"""EXP-036c: floor SCOPE evaluation on the clean protocol (prompt §12/§15).

The floor's physical basis is the ACTUAL containment invariant (actual total
contains every actual stage duration; EXP-032: 0/1575 violations). Applying it
with a stage estimate whose model is documented as experimental/small-sample
injects a disclosed-unreliable estimate into the ETA. This script quantifies:

  F0: raw total (no floor)
  F1: floor = max over ALL five ML stage predictions (current serving rule)
  F2: floor = max over stage predictions whose lifecycle status is NOT
      experimental/small-sample (evidence-adequate stages only)

Metrics: MAE / RMSE / MedAE / within-20% / within-2x / floor>actual counts.
Run from repo root:
  uv run --project backend python backend/scripts/exp036c_floor_scope.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _prep_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == "string" or isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].fillna("__MISSING__").astype("string")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def main() -> None:
    from dltime.feature_assembly import TargetTransformer
    from dltime.modeling import add_chronological_rank, chronological_holdout_split, compute_metrics
    from dltime.stage_models import build_stage_frame, _make_estimator, STAGE_MODEL_SERVING_KINDS
    from dltime.stage_serving import _StagePredictor
    from dltime.stage_targets import ML_STAGE_NAMES
    from dltime.sources import load_all_sources

    tt = TargetTransformer(mode="log1p")
    bundle = load_all_sources()

    total_frame, total_feats, _ = build_stage_frame(stage="staging", bundle=bundle)
    completed = total_frame["LDR_Status"].isin(
        {"SUCCEED", "COMPLETED_WITH_WARNING", "COMPLETED_WITH_ERROR", "COMPLETED_WITH_ERROR_AND_WARNING", "ERROR"}
    )
    id_ok = total_frame["Loader_Name"].notna() & total_frame["Sprint"].notna()
    target = pd.to_numeric(total_frame["total_timetaken_in_min"], errors="coerce") * 60.0
    pop = (completed & id_ok & (target > 0)).fillna(False)
    frame = add_chronological_rank(total_frame[pop].copy())
    y_raw = target[pop].loc[frame.index]
    train, test, split_info = chronological_holdout_split(frame)
    train_ids = set(train["LDR_Execution_Id"])
    test_ids = set(test["LDR_Execution_Id"])

    # ---- frozen stage models (train-period rows only) -----------------------
    stage_predictors: dict = {}
    for stage in ML_STAGE_NAMES:
        kind = STAGE_MODEL_SERVING_KINDS[stage]["model_kind"]
        sframe, sfeats, _diag = build_stage_frame(stage=stage, bundle=bundle)
        elig = sframe[sframe["stage_eligible"].astype(bool)]
        elig_train = elig[elig["LDR_Execution_Id"].isin(train_ids)]
        if len(elig_train) < 10:
            continue
        X = _prep_numeric(elig_train[sfeats].copy())
        medians = X.median(numeric_only=True)
        Xenc = pd.get_dummies(X.astype(object).where(X.notna(), None), dummy_na=True)
        model = _make_estimator(kind)
        model.fit(Xenc, tt.transform(pd.to_numeric(elig_train[f"stage_{stage}_sec"], errors="coerce")))
        pred = _StagePredictor(stage, kind, model, sfeats, len(elig_train), medians)
        pred._enc_columns = Xenc.columns
        stage_predictors[stage] = pred

    # ---- per-stage holdout predictions (frozen models) ----------------------
    preds_by_stage: dict[str, np.ndarray] = {}
    for stage, pred in stage_predictors.items():
        sframe_full = pred  # placeholder to keep linters quiet
        del sframe_full
        sframe, sfeats, _diag = build_stage_frame(stage=stage, bundle=bundle)
        merged = sframe.set_index("LDR_Execution_Id")
        vals: list[float] = []
        for eid in test["LDR_Execution_Id"]:
            try:
                row_ctx = merged.loc[[eid]]
                vals.append(float(pred.predict_seconds(row_ctx)))
            except Exception:
                vals.append(np.nan)
        preds_by_stage[stage] = np.asarray(vals, dtype=float)

    # which stages are evidence-adequate? The architecture's own small-sample
    # disclosure: pre_load trains on n=13 (EXPERIMENTAL_SMALL_SAMPLE, serving
    # label). Adequacy criterion here = stage train n >= 30.
    train_n = {}
    for s in ML_STAGE_NAMES:
        sframe, _f, _d = build_stage_frame(stage=s, bundle=bundle)
        elig = sframe[sframe["stage_eligible"].astype(bool)]
        train_n[s] = int(elig[elig["LDR_Execution_Id"].isin(train_ids)].shape[0])
    print("stage train-n (clean protocol):", train_n)
    adequate_stages = [s for s in ML_STAGE_NAMES if s in stage_predictors and train_n[s] >= 30]

    # ---- raw total predictions from the exp036 clean CSV --------------------
    import os

    csv_path = (
        "clean_holdout_predictions.csv"
        if os.path.exists("clean_holdout_predictions.csv")
        else os.path.join("backend", "scripts", "clean_holdout_predictions.csv")
    )
    rows = list(pd.read_csv(csv_path).itertuples(index=False))
    order = [r.LDR_Execution_Id for r in rows]
    id_to_pos = {eid: i for i, eid in enumerate(order)}
    raw = np.asarray([r.raw for r in rows], dtype=float)
    actual = np.asarray([r.actual for r in rows], dtype=float)

    def stacked(stage_list: list[str]) -> np.ndarray:
        mat = np.vstack([preds_by_stage[s][[id_to_pos[e] for e in order]] for s in stage_list])
        return np.nanmax(mat, axis=0)

    floor_all = stacked([s for s in ML_STAGE_NAMES if s in stage_predictors])
    floor_adequate = stacked(adequate_stages) if adequate_stages else np.full(len(raw), np.nan)

    variants = {
        "F0 raw (no floor)": raw,
        "F1 floor all ML stages": np.maximum(raw, np.where(np.isfinite(floor_all), floor_all, raw)),
        "F2 floor evidence-adequate stages": np.maximum(
            raw, np.where(np.isfinite(floor_adequate), floor_adequate, raw)
        ),
    }
    print(f"\nholdout n={len(raw)}  (TRAIN last {split_info['train_last_start']})")
    for name, eta in variants.items():
        m = compute_metrics(actual, eta)
        engaged = np.isfinite(eta) & (eta > raw + 1e-9)
        above = int(((eta > actual) & engaged).sum())
        print(
            f"{name}: MAE {m['mae']:.1f} / RMSE {m['rmse']:.1f} / MedAE {m['median_ae']:.2f} / "
            f"±20% {m['within_20pct']:.1f}% / 2× {m['within_2x']:.1f}% / engaged {int(engaged.sum())} / "
            f"engaged&>actual {above}"
        )
    print("\nDONE")


if __name__ == "__main__":
    main()
