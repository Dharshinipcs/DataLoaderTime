"""EXP-036b: diagnose the floor's failures in the clean protocol (prompt §12/§15).

Questions answered before any policy change:
  1. Wrapper sanity: on TRAIN-period rows the frozen clean-protocol stage
     models must roughly track actual stage durations (they were fit there).
  2. Calibration on HOLDOUT: predicted stage duration vs actual stage duration
     (median ratio), for the stages that drive the floor (staging, data_loading
     have the largest n).
  3. Raw total ETA vs actual: same quantity, for context.
  4. In the SERVED (ad34) pipeline — the architecture that ships — how often
     does the floor move the ETA, how much, and toward/away from the actual?
Run from repo root:
  uv run --project backend python backend/scripts/exp036b_floor_diagnosis.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _encode_fit(X: pd.DataFrame):
    return pd.get_dummies(X.astype(object).where(X.notna(), None), dummy_na=True)


def _prep_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == "string" or isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].fillna("__MISSING__").astype("string")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def _fit_stage(stage: str, bundle, train_ids: set, tt):
    from dltime.stage_models import _make_estimator, build_stage_frame
    from dltime.stage_serving import _StagePredictor

    sframe, sfeats, _diag = build_stage_frame(stage=stage, bundle=bundle)
    elig = sframe[sframe["stage_eligible"].astype(bool)]
    elig_train = elig[elig["LDR_Execution_Id"].isin(train_ids)]
    X = _prep_numeric(elig_train[sfeats].copy())
    medians = X.median(numeric_only=True)
    Xenc = _encode_fit(X.astype(object).where(X.notna(), None))
    y_log = tt.transform(pd.to_numeric(elig_train[f"stage_{stage}_sec"], errors="coerce"))
    kind = {
        "staging": "extra_trees",
        "prevalidation": "extra_trees",
        "transformation": "random_forest",
        "pre_load": "extra_trees",
        "data_loading": "ridge",
    }[stage]
    if len(elig_train) < 10:
        return None, (sframe, None, len(elig_train))
    model = _make_estimator(kind)
    model.fit(Xenc, y_log)
    pred = _StagePredictor(stage, kind, model, sfeats, len(elig_train), medians)
    pred._enc_columns = Xenc.columns
    return pred, (sframe, sfeats, len(elig_train))


def main() -> None:
    from dltime.feature_assembly import TargetTransformer
    from dltime.modeling import add_chronological_rank, chronological_holdout_split
    from dltime.stage_models import build_stage_frame
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

    print("== stage sanity on TRAIN rows (fit rows) ==")
    stage_predictors: dict = {}
    stage_info: dict = {}
    for stage in ML_STAGE_NAMES:
        pred, info = _fit_stage(stage, bundle, train_ids, tt)
        stage_predictors[stage] = pred
        stage_info[stage] = info
        if pred is None:
            print(f"  {stage}: too few train rows ({info[2]}), skipped")
            continue
        sframe = info[0]
        elig = sframe[sframe["stage_eligible"].astype(bool)]
        in_fit = elig[elig["LDR_Execution_Id"].isin(train_ids)]
        sub = in_fit.head(60)
        try:
            merged = sframe.set_index("LDR_Execution_Id")
            preds = []
            actuals = []
            for eid in sub["LDR_Execution_Id"]:
                row_ctx = merged.loc[[eid]]
                preds.append(float(pred.predict_seconds(row_ctx)))
                actuals.append(float(row_ctx[f"stage_{stage}_sec"].iloc[0]))
            preds = np.asarray(preds)
            actuals = np.asarray(actuals)
            ok = np.isfinite(preds) & np.isfinite(actuals) & (actuals > 0)
            ratio = preds[ok] / actuals[ok]
            print(
                f"  {stage}: n_train={len(in_fit)} median(pred/actual) on fit rows="
                f"{np.median(ratio):.2f}  (sanity: should be ~0.5-2)"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {stage}: sanity failed: {exc}")

    print("\n== holdout calibration: predicted vs actual stage duration ==")
    for stage in ML_STAGE_NAMES:
        pred = stage_predictors[stage]
        if pred is None:
            continue
        sframe, sfeats, _n = stage_info[stage]
        sframe = sframe[sframe["LDR_Execution_Id"].isin(test_ids)]
        sframe = sframe[sframe["stage_eligible"].astype(bool)]
        if len(sframe) == 0:
            print(f"  {stage}: no holdout stage rows")
            continue
        merged_full = stage_info[stage][0].set_index("LDR_Execution_Id")
        preds, actuals = [], []
        for eid in sframe["LDR_Execution_Id"]:
            try:
                row_ctx = merged_full.loc[[eid]]
                preds.append(float(pred.predict_seconds(row_ctx)))
                actuals.append(float(row_ctx[f"stage_{stage}_sec"].iloc[0]))
            except Exception:
                continue
        preds = np.asarray(preds)
        actuals = np.asarray(actuals)
        ok = np.isfinite(preds) & np.isfinite(actuals) & (actuals > 0)
        if ok.sum() == 0:
            print(f"  {stage}: no comparable holdout rows")
            continue
        ratio = preds[ok] / actuals[ok]
        print(
            f"  {stage}: n={int(ok.sum())} median(pred/actual)={np.median(ratio):.2f} "
            f"p25={np.percentile(ratio, 25):.2f} p75={np.percentile(ratio, 75):.2f}"
        )

    print("\n== raw total ETA vs actual on holdout ==")
    y = y_raw.loc[test.index].to_numpy(dtype=float)
    # re-load the raw predictions produced by exp036 (same protocol, same split)
    import csv
    import os

    csv_path = os.path.join("backend", "scripts", "clean_holdout_predictions.csv") if os.path.exists(os.path.join("backend", "scripts", "clean_holdout_predictions.csv")) else "clean_holdout_predictions.csv"
    if os.path.exists(csv_path):
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
        raw = np.asarray([float(r["raw"]) for r in rows])
        actual = np.asarray([float(r["actual"]) for r in rows])
        floor = np.asarray([float(r["floor"]) for r in rows])
        eta = np.asarray([float(r["eta"]) for r in rows])
        print(f"  n={len(rows)} (from clean_holdout_predictions.csv)")
        print(f"  median(actual/raw): {np.median(actual / raw):.2f}")
        print(f"  median(actual/floor): {np.median(actual / floor):.2f}")
        print(f"  floor > actual: {(floor > actual).sum()}/{len(rows)}")
        print(f"  raw > actual:   {(raw > actual).sum()}/{len(rows)}")
        # how much of the floor's error is stage-prediction over-prediction?
        floor_minus_raw = floor - raw
        print(f"  median floor lift over raw: {np.median(floor_minus_raw):.1f}s")
        # would a capped floor help? (least-invasive alternatives, evidence only)
        for cap in (0.5, 0.75, 1.0):
            capped = np.maximum(raw, np.minimum(floor, cap * raw))
            mae = np.abs(capped - actual).mean()
            within20 = (np.abs(capped - actual) <= 0.2 * actual).mean()
            print(f"  alt floor = min(stage_max, {cap:.2f}x raw): MAE {mae:.1f} / ±20% {within20 * 100:.1f}%")
    else:
        print("  clean_holdout_predictions.csv not found; run exp036 first")

    print("\nDONE")


if __name__ == "__main__":
    main()
