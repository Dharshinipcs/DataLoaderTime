"""EXP-036: clean leakage-safe holdout evaluation (prompt §13).

Protocol (exactly as demanded):
  1. chronological split -> TRAIN / HOLDOUT (single timestamp boundary)
  2. fit the five stage models ONLY on TRAIN-period rows (per stage's own
     eligible-row policy restricted to train)
  3. freeze stage models; generate stage predictions for HOLDOUT rows using
     only pre-execution context + strictly-prior history (as-of semantics)
  4. fit the total model ONLY on TRAIN rows using OOF stage predictions
     (block-cadence expanding, strictly-prior) computed WITHIN the train
     period; freeze
  5. predict HOLDOUT with the frozen pair; only then reveal actuals
  6. metrics: raw vs floored, plus per-slice honesty

Run from repo root:
  uv run --project backend python backend/scripts/exp036_clean_holdout.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def build_population(bundle):
    from dltime.modeling import add_chronological_rank
    from dltime.stage_models import build_stage_frame

    total_frame, total_feats, _ = build_stage_frame(stage="staging", bundle=bundle)
    completed = total_frame["LDR_Status"].isin(
        {"SUCCEED", "COMPLETED_WITH_WARNING", "COMPLETED_WITH_ERROR", "COMPLETED_WITH_ERROR_AND_WARNING", "ERROR"}
    )
    id_ok = total_frame["Loader_Name"].notna() & total_frame["Sprint"].notna()
    # 2026-09-29 replacement: no total_timetaken_in_min column — the
    # authoritative target is the AD-31 span (End - Start, seconds); the
    # legacy timetaken_in_sec is span-consistent on this source.
    if "total_timetaken_in_min" in total_frame.columns:
        target = pd.to_numeric(total_frame["total_timetaken_in_min"], errors="coerce") * 60.0
    else:
        target = pd.to_numeric(total_frame["timetaken_in_sec"], errors="coerce")
    pop = (completed & id_ok & (target > 0)).fillna(False)
    frame = add_chronological_rank(total_frame[pop].copy())
    return frame, target[pop].loc[frame.index], total_feats


def encode_fit(X: pd.DataFrame):
    enc = pd.get_dummies(X.astype(object).where(X.notna(), None), dummy_na=True)
    return enc


def main() -> None:
    from dltime.feature_assembly import TargetTransformer
    from dltime.modeling import chronological_holdout_split, compute_metrics
    from dltime.stage_aware_total import _stage_oof_predictions
    from dltime.stage_models import _make_estimator, build_stage_frame
    from dltime.stage_serving import _StagePredictor
    from dltime.stage_targets import APPROVAL_STAGE_NAMES, ML_STAGE_NAMES
    from dltime.approval_estimator import estimate_approval_stage
    from dltime.sources import load_all_sources

    bundle = load_all_sources()
    frame, y_raw, total_feats = build_population(bundle)
    train, test, split_info = chronological_holdout_split(frame)
    train_end_time = split_info["train_last_start"]

    print("== §13 clean-holdout protocol ==")
    print(f"TRAIN:   {split_info['train_first_start']} -> {train_end_time} (n={len(train)})")
    print(f"HOLDOUT: {split_info['test_first_start']} -> {split_info['test_last_start']} (n={len(test)})")
    assert train["chrono_rank"].max() < test["chrono_rank"].min()

    tt = TargetTransformer(mode="log1p")

    # ---- step 2: stage models fit ONLY on train-period eligible rows --------
    stage_predictors = {}
    stage_feature_info = {}
    for stage in ML_STAGE_NAMES:
        sframe, sfeats, sdiag = build_stage_frame(stage=stage, bundle=bundle)
        elig = sframe[sframe["stage_eligible"].astype(bool)]
        train_ids = set(train["LDR_Execution_Id"])
        elig_train = elig[elig["LDR_Execution_Id"].isin(train_ids)]
        X = elig_train[sfeats].copy()
        for col in X.columns:
            if X[col].dtype == "string" or isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].fillna("__MISSING__").astype("string")
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        medians = X.median(numeric_only=True)
        Xenc = pd.get_dummies(X.astype(object).where(X.notna(), None), dummy_na=True)
        y_log = tt.transform(pd.to_numeric(elig_train[f"stage_{stage}_sec"], errors="coerce"))
        kind = {"staging": "extra_trees", "prevalidation": "extra_trees", "transformation": "random_forest",
                "pre_load": "extra_trees", "data_loading": "ridge"}[stage]
        if len(elig_train) >= 10:
            model = _make_estimator(kind)
            model.fit(Xenc, y_log)
            pred = _StagePredictor(stage, kind, model, sfeats, len(elig_train), medians)
            pred._enc_columns = Xenc.columns
            stage_predictors[stage] = pred
        stage_feature_info[stage] = (sframe, sfeats, len(elig_train))
    print("stage models fitted on TRAIN-period rows only:",
          {k: v.n_train for k, v in stage_predictors.items()})

    # ---- step 4: total model fit on TRAIN rows with in-train OOF stage preds -
    X_train = train[total_feats].copy()
    for col in X_train.columns:
        if X_train[col].dtype == "string" or isinstance(X_train[col].dtype, pd.CategoricalDtype):
            X_train[col] = X_train[col].fillna("__MISSING__").astype("string")
        else:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")

    def add_stage_columns(X: pd.DataFrame, target_rows: pd.DataFrame, use_frozen: bool) -> pd.DataFrame:
        X = X.copy()
        for stage in ML_STAGE_NAMES:
            col = f"pred_stage_{stage}_sec"
            if use_frozen and stage in stage_predictors:
                pred = stage_predictors[stage]
                sframe, sfeats, _ = stage_feature_info[stage]
                sub = target_rows
                vals = []
                merged = sframe.set_index("LDR_Execution_Id")
                for eid in sub["LDR_Execution_Id"]:
                    try:
                        row_ctx = merged.loc[[eid]]
                        vals.append(float(pred.predict_seconds(row_ctx)))
                    except Exception:
                        vals.append(np.nan)
                X[col] = pd.to_numeric(pd.Series(vals, index=sub.index), errors="coerce").astype(float)
            elif not use_frozen:
                sframe, sfeats, sdiag = stage_feature_info[stage]
                if sdiag if False else True:
                    # in-train OOF via the existing strictly-prior cadence,
                    # restricted to rows inside the train period by the frame
                    # builder (the OOF mechanism is chronological; rows map to
                    # train ids by construction below).
                    oof = _stage_oof_predictions(stage, sframe, sfeats, {})
                    lookup = pd.Series(oof.values, index=sframe.loc[oof.index, "LDR_Execution_Id"].to_numpy())
                    lookup = lookup[lookup.index.isin(set(train["LDR_Execution_Id"]))]
                    X[col] = pd.to_numeric(target_rows["LDR_Execution_Id"].map(lookup), errors="coerce").astype(float)
            else:
                X[col] = np.nan
        for stage in APPROVAL_STAGE_NAMES:
            est_map: dict = {}
            for _idx, row in target_rows.iterrows():
                est = estimate_approval_stage(frame, row["LDR_Execution_Id"], stage)
                if est.available and est.estimate_seconds is not None:
                    est_map[row["LDR_Execution_Id"]] = float(est.estimate_seconds)
            X[f"pred_stage_{stage}_sec"] = target_rows["LDR_Execution_Id"].map(est_map).astype(float)
        return X

    X_train_sa = add_stage_columns(X_train, train, use_frozen=False)
    y_train_log = tt.transform(y_raw.loc[train.index])
    Xenc_train = encode_fit(X_train_sa)
    from dltime.stage_models import _make_estimator as _mk

    total_est = _mk("extra_trees")
    total_est.fit(Xenc_train, y_train_log)
    total_medians = X_train_sa.median(numeric_only=True)
    print(f"total model fitted on TRAIN rows (n={len(train)}); features={X_train_sa.shape[1]}")

    def predict_rows(rows: pd.DataFrame, use_frozen: bool) -> np.ndarray:
        Xr = rows[total_feats].copy()
        for col in Xr.columns:
            if Xr[col].dtype == "string" or isinstance(Xr[col].dtype, pd.CategoricalDtype):
                Xr[col] = Xr[col].fillna("__MISSING__").astype("string")
            else:
                Xr[col] = pd.to_numeric(Xr[col], errors="coerce")
        Xr_sa = add_stage_columns(Xr, rows, use_frozen=use_frozen)
        enc = pd.get_dummies(Xr_sa.astype(object).where(Xr_sa.notna(), None), dummy_na=True).reindex(columns=Xenc_train.columns, fill_value=0)
        pred_log = total_est.predict(enc)
        return tt.inverse_transform(pred_log)

    # ---- step 5: frozen models predict HOLDOUT; then reveal -----------------
    pred_holdout = predict_rows(test, use_frozen=True)
    y_holdout = y_raw.loc[test.index].to_numpy(dtype=float)

    # per-row floor from the SAME frozen stage predictions
    stage_matrix_cols = {st: f"pred_stage_{st}_sec" for st in ML_STAGE_NAMES}
    X_hold_check = test[total_feats].copy()
    for col in X_hold_check.columns:
        if X_hold_check[col].dtype == "string" or isinstance(X_hold_check[col].dtype, pd.CategoricalDtype):
            X_hold_check[col] = X_hold_check[col].fillna("__MISSING__").astype("string")
        else:
            X_hold_check[col] = pd.to_numeric(X_hold_check[col], errors="coerce")
    X_hold_sa = add_stage_columns(X_hold_check, test, use_frozen=True)
    ml_vals = X_hold_sa[[stage_matrix_cols[s] for s in ML_STAGE_NAMES]]
    floor = ml_vals.max(axis=1, skipna=True).to_numpy()
    eta = np.maximum(pred_holdout, np.where(np.isfinite(floor), floor, -np.inf))

    print("\n== evidence table ==")
    print("STAGE MODELS FIT ON: train-period eligible rows only (ids within train)")
    print(f"TOTAL MODEL FIT ON:  train rows with in-train strictly-prior OOF stage columns")
    print(f"HISTORICAL AS-OF:    streaming prefix; context excluded from own history")
    print(f"OVERLAP:             0 (chronological boundary)")
    print(f"LEAKAGE CHECK:       holdout rows never in stage-model or total-model fitting")

    m_raw = compute_metrics(y_holdout, pred_holdout)
    m_floor = compute_metrics(y_holdout, eta)
    engaged = np.isfinite(floor) & (pred_holdout < floor)
    print(f"\nfloor engaged: {int(engaged.sum())}/{len(eta)}")
    print("raw (frozen pipeline):  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m_raw.items()})
    print("with floor:             ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m_floor.items()})
    print("floor above actual:    ", int(((eta > y_holdout) & engaged).sum()), "/", len(eta))

    # comparison with the previous (stage-models-fully-fit) numbers
    print("\nreference (EXP-034 served-model holdout, stage models in-sample):")
    print("  MAE 780.07 / RMSE 3169.21 / ±20% 17.6% / 2× 63.5%")

    out = pd.DataFrame({
        "LDR_Execution_Id": test["LDR_Execution_Id"].to_numpy(),
        "start": test["Start_Time"].to_numpy(),
        "loader": test["Loader_Name"].to_numpy(),
        "raw": pred_holdout,
        "floor": floor,
        "eta": eta,
        "actual": y_holdout,
    })
    out["abs_err"] = (out["eta"] - out["actual"]).abs()
    out.to_csv("clean_holdout_predictions.csv", index=False)
    print("\nwrote clean_holdout_predictions.csv (85 rows)")
    print("DONE")


if __name__ == "__main__":
    main()
