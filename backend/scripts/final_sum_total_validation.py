"""FINAL PRODUCTIZATION PASS — Part 7 total-policy validation.

Question: may the product display TOTAL ETA = sum(five operational stage
predictions), retiring the stage-aware total model from the serving path?

Protocol (reuses the accepted EXP-036 clean leakage-safe holdout machinery):
  1. identical chronological TRAIN/HOLDOUT split as EXP-036;
  2. stage models fit ONLY on TRAIN-period eligible rows, then frozen;
  3. frozen stage models predict the HOLDOUT rows (pre-execution context +
     strictly-prior history only);
  4. compare on the same revealed actuals:
       A. stage-aware total (frozen pipeline, unfloored raw output; taken from
          the EXP-036 evidence file, ids/actuals cross-checked),
       B. simple sum of the five operational stage predictions,
       C. B + the two contextual approval estimates (for context only);
  5. extreme-case forensics (rows where the sum is dramatically off) and
     stage-window coverage forensics (how much of the actual total the
     measured stage windows account for on holdout rows).

Run from repo root:
  uv run --project backend python backend/scripts/final_sum_total_validation.py
"""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# repo root = two levels above this file (backend/scripts/<name>.py)
ROOT = Path(__file__).resolve().parents[2]


def load_exp036():
    spec = importlib.util.spec_from_file_location(
        "exp036", ROOT / "backend" / "scripts" / "exp036_clean_holdout.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    from dltime.modeling import chronological_holdout_split, compute_metrics
    from dltime.stage_models import _make_estimator, build_stage_frame
    from dltime.stage_serving import _StagePredictor
    from dltime.stage_targets import APPROVAL_STAGE_NAMES, ML_STAGE_NAMES
    from dltime.approval_estimator import estimate_approval_stage
    from dltime.feature_assembly import TargetTransformer
    from dltime.sources import load_all_sources

    mod = load_exp036()
    bundle = load_all_sources()
    frame, y_raw, total_feats = mod.build_population(bundle)
    train, test, split_info = chronological_holdout_split(frame)
    print(f"split reproduced: TRAIN n={len(train)} / HOLDOUT n={len(test)}")

    tt = TargetTransformer(mode="log1p")

    # ---- frozen stage models on TRAIN-period rows (identical to EXP-036) ----
    stage_predictors = {}
    stage_frames = {}
    for stage in ML_STAGE_NAMES:
        sframe, sfeats, _sdiag = build_stage_frame(stage=stage, bundle=bundle)
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
        kind = {"staging": "extra_trees", "prevalidation": "extra_trees",
                "transformation": "random_forest", "pre_load": "extra_trees",
                "data_loading": "ridge"}[stage]
        model = _make_estimator(kind)
        model.fit(Xenc, y_log)
        pred = _StagePredictor(stage, kind, model, sfeats, len(elig_train), medians)
        pred._enc_columns = Xenc.columns
        stage_predictors[stage] = pred
        stage_frames[stage] = sframe
    print("frozen stage models:", {k: v.n_train for k, v in stage_predictors.items()})

    # ---- stage predictions on the HOLDOUT (frozen; strictly pre-execution) --
    merged = {st: sf.set_index("LDR_Execution_Id") for st, sf in stage_frames.items()}
    stage_pred = {st: [] for st in ML_STAGE_NAMES}
    for eid in test["LDR_Execution_Id"]:
        for st in ML_STAGE_NAMES:
            try:
                stage_pred[st].append(float(stage_predictors[st].predict_seconds(merged[st].loc[[eid]])))
            except Exception:
                stage_pred[st].append(np.nan)
    stage_df = pd.DataFrame(stage_pred, index=test.index)
    sum_pred = stage_df.sum(axis=1, skipna=True).to_numpy(dtype=float)

    # approvals on the holdout (contextual statistics; NOT in the product sum)
    appr = np.zeros(len(test))
    appr_ok = np.ones(len(test), dtype=bool)
    for i, (_idx, row) in enumerate(test.iterrows()):
        for st in APPROVAL_STAGE_NAMES:
            est = estimate_approval_stage(frame, row["LDR_Execution_Id"], st)
            if est.available and est.estimate_seconds is not None:
                appr[i] += float(est.estimate_seconds)
            else:
                appr_ok[i] = False
    y = y_raw.loc[test.index].to_numpy(dtype=float)

    # ---- A: stage-aware total from the EXP-036 evidence file ----------------
    csv = pd.read_csv(ROOT / "clean_holdout_predictions.csv")
    ids_test = set(test["LDR_Execution_Id"].astype(np.int64))
    assert set(csv["LDR_Execution_Id"].astype(np.int64)) == ids_test, "holdout ids drifted"
    csv_aligned = csv.set_index("LDR_Execution_Id").loc[
        test["LDR_Execution_Id"].astype(np.int64)
    ]
    assert np.allclose(csv_aligned["actual"].to_numpy(), y, rtol=1e-9), "actuals drifted"
    aware_pred = csv_aligned["raw"].to_numpy(dtype=float)

    # ---- metrics -------------------------------------------------------------
    def fmt(m):
        return {k: (round(v, 3) if isinstance(v, (int, float)) else v) for k, v in m.items()}

    print("\n== holdout comparison (identical 85 rows, identical actuals) ==")
    print("A stage-aware total :", fmt(compute_metrics(y, aware_pred)))
    print("B simple 5-stage sum:", fmt(compute_metrics(y, sum_pred)))
    print("C B + approvals     :", fmt(compute_metrics(y, sum_pred + appr)))

    # ---- extreme-case forensics ---------------------------------------------
    ratio = np.divide(sum_pred, y, out=np.full_like(y, np.nan), where=y > 0)
    print("\n== B vs actual: extreme cases ==")
    for lo, hi, label in [(0, 0.5, "sum < 0.5x actual (severe under)"),
                          (0.5, 0.8, "sum 0.5-0.8x actual"),
                          (0.8, 1.25, "sum within 0.8-1.25x"),
                          (1.25, 2.0, "sum 1.25-2x actual"),
                          (2.0, np.inf, "sum > 2x actual (severe over)")]:
        n = int(((ratio >= lo) & (ratio < hi)).sum())
        print(f"  {label}: {n}/{len(ratio)}")
    worst = pd.DataFrame({
        "loader": test["Loader_Name"].to_numpy(),
        "actual": y, "sum": sum_pred, "aware": aware_pred, "sum/actual": ratio,
    }).reindex(pd.Series(ratio).abs().sub(1).abs().sort_values(ascending=False).index)
    print(worst.head(10).to_string(index=False))

    # ---- stage-window coverage forensics on holdout rows ---------------------
    cov = []
    for eid in test["LDR_Execution_Id"]:
        total = 0.0
        n_stages = 0
        for st in ML_STAGE_NAMES:
            sf = merged[st]
            if eid in sf.index and bool(sf.loc[[eid], "stage_eligible"].iloc[0]):
                v = pd.to_numeric(sf.loc[[eid], f"stage_{st}_sec"], errors="coerce").iloc[0]
                if pd.notna(v):
                    total += float(v)
                    n_stages += 1
        cov.append((n_stages, total))
    cov_df = pd.DataFrame(cov, columns=["n_stage_windows", "stage_window_sum"], index=test.index)
    cov_df["actual"] = y
    have = cov_df[cov_df["n_stage_windows"] > 0]
    share = (have["stage_window_sum"] / have["actual"]).replace([np.inf], np.nan).dropna()
    print(f"\n== stage-window coverage on holdout (measured windows vs actual total) ==")
    print(f"rows with >=1 measured stage window: {len(have)}/{len(cov_df)}")
    print(f"window-sum/actual: median {share.median():.3f}  p25 {share.quantile(.25):.3f}  p75 {share.quantile(.75):.3f}")

    out = pd.DataFrame({
        "LDR_Execution_Id": test["LDR_Execution_Id"].to_numpy(),
        "loader": test["Loader_Name"].to_numpy(),
        "actual": y,
        "aware_total": aware_pred,
        "stage_sum": sum_pred,
        "stage_sum_plus_approvals": sum_pred + appr,
        **{f"pred_{st}": stage_df[st].to_numpy() for st in ML_STAGE_NAMES},
    })
    out.to_csv(ROOT / "final_total_policy_comparison.csv", index=False)
    print("\nwrote final_total_policy_comparison.csv")
    print("DONE")


if __name__ == "__main__":
    main()
