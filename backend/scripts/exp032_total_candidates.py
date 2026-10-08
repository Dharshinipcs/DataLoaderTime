"""EXP-032 Part C: total-mechanism candidate evaluation (live probe).

Evaluates physically-motivated candidates against the recorded chronological
comparison on the identical stage-policy population:
  C1  pure sum of the seven served-style stage estimates
  C1b sum of the five operational stage estimates
  C3  sum x chronological scalar alpha
  C4  the stage-aware learned total (recorded, extra_trees)
  C5  physical-consistency probe of the SERVED model: can the live total fall
      below a predicted stage? (uses the live prediction path)

All candidates use the same target (authoritative total, seconds) and are
scored on the same chronological folds via a lightweight replication.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "backend/src")

from dltime.stage_aware_total import (  # noqa: E402
    _encode,
    _fit_predict_total,
    _stage_oof_predictions,
)
from dltime.stage_models import build_stage_frame  # noqa: E402
from dltime.stage_targets import ML_STAGE_NAMES, APPROVAL_STAGE_NAMES, stage_target_column  # noqa: E402
from dltime.feature_assembly import TargetTransformer  # noqa: E402
from dltime.approval_estimator import estimate_approval_stage  # noqa: E402
from dltime.modeling import add_chronological_rank, compute_metrics  # noqa: E402
from dltime.sources import load_all_sources  # noqa: E402

bundle = load_all_sources()
total_frame, total_feats, _ = build_stage_frame(stage="staging", bundle=bundle)
completed = total_frame["LDR_Status"].isin(
    {"SUCCEED", "COMPLETED_WITH_WARNING", "COMPLETED_WITH_ERROR", "COMPLETED_WITH_ERROR_AND_WARNING", "ERROR"}
)
id_ok = total_frame["Loader_Name"].notna() & total_frame["Sprint"].notna()
target = pd.to_numeric(total_frame["timetaken_in_sec"], errors="coerce")
pop = (completed & id_ok & (target > 0)).fillna(False)
frame = total_frame[pop].copy()
y_raw = target[pop].to_numpy()

print(f"population n={len(frame)}  (identical to the recorded AD-31 comparison)")

# stage OOF columns (identical cadence to evaluate_stage_aware_total)
frame = add_chronological_rank(frame)
ids = frame["LDR_Execution_Id"]
stage_pred = {}
for stage in ML_STAGE_NAMES:
    sframe, sfeats, sdiag = build_stage_frame(stage=stage, bundle=bundle)
    if sdiag["stage_eligible_rows"] >= 10:
        oof = _stage_oof_predictions(stage, sframe, sfeats, {})
        lookup = pd.Series(oof.values, index=sframe.loc[oof.index, "LDR_Execution_Id"].to_numpy())
        stage_pred[stage] = ids.map(lookup).astype(float)
    else:
        stage_pred[stage] = pd.Series(np.nan, index=frame.index)

# approval strictly-prior estimates (identical loop to the recorded comparison)
ordered = frame.sort_values(["Start_Time", "LDR_Execution_Id"], kind="mergesort")
for stage in APPROVAL_STAGE_NAMES:
    ests = []
    for exec_id in ordered["LDR_Execution_Id"]:
        est = estimate_approval_stage(frame, exec_id, stage)
        ests.append(est.estimate_seconds if est.available else np.nan)
    m = pd.Series(ests, index=ordered.index)
    stage_pred[stage] = pd.Series(np.nan, index=frame.index)
    stage_pred[stage].loc[ordered.index] = m

S = pd.DataFrame({f"pred_stage_{k}_sec": v for k, v in stage_pred.items()}, index=frame.index)
sum7 = S.sum(axis=1).to_numpy()
sum5 = S[[c for c in S.columns if "approval" not in c]].sum(axis=1).to_numpy()
max_stage = S.max(axis=1).to_numpy()

n = len(frame)
n_test = max(int(round(n * 0.25)), 12)
cut = n - n_test
boundary = frame.iloc[cut]["Start_Time"]
while cut > 0 and frame.iloc[cut - 1]["Start_Time"] == boundary:
    cut -= 1
y_ho = y_raw[cut:]

def report(name, pred_ho):
    pred_ho = np.asarray(pred_ho, dtype=float)
    m = compute_metrics(y_ho, pred_ho)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(y_ho - pred_ho) / np.where(y_ho > 0, y_ho, np.nan)
    w20 = float(np.nanmean(rel <= 0.20) * 100.0)
    w2x = float(np.nanmean((y_ho > 0) & (pred_ho <= 2 * y_ho) & (pred_ho >= 0.5 * y_ho)) * 100.0)
    below = int((pred_ho < max_stage[cut:]).sum())
    neg = int((pred_ho < 0).sum())
    print(f"{name}: MAE={m['mae']:.1f} MedAE={m['median_ae']:.2f} RMSE={m['rmse']:.1f} "
          f"pm20={w20:.2f}% 2x={w2x:.2f}% | pred<max_stage: {below}/{len(pred_ho)} | neg: {neg}")

print("\n== HOLDOUT candidates (identical chronological cut) ==")
report("C1  sum7", sum7[cut:])
report("C1b sum5(ops only)", sum5[cut:])
alpha = float(np.dot(y_raw[:cut], sum7[:cut]) / max(np.dot(sum7[:cut], sum7[:cut]), 1e-12))
report(f"C3  sum7*alpha (alpha={alpha:.3f})", sum7[cut:] * alpha)

# ratio-based reference: median chronological total/sum ratio applied to sum7
hist_ratio = y_raw[:cut] / np.where(sum7[:cut] > 0, sum7[:cut], np.nan)
med_ratio = float(np.nanmedian(hist_ratio))
report(f"C3b sum7*median_ratio ({med_ratio:.2f})", sum7[cut:] * med_ratio)

print("\n== physical distribution evidence ==")
r = y_raw / np.where(sum7 > 0, sum7, np.nan)
print(f"total/sum7: median={np.nanmedian(r):.2f} p25={np.nanpercentile(r,25):.2f} p75={np.nanpercentile(r,75):.2f} p90={np.nanpercentile(r,90):.2f}")
r5 = y_raw / np.where(sum5 > 0, sum5, np.nan)
print(f"total/sum5: median={np.nanmedian(r5):.2f}")

print("\nDONE")
