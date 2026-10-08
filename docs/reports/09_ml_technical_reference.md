# Report 9 — ML Technical Reference

**Status: implemented; reflects the code as of 2026-10-08.**

## Training pipeline (`retraining.py`)

1. `load_all_sources()` — validated workbook bundle (Report 7).
2. Population via `backend/scripts/exp036_clean_holdout.py::build_population` (the same protocol module used by the original EXP-036 experiments — training and experiments cannot drift apart).
3. `chronological_holdout_split` — time-ordered TRAIN/TEST split; no shuffling, no ID overlap.
4. Per stage (`ML_STAGE_NAMES`): `build_stage_frame` → eligible rows → train-period slice → encoding:
   - strings → `__MISSING__` → `get_dummies(..., dummy_na=True)`;
   - numerics → coerce + median impute (medians from train only).
5. Target: `stage_<stage>_sec` in `log1p` space (`TargetTransformer`).
6. Estimator per `STAGE_KIND_MAP`: staging/prevalidation/pre_load = ExtraTrees, transformation = RandomForest, data_loading = Ridge.
7. Refusals (fail-closed): unknown stage; zero eligible rows; < 5 eligible train rows for an individual-stage request.
8. Evaluation on the frozen holdout → `evaluation` record; bundle persisted as `artifacts/stage-models-<model_id>.pkl`; registry + experiment records appended.

## Serving pipeline (`/predict`)

- Same feature assembly, **strictly-prior** history features only.
- Five stage predictions in seconds → **served total = exact sum** (EXP-038); approvals excluded; `floor_applied = max(ML stage predictions)` only if it exceeds the sum (serving-only physical floor; containment invariant: floor ≤ actual held 490/490 in EXP-033 checks).
- Reference evidence (not serving): learned stage-aware total + direct total (`stage_aware_reference_seconds`, `direct_reference_seconds` byte-stable diagnostics).
- Uncertainty: conformal calibration on log1p residuals (85-residual window; known heavy-tail limitation disclosed when width-factor ≥ 12× → LOW reliability).
- Response identity: promoted model id, `configuration_provenance` (user_supplied vs definition_derived per input), floor fields, risk classification (LOW_CONFIDENCE / OOD / INSUFFICIENT_HISTORY → no fabricated evidence).

## Monitoring / drift

- Readiness: eligible new actuals count vs threshold (currently 30) + performance degradation = matched-outcome MedAE vs registry baseline per stage, only after enough matched outcomes ("needs more matched outcomes" is the honest state until then).
- Population/feature drift endpoints: `/drift/population`, `/drift/features`, `/drift/performance`; aggregated human-readable verdict on `/drift/overview` (UI: Monitoring page).
- Recommendation mapping (UI, 2026-10-08): NONE → Continue monitoring; STAGES → one button per degraded stage; ALL → "Retrain all 5 stage models".

## Registry API

`GET /models` → `{models, promoted_model_id, lifecycle_states}`; `GET /models/lifecycle`; `GET /monitoring/stage-lineage`; `GET /retraining/serving-status`; compatibility check per model id. Frontend types in `frontend/src/api.ts` mirror these (single-or-array `registered`/`evaluation` for multi-stage candidates).

## Known limitations (disclosed, not hidden)

- 85-residual conformal window caps calibration granularity (heavy tails).
- `pre_load` train n = 13 → floor scope excludes it; its prediction remains a visible, disclosed estimate.
- Uninstrumented gap mass (~56% of median total) means stage windows do not cover the whole run — the served total is a product decision (EXP-038), with the accuracy trade-off recorded (EXP-037).
