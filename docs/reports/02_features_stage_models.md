# Report 2 — Features and the Five Stage Models

**Status: implemented and verified** (backend suite green; serving path live-verified).

## The 11-input pre-execution contract

The user supplies only pre-execution context (no outcome data — that would leak):

| Input | Notes |
| --- | --- |
| `loader_name` | which loader executes |
| `sprint` | sprint identifier |
| `dataset` | dataset being loaded |
| `connection` | source connection (e.g. `MDM_DM_TEST`) |
| Calling type / Type | workflow style, component context |
| Component definitions | prevalidation / transformation components: counts, types, order (derived canonically from definitions; user fields are audit overrides) |
| Approval estimates | 2 contextual approval estimates — **context only, excluded from the ML total and the floor** |

Applicability: the *applicable* configuration is **derived** from canonical component definitions; user fields are audit overrides (AD-35 evidence pack: override-independence probes A–F, 0 failures, regression-pinned in `tests/test_input_applicability_ad35.py`).

## The five ML stages

`backend/src/dltime/stage_targets.py` → `ML_STAGE_NAMES`:

| Stage | Estimator family | Target |
| --- | --- | --- |
| staging | ExtraTrees | `stage_staging_sec` |
| prevalidation | ExtraTrees | `stage_prevalidation_sec` |
| transformation | RandomForest | `stage_transformation_sec` |
| pre_load | ExtraTrees | `stage_pre_load_sec` |
| data_loading | Ridge | `stage_data_loading_sec` |

- Same families for production and candidates (`STAGE_KIND_MAP` in `retraining.py`) so a candidate is apples-to-apples with serving.
- Targets are seconds, modeled in `log1p` space (`TargetTransformer`), predicted back-transformed to seconds.
- Each stage row carries `stage_eligible` (from runtime stage windows); stages with no data **refuse** rather than fabricate.
- **Served total = staging + prevalidation + transformation + pre_load + data_loading** (EXP-038 product decision; deterministic from the five displayed values; additivity 0.0). A learned stage-aware total and a direct total are retained as labelled **reference evidence** only.
- **Physical floor**: `max(ML stage predictions)`, serving-only (never in training/eval/calibration), approvals excluded; disclosed as `physical_floor_seconds` + `floor_applied` in the payload.

## Feature assembly

`stage_models.build_stage_frame(stage, bundle)` builds per-stage frames from the loaded workbook bundle: pre-execution context, per-stage strictly-prior history features (`stage_hist_*`), and component-derived counts. String columns → `__MISSING__` → one-hot (`dummy_na=True`); numeric → median-imputed. History features are **strictly prior** (poison test proves a future row cannot move a probe's history features).
