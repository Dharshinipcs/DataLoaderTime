# Project File Guide — what each file is for

_Last updated 2026-10-04 after the final structure cleanup. Read this when you want to find
where a concept lives. Documentation itself is in `docs/` (see bottom)._

## Root

| Path | Meaning |
|---|---|
| `frontend/` | React + Vite + TypeScript UI (the whole user-facing system). |
| `backend/` | FastAPI service: prediction API, ML models, stores, tests. |
| `train/data/` | The ORIGINAL corrected export (never modified by any code). |
| `train/data_filled/` | Training copy: real rows + deterministic donor-based synthetic coverage rows (id >= 899001). Built once by `backend/scripts/build_filled_dataset.py`. |
| `docs/` | All documentation and reports. |

## Frontend (`frontend/src/`)

| File | Meaning |
|---|---|
| `App.tsx` | Shell: sidebar navigation, page routing, notification bell. |
| `api.ts` | Typed REST client for every backend endpoint (single BASE). |
| `lib/format.ts` | `formatSeconds` and friends (seconds-first display). |
| `pages/PredictionStudio.tsx` | Pre-Execution Features page: 4 keys, component rows (Display Name / Calling Type / Type / Order), live counts, Predict, result card, insights. |
| `pages/LoaderIntelligence.tsx` | Loader directory + filters + 10-per-view paging + simple inline intelligence panel (trend, counts, typical/recent/range). |
| `pages/PredictionHistory.tsx` | Audit trail: filters top, graph + accuracy cards left, 10-per-view predictions list right, click → detail drawer. |
| `pages/DriftRetraining.tsx` | Compact two-column: LEFT drift evidence, RIGHT retraining decision + model versions + combined history timeline. System scope. |
| `components/ActualOutcomeForm.tsx` | Post-execution form: fixed `YYYY-MM-DD HH:MM:SS` timestamps, seconds-only inputs, derived-duration helper. |
| `components/ui.tsx` | Card / Badge / Metric / PageHeader primitives. |
| `components/NotificationBell.tsx` | System-level notification center (AD-33). |

## Backend core (`backend/src/dltime/`)

| File | Meaning |
|---|---|
| `api.py` | FastAPI app: `/predict`, `/feedback/actual`, `/drift/overview`, `/models/*`, `/retraining/*`, `/loaders`, `/events`. |
| `inference.py` | Prediction orchestration: identity resolution, three loader cases, context-row overrides (`_apply_component_aggregates`), payload assembly. |
| `stage_v2.py` | The 5 stage ML models: family comparison, selection, serving bundle (artifact), loader trust gate, sum-of-available-stages total. |
| `stage_intel.py` | Per-stage intelligence entries (prediction + approval estimates with evidence tiers). |
| `approval_estimator.py` | The 2 approval analytics: hierarchical historical MEDIAN from the loader's OWN approval history (1 = itself, 2+ = median), tier evidence. |
| `sum_total_serving.py` | Total policy + human-readable explanation sentences. |
| `definition_features.py` | `def_*` pre-execution features from the definition files: per-section (prevalidation/transformation) execution-order stats, per-type counts, counts, dominant calling type. |
| `feature_registry.py` | Registry of every feature (source, group, leakage category). Leakage validator consumes it. |
| `feature_assembly.py` | Training-frame assembly + leakage validation + log1p target transformer. |
| `historical.py` | `hist_*` strictly-prior duration aggregates (as-of, no leakage). |
| `stage_models.py` | Per-stage training frames (`build_stage_frame`). |
| `stage_targets.py` | Stage target derivation chain (minutes×60 → span → missing). |
| `resolution.py` | Dataset+Sprint+Loader identity resolution against registered definitions. |
| `uncertainty.py` | Calibrated prediction interval (log-space quantile models). |
| `risk.py` | Transient risk toasts (interval-width signal; reliability label removed 2026-10-03). |
| `explain.py` / `explain_evidence.py` | "Why this ETA?" evidence sentences + what-ifs. |
| `feedback.py` | Actual-outcome validation, linkage, TOTAL + PER-MODEL (7-way) error computation, feedback records. |
| `store.py` | Append-only JSONL stores under `backend/store/`. |
| `monitoring.py` (in api drift section) | Drift overview: PSI/KS data drift, prediction drift, model + stage health, retraining readiness. |
| `retraining.py` | Candidate training/evaluation (never auto-promotes). |
| `model_lab.py` | Model registry: register/promote/rollback + lifecycle log (human-gated). |
| `notifications.py` | Alert types + severities (drift, accuracy, OOD). |
| `sources.py` | Loads the raw XLSX sources (the union point for retraining data pickup). |

## Backend artifacts & stores

| Path | Meaning |
|---|---|
| `backend/artifacts/stage-models-v2.pkl` | The serving bundle: 5 fitted stage models + metadata. Rebuilt by `python backend/src/dltime/stage_v2.py`. |
| `backend/store/prediction_events.jsonl` | Every prediction attempt (full pre-execution input + state). |
| `backend/store/actual_outcomes.jsonl` | Every saved post-execution record (fixed-format timestamps, seconds + minutes mirror). |
| `backend/store/feedback_records.jsonl` | Linked prediction↔actual records with total + per-model errors. |
| `backend/store/retraining_decisions.jsonl` | Recorded human retraining decisions. |
| `backend/store/notifications.jsonl` | Notification-center events. |
| `backend/store/model_registry.jsonl` + `lifecycle_log.jsonl` (in store dir when created) | Model versions + promotions/rollbacks. |

## Backend scripts (`backend/scripts/`)

| File | Meaning |
|---|---|
| `build_filled_dataset.py` | One-time synthetic-coverage builder for `train/data_filled/`. |
| `build_stage_v2_reports.py` | Regenerates `docs/STAGE_V2_TECHNICAL_REPORT.md` from the artifact + comparison JSON. |
| `exp032…exp036*, final_sum_total_validation, reanchor_*` | Historical experiment/evidence scripts kept because `retraining.py`, `sum_total_calibration.py` and `api_lab.py` load the EXP-036 protocol from `exp036_clean_holdout.py`. |

## Tests

| Path | Meaning |
|---|---|
| `backend/tests/` | Pytest suites (run per-file; the full run exceeds one command timeout). |
| `frontend/src/**/*.test.tsx` | Vitest + Testing-Library UI suites (56 tests). |

## Documentation (`docs/`)

| File | Meaning |
|---|---|
| `STAGE_V2_TECHNICAL_REPORT.md` | THE technical report: ML models per stage, inputs, hyperparameter grids, what was compared, selection metric, approvals analytics, save schema, drift/retraining mechanics. |
| `stage_v2_comparison.json` | Raw per-stage comparison evidence (regenerated with the artifact). |
| `RETRAINING_AND_POST_EXECUTION_REPORT.md` | Save flow: where records go, exact schemas, per-model error method, how retraining picks up new actuals. |
| `ML_STAGE_PATTERN_REPORT.md` | What each stage model actually learned (feature importances read from the artifact). |
| `MANUAL_TEST_PLAN.md` | The end-to-end user test plan — follow it top to bottom, including the retraining/promotion/rollback drill. |
| `FILE_GUIDE.md` | This file. |
