# Report 6 — Database and Storage

**Status: implemented and verified.** There is **no database server** — all persistence is append-only JSONL files plus model artifacts, which keeps the app portable and diff-able.

## Stores

### Operational store — `backend/store/`

| File | Contents |
| --- | --- |
| `prediction_events.jsonl` | every `/predict` response (inputs, ETA, stage values, floor/provenance, model identity) |
| `actual_outcomes.jsonl` | real outcomes posted via `/feedback/actual`, with eligibility classification |
| `feedback_records.jsonl` | linkage records (exact `workflow_execution_id` match, MATCHED / UNMATCHED / AMBIGUOUS) |
| `notifications.jsonl` | alert-center items (persistent risks, readiness, promotions) |
| `retraining_decisions.jsonl` | human decisions (START_RETRAINING / CONTINUE_MONITORING) with reasons |

### Model registry — `backend/artifacts/registry/`

| File | Contents |
| --- | --- |
| `models.jsonl` | one record per registered model: `model_id`, `lifecycle_state` (`REGISTERED` → `CANDIDATE_EVALUATED` → `PROMOTED` / `RETIRED` / `REJECTED`), artifact path, stage (for single-stage candidates), created_at |
| `experiments.jsonl` | training/evaluation records produced by `train-candidate` |
| `lifecycle_events.jsonl` | REGISTERED / PROMOTION_APPROVED / RETIRED / REJECTED / rollback events (the "View Timeline" source) |
| `serving_switch.json` | which model id is actually serving + `candidate_override_active` |
| `stage_performance_lineage.jsonl` | per-stage: serving model, holdout MedAE, post-promotion live MedAE on matched outcomes |
| `served_baseline.json` | frozen baseline metrics used by readiness/degradation checks |

### Artifacts

- `backend/artifacts/stage-models-<model_id>.pkl` — the frozen five-stage bundle per model version (e.g. `stage-models-full_stage_bundle_20261007T044154Z.pkl`).
- Historical single-purpose artifacts (`stage-aware-total-*.pkl`, `stage-models-ad32/ad34.pkl`) retained for traceability/rollback; naming migrated 2026-10-07 (0 stale references in registry/lifecycle/experiments/lineage).

### Source data

- `train/data/*.xlsx` workbooks; `DATA_DIR = Path(os.environ.get("DLTIME_DATA_DIR", PROJECT_ROOT/"train"/"data"))` in `backend/src/dltime/config.py` — swap the folder via env var without code changes (see Report 7).

## Invariants

- Registry writes are the **only** path to lifecycle changes; `register()` validates state + unique id + artifact existence; `promote()` only from `CANDIDATE_EVALUATED` and retires the prior promoted model.
- Append-only JSONL: records are never rewritten (a rollback writes new events rather than editing history).
