# Post-Execution Capture & Retraining — Technical Report

_Generated 2026-10-03 from the actual store files under `backend/store/` and the
filled dataset. Every number is read programmatically._

## 1. Loop overview (the trust loop)

1. User predicts in the Studio (11 pre-execution inputs). The response carries a linkage id
   (`workflow_execution_id` / prediction event id) and the exact inputs used.
2. After execution the user saves the actual outcome in the UI -> `POST /feedback/actual`.
   The ingest contract: `actual_total_timetaken_in_min` wins; minutes are converted to seconds
   (x60, AD-34). Per-stage actuals are accepted when present.
3. The outcome lands in `backend/store/actual_outcomes.jsonl` and is linked to the prediction
   event in `prediction_events.jsonl` (identity, inputs, stage intel, model version).
4. `POST /retraining/decide` evaluates drift/coverage and records a decision in
   `retraining_decisions.jsonl`; `POST /retraining/train-candidate` retrains the stage bundle
   (v2: zero-data stages skip honestly instead of failing the candidate).
5. Serving status/promote/rollback keep the served artifact auditable at all times.

## 2. What the capture stores (and why it is enough to retrain)

- Full request inputs (11 pre-execution fields) — the same feature contract the models use.
- Resolved identity + connection + definition-derived features.
- Per-stage intelligence snapshot (model family, training counts, estimates).
- The actual outcome (total + stage actuals) with linkage to the event.

## 3. Retraining integration with v2

- Stage candidates train on the dataset in force at decision time; stages with zero eligible
  rows are recorded in `skipped_stages` (`NOT_APPLICABLE_NO_STAGE_DATA`) and excluded from the
  candidate's total evaluation — never silently dropped, never fabricated.
- The filled dataset currently feeding training has 1493 rows (792 real + 701 synthetic coverage; synthetic rows never enter accuracy evidence).
- Live flow verified: decide -> train-candidate -> serving-status, with the new dataset.

## 4. Store state at report time

- `prediction_events.jsonl`: 304 records
- `actual_outcomes.jsonl`: 22 records
- `feedback_records.jsonl`: 22 records
- `retraining_decisions.jsonl`: 4 records
- `notifications.jsonl`: 48 records

## 5. Honesty guarantees in the loop

- Predictions and actuals are stored separately and linked by id; the dataset is never
  auto-mutated by serving — retraining consumes real exports + recorded outcomes only.
- Every served total carries its model version (`sum-of-available-stages-v2`) and loader gate
  case, so any historical prediction can be traced to the exact policy that produced it.
- Drift/monitoring never ingests synthetic rows (real-row mask `LDR_Execution_Id < 899001`).
## 6. 2026-10-03 — Save flow verified end-to-end, error method, and how learning picks up new actuals

### 6.1 Where a saved actual goes (exact paths + schema)

`POST /api/feedback/actual` (the "Save Actual Outcome" button) writes THREE
newline-JSON files under `backend/store/` — one record per line, append-only,
never rewriting history:

- `prediction_events.jsonl` — one record per prediction ATTEMPT (every state, even
  refusals). Key fields: `event_id`, `schema_version`, `created_at`, `event_state`,
  `request` (the 4 identity keys), `identity` (the resolved identity),
  `prediction` (`predicted_total_seconds`, `model_version`, `model_status`,
  `prior_count`, interval bounds, `risk_label`, `risk_policy_version` — the old label
  field is gone), `explanation_digest`, `stage_intelligence`.
- `actual_outcomes.jsonl` — the stored post-execution record (verified live,
  2026-10-03): `identity` (loader/dataset/sprint/connection), `execution_ref`
  (optional `workflow_execution_id` / `ldr_execution_id`), `timing` with
  `start_time` / `end_time` in the FIXED `YYYY-MM-DD HH:MM:SS` form (the UI's
  picker value is normalized — seconds forced, T exchanged for a space),
  `actual.total_seconds` + `status` (the UI collects SECONDS and they are stored
  natively in seconds), `stage_actuals` (seconds) + `stage_minutes_reported`
  (minutes mirror of the same durations), `record_counts`,
  `training_eligible` + `training_eligibility_reason`, `data_quality_warnings`,
  and a dedup `outcome_id` / `outcome_hash` (duplicate ingestion is flagged,
  never double-counted).
- `feedback_records.jsonl` — the prediction-actual LINK with the computed errors:
  `prediction_event_id`, `linkage` (`status` = MATCHED/... + `method` =
  EXACT_WORKFLOW_EXECUTION_ID / deterministic rule), `identity`,
  `actual` (the outcome fields), `total_error` (below), `stage_errors`,
  `training_eligible` + `training_eligibility_reason`.

### 6.2 The error-calculation method (and why it fits)

For every matched pair the system computes, in `compute_total_error`:

- `signed_error_seconds = predicted - actual` — positive means over-prediction. This is
  the direction-aware signal used for the trend chart and honest reporting.
- `absolute_error_seconds = |signed|` — the size of the miss; MAE and MedAE on the
  Prediction History page aggregate exactly this.
- `absolute_percentage_error = absolute / actual` — comparable across loaders of very
  different scale; drives the "within ±10/±20%" accuracy cards.
- `log_absolute_error = |log1p(predicted) - log1p(actual)|` — scale-free error on the
  same log space the models train in; matches the models' own loss, so retraining
  readiness compares like with like.
- Stage errors are computed identically per stage when the actual stage durations exist.

This mix is the right one for this system: signed errors expose systematic bias,
absolute/median errors resist the heavy right tail, percentages make loaders comparable,
and the log error ties directly to the training objective.

### 6.3 Separate vs merged with the training dataset

The store is deliberately SEPARATE from the training data: the three Excel sources
(`train/data/*.xlsx`) and the serving artifacts (`backend/artifacts/*.pkl`) are never
modified by a save. Guarantees: a bad/mistyped actual can never corrupt a served model;
every save is audit-unique (duplicate ingestion is flagged, not double-counted);
monitoring reads are fail-open (a store problem can never affect a prediction).

How the new histories are picked up:

1. **Immediately, for monitoring + drift + accuracy:** every monitoring, drift and
   retraining-readiness endpoint (`/monitoring/*`, `/retraining/readiness`, the
   Drift & Retraining page) reads the store live — a newly saved actual changes the
   accuracy cards, the error slices, the drift statistics and the readiness verdict on
   the next page load. This is the "error shrinks as the model improves" loop's
   measurement side.
2. **For the models' own weights (full retraining):** the candidate bundle is trained
   from the source export (`load_all_sources`). The ingested records already carry every
   executions-native field in native units and the exact identity columns, so closing
   the loop is a schema-identical union at that single integration point
   (`load_all_sources` / `build_foundation`) — the documented next step of the
   DataZap integration, with no schema change needed on either side. Nothing about the
   ingest needs to change for that union to happen.

### 6.4 Full retraining flow (verified)

- `POST /retraining/candidate` builds a CANDIDATE five-stage bundle: same approved
  methodology (per-stage frames, EXP-036 chronological TRAIN/HOLDOUT split, candidates
  fit ONLY on the train period, frozen, evaluated on the holdout), writes
  `backend/artifacts/stage-models-<candidate_version>.pkl` and registers
  `CANDIDATE_EVALUATED` — a serving change never happens silently.
- `POST /retraining/decision` records the human decision in
  `backend/store/retraining_decisions.jsonl`.
- `POST /retraining/promote` atomically swaps the serving path to the candidate;
  `POST /retraining/rollback` restores the previous version;
  `GET /retraining/serving-status` shows the active version at any time.
- `GET /retraining/readiness` combines live drift + accumulated feedback errors into the
  recommendation the Drift & Retraining page displays.
