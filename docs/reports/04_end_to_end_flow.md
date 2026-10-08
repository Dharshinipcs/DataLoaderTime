# Report 4 — End-to-End Flow

**Status: implemented and live-verified** (predict → outcome → monitoring → retrain → promote loop exercised on the real stack).

## The loop

```
 train/data/*.xlsx  (replacement-friendly; see Report 7)
        │  load_all_sources() — schema-validated
        ▼
 POST /predict  ──►  prediction_events.jsonl  ──►  Prediction Studio result
   (11 pre-execution inputs)                        (ETA = Σ 5 stages + 2 approval
        │                                            estimates; insights disclosure)
        │  later, real run finishes
        ▼
 POST /feedback/actual  ──►  actual_outcomes.jsonl + feedback_records.jsonl
   (identity + start/end or minutes)     eligibility gate (leakage rules)
        │                                 match by exact workflow_execution_id
        ▼
 GET /drift/overview ──► monitoring signals:
   • new outcomes vs threshold (e.g. 23 of 30 needed)
   • performance degradation (matched MedAE vs registry baseline, per stage)
   • data drift flags
        ▼
 Recommendation card (system advice, never auto-acted):
   • Continue monitoring → no retrain action offered
   • one degraded stage → "Retrain <Stage>" button for that stage
   • several → per-stage buttons for each degraded stage
   • all five → "Retrain all 5 stage models"
        ▼
 POST /retraining/train-candidate ──► candidate REGISTERED + CANDIDATE_EVALUATED
        ▼
 Model Versions (right column, newest first, scrollable) shows the candidate
 with "Use for promote" ──► prefills Human Decision (left column)
        ▼
 POST /models/promote  (human only) ──► serving_switch.json updated,
 previous promoted model RETIRED, PROMOTION_APPROVED event recorded
        ▼
 next /predict serves the new bundle ──► cycle repeats
```

## Persistence at each hop

- Every prediction → `backend/store/prediction_events.jsonl`
- Every actual → `actual_outcomes.jsonl` (+ `feedback_records.jsonl`)
- Every human decision → `retraining_decisions.jsonl`
- Registry changes → `backend/artifacts/registry/models.jsonl` + `lifecycle_events.jsonl`
- Serving switch → `serving_switch.json`

## Golden smoke case

```json
{"loader_name":"MigrationSummarySubTableLoaderDataset","sprint":"TEST",
 "dataset":"Migration_Summary_Sub_Fact_Table_LDR_Datase_new","connection":"MDM_DM_TEST"}
```

Fail-closed behavior: missing stage data ⇒ no total; ambiguous outcome reference ⇒ `AMBIGUOUS`, never double-matched; duplicate prediction references are rejected.
