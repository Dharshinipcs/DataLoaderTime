# Report 7 — Data Replacement and Retraining Behavior

**Status: VERIFIED (implemented; no code change was needed).** Replacing the training data is designed to be a drop-in file swap.

## The contract

1. Put replacement workbooks in `train/data/` **with the same file names and the same schema** (column names required by `SOURCE_SPECS` in `backend/src/dltime/sources.py`).
   - Optional: point `DLTIME_DATA_DIR` at a different folder instead of touching `train/data/`.
2. Nothing else to do at prediction time: `/predict` loads through `load_all_sources()` on demand, so the next prediction already reflects the replaced files.
3. For retraining: `POST /retraining/train-candidate` calls `load_all_sources()` itself — the candidate is trained on whatever is currently in the data folder. No code change, no import step, no cache to clear.

## Why it is safe

- **Schema validation at load**: `SOURCE_SPECS` validates required columns per source file; a wrong/missing column fails loudly at load time instead of silently training on bad frames.
- **Eligibility re-derives from the new data**: which rows are eligible for each stage (`stage_eligible`), statuses, and chronological split boundaries all recompute from the workbook contents (eligibility rules in `eligibility.py`, split in `modeling.chronological_holdout_split`).
- **Leakage rules are structural**: predict-then-reveal ordering, strictly-prior history features and train-only fitting are enforced by code paths, independent of which rows are in the files.
- **Human gate unchanged**: a retrain on new data still only produces a candidate; serving changes only after `POST /models/promote`.

## Verification performed (2026-10-08)

- `load_all_sources()` executed against the current `train/data` workbooks: **792 execution rows loaded in ~0.6 s**, all `SOURCE_SPECS` validations passing.
- Confirmed by code inspection + targeted greps that the training path (`retraining.py` → `load_all_sources`) and the serving path read the same `DATA_DIR` location; no hardcoded row counts, no cached intermediate frames.

## Failure modes (intended)

| Situation | Behavior |
| --- | --- |
| Column missing / renamed | load-time validation error naming the source spec — fix the file |
| Same schema, fewer/more rows | fine; thresholds (e.g. 30 matched outcomes) and min-train-row guards apply to whatever loads |
| Stage has no eligible rows after replacement | candidate training **refuses** for that stage (`LabError`), never silently skips |
| Empty stage requested individually | explicit refusal: "stage has no stage data on the current source" |
