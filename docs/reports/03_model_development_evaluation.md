# Report 3 — Model Development and Evaluation

**Status: implemented; protocol and results below are from recorded experiment evidence** (docs/EXPERIMENT_LOG.md, docs/PROJECT_STATE.md). Live numbers change as new models are promoted — check the Drift & Retraining page for current serving state.

## Development history (short form)

- EXP-033 settled stage **order** (20 runtime orders; timestamps decide, never config order) and **coverage** (union of stage windows ≈ 44% of total duration → simple summation of stages as a *model of reality* is invalid, but a **sum as the served product quantity** was later mandated by the product owner).
- EXP-036 established the **clean leakage-safe holdout protocol** used by every candidate evaluation since.
- EXP-037 evaluated "simple sum of 5 stages" vs the learned stage-aware total on that holdout: sum MedAE 10.93s vs 1.82s — evidence **against** the sum on accuracy.
- EXP-038 (product decision): the served/displayed total is nevertheless the **exact sum of the five operational stage predictions** (explainability and simplicity won); the stage-aware artifact stays on disk as reference evidence and rollback path. Uncertainty recalibrated for the sum (`unc-log1p-conformal-v1-sum-of-five-stages.json`, 85 residuals, Q90 2.92, under-prediction share disclosed).

## The evaluation protocol (EXP-036, still authoritative)

1. Chronological split: **TRAIN 256 rows [2025-10-10 → 2026-07-15] / HOLDOUT 85 rows [→ 2026-07-30]**; overlap 0, IDs disjoint.
2. Stage models for evaluation are fitted **only on train-period eligible rows**; frozen; holdout predicted **before** actuals are revealed (predict-then-reveal).
3. Metrics reported on the holdout: MAE, MedAE, ±10% / ±20%, within-2× band, sane 0.8–1.25× band.
4. Known-loader vs new-loader vs low-history slices reported **honestly** (small-n slices are labelled as such).

## Candidate lifecycle (how a new model becomes serving)

1. `POST /retraining/train-candidate` (optionally `stages: [...]` for one or a subset) → trains on the current data via the EXP-036 protocol → registers a model (`REGISTERED`) and evaluation (`CANDIDATE_EVALUATED`) in the registry.
2. The new candidate appears **immediately at the top of Model Versions** on the Drift & Retraining page (newest first, scrollable), with a "Candidate awaiting decision" summary line and a **"Use for promote"** action on its row.
3. Human promotes via `POST /models/promote` (allowed only from `CANDIDATE_EVALUATED`; retires the previous promoted model; writes a `PROMOTION_APPROVED` lifecycle event) — **automatic promotion does not exist** (`auto_retraining OFF`, `human_gate_required TRUE`).
4. `POST /models/rollback` restores the most recent prior promoted model; `POST /models/{id}/reject` records a rejection.

## Current state at time of writing

- Registry: 13 models, all on the `full_stage_bundle_<timestamp>` / stage-aware naming; serving switch active on a promoted `full_stage_bundle_*` bundle (one rollback recorded during verification).
- Stage lineage (`stage_performance_lineage.jsonl`) ties each serving stage to its model, holdout MedAE and post-promotion live MedAE.
