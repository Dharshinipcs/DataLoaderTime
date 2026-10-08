# Report 1 — SHAP Explainability

**Status: implemented and verified** (backend targeted suite 64/64 including `tests/test_explain_task08.py`; frontend 56/56).

## What the system explains

The served ETA is the **sum of the five operational stage predictions** (staging, prevalidation, transformation, pre_load, data_loading) plus two **approval-based contextual estimates**. The product therefore explains at two levels:

1. **Per-stage attribution** — for each ML stage, which pre-execution inputs pushed that stage's prediction up or down.
2. **Global model behavior** — which features dominate each stage model across the evaluation holdout.

## How it works

- Library: `shap>=0.52.0` (see [pyproject.toml](../../pyproject.toml)).
- Per-stage: `backend/src/dltime/stage_shap_explanation.py` runs `shap.TreeExplainer` on the encoded prediction row for the tree-based stage models (ExtraTrees / RandomForest) and the linear equivalent for the Ridge stage (`data_loading`). The attribution is computed **on the stage model's own output**, so stage SHAP values decompose that stage's seconds exactly.
- Global: `backend/src/dltime/explain_evidence.py` computes global SHAP importances for the **authoritative (promoted) artifact** over holdout rows — never over a retired model.
- Served via:
  - `POST /explain/prediction` — evidence for one prediction (stage-by-stage drivers, in-scope notes, provenance),
  - `GET /explanation/global` — global importances for the promoted artifact.

## UI surface (Prediction Studio)

- "Prediction Insights" disclosure: uncertainty range, confidence wording, per-stage explanation, provenance and model identity.
- Risk notice is compact and honest (LOW_CONFIDENCE / OOD / INSUFFICIENT_HISTORY show **no fabricated ETA evidence** — pinned by tests).
- 2026-10-07 copy cleanup removed hedging artifacts ("so we predict …", "(median of N recorded approvals)") — behavior unchanged, wording only.

## Honesty guarantees (regression-pinned)

- Approvals are **never** part of the ML total (they are contextual estimates only).
- SHAP is computed on the promoted model actually serving — model identity is carried in the response.
- Fail-closed: if a stage model is missing, no total and no fake attribution are produced.
