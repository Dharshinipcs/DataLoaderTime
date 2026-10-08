# Report 8 — DataZap Integration Reference (DESIGN ONLY)

> ⚠️ **Status: design reference only — nothing in this report is implemented.** No DataZap code, webhooks, or credentials exist in this repository today.

## Goal

Let DataZap (the team's data-delivery automation) refresh this app's training data and trigger the existing human-gated retraining loop, without weakening any safety property.

## Proposed integration points (all already exist as stable seams)

| Seam | Mechanism today | DataZap role |
| --- | --- | --- |
| Data delivery | Replace `train/data/*.xlsx` with **same file names + same schema** (Report 7), or set `DLTIME_DATA_DIR` | DataZap writes the workbooks into the data folder on schedule/event |
| Load validation | `SOURCE_SPECS` schema validation at load time | acts as the automatic acceptance test for each delivery |
| Health check | `GET /health` | DataZap post-delivery probe |
| Retrain trigger | `POST /retraining/train-candidate` | optional trigger after a successful delivery |
| Outcome ingestion | `POST /feedback/actual` | could deliver actual outcomes if they exist in DataZap's systems |
| Promotion | **Human only** (`POST /models/promote`, UI gate) | deliberately NOT automated |

## Suggested sequence (future implementation)

1. DataZap writes replacement workbooks atomically (write temp, rename) to `train/data/`.
2. DataZap calls `GET /health` and a cheap load-validation call; on schema failure it alerts and **keeps the previous files** (atomic rename makes this automatic).
3. Optionally DataZap posts `/retraining/train-candidate`; the candidate lands in Model Versions for **human review** (never auto-promoted).
4. Everything else (drift, recommendation, promote/rollback) proceeds exactly as in Report 4.

## Explicit non-goals (by design)

- No auto-promotion, no credential storage in the repo, no write access to the registry from DataZap.
- If DataZap outages occur, the app keeps running on the last valid dataset — the design degrades to manual replacement.

## What would need to be built (if pursued)

- A small authenticated trigger endpoint or CLI wrapper for step 3 (optional).
- Documented DataZap job definitions for step 1–2.
- Nothing in the ML or lifecycle code paths.
