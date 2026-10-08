# Report 5 — Technical Architecture

**Status: implemented and verified.**

## Shape

Two-process local app, no external services:

| Piece | Tech | Where | Run |
| --- | --- | --- | --- |
| Backend | Python + FastAPI (`uv` package mgmt) | [backend/src/dltime](../../backend/src/dltime) | `cd backend && uv run uvicorn dltime.api:app --host 127.0.0.1 --port 8000` (no `--reload`; restart after backend edits) |
| Frontend | React 18 + TypeScript + Vite | [frontend/src](../../frontend/src) | `cd frontend && npm run dev` (port 5173) |
| Data | Excel workbooks | [train/data](../../train/data) | read at request/training time |

## Backend module map (main responsibilities)

- `api.py` / `api_lab.py` — HTTP surface (see Report 4 for endpoint list): predict, explain, feedback, monitoring, drift, retraining, models/lifecycle, loaders, notifications, laboratory experiments.
- `sources.py` + `config.py` — workbook loading, `SOURCE_SPECS` schema validation, `DATA_DIR` (env-overridable).
- `feature_assembly.py`, `stage_models.py`, `stage_targets.py`, `stage_serving.py` — features, stage training frames, serving-time stage prediction + floor.
- `sum_total_serving.py`, `stage_aware_serving.py` — served total (Σ stages) vs reference evidence path.
- `retraining.py` — candidate training protocol, registry writes, lifecycle transitions.
- `registry.py` / `model_lab.py` — JSONL registry, experiments.
- `eligibility.py`, `leakage.py`, `schemas.py` — outcome eligibility and locked leakage rules.
- `explain.py`, `explain_evidence.py`, `stage_shap_explanation.py` — SHAP (Report 1).
- `notifications.py` — alert center policies (persistent vs transient risk toasts).

## Frontend

- 4 pages under `App.tsx` hash navigation: Pre-Execution Features (Prediction Studio), Loader Intelligence, Prediction History, Drift & Retraining.
- `api.ts` — typed client; `TrainCandidateResponse` supports single object **or** arrays (multi-stage candidates).
- Design system in `ui.tsx` (Card, PageHeader, badges).
- Tests: Vitest + Testing Library, 56 tests (5 files). Type-check `tsc -p tsconfig.app.json`; production build via `npm run build`.

## Verification commands

- Backend targeted: `cd backend && uv run pytest tests/test_retraining_workflow.py tests/test_explain_task08.py tests/test_stage_v2.py tests/test_drift_overview.py -q` (64 passing at time of writing; full suite larger).
- Frontend: `npm test -- --run`, `npx tsc -p tsconfig.app.json --noEmit`, `npm run build`.

## Conventions / constraints

- No `--reload` on uvicorn (Windows/Git-Bash workflow); kill the `netstat` PID and restart after backend edits.
- Frontend dev server listens on `[::1]:5173`.
- `python` alias is broken in this workspace — use `uv run python`.
