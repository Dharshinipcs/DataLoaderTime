# dltime Frontend — Prediction Studio

React 19 + TypeScript + Vite. Single-page app; no router needed (locked rule: no unnecessary infrastructure).

## Commands (Windows: use `npm.cmd` if `npm.ps1` is blocked)

```bash
cd frontend
npm install           # dependencies (React, Vite, Vitest, Testing Library)
npm run dev           # Vite dev server, http://localhost:5173 (proxies /api → 127.0.0.1:8000)
npm test              # Vitest in run mode
npm run build         # tsc -b && vite build (full typecheck + production bundle)
```

## Contract (Task 04, authoritative — not duplicated or replaced)

`POST /predict` with `{ loader_name, sprint, dataset, connection?, workflow_execution_id?, planned_start_time? }`.
Response states: `PREDICTED | INSUFFICIENT_HISTORY | UNRESOLVED_IDENTITY | AMBIGUOUS_IDENTITY | INVALID_INPUT`.

The UI exposes **only legitimate pre-execution identity fields**. Historical statistics, previous duration,
time-since-previous, similarity, encoded features, SHAP, residuals, actual duration, end time, status,
stage durations and error counts are **never** manual inputs — the backend derives them.

Connection is validation-only: disagreement with backend resolution returns `INVALID_INPUT`; the UI shows
the backend's message and never guesses or overrides the resolved connection.
