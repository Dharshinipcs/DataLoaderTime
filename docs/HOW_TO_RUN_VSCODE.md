# HOW TO RUN THE PROJECT IN NORMAL VS CODE

All commands are copy-paste-ready for the VS Code integrated terminal (PowerShell) on this machine. Project root: `C:\Users\dharcons527\Documents\DataLoaderTime`.

## 1. Open the project in VS Code

```powershell
code C:\Users\dharcons527\Documents\DataLoaderTime
```

(Or: VS Code -> File -> Open Folder -> `Documents\DataLoaderTime`.) Then open a terminal in VS Code with `` Ctrl+` `` — it starts at the project root.

## 2. One-time environment setup (already done on this machine; only needed on a fresh checkout)

The Python environment is managed by `uv`:

```powershell
uv sync
```

Frontend dependencies:

```powershell
cd frontend
npm install
cd ..
```

## 3. Start the backend (terminal 1)

Run from the **project root** (not from `backend/`):

```powershell
uv run uvicorn dltime.api:app --host 127.0.0.1 --port 8000
```

- First start takes ~30-60 s (loads the Excel sources, rebuilds the foundation, loads the model artifact).
- Ready when you see `Uvicorn running on http://127.0.0.1:8000`.
- Verify: open http://127.0.0.1:8000/health in a browser -> `{"status":"ok","version":"0.1.0"}`.
- API docs (optional): http://127.0.0.1:8000/docs

## 4. Start the frontend (terminal 2)

Click the `+` in the terminal panel for a second terminal, then:

```powershell
cd frontend
npm run dev
```

- If PowerShell blocks `npm.ps1`, use `npm.cmd run dev` instead.
- Ready when you see `Local: http://localhost:5173/`.

## 5. Open the app in the browser

http://localhost:5173/

Three tabs are in the top navigation:
- **Prediction Studio** — enter Loader Name / Sprint / Dataset / Connection (+ optional Planned Start) -> Predict. All four identity fields are required (Task 10.1).
- **Stage Analytics** — dataset-wide stage statistics.
- **Loader Intelligence** — enter a loader name -> history evidence, context, cold-start explanation.
- **Prediction Monitoring** — prediction attempts vs actual outcomes, operational accuracy, stage accuracy, training-eligibility feedback (Task 11).

The frontend proxies `/api/*` to the backend on port 8000 automatically (configured in `frontend/vite.config.ts`); no extra environment variables are needed.

## 6. Record an actual execution outcome (manual local ingestion, Task 11)

There is no production connector yet, so actual outcomes are recorded manually. The easiest way is Swagger:

1. With the backend running, open http://127.0.0.1:8000/docs
2. Expand **POST /feedback/actual** -> **Try it out** and paste a body like:

```json
{
  "loader_name": "MigrationSummarySubTableLoaderDataset",
  "dataset": "Migration_Summary_Sub_Fact_Table_LDR_Datase_new",
  "sprint": "TEST",
  "connection": "MDM_DM_TEST",
  "workflow_execution_id": "WF-DEMO-001",
  "start_time": "2026-08-15T14:41:43",
  "end_time": "2026-08-15T14:42:03",
  "actual_total_seconds": 20.4,
  "status": "SUCCESS",
  "stage_actuals": {"prevalidation": 0.8, "transformation": 1.2, "pre_load": 6.1, "data_loading": 5.3, "data_loading_1": 7.0}
}
```

3. **Execute**. The response tells you the linkage result (`MATCHED` / `UNMATCHED` / `AMBIGUOUS`) and, when matched, the total and per-stage errors plus training eligibility.

How linkage works: put the **same `workflow_execution_id` into the optional Workflow Execution Id field in Prediction Studio** when you make the prediction (or send it as `workflow_execution_id` in `POST /predict`). Later, submitting an actual outcome with the same reference links the pair — this is the only defensible linkage (identity+time guessing is intentionally NOT supported: 57% of historical executions share their identity and day with another execution).

PowerShell equivalent (no Swagger):

```powershell
$body = @'
{"loader_name":"MigrationSummarySubTableLoaderDataset","dataset":"Migration_Summary_Sub_Fact_Table_LDR_Datase_new","sprint":"TEST","connection":"MDM_DM_TEST","workflow_execution_id":"WF-DEMO-001","start_time":"2026-08-15T14:41:43","end_time":"2026-08-15T14:42:03","actual_total_seconds":20.4,"status":"SUCCESS"}
'@
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/feedback/actual -ContentType "application/json" -Body $body
```

Behavior notes:
- Re-submitting the IDENTICAL outcome is idempotent (`duplicate: true`, no new record).
- Submitting DIFFERENT content for the same execution reference is rejected with HTTP 409 and nothing is overwritten.
- Records are append-only under `backend/store/` (prediction_events / actual_outcomes / feedback_records, JSONL). To start monitoring over fresh, stop the backend and delete those three files — they are local derived data, never source data.

## 7. Stop the backend / frontend

In each terminal, press `Ctrl+C` (may need two presses).

If a process stays bound to a port (e.g. after a crash), find and kill it:

```powershell
netstat -ano | findstr ":8000" | findstr "LISTENING"
taskkill /F /PID <pid-from-above>
```

Same for the frontend with `:5173`.

## 8. Run the tests (optional, from the project root)

```powershell
# Backend (full suite takes ~12-15 min; verified in segments during development)
uv run pytest -q

# Frontend tests
cd frontend
npm.cmd test -- --run

# Frontend production build check
npm.cmd run build
cd ..
```

## Quick troubleshooting

| Symptom | Cause / fix |
|---|---|
| Frontend shows network errors on Predict | Backend not running — start it (step 3) and confirm `/health`. |
| `npm.ps1 cannot be loaded` | Use `npm.cmd` instead of `npm`. |
| First Predict is slow (~20-45 s) | Cold service build (Excel load + model + calibration). Later requests are fast. |
| `uv: command not found` | `uv 0.11.21` must be installed; `pip install uv` then `uv sync`. |
| Port 8000 already in use | A stale backend is running — kill it per step 7, then restart. |
| Monitoring numbers look stale / you want a clean slate | Stop the backend and delete `backend/store/*.jsonl` (local derived data only). |
| `/feedback/actual` returns 409 | An outcome with the same execution reference but different content already exists; the store is append-only by design. |
