# MANUAL TEST PLAN — full system, follow top to bottom

_2026-10-04. Every value below is live-verified against the real data. Start the servers first:
backend `:8000` (`cd backend && ../.venv/Scripts/python -m uvicorn dltime.api:create_app --factory --host 127.0.0.1 --port 8000`), frontend `:5173` (`cd frontend && npm run dev`). The first Predict takes ~1–2 min (model warm-up)._

---

## PART A — Pre-Execution Features (Prediction Studio)

### A1. Reference prediction (the golden case)
1. Open **Pre-Execution Features**. Enter exactly:
   - Sprint: `TEST` · Dataset Name: `Migration_Summary_Sub_Fact_Table_LDR_Datase_new` · Loader Name: `MigrationSummarySubTableLoaderDataset` · Connection: `MDM_DM_TEST`
2. Click **Predict** (the first predict after a server start warms up for ~1–2 min).
3. MUST see: total **11.465 s** (shown as `11.47 seconds`, ≈ 0.19 min), a **Stage Prediction** section with EXACTLY the 3 applicable ML stages — Staging **0.600 s** · Pre-load **10.388 s** · Data loading **0.477 s** — and NO approval rows (this loader has no own approval history; the GLOBAL broad-context estimates are correctly hidden, never summed).
4. Open **ⓘ Prediction Insights** → Prediction range (90%) ≈ **8.43 s — 14.5 s** (conformal, 90% level); "Why this ETA?" shows only 2–3 plain lines (plain-explain-v1 + the unentered-components note) — no evidence tables, no technical collapsible, no risk label inline.
5. The header must read **"Pre-Execution Features"** — the old "Prediction Studio" title is gone. The page has NO explanatory footer sentences and NO risk pop-up/toast (2026-10-05: fully removed).

### A2. Component rows — Type field + live counts + per-section order
1. Under **Prevalidation / Transformation Components**, add rows:
   - Row 1: Display Name `Fee Check`, Calling Type `PREVALIDATION`, Type `dataobject`, Order `1`
   - Row 2: Display Name `Fee Transform`, Calling Type `TRANSFORMATION`, Type `table`, Order `1`
2. MUST see the live counts line update **before** Predict: `Components detected: 1 prevalidation · 1 transformation …`.
3. Click **Predict**. The total MOVES vs A1 (models consume the real component features — verified live: adding components shifted the total massively). Every row field (incl. **Type** dropdown: crossreference / dataobject / dataextract / dataset / query / table) is sent to the backend.
4. **Order meaning (2026-10-04):** execution order is PER SECTION — prevalidation rows are ordered 1..n among themselves, transformation rows 1..m among themselves. Two independent sequences. The models now carry per-section order features (max/mean/std) learned from history.
5. Inspect the request (DevTools → Network → predict): `preval_trans_components` carries `display_name`, `calling_type`, `type`, `execution_order` for each row.

### A3. The three loader cases
1. **Completely new loader (no definition, no history):** Loader Name `My_Brand_New_Loader_XYZ`, Sprint `TEST`, Dataset `My_Dataset_XYZ`, Connection `MDM_DM_TEST` → Predict.
   MUST see the short refusal: **"New loader 'My_Brand_New_Loader_XYZ' — it has no prior executions yet, so there is no history to predict from."** + "0 (required minimum: 3)". No total, no fabricated number.
2. **Registered but never executed:** Loader `2210_TEST_ALL_CONFIG`, Sprint `SUPPORT`, Dataset `A_TABLE`, Connection `test1` → same short "New loader" refusal (this loader IS registered, 0 priors).
3. **Insufficient history:** Loader `ACS_LDR_DATE_01`, Sprint `DEV`, Dataset `ACS_LDR_DATE_01`, Connection `PFM_Meta` →
   MUST see: **"Insufficient history to predict: loader 'ACS_LDR_DATE_01' has only 2 prior execution(s); the trust threshold is 3."**
4. In ALL three refusals: click **Enter Actual Post-Execution Values** → the form opens with an EMPTY total (nothing prefabricated). Saving here builds the loader's history (Part B).

---

## PART B — Post-execution save (builds history + trains the models)

### B1. Save an actual for the reference loader
1. Run A1 (reference Predict), then click **Enter Actual Post-Execution Values**.
2. Enter:
   - Start: `2026-08-10 10:00:00`, End: `2026-08-10 10:00:18` (fixed `YYYY-MM-DD HH:MM:SS` pickers with seconds)
   - Status: `SUCCEED`
   - Total time taken (sec): `18` · Staging (sec): `3` (any other stage fields optional)
3. MUST see the kept helper: **"Duration derived from Start/End: 0.30 min (18 seconds)…"** — no need to compute anything.
4. Click **Save actual outcome** → success message with linkage status.
5. **Where it is stored (check on disk):** `backend/store/actual_outcomes.jsonl` (last line). Verify the record carries:
   - `timing.start_time: "2026-08-10 10:00:00"` (exact fixed format — the normalizer restores dropped `:00` seconds automatically)
   - `actual.total_seconds: 18.0` and `stage_actuals.staging: 3.0` (seconds-native; `stage_minutes_reported.staging: 0.05` minutes mirror)
   - `training_eligible: true`
6. **Errors are computed for ALL 7 predictors** (5 ML stages + 2 approval analytics): when the save links to a prediction, `backend/store/feedback_records.jsonl` (last line) carries `total_error` (signed / absolute / % / log-space / within-10-20-30% / within-2x) AND `stage_errors` — one block per predicted stage **and** per approval estimate where the actual exists: `signed_error_seconds`, `absolute_error_seconds`, `absolute_percentage_error`, `log_absolute_error`, `within_20_percent`, `within_2x`. This per-model error block is what monitoring uses to flag a degrading stage and what retraining improves.

### B2. Three saves = a loader starts predicting
1. Repeat B1 three times for the SAME loader (`ACS_LDR_DATE_01`, sprint `DEV`, dataset `ACS_LDR_DATE_01`, connection `PFM_Meta`) with slightly different totals (e.g. 15, 17, 16 sec) — remember to change Start/End so the hashes differ.
2. Predict again for it → the refusal is gone once it reaches 3 recorded executions (the trust threshold) and a real stage-sum total appears.

---

## PART C — Prediction History

1. Open **Prediction History**. Layout MUST be: filters on top; LEFT = predicted-vs-actual graph + accuracy cards (they reflect live state and improve as the model is retrained); RIGHT = predictions list with **10 rows per view + Prev/Next buttons** (`← Prev 10 · Page x of y · Next 10 →`). No Status/Risk columns.
2. No "Audit trail" or explanation sentences anywhere on the page.
3. Click any row → detail drawer opens **from the right**; close it.

---

## PART D — Loader Intelligence

1. Open **Loader Intelligence**. The directory table has NO "Intelligence" column (the filter stays under More Filters). **10 rows per view** with `← Prev 10 / Next 10 →` paging at the bottom.
2. No "Historical intelligence" / "Understand each loader…" sentences.
3. Click loader **MigrationSummarySubTableLoaderDataset** → the simple panel opens right after the filters showing ONLY: historical executions, completed successfully / with warnings / killed counts, typical execution time, recent execution time, typical min–max range, and the execution-time trend chart. No stage-intelligence tabs.

---

## PART E — Drift & Retraining (the decision page)

_Layout: LEFT = drift evidence; RIGHT = retraining decision + model versions + history timeline. System scope only. Human supplies ONLY the reason — no name field. Nothing retrains or promotes automatically._

### E1. Read the evidence (no actions)
1. Open **Drift & Retraining**. MUST see, without paragraphs of prose:
   - LEFT: drift health strip (Data drift / Model health badges + verdict), **Data drift** card (input groups + execution mix with statistic values behind the toggle), **Prediction drift & model performance** card (status, median absolute error, matched outcomes, per-stage health table — click a stage name to drill into `recent MedAE vs baseline`).
   - RIGHT: **Retraining** card (new outcomes x of y needed + two buttons), **Model versions** (serving version + registered models + promote/rollback), **History** timeline (promotions, rollbacks, retrainings, decisions — newest first with timestamps; "Show all N events" expands).

### E2. Record a decision (no retraining)
1. Type reason `no drift observed`, click **Continue monitoring**.
2. MUST see "Decision recorded: continue monitoring…" and the decision appears in the **History** timeline.

### E3. Train a candidate (retraining, real ML)
1. Click **Retrain now (train candidate)**. This trains + evaluates a candidate on the SAME chronological protocol — takes a few minutes; production keeps serving meanwhile.
2. MUST see the evaluation table: per-stage `Candidate median error` vs `Current baseline` + the `Total (sum of stages)` row + verdict badge (`Candidate ready for human review` / `not sufficiently better`) + **Production model unchanged**. THIS is the "last error vs now" comparison.
3. The candidate id appears (e.g. `stage-serving-candidate-…`).

### E4. Promote / rollback (human decision with reason only)
1. Copy the candidate id into **Model id to promote**, type a reason (e.g. `better holdout error`), click **Promote candidate** → success notice; **History** timeline gains a promoted event; the Model versions card shows the new serving version.
2. Predict again in the Studio (A1) → the total is served by the NEW bundle; compare the accuracy cards in Prediction History over subsequent saves.
3. **Rollback test:** click **Roll back** with any reason → the previous promoted model serves again; timeline records it. (A rollback needs a previously promoted candidate; on the standard production bundle it records the decision and keeps the standard path — the notice tells you which.)

### E5. Where everything is stored (check on disk)
- Decisions: `backend/store/retraining_decisions.jsonl`
- Candidate registration + promotion/rollback: `backend/store/model_registry.jsonl` + lifecycle log entries in `backend/store/` (lifecycle shown on the page's History timeline)

---

## PART F — Structural checks after this delivery

1. Repo root has **no PNG files** and no `_*.md` temp fragments.
2. `docs/` contains ONLY: `FILE_GUIDE.md`, `MANUAL_TEST_PLAN.md`, `ML_STAGE_PATTERN_REPORT.md`, `RETRAINING_AND_POST_EXECUTION_REPORT.md`, `STAGE_V2_TECHNICAL_REPORT.md`, `stage_v2_comparison.json`, `filled_dataset_README.md`, `verification/`.
3. The sidebar has no "Execution-time intelligence" tagline and no total-formula sentence; the Drift page has no "Is the data seen by the model…" paragraphs.
4. No page shows a "reliability" label anywhere (removed system-wide).

---

## PART G — Regression pins

| Check | Expected |
|---|---|
| Backend suite `test_sum_total_arithmetic.py` | 9 passed |
| Frontend `npx vitest run` | 57 passed (58 total incl. parametrize) |
| Frontend `npx tsc --noEmit` | exit 0 |
| Sum arithmetic (mock) | 10+20+30+40+50 = exactly 150.00 seconds; changing one stage moves the total by exactly that delta |
| New-loader / below-threshold refusals | short honest messages, no fabricated ETA (A3) |

---

## PART W — Retraining, promotion & rollback (numeric proof)

Quote these BASELINE numbers on the Drift & Retraining page BEFORE retraining (live 2026-10-05):
- Model health: Median absolute error **8.45 s**, matched outcomes **8** (minimum 20).
- Data drift: **2 of 5 signals** indicate change; review readiness **19 of 30** new eligible outcomes.
- Drifting inputs: Historical duration context, Def Trans Order Std, Loader configuration, Def Preval Order Mean/Std, Execution Order.

Steps with what to point at:
1. Save actual outcomes via Prediction Studio → "Enter Actual Post-Execution Values" (same golden identity, values above). Each save now shows `Linkage: MATCHED · IDENTITY_MOST_RECENT · Training-eligible: yes` — the count on the drift page ("19 new outcomes") climbs with every save. That is proof outcomes flow into monitoring.
2. **Continue monitoring** → a recorded decision appears in the timeline.
3. **Retrain now** → Model Versions gains a row with state `candidate evaluated` (e.g. `stage-serving-candidate-20260930T060643Z`) while the SERVING row is unchanged — proof "retrain" never auto-replaces.
4. The evaluation card under the button shows the candidate's chronological-holdout error vs the serving model. Promote only when the candidate's error is lower — the numbers ARE the justification.
5. Promote (type a reason, e.g. "lower holdout error on fresh outcomes") → serving version flips to the candidate; Model Health then tracks the new model's errors as new outcomes arrive.
6. **Rollback** → serving flips back; both moves appear in the timeline in order.
Numeric proof = candidate row state + evaluation-card error numbers + Model Health count climbing + timeline ordering.

---

## PART H — "Why this ETA?" plain explanation (2026-10-06)

_Backend `plain-explain-v2`: up to 3 plain sentences built from the same layers that run for every prediction — the per-stage prediction vs this loader's own recorded history, the calibrated interval (Task 07), and the stages that do not apply. The UI shows this text in **ⓘ Prediction Insights → Why this ETA?**; the sum-equation line remains only as a fallback when the backend layer is unavailable._

1. Run the golden case A1 → click **ⓘ Prediction Insights**.
2. MUST see 2–3 sentences, e.g.:
   - "Most of the expected time comes from the **pre-load** step, which this run's inputs make about **35% longer** than your loader's usual **9.8s** — so we predict **11s** overall."
   - "On similar past runs this model's misses stayed within about **X–Y s**, so the actual time should land close to the prediction." (or "…within about **Z s**, so treat the range as realistic rather than the exact number." when the interval is wide)
   - If some stages don't apply: "**Prevalidation** is not part of this run's configuration, so it adds nothing to the total."
3. Compare: the "usual 9.8s" number must match the loader's own recorded median for that stage (Loader Intelligence → typical execution time); the "X–Y s" range must match the **Prediction range** block right below in the same panel.
4. Brand-new loader (A3 case 1): no prediction is shown, so no insights either. A loader WITH history but no own stage medians shows the "set up much like your loader's past executions" variant.
5. The ETA itself is NEVER changed by the explanation (read-only layer).

---

## PART X — Drift & Retraining rework (2026-10-06): plain summary, live column, per-stage retraining

### X1. Plain-language summary (top of page)
1. Open **Drift & Retraining**. The page now opens with ONE sentence per signal (no paragraphs):
   - Data drift: names the shifted inputs and their count, e.g. "**3 of 5** monitored inputs have shifted: Def Trans Order Std, Loader configuration, …", with a drill-through to the full KS table.
   - Prediction drift: overall reference-vs-recent range comparison sentence.
   - Model health: matched-outcome error vs baseline sentence.
2. Each sentence ends in a link/toggle that expands exactly the corresponding detail table — nothing else.

### X2. Model error comparison table (Stage | Baseline | Previous | Current serving (since promotion) | After retraining | Linked outcomes | Health)
1. **Baseline / Previous** come from the persisted lineage file `backend/artifacts/registry/stage_performance_lineage.jsonl` (one snapshot per stage written on every promote/rollback — open the file and verify `{stage, model_version, holdout_medae, role, at}` lines).
2. **Current serving (since promotion)** = LIVE error computed ONLY from matched outcomes recorded AFTER the serving epoch: verify `backend/store/feedback_records.jsonl` — an outcome with `created_at` BEFORE the promotion timestamp must NOT be counted in this cell, but IS still visible in the all-outcome MedAE drill-through.
3. Save a new actual via the Studio (B1) → within **30 s** (auto-poll) the Current-serving cell and Linked-outcomes count move WITHOUT a page reload. The number itself shows from the FIRST matched outcome (n ≥ 1); the Health badge stays grey/insufficient until 20 outcomes (policy minimum) — the count is always disclosed next to the value.
4. Zero post-promotion outcomes → the cell shows **—**. Total row has its own live cell from `total_post_promotion`.
5. Click the Current-serving cell → drill-through shows: serving model id, the promotion time (the epoch), "updates as new matched outcomes arrive", and the all-outcome MedAE for context.

### X3. Per-stage retraining
1. On a row that warrants it click **Retrain this stage** → only ONE candidate job runs for that stage; the candidate id is `stage-serving-candidate-<stage>-<v>`.
2. After it completes, the candidate error appears ONLY in that stage's **After retraining** column (other rows keep —).
3. Backend check: `POST /retraining/train-candidate` with body `{"stages": ["staging"]}` returns a single-stage result; without `stages` it retrains all five (bundle path unchanged).

### X4. Per-stage promote / rollback
1. Promote the single-stage candidate (reason required) → ONLY that stage's serving model switches: verify `backend/store/serving_switch.json` gains a `stage_overrides` entry for that stage while the other four stages keep the standard artifact; that stage's lineage snapshot is written (role current).
2. **Rollback** with `stages: ["staging"]` (per-stage) → the override is removed, the standard stage model serves again, lineage records the reversal.
3. Whole-bundle buttons still work exactly as before (E4): five lineage snapshots, no overrides.
4. Every promote/rollback stays human-gated: nothing retrains, promotes, or rolls back automatically (History timeline records each move in order).

### X5. "After retraining" fallback
1. Reload the page after a candidate exists (e.g. after a fresh server start, in-session evaluation gone) → the **After retraining** column falls back to the latest candidate's per-stage MedAE from the registry (`retraining.latest_candidate` in the `/drift/overview` payload); show all-five candidates state when the last run was a bundle.

### X6. API pins (curl, read-only)
- `GET /monitoring/stage-lineage` → `{stages: {<stage>: {baseline, previous, current}}, active_candidate}` per stage.
- `GET /drift/overview` → `model_performance.serving_epoch` (per-stage `since` + source `registry|serving-switch`), per-stage `post_promotion` block, `total_post_promotion`, and `retraining.latest_candidate`.
- Promote a candidate, then confirm `serving_epoch.<stage>.source` becomes `serving-switch` with the switch's `activated_at`; roll back and confirm it reverts to `registry` with the restored model's `promoted_at`.
