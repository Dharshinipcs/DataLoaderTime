# EXPERIMENT_LOG.md

## DR-VERIF — FINAL AUDIT + RETRAINING-WORKFLOW VERIFICATION (2026-09-29, not a model experiment)

**Scope.** Human-gated retraining workflow completed and live-verified (governance drill, NOT a promotion-on-merit: candidates were evaluated as NOT_SUFFICIENTLY_BETTER and the promotion/rollback was an explicitly approved drill to prove the full journey). Candidate bundles trained with the SAME approved stage families and the SAME EXP-036 chronological protocol; evaluation compares against the RECORDED served baseline (backend/artifacts/registry/served_baseline.json: MedAE 1.823s, n=85, raw frozen pipeline — provenance `backend/scripts/exp036_output.txt` line 15). Candidate A holdout MedAE 10.9336s = 5.998× baseline → NOT_SUFFICIENTLY_BETTER (live); candidate B likewise → REJECTED live. Governed drill: promote (bundle gates pass) → serving switch verified by prediction change 16.649→18.524s → rollback → 16.649s restored; registry lifecycle PROMOTION_APPROVED/RETIRED/ROLLBACK appended; exactly one PROMOTED model preserved (ad34 at freeze). Human decisions persisted (CONTINUE_MONITORING + START_RETRAINING). Notifications for the retraining lifecycle reuse the AD-33 store (open-key dedupe verified live). Reliability re-anchor for the additive point±Q90 interval recorded in backend/scripts/reanchor_reliability_width.py (anchors LOW≥0.472 / MEDIUM≥0.354 over the EXP-036 holdout widths). Evidence: docs/verification/drift_retraining/ (DR-001..DR-028).

**Limitation preserved.** 8/20 matched outcomes — INSUFFICIENT to claim real model performance degradation; CONTINUE_MONITORING verdict stands on real drift + insufficient performance evidence.

## EXP-038 — FINAL PRODUCT DECISION: served TOTAL = exact sum of the five operational stage predictions

**Trigger.** After reading the EXP-037 evidence, the product owner EXPLICITLY MANDATED the sum anyway as the final product decision: the served/displayed TOTAL ETA must be the exact deterministic sum of the five operational stage predictions (staging + prevalidation + transformation + pre_load + data_loading), retiring the stage-aware total model from the serving path. Mandate: no approvals in the total, no floor, no hidden learned adjustment, no correction factor; backend = single source of truth for the arithmetic, consistent across studio/history/monitoring/tests/docs; 11-input contract, leakage rules, stage-history isolation, ambiguity handling, feedback flow, and retraining governance unchanged; approval estimates remain contextual and excluded. EXP-037 evidence/CSV are RETAINED unchanged as the traceability record of WHY the sum is less accurate on this dataset — the decision is explicitly product-driven (transparency/auditability of the displayed arithmetic), not an accuracy claim.

**Implementation (one authoritative arithmetic).** `dltime/sum_total_serving.py`: `sum_total_from_stage_intel` sums ONLY available ML stage predictions for the five operational stages, in canonical order, using the EXACT displayed values; fail-closed (any missing/unavailable operational stage ⇒ NO total is produced — SumTotalError ⇒ INSUFFICIENT_HISTORY refusal, never a partial or padded sum); `sum_total_explanation` replaces SHAP-of-the-total with the honest decomposition (the five stage predictions as exact additive components in linear seconds, base 0, additivity error 0.0, approvals listed as explicit non-components with contribution 0.0). `inference.py` predict(): stage-aware total call site + AD-34/AD-38 floor RETIRED from serving (`floor_applied` always False, `physical_floor_seconds` always None); retired models still evaluated per request and returned as clearly-labelled reference evidence only (`stage_aware_reference_seconds` NEW — the retired stage-aware output; `direct_reference_seconds` unchanged); model identity = `sum-of-five-stages-v1` / `DETERMINISTIC_POLICY` / serving_role AUTHORITATIVE_TOTAL; the 36-feature stage-aware vector is retained as audit context in `features`. `uncertainty.py` layer unchanged; calibration switched to a NEW dedicated artifact `unc-log1p-conformal-v1-sum-of-five-stages.json` built by `sum_total_calibration.py` on the EXP-036 clean protocol (identical TRAIN 256 / HOLDOUT 85 split, stage models fit ONLY on train-period rows then frozen, holdout predicted strictly pre-execution; residuals = |log1p(actual) − log1p(five-stage sum)|; Q80 1.6304 / Q90 2.9195 on 85 residuals; under-prediction share 77.6% — honestly disclosed). Retired stage-aware artifact/calibration files stay on disk untouched for traceability/rollback. Event/feedback (`feedback.py`): events record `stage_aware_reference_seconds` (floor fields retired); `compute_total_error` compares the SERVED sum against the validated actual — prediction-vs-actual stays honest (§30), no redefinition of the actual. `/models/status` reports serving identity `sum-of-five-stages-v1` (governed registry entry stage-aware-total-ad34 stays PROMOTED as governance history); `/overview` model block describes the sum policy (feature_count 5); `/explanation/global` clearly labelled as supplementary stage-model attribution on the retired model's holdout (`applies_to_served_total=false`). Frontend: `stage-sum-note` reworded to "the total ETA is **exactly the sum** of the five operational stage predictions shown below (live equation with the displayed values) — nothing is added, floored, or adjusted, and the approval estimates are not included"; hero carries `total-definition-note` ("Exact sum of the five operational stage predictions — approvals excluded, no adjustments"); "Why this ETA?" renders the exact per-stage decomposition in seconds; approvals keep "Not included in predicted total"; model identity line states the deterministic five-stage sum.

**Verification.** Controlled arithmetic (tests/test_sum_total_arithmetic.py, 8 tests): synthetic 10+20+30+40+50 → served total EXACTLY 150 (through the REAL serving path via stage-intel injection; approvals 98.4/2.7 excluded; floor never engages); single-stage delta (data_loading 50→60) moves the total by EXACTLY +10 → 160; missing-stage refusal; explanation additivity exact. Real-reference arithmetic: served ETA 16.291 = 0.6 + 0.6 + 0.482 + 12.902 + 1.707 (byte-deterministic); retired direct reference byte-stable at 16.112395656629168; retired stage-aware reference 4.180811920161842. Backend full suite 396 passed / 0 failed (all 29 test files run in groups: 132 + 57 + 61 + 75 + 53 + 18); pins updated to the approved semantics WITHOUT weakening (total==sum identity asserted everywhere; within-band flips in the 17s-actual feedback test recomputed from the served 16.291 honestly). Frontend 40 passed (37 + 3 new); production build green (306.54 kB); live browser E2E + leakage re-verification below (Part 35 checklist).

**Decision.** Sum-of-five-stages policy IS the authoritative served total (product decision recorded here and in PROJECT_STATE.md). Accuracy trade-off disclosed and accepted per EXP-037 (which remains the standing accuracy evidence AGAINST the sum on this dataset). Reversal path (if the product ever re-opens): the stage-aware artifact + calibration remain on disk; `stage_aware_reference_seconds` is already computed per request; restoring serving = re-enabling the retired call site.
## EXP-037 — final productization: simple-sum total validation (Part 7 decision gate)

**Trigger.** Final productization pass requested TOTAL ETA = sum(five operational stage predictions), retiring the stage-aware total from serving — but only if the simple sum is demonstrably acceptable (`backend/scripts/final_sum_total_validation.py`).

**Protocol.** Reused the accepted EXP-036 clean leakage-safe holdout machinery verbatim: identical chronological split (TRAIN 256 / HOLDOUT 85), stage models fitted ONLY on train-period eligible rows (staging 244, prevalidation 49, transformation 41, pre_load 13, data_loading 232), frozen, holdout predicted with pre-execution context + strictly-prior history; stage-aware totals cross-checked against `clean_holdout_predictions.csv` (ids + actuals asserted identical).

**Result (identical 85 rows, identical actuals).**

| Metric | A stage-aware total | B simple 5-stage sum | C B + approvals |
|---|---|---|---|
| MAE | 781.0 | 784.0 | 823.8 |
| MedAE | **1.82** | 10.93 | 111.1 |
| ±10% | 9.4% | **0.0%** | 0.0% |
| ±20% | 11.8% | **0.0%** | 4.7% |
| ±30% | 29.4% | 1.2% | 4.7% |
| within 2× | **51.8%** | 9.4% | 5.9% |
| SMAPE | 79.8% | 124.0% | 156.4% |

B vs actual: severe over (>2×) 62/85, severe under (<0.5×) 15/85, sane band (0.8–1.25×) **0/85** (worst 14.7×). Approvals summed in make it worse (C). Physical cause reconfirmed on holdout: measured stage-window sum / actual median 0.750 (p25 0.500) — the un-instrumented gap/overhead mass (EXP-033) plus stage overshoot on small executions make the sum structurally biased, not a sampling artifact.

**Decision.** Simple sum NOT demonstrably acceptable → per the decision rule, NO total-architecture change: the stage-aware total remains the serving ETA; the product presents it simply ("Stage 1 + … + Stage 5 → TOTAL" wording is NOT used; the Seven-Stage View keeps its never-summed note). Evidence: `final_total_policy_comparison.csv` (repo root).

## EXP-036 / 036b / 036c — clean leakage-safe holdout + floor-scope correction (AD-38 §1)

**Trigger.** The AD-36 audit disclosed nuance L1 (stage models fit on all eligible rows) meant the holdout stage columns were in-sample for the stage models. AD-38 §13 demanded a fully clean protocol before acceptance.

**Protocol (EXP-036, `scripts/exp036_clean_holdout.py`).** Chronological split TRAIN 256 rows [2025-10-10 → 2026-07-15 17:37] / HOLDOUT 85 rows [→ 2026-07-30 14:43]; the five stage models fitted ONLY on train-period eligible rows (staging 244, prevalidation 49, transformation 41, pre_load 13, data_loading 232); total model fitted on train rows with in-train strictly-prior OOF stage columns; models frozen; holdout predicted by the frozen pair; actuals revealed last. Overlap 0; leakage check pass. `clean_holdout_predictions.csv` (85 rows) written.

**Result and diagnosis.** Raw frozen-pipeline holdout: MAE 781.0 / RMSE 3172.5 / MedAE 1.82 / ±20% 11.8% / 2× 51.8%. Flooring on ALL five ML stage predictions engaged 84/85 rows and collapsed ±20% to 1.2% (floor above actual 66/85) — NOT a wrapper bug (EXP-036b: stage predictors reproduce train medians on fit rows, staging median pred/actual = 1.00) but genuine holdout stage miscalibration: pre_load median pred/actual = 18.6× (n_train 13, disclosed small-sample); staging 1.00×, data_loading 1.21×, transformation 2.08×.

**Floor-scope decision (EXP-036c, `scripts/exp036c_floor_scope.py`).** Floor restricted to evidence-adequate stage models (train n ≥ 30; pre_load excluded): MAE 779.7 / MedAE 1.38 / ±20% 20.0% / 2× 63.5% / engaged 39 / floor-above-actual 10 — beats BOTH raw and the all-stage floor on every metric, while preserving the physical containment invariant where stage evidence is trustworthy. Adopted in serving (AD-38 §1); pre_load remains a visible disclosed estimate in the Seven-Stage View. Canonical live ETA for the reference context is now the raw stage-aware 4.180811920161842 (floor 1.707 s not engaged); direct reference byte-stable at 16.112395656629168.

## EXP-035 — input-contract probes (AD-38 §6)

`scripts/exp035_input_contract_probe.py` (live API): probes A–F each change exactly the derived def_* features for the changed configuration input (prevalidation name, transformation name, execution order, counts, calling type, type) with unrelated features untouched; user echo verification (supplied_configuration, counts, order) passes; 0 failures. Regression-pinned in tests/test_input_applicability_ad35.py (override independence A/B/C/D, marker ETA-invariance, history-isolation poison).

## EXP-034 — Senior-audit evidence: unseen prediction proof, floor impact, applicability manifests

**Trigger.** Final senior audit: prove unseen-execution testing, re-validate the floor on held-out data, quantify component cardinality, and audit lifecycle-truthful UI. No architecture change.

**E1 — train/test evidence (scripts/exp034_audit_evidence.py).** Population 341 rows. Train 2025-10-10 11:26:01 → 2026-07-15 17:37:18 (n=256); test 2026-07-15 17:39:28 → 2026-07-30 14:43:48 (n=85). ID overlap 0; max(train chrono_rank) < min(test chrono_rank); artifact training_row_count == train rows; encoding/medians fitted train-side only (enc_columns pinned); historical features strictly as-of by streaming prefix. **Leakage: none found.**

**E2 — unseen predict-then-reveal (n=85).** Aggregate (with floor): MAE 780.07 / MedAE 2.02 / RMSE 3169.21 / ±10% 16.5% / ±20% 17.6% / ±30% 30.6% / 2× 63.5%. Slices: known-loader n=3 (MAE 19.16, 0/3 within 2× — too small to claim anything); new-loader n=82 (MAE 807.91, 2× 65.9%); low-history n=40 (MAE 1598.38). Honest reading: heavy-tailed data — median error ~2s while MAE is dominated by rare multi-hour/day executions; no excellence claimed; the model is a limited experimental candidate.

**E3 — floor impact on holdout.** Engaged 2/85; no-floor MAE 780.786 → with-floor 780.068 (RMSE 3170.75 → 3169.21; MedAE/±20%/2× unchanged); floor raised the prediction ABOVE actual in 0/85 cases. The floor is a pure safety invariant on unseen data, consistent with the historical containment invariant (490/490 + 0/1575, EXP-033).

**E4 — stage models.** Eligible rows: staging 326 / prevalidation 65 / transformation 60 / pre_load 19 (EXPERIMENTAL_SMALL_SAMPLE) / data_loading 313; 29 features each (config + stage-specific history); log1p targets; strictly-prior history policy. Served from stage-models-ad34.pkl with per-stage context attach (AD-35 poison test).

**Applicability manifests (scripts/exp034b_config_manifests.py).** Per-configuration component counts from the REAL definitions: configs with >1 prevalidation row 105/278 (max 147 rows, e.g. Customer Master/DEV/CUSTOMER_PROFILE_DEMO = 47 distinct names); >1 transformation 45/278 (max 11); prevalidation-only 153; transformation-only 71. Real scenario manifests recorded for acceptance Cases A–J (e.g. test_acs_d_test/DEV: 2 prev + 4 trans with orders; 2210_TEST_ALL_CONFIG/SUPPORT: 6+5). Conclusion recorded in AD-36 §4: single display-name field = audit context for ONE component; counts/order/type always canonical-derived; multi-component structured input deferred explicitly.

**Lifecycle-truthful UI.** Studio status line now derives wording from serving_role (authoritative vs reference); notification panel repositioned (fixed left-16 top-16) — clipping eliminated, verified live; global SHAP confirmed bound to the authoritative ad34 artifact over holdout rows. LOW-reliability logic audited and CONFIRMED correct (width-factor ≥12× + <100-residual calibration cap); live reasons disclosed.

**Regression tests added (tests/test_input_applicability_ad35.py, 8 passing).** Override independence A/B/C/D (markers write only their own column), neutral-marker ETA invariance (audit markers are not fitted features), count-override isolation, stage-specific history poison isolation (future-row poisoning leaves the probe's stage_hist_* byte-identical — automated regression for the EXP-032 Part-J bug class), floor = max ML stage predictions only (approvals excluded), authoritative role/status contract.

**Full suites.** Backend 385 passed / 0 failed (single 20:06 run; 377 prior + 8 new). Frontend 54/54; build green. Docs created: docs/ACCEPTANCE_TEST_CASES.md (developer A–O matrix + 10 user scenarios), docs/UNSEEN_PREDICTION_EVIDENCE.md (this evidence, formalized).

## EXP-033 — Final closure forensics: stage ORDER, coverage vs summation validity, max-floor justification

**Trigger.** Final acceptance prompt: no universal stage-order assumption; zero-overlap ≠ complete-coverage; explicit total-composition decision; max(...) floor justification; Overview p95 cleanup.

**Part A — stage order (scripts/exp033_stage_order_forensics.py, execution lens: 492/512 executions with ≥1 valid stage window; row lens 739/759 rows).** (A1) NO universal order: **20 distinct observed runtime orders** (6 for n=2 stages, 6 for n=3, 4 for n=4, 1 for n=5); top patterns staging>data_loading 216, staging>prevalidation 48, staging>transformation>data_loading 27, staging>prevalidation>data_loading 23, staging>transformation>prevalidation 17, staging>transformation>prevalidation>data_loading 22; the full 5-stage sequence occurs in only **4/492** executions. (A2) single-stage executions 80 (staging 64, data_loading 14, prevalidation 2); stage sets are loader-specific signatures (e.g. Asset Master AS91 = staging+prevalidation 16/18; doublesen_ldr = staging+data_loading 34/34; PreloadLDR = pre_load+staging+transformation ±data_loading). (A3) `execution_order` config: the definitions detail contains ONLY calling_type ∈ {PREVALIDATION 1278, TRANSFORMATION 202} rows — no staging/pre_load/data_loading component rows exist — so config order can only describe the prev/trans component sequence; 10/229 loaders carry >1 distinct config sequence; 33/35 both-component sequences place PREVALIDATION first. **Runtime conflicts with config in 57/57 executions where both windows exist** (runtime shows transformation first in all 57; e.g. ExtraConfigLdr2902 observed staging>transformation>prevalidation>data_loading vs config prevalidation>transformation). VERDICT: pre-execution `execution_order` and runtime timestamp order are DIFFERENT artifacts (component-row ordering vs measured execution); the runtime order is authoritative for physics; the only safe inference from config is component composition, never operational order.

**Part B — zero-overlap vs coverage (execution lens).** 583 consecutive-window pairs: **0 overlaps**, 326 touching, 257 gaps; sum(windows) == union(windows) exactly (max |sum−union| = 0.000000s). BUT complete coverage of Start_Time→End_Time in only **13/490 executions (2.7%)**; union/span median **0.42** (row lens 0.30; union/total median 0.44 exec lens, 0.31 row lens); head gap median 0.6s, tail median 0.1s (27 tail outliers >1 day flagged as data-quality, still span-consistent with min×60); per-pattern: partial_multi n=407 median 0.46, single_stage n=79 median 0.04, full_sequence n=4 median 0.48 — **no pattern category achieves coverage**; per-loader medians range 0.00 (Test_Map_check_2) to 0.98 (Check_Case_ConsoliRepoer_ldr) — conclusion is universal, not loader-specific. Containment invariant total ≥ union holds **490/490** (all categories); duration fields never exceed total **0/1575**; approvals never exceed total (0/21, 0/6).

**Part C/D — total-composition decision (explicit).** (1) Simple sum of predicted stage durations: INVALID — mathematically requires window-union == total, measured 0.44 median; ≈56% of the median total is uninstrumented queue/gap/overhead time that stages cannot see; missing stages make sums unstable. (2) max(predicted stages) alone: INVALID — a lower bound, not an estimator; discards all gap mass and non-largest-stage evidence. (3) Learned stage-aware total (ad34): RETAINED as the ONLY candidate that can represent uninstrumented overhead (trained on the authoritative total target, not on stage values; leakage-safe OOF stage features). (4) Floor: RETAINED with precise invariant — **every historical execution total ≥ every one of its stage durations** (containment 490/490 execution lens + 737/737 row lens, duration-field form 0/1575), therefore max(ML stage predictions) is a data-consistent lower bound on any future total from the same process; applied ONLY at serving, disclosed (floor_applied/physical_floor_seconds); never in training/eval/calibration/SHAP; approvals are EXCLUDED from the floor (approval fields never exceed total and approvals post-date the loading work — including them would import a >total-median estimate (98.4s datamart) into short executions). Conclusion: `ETA = max(learned stage-aware total, max ML stage predictions)` is the evidence-backed composition; unchanged from AD-34.

**Part F — Overview p95 cleanup.** pipeline.py report.target: value source verified authoritative already (reconstruction derives the column from total_timetaken_in_min×60; the corrupted legacy field never survives reconstruction) but labels were untruthful; now carries definition + target_source_counts + span-consistency (checked/within_5pct); api.py surfaces duration_stat_definition etc.; Overview UI sub-label now “authoritative total (min×60)”. Audit of every `timetaken_in_sec` consumer (eligibility gate, feature_assembly TARGET_COLUMN, historical features, loader_intel, feature_registry hist specs, leakage schemas, direct-model path): all read the RECONSTRUCTED frame (AD-31-derived value); the direct model remains a documented legacy/reference path (byte-stable rollback, AD-34). No authoritative-path use of corrupted legacy seconds exists.

**Part K — live representative-case probes (scripts/exp033_consistency_cases.py, stack up).** canonical-context-floor: PREDICTED 12.902 s, floor 12.902, floor_applied=true, ETA ≥ max ML stage (5 ML stages); 11-input-overrides: same, ETA invariant under marker overrides; example-ldr-alt-identity: PREDICTED 12.902 (global small-sample pre_load dominates); asset-master + bapi: INSUFFICIENT_HISTORY (no ETA fabricated); test-transform sparse: UNRESOLVED_IDENTITY (fail-closed). **Violations: 0.**

**Parts E/G/H/I/J/L/M/N — re-verified live.** Direct reference byte-stable 16.112395656629168; 11-field contract intact (form + API + planned_start_time:null; workflow-ref linkage-only note); 5 ML stage models + 2 contextual statistical approvals confirmed in payload and tests (test_stage_intel_prompt2.py: five_ml_stages, approvals_contextual_statistical, never_summed, no_alias); stage-history parity: _attach_context_stage_history(stage_specific=True) + generic columns carry staging history (inference.py docstring documents the corrected Part-J semantics); poison suite green (future-row ETA byte-identical, stage-history strictly-prior, current-row contamination blocked). Feedback roundtrip: outcome UNMATCHED (deterministic, EXP-021) then predict→outcome with exact workflow ref → **MATCHED** with computed error (−32.098s) → monitoring metrics/slices updated live. Notifications: 13 alerts, 6 alert types (MODEL_DRIFT feature/population, RETRAINING_RECOMMENDED, STAGE_DATA_COVERAGE ×5, INSUFFICIENT_HISTORY ×2, LOW_PREDICTION_RELIABILITY ×2, MODEL_STALE), unique keys, bell badge live updates, mark-all-read → 0 unread, persistence via notifications.jsonl. Registry: ad34 PROMOTED, ad32 + direct RETIRED; readiness RETRAINING_RECOMMENDED_FOR_REVIEW, automatic_retraining OFF, human_gate_required true. UI stale copy fixed: Model Lab feature version read from promoted artifact (was hardcoded “pre-execution+historical-v30”); DataSources policies updated to the 11-input contract + AD-31a pre_load independence + physical-floor statement (removed the false “pre_load is a duplicate of data_loading” claim).

**Tests.** Backend 377/0 (chunks 86+147+117+27); frontend 54/54; build green (rerun after UI fixes).

## EXP-032 — Final forensic audit: stage-timing physics, total-target correction, 11-input contract, per-stage history

**Trigger.** Final audit prompt: (1) historical stage timing/overlap/total physical consistency; (2) prevalidation vs transformation input semantics; (3) the live red flag where the stage-aware ETA (2.01 s / 4.18 s) fell below the predicted pre_load stage.

**Part B — stage-timing physics (scripts/exp032_stage_timing_audit.py).** 737 executions with valid stage windows: stage order staging→prevalidation→transformation→pre_load→data_loading (prevalidation|transformation order reversed in all 57 evaluable rows — the source labels the pair inconsistently, timestamps decide); **0 overlapping windows in all 10 pairs** (sequential with gaps); containment total ≥ union(stage windows) **737/737 (100%)**; union covers only ~31% of total (median) — non-stage queue/gap time dominates; total/Σstage-durations median 0.59 (the earlier 5.22 was a span-vs-duration artifact of sparse stage windows, not a multiplier). Approvals: duration-only evidence (no timestamp windows) — treated as separate post-load analytics.

**Part C — total-target root cause.** `total_timetaken_in_min×60` matches the Start/End span **757/757 (100%)**; legacy `timetaken_in_sec` agrees only **15.1%** (170/172 checked rows disagree, up to 1000×) — corrupted. The stage-aware total (and direct model) had been trained on the corrupted field; the reference loader's true totals are 3.6–20.4 s (median 3.6), so a ~4 s raw ETA was correct for its context while pre_load 12.9 s is a global small-sample fallback for a loader with zero pre_load history. Sum-based candidates rejected on physics (stage vector alone R²≈0.15 log-space; union ≈31% of total); learned stage-aware total (C4) retained on the corrected target with a **serving-side physical floor** (duration fields never exceed total: 0/1575). Duration-vs-total audit: scripts/exp032_duration_vs_total.py.

**Corrections applied.** (1) Retarget: stage_aware_total.py + stage_aware_serving.py train/evaluate/calibrate on `total_timetaken_in_min×60`; artifacts rebuilt as `stage-aware-total-ad34` / `stage-models-ad34` / `unc-log1p-conformal-v1-stage-aware-ad34.json` (85 residuals); ad32 retained for rollback. (2) Floor: ETA = max(stage-aware ETA, max ML stage prediction); disclosed via floor_applied/physical_floor_seconds + UI bullet. (3) Part J: serving stage-history loop overwrite fixed — each stage model gets its own stage_hist_* context; generic contract columns carry staging history; stage-specific poison test passes (poisoning prior staging/transformation/data_loading history changes only its own stage's prediction). (4) Part F (scripts/exp032_definitions_audit.py): definitions rows are per-component — calling_type PREVALIDATION=1278 vs TRANSFORMATION=202; preval_trans_display_name is a component label (DO/CR/QUERY/…), not an operation identifier; a single execution config carries many components of both kinds → **11-input contract** (separate prevalidation_display_name + transformation_display_name) in PredictionStudio + API + override resolution; A/B/C/D probes confirm independence (markers are audit context, not fitted features — ETA unchanged across them, disclosed). (5) Part H: outcome total switched to `actual_total_timetaken_in_min` (seconds backward-compatible).

**Corrected-target chronological comparison (identical folds, holdout n=85).** Stage-aware: pooled OOF MAE 498.33 / MedAE 4.43 / RMSE 1462.60 / ±20% 19.31% / 2× 36.20%; holdout MAE 758.73 / MedAE 4.80 / RMSE 3031.59 / ±20% 4.71% / 2× 23.53%. Direct: pooled 513.09 / 11.87 / 1507.84 / 7.56% / 31.45%; holdout 779.30 / 4.09 / 3158.08 / 8.24% / 42.35%. Mixed evidence persists (stage-aware wins pooled MAE/MedAE/RMSE/±20%/2×; direct wins holdout MedAE/±20%/2×); stage-aware remains authoritative by AD-32 architecture with honest reporting; direct = reference/rollback (still byte-reproducing 16.112395656629168).

**Live E2E (browser + API, ad34).** Canonical prediction: staging 0.60 / prevalidation 0.60 / transformation 0.482 / pre_load 12.902 / data_loading 1.71 (5× ML_PREDICTED) + datamart 98.4 s (GLOBAL n=7) / dataloading 2.7 s (GLOBAL n=6) → **ETA 12.902 s with floor engaged (raw stage-aware 4.18 s < pre_load prediction)**, disclosed in payload + UI; direct reference 16.112 s. Reliability LOW (interval 29.6× point; 85 residuals), risk LOW_CONFIDENCE + ETA-vs-loader-median signal. Outcome submitted via UI (total_min semantics): training-eligible ELIGIBLE, linkage UNMATCHED (no workflow ref supplied — EXP-021 identity+time matching unsafe, deterministic by design); alerts MODEL_STALE (INFO, correct ad34 version) + LOW_PREDICTION_RELIABILITY (WARNING) fired from real conditions. All pages re-verified live (Overview/Studio/History/Monitoring/Laboratory/notifications).

**Tests.** Backend 377 passed / 0 failed (26 files, chunks 96+134+68+79, rerun after every fix). Frontend 54/54; production build green. Store cleaned after probes except the demo prediction/outcome/notification roundtrip.

## EXP-031 — AD-32 authoritative serving switch + AD-33 notification center (final implementation pass)

**Trigger.** Locked final direction: the stage-aware total must become the authoritative prediction path; feedback + notification + monitoring completion to close the end-to-end lifecycle.

**Serving build (real corrected-data runs).** `stage-aware-total-ad32` artifact: extra_trees, log1p, 36 features (29 context + 7 stage-derived), encoded columns 1134, train n=341-eligible population, holdout 85. Calibration: `unc-log1p-conformal-v1-stage-aware.json`, 85 chronological residuals from the exact served fit. OOD profile: numeric `def_*` q01/IQR bounds. Stage artifacts: all five persisted (`stage-serving-models.pkl`, version-checked).

**Chronological comparison on the identical population (seconds, inverse-transformed):** stage-aware pooled OOF MAE 498.3 / RMSE 1462.6 vs direct 523.8 / 1507.8; holdout MAE 758.8 vs 779.3; holdout ±20% 9.41% vs 8.24%; direct historically retains holdout MedAE/2× — evidence mixed, architecture decision (AD-32) makes stage-aware authoritative with honest reporting; direct = REFERENCE_ROLLBACK (RETIRED in registry), still byte-reproducing 16.112395656629168.

**Live seven-stage prediction (canonical Scenario A context, actual API response):** staging 6.019 s (ML extra_trees), prevalidation 381.039 s (ML extra_trees), transformation 0.343 s (ML random_forest), pre_load 13.200 s (ML extra_trees, SMALL-SAMPLE n=19), data_loading 1.624 s (ML ridge), datamart approval 98.4 s (GLOBAL tier n=7), dataloading approval 2.7 s (GLOBAL tier n=6) → stage-aware total ETA 2.014 s (AUTHORITATIVE_TOTAL); direct reference 16.112 s. Uncertainty recalibrated to the stage-aware model (n=85); SHAP = TreeExplainer on the served artifact, additivity |err| ≈ 1.9e-13; risk LOW_CONFIDENCE via LOW_RELIABILITY (locked hierarchy).

**AD-33 notifications.** 12 alert types wired to existing verdicts only; persisted notifications.jsonl; dedupe on open keys; read/unread persistent; endpoints verified by 18 new tests (test_notifications_ad33.py) incl. evaluate→mark-read→re-raise roundtrip.

**Feedback.** XLSX-faithful ActualOutcomeBody (§26) + ActualOutcomeForm (Studio + History); errors computed on ingest; validation warnings surfaced + alert-raised; training-eligibility unchanged (AD-24 semantics).

**Tests.** Backend 377 passed / 0 failed (26 files; chunks 96+134+68+79, all rerun green after fixes). Frontend 54/54; production build green. Store cleaned after probes; no automatic retraining; promotion performed once through the documented human gate with recorded evidence.

One entry per investigation/experiment. Facts only; no narrative.

## EXP-001 — Source schema & structural audit (Task 01)
- Data: real files in `train/data/` (sha256 in provenance).
- Method: read-only pandas load; per-column non-null/unique profile; groupby constancy analysis per execution.
- Result: 1480×8 / 278×6 / 792×38; 543 unique `LDR_Execution_Id`; 74 multi-row executions; max 9 rows; 0 fully duplicated rows.
- Decision: schemas locked into `sources.SOURCE_SPECS`; facts asserted by tests.
- Accepted: reproduces the locked audit exactly.

## EXP-002 — Reconstruction semantics (Task 01)
- Data: 792 raw execution rows.
- Method: per-column constancy check across rows of each execution; inspection of 9-row executions; duplication check.
- Result: Loader_Name, Sprint, LDR_Status, ldr_workflow_execution_id, Start_Time, End_Time, Time_Taken, timetaken_in_sec, transformation/pre_load/data_loading/approval timings are constant within every execution; Dataset_display_name + record/error counters vary per row. Multi-row = ONE workflow execution loading several datasets (e.g. exec 90559: 9 datasets, one workflow id 272023, identical Start/End/target).
- Decision: reconstruction policy per AD-03; one row per LDR_Execution_Id + component rows preserved; 543 executions; 469 single-row / 74 multi-row / 323 rows in multi-row execs.
- Accepted: tests `test_reconstruct.py` assert every locked number.

## EXP-003 — Status semantics & eligibility policy (Task 01)
- Data: 543 reconstructed executions; status × target-availability × resolution.
- Result: exec-level distribution SUCCEED 188, STOPPED 178, CWW 89, CWE 61, CWEW 12, KILLED 8, ERROR 7. Target present for 100% of completed-status executions; missing only for 2 STOPPED executions. STOPPED/KILLED are interrupted outcomes (337/8 raw rows) — no usable completed-duration semantics.
- Decision: v1 training policy = completed statuses only (AD-08) → 353 training-eligible executions. Configurable, revisited after duration-vs-status evidence in Task 02/03.
- Accepted: eligibility framework made policy-configurable, not hard-coded.

## EXP-004 — Connection resolution (Task 01)
- Data: executions (Dataset, Sprint, Loader) vs both definition sources.
- Method: exact normalized (strip, case-sensitive) lookup; 0/1/multi match rule; per-execution union aggregation.
- Result: raw rows 382 RESOLVED / 410 UNRESOLVED / 0 AMBIGUOUS (exact match to locked audit). Keys (dropna=False): 44/95/0. Executions: 274/269/0. Definition keys with conflicting connections: 0. Missing-component keys: 3 (counted UNRESOLVED; the locked 95 uses this convention). 27 mixed-resolution executions resolve via union rule (strict alternative = 247).
- Decision: AD-04. The single sprint-mismatch key (`testing_table|SUPPORT|testing-table` → defs say DEV) is NOT remapped.
- Accepted: reproduces the locked audit without any silent choice.

## EXP-005 — Target validation (Task 01)
- Data: `timetaken_in_sec` raw rows (790 non-null) and executions (541 non-null).
- Method: distribution stats; span consistency `End_Time − Start_Time` vs target.
- Result: raw median 65.62 / mean 870,882.67 / p95 665,974.4 / max 43,000,000 (all locked values reproduced). 775/790 rows match span within 1s; 12 rows exceed span by >1s (target > span); 3 rows span > target. 36 executions >100,000s; 21 corroborated by span within 1s. Exec-level median 26.325 / mean 1,232,600.23 / p95 648,287.
- Decision: target accepted as-is; extremes retained (AD-05/06); robust metrics mandatory in later evaluation.
- Accepted: no outliers removed.

## EXP-006 — Leakage classification coverage (Task 01)
- Data: all 52 raw source fields.
- Method: registry completeness test + guard tests (forbidden list, unclassified field, frame-level guard).
- Result: every raw field classified into exactly one of 7 categories; guards raise `LeakageViolation` for target/post-execution/record-counter fields and for unclassified fields.
- Decision: AD-07 enforcement layer accepted; will be extended to engineered features as they are created (Task 02+).

## EXP-007 — AD-04 review: the 27 mixed-resolution executions (read-only investigation, no code changed)
- Data: real sources; 27 executions with both RESOLVED and non-RESOLVED component rows.
- Method: per-component definition diagnostics (triple match; (Sprint,Loader)→connection cross-check; dataset-presence check); policy comparison A/B/C.
- Result:
  - All 27 are multi-row (2/3/9 rows); all 27 have exactly ONE unique resolved connection across components (27/27); 0 with multiple distinct connections; 0 with missing identity; 0 AMBIGUOUS components; 0 contradictory evidence.
  - All 50 unresolved components: their (Sprint, Loader) definition pair maps to exactly the SAME connection as their execution's resolved connection (0 contradictions) — failures are dataset-dimension lookup gaps (e.g. EXTENSION, INVESTMENT, Childchi, ChildSec, preLoadLDR2 absent from definitions by triple).
  - Cross-check: 247 fully-resolved executions agree on one connection in 247/247 cases (0 disagreements).
  - Structural fact: `ldr_workflow_execution_id` is constant within every execution; 0 multi-row executions are fully resolved by triple lookup (74/74 have ≥1 gap) — full-triple coverage does not exist for any multi-dataset workflow.
  - Policy counts: A (any-resolves/union) 274 RESOLVED / 269 UNRESOLVED / 0 AMBIGUOUS; B (all-must-resolve) 247/296/0 (ALL 74 multi-row executions become UNRESOLVED); C (component-level only) ≈ B at execution level unless partial-evidence handling is added.
  - Eligibility cross-tab: 135 of 353 training-eligible executions are RESOLVED (10 of the 27 mixed are training-eligible: 4 PreloadLDR, 5 Check_Case, 1 Asset Master CWW).
- Decision: recommend keeping Policy A. It cannot silently choose among conflicting connections (multiple distinct ⇒ AMBIGUOUS, locked rule), and the evidence shows one workflow execution = one connection with zero contradictions. **APPROVED by user (Task 02 kickoff); AD-04 updated with precise distinct-connections rule.**

## EXP-008 — Task 02 chronological historical features (real data)
- Data: 543 reconstructed executions (350 training-eligible).
- Method: deterministic chronological ordering (Start_Time, tie-break LDR_Execution_Id); streaming prefix computation; prior_filter='training_eligible' (only trustworthy prior outcomes feed history). Groups: GLOBAL, LOADER, CONNECTION (Loader+execution_connection), DATASET (Loader+representative dataset), COMPOSITE (Dataset+Sprint+Loader), PREVIOUS.
- Result: coverage over 543 execs: GLOBAL 538, LOADER 371, CONNECTION 203, DATASET 346, COMPOSITE 341; first chronological execution has zero priors (tested). Training rows: ≥1 prior 349/350; prev-loader-duration 262; valid time-since-prev 262 (clipped ≥0); prior-loader-count median 2 / p75 6 / p90 10 / max 33; zero-history 88.
- Decision: AD-11 adopted (strictly-prior rule; synthetic leakage probes + real-data assertions in test_historical.py).
- Accepted: chronology guarantees proven by tests; no current/future contribution possible.

## EXP-009 — History eligibility evidence (no threshold locked)
- Data: prior-loader-history counts for 350 training rows.
- Result: distribution min 0 / p25 0.25 / median 2 / p75 6 / p90 10 / max 33; zero-history 88; <3: 184; ≥3: 166; ≥5: 114; ≥10: 42. Eligibility outcomes by candidate threshold: min=3 → 166 sufficient / 184 insufficient; min=5 → 114/236; min=10 → 42/308.
- Decision: HistoryEligibilityEngine is configurable; thresholds remain PENDING (AD-12) until Task 03/04 chronological evaluation shows which level yields reliable ETA. No arbitrary rule locked; insufficient-history contexts will return explicit INSUFFICIENT_HISTORY (no silent cold-start ETA).

## EXP-010 — Execution-level dataset completeness (corrective finding)
- Data: 543 reconstructed executions after adding exec-level `Dataset_display_name` (first component dataset, deterministic) in Task 02.
- Result: 8 executions have NO dataset at all (40799/40804/40806/40809 missing Loader+Sprint+Dataset; 89883 KILLED; 90579/90580/90581 SUCCEED). Under the locked canonical identity (Dataset+Sprint+Loader) they cannot form a valid identity; 3 of them (90579/90580/90581) were previously counted training-eligible because the dataset column was absent from the Task 01 frame.
- Decision: identity completeness now includes dataset (consistent with locked canonical identity). Training-eligible count 353 → 350 (535 valid / 533 target-valid / 350 training-eligible). Task 01 tests updated to the corrected real values; no data modified.

## EXP-011 — Definition-derived configuration features (real data)
- Data: LOADER_DEFINATIONS.xlsx + LOADER_DEFINATIONS_1.xlsx joined by canonical identity; execution-level aggregation over the execution's dataset set (multi-dataset aware).
- Result: definition evidence exists for exactly the 274 RESOLVED executions (same lookup semantics as connection resolution); among 350 training rows, 135 carry definition features. prevalidation_count legitimately includes 0 (56 identity keys); def_component_count ≥ 1; def_dominant_calling_type ∈ {PREVALIDATION, TRANSFORMATION} (categorical; encoding decision deferred to Task 03).
- Decision: 8 config features registered as DERIVED_FROM_PRE_EXECUTION/PRE_EXECUTION (AD-13); unmatched executions get <NA> (no fabrication); def_match recorded as analytical diagnostic.

## EXP-012 — Task 03 chronological model training & evaluation (real data)
- Data: 350 training-eligible executions; 30 leakage-safe features; target timetaken_in_sec (raw and log1p back-transformed via expm1).
- Split (chronological, no shuffle, AD-11):
  - Expanding-window CV (5 folds): train 175→315 rows, validate 35 rows each; windows only grow forward.
  - Final holdout: train 262 rows (2025-10-10 … 2026-07-15), test 88 rows (2026-07-15 … 2026-08-13); boundary = single timestamp, no ties straddle it.
- Models: baselines global median / global mean (DummyRegressor, chronological windows); Ridge(α=1, standardized, one-hot); RandomForest(300, min_leaf=2); ExtraTrees candidate (200, min_leaf=2, max_features=1.0, seed 42); ExtraTrees default; HistGradientBoosting(300, lr=0.05). XGBoost/LightGBM/CatBoost not installed — reported, not installed.
- Results (pooled out-of-fold, raw seconds):
  | model | mode | MAE | MedAE | RMSE | ≤30% | ≤2× | hold MAE | hold MedAE | hold ≤30% |
  |---|---|---|---|---|---|---|---|---|---|
  | extra_trees_candidate | log1p | **364.6** | **5.16** | 1711 | **25.7%** | **48.6%** | 696.7 | 5.74 | 13.6% |
  | baseline_global_median | log1p | 507.1 | 5.49 | 2299 | 19.4% | 30.3% | 764.3 | 4.67 | 18.2% |
  | extra_trees_default | log1p | 1856 | 7.89 | 13660 | 22.9% | 37.1% | 698.8 | 11.55 | 13.6% |
  | hist_gradient_boosting | log1p | 491.2 | 15.96 | 1851 | 16.0% | 31.4% | 733.8 | **2.95** | **29.5%** |
  | ridge | log1p | (exploded, see below) | 16.66 | — | 4.0% | 16.6% | 719.3 | 15.15 | 0.0% |
  | random_forest | log1p | 510.0 | 23.70 | 1642 | 26.3% | 37.7% | 707.7 | 21.24 | 19.3% |
  | baseline_global_mean | log1p | 521.9 | 33.57 | 2293 | 6.3% | 9.7% | 779.1 | 29.64 | 3.4% |
  | all raw-target models | raw | MAE 5.3e3 … 3.4e6 | 1700 … 6.8e5 | — | 0–12% | 2–15% | — | — | — |
- Raw vs log1p: log1p wins decisively on every model (raw-target ExtraTrees candidate: pooled MAE 347,893 s / MedAE 250,923 s). Right-skew makes raw-target squared-error training dominated by extreme durations. No observation deleted; only the transformation differs.
- Candidate detail (fold-level, ExtraTrees log1p): MAE per fold 129 / 252 / 933 / 476 / 32 (highly variable — small folds + extreme values dominate some folds); pooled MedAE 5.16 s.
- Seen vs unseen loader (pooled OOF): seen (≥1 prior loader exec) n=126: MAE 219.3, MedAE 3.81, ≤30% 31.0%, ≤2× 54.0%. Unseen (0 priors) n=49: MAE 738.3, MedAE 15.10, ≤30% 12.2%, ≤2× 34.7%. ML helps exactly where history exists — consistent with the INSUFFICIENT_HISTORY product rule.
- History-count bins (pooled OOF, ExtraTrees log1p vs global-median baseline):
  | bin | n | ET MedAE | ET ≤2× | baseline MedAE | baseline ≤2× |
  |---|---|---|---|---|---|
  | 0 | 49 | 8.19 | 50.0% | 4.62 | 42.9% |
  | 1–2 | 75 | 8.19 | 50.0% | 5.55 | 30.4% |
  | 3–4 | 28 | 6.19 | 35.7% | 5.74 | 17.9% |
  | 5–9 | 44 | **1.30** | **70.5%** | 4.97 | 27.3% |
  | 10+ | 8 | 4.21 | 50.0% | 5.06 | 12.5% |
  Bin 0 and 1–2 share identical MedAE 8.19 for ET because loader-history features are all-missing at count 0 — same information state.
- Ridge numerical explosion: standardized log-space Ridge produced predictions up to 1.12e16 in fold 1 (collinear one-hot/median-imputed features ⇒ ill-conditioned solve); recorded, not tuned (no time budget for iterative solver search; linear model not a serious candidate on this data).
- Encoding decision: `def_dominant_calling_type` (2 real categories) one-hot with handle_unknown=ignore + explicit "__MISSING__" sentinel; no ordinal label-encoding (documented AD-14).
- Reproducibility: fixed seeds everywhere; deterministic chronological splits; `aggregate_fold_metrics` keeps pooled-OOF, mean-across-folds, median-across-folds, worst-fold strictly separated.
- Decision: ExtraTrees(log1p, locked candidate hyperparameters) selected as CANDIDATE (artifact total-et-v3-candidate, status CANDIDATE_EVALUATED — not production). Baseline evidence: beats global median on pooled MedAE (5.16 vs 5.49), MAE (365 vs 507), ≤30% (25.7 vs 19.4), ≤2× (48.6 vs 30.3). History-eligibility recommendation (AD-12 evidence, NOT locked): ≥3 prior loader executions required for authoritative ETA; bins 0–2 show MedAE ≈ 8 s / ≤2× ≈ 50% (unreliable), bins 5–9 show MedAE 1.3 s / ≤2× 70% (reliable); 10+ bin too small (n=8) for reliable conclusions.
- Accepted: candidate model accepted for Task 04 inference integration; threshold decision remains pending explicit approval.

## EXP-013 — Task 04 inference integration (real data, end-to-end)
- Artifact: `total-et-v3-candidate` (CANDIDATE_EVALUATED), ExtraTrees(n_estimators=200, min_samples_leaf=2, max_features=1.0, random_state=42), log1p→expm1, 30 features in exact Task 02 order; fitted on the recorded 262-row chronological training window (2025-10-10 11:26:01.616 → 2026-07-15 17:48:04.909); 88 holdout rows excluded. Built via the same encoding path as Task 03 evaluation (`fit_candidate_pipeline`); persisted to `backend/artifacts/total-et-v3-candidate.pkl`; load-time integrity checks (version/status/feature order/preprocessor feature_names_in_; PRODUCTION status refused).
- Inference path: clean pre-execution request → validate → canonical-identity resolution (locked rules) → synthetic context row (target <NA>, is_training_eligible=False) appended to `executions_eligible` → UNMODIFIED Task 02 definition+chronological feature path → AD-16 gate (min_prior_loader_executions=3, approved) → artifact predict → expm1.
- Out-of-sample demonstration (holdout identity replay, no fabricated values):
  - `MigrationSummarySubTableLoaderDataset | TEST | Migration_Summary_Sub_Fact_Table_LDR_Datase_new`: actual 17.837 s, predicted 18.155 s (abs err 0.318 s, 1.78%), context saw 11 priors, state PREDICTED.
  - `MigrationSummaryMainFactTableLoader | TEST | Migration_Summary_Main_Fact_Table_Loader`: actual 0.825 s, predicted 7.756 s (large relative error on a sub-second execution — small-duration percentage errors are inherently unstable; limitation recorded, consistent with Task 03 tolerance-metric emphasis).
  - Never-executed definition loader `2210_TEST_ALL_CONFIG | SUPPORT | A_TABLE` (0 priors): state INSUFFICIENT_HISTORY, no ETA. Blank sprint: INVALID_INPUT.
- Determinism: parallel predict (`n_jobs=-1`) drifted in the last ulps between identical requests; serving estimator pinned to n_jobs=1 → bit-identical repeated predictions (test-enforced).
- Test-enforced guarantees: 39 new tests (artifact build/save/load/integrity; all 5 response states; feature order == artifact order; leakage probes — poisoned strictly-future target leaves features/history/prediction unchanged, context row ineligible and target-less; AD-16 configurable — min=5 flips a 3-4-prior case to gated; deterministic HTTP repeats; schema extra="forbid" rejects smuggled post-execution fields with 422).
- Full suite: 129 passed (90 pre-existing + 39 new).
- Decision: inference integration accepted for review; artifact remains CANDIDATE_EVALUATED (not production); no frontend/uncertainty/SHAP/risk/monitoring implemented.

## EXP-014 — Task 05 Prediction Studio (frontend + live end-to-end verification)
- Frontend: new `frontend/` (Vite + React 19 + TypeScript + Tailwind v4; Vitest + Testing Library + jsdom). Route: single page at `/` (Prediction Studio; no router — locked no-unnecessary-infra rule).
- Contract: mirrors Task 04 `POST /predict` exactly via typed client `src/api.ts`; dev proxy `/api` → `127.0.0.1:8000`; no second API invented. Backend change limited to CORS middleware (localhost/127.0.0.1:5173, GET/POST, Content-Type) + 3 tests; /predict semantics untouched.
- User-facing inputs (complete list): loader_name*, sprint*, dataset*, connection (optional, validation-only), planned_start_time (optional, datetime-local → ISO). `workflow_execution_id` deliberately NOT exposed (internal reference, no user value). NO manual inputs for any historical/derived/post-execution value — enforced by a component test that asserts no such labeled input exists.
- Frontend tests: 10 passed — required-field validation (no API call), exact POST contract body, PREDICTED rendering, INSUFFICIENT_HISTORY (no ETA), UNRESOLVED/AMBIGUOUS/INVALID_INPUT rendering, network-failure + retry, loading + disabled button, absence of forbidden inputs. Build: `tsc -b && vite build` green.
- Live E2E (real backend on :8000, Vite on :5173, real browser driven through the actual UI):
  - A: `MigrationSummarySubTableLoaderDataset | TEST | Migration_Summary_Sub_Fact_Table_LDR_Datase_new` → PREDICTED, 18.16 seconds, total-et-v3-candidate, 11 priors, RESOLVED → MDM_DM_TEST (matches Task 04 out-of-sample value 18.155 s).
  - B: `2210_TEST_ALL_CONFIG | SUPPORT | A_TABLE` (real never-executed definition loader) → INSUFFICIENT_HISTORY, 0 of required 3 priors, NO_DIRECT_HISTORY, no ETA, connection-aware warning shown.
  - C: `nonexistent_loader_e2e | X | Y` → UNRESOLVED_IDENTITY, no identity chosen.
  - D: valid identity + `connection=wrong_connection_xyz` → INVALID_INPUT `provided_connection_disagrees_with_definition_resolution: 'wrong_connection_xyz' != 'MDM_DM_TEST'` (no override, no retry-with-guess).
  - E: cleared required field → client-side errors, no fetch issued, previous result cleared (no stale ETA ambiguity).
- Backend suite after Task 05: 132 passed (129 + 3 CORS/API).
- Decision: Prediction Studio accepted for review; backend identity semantics unchanged (Section 17 observation: lookup-gap datasets legitimately return UNRESOLVED_IDENTITY; UI communicates it clearly — no semantic change made).
- Limitations: artifact remains CANDIDATE_EVALUATED; connection value is free-text (no dropdown — datalist sourcing is a later UX task); planned_start_time accepts past dates (backend chronology guards handle it; explicit past-date warning is a later task).

## EXP-015 — Task 06 stage intelligence (supplementary only)
- Data: real reconstructed executions (543; 350 training-eligible), stage columns at execution level.
- Discrepancy investigation: the task brief's quoted stage facts (staging 148/159 pos, median 0.60, CV 4.92, corr vs total 0.9767; prevalidation 19; transformation 37; data_loading 17; data_loading_1 162; datamart 7; dataloading approval 1) do NOT reproduce from the real files. A systematic scan of ALL timestamp pairs and duration columns (raw-row and execution level, correlation fingerprints included) found no 'staging' stage anywhere in the dataset and no segment matching the quoted statistics. Near-fingerprint hits were numeric artifacts (error-record counts), not stage semantics. REAL execution-level positive counts: data_loading_1 332, prevalidation 127, transformation 110, pre_load 51 (derived span), data_loading 51, datamart_approval 12, dataloading_approval 6.
- Pre_load/data_loading equivalence check: derived span exec_pre_load_start_time → exec_pre_load_end_time equals the literal data_loading duration column on ALL 51 observed rows (max abs diff 0.0) — same phase, two sources; reported as alias (duplicate_of), never as a 6th stage.
- Method: evidence-based classification (AD-19 thresholds: MODELABLE ≥30 pos, ANALYTICS_ONLY ≥6) → 5 MODELABLE, 2 ANALYTICS_ONLY, 0 TOO_SPARSE/NOT_OBSERVED. Estimator: prior median from strictly-earlier training-eligible executions only (loader median if ≥3 same-loader priors, else global prior median); min 3 priors for ANY estimate; low-evidence warning below 10. No per-stage ML (sample sizes do not justify it).
- Leakage: stage history reuses the Task 04 context-row mechanism (context row stage-NA, is_training_eligible=False, appended to the frame; deterministic Start_Time + LDR_Execution_Id ordering). Tests prove current/future poisoned stage values leave history identical, ties are order-independent, and sparse stages never receive estimates.
- Result (live API, real eligible case MigrationSummarySubTableLoaderDataset|TEST|Migration_Summary_Sub_Fact_Table_LDR_Datase_new): total ETA 18.155 s UNCHANGED (total-et-v3-candidate); stage estimates prevalidation 0.663 s (62 priors), transformation 1.514 s (4, low-evidence warning), pre_load/data_loading 5.465 s (23), data_loading_1 1.949 s (11); datamart_approval/dataloading_approval explicitly unavailable (ANALYTICS_ONLY). Stage analytics endpoint reproduces the real statistics (e.g. data_loading_1 median 2.075 s, p90 21.516 s, CV 17.415).
- UI: supplementary Stage Intelligence section (PREDICTED only) + Stage Analytics page with modelability badges, alias disclosure, sparse-coverage annotations, thresholds and NEVER-summed note. Live E2E verified through the real browser + real backend.
- Tests: +14 backend (146 total), +5 frontend (15 total); build green.
- Decision: stage intelligence accepted as supplementary only; stage-sum total NOT implemented (hard rule, test-enforced); the brief-vs-data discrepancy is recorded here for user review, not silently resolved.
- Limitations: robust-median stage estimates (not ML); per-loader evidence can be small at prediction time (warning surfaced); 'staging' does not exist in this dataset; no stage-level uncertainty.

## EXP-016 — Task 07 total-ETA uncertainty + reliability
- Data: serving artifact total-et-v3-candidate (trained on the 262 earliest training-eligible rows) + the 88-row final chronological holdout (2026-07-15 17:48:04.915 → 2026-08-13 14:41:43.809) — the artifact never saw these rows; all strictly before any serving context.
- Residual behavior (n=88): |log1p residual| median 0.690, mean 0.977, max 4.381; raw |residual| median 5.74 s, p90 395 s, max 15,007 s; 81.8% of holdout targets are UNDER-predictions (model skews low). Extreme durations: 15.4 ks actual vs 367 s predicted (log-res +3.73) — the worst case drives q90.
- Methods evaluated (honest chronological validation: calibrate on earliest 53 holdout rows → evaluate coverage on latest 35 rows; never the reverse):
  - log1p abs-residual conformal: q80 1.633 → coverage 0.886, med width 66.8 s (5.3×); q90 2.387 → coverage 1.000, med width 146.4 s (11.6×)
  - raw abs-residual: q90 805 s → coverage 1.000 but med width 817.7 s (65×) — unusable skew amplification
  - relative error: coverage 0.971 @ 0.8, med width 2.4× — decent but calibration semantics depend on actual magnitude (weaker at extremes)
  - scaled-log |res|/(1+pred_log): coverage 0.800 @ 0.8 and 0.914 @ 0.9 — exactly nominal, but chosen against: less robust at n=53 calibration and harder to explain
  - Quantile regression: NOT implemented — with 350 training rows and a heavy-skew target, a separate quantile model would need its own leakage/chronology validation and was judged not justified for the deadline (documented rejection, not omission).
- CHOSEN: log1p-space absolute-residual conformal (full-holdout calibration: q80=1.5653, q90=2.2292, floor 1.5653). Rationale: most stable width behavior across the skew, simple to explain, leakage-safe by construction, and honest split coverage closest to nominal without over-trusting a 35-row evaluation.
- Coverage by history bin (chosen method, cal→eval rows, 90%): bin 0: n=11 coverage 0.909; 1–2: n=6 cov 1.000; 3–4: n=3 cov 1.000; 5–9: n=12 cov 1.000; 10+: n=3 cov 1.000 — no bin degrades; bins are small, so this is directional evidence only.
- Reliability design (deterministic, no fake probability): width factor thresholds (≥12× LOW, ≥9× MEDIUM, else HIGH) + AD-16 history checks (below min → LOW, below 2× min → cap MEDIUM) + v1 calibration-evidence cap (single 88-residual window ⇒ HIGH not claimable). Reference case (ETA 18.155 s, 11 priors): 9.7× width → MEDIUM with two explicit reasons.
- Serving results: PREDICTED gains `uncertainty` {available, method, calibration_version, interval_level 0.9, lower/upper/width, calibration_rows 88, window, interpretation, reliability{label,reasons}}; non-PREDICTED states carry none (test-enforced). Reference: 18.155 s → [1.061, 176.990] @ 90%, width 175.929 s, MEDIUM — point ETA bit-identical to Task 05/06 (regression-tested).
- Tests: +12 backend (158 total), +3 frontend (18 total); build green. Live E2E verified (real backend + real browser): PREDICTED shows ETA + Expected Range + MEDIUM badge + honest interpretation; insufficient-history loader shows no ETA and no range.
- Decision: uncertainty layer accepted as evidence-backed; interval interpretation kept strictly non-probabilistic; recalibration policy left for a later task (calibration is static until then).
- Limitations: n=88 calibration residuals (single window); empirical coverage evidence from a 35-row evaluation slice only; wide intervals are inherent to the skew (median ~10× at 90%); level 0.8 stored but not served; no online/rolling recalibration.

## EXP-017 — Task 08 SHAP explainability for the total ETA

Environment/compatibility investigation (real venv):
- shap not installed initially; `uv add shap` → shap 0.52.0. Side effect recorded: venv re-resolved to Python 3.14.6 (was 3.12.13), sklearn 1.9.1, pandas 3.0.5.
- Behavior-preservation gates after the interpreter move: full backend suite green (175 passed), reference serving identity returns ETA 18.155241 s byte-identical, calibration artifact loads, additivity exact.

Investigation findings (temporary script tmp_shap08_investigation.py, deleted after harvest):
- TreeExplainer on the persisted ExtraTrees/log1p artifact: output shape (n, 32) additive — base + Σshap = model log1p output (error ~1e-15).
- Encoded matrix layout: 32 columns = 29 `num__<feature>` (1:1 with registered numerics) + 3 `cat__def_dominant_calling_type_<VALUE>` one-hots → mapping must SUM the one-hot block back to the registered categorical feature.
- Explainer construction ~1.5–2 s once; per-request SHAP on one row ~40 ms → cached service explainer.

Implementation (permanent):
- backend/src/dltime/explain.py: cached ExplainService; transformer-derived encoded→registered mapping (validated to cover 32 columns/30 features exactly once); local_explanation with request-time additivity gate (≤1e-6, else withheld); DISPLAY_NAMES for all 30 registered features; global_explanation (mean-|SHAP|) over a documented fixed subset.
- inference.py: PREDICTED-path `explanation` block from the SAME feature vector X; failure → warning, ETA/uncertainty untouched. api.py: `explanation` response field + GET /explanation/global (88-row chronological holdout; cached module-level explainer).
- Tests: backend/tests/test_explain_task08.py (17) + frontend (4).

Local explanation — reference serving identity (real data):
- base 3.522817 log1p; top contributions: Global prior mean duration +0.2548 (higher), Loader prior median −0.1468 (lower), Composite prior median −0.1395 (lower), Loader prior mean −0.1000 (lower), Global prior count +0.0941 (higher). Additivity error 4.441e-16. expm1(base+Σ) = 18.1552 s = served ETA.

Global summary (GET /explanation/global, 88 holdout rows):
- Top mean-|SHAP| (log1p space): Global prior duration variability 0.1566, Loader prior mean 0.1405, Dominant calling type 0.1390, Composite prior median 0.1236, Composite prior mean 0.1225. Interpretation: historical-evidence features dominate the model's learned behavior; no causal claim.

Live E2E (real backend :8000 + Prediction Studio UI):
- Reference identity → PREDICTED 18.16 s; uncertainty [1.061 s — 2m57s] MEDIUM unchanged; 'Why this ETA?' renders the 5 real contributors with correct ↑/↓ directions and the model-attribution note (screenshots in transcript).
- INSUFFICIENT_HISTORY identity → no ETA, no uncertainty, no explanation.
- Global endpoint: 200, n_obs=88, deterministic.

Limitations:
- SHAP explains the model's learned behavior — NOT causal effects (wording enforced).
- TreeExplainer on ExtraTrees: contributions are exact for the tree ensemble's log-space output but the base value reflects training-distribution expectations.
- Global summary is descriptive over 88 out-of-sample rows (subset documented in the response payload).

## EXP-018 — Task 09 operational risk engine

Investigation (temporary script tmp_risk09_investigation.py + focused probes, deleted after harvest; real data):
- Interval width/ETA ratio (88-row holdout): p50=10.02, p90=11.48, max=12.85 — already fully encoded by Task 07 reliability → REJECTED as a separate signal (double counting).
- ETA / prior-loader-median: p50=1.79, p90=3.12, max=3.89; share >5x = 0.000 → WARNING threshold 3x (~10% flag rate), HIGH_RISK threshold 5x (unprecedented in real data).
- Robust z=(ETA−median)/prior std: p50=4.72; share z>3 = 38.6% → prior std is not a usable scale → REJECTED.
- OOD via training-range fence: all-feature probe gave spurious max excess 29,605 (skewed hist_* IQRs) → config features ONLY. Config-only fence (7 def_* numeric features, 262-row training window): train flag-rate 0.004, holdout (n=88) flagged 0 — a flag is meaningful. Synthetic probes: component_count x2→18.94, x5→56.44, x10→118.94 (attribution: def_component_count). All-config-missing → excess 0 (covered by the existing definition-match warning).
- Prior-count distribution among 350 eligible: 3–5:71, 6–9:53, 10–19:28, 20+:14; 88 rows lack a positive loader median (ETA-vs-median signal correctly skips them).

Strategy comparison: rule hierarchy vs additive score vs hybrid — hierarchy selected: score inputs overlap (width↔reliability↔history), weights would double-count; no failure labels exist to train a classifier; hybrid adds complexity without evidence.

Implementation:
- backend/src/dltime/risk.py: RISK_POLICY_VERSION="risk-hierarchy-v1"; assess_risk (pure, deterministic); training_range_profile/config_range_excess; insufficient_history_risk; _low_stage_evidence (dataset-MODELABLE availability only; magnitudes never used; never summed).
- Precedence: LOW_CONFIDENCE > OUT_OF_DISTRIBUTION > HIGH_RISK (corroborated) > WARNING > NORMAL. Reasons carry code/message/evidence; evidence keys whitelisted prediction-time values only.
- inference.py: risk on PREDICTED (try/except → warning, ETA/uncertainty/explanation untouched) + opaque INSUFFICIENT_HISTORY label on that state; stage intelligence computed once per request and shared; OOD profile + MODELABLE stage set built once per service. api.py: `risk` response field.
- Tests: backend/tests/test_risk_task09.py (17) + frontend (6).

Reference case (live): PREDICTED 18.155241 s unchanged; uncertainty [1.061, 176.990] @90% MEDIUM unchanged; SHAP additivity 4.441e-16 unchanged; RISK=WARNING — ETA_ABOVE_LOADER_HISTORY "ETA is 4.1x the prior loader median (threshold 3.0x)", evidence {prior_loader_executions: 11, config_range_excess: 0.0, eta_vs_prior_loader_median: 4.078}, policy risk-hierarchy-v1. Repeated requests → identical risk (deterministic).
Case B (live): INSUFFICIENT_HISTORY — risk label INSUFFICIENT_HISTORY only; no ETA/uncertainty/explanation.
UI (live): Operational Risk card with label + evidence-annotated reason + "not probabilities and not failure predictions" note; PREDICTED and INSUFFICIENT-HISTORY variants verified with screenshots.

Limitations:
- Thresholds 3x/5x/0.5 are provisional policy anchored to observed distributions (n=88 holdout), not validated failure predictors; they are documented and configurable.
- No ground-truth failure/risk labels exist; risk cannot claim predictive validity for failures.
- OOD fence uses training-window config ranges; chronological drift may shift ranges over time (recalibration belongs to later monitoring tasks).
- eta_vs_prior_loader_median unavailable for 88/350 eligible contexts without a positive loader median (signal degrades gracefully to other signals).

## EXP-019 — Task 10 loader intelligence / adaptive history / cold-start evidence

Investigation (temporary script tmp_li10_investigation.py, deleted after harvest; real reconstructed data):
- Loader-level eligible-history distribution (88 loaders with any eligible history): 1×35, 2×10, 3-4×22, 5-9×10, 10+×11 (350 eligible rows total).
- Zero-history population: 194 of 229 definition loaders never have an eligible execution; 45 loaders have 1-2 priors (51.1% of loaders with any history are AD-16-insufficient).
- Sufficient loaders (n>=3): 43; all with computable CV (median CV 0.20, p90 0.89); loader-median p50 7.43 s, p90 994.6 s.
- Stability (n>=6 loaders, n=19): old-half vs new-half median ratio p25 0.96 / p50 1.04 / p75 1.33; 78.9% within [0.5,2.0] → trend display justified at n>=6, labeled from the ratio.
- Granularity: 35 loader+connection pairs (NO loader uses >1 connection); 96 loader+dataset pairs (only 5 loaders use >1 dataset); 38 composite identities (19 with >=3).
- Cold-start contextual feasibility: 1-2-prior loaders have median 0 same-dataset other-loader rows (only 11% >=3); never-executed loaders' definition datasets overlap executed datasets in only 7/177 cases → rich similarity intelligence is NOT supported by this data; only same-dataset context when >=3 real rows exist.
- EXP-012 history-bin holdout coverage unchanged (bin0 0.909 n=11; others 1.000, small n) — recorded, not replaced.

Implementation:
- backend/src/dltime/loader_intel.py: build_loader_intel (read-only over the reconstructed frame + definitions), evidence_level_for, honesty rules (typical stats withheld <3; trend >=6), cold_start section (contextual only when >=3 same-dataset other-loader rows; explicit no-evidence note otherwise), definition context. LOADER_INTEL_VERSION=loader-intel-v1.
- api.py: GET /loaders/{loader_name} (URL-decoded exact match; no second identity system).
- Frontend: LoaderIntelligence page (search, evidence-level badge, ETA-gate explanation, history facts, context, cold-start section with disclaimers, limitations), App nav third tab, typed fetchLoaderIntelligence.
- Tests: backend 13 (level boundaries, strong/sufficient/emerging/no-history/unknown, multi-dataset, single-execution honesty, min-history eligibility, URL decoding, read-only regression on /predict T04-T09 outputs) + frontend 5 (encoding+STRONG render, EMERGING withholding, contextual-not-prediction, no-context note, API error).

Live E2E (real backend :8000 + real UI, screenshots in transcript):
- Reference /predict unchanged: ETA 18.155241 s, uncertainty [1.061, 176.990] MEDIUM, SHAP additivity 4.441e-16, risk WARNING, stages available.
- /loaders/STRONG: STRONG, n=11, median 4.452, CV 0.883, trend stable, ETA-eligible.
- /loaders/cold-with-context: EMERGING, n=1, no typical stats, contextual 15-row same-dataset evidence with NOT-a-prediction disclaimer.
- /loaders/2210_TEST_ALL_CONFIG: NO_HISTORY + definition context (50 rows, A_TABLE..E_TABLE) + explicit no-context note.
- /loaders/unknown: NO HISTORY, unknown-to-definitions badge, no-evidence note.

Limitations:
- Evidence-level thresholds are provisional (documented); the ETA gate stays AD-16=3 regardless.
- Trend basis is a coarse half-vs-half median comparison; per-execution trend lines belong to Task 12 (trends).
- Contextual cold-start evidence exists for few loaders (real data property, stated in responses).
- No similarity scoring implemented (would fabricate authority); deferred unless explicitly approved.

## EXP-020 — Task 10.1 product refinement pass (mandatory Connection + presentation hierarchy)

**Scope:** UI/contract refinement ONLY; zero engine changes (no model, features, calibration, SHAP, risk, stage, or eligibility modifications).

**Mandatory Connection (AD-18 revised, user-approved):**
- `PredictionRequestBody.connection` now required (HTTP 422 on missing/blank); `PredictionRequest.connection` required — service-level defense-in-depth returns INVALID_INPUT `missing_required_field: connection` even if the HTTP schema is bypassed. AD-04 definition resolution remains the single source of truth; supplied connection is validated against it (disagreement ⇒ INVALID_INPUT, unchanged); input never overrides resolution.
- All 6 existing backend test files updated for the required field; +3 new tests (schema-level 422, service-level INVALID_INPUT, wrong-connection still INVALID_INPUT).

**Presentation hierarchy (PREDICTED):** ETA → model/evidence strip + explicit "Model status: CANDIDATE_EVALUATED — evaluated candidate, not production-final" note → Stage Intelligence (unavailable stages OMITTED from the table; single "N stages are omitted — insufficient historical evidence" note; backend stage payload unchanged) → Expected Range + Reliability → "Why this ETA?" → Operational Risk ONLY when label ≠ NORMAL (alert semantics; the API risk block itself is unchanged and still present on every PREDICTED response).

**Human-readable explanation:** new `human_readable_summary` field in the explanation block, generated deterministically in `explain.py` from the ACTUAL top-ranked SHAP entries (strongest signal with direction + real value; second same-direction real entry; strongest opposing signal; explicit "attributions, not causes" clause). No new computation; exact SHAP values, additivity gate, output-space semantics, and the full ranked list (now in a collapsed "Show technical model contributions" detail) unchanged. Reference-case summary verified live.

**Reference E2E (live backend + browser):** ETA **18.155241123720305 s — byte-identical**; interval [1.061, 176.990] @ 90%; reliability MEDIUM; SHAP additivity 4.441e-16; risk WARNING (ETA_ABOVE_LOADER_HISTORY); 11 priors; resolved MDM_DM_TEST; hierarchy order verified in the DOM (Stage Intelligence → Expected Range → Why this ETA → risk alert). NORMAL case verified live: `bb_sample/DEV/bb_sample/QA_DM_PSQL` (exactly 3 priors) → PREDICTED ~80.5 s, NO risk card, ETA/range/summary intact. Missing-connection case verified live: client error, no network request.

**Manual matrix corrections (data-verified):** TC-02/03/05/07 identities from the old matrix (`DGE Migration Summary Main Fact Table Loader_new`, `LDR_TAM_TABLE_001`, `MigrationSummaryMainFactTableLoaderNoTf`) do NOT resolve (AD-04 lookup-gap family) — replaced with verified identities: `bb_sample/QA_DM_PSQL` (n=3, PREDICTED/NORMAL), `TST_TST_NEW_ONE/PFM_Datamart` (n=2, gate boundary), `ACS_LDR_DATE_01/PFM_Meta` (n=2). Ambiguous-identity manual case explicitly documented as unavailable (0 AMBIGUOUS in production data; covered by the synthetic-resolver unit test).

**Tests/build:** backend 208 passed (208 = 205 + 3, verified in segments); frontend 35 passed; `npm run build` green.

## EXP-021 — Task 11 linkage investigation + operational-loop verification

Date: 2026-09-18. Method: direct inspection of the reconstructed execution frame (read-only) + live end-to-end run of the implemented loop.

### A. Available identifiers / linkage evidence
- `LDR_Execution_Id`: 543/543 executions, UNIQUE — but a post-execution server identifier; cannot exist at prediction time. Not usable to identify a future execution pre-execution.
- `ldr_workflow_execution_id`: present 543/543, 466 DISTINCT values → the caller-side reference a user/operator knows when submitting the job. Selected as the ONLY defensible pre-execution↔post-execution join key (EXACT_WORKFLOW_EXECUTION_ID).
- Canonical identity + time: UNSAFE as a fallback. 310/543 executions (57.1%) share their identity + calendar day with ≥1 other execution; worst identity-day holds 32 executions. A ±24h window would be guesswork; forbidden by PART C. → UNMATCHED instead of guessing.
- `planned_start_time` behavior: defaults to latest recorded start + 1 day (serving contract) — reinforces that prediction-time timing is synthetic, never evidence.

### B. Prediction event store verification (live)
- Reference identity (11 priors): PREDICTED → event persisted with request (incl. caller reference WF-E2E-11-001), identity, prediction (ETA byte-identical 18.155241123720305), SHAP digest (additivity 4.4e-16), 7-row stage evidence, full 30-feature audit vector.
- 2210_TEST_ALL_CONFIG + correct connection `test1`: INSUFFICIENT_HISTORY → event persisted, no ETA/uncertainty/SHAP; wrong connection first rejected INVALID_INPUT (mandatory-connection contract intact).

### C. Ingestion / linkage / error verification (live)
- Actual 20.4 s with reference WF-E2E-11-001 → MATCHED via EXACT_WORKFLOW_EXECUTION_ID; signed −2.244759 s; APE 11.0037% (within-20/30%/2× true, within-10 false — arithmetic verified); 5 stage errors computed, approval stages not scored.
- Duplicate submission → `duplicate: true`, same feedback_id, no new row. Conflicting value → HTTP 409 conflicting_actual_outcome, store unchanged.
- Outcome with never-predicted reference → UNMATCHED, prediction_event_id null, no fabricated error.
- Non-golden status → training_eligible=false, reason UNKNOWN_STATUS (existing Task 01 chain, reused not reinvented).
- Monitoring summary after the flow: attempts 1 / PREDICTED 1 / outcomes 2 / matched 1 / unmatched 1; MAE 2.244759 s, MAPE 11.0037%, tolerances 0/100/100/100; training feedback 0 eligible / 1 not-eligible (UNKNOWN_STATUS) — all recomputed from the store, labeled "NOT Task 03 chronological model evaluation". Loader filter verified both ways.

### D. Store hygiene finding (fixed during the task)
- Initial backend runs polluted the real `backend/store/` with 71 test-generated events (non-monitoring test files exercise the real predict path; store location was module-level). Fixed: autouse conftest fixture redirects the store per-test (hermetic); `resolve_store_dir()` resolves at call time; polluted files deleted and re-verified (full backend suite leaves 0 store files). Live servers use `backend/store/` normally.

### E. Regression
- Backend 239 passed (161 core + 34 T08/T09 + 44 T10/T11, hermetic); frontend 39 passed; build green. Reference ETA/interval/SHAP/risk unchanged.

## EXP-022 — Prompt 1 of 5 / Module 1: Data Foundation audit (investigation-only)

**Mode:** read-only inspection + test locking. NO production code changed. All facts produced by code against the real `train/data` sources (sha256-provenanced loads).

### A. Pre-execution field audit (complete classification)

| Field | Source | Classification |
|---|---|---|
| `sprint_name`, `ldr_connection_name`, `loader_display_name`, `datasetname` | both definition files | **Mandatory identity input** (PRE_EXECUTION; product-facing Sprint/Connection/Loader/Dataset) |
| `preval_trans_display_name` | detail | Additional legitimate pre-execution input (PRE_EXECUTION) — free-text component label, 720 distinct / 1480 rows; NOT promoted to a feature (see D) |
| `type`, `calling_type`, `execution_order` | detail | Additional legitimate pre-execution input (PRE_EXECUTION) → already feature-ized (def_type_nunique, def_dominant_calling_type, def_execution_order_*) |
| `prevalidation_count`, `transformation_count` | summary | Additional legitimate pre-execution input (PRE_EXECUTION) → def_prevalidation_count / def_transformation_count |
| def_* (8) | engineered | DERIVED_FROM_PRE_EXECUTION features |
| hist_* (22) | engineered | DERIVED_FROM_POST_EXECUTION, strictly-prior only |
| `Start_Time`, `End_Time`, `Time_Taken`, `LDR_Status`, record/error counters, all stage timing/duration columns (18) | executions | POST_EXECUTION / leakage — forbidden as prediction inputs (test-enforced) |
| `timetaken_in_sec` | executions | TARGET |
| `LDR_Execution_Id`, `ldr_workflow_execution_id`, `Loader_Name`, `Sprint`, `Dataset_display_name` | executions | IDENTIFIER (identity components / linkage-only; never model features) |
| — | — | `Planned Start Time` is NOT a source field (product-level optional input only; verified absent from all three files) |

### B. Reconstruction / resolution / target / eligibility — revalidated, all unchanged

- Reconstruction: 792 raw → 543 executions; 74 multi-row; max 9 rows/exec; 469 single-row; 323 rows in multi-row execs; 541 with target.
- AD-04 resolution (execution level): 274 RESOLVED / 269 UNRESOLVED / 0 AMBIGUOUS.
- Eligibility tiers: 535 valid / 533 target-valid / **350 training-eligible** (exactly the previously recorded count; no drift).
- Training frame: 350×30; extremes retained (17 rows >100,000 s; max 43,000,000 s) — anomaly handling never drops legitimate long runs.

### C. Stage evidence — CORRECTED and strengthened (changes the recorded pre_load fact's scope, not its conclusion)

- `pre_load` derived span == `data_loading_timetaken_in_sec` literal: **75/75 raw component rows exact (max abs diff 0.0)**, execution level 51/51; prior recording cited 51 execution-level observations only.
- NEW: the two measurements ALWAYS co-occur (75 rows with pre_load window ↔ 75 rows with dl literal; identical row support; same 51 executions) — same phase, two sources.
- NEW: the equality is of duration VALUES, not of time windows — wherever both windows exist (40 rows), the pre_load window **strictly precedes** the data_loading window (pre_load_end ≤ data_loading_start on 40/40; median gap 35.7 s; max gap 406,748 s). The data_loading literal does NOT match its own timestamp span (0/40 exact) — the literal is the authoritative duration; the pre_load window is a separate (earlier) window that happens to carry the same duration value. `pre_load` therefore remains a **verified alias of the data_loading duration, never a fifth independent stage** — and additionally must never be interpreted as the same time window.
- data_loading_1 (second loading stage) is independent: never equal to dl (0/40), always ≤ total (421/421), and its value equals the dl timestamp-window span on 415/422 rows (a different measurement source again).

### D. Stage/total relationship conclusion (for Prompt 2)

- Stages DO NOT partition the total: every stage ≤ total on ALL co-observed rows (pv 289/289, tf 118/118, dl 75/75, dl1 421/421); pv+dl+dl1 covers a median of **4.5%** of total (n=5 rows with all three); tf+dl1 covers median 23.9% (n=64).
- Stage windows are not sequential: prevalidation_end > transformation_start on **85/85** co-observed rows (overlapping windows).
- **Conclusion: the five operational stage targets are overlapping, partially-observed components of the execution — the total ETA must be modeled DIRECTLY (as Task 03/04 already do); stage predictions must never be summed into it.** Prompt 2 should treat each of the 5 stages as an independent supervised target (evidence-gated), not as decomposed parts of the total.

### E. Feature inventory conclusion (30-feature contract stands)

- All 8 definition-derived features map to legitimate pre-execution fields; all 22 historical features are strictly-prior (chronology tests + new mid-history poison probe at the context instant).
- Sparse evidence is handled by explicit NA (172 full-frame / 88 training-row zero-loader-history executions) — no fabrication.
- NO feature additions justified in Module 1: the only unfeature-ized pre-execution field is `preval_trans_display_name` (720 free-text labels; component identity is already captured by def_component_count/def_type_nunique; the median-lookup concern does not apply since the model learns over 30 features incl. 22 history stats — confirmed EXP-012 evidence that trees exploit them non-trivially). Adding it now would add cardinality without evidence and alter the locked serving artifact (30-feature contract). Re-evaluation belongs to Prompt 4's laboratory with chronological evidence, not to this module.

### F. Tests added

`backend/tests/test_foundation_module1.py` — 31 tests locking: full source-field classification (incl. `ldr_workflow_execution_id` = IDENTIFIER/linkage-only, never a feature; `Planned Start Time` absent from sources), pre_load alias value-equality at raw (75/75) and execution (51/51) level, pre_load window precedence (40/40), stage containment (no stage > total), non-partition (sum ratio < 0.2), window overlap (85/85), approval stages excluded as targets, reconstruction/resolution/eligibility counts, 350×30 frame with extremes retained, exactly-30 feature contract with allowed categories only, identity components never features, first-execution zero priors, zero-history → NA (172/88 scopes), loader-count spot checks vs independent recomputation, context-row history isolation (expected == 11), mid-history future-poison probe (bit-identical), no post-execution columns in X, long-tail retention.

**Result:** backend 270/270 passed (179+60+31 segments; ~2.5 min + ~8 min). Frontend untouched (no frontend changes in Module 1 → frontend suite/build not re-run; last green state 39/39 + build green stands). Reference prediction untouched (no serving-path change; verified by the full suite incl. byte-identical reference tests).

## EXP-023 — Prompt 2 of 5: Total + 5-Stage Intelligence Engine experiments

**Mode:** chronological experiments on the real 350-row training population (expanding-window CV, 5 folds, + final chronological holdout 262/88; no random splits; all runs via the production `modeling.py` path). Temporary experiment scripts deleted after evidence harvest.

### A. Total-model comparison (pooled OOF / holdout, raw seconds)

| model | tgt | pooled MAE | pooled MedAE | pooled RMSE | |20% | |2x | holdout MAE | holdout MedAE | holdout |2x |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_global_median | log1p | 507.1 | 5.49 | 2299.1 | 16.0 | 30.3 | 764.3 | 4.67 | 38.6 |
| baseline_global_mean | log1p | 521.9 | 33.57 | 2293.3 | 2.3 | 9.7 | 779.1 | 29.64 | 4.5 |
| ridge | log1p | ~1e100 (exploded) | 16.66 | ~1e100 | 2.9 | 16.6 | 719.3 | 15.15 | 18.2 |
| random_forest | log1p | 510.0 | 23.70 | 1642.1 | 20.0 | 37.7 | 707.7 | 21.24 | 30.7 |
| **extra_trees_candidate** | **log1p** | **364.6** | **5.16** | **1711.4** | **16.0** | **48.6** | **696.7** | **5.74** | **35.2** |
| extra_trees_default | log1p | 1856.0 | 7.89 | 13659.9 | 15.4 | 37.1 | 698.8 | 11.55 | 20.5 |
| hist_gradient_boosting | log1p | 491.2 | 15.96 | 1850.7 | 12.0 | 31.4 | 733.8 | **2.95** | **62.5** |
| (all raw-target variants) | raw | 3e5-3e7 | 2.5e5-6.8e5 | - | <=10.9 | <=15.4 | - | - | <=3.4 |

Per-fold MedAE (stability): ET [39.63, 1.94, 7.43, 5.19, 1.32] vs HGB [15.96, 3.75, 194.09, 61.92, 3.03] — HGB's attractive holdout numbers hide catastrophic fold instability; ET selected on the full metric set (accuracy, median case, tail behavior, chronological stability, inference cost, SHAP compatibility). XGBoost/LightGBM/CatBoost: not installed in the environment (reported, not installed).

### B. DEFENSIBILITY: learned model vs median lookup (pooled OOF, log1p space)

| population | n | ET MedAE | ET MAE | ET |2x | median-baseline MedAE | MAE | |2x |
|---|---|---|---|---|---|---|---|
| ALL OOF rows | 175 | 5.16 | 364.6 | 48.6% | 5.49 | 507.1 | 30.3% |
| **SERVED (>=3 priors)** | **80** | **2.41** | **70.7** | **56.2%** | **5.57** | **109.4** | **22.5%** |
| seen (>0 priors) | 126 | 3.81 | 219.3 | 54.0% | 5.55 | 422.2 | 25.4% |
| gated out (<3) | 95 | 8.70 | 612.1 | 42.1% | 5.49 | 842.1 | 36.8% |

Per-bin (served rows): 3-4 priors ET MedAE 6.19 vs median 5.74; 5-9 priors ET 1.30 vs 4.97; 10+ ET 4.21 vs 5.06. The learned model's advantage concentrates exactly where the gate serves.

### C. Slices (ET log1p, pooled OOF)

History bins: 0 → n=49, MedAE 15.10, |2x| 34.7%; 1-2 → n=46, 8.19, 50.0%; 3-4 → n=28, 6.19, 35.7%; 5-9 → n=44, 1.30, 70.5%; 10+ → n=8, 4.21, 50.0%. Seen/unseen: seen MedAE 3.81 (n=126) vs unseen 15.10 (n=49). Context availability: composite-hist>0 MedAE 3.52 (n=121); loader+dataset>0 3.67 (n=124); loader+conn>0 8.61 (n=66); def-match rows 14.82 (n=86).

### D. Feature experiments

Leave-one-group-out (pooled MedAE / MAE): ALL-30 = 5.16 / 364.6. Removing PRE_EXECUTION_CONFIG → 5.88 / 659.9 (MAE +81%); LOADER → 6.01 / 360.0; LOADER_CONNECTION → 6.22 / 351.2; LOADER_DATASET → 6.15 / 363.8; COMPOSITE → 5.92 / 361.9; PREVIOUS → 5.26 / 375.0; GLOBAL → 4.71 / 443.2 (MedAE better, MAE +22% worse — tail cost). def-only-8 → MedAE 30.93; loader-hist-only-4 → MedAE 2.73 but MAE 467.8 (lookup behavior — the full 30-feature model is what generalizes). `preval_trans_display_name` as `def_preval_name_nunique` (coverage 129/350): MedAE 5.16 → 4.69, MAE 364.6 → 363.3 — promising but NOT adopted (would change the locked 30-feature serving artifact); referred to the Model Laboratory phase. ET config grid: msl=2/n_est=200 confirmed best balanced (msl=1 unstable, msl=5 slightly better MedAE but much worse MAE/RMSE).

### E. Stage modeling (per-stage chronological holdout: 70% train / 30% test of stage-positive rows)

| stage | train-eligible positives | loaders | test n | A: prior-median MedAE/MAE/|2x | B: ET-30 MedAE/MAE | C: ET-30+stage-hist MedAE/MAE |
|---|---|---|---|---|---|---|
| prevalidation | 62 | 22 | 19 | **0.28 / 2424.6 / 63.2%** | 15.17 / 2573.96 | 11.17 / 2539.5 |
| transformation | 58 | 19 | 17 | **0.13 / 3.53 / 64.7%** | 0.46 / 3.30 | 0.57 / 3.49 |
| data_loading (=pre_load) | 23 | 16 | 7 | 5.50 / 25.38 / 0% | 0.25 / 23.14 | 0.31 / 23.17 |
| data_loading_1 | 321 | 79 | 96 | **1.04 / 8.48 / 71.9%** | 1.16 / 12.79 | 0.93 / 12.09 |

Learned stage models REJECTED: worse or statistically inconclusive on every stage (data_loading n=7 cannot support any conclusion — stated honestly). Estimator A also produces estimates on far more rows (evidence-gated global fallback) while B/C are silent on sparse contexts.

### F. Cold-start context evidence

Of 184 gate-failing training rows: 58 (25 distinct loaders) have >=3 same-CONNECTION other-loader eligible rows; 35 have >=3 same-dataset rows. Connection context therefore materially extends Loader Intelligence coverage (implemented, strictly CONTEXTUAL). String-similarity loader matching evaluated and rejected: loader names are free-text (e.g. 'bb_sample', 'TST_TST_NEW_ONE') with no defensible similarity semantics.

**Test lock:** `backend/tests/test_stage_intel_prompt2.py` (17 tests: EXP-012 reproduction, learned-vs-median proof, baseline reporting, 262/88 holdout, optional-modeler honesty, stage-intel tiers, alias suppression, approval exclusion, learned-stage rejection direction, serving payload, reference ETA byte-identical, mandatory connection 422, wrong connection, insufficient-history, unresolved identity). Full suite: backend 287/287 (segments), frontend 40/40, build green. Reference ETA byte-identical: 18.155241123720305.

## EXP-024 — Prompt 3 of 5: Reliability + Explanation + Risk + Monitoring validation

**Calibration provenance revalidation (§2).** `unc-log1p-conformal-v1` was audited, not assumed:
its recorded window (2026-07-15 17:48:04.915 → 2026-08-13 14:41:43.809, n=88) is exactly the final
chronological holdout strictly after the serving artifact's training window (262 earliest
training-eligible rows), and `build_calibration` hard-refuses an overlapping window. Rebuilding the
calibration from the CURRENT Prompt-2 candidate reproduced q90=2.2291534416514427 and
q80=1.5653350845931193 exactly (assertion-locked). Verdict: model/artifact/population/feature
construction all compatible — **no recalibration; artifact and version preserved.**

**Coverage evidence (§4, n=88 holdout; calibrate-window residuals → same-window coverage as a
revalidation exercise; small slices directional only).**
| slice | n | 90% coverage | 90% med width | 80% coverage |
|---|---|---|---|---|
| overall | 88 | **0.909** | 111.5 s | **0.807** |
| history 3–4 | 14 | 0.857 | 101.7 s | 0.857 |
| history 5–9 | 28 | 1.000 | 102.9 s | 1.000 |
| history 10+ | 6 | 1.000 | 88.4 s | 1.000 |
| history 1–2 | 21 | 0.952 | 105.2 s | 0.810 |
| seen loaders | 69 | 0.957 | 103.9 s | 0.913 |
| unseen loaders | 19 | **0.737** | 187.5 s | **0.421** |

The ONLY sub-nominal slice (unseen loaders, 0 prior loader history) is structurally unreachable on
the served path: 0 priors → INSUFFICIENT_HISTORY → no ETA → no interval. n=19 is too small to
justify threshold changes; recorded as a limitation, not silently averaged away. Overall 90%
coverage 0.909 ≈ nominal 0.90 on n=88. Interval construction: log1p-symmetric, lower bound clamped
at 0 (durations cannot be negative), point always inside, finite-value guards.

**SHAP revalidation (§6/§8).** Additivity: reference-case max |base+Σcontrib−output| = 4.44e-16;
<1e-6 across 20 chronological holdout rows (new test). Mapping: 32 encoded columns → 30 registered
features, each encoded column mapped exactly once. Banned fields (workflow reference, execution id,
target, post-execution counters) disjoint from the explained feature set. Global explainer uses the
same serving artifact + preprocessing (unchanged from Task 08).

**Risk revalidation (§10–13).** Hierarchy and thresholds unchanged (risk-hierarchy-v1, EXP-018
evidence; 0/88 holdout contexts OOD-flagged at 0.5 IQR-units). Added the §12 interpretation
disclaimer to every payload: "operational attention level … not a probability of failure and not a
statement about model confidence" (PREDICTED + INSUFFICIENT_HISTORY paths). State isolation
re-tested: non-prediction states carry no risk; INSUFFICIENT_HISTORY carries no ETA.

**Monitoring hardening (§14–18).** Linkage contract re-verified: EXACT_WORKFLOW_EXECUTION_ID only,
no time-window method exists in feedback.py (source-inspection test); identity guard rejects
mismatched references; shared references → AMBIGUOUS; duplicates idempotent; conflicts 409. New:
the caller reference is persisted in the prediction-event identity block (metadata only, never a
feature — test-enforced). Metrics extended with rmse_seconds, mean_signed_error_seconds (bias),
explicit denominators {matched_pairs, pairs_with_percentage_error} (zero/non-positive actuals
excluded from percentage math, never silently dropped), slices by reliability / history bucket
(EXP-012 bins) / loader, and per-stage MAE/MedAE with available_on_matched/scored_on_matched
coverage. Empty populations → metrics null (no fabricated zeros).

**Reference E2E.** PREDICTED 18.155241123720305 / total-et-v3-candidate / 11 priors /
[1.061, 176.990] @ 90% / MEDIUM / WARNING / SHAP additivity 4.44e-16 — byte-identical.

**Test lock:** `backend/tests/test_reliability_prompt3.py` (28 tests: calibration compatibility +
interval contract, coverage revalidation incl. served-path slices ≥0.857, SHAP additivity/mapping/
banned-features, contribution-language policy, stage honesty, risk contract + disclaimer, identity-
block linkage reference, monitoring denominators/slices/empty-populations/no-time-window, §21
section contract, five-state regression).

## EXP-025 — Prompt 4 of 5 / Module 7: Model Laboratory evidence (governed evaluation; nothing adopted)

All numbers below were **reproduced during the Prompt-4 verification audit** from the deterministic
chronological pipeline (ExtraTrees log1p, n_estimators=200, min_samples_leaf=2, max_features=1.0,
random_state=42; expanding-window OOF n=175 over 262 training rows + final holdout n=88; total
training-eligible frame 350 rows). No serving artifact, feature contract, or calibration changed.

### Feature experiment: `def_preval_name_nunique` (§9) — NOT adopted, retained as lab evidence

Derived as-of from the legitimate source field `preval_trans_display_name` (definition data,
available pre-execution, deterministic, leakage-safe; coverage 129/350 rows). Feature-set version
`pre-execution+historical-v31+preval_name_nunique`. Registry: `EXP-LAB-025-FEATURE` (status EVALUATED).

Pooled chronological (OOF, n=175):

| metric | baseline (30) | candidate (31) | direction |
|---|---|---|---|
| MAE | 364.6 | 363.3 | candidate ≈ neutral-better |
| MedAE | 5.16 | 4.69 | candidate better |
| RMSE | 1711.4 | 1708.5 | ≈ neutral |
| ±20% | 16.0% | 20.6% | candidate better |
| within-2× | 48.6% | 49.1% | ≈ neutral |

Fold MedAEs — baseline [39.63, 1.95, 7.43, 5.19, 1.32]; candidate [33.79, 1.38, 8.36, 4.66, 1.22]
(candidate better on folds 1/2/4/5, worse on fold 3: 7.43→8.36).

Chronological holdout (n=88): baseline MAE 696.7 / MedAE 5.74 / RMSE 2870.5 / ±20% 10.2 / 2× 35.2;
candidate MAE 705.2 / MedAE 5.31 / RMSE 2898.3 / ±20% 11.4 / 2× 39.8 — mixed (MedAE/±20%/2× better,
MAE/RMSE worse); n=88, directional only.

Served population (≥3 prior eligible executions, n=80) — **MIXED, decisive neither way**:

| metric | baseline | candidate |
|---|---|---|
| MAE | 70.7 | 61.0 (better) |
| MedAE | 2.41 | 2.64 (worse) |
| RMSE | 272.1 | 221.7 (better) |
| ±20% | 25.0% | 33.8% (better) |
| within-2× | 56.2% | 57.5% (≈ neutral-better) |

Known loaders (n=126): MedAE 3.81→3.70, MAE 219.3→217.1, RMSE 847.8→870.7 (mixed, ≈ neutral).
Unseen loaders (n=49): MedAE 15.10→12.16, ±20% 4.1→8.2, 2× 34.7→34.7, MAE 738.3→739.0 (MedAE better).

History buckets (OOF, MedAE): 0 priors 15.10→12.16 (better); 1–2: 8.19→7.44 (better); 3–4:
6.19→6.43 (worse); 5–9: 1.30→1.21 (better, ±20% 27.3→40.9); 10+: 4.21→3.61 (better; n=8, too small
to conclude).

**Decision (recorded, §38 — no single-metric claims):** the candidate improves most median-sensitive
and low-history slices but *worsens the served-population MedAE (2.41→2.64) — the population the
system actually serves — and adoption would change the locked serving artifact*. Retained in the
Model Laboratory as EVALUATED experimental evidence; NOT adopted, NOT promoted. Any adoption is a
human governance decision via the §15/§16 promotion gates.

### Feature-group ablation (§10, `run_group_ablation`, chronological contract identical to §9)

| ablation | n_feat | pooled MedAE | pooled MAE | pooled RMSE | ±20% | within-2× | holdout MedAE |
|---|---|---|---|---|---|---|---|
| ALL_FEATURES (baseline) | 30 | 5.16 | 364.6 | 1711.4 | 16.0 | 48.6 | 5.74 |
| − CONFIGURATION | 22 | 5.88 | **659.9** | **2328.1** | 23.4 | 42.3 | 4.53 |
| − GLOBAL history | 26 | **4.71** | 443.2 | 2115.4 | **33.1** | **53.7** | **2.77** |
| − LOADER history | 26 | 6.01 | 360.0 | 1688.3 | 18.3 | 46.9 | 6.63 |
| − LOADER+CONNECTION | 26 | 6.22 | 351.2 | 1660.6 | 16.0 | 46.3 | 5.10 |
| − LOADER+DATASET | 26 | 6.15 | 363.8 | 1692.6 | 17.1 | 46.3 | 7.49 |
| − COMPOSITE | 26 | 5.92 | 361.9 | 1685.8 | 17.1 | 45.1 | 8.14 |
| − PREVIOUS execution | 28 | 5.26 | 375.0 | 1722.4 | 23.4 | 46.9 | 6.16 |

Interpretation (honest, per §38 — measured during the audit, mixed results stated as mixed):

- **Pooled MedAE degrades for every removal except GLOBAL** (config 5.16→5.88, loader →6.01,
  loader+conn →6.22, loader+dataset →6.15, composite →5.92, previous →5.26).
- **GLOBAL removal is a genuine tradeoff, not a win:** MedAE 5.16→4.71, ±20% 16.0→33.1,
  within-2× 48.6→53.7 and holdout MedAE 5.74→2.77 all improve, but MAE 364.6→443.2 and
  RMSE 1711→2115 degrade sharply — global features buy long-tail robustness for a heavily
  right-skewed target at the cost of median-case accuracy. GLOBAL is kept; this is a documented
  judgment on the mean/median tradeoff (§11), not a single-metric conclusion.
- **CONFIGURATION removal shows the same skew pattern in mirror image:** holdout MedAE improves
  (5.74→4.53) and ±20% rises (16.0→23.4) while pooled MAE nearly doubles (364.6→659.9) and RMSE
  jumps to 2328 — the configuration group carries tail protection.
- LOADER+CONNECTION removal slightly *improves* MAE/RMSE (351.2/1660.6) while degrading MedAE —
  recorded as measured; no group is removed on this evidence (§38: no single-metric decisions).

All figures are exact reproduced chronological results (same contract as the §9 experiment).

### Drift (§21–25) — EXP-025 audit run

Reference baseline = the serving model's chronological training window (262 rows, 2025-10-10→
2026-07-15). Current window = all executions after the reference end (n=137, of which 88
training-eligible) — model-unseen by construction, no future mixing. Methods: two-sample KS for
numeric features (PSI also implemented), max-proportion-change + total-variation for categoricals.
Thresholds KS ≥0.15 WARNING / ≥0.30 SIGNIFICANT — **project-defined heuristics**, not universal
truths. Window <20 or reference <30 rows → INSUFFICIENT_SAMPLE (never a verdict).

Findings (signals, not model failure): feature drift SIGNIFICANT overall — `hist_global_prior_count`
KS 1.00 (structural: later rows necessarily have longer prior histories) and `def_execution_order_mean`
KS 0.42; population drift SIGNIFICANT — loader TV 0.91, dataset TV 0.89, sprint TV 0.199 (WARNING),
history-availability NO_MEANINGFUL_DRIFT — the holdout period introduces new loaders/datasets,
expected in this deployment.

### Prediction-performance drift (§21C/§9) — INSUFFICIENT_SAMPLE

0 matched prediction→actual outcomes exist (no production predictions linked to actuals yet; live
minimum 20 required). No performance-drift metric computed; no claim made. Performance drift can
only ever be calculated on matched actuals — enforced by code and tests.

### Retraining readiness (§26) — RETRAINING_RECOMMENDED_FOR_REVIEW (human gate)

Signals: SIGNIFICANT feature + population drift → recommendation signal fires. Not-eligible signals:
0 new eligible actuals beyond training (needs ≥30), performance degradation not assessable (no
matched actuals), calibration degradation not assessable. `automatic_retraining=OFF` always; the
state recommends a human **review**, not retraining itself.

### Eligibility policy status (§6)

Current behavior formalized as policy **v1: completed-status training eligibility** (ELIGIBLE
requires a valid execution chain and completed status; non-golden statuses → UNKNOWN_STATUS → not
eligible; 350/533 target-valid rows eligible). Changing the status→eligibility mapping is a
controlled model-governance decision requiring explicit human approval before any retraining under
a new policy. Recorded in AD-28.

**Reference E2E (post-Module-7):** PREDICTED 18.155241123720305 / total-et-v3-candidate / 11 priors /
[1.061, 176.990] @ 90% / MEDIUM / WARNING / SHAP additivity 4.44e-16 — byte-identical. Registry
`/models` empty (no promotion ever performed; serving model remains the Prompt-2 candidate outside
the registry). **Test lock:** `backend/tests/test_model_lab_prompt4.py` (33 tests: registry
lifecycle, promotion/rollback/compatibility gates, drift incl. small-sample guard, readiness,
ablation/feature-experiment contracts, leakage boundary) + 3 frontend Model Laboratory tests.

## EXP-026 — Prompt 5 of 5 / Module 8: end-to-end validation record (no model changes)

**E2E chain (test_module8_e2e.py, 8 tests, real artifact + real sources, hermetic store):**
- Valid chain: PREDICTED, identity RESOLVED, 11 priors, 30/30 registry features, ETA 18.155241123720305 (abs=0.0), interval [1.061, 176.990] @ 90% finite (lower ≥ 0, point inside), reliability MEDIUM with reasons, `unc-log1p-conformal-v1`, 7-stage payload with `pre_load` suppressed (`duplicate_of: data_loading`, estimate null; approval stages unavailable; no `total` key), SHAP additivity ≤ 1e-6 with base+Σ≈output and "not causal" interpretation, risk interpretation disclaimer, `total-et-v3-candidate` metadata, event persisted with the exact 30-feature vector.
- Negative states: INSUFFICIENT_HISTORY (2210_TEST_ALL_CONFIG identity), UNRESOLVED_IDENTITY, INVALID_INPUT (connection disagreement), schema 422 (missing connection) — no ETA/uncertainty/explanation; events persisted with `prediction: null`.
- Future-row poisoning: 2 strictly-future eligible-looking executions (43M s targets, distinct identity) → ETA byte-identical, priors still 11.
- Fail-closed history: eligibility-evidence-free frame → 0 usable priors → INSUFFICIENT_HISTORY (no fabricated history).
- Lab safety (API): serving model identity = `total-et-v3-candidate`; promoted-model count ≤ 1; 31-feature candidate not PROMOTED; experimental/incompatible promotion refused (409/422).
- Drift/readiness (zero-outcome hermetic state): performance drift `INSUFFICIENT_OUTCOMES` (`has_actual_outcomes: false`); population drift status vocabulary unchanged; readiness states valid with `automatic_retraining: OFF`, human gate required.

**Hardening defects found → fixed (behavior-preserving otherwise):**
1. `inference.py`: connection-disagreement INVALID_INPUT attempts were not persisted to the monitoring store (Task 11 required ALL five states). Fixed; all prediction APIs unchanged.
2. `model_lab.py`: zero matched outcomes returned `INSUFFICIENT_SAMPLE`; now returns `INSUFFICIENT_OUTCOMES` (Module-8 §13). Small-nonzero windows unchanged.

**Regression:** backend 356 passed (348 prior + 8 new; four segment runs 77+82+95+102, zero failures); frontend 43/43; `npm run build` green. Store empty after cleanup of two probe-generated events (both reference-identity probes from this module's diagnosis). Reference prediction verified byte-identical within the E2E suite.

**Limitations (unchanged, carried):** matched-outcome count remains 0 (no production outcomes linked yet) → performance drift/readiness signals stay unassessed, NOT_READY; reliability capped at MEDIUM; unseen-loader interval coverage sub-nominal but unreachable on the served path; stage estimates are prior-median evidence, not learned models; XGBoost/LightGBM/CatBoost never installed (NOT_AVAILABLE in lab).

## EXP-027 — Final UI integration: browser-acceptance record (no model changes)

**Live validation (Vite + uvicorn, real XLSX-backed sources):**
- Overview: real facts — 102 loaders, 543 executions, 350 training-eligible, median 26.32 s (p95 648,287 s); model card `total-et-v3-candidate` CANDIDATE_EVALUATED, 30 features, `unc-log1p-conformal-v1` (88 rows); drift badges SIGNIFICANT DRIFT ×2 + INSUFFICIENT OUTCOMES; readiness RETRAINING_RECOMMENDED_FOR_REVIEW with auto: OFF; empty-store truthful empty state.
- Scenario 1 (known loader, live UI): identity filled → PREDICTED **18.16 seconds**, Known Loader badge, range 1.06 s–2m 57s, MEDIUM reliability, five operational stages with human-readable names + evidence tiers, pre_load duplicate disclosure, Approval Stages section ("no stable historical estimate", "Not included in predicted total"), SHAP summary + bars, WARNING risk.
- Scenario 2 (insufficient history): TESTING_NEW_34854_dev (2 priors < 3) → withheld, note "found: 2 (required minimum: 3)", no ETA.
- Scenario 3 (unknown loader): TotallyUnknownLoaderXYZ on Loader Intelligence → "unknown to definitions", honest no-context cold-start statement, no fabricated evidence.
- Scenario 4 (pre-execution variation): planned 2027-06-01 → PREDICTED 15.00 s (different learned estimate through the real feature pipeline; not a lookup).
- Scenario 5 (invalid input): connection disagreement → INVALID_INPUT card with exact reason, no ETA.
- Scenario 6 (unresolved identity): live UI → "Identity could not be resolved", no ETA.
- Scenario 7 (ambiguous): NOT APPLICABLE live — real data contains 0 AMBIGUOUS resolution rows; covered by automated tests.
- Monitoring & Drift page: zero-matched-outcome truthful states; History page: real events incl. withheld; Model Lab: registry experiments + mixed-evidence feature experiment + rollback control; Data & Settings: 3 source workbooks + reconstruction quality + fixed policies.

**Defect found & fixed in validation:** tz-aware planned_start_time (Z suffix) → ChronologyError 500. Normalized to naive UTC at validation; regression test added (test_module8_e2e). Restart-refresh verified in browser (15.00 s result).

**Tests:** backend 360 passed (47+106+13+26+34+125+9 across all 25 files); frontend 54/54 (8 files); build green. Store cleaned (10 validation probes removed); registry/artifacts untouched; reference prediction byte-identical.

## EXP-028 — Corrected dataset audit + stage-aware architecture experiments (AD-31)

**Corrected-source audit (train/data/Loader_Executions.xlsx, 759 raw rows; replaces the 543-row old file).** Stage valid targets: staging 469, prevalidation 135, transformation 102, pre_load 47, data_loading 322; approvals datamart 12 / dataloading 6. `Dataset_display_name` populated on 36/759 rows (drives the AD-31 §3 stage-identity deviation). Duration-vs-timestamp derivation applied where the duration field is missing and valid start/end exist; no fabricated targets; consistency checks record material duration/timestamp discrepancies. No staging placeholder: staging is a REAL stage trained from `staging_timetaken_in_min × 60`.

**Stage model experiments (5 candidates × 4 trainable stages; 33 registry records incl. 4 SELECTION + EXP-STAGE-AWARE-TOTAL-v1).** Chronological holdout MAE (seconds, log1p targets):

| Stage | Winner (chronological) | Holdout MAE | n | Status |
|---|---|---|---|---|
| Staging | extra_trees | 35.83 s | 82 | registry candidate (never promoted) |
| Prevalidation | extra_trees | 2371.16 s | 16 | registry candidate |
| Transformation | historical_median_baseline | 3.11 s | 15 | baseline wins on evidence |
| Pre-load | NOT IMPLEMENTED | — | 19 eligible (< min) | honest ML_UNAVAILABLE at serving |
| Data Loading | ridge | 3.41 s | 78 | registry candidate |

Selections are `SELECTED_FOR_REGISTRY_ONLY` — no stage model is promoted or served as production; serving exposes stage estimates through the stage-intelligence layer with method/model-status labels.

**Approval contextual estimator.** Hierarchical median with disclosed evidence tiers (exact group → Loader+Dataset → Loader+Connection → Loader → global). On the corrected source both approvals fall back to the GLOBAL tier at serving (12/6 positives, sparse context coverage); estimates, tier, and n are disclosed per request. No hardcoded constants.

**Stage-aware total (EXP-STAGE-AWARE-TOTAL-v1, decision `CANDIDATE_ELIGIBLE_FOR_REVIEW_NOT_PROMOTED`).** Direct-30 features + leakage-safe OOF 7-stage vector, compared chronologically on the identical population: stage-aware wins pooled MAE (508.6 vs 523.8), pooled RMSE (1479.1 vs 1524.7), holdout MAE (758.9 vs 779.3), holdout RMSE (3032.4 vs 3158.1); loses holdout MedAE (4.80 vs 4.09), within-20% (4.7% vs 8.2%), within-2× (23.5% vs 42.4%). Mixed evidence ⇒ per the locked rule the **direct total `total-et-v3-candidate` remains production**; the stage-aware total is experimental. Nothing was promoted, retrained, or rolled back.

**Corrected-regime total model facts (recorded for traceability).** Pooled OOF: MAE 469.2 / MedAE 3.19 / RMSE 2146.0; holdout (85 rows): MAE 729.52 / MedAE 17.87; served-population ET MedAE 2.90 vs 5.40 median-baseline (within-2× 53.8% vs 28.8% — defensibility direction preserved). Calibration regenerated for the served artifact: 85 residual rows, Q80 2.035, Q90 3.7122; holdout coverage 0.906 @ 90 / 0.812 @ 80. Deterministic reference prediction after the AD-31 rebuild: **16.112395656629168 s** (`total-et-v3-candidate`, PREDICTED).

**Defect fixes found during the pass:** (1) `expanding_window_folds` left one row uncovered between the last two folds on n=341 — fold builder fixed so validation blocks partition the post-initial window contiguously (171/171 covered on the regression probe); (2) the serving artifact's embedded evaluation constants were stale hardcoded EXP-012 values — now computed from the corrected-regime evaluation (fabrication-risk removal); (3) on-disk calibration predated the fold fix — regenerated for the exact served fit (§18).

## EXP-030 — Forensic validation pass: five real ML stage models completed (AD-31a)

**Trigger.** The forensic acceptance pass ruled that the EXP-028 stage outcome did NOT satisfy the five-ML-stage requirement: pre_load was `NOT_IMPLEMENTED` (n=19 < the old 40-row gate), and transformation was served by the historical median baseline (a non-ML reference winning a comparison is not an ML model).

**Policy change (AD-31a, documented in `stage_models.py`).** Two-tier evidence policy: n ≥ 40 → full chronological evaluation (4 folds + holdout); 12 ≤ n < 40 → SMALL-SAMPLE evaluation (2 folds + holdout) — the stage still gets a REAL fitted ML model with the limitation disclosed; n < 12 → `NOT_ENOUGH_EVIDENCE`. Selection is now holdout-MAE among REAL ML candidates only (extra_trees / random_forest / hist_gradient_boosting / ridge); the historical median baseline remains registered as a non-ML reference and is never the served stage model.

**Final sweep (real chronological evidence, seconds, log1p targets; all five candidates per stage):**

| Stage | Selected ML model | Holdout MAE | n_total | n_holdout | Status |
|---|---|---|---|---|---|
| Staging | extra_trees | 35.833 | 326 | 82 | OK |
| Prevalidation | extra_trees | 2371.157 | 65 | 16 | OK |
| Transformation | random_forest | 3.361 | 60 | 15 | OK |
| Pre-load | extra_trees | 16.929 | 19 | 12 (min-12 holdout; n_train 7 on the smallest fold) | OK_SMALL_SAMPLE |
| Data Loading | ridge | 3.415 | 313 | 78 | OK |

Transformation baseline reference: historical median 3.106 (registered `…-historical_median_baseline-v2`, never served). Pre-load small-sample metrics are honest but high-variance (pooled MAE 21.12; disclosed in every availability_reason). All records appended as a versioned `…-v2` series (append-only registry; 63 records total) plus `EXP-STAGE-AWARE-TOTAL-v2` (block-cadence OOF now includes pre_load via the documented small-sample cadence min_train=10 / REFIT_EVERY=5; strictly-prior safety unchanged). Re-run comparison: stage-aware pooled MAE 498.3 vs 513.1, pooled RMSE 1462.6 vs 1507.8, holdout MAE 758.7 vs 779.3, holdout RMSE 3031.6 vs 3158.1; loses holdout MedAE 4.80 vs 4.09, ±20% 4.71 vs 8.24, ±2× 23.53 vs 42.35 — same MIXED verdict: `CANDIDATE_EXPERIMENTAL`, direct `total-et-v3-candidate` remains the served model. Nothing promoted.

**Leakage probe (live):** three strictly-future rows carrying 43M-second stage targets leave all five stage predictions byte-identical (PASS). Reference ETA unchanged: 16.112395656629168.

**Tests:** backend 359 passed (0 failed; 25 files, four deterministic chunks 86/71/78/124); frontend 54/54 (Prediction Studio fixture updated to the five-predicting-stage contract, pre-load now rendered as a real ML stage row); build green. Live browser: Overview / Prediction Studio (10 legitimate inputs, no planned-start, no workflow-ID) / result (ETA 16.11 s, 90% range 0–11m40s, LOW reliability, five ML stage rows incl. Pre-load 13.20 s trained on 19) / Stage Analytics / Model Laboratory (v2 records + stage-aware comparison card) / Prediction History / Monitoring & Drift / Data & Settings / Loader Intelligence — all PASS.
