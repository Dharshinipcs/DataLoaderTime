# ARCHITECTURE_DECISIONS.md

## AD-39 — Human-gated retraining workflow completed + audited freeze (2026-09-29; evidence: DR-001..DR-028, docs/verification/drift_retraining/)

**Context.** The retraining workflow (AD-28 skeleton) was completed to its locked end-to-end form and then the whole system passed a final audit/freeze. NO change to the authoritative prediction logic, the five stage models, or the sum-of-five-stages total policy.

**Decisions.**
1. **Retraining decisions are persisted human records** (`dltime/retraining.py`, store `backend/store/retraining_decisions.jsonl`): CONTINUE_MONITORING (keep serving; monitoring continues) and START_RETRAINING (candidate workflow only — serving untouched until a separate explicit promotion). Decisions snapshot the assessment verdict at decision time.
2. **Candidate bundles reuse the approved methodology exactly**: same five stage families (staging/prevalidation extra_trees, transformation random_forest, pre_load extra_trees, data_loading ridge), same `build_stage_frame` feature contract, same EXP-036 chronological train/holdout split; candidates fit ONLY on train-period rows. Registered CANDIDATE_EVALUATED + an EXP-RETRAIN-* experiment record; serving is untouched by training.
3. **Candidate evaluation compares against the RECORDED served baseline** (`backend/artifacts/registry/served_baseline.json`, EXP-036 raw frozen pipeline, MedAE 1.823s n=85) plus per-stage SELECTION-experiment baselines; verdicts are ACCEPTABLE_FOR_REVIEW / NOT_SUFFICIENTLY_BETTER / INSUFFICIENT_EVIDENCE — never auto-promoting. A candidate at ratio ≤ 1.10 still goes to a HUMAN, never straight to production.
4. **Bundle promotion gates** (`candidate_bundle_report`): artifact exists/loads, version match, exactly the five stages, approved model kinds, callable predict path. Promotion of a `stage-serving-candidate-*` bundle additionally switches stage serving via a PERSISTED file switch (`retraining_serving.py`, fail-closed on corrupt/missing artifacts, live serving-cache reset) and is reversed by rollback. The uncertainty calibration stays bound to the deterministic sum policy — no calibration swap exists or is needed.
5. **Retraining lifecycle notifications** reuse the AD-33 store with open-key dedupe: RETRAINING_STARTED, CANDIDATE_READY, CANDIDATE_REJECTED, MODEL_PROMOTED, ROLLBACK_COMPLETED, EVIDENCE_INSUFFICIENT (low-priority).
6. **Reliability re-anchor for the additive interval** (point ± Q90, lower clamped ≥0): assess_reliability anchors derived from EXP-036 holdout widths (`backend/scripts/reanchor_reliability_width.py`): LOW ≥ 0.472, MEDIUM ≥ 0.354 width-factor. Live regression pinned: 16.649 → [13.73, 19.568] @90%, width 5.839, reliability MEDIUM.

**Verification.** Live drill (actor verification-run): candidate A trained+evaluated (10.9336s vs 1.823s = 5.998× → NOT_SUFFICIENTLY_BETTER), candidate B rejected live; governed promote → serving verified switched (prediction 16.649→18.524s) → rollback → 16.649s restored; lifecycle PROMOTION_APPROVED/RETIRED/ROLLBACK appended; exactly one PROMOTED model preserved (ad34). Full backend suite 426/0; frontend 53/53; tsc + build green; browser 17/17 with 0 console errors. Honest limitation unchanged: 8/20 matched outcomes is INSUFFICIENT to claim real model degradation.

## AD-38 — Productization closure: floor scope correction, transient risk policy, derived-duration feedback, configuration provenance (evidence: EXP-035/036/036b/036c)

**Context.** Final productization pass before formal manual acceptance testing. Goal: the product must run as a standalone hand-off (user-entered pre-execution configuration → prediction → actual outcome → accumulated history → monitoring → governed retraining) without source-code edits. No architecture redesign; the locked stage-aware architecture survived every audit; the changes below are evidence-backed corrections.

**Decisions.**
1. **Floor scope corrected to evidence-adequate stage models (the one architecture-relevant correction).** The clean leakage-safe holdout protocol (EXP-036: stage models fit ONLY on train-period rows, in-train strictly-prior OOF stage columns for the total fit, frozen models predict the 85-row holdout, actuals revealed last) exposed a genuine weakness of flooring on ALL five ML stage predictions: pre_load's stage model trains on n=13 (architecture label EXPERIMENTAL_SMALL_SAMPLE) and its holdout stage predictions ran a median 18.6× above actual stage durations, flooring 84/85 holdout rows and collapsing ±20% from 11.8% to 1.2% (floor above actual in 66/85). The ACTUAL containment invariant (total ≥ every stage duration, 0 violations data-universal, EXP-033) remains the physical basis — but a disclosed-unreliable stage ESTIMATE must not unilaterally override the total. **Decision:** the floor uses only stage models with training n ≥ 30 (staging 244/326, prevalidation 49/65, transformation 41/60, data_loading 232/313 qualify; pre_load excluded). Clean-protocol holdout with the corrected floor: MAE 781.0→779.7, MedAE 1.82→1.38, ±20% 11.8%→20.0%, 2× 51.8%→63.5%, engaged 39/85, floor-above-actual 10/85 — better than raw AND than the all-stage floor on every metric. Small-evidence stage estimates remain fully visible in the Seven-Stage View (disclosed experimental estimates); they just cannot set the floor. Canonical context ETA changes from 12.902 (pre_load-floored) to the raw stage-aware 4.180811920161842 (floor 1.707 not engaged); the direct reference stays byte-stable at 16.112395656629168.
2. **Transient vs persistent risk policy (§16/§26).** Per-prediction risk signals (LOW_CONFIDENCE, OOD, LOW reliability, NEW_LOADER) are shown TRANSIENTLY on the Prediction Result (auto-dismissing toast + the full Operational Risk card) and NO LONGER enter the persistent Alert Center; HIGH_RISK (systemic pattern), withheld predictions, and first-time loader discovery remain persistent. Implemented in `notifications.alerts_from_prediction_event` + the frontend `TransientRiskToast`; regression tests pin the policy.
3. **Actual-outcome derived duration (§18).** The form derives total minutes from Start/End when the user does not enter them (never asks the user to hand-calculate), sends `actual_total_derived_from_timestamps` for provenance, and the backend records `duration_provenance.total = derived_from_start_end_timestamps` on the stored outcome. Explicit minutes win; consistency checks (end-before-start, span mismatch, count consistency) unchanged.
4. **Configuration provenance disclosure (§3/§7).** Every prediction response now carries `identity.configuration_provenance`: each configuration input is labeled `user_supplied` or `definition_derived`. The product never silently substitutes the training workbook for user input: definition features are joined from the registered definitions (the deployed contract), user-supplied values override the CONTEXT row only (audit echo + markers), and provenance is disclosed so a blank field is never presented as user-asserted data. The studio renders provenance badges.
5. **Product workflow verified as a hand-off (§20/§21/§22/§24).** Outcomes persist to `actual_outcomes.jsonl` with eligibility classification; monitoring retraining-readiness counts eligible accumulated actuals (threshold 30) + performance degradation (matched-feedback MedAE vs registry baseline) + drift; recommendation is notification-only; candidate training exists in the Model Lab path with registry registration (CANDIDATE_EVALUATED) and explicit human promotion (registry refuses non-candidate promotion); automatic retraining/promotion remains OFF.

**Consequences.** `inference.py` (floor scope + provenance), `notifications.py` (transient policy), `api.py` (`actual_total_derived_from_timestamps` field), `PredictionStudio.tsx` (toast + provenance badges), `ActualOutcomeForm.tsx` (derived duration), tests updated to the new canonical ETA; backend suite re-run (386/0), frontend 54/54, build green; live E2E re-verified (ETA 4.180811920161842, floor_applied=false, direct reference byte-identical, provenance badges rendered). Artifacts unchanged (no retraining; the floor is a serving-time rule).

## AD-36 — Senior-audit corrections: lifecycle-truthful UI, notification-panel positioning, applicability honesty (evidence: EXP-034)

**Context.** Final broad correctness audit before formal acceptance testing (predict-then-reveal evidence, input-applicability semantics, UI lifecycle truth). Findings required corrections; none changed the model architecture.

**Findings and decisions.**
1. **Lifecycle truth = registry, not artifact metadata.** The artifact metadata status (`CANDIDATE_EVALUATED`) is the training-snapshot label; the registry lifecycle (PROMOTED ad34) is the governance truth. The Prediction Studio status line previously mislabeled the authoritative model "evaluated candidate, not production-final". **Decision:** the UI derives wording from `serving_role` — AUTHORITATIVE_TOTAL → "authoritative serving path (registry lifecycle governs; artifact status is the training snapshot)"; REFERENCE_ROLLBACK → "reference/rollback path". The artifact status remains displayed, correctly framed.
2. **Notification panel positioning.** The panel was absolutely positioned `right-0` from a bell anchored at the sidebar's left region, clipping it off the viewport's left edge. **Decision:** fixed-position panel anchored below the shell header (`fixed left-16 top-16`, scrollable, max-w guard). Verified live: no clipping at 1400×900, badge/mark-read/mark-all unaffected.
3. **Applicability semantics (the most important audit concern).** The audit demanded that "optional" not mean "irrelevant": applicable configuration must be required, canonically derived, or clearly shown as derived. **Decision (least invasive, evidence-backed):** the canonical definitions ALREADY drive the fitted features (def_* aggregates + counts + orders are derived from the resolved loader+sprint+dataset configuration — the primary path); user-entered 11 fields are AUDIT OVERRIDES for this execution's real configuration, echoed in `supplied_configuration` + markers, never fabricated into unrelated features. A blank applicable field therefore does NOT misrepresent the prediction: the fitted configuration comes from the canonical definition (shown via `def_match` + derivation disclosures), and an unmatched identity raises an explicit warning. This semantics is now documented (UI copy + EXP-034b manifests) rather than redesigned.
4. **Component cardinality honesty.** Source manifests: 105/278 configs carry >1 PREVALIDATION row (max 147), 45/278 >1 TRANSFORMATION (max 11); 153 prev-only, 71 trans-only. A single text field per family cannot faithfully name every component of a 147-component config. **Decision:** the single display-name field stays (least invasive) but its semantics are documented and enforced: it names ONE component's display name as audit context; counts/type/order come canonically from the definitions summary/detail (never from the single field); the system must never pretend the field enumerates all components. Multi-component UI (structured input) is explicitly deferred as a future enhancement, not silently claimed as present.
5. **Floor re-validated on unseen data (not just historical invariants).** Predict-then-reveal on all 85 holdout rows: floor engaged 2/85; MAE 780.79→780.07 (never harmful); 0/85 raised above actual. The distinction between ACTUAL invariants (containment, data-universal) and PREDICTED disagreement (estimates may exceed actuals; the floor never claims otherwise) is documented in docs/UNSEEN_PREDICTION_EVIDENCE.md §4.
6. **Reliability logic CONFIRMED correct.** LOW is triggered by interval width-factor ≥12× (conformal width on heavy-tailed data) plus the documented single-window calibration cap (<100 residuals caps at MEDIUM). No change; live reasons disclosed.

**Consequences.** Backend 385/0 (single run incl. 8 new AD-35 regression tests: override independence A/B/C/D, neutral-marker ETA invariance, count-override isolation, stage-specific history poison isolation, floor scope, role/status contract); frontend 54/54; build green. No model/artifact rebuild was required (no contract, target, or fitting change). Formal acceptance testing may proceed against docs/ACCEPTANCE_TEST_CASES.md.

## AD-35 — Stage order, coverage, and the total-composition/floor decision (final closure; evidence: EXP-033)

**Context.** EXP-033 (scripts/exp033_stage_order_forensics.py, exp033_consistency_cases.py) completed the final-closure forensics that EXP-032 left open: actual per-execution stage ORDER (no universal-order assumption), union-of-stage-windows COVERAGE of Start_Time→End_Time, the explicit comparison of total-composition candidates (sum / max / learned / learned+floor), and the precise scope of the max-floor rule.

**Findings.** (a) NO universal operational stage order exists: 20 distinct observed runtime orders across 492 executions with stage evidence; the full 5-stage sequence occurs in only 4/492 executions; stage SETS are loader-specific signatures; single-stage executions exist (staging-only 64, data_loading-only 14, prevalidation-only 2). (b) Pre-execution `execution_order` describes ONLY the PREVALIDATION/TRANSFORMATION component sequence (no staging/pre_load/data_loading component rows exist in the definitions detail) and CONFLICTS with runtime in 57/57 executions where both windows exist (config favors prevalidation-first 33/35; runtime shows transformation first 57/57) — config order and runtime order are different artifacts; timestamps decide the physics. (c) Zero overlap is confirmed (583 consecutive pairs: 0 overlap, 326 touching, 257 gaps; union == sum exactly) but complete coverage of the execution span holds in only 13/490 executions (2.7%); union/total median 0.44 — more than half of the median execution total is uninstrumented queue/gap/overhead time. (d) The containment invariant (total ≥ every stage window AND ≥ every stage duration field) holds 490/490 + 0/1575 violations, across every pattern category; approvals never exceed total (0/21, 0/6) and post-date the loading work.

**Decision.**
1. **Total composition = learned stage-aware total (ad34) with the max-ML-stage physical floor.** Sum of stage predictions is rejected (requires coverage ≈ 1; measured median 0.44; breaks on missing stages; cannot represent gap mass). Max-of-stages alone is rejected (a lower bound, not an estimator — discards gap mass and all non-largest-stage evidence). The learned total is retained as the only candidate trained on the authoritative total target with leakage-safe stage features; it can represent uninstrumented overhead. The floor is retained because the containment invariant is verified data-universal: `ETA = max(stage-aware ETA, max(five ML stage predictions))`.
2. **Floor scope is precise and disclosed:** applied ONLY at serving; approvals are EXCLUDED (their durations never exceed total and they post-date loading work — including them would import large contextual medians into short executions); the floor never touches training, OOF evaluation, calibration, or SHAP; engagement is disclosed via `floor_applied` / `physical_floor_seconds` and the UI bullet.
3. **Order semantics:** the product never assumes an operational order. Runtime ordering is a post-execution observation (analytics only); the pre-execution contract carries `execution_order` + counts as configuration DESCRIPTION inputs, never as order assertions about stage execution; stage features remain per-stage and order-agnostic.
4. **Overview duration statistics are authoritative:** `report.target` values derive from `total_timetaken_in_min×60` (span fallback) via reconstruction; the API surfaces `duration_stat_definition` + target-source counts + span consistency; the UI labels the statistic “authoritative total (min×60)”. The corrupted legacy `timetaken_in_sec` never feeds any authoritative statistic, model, or label; the direct reference/rollback path remains byte-stable and documented.

**Consequences.** No serving-path change vs AD-34 (the floor was already implemented correctly — including the approval exclusion and serving-only scope); corrections in this pass are truth-labeling and stale-copy fixes: Model Lab feature-version literal, DataSources policy bullets (11-input contract, pre_load independence per AD-31a, floor statement). Documented so the decision does not need to be re-derived.

## AD-32 — Stage-aware total is the AUTHORITATIVE serving path; direct total is reference/rollback

**Context.** The locked product direction names the stage-aware total as the intended final prediction architecture. AD-31/AD-31a built and evaluated it but left the direct total served. The final implementation pass promotes the architecture decision: the served ETA must come from the stage-aware total model, with the direct model retained only as a historical/rollback reference.

**Decision.**
1. `stage_aware_serving.py` builds and persists a serving artifact `stage-aware-total-ad32` (extra_trees, log1p target, 36 features = 29 pre-execution context columns + 7 stage-derived columns `pred_stage_*_sec`), plus a dedicated conformal calibration (`unc-log1p-conformal-v1-stage-aware.json`, 85 chronological residuals) tied to that exact model version, plus an OOD training-range profile restricted to numeric `def_*` features.
2. Serving (AD-32 §12): 10 legitimate pre-execution inputs → internal config features → strictly-prior historical features → five ML stage predictions + two contextual statistical approval estimates → 36-feature stage-aware vector → stage-aware ETA → recalibrated uncertainty/reliability → SHAP of the actual served artifact (tree path additive to ~1e-13) → risk. The direct total is still evaluated per request ONLY as `direct_reference_seconds` (rollback evidence) and is never shown as the ETA.
3. Stage models are persisted (`stage-serving-models.pkl`) instead of being refit in-process; a version-integrity check refuses a stale artifact at load time.
4. Registry: `stage-aware-total-ad32` registered as `CANDIDATE_EVALUATED` with the full gate table (leakage/chronological/artifact/feature-contract/registry/live/calibration/SHAP/tests) and promoted through the human-gated `promote()` with that evidence; the direct `total-et-v3-candidate` is marked RETIRED (reference/rollback).
5. Post-execution feedback (§25-29): `POST /feedback/actual` accepts the XLSX-faithful outcome schema (record/error counts, seven stage durations in MINUTES, execution total in SECONDS, linkage refs); errors (total + per-stage) are system-computed and never re-enter prediction; the ActualOutcomeForm (Prediction Studio + Prediction History) is the temporary manual ingestion source until DataZap posts the same body to the same endpoint.
6. The seven-stage view states explicitly (§14) that the total is the stage-aware model's output and NOT the sum of the seven estimates.

**Consequences.** Uncertainty/SHAP/risk are calibrated to and explain the served model (no stale direct-model artifacts). The direct artifact/constants remain for rollback. Evidence remains mixed (stage-aware wins MAE/RMSE/±20% pooling; direct historically held MedAE/2× on holdout), so recalibration and evidence review continue as validated feedback accumulates.

## AD-34 — Evidence-based total-time physics: authoritative target, physical floor, per-stage history, 11-input contract

**Context.** EXP-032 forensic audit of the corrected `train/data/Loader_Executions.xlsx` (scripts/exp032_*.py) established: (a) stage windows are strictly sequential with gaps — 0 overlaps across all 10 pairs of staging/prevalidation/transformation/pre_load/data_loading; (b) the containment invariant total ≥ union(stage windows) holds 737/737 and duration fields never exceed the total (0/1575); (c) the union of stage windows covers only ~31% of total duration (median) — execution time is dominated by non-stage queue/gap time, so any sum-based total mechanism is physically invalid; (d) `total_timetaken_in_min×60` matches the Start/End span 757/757 (100%) while legacy `timetaken_in_sec` agrees only 15.1% — the legacy field is corrupted and the min×60 field is the true physical total; (e) the stage-aware total had been trained on the corrupted legacy field (root cause of the live red flag where the ETA fell below the pre_load prediction); (f) LOADER_DEFINATIONS rows are per-component (`calling_type` ∈ {PREVALIDATION: 1278, TRANSFORMATION: 202}) and `preval_trans_display_name` is a component label, not an operation identifier — prevalidation and transformation are independently applicable within one execution.

**Decision.**
1. **Authoritative target retarget:** the stage-aware total (and its OOF evaluation + calibration) trains on `total_timetaken_in_min × 60` — the same corrected-data target semantics as the stage targets — not on the corrupted legacy `timetaken_in_sec`. Artifact regenerated as `stage-aware-total-ad34` (+ `stage-models-ad34.pkl`, `unc-log1p-conformal-v1-stage-aware-ad34.json`); ad32 artifacts retained for rollback. Corrected-target chronological comparison (identical folds, n=85 holdout): stage-aware pooled 498.33/4.43/1462.6/19.31%/36.20% (MAE/MedAE/RMSE/±20%/2×) vs direct 513.09/11.87/1507.84/7.56%/31.45%; holdout stage-aware 758.73/4.80/3031.59/4.71%/23.53% vs direct 779.30/4.09/3158.08/8.24%/42.35%. Evidence remains mixed; stage-aware stays authoritative by architecture (AD-32) with honest reporting.
2. **Physical floor at serving:** the served ETA is `max(stage-aware ETA, max(five ML stage predictions))`. EXP-032 proved total ≥ every stage duration and total ≥ union(stage windows) in 100% of comparable executions, so a predicted total below a predicted stage is physically impossible. Floor engagement is disclosed (`floor_applied`, `physical_floor_seconds`, UI bullet: "raised to the longest predicted stage"). The floor never alters training, evaluation, calibration, or SHAP — it is a serving-side disclosure-backed clamp on the final displayed ETA only.
3. **Per-stage serving history (Part J correction):** the serving path must attach each stage model's OWN `stage_hist_*` context (the previous loop overwrote the columns, so every stage received data_loading's history). Generic contract columns carry the staging-frame history (build-time contract source). Verified by a stage-specific poison test: poisoning prior staging/transformation/data_loading history changes only the corresponding stage's prediction.
4. **11 legitimate pre-execution inputs:** separate `prevalidation_display_name` and `transformation_display_name` replace the combined `preval_trans_display_name` user field (source semantics: definitions rows are per-component; a single execution config carries components of both kinds). Both flow through config override resolution independently (controlled A/B/C/D probes confirm no cross-corruption); neither is a fitted-contract feature — the registered definitions supply the fitted feature values, user fields act as configuration override/audit inputs surfaced in `result.features` markers.
5. **Feedback total semantics:** the outcome form/API uses `actual_total_timetaken_in_min` (source field semantics; seconds accepted backward-compatible). XLSX column classification otherwise unchanged (AD-32 §26).

**Consequences.** The stage-aware ETA is now trained, evaluated, calibrated, and served on the same physical quantity as the stage models; the live total can never be smaller than a predicted contained stage. Sum-based total candidates (C1/C2/C3) are rejected on physics evidence (stage vector alone R²≈0.15 log-space; union ≈31% of total; total/Σstages median 0.59) and documented in EXP-032. Pre-load's global fallback for zero-history loaders remains disclosed as a small-sample limitation, not hidden by the floor.

## AD-33 — System-level notification center (alerts from existing signals only)

**Context.** The product needed a system-level alert surface (bell icon) without inventing new detection logic or thresholds.

**Decision.** `notifications.py` derives alerts EXCLUSIVELY from existing subsystem verdicts: prediction-event risk/history states (NEW_LOADER, INSUFFICIENT_HISTORY, OUT_OF_DISTRIBUTION, LOW_PREDICTION_RELIABILITY, UNUSUAL_LOADER_EXECUTION_PATTERN from HIGH_RISK), drift statuses (MODEL_DRIFT), retraining readiness (RETRAINING_RECOMMENDED, MODEL_STALE, PREDICTION_ACCURACY_DEGRADED), registry lifecycle (NEW_MODEL_CANDIDATE), feedback data-quality warnings (TRAINING_DATA_QUALITY), and stage serving lifecycle labels (STAGE_DATA_COVERAGE — pre_load small-sample discloses honestly). Severity taxonomy INFO/WARNING/HIGH/CRITICAL with textual label + icon always rendered. Storage: `notifications.jsonl` via MonitoringStore (schema-gated); append at event/outcome ingestion; `append_notifications` deduplicates per open (type, key); notifications stay until explicitly marked read (`{id}/read`, `read-all`); read state persists. `/notifications/evaluate` reuses the SAME drift/readiness computations as the dashboards. No fake alerts, no automatic actions.

**Consequences.** The bell surfaces real conditions only; unread counts persist across restarts; future alert types must wrap an existing verdict to be added.

## AD-31a — Every operational stage serves a REAL fitted ML model (forensic-validation correction of AD-31)

**Context.** AD-31 required five ML operational stages, but the EXP-028 evidence policy produced pre_load = NOT_IMPLEMENTED (n=19 < the 40-row gate) and transformation served by the historical median baseline. The forensic pass ruled both outcomes violate the locked five-ML-stage requirement: a sparse dataset is a limitation to disclose, not permission to remove a required stage; and a non-ML baseline winning a comparison is a reference result, not an ML model.

**Decision.**
1. Two-tier evidence policy in `stage_models.py`: n ≥ 40 → full chronological evaluation (4 folds + holdout, min 12); 12 ≤ n < 40 → SMALL-SAMPLE evaluation (2 folds + holdout) with the limitation disclosed in every serving payload (`EXPERIMENTAL_SMALL_SAMPLE`, reason text carries SMALL-SAMPLE); n < 12 → `NOT_ENOUGH_EVIDENCE` (a defensible ≥12-row holdout is impossible).
2. Stage selection = best holdout MAE among REAL ML candidates only (extra_trees, random_forest, hist_gradient_boosting, ridge). The historical median baseline is kept registered as a non-ML comparison reference and is NEVER the served stage model.
3. `stage_aware_total` OOF includes all five stages: small-sample stages use documented cadence (min_train=10, REFIT_EVERY=5) — training data still strictly precedes each OOF prediction; the approximation (block reuse) is documented and leakage-safe.
4. Serving registry single-sourced (`stage_models.STAGE_MODEL_SERVING_KINDS`); `stage_intel.STAGE_MODEL_SERVING` re-exports it.
5. No promotion changes: stage models remain EXPERIMENTAL candidates; the stage-aware total remains experimental pending explicit human review; direct `total-et-v3-candidate` remains the served ETA.

**Consequences.** All five operational stages now produce real numeric ML predictions live (verified in the browser and API). Pre-load's metrics are honest but high-variance (n=19) — its payload always discloses this. If a future stage's eligible count falls below 12, the system reverts to honest ML_UNAVAILABLE for that stage. Baseline rows stay in the registry as the permanent fairness reference.

Locked decisions only. A decision changes here ONLY with explicit user approval.

## AD-01 — Execution identity and grouping
- Reconstruction grouping key: `LDR_Execution_Id` (one logical execution per id).
- Canonical business identity: **Dataset + Connection + Sprint + Loader Name**.
- Column mapping: executions `Dataset_display_name` / `Sprint` / `Loader_Name`; definitions `datasetname` / `ldr_connection_name` / `sprint_name` / `loader_display_name`.

## AD-02 — Source roles (read-only)
- `LOADER_DEFINATIONS.xlsx` = PRE_EXECUTION_DEFINITION_DETAIL (1480×8)
- `LOADER_DEFINATIONS_1.xlsx` = PRE_EXECUTION_DEFINITION_SUMMARY (278×6)
- `Loader_Executions.xlsx` = POST_EXECUTION_HISTORY (792×38)
- Raw files are never modified; explicit schema validation; sha256 provenance per load.

## AD-03 — Raw rows are not executions
- Training happens at execution level (543), never on 792 raw rows.
- Per-column reconstruction policy (explicit registry in `reconstruct.COLUMN_POLICIES`):
  EXECUTION_LEVEL_STRICT (identity/status/times/target/workflow id — conflict ⇒ ReconstructionError, never guessed),
  COUNT_DISTINCT (dataset dimension), AGG_SUM (record/error counters summed per component),
  COMPONENT_DETAIL (stage fields kept per component row; conservative widening to execution level when single-valued across rows).
- Unknown column ⇒ fail closed.

## AD-04 — Connection resolution (locked rules)
- 0 matches → UNRESOLVED; 1 match → RESOLVED; >1 → AMBIGUOUS. Never silently choose.
- Lookup built from both definition sources; identity normalized by strip, case-sensitive (convention matching the locked audit facts).
- Missing identity component ⇒ UNRESOLVED.
- **APPROVED (user, Task 02 kickoff) — execution-level rule:** collect all successfully resolved component connections of an execution; 0 distinct ⇒ UNRESOLVED, exactly 1 distinct ⇒ RESOLVED, >1 distinct ⇒ AMBIGUOUS. Never silently choose between different connections.
- Evidence (EXP-007): all 27 mixed executions have exactly one distinct resolved connection (0 multi-connection, 0 contradictory evidence); all 50 unresolved components' (Sprint,Loader) definition pairs map to the same connection as their execution; 247/247 fully-resolved executions agree on one connection; 0/74 multi-row executions have full triple coverage, so all-must-resolve policies would mark every multi-dataset execution UNRESOLVED.
- Result: 382/410/0 raw rows; 44/95/0 keys (dropna=False convention, incl. 3 missing-component keys); 274/269/0 executions; 0 AMBIGUOUS anywhere.

## AD-05 — Target
- Target: `timetaken_in_sec` (total execution duration), execution level.
- Validated against `End_Time − Start_Time` (775/790 rows within 1s at raw level).
- Locked raw-row stats revalidated by tests: median 65.62, mean 870,882.67, p95 665,974.4, max 43,000,000.
- Execution-level stats (modeling view): median 26.325, mean 1,232,600.23, p95 648,287, max 43,000,000.

## AD-06 — Extremes are never silently discarded
- 36 executions >100,000s retained (spans corroborate 21 of them within 1s). Outlier handling for modeling is a later explicit decision.

## AD-07 — Leakage boundary (locked)
- Prediction-time feature sets may contain only PRE_EXECUTION, DERIVED_FROM_PRE_EXECUTION, or historical DERIVED_FROM_POST_EXECUTION computed exclusively from PRIOR executions.
- Forbidden as current-execution prediction inputs: target (`timetaken_in_sec`), `Time_Taken`, `End_Time`, `Start_Time`, `LDR_Status`, record/error counters, all stage durations.
- Enforcement: single registry (`schemas.build_default_registry`, all 52 raw fields classified), runtime guards (`leakage.validate_feature_set`, `leakage.guard_frame`), tests fail on violation; unclassified field ⇒ error (fail closed).
- Identity components (`Loader_Name`, `Sprint`, `Dataset_display_name`, ids) = IDENTIFIER: allowed for lookup/joins/eligibility, never model features without an explicit locked decision.

## AD-08 — Training eligibility (v1 policy, configurable)
- Tiers: valid_execution (complete identity + known status) → target_valid (+ finite positive target) → training_eligible (+ status policy).
- v1 status policy: completed statuses eligible (SUCCEED, COMPLETED_WITH_WARNING, COMPLETED_WITH_ERROR, COMPLETED_WITH_ERROR_AND_WARNING, ERROR); interrupted (STOPPED, KILLED) not. Rationale: completed runs count outcomes; interrupted runs do not (EXPERIMENT_LOG EXP-003). Revisit after status-vs-duration evidence review (Task 02/03).
- v1 result: 539 valid / 537 target-valid / 353 training-eligible.
- Connection resolution is recorded per execution but does not gate eligibility in v1 (identity completeness gates instead); a config flag exists to enable the gate later.

## AD-11 — Chronological historical-feature rule (Task 02)
- Historical features for execution E use ONLY executions strictly earlier than E (start timestamp); E itself and any later execution can never contribute.
- Primary ordering: execution Start_Time; deterministic tie-break by LDR_Execution_Id (documented, no random ordering).
- Historical aggregates never read the current or future target/duration; previous-loader-duration and time-since-previous are computed from strictly prior executions; time-since-previous is clipped to ≥ 0; no prior ⇒ explicit missing (<NA>), never a fabricated value.
- Enforced by dedicated chronology tests and the leakage validator (AD-07 remains authoritative).

## AD-12 — History eligibility mechanism (Task 02)
- Configurable HistoryEligibilityEngine; NO hard-coded universal threshold. A v1 default candidate (min_prior_loader_executions=3) exists ONLY as a placeholder for model-validation evidence and is recorded as PENDING (not a locked rule); final values must come from Task 03/04 validation experiments.
- Insufficient-history contexts must never receive a silently generated authoritative ETA from similarity/cold-start logic (cold start is a later task).

## AD-13 — Pre-execution configuration features (Task 02)
- Definition-derived features (prevalidation_count, transformation_count, execution-order statistics, component-type diversity, dominant calling_type) are PRE_EXECUTION/DERIVED_FROM_PRE_EXECUTION and may enter prediction feature sets; all are registered in the feature registry and pass the leakage validator.
- Definition features are joined by exact identity key only; unmatched ⇒ <NA> (no fabrication).

## AD-14 — Categorical encoding (Task 03)
- `def_dominant_calling_type` (2 real categories: PREVALIDATION, TRANSFORMATION) is one-hot encoded (`handle_unknown=ignore`) with explicit `__MISSING__` sentinel for <NA>; missing = information genuinely unavailable pre-execution, never a fabricated category.
- No ordinal label-encoding (categories are not ordered; ordinals would invent false magnitude).
- Numeric features: median imputation; StandardScaler applied only to Ridge (scale-sensitive), not to tree models.

## AD-15 — Chronological evaluation protocol & candidate selection (Task 03)
- Primary evaluation is strictly chronological: expanding-window CV (5 folds, train 175→315 rows, validate 35 each) + final chronological holdout (train 262 rows ≤ 2026-07-15 17:48:04.909; test 88 rows after). No shuffle; no random split anywhere.
- Tie-break on Start_Time = LDR_Execution_Id; boundary timestamps never straddle a split.
- Robust metrics mandatory: MedianAE and tolerance metrics (±10/20/30%, 2×) reported alongside MAE/RMSE; pooled-OOF vs mean/median-across-folds vs worst-fold reported separately.
- Candidate model: ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, max_features=1.0, random_state=42) on log1p target (expm1 back-transform). Artifact `total-et-v3-candidate`, status CANDIDATE_EVALUATED — NOT production/final until Task 04 inference integration is reviewed.
- log1p selected over raw target by evidence (EXP-012), not assumption; raw-target variants remain available via TargetTransformer for future re-evaluation.

## AD-16 — History-eligibility evidence & approved threshold (Task 03 → Task 04)
- Evidence (EXP-012): bins 0 and 1–2 priors ⇒ MedAE ≈ 8.2 s, ≤2× ≈ 50% (unreliable); bins 5–9 ⇒ MedAE 1.30 s, ≤2× 70.5% (reliable); 10+ bin n=8, too small for conclusions. Unseen-loader rows: MedAE 15.1 s vs seen-loader 3.8 s.
- **APPROVED (user, Task 04 kickoff):** configurable policy value `min_prior_loader_executions = 3` (HistoryEligibilityConfig) is the current gate for an authoritative ETA. It is a policy VALUE, not a hard-coded rule: revise anytime from validation evidence without code changes.
- Insufficient-history contexts return explicit INSUFFICIENT_HISTORY; no cold-start/similarity ETA is ever generated (cold start remains a later, separate task).
- The inference service resolves identity BEFORE the gate: UNRESOLVED/AMBIGUOUS identities return their own states and never reach eligibility or prediction.

## AD-17 — Serving artifact & inference path (Task 04)
- Serving artifact `total-et-v3-candidate` (status CANDIDATE_EVALUATED — NOT production): the AD-15 candidate configuration fitted on the exact Task 03 chronological training window (262 rows, 2025-10-10 → 2026-07-15 17:48:04.909; the 88 holdout rows excluded by construction). Built via the same `fit_candidate_pipeline` path as evaluation (byte-identical preprocessing); not a retraining decision.
- Persisted under `backend/artifacts/`; loading is integrity-checked (version/status/feature-order/preprocessor-order must match exactly, else ArtifactError; PRODUCTION status refused — no approval exists for it).
- Serving runtime knob: estimator `n_jobs=1` for bit-identical repeated predictions (parallel float accumulation changed results in the last ulps; not a modeling hyperparameter — recorded hyperparameters unchanged).
- Inference contract: request carries ONLY identity (loader/sprint/dataset) + optional reference fields; the backend constructs all historical/configuration features via the UNMODIFIED Task 02 path — a synthetic context row (target <NA>, is_training_eligible=False, deterministic execution id, Start_Time = latest recorded + 1 day or caller's planned_start_time) is appended to the execution frame and processed by the real chronological engine, so only strictly-earlier eligible executions can contribute (AD-11 preserved end-to-end).
- A caller-provided connection is validation only: disagreement with definition resolution ⇒ INVALID_INPUT; it never overrides resolution.
- Response states: PREDICTED / INSUFFICIENT_HISTORY / UNRESOLVED_IDENTITY / AMBIGUOUS_IDENTITY / INVALID_INPUT; identity states are checked before the eligibility gate.

## AD-18 — Prediction Studio boundary (Task 05; REVISED Task 10.1)
- The Task 04 POST /predict contract is authoritative and is the ONLY API the frontend calls (typed client mirrors it; no second API).
- User-facing inputs are pre-execution identity fields: loader_name, sprint, dataset, **connection — all four REQUIRED as of Task 10.1 (user-approved product decision)** — plus planned_start_time (optional). workflow_execution_id is deliberately not exposed (internal reference). No historical/derived/post-execution value is ever a manual input — the backend derives all of them; a frontend test enforces the absence of such inputs.
- **Task 10.1 mandatory-Connection semantics:** requiring the user to state the connection does NOT change AD-04 authority. The backend still resolves the authoritative connection from the loader definitions; the supplied value is validated against that resolution; disagreement ⇒ INVALID_INPUT (unchanged); input NEVER overrides resolution; unresolved/ambiguous identities keep their own state semantics. There is no second source of truth — the required field is a consistency check, not an override. Missing/blank connection is rejected (HTTP 422 at schema level; service-level INVALID_INPUT `missing_required_field: connection` as defense in depth).
- **Task 10.1 presentation refinement (no engine changes):** PREDICTED results render in a fixed hierarchy — ETA → model/evidence strip (with explicit candidate-status note) → supplementary Stage Intelligence (unavailable stages omitted from the table with a single count note; never summed) → Expected Range/Reliability → "Why this ETA?" (backend-generated deterministic human-readable summary from the actual top SHAP directions/values; full ranked contributions and additivity evidence collapsed underneath; attribution-not-cause wording) → Operational Risk rendered ONLY when the label is not NORMAL (alert behavior; the API risk block itself is unchanged and still present on every PREDICTED response).
- The frontend never computes an ETA. Every state renders the backend's actual state verbatim; non-prediction states never become predictions; connection disagreement is displayed, never overridden or retried with a guess.
- Backend change for integration is limited to CORS middleware for the local Vite dev origins; all Task 01–04 data/leakage/model semantics untouched.

## AD-19 — Stage intelligence is supplementary only (Task 06)
- HARD RULE: stage estimates are NEVER summed into, and never replace, the authoritative total ETA (the Task 04 ExtraTrees/log1p total model). Stage intelligence exists for composition visibility, slow-stage identification, and the Stage Analytics view.
- Stage inventory is defined by what the REAL reconstructed data contains (no 'staging' column exists in this dataset — EXP-015 discrepancy vs the task brief's quoted facts, recorded not silently accepted). `pre_load` (derived span) is VERIFIED IDENTICAL to the literal `data_loading` column on every observed row — surfaced as an alias (`duplicate_of`), never an independent stage.
- Evidence-based modelability (configurable StageConfig, real-data counts): MODELABLE ≥30 positive observations (data_loading_1 332, prevalidation 127, transformation 110, pre_load 51, data_loading 51); ANALYTICS_ONLY ≥6 (datamart_approval 12, dataloading_approval 6); below that TOO_SPARSE / NOT_OBSERVED. ONLY MODELABLE stages receive estimates; sparser stages explicitly report unavailable — no fabricated values.
- Estimator: robust prior median (loader median when ≥3 prior same-loader positives, else global prior median), computed ONLY from strictly-earlier training-eligible executions (AD-11 chronology reused; the target-less, stage-NA, ineligible context row can never enter its own aggregates). ML per stage is NOT justified at these sample sizes; no separate stage models trained.
- Stage durations are POST_EXECUTION observations: usable for prior-only stage statistics and analytics, never as current-execution features in the total model. POST /predict semantics unchanged; `stage_intelligence` is a supplementary response section (PREDICTED only; non-prediction states carry none) and GET /stages/analytics is a separate read-only endpoint.

## AD-20 — Total-ETA uncertainty + reliability (Task 07)
- A layer AROUND the point ETA, never a replacement: the authoritative ExtraTrees/log1p model, its features, hyperparameters, and serving path are UNCHANGED; no interval width may alter the point prediction, and no alternative model may be chosen for narrower intervals.
- Method (chosen from chronological evidence, EXP-016): log1p-space absolute-residual conformal intervals. |log1p(actual) − log1p(predicted)| on the calibration window → empirical quantile q (method='higher', floor 1.5653 to prevent degenerate overconfident intervals) → interval [max(expm1(pred_log − q), 0), expm1(pred_log + q)]. Rejected alternatives (evidence in EXP-016): raw-space |residual| (unstable for skew, 65× median width), relative-error (uses actual magnitude — weaker calibration semantics at extreme durations), scaled-log variant (nominal coverage on n=35 but a less robust estimator choice).
- Calibration data scope (exact): the 88-row final chronological holdout (2026-07-15 17:48:04.915 → 2026-08-13 14:41:43.809) — strictly AFTER the 262-row model training window and strictly BEFORE any serving context. Residuals are out-of-sample for the exact serving artifact. No current/future execution can enter; enforced by a build-time chronology check + tests.
- Versioned calibration artifact `unc-log1p-conformal-v1` (JSON beside the model artifact): method, levels {0.8, 0.9}, quantiles, calibration window/rows, residual stats, bound model version + feature count. Loading fails closed on version/model/feature-count mismatch or unreadable content. Serving uses level 0.9; 0.8 is stored but not served.
- Honesty rules: "90%" is the conformal TARGET (share of calibration residuals within q), NOT a verified probability; honest held-out coverage (calibrate 53 → evaluate 35 rows) is reported in EXP-016 (88.6% @ 0.8 target, 100% @ 0.9) and stored in the metadata. Reliability is a deterministic evidence-based label (MEDIUM/LOW in v1; HIGH not claimable from a single 88-residual window): width factor (≥12× LOW, ≥9× MEDIUM, else HIGH), loader history <2×AD-16 minimum caps at MEDIUM, below minimum is LOW; calibration-evidence cap binds always in v1.
- Scope: uncertainty appears ONLY on PREDICTED responses (`uncertainty` field); non-prediction states never receive a fabricated range; AD-16 gating is unchanged (no ETA ⇒ no interval). Risk-engine/OOD logic stays out (Task 09).

## AD-21 — SHAP explainability (Task 08)
- Explanatory enrichment ONLY, bound to the authoritative serving artifact: the point ETA, features, hyperparameters, history eligibility, and the Task 07 uncertainty layer are UNCHANGED by explanation; SHAP must never become a second prediction model or alter any prediction output.
- Explainer: `shap.TreeExplainer` on the persisted ExtraTrees (shap 0.52.0), built ONCE per service and cached (deterministic; model-version-keyed). Contributions explain the model's **log1p** output: base + Σcontributions = model log1p output (additivity enforced at request time, gate ≤1e-6, else explanation is withheld with a warning). Contributions are NEVER converted to or labelled as seconds; the UI shows the ETA separately and labels magnitudes as log-scale model contributions.
- Feature mapping: derived from the FITTED ColumnTransformer layout (never string-guessed): 32 encoded columns → 30 registered features (29 numeric 1:1; `def_dominant_calling_type` one-hot ×3 aggregated by exact SUM so the encoded contributions are preserved, never max-picked). Mapping is validated to cover every encoded column and every registered feature exactly once; layout drift raises and withholds the explanation.
- Leakage: explanation is computed from the exact prediction-time feature vector the point prediction used (no second feature pipeline). Target/post-execution fields cannot enter by construction; test-enforced (target tampering and frame-growth probes leave SHAP values bit-identical).
- States: `explanation` appears ONLY on PREDICTED responses; non-prediction states never receive a fabricated explanation. Failure behavior: any explanation failure appends `Explanation unavailable: …` to warnings and NEVER removes or changes the point ETA or uncertainty (test-enforced in both directions). Explanations are model attributions, not causal effects — enforced wording in UI and docs.
- Global summary: lightweight GET `/explanation/global` — mean-|SHAP| per registered feature over the FIXED 88-row chronological holdout (out-of-sample for the serving artifact; documented subset; cached explainer; deterministic). Not a Model Laboratory (Task 13); no retraining.

## AD-22 — Operational risk engine (Task 09)
- Independent INTERPRETATION layer: consumes the point ETA (T04), uncertainty/reliability (T07), history evidence, config features, and stage-intelligence AVAILABILITY read-only. Risk NEVER modifies the ETA, interval, SHAP, eligibility, features, or artifact, and never turns a PREDICTED response into another state. Risk is an operational classification — NOT a probability of failure and not model confidence.
- Strategy (EXP-018 evidence): transparent rule HIERARCHY with machine-readable reasons (code/message/evidence). Additive weighted score REJECTED — its candidate inputs overlap (interval width ↔ reliability ↔ history count), guaranteeing double counting; a supervised classifier is unjustified (no ground-truth failure labels).
- Precedence (deterministic): LOW_CONFIDENCE (reliability=LOW, Task 07 label is the single source of truth) > OUT_OF_DISTRIBUTION (config beyond training range) > HIGH_RISK (ETA >5x prior loader median PLUS ≥1 independent corroborating concern) > WARNING (single concern) > NORMAL. HIGH_RISK corroboration prevents dramatizing a large-but-consistent ETA; a large ETA alone is never HIGH_RISK.
- OOD (implemented, no Isolation Forest): robust training-range fence over the 7 numeric config features (def_*) of the artifact's EXACT 262-row training window — excess in robust-IQR units beyond [p1,p99], flag >0.5 (provisional, documented). Evidence: 0/88 real holdout contexts flagged; 0.4% of training rows; synthetic probes scale cleanly (x2→18.9, x5→56.4, x10→118.9) with correct feature attribution. Historical-aggregate features are EXCLUDED from the fence (skewed scales produce meaningless ratios — spurious 29,605-unit excess in the initial probe). All-config-missing contexts yield excess 0 (the definition-match warning already covers them).
- Signals REJECTED as double-counting: interval width ratio (p50=10.0 of ETA — already integral to T07 reliability) and prior-count-based signals (already in reliability); robust z (ETA−median)/prior std REJECTED (flags 38.6% of real holdout — prior std is not a usable scale).
- Stage intelligence: availability/evidence ONLY (a dataset-MODELABLE stage unavailable for this context = LOW_STAGE_EVIDENCE). Stage magnitudes never enter risk; estimates never summed (hard rule unchanged).
- States: PREDICTED carries the full risk block; INSUFFICIENT_HISTORY carries only the opaque INSUFFICIENT_HISTORY label + reason (no fabricated score for a nonexistent ETA); UNRESOLVED/AMBIGUOUS/INVALID carry NO risk at all. Failure isolation identical to SHAP/uncertainty: risk failure appends a warning and never alters the prediction (test-enforced). Policy version: `risk-hierarchy-v1`.

## AD-23 — Loader Intelligence + adaptive history evidence + cold-start context (Task 10)
- A read-only INTELLIGENCE layer answering "what do we know about this loader?" — never a prediction path. The ETA gate remains exactly AD-16 (min 3 prior eligible executions, configurable); no intelligence output can produce, hint, or substitute an ETA, and insufficient-history loaders keep receiving NO authoritative ETA.
- History-evidence levels (EXP-019 real distributions; provisional, documented): NO_HISTORY (0 eligible) / EMERGING (1-2) / SUFFICIENT (3-9, == the AD-16 ETA-eligible population by design) / STRONG (>=10, the top EXP-012 bin). Levels classify EVIDENCE only; they do not move the gate.
- Honesty rules: typical-duration/variability statistics are WITHHELD below 3 eligible observations (a 1-2-sample "typical" would mislead); trend (recent-half vs older-half median ratio) is shown only at n>=6 (19 real loaders; median ratio 1.04, 79% within [0.5,2] — stable on average, so trend is labeled from the ratio with the basis documented).
- Cold-start context: for gate-failing loaders the ONLY contextual evidence is same-DATASET history from OTHER loaders, shown only when >=3 real observations exist, always labeled CONTEXTUAL with an explicit "NOT a prediction / NEVER used as its ETA" disclaimer; otherwise an explicit no-context note. Similar-loader/similarity ETA is explicitly NOT implemented (would fabricate authority from weak evidence).
- API: single read-only endpoint GET /loaders/{loader_name} (URL-decoded exact loader-name match per the existing canonical identity semantics; no second identity system). Datasets/connections/sprints/definition rows are exposed as context. Frontend: new Loader Intelligence page; Prediction Studio untouched (a separate inspection surface, not duplicated into the studio).
- Invariants re-verified by tests: /predict ETA/uncertainty/SHAP/risk/stage outputs byte-identical; insufficient-history contract intact; stage estimates still never summed.

## AD-09 — Out of scope until explicitly tasked
- No model training, no frontend/Prediction Studio, no cold-start prediction, no stage prediction, no DataZap integration, no automatic retraining. (Stage *intelligence* was added by Task 06 per AD-19; per-stage ML models remain out of scope.)

## AD-10 — Toolchain
- `uv` + Python 3.12; deps: pandas, numpy, openpyxl, scikit-learn; dev: pytest.
- Task 08 addition (approved scope: SHAP explainability): `shap` dependency added (0.52.0; re-resolved venv to Python 3.14.6 — verified behavior-preserving: full suite green, reference ETA byte-identical).
- Task 04 addition (approved scope: FastAPI prediction endpoint): `fastapi` + `uvicorn` (runtime), `httpx` (dev, API tests). No other infrastructure.
- Task 05 addition (approved scope: Prediction Studio frontend): `frontend/` Vite + React 19 + TypeScript + Tailwind v4; dev test stack Vitest + Testing Library. Backend CORS restricted to the two local dev origins (localhost/127.0.0.1:5173). No router (single-page Prediction Studio); no other infrastructure.

## AD-24 — Prediction Monitoring + actual-outcome feedback (Task 11)

Status: APPROVED (Task 11 scope; locked roadmap). Prediction engine and all Task 01–10.1 layers untouched.

- Purpose: the operational loop ONLY — persist every prediction attempt, ingest later actual outcomes, link when DEFENSIBLE, compute total/stage error, and record training eligibility as evidence for LATER tasks. NO retraining, NO drift detection, NO model registry, NO automatic promotion (Tasks 12/13; AD-09 unchanged).
- Linkage hierarchy (PART A investigation, EXP-021): `LDR_Execution_Id` is unique (543/543) but is a POST-execution identifier — it cannot exist at prediction time. `ldr_workflow_execution_id` exists on all historical executions (466 distinct values) and is the ONE reference a caller can legitimately supply pre-execution with the executed job. Temporal/identity fallback is PROHIBITED: 57.1% of executions share their canonical identity + calendar day with at least one other execution (max 32 same-day for one identity) — false linkage would corrupt monitoring and future training data. Final rule: EXACT_WORKFLOW_EXECUTION_ID (caller-supplied at prediction time, exact match at ingestion) → else UNMATCHED → AMBIGUOUS only on genuine reference collisions (never silently chosen). The internal synthetic execution id is never persisted as a linkage reference.
- Persistence: local append-only JSONL under `backend/store/` (prediction_events / actual_outcomes / feedback_records; schema t11-v1). Records are NEVER mutated or deleted; linkage is a separate feedback record referencing the prediction event id. Corrupt/foreign-schema lines fail closed as visible store warnings. Repository abstraction (`MonitoringStore`) keeps the engine independent of storage; a later DataZap/database backend can replace it without touching prediction. No secrets stored.
- Every prediction attempt is persisted (all five states) inside the existing failure-isolation envelope: a store failure appends a warning and never changes state/ETA/uncertainty/SHAP/risk (same pattern as SHAP/risk isolation, test-enforced).
- Post-execution data boundary: actual outcomes live ONLY in the monitoring path. The /predict contract has no post-execution fields, so ingested actuals can never enter pre-execution feature assembly (test-enforced). Persisted events are immutable; later history cannot alter an already-persisted event.
- Total error definitions: signed_error = predicted − actual; absolute_error = |signed|; APE only when actual > 0 (else unavailable); log_absolute_error; tolerance flags within 10/20/30% and within 2×. Operational metrics are clearly labeled "NOT Task 03 chronological model evaluation" (offline/online separation, AD-07 preserved).
- Stage errors: computed ONLY where BOTH the served stage estimate and the actual stage value exist; approval/ANALYTICS_ONLY stages are never scored; stage values are never summed against the total (AD-19 unchanged).
- Training eligibility: reuse of the EXISTING Task 01 eligibility semantics (valid target + eligibility chain + status policy) — no new rules invented. ELIGIBLE requires the full existing chain; otherwise a deterministic reason (e.g. UNKNOWN_STATUS, INVALID_TARGET, UNRESOLVED_IDENTITY). Training eligibility is evidence for Task 13; nothing retrains in Task 11.
- Monitoring APIs (read-only): GET /monitoring/summary?loader=, GET /monitoring/events?limit=, GET /monitoring/feedback?limit=. Manual local ingestion via POST /feedback/actual (documented in HOW_TO_RUN_VSCODE.md); no fake production connector.

## AD-25 — Module 1 foundation audit (Prompt 1 of 5; evidence-locked clarifications, no locked-decision changes)

Status: RECORDED (Prompt 1 of 5 complete; investigation-only module — every prior locked decision revalidated and left unchanged).

- **Complete pre-execution field classification** is now test-enforced (`backend/tests/test_foundation_module1.py`): the 4 canonical identity fields (`sprint_name`, `ldr_connection_name`, `loader_display_name`, `datasetname`) and the 6 additional legitimate pre-execution fields (`preval_trans_display_name`, `type`, `calling_type`, `execution_order`, `prevalidation_count`, `transformation_count`) are PRE_EXECUTION; all stage timing/duration columns, record/error counters, status and time columns are POST_EXECUTION; `timetaken_in_sec` is TARGET; `LDR_Execution_Id`, `ldr_workflow_execution_id`, `Loader_Name`, `Sprint`, `Dataset_display_name` are IDENTIFIER (identity/linkage only, never model features). `ldr_workflow_execution_id` remains linkage-only for Task 11 monitoring — explicitly NOT part of the pre-execution feature inventory. `Planned Start Time` is a product-level optional input, NOT a source field (verified absent from all three source files).
- **pre_load semantics (corrected evidence, EXP-022):** the pre_load derived span equals the literal `data_loading_timetaken_in_sec` on 75/75 raw rows (51/51 executions) AND the two measurements always co-occur — but the pre_load timestamp window strictly precedes the data_loading window (40/40 rows; median gap 35.7 s). The equality is of duration VALUES from two sources (alias), never of time windows, and never an independent phase. AD-19's alias conclusion stands with stronger evidence and an explicit window-precedence caveat.
- **Stage/total relationship (conclusion for the modeling phase):** stage durations never partition the total — every stage ≤ total on all co-observed rows; pv+dl+dl1 covers a median of ~4.5% of total (n=5); prevalidation and transformation windows overlap on 85/85 rows. The five operational stages (prevalidation, transformation, pre_load→data_loading, data_loading_1; approvals excluded) are independent, partially-observed targets. The total ETA must continue to be modeled DIRECTLY; stage predictions are supplementary and must never be summed into the total (AD-19 unchanged, now evidence-locked).
- **Feature inventory:** the 30-feature representation is confirmed complete against the full legitimate pre-execution inventory; no additions justified in Module 1 (`preval_trans_display_name` is a 720-value free-text label whose structural information def_component_count/def_type_nunique already encode; adding it would alter the locked serving artifact without evidence). Any feature-space change belongs to the Model Laboratory phase with chronological evidence.
- **Anomaly handling:** long legitimate executions are retained by construction (17 training rows >100,000 s; max 43M s); no outlier removal exists anywhere in the pipeline (verified) — AD-06 unchanged.

## AD-26 — Total + 5-Stage Intelligence Engine (Prompt 2 of 5; evidence-based modeling decisions)

Status: RECORDED (Prompt 2 complete; total serving artifact UNCHANGED — no locked decision revised, new evidence recorded).

- **Total model candidate CONFIRMED (not replaced):** full chronological re-evaluation (EXP-023) reproduces EXP-012 exactly (ExtraTrees log1p pooled OOF MAE 364.6 / MedAE 5.16 / within-2x 48.6%). Selection rationale across metrics: best pooled MedAE among learned models, best within-2x on the served population, stable fold behavior (fold MedAEs 1.3-39.6 s), best holdout MAE, cheap inference, SHAP-compatible. HGB-log1p REJECTED for chronological instability (holdout MedAE 2.95 but fold MedAE up to 194.1 s, pooled 15.96); Ridge numerically unstable (recorded, not tuned); raw-target variants uniformly worse; XGBoost/LightGBM/CatBoost not installed (reported, not added). The artifact remains `total-et-v3-candidate` (CANDIDATE_EVALUATED) — NOT production-final; formal promotion belongs to the Model Laboratory phase.
- **Learned-not-lookup defensibility (new evidence, test-locked):** on the AD-16-served population (>=3 priors, n=80) the candidate beats the global-median baseline: MedAE 2.41 vs 5.57 s, MAE 70.7 vs 109.4 s, within-2x 56.2% vs 22.5%. The ETA is a learned model, not a median lookup.
- **Feature contract UNCHANGED at 30:** leave-one-group-out ablation shows every historical group contributes (removal degrades MedAE: PRE_EXECUTION_CONFIG ->5.88, LOADER ->6.01, LOADER_CONNECTION ->6.22, LOADER_DATASET ->6.15, COMPOSITE ->5.92; GLOBAL removal improves MedAE slightly but degrades MAE by +22% — retained for tail behavior). `preval_trans_display_name` (as def_preval_name_nunique, 129/350 coverage) improved pooled MedAE 5.16 -> 4.69 but is NOT adopted in Prompt 2: adopting it would alter the locked serving artifact and feature contract; referred to the Model Laboratory phase for full evaluation (per-feature ablation, seen/unseen slices) before any contract change.
- **Learned stage models REJECTED (evidence, not assumption):** per-stage chronological comparison of (A) the Task 06 evidence-gated prior-median estimator vs (B) learned ExtraTrees on the 30 total-model features vs (C) learned + chronological stage-history features: the estimator wins or ties everywhere with meaningful test data — prevalidation MedAE 0.28 vs 15.17 s; transformation 0.13 vs 0.46 s; data_loading_1 1.04 / MAE 8.48 vs 0.93 / 12.79 s; data_loading n=7 (inconclusive, and both approaches weak). With 23-321 training-eligible positives per stage, learned per-stage models are unsupported; the median-lookup concern does not apply to stages because the estimator is evidence-gated and used as SUPPLEMENTARY intelligence, never the authoritative total.
- **Stage availability policy (`stage_intel.py`, stage-intel-v2):** every stage response carries an evidence tier: DIRECT (>=3 prior same-loader stage observations; loader median) / LEARNED (>=3 prior global stage observations; global prior median — the broader contextual evidence the product requires, replacing the old blanket "fewer than 3 -> never predict" reading) / UNAVAILABLE (fewer than 3 total prior positives, or stage not dataset-MODELABLE per AD-19). The estimator itself is byte-identical to Task 06 (chronology AD-11 reused); only the classification/policy layer is new. Approval stages are never estimated (AD-19 unchanged).
- **pre_load/data_loading alias policy (serving-enforced):** exactly one estimate per measured duration. data_loading (literal column) carries the estimate; pre_load is returned as `duplicate_of: data_loading` with `available=False` and an explicit non-double-counting note (same duration value on every co-observed row — EXP-022 — while its timestamp window differs). Stage values are NEVER summed into the total (AD-19 hard rule unchanged and re-probed by tests).
- **Cold-start connection context:** Loader Intelligence now also exposes same-CONNECTION other-loader eligible history as CONTEXTUAL intelligence when >=3 observations exist (EXP-023: 58/184 gate-failing training rows qualify; 25 distinct loaders) — dataset context remains preferred when both exist. Context remains strictly descriptive: never an ETA, never a gate change (AD-23 unchanged). Cold-start ETA policy unchanged: sufficient direct history -> authoritative ETA; otherwise NO ETA (global-median or similarity ETAs are explicitly not fabricated; string-similarity matching was evaluated and rejected as indefensible on this data).

## AD-27 — Modules 5–6 hardening (Prompt 3 of 5; no locked-decision changes)

1. **Calibration preserved, not recalibrated.** `unc-log1p-conformal-v1` is bound to the serving
   candidate (model_version + feature count integrity-checked on every load; window guard refuses
   training overlap). Prompt-3 audit reproduced its quantiles exactly from the current artifact →
   recalibration would have been churn, not correction.
2. **Reliability stays evidence-based and capped.** The deterministic width-factor + history +
   calibration-evidence v1 cap (HIGH not claimable from a single 88-residual window) is retained;
   coverage evidence (EXP-024) does not justify threshold changes. Unseen-loader under-coverage is
   unreachable on the served path (no ETA below the history gate) and recorded as a limitation.
3. **Contribution-language explanation policy.** The human-readable SHAP summary uses contribution
   language only and always closes with "model contributions, not causal explanations"; causal
   phrasing is assertion-tested as absent. SHAP math, output space, and mappings unchanged.
4. **Risk interpretation travels with the payload.** Every risk payload carries the disclaimer that
   risk is an operational attention level — not a probability of failure, not model confidence. No
   failure-probability model exists or is authorized in this project.
5. **Monitoring linkage metadata in the event identity block.** The caller workflow reference is
   persisted in `identity.ldr_workflow_execution_id` for audit/linkage; it remains linkage-only —
   never a model feature, never explained, never a required input. Exact-reference-only linkage
   (no time windows) is re-asserted by test.
6. **Monitoring metrics report denominators.** Percentage-based metrics declare their valid
   denominator (positive actuals); matched-pair counts are explicit; empty populations yield null
   metrics, never zeros. Stage errors are scored only where both sides exist; unavailable/duplicate
   stages never scored.
7. **Prediction Studio section order locked (§21).** Execution Context → Total ETA → Expected
   Range + Reliability → Stage Intelligence (supplementary) → Why This ETA → risk alert only when
   non-NORMAL → technical details collapsed.

## AD-28 — Module 7 Model Governance (Prompt 4 of 5; human-gated lifecycle, no locked-model changes)

Locked decisions for the Model Laboratory / drift / retraining-readiness / model-registry layer.

1. **Governed, human-gated model lifecycle.** `model_lab.py` provides an append-only ExperimentRegistry
   (required-field validation, unique IDs, historical records never rewritten), a ModelRegistry with
   explicit states `EXPERIMENTAL → CANDIDATE_EVALUATED → PROMOTED → RETIRED` plus terminal `REJECTED`
   and an append-only LifecycleLog. `register ≠ promote`: a registry entry is never active merely by
   existing. At most one PROMOTED total model at a time (promotion supersedes — retires — the previous
   promoted model, never deletes it). Rollback replays PROMOTION_APPROVED history, verifies the
   previous artifact + compatibility gates, and restores it; every promote/rollback/reject requires
   explicit `approved_by` + `reason`.
2. **Promotion compatibility gates (fail-closed).** `compatibility_report()` verifies before any
   promotion: artifact exists and loads; feature registry/version match with the leakage boundary;
   preprocessing feature-order match; inference contract (predict callable + expected feature count);
   uncertainty-calibration compatibility (version + feature count); SHAP TreeExplainer constructibility;
   no forbidden/post-execution feature (workflow id, target, counters, actuals, residuals). A model
   that fails any gate cannot serve with the existing uncertainty/explanation layers and is refused
   with the failing-gate list. The current candidate `total-et-v3-candidate` passes all gates.
3. **Serving vs registry distinction.** The active serving model remains the Prompt-2 candidate
   `total-et-v3-candidate` (AD-26). No promotion has been performed; `/models/status` explicitly
   reports `serving_is_promoted=false`. The candidate becomes "production" only through the governed
   promotion action — never by assertion.
4. **Drift methodology (signals, not verdicts).** Numeric features: two-sample Kolmogorov–Smirnov
   (PSI also implemented); categoricals: max-proportion-change + total variation. Thresholds
   KS ≥0.15 WARNING / ≥0.30 SIGNIFICANT are **project-defined heuristics**, not statistical truths.
   Reference baseline = the serving model's chronological training window (262 rows,
   2025-10-10→2026-07-15); current window = all executions strictly after it (n=137 in EXP-025, of
   which 88 training-eligible) — never future data as permanent baseline, never a rolling window
   (it would overlap the reference period here). Windows below 20 (current) / 30 (reference) rows
   return INSUFFICIENT_SAMPLE — small samples never generate drift claims. Drift status is an
   attention signal; it is never reported as model failure.
5. **Prediction-performance drift is gated on matched actuals.** Performance drift is computed only
   from matched prediction→actual pairs (minimum 20). With 0 matched actuals (current live state)
   the answer is INSUFFICIENT_SAMPLE; no metric is fabricated from unmatched predictions.
6. **Retraining readiness — recommend, never trigger.** Readiness is a structured signal aggregate:
   ≥30 new eligible actuals → READY_FOR_REVIEW; persistent performance degradation (window MedAE
   > 2× training baseline, heuristic) or SIGNIFICANT drift → RETRAINING_RECOMMENDED_FOR_REVIEW.
   `automatic_retraining` is permanently OFF; no prediction or monitoring path can start training.
   A human decision is required to run the §27 workflow, and its output is a registered candidate
   awaiting promotion gates — never an auto-promotion.
7. **Training eligibility policy formalized (v1; change requires governance).** Current behavior is
   locked as **v1: completed-status eligibility** — a row is training-eligible only with a valid
   execution/target chain AND completed status (non-golden statuses → UNKNOWN_STATUS → not eligible;
   350/533 target-valid rows). The Task-11-identified question of whether other terminal statuses
   should be eligible remains OPEN and is recorded as a controlled model-governance decision: changing
   it requires explicit human approval BEFORE any retraining under a changed policy. Prompt 4 did not
   change eligibility and trained nothing.
8. **Feature-experiment adoption is a promotion-class decision.** `def_preval_name_nunique`
   (from `preval_trans_display_name`) is retained as EVALUATED lab evidence (EXP-025): it improves
   pooled MedAE 5.16→4.69 and most low-history slices, but the served-population result is mixed
   (MedAE 2.41→2.64 worse; MAE/RMSE/±20% better) and adoption would change the locked serving
   artifact. Not adopted in Prompt 4; adoption requires a new artifact version through §15/§16 gates.
9. **Reproducibility.** Every experiment record carries random_state, feature-set version, target
   transformation, hyperparameters, chronological validation method/config, and training/holdout
   populations. No commit hash is fabricated (the project has no VCS); lineage uses the existing
   artifact-metadata + experiment-registry mechanism.
10. **Ablation interpretation rule.** Group-removal results are judged across the full metric set
    (MedAE/MAE/RMSE/±20%/within-2×/holdout) with explicit mean-vs-median tradeoff language (§11/§38).
    GLOBAL-history removal improves MedAE/±20%/within-2×/holdout MedAE but degrades MAE/RMSE — the
    30-feature registry is kept; documented as a judgment, not a single-metric conclusion.

## AD-29 — Module 8 Final Integration & Hardening (Prompt 5 of 5; delivery)

Scope: integration validation + concrete-defect hardening only. No redesign of Modules 1–7; no model, feature-contract, or eligibility change; no promotion/retraining.

1. **Final integration boundary.** The locked lifecycle is proven end-to-end by `backend/tests/test_module8_e2e.py`: REAL source data → identity/eligibility → pre-execution feature assembly (keys ⊆ prediction-only registry, 30) → history snapshot (AD-16 gate) → total ETA → conformal uncertainty (`unc-log1p-conformal-v1`) → 5-stage intelligence (+2 approval stages analytics-only; `pre_load` alias suppressed, never summed) → SHAP explanation → risk → prediction event → exact-reference linkage → monitoring → Model Lab → drift/readiness. Each layer's unit guarantees stay in their module test files; Module 8 proves composition.

2. **Defect fix — INVALID_INPUT audit gap.** Task 11's contract persists a prediction event for ALL five states, but the connection-disagreement INVALID_INPUT path (Task 10.1) returned early without persisting. Fixed: the attempt is now audited like every other state. Behavior of valid predictions unchanged.

3. **Defect fix — performance-drift conformance.** `prediction_performance_drift` reported `INSUFFICIENT_SAMPLE` when zero matched outcomes exist. Module-8 §13 distinguishes "no actuals at all" from "window too small": zero matched outcomes now returns **`INSUFFICIENT_OUTCOMES`**; `INSUFFICIENT_SAMPLE` remains for small-but-nonzero windows and for feature/population drift small samples. Existing status vocabulary otherwise unchanged (no invented states).

4. **Fail-closed history contract (documented, test-enforced).** If the executions frame ever loses the eligibility columns, the history engine treats all rows as unusable (0 priors) → INSUFFICIENT_HISTORY. The system never fabricates history from unevaluated rows. Future-row poisoning (eligible-looking strictly-future rows with extreme targets) provably cannot alter a prediction.

5. **Production-model protection.** `total-et-v3-candidate` + 30-feature contract + calibration artifact are untouched; the 31-feature candidate remains EXPERIMENTAL evidence (EXP-025/AD-28); adoption is a human governance decision requiring a new artifact version + compatibility gates. No promotion, rollback, retraining, or drift verdict was executed against production during Module 8.

6. **Delivery state.** Backend 356 / frontend 43 / build green / store clean / reference prediction byte-identical. Known limitations carried forward unchanged (see PROJECT_STATE.md).

## AD-30 — Final Product / UI Integration Pass (post-Prompt-5 acceptance)

Scope: product-UI integration + browser-validated hardening only. No ML, contract, gate, or lifecycle changes; no promotion/retraining.

1. **Visual system (locked palette).** Tailwind v4 `@theme` tokens carry the authoritative vintage palette verbatim (ivory #F7F4EE workspace, parchment #FFFDF8 cards, charcoal #263238 nav, bronze #8A6A4A primary, sage #6F7D73 secondary, gold #C6A15B sparing highlights, linen #DDD7CC borders, forest/ochre/terra/ember semantics). Shared primitives (`components/ui.tsx`) enforce one coherent system: cards, metrics, labeled badges (color never sole carrier), table helpers, empty/loading/error states. Charcoal sidebar navigation; bronze CTA "Predict Execution Time".

2. **Overview facts endpoint.** Read-only `GET /overview`: static facts (foundation reconstruction, eligibility counts, source provenance rows/cols, artifact metadata, calibration version+rows) cached per process; monitoring counts and recent events read FRESH per request; frame drift cached per feedback-activity level; performance drift computed live from matched feedback. Coverage_90 deliberately null: measured 0.909 lives only in EXP-016 documentation, not in a machine-readable artifact — the UI states it as documented offline evidence, never live accuracy.

3. **Prediction Result hierarchy (locked order).** Execution Context form → Total ETA hero (largest charcoal serif value + Loader-State badge derived from real prior counts vs the AD-16 gate) → range + reliability → evidence grid → five operational stages (human-readable names; pre_load shown only as a duplicate-of note, never a second bar) → separate "Approval Stages (Supplementary Analytics — Not included in predicted total)" card → Why-this-ETA SHAP list (sage raises / terracotta lowers; contribution language, additivity disclosed) → risk card (semantic palette per locked mapping; NORMAL suppressed) → context/warnings.

4. **Defect fix — tz-aware planned_start_time.** Browser `datetime-local` → `toISOString()` yields a Z-suffix timestamp; mixed into the naive Start_Time column it coerced to NaT and tripped ChronologyError (HTTP 500). Fixed at the validation boundary: tz-aware values are converted to naive UTC before entering the context row; naive input unchanged. Regression test proves PREDICTED with a legitimately different learned ETA (feature-pipeline variation, not an exact-match lookup).

5. **Truthful states preserved.** Overview/History/Monitoring show real zero/empty states (no fabricated recent predictions, metrics, or drift); performance drift remains INSUFFICIENT_OUTCOMES; readiness remains evidence-based with auto-retraining OFF; approvals never summed; SHAP never causal.

6. **Delivery state.** Backend 360 / frontend 54 / build green; store cleaned of validation probes; reference ETA byte-identical (verified live in the browser).

## AD-31 — Final modeling architecture correction: stage-aware execution intelligence (corrected Loader_Executions.xlsx)

Trigger: the authoritative execution-history source was replaced with the corrected `train/data/Loader_Executions.xlsx` (759 raw execution rows, real per-stage duration fields). The prior stage-intelligence design (evidence-gated prior medians, `pre_load` as a `duplicate_of: data_loading` alias, approvals analytics-only) was built on the old dataset's sparse/aliased stage columns and is superseded. Everything else (leakage contract, uncertainty, risk, monitoring, Model Laboratory governance, UI system) is preserved.

1. **Five operational stages are now dedicated ML models, not prior medians.** The corrected source carries explicit stage duration fields: `staging_timetaken_in_min` (469 valid targets), `prevalidation_*`/`prevaliadtion_timetaken_in_sec` (135), `transformation_timetaken_in_sec` (102), `pre_load_timetaken_in_min` (47), `data_loading_timetaken_in_sec_1` (322). Minutes-based targets are ×60 converted to seconds; missing durations may be derived from valid stage start/end timestamps; if neither exists the target stays missing — never fabricated/imputed. Each stage trains on a log1p target (right-skew) with expanding-window chronological evaluation and a 5-candidate comparison (historical-median baseline, ExtraTrees, RandomForest, HistGradientBoosting, Ridge).

2. **`pre_load` is a real, independent stage; the alias is removed.** The old `duplicate_of: data_loading` interpretation came from the old file's column collision. The corrected source provides explicit `pre_load_*` fields. Documented deviation: only 19 pre-load rows pass stage eligibility (positive target + identity), below the minimum training population, so the stage model is NOT fitted and serving reports honest `ML_UNAVAILABLE` for Pre-load — no substitute estimate is invented.

3. **Stage training policy (documented deviation from the total-model identity gate).** Total-model eligibility requires the full canonical identity; on the corrected source `Dataset_display_name` is populated on only 36/759 rows, so a per-stage identity gate would eliminate training entirely. Stage models therefore train on loader+sprint identity + completed status + positive stage target. This is a per-stage population policy, recorded in every stage registry record; it does NOT change the total-model eligibility contract.

4. **Approval stages are contextual hierarchical statistical estimates — never ML models, never summed.** Datamart Approval (12 positives) and Dataloading Approval (6 positives) are too sparse for regression. The estimator (AD-31 §6) uses robust medians over a progressive evidence hierarchy: exact contextual group → Loader+Dataset → Loader+Connection → Loader → broader defensible context → global fallback. Every estimate carries its tier, sample count n, and fallback level; no universal constants (no "2 s / 3 s" hardcoding). Approval estimates are labelled contextual statistical estimates and are excluded from the total ETA.

5. **Stage-aware total model replaces naive stage summing — and remains EXPERIMENTAL.** The EXP-028 audit shows operational stage windows overlap and cover only a fraction of the total (median total/Σstages ratio 5.22), so Σ(7 stages) is invalid as a final ETA. A stage-aware total model (direct-30 features + 7-vector of OOF stage predictions/approval estimates, leakage-safe) was trained and compared chronologically: it beats the direct baseline on pooled MAE (508.6 vs 523.8), pooled RMSE (1479.1 vs 1524.7), holdout MAE (758.9 vs 779.3) and holdout RMSE (3032.4 vs 3158.1), but LOSES on holdout MedAE (4.80 vs 4.09), within-20% (4.7% vs 8.2%) and within-2× (23.5% vs 42.4%). Per the locked rule (mixed evidence ⇒ no adoption), the direct total `total-et-v3-candidate` remains the served production model; `stage_aware_total_v1` is a registered candidate eligible for human review. No automatic promotion.

6. **`ldr_workflow_execution_id` is linkage-only.** It is post-execution linkage metadata for monitoring/feedback joins (exact-ID linkage, no time-window guessing). It is excluded from the feature registry, from Prediction Studio inputs, and from feature engineering. Automated tests enforce its absence from prediction inputs.

7. **`planned_start_time` is NOT a source field.** It does not exist in the authoritative dataset schema; it is removed from the Prediction Studio contract and from the feature registry. The AD-30 tz-normalization fix remains for any infrastructure that still receives a timestamp, but a planned-start value never becomes a model input. Prediction inputs are exactly the legitimate pre-execution fields: sprint, connection, loader, dataset, preval/trans display name, type, calling type, execution order, prevalidation count, transformation count.

8. **Prediction Studio contract corrected.** The form exposes the ten legitimate pre-execution fields above (§2 of the correction brief) and never exposes engineered features, historical statistics, prior durations, workflow ID, planned start, or post-execution fields. UI communicates the split: the user provides execution configuration; the system derives ML features internally. The result view renders the seven-stage execution view: five ML stage predictions (Staging/Prevalidation/Transformation/Pre-load/Data Loading) + two contextual approval estimates, each with method, model/evidence status, and sample counts.

9. **Uncertainty/SHAP compatibility.** The served conformal calibration (`unc-log1p-conformal-v1`) was regenerated on the corrected regime for the exact served artifact (85 residual rows; Q80 2.035, Q90 3.7122; holdout coverage 0.906 @ 90 / 0.812 @ 80) — the old 88-row calibration no longer matched the refit (§18 contract). SHAP continues to run on the served direct-total model's 30-feature contract and never exposes post-execution fields as drivers; stage/approval components are presented as separate method-labelled sections, not SHAP features.
