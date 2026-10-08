# Stage Models v2 — Technical Report (Main Goal)

_Generated 2026-10-03 from the actual artifacts on disk (`backend/artifacts/stage-models-v2.pkl`, `docs/stage_v2_comparison.json`). Every number below is read programmatically from those files — nothing hand-copied._

## 1. Problem framing

The system's main goal: given ONLY the 11 pre-execution inputs a user types in the Studio UI,
predict each operational stage's duration with real ML models, and serve the total as the
exact sum of the stage predictions plus the two approval-stage contextual estimates.
The v2 rebuild replaced the previous opaque single-artifact serving with:

1. a per-stage model comparison (5 families x small grids, chronological CV) — the winner per
   stage is chosen by evidence, never hardcoded;
2. a strict real-vs-synthetic data policy (synthetic coverage rows train the production model
   but NEVER enter accuracy evidence);
3. a loader trust gate with three explicit availability cases;
4. a transparent sum: total = stage predictions + approvals, computed and verified arithmetically.

## 2. Data regime

- Raw corrected export `train/data/Loader_Executions.xlsx`: **792 rows**, 543 executions, 42 columns.
- Staging timing fields (`staging_start_time`, `staging_end_time`, `staging_timetaken_in_min`) were merged from the corrected export by `LDR_Execution_Id`: **716/792** real rows carry observed staging durations (span-verified). No existing cell was modified.
- Filled dataset `train/data_filled/`: **1493 rows** = 792 real + 701 synthetic/edge coverage rows (id >= 899001).
- Synthetic rows carry **0** fabricated staging values — staging exists only where it was truly observed.
- Real-row mask: `LDR_Execution_Id < 899001`. Synthetic rows are training-only coverage for rare
  4-key combos; they are excluded from CV holdouts, holdout evidence, monitoring and any live claim.
- Stage targets derive from the AD-31 chain (supplied minutes x 60, else span, else missing).

## 3. Feature contract (strictly pre-execution)

Per stage, identical contract:

- `def_*`: the AD-31 legitimate pre-execution configuration features derived from the definition
  files for the resolved identity (component names, counts, execution order, calling type, ...).
- `stage_hist_<stage>_*`: as-of strictly-prior aggregates of the SAME stage's durations
  (count/mean/median/std by loader, loader+connection, loader+dataset, and global).
  The context row itself is never usable (`is_training_eligible=False`); no post-execution
  column may enter features (leakage registry enforces this).
- Target: `stage_<stage>_sec` (seconds), modelled as `log1p` and inverted with `expm1`.

## 4. Comparison protocol

For each of the five operational stages (staging now included), 5 candidate families were
compared with small fixed hyperparameter grids:

| family | grid (fixed, documented) |
|---|---|
| ridge | alpha in {1, 10, 100} |
| random_forest | (400, leaf=2, max_features=0.5), (400, leaf=1, max_features=1.0) |
| extra_trees | (400, leaf=2, max_features=0.5), (400, leaf=1, max_features=1.0) |
| hist_gradient_boosting | (400 iter, lr=0.05, leaves=15, leaf_samples=10), (400 iter, lr=0.03, leaves=31, leaf_samples=5) |
| gradient_boosting | (300 est, lr=0.05, depth=2), (300 est, lr=0.03, depth=3) |

During CV, tree families fit a reduced forest (150 estimators) for speed; the final refit
always uses the full grid values. Seed 42 everywhere.

Selection metric: **chronological expanding-window CV MedAE (seconds scale), RMSE tie-break**,
with the small-sample cadence (<40 eligible rows -> 2 folds, min_train=8; >=40 -> more folds).
All folds are chronological; **synthetic rows are excluded from every evaluation split** and
from the holdout. The final production model per stage is refit on ALL eligible rows (real +
synthetic) with the winning family/params; its **holdout evidence is separately refit on real
rows only** so the reported accuracy is uncontaminated by synthetic donors.

## 5. Per-stage results

### staging

- eligible rows: 326 real (+0 synthetic for training)
- **selected: Extra Trees (log1p target)** — params `{"n_estimators": 400, "min_samples_leaf": 1, "max_features": 1.0}`
- real-only holdout: n=81, MedAE=0.0s, RMSE=88.62s, MAE=18.29s, within±20%: 0.9136

| rank | family | CV MedAE (s) | CV RMSE (s) | params |
|---|---|---|---|---|
| 1 | extra_trees | 0.14326518116252104 | 109.33611999924781 | n_estimators=400, min_samples_leaf=1, max_features=1.0 |
| 2 | extra_trees | 0.3894920403467345 | 108.2390889324269 | n_estimators=400, min_samples_leaf=2, max_features=0.5 |
| 3 | random_forest | 0.5346416267617478 | 108.06878981855617 | n_estimators=400, min_samples_leaf=2, max_features=0.5 |
| 4 | hist_gradient_boosting | 0.5360213583401315 | 106.8539464001559 | max_iter=400, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=5 |
| 5 | random_forest | 0.6020375888540608 | 109.21494417010418 | n_estimators=400, min_samples_leaf=1, max_features=1.0 |
| 6 | hist_gradient_boosting | 0.799022358046252 | 107.8173559677008 | max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10 |
| 7 | gradient_boosting | 0.9953669433907282 | 109.32278310560672 | n_estimators=300, learning_rate=0.03, max_depth=3 |
| 8 | gradient_boosting | 1.0095972837920433 | 109.45338456259631 | n_estimators=300, learning_rate=0.05, max_depth=2 |
| 9 | ridge | 1.0874971272926541 | 109.23133235411183 | alpha=100.0 |
| 10 | ridge | 1.174725168429392 | 108.90522120596927 | alpha=10.0 |
| 11 | ridge | 1.1982641800213576 | 108.86325235261558 | alpha=1.0 |

### prevalidation

- eligible rows: 66 real (+362 synthetic for training)
- **selected: Extra Trees (log1p target)** — params `{"n_estimators": 400, "min_samples_leaf": 2, "max_features": 0.5}`
- real-only holdout: n=16, MedAE=2.86s, RMSE=4126.53s, MAE=1483.32s, within±20%: 0.0

| rank | family | CV MedAE (s) | CV RMSE (s) | params |
|---|---|---|---|---|
| 1 | extra_trees | 8.729869715630944 | 726.7700731761272 | n_estimators=400, min_samples_leaf=2, max_features=0.5 |
| 2 | extra_trees | 45.00620145637495 | 456.3704178227424 | n_estimators=400, min_samples_leaf=1, max_features=1.0 |
| 3 | gradient_boosting | 247.45649094754435 | 1008.7324252188332 | n_estimators=300, learning_rate=0.05, max_depth=2 |
| 4 | gradient_boosting | 282.7760255876704 | 1010.2346640180489 | n_estimators=300, learning_rate=0.03, max_depth=3 |
| 5 | random_forest | 779.4389846163597 | 1489.1554332064502 | n_estimators=400, min_samples_leaf=1, max_features=1.0 |
| 6 | random_forest | 793.8587530724755 | 1494.42722469755 | n_estimators=400, min_samples_leaf=2, max_features=0.5 |
| 7 | hist_gradient_boosting | 849.4214115190812 | 1437.3918533072874 | max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10 |
| 8 | ridge | 884.5611154881718 | 1538.8325832151536 | alpha=1.0 |
| 9 | hist_gradient_boosting | 1003.3856493877458 | 1670.0124832137142 | max_iter=400, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=5 |
| 10 | ridge | 1005.1306306799911 | 1699.7310431016617 | alpha=10.0 |
| 11 | ridge | 1053.8545861045754 | 1756.3994557861306 | alpha=100.0 |

### transformation

- eligible rows: 62 real (+154 synthetic for training)
- **selected: HistGradientBoosting (log1p target)** — params `{"max_iter": 400, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 10}`
- real-only holdout: n=15, MedAE=0.66s, RMSE=8.16s, MAE=4.2s, within±20%: 0.0667

| rank | family | CV MedAE (s) | CV RMSE (s) | params |
|---|---|---|---|---|
| 1 | hist_gradient_boosting | 1.9341478182576972 | 6.037334521433365 | max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10 |
| 2 | extra_trees | 2.019491775081793 | 6.587276395329225 | n_estimators=400, min_samples_leaf=2, max_features=0.5 |
| 3 | hist_gradient_boosting | 2.0693668185969214 | 5.611978149426868 | max_iter=400, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=5 |
| 4 | extra_trees | 2.761747394096213 | 6.19485441528606 | n_estimators=400, min_samples_leaf=1, max_features=1.0 |
| 5 | gradient_boosting | 3.1402479498353033 | 7.321120892622737 | n_estimators=300, learning_rate=0.03, max_depth=3 |
| 6 | gradient_boosting | 3.995065090941544 | 7.560350597077142 | n_estimators=300, learning_rate=0.05, max_depth=2 |
| 7 | ridge | 5.360733843880484 | 9.206224175258154 | alpha=1.0 |
| 8 | random_forest | 5.4168075430568505 | 10.768791528214242 | n_estimators=400, min_samples_leaf=1, max_features=1.0 |
| 9 | random_forest | 5.676056931435323 | 10.641281306852992 | n_estimators=400, min_samples_leaf=2, max_features=0.5 |
| 10 | ridge | 6.83528915868124 | 11.080886101050513 | alpha=10.0 |
| 11 | ridge | 8.748999000678637 | 12.87057155309359 | alpha=100.0 |

### pre_load

- eligible rows: 23 real (+0 synthetic for training)
- **selected: Gradient Boosting (log1p target)** — params `{"n_estimators": 300, "learning_rate": 0.05, "max_depth": 2}`
- real-only holdout: n=12, MedAE=5.81s, RMSE=35.97s, MAE=16.89s, within±20%: 0.0833

| rank | family | CV MedAE (s) | CV RMSE (s) | params |
|---|---|---|---|---|
| 1 | gradient_boosting | 1.2299703297740239 | 1.6132720010993327 | n_estimators=300, learning_rate=0.05, max_depth=2 |
| 2 | gradient_boosting | 1.2952704119265501 | 1.69306481602611 | n_estimators=300, learning_rate=0.03, max_depth=3 |
| 3 | extra_trees | 1.3261838198588132 | 1.5534622488990562 | n_estimators=400, min_samples_leaf=1, max_features=1.0 |
| 4 | random_forest | 1.4318720688237327 | 1.4509246579429118 | n_estimators=400, min_samples_leaf=1, max_features=1.0 |
| 5 | hist_gradient_boosting | 1.6178356728723724 | 1.3628298090158353 | max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10 |
| 6 | hist_gradient_boosting | 1.6178356728723724 | 1.3628298090158353 | max_iter=400, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=5 |
| 7 | ridge | 2.0775827850917903 | 1.8040687178457948 | alpha=100.0 |
| 8 | extra_trees | 2.727500002232663 | 2.4311605446712106 | n_estimators=400, min_samples_leaf=2, max_features=0.5 |
| 9 | ridge | 3.073557830702894 | 2.8188763505037224 | alpha=10.0 |
| 10 | random_forest | 3.1534143123130534 | 2.888504828462292 | n_estimators=400, min_samples_leaf=2, max_features=0.5 |
| 11 | ridge | 3.4247420313091883 | 3.2262225746474646 | alpha=1.0 |

### data_loading

- eligible rows: 323 real (+457 synthetic for training)
- **selected: Extra Trees (log1p target)** — params `{"n_estimators": 400, "min_samples_leaf": 1, "max_features": 1.0}`
- real-only holdout: n=80, MedAE=1.77s, RMSE=229.75s, MAE=29.82s, within±20%: 0.2

| rank | family | CV MedAE (s) | CV RMSE (s) | params |
|---|---|---|---|---|
| 1 | extra_trees | 1.87946672273176 | 6.49188164508074 | n_estimators=400, min_samples_leaf=1, max_features=1.0 |
| 2 | random_forest | 2.050104537802006 | 6.389121769173463 | n_estimators=400, min_samples_leaf=1, max_features=1.0 |
| 3 | ridge | 2.357220021999237 | 4.361906483226259 | alpha=1.0 |
| 4 | hist_gradient_boosting | 2.5731307743695897 | 6.563594774687589 | max_iter=400, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=5 |
| 5 | random_forest | 3.153837992550228 | 5.271278475021544 | n_estimators=400, min_samples_leaf=2, max_features=0.5 |
| 6 | hist_gradient_boosting | 3.4788022093965116 | 6.8752403153655335 | max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10 |
| 7 | ridge | 3.6918204582827685 | 5.883140531306985 | alpha=10.0 |
| 8 | gradient_boosting | 4.376782413910465 | 8.165417090705184 | n_estimators=300, learning_rate=0.03, max_depth=3 |
| 9 | gradient_boosting | 4.40744413485063 | 8.06186452199424 | n_estimators=300, learning_rate=0.05, max_depth=2 |
| 10 | extra_trees | 4.905135385477093 | 7.0097658760050265 | n_estimators=400, min_samples_leaf=2, max_features=0.5 |
| 11 | ridge | 5.545744405461226 | 8.654389192805048 | alpha=100.0 |

## 6. Serving policy (the three loader cases)

Threshold: a loader is trusted when it has **>= 3 strictly-prior REAL training-eligible
executions** (`STAGE_LOADER_MIN_PRIOR = 3`). Cases:

1. **KNOWN_LOADER_TRUSTED** — total served = exact sum of the 5 ML stage predictions + the
   two contextual statistical approval estimates (`sum-of-available-stages-v2`). The arithmetic
   identity (total == sum of parts) is verified in the smoke test.
2. **UNKNOWN_LOADER** — per-stage indicative estimates are shown, **no total**; message states
   the threshold and why.
3. **BELOW_THRESHOLD** (1-2 priors) — same as case 2 plus exactly how many more executions
   are needed.

Approvals are contextual hierarchical statistical estimates (sparse by design, AD-31) and are
part of the total per the user contract; they are labelled statistical, not ML. Staging, when
untrained, is reported honestly unavailable and the sum refuses (no partial total).

## 7. Artifacts & traceability

- `backend/artifacts/stage-models-v2.pkl` — serving bundle (per-stage predictors + comparisons).
- `docs/stage_v2_comparison.json` — the full leaderboard/metrics evidence this report renders.
- Serving is all-or-nothing v2: legacy stage predictors are never mixed in (a missing stage is
  honest unavailability, never a stale-model fallback).
## 8. 2026-10-03 update — the ML models in detail (user report request)

### 8.1 The five real ML stage models (read from `docs/stage_v2_comparison.json`)

Each operational stage is served by its OWN trained regressor — real scikit-learn
ensembles, never hardcoded constants, never lookup tables. The winner per stage was
selected by evidence: a 5-family x small-grid comparison under 5-fold chronological
cross-validation, scored by MedAE (median absolute error, seconds, on the log1p
inverted scale) and tie-checked with RMSE.

| Stage | Winner (family) | Hyperparameters | CV MedAE (s) | CV RMSE (s) | Real/synthetic training rows | Holdout (real-only) evidence |
|---|---|---|---|---|---|---|
| staging | ExtraTreesRegressor | n_estimators=400, min_samples_leaf=1, max_features=1.0 | 0.143 | 109.34 | 326 real / 0 synthetic | n=81: MedAE ~0.0 s, MAE 18.29 s, within-20% = 91.4% |
| prevalidation | ExtraTreesRegressor | n_estimators=400, min_samples_leaf=2, max_features=0.5 | 8.730 | 726.77 | 66 real / 362 synthetic | n=16: MedAE 2.86 s (wide tails; honest within-20% = 0%) |
| transformation | HistGradientBoostingRegressor | max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10 | 1.934 | 6.04 | 62 real / 154 synthetic | n=15: MedAE 0.66 s |
| pre_load | GradientBoostingRegressor | n_estimators=300, learning_rate=0.05, max_depth=2 | 1.230 | 1.61 | 23 real / 0 synthetic | n=12: MedAE 5.81 s (small-sample, carried honestly) |
| data_loading | ExtraTreesRegressor | n_estimators=400, min_samples_leaf=1, max_features=1.0 | 1.879 | 6.49 | 323 real / 457 synthetic | n=80: MedAE 1.77 s |

What "learning" means here, concretely:

- **Features (strictly pre-execution):** each stage model consumes the same registered
  feature contract — the `def_*` definition/configuration features of the resolved
  identity (component display names, per-calling-type counts, execution order
  max/mean/std, component count, dominant calling type, type mix) plus the
  `stage_hist_<stage>_*` strictly-prior chronological aggregates (count/mean/median/std
  by loader, loader+connection, loader+dataset, global). No post-execution column can
  enter a feature (the leakage registry enforces this mechanically).
- **Target:** the stage's duration in seconds, modelled as `log1p` and inverted with
  `expm1` at serving time (right-skewed durations need the log space).
- **When you change a pre-execution input, the prediction really moves:** the component
  rows you type re-derive the training-time aggregates for the context row
  (`_apply_component_aggregates`), so a different component mix/order/count/type changes
  the model's input vector and therefore its output. Nothing is a fixed table.

### 8.2 The two approval stages are deliberately STATISTICAL, not ML (2026-10-03 rule)

Approvals are sparse by construction, so training a regressor on a handful of rows would
fabricate confidence. Instead each approval stage uses the hierarchical historical
MEDIAN over strictly-prior executions, walking the evidence ladder from narrowest to
broadest (CONTEXT_GROUP -> LOADER_DATASET -> LOADER_CONNECTION -> LOADER -> COMPOSITE ->
GLOBAL). A tier wins when it has enough recorded observations:

- **1 recorded approval timing for this loader -> that timing itself** is the estimate
  (the median of one observation is the observation).
- **2+ recorded timings -> the median** (robust to the heavy right tail; the mean is
  reported in evidence but the median is the served statistic).
- **Loader-own evidence ONLY (2026-10-03 final rule):** every loader-matched tier
  (CONTEXT_GROUP / LOADER_DATASET / LOADER_CONNECTION / LOADER / COMPOSITE all require
  the loader to match) is INCLUDED in the total and DISPLAYED. The GLOBAL fallback
  (other loaders' history) is disclosed in the audit payload with
  `included_in_total=false` but **hidden from the prediction result** — a loader with no
  approval history of its own shows NO approval section at all.
- Approvals that are included are ADDED to the total:
  `total = sum(ML stage predictions) + sum(loader-own approval medians)` — exact
  arithmetic, verified by `test_sum_total_arithmetic`.

### 8.3 The three loader cases (unchanged contract, verified 2026-10-03)

1. **KNOWN loader (enough history):** five stage models predict; loader-own approvals
   join; total served with the calibrated 90% interval. Reference: loader
   `MigrationSummarySubTableLoaderDataset` (sprint TEST) -> **6.544 s**
   (0.6 + 3.339 + 2.605; this loader has no approval history of its own, so nothing
   approval-related is displayed).
2. **UNKNOWN loader (no prior executions) — BOTH kinds, same treatment
   (2026-10-03 user final spec):** a loader registered in the definitions but
   never run, AND a completely new loader with no definition match at all, both
   get the SAME short refusal — "New loader ... — it has no prior executions
   yet..." (`INSUFFICIENT_HISTORY`, `UNKNOWN_LOADER` case, zero priors, no ETA
   invented). The identity block stays honest (`resolution_status: UNRESOLVED`
   for the undefined kind) and the attempt is persisted with the full
   pre-execution inputs, so the store keeps collecting while post-execution
   entry builds the loader's history toward the trust threshold. Only a
   genuinely AMBIGUOUS identity (same dataset+sprint+loader mapped to multiple
   connections) remains a hard refusal — the system never silently chooses.
3. **BELOW THRESHOLD (some history, under the trust threshold):** same honest refusal
   with the exact counts ("has only N prior execution(s); the trust threshold is M").

### 8.4 Component TYPE is now a first-class learned field (2026-10-03)

Every component row now carries `type` — one of **crossreference, dataobject,
dataextract, dataset, query, table** — alongside display name, calling type
(PREVALIDATION/TRANSFORMATION) and execution order. The backend normalizes it
(`PrevalTransComponent.component_type`), echoes it in the audit block, and records
REAL training features from it: the true `def_type_nunique` and per-type counts
(`def_type_crossreference_count`, `def_type_dataobject_count`, `def_type_dataextract_count`,
`def_type_dataset_count`, `def_type_query_count`, `def_type_table_count`). These join the
training frame on the next retrain so the stage models learn the type mix — no
hardcoding anywhere. The Studio also shows the live prevalidation/transformation
counts computed from your rows BEFORE you click Predict, and posts every field to the
backend.

### 8.5 The reliability label layer was REMOVED (2026-10-03)

The HIGH/MEDIUM/LOW label (and the `WIDE_INTERVAL`-replaced alert type, the event
field, the UI badge, the `by_reliability` error slice and all documentation) was removed
everywhere. What stays: the calibrated 90% interval around the point ETA, the live
error metrics (MAE / MedAE / within-%) on the Prediction History page, and the drift
statistics. The only behavioral descendant: the LOW_CONFIDENCE risk signal now fires
directly from the interval-width criterion (width >= 0.472x the point estimate) without
any label named like that.


---

## 8. 2026-10-04 update — per-section execution order + component-type features (user spec)

### 8.1 The corrected execution-order semantics

The user clarified: **execution_order is maintained separately per section** — prevalidation
components are ordered 1..n among themselves, and transformation components 1..m among
themselves (two independent sequences). The pooled order stats are kept for continuity, and
the following PER-SECTION and PER-TYPE features were added to the feature contract (served
in `inference._apply_component_aggregates`, trained in `definition_features.py`, registered
in `feature_registry.py`; every one is DERIVED_FROM_PRE_EXECUTION):

| feature | meaning |
|---|---|
| `def_preval_order_max/_mean/_std` | shape of the prevalidation 1..n sequence |
| `def_trans_order_max/_mean/_std` | shape of the transformation 1..m sequence |
| `def_preval_type_nunique` / `def_trans_type_nunique` | type variety within each section |
| `def_type_crossreference_count` … `def_type_table_count` | count of each of the 6 component types (crossreference, dataobject, dataextract, dataset, query, table) |

The models were RETRAINED on this extended contract — the serving artifact
`backend/artifacts/stage-models-v2.pkl` and `docs/stage_v2_comparison.json` were regenerated
2026-10-04 (runner: `_run_v2_training.py`, since deleted). The per-stage leaderboard numbers
in sections 5.x above reflect the NEW comparison run whenever regenerated on/after this date.

### 8.2 Why the models can now learn what the user asked

- **Ordering patterns**: a stage whose section runs 5 ordered components produces very
  different durations than one that runs 1 — and the within-section ORDER depth/position is
  now separable per section instead of pooled across both.
- **Type-mix patterns**: the 6 per-type counts let the trees split on e.g. "2 dataextract
  components" vs "table-heavy" configurations, on top of the existing
  `def_type_nunique` + dominant calling type.
- Everything remains strictly pre-execution: the leakage registry blocks any post-execution
  column, and the same feature contract is enforced identically at training time
  (definition files) and serving time (user-supplied component rows).

### 8.3 Loader-case behavior (unchanged contract, re-verified)

1. **KNOWN loader (>= 3 priors)**: stage-sum total available; approvals joined ONLY from the
   loader's OWN approval history (LOADER tier; 1 timing = itself, 2+ = median).
2. **UNKNOWN / new loader** (registered-but-never-run OR completely undefined): short honest
   refusal, no total, post-execution entry stays available so real executions build history.
3. **INSUFFICIENT history (1–2 priors)**: refusal names the count and the threshold (3).
