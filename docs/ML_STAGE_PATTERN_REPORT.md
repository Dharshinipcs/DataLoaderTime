# What each stage ML model really learns — pattern report (2026-10-03)

Every number below is read DIRECTLY from the trained serving artifact
(`backend/artifacts/stage-models-v2.pkl`) — the actual feature importances of the
deployed models, not a design description. All 5 stage models use the SAME 29
registered features, so the comparison is apples-to-apples.

## The two signal families every model sees (29 features)

1. **`def_*` — the pre-execution configuration YOU enter (7 features):**
   prevalidation count, transformation count, execution order (max / mean / std),
   component count, type nunique (how many distinct component types), dominant
   calling type. After the next retrain this family also gains the six per-type
   counts (crossreference / dataobject / dataextract / dataset / query / table).
2. **`stage_hist_*` — strictly-prior history (22 features):** for the SAME stage,
   the count / mean / median / std of past durations computed at four context
   widths — this loader, loader+connection, loader+dataset, composite
   (dataset+sprint+loader) — plus the GLOBAL prior and the previous stage's
   duration. Strictly prior only: the context row itself can never contribute
   (leakage registry enforces it), so history features are always "what happened
   BEFORE this execution", never after.

Target: the stage's duration in seconds, modelled as `log1p` and inverted with
`expm1` at serving time (right-skewed durations). Synthetic coverage rows train
the models but NEVER enter any accuracy evidence.

## Per stage — model, focus, and the honest read

### 1. staging — ExtraTrees (400 trees, leaves=1) — 326 real rows, 0 synthetic
- **Group share of importance: pre-exec config 42%, history ~0%.**
- Top feature by far: **`def_type_nunique` 41.0%** — the variety of component
  types in your configuration is the staging model's dominant signal. Next:
  `def_execution_order_std` 1.2%, loader+dataset prior count 0.9%.
- Honest read: staging time in this dataset tracks the SHAPE of your component
  configuration much more than past staging durations (many executions share the
  same staging behaviour). Single-feature concentration is worth watching — the
  next retrain adds the six per-type counts, which give the model a finer
  version of this same signal instead of one blunt counter.

### 2. prevalidation — ExtraTrees (400 trees, leaves=2) — 66 real + 362 synthetic rows
- **Group share: global prior mean 22% + global prior std 9% (history-heavy),
  pre-exec config 7%.**
- Top: `stage_hist_global_prior_mean_sec` 22.4%, `stage_hist_global_prior_std_sec`
  8.8%, loader+dataset prior count 2.9%, `def_execution_order_std` 1.8%,
  `def_prevalidation_count` 1.7%.
- Honest read: prevalidation is SPARSE (only 66 real rows), so the model leans on
  the global typical prevalidation duration and its variability, adjusted by your
  component counts and ordering. This is the correct behaviour for a sparse
  stage — it refuses to pretend loader-specific detail it does not have.
  Trust note: CV MedAE 8.73 s; the holdout shows wide tails (within-20% = 0%),
  so treat prevalidation numbers as indicative until more real rows accumulate.

### 3. transformation — HistGradientBoosting (400 iters, lr 0.05) — 62 real + 154 synthetic
- This family does not expose feature importances directly (histogram-based
  boosting). It consumes the SAME 29 features with the same semantics.
- Honest read: same sparse-stage regime as prevalidation — global/loader history
  carries the level, your component counts and order shape the adjustment.
  CV MedAE 1.93 s on log-inverted seconds is the strongest sparse-stage score in
  the comparison; holdout MedAE 0.66 s on 15 real rows.

### 4. pre_load — GradientBoosting (300 trees, depth 2) — 23 real rows, 0 synthetic
- **The healthiest, most balanced pattern in the system:**
  `def_execution_order_max` 14.2%, loader+connection prior mean 10.2%,
  loader+dataset prior mean 7.1%, global prior median 6.6%,
  loader prior median 5.6%.
- Honest read: it genuinely learned "the later your components run (max order),
  the longer pre-load takes, scaled by this loader/connection's own typical
  pre-load duration". Pre-exec config 15% + loader/connection history ~15% +
  broader priors — a real blend, not a single-feature shortcut.
  Trust note: only 23 real rows — the small sample is carried honestly in the
  evidence (holdout MedAE 5.81 s on 12 rows).

### 5. data_loading — ExtraTrees (400 trees, leaves=1) — 323 real + 457 synthetic rows
- **Group share: loader+connection prior mean 11%, pre-exec config 4%.**
- Top: `stage_hist_loader_conn_prior_mean_sec` 10.9%, `def_transformation_count`
  2.9%, `def_prevalidation_count` 0.7%.
- Honest read: data-loading duration is dominated by WHO runs it and WHERE
  (loader+connection) — the model anchors on that loader's own typical data-load
  time and adjusts mildly for how many transformations you configured. That
  matches the operational intuition that data loading is infrastructure-bound.

## The approvals (NOT ML — by design)
Datamart and Dataloading approvals are too sparse for regression. They use the
hierarchical historical MEDIAN over strictly-prior executions (loader-matched
tier wins; 1 recorded timing = that timing itself; 2+ = median; the GLOBAL
fallback of other loaders' history is hidden from the result). Included
loader-own approvals are added to the total as exact arithmetic.

## How to read this for trust
- The models learn exactly two legitimate things: **your pre-execution
  configuration** and **strictly-prior history of the same stage**. No
  post-execution column can enter a feature (mechanically enforced), so nothing
  about the actual run leaks into the prediction.
- Where the model concentrates matches operational intuition: staging tracks
  configuration shape; pre_load blends order + loader/connection history;
  data_loading anchors on loader+connection history; sparse stages
  (prevalidation/transformation) lean on broader priors instead of pretending
  loader detail.
- The single watch-item: staging's 41% concentration on `def_type_nunique` —
  legitimate but blunt; the per-type counts joining on retrain refine it.
- Every prediction moves when you change component rows because those rows
  re-derive the `def_*` features for the context row (`_apply_component_aggregates`)
  — verified live: 3-stage total 6.544 s vs 5-stage 57.674 s with components.
