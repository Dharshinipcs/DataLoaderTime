# Filled dataset — what was created and how (2026-09-29)

**Originals in [train/data/](../train/data/) are untouched.** All created files live in
[train/data_filled/](../train/data_filled/). Original real rows are preserved
value-for-value (verified programmatically: 0 diffs on every column).

## Files

| File | Contents |
|---|---|
| `train/data_filled/Loader_Executions.xlsx` | 792 real rows (unchanged) + 693 synthetic coverage rows (ids ≥ 900000) + 8 edge-case rows (ids 899001–899008) |
| `train/data_filled/LOADER_DEFINATIONS.xlsx` | original 1480 rows + `is_synthetic` column (all False — no synthetic rows needed) |
| `train/data_filled/LOADER_DEFINATIONS_1.xlsx` | original 278 rows + `is_synthetic` column (all False — no synthetic rows needed) |
| `train/data_filled/README_filled_dataset.txt` | plain-text summary of the above |

## Why

The 4-key join (Sprint + Connection + Loader Name + Dataset Name) showed that of the
**278 pre-execution loaders only 44 had execution history** (792-row replaced
`Loader_Executions.xlsx`). Training per-loader stage models on 44/278 coverage would
leave 234 loaders with zero loader-specific signal. The fill closes that coverage gap
so retraining can learn *loader-specific* behavior for every configured loader.

## Method (donor-based synthesis, deterministic, seed = 42)

1. **Donor selection** — for each of the 234 missing 4-keys, the real loader with the
   closest configuration signature is chosen as donor. Signature distance =
   |Δ prevalidation_count| + |Δ transformation_count| + 0.5·|Δ component rows|
   + 2·(mismatch of has-prevalidation / has-transformation); donors with real stage
   timings get a 0.7 distance multiplier (preferred).
2. **Clone + rescale** — 2–4 of the donor's real executions are cloned per key.
   The donor's stage-timing *shape* is kept (which stages ran, stage fractions,
   queue/gap share); magnitudes are scaled by √(target-config-counts / donor-config-counts),
   bounded to ×0.5–×3, times a lognormal jitter (σ = 0.10 per stage, 0.12 total).
3. **Realistic dressing** — record counts scale with the same factor; error counters
   follow the donor status's real error profile; Postgres-interval text fields
   (`…_timetaken`) are rendered in the file's native format; stage windows are placed
   sequentially inside the total span (matching the real file's structure).
4. **Timestamp isolation** — synthetic executions start after 2026-11-01, strictly
   after the last real `End_Time` (2026-08-13), and before 2027-01-01, so real vs
   synthetic never interleave in time.
5. **Edge cases (8 rows)** — tiny (2 records), huge (1.5M), all-error, zero-duration
   (KILLED), no-end-time (target honestly unavailable), slow-preval-dominant,
   no-stage-fields, boundary counts. These stretch the model's input space on purpose.

## Identification of synthetic rows

- Executions file: `LDR_Execution_Id` ≥ 900000 (coverage), 899001–899008 (edge cases);
  workflow ids in reserved ranges too. Real rows keep their original ids (< 899001).
- Definition files: `is_synthetic` column (False everywhere today; True if you ever
  add synthetic definition rows).

## Verification performed

- Original rows byte-value-identical: **0 diffs** across all three files.
- All three filled files pass `dltime.sources` schema validation.
- `reconstruct_executions` succeeds on the filled file (1244 executions,
  1240 with targets; the no-end-time edge row is honestly target-less).
- 4-key coverage after fill: **278 / 278**.
- Backend test suite re-pinned to the replaced source (see below).

## Backend compatibility fix (required by the replaced file, done before the fill)

`reconstruct.py` is fail-closed: the new file's `Connection_Name` column had no
policy entry and was rejected. Added as `EXECUTION_LEVEL_STRICT` (identity), and
`Connection_Name` was registered as an `IDENTIFIER` in `schemas.py`. Stale data pins
in 8 test files were re-computed from the real replaced file and updated
(792 rows → 543 executions, 353 training-eligible, etc.).

## Honest usage boundary (important)

Synthetic rows exist to give the models **training coverage**. They must never be
mixed into: live accuracy claims, drift/monitoring evidence, uncertainty calibration,
or anything shown as production proof. Keep the id-range filter whenever training.
Real outcomes from tomorrow's meeting data should gradually replace synthetic rows
1:1 — retrain with the flag filter, then drop the synthetic rows for that loader
once real coverage exists.
