FILLED DATASET (created 2026-09-30)
=====================================================
Originals in train/data/ are UNTOUCHED. Everything here is a working copy.

Loader_Executions.xlsx
  - Original 792 rows: UNCHANGED (values preserved exactly; Connection_Name
    left blank where the real file is blank).
  - Rows with LDR_Execution_Id >= 900000: SYNTHETIC coverage rows
    (693 rows, 693 executions)
    for the 234 pre-execution 4-keys that had no execution history.
  - Rows with LDR_Execution_Id 899001..899008:
    SYNTHETIC edge-case rows (8) - tiny/huge/error/zero-duration/
    no-end-time/slow-preval/no-stage/boundary signatures.
  - Synthetic timestamps sit AFTER the last real End_Time (2027-06-02 02:15:50) and
    BEFORE 2027-01-01, so real vs synthetic never interleave in time.
  - Workflow ids of synthetic rows are also in reserved ranges (899xxx/9xxxxx).

LOADER_DEFINATIONS.xlsx / LOADER_DEFINATIONS_1.xlsx
  - Original rows unchanged + extra column `is_synthetic` (False for all
    originals). No synthetic rows were needed: the definitions already cover
    all 278 4-keys; only executions were missing.

Method: donor-based synthesis (deterministic, seed=42). Each missing 4-key
was matched to the real loader with the closest config signature
(prevalidation/transformation counts, component rows, type mix); the donor's
real executions were cloned with stage timings re-scaled by the config-count
ratio and a small lognormal jitter; records/errors follow the donor status's
real error profile. Synthetic rows are for TRAINING COVERAGE only - never mix
them with real rows for live accuracy claims, drift evidence, or monitoring.
