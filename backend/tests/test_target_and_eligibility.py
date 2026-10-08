"""Parts E & F: target validation and training-eligibility tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dltime.eligibility import (
    COMPLETED_STATUSES,
    EligibilityConfig,
    apply_eligibility,
)
from dltime.resolution import ResolutionStatus


def test_raw_row_target_facts_match_locked_values(bundle):
    """2026-09-29 replacement facts for the legacy raw ``timetaken_in_sec``
    column: unlike the previous corrected file, this export's legacy column is
    largely span-consistent (775/790 within 1s). It remains superseded by the
    AD-31 span-derived target (no total_timetaken_in_min column exists), and
    the 15 outliers still justify never trusting it blindly."""
    t = pd.to_numeric(bundle.executions["timetaken_in_sec"], errors="coerce")
    assert t.notna().sum() == 790
    assert (t > 0).sum() == 790
    assert abs(t.median() - 65.62) < 0.01
    assert abs(t.mean() - 870882.67) < 0.01
    assert abs(t.quantile(0.95) - 665974.45) < 0.1
    assert abs(t.max() - 43000000.0) < 0.01


def test_execution_level_target_facts(foundation):
    """Execution-level target facts produced by reconstruction (not locked;
    recorded in EXPERIMENT_LOG.md EXP-002 for transparency)."""
    t = foundation.target
    assert t["non_null"] == 541
    assert t["positive"] == 541
    assert t["zero"] == 0 and t["negative"] == 0
    assert abs(t["median"] - 26.325) < 0.001
    assert abs(t["mean"] - 1232839.241) < 0.01
    assert abs(t["p95"] - 648287.302) < 0.1
    assert abs(t["max"] - 43059805.424) < 0.01


def test_extreme_durations_are_retained(foundation):
    ex = foundation.frames["executions"]
    t = pd.to_numeric(ex["timetaken_in_sec"], errors="coerce")
    assert (t > 100000).sum() == 36  # long executions kept, never dropped for being extreme
    assert abs(t.max() - 43059805.424) < 0.01


def test_start_end_target_consistency(bundle, foundation):
    """2026-09-29 replacement: the legacy raw in-sec column is now largely
    span-consistent (775/790 within 1s; 15 outliers remain), and the AD-31
    derivation is span-first because the file has no total_timetaken_in_min
    column — so the derived target equals the span by construction."""
    d = bundle.executions.dropna(subset=["Start_Time", "End_Time", "timetaken_in_sec"]).copy()
    span = (pd.to_datetime(d["End_Time"]) - pd.to_datetime(d["Start_Time"])).dt.total_seconds()
    diff = (span - pd.to_numeric(d["timetaken_in_sec"], errors="coerce")).abs()
    assert (diff <= 1).sum() == 775
    assert (diff <= 5).sum() == 775
    # Derived execution-level target (span, per AD-31 rule 2 on this file)
    # trivially agrees with the span; the assertion stays as a tripwire
    # against any future derivation regression.
    ex = foundation.frames["executions"].dropna(
        subset=["Start_Time", "End_Time", "timetaken_in_sec"]
    )
    ex_span = (pd.to_datetime(ex["End_Time"]) - pd.to_datetime(ex["Start_Time"])).dt.total_seconds()
    ex_diff = (ex_span - pd.to_numeric(ex["timetaken_in_sec"], errors="coerce")).abs()
    assert (ex_diff <= ex_span.abs() * 0.01 + 5).mean() >= 0.9


def test_status_distribution_matches_real_data(foundation):
    dist = foundation.eligibility["status_distribution"]
    assert dist == {
        "SUCCEED": 188,
        "STOPPED": 178,
        "COMPLETED_WITH_WARNING": 89,
        "COMPLETED_WITH_ERROR": 61,
        "COMPLETED_WITH_ERROR_AND_WARNING": 12,
        "ERROR": 7,
        "KILLED": 8,
    }


def test_three_tier_eligibility_counts(foundation):
    counts = foundation.eligibility["counts"]
    assert counts["total_executions"] == 543
    # 4 invalid executions lack canonical identity parts (Loader/Sprint/Dataset).
    assert counts["valid_executions"] == 539
    assert counts["target_valid"] == 537      # minus executions with no derivable target
    assert counts["training_eligible"] == 353  # completed statuses only, AD-31 policy


def test_stopped_and_killed_excluded_by_policy(foundation):
    table = foundation.eligibility["status_x_eligible"]
    assert table["STOPPED"]["eligible"] == 0
    assert table["KILLED"]["eligible"] == 0
    assert table["SUCCEED"]["eligible"] == 184
    assert table["COMPLETED_WITH_WARNING"]["eligible"] == 89
    assert table["COMPLETED_WITH_ERROR"]["eligible"] == 61
    assert table["COMPLETED_WITH_ERROR_AND_WARNING"]["eligible"] == 12
    assert table["ERROR"]["eligible"] == 7
    # eligible sum check
    assert sum(v["eligible"] for v in table.values()) == 353


def test_reasons_are_explicit(foundation):
    ex = foundation.frames["executions_eligible"]
    stopped = ex[ex["LDR_Status"] == "STOPPED"].iloc[0]
    assert "status_not_training_eligible" in stopped["eligibility_reasons"]


def test_missing_identity_blocks_validity(foundation):
    ex = foundation.frames["executions_eligible"]
    invalid = ex[~ex["is_valid_execution"]]
    reasons = set()
    for r in invalid["eligibility_reasons"]:
        reasons.update(r.split(";"))
    assert "missing_loader_name" in reasons or "missing_sprint" in reasons


def test_config_can_enable_connection_gate(foundation):
    ex = foundation.frames["executions_eligible"]
    strict_cfg = EligibilityConfig(require_resolved_connection=True)
    strict = apply_eligibility(ex.drop(
        columns=["is_valid_execution", "is_target_valid", "is_training_eligible",
                 "eligibility_reasons"], errors="ignore"), strict_cfg)
    assert strict.counts["training_eligible"] == 350  # 2026-09-29 replacement: Connection_Name makes the join dense (275 exec RESOLVED)
    assert strict.counts["training_eligible"] > 0


def test_all_statuses_classified():
    known = {"STOPPED", "SUCCEED", "COMPLETED_WITH_WARNING", "COMPLETED_WITH_ERROR",
             "COMPLETED_WITH_ERROR_AND_WARNING", "KILLED", "ERROR"}
    assert known == COMPLETED_STATUSES | {"STOPPED", "KILLED"}
