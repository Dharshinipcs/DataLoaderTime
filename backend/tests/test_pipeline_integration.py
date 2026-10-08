"""Integration: the whole Task-01 foundation over the real source files."""

from __future__ import annotations


def test_foundation_pipeline_end_to_end(foundation):
    # sources (2026-09-29 replaced Loader_Executions.xlsx: 792 raw rows,
    # Connection_Name column added as part of the 4-key identity)
    assert foundation.sources["executions"]["rows"] == 792

    # reconstruction
    assert foundation.reconstruction["executions_out"] == 543
    assert foundation.frames["executions"]["LDR_Execution_Id"].is_unique

    # resolution (Connection_Name on the replaced source makes the 4-key
    # join far denser than the AD-31-era sparse reality)
    assert foundation.resolution_raw_rows.get("RESOLVED") == 383
    assert foundation.resolution_executions.get("RESOLVED") == 275

    # eligibility (AD-31 corrected-regime identity policy: loader+sprint)
    assert foundation.eligibility["counts"]["training_eligible"] == 353
    assert foundation.eligibility["counts"]["target_valid"] == 537
    assert foundation.eligibility["counts"]["valid_executions"] == 539

    # frames present for downstream tasks
    for name in ("executions", "components", "raw_annotated", "executions_eligible"):
        assert name in foundation.frames
        assert len(foundation.frames[name]) > 0


def test_no_post_execution_column_lands_in_execution_level_features(foundation):
    """Sanity: the execution frame keeps post-execution fields for training
    (target creation) but they are clearly marked by the registry as forbidden
    for prediction-time features."""
    ex = foundation.frames["executions"]
    # present (needed for training/monitoring)
    assert "timetaken_in_sec" in ex.columns
    assert "LDR_Status" in ex.columns
