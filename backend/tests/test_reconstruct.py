"""Part C: execution reconstruction tests against the locked facts."""

from __future__ import annotations

import pandas as pd
import pytest

from dltime.reconstruct import (
    COLUMN_POLICIES,
    ColumnPolicy,
    ReconstructionError,
    reconstruct_executions,
)


def test_reconstruction_counts_match_locked_facts(foundation):
    diag = foundation.reconstruction
    # 2026-09-29 source replacement pins (792 raw rows -> 543 executions)
    assert diag["raw_rows_in"] == 792
    assert diag["executions_out"] == 543
    assert diag["multi_row_execs"] == 74
    assert diag["max_rows_per_exec"] == 9
    assert diag["single_row_execs"] == 469
    assert diag["rows_in_multi_row_execs"] == 323


def test_one_row_per_execution(foundation):
    ex = foundation.frames["executions"]
    assert ex["LDR_Execution_Id"].is_unique
    assert len(ex) == ex["LDR_Execution_Id"].nunique()


def test_component_rows_preserve_all_raw_rows(foundation):
    comp = foundation.frames["components"]
    assert len(comp) == 792
    # every raw row kept exactly once
    assert comp["LDR_Execution_Id"].notna().all()
    per_exec = comp.groupby("LDR_Execution_Id").size()
    ex = foundation.frames["executions"].set_index("LDR_Execution_Id")
    assert (per_exec == ex.loc[per_exec.index, "n_component_rows"]).all()


def test_execution_level_fields_constant_within_execution(bundle):
    """AD-31 note: the legacy raw ``timetaken_in_sec`` column is per-component
    in the corrected source (contradictory values within an execution) — it is
    excluded from the execution-level constancy contract by design; the target
    derives from total_timetaken_in_min x 60 / the valid span instead."""
    raw = bundle.executions
    for col in ["LDR_Status", "Start_Time", "End_Time",
                "ldr_workflow_execution_id", "Loader_Name", "Sprint"]:
        varies = raw.groupby("LDR_Execution_Id")[col].nunique(dropna=True)
        assert (varies <= 1).all(), f"execution-level column {col} varies within an execution"


def test_conflicting_execution_level_value_raises():
    raw = pd.DataFrame(
        {
            "LDR_Execution_Id": [1, 1],
            "Loader_Name": ["A", "A"],
            "Sprint": ["S", "S"],
            "Dataset_display_name": ["D1", "D2"],
            "LDR_Status": ["SUCCEED", "STOPPED"],  # conflict
            "Start_Time": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01")],
            "End_Time": [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-02")],
            "Time_Taken": ["1s", "1s"],
            "timetaken_in_sec": [10.0, 10.0],
            "ldr_workflow_execution_id": [7, 7],
            "Total_Records": [1, 2],
            "Success_Records": [1, 2],
            "Error_Records": [0, 0],
            "Prevalidation_Errors": [0, 0],
            "Format_Errors": [0, 0],
            "Constraint_Errors": [0, 0],
            "Application_Errors": [0, 0],
            "Cascade_Errors": [0, 0],
        }
    )
    with pytest.raises(ReconstructionError, match="LDR_Status"):
        reconstruct_executions(raw)


def test_multi_dataset_execution_dataset_dimension(foundation):
    ex = foundation.frames["executions"]
    multi = ex[ex["n_component_rows"] == 9]
    assert len(multi) > 0
    assert (multi["n_datasets"] == multi["datasets"].str.split(";").str.len()).all()


def test_counter_sum_consistency(bundle, foundation):
    raw = bundle.executions
    ex = foundation.frames["executions"].set_index("LDR_Execution_Id")
    raw_sums = raw.groupby("LDR_Execution_Id")["Total_Records"].sum(min_count=1)
    assert (ex.loc[raw_sums.index, "Total_Records"].fillna(0) == raw_sums.fillna(0)).all()


def test_unknown_column_fails_closed():
    raw = pd.DataFrame(
        {
            "LDR_Execution_Id": [1],
            "Loader_Name": ["A"],
            "Sprint": ["S"],
            "Dataset_display_name": ["D1"],
            "LDR_Status": ["SUCCEED"],
            "Start_Time": [pd.Timestamp("2026-01-01")],
            "End_Time": [pd.Timestamp("2026-01-02")],
            "Time_Taken": ["1d"],
            "timetaken_in_sec": [86400.0],
            "ldr_workflow_execution_id": [7],
            "Totally_New_Column": [1],
        }
    )
    with pytest.raises(ReconstructionError, match="Totally_New_Column"):
        reconstruct_executions(raw)


def test_every_source_column_has_explicit_policy(bundle):
    missing = [c for c in bundle.executions.columns if c not in COLUMN_POLICIES]
    assert not missing, f"unclassified source columns: {missing}"
