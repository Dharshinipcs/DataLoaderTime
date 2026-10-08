"""Task 02: chronological historical features — chronology & prior-only tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dltime.historical import (
    HIST_FEATURE_NAMES,
    ChronologyError,
    build_historical_features,
)


def _exec_frame(rows: list[dict]) -> pd.DataFrame:
    base = {
        "LDR_Execution_Id": None,
        "Loader_Name": "L",
        "Sprint": "S",
        "Dataset_display_name": "D",
        "LDR_Status": "SUCCEED",
        "Start_Time": None,
        "End_Time": None,
        "timetaken_in_sec": None,
        "ldr_workflow_execution_id": 1,
        "execution_connection": pd.NA,
        "Start_Time_default": None,
        "is_training_eligible": True,
        "is_valid_execution": True,
        "is_target_valid": True,
    }
    out = []
    for i, r in enumerate(rows):
        row = dict(base)
        row.update(r)
        row.pop("Start_Time_default", None)
        # deterministic default timestamps so helper-built frames are orderable
        if row.get("Start_Time") is None:
            row["Start_Time"] = pd.Timestamp("2026-01-01 10:00:00") + pd.Timedelta(hours=i)
        row["LDR_Execution_Id"] = row.get("LDR_Execution_Id", 1000 + i)
        out.append(row)
    return pd.DataFrame(out)


def test_real_data_first_execution_has_zero_priors(foundation):
    full = foundation.frames["executions_eligible"]
    result = build_historical_features(full, prior_filter="training_eligible")
    ordered = result.frame.sort_values(["Start_Time", "LDR_Execution_Id"], kind="mergesort")
    first = ordered.iloc[0]
    assert first["hist_global_prior_count"] == 0
    assert pd.isna(first["hist_global_prior_mean_sec"])
    assert pd.isna(first["hist_prev_loader_duration_sec"])
    assert pd.isna(first["hist_time_since_prev_loader_sec"])


def test_real_data_prior_counts_are_strictly_increasing_somewhere(foundation):
    result = build_historical_features(
        foundation.frames["executions_eligible"], prior_filter="training_eligible"
    )
    ordered = result.frame.sort_values(["Start_Time", "LDR_Execution_Id"], kind="mergesort")
    counts = ordered["hist_global_prior_count"].to_numpy()
    # counts must increase by exactly the number of usable priors as we walk time
    assert counts[-1] == counts[0] + int(
        result.frame.sort_values(["Start_Time", "LDR_Execution_Id"], kind="mergesort")[
            "is_training_eligible"
        ].fillna(False).astype(bool).to_numpy()[:-1].sum()
    )


def test_synthetic_current_and_future_excluded():
    """Leakage probe: later executions' targets must not enter earlier priors."""
    df = _exec_frame(
        [
            {"Start_Time": pd.Timestamp("2026-01-01 10:00:00"), "timetaken_in_sec": 100.0},
            {"Start_Time": pd.Timestamp("2026-01-02 10:00:00"), "timetaken_in_sec": 200.0},
            {"Start_Time": pd.Timestamp("2026-01-03 10:00:00"), "timetaken_in_sec": 400.0},
        ]
    )
    result = build_historical_features(df, prior_filter="all")
    f = result.frame.sort_values("Start_Time")
    r1, r2, r3 = f.iloc[0], f.iloc[1], f.iloc[2]
    # exec 1: nothing prior
    assert r1["hist_global_prior_count"] == 0
    # exec 2: only exec 1
    assert r2["hist_global_prior_count"] == 1
    assert r2["hist_global_prior_mean_sec"] == 100.0
    assert r2["hist_prev_loader_duration_sec"] == 100.0
    # exec 3: execs 1+2, never itself (400) and never future values
    assert r3["hist_global_prior_count"] == 2
    assert r3["hist_global_prior_mean_sec"] == 150.0
    assert r3["hist_prev_loader_duration_sec"] == 200.0
    assert r3["hist_global_prior_median_sec"] == 150.0


def test_synthetic_same_timestamp_tie_broken_by_execution_id():
    """Tie handling: same Start_Time ordered by LDR_Execution_Id deterministically."""
    df = _exec_frame(
        [
            {"LDR_Execution_Id": 200, "Start_Time": pd.Timestamp("2026-01-01 10:00:00"), "timetaken_in_sec": 100.0},
            {"LDR_Execution_Id": 100, "Start_Time": pd.Timestamp("2026-01-01 10:00:00"), "timetaken_in_sec": 300.0},
            {"LDR_Execution_Id": 300, "Start_Time": pd.Timestamp("2026-01-01 10:00:00"), "timetaken_in_sec": 500.0},
        ]
    )
    result = build_historical_features(df, prior_filter="all")
    by_id = result.frame.set_index("LDR_Execution_Id")
    # id 100 is processed first: no priors
    assert by_id.loc[100, "hist_global_prior_count"] == 0
    # id 200 processed second: sees only id 100 (300s)
    assert by_id.loc[200, "hist_global_prior_count"] == 1
    assert by_id.loc[200, "hist_global_prior_mean_sec"] == 300.0
    # id 300 sees both
    assert by_id.loc[300, "hist_global_prior_count"] == 2
    assert by_id.loc[300, "hist_global_prior_mean_sec"] == 200.0


def test_time_since_prev_is_non_negative():
    df = _exec_frame(
        [
            {"Start_Time": pd.Timestamp("2026-01-02 10:00:00"), "timetaken_in_sec": 100.0},
            # earlier id but later time would break order; instead: same timestamp, later id
            {"LDR_Execution_Id": 1005, "Start_Time": pd.Timestamp("2026-01-02 10:00:00"), "timetaken_in_sec": 200.0},
        ]
    )
    result = build_historical_features(df, prior_filter="all")
    ts = result.frame["hist_time_since_prev_loader_sec"].dropna()
    assert (ts >= 0).all()


def test_group_keys_isolate_history():
    """A different loader/dataset must not share history."""
    df = _exec_frame(
        [
            {"Loader_Name": "A", "timetaken_in_sec": 100.0},
            {"Loader_Name": "B", "timetaken_in_sec": 900.0},
            {"Loader_Name": "A", "timetaken_in_sec": 300.0},
        ]
    )
    result = build_historical_features(df, prior_filter="all")
    rows = result.frame.sort_values("Start_Time")
    a2 = rows.iloc[2]
    assert a2["hist_loader_prior_count"] == 1  # only first A execution
    assert a2["hist_loader_prior_mean_sec"] == 100.0
    assert a2["hist_global_prior_count"] == 2  # global sees both loaders


def test_connection_group_uses_resolved_connection():
    df = _exec_frame(
        [
            {"execution_connection": "C1", "timetaken_in_sec": 100.0},
            {"execution_connection": "C2", "timetaken_in_sec": 900.0},
            {"execution_connection": "C1", "timetaken_in_sec": 300.0},
            {"execution_connection": pd.NA, "timetaken_in_sec": 700.0},
        ]
    )
    result = build_historical_features(df, prior_filter="all")
    rows = result.frame.sort_values("Start_Time")
    assert rows.iloc[2]["hist_loader_conn_prior_count"] == 1  # only C1 prior
    assert rows.iloc[3]["hist_loader_conn_prior_count"] == 0  # no connection -> no conn history


def test_ineligible_prior_excluded():
    df = _exec_frame(
        [
            {"timetaken_in_sec": 100.0, "is_training_eligible": True},
            {"timetaken_in_sec": 999.0, "is_training_eligible": False},  # STOPPED-like
            {"timetaken_in_sec": 300.0, "is_training_eligible": True},
        ]
    )
    result = build_historical_features(df, prior_filter="training_eligible")
    rows = result.frame.sort_values("Start_Time")
    third = rows.iloc[2]
    assert third["hist_global_prior_count"] == 1
    assert third["hist_global_prior_mean_sec"] == 100.0  # 999 never enters


def test_all_names_present_and_natural_order():
    df = _exec_frame(
        [
            {"Start_Time": pd.Timestamp("2026-01-01"), "timetaken_in_sec": 10.0},
            {"Start_Time": pd.Timestamp("2026-01-02"), "timetaken_in_sec": 20.0},
        ]
    )
    result = build_historical_features(df, prior_filter="all")
    for name in HIST_FEATURE_NAMES:
        assert name in result.frame.columns, name


def test_missing_start_time_raises():
    df = _exec_frame(
        [
            {"Start_Time": pd.NaT, "timetaken_in_sec": 10.0},
        ]
    )
    with pytest.raises(ChronologyError):
        build_historical_features(df, prior_filter="all")
