"""Final-requirements tests (user spec 2026-09-30).

Covers:
  1. Applicable-stage determination from pre-execution entries.
  2. sum_total_v2 skips non-applicable stages instead of refusing the total.
  3. Approval statistical ladder: loader-specific evidence (2-of-4 sparse
     observations) wins over GLOBAL; global fallback fires with 1+ observation.
  4. Explanation: approvals are included components; additivity holds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dltime.stage_intel import (
    determine_applicable_stages,
    filter_stage_intel_to_applicable,
)
from dltime.stage_v2 import SUM_VERSION, sum_total_v2


# --------------------------------------------------------------- 1. applicable
def test_applicable_default_excludes_prevalidation_transformation():
    got = determine_applicable_stages({})
    assert "staging" in got and "pre_load" in got and "data_loading" in got
    assert "datamart_approval" in got and "dataloading_approval" in got
    assert "prevalidation" not in got and "transformation" not in got


@pytest.mark.parametrize(
    "overrides,stage",
    [
        ({"prevalidation_count": 2}, "prevalidation"),
        ({"prevalidation_display_name": "P"}, "prevalidation"),
        ({"transformation_count": 3}, "transformation"),
        ({"transformation_display_name": "T"}, "transformation"),
        ({"preval_trans_display_name": "DO"}, "prevalidation"),
    ],
)
def test_applicable_conditional_entries(overrides, stage):
    assert stage in determine_applicable_stages(overrides)


def test_applicable_nan_count_does_not_qualify():
    assert "prevalidation" not in determine_applicable_stages(
        {"prevalidation_count": float("nan")}
    )


# ------------------------------------------------------------------- 2. filter
def _stage_entry(stage, available=True, seconds=1.0):
    return {
        "stage": stage,
        "available": available,
        "estimate_seconds": seconds if available else None,
        "method": "ml_stage_model" if stage not in ("datamart_approval", "dataloading_approval")
        else "contextual_hierarchical_statistical",
    }


def _all_stages():
    return [
        _stage_entry("staging", seconds=1.0),
        _stage_entry("prevalidation", available=False),
        _stage_entry("transformation", available=False),
        _stage_entry("pre_load", seconds=2.0),
        _stage_entry("data_loading", seconds=3.0),
        _stage_entry("datamart_approval", seconds=0.5),
        _stage_entry("dataloading_approval", seconds=0.5),
    ]


def test_filter_marks_not_applicable():
    applicable = determine_applicable_stages({})
    out = filter_stage_intel_to_applicable({"stages": _all_stages()}, applicable)
    names = {s["stage"] for s in out["stages"]}
    assert "prevalidation" not in names and "transformation" not in names
    assert set(out["not_applicable_stages"]) == {"prevalidation", "transformation"}


def test_sum_skips_non_applicable_instead_of_refusing():
    case = {"case": "KNOWN_LOADER_TRUSTED", "total_available": True, "message": "ok"}
    intel = {
        "applicable_stages": list(determine_applicable_stages({})),
        "stages": _all_stages(),
    }
    result = sum_total_v2(intel, case)
    assert result["sum_available"] is True
    # 1 + 2 + 3 operational + 0.5 + 0.5 approvals
    assert result["total_seconds"] == pytest.approx(7.0)
    assert sorted(result["not_applicable_stages"]) == ["prevalidation", "transformation"]


def test_sum_still_refuses_when_applicable_stage_unavailable():
    case = {"case": "KNOWN_LOADER_TRUSTED", "total_available": True, "message": "ok"}
    stages = _all_stages()
    for s in stages:
        if s["stage"] == "pre_load":
            s["available"] = False
            s["estimate_seconds"] = None
    intel = {
        "applicable_stages": list(determine_applicable_stages({})),
        "stages": stages,
    }
    result = sum_total_v2(intel, case)
    assert result["sum_available"] is False
    assert "pre_load" in result["unavailable_stages"]


def test_sum_version_unchanged():
    assert SUM_VERSION == "sum-of-available-stages-v2"


# ------------------------------------------------------- 3. approval estimator
def _approval_frame(n_loader_rows: int, loader_positive: int, global_positive: int):
    """Frame with the context row appended (last, by Start_Time)."""
    rows = []
    ts = pd.Timestamp("2026-01-01")
    eid = 1
    for i in range(n_loader_rows):
        val = float(i + 1) * 10 if i < loader_positive else np.nan
        rows.append(
            {
                "LDR_Execution_Id": eid,
                "Start_Time": ts,
                # sprint/connection/dataset vary so only the LOADER tier can
                # group these rows (narrower tiers must not fire)
                "Sprint": f"S{i % 2}",
                "execution_connection": f"C{i % 2}",
                "Loader_Name": "L",
                "Dataset_display_name": f"D{i % 2}",
                "stage_datamart_approval_sec": val,
                "is_training_eligible": True,
            }
        )
        eid += 1
        ts += pd.Timedelta(hours=1)
    for i in range(global_positive):
        rows.append(
            {
                "LDR_Execution_Id": eid,
                "Start_Time": ts,
                "Sprint": "S2",
                "execution_connection": "C2",
                "Loader_Name": "OTHER",
                "Dataset_display_name": "D2",
                "stage_datamart_approval_sec": 500.0 + i,
                "is_training_eligible": True,
            }
        )
        eid += 1
        ts += pd.Timedelta(hours=1)
        rows.append(
            {
                "LDR_Execution_Id": eid,
                "Start_Time": ts,
                "Sprint": "S1",
                "execution_connection": "C1",
                "Loader_Name": "L",
                "Dataset_display_name": "D",
                "stage_datamart_approval_sec": np.nan,
                "is_training_eligible": False,
            }
        )
    return pd.DataFrame(rows)


CTX_ID = None


def _ctx_id(frame):
    return int(frame["LDR_Execution_Id"].iloc[-1])


def test_approval_loader_own_tier_wins_with_two_observations():
    """2026-10-03: the NARROWEST loader-own tier with evidence wins (a single
    recorded timing is used as itself; 2+ use the median). Every loader-matched
    tier IS this loader's own history, so the estimate is included in the total
    and displayed — the GLOBAL fallback stays hidden/excluded."""
    from dltime.approval_estimator import ApprovalTier, estimate_approval_stage

    frame = _approval_frame(n_loader_rows=4, loader_positive=2, global_positive=50)
    est = estimate_approval_stage(frame, _ctx_id(frame), "datamart_approval")
    assert est.available is True
    # narrowest loader-own tier first; a SINGLE recorded timing is used as itself
    assert est.tier == ApprovalTier.LOADER_CONNECTION
    assert est.n_evidence == 1
    assert est.estimate_seconds == pytest.approx(20.0)
    assert est.to_payload()["included_in_total"] is True
    # with the statistical threshold raised to 2 observations the loader tier
    # pools both of this loader's recorded timings -> median (10.0, 20.0) = 15.0
    est2 = estimate_approval_stage(
        frame, _ctx_id(frame), "datamart_approval", min_tier_observations=2
    )
    assert est2.available is True
    assert est2.tier == ApprovalTier.LOADER
    assert est2.n_evidence == 2
    assert est2.estimate_seconds == pytest.approx(15.0)
    assert est2.to_payload()["included_in_total"] is True


def test_approval_global_fallback_with_single_observation():
    from dltime.approval_estimator import ApprovalTier, estimate_approval_stage

    frame = _approval_frame(n_loader_rows=4, loader_positive=0, global_positive=1)
    est = estimate_approval_stage(frame, _ctx_id(frame), "datamart_approval")
    assert est.available is True
    assert est.tier == ApprovalTier.GLOBAL
    assert est.n_evidence == 1
    assert est.estimate_seconds == pytest.approx(500.0)


def test_approval_unavailable_when_zero_evidence_anywhere():
    from dltime.approval_estimator import estimate_approval_stage

    frame = _approval_frame(n_loader_rows=4, loader_positive=0, global_positive=0)
    est = estimate_approval_stage(frame, _ctx_id(frame), "datamart_approval")
    assert est.available is False
    assert est.estimate_seconds is None


# ---------------------------------------------------------------- 4. explanation
def test_explanation_includes_approvals_and_additivity():
    from dltime.sum_total_serving import sum_total_explanation

    case_intel = {
        "applicable_stages": [
            "staging", "pre_load", "data_loading",
            "datamart_approval", "dataloading_approval",
        ],
        "stages": _all_stages(),
    }
    total = 1.0 + 2.0 + 3.0 + 0.5 + 0.5
    expl = sum_total_explanation(case_intel, total)
    names = {f["stage"] if "stage" in f else f["feature"] for f in expl["top_features"]}
    assert "pred_stage_datamart_approval_sec" in names
    assert all(f["included_in_total"] for f in expl["top_features"])
    contributions = sum(f["contribution_seconds"] for f in expl["top_features"])
    assert contributions == pytest.approx(expl["model_output_seconds"])
    assert "applicable stage predictions plus the approval estimates" in expl["human_readable_summary"]
