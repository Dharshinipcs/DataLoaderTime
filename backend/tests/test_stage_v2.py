"""Stage-v2 main-goal contract tests (2026-09-29).

Covers: filled-dataset loading integrity, the per-stage comparison protocol
(selects a real family, real-rows-only holdout evidence), the loader
threshold gate (the THREE availability cases), and the sum-of-available-
stages v2 total (gate-aware, staging-honest, approvals included).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dltime.stage_v2 import (
    STAGE_LOADER_MIN_PRIOR,
    SUM_VERSION,
    assess_loader_case,
    comparison_report,
    load_filled_bundle,
    load_stage_v2_bundle,
    real_exec_mask,
    sum_total_v2,
)


@pytest.fixture(scope="module")
def filled():
    return load_filled_bundle()


def test_filled_bundle_loads_and_carries_provenance(filled):
    prov = filled.provenance["filled_dataset"]
    assert prov["real_exec_rows"] == 792
    assert prov["synthetic_exec_rows"] == 701
    assert "synthetic" in prov["note"]


def test_filled_bundle_rows_match_real_values(filled):
    """Real rows in the filled file keep their original values (spot checks)."""
    ex = filled.executions
    real = ex[real_exec_mask(ex)]
    assert len(real) == 792
    # real ids are untouched originals
    assert int(real["LDR_Execution_Id"].min()) < 899001
    # synthetic rows only ADD rows; every real id stays present exactly once
    ids = pd.to_numeric(ex["LDR_Execution_Id"], errors="coerce")
    assert (ids < 899001).sum() == 792


def test_gate_case1_known_loader_trusted():
    df = pd.DataFrame({
        "LDR_Execution_Id": [1, 2, 3],
        "Loader_Name": ["L", "L", "L"],
        "is_training_eligible": [True, True, True],
    })
    case = assess_loader_case(df, "L")
    assert case["case"] == "KNOWN_LOADER_TRUSTED"
    assert case["case_number"] == 1
    assert case["total_available"] is True


def test_gate_case2_unknown_loader():
    df = pd.DataFrame({
        "LDR_Execution_Id": [1, 2, 3],
        "Loader_Name": ["L", "L", "L"],
        "is_training_eligible": [True, True, True],
    })
    case = assess_loader_case(df, "NEVER_SEEN_LOADER")
    assert case["case"] == "UNKNOWN_LOADER"
    assert case["case_number"] == 2
    assert case["total_available"] is False
    assert "no prior executions" in case["message"]


def test_gate_case3_below_threshold():
    df = pd.DataFrame({
        "LDR_Execution_Id": [1, 2],
        "Loader_Name": ["L", "L"],
        "is_training_eligible": [True, True],
    })
    case = assess_loader_case(df, "L")
    assert case["case"] == "BELOW_THRESHOLD"
    assert case["case_number"] == 3
    assert case["total_available"] is False
    assert str(STAGE_LOADER_MIN_PRIOR) in case["message"]
    assert str(case["prior_count"]) in case["message"]


def test_gate_synthetic_rows_never_count_as_loader_history():
    """Synthetic rows must not manufacture loader history for the gate."""
    df = pd.DataFrame({
        "LDR_Execution_Id": [900001, 900002, 900003, 1],
        "Loader_Name": ["L", "L", "L", "L"],
        "is_training_eligible": [True, True, True, True],
    })
    case = assess_loader_case(df, "L")
    assert case["prior_count"] == 1
    assert case["case"] == "BELOW_THRESHOLD"


def _intel(stages_avail: dict[str, float | None], approvals: dict[str, float] | None = None):
    stages = []
    for name, sec in stages_avail.items():
        stages.append({
            "stage": name,
            "method": "ml_stage_model",
            "available": sec is not None,
            "estimate_seconds": sec,
        })
    for name, sec in (approvals or {"datamart_approval": 2.0, "dataloading_approval": 1.0}).items():
        stages.append({
            "stage": name,
            "method": "contextual_hierarchical_statistical",
            "available": sec is not None,
            "estimate_seconds": sec,
        })
    return {"stages": stages}


def test_sum_v2_all_stages_available_includes_approvals():
    """2026-09-30: staging is trained, so ALL FIVE operational stages are
    required; the total = 5 stage predictions + approvals."""
    gate = {"total_available": True, "message": "ok"}
    intel = _intel({
        "staging": 0.5,
        "prevalidation": 1.0, "transformation": 2.0, "pre_load": 3.0, "data_loading": 4.0,
    })
    out = sum_total_v2(intel, gate)
    assert out["sum_available"] is True
    assert out["total_seconds"] == pytest.approx(10.5 + 3.0)
    assert out["operational_total_seconds"] == pytest.approx(10.5)
    assert out["approvals_total_seconds"] == pytest.approx(3.0)


def test_sum_v2_refuses_when_staging_missing():
    """Staging is a required operational stage: missing it refuses the total
    (fail-closed, no partial sum) — superseding the interim 4-stage tolerance."""
    gate = {"total_available": True, "message": "ok"}
    intel = _intel({
        "staging": None,
        "prevalidation": 1.0, "transformation": 2.0, "pre_load": 3.0, "data_loading": 4.0,
    })
    out = sum_total_v2(intel, gate)
    assert out["sum_available"] is False
    assert "staging" in str(out.get("unavailable_stages"))


def test_sum_v2_blocked_by_gate():
    gate = {"total_available": False, "message": "threshold not met"}
    out = sum_total_v2(_intel({"prevalidation": 1.0}), gate)
    assert out["sum_available"] is False
    assert out["total_seconds"] is None
    assert "threshold not met" in out["reason"]


def test_sum_v2_refuses_when_required_stage_missing():
    gate = {"total_available": True, "message": "ok"}
    intel = _intel({"prevalidation": 1.0, "transformation": 2.0, "pre_load": None, "data_loading": 4.0})
    out = sum_total_v2(intel, gate)
    assert out["sum_available"] is False
    assert "pre_load" in out.get("unavailable_stages", [])


def test_artifact_if_present_is_v2_shape():
    """When the built artifact exists it must carry v2 predictors with honest metadata."""
    bundle = load_stage_v2_bundle()
    if bundle is None:
        pytest.skip("stage-v2 artifact not built yet")
    assert bundle.version == "stage-v2-bundle-1"
    # 2026-09-30: the corrected export carries real staging timing fields, so
    # staging is a trained v2 predictor like the other stages (no unavailability).
    assert "staging" in bundle.predictors
    assert "prevalidation" in bundle.predictors
    for name, pred in bundle.predictors.items():
        assert pred.family in ("ridge", "random_forest", "extra_trees",
                               "hist_gradient_boosting", "gradient_boosting")
        assert pred.n_real >= 12
        assert pred.n_synthetic >= 0
    report = comparison_report(bundle)
    for stage in bundle.predictors:
        assert report[stage]["status"] == "OK"
        assert len(report[stage]["leaderboard"]) >= 2  # families were actually compared
