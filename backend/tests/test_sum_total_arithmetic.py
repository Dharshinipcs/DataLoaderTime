"""Controlled total arithmetic — the sum policy's core guarantee (prompt §35).

These tests pin the FINAL PRODUCT DECISION arithmetic with controlled values,
independent of any real model output:

  1. 10 + 20 + 30 + 40 + 50  ->  TOTAL == 150  (exactly; approval estimates
     excluded in LEGACY payloads, no floor, no hidden adjustment, no
     correction factor);
  2. changing ONE stage prediction changes the total by EXACTLY that delta
     (10/20/30/40/60 -> 160; +10 on data_loading moves the total +10);
  3. the deterministic sum helper itself (sum_total_serving) obeys the same
     identity on synthetic stage payloads;
  4. in LEGACY payloads (no applicable_stages key) approvals (even absurdly
     large contextual estimates) NEVER enter the total; in applicable-mode
     payloads they enter as statistical components;
  5. the persisted prediction event records the SAME total the response
     served, and the feedback error math compares THAT value against the
     actual (prediction-vs-actual stays honest under the new definition).

The serving-path identity (response total == sum of the five DISPLAYED
operational stage values) is additionally re-probed against the REAL
artifacts + REAL sources at the end of this module.
"""

from __future__ import annotations

import math

import pytest


# ------------------------------------------------------------- unit: sum helper
def _intel(stages: dict[str, float | None], approvals: dict[str, float] | None = None, available: bool = True):
    """A synthetic Seven-Stage payload with exact controlled numbers."""
    def entry(name: str, v: float | None, method: str):
        return {
            "stage": name,
            "estimate_seconds": v,
            "available": v is not None and available,
            "method": method,
        }

    return {
        "available": True,
        "stages": [entry(name, v, "ml_stage_model") for name, v in stages.items()]
        + [entry(name, v, "contextual_hierarchical_statistical") for name, v in (approvals or {}).items()],
    }


FIVE = {"staging": 10.0, "prevalidation": 20.0, "transformation": 30.0, "pre_load": 40.0, "data_loading": 50.0}


def test_sum_helper_exact_150():
    from dltime.sum_total_serving import sum_total_from_stage_intel

    meta = sum_total_from_stage_intel(_intel(FIVE, {"datamart_approval": 98.4, "dataloading_approval": 2.7}))
    assert meta["ml_stage_sum_seconds"] == pytest.approx(150.0, abs=0.0)
    assert meta["model_version"] == "sum-of-available-stages-v2"
    assert meta["ml_stages_summed"] == ["staging", "prevalidation", "transformation", "pre_load", "data_loading"]
    # legacy mode: approvals recorded as excluded evidence, never summed
    assert {a["stage"] for a in meta["approvals_excluded"]} == {"datamart_approval", "dataloading_approval"}


def test_sum_helper_delta_single_stage():
    from dltime.sum_total_serving import sum_total_from_stage_intel

    base = sum_total_from_stage_intel(_intel(FIVE))["ml_stage_sum_seconds"]
    changed = sum_total_from_stage_intel(_intel({**FIVE, "data_loading": 60.0}))["ml_stage_sum_seconds"]
    assert base == pytest.approx(150.0, abs=0.0)
    assert changed == pytest.approx(160.0, abs=0.0)
    assert changed - base == pytest.approx(10.0, abs=0.0)  # exactly the data_loading delta


def test_sum_helper_refuses_when_a_stage_is_missing():
    from dltime.sum_total_serving import SumTotalError, sum_total_from_stage_intel

    broken = {**FIVE, "pre_load": None}
    with pytest.raises(SumTotalError):
        sum_total_from_stage_intel(_intel(broken))


def test_sum_helper_refuses_approvals_only_payload():
    from dltime.sum_total_serving import SumTotalError, sum_total_from_stage_intel

    with pytest.raises(SumTotalError):
        sum_total_from_stage_intel(_intel({}, {"datamart_approval": 98.4}))


def test_sum_explanation_additivity_exact():
    from dltime.sum_total_serving import sum_total_explanation

    # Legacy payload (no applicable_stages key): operational stages only,
    # approvals rendered as explicit non-components (pre-2026-09-30 shape).
    payload = sum_total_explanation(
        _intel(FIVE, {"datamart_approval": 98.4, "dataloading_approval": 2.7}), 150.0
    )
    assert payload["method"] == "deterministic_stage_sum"
    assert payload["additivity_abs_error"] == 0.0
    assert payload["model_output_seconds"] == pytest.approx(150.0, abs=0.0)
    ranked_sum = sum(f["contribution_seconds"] for f in payload["top_features"])
    assert ranked_sum == pytest.approx(150.0, abs=0.0)
    non_components = [f for f in payload["features"] if f["included_in_total"] is False]
    assert {f["feature"] for f in non_components} == {
        "pred_stage_datamart_approval_sec",
        "pred_stage_dataloading_approval_sec",
    }
    assert all(f["contribution_seconds"] == 0.0 for f in non_components)


def test_sum_explanation_applicable_includes_approvals():
    from dltime.sum_total_serving import sum_total_explanation

    # Applicable-mode payload (user final spec, 2026-09-30): approvals that
    # are applicable ARE included statistical components of the total.
    intel = _intel(FIVE, {"datamart_approval": 98.4, "dataloading_approval": 2.7})
    intel["applicable_stages"] = [
        "staging", "prevalidation", "transformation", "pre_load", "data_loading",
        "datamart_approval", "dataloading_approval",
    ]
    payload = sum_total_explanation(intel, 251.1)
    assert payload["model_output_seconds"] == pytest.approx(251.1, abs=0.0)
    ranked_sum = sum(f["contribution_seconds"] for f in payload["top_features"])
    assert ranked_sum == pytest.approx(251.1, abs=0.0)
    approvals = [f for f in payload["top_features"] if "approval" in f["feature"]]
    assert {f["feature"] for f in approvals} == {
        "pred_stage_datamart_approval_sec",
        "pred_stage_dataloading_approval_sec",
    }
    assert all(f["included_in_total"] for f in payload["top_features"])


# ------------------------------------------------- integration: serving + event
@pytest.fixture(scope="module")
def _svc():
    from dltime.inference import PredictionService

    return PredictionService.from_sources()


def _reference_request():
    from dltime.inference import PredictionRequest

    return PredictionRequest(
        loader_name="MigrationSummarySubTableLoaderDataset",
        sprint="TEST",
        dataset="Migration_Summary_Sub_Fact_Table_LDR_Datase_new",
        connection="MDM_DM_TEST",
    )


def test_served_total_equals_displayed_stage_sum_on_real_models(_svc):
    """The REAL serving path (real artifacts, real sources) obeys the identity:
    response total == sum of the displayed applicable ML stage values plus the
    LOADER-TIER approval statistical estimates (2026-10-02 user final spec:
    only the loader's OWN approval history joins the total)."""
    # applicable-stage contract (user final spec, 2026-09-30): a request
    # without prevalidation/transformation entries runs 3 applicable stages;
    # the reference loader has NO loader-tier approval evidence, so NO approval
    # estimate joins the total (the GLOBAL-tier 98.592+2.498 are disclosed but
    # excluded per the 2026-10-02 final spec).
    r = _svc.predict(_reference_request())
    assert r.state == "PREDICTED"
    si = r.stage_intelligence or {}
    vals = {
        s["stage"]: s["estimate_seconds"]
        for s in si.get("stages", [])
        if s.get("method") == "ml_stage_model" and s.get("estimate_seconds") is not None
    }
    assert set(vals) == {"staging", "pre_load", "data_loading"} or set(vals) == {
        "staging", "prevalidation", "transformation", "pre_load", "data_loading"
    }
    approval_sum = sum(
        s["estimate_seconds"]
        for s in si.get("stages", [])
        if s.get("method") == "contextual_hierarchical_statistical"
        and s.get("estimate_seconds") is not None
        and s.get("included_in_total") is not False
    )
    assert r.predicted_seconds == pytest.approx(sum(vals.values()) + approval_sum, rel=0, abs=1e-9)
    # 2026-10-02 re-pin: the reference identity carries no LOADER-tier approval
    # history, so the total is the 3-stage ML sum 0.6 + 10.388 + 3.042 = 11.465
    # (the GLOBAL-tier approval estimates 98.592 + 2.498 are excluded).
    assert r.predicted_seconds == pytest.approx(11.465, rel=0, abs=1e-6)
    # no floor, no hidden adjustment
    assert r.floor_applied is False
    assert r.physical_floor_seconds is None


def test_monkeypatched_stage_values_drive_total_exactly(_svc, monkeypatch):
    """Injecting controlled stage estimates (10/20/30/40/50) through the REAL
    stage-model cache must serve TOTAL == 100 + available approval estimates —
    arithmetic isolated from the fitted models proves there is no hidden
    learned adjustment."""
    from dltime.stage_intel import build_stage_intel_for_context

    real_builder = build_stage_intel_for_context

    def fake_builder(combined, execution_id, stage_models=None):
        result = real_builder(combined, execution_id, stage_models=stage_models)
        fixed = {"staging": 10.0, "prevalidation": 20.0, "transformation": 30.0, "pre_load": 40.0, "data_loading": 50.0}
        for s in result.per_stage.values():
            if s.get("method") == "ml_stage_model" and s.get("stage") in fixed:
                s["estimate_seconds"] = fixed[s["stage"]]
                s["available"] = True
                s["availability"] = "ML_PREDICTED"
        return result

    monkeypatch.setattr("dltime.stage_intel.build_stage_intel_for_context", fake_builder)
    r = _svc.predict(_reference_request())
    assert r.state == "PREDICTED"
    # applicable stages for the no-prevalidation/transformation request:
    # staging + pre_load + data_loading = 10 + 40 + 50 = 100. The reference
    # loader has NO loader-tier approval history, so NO approval estimate
    # joins the total (2026-10-02 final spec).
    appr = sum(
        s["estimate_seconds"]
        for s in (r.stage_intelligence or {}).get("stages", [])
        if s.get("method") == "contextual_hierarchical_statistical"
        and s.get("estimate_seconds") is not None
        and s.get("included_in_total") is not False
    )
    assert appr == pytest.approx(0.0, abs=1e-9)
    assert r.predicted_seconds == pytest.approx(100.0 + appr, rel=0, abs=1e-9)
    # the displayed applicable stages are exactly the injected values
    displayed = {
        s["stage"]: s["estimate_seconds"]
        for s in (r.stage_intelligence or {}).get("stages", [])
        if s.get("method") == "ml_stage_model"
    }
    assert displayed == {"staging": 10.0, "pre_load": 40.0, "data_loading": 50.0}
    # explanation decomposes the same applicable total exactly
    e = r.explanation or {}
    assert e.get("model_output_seconds") == pytest.approx(100.0 + appr, abs=1e-9)
    assert sum(f["contribution_seconds"] for f in e["top_features"]) == pytest.approx(100.0 + appr, abs=1e-9)


def test_prediction_event_records_served_total_consistently(_svc, tmp_path, monkeypatch):
    """The persisted event stores the SAME total the response served, and the
    feedback error math compares THAT value against the actual."""
    import dltime.store as store_mod

    monkeypatch.setattr(store_mod, "STORE_DIR", tmp_path)
    from dltime.store import MonitoringStore

    store = MonitoringStore(tmp_path)

    captured: dict = {}

    def fake_append(event):
        captured["event"] = event
        return None

    monkeypatch.setattr(store, "append_prediction_event", fake_append)
    monkeypatch.setattr("dltime.store.MonitoringStore", lambda *a, **k: store)

    r = _svc.predict(_reference_request())
    event = captured["event"]
    assert event["event_state"] == "PREDICTED"
    assert event["prediction"]["predicted_total_seconds"] == pytest.approx(r.predicted_seconds, abs=1e-9)
    assert event["prediction"]["model_version"] == "sum-of-available-stages-v2"

    # feedback math on the matched pair uses the served value
    from dltime.feedback import compute_total_error

    errs = compute_total_error(float(event["prediction"]["predicted_total_seconds"]), 100.0)
    assert errs["signed_error_seconds"] == pytest.approx(r.predicted_seconds - 100.0, abs=1e-6)
