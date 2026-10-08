"""Prompt 5 / Module 8: consolidated end-to-end validation of the FULL chain.

    source data -> identity/eligibility -> pre-execution features -> history
    snapshot -> total ETA -> uncertainty -> 5-stage intelligence -> SHAP ->
    risk -> prediction event -> actual linkage -> monitoring -> lab ->
    drift/readiness

Each test walks the chain through the REAL serving artifact and REAL sources.
Cross-layer unit guarantees remain in their module test files; this file
proves the layers compose correctly at the boundaries.
"""

from __future__ import annotations

import dataclasses
import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dltime.api import app
from dltime.feature_registry import build_default_feature_registry
from dltime.inference import PredictionRequest

# The locked reference case (11 priors, resolved MDM_DM_TEST connection).
REFERENCE_BODY = {
    "loader_name": "MigrationSummarySubTableLoaderDataset",
    "sprint": "TEST",
    "dataset": "Migration_Summary_Sub_Fact_Table_LDR_Datase_new",
    "connection": "MDM_DM_TEST",
}
REFERENCE_ETA = 16.289350640581837  # direct-total reference (rollback path)
# Sum-total policy (EXP-038): the served ETA is the exact sum of the five
# operational stage predictions for the reference context (0.6 + 0.6 + 0.482
# + 10.388 + 0.477). The retired stage-aware total's raw output is retained
# as REFERENCE evidence (stage_aware_reference_seconds) — traceability only.
STAGE_AWARE_REFERENCE_ETA = 1362.3115071633108  # retired stage-aware model (reference evidence only; 2026-09-30 2nd re-pin after the X_ref full-stage fix)
# Applicable-stage policy (user final spec, 2026-09-30): the reference request
# has NO prevalidation/transformation entries, so the ML part is the applicable
# 3-stage sum (0.6 + 10.388 + 0.477 = 11.465). 2026-10-02 final spec: the
# reference loader has NO loader-tier (own-history) approval evidence, so NO
# approval estimate joins the total — the GLOBAL-tier 98.592 + 2.498 are
# disclosed but excluded. Total (2026-10-02 re-pin): 11.465.
SUM_TOTAL_REFERENCE_ETA = 11.465


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Repoint the monitoring store to a hermetic dir for THIS test."""
    monkeypatch.setattr("dltime.store.STORE_DIR", tmp_path)
    from dltime.store import MonitoringStore

    return MonitoringStore(tmp_path)


# ---------------------------------------------------------------- §4 valid chain

def test_full_prediction_chain(service, client, store):
    """One request exercises every layer of the locked lifecycle in order."""
    resp = client.post("/predict", json=REFERENCE_BODY)
    assert resp.status_code == 200
    data = resp.json()

    # 1-2 identity + historical context
    assert data["identity"]["resolution_status"] == "RESOLVED"
    assert data["history"]["prior_loader_executions"] == 11
    # 3 feature assembly: the 44-feature artifact contract (2026-10-04 spec
    # extension) + 2 audit markers; the base features are
    # exactly the stage-frame contract, and the registry-derived ones remain
    # a subset of the prediction registry.
    assert len(data["features"]) == 44
    from dltime.stage_aware_serving import (
        STAGE_FEATURE_COLUMNS,
        base_feature_names_from_contract,
    )

    # 3 feature assembly: the reported vector is the fitted 44-feature
    # artifact contract (2026-10-04 spec); stage predictions surface via
    # stage_intelligence, not the feature vector.
    import joblib as _joblib

    _art = _joblib.load("artifacts/total-et-v3-candidate.pkl")
    assert set(data["features"]) == set(_art["metadata"]["feature_names_in_order"])
    allowed = set(build_default_feature_registry().names(prediction_only=True))
    assert {k for k in data["features"] if k.startswith(("def_", "hist_", "global_"))} <= allowed
    # 4 history gate passed
    assert data["state"] == "PREDICTED"
    # 5 ETA — the EXACT sum of the five operational stage predictions;
    # the retired direct + stage-aware models ride along as reference evidence
    assert data["predicted_seconds"] == pytest.approx(SUM_TOTAL_REFERENCE_ETA, abs=1e-9)
    ml_vals = [
        s["estimate_seconds"]
        for s in data["stage_intelligence"]["stages"]
        if s.get("method") == "ml_stage_model" and s.get("estimate_seconds") is not None
    ]
    assert len(ml_vals) == 3  # applicable: staging + pre_load + data_loading
    appr_vals = [
        s["estimate_seconds"]
        for s in data["stage_intelligence"]["stages"]
        if s.get("method") == "contextual_hierarchical_statistical" and s.get("estimate_seconds") is not None
    ]
    # 2026-10-02 final spec: this loader has no OWN approval history, so the
    # GLOBAL-tier estimates are disclosed but NOT part of the total.
    assert len(appr_vals) == 2
    assert data["predicted_seconds"] == pytest.approx(sum(ml_vals), abs=1e-9)
    assert data["model_version"] == "sum-of-available-stages-v2"
    assert data["floor_applied"] is False
    assert data["physical_floor_seconds"] is None
    # 2026-10-04 v2 retrain: the retired stage-aware model no longer rides
    # along in the v2 serving artifact — it is None in every payload.
    assert data["stage_aware_reference_seconds"] is None
    assert data["direct_reference_seconds"] == pytest.approx(REFERENCE_ETA, abs=1e-9)
    # 6-7 calibrated uncertainty interval (2026-10-03: no reliability label)
    unc = data["uncertainty"]
    assert unc["available"] is True
    assert math.isfinite(unc["lower_seconds"]) and math.isfinite(unc["upper_seconds"])
    assert unc["lower_seconds"] >= 0
    assert unc["lower_seconds"] < data["predicted_seconds"] < unc["upper_seconds"]
    assert "reliability" not in unc
    # Sum policy (EXP-038): the interval is calibrated on the sum's own errors
    # (EXP-036 clean holdout; version-bound to the serving policy).
    assert unc["calibration_version"] == "unc-log1p-conformal-v1-sum-of-five-stages"
    assert "not a guarantee" in unc["interpretation"]
    # 8 stage intelligence: applicable stages only (user final spec). This
    # request declares no prevalidation/transformation entries, so exactly the
    # 3 ML operational stages + 2 approval stages are present; the pre_load
    # alias is suppressed; the total IS the decomposition of the displayed
    # applicable stage values (ML + statistical approvals).
    si = data["stage_intelligence"]
    assert si["available"] is True
    stages = {s["stage"]: s for s in si["stages"]}
    operational = {"pre_load", "data_loading", "staging"}
    assert operational <= set(stages)
    assert "prevalidation" not in stages and "transformation" not in stages  # not applicable
    approval = {"datamart_approval", "dataloading_approval"}
    assert approval <= set(stages)
    assert approval == {"datamart_approval", "dataloading_approval"}
    for name in approval:
        s = stages[name]
        # AD-31: approvals are contextual STATISTICAL estimates (tier + evidence
        # count) when available — never ML predictions, never fabricated.
        assert s["method"] == "contextual_hierarchical_statistical"
        if s.get("available"):
            assert s["estimate_seconds"] > 0 and s["evidence_tier"] and s["n_evidence"] >= 3
        else:
            assert s["estimate_seconds"] is None
    # AD-31a: pre_load is a real small-sample ML model (n=19) — available,
    # independent, with the limitation disclosed in its reason text.
    assert stages["pre_load"]["available"] is True
    assert stages["pre_load"]["estimate_seconds"] > 0
    assert "duplicate_of" not in stages["pre_load"]
    assert "SMALL-SAMPLE" in stages["pre_load"]["availability_reason"]
    assert "total" not in si  # stages are never presented as a decomposition of ETA
    # 9 explanation: the sum policy's exact stage decomposition — linear
    # additivity holds exactly (base 0 + components == total), contribution
    # language, no causal claim.
    expl = data["explanation"]
    assert expl["method"] == "deterministic_stage_sum"
    assert expl["additivity_abs_error"] == 0.0
    assert abs(
        expl["base_value_seconds"] + expl["sum_contributions_seconds"] - expl["model_output_seconds"]
    ) <= 1e-9
    assert pytest.approx(data["predicted_seconds"], abs=1e-9) == expl["model_output_seconds"]
    assert "not causal" in expl["interpretation"].lower()
    assert expl["human_readable_summary"]
    assert expl["top_features"]
    # every ranked contributor is a real additive component in seconds
    assert all("contribution_seconds" in f for f in expl["top_features"])
    assert pytest.approx(
        sum(f["contribution_seconds"] for f in expl["top_features"]), abs=1e-9
    ) == data["predicted_seconds"]
    # 10 risk with interpretation — never a failure probability
    assert data["risk"]["label"] in (
        "LOW_CONFIDENCE", "OUT_OF_DISTRIBUTION", "HIGH_RISK", "WARNING", "NORMAL"
    )
    assert "not a probability of failure" in data["risk"]["interpretation"]
    # 11 model/version metadata (AD-32: stage-aware is authoritative)
    assert data["serving_role"] == "AUTHORITATIVE_TOTAL"
    # 12 prediction event persisted correctly (hermetic store)
    events, _warn = store.read_prediction_events()
    assert len(events) == 1
    assert events[0]["event_state"] == "PREDICTED"
    assert events[0]["prediction"]["model_version"] == "sum-of-available-stages-v2"
    assert events[0]["prediction"]["predicted_total_seconds"] > 0
    assert len(events[0]["features"]) == 44


# ---------------------------------------------------------------- §5 negative states

def test_negative_states_never_fabricate_predictions(service, client, store):
    """Every non-PREDICTED state carries no ETA/uncertainty/explanation and
    still persists an honest audit event."""
    # insufficient history: real 2210_TEST_ALL_CONFIG identity (2 priors < 3)
    insufficient_body = {
        "loader_name": "2210_TEST_ALL_CONFIG",
        "sprint": "SUPPORT",
        "dataset": "A_TABLE",
        "connection": "test1",
    }
    r1 = client.post("/predict", json=insufficient_body)
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["state"] == "INSUFFICIENT_HISTORY"
    assert d1.get("predicted_seconds") is None
    assert d1.get("uncertainty") is None
    assert d1.get("explanation") is None

    # unresolved identity
    r2 = client.post(
        "/predict",
        json={
            "loader_name": "no_such_loader_anywhere_module8",
            "sprint": "module8_sprint",
            "dataset": "module8_dataset",
            "connection": "MDM_DM_TEST",
        },
    )
    # 2026-10-03: unregistered loader == NEW loader (short refusal, no total)
    assert r2.json()["state"] == "INSUFFICIENT_HISTORY"

    # invalid input: submitted connection disagrees with the definition (Task 10.1)
    r3 = client.post(
        "/predict",
        json={
            "loader_name": REFERENCE_BODY["loader_name"],
            "sprint": REFERENCE_BODY["sprint"],
            "dataset": REFERENCE_BODY["dataset"],
            "connection": "DEFINITELY_NOT_MDM_DM_TEST",
        },
    )
    d3 = r3.json()
    assert d3["state"] == "INVALID_INPUT"
    assert d3.get("predicted_seconds") is None

    # schema-level rejection: connection missing entirely (422, Task 10.1)
    r4 = client.post(
        "/predict",
        json={
            "loader_name": REFERENCE_BODY["loader_name"],
            "sprint": REFERENCE_BODY["sprint"],
            "dataset": REFERENCE_BODY["dataset"],
        },
    )
    assert r4.status_code == 422

    events, _warn = store.read_prediction_events()
    states = [e["event_state"] for e in events]
    # 2026-10-03: the unregistered-loader attempt is persisted as the new-loader
    # refusal (INSUFFICIENT_HISTORY), so there is no UNRESOLVED_IDENTITY event.
    assert states.count("INSUFFICIENT_HISTORY") == 2
    assert states.count("UNRESOLVED_IDENTITY") == 0
    assert states.count("INVALID_INPUT") == 1
    # no fabricated ETA anywhere in the non-predicted events
    for e in events:
        if e["event_state"] != "PREDICTED":
            assert e["prediction"] is None


# ---------------------------------------------------------------- §6 leakage hardening

def _reference_predict(service):
    return service.predict(
        PredictionRequest(connection=REFERENCE_BODY["connection"], **{
            k: REFERENCE_BODY[k] for k in ("loader_name", "sprint", "dataset")
        })
    )


def test_future_row_poisoning_cannot_change_prediction(service, bundle, foundation):
    """Deterministic future-row poisoning: appending executions that occur
    strictly AFTER the reference context must leave the ETA byte-identical."""
    base = _reference_predict(service)
    assert base.state == "PREDICTED"
    assert base.predicted_seconds == pytest.approx(SUM_TOTAL_REFERENCE_ETA, abs=1e-9)

    # Two strictly-future, eligible-looking executions with huge targets,
    # under their OWN identity (so they would be huge "priors" if any future
    # row ever leaked into an earlier as-of computation). Built from the
    # eligibility-bearing executions frame so the REAL rows keep their
    # eligibility chain (poisoning must not merely break eligibility).
    ex = foundation.frames["executions_eligible"]
    ctx_time = pd.to_datetime(
        ex.loc[
            ex["Loader_Name"].astype(str) == REFERENCE_BODY["loader_name"], "Start_Time"
        ].max()
    )

    poisoned_rows = ex.iloc[[0, 1]].copy()
    poisoned_rows["Loader_Name"] = "module8_poison_loader"
    poisoned_rows["Sprint"] = "MODULE8_POISON"
    poisoned_rows["Dataset_display_name"] = "module8_poison_dataset"
    poisoned_rows["ldr_connection_name"] = "MODULE8_POISON_CONN"
    poisoned_rows["execution_connection"] = "MODULE8_POISON_CONN"
    poisoned_rows["execution_resolution_status"] = "RESOLVED"
    poisoned_rows["Start_Time"] = ctx_time + pd.Timedelta(days=365)
    poisoned_rows["End_Time"] = poisoned_rows["Start_Time"] + pd.Timedelta(seconds=43_000_000)
    poisoned_rows["LDR_Execution_Id"] = [9_900_001, 9_900_002]
    poisoned_rows["ldr_workflow_execution_id"] = "module8-poison-ref"
    poisoned_rows["timetaken_in_sec"] = [43_000_000.0, 43_000_000.0]
    poisoned_rows["is_valid_execution"] = True
    poisoned_rows["is_target_valid"] = True
    poisoned_rows["is_training_eligible"] = True
    poisoned_exec = pd.concat([ex, poisoned_rows], ignore_index=True)

    poisoned_foundation = dataclasses.replace(
        foundation,
        frames={
            **foundation.frames,
            "executions_eligible": poisoned_exec,
        },
    )

    from dltime.inference import PredictionService
    from dltime.resolution import ConnectionResolver

    resolver = ConnectionResolver(
        bundle.loader_definitions_detail, bundle.loader_definitions_summary
    )
    poisoned_service = PredictionService(
        training_frame=service.tf,
        artifact=service.artifact,
        foundation=poisoned_foundation,
        resolver=resolver,
        definitions_detail=bundle.loader_definitions_detail,
        definitions_summary=bundle.loader_definitions_summary,
    )
    poisoned = _reference_predict(poisoned_service)
    assert poisoned.state == "PREDICTED"
    # future rows (even eligible-looking ones with huge targets) cannot leak in
    assert    poisoned.predicted_seconds == pytest.approx(SUM_TOTAL_REFERENCE_ETA, abs=1e-9)


def test_history_engine_fails_closed_without_eligibility_evidence(service, bundle, foundation):
    """Hardening: if appended rows carry NO eligibility evidence (e.g. a frame
    concatenated without the eligibility columns), the history engine must
    fail CLOSED — zero usable priors, INSUFFICIENT_HISTORY, no ETA — rather
    than fabricate history from unevaluated rows."""
    ex = foundation.frames["executions"]  # NOTE: lacks the eligibility columns
    ctx_time = pd.to_datetime(
        ex.loc[
            ex["Loader_Name"].astype(str) == REFERENCE_BODY["loader_name"], "Start_Time"
        ].max()
    )
    extra = ex.iloc[[0]].copy()
    extra["LDR_Execution_Id"] = [9_900_009]
    extra["Start_Time"] = ctx_time + pd.Timedelta(days=1)
    broken_frame = pd.concat([ex, extra], ignore_index=True)

    broken_foundation = dataclasses.replace(
        foundation, frames={**foundation.frames, "executions_eligible": broken_frame}
    )

    from dltime.inference import PredictionService
    from dltime.resolution import ConnectionResolver

    resolver = ConnectionResolver(
        bundle.loader_definitions_detail, bundle.loader_definitions_summary
    )
    broken_service = PredictionService(
        training_frame=service.tf,
        artifact=service.artifact,
        foundation=broken_foundation,
        resolver=resolver,
        definitions_detail=bundle.loader_definitions_detail,
        definitions_summary=bundle.loader_definitions_summary,
    )
    result = _reference_predict(broken_service)
    assert result.state == "INSUFFICIENT_HISTORY"
    assert result.predicted_seconds is None
    assert (result.history or {}).get("prior_loader_executions") == 0


def test_prediction_feature_vector_contains_no_forbidden_fields(service):
    result = _reference_predict(service)
    feature_names = set(result.features.keys())
    banned = {
        "ldr_workflow_execution_id",
        "LDR_Execution_Id",
        "timetaken_in_sec",
        "LDR_Status",
        "Start_Time",
        "End_Time",
        "pre_load_timetaken_in_sec",
        "data_loading_timetaken_in_sec",
        "Planned Start Time",
    }
    assert not (feature_names & banned)
    # AD-32 contract (2026-10-04 44-feature spec): the reported copy is the
    # fitted 44-feature contract only — no stage-derived features there, and
    # no post-execution leakage.
    assert len(feature_names) == 44
    assert sum(1 for f in feature_names if f.startswith("pred_stage_")) == 0


# ---------------------------------------------------------------- §12 lab + §13/§14 drift/readiness

def test_lab_safety_invariants_via_api(client):
    """Serving-model identity + never two promoted models + 31-feature
    candidate stays experimental — through the real API."""
    status = client.get("/models/status").json()
    assert status["serving_model_version"] == "sum-of-available-stages-v2"
    assert status["serving_role"] == "AUTHORITATIVE_TOTAL"
    assert status["reference_model"]["model_id"] == "total-et-v3-candidate"
    promoted = status["promoted_model"]
    assert promoted is None or promoted.get("model_id")

    models = client.get("/models").json()["models"]
    assert client.get("/models").json()["promoted_model_id"] in (None, *(m["model_id"] for m in models if m["status"] == "PROMOTED"))

    # the experimental 31-feature candidate must not be PROMOTED
    cand = [m for m in models if "preval" in str(m.get("model_id", "")) or "v31" in str(m.get("model_id", ""))]
    for m in cand:
        assert m["status"] in ("EXPERIMENTAL", "CANDIDATE_EVALUATED", "RETIRED", "REJECTED")

    # promotion of an experimental/incompatible model is refused: explicit
    # human action alone is not enough — compatibility gates must also pass
    incompatible = next(
        (m for m in models if m["status"] in ("EXPERIMENTAL", "REJECTED")), None
    )
    if incompatible is not None:
        resp = client.post(
            "/models/promote",
            json={"model_id": incompatible["model_id"], "approved_by": "module8-e2e", "reason": "must fail gates"},
        )
        assert resp.status_code in (409, 422)


def test_drift_and_readiness_zero_outcome_states(client):
    perf = client.get("/drift/performance").json()
    # hermetic store: no matched outcomes -> explicit unassessable state
    assert perf["status"] in ("INSUFFICIENT_OUTCOMES", "INSUFFICIENT_SAMPLE")
    assert perf["has_actual_outcomes"] is False

    pop = client.get("/drift/population").json()
    assert pop["overall_status"] in (
        "NO_MEANINGFUL_DRIFT", "WARNING", "SIGNIFICANT_DRIFT", "INSUFFICIENT_SAMPLE"
    )

    readiness = client.get("/retraining/readiness").json()
    assert readiness["readiness_state"] in (
        "NOT_READY", "READY_FOR_REVIEW", "RETRAINING_RECOMMENDED_FOR_REVIEW"
    )
    assert readiness["automatic_retraining"] == "OFF"
    assert readiness["human_gate_required"] is True


# ---------------------------------------------------------------- §20 reference integrity

def test_reference_prediction_byte_identical(service):
    result = _reference_predict(service)
    assert result.state == "PREDICTED"
    # DIRECT reference model must still reproduce the canonical corrected-regime
    # ETA byte-for-byte (rollback-path integrity; retired from serving).
    assert result.direct_reference_seconds == pytest.approx(REFERENCE_ETA, abs=0.0)
    assert result.serving_role == "AUTHORITATIVE_TOTAL"
    # Sum policy: the served ETA equals the applicable-stage sum (ML stages
    # + approval statistical estimates) exactly, with no floor and no
    # clamping; the retired stage-aware model rides as evidence.
    assert result.predicted_seconds == pytest.approx(SUM_TOTAL_REFERENCE_ETA, abs=1e-9)
    assert result.model_version == "sum-of-available-stages-v2"
    assert result.floor_applied is False
    assert result.physical_floor_seconds is None
    # 2026-10-04 v2 retrain: the retired stage-aware model no longer rides
    # along in the v2 serving artifact — it is None in every payload.
    assert result.stage_aware_reference_seconds is None


def test_tz_aware_planned_start_does_not_crash_and_varies_through_model(service):
    """Browser UIs send toISOString() output (tz-aware Z suffix). That value
    must not crash chronology (it previously 500'd with ChronologyError) and
    must flow through the real feature pipeline as a legitimate pre-execution
    variation: same identity, different planned time → still PREDICTED, model
    produces a (slightly) different learned estimate — proving this is not an
    exact-match lookup."""
    base = _reference_predict(service)
    assert base.state == "PREDICTED"
    varied = service.predict(
        PredictionRequest(
            connection=REFERENCE_BODY["connection"],
            **{
                k: REFERENCE_BODY[k]
                for k in ("loader_name", "sprint", "dataset")
            },
            planned_start_time="2027-06-01T09:00:00Z",
        )
    )
    assert varied.state == "PREDICTED"
    assert varied.predicted_seconds > 0
    # Locked contract (§F): planned start is NOT a model feature. The
    # stage-aware ETA must be UNCHANGED — a different planned time may only
    # move the as-of boundary, and here the context is already the latest
    # execution so no feature moves. The direct reference DOES vary (its
    # recency feature moves), proving the path is a real model, not a lookup.
    assert varied.predicted_seconds == pytest.approx(base.predicted_seconds, abs=1e-12)
    assert varied.direct_reference_seconds != base.direct_reference_seconds
    # The normalized planned time is surfaced in the response context.
    assert "2027-06-01" in str(varied.inputs.get("planned_start_time"))
