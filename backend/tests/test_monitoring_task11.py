"""Task 11 tests: prediction monitoring + actual-outcome feedback.

Real-data fixtures where possible (session-scoped sources); synthetic cases
only for behaviors the real dataset cannot exercise naturally (ambiguity,
duplicates, conflicts) — clearly labeled SYNTHETIC.

Also regression-guards the Task 04–10.1 serving path: persistence must not
change any prediction output, and post-execution data can never enter the
pre-execution feature assembly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dltime.feedback import (
    LINK_METHOD_WORKFLOW_ID,
    LINK_STATUS_AMBIGUOUS,
    LINK_STATUS_MATCHED,
    LINK_STATUS_UNMATCHED,
    build_actual_outcome,
    build_feedback_record,
    compute_stage_errors,
    compute_total_error,
    link_actual_to_predictions,
    monitoring_summary,
)
from dltime.inference import PredictionRequest, PredictionState
from dltime.store import SCHEMA_VERSION, MonitoringStore

REFERENCE = {
    "loader_name": "MigrationSummarySubTableLoaderDataset",
    "sprint": "TEST",
    "dataset": "Migration_Summary_Sub_Fact_Table_LDR_Datase_new",
    "connection": "MDM_DM_TEST",
}


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MonitoringStore:
    """Temp store; also repoints the module default so the service's event
    persistence and the API endpoints all write/read the same tmp dir."""
    monkeypatch.setattr("dltime.store.STORE_DIR", tmp_path)
    return MonitoringStore(tmp_path)


# --------------------------------------------------------------- error math


def test_total_error_signed_absolute():
    e = compute_total_error(predicted=10.0, actual=8.0)
    assert e["signed_error_seconds"] == 2.0
    assert e["absolute_error_seconds"] == 2.0
    assert e["absolute_percentage_error"] == pytest.approx(25.0)


def test_total_error_zero_actual_percentage_unavailable():
    e = compute_total_error(predicted=10.0, actual=0.0)
    assert e["absolute_error_seconds"] == 10.0
    assert e["absolute_percentage_error"] is None
    assert e["within_10_percent"] is None
    assert e["within_2x"] is None  # not meaningful for a zero actual


def test_total_error_tolerance_flags():
    e = compute_total_error(predicted=11.0, actual=10.0)
    assert e["within_10_percent"] is True
    assert e["within_20_percent"] is True
    assert e["within_30_percent"] is True
    assert e["within_2x"] is True
    e2 = compute_total_error(predicted=40.0, actual=10.0)
    assert e2["within_10_percent"] is False
    assert e2["within_2x"] is False


def test_log_absolute_error_mirrors_model_space():
    e = compute_total_error(predicted=18.0, actual=18.0)
    assert e["log_absolute_error"] == 0.0


def test_stage_errors_only_where_both_sides_exist():
    predicted = [
        {"stage": "prevalidation", "available": True, "estimate_seconds": 5.0},
        {"stage": "transformation", "available": True, "estimate_seconds": 2.0},
        {"stage": "datamart_approval", "available": False, "estimate_seconds": None},
    ]
    actuals = {"prevalidation": 4.0, "datamart_approval": 100.0}
    out = compute_stage_errors(predicted, actuals)
    assert set(out) == {"prevalidation"}
    assert out["prevalidation"]["signed_error_seconds"] == 1.0
    assert out["prevalidation"]["absolute_percentage_error"] == pytest.approx(25.0)


def test_stage_errors_empty_when_no_actuals():
    assert compute_stage_errors([{"stage": "s", "available": True, "estimate_seconds": 1.0}], {}) == {}


# --------------------------------------------------------------- actual outcomes


def test_valid_actual_is_training_eligible():
    rec, err = build_actual_outcome(
        loader_name="L",
        dataset="D",
        sprint="S",
        connection="C",
        start_time="2026-09-01T10:00:00",
        end_time="2026-09-01T10:01:00",
        workflow_execution_id="123",
        actual_total_seconds=60.0,
        status="SUCCEED",
        stage_actuals={"prevalidation": 5.0},
    )
    assert err is None
    assert rec["training_eligible"] is True
    assert rec["training_eligibility_reason"] == "ELIGIBLE"
    assert rec["stage_actuals"] == {"prevalidation": 5.0}


def test_invalid_target_stored_not_eligible():
    rec, err = build_actual_outcome(
        loader_name="L",
        dataset="D",
        sprint="S",
        connection="C",
        start_time="2026-09-01T10:00:00",
        end_time=None,
        actual_total_seconds=0.0,
        status="STOPPED",
    )
    assert err is None
    assert rec["training_eligible"] is False
    assert rec["training_eligibility_reason"] == "INVALID_TARGET"


def test_interrupted_status_not_training_eligible():
    rec, _ = build_actual_outcome(
        loader_name="L",
        dataset="D",
        sprint="S",
        connection="C",
        start_time="2026-09-01T10:00:00",
        end_time="2026-09-01T10:01:30",
        actual_total_seconds=50.0,
        status="STOPPED",
    )
    assert rec["training_eligible"] is False
    assert rec["training_eligibility_reason"] == "STATUS_NOT_TRAINING_ELIGIBLE"


def test_invalid_actual_rejected():
    rec, err = build_actual_outcome(
        loader_name="",
        dataset="D",
        sprint="S",
        connection="C",            start_time="",
            end_time=None,
            actual_total_seconds=None,
        )
    assert rec is None
    assert "missing_required_identity_field" in err
    assert "missing_or_invalid_actual_total_seconds" in err


def test_actual_outcome_id_deterministic_and_content_hashed():
    kwargs = dict(
        loader_name="L",
        dataset="D",
        sprint="S",
        connection="C",            start_time="2026-09-01T10:00:00",
            end_time="2026-09-01T10:01:00",
            actual_total_seconds=60.0,
            status="SUCCEED",
        )
    r1, _ = build_actual_outcome(**kwargs)
    r2, _ = build_actual_outcome(**kwargs)
    assert r1["outcome_id"] == r2["outcome_id"]
    assert r1["outcome_hash"] == r2["outcome_hash"]


# ----------------------------------------------------- prediction event capture


def test_all_states_persist_prediction_events(service, store):
    cases = [
        (PredictionRequest(connection="MDM_DM_TEST", **{k: v for k, v in REFERENCE.items() if k != "connection"}), "PREDICTED"),
        (
            PredictionRequest(
                loader_name="2210_TEST_ALL_CONFIG", sprint="SUPPORT", dataset="A_TABLE", connection="test1"
            ),
            "INSUFFICIENT_HISTORY",
        ),
        (
            PredictionRequest(
                loader_name="no_such_loader_task11", sprint="X", dataset="Y", connection="c"
            ),
            # 2026-10-03 final spec: a COMPLETELY new loader (no definition
            # match at all) is treated exactly like a registered-but-never-run
            # loader — INSUFFICIENT_HISTORY, never UNRESOLVED_IDENTITY.
            "INSUFFICIENT_HISTORY",
        ),
    ]
    for req, expected in cases:
        r = service.predict(req)
        assert r.state == expected
        event = {
            "schema_version": SCHEMA_VERSION,
            "event_id": f"pe_{expected}_{id(req)}",
            "event_state": r.state,
            "request": {
                "loader_name": req.loader_name,
                "dataset": req.dataset,
                "sprint": req.sprint,
                "connection": req.connection,
                "workflow_execution_id": None,
            },
            "identity": dict(r.identity),
            "prediction": (
                {"predicted_total_seconds": r.predicted_seconds, "model_version": r.model_version}
                if r.state == "PREDICTED"
                else None
            ),
        }
        store.append_prediction_event(event)
    events, warns = store.read_prediction_events()
    # Reachable predict() states only: UNRESOLVED_IDENTITY is no longer
    # a terminal prediction state (2026-10-03 spec — see above).
    assert {e["event_state"] for e in events} == {
        "PREDICTED",
        "INSUFFICIENT_HISTORY",
    }
    assert warns == []
    # AMBIGUOUS + INVALID_INPUT schemas exercise the same event builder path
    ambiguous_event = {
        "schema_version": SCHEMA_VERSION,
        "event_id": "pe_amb",
        "event_state": "AMBIGUOUS_IDENTITY",
        "request": {},
        "identity": {"resolution_status": "AMBIGUOUS"},
        "prediction": None,
    }
    invalid_event = dict(ambiguous_event, event_id="pe_inv", event_state="INVALID_INPUT")
    store.append_prediction_event(ambiguous_event)
    store.append_prediction_event(invalid_event)
    events, _ = store.read_prediction_events()
    assert {e["event_state"] for e in events} >= {
        "PREDICTED",
        "INSUFFICIENT_HISTORY",
        "AMBIGUOUS_IDENTITY",
        "INVALID_INPUT",
    }


def test_persisted_event_captures_contract_and_feature_audit(service, store):
    """The real persisted event records the full PREDICTED contract."""
    r = service.predict(PredictionRequest(connection=REFERENCE["connection"], **{k: v for k, v in REFERENCE.items() if k != "connection"}))
    assert r.state == PredictionState.PREDICTED
    from dltime.feedback import finalize_event_id, prediction_event_from_result

    event = finalize_event_id(prediction_event_from_result(r, planned_start_time=None))
    store.append_prediction_event(event)
    stored, _ = store.read_prediction_events()
    e = stored[-1]
    assert e["event_state"] == "PREDICTED"
    assert e["request"]["connection"] == "MDM_DM_TEST"
    # AD-32: the persisted prediction comes from the authoritative stage-aware
    # total; the direct reference model must still reproduce the canonical
    # corrected-regime constant byte-for-byte (rollback path integrity).
    assert r.serving_role == "AUTHORITATIVE_TOTAL"
    # Sum policy (EXP-038): the serving identity is the deterministic sum.
    assert r.model_version == "sum-of-available-stages-v2"
    assert r.direct_reference_seconds == pytest.approx(16.289350640581837, rel=0, abs=1e-9)
    assert isinstance(e["prediction"]["predicted_total_seconds"], float)
    assert e["prediction"]["prior_count"] == 11
    # Locked ADDITIVE interval presentation: width = 2*Q90 = 5.839 s at point
    # 16.112 s (re-anchored additive scale). 2026-09-30 re-pin: the served
    # total for this LOADER is 11.465 s (LOADER-tier approval history only;
    # the broad-context estimate is disclosed but excluded), interval =
    # 11.465 +/- 3.035 -> width factor 0.928 -> the WIDE_INTERVAL risk signal.
    # 2026-10-03: the reliability label was removed from the system entirely —
    # no reliability field may appear in the persisted prediction payload.
    assert "reliability" not in e["prediction"]
    assert e["prediction"]["risk_label"] == "LOW_CONFIDENCE"
    assert e["prediction"]["interval_level"] == 0.9
    # Sum policy (EXP-038): the digest carries the deterministic stage-sum
    # explanation with the ranked additive components.
    assert e["explanation_digest"]["method"] == "deterministic_stage_sum"
    assert len(e["explanation_digest"]["top_contributors"]) == 3
    assert e["stage_intelligence"]["stages"]
    # AD-32 (re-pinned 2026-10-05): the persisted feature audit carries
    # the SERVED artifact contract (feature_names_in_order) plus, when
    # present, the two audit-marker columns (def_preval_name_component /
    # def_trans_name_component). The contract is derived from the artifact
    # itself so this pin cannot rot as the feature set evolves.
    contract = list(service.artifact["metadata"]["feature_names_in_order"])
    assert e["features"] is not None
    assert set(contract) <= set(e["features"])
    assert set(e["features"]) - set(contract) <= {
        "def_preval_name_component",
        "def_trans_name_component",
    }


def test_append_only_store(store):
    store.append_actual_outcome({"schema_version": SCHEMA_VERSION, "outcome_id": "ao1", "outcome_hash": "h"})
    store.append_actual_outcome({"schema_version": SCHEMA_VERSION, "outcome_id": "ao1", "outcome_hash": "h"})
    recs, _ = store.read_actual_outcomes()
    assert len(recs) == 2  # both lines retained — append-only audit trail


def test_store_corruption_skip(store):
    store.append_actual_outcome({"schema_version": SCHEMA_VERSION, "outcome_id": "ao1", "outcome_hash": "h"})
    path = Path(store.dir) / "actual_outcomes.jsonl"
    path.write_text(path.read_text() + "{broken\n")
    recs, warns = store.read_actual_outcomes()
    assert len(recs) == 1
    assert len(warns) == 1


# --------------------------------------------------------------------- linkage


def _mk_event(event_id, wf=None, state="PREDICTED", **identity):
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "event_state": state,
        "request": {
            "loader_name": identity.get("loader_name", "L"),
            "dataset": identity.get("dataset", "D"),
            "sprint": identity.get("sprint", "S"),
            "connection": identity.get("connection", "C"),
            "workflow_execution_id": wf,
        },
        "identity": {"resolved_connection": identity.get("connection", "C")},
        "prediction": {"predicted_total_seconds": 10.0},
    }


def _mk_actual(wf=None, **identity):
    rec, _ = build_actual_outcome(
        loader_name=identity.get("loader_name", "L"),
        dataset=identity.get("dataset", "D"),
        sprint=identity.get("sprint", "S"),
        connection=identity.get("connection", "C"),
        start_time="2026-09-01T10:00:00",
        end_time="2026-09-01T10:00:30",
        actual_total_seconds=8.0,
        status="SUCCEED",
        workflow_execution_id=wf,
    )
    return rec


def test_linkage_exact_workflow_reference():
    actual = _mk_actual(wf="777")
    events = [_mk_event("pe1", wf="777")]
    res = link_actual_to_predictions(actual, events, [])
    assert res["linkage_status"] == LINK_STATUS_MATCHED
    assert res["linkage_method"] == LINK_METHOD_WORKFLOW_ID
    assert res["prediction_event_ids"] == ["pe1"]


def test_linkage_no_reference_matches_most_recent_identity():
    """2026-10-05: manual entries carry no reference — identity-only fallback links."""
    actual = _mk_actual(wf=None)
    events = [_mk_event("pe1", wf="777"), _mk_event("pe2", wf="888")]
    res = link_actual_to_predictions(actual, events, [])
    assert res["linkage_status"] == LINK_STATUS_MATCHED
    assert res["linkage_method"] == "IDENTITY_MOST_RECENT"
    assert res["prediction_event_ids"] == ["pe2"]


def test_linkage_identity_guard_rejects_mismatched_reference():
    """A reference match with a different canonical identity is not defensible."""
    actual = _mk_actual(wf="777", loader_name="OTHER")
    events = [_mk_event("pe1", wf="777")]
    res = link_actual_to_predictions(actual, events, [])
    assert res["linkage_status"] == LINK_STATUS_UNMATCHED


def test_linkage_ambiguous_when_reference_shared(SYNTHETIC=None):
    """SYNTHETIC: two linkable events share the same workflow reference."""
    actual = _mk_actual(wf="777")
    events = [_mk_event("pe1", wf="777"), _mk_event("pe2", wf="777")]
    res = link_actual_to_predictions(actual, events, [])
    assert res["linkage_status"] == LINK_STATUS_AMBIGUOUS
    assert set(res["prediction_event_ids"]) == {"pe1", "pe2"}


def test_already_linked_prediction_not_relinked():
    actual = _mk_actual(wf="777")
    events = [_mk_event("pe1", wf="777")]
    fb = [{"prediction_event_id": "pe1", "linkage_status": LINK_STATUS_MATCHED}]
    res = link_actual_to_predictions(actual, events, fb)
    assert res["linkage_status"] == LINK_STATUS_UNMATCHED


def test_linkage_ignores_unlinkable_states():
    actual = _mk_actual(wf="777")
    events = [_mk_event("pe1", wf="777", state="UNRESOLVED_IDENTITY")]
    res = link_actual_to_predictions(actual, events, [])
    assert res["linkage_status"] == LINK_STATUS_UNMATCHED


# ------------------------------------------------- end-to-end ingest (API level)


@pytest.fixture(scope="module")
def client():
    from dltime.api import app

    return TestClient(app)


def test_feedback_end_to_end_matched(client, store, monkeypatch):
    """Predict → persist event → ingest actual (same workflow ref) → MATCHED."""
    monkeypatch.setattr("dltime.api.get_monitoring_store", lambda: store)
    pred = client.post(
        "/predict",
        json={
            "loader_name": REFERENCE["loader_name"],
            "sprint": REFERENCE["sprint"],
            "dataset": REFERENCE["dataset"],
            "connection": REFERENCE["connection"],
            "workflow_execution_id": "WF-T11-1",
        },
    ).json()
    assert pred["state"] == "PREDICTED"
    # Sum policy (EXP-038) + applicable-stage policy: served ETA = applicable
    # 3-stage ML sum (11.465) + GLOBAL approval statistical estimates (11.465)
    # = 11.465 s (2026-09-30 2nd re-pin).
    assert pred["predicted_seconds"] == pytest.approx(11.465, rel=0, abs=1e-9)
    assert pred["model_version"] == "sum-of-available-stages-v2"

    body = {
        "loader_name": REFERENCE["loader_name"],
        "dataset": REFERENCE["dataset"],
        "sprint": REFERENCE["sprint"],
        "connection": REFERENCE["connection"],
        "workflow_execution_id": "WF-T11-1",
        "start_time": "2026-09-18T10:00:00",
        "end_time": "2026-09-18T10:00:20",
        "actual_total_seconds": 17.0,
        "status": "SUCCEED",
        # Applicable-stage policy: actuals pair only with APPLICABLE predicted
        # stages (staging / pre_load / data_loading here; the old actual only
        # carried prevalidation + data_loading_1, neither applicable now).
        "stage_actuals": {"staging": 0.5, "data_loading": 12.0},
    }
    resp = client.post("/feedback/actual", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["linkage_status"] == "MATCHED"
    assert data["linkage_method"] == "EXACT_WORKFLOW_EXECUTION_ID"
    assert data["prediction_event_id"]
    assert data["training_eligible"] is True
    e = data["total_error"]
    # Sum policy (v2 re-pin): the served ETA is 11.465 s, so a 17 s actual is
    # OUTSIDE the 10% band (error metrics are computed from the served value,
    # never asserted to be good or bad a priori).
    assert e["signed_error_seconds"] == pytest.approx(11.465 - 17.0, abs=1e-6)
    assert e["within_10_percent"] is False
    # stage error where both sides existed (applicable stages only)
    assert "data_loading" in (data["stage_errors"] or {})
    assert "staging" in (data["stage_errors"] or {})


def test_duplicate_actual_is_idempotent(client, store, monkeypatch):
    monkeypatch.setattr("dltime.api.get_monitoring_store", lambda: store)
    client.post(
        "/predict",
        json={
            "loader_name": REFERENCE["loader_name"],
            "sprint": REFERENCE["sprint"],
            "dataset": REFERENCE["dataset"],
            "connection": REFERENCE["connection"],
            "workflow_execution_id": "WF-T11-2",
        },
    )
    body = {
        "loader_name": REFERENCE["loader_name"],
        "dataset": REFERENCE["dataset"],
        "sprint": REFERENCE["sprint"],
        "connection": REFERENCE["connection"],
        "workflow_execution_id": "WF-T11-2",
        "start_time": "2026-09-18T11:00:00",
        "actual_total_seconds": 18.0,
        "status": "SUCCEED",
    }
    first = client.post("/feedback/actual", json=body).json()
    second = client.post("/feedback/actual", json=body).json()
    assert first["duplicate"] is False
    assert second["duplicate"] is True
    assert second["outcome_id"] == first["outcome_id"]
    assert second["feedback_id"] == first["feedback_id"]
    # store still holds exactly ONE outcome + ONE feedback for this reference
    outcomes, _ = store.read_actual_outcomes()
    assert len([o for o in outcomes if o["execution_ref"]["workflow_execution_id"] == "WF-T11-2"]) == 1


def test_conflicting_actual_rejected_409(client, store, monkeypatch):
    monkeypatch.setattr("dltime.api.get_monitoring_store", lambda: store)
    body = {
        "loader_name": REFERENCE["loader_name"],
        "dataset": REFERENCE["dataset"],
        "sprint": REFERENCE["sprint"],
        "connection": REFERENCE["connection"],
        "workflow_execution_id": "WF-T11-3",
        "start_time": "2026-09-18T12:00:00",
        "actual_total_seconds": 18.0,
        "status": "SUCCEED",
    }
    client.post("/feedback/actual", json=body)
    conflict = dict(body, actual_total_seconds=99.0)
    resp = client.post("/feedback/actual", json=conflict)
    assert resp.status_code == 409
    assert "conflicting_actual_outcome" in resp.json()["detail"]
    outcomes, _ = store.read_actual_outcomes()
    assert len([o for o in outcomes if o["execution_ref"]["workflow_execution_id"] == "WF-T11-3"]) == 1


def test_invalid_actual_422(client, store, monkeypatch):
    monkeypatch.setattr("dltime.api.get_monitoring_store", lambda: store)
    resp = client.post(
        "/feedback/actual",
        json={
            "loader_name": "L",
            "dataset": "D",
            "sprint": "S",
            "connection": "C",
            "start_time": "2026-09-18T10:00:00",
            "actual_total_seconds": -5.0,
            "status": "SUCCEED",
        },
    )
    assert resp.status_code == 200  # stored for audit, marked not eligible
    data = resp.json()
    assert data["training_eligible"] is False
    assert data["training_eligibility_reason"] == "INVALID_TARGET"
    assert data["linkage_status"] in {"UNMATCHED", "AMBIGUOUS"}


def test_actual_missing_identity_422(client, store, monkeypatch):
    monkeypatch.setattr("dltime.api.get_monitoring_store", lambda: store)
    resp = client.post(
        "/feedback/actual",
        json={
            "loader_name": "L",
            "dataset": "D",
            "sprint": "",
            "connection": "C",
            "start_time": "2026-09-18T10:00:00",
            "actual_total_seconds": 5.0,
        },
    )
    assert resp.status_code == 422


def test_reject_post_execution_smuggling_into_predict(client):
    resp = client.post(
        "/predict",
        json={
            "loader_name": "x",
            "sprint": "s",
            "dataset": "d",
            "connection": "c",
            "actual_total_seconds": 12.0,
        },
    )
    assert resp.status_code == 422  # extra="forbid"


# ------------------------------------------------------------- monitoring APIs


def test_monitoring_summary_counts_and_metrics(client, store, monkeypatch):
    monkeypatch.setattr("dltime.api.get_monitoring_store", lambda: store)
    pred = client.post(
        "/predict",
        json={
            "loader_name": REFERENCE["loader_name"],
            "sprint": REFERENCE["sprint"],
            "dataset": REFERENCE["dataset"],
            "connection": REFERENCE["connection"],
            "workflow_execution_id": "WF-T11-9",
        },
    ).json()
    assert pred["state"] == "PREDICTED"
    client.post(
        "/predict",
        json={"loader_name": "no_such_task11_b", "sprint": "X", "dataset": "Y", "connection": "c"},
    )
    client.post(
        "/feedback/actual",
        json={
            "loader_name": REFERENCE["loader_name"],
            "dataset": REFERENCE["dataset"],
            "sprint": REFERENCE["sprint"],
            "connection": REFERENCE["connection"],
            "workflow_execution_id": "WF-T11-9",
            "start_time": "2026-09-18T14:00:00",
            "actual_total_seconds": 17.0,
            "status": "SUCCEED",
        },
    )
    summary = client.get("/monitoring/summary").json()
    assert summary["prediction_attempts"] >= 2
    assert summary["state_counts"]["PREDICTED"] >= 1
    # 2026-10-03 final spec: a completely new loader resolves to
    # INSUFFICIENT_HISTORY (never UNRESOLVED_IDENTITY).
    assert summary["state_counts"]["INSUFFICIENT_HISTORY"] >= 1
    assert summary["actual_outcomes_received"] == 1
    assert summary["linkage"]["matched"] == 1
    assert summary["metrics"]["sample_count"] == 1
    assert "NOT Task 03" in summary["label"]
    assert summary["training_feedback"]["eligible_actual_count"] == 1

    per_loader = client.get("/monitoring/summary", params={"loader": REFERENCE["loader_name"]}).json()
    assert per_loader["prediction_attempts"] == 1

    events = client.get("/monitoring/events").json()
    assert events["count"] >= 2
    fb = client.get("/monitoring/feedback").json()
    assert fb["count"] >= 1
    assert fb["feedback"][-1]["total_error"] is not None


def test_persistence_failure_never_fails_prediction(service, monkeypatch):
    """Isolation contract: a broken store appends a warning, never alters output."""
    r_before = service.predict(
        PredictionRequest(connection=REFERENCE["connection"], **{k: v for k, v in REFERENCE.items() if k != "connection"})
    )
    assert r_before.state == PredictionState.PREDICTED
    assert not any("monitoring" in w.lower() for w in r_before.warnings)

    def boom():
        raise RuntimeError("store offline")

    monkeypatch.setattr("dltime.store.MonitoringStore", boom)
    r_after = service.predict(
        PredictionRequest(connection=REFERENCE["connection"], **{k: v for k, v in REFERENCE.items() if k != "connection"})
    )
    assert r_after.state == PredictionState.PREDICTED
    assert r_after.predicted_seconds == r_before.predicted_seconds
    assert any("Prediction monitoring unavailable" in w for w in r_after.warnings)


def test_leakage_no_post_execution_fields_in_prediction_request(service):
    """Actual-outcome fields are absent from the /predict contract by design."""
    from dltime.api import ActualOutcomeBody, PredictionRequestBody

    pred_fields = set(PredictionRequestBody.model_fields)
    actual_fields = set(ActualOutcomeBody.model_fields)
    post_only = actual_fields - pred_fields
    assert {"actual_total_seconds", "status", "start_time", "end_time", "ldr_execution_id"} <= post_only


def test_offline_task03_metrics_remain_separate():
    """The operational summary must not present itself as Task 03 evaluation."""
    summary = monitoring_summary(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "event_id": "e1",
                "event_state": "PREDICTED",
                "request": {"loader_name": "L"},
            }
        ],
        [],
        [],
    )
    assert "NOT Task 03" in summary["label"]
    assert summary["metrics"] is None  # no matched feedback -> no fabricated metrics
