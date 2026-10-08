"""AD-33 notification-center tests — fixed-scenario policy (2026-10-05).

The Alert Center carries ONLY the THREE fixed scenarios:
  1. RETRAINING_RECOMMENDED — retraining needed (readiness reached review)
  2. NEW_MODEL_CANDIDATE    — stage-model promotion suggestion (candidate ready)
  3. MODEL_DRIFT            — drift detected (feature/population)

Per-prediction events, actual outcomes, model-stale counters, and accuracy
degradation generate NO notifications, so the bell cannot flood. Dedupe,
read/unread persistence and the §34 stage_prediction_distributions summary
are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dltime import notifications
from dltime.model_lab import stage_prediction_distributions
from dltime.store import MonitoringStore


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MonitoringStore:
    monkeypatch.setattr("dltime.store.STORE_DIR", tmp_path)
    return MonitoringStore(tmp_path)


def _rec(alert_type: str, key: str, message: str = "m") -> dict:
    return notifications._record(alert_type=alert_type, dedupe_key=key, message=message)


def _event(**overrides) -> dict:
    base = {
        "event_id": "ev_1",
        "event_state": "PREDICTED",
        "identity": {
            "loader_name": "LoaderA",
            "dataset": "DS",
            "sprint": "SPR",
            "connection": "CON",
        },
        "history": {"prior_loader_executions": 10},
        "prediction": {"risk_label": "NORMAL", "model_version": "m1"},
    }
    base.update(overrides)
    return base


# ------------------------------------------------ fixed-scenario generation


def test_prediction_events_never_generate_alerts():
    """Fixed-scenario policy: NO alert from any prediction event — new loader,
    withheld prediction, or any per-prediction risk label (bell cannot flood)."""
    for ev in (
        _event(),
        _event(history={"prior_loader_executions": 0}),
        _event(event_state="INSUFFICIENT_HISTORY"),
        _event(prediction={"risk_label": "OUT_OF_DISTRIBUTION", "model_version": "m1"}),
        _event(prediction={"risk_label": "LOW_CONFIDENCE", "model_version": "m1"}),
        _event(prediction={"risk_label": "HIGH_RISK", "model_version": "m1"}),
    ):
        assert notifications.alerts_from_prediction_event(ev) == []


def test_actual_outcomes_never_generate_alerts():
    """Data-quality / model-stale signals stay on their dashboards, not the bell."""
    outcome = {
        "outcome_id": "out_1",
        "identity": {"loader_name": "LoaderA", "dataset": "DS"},
        "data_quality_warnings": ["end_time before start_time"],
        "training_eligible": False,
    }
    assert notifications.alerts_from_actual_outcome(outcome) == []
    assert notifications.alerts_from_actual_outcome(outcome, eligible_outcome_count=30) == []


def test_system_evaluation_drift_and_retraining_only(store, monkeypatch):
    monkeypatch.setattr(notifications, "_registry_alerts", lambda registry: [])
    records = notifications.evaluate_system_notifications(
        drift_feature_status="SIGNIFICANT_DRIFT",
        drift_population_status="NO_DRIFT",
        readiness={
            "readiness_state": "RETRAINING_RECOMMENDED_FOR_REVIEW",
            "signals": {
                "eligible_new_actuals": {"count": 30, "threshold": 30, "met": True},
                "performance_degradation": {"assessed": True, "met": True,
                                            "window_medae": 100.0, "baseline_medae": 10.0},
            },
        },
    )
    types = {r["alert_type"] for r in records}
    assert types == {notifications.AlertType.MODEL_DRIFT, notifications.AlertType.RETRAINING_RECOMMENDED}


def test_system_evaluation_clean_state_no_alerts(store, monkeypatch):
    monkeypatch.setattr(notifications, "_stage_coverage_alerts", lambda: [])
    monkeypatch.setattr(notifications, "_registry_alerts", lambda registry: [])
    records = notifications.evaluate_system_notifications(
        drift_feature_status="NO_DRIFT",
        readiness={
            "readiness_state": "NOT_READY",
            "signals": {
                "eligible_new_actuals": {"count": 0, "threshold": 30, "met": False},
                "performance_degradation": {"assessed": False, "met": False},
            },
        },
    )
    assert records == []


def test_promotion_suggestion_alert_from_registry_candidate():
    """Fixed scenario 2: stage-model promotion suggestion (candidate ready)."""

    class FakeRegistry:
        def list(self):
            return [{"model_id": "cand-1", "status": "CANDIDATE_EVALUATED", "model_family": "extra_trees"}]

    records = notifications._registry_alerts(FakeRegistry())
    assert len(records) == 1
    assert records[0]["alert_type"] == notifications.AlertType.NEW_MODEL_CANDIDATE
    assert records[0]["context"]["model_id"] == "cand-1"


def test_lifecycle_events_restricted_to_fixed_scenarios(store, monkeypatch):
    monkeypatch.setattr("dltime.store.STORE_DIR", store.dir)
    from dltime import retraining

    # allowed: drift / retraining-needed / promotion suggestion
    assert retraining.lifecycle_notifications(
        event_kind="DRIFT_DETECTED", store=store, context={"model_version": "m1"}
    ) != []
    assert retraining.lifecycle_notifications(
        event_kind="RETRAINING_REVIEW_RECOMMENDED", store=store, context={"model_version": "m1"}
    ) != []
    assert retraining.lifecycle_notifications(
        event_kind="CANDIDATE_READY", store=store, context={"model_version": "m1"}
    ) != []
    # everything else stays out of the bell
    for kind in ("DEGRADATION_DETECTED", "RETRAINING_STARTED", "CANDIDATE_REJECTED",
                 "MODEL_PROMOTED", "ROLLBACK_COMPLETED", "EVIDENCE_INSUFFICIENT"):
        assert retraining.lifecycle_notifications(
            event_kind=kind, store=store, context={"model_version": "m1"}
        ) == []


def test_severity_and_title_present(store):
    rec = _rec(notifications.AlertType.MODEL_DRIFT, "drift:feature")
    assert rec["severity"] == notifications.Severity.HIGH
    assert rec["title"] == notifications.TITLE_FOR_TYPE[rec["alert_type"]]
    assert rec["read"] is False and rec["read_at"] is None


# ---------------------------------------------------------------- dedupe


def test_dedupe_no_duplicate_open_notification(store):
    first = notifications.append_notifications(store, [_rec("MODEL_DRIFT", "drift:feature")])
    second = notifications.append_notifications(store, [_rec("MODEL_DRIFT", "drift:feature")])
    assert len(first) == 1
    assert second == []  # open duplicate suppressed
    assert notifications.list_notifications(store)["unread_count"] == 1


def test_reread_notification_allows_reoccurrence(store):
    notifications.append_notifications(store, [_rec("MODEL_DRIFT", "drift:feature")])
    feed = notifications.list_notifications(store)
    notifications.mark_notification_read(store, feed["notifications"][0]["notification_id"])
    second = notifications.append_notifications(store, [_rec("MODEL_DRIFT", "drift:feature")])
    assert len(second) == 1  # re-raised after being read
    assert notifications.list_notifications(store)["unread_count"] == 1


def test_mark_all_read_persists(store):
    notifications.append_notifications(store, [_rec("MODEL_DRIFT", "drift:feature")])
    result = notifications.mark_all_notifications_read(store)
    assert result["marked"] is True
    assert notifications.list_notifications(store)["unread_count"] == 0
    fresh = MonitoringStore(store.dir)
    assert notifications.list_notifications(fresh)["unread_count"] == 0
    assert all(r["read_at"] for r in notifications.list_notifications(fresh)["notifications"])


# --------------------------------------------------- §34 stage distributions


def test_stage_distribution_summary_from_events():
    events = [
        {
            "stage_intelligence": {
                "stages": [
                    {"stage": "staging", "estimate_seconds": 10.0, "method": "ml_stage_model"},
                    {"stage": "prevalidation", "estimate_seconds": 100.0, "method": "ml_stage_model"},
                ]
            }
        },
        {
            "prediction": {  # nested payload variant must also be read
                "stage_intelligence": {
                    "stages": [
                        {"stage": "staging", "estimate_seconds": 30.0, "method": "ml_stage_model"}
                    ]
                }
            }
        },
    ]
    out = stage_prediction_distributions(events)
    assert out["event_count"] == 2
    assert out["stages"]["staging"] == {
        "n": 2, "mean_seconds": 20.0, "median_seconds": 20.0,
        "min_seconds": 10.0, "max_seconds": 30.0, "method": "ml_stage_model",
    }
    assert out["stages"]["prevalidation"]["n"] == 1


def test_stage_distribution_empty_is_honest():
    out = stage_prediction_distributions([])
    assert out["stages"] == {} and out["event_count"] == 0


# --------------------------------------------------------------- API surface


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from dltime.api import create_app

    return TestClient(create_app())


def test_notifications_endpoints_roundtrip(client, store, monkeypatch):
    monkeypatch.setattr("dltime.api.get_monitoring_store", lambda: store)
    notifications.append_notifications(store, [_rec("MODEL_DRIFT", "drift:feature")])

    listing = client.get("/notifications").json()
    assert listing["unread_count"] == 1

    nid = listing["notifications"][0]["notification_id"]
    marked = client.post(f"/notifications/{nid}/read").json()
    assert marked["marked"] is True
    assert client.get("/notifications").json()["unread_count"] == 0

    cleared = client.post("/notifications/read-all").json()
    assert cleared["marked"] is False  # already read
    assert client.get("/notifications").json()["unread_count"] == 0


def test_notifications_store_is_jsonl(store):
    notifications.append_notifications(store, [_rec("MODEL_DRIFT", "drift:feature")])
    path = Path(store.dir) / "notifications.jsonl"
    assert path.exists()
    rec = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert rec["alert_type"] == notifications.AlertType.MODEL_DRIFT
