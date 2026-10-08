"""Module: final UI integration — GET /overview presentation endpoint.

Overview facts must come from REAL data (sources/reconstruction/artifact/
store). The endpoint is read-only: a GET must never write to the monitoring
store, and with an empty store it must report zero activity — never fake
recent predictions or fake accuracy.
"""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore")


def _client():
    from fastapi.testclient import TestClient

    from dltime.api import create_app

    return TestClient(create_app())


@pytest.fixture(autouse=True)
def _fresh_api_store(monkeypatch):
    """Follow the test_monitoring_task11 convention: the api module caches its
    MonitoringStore provider (lru_cache), which would pin the first test's tmp
    directory. Patch the provider to resolve the CURRENT (monkeypatched)
    STORE_DIR on every call so /overview and /predict always agree."""
    import dltime.api
    from dltime.store import MonitoringStore

    monkeypatch.setattr("dltime.api.get_monitoring_store", lambda: MonitoringStore())


def test_overview_reports_real_data_and_empty_store():
    c = _client()
    r = c.get("/overview")
    assert r.status_code == 200
    body = r.json()

    # Data facts are the real reconstruction numbers (positive, non-zero).
    data = body["data"]
    assert data["total_executions"] > 0
    assert data["total_loaders"] > 0
    assert data["training_eligible"] > 0
    assert data["median_duration_sec"] > 0
    assert data["p95_duration_sec"] >= data["median_duration_sec"]

    # Model facts describe the AUTHORITATIVE serving policy (sum of the five
    # operational stage predictions; EXP-038).
    model = body["model"]
    assert model["model_version"] == "sum-of-available-stages-v2"
    assert model["feature_count"] == 5
    assert model["serving_role"] == "AUTHORITATIVE_TOTAL"
    assert model["calibration"]["version"] == "unc-log1p-conformal-v1-sum-of-five-stages"
    # Calibration rows come from the REAL chronological calibration window —
    # never a fake round number or a fabricated "production" accuracy.
    cov = model["calibration"]["coverage_90"]
    assert cov is None or 0.0 <= cov <= 1.0

    # Empty store: zero activity, no fabricated recent events.
    mon = body["monitoring"]
    assert mon["prediction_attempts"] == 0
    assert mon["state_counts"] == {}
    assert mon["actual_outcomes"] == 0
    assert mon["matched_outcomes"] == 0
    assert body["recent_events"] == []

    # Drift/readiness use the real framework states.
    assert body["drift"]["performance_status"] in {"INSUFFICIENT_OUTCOMES", "INSUFFICIENT_SAMPLE"}
    assert body["retraining"]["readiness_state"] in {
        "NOT_READY",
        "READY_FOR_REVIEW",
        "RETRAINING_RECOMMENDED_FOR_REVIEW",
    }
    assert body["retraining"]["automatic_retraining"] == "OFF"


def test_overview_get_does_not_mutate_store():
    from dltime.store import STORE_DIR

    before = sorted(p.name for p in STORE_DIR.iterdir()) if STORE_DIR.exists() else []
    c = _client()
    assert c.get("/overview").status_code == 200
    after = sorted(p.name for p in STORE_DIR.iterdir()) if STORE_DIR.exists() else []
    assert before == after


def test_overview_recent_events_reflect_real_store_entries():
    """With real store content, recent_events mirror actual recorded events."""
    c = _client()
    body_probe = {
        "loader_name": "MigrationSummarySubTableLoaderDataset",
        "sprint": "TEST",
        "dataset": "Migration_Summary_Sub_Fact_Table_LDR_Datase_new",
        "connection": "MDM_DM_TEST",
    }
    resp = c.post("/predict", json=body_probe)
    assert resp.status_code == 200
    overview = c.get("/overview").json()
    assert overview["monitoring"]["prediction_attempts"] >= 1
    states = {e["event_state"] for e in overview["recent_events"]}
    assert "PREDICTED" in states
    pred_entry = next(e for e in overview["recent_events"] if e["event_state"] == "PREDICTED")
    assert pred_entry["predicted_seconds"] is not None
    assert pred_entry["loader_name"] == body_probe["loader_name"]
