"""Task 04: FastAPI /predict endpoint behaviour (clean pre-execution contract)."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dltime.api import app
from dltime.inference import PredictionRequest


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _first_predicted_body(service, client: TestClient) -> dict:
    """Find a real RESOLVED eligible identity, then hit the API."""
    X = service.tf.X
    frame = service.tf.frame.loc[X.index]
    keep = [
        i
        for i in frame.index
        if service.resolver.resolve(
            frame.at[i, "Dataset_display_name"], frame.at[i, "Sprint"], frame.at[i, "Loader_Name"]
        )[0].value
        == "RESOLVED"
        and pd.notna(frame.at[i, "execution_connection"])
        and float(X.at[i, "hist_loader_prior_count"]) >= 3
    ]
    assert keep, "no resolved >=3-prior training case"
    row = frame.loc[keep[-1]]
    body = {
        "loader_name": str(row["Loader_Name"]),
        "sprint": str(row["Sprint"]),
        "dataset": str(row["Dataset_display_name"]),
        "connection": str(row["execution_connection"]),
    }
    resp = client.post("/predict", json=body)
    assert resp.status_code == 200
    return body, resp.json()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_predicted_state(service, client):
    body, data = _first_predicted_body(service, client)
    assert data["state"] == "PREDICTED"
    assert data["predicted_seconds"] > 0
    # Sum policy (EXP-038) + applicable-stage policy: deterministic sum of the
    # APPLICABLE stages (ML + statistical approval components).
    assert data["model_version"] == "sum-of-available-stages-v2"
    assert data["serving_role"] == "AUTHORITATIVE_TOTAL"
    assert data["direct_reference_seconds"] > 0
    # the served total is exactly the displayed applicable ML stages plus the
    # available approval statistical estimates
    ml_vals = [
        s["estimate_seconds"]
        for s in data["stage_intelligence"]["stages"]
        if s.get("method") == "ml_stage_model" and s.get("estimate_seconds") is not None
    ]
    appr_vals = [
        s["estimate_seconds"]
        for s in data["stage_intelligence"]["stages"]
        if s.get("method") == "contextual_hierarchical_statistical"
        and s.get("estimate_seconds") is not None
        and s.get("included_in_total") is not False
    ]
    assert len(ml_vals) == 3  # applicable: no prevalidation/transformation entries
    # 2026-10-02 final spec: only LOADER-tier approval evidence joins the total.
    assert data["predicted_seconds"] == pytest.approx(sum(ml_vals) + sum(appr_vals), abs=1e-9)
    assert data["identity"]["resolution_status"] == "RESOLVED"
    assert data["history"]["prior_loader_executions"] >= 3
    assert isinstance(data["warnings"], list)


def test_predict_insufficient_history_state(service, client):
    """A genuine definition triple whose loader never executed (0 priors)."""
    executed = set(service.tf.frame["Loader_Name"].dropna().astype(str))
    d = service.definitions_detail
    cand = d[~d["loader_display_name"].astype(str).isin(executed)].dropna(
        subset=["datasetname", "sprint_name", "loader_display_name", "ldr_connection_name"]
    )
    assert not cand.empty
    r = cand.iloc[0]
    body = {
        "loader_name": str(r["loader_display_name"]),
        "sprint": str(r["sprint_name"]),
        "dataset": str(r["datasetname"]),
        "connection": str(r["ldr_connection_name"]),
    }
    resp = client.post("/predict", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "INSUFFICIENT_HISTORY"
    assert data["identity"]["resolution_status"] == "RESOLVED"
    assert data["predicted_seconds"] is None


def test_predict_unresolved_identity_state(service, client):
    body = {
        "loader_name": "no_such_loader_anywhere_task04",
        "sprint": "task04_sprint",
        "dataset": "task04_dataset",
        "connection": "task04_connection",
    }
    resp = client.post("/predict", json=body)
    assert resp.status_code == 200
    # 2026-10-03: a never-registered loader is treated as a NEW loader — the
    # same short refusal as a registered-but-never-run loader (no total).
    assert resp.json()["state"] == "INSUFFICIENT_HISTORY"


def test_predict_rejects_extra_fields(service, client):
    body = {
        "loader_name": "x",
        "sprint": "s",
        "dataset": "d",
        "connection": "c",
        "timetaken_in_sec": 123.0,
    }
    resp = client.post("/predict", json=body)
    assert resp.status_code == 422  # extra="forbid" schema guard


def test_predict_rejects_blank_required_fields(client):
    resp = client.post(
        "/predict", json={"loader_name": "  ", "sprint": "s", "dataset": "d", "connection": "c"}
    )
    assert resp.status_code == 422


def test_predict_rejects_missing_connection(client):
    """Task 10.1: Connection is a compulsory pre-execution input (422, schema-level)."""
    resp = client.post("/predict", json={"loader_name": "x", "sprint": "s", "dataset": "d"})
    assert resp.status_code == 422
    resp_blank = client.post(
        "/predict", json={"loader_name": "x", "sprint": "s", "dataset": "d", "connection": "  "}
    )
    assert resp_blank.status_code == 422


def test_predict_missing_connection_reaches_service_as_invalid_input(service, client):
    """Service-level belt: even bypassing the HTTP schema, a missing connection
    is INVALID_INPUT before identity resolution (no silent default connection)."""
    result = service.predict(
        PredictionRequest(loader_name="x", sprint="s", dataset="d", connection=None)
    )
    assert result.state == "INVALID_INPUT"
    assert result.reason == "missing_required_field: connection"


def test_predict_invalid_input_state(client):
    resp = client.post(
        "/predict",
        json={
            "loader_name": "x",
            "sprint": "s",
            "dataset": "d",
            "connection": "c",
            "planned_start_time": "junk",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["state"] == "INVALID_INPUT"


def test_predict_is_deterministic_over_http(service, client):
    body, first = _first_predicted_body(service, client)
    second = client.post("/predict", json=body).json()
    assert first["state"] == second["state"] == "PREDICTED"
    assert first["predicted_seconds"] == second["predicted_seconds"]
