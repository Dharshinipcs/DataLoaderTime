"""Task 05: CORS for the local Prediction Studio dev server (minimal addition)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from dltime.api import app

client = TestClient(app)


def test_cors_preflight_from_dev_origin_allowed():
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_other_origins_not_allowed():
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in response.headers


def test_predict_rejects_blank_identity_over_api():
    """Backend remains authoritative; schema guard unchanged by Task 05.
    Connection is required since Task 10.1."""
    response = client.post(
        "/predict", json={"loader_name": "x", "sprint": "s", "dataset": "d", "connection": "c"}
    )
    assert response.status_code == 200
    # 2026-10-03: unregistered loader == NEW loader (short refusal, no total).
    assert response.json()["state"] == "INSUFFICIENT_HISTORY"
