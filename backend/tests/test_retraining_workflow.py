"""Retraining workflow tests (locked plan §9-16).

Hermetic: fakes the registry, monitoring store, decisions file, and the
candidate-bundle training path (the REAL training protocol is exercised live
in the Phase 3 E2E; here we verify governance, persistence, evaluation logic,
serving-switch behavior, and notifications).
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dltime import retraining, retraining_serving
from dltime.model_lab import LabError, ModelState, ModelRegistry, _JsonlStore


# ------------------------------------------------------------------ fixtures


class _FakeStore:
    """Minimal MonitoringStore stand-in for notifications persistence."""

    def __init__(self):
        self.records: list[dict] = []

    def read_notifications(self):
        return list(self.records), []

    def append_notification(self, rec):
        self.records.append(rec)

    def rewrite_notifications(self, recs):
        self.records = list(recs)


@pytest.fixture()
def tmp_decisions(tmp_path, monkeypatch):
    monkeypatch.setattr(retraining, "DECISIONS_FILE", tmp_path / "retraining_decisions.jsonl")
    return tmp_path / "retraining_decisions.jsonl"


@pytest.fixture()
def tmp_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(retraining_serving, "SWITCH_FILE", tmp_path / "serving_switch.json")
    return tmp_path / "serving_switch.json"


# ------------------------------------------------------------- decision records


def test_record_decision_persists_and_lists(tmp_decisions):
    rec = retraining.record_decision(
        decision_type="CONTINUE_MONITORING", reason="Drift without degradation", decided_by="tester"
    )
    assert rec["decision_type"] == "CONTINUE_MONITORING"
    assert rec["reason"] == "Drift without degradation"
    assert rec["model_version_at_decision"]
    assert rec["assessment_snapshot"]["verdict"] is None
    listed = retraining.list_decisions()
    assert listed[0]["decision_id"] == rec["decision_id"]


def test_record_decision_rejects_unknown_type(tmp_decisions):
    with pytest.raises(LabError):
        retraining.record_decision(decision_type="AUTO_RETRAIN", reason="x", decided_by="t")


def test_start_retraining_decision_records_type(tmp_decisions):
    rec = retraining.record_decision(
        decision_type="START_RETRAINING", reason="both signals met", decided_by="tester"
    )
    assert rec["decision_type"] == "START_RETRAINING"
    assert rec["assessment_snapshot"]["verdict"] is None


# ------------------------------------------------------- candidate evaluation


def test_candidate_evaluation_acceptable_when_better():
    ev = retraining.candidate_evaluation(
        candidate_version="cand-1",
        per_stage={
            "staging": {"median_ae": 0.5, "n": 30},
            "__total__": {},
        },
        baseline_per_stage={"staging": {"median_ae": 1.0}},
        total={"median_ae": 1.5, "n": 60},
        baseline_total={"median_ae": 2.0},
    )
    assert ev["verdict"] == "ACCEPTABLE_FOR_REVIEW"
    assert ev["human_gate_required"] is True
    assert ev["automatic_promotion"] == "OFF"
    assert ev["total"]["ratio"] == 0.75


def test_candidate_evaluation_rejects_worse_candidate():
    ev = retraining.candidate_evaluation(
        candidate_version="cand-2",
        per_stage={},
        baseline_per_stage={},
        total={"median_ae": 4.0, "n": 60},
        baseline_total={"median_ae": 2.0},
    )
    assert ev["verdict"] == "NOT_SUFFICIENTLY_BETTER"


def test_candidate_evaluation_insufficient_without_baseline():
    ev = retraining.candidate_evaluation(
        candidate_version="cand-3", per_stage={}, baseline_per_stage=None,
        total={"median_ae": 1.0, "n": 60}, baseline_total=None,
    )
    assert ev["verdict"] == "INSUFFICIENT_EVIDENCE"


# ------------------------------------------------ registry candidate handling


def test_candidate_registers_as_candidate_evaluated(tmp_path):
    store = _JsonlStore(directory=tmp_path)
    reg = ModelRegistry(store=store)
    registered = {
        "model_id": "stage-serving-candidate-20260929T000000",
        "status": ModelState.CANDIDATE_EVALUATED,
        "artifact_path": str(tmp_path / "missing.pkl"),
        "model_family": "sum-of-five-stages-candidate",
        "schema_version": "t13-v1",
        "feature_set_version": "stage-frames-v2",
        "feature_count": 0,
        "target_mode": "log1p",
        "training_population": {},
        "validation_population": {},
    }
    rec = reg.models.register(registered) if hasattr(reg, "models") else reg.register(registered)
    assert rec["status"] == ModelState.CANDIDATE_EVALUATED
    # promote refuses while the artifact is missing (safety gate)
    with pytest.raises(LabError):
        reg.promote(rec["model_id"], approved_by="t", reason="should fail: no artifact")


def test_promotion_requires_candidate_state(tmp_path):
    store = _JsonlStore(directory=tmp_path)
    reg = ModelRegistry(store=store)
    rec = {
        "model_id": "m-exp",
        "status": ModelState.EXPERIMENTAL,
        "artifact_path": "nowhere.pkl",
        "model_family": "x",
        "schema_version": "t13-v1",
        "feature_set_version": "v",
        "feature_count": 0,
        "target_mode": "log1p",
        "training_population": {},
        "validation_population": {},
    }
    reg.register(rec)
    with pytest.raises(LabError):
        reg.promote("m-exp", approved_by="t", reason="experimental cannot promote")


# ------------------------------------------------------------- serving switch


def test_serving_switch_activate_and_reset(tmp_switch, tmp_path):
    art = tmp_path / "stage-models-cand-x.pkl"
    art.write_bytes(b"blob")
    retraining_serving.activate("stage-serving-candidate-x", artifact_path=str(art))
    assert retraining_serving.active_version() == "stage-serving-candidate-x"
    assert retraining_serving.active_artifact_path() == art
    retraining_serving.reset()
    assert retraining_serving.active_version() is None


def test_serving_switch_refuses_missing_artifact(tmp_switch, tmp_path):
    with pytest.raises(LabError):
        retraining_serving.activate("stage-serving-candidate-nope", artifact_path=str(tmp_path / "gone.pkl"))


def test_serving_switch_corrupt_file_fails_closed(tmp_switch):
    tmp_switch.parent.mkdir(parents=True, exist_ok=True)
    tmp_switch.write_text("{corrupt", encoding="utf-8")
    assert retraining_serving.active_version() is None


# ---------------------------------------------------------------- notifications


def test_lifecycle_notifications_persist_and_dedupe():
    store = _FakeStore()
    # Fixed-scenario policy: only the three scenario kinds reach the
    # Alert Center; other lifecycle echoes stay in the retraining store.
    first = retraining.lifecycle_notifications(event_kind="CANDIDATE_READY", store=store, context={"model_id": "m1"})
    assert len(first) == 1
    second = retraining.lifecycle_notifications(event_kind="CANDIDATE_READY", store=store, context={"model_id": "m1"})
    assert second == []  # deduped while open
    drift = retraining.lifecycle_notifications(event_kind="DRIFT_DETECTED", store=store, context={"model_id": "m1"})
    assert len(drift) == 1  # different type -> new notification
    assert store.records[0]["severity"] == "INFO"
    # Non-scenario lifecycle kinds never reach the Alert Center.
    assert retraining.lifecycle_notifications(event_kind="MODEL_PROMOTED", store=store, context={"model_id": "m1"}) == []


def test_lifecycle_notification_types_cover_locked_plan():
    store = _FakeStore()
    scenario_kinds = [
        "DRIFT_DETECTED", "RETRAINING_REVIEW_RECOMMENDED", "CANDIDATE_READY",
    ]
    echo_kinds = [
        "DEGRADATION_DETECTED", "RETRAINING_STARTED", "CANDIDATE_REJECTED",
        "MODEL_PROMOTED", "ROLLBACK_COMPLETED", "EVIDENCE_INSUFFICIENT",
    ]
    for kind in scenario_kinds:
        recs = retraining.lifecycle_notifications(event_kind=kind, store=store, context={"model_version": f"mv-{kind}"})
        assert len(recs) == 1, kind
        assert recs[0]["severity"] in ("INFO", "WARNING", "HIGH", "CRITICAL")
    for kind in echo_kinds:
        assert retraining.lifecycle_notifications(event_kind=kind, store=store, context={"model_version": f"mv-{kind}"}) == [], kind
    assert len(store.records) == len(scenario_kinds)


# ------------------------------------------------------------------ API wiring


def _wired_client(registry, lab_ctx):
    from dltime.api_lab import wire_model_lab

    app = FastAPI()
    wire_model_lab(app, get_registry=lambda: registry, get_lab_context=lambda: lab_ctx)
    return TestClient(app)


class _FakeExperiments:
    def list(self):
        return []

    def register(self, rec):
        return rec


class _FakeModels:
    def __init__(self):
        self._recs: list[dict] = []

    def list(self):
        return list(self._recs)

    def get(self, model_id):
        for r in self._recs:
            if r.get("model_id") == model_id:
                return r
        return None

    def promoted_model(self):
        return None

    def register(self, rec):
        self._recs.append(rec)
        return rec

    def promote(self, model_id, **kw):
        rec = self.get(model_id)
        rec["status"] = ModelState.PROMOTED
        rec["promotion"] = kw
        return rec

    def reject(self, model_id, **kw):
        rec = self.get(model_id)
        rec["status"] = ModelState.REJECTED
        return rec

    def rollback(self, **kw):
        return {"model_id": "restored"}


class _FakeRegistry:
    def __init__(self):
        self.models = _FakeModels()
        self.experiments = _FakeExperiments()


def test_api_decide_endpoint(tmp_decisions):
    reg = _FakeRegistry()
    client = _wired_client(reg, {"monitoring_store": _FakeStore()})
    resp = client.post(
        "/retraining/decide",
        json={"decision_type": "CONTINUE_MONITORING", "reason": "drift only", "decided_by": "tester"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["decision"]["decision_type"] == "CONTINUE_MONITORING"
    # invalid decision type -> 422
    bad = client.post(
        "/retraining/decide",
        json={"decision_type": "NOT_A_TYPE", "reason": "x", "decided_by": "t"},
    )
    assert bad.status_code == 422


def test_api_decisions_listing(tmp_decisions):
    reg = _FakeRegistry()
    client = _wired_client(reg, {"monitoring_store": _FakeStore()})
    client.post("/retraining/decide", json={"decision_type": "START_RETRAINING", "reason": "r", "decided_by": "t"})
    resp = client.get("/retraining/decisions")
    assert resp.status_code == 200
    assert resp.json()["decisions"][0]["decision_type"] == "START_RETRAINING"


def test_api_serving_status(tmp_switch):
    reg = _FakeRegistry()
    client = _wired_client(reg, {"monitoring_store": _FakeStore()})
    resp = client.get("/retraining/serving-status")
    assert resp.status_code == 200
    body = resp.json()
    assert "serving_version" in body and "candidate_override_active" in body
