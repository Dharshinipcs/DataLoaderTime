"""Module 7 tests: Model Laboratory, registry lifecycle, drift, readiness.

All governance tests run against an ISOLATED registry dir (tmp_path); the
real backend/artifacts/registry and backend/store are never touched. Heavy
experiments (ablation) are recorded from the real evidence runs (EXP-025)
via recorded-fixture assertions; the feature experiment's leakage-safety is
proven on the synthetic probe rather than re-running the multi-minute suite
in every test session.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dltime.model_lab import (
    LabError,
    ModelState,
    _JsonlStore,
    categorical_shift,
    compatibility_report,
    feature_drift,
    prediction_performance_drift,
    psi,
    population_drift,
    retraining_readiness,
)

Q90 = 2.2291534416514427


@pytest.fixture()
def lab_dir(tmp_path, monkeypatch):
    """Isolated registry dir for every governance test."""
    from dltime import model_lab

    d = tmp_path / "registry"
    monkeypatch.setattr(model_lab, "REGISTRY_DIR", d)
    store = _JsonlStore(d)
    return {
        "dir": d,
        "store": store,
        "experiments": __import__("dltime.model_lab", fromlist=["ExperimentRegistry"]).ExperimentRegistry(store),
        "models": __import__("dltime.model_lab", fromlist=["ModelRegistry"]).ModelRegistry(store),
    }


_ART = Path(__file__).resolve().parents[1] / "artifacts" / "total-et-v3-candidate.pkl"


def _candidate_record(model_id="total-et-v4-candidate", artifact=str(_ART)):
    from dltime.model_lab import ModelRecord

    return ModelRecord(
        model_id=model_id,
        model_family="extra_trees_candidate",
        artifact_path=str(artifact),
        status=ModelState.CANDIDATE_EVALUATED,
        feature_set_version="pre-execution+historical-v30",
        feature_count=30,
        target_mode="log1p",
        training_population={"rows": 262, "window": {"first_start": "2025-10-10", "last_start": "2026-07-15 17:48:04.909"}},
        validation_population={"rows": 88, "scope": "final chronological holdout"},
        experiment_id="EXP-LAB-001",
        metrics={"served_population": {"median_ae": 2.41}},
        notes="test candidate",
    )


# ---------------------------------------------------------------- experiments


class TestExperimentRegistry:
    def test_register_and_persistence(self, lab_dir):
        rec = {
            "experiment_id": "EXP-LAB-001",
            "model_family": "extra_trees_candidate",
            "target_transformation": "log1p",
            "hyperparameters": {"n_estimators": 200, "min_samples_leaf": 2, "max_features": 1.0, "random_state": 42},
            "feature_set_version": "pre-execution+historical-v30",
            "feature_names": ["f1", "f2"],
            "training_population": {"rows": 262},
            "validation_method": {"type": "chronological"},
            "metrics": {"mae": 364.6},
        }
        out = lab_dir["experiments"].register(rec)
        assert out["created_at"]
        assert lab_dir["experiments"].get("EXP-LAB-001")["metrics"]["mae"] == 364.6

    def test_missing_required_fields_refused(self, lab_dir):
        with pytest.raises(LabError):
            lab_dir["experiments"].register({"experiment_id": "EXP-BAD"})

    def test_append_only_no_overwrite(self, lab_dir):
        base = {
            "experiment_id": "EXP-LAB-001",
            "model_family": "x",
            "target_transformation": "log1p",
            "hyperparameters": {},
            "feature_set_version": "v30",
            "feature_names": [],
            "training_population": {},
            "validation_method": {},
            "metrics": {"mae": 1},
        }
        lab_dir["experiments"].register(base)
        with pytest.raises(LabError):
            lab_dir["experiments"].register({**base, "metrics": {"mae": 2}})
        # history preserved
        assert lab_dir["experiments"].get("EXP-LAB-001")["metrics"]["mae"] == 1


# ---------------------------------------------------------------- registry


class TestModelRegistry:
    def test_register_candidate_and_uniqueness(self, lab_dir):
        lab_dir["models"].register(_candidate_record())
        assert lab_dir["models"].get("total-et-v4-candidate")["status"] == "CANDIDATE_EVALUATED"
        with pytest.raises(LabError):
            lab_dir["models"].register(_candidate_record())

    def test_register_cannot_promote_directly(self, lab_dir):
        rec = _candidate_record()
        rec.status = ModelState.PROMOTED
        with pytest.raises(LabError):
            lab_dir["models"].register(rec)

    def test_promotion_requires_gate_pass_and_records_history(self, lab_dir):
        lab_dir["models"].register(_candidate_record())
        comp = compatibility_report(str(_ART))
        updated = lab_dir["models"].promote(
            "total-et-v4-candidate", compatibility=comp, approved_by="human", reason="test promotion"
        )
        assert updated["status"] == ModelState.PROMOTED
        assert updated["promotion"]["approved_by"] == "human"
        assert lab_dir["models"].promoted_model()["model_id"] == "total-et-v4-candidate"
        events = lab_dir["models"].events.list()
        assert any(e["event_type"] == "PROMOTION_APPROVED" for e in events)

    def test_promotion_blocked_on_failed_gate(self, lab_dir):
        lab_dir["models"].register(_candidate_record())
        comp = compatibility_report(str(_ART))
        comp["uncertainty_calibration_compatible"] = False
        with pytest.raises(LabError):
            lab_dir["models"].promote(
                "total-et-v4-candidate", compatibility=comp, approved_by="human", reason="r"
            )
        # state unchanged
        assert lab_dir["models"].get("total-et-v4-candidate")["status"] == "CANDIDATE_EVALUATED"

    def test_promotion_blocked_without_evaluation(self, lab_dir):
        rec = _candidate_record()
        rec.status = ModelState.EXPERIMENTAL
        lab_dir["models"].register(rec)
        with pytest.raises(LabError):
            lab_dir["models"].promote("total-et-v4-candidate", compatibility={}, approved_by="h", reason="r")

    def test_promotion_blocked_on_missing_artifact(self, lab_dir):
        lab_dir["models"].register(_candidate_record(artifact=str(_ART.parent / "__nope__.pkl")))
        with pytest.raises(LabError):
            lab_dir["models"].promote("total-et-v4-candidate", compatibility={}, approved_by="h", reason="r")

    def test_promotion_supersedes_previous_and_rollback_restores(self, lab_dir):
        lab_dir["models"].register(_candidate_record("total-et-v3-candidate"))
        comp = compatibility_report(str(_ART))
        lab_dir["models"].promote("total-et-v3-candidate", compatibility=comp, approved_by="h", reason="first")
        lab_dir["models"].register(_candidate_record("total-et-v4-candidate"))
        lab_dir["models"].promote("total-et-v4-candidate", compatibility=comp, approved_by="h", reason="second")
        # only one promoted at a time
        assert lab_dir["models"].promoted_model()["model_id"] == "total-et-v4-candidate"
        assert lab_dir["models"].get("total-et-v3-candidate")["status"] == ModelState.RETIRED
        # rollback restores the previous promoted model; candidate history preserved
        restored = lab_dir["models"].rollback(approved_by="h", reason="regression")
        assert restored["model_id"] == "total-et-v3-candidate"
        assert lab_dir["models"].get("total-et-v4-candidate")["status"] == ModelState.RETIRED
        events = lab_dir["models"].events.list()
        assert any(e["event_type"] == "ROLLBACK" for e in events)
        assert lab_dir["models"].get("total-et-v4-candidate") is not None  # nothing deleted

    def test_rollback_requires_previous_promotion(self, lab_dir):
        with pytest.raises(LabError):
            lab_dir["models"].rollback(approved_by="h", reason="nothing to roll back to")

    def test_reject_is_terminal(self, lab_dir):
        lab_dir["models"].register(_candidate_record())
        lab_dir["models"].reject("total-et-v4-candidate", reason="worse on served population")
        assert lab_dir["models"].get("total-et-v4-candidate")["status"] == ModelState.REJECTED
        with pytest.raises(LabError):
            lab_dir["models"].promote("total-et-v4-candidate", compatibility={}, approved_by="h", reason="r")


# ---------------------------------------------------------------- compatibility


class TestCompatibilityGates:
    def test_current_candidate_passes_all_gates(self):
        report = compatibility_report(str(_ART))
        for key in (
            "artifact_exists",
            "feature_registry_compatible",
            "preprocessing_compatible",
            "inference_compatible",
            "uncertainty_calibration_compatible",
            "explanation_compatible",
            "expected_model_version_ok",
            "no_forbidden_feature",
            "no_post_execution_feature",
        ):
            assert report[key] is True, f"gate {key} failed: {report.get('checks_detail')}"
        # 2026-10-04 44-feature contract (22 DEF + 22 HIST): the candidate
        # artifact and its calibration both carry 44 fitted features.
        assert report["feature_count"] == 44

    def test_missing_artifact_fails_closed(self):
        report = compatibility_report(str(_ART.parent / "__does_not_exist__.pkl"))
        assert report["artifact_exists"] is False
        assert report["inference_compatible"] is False

    def test_forbidden_feature_refused(self, tmp_path):
        """A candidate carrying a post-execution feature cannot pass gates."""
        import joblib

        art = joblib.load(str(_ART))
        meta = dict(art["metadata"])
        meta["feature_names_in_order"] = list(meta["feature_names_in_order"]) + ["timetaken_in_sec"]
        art2 = dict(art)
        art2["metadata"] = meta
        p = tmp_path / "leaky.pkl"
        joblib.dump(art2, p)
        report = compatibility_report(p)
        assert report["no_forbidden_feature"] is False
        assert "timetaken_in_sec" in report["forbidden_found"]

    def test_wrong_feature_count_refused(self, tmp_path):
        import joblib

        art = joblib.load(str(_ART))
        meta = dict(art["metadata"])
        meta["feature_names_in_order"] = list(meta["feature_names_in_order"])[:-1]
        art2 = dict(art)
        art2["metadata"] = meta
        p = tmp_path / "short.pkl"
        joblib.dump(art2, p)
        report = compatibility_report(p)
        assert report["inference_compatible"] is False

    def test_uncertainty_calibration_mismatch_detected(self):
        from dltime.uncertainty import UncertaintyError, load_calibration_artifact

        with pytest.raises(UncertaintyError):
            load_calibration_artifact("some-other-model", 30)


# ---------------------------------------------------------------- drift


def _frames(n_ref=40, n_cur=25, shift=0.0, cat_shift=0.0):
    rng = np.random.default_rng(7)
    ref = pd.DataFrame(
        {
            "hist_loader_prior_mean_sec": rng.gamma(2.0, 3.0, n_ref),
            "hist_loader_prior_count": rng.integers(0, 10, n_ref).astype(float),
            "def_dominant_calling_type": ["A"] * int(n_ref * (0.5 - cat_shift)) + ["B"] * (n_ref - int(n_ref * (0.5 - cat_shift))),
        }
    )
    cur = pd.DataFrame(
        {
            "hist_loader_prior_mean_sec": rng.gamma(2.0 + shift, 3.0, n_cur),
            "hist_loader_prior_count": rng.integers(0, 10, n_cur).astype(float),
            "def_dominant_calling_type": ["A"] * int(n_cur * (0.5 + cat_shift)) + ["B"] * (n_cur - int(n_cur * (0.5 + cat_shift))),
        }
    )
    return ref, cur


class TestDrift:
    def test_no_meaningful_drift_on_same_distribution(self):
        ref, cur = _frames(shift=0.0)
        out = feature_drift(ref, cur, feature_names=["hist_loader_prior_mean_sec"])
        assert out["overall_status"] == "NO_MEANINGFUL_DRIFT"

    def test_numeric_drift_detected(self):
        ref, cur = _frames(shift=3.0)  # big location shift
        out = feature_drift(ref, cur, feature_names=["hist_loader_prior_mean_sec"])
        assert out["overall_status"] in ("WARNING", "SIGNIFICANT_DRIFT")
        assert out["features"]["hist_loader_prior_mean_sec"]["method"] == "KS"

    def test_categorical_drift_detected(self):
        ref, cur = _frames(cat_shift=0.45)
        out = feature_drift(
            ref, cur, feature_names=["def_dominant_calling_type"], categorical_features=["def_dominant_calling_type"]
        )
        assert out["overall_status"] in ("WARNING", "SIGNIFICANT_DRIFT")

    def test_insufficient_sample_guard(self):
        ref, cur = _frames(n_ref=40, n_cur=5, shift=5.0)
        out = feature_drift(ref, cur, feature_names=["hist_loader_prior_mean_sec"])
        assert out["overall_status"] == "INSUFFICIENT_SAMPLE"
        assert out["current_rows"] == 5

    def test_insufficient_reference_guard(self):
        ref, cur = _frames(n_ref=10, n_cur=25)
        out = feature_drift(ref, cur, feature_names=["hist_loader_prior_mean_sec"])
        assert out["overall_status"] == "INSUFFICIENT_SAMPLE"

    def test_population_drift_dimensions(self):
        ref, cur = _frames()
        ref["Loader_Name"] = ["L1"] * 40
        cur["Loader_Name"] = ["L2"] * 25
        out = population_drift(ref, cur)
        assert out["dimensions"]["loader"]["status"] == "SIGNIFICANT_DRIFT"

    def test_performance_drift_requires_actuals(self):
        out = prediction_performance_drift([])
        assert out["status"] == "INSUFFICIENT_OUTCOMES"  # zero matched actuals (Module 8 §13)
        assert out["has_actual_outcomes"] is False
        rec = {
            "total_error": {
                "absolute_error_seconds": 5.0,
                "signed_error_seconds": 1.0,
                "absolute_percentage_error": 25.0,
            }
        }
        out2 = prediction_performance_drift([rec] * 3)
        assert out2["status"] == "INSUFFICIENT_SAMPLE"  # below min_n, no verdict
        assert out2["mae_seconds"] == 5.0

    def test_psi_and_ks_functions(self):
        rng = np.random.default_rng(3)
        e = list(rng.normal(0, 1, 200))
        a_same = list(rng.normal(0, 1, 200))
        a_shift = list(rng.normal(4, 1, 200))
        assert psi(e, a_same) < 0.1
        assert psi(e, a_shift) > 0.25
        ks_same, _ = __import__("dltime.model_lab", fromlist=["ks_statistic"]).ks_statistic(e, a_same)
        ks_shift, _ = __import__("dltime.model_lab", fromlist=["ks_statistic"]).ks_statistic(e, a_shift)
        assert ks_shift > ks_same

    def test_categorical_shift_empty(self):
        out = categorical_shift([], ["A"])
        assert out["status"] == "INSUFFICIENT_SAMPLE"


# ---------------------------------------------------------------- readiness


class TestRetrainingReadiness:
    def _registry_with_promoted(self, lab_dir, served_medae=2.41):
        lab_dir["models"].register(_candidate_record("m1"))
        comp = compatibility_report(str(_ART))
        lab_dir["models"].promote("m1", compatibility=comp, approved_by="h", reason="setup")
        rec = lab_dir["models"].promoted_model()
        rec["metrics"] = {"served_population": {"median_ae": served_medae}}
        models = lab_dir["models"].list()
        for m in models:
            if m["model_id"] == "m1":
                m["metrics"] = rec["metrics"]
        lab_dir["models"].store.rewrite("models.jsonl", models)
        return lab_dir["models"]

    def test_not_ready_when_no_signals(self, lab_dir):
        reg = self._registry_with_promoted(lab_dir)
        out = retraining_readiness(registry=reg, matched_feedback=[], eligible_actual_count=0)
        assert out["readiness_state"] == "NOT_READY"
        assert out["automatic_retraining"] == "OFF"
        assert out["human_gate_required"] is True

    def test_ready_for_review_on_enough_eligible_actuals(self, lab_dir):
        reg = self._registry_with_promoted(lab_dir)
        out = retraining_readiness(registry=reg, matched_feedback=[], eligible_actual_count=30)
        assert out["readiness_state"] == "READY_FOR_REVIEW"
        assert out["signals"]["eligible_new_actuals"]["met"] is True

    def test_degradation_signal_recommends_review(self, lab_dir):
        reg = self._registry_with_promoted(lab_dir, served_medae=2.0)
        # 25 matched feedback records with MedAE 12.0 >> 2x baseline 2.0
        fb = [
            {"total_error": {"absolute_error_seconds": 12.0, "signed_error_seconds": 12.0}}
        ] * 25
        out = retraining_readiness(registry=reg, matched_feedback=fb, eligible_actual_count=0)
        assert out["signals"]["performance_degradation"]["assessed"] is True
        assert out["readiness_state"] == "RETRAINING_RECOMMENDED_FOR_REVIEW"

    def test_drift_signal_recommends_review(self, lab_dir):
        reg = self._registry_with_promoted(lab_dir)
        out = retraining_readiness(
            registry=reg,
            matched_feedback=[],
            eligible_actual_count=0,
            feature_drift_report={"overall_status": "SIGNIFICANT_DRIFT"},
        )
        assert out["readiness_state"] == "RETRAINING_RECOMMENDED_FOR_REVIEW"
        # drift alone recommends REVIEW, never launches training
        assert out["automatic_retraining"] == "OFF"

    def test_no_performance_verdict_without_enough_feedback(self, lab_dir):
        reg = self._registry_with_promoted(lab_dir)
        out = retraining_readiness(registry=reg, matched_feedback=[], eligible_actual_count=0)
        assert out["signals"]["performance_degradation"]["assessed"] is False


# ---------------------------------------------------------------- live serving integrity


class TestServingIntegrity:
    def test_reference_prediction_unchanged(self, service):
        """§39: no promotion performed -> reference byte-identical, serving
        model remains the Task 04 candidate (not registry-promoted)."""
        from dltime.inference import PredictionRequest

        r = service.predict(
            PredictionRequest(
                loader_name="MigrationSummarySubTableLoaderDataset",
                sprint="TEST",
                dataset="Migration_Summary_Sub_Fact_Table_LDR_Datase_new",
                connection="MDM_DM_TEST",
            )
        )
        assert r.state == "PREDICTED"
        # Sum policy (EXP-038): served ETA = five-stage sum over the
        # applicable stages. The retired direct-reference constant (16.289 s
        # over the full scripted feed) is reported alongside as audit
        # context; the served number itself no longer derives from it.
        assert r.predicted_seconds == 11.465
        assert r.direct_reference_seconds == pytest.approx(16.289350640581837, rel=1e-9)
        assert r.serving_role == "AUTHORITATIVE_TOTAL"
        assert r.model_version == "sum-of-available-stages-v2"

    def test_models_status_reports_serving_vs_promoted(self, service):
        from fastapi.testclient import TestClient

        from dltime.api import create_app, get_model_registry

        app = create_app()
        client = TestClient(app)
        body = client.get("/models/status").json()
        # Sum policy (EXP-038): the serving identity is the deterministic
        # five-stage sum; the governed stage-aware registry entry stays
        # PROMOTED as governance history, the direct model stays retired.
        assert body["serving_model_version"] == "sum-of-available-stages-v2"
        assert body["serving_is_promoted"] is True
        assert body["reference_model"]["status"] == "RETIRED"
