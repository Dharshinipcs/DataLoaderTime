"""Drift & Retraining overview tests (locked plan).

The overview must COMPOSE the existing lab signals (feature/population drift,
performance drift, readiness, registry baselines, lifecycle) without inventing
new statistics. All fakes are hermetic: no real artifacts, registry files, or
backend/store data are touched.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dltime.model_lab import (
    MIN_WINDOW_ROWS,
    drift_overview,
    prediction_drift_summary,
    stage_model_health,
)


# ------------------------------------------------------------------ fakes

def _feedback(n: int, *, stage: str | None = None, abs_err: float = 1.0) -> list[dict]:
    out: list[dict] = []
    for i in range(n):
        rec: dict = {
            "linkage": {"status": "MATCHED"},
            "total_error": {
                "absolute_error_seconds": abs_err,
                "signed_error_seconds": -abs_err,
                "absolute_percentage_error": 5.0,
            },
        }
        if stage is not None:
            rec["stage_errors"] = {stage: {"absolute_error_seconds": abs_err}}
        out.append(rec)
    return out


def _selection_experiment(stage_suffix: str, medae: float) -> dict:
    return {
        "experiment_id": f"EXP-STAGE-{stage_suffix}-SELECTION",
        "metrics": {"some_kind": {"median_ae": medae, "n": 40}},
    }


def _fake_registry(baseline_medae: float | None = 2.41):
    promoted = {
        "model_id": "stage-aware-total-ad34",
        "status": "PROMOTED",
    }
    if baseline_medae is not None:
        promoted["metrics"] = {"served_population": {"median_ae": baseline_medae}}

    class _Models:
        def promoted_model(self):
            return promoted

    class _Experiments:
        def list(self):
            return []

    class _Registry:
        models = _Models()
        experiments = _Experiments()

    return _Registry()


# --------------------------------------------------------- stage_model_health

class TestStageModelHealth:
    def test_insufficient_outcomes_never_claim_health(self):
        out = stage_model_health(_feedback(3, stage="pre_load"), selection_experiments=[])
        assert out["overall_status"] == "INSUFFICIENT_EVIDENCE"
        for rec in out["stages"].values():
            assert rec["status"] == "INSUFFICIENT_EVIDENCE"
            assert rec["recent_outcomes"] <= 3
            assert rec["min_required"] == MIN_WINDOW_ROWS

    def test_degradation_against_selection_baseline(self):
        experiments = [_selection_experiment("PRE_LOAD", medae=1.0)]
        # 20 linked outcomes with clearly elevated pre-load error (>= 2x baseline).
        out = stage_model_health(
            _feedback(MIN_WINDOW_ROWS, stage="pre_load", abs_err=5.0),
            selection_experiments=experiments,
        )
        pre = out["stages"]["pre_load"]
        assert pre["status"] == "DEGRADATION_DETECTED"
        assert pre["baseline_medae_seconds"] == 1.0
        assert pre["recent_medae_seconds"] == 5.0
        assert out["overall_status"] == "DEGRADATION_DETECTED"

    def test_normal_performance_within_baseline_band(self):
        experiments = [_selection_experiment("PRE_LOAD", medae=10.0)]
        out = stage_model_health(
            _feedback(MIN_WINDOW_ROWS, stage="pre_load", abs_err=5.0),
            selection_experiments=experiments,
        )
        assert out["stages"]["pre_load"]["status"] == "PERFORMING_NORMALLY"

    def test_unknown_stage_inputs_are_ignored(self):
        out = stage_model_health(_feedback(MIN_WINDOW_ROWS, stage="bogus_stage"), selection_experiments=[])
        assert all(r["status"] == "INSUFFICIENT_EVIDENCE" for r in out["stages"].values())


# --------------------------------------------- post-promotion (live) error

class TestPostPromotionError:
    """The live "Current serving (since promotion)" error: matched
    outcomes recorded AFTER the stage's serving epoch only. Shown
    from the first outcome (n >= 1); the health badge stays
    policy-gated at MIN_WINDOW_ROWS."""

    EPOCH = {"model_id": "cand-1", "since": "2026-01-15T00:00:00Z", "source": "registry"}

    def _feedback(self, *items: tuple[str, str, float]) -> list[dict]:
        out: list[dict] = []
        for created, stage, err in items:
            out.append(
                {
                    "linkage": {"status": "MATCHED"},
                    "created_at": created,
                    "stage_errors": {stage: {"absolute_error_seconds": err}},
                    "total_error": {"absolute_error_seconds": err},
                }
            )
        return out

    def test_outcomes_before_promotion_excluded_after_included(self):
        fb = self._feedback(
            ("2026-01-01T00:00:00Z", "pre_load", 9.0),
            ("2026-01-14T23:59:59Z", "pre_load", 9.0),
            ("2026-01-15T00:00:00Z", "pre_load", 2.0),
            ("2026-01-20T00:00:00Z", "pre_load", 4.0),
        )
        out = stage_model_health(
            fb, selection_experiments=[], serving_epochs={"pre_load": dict(self.EPOCH)}
        )
        post = out["stages"]["pre_load"]["post_promotion"]
        assert post is not None
        assert post["outcomes"] == 2  # the two pre-promotion outcomes are excluded
        assert post["median_absolute_error_seconds"] == 3.0
        assert post["promoted_at"] == self.EPOCH["since"]
        assert post["model_id"] == "cand-1"
        assert post["source"] == "registry"
        # the all-outcome MedAE stays available as drill-through context
        assert out["stages"]["pre_load"]["recent_outcomes"] == 4

    def test_empty_window_yields_no_post_promotion_value(self):
        fb = self._feedback(("2026-01-01T00:00:00Z", "pre_load", 9.0))
        out = stage_model_health(
            fb, selection_experiments=[], serving_epochs={"pre_load": dict(self.EPOCH)}
        )
        assert out["stages"]["pre_load"]["post_promotion"] is None

    def test_no_serving_epoch_yields_no_post_promotion(self):
        fb = self._feedback(("2026-01-20T00:00:00Z", "pre_load", 2.0))
        out = stage_model_health(fb, selection_experiments=[], serving_epochs=None)
        assert out["stages"]["pre_load"]["post_promotion"] is None
        assert out["total_post_promotion"] is None

    def test_total_post_promotion_scoped_to_latest_epoch(self):
        fb = self._feedback(
            ("2026-01-01T00:00:00Z", "staging", 10.0),
            ("2026-01-16T00:00:00Z", "staging", 2.0),
            ("2026-01-17T00:00:00Z", "staging", 6.0),
        )
        epochs = {
            "staging": {"model_id": "a", "since": "2026-01-10T00:00:00Z", "source": "registry"},
            "pre_load": {"model_id": "b", "since": "2026-01-16T12:00:00Z", "source": "serving-switch"},
        }
        out = stage_model_health(fb, selection_experiments=[], serving_epochs=epochs)
        total = out["total_post_promotion"]
        assert total is not None
        # the total row uses the LATEST epoch (current composition's activation)
        assert total["promoted_at"] == "2026-01-16T12:00:00Z"
        assert total["outcomes"] == 1  # only the 2026-01-17 outcome qualifies
        assert total["median_absolute_error_seconds"] == 6.0

    def test_health_badge_stays_policy_gated_with_post_promotion_present(self):
        fb = self._feedback(("2026-01-20T00:00:00Z", "pre_load", 5.0))
        out = stage_model_health(
            fb,
            selection_experiments=[_selection_experiment("PRE_LOAD", medae=1.0)],
            serving_epochs={"pre_load": dict(self.EPOCH)},
        )
        # one post-promotion outcome: the live NUMBER shows, the badge does not
        assert out["stages"]["pre_load"]["post_promotion"]["outcomes"] == 1
        assert out["stages"]["pre_load"]["status"] == "INSUFFICIENT_EVIDENCE"

# ---------------------------------------------------- prediction_drift_summary

class TestPredictionDriftSummary:
    def test_shifted_predictions_flag_warning(self):
        events = (
            [{"created_at": f"2026-01-0{i}T00:00:00Z", "event_state": "PREDICTED",
              "prediction": {"predicted_total_seconds": 10.0 + i}} for i in range(1, 9)]
            + [{"created_at": f"2026-02-0{i}T00:00:00Z", "event_state": "PREDICTED",
                "prediction": {"predicted_total_seconds": 30.0 + i}} for i in range(1, 9)]
        )
        out = prediction_drift_summary(events)
        assert out["status"] in ("WARNING", "SIGNIFICANT_DRIFT")
        assert out["reference_predictions"]["max_seconds"] < out["recent_predictions"]["min_seconds"]

    def test_empty_inputs_are_insufficient_not_drift(self):
        out = prediction_drift_summary([])
        assert out["status"] == "INSUFFICIENT_SAMPLE"

    def test_unavailable_predictions_are_excluded(self):
        events = [
            {"created_at": "2026-01-01T00:00:00Z", "event_state": "UNRESOLVED_IDENTITY"},
            {"created_at": "2026-02-01T00:00:00Z", "event_state": "PREDICTED",
             "prediction": {"predicted_total_seconds": 15.0}},
        ]
        out = prediction_drift_summary(events)
        assert out["recent_predictions"]["count"] == 1
        assert out["reference_predictions"]["count"] == 0
        assert out["status"] == "INSUFFICIENT_SAMPLE"


# ---------------------------------------------------------------- composition

def _fd_report(status: str, rows: tuple[int, int] = (300, 25), top: list[dict] | None = None):
    return {
        "overall_status": status,
        "reference_rows": rows[0],
        "current_rows": rows[1],
        "top_shifted_features": [t["name"] for t in (top or [])],
        "features": {t["name"]: {"status": t.get("status", status), "statistic": 0.42, "method": "KS"} for t in (top or [])},
    }


def _perf_report(status: str, n: int = 0):
    return {"status": status, "sample_count": n, "has_actual_outcomes": n > 0, "available": n >= MIN_WINDOW_ROWS}


class TestDriftOverview:
    def test_healthy_composition(self):
        out = drift_overview(
            registry=_fake_registry(),
            matched_feedback=[],
            prediction_events=[],
            feature_drift_report=_fd_report("NO_MEANINGFUL_DRIFT"),
            population_drift_report={"overall_status": "NO_MEANINGFUL_DRIFT", "dimensions": {}},
            performance_report=_perf_report("INSUFFICIENT_OUTCOMES"),
            selection_experiments=[],
            eligible_actual_count=4,
        )
        assert out["overview_version"] == "drift-overview-v1"
        assert out["headline"]["data_health"] == "STABLE"
        assert out["headline"]["action"] == "NO_RETRAINING_NEEDED"
        assert out["retraining"]["automatic_retraining"] == "OFF"
        assert out["retraining"]["human_gate_required"] is True
        ids = [e["id"] for e in out["assessment"]["evidence"]]
        assert ids == ["new_eligible_outcomes", "data_drift", "prediction_drift", "model_performance", "stage_degradation"]
        assert "4 new eligible execution outcomes" in out["assessment"]["evidence"][0]["label"]

    def test_data_drift_alone_is_watch_not_retrain(self):
        out = drift_overview(
            registry=_fake_registry(),
            matched_feedback=[],
            prediction_events=[],
            feature_drift_report=_fd_report("SIGNIFICANT_DRIFT"),
            population_drift_report={"overall_status": "NO_MEANINGFUL_DRIFT", "dimensions": {}},
            performance_report=_perf_report("INSUFFICIENT_OUTCOMES"),
            selection_experiments=[],
            eligible_actual_count=40,
        )
        assert out["headline"]["data_health"] == "DRIFT_DETECTED"
        assert out["assessment"]["verdict"] == "CONTINUE_MONITORING"
        # The EXISTING readiness policy may flag review (signals only); the page
        # verdict stays monitoring because performance is not assessable/degraded.
        assert out["retraining"]["human_gate_required"] is True
        assert out["retraining"]["automatic_retraining"] == "OFF"

    def test_stage_degradation_triggers_review_with_evidence(self):
        experiments = [_selection_experiment("PRE_LOAD", medae=1.0)]
        feedback = _feedback(MIN_WINDOW_ROWS, stage="pre_load", abs_err=5.0)
        out = drift_overview(
            registry=_fake_registry(baseline_medae=None),  # readiness baseline path off
            matched_feedback=feedback,
            prediction_events=[],
            feature_drift_report=_fd_report("NO_MEANINGFUL_DRIFT"),
            population_drift_report={"overall_status": "NO_MEANINGFUL_DRIFT", "dimensions": {}},
            performance_report=_perf_report("MONITORED", n=MIN_WINDOW_ROWS),
            selection_experiments=experiments,
            eligible_actual_count=40,
        )
        assert out["model_performance"]["stage_health"]["stages"]["pre_load"]["status"] == "DEGRADATION_DETECTED"
        assert out["assessment"]["verdict"] == "RETRAINING_REVIEW_RECOMMENDED"
        met = {e["id"] for e in out["assessment"]["evidence"] if e["met"]}
        assert "stage_degradation" in met

    def test_feature_rows_are_grouped_and_jargon_free(self):
        top = [
            {"name": "stage_hist_prev_stage_sec", "status": "NO_MEANINGFUL_DRIFT"},
            {"name": "execution_order", "status": "WARNING"},
            {"name": "something_unusual", "status": "WARNING"},
        ]
        out = drift_overview(
            registry=_fake_registry(),
            matched_feedback=[],
            prediction_events=[],
            feature_drift_report=_fd_report("WARNING", top=top),
            population_drift_report={"overall_status": "NO_MEANINGFUL_DRIFT", "dimensions": {}},
            performance_report=_perf_report("INSUFFICIENT_OUTCOMES"),
            selection_experiments=[],
            eligible_actual_count=0,
        )
        labels = [r["feature"] for r in out["data_drift"]["feature_rows"]]
        assert "Historical duration context" in labels
        assert "Execution Order" in labels
        assert "Something Unusual" in labels
        for row in out["data_drift"]["feature_rows"]:
            assert "monitored_inputs" in row
            assert all("_" not in row["feature"] for _ in [0])  # human labels, no raw snake_case
            assert not any(k.startswith("stage_hist") for k in row)

    def test_assessment_wording_rules(self):
        out = drift_overview(
            registry=_fake_registry(),
            matched_feedback=[],
            prediction_events=[],
            feature_drift_report=_fd_report("NO_MEANINGFUL_DRIFT"),
            population_drift_report={"overall_status": "NO_MEANINGFUL_DRIFT", "dimensions": {}},
            performance_report=_perf_report("INSUFFICIENT_OUTCOMES"),
            selection_experiments=[],
            eligible_actual_count=0,
        )
        text = " ".join(out["notes"] + [out["assessment"]["narrative"]]).lower()
        assert "automatically" in text or "automatic" in text
        assert "psi" not in text and "n_train" not in text

    def test_serving_epoch_and_latest_candidate_in_payload(self):
        experiments = [
            _selection_experiment("PRE_LOAD", medae=1.0),
            {
                "experiment_id": "EXP-RETRAIN-20260120",
                "created_at": "2026-01-20T00:00:00Z",
                "metrics": {"pre_load": {"median_ae": 0.8, "n": 20}},
            },
        ]
        out = drift_overview(
            registry=_fake_registry(),
            matched_feedback=[],
            prediction_events=[],
            feature_drift_report=_fd_report("NO_MEANINGFUL_DRIFT"),
            population_drift_report={"overall_status": "NO_MEANINGFUL_DRIFT", "dimensions": {}},
            performance_report=_perf_report("INSUFFICIENT_OUTCOMES"),
            selection_experiments=experiments,
            eligible_actual_count=0,
            serving_epochs={
                "pre_load": {
                    "model_id": "cand-1",
                    "since": "2026-01-15T00:00:00Z",
                    "source": "registry",
                }
            },
        )
        assert out["model_performance"]["serving_epoch"]["pre_load"]["source"] == "registry"
        cand = out["retraining"]["latest_candidate"]
        assert cand is not None
        assert cand["experiment_id"] == "EXP-RETRAIN-20260120"
        assert cand["per_stage"]["pre_load"]["median_ae"] == 0.8
        assert cand["per_stage"]["pre_load"]["n"] == 20


# ----------------------------------------------------------------- API wiring

class TestDriftOverviewEndpoint:
    def test_get_drift_overview_endpoint(self):
        from dltime.api_lab import wire_model_lab

        app = FastAPI()
        registry = _fake_registry()

        ctx = {
            "reference_frame": pd.DataFrame({"a": [1, 2, 3]}),
            "current_frame": pd.DataFrame({"a": [1]}),
            "feature_names": [],
            "categorical_features": [],
            "reference_window": {"first_start": "2025-10-10", "last_start": "2026-07-15", "rows": 256},
            "monitoring_store": type(
                "_Store",
                (),
                {
                    "read_feedback_records": staticmethod(lambda: ([], [])),
                    "read_prediction_events": staticmethod(lambda: ([], [])),
                    "read_actual_outcomes": staticmethod(lambda: ([], [])),
                },
            )(),
        }
        wire_model_lab(app, get_registry=lambda: registry, get_lab_context=lambda: ctx)
        client = TestClient(app)
        r = client.get("/drift/overview")
        assert r.status_code == 200
        body = r.json()
        assert body["overview_version"] == "drift-overview-v1"
        assert set(body["headline"]) == {"data_health", "model_health", "new_eligible_outcomes", "action"}
        assert body["headline"]["data_health"] == "INSUFFICIENT_DATA"
        assert body["assessment"]["verdict"] in ("NO_RETRAINING_NEEDED", "CONTINUE_MONITORING")
        assert body["retraining"]["automatic_retraining"] == "OFF"

    def test_loader_scope_filters_frames_and_reports_scope(self):
        import pandas as pd
        from dltime.api_lab import wire_model_lab

        app = FastAPI()
        registry = _fake_registry()
        big_ref = pd.DataFrame({"Loader_Name": ["L1"] * 40, "a": list(range(40))})
        big_cur = pd.DataFrame({"Loader_Name": ["L1", "L2"] * 15, "a": list(range(30))})
        ctx = {
            "reference_frame": big_ref,
            "current_frame": big_cur,
            "feature_names": ["a"],
            "categorical_features": [],
            "reference_window": {"first_start": "2025-10-10", "last_start": "2026-07-15", "rows": 40},
            "monitoring_store": type(
                "_Store",
                (),
                {
                    "read_feedback_records": staticmethod(lambda: ([], [])),
                    "read_prediction_events": staticmethod(lambda: ([], [])),
                    "read_actual_outcomes": staticmethod(lambda: ([], [])),
                },
            )(),
        }
        wire_model_lab(app, get_registry=lambda: registry, get_lab_context=lambda: ctx)
        client = TestClient(app)
        r = client.get("/drift/overview", params={"loader": "L1"})
        assert r.status_code == 200
        body = r.json()
        assert body["scope"] == {"kind": "loader", "value": "L1"}
        # L1 has 15 recent rows — below the policy minimum, so no verdict is issued.
        assert body["data_drift"]["overall_status"] == "INSUFFICIENT_SAMPLE"
        r2 = client.get("/drift/overview")
        assert r2.json()["scope"] == {"kind": "system", "value": None}
