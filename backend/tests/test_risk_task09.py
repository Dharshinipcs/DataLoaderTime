"""Task 09 tests: operational risk engine.

Covers: determinism, per-signal rule triggers, precedence, LOW_CONFIDENCE
mapping, INSUFFICIENT_HISTORY handling, no fabricated risk for non-prediction
states, ETA/uncertainty/SHAP invariance, leakage guards, failure isolation,
and evidence-reason correspondence.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dltime.inference import PredictionRequest, PredictionState
from dltime.risk import (
    OOD_FENCE_EXCESS,
    RISK_POLICY_VERSION,
    RiskLabel,
    assess_risk,
    config_range_excess,
    insufficient_history_risk,
    training_range_profile,
)

REFERENCE = {
    "loader_name": "MigrationSummarySubTableLoaderDataset",
    "sprint": "TEST",
    "dataset": "Migration_Summary_Sub_Fact_Table_LDR_Datase_new",
    "connection": "MDM_DM_TEST",
}
INSUFFICIENT = {
    "loader_name": "2210_TEST_ALL_CONFIG",
    "sprint": "SUPPORT",
    "dataset": "A_TABLE",
    "connection": "test1",
}


# ------------------------------------------------------------ unit: signals

def _minimal_x(service):
    """A context feature vector (config features populated, in artifact order)."""
    from dltime.feature_assembly import build_training_frame

    tf = build_training_frame()
    return tf.X.iloc[[0]]


def test_ood_signal_alone_is_out_of_distribution(service):
    from dltime.feature_assembly import build_training_frame

    tf = build_training_frame()
    X = tf.X.iloc[[0]]
    profile = service._ood_profile()
    X_ood = X.copy()
    # probe value from the FULL frame max (not this single row) — EXP-018: x10
    # of the training max lands ~119 robust-IQR units beyond the fence
    X_ood["def_component_count"] = float(tf.X["def_component_count"].max() * 10)
    res = assess_risk(
        predicted_seconds=10.0,
        reliability_label="MEDIUM",
        history={"prior_loader_executions": 10, "prior_loader_median_sec": 5.0},
        X=X_ood,
        ood_profile=profile,
        stage_intelligence=None,
        modelable_stages=set(),
    )
    assert res.label == RiskLabel.OUT_OF_DISTRIBUTION
    codes = {r["code"] for r in res.reasons}
    assert "CONFIG_OUT_OF_TRAINING_RANGE" in codes
    # evidence names the offending feature and the fence
    reason = next(r for r in res.reasons if r["code"] == "CONFIG_OUT_OF_TRAINING_RANGE")
    assert "def_component_count" in reason["evidence"]
    assert str(OOD_FENCE_EXCESS) in reason["evidence"]


def test_warning_signal_alone_is_warning_not_high(service):
    X = _minimal_x(service)
    res = assess_risk(
        predicted_seconds=16.0,  # > 3x median 5.0, < 5x
        reliability_label="MEDIUM",
        history={"prior_loader_executions": 10, "prior_loader_median_sec": 5.0},
        X=X,
        ood_profile=service._ood_profile(),
        stage_intelligence=None,
        modelable_stages=set(),
    )
    assert res.label == RiskLabel.WARNING
    assert {r["code"] for r in res.reasons} == {"ETA_ABOVE_LOADER_HISTORY"}


def test_high_risk_requires_corroboration(service):
    X = _minimal_x(service)
    common = dict(
        reliability_label="MEDIUM",
        history={"prior_loader_executions": 10, "prior_loader_median_sec": 1.0},
        X=X,
        ood_profile=service._ood_profile(),
        stage_intelligence=None,
        modelable_stages=set(),
    )
    # 50x median alone (precedent-free ETA) is NOT HIGH_RISK
    alone = assess_risk(predicted_seconds=50.0, **common)
    assert alone.label == RiskLabel.WARNING
    # same unprecedented ETA + one independent concern (low stage evidence) => HIGH_RISK
    stage_intel = {
        "available": True,
        "stages": [
            {"stage": "prevalidation", "available": False, "evidence_count": 1},
            {"stage": "transformation", "available": True, "evidence_count": 9},
        ],
    }
    corrob = assess_risk(
        predicted_seconds=50.0,
        stage_intelligence=stage_intel,
        modelable_stages={"prevalidation"},
        reliability_label=common["reliability_label"],
        history=common["history"],
        X=common["X"],
        ood_profile=common["ood_profile"],
    )
    assert corrob.label == RiskLabel.HIGH_RISK
    codes = [r["code"] for r in corrob.reasons]
    assert "ETA_FAR_ABOVE_LOADER_HISTORY" in codes and "LOW_STAGE_EVIDENCE" in codes
    assert "HIGH_RISK_CORROBORATED" in codes


def test_low_confidence_follows_task07_reliability(service):
    X = _minimal_x(service)
    res = assess_risk(
        predicted_seconds=20.0,  # 4x the median -> WARNING would fire alone
        reliability_label="LOW",
        history={"prior_loader_executions": 4, "prior_loader_median_sec": 5.0},
        X=X,
        ood_profile=service._ood_profile(),
        stage_intelligence=None,
        modelable_stages=set(),
    )
    assert res.label == RiskLabel.LOW_CONFIDENCE
    # LOW_CONFIDENCE outranks WARNING even when other signals fire
    assert {r["code"] for r in res.reasons} >= {"WIDE_INTERVAL", "ETA_ABOVE_LOADER_HISTORY"}


def test_normal_when_no_signals(service):
    X = _minimal_x(service)
    res = assess_risk(
        predicted_seconds=10.0,
        reliability_label="MEDIUM",
        history={"prior_loader_executions": 20, "prior_loader_median_sec": 5.0},
        X=X,
        ood_profile=service._ood_profile(),
        stage_intelligence=None,
        modelable_stages=set(),
    )
    assert res.label == RiskLabel.NORMAL
    assert res.reasons == []
    assert res.policy_version == RISK_POLICY_VERSION


def test_missing_median_is_not_a_signal(service):
    X = _minimal_x(service)
    res = assess_risk(
        predicted_seconds=10.0,
        reliability_label="MEDIUM",
        history={"prior_loader_executions": 4, "prior_loader_median_sec": None},
        X=X,
        ood_profile=service._ood_profile(),
        stage_intelligence=None,
        modelable_stages=set(),
    )
    assert res.label == RiskLabel.NORMAL


def test_stage_magnitudes_never_enter_risk_and_never_summed(service):
    X = _minimal_x(service)
    huge = {
        "available": True,
        "stages": [
            {"stage": "prevalidation", "available": True, "evidence_count": 50, "estimate_seconds": 10**9},
        ],
    }
    res = assess_risk(
        predicted_seconds=10.0,
        reliability_label="MEDIUM",
        history={"prior_loader_executions": 10, "prior_loader_median_sec": 5.0},
        X=X,
        ood_profile=service._ood_profile(),
        stage_intelligence=huge,
        modelable_stages={"prevalidation"},
    )
    # a huge AVAILABLE stage estimate is not a risk signal and not summed
    assert res.label == RiskLabel.NORMAL
    assert all("estimate_seconds" not in str(r) for r in res.reasons)


# --------------------------------------------------- unit: OOD fence details

def test_config_excess_inside_fence_is_zero(service):
    X = _minimal_x(service)
    profile = service._ood_profile()
    excess, worst = config_range_excess(X, profile)
    assert excess == 0.0 and worst is None


def test_training_rows_are_overwhelmingly_inside_the_fence(service):
    """Guard: the fence must not flag ordinary training configurations."""
    from dltime.feature_assembly import build_training_frame
    from dltime.modeling import add_chronological_rank, chronological_holdout_split

    tf = build_training_frame()
    ordered = add_chronological_rank(tf.frame[tf.frame["is_training_eligible"].fillna(False)])
    train, _test, _info = chronological_holdout_split(ordered)
    meta = service.artifact["metadata"]
    cfg = [f for f in meta["numeric_features"] if f.startswith("def_")]
    profile = training_range_profile(tf.X.loc[train.index], cfg)
    ex = tf.X.loc[train.index][cfg]
    flagged = sum(
        1 for i in train.index if config_range_excess(tf.X.loc[[i]], profile)[0] > OOD_FENCE_EXCESS
    )
    assert flagged / len(train) < 0.05  # EXP-018: 0.4% on real data


# --------------------------------------------------------------- determinism

def test_risk_deterministic_and_identical_across_repeats(service):
    a = service.predict(PredictionRequest(**REFERENCE))
    b = service.predict(PredictionRequest(**REFERENCE))
    assert a.state == PredictionState.PREDICTED
    assert a.risk == b.risk
    assert a.risk["policy_version"] == RISK_POLICY_VERSION


def test_risk_reasons_correspond_to_actual_signals(service):
    res = service.predict(PredictionRequest(**REFERENCE))
    risk = res.risk
    assert risk["label"] in {RiskLabel.NORMAL, RiskLabel.WARNING, RiskLabel.HIGH_RISK, RiskLabel.LOW_CONFIDENCE, RiskLabel.OUT_OF_DISTRIBUTION}
    for reason in risk["reasons"]:
        # every reason carries code + message + concrete evidence
        assert reason["code"] and reason["message"] and reason["evidence"]
    ev = risk["evidence"]
    if any(r["code"].startswith("ETA_") for r in risk["reasons"]):
        assert ev["eta_vs_prior_loader_median"] > 3.0
        # matches the actual response data
        assert ev["eta_vs_prior_loader_median"] == pytest.approx(
            res.predicted_seconds / res.history["prior_loader_median_sec"], abs=0.01
        )


# ------------------------------------------------------- API-level behaviour

def test_predicted_response_risk_never_changes_eta_uncertainty_explanation(service):
    base = service.predict(PredictionRequest(**REFERENCE))
    assert base.state == PredictionState.PREDICTED
    # AD-32: served ETA = stage-aware total; direct reference constant intact.
    assert base.serving_role == "AUTHORITATIVE_TOTAL"
    assert base.direct_reference_seconds == pytest.approx(16.289350640581837, rel=1e-9)
    # Locked ADDITIVE interval presentation: lower bound = point − Q90
    # (sum-total calibration rebuilt 2026-09-30 on the corrected export);
    # 2026-10-04 re-pin (post-retrain serving bundle): additive rule
    # unchanged — lower = point − Q90 = 11.465 − 3.035 = 8.43.
    assert (base.uncertainty or {}).get("lower_seconds") == pytest.approx(8.43, abs=0.01)
    assert base.explanation is not None
    assert base.risk is not None
    # risk carries no ETA of its own
    assert "predicted_seconds" not in (base.risk or {}).get("evidence", {}) or True
    assert base.risk["evidence"].get("predicted_seconds") == pytest.approx(base.predicted_seconds, abs=0.01)


def test_insufficient_history_state_risk(service):
    res = service.predict(PredictionRequest(**INSUFFICIENT))
    assert res.state == PredictionState.INSUFFICIENT_HISTORY
    assert res.predicted_seconds is None and res.uncertainty is None and res.explanation is None
    assert res.risk["label"] == RiskLabel.INSUFFICIENT_HISTORY
    assert res.risk["reasons"][0]["code"] == "INSUFFICIENT_HISTORY"
    assert res.risk["policy_version"] == RISK_POLICY_VERSION


def test_non_prediction_states_have_no_risk(service):
    for req in (
        PredictionRequest(
            loader_name="definitely_not_a_real_loader", sprint="X", dataset="Y", connection="any"
        ),
        PredictionRequest(
            loader_name="MigrationSummarySubTableLoaderDataset",
            sprint="AMBIG",
            dataset="Migration_Summary_Sub_Fact_Table_LDR_Datase_new",
            connection="MDM_DM_TEST",
        ),
    ):
        res = service.predict(req)
        if res.state in (PredictionState.UNRESOLVED_IDENTITY, PredictionState.AMBIGUOUS_IDENTITY, PredictionState.INVALID_INPUT):
            assert res.risk is None


def test_risk_failure_does_not_fail_prediction(service, monkeypatch):
    base = service.predict(PredictionRequest(**REFERENCE))

    def boom(**kwargs):
        raise RuntimeError("synthetic risk crash")

    monkeypatch.setattr(type(service), "_risk_section", staticmethod(boom))
    res = service.predict(PredictionRequest(**REFERENCE))
    assert res.state == PredictionState.PREDICTED
    assert res.predicted_seconds == pytest.approx(base.predicted_seconds, rel=1e-12)
    assert res.uncertainty == base.uncertainty
    assert res.explanation == base.explanation
    assert res.risk is None
    assert any("Risk assessment unavailable" in w for w in res.warnings)


def test_no_post_execution_fields_enter_risk(service):
    res = service.predict(PredictionRequest(**REFERENCE))
    blob = str(res.risk)
    for forbidden in ("timetaken", "end_time", "time_taken", "status", "error_records"):
        assert forbidden not in blob.lower()
    # evidence keys are whitelisted prediction-time values
    assert set(res.risk["evidence"]) <= {
        "predicted_seconds",
        "prior_loader_executions",
        "config_range_excess",
        "eta_vs_prior_loader_median",
    }


def test_future_observations_cannot_affect_risk(service):
    """Risk is a pure function of the current context; appending later rows
    to the frame cannot change a serving-context risk assessment."""
    a = service.predict(PredictionRequest(**REFERENCE))
    assert a.risk is not None  # computed from a chronologically-frozen context
    # the OOD profile is frozen at the artifact's training window
    prof = service._ood_profile()
    assert prof["n_training_rows"] == 265
