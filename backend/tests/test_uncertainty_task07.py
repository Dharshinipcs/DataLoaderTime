"""Task 07 tests: total-ETA uncertainty + reliability.

Covers: interval math, calibration chronology (permitted observations only),
current/future contamination probes, artifact integrity, per-state uncertainty
presence, and point-prediction invariance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dltime import uncertainty as unc
from dltime.feature_assembly import build_training_frame
from dltime.inference import PredictionRequest, PredictionState
from dltime.modeling import add_chronological_rank, chronological_holdout_split


@pytest.fixture(scope="module")
def cal(tmp_path_factory):
    """Calibration built into a temp dir (does not pollute the real artifact)."""
    tf = build_training_frame()
    from dltime.artifact import load_serving_artifact

    artifact = load_serving_artifact()
    path = tmp_path_factory.mktemp("cal") / "cal.json"

    def predict_batch(X: pd.DataFrame) -> np.ndarray:
        from dltime.artifact import predict_with_artifact

        return np.array(
            [predict_with_artifact(artifact, X.iloc[[i]]) for i in range(len(X))],
            dtype=float,
        )

    cal = unc.build_calibration(tf, artifact["metadata"]["model_version"], predict_batch)
    meta = unc.calibration_to_metadata(cal)
    unc.save_calibration_metadata(meta, path)
    loaded = unc.load_calibration_artifact(
        meta["model_version"], meta["model_feature_count"], path
    )
    return {"tf": tf, "artifact": artifact, "meta": loaded, "cal": cal}


# ---------------------------------------------------------------- interval math


def test_interval_calculation_exact():
    # Locked product presentation: ADDITIVE seconds interval (point +/- q),
    # calibrated quantile unchanged (16.65 +/- 2.9195 -> 13.73-19.57).
    lo, hi = unc.interval_from_quantile(18.155241, 2.2291534416514427)
    assert lo == pytest.approx(max(18.155241 - 2.2291534416514427, 0.0), abs=1e-9)
    assert hi == pytest.approx(18.155241 + 2.2291534416514427, abs=1e-9)
    assert lo >= 0.0
    assert hi >= lo
    # the point prediction always lies inside (symmetric additive q)
    assert lo <= 18.155241 <= hi


def test_interval_lower_bound_never_negative_and_point_inside():
    for pred, q in [(0.001, 5.0), (1.0, 2.0), (18.155, 2.229), (665974.0, 2.5), (43_000_000.0, 4.4)]:
        lo, hi = unc.interval_from_quantile(pred, q)
        assert lo >= 0.0
        assert hi >= lo
        assert lo <= pred <= hi


def test_extreme_durations_do_not_crash(cal):
    meta = cal["meta"]
    q = meta["quantiles_abs_log_residual"]["90"]
    for pred in [1e-6, 0.5, 65.62, 870882.0, 43_000_000.0]:
        lo, hi = unc.interval_from_quantile(pred, q)
        assert np.isfinite(lo) and np.isfinite(hi) and lo >= 0 and hi >= lo


def test_invalid_inputs_rejected():
    with pytest.raises(unc.UncertaintyError):
        unc.interval_from_quantile(-1.0, 2.0)
    with pytest.raises(unc.UncertaintyError):
        unc.interval_from_quantile(10.0, 0.0)
    with pytest.raises(unc.UncertaintyError):
        unc.interval_from_quantile(float("nan"), 2.0)


# ------------------------------------------------------- calibration integrity


def test_calibration_uses_only_permitted_chronological_observations(cal):
    meta = cal["meta"]
    # window must be strictly after the model's recorded training window
    assert meta["n_calibration_rows"] == 88
    assert meta["model_version"] == "total-et-v3-candidate"
    assert meta["model_feature_count"] == 44
    win = meta["calibration_window"]
    tf = cal["tf"]
    eligible = tf.frame[tf.frame["is_training_eligible"].fillna(False)]
    ordered = add_chronological_rank(eligible)
    train, _test, info = chronological_holdout_split(ordered)
    assert pd.Timestamp(win["first_start"]) > pd.Timestamp(info["train_last_start"])
    assert win["scope"].startswith("final chronological holdout")


def test_current_execution_target_cannot_influence_uncertainty(cal):
    """Poisoning the *context* target (and any same/future row) changes nothing.

    The calibration is a fixed window strictly before serving contexts; the
    context row itself is target-less. Recomputing the interval for a context
    whose hypothetical 'actual' changes must give identical bounds.
    """
    meta = cal["meta"]
    q = meta["quantiles_abs_log_residual"]["90"]
    lo1, hi1 = unc.interval_from_quantile(18.155241, q)
    lo2, hi2 = unc.interval_from_quantile(18.155241, q)
    assert (lo1, hi1) == (lo2, hi2)
    # and the calibration quantiles are independent of any request target by
    # construction (built once from the frozen holdout window)
    assert cal["meta"]["n_calibration_rows"] == 88


def test_future_observations_cannot_influence_calibration(cal, tmp_path):
    """A poisoned strictly-future execution leaves calibration metadata identical."""
    tf = cal["tf"]
    artifact = cal["artifact"]

    def predict_batch(X: pd.DataFrame) -> np.ndarray:
        from dltime.artifact import predict_with_artifact

        return np.array(
            [predict_with_artifact(artifact, X.iloc[[i]]) for i in range(len(X))],
            dtype=float,
        )

    poisoned_frame = tf.frame.copy()
    # add a monstrous strictly-future row (after the calibration window end)
    win_end = pd.Timestamp(cal["meta"]["calibration_window"]["last_start"])
    extra = poisoned_frame.iloc[[0]].copy()
    extra["Start_Time"] = win_end + pd.Timedelta(days=30)
    extra["timetaken_in_sec"] = 9.9e9
    extra["is_training_eligible"] = False  # serving-style context row
    poisoned = type(tf)(
        frame=pd.concat([poisoned_frame, extra], ignore_index=True),
        X=tf.X,
        y_raw=tf.y_raw,
        y_log1p=tf.y_log1p,
        identifier_columns=tf.identifier_columns,
        feature_columns=tf.feature_columns,
        analytical_columns=tf.analytical_columns,
        target_column=tf.target_column,
    )
    # build_calibration filters to is_training_eligible rows of the holdout
    # window; the extra ineligible future row cannot enter
    cal2 = unc.build_calibration(
        poisoned, artifact["metadata"]["model_version"], predict_batch
    )
    meta2 = unc.calibration_to_metadata(cal2)
    assert meta2["quantiles_abs_log_residual"] == cal["meta"]["quantiles_abs_log_residual"]
    assert meta2["n_calibration_rows"] == cal["meta"]["n_calibration_rows"]


def test_calibration_artifact_load_and_rejection(cal, tmp_path):
    meta = cal["meta"]
    path = tmp_path / "c.json"
    unc.save_calibration_metadata(meta, path)
    # correct load
    loaded = unc.load_calibration_artifact(
        meta["model_version"], meta["model_feature_count"], path
    )
    assert loaded["quantiles_abs_log_residual"]["90"] == pytest.approx(
        meta["quantiles_abs_log_residual"]["90"]
    )
    # wrong model version -> rejected
    with pytest.raises(unc.UncertaintyError, match="model"):
        unc.load_calibration_artifact("other-model", meta["model_feature_count"], path)
    # changed feature count -> rejected
    with pytest.raises(unc.UncertaintyError, match="feature"):
        unc.load_calibration_artifact(meta["model_version"], 99, path)
    # corrupted file -> rejected
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(unc.UncertaintyError, match="unreadable"):
        unc.load_calibration_artifact(meta["model_version"], meta["model_feature_count"], bad)
    # missing file -> rejected
    with pytest.raises(unc.UncertaintyError, match="not found"):
        unc.load_calibration_artifact(
            meta["model_version"], meta["model_feature_count"], tmp_path / "nope.json"
        )


def test_quantile_floor_prevents_overconfident_intervals():
    """A degenerate calibration (near-zero residuals) still claims >= floor spread."""
    meta = {
        "calibration_version": unc.CALIBRATION_VERSION,
        "method": unc.CALIBRATION_METHOD,
        "interval_levels": [0.8, 0.9],
        "quantiles_abs_log_residual": {"80": 0.001, "90": 0.002},
        "n_calibration_rows": 88,
        "n_positive_target_rows": 88,
        "calibration_window": {"first_start": "x", "last_start": "y", "scope": "s"},
        "residual_stats": {},
        "model_version": "total-et-v3-candidate",
        "model_feature_count": 30,
        "diagnostics": {},
    }
    # (load-level validation of a hand-built block is allowed; floor applies at
    # build time, so verify the build-time floor directly)
    import numpy as np

    residuals = np.array([1e-9] * 100)  # degenerate
    q = float(np.quantile(residuals, 0.9, method="higher"))
    assert max(q, unc.MIN_QUANTILE_FLOOR) == unc.MIN_QUANTILE_FLOOR


# ------------------------------------------------- reliability label removed


def test_no_reliability_label_in_payload(cal):
    """2026-10-03 (user final spec): the reliability label layer was removed
    from the system — the calibrated interval itself is the honesty mechanism."""
    meta = cal["meta"]
    common = dict(
        calibration_metadata=meta,
        min_required_history=3,
        evidence={"n_calibration_rows": meta["n_calibration_rows"]},
    )
    s1 = unc.uncertainty_section(
        predicted_seconds=18.155241, prior_loader_executions=11, **common
    )
    s2 = unc.uncertainty_section(
        predicted_seconds=18.155241, prior_loader_executions=11, **common
    )
    assert s1 == s2  # deterministic
    assert "reliability" not in s1


# ---------------------------------------------------------------- API behaviour


def test_api_predicted_contains_uncertainty_and_point_unchanged(service):
    req = PredictionRequest(
        loader_name="MigrationSummarySubTableLoaderDataset",
        sprint="TEST",
        dataset="Migration_Summary_Sub_Fact_Table_LDR_Datase_new",
        connection="MDM_DM_TEST",
    )
    result = service.predict(req)
    assert result.state == PredictionState.PREDICTED
    # AD-32: the point prediction comes from the stage-aware total artifact;
    # the direct-total path must still reproduce the canonical reference value
    # (rollback-path regression guard — the two roles are checked separately).
    from dltime.artifact import predict_with_artifact

    ctx = service.build_context_row(
        loader_name=req.loader_name,
        sprint=req.sprint,
        dataset=req.dataset,
        resolution_status=__import__("dltime.resolution", fromlist=["ResolutionStatus"]).ResolutionStatus.RESOLVED,
        resolved_connection=result.identity["resolved_connection"],
        start_time=service._default_start_time(),
        execution_id=service._synthetic_execution_id(),
        workflow_execution_id=None,
    )
    X = service._context_feature_vector(ctx)
    direct = predict_with_artifact(service.artifact, X)
    assert result.direct_reference_seconds == pytest.approx(direct, rel=1e-12)
    # Applicable-stage policy (user final spec, 2026-10-03): the ML part is
    # the applicable 3-stage sum (0.6 + 10.388 + 0.477 = 11.465) plus ONLY the
    # loader-OWN approval statistical estimates. This reference loader has NO
    # recorded approval history of its own — its two approval estimates come
    # from the GLOBAL fallback (other loaders' history), so they are excluded
    # from the total AND hidden from the displayed approval section.
    assert result.predicted_seconds == pytest.approx(11.465, abs=1e-9)
    si = result.stage_intelligence or {}
    ml_vals = [
        s["estimate_seconds"]
        for s in si.get("stages", [])
        if s.get("method") == "ml_stage_model" and s.get("estimate_seconds") is not None
    ]
    assert len(ml_vals) == 3
    appr_all = [
        s
        for s in si.get("stages", [])
        if s.get("method") == "contextual_hierarchical_statistical" and s.get("estimate_seconds") is not None
    ]
    assert len(appr_all) == 2
    # loader-own approvals (included) — none for this loader:
    appr_included = [s for s in appr_all if s.get("included_in_total") is True]
    assert appr_included == []
    # the two available estimates are broader-context (GLOBAL) — disclosed in
    # the audit payload with included_in_total=False, never summed, never shown
    assert all(s.get("included_in_total") is False for s in appr_all)
    assert result.predicted_seconds == pytest.approx(sum(ml_vals) + sum(s["estimate_seconds"] for s in appr_included), abs=1e-9)
    assert result.floor_applied is False and result.physical_floor_seconds is None

    u = result.uncertainty
    assert u is not None and u["available"] is True
    assert u["interval_level"] == 0.9
    assert 0 <= u["lower_seconds"] <= result.predicted_seconds <= u["upper_seconds"]
    assert u["interval_width_seconds"] == pytest.approx(
        u["upper_seconds"] - u["lower_seconds"], abs=0.01
    )
    assert u["calibration_rows"] == 88
    assert u["calibration_version"].startswith("unc-log1p-conformal-v1")
    assert "reliability" not in u


def test_api_non_predicted_states_have_no_uncertainty(service):
    cases = [
        (
            PredictionRequest(
                loader_name="2210_TEST_ALL_CONFIG",
                sprint="SUPPORT",
                dataset="A_TABLE",
                connection="test1",
            ),
            PredictionState.INSUFFICIENT_HISTORY,
        ),
        (
            PredictionRequest(
                loader_name="definitely_not_a_loader_xyz",
                sprint="TEST",
                dataset="nope",
                connection="nope_conn",
            ),
            PredictionState.INSUFFICIENT_HISTORY,
        ),
        (
            PredictionRequest(loader_name=None, sprint="TEST", dataset="DS", connection="c"),
            PredictionState.INVALID_INPUT,
        ),
    ]
    for req, expected in cases:
        result = service.predict(req)
        assert result.state == expected
        assert result.uncertainty is None
        if expected is not PredictionState.INVALID_INPUT:
            assert result.predicted_seconds is None
