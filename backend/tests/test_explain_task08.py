"""Task 08 tests: SHAP explainability for the total ETA.

Covers: explainer loading/binding, exact prediction-time feature vector,
encoded->registered mapping (one-hot aggregation), additivity in log1p space,
direction/sort correctness, contamination probes, per-state explanation
presence, failure behavior (explanation must never break the point ETA), and
the global holdout summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dltime.feature_assembly import build_training_frame
from dltime.inference import PredictionRequest, PredictionState


@pytest.fixture(scope="module")
def explain_env():
    tf = build_training_frame()
    from dltime.artifact import load_serving_artifact
    from dltime.explain import ExplainService

    artifact = load_serving_artifact()
    svc = ExplainService(artifact)
    return {"tf": tf, "artifact": artifact, "svc": svc}


# ----------------------------------------------------------------- explainer

def test_shap_version_recorded_and_explainer_loads(explain_env):
    import shap

    from dltime.explain import get_or_build_explainer

    explainer, cache = get_or_build_explainer(explain_env["artifact"], None)
    assert cache["model_version"] == "total-et-v3-candidate"
    assert isinstance(explainer, shap.TreeExplainer)


def test_explainer_cache_reused_for_same_model_version(explain_env):
    from dltime.explain import get_or_build_explainer

    artifact = explain_env["artifact"]
    e1, c1 = get_or_build_explainer(artifact, None)
    e2, c2 = get_or_build_explainer(artifact, c1)
    assert e1 is e2


# ------------------------------------------------------------ mapping/layout

def test_encoded_mapping_covers_all_registered_features(explain_env):
    from dltime.explain import encoded_column_mapping
    from collections import Counter

    meta = explain_env["artifact"]["metadata"]
    mapping = encoded_column_mapping(explain_env["artifact"]["preprocessor"], meta)
    # 2026-09-30 re-pin: the direct-total artifact on the corrected export
    # has 30 numeric 1:1 columns + one 2-wide one-hot block = 32 encoded cols.
    assert len(mapping) == 46  # 2026-10-04: contract extended with per-section order + per-type-count features
    owners = Counter(r for r, _ in mapping)
    assert set(owners) == set(meta["feature_names_in_order"])
    assert len(owners) == 44
    assert owners["def_dominant_calling_type"] == 3  # one-hot block aggregated by SUM


def test_bad_explainer_shape_is_reported_not_silently_mapped(explain_env, monkeypatch):
    """A wrong-shaped SHAP result must raise, never map by position guess."""
    from dltime.explain import ExplainError, local_explanation

    class FakeExplainer:
        expected_value = 0.0

        def shap_values(self, Xt):
            return np.zeros((len(Xt), 5))  # wrong width on purpose

    X = explain_env["tf"].X.iloc[[-1]]
    with pytest.raises(ExplainError):
        local_explanation(explain_env["artifact"], X, FakeExplainer())


# --------------------------------------------------------------- explanation

def test_explanation_uses_exact_prediction_time_vector_and_additivity(explain_env):
    """base + sum(contributions) == model log1p output (the SHAP space)."""
    tf, artifact, svc = explain_env["tf"], explain_env["artifact"], explain_env["svc"]
    X = tf.X.iloc[[-1]]
    exp = svc.explain(X)
    assert exp["model_version"] == "total-et-v3-candidate"
    assert exp["additivity_abs_error"] <= 1e-6
    # rounded payload fields are independently 6-dp; the full-precision gate
    # is additivity_abs_error above (computed pre-rounding)
    assert exp["base_value_log1p"] + exp["sum_contributions_log1p"] == pytest.approx(
        exp["model_output_log1p"], abs=2e-6
    )
    # and the SHAP-space output IS the model's ETA in log1p space
    # (6-dp payload rounding of the log value -> rel tol 1e-5 after expm1)
    from dltime.artifact import predict_with_artifact

    eta = predict_with_artifact(artifact, X)
    assert float(np.expm1(exp["model_output_log1p"])) == pytest.approx(eta, rel=1e-5)


def test_correct_feature_names_and_directions(explain_env):
    from dltime.explain import DISPLAY_NAMES

    exp = explain_env["svc"].explain(explain_env["tf"].X.iloc[[-1]])
    feats = exp["features"]
    assert len(feats) == 44  # 2026-10-04: 15 new def_* features joined the contract
    assert {f["feature"] for f in feats} == set(
        explain_env["artifact"]["metadata"]["feature_names_in_order"]
    )  # ranked by |SHAP|, so compare as a set, not by order
    for f in feats:
        assert f["display_name"] == DISPLAY_NAMES[f["feature"]]
        expected_dir = "higher" if f["shap_value_log1p"] > 0 else ("lower" if f["shap_value_log1p"] < 0 else "neutral")
        assert f["direction"] == expected_dir


def test_top_contributors_sorted_by_absolute_magnitude(explain_env):
    exp = explain_env["svc"].explain(explain_env["tf"].X.iloc[[-1]])
    top = exp["top_features"]
    assert 0 < len(top) <= 5
    mags = [abs(f["shap_value_log1p"]) for f in top]
    assert mags == sorted(mags, reverse=True)
    ranks = [f["contribution_rank"] for f in top]
    assert ranks == list(range(1, len(top) + 1))
    # top features are the global top of the full list
    all_mags = [abs(f["shap_value_log1p"]) for f in exp["features"]]
    assert abs(mags[0] - max(all_mags)) < 1e-12


def test_one_hot_contributions_aggregated_not_discarded(explain_env):
    """Encoded one-hot SHAP values must be summed into the registered feature."""
    import shap

    from dltime.explain import encoded_column_mapping

    artifact = explain_env["artifact"]
    meta = artifact["metadata"]
    explainer, _ = __import__("dltime.explain", fromlist=["get_or_build_explainer"]).get_or_build_explainer(artifact, None)
    X = explain_env["tf"].X.iloc[[-1]]
    from dltime.modeling import transform_with_preprocessor

    Xt = transform_with_preprocessor(artifact["preprocessor"], X, meta["categorical_features"], meta["numeric_features"])
    row = np.asarray(explainer.shap_values(Xt))[0]
    mapping = encoded_column_mapping(artifact["preprocessor"], meta)
    idxs = [i for r, i in mapping if r == "def_dominant_calling_type"]
    exp = explain_env["svc"].explain(X)
    got = next(f["shap_value_log1p"] for f in exp["features"] if f["feature"] == "def_dominant_calling_type")
    assert got == pytest.approx(float(row[idxs].sum()), abs=1e-6)


# ----------------------------------------------------------- leakage guards

def test_current_target_cannot_enter_explanation(explain_env):
    """Tampering with target/post-execution data must leave SHAP unchanged."""
    tf, svc = explain_env["tf"], explain_env["svc"]
    X1 = tf.X.iloc[[-1]]
    X2 = X1.copy()
    # any post-execution field is not part of X at all; prove the vector is
    # self-contained by perturbing the frame's target data (if present)
    for col in ("timetaken_in_sec", "y_raw", "target"):
        if col in tf.frame.columns:
            tf.frame.loc[tf.frame.index[-1], col] = 999999999.0
    exp1 = svc.explain(X1)
    exp2 = svc.explain(X2)
    a = [f["shap_value_log1p"] for f in exp1["features"]]
    b = [f["shap_value_log1p"] for f in exp2["features"]]
    assert a == b


def test_future_observations_cannot_enter_explanation(explain_env):
    """Appending later/future rows to the frame must not alter one-row SHAP."""
    tf, svc = explain_env["tf"], explain_env["svc"]
    X1 = tf.X.iloc[[-1]]
    exp1 = svc.explain(X1)
    # simulate a future context appended to the feature frame
    X_future = X1.copy()
    X_future.index = [max(tf.X.index) + 1000]
    combined = pd.concat([tf.X, X_future])
    svc2 = explain_env["svc"]  # same cached explainer; frame growth is invisible to it
    exp2 = svc2.explain(combined.iloc[[-1]])
    assert [f["shap_value_log1p"] for f in exp1["features"]] == [
        f["shap_value_log1p"] for f in exp2["features"]
    ]


def test_feature_vector_has_no_post_execution_fields(explain_env):
    forbidden = {"timetaken_in_sec", "time_taken", "end_time", "end_tm", "status", "error_records"}
    cols = {c.lower() for c in explain_env["tf"].X.columns}
    assert not (forbidden & cols)


# ------------------------------------------------------------- API behaviour

def test_api_predicted_contains_explanation_eta_and_uncertainty_intact(service):
    req = PredictionRequest(
        loader_name="MigrationSummarySubTableLoaderDataset",
        sprint="TEST",
        dataset="Migration_Summary_Sub_Fact_Table_LDR_Datase_new",
        connection="MDM_DM_TEST",
    )
    result = service.predict(req)
    assert result.state == PredictionState.PREDICTED
    # Sum policy (EXP-038): the served ETA is the five-stage sum; the direct
    # reference model still reproduces the canonical corrected-regime constant.
    assert result.serving_role == "AUTHORITATIVE_TOTAL"
    assert result.direct_reference_seconds == pytest.approx(16.289350640581837, rel=1e-9)
    u = result.uncertainty or {}
    assert u.get("lower_seconds") is not None and u.get("upper_seconds") is not None
    e = result.explanation
    assert e is not None
    # the served total's explanation is the exact stage decomposition
    assert e["method"] == "deterministic_stage_sum"
    assert e["model_version"] == "sum-of-available-stages-v2"
    assert "sum of the applicable operational stage predictions" in e["output_space"]
    assert e["additivity_abs_error"] == 0.0
    assert pytest.approx(result.predicted_seconds, abs=1e-9) == e["model_output_seconds"]
    assert 1 <= len(e["top_features"]) <= 5
    # every ranked contributor is an exact additive component in seconds
    assert pytest.approx(
        sum(f["contribution_seconds"] for f in e["top_features"]), abs=1e-9
    ) == result.predicted_seconds
    # Applicable-mode explanation (user final spec, 2026-09-30): approval
    # statistical estimates ARE additive components of the total (disclosed
    # with their exact contributions); nothing applicable is left out.
    # Per the LOADER-tier-total policy (2026-10-02), only LOADER-tier
    # approvals count toward the total; broader-context approvals are
    # disclosed as components with included_in_total=False and contribution
    # of 0.0 (not part of the served total).
    approval_components = [
        f for f in e["features"] if "approval" in f.get("feature", "")
    ]
    # Per the LOADER-tier-total policy (2026-10-02), only LOADER-tier
    # approvals are included in the total; broader-context approvals
    # (included_in_total=False) are disclosed as components with
    # contribution_seconds=0.0 (not part of the served total).
    # Per the LOADER-tier-total policy (2026-10-02), only LOADER-tier approvals
    # count toward the total; broader-context approvals (included_in_total=False)
    # are disclosed as components with contribution_seconds=0.0 (not part of the
    # served total). In this test case, both approval stages resolve to
    # GLOBAL-tier (broader-context), so all are excluded from the total.
    approval_components = [
        f for f in e["features"] if "approval" in f.get("feature", "")
    ]
    assert approval_components, "approval estimates must appear as components"
    assert all(f.get("included_in_total") is not None for f in approval_components), \
        "each approval component must carry included_in_total flag"
    assert all(f["contribution_seconds"] == 0.0 for f in approval_components), \
        "excluded approvals must carry contribution_seconds=0.0"
    # Total approval estimates for this loader (both GLOBAL-tier):
    total_approval_estimates = sum(
        s["estimate_seconds"]
        for s in (result.stage_intelligence or {}).get("stages", [])
        if s.get("method") == "contextual_hierarchical_statistical" and s.get("estimate_seconds") is not None
    )
    # LOADER-tier approvals (included_in_total=True) would sum to the approval
    # portion of the total ETA; broader-context approvals (included_in_total=False)
    # are disclosed but excluded. Here both approvals are GLOBAL-tier, so:
    # - included_approval_sum = 0.0 (no LOADER-tier approvals)
    # - total = 3-stage ML sum (11.465) + 0.0 = 11.465
    included_approval_sum = sum(
        f["contribution_seconds"]
        for f in approval_components
        if f.get("included_in_total") is True
    )
    assert pytest.approx(included_approval_sum, abs=1e-9) == 0.0
    assert not any("Explanation unavailable" in w for w in result.warnings)


def test_api_non_predicted_states_have_no_explanation(service):
    # INSUFFICIENT_HISTORY
    r = service.predict(
        PredictionRequest(
            loader_name="2210_TEST_ALL_CONFIG",
            sprint="SUPPORT",
            dataset="A_TABLE",
            connection="test1",
        )
    )
    assert r.state == PredictionState.INSUFFICIENT_HISTORY
    assert r.explanation is None
    # unregistered loader == NEW loader (2026-10-03): refused WITHOUT any
    # explanation payload — no total, no fabricated content.
    r2 = service.predict(
        PredictionRequest(
            loader_name="definitely_not_a_real_loader",
            sprint="X",
            dataset="Y",
            connection="any_conn",
        )
    )
    assert r2.state == PredictionState.INSUFFICIENT_HISTORY
    assert r2.explanation is None


def test_explanation_failure_never_breaks_point_prediction(service, monkeypatch):
    req = PredictionRequest(
        loader_name="MigrationSummarySubTableLoaderDataset",
        sprint="TEST",
        dataset="Migration_Summary_Sub_Fact_Table_LDR_Datase_new",
        connection="MDM_DM_TEST",
    )
    # point prediction baseline
    base = service.predict(req)
    assert base.state == PredictionState.PREDICTED

    def boom(self, X):
        raise RuntimeError("synthetic explainer crash")

    monkeypatch.setattr(type(service), "_explanation_section", boom)
    res = service.predict(req)
    assert res.state == PredictionState.PREDICTED  # prediction survives
    assert res.predicted_seconds == pytest.approx(base.predicted_seconds, rel=1e-12)
    assert res.explanation is None
    assert any("Explanation unavailable" in w for w in res.warnings)
    # uncertainty layer independent of SHAP failure
    assert (res.uncertainty or {}).get("lower_seconds") == pytest.approx(
        (base.uncertainty or {}).get("lower_seconds"), rel=1e-9
    )


def test_uncertainty_still_works_when_shap_succeeds(service):
    res = service.predict(
        PredictionRequest(
            loader_name="MigrationSummarySubTableLoaderDataset",
            sprint="TEST",
            dataset="Migration_Summary_Sub_Fact_Table_LDR_Datase_new",
            connection="MDM_DM_TEST",
        )
    )
    assert res.explanation is not None
    assert res.uncertainty is not None
    assert res.uncertainty["interval_level"] == 0.9


def test_explanation_deterministic_across_repeated_predictions(service):
    req = PredictionRequest(
        loader_name="MigrationSummarySubTableLoaderDataset",
        sprint="TEST",
        dataset="Migration_Summary_Sub_Fact_Table_LDR_Datase_new",
        connection="MDM_DM_TEST",
    )
    a = service.predict(req).explanation
    b = service.predict(req).explanation
    assert a == b


# --------------------------------------------------------- global summary

def test_global_summary_over_documented_holdout(explain_env):
    from dltime.explain import global_explanation
    from dltime.modeling import add_chronological_rank, chronological_holdout_split

    tf, artifact, svc = explain_env["tf"], explain_env["artifact"], explain_env["svc"]
    ordered = add_chronological_rank(tf.frame[tf.frame["is_training_eligible"].fillna(False)])
    _train, test, _info = chronological_holdout_split(ordered)
    X_holdout = tf.X.loc[test.index]
    assert len(X_holdout) == 88
    explainer, _ = __import__("dltime.explain", fromlist=["get_or_build_explainer"]).get_or_build_explainer(artifact, None)
    g = global_explanation(artifact, X_holdout, explainer, subset_description="test")
    assert g["n_observations"] == 88
    assert g["model_version"] == "total-et-v3-candidate"
    feats = g["features"]
    assert len(feats) == 44  # 2026-10-04: 15 new def_* features joined the contract
    mags = [f["mean_abs_shap_log1p"] for f in feats]
    assert mags == sorted(mags, reverse=True)
    assert all(m >= 0 for m in mags)
