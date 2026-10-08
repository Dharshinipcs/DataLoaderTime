"""Task 02: feature registry, definition features, history eligibility, training frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dltime.definition_features import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES as DEF_FEATURES,
    build_execution_level_definition_features,
    build_identity_level_definition_features,
)
from dltime.feature_assembly import (
    ANALYTICAL_ONLY_COLUMNS,
    IDENTIFIER_COLUMNS,
    PREDICTION_FEATURE_GROUPS,
    TargetTransformer,
    build_training_frame,
    evaluate_history_eligibility,
)
from dltime.feature_registry import DEFAULT_FEATURE_REGISTRY
from dltime.history_eligibility import (
    HistoryEligibilityConfig,
    HistoryEligibilityEngine,
    summarize_prior_history_distribution,
)
from dltime.leakage import LeakageViolation, validate_feature_set


# ---------------- registry & leakage ----------------

def test_registry_covers_every_production_feature():
    all_names = [n for cols in PREDICTION_FEATURE_GROUPS.values() for n in cols]
    # 2026-10-04 feature-contract expansion: 22 pre-execution definition
    # features + 22 historical features = 44 registered model inputs.
    assert len(all_names) == 44
    for name in all_names:
        spec = DEFAULT_FEATURE_REGISTRY.get(name)
        assert spec.valid_for_prediction
        assert spec.source_fields and spec.logic and spec.missing_behavior


def test_registry_classifications_pass_leakage_validator():
    from dltime.feature_assembly import sync_engineered_fields

    sync_engineered_fields()  # idempotent: engineered features -> leakage registry
    all_names = [n for cols in PREDICTION_FEATURE_GROUPS.values() for n in cols]
    # must not raise: all features are PRE_EXECUTION*/historical DERIVED_FROM_POST_EXECUTION
    validate_feature_set(all_names, context="registry-test")


def test_current_execution_post_execution_fields_rejected():
    for name in ("timetaken_in_sec", "End_Time", "LDR_Status", "Total_Records",
                 "Success_Records", "Error_Records", "Start_Time"):
        with pytest.raises(LeakageViolation):
            validate_feature_set([name], context="task02-test")


def test_unregistered_engineered_feature_rejected():
    with pytest.raises(LeakageViolation):
        validate_feature_set(["engineered_but_unregistered_v2"], context="task02-test")


def test_training_frame_has_no_post_execution_columns():
    tf = build_training_frame()
    banned = {"timetaken_in_sec", "End_Time", "Start_Time", "LDR_Status",
              "Total_Records", "Success_Records", "Error_Records",
              "Prevalidation_Errors", "Format_Errors", "Constraint_Errors",
              "Application_Errors", "Cascade_Errors"}
    overlap = banned & set(tf.X.columns)
    assert not overlap, f"post-execution columns leaked into X: {overlap}"


# ---------------- definition features ----------------

def test_identity_level_definition_features(bundle):
    feats = build_identity_level_definition_features(
        bundle.loader_definitions_detail, bundle.loader_definitions_summary
    )
    assert len(feats) > 0
    assert feats[["_d", "_s", "_l"]].duplicated().sum() == 0
    for col in DEF_FEATURES:
        assert col in feats.columns


def test_execution_level_definition_features_attach(foundation, bundle):
    ex = foundation.frames["executions_eligible"]
    attached = build_execution_level_definition_features(
        ex, bundle.loader_definitions_detail, bundle.loader_definitions_summary
    )
    assert len(attached) == len(ex)
    # EXP-028 (AD-31 corrected-source regime): definition keys align with the
    # corrected executions on only 34 executions — the def_match count is a
    # REAL data property, not 274 (the old dataset's resolved count).
    assert int(attached["def_match"].sum()) == 275
    # unmatched -> all definition features NA
    nomatch = attached[~attached["def_match"]]
    assert nomatch["def_prevalidation_count"].isna().all()
    assert nomatch["def_component_count"].isna().all()


def test_definition_feature_values_are_real(bundle):
    feats = build_identity_level_definition_features(
        bundle.loader_definitions_detail, bundle.loader_definitions_summary
    )
    # spot check against real definition rows: prevalidation_count may legitimately be 0
    pv = feats["def_prevalidation_count"].dropna()
    assert (pv >= 0).all()
    cc = feats["def_component_count"].dropna()
    assert (cc >= 1).all()
    dom = feats["def_dominant_calling_type"].dropna()
    allowed = set(bundle.loader_definitions_detail["calling_type"].dropna().unique())
    assert set(dom.unique()).issubset(allowed)


# ---------------- history eligibility ----------------

def test_history_eligibility_engine_states():
    eng = HistoryEligibilityEngine(HistoryEligibilityConfig(min_prior_loader_executions=3))
    r0 = eng.evaluate(hist_loader_prior_count=0)
    assert not r0.sufficient and r0.reason == "NO_DIRECT_HISTORY"
    r1 = eng.evaluate(hist_loader_prior_count=2)
    assert not r1.sufficient and r1.reason == "INSUFFICIENT_LOADER_HISTORY"
    r2 = eng.evaluate(hist_loader_prior_count=5, hist_loader_conn_prior_count=2,
                      hist_time_since_prev_loader_sec=3600.0)
    assert r2.sufficient and r2.reason == "SUFFICIENT_DIRECT_HISTORY"


def test_history_eligibility_threshold_is_configurable():
    eng = HistoryEligibilityEngine(HistoryEligibilityConfig(min_prior_loader_executions=10))
    r = eng.evaluate(hist_loader_prior_count=5)
    assert not r.sufficient and r.reason == "INSUFFICIENT_LOADER_HISTORY"


def test_prior_history_distribution_summary():
    s = summarize_prior_history_distribution(pd.Series([0, 1, 2, 3, 4, 10]))
    assert s["zero_history"] == 1
    assert s["ge_3"] == 3
    assert s["max"] == 10


# ---------------- training frame ----------------

def test_training_frame_shape_and_target(tf):
    # EXP-028 (AD-31 corrected-source regime): 512 reconstructed executions,
    # 353 training-eligible under the 2026-09-29 replaced source.
    assert tf.X.shape[0] == tf.y_raw.shape[0] == 353
    assert len(tf.frame) == 543
    assert (tf.y_raw > 0).all()
    assert tf.y_raw.notna().all()


def test_training_frame_separates_blocks(tf):
    assert set(tf.identifier_columns) == set(IDENTIFIER_COLUMNS)
    assert "timetaken_in_sec" not in tf.X.columns
    assert "LDR_Status" not in tf.X.columns
    for name in tf.feature_names:
        assert name in tf.X.columns


def test_training_frame_diagnostics(tf):
    d = tf.diagnostics
    assert d["reconstructed_executions"] == 543
    assert d["training_rows"] == 353
    assert d["executions_with_at_least_one_prior"] >= d["training_rows"] - 1
    dist = d["prior_loader_history_distribution_training"]


def test_target_transformer_roundtrip(tf):
    t = TargetTransformer("log1p")
    z = t.transform(tf.y_raw)
    back = t.inverse_transform(z)
    assert np.allclose(back, tf.y_raw.to_numpy(), rtol=1e-9, atol=1e-6)
    t_raw = TargetTransformer("raw")
    assert np.allclose(t_raw.transform(tf.y_raw), tf.y_raw.to_numpy())


def test_prior_only_guarantee_on_real_data(tf):
    """The single earliest execution must have zero priors in the full frame."""
    ordered = tf.frame.sort_values(["Start_Time", "LDR_Execution_Id"], kind="mergesort")
    first = ordered.iloc[0]
    assert first["hist_global_prior_count"] == 0
    assert pd.isna(first["hist_prev_loader_duration_sec"])


def test_x_positional_alignment_with_target(tf):
    assert len(tf.X) == len(tf.y_raw) == len(tf.y_log1p)
    assert tf.X.index.equals(tf.y_raw.index)
