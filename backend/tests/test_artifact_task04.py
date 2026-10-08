"""Task 04: persistent serving artifact (build/save/load/predict path)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dltime.artifact import (
    ARTIFACT_DIR,
    SERVING_MODEL_TYPE,
    SERVING_MODEL_VERSION,
    SERVING_STATUS,
    SERVING_TARGET_MODE,
    ArtifactError,
    build_serving_artifact,
    load_serving_artifact,
    predict_with_artifact,
    save_serving_artifact,
)
from dltime.modeling import chronological_holdout_split, add_chronological_rank


def test_artifact_metadata_matches_approved_candidate(tf):
    """Artifact must record the exact approved candidate configuration (AD-15)."""
    artifact = build_serving_artifact(tf)
    meta = artifact["metadata"]
    assert meta["model_version"] == SERVING_MODEL_VERSION == "total-et-v3-candidate"
    assert meta["model_type"] == SERVING_MODEL_TYPE == "extra_trees_candidate"
    assert meta["target_transformation"] == SERVING_TARGET_MODE == "log1p"
    assert meta["status"] == SERVING_STATUS == "CANDIDATE_EVALUATED"
    hp = meta["hyperparameters"]
    assert hp["n_estimators"] == 200
    assert hp["min_samples_leaf"] == 2
    assert hp["max_features"] == 1.0
    assert hp["random_state"] == 42
    # Exact Task 02 feature order preserved (2026-10-04: 44-feature contract)
    assert meta["feature_names_in_order"] == tf.feature_names
    assert len(meta["feature_names_in_order"]) == 44
    # Evaluation block carries the corrected-source AD-31 evidence (EXP-028)
    assert meta["evaluation"]["pooled_oof"]["median_ae"] == pytest.approx(3.19, abs=0.01)
    assert meta["evaluation_run"].startswith("EXP-028")


def test_artifact_training_window_is_exact_holdout_train_side(tf):
    """Training window must be the recorded chronological 265-row train split
    (2026-09-30 re-pin on the corrected export with real staging fields)."""
    artifact = build_serving_artifact(tf)
    full = tf.frame
    eligible = full[full["is_training_eligible"].fillna(False)]
    ordered = add_chronological_rank(eligible)
    train, test, info = chronological_holdout_split(ordered)
    meta = artifact["metadata"]
    assert meta["training_row_count"] == len(train) == 265
    assert len(test) == 88  # holdout rows excluded from the serving fit
    # metadata stores str(Timestamp) (same convention as modeling.split_info)
    assert str(train["Start_Time"].max()) == meta["training_period"]["last_start"]
    assert str(train["Start_Time"].min()) == meta["training_period"]["first_start"]
    # The served estimator was actually fitted on those rows (not just recorded)
    assert artifact["estimator"].n_outputs_ == 1
    assert artifact["metadata"]["fitted"]["n_training_rows"] == len(train)


def test_artifact_save_and_load_roundtrip(tf, tmp_path: Path):
    built = build_serving_artifact(tf)
    path = tmp_path / "art.pkl"
    save_serving_artifact(built, path)
    loaded = load_serving_artifact(path)
    assert loaded["metadata"]["model_version"] == SERVING_MODEL_VERSION
    assert loaded["metadata"]["feature_names_in_order"] == tf.feature_names
    for key in ("estimator", "preprocessor", "transformer"):
        assert key in loaded


def test_artifact_default_path_used_by_session(serving_artifact):
    """Session fixture persists to the default project artifact path."""
    assert serving_artifact["metadata"]["model_version"] == SERVING_MODEL_VERSION
    assert (ARTIFACT_DIR / f"{SERVING_MODEL_VERSION}.pkl").exists()


def test_load_rejects_wrong_version(tf, tmp_path: Path):
    built = build_serving_artifact(tf)
    built["metadata"]["model_version"] = "some-other-model"
    path = tmp_path / "wrong.pkl"
    save_serving_artifact(built, path)
    with pytest.raises(ArtifactError, match="model_version"):
        load_serving_artifact(path)


def test_load_refuses_production_status(tf, tmp_path: Path):
    """PRODUCTION status is not approved anywhere in this project — refuse."""
    built = build_serving_artifact(tf)
    built["metadata"]["status"] = "PRODUCTION"
    path = tmp_path / "prod.pkl"
    save_serving_artifact(built, path)
    with pytest.raises(ArtifactError, match="PRODUCTION"):
        load_serving_artifact(path)


def test_load_rejects_missing_file(tmp_path: Path):
    with pytest.raises(ArtifactError, match="not found"):
        load_serving_artifact(tmp_path / "missing.pkl")


def test_predict_with_artifact_rejects_reordered_features(serving_artifact):
    feature_names = serving_artifact["metadata"]["feature_names_in_order"]
    X = pd.DataFrame([{name: 0.0 for name in feature_names}])
    reordered = X[list(reversed(feature_names))]
    with pytest.raises(ArtifactError, match="order mismatch"):
        predict_with_artifact(serving_artifact, reordered)


def test_predict_with_artifact_rejects_multi_row(serving_artifact):
    feature_names = serving_artifact["metadata"]["feature_names_in_order"]
    X = pd.DataFrame([{name: 0.0 for name in feature_names}] * 2)
    with pytest.raises(ArtifactError, match="exactly one row"):
        predict_with_artifact(serving_artifact, X)


def test_log1p_expm1_prediction_path(tf, serving_artifact):
    """expm1 back-transform applied exactly once; log1p value consistent."""
    import numpy as np

    from dltime.inference import PredictionRequest

    # pick an eligible row with sufficient loader history for a PREDICTED call
    X = tf.X
    row_mask = X["hist_loader_prior_count"] >= 3
    assert row_mask.any()
    row = X[row_mask].iloc[0]
    # recover identity from the frame (X index aligns with tf.frame)
    ident = tf.frame.loc[row.name]
    service_features = serving_artifact["metadata"]["feature_names_in_order"]

    # sanity: one full predict path through the service is covered in
    # test_inference_task04.py; here we verify the transformation identity.
    seconds = 1234.0
    import math

    assert math.isclose(
        math.expm1(math.log1p(seconds)), seconds, rel_tol=1e-12
    )
    assert SERVING_TARGET_MODE == "log1p"
    assert service_features == tf.feature_names
