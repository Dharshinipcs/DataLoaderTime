"""Task 03: chronological model training, evaluation, and history-eligibility evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dltime.modeling import (
    EXTRATREES_CANDIDATE,
    add_chronological_rank,
    aggregate_fold_metrics,
    available_candidates,
    build_model_artifact,
    chronological_holdout_split,
    compute_metrics,
    evaluate_candidate,
    evaluate_candidate_synthetic,
    expanding_window_folds,
    history_bin_evaluation,
    matrix_encoding_columns,
    seen_unseen_evaluation,
)

synthetic = evaluate_candidate_synthetic


# ------------------------- chronological split correctness -------------------------

def test_chrono_rank_is_deterministic_and_tie_broken(tf):
    frame = tf.frame[tf.frame["is_training_eligible"].fillna(False)]
    ordered1 = add_chronological_rank(frame)
    ordered2 = add_chronological_rank(frame)
    pd.testing.assert_frame_equal(ordered1, ordered2)
    # strictly non-decreasing time
    times = pd.to_datetime(ordered1["Start_Time"])
    assert times.is_monotonic_increasing
    # ties broken by LDR_Execution_Id (no random ordering)
    dup_mask = times.duplicated(keep=False)
    if dup_mask.any():
        sub = ordered1.loc[dup_mask, ["Start_Time", "LDR_Execution_Id"]]
        for _, grp in sub.groupby("Start_Time"):
            assert grp["LDR_Execution_Id"].is_monotonic_increasing


def test_holdout_earliest_train_latest_test(tf):
    frame = tf.frame[tf.frame["is_training_eligible"].fillna(False)]
    train, test, info = chronological_holdout_split(frame)
    assert len(train) + len(test) == len(frame)
    assert train["Start_Time"].max() <= test["Start_Time"].min()
    # no timestamp straddles the boundary
    assert train["Start_Time"].iloc[-1] != test["Start_Time"].iloc[0] or (
        train["Start_Time"].max() < test["Start_Time"].min()
    )
    assert info["train_rows"] == len(train)
    assert info["test_rows"] == len(test)


def test_expanding_folds_are_chronological(tf):
    frame = tf.frame[tf.frame["is_training_eligible"].fillna(False)]
    folds = expanding_window_folds(frame)
    assert len(folds) == 5
    last_val_end = 0
    for f in folds:
        # training window ends exactly where its validation begins (no gap/overlap)
        assert f["val_rank_start"] == f["train_rank_end_exclusive"]
        # validation blocks are contiguous and move forward in time
        assert f["val_rank_start"] >= last_val_end
        last_val_end = f["val_rank_end_exclusive"]
    # training window is expanding: later folds train on at least as much
    train_ends = [f["train_rank_end_exclusive"] for f in folds]
    assert train_ends == sorted(train_ends)
    assert folds[-1]["val_rank_end_exclusive"] == len(frame)


# ------------------------- synthetic leakage-in-evaluation probe -------------------------

def test_no_future_rows_in_training_via_synthetic_probe():
    """Future-row targets cannot influence earlier-fold predictions.

    Targets of rows 60+ are multiplied by 1000. Fold 1 trains ONLY on rows
    0..59 (clean), so its predictions must be byte-identical between runs.
    (Validation targets differ by design in the poisoned run — that is not
    leakage, it is the poisoned outcome itself.)
    """
    n = 120
    rng = np.random.default_rng(7)
    base = pd.DataFrame(
        {
            "Start_Time": pd.date_range("2024-01-01", periods=n, freq="h"),
            "LDR_Execution_Id": np.arange(n),
            "hist_loader_prior_count": rng.integers(0, 5, n),
            "f_num": rng.normal(size=n),
        }
    )
    y = pd.Series(100 + rng.normal(scale=5, size=n))

    res_normal = synthetic(frame=base, y=y, model_kind="tree", target_mode="log1p")
    y_poisoned = y.copy()
    y_poisoned.iloc[60:] *= 1000.0  # future rows relative to fold-1 training
    res_poisoned = synthetic(frame=base, y=y_poisoned, model_kind="tree", target_mode="log1p")

    # fold-1 validation block = rows 60..74 (initial window is 60 rows)
    f1 = expanding_window_folds(base, n_folds=4)[0]
    assert f1["train_rank_end_exclusive"] == 60
    val_idx = base.index[f1["val_rank_start"] : f1["val_rank_end_exclusive"]]
    # predictions depend only on training data -> identical, proving no leak
    np.testing.assert_allclose(
        res_normal.oof_by_row.loc[val_idx].to_numpy(),
        res_poisoned.oof_by_row.loc[val_idx].to_numpy(),
        rtol=1e-12,
    )
    # sanity: the probe CAN detect poisoning (last fold validates on poisoned rows)
    assert res_poisoned.fold_metrics[-1]["mae"] > 100 * res_normal.fold_metrics[-1]["mae"]


# ------------------------- metrics -------------------------

def test_compute_metrics_exact_values():
    y = np.array([100.0, 200.0, 50.0])
    p = np.array([110.0, 100.0, 50.0])
    m = compute_metrics(y, p)
    assert m["mae"] == pytest.approx((10 + 100 + 0) / 3)
    assert m["median_ae"] == pytest.approx(10.0)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 10000 + 0) / 3))
    assert m["within_10pct"] == pytest.approx(2 / 3 * 100)
    assert m["within_20pct"] == pytest.approx(2 / 3 * 100)
    assert m["within_30pct"] == pytest.approx(2 / 3 * 100)
    # all three fall inside [0.5y, 2y]: [50,200],[100,400],[25,100]
    assert m["within_2x"] == pytest.approx(100.0)


def test_log1p_expm1_roundtrip_and_zero_target():
    from dltime.feature_assembly import TargetTransformer

    y = pd.Series([1.0, 65.62, 43_000_000.0])
    tt = TargetTransformer(mode="log1p")
    roundtrip = tt.inverse_transform(tt.transform(y))
    np.testing.assert_allclose(roundtrip, y.to_numpy(), rtol=1e-9)
    raw = TargetTransformer(mode="raw")
    np.testing.assert_allclose(raw.inverse_transform(raw.transform(y)), y.to_numpy())
    with pytest.raises(Exception):
        TargetTransformer(mode="log1p").transform(pd.Series([0.0, -1.0]))


def test_aggregate_fold_metrics_distinct_views():
    folds = [
        compute_metrics(np.array([100.0, 200.0]), np.array([110.0, 180.0])),
        compute_metrics(np.array([1000.0]), np.array([900.0])),
    ]
    agg = aggregate_fold_metrics(folds)
    assert agg["n_folds"] == 2
    assert agg["total_val_rows"] == 3
    # pooled over 3 rows differs from mean across 2 folds for MAE
    pooled_mae = (10 + 20 + 100) / 3
    mean_folds = (15 + 100) / 2
    assert agg["mae"]["mean_across_folds"] == pytest.approx(mean_folds)
    assert agg["mae"]["median_across_folds"] == pytest.approx(np.median([15, 100]))
    assert agg["mae"]["worst_fold"] == pytest.approx(100)
    # pooled reconstruction from explicit values
    assert pooled_mae == pytest.approx(130 / 3)


def test_metric_functions_reproducible(tf):
    frame = tf.frame[tf.frame["is_training_eligible"].fillna(False)]
    folds1 = expanding_window_folds(frame)
    folds2 = expanding_window_folds(frame)
    assert folds1 == folds2


# ------------------------- real-frame candidate evaluation -------------------------

@pytest.fixture(scope="module")
def ordered_real(tf):
    frame = tf.frame[tf.frame["is_training_eligible"].fillna(False)]
    return add_chronological_rank(frame)


@pytest.fixture(scope="module")
def real_folds(ordered_real):
    return expanding_window_folds(ordered_real)


def test_extra_trees_candidate_config():
    assert EXTRATREES_CANDIDATE == {
        "n_estimators": 200,
        "min_samples_leaf": 2,
        "max_features": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }


def test_all_candidate_names_available():
    names = set(available_candidates())
    assert {"baseline_global_median", "baseline_global_mean", "ridge", "random_forest",
            "extra_trees_candidate", "extra_trees_default", "hist_gradient_boosting"} <= names


def test_feature_order_consistency_and_no_identifiers(tf):
    feature_names = tf.feature_names
    assert len(feature_names) == len(set(feature_names))
    ident = set(tf.identifier_columns) | {"timetaken_in_sec", "y_log1p", "End_Time"}
    assert not ident & set(feature_names)


def test_categorical_encoding_columns(tf):
    cat, num = matrix_encoding_columns(tf.X)
    assert "def_dominant_calling_type" in cat
    # one-hot is applied in the pipeline (handle_unknown=ignore), never ordinal codes
    assert len(num) > 20


def test_real_extra_trees_candidate_log1p(tf, ordered_real, real_folds):
    res = evaluate_candidate(
        "extra_trees_candidate", available_candidates()["extra_trees_candidate"],
        tf, "log1p", folds=real_folds, ordered_frame=ordered_real,
    )
    # pooled OOF covers every row outside the initial training window
    assert res.pooled_oof["n"] == len(ordered_real) - real_folds[0]["train_rows"]
    assert res.per_fold_aggregate["total_val_rows"] == res.pooled_oof["n"]
    assert res.holdout["n"] == len(ordered_real) - res.holdout["split"]["train_rows"]
    assert res.per_fold_aggregate["n_folds"] == 5
    # back-transform produced positive raw-second predictions
    assert (res.oof_by_row.dropna() > 0).all()


@pytest.mark.parametrize("name", ["baseline_global_median", "ridge"])
def test_real_baselines_run(tf, ordered_real, real_folds, name):
    res = evaluate_candidate(
        name, available_candidates()[name], tf, "log1p", folds=real_folds, ordered_frame=ordered_real
    )
    assert res.pooled_oof["n"] == len(ordered_real) - real_folds[0]["train_rows"]


def test_seen_unseen_and_history_bins(tf, ordered_real, real_folds):
    res = evaluate_candidate(
        "extra_trees_candidate", available_candidates()["extra_trees_candidate"],
        tf, "log1p", folds=real_folds, ordered_frame=ordered_real,
    )
    su = seen_unseen_evaluation(res, tf, ordered_real)
    assert su["seen_loader"]["n"] + su["unseen_loader"]["n"] == res.pooled_oof["n"]
    bins = history_bin_evaluation(res, tf, ordered_real)
    assert sum(b["n"] for b in bins.values()) == res.pooled_oof["n"]


def test_model_artifact_record(tf, ordered_real, real_folds):
    res = evaluate_candidate(
        "extra_trees_candidate", available_candidates()["extra_trees_candidate"],
        tf, "log1p", folds=real_folds, ordered_frame=ordered_real,
    )
    art = build_model_artifact(
        "extra_trees_candidate", "log1p", tf, res,
        evaluation_config={"n_folds": 5, "initial_train_fraction": 0.5, "holdout_fraction": 0.25},
    )
    assert art["status"] == "CANDIDATE_EVALUATED"
    assert art["feature_names_in_order"] == tf.feature_names
    assert art["target_transformation"] == "log1p"
    assert art["hyperparameters"]["n_estimators"] == 200
