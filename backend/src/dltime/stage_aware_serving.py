"""Stage-aware TOTAL serving (AD-32): the authoritative total ETA path.

Promotes the AD-31 stage-aware total architecture from an evaluation-only
experiment to the SERVING model, per the locked final product direction:

    10 pre-execution inputs
      -> internal feature engineering (exact Task-02/stage paths)
      -> 5 ML stage predictions (real fitted stage models)
      + 2 contextual statistical approval estimates
      -> STAGE-AWARE TOTAL regression model  -> FINAL ETA

The direct total model (total-et-v3-candidate) is retained unchanged as a
REFERENCE/ROLLBACK artifact — it is no longer the primary ETA.

Leakage contract (unchanged, AD-31 §8/§17):
  * TRAINING stage columns are strictly-prior OUT-OF-FOLD predictions
    (block-cadence expanding OOF — train data always precedes the block start)
    plus strictly-prior hierarchical approval estimates; actual stage durations
    never enter any feature matrix.
  * INFERENCE stage columns are the served stage models' predictions for the
    context row (whose own target is NA and eligibility False) plus the
    approval estimator's strictly-prior estimates.

Artifact: backend/artifacts/stage-aware-total-ad34.pkl (joblib), carrying
estimator + encoded-column contract + imputation medians + full metadata.
Uncertainty: dedicated conformal calibration bound to this exact model version
(unc-log1p-conformal-v1-stage-aware-ad34.json). SHAP: TreeExplainer over the SAME
encoded matrix the prediction used, with additivity enforced.

Nothing here auto-promotes beyond the recorded governance action taken at
build time; the registry + lifecycle log remain the audit trail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dltime.config import PROJECT_ROOT

STAGE_AWARE_MODEL_VERSION = "stage-aware-total-ad34"  # AD-34: corrected total target (EXP-032)
STAGE_AWARE_ARTIFACT_PATH = PROJECT_ROOT / "backend" / "artifacts" / f"{STAGE_AWARE_MODEL_VERSION}.pkl"
STAGE_AWARE_CALIBRATION_PATH = PROJECT_ROOT / "backend" / "artifacts" / "unc-log1p-conformal-v1-stage-aware-ad34.json"

# The 7 stage-derived feature columns (identical names to the AD-31 evaluation).
STAGE_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"pred_stage_{s}_sec"
    for s in (
        "staging", "prevalidation", "transformation", "pre_load",
        "data_loading", "datamart_approval", "dataloading_approval",
    )
)
# Base (context) features = the stage-frame feature contract (def_* + stage_hist_*).
BASE_FEATURE_PREFIXES = ("def_", "stage_hist_")

# Human-readable display names for explanation (contribution language only).
STAGE_AWARE_DISPLAY_NAMES: dict[str, str] = {
    "pred_stage_staging_sec": "Predicted Staging duration",
    "pred_stage_prevalidation_sec": "Predicted Prevalidation duration",
    "pred_stage_transformation_sec": "Predicted Transformation duration",
    "pred_stage_pre_load_sec": "Predicted Pre-load duration",
    "pred_stage_data_loading_sec": "Predicted Data Loading duration",
    "pred_stage_datamart_approval_sec": "Estimated Datamart Approval duration",
    "pred_stage_dataloading_approval_sec": "Estimated Dataloading Approval duration",
}


class StageAwareArtifactError(RuntimeError):
    """Raised when the stage-aware artifact cannot be built/loaded safely."""


def base_feature_names(frame: pd.DataFrame) -> list[str]:
    """Context feature contract: def_* + stage_hist_* in deterministic frame order."""
    return [
        c for c in frame.columns
        if c.startswith(BASE_FEATURE_PREFIXES) and c != "def_match"
    ]


def feature_contract() -> list[str]:
    """The documented 36-feature stage-aware total contract (29 base + 7 stage)."""
    return base_feature_names_from_contract() + list(STAGE_FEATURE_COLUMNS)


def base_feature_names_from_contract() -> list[str]:
    """The context feature names, in the order build_stage_frame produces them.

    Derived from the authoritative stage-frame builder (same bundle context at
    build time), so the contract can never silently drift from the source.
    """
    from dltime.sources import load_all_sources
    from dltime.stage_models import build_stage_frame

    frame, _feats, _diag = build_stage_frame(stage="staging", bundle=load_all_sources())
    return base_feature_names(frame)


# --------------------------------------------------------------------- build

def build_stage_aware_serving_artifact(*, bundle=None) -> tuple[dict, dict]:
    """Build (fit) the stage-aware total artifact on the chronological train split.

    Returns (artifact, build_info). The evaluation metrics embedded in the
    artifact metadata are COMPUTED here (fresh chronological evidence), never
    copied from an old experiment record.
    """
    from dltime.feature_assembly import TargetTransformer
    from dltime.modeling import (
        add_chronological_rank,
        chronological_holdout_split,
        compute_metrics,
        expanding_window_folds,
    )
    from dltime.stage_aware_total import evaluate_stage_aware_total
    from dltime.stage_models import _make_estimator

    comparison = evaluate_stage_aware_total(bundle=bundle, total_model_kind="extra_trees", n_folds=4)

    # Rebuild the EXACT training matrix the comparison used, then fit on the
    # chronological train side of the same population.
    from dltime.stage_models import build_stage_frame
    from dltime.stage_targets import ML_STAGE_NAMES, APPROVAL_STAGE_NAMES, stage_target_column
    from dltime.approval_estimator import estimate_approval_stage
    from dltime.sources import load_all_sources

    bundle = bundle or load_all_sources()
    total_frame, total_feats, _ = build_stage_frame(stage="staging", bundle=bundle)
    completed = total_frame["LDR_Status"].isin(
        {"SUCCEED", "COMPLETED_WITH_WARNING", "COMPLETED_WITH_ERROR", "COMPLETED_WITH_ERROR_AND_WARNING", "ERROR"}
    )
    id_ok = total_frame["Loader_Name"].notna() & total_frame["Sprint"].notna()
    # AD-34 target correction (EXP-032): authoritative total =
    # total_timetaken_in_min * 60 (Start/End-span-consistent 757/757); the
    # legacy timetaken_in_sec column is corrupted and never the target.
    target = pd.to_numeric(total_frame["total_timetaken_in_min"], errors="coerce") * 60.0
    pop = (completed & id_ok & (target > 0)).fillna(False)
    frame = add_chronological_rank(total_frame[pop].copy())
    y_raw = target[pop].loc[frame.index]

    X = frame[total_feats].copy()
    for col in X.columns:
        if X[col].dtype == "string" or isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].fillna("__MISSING__").astype("string")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    # stage columns (OOF stage predictions + strictly-prior approval estimates)
    from dltime.stage_aware_total import _stage_oof_predictions

    for stage in ML_STAGE_NAMES:
        sframe, sfeats, sdiag = build_stage_frame(stage=stage, bundle=bundle)
        if sdiag["stage_eligible_rows"] >= 10:
            oof = _stage_oof_predictions(stage, sframe, sfeats, {})
            lookup = pd.Series(oof.values, index=sframe.loc[oof.index, "LDR_Execution_Id"].to_numpy())
            X[f"pred_stage_{stage}_sec"] = pd.to_numeric(frame["LDR_Execution_Id"].map(lookup), errors="coerce").astype(float)
        else:
            X[f"pred_stage_{stage}_sec"] = np.nan
    for stage in APPROVAL_STAGE_NAMES:
        estimates: dict[Any, float] = {}
        for _idx, row in frame.iterrows():
            est = estimate_approval_stage(frame, row["LDR_Execution_Id"], stage)
            if est.available and est.estimate_seconds is not None:
                estimates[row["LDR_Execution_Id"]] = float(est.estimate_seconds)
        X[f"pred_stage_{stage}_sec"] = frame["LDR_Execution_Id"].map(estimates).astype(float)

    feature_names = list(X.columns)
    expected = feature_contract()
    if feature_names != expected:
        raise StageAwareArtifactError(
            f"Stage-aware feature contract drifted: got {len(feature_names)} features, expected {len(expected)}."
        )

    train, _test, split_info = chronological_holdout_split(frame)
    X_train = X.loc[train.index]
    y_train = y_raw.loc[train.index].to_numpy(dtype=float)
    y_log = TargetTransformer(mode="log1p").transform(pd.Series(y_train))
    y_log_vals = (y_log.to_numpy() if hasattr(y_log, "to_numpy") else np.asarray(y_log, dtype=float))

    enc = X_train.astype(object).where(X_train.notna(), None)
    Xenc = pd.get_dummies(enc, dummy_na=True)
    est = _make_estimator("extra_trees")
    est.fit(Xenc, y_log_vals)
    medians = X_train.median(numeric_only=True)

    # ---- fresh holdout evaluation with THIS exact fitted configuration -------
    X_test = X.loc[frame.index.difference(train.index)]
    y_test = y_raw.loc[frame.index.difference(train.index)].to_numpy(dtype=float)
    Xte_enc = pd.get_dummies(
        X_test.astype(object).where(X_test.notna(), None), dummy_na=True
    ).reindex(columns=Xenc.columns, fill_value=0)
    pred_log = np.asarray(est.predict(Xte_enc), dtype=float)
    pred = TargetTransformer(mode="log1p").inverse_transform(pred_log)
    holdout = compute_metrics(y_test, pred)

    metadata = {
        "model_version": STAGE_AWARE_MODEL_VERSION,
        "status": "CANDIDATE_EVALUATED",
        "model_type": "stage_aware_total_extra_trees",
        "architecture": "stage_aware_total",
        "role": "AUTHORITATIVE_TOTAL",
        "hyperparameters": est.get_params(),
        "target_transformation": "log1p",
        "feature_names_in_order": feature_names,
        "base_feature_count": len(total_feats),
        "stage_feature_columns": list(STAGE_FEATURE_COLUMNS),
        "stage_sources": {
            "ml_stages": "five fitted ML stage models (OOF during training; served models at inference)",
            "approval_stages": "contextual hierarchical statistical estimates (strictly prior)",
            "leakage": "block-cadence expanding OOF — stage training data strictly precedes each predicted row; actual stage durations never enter features",
        },
        "encoding": "pd.get_dummies(dummy_na=True) over the object frame; column contract pinned",
        "training_row_count": int(split_info["train_rows"]),
        "training_period": {
            "first_start": split_info["train_first_start"],
            "last_start": split_info["train_last_start"],
        },
        "holdout_rows": int(len(frame) - split_info["train_rows"]),
        "holdout_period": {
            "first_start": split_info["test_first_start"],
            "last_start": split_info["test_last_start"],
        },
        "evaluation": {
            "split_method": "expanding-window chronological CV + final chronological holdout (identical population/folds for both architectures)",
            "config": {"holdout_fraction": 0.25, "min_holdout": 25, "n_folds": 4},
            "pooled_oof": comparison.stage_aware.get("pooled_oof"),
            "holdout": holdout,
        },
        "direct_reference_evaluation": {
            "note": "direct-total baseline on the identical population/folds (reference/rollback model)",
            "pooled_oof": comparison.direct.get("pooled_oof"),
            "holdout": comparison.direct.get("holdout"),
        },
        "evaluation_run": "EXP-030 (AD-32 stage-aware serving build; fresh chronological evidence)",
        "fitted": {
            "feature_out_count": int(Xenc.shape[1]),
            "sklearn_version": __import__("sklearn").__version__,
            "n_training_rows": int(len(train)),
        },
    }
    artifact = {
        "metadata": metadata,
        "estimator": est,
        "enc_columns": list(Xenc.columns),
        "impute_medians": {k: float(v) for k, v in medians.items()},
        "transformer": TargetTransformer(mode="log1p"),
    }
    build_info = {"comparison_verdict": comparison.verdict, "population_rows": int(len(frame))}
    return artifact, build_info


def save_stage_aware_artifact(artifact: dict, path: Path | None = None) -> Path:
    path = path or STAGE_AWARE_ARTIFACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(artifact, path)
    return path


def load_stage_aware_artifact(path: Path | None = None) -> dict:
    """Load + integrity-check the stage-aware artifact (fail closed)."""
    path = path or STAGE_AWARE_ARTIFACT_PATH
    if not path.exists():
        raise StageAwareArtifactError(f"Stage-aware artifact not found: {path}")
    import joblib

    try:
        artifact = joblib.load(path)
    except Exception as exc:
        raise StageAwareArtifactError(f"Stage-aware artifact unreadable at {path}: {exc}") from exc
    meta = artifact.get("metadata", {})
    if meta.get("model_version") != STAGE_AWARE_MODEL_VERSION:
        raise StageAwareArtifactError(
            f"Stage-aware artifact version mismatch: {meta.get('model_version')!r}"
        )
    if meta.get("feature_names_in_order") != feature_contract():
        raise StageAwareArtifactError("Stage-aware feature contract mismatch; refusing to serve.")
    if list(artifact.get("enc_columns", [])) != meta.get("fitted", {}).get("enc_columns", artifact.get("enc_columns", [])):
        pass  # enc_columns are the authority; consistency enforced at predict time
    for key in ("estimator", "enc_columns", "impute_medians", "transformer"):
        if key not in artifact:
            raise StageAwareArtifactError(f"Stage-aware artifact missing component '{key}'.")
    return artifact


def artifact_exists() -> bool:
    return STAGE_AWARE_ARTIFACT_PATH.exists()


# ------------------------------------------------------------------ inference

def _encode_row(X: pd.DataFrame, enc_columns: list[str]) -> pd.DataFrame:
    """Encode context feature rows EXACTLY like the artifact's training encode."""
    enc = pd.get_dummies(X.astype(object).where(X.notna(), None), dummy_na=True)
    return enc.reindex(columns=enc_columns, fill_value=0)


def predict_stage_aware_seconds(artifact: dict, X: pd.DataFrame) -> float:
    """One-row stage-aware prediction; returns raw seconds (expm1-inverse)."""
    meta = artifact["metadata"]
    if list(X.columns) != meta["feature_names_in_order"]:
        raise StageAwareArtifactError("Stage-aware feature order mismatch at predict time.")
    if len(X) != 1:
        raise StageAwareArtifactError("predict_stage_aware_seconds expects exactly one row.")
    Xf = X.copy()
    medians = artifact["impute_medians"]
    for col in Xf.columns:
        if Xf[col].dtype == "string" or isinstance(Xf[col].dtype, pd.CategoricalDtype):
            Xf[col] = Xf[col].fillna("__MISSING__").astype("string")
        else:
            Xf[col] = pd.to_numeric(Xf[col], errors="coerce").fillna(medians.get(col, 0.0))
    Xenc = _encode_row(Xf, artifact["enc_columns"])
    pred_log = float(artifact["estimator"].predict(Xenc)[0])
    return float(artifact["transformer"].inverse_transform([pred_log])[0])


def explain_stage_aware(artifact: dict, X: pd.DataFrame) -> dict[str, Any]:
    """SHAP explanation of the ACTUAL served stage-aware prediction.

    TreeExplainer over the exact encoded row; contributions aggregated back to
    the 36 registered features by encoded-name mapping (longest-prefix match on
    the pinned get_dummies columns). Additivity is enforced (raises on
    violation). Contributions are model attributions in log1p space — never
    causal effects, never seconds.
    """
    import shap

    meta = artifact["metadata"]
    if list(X.columns) != meta["feature_names_in_order"] or len(X) != 1:
        raise StageAwareArtifactError("Explanation requires the single-row exact contract vector.")
    Xf = X.copy()
    medians = artifact["impute_medians"]
    for col in Xf.columns:
        if Xf[col].dtype == "string" or isinstance(Xf[col].dtype, pd.CategoricalDtype):
            Xf[col] = Xf[col].fillna("__MISSING__").astype("string")
        else:
            Xf[col] = pd.to_numeric(Xf[col], errors="coerce").fillna(medians.get(col, 0.0))
    Xenc = _encode_row(Xf, artifact["enc_columns"])

    explainer = shap.TreeExplainer(artifact["estimator"])
    sv = np.asarray(explainer.shap_values(Xenc))
    if sv.ndim == 3:
        sv = sv[:, :, 0]
    if sv.ndim != 2 or sv.shape[0] != 1:
        raise StageAwareArtifactError(f"Unexpected SHAP output shape {np.shape(sv)}.")

    base = float(np.asarray(explainer.expected_value).ravel()[0])
    log_output = float(artifact["estimator"].predict(Xenc)[0])
    additivity_err = abs(base + float(sv[0].sum()) - log_output)
    if additivity_err > 1e-6:
        raise StageAwareArtifactError(
            f"SHAP additivity violated: base+sum={base + float(sv[0].sum()):.9f} vs model {log_output:.9f}"
        )

    # encoded column -> registered feature (longest registered prefix wins)
    registered = list(meta["feature_names_in_order"])
    col_owner: dict[int, str] = {}
    for idx, enc_name in enumerate(artifact["enc_columns"]):
        owner = None
        for feat in sorted(registered, key=len, reverse=True):
            if enc_name == feat or enc_name.startswith(feat + "_"):
                owner = feat
                break
        if owner is None:
            raise StageAwareArtifactError(f"Encoded column {enc_name!r} maps to no registered feature.")
        col_owner[idx] = owner

    aggregated: dict[str, float] = {name: 0.0 for name in registered}
    for idx, owner in col_owner.items():
        aggregated[owner] += float(sv[0][idx])

    row_raw = X.iloc[0]
    features: list[dict[str, Any]] = []
    for name in registered:
        val = aggregated[name]
        direction = "higher" if val > 0 else ("lower" if val < 0 else "neutral")
        raw = row_raw[name]
        features.append({
            "feature": name,
            "display_name": STAGE_AWARE_DISPLAY_NAMES.get(name, _display_for_base(name)),
            "value": _jsonable(raw),
            "shap_value_log1p": round(val, 6),
            "direction": direction,
        })
    features.sort(key=lambda f: -abs(f["shap_value_log1p"]))
    for rank, f in enumerate(features, start=1):
        f["contribution_rank"] = rank

    from dltime.explain import human_readable_summary

    payload = {
        "method": "shap_tree_explainer",
        "shap_version": str(shap.__version__),
        "model_version": STAGE_AWARE_MODEL_VERSION,
        "output_space": (
            "log1p(total_execution_seconds): contributions are additive in the "
            "model's log-scale output; the ETA in seconds is shown separately"
        ),
        "base_value_log1p": round(base, 6),
        "sum_contributions_log1p": round(float(sv[0].sum()), 6),
        "model_output_log1p": round(log_output, 6),
        "additivity_abs_error": float(f"{additivity_err:.3e}"),
        "interpretation": (
            "Model contributions, not causal effects. Each entry shows how the "
            "feature moved the model's log-scale ETA for this prediction."
        ),
        "top_features": features[:5],
        "features": features,
    }
    payload["human_readable_summary"] = human_readable_summary(payload)
    return payload


def _display_for_base(name: str) -> str:
    from dltime.explain import DISPLAY_NAMES

    return DISPLAY_NAMES.get(name, name)


def _normalize_contract_frame(X: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Contract-ordered frame with the artifact's exact imputation semantics."""
    Xf = X.copy()
    medians = artifact["impute_medians"]
    for col in Xf.columns:
        if Xf[col].dtype == "string" or isinstance(Xf[col].dtype, pd.CategoricalDtype):
            Xf[col] = Xf[col].fillna("__MISSING__").astype("string")
        else:
            Xf[col] = pd.to_numeric(Xf[col], errors="coerce").fillna(medians.get(col, 0.0))
    return Xf


def stage_aware_holdout_matrix(
    artifact: dict, *, bundle=None
) -> tuple[pd.DataFrame, pd.Series]:
    """36-feature holdout matrix + actual totals for the GLOBAL summary.

    Context features come from the stage-frame population's final
    chronological holdout (rows the total model never trained on). Stage
    columns use the SERVED stage models (in-sample for the stage models —
    disclosed; this is a supplementary explanation, not an evaluation) and
    strictly-prior approval estimates. Never used for metrics.
    """
    from dltime.approval_estimator import estimate_approval_stage
    from dltime.modeling import add_chronological_rank, chronological_holdout_split
    from dltime.stage_models import build_stage_frame
    from dltime.stage_targets import APPROVAL_STAGE_NAMES, ML_STAGE_NAMES
    from dltime.stage_serving import load_stage_serving_models

    bundle = bundle or __import__("dltime.sources", fromlist=["load_all_sources"]).load_all_sources()
    total_frame, total_feats, _ = build_stage_frame(stage="staging", bundle=bundle)
    completed = total_frame["LDR_Status"].isin(
        {"SUCCEED", "COMPLETED_WITH_WARNING", "COMPLETED_WITH_ERROR", "COMPLETED_WITH_ERROR_AND_WARNING", "ERROR"}
    )
    id_ok = total_frame["Loader_Name"].notna() & total_frame["Sprint"].notna()
    # AD-34 target correction (EXP-032): authoritative total =
    # total_timetaken_in_min * 60; legacy timetaken_in_sec is corrupted.
    target = pd.to_numeric(total_frame["total_timetaken_in_min"], errors="coerce") * 60.0
    pop = (completed & id_ok & (target > 0)).fillna(False)
    frame = add_chronological_rank(total_frame[pop].copy())
    _train, test, _info = chronological_holdout_split(frame)
    hold = frame.loc[test.index]

    X = hold[total_feats].copy()
    for col in X.columns:
        if X[col].dtype == "string" or isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].astype("string")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)

    predictors = load_stage_serving_models(bundle=bundle)
    for stage in ML_STAGE_NAMES:
        pred = predictors.get(stage)
        vals: list[float | None] = []
        if pred is not None:
            sub = hold[getattr(pred, "feature_names")]
            for i in range(len(sub)):
                try:
                    vals.append(float(pred.predict_seconds(sub.iloc[[i]])))
                except Exception:
                    vals.append(None)
        X[f"pred_stage_{stage}_sec"] = pd.to_numeric(
            pd.Series(vals, index=X.index), errors="coerce"
        ).astype(float)
    for stage in APPROVAL_STAGE_NAMES:
        est_map: dict[Any, float] = {}
        for _idx, row in hold.iterrows():
            est = estimate_approval_stage(frame, row["LDR_Execution_Id"], stage)
            if est.available and est.estimate_seconds is not None:
                est_map[row["LDR_Execution_Id"]] = float(est.estimate_seconds)
        X[f"pred_stage_{stage}_sec"] = (
            hold["LDR_Execution_Id"].map(est_map).astype(float)
        )

    expected = feature_contract()
    if list(X.columns) != expected:
        raise StageAwareArtifactError("Global-summary matrix drifted from the 36-feature contract.")
    y = target[pop].loc[test.index]
    return X, y


def global_explanation_stage_aware(
    artifact: dict, X: pd.DataFrame, explainer: Any | None = None
) -> dict[str, Any]:
    """Mean |SHAP| per registered feature for the AUTHORITATIVE total model.

    Same aggregation contract as the legacy global summary (encoded columns
    mapped back to registered features) but over the stage-aware 36-feature
    contract. Model contributions only — never causal claims.
    """
    import shap

    meta = artifact["metadata"]
    if list(X.columns) != meta["feature_names_in_order"] or len(X) == 0:
        raise StageAwareArtifactError("Global summary requires the exact 36-feature contract frame.")
    Xf = _normalize_contract_frame(X, artifact)
    Xenc = _encode_row(Xf, artifact["enc_columns"])

    if explainer is None:
        explainer = shap.TreeExplainer(artifact["estimator"])
    sv = np.asarray(explainer.shap_values(Xenc))
    if sv.ndim == 3:
        sv = sv[:, :, 0]
    if sv.ndim != 2 or sv.shape[1] != Xenc.shape[1]:
        raise StageAwareArtifactError(f"Unexpected SHAP output shape {np.shape(sv)} for global summary.")

    registered = list(meta["feature_names_in_order"])
    agg = np.zeros((sv.shape[0], len(registered)))
    col_of = {name: i for i, name in enumerate(registered)}
    for idx, enc_name in enumerate(artifact["enc_columns"]):
        owner = None
        for feat in sorted(registered, key=len, reverse=True):
            if enc_name == feat or enc_name.startswith(feat + "_"):
                owner = feat
                break
        if owner is None:
            raise StageAwareArtifactError(f"Encoded column {enc_name!r} maps to no registered feature.")
        agg[:, col_of[owner]] += sv[:, idx]

    mean_abs = np.abs(agg).mean(axis=0)
    mean_signed = agg.mean(axis=0)
    order = np.argsort(-mean_abs)
    features = [
        {
            "feature": registered[i],
            "display_name": STAGE_AWARE_DISPLAY_NAMES.get(registered[i], _display_for_base(registered[i])),
            "mean_abs_shap_log1p": round(float(mean_abs[i]), 6),
            "mean_signed_shap_log1p": round(float(mean_signed[i]), 6),
        }
        for i in order
    ]
    return {
        "method": "shap_tree_explainer",
        "model_version": STAGE_AWARE_MODEL_VERSION,
        "n_observations": int(len(X)),
        "output_space": "log1p(total_execution_seconds) — model contributions, not causal effects",
        "subset_description": (
            "Final chronological holdout of the stage-aware population (rows the "
            "total model never trained on). Stage columns computed with the served "
            "stage models (supplementary explanation only — never an evaluation)."
        ),
        "features": features,
    }


def _jsonable(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


# ---------------------------------------------------------------- calibration

def build_stage_aware_calibration(artifact: dict, *, bundle=None) -> dict[str, Any]:
    """Conformal calibration for the EXACT stage-aware artifact.

    Identical method/semantics to the Task 07 layer (log1p absolute-residual
    conformal on the artifact's own chronological holdout — rows it never
    trained on), bound to the stage-aware model version + 36-feature contract.
    """
    from dltime.feature_assembly import TargetTransformer
    from dltime.modeling import add_chronological_rank, chronological_holdout_split
    from dltime.stage_aware_total import _stage_oof_predictions
    from dltime.stage_models import build_stage_frame
    from dltime.stage_targets import ML_STAGE_NAMES, APPROVAL_STAGE_NAMES, stage_target_column
    from dltime.approval_estimator import estimate_approval_stage
    from dltime.sources import load_all_sources
    from dltime.uncertainty import (
        CALIBRATION_METHOD,
        CALIBRATION_VERSION,
        INTERVAL_LEVELS,
        MIN_QUANTILE_FLOOR,
    )

    bundle = bundle or load_all_sources()
    total_frame, total_feats, _ = build_stage_frame(stage="staging", bundle=bundle)
    completed = total_frame["LDR_Status"].isin(
        {"SUCCEED", "COMPLETED_WITH_WARNING", "COMPLETED_WITH_ERROR", "COMPLETED_WITH_ERROR_AND_WARNING", "ERROR"}
    )
    id_ok = total_frame["Loader_Name"].notna() & total_frame["Sprint"].notna()
    # AD-34 target correction (EXP-032): authoritative total =
    # total_timetaken_in_min * 60; legacy timetaken_in_sec is corrupted.
    target = pd.to_numeric(total_frame["total_timetaken_in_min"], errors="coerce") * 60.0
    pop = (completed & id_ok & (target > 0)).fillna(False)
    frame = add_chronological_rank(total_frame[pop].copy())

    X = frame[total_feats].copy()
    for col in X.columns:
        if X[col].dtype == "string" or isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].fillna("__MISSING__").astype("string")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    for stage in ML_STAGE_NAMES:
        sframe, sfeats, sdiag = build_stage_frame(stage=stage, bundle=bundle)
        if sdiag["stage_eligible_rows"] >= 10:
            oof = _stage_oof_predictions(stage, sframe, sfeats, {})
            lookup = pd.Series(oof.values, index=sframe.loc[oof.index, "LDR_Execution_Id"].to_numpy())
            X[f"pred_stage_{stage}_sec"] = pd.to_numeric(frame["LDR_Execution_Id"].map(lookup), errors="coerce").astype(float)
        else:
            X[f"pred_stage_{stage}_sec"] = np.nan
    for stage in APPROVAL_STAGE_NAMES:
        estimates: dict[Any, float] = {}
        for _idx, row in frame.iterrows():
            est = estimate_approval_stage(frame, row["LDR_Execution_Id"], stage)
            if est.available and est.estimate_seconds is not None:
                estimates[row["LDR_Execution_Id"]] = float(est.estimate_seconds)
        X[f"pred_stage_{stage}_sec"] = frame["LDR_Execution_Id"].map(estimates).astype(float)

    _train, test, split_info = chronological_holdout_split(frame)
    X_cal = X.loc[test.index]
    y_cal = target[pop].loc[test.index].to_numpy(dtype=float)
    preds = np.array(
        [predict_stage_aware_seconds(artifact, X_cal.iloc[[i]]) for i in range(len(X_cal))],
        dtype=float,
    )
    positive = y_cal > 0
    abs_res_log = np.abs(np.log1p(y_cal[positive]) - np.log1p(preds[positive]))
    quantiles = {}
    for level in INTERVAL_LEVELS:
        q = float(np.quantile(abs_res_log, level, method="higher"))
        quantiles[f"{int(round(level * 100))}"] = max(q, MIN_QUANTILE_FLOOR)

    meta = {
        "calibration_version": CALIBRATION_VERSION,
        "method": CALIBRATION_METHOD,
        "interval_levels": list(INTERVAL_LEVELS),
        "quantiles_abs_log_residual": quantiles,
        "n_calibration_rows": int(len(X_cal)),
        "n_positive_target_rows": int(positive.sum()),
        "calibration_window": {
            "first_start": str(split_info["test_first_start"]),
            "last_start": str(split_info["test_last_start"]),
            "scope": "final chronological holdout; strictly after the stage-aware model training window; out-of-sample for the serving artifact",
        },
        "residual_stats": {
            "median_abs_log_residual": float(np.median(abs_res_log)),
            "mean_abs_log_residual": float(abs_res_log.mean()),
            "max_abs_log_residual": float(abs_res_log.max()),
            "share_under_predictions": float(
                (np.log1p(y_cal[positive]) < np.log1p(preds[positive])).mean()
            ),
        },
        "model_version": STAGE_AWARE_MODEL_VERSION,
        "model_feature_count": len(feature_contract()),
        "diagnostics": {
            "quantile_floor": MIN_QUANTILE_FLOOR,
            "quantile_interpolation": "higher",
            "note": (
                "Levels are conformal TARGETS (share of calibration residuals "
                "within q), not verified probabilities. Fresh EXP-030 evidence."
            ),
        },
    }
    STAGE_AWARE_CALIBRATION_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_stage_aware_calibration() -> dict[str, Any]:
    """Load + integrity-check the stage-aware calibration (fail closed)."""
    from dltime.uncertainty import (
        CALIBRATION_METHOD,
        CALIBRATION_VERSION,
        INTERVAL_LEVELS,
    )

    if not STAGE_AWARE_CALIBRATION_PATH.exists():
        raise StageAwareArtifactError(
            f"Stage-aware calibration not found: {STAGE_AWARE_CALIBRATION_PATH}"
        )
    meta = json.loads(STAGE_AWARE_CALIBRATION_PATH.read_text(encoding="utf-8"))
    if meta.get("calibration_version") != CALIBRATION_VERSION:
        raise StageAwareArtifactError("Stage-aware calibration version mismatch.")
    if meta.get("method") != CALIBRATION_METHOD:
        raise StageAwareArtifactError("Unknown stage-aware calibration method.")
    if meta.get("model_version") != STAGE_AWARE_MODEL_VERSION:
        raise StageAwareArtifactError(
            f"Calibration bound to {meta.get('model_version')!r}, serving {STAGE_AWARE_MODEL_VERSION!r}"
        )
    if meta.get("model_feature_count") != len(feature_contract()):
        raise StageAwareArtifactError("Stage-aware calibration feature count mismatch.")
    for lvl in INTERVAL_LEVELS:
        if str(int(round(lvl * 100))) not in meta.get("quantiles_abs_log_residual", {}):
            raise StageAwareArtifactError(f"Stage-aware calibration missing quantile for {lvl}.")
    return meta
