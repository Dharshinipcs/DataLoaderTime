"""SHAP explainability for the authoritative total model (Task 08, AD-21).

Explanatory enrichment ONLY — never a second prediction model. The point ETA,
features, eligibility, and uncertainty layers are untouched; if explanation
fails, the PREDICTED response simply carries an explanatory warning.

Method: shap.TreeExplainer on the persisted ExtraTrees serving artifact
(total-et-v3-candidate, log1p target). SHAP values therefore explain the
model's LOG-SCALE output: base + sum(contributions) == model log1p output
(additivity is validated at request time and reported in the response).
No linear "seconds" conversion of contributions is offered — the ETA in
seconds is displayed separately by the UI.

Encoded-column mapping: the preprocessor expands the 30 registered features
into 32 model columns (29 numeric 1:1 + one-hot of def_dominant_calling_type).
The mapping is DERIVED from the fitted ColumnTransformer (never guessed from
strings) and every encoded column is aggregated back to its registered
feature by summation — contributions are preserved exactly.

Leakage: explanation is computed from the exact prediction-time feature
vector already assembled for the point prediction (same Task 02 path). No
second feature pipeline exists; no post-execution field can enter it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

EXPLAIN_VERSION = "shap-treeexplainer-v1"
TOP_K = 5

# Human-readable names for the 30 registered features (display only; no
# semantic invention — each mirrors the feature's documented definition).
DISPLAY_NAMES: dict[str, str] = {
    "def_prevalidation_count": "Prevalidation configuration count",
    "def_transformation_count": "Transformation configuration count",
    "def_execution_order_max": "Max execution order",
    "def_execution_order_mean": "Mean execution order",
    "def_execution_order_std": "Execution order variability",
    "def_component_count": "Definition component count",
    "def_type_nunique": "Distinct component types",
    "def_dominant_calling_type": "Dominant calling type",
    "def_preval_name_component": "Prevalidation component display name",
    "def_trans_name_component": "Transformation component display name",
    # 2026-10-04 (user spec): per-section execution order + per-type counts
    "def_preval_order_max": "Prevalidation section last order",
    "def_preval_order_mean": "Prevalidation section mean order",
    "def_preval_order_std": "Prevalidation section order variability",
    "def_trans_order_max": "Transformation section last order",
    "def_trans_order_mean": "Transformation section mean order",
    "def_trans_order_std": "Transformation section order variability",
    "def_preval_type_nunique": "Distinct types in prevalidation section",
    "def_trans_type_nunique": "Distinct types in transformation section",
    "def_type_crossreference_count": "Crossreference-type component count",
    "def_type_dataobject_count": "Dataobject-type component count",
    "def_type_dataextract_count": "Dataextract-type component count",
    "def_type_dataset_count": "Dataset-type component count",
    "def_type_query_count": "Query-type component count",
    "def_type_table_count": "Table-type component count",
    "hist_global_prior_count": "Global prior execution count",
    "hist_global_prior_mean_sec": "Global prior mean duration",
    "hist_global_prior_median_sec": "Global prior median duration",
    "hist_global_prior_std_sec": "Global prior duration variability",
    "hist_loader_prior_count": "Loader prior execution count",
    "hist_loader_prior_mean_sec": "Loader prior mean duration",
    "hist_loader_prior_median_sec": "Loader prior median duration",
    "hist_loader_prior_std_sec": "Loader prior duration variability",
    "hist_loader_conn_prior_count": "Loader+connection prior count",
    "hist_loader_conn_prior_mean_sec": "Loader+connection prior mean",
    "hist_loader_conn_prior_median_sec": "Loader+connection prior median",
    "hist_loader_conn_prior_std_sec": "Loader+connection prior variability",
    "hist_loader_dataset_prior_count": "Loader+dataset prior count",
    "hist_loader_dataset_prior_mean_sec": "Loader+dataset prior mean",
    "hist_loader_dataset_prior_median_sec": "Loader+dataset prior median",
    "hist_loader_dataset_prior_std_sec": "Loader+dataset prior variability",
    "hist_composite_prior_count": "Composite identity prior count",
    "hist_composite_prior_mean_sec": "Composite prior mean duration",
    "hist_composite_prior_median_sec": "Composite prior median duration",
    "hist_composite_prior_std_sec": "Composite prior duration variability",
    "hist_prev_loader_duration_sec": "Previous loader duration",
    "hist_time_since_prev_loader_sec": "Time since previous loader execution",
}


class ExplainError(RuntimeError):
    """Raised when a SHAP explanation cannot be produced."""


def shap_version() -> str:
    import shap

    return str(shap.__version__)


def get_or_build_explainer(artifact: dict, cached: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    """Return (TreeExplainer, cache_entry). The explainer is deterministic and
    model-bound, so it is built once per service and reused across requests."""
    if cached is not None and cached.get("model_version") == artifact["metadata"]["model_version"]:
        return cached["explainer"], cached
    import shap

    explainer = shap.TreeExplainer(artifact["estimator"])
    cache = {
        "explainer": explainer,
        "model_version": artifact["metadata"]["model_version"],
    }
    return explainer, cache


def encoded_column_mapping(preprocessor: Any, metadata: dict[str, Any]) -> list[tuple[str, int]]:
    """Map every encoded model column to its registered feature.

    Derived from the FITTED ColumnTransformer layout, not from name strings:
      - numeric pipeline outputs map 1:1, in order, to metadata numeric_features
      - the categorical pipeline's one-hot block maps (entirely) to the
        registered categorical feature(s) it encodes
    Every encoded index appears exactly once; contributions therefore sum
    back to the model output without loss.
    """
    feature_names = list(metadata["feature_names_in_order"])
    numeric = list(metadata["numeric_features"])
    categorical = list(metadata["categorical_features"])

    encoded_names = [str(x) for x in preprocessor.get_feature_names_out()]
    n_out = len(encoded_names)

    mapping: list[tuple[str, int]] = []
    cursor = 0
    for name, _trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        # Output width of this transformer block via output_indices_ (fitted truth).
        block = preprocessor.output_indices_[name]
        width = block.stop - block.start
        if name == "num":
            if len(cols) != width:
                raise ExplainError(
                    f"Numeric block width {width} != {len(cols)} numeric features; layout changed."
                )
            for col in cols:
                col = str(col)
                if col not in feature_names:
                    raise ExplainError(f"Numeric encoded column {col!r} is not a registered feature.")
                mapping.append((col, cursor))
                cursor += 1
        elif name == "cat":
            if len(categorical) != 1:
                # General case: slice this block's encoded names to their
                # registered prefixes. With the current single-categorical
                # layout this branch is defensive only.
                block_names = encoded_names[cursor : cursor + width]
                for enc in block_names:
                    owner = next(
                        (f for f in categorical if enc[len("cat__") :].startswith(f)),
                        None,
                    )
                    if owner is None:
                        raise ExplainError(f"Cannot map encoded column {enc!r} to a registered feature.")
                    mapping.append((owner, cursor))
                    cursor += 1
            else:
                owner = categorical[0]
                block_names = encoded_names[cursor : cursor + width]
                prefix = f"cat__{owner}_"
                for enc in block_names:
                    if not enc.startswith(prefix):
                        raise ExplainError(
                            f"Encoded column {enc!r} does not belong to categorical feature {owner!r}."
                        )
                    mapping.append((owner, cursor))
                    cursor += 1
        else:
            raise ExplainError(f"Unexpected preprocessor block {name!r}; refusing to explain.")
    if cursor != n_out:
        raise ExplainError(f"Mapped {cursor} encoded columns but model matrix has {n_out}.")
    covered = {reg for reg, _ in mapping}
    if covered != set(feature_names):
        missing = sorted(set(feature_names) - covered)
        raise ExplainError(f"Registered features missing from encoded mapping: {missing}")
    return mapping


def local_explanation(artifact: dict, X: pd.DataFrame, explainer: Any) -> dict[str, Any]:
    """SHAP explanation for one prediction-time feature vector (log1p space).

    Uses the EXACT transformed matrix the point prediction used. Returns the
    full ranked contribution list plus the top-K summary for the UI.
    """
    meta = artifact["metadata"]
    feature_names = list(meta["feature_names_in_order"])
    if list(X.columns) != feature_names or len(X) != 1:
        raise ExplainError("Explanation requires the single-row exact-artifact feature vector.")

    from dltime.modeling import transform_with_preprocessor

    Xt = transform_with_preprocessor(
        artifact["preprocessor"], X, meta["categorical_features"], meta["numeric_features"]
    )

    sv = explainer.shap_values(Xt)
    arr = np.asarray(sv)
    if arr.ndim == 3:  # some shap versions: (n_samples, n_features, n_outputs)
        arr = arr[:, :, 0]
    if arr.ndim != 2 or arr.shape[0] != 1:
        raise ExplainError(f"Unexpected SHAP output shape {np.shape(sv)}.")
    row = arr[0]

    base_raw = explainer.expected_value
    base = float(np.asarray(base_raw).ravel()[0])
    log_output = float(artifact["estimator"].predict(Xt)[0])
    additivity_err = abs(base + float(row.sum()) - log_output)
    if additivity_err > 1e-6:
        raise ExplainError(
            f"SHAP additivity violated: base+sum={base + float(row.sum()):.9f} vs model {log_output:.9f}"
        )

    mapping = encoded_column_mapping(artifact["preprocessor"], meta)
    if len(mapping) != row.shape[0]:
        raise ExplainError("Encoded-column mapping does not cover the model matrix.")

    aggregated: dict[str, float] = {name: 0.0 for name in feature_names}
    for registered, idx in mapping:
        aggregated[registered] += float(row[idx])

    row_raw = X.iloc[0]
    features: list[dict[str, Any]] = []
    for name in feature_names:
        val = aggregated[name]
        direction = "higher" if val > 0 else ("lower" if val < 0 else "neutral")
        raw_value = row_raw[name]
        features.append(
            {
                "feature": name,
                "display_name": DISPLAY_NAMES.get(name, name),
                "value": _jsonable(raw_value),
                "shap_value_log1p": round(val, 6),
                "direction": direction,
            }
        )
    features.sort(key=lambda f: -abs(f["shap_value_log1p"]))
    for rank, f in enumerate(features, start=1):
        f["contribution_rank"] = rank

    return {
        "method": "shap_tree_explainer",
        "shap_version": shap_version(),
        "model_version": meta["model_version"],
        "output_space": (
            "log1p(total_execution_seconds): contributions are additive in the "
            "model's log-scale output; the ETA in seconds is shown separately"
        ),
        "base_value_log1p": round(base, 6),
        "sum_contributions_log1p": round(float(row.sum()), 6),
        "model_output_log1p": round(log_output, 6),
        "additivity_abs_error": float(f"{additivity_err:.3e}"),
        "interpretation": (
            "Model contributions, not causal effects. Each entry shows how the "
            "feature moved the model's log-scale ETA for this prediction."
        ),
        "top_features": features[:TOP_K],
        "features": features,
        "human_readable_summary": human_readable_summary({"features": features}),
    }


def human_readable_summary(payload: dict[str, Any]) -> str:
    """Deterministic plain-language summary built from the ACTUAL top SHAP
    entries (Task 10.1; Prompt 3 §6/§7 wording policy). No new computation,
    no invented evidence, no causal wording: every clause is template-filled
    from real returned directions and magnitudes; the second-largest real
    entry is quantified as 'also'. Contribution language only ('pushes the
    estimate higher/lower', 'was associated with') — never 'caused'/'will
    cause'/'guarantees' — and always closed by the model-attribution
    disclaimer distinguishing model contribution from causal explanation."""
    ranked = [f for f in payload.get("features", []) if f.get("direction") != "neutral"]
    if not ranked:
        return (
            "No single feature dominated this estimate: the model's historical "
            "baseline dominated, with only small individual adjustments."
        )
    top = ranked[0]
    up = [f for f in ranked if f["direction"] == "higher"]
    down = [f for f in ranked if f["direction"] == "lower"]
    parts: list[str] = [
        f"The strongest model signal here is {top['display_name']} "
        f"(value {top['value']}), pushing the estimate {top['direction']}."
    ]
    for entry, label in ((up[1] if len(up) > 1 else None, "upward"), (down[0] if down else None, "downward")):
        if entry is not None and entry is not top:
            parts.append(
                f"{entry['display_name']} also pushes {label} "
                f"(value {entry['value']})."
            )
    if up and down:
        parts.append(
            f"On balance, {len(up)} signal(s) pushed the estimate upward and "
            f"{len(down)} pushed it downward among the largest model contributors."
        )
    parts.append(
        "These are model signals from historical evidence — model "
        "contributions, not causal explanations."
    )
    return " ".join(parts)


def _jsonable(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def global_explanation(artifact: dict, X: pd.DataFrame, explainer: Any, *, subset_description: str) -> dict[str, Any]:
    """Lightweight global summary: mean |SHAP| per registered feature over a
    FIXED, documented chronological subset (Task 03 final holdout — rows the
    artifact never trained on). Not a Model Laboratory; deterministic."""
    meta = artifact["metadata"]
    feature_names = list(meta["feature_names_in_order"])
    if list(X.columns) != feature_names or len(X) == 0:
        raise ExplainError("Global explanation requires the exact-artifact feature frame.")

    from dltime.modeling import transform_with_preprocessor

    Xt = transform_with_preprocessor(
        artifact["preprocessor"], X, meta["categorical_features"], meta["numeric_features"]
    )
    sv = np.asarray(explainer.shap_values(Xt))
    if sv.ndim == 3:
        sv = sv[:, :, 0]
    if sv.ndim != 2 or sv.shape[1] != Xt.shape[1]:
        raise ExplainError(f"Unexpected SHAP output shape {np.shape(sv)} for global summary.")

    mapping = encoded_column_mapping(artifact["preprocessor"], meta)
    agg = np.zeros((sv.shape[0], len(feature_names)))
    col_of = {name: i for i, name in enumerate(feature_names)}
    for registered, idx in mapping:
        agg[:, col_of[registered]] += sv[:, idx]

    mean_abs = np.abs(agg).mean(axis=0)
    mean_signed = agg.mean(axis=0)
    order = np.argsort(-mean_abs)
    features = [
        {
            "feature": feature_names[i],
            "display_name": DISPLAY_NAMES.get(feature_names[i], feature_names[i]),
            "mean_abs_shap_log1p": round(float(mean_abs[i]), 6),
            "mean_signed_shap_log1p": round(float(mean_signed[i]), 6),
        }
        for i in order
    ]
    return {
        "method": "shap_tree_explainer_global_mean_abs",
        "shap_version": shap_version(),
        "model_version": meta["model_version"],
        "output_space": (
            "log1p(total_execution_seconds): mean absolute contribution to the "
            "model's log-scale output; not causal effects"
        ),
        "n_observations": int(len(X)),
        "subset": subset_description,
        "features": features,
    }


@dataclass
class ExplainService:
    """Cached-explainer accessor for the prediction service."""

    artifact: dict
    _cache: dict[str, Any] | None = None

    def explain(self, X: pd.DataFrame) -> dict[str, Any]:
        explainer, self._cache = get_or_build_explainer(self.artifact, self._cache)
        return local_explanation(self.artifact, X, explainer)
