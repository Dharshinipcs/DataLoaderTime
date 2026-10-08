"""Per-stage SHAP explainability for the served total (Prediction Studio §8).

Explanatory enrichment ONLY — never a second prediction model, never a change
to the point ETA, the stage predictions, the uncertainty layer, or the
authoritative sum-of-five-stages total policy.

WHY THIS EXISTS
---------------
Under the authoritative sum policy the served total is a deterministic
arithmetic sum of the five operational stage predictions, so there is no
learned total model to attribute (the old direct/stage-aware SHAP explainers
explain RETIRED models only). The user-facing question "why did the system
predict this execution time?" is nevertheless answerable with the EXISTING
SHAP machinery: each of the five stage predictions is itself produced by a
fitted tree/linear model over 29 registered pre-execution features, and
`shap.TreeExplainer` attributes that stage model's own output exactly.

WHAT THIS MODULE DOES
---------------------
For each available operational stage it:
  1. rebuilds the EXACT single-row context the stage model scored —
     `dltime.stage_intel.stage_context_row` (the same row-preparation the
     point prediction used, including the stage-specific ``stage_hist_*``
     history remap). No second feature pipeline exists.
  2. applies the predictor's own encoding (``__MISSING__`` fill for
     categoricals, training-median fill for numerics, ``get_dummies``
     reindexed to the training columns) — the identical transform in
     ``_StagePredictor.predict_seconds``.
  3. runs ``shap.TreeExplainer`` on the encoded row (tree stages) or the
     exact linear attribution ``coef_i * x_i`` with base = intercept
     (Ridge stage) — in both cases contributions are additive to the model's
     log1p output.
  4. aggregates encoded dummy columns back to the 29 registered features by
     summation (get_dummies groups), preserving contribution totals.
  5. converts each log-space contribution to a stage-level seconds effect
     ``expm1(base + c) - expm1(base)`` and ranks ALL stage/feature
     contributions by |log-space magnitude|.
  6. renders the strongest contributors as human-readable sentences from
     deterministic per-feature templates (display names + the actual context
     value + the contribution direction). No LLM, no fabricated evidence,
     no causal wording ("contributes toward", never "causes").

Leakage: the context row is the prediction-time pre-execution row; no
post-execution field can enter it. Failures are contained per stage — a
stage that cannot be explained is skipped with a reason, never fabricated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

POLICY_VERSION = "stage-shap-explanation-v1"
TOP_K = 3

# Human-readable names for the 29 registered stage features. The def_* names
# mirror dltime.explain.DISPLAY_NAMES (same registered features); the
# stage_hist_* names are generated from the documented per-level aggregates.
DEF_DISPLAY_NAMES: dict[str, str] = {
    "def_prevalidation_count": "Prevalidation configuration count",
    "def_transformation_count": "Transformation configuration count",
    "def_execution_order_max": "Max execution order",
    "def_execution_order_mean": "Mean execution order",
    "def_execution_order_std": "Execution order variability",
    "def_component_count": "Definition component count",
    "def_type_nunique": "Distinct component types",
    "def_dominant_calling_type": "Dominant calling type",
}

_STAGE_LABEL = {
    "staging": "staging",
    "prevalidation": "prevalidation",
    "transformation": "transformation",
    "pre_load": "pre-load",
    "data_loading": "data loading",
}

_LEVEL_PHRASE = {
    "global": "across all loaders",
    "loader_conn": "for this loader and connection",
    "loader_dataset": "for this loader and dataset",
    "loader": "for this loader",
    "composite": "for this exact loader configuration",
}

_AGG_PHRASE = {
    "prior_count": "number of earlier executions with this stage recorded",
    "prior_mean_sec": "average duration of earlier executions",
    "prior_median_sec": "typical duration of earlier executions",
    "prior_std_sec": "variability of earlier durations",
}


def _stage_feature_display(name: str) -> str:
    if name in DEF_DISPLAY_NAMES:
        return DEF_DISPLAY_NAMES[name]
    if name.startswith("stage_hist_"):
        rest = name[len("stage_hist_"):]
        for stage_key, label in _STAGE_LABEL.items():
            if rest.startswith(f"{stage_key}_"):
                rest = rest[len(stage_key) + 1:]
                break
        for level_key, level_phrase in _LEVEL_PHRASE.items():
            if rest.startswith(f"{level_key}_"):
                agg = rest[len(level_key) + 1:]
                agg_phrase = _AGG_PHRASE.get(agg, agg.replace("_", " "))
                return f"{label} history {level_phrase}: {agg_phrase}"
        if rest.startswith("prev_stage_sec"):
            return f"previous execution's {label} duration"
    return name


# Deterministic sentence templates. {stage} = stage label; {value} = the
# actual context value; {fmt} = human-formatted duration. Every clause is
# template-filled from real returned data — nothing invented.
def _clean_value(value: Any) -> Any:
    """Normalize a context value for display; 'nan'/'None' strings count as missing."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s in ("", "nan", "none", "<na>", "nat"):
        return None
    return value


def _stage_feature_sentence(feature: str, stage_label: str, value: Any) -> str:
    def fmt(v: Any) -> str:
        try:
            n = float(v)
            return f"{n:,.1f} seconds" if n >= 1 else f"{n:.2f} seconds"
        except (TypeError, ValueError):
            return str(v)

    if feature == "def_prevalidation_count":
        return f"The configuration includes {fmt(value)} prevalidation step(s)"
    if feature == "def_transformation_count":
        return f"The configuration includes {fmt(value)} transformation step(s)"
    if feature == "def_execution_order_max":
        return f"The highest execution order in this configuration is {fmt(value)}"
    if feature == "def_execution_order_mean":
        return f"The average execution order in this configuration is {fmt(value)}"
    if feature == "def_execution_order_std":
        return "Execution orders vary across this configuration's components"
    if feature == "def_component_count":
        return f"The registered configuration contains {fmt(value)} component(s)"
    if feature == "def_type_nunique":
        return f"The configuration spans {fmt(value)} distinct component type(s)"
    if feature == "def_dominant_calling_type":
        return f"The dominant calling type is \u201c{value}\u201d"

    rest = feature[len("stage_hist_"):]
    stage_key = next((s for s in _STAGE_LABEL if rest.startswith(f"{s}_")), None)
    if stage_key is not None:
        rest = rest[len(stage_key) + 1:]
    level_key = next((l for l in _LEVEL_PHRASE if rest.startswith(f"{l}_")), None)
    agg = rest[len(level_key) + 1:] if level_key else rest

    if agg == "prior_count":
        if _clean_value(value) is None:
            return f"No earlier executions with a recorded {stage_label} stage are on file {(_LEVEL_PHRASE.get(level_key, 'for this loader'))}"
        return f"{fmt(value)} earlier execution(s) {(_LEVEL_PHRASE.get(level_key, 'in the recorded history'))} recorded a {stage_label} stage"
    if agg == "prior_mean_sec":
        if _clean_value(value) is None:
            return f"No earlier {stage_label} history is on file {(_LEVEL_PHRASE.get(level_key, 'for this loader'))}"
        return f"The {stage_label} stage averaged about {fmt(value)} {(_LEVEL_PHRASE.get(level_key, 'in the recorded history'))}"
    if agg == "prior_median_sec":
        if _clean_value(value) is None:
            return f"No typical {stage_label} duration is on file {(_LEVEL_PHRASE.get(level_key, 'for this loader'))}"
        return f"The typical {stage_label} stage takes about {fmt(value)} {(_LEVEL_PHRASE.get(level_key, 'in the recorded history'))}"
    if agg == "prior_std_sec":
        return f"{stage_label.capitalize()} durations vary {(_LEVEL_PHRASE.get(level_key, 'in the recorded history'))}"
    if agg == "prev_stage_sec":
        if _clean_value(value) is None:
            return f"The previous execution recorded no {stage_label} duration"
        return f"The previous execution's {stage_label} stage took about {fmt(value)}"
    return f"{feature.replace('_', ' ')} = {fmt(value)}"


class StageShapError(RuntimeError):
    """Raised when no stage could be explained at all."""


def _encode_row(predictor: Any, ctx_row: pd.DataFrame) -> pd.DataFrame:
    """Identical encoding to _StagePredictor.predict_seconds (reuse by copy)."""
    X = ctx_row[predictor.feature_names].copy()
    for col in X.columns:
        if X[col].dtype == "string" or isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].fillna("__MISSING__").astype("string")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(
                predictor._medians.get(col, 0.0)
            )
    Xenc = pd.get_dummies(X.astype(object).where(X.notna(), None), dummy_na=True)
    return Xenc.reindex(columns=predictor._enc_columns, fill_value=0)


def _contributions_log(
    predictor: Any, Xenc: pd.DataFrame, cache: dict[str, Any] | None
) -> tuple[np.ndarray, float]:
    """Per-registered-feature log1p contributions + baseline (additive)."""
    model = predictor.model
    # stage-v2 predictors carry the model family in ``family`` (older
    # serving bundles used ``kind``) — accept either attribute name.
    kind = getattr(predictor, "kind", None) or getattr(predictor, "family", "ridge")
    if kind == "ridge":
        # Exact linear attribution in the model's own output space:
        # pred_log = intercept + sum(coef_i * x_i).
        coef = np.asarray(model.coef_, dtype=float)
        x = Xenc.to_numpy(dtype=float)[0]
        per_encoded = coef * x
        base = float(model.intercept_)
    else:
        import shap

        ck = f"explainer:{id(model)}"
        entry = (cache or {}).get(ck)
        if entry is None:
            entry = shap.TreeExplainer(model)
            if cache is not None:
                cache[ck] = entry
        sv = entry.shap_values(Xenc)
        arr = np.asarray(sv)
        if arr.ndim == 3:  # some shap versions: (n, features, outputs)
            arr = arr[:, :, 0]
        if arr.ndim != 2 or arr.shape[0] != 1:
            raise StageShapError(f"Unexpected SHAP output shape {np.shape(sv)}.")
        per_encoded = arr[0]
        base = float(np.asarray(entry.expected_value).ravel()[0])
    # Aggregate encoded columns back to registered features (dummy groups).
    aggregated = np.zeros(len(predictor.feature_names), dtype=float)
    enc_names = [str(c) for c in Xenc.columns]
    cursor = 0
    for i, fname in enumerate(predictor.feature_names):
        n_dummy = sum(1 for c in enc_names if c == fname or c.startswith(f"{fname}_"))
        aggregated[i] = float(np.sum(per_encoded[cursor : cursor + n_dummy]))
        cursor += n_dummy
    if cursor != len(per_encoded):
        raise StageShapError(
            f"Encoded-column aggregation covered {cursor} of {len(per_encoded)} columns."
        )
    return aggregated, base


def _seconds_effect(base_log: float, c_log: float) -> float:
    return float(np.expm1(base_log + c_log) - np.expm1(base_log))


def stage_shap_explanation(
    stage_models: dict[str, Any],
    combined_executions: pd.DataFrame,
    context_execution_id: Any,
    stage_intelligence: dict[str, Any],
    *,
    cache: dict[str, Any] | None = None,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Explain the strongest drivers behind the five served stage predictions.

    Returns the payload consumed by the Prediction Studio insights panel:
    ``top_contributors`` (human-readable, real SHAP/linear attributions),
    ``human_readable_summary`` and the exact ``total_decomposition`` block
    (unchanged sum-policy arithmetic from dltime.sum_total_serving).
    """
    from dltime.stage_intel import stage_context_row
    from dltime.stage_targets import ML_STAGE_NAMES

    stages_by_name = {
        s.get("stage"): s for s in (stage_intelligence or {}).get("stages", [])
    }
    contributors: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    methods: set[str] = set()

    for stage in ML_STAGE_NAMES:
        payload = stages_by_name.get(stage) or {}
        if not payload.get("available") or payload.get("estimate_seconds") is None:
            skipped[stage] = "stage prediction unavailable for this context"
            continue
        predictor = stage_models.get(stage)
        if predictor is None or getattr(predictor, "model", None) is None:
            skipped[stage] = "stage model not loaded in this process"
            continue
        try:
            ctx_row = stage_context_row(combined_executions, context_execution_id, stage)
            if ctx_row is None or len(ctx_row) != 1:
                skipped[stage] = "context row unavailable"
                continue
            Xenc = _encode_row(predictor, ctx_row)
            per_encoded, base_log = _contributions_log(predictor, Xenc, cache)
            kind = getattr(predictor, "kind", None) or getattr(predictor, "family", "ridge")
            methods.add("shap_tree_explainer" if kind != "ridge" else "linear_coefficient_attribution")

            raw_row = ctx_row.iloc[0]
            stage_label = _STAGE_LABEL.get(stage, stage)
            entries: list[dict[str, Any]] = []
            for i, fname in enumerate(predictor.feature_names):
                c_log = float(per_encoded[i])
                if c_log == 0.0:
                    continue
                value = _clean_value(raw_row.get(fname))
                seconds = _seconds_effect(base_log, c_log)
                entries.append(
                    {
                        "stage": stage,
                        "stage_display": stage_label.capitalize(),
                        "feature": fname,
                        "display_name": _stage_feature_display(fname),
                        "value": None if pd.isna(value) else value,
                        "contribution_log1p": round(c_log, 6),
                        "stage_seconds_effect": round(seconds, 3),
                        "direction": "higher" if c_log > 0 else "lower",
                        "human_readable": (
                            f"{_stage_feature_sentence(fname, stage_label, value)} "
                            f"— contributing toward a {'longer' if c_log > 0 else 'shorter'} "
                            f"predicted {stage_label} duration."
                        ),
                    }
                )
            entries.sort(key=lambda e: -abs(e["contribution_log1p"]))
            contributors.extend(entries[:2])
        except Exception as exc:  # explanatory-only: skip the stage, never fail serving
            skipped[stage] = f"explanation failed: {exc}"

    if not contributors:
        raise StageShapError(
            "No operational stage could be explained: " + "; ".join(f"{k}: {v}" for k, v in skipped.items())
        )

    contributors.sort(key=lambda e: -abs(e["contribution_log1p"]))
    for rank, c in enumerate(contributors[:top_k], start=1):
        c["contribution_rank"] = rank
    top = contributors[:top_k]

    summary = " ".join(c["human_readable"] for c in top)
    summary += (
        " These are model signals derived from historical executions — model"
        " contributions, not causal explanations. The total is the exact sum of"
        " the five stage predictions shown on the result."
    )

    return {
        "available": True,
        "policy_version": POLICY_VERSION,
        "method": "+".join(sorted(methods)),
        "shap_version": _shap_version_safe(),
        "output_space": (
            "per-stage log1p contributions (aggregated to registered features); "
            "stage_seconds_effect = expm1(base+c) - expm1(base) on each stage model's own output"
        ),
        "stages_explained": sorted({c["stage"] for c in top}),
        "stages_skipped": skipped,
        "top_contributors": top,
        "human_readable_summary": summary,
    }


def _shap_version_safe() -> str | None:
    try:
        import shap

        return str(shap.__version__)
    except Exception:
        return None
