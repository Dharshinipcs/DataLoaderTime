"""Stage intelligence serving (AD-31, corrected architecture; prompt §23-§24).

SUPERSEDES the prior-median stage intelligence (stage-intel-v2) and the
pre_load/data_loading alias policy (EXP-022 evidence is obsolete on the
corrected source: pre_load and data_loading are verified DIFFERENT stages).

The locked serving contract for a prediction context:

    FIVE ML STAGES  staging, prevalidation, transformation, pre_load,
                    data_loading -> predictions from the trained stage models
                    (dltime.stage_models), each reported with its model kind,
                    training count, and availability:
                        ML_PREDICTED   stage model served the estimate
                                       (all five stages; pre_load is a real
                                       fitted model with a disclosed
                                       SMALL-SAMPLE limitation, AD-31a)
                        ML_UNAVAILABLE stage model not implementable
                                       (insufficient eligible evidence,
                                       n < 12) or not loaded in this process
                    ML stage outputs are PREDICTIONS, never actuals.

    TWO APPROVAL STAGES  datamart_approval, dataloading_approval ->
                    contextual hierarchical STATISTICAL estimates
                    (dltime.approval_estimator): evidence tier, sample count,
                    explicit non-ML labelling (CONTEXTUAL_STATISTICAL).

Stage estimates for the five operational stages ARE the authoritative
served total's exact additive components (sum-of-five-stages policy,
EXP-038): the served TOTAL = staging + prevalidation + transformation +
pre_load + data_loading, with no floor and no learned adjustment. The two
approval estimates are contextual statistics and are NEVER summed into
the total. (This replaces the AD-19 'never summed' rule; recorded in
EXPERIMENT_LOG.md EXP-038.)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dltime.stage_models import STAGE_MIN_SMALL_SAMPLE_ROWS, STAGE_MIN_TRAIN_ROWS

STAGE_INTEL_VERSION = "stage-intel-v3-ad31"

ML_STAGES: tuple[str, ...] = ("staging", "prevalidation", "transformation", "pre_load", "data_loading")
APPROVAL_STAGES: tuple[str, ...] = ("datamart_approval", "dataloading_approval")
ALL_STAGES: tuple[str, ...] = ML_STAGES + APPROVAL_STAGES

# --------------------------------------------------------------------------
# Applicable-stage determination (user final spec, 2026-09-30)
# --------------------------------------------------------------------------
# Not every loader execution runs all seven stages. From the user's
# pre-execution entries we KNOW which stages are part of THIS execution:
#   - staging, pre_load, data_loading and the two approvals are ALWAYS part
#     of an execution (the actual loading path + its approvals);
#   - prevalidation and transformation run only when the user supplies the
#     corresponding pre-execution entries (display name or count).
# Stages that are not applicable are EXCLUDED from the stage view and the
# total — a 3-stage loader must never receive 5 stage timings.
APPLICABLE_ALWAYS: frozenset[str] = frozenset(
    ("staging", "pre_load", "data_loading", "datamart_approval", "dataloading_approval")
)
_APPLICABLE_CONDITIONAL: dict[str, tuple[str, ...]] = {
    "prevalidation": (
        "prevalidation_display_name",
        "preval_trans_display_name",  # legacy combined field
        "prevalidation_count",
    ),
    "transformation": (
        "transformation_display_name",
        "preval_trans_display_name",  # legacy combined field
        "transformation_count",
    ),
}


def determine_applicable_stages(config_overrides: dict | None) -> tuple[str, ...]:
    """The stages THIS execution will run, from the user's pre-execution entries.

    2026-10-02: the per-component rows (``preval_trans_components`` — one row
    per PREVALIDATION/TRANSFORMATION component with its calling_type) are the
    authoritative signal when present: a stage is applicable exactly when the
    user entered at least one component of that type. The scalar fallbacks
    (single display name / count fields) still work unchanged.
    """
    given = {
        k for k, v in (config_overrides or {}).items()
        if v is not None and not (isinstance(v, float) and pd.isna(v))
    }
    components = (config_overrides or {}).get("preval_trans_components") or ()
    component_types = {
        str(getattr(c, "calling_type", "") or "").upper() for c in components
    }
    applicable = list(APPLICABLE_ALWAYS)
    for stage, fields in _APPLICABLE_CONDITIONAL.items():
        if stage == "prevalidation" and "PREVALIDATION" in component_types:
            applicable.append(stage)
            continue
        if stage == "transformation" and "TRANSFORMATION" in component_types:
            applicable.append(stage)
            continue
        if any(f in given for f in fields):
            applicable.append(stage)
    return tuple(applicable)


def filter_stage_intel_to_applicable(stage_intel: dict, applicable: tuple[str, ...]) -> dict:
    """Keep only applicable stages in a stage-intel payload (non-destructive
    for other keys). Filtered-out stages are listed under ``not_applicable``
    for transparency."""
    if not stage_intel:
        return stage_intel
    stages = stage_intel.get("stages") or []
    kept = [s for s in stages if isinstance(s, dict) and s.get("stage") in applicable]
    dropped = [s.get("stage") for s in stages if isinstance(s, dict) and s.get("stage") not in applicable]
    out = dict(stage_intel)
    out["stages"] = kept
    out["applicable_stages"] = list(applicable)
    out["not_applicable_stages"] = [d for d in dropped if d]
    return out

# Stage model serving registry — SINGLE SOURCE OF TRUTH is
# dltime.stage_models.STAGE_MODEL_SERVING_KINDS (chronological selection
# records; holdout-MAE rule among ML candidates — AD-31a: every operational
# stage serves a real fitted ML model; the historical median baseline stays a
# registered non-ML reference only). pre_load: EXPERIMENTAL_SMALL_SAMPLE
# (n=19 — real fitted model with a disclosed small-sample limitation).
from dltime.stage_models import STAGE_MODEL_SERVING_KINDS as STAGE_MODEL_SERVING

STAGE_MODEL_SERVING_VERSION = "stage-serving-ad31a"


class AvailabilityState:
    ML_PREDICTED = "ML_PREDICTED"
    ML_UNAVAILABLE = "ML_UNAVAILABLE"
    CONTEXTUAL_STATISTICAL = "CONTEXTUAL_STATISTICAL"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class StageIntelResult:
    per_stage: dict[str, dict]
    version: str = STAGE_INTEL_VERSION


def stage_context_row(
    combined_executions: pd.DataFrame,
    context_execution_id,
    stage: str,
) -> pd.DataFrame | None:
    """The EXACT single context row a given stage's model was scored on.

    Same ordering (`Start_Time`, `LDR_Execution_Id` mergesort) and the same
    stage-specific ``stage_hist_*`` remap used inside
    `build_stage_intel_for_context` — extracted so the per-stage SHAP
    explanation (Prediction Studio insights) explains the identical feature
    vector the point prediction used. No second feature pipeline exists.
    Returns None when the context row cannot be located.
    """
    df = combined_executions.copy()
    df["_ts"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    ordered = df.sort_values(["_ts", "LDR_Execution_Id"], kind="mergesort").reset_index(drop=True)
    hits = ordered.index[ordered["LDR_Execution_Id"] == context_execution_id]
    if len(hits) == 0:
        return None
    ctx_pos = int(hits[0])
    ctx_row = ordered.iloc[[ctx_pos]]
    stage_specific_present = any(
        col.startswith("stage_hist_") and col.count("_") > 2 and col.split("_")[2] in ML_STAGES
        for col in ordered.columns
    )
    if stage_specific_present:
        ctx_row = ctx_row.copy()
        for col in list(ctx_row.columns):
            if col.startswith("stage_hist_") and not col.startswith(f"stage_hist_{stage}_"):
                specific = f"stage_hist_{stage}_{col[len('stage_hist_'):]}"
                if specific in ctx_row.columns:
                    ctx_row[col] = ctx_row[specific]
    return ctx_row


def build_stage_intel_for_context(
    combined_executions: pd.DataFrame,
    context_execution_id,
    stage_models: dict[str, object] | None = None,
) -> StageIntelResult:
    """Seven-stage intelligence for one prediction context (AD-31 contract).

    combined_executions MUST already contain the context row (stage targets
    NA, is_training_eligible=False) — identical contract to the historical
    engines, so strictly-prior guarantees hold for the approval estimator.

    stage_models: optional mapping stage -> fitted predictor object exposing
    ``predict_seconds(context_row_frame) -> float``. When absent (or for a
    NOT_IMPLEMENTED stage), ML stages report ML_UNAVAILABLE with the exact
    reason — never a fabricated number.
    """
    from dltime.approval_estimator import estimate_approval_stage
    from dltime.stage_targets import stage_target_column

    per_stage: dict[str, dict] = {}
    models = stage_models or {}

    # context evidence counts (strictly prior, training-eligible)
    df = combined_executions.copy()
    df["_ts"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    ordered = df.sort_values(["_ts", "LDR_Execution_Id"], kind="mergesort").reset_index(drop=True)
    # EXP-032 (Part J): stage-specific attach present?
    stage_specific_present = any(
        col.startswith("stage_hist_") and col.count("_") > 2 and col.split("_")[2] in ML_STAGES
        for col in ordered.columns
    )
    hits = ordered.index[ordered["LDR_Execution_Id"] == context_execution_id]
    ctx_pos = int(hits[0])
    usable = ordered["is_training_eligible"].fillna(False).astype(bool)

    for stage in ALL_STAGES:
        col = stage_target_column(stage)
        if col in ordered.columns:
            v = pd.to_numeric(ordered[col], errors="coerce")
            prior_mask = (ordered.index < ctx_pos) & usable & (v > 0)
            loader = ordered.loc[ctx_pos, "Loader_Name"]
            same_loader = (
                ordered["Loader_Name"].astype("string")
                == pd.Series([None if pd.isna(loader) else str(loader)], dtype="string")[0]
            )
            n_prior = int(prior_mask.sum())
            n_prior_loader = int((prior_mask & same_loader).sum())
        else:
            n_prior, n_prior_loader = 0, 0

        if stage in APPROVAL_STAGES:
            est = estimate_approval_stage(combined_executions, context_execution_id, stage)
            payload = est.to_payload()
            payload.update(
                {
                    "availability": (
                        AvailabilityState.CONTEXTUAL_STATISTICAL
                        if est.available
                        else AvailabilityState.UNAVAILABLE
                    ),
                    "prior_positive_count": n_prior,
                    "prior_loader_positive_count": n_prior_loader,
                    "policy_version": STAGE_INTEL_VERSION,
                }
            )
            per_stage[stage] = payload
            continue

        # ---- ML stage ----
        serving = dict(STAGE_MODEL_SERVING.get(stage, {}))
        predictor = models.get(stage)
        v2_unavailable_reason = None
        if predictor is not None and getattr(predictor, "version", "") == "stage-v2":
            # v2 predictors carry their OWN truth (selected family + real-row
            # training count on the filled dataset); the static registry is
            # superseded for display purposes — INCLUDING its availability
            # status: a stage the static registry marked NOT_ENOUGH_EVIDENCE
            # (staging on the old export) is AVAILABLE once a v2 predictor
            # exists for it.
            serving["model_kind"] = getattr(predictor, "family", serving.get("model_kind"))
            serving["n_train"] = getattr(predictor, "n_real", serving.get("n_train"))
            serving["status"] = (
                "EXPERIMENTAL_SMALL_SAMPLE"
                if predictor.n_real < STAGE_MIN_TRAIN_ROWS
                else "EXPERIMENTAL"
            )
        elif predictor is None:
            try:
                from dltime.stage_v2 import load_stage_v2_bundle

                _v2 = load_stage_v2_bundle()
                if _v2 is not None and stage in _v2.unavailable_stages:
                    v2_unavailable_reason = _v2.unavailable_stages[stage]
            except Exception:
                v2_unavailable_reason = None
        if serving.get("status") == "NOT_ENOUGH_EVIDENCE" or predictor is None:
            reason = (
                f"Stage model not implemented: only {serving.get('n_train')} eligible "
                f"training observations on the corrected source (minimum "
                f"{STAGE_MIN_SMALL_SAMPLE_ROWS} for even a small-sample "
                "chronological evaluation)."
                if serving.get("status") == "NOT_ENOUGH_EVIDENCE"
                else "Stage model not loaded in this process."
            )
            if v2_unavailable_reason:
                reason = v2_unavailable_reason
            per_stage[stage] = {
                "stage": stage,
                "estimate_seconds": None,
                "available": False,
                "method": "ml_stage_model",
                "availability": AvailabilityState.ML_UNAVAILABLE,
                "availability_reason": reason,
                "model_kind": serving.get("model_kind"),
                "model_status": serving.get("status"),
                "n_training_observations": serving.get("n_train"),
                "prior_positive_count": n_prior,
                "prior_loader_positive_count": n_prior_loader,
                "policy_version": STAGE_INTEL_VERSION,                    "note": (
                        "ML stage predictions are model outputs, never actuals. "
                        "With this stage unavailable the deterministic five-stage "
                        "sum is refused (no total is fabricated)."
                    ),
            }
        else:
            ctx_row = ordered.iloc[[ctx_pos]]
            # EXP-032 (Part J): each stage model must receive its OWN
            # stage-specific historical context. When the frame carries the
            # stage-specific attach (stage_hist_<stage>_*), the generic
            # stage_hist_* columns the predictor was trained on are filled
            # from THIS stage's history — no cross-stage overwrite.
            if stage_specific_present:
                ctx_row = ctx_row.copy()
                for col in list(ctx_row.columns):
                    if col.startswith("stage_hist_") and not col.startswith(f"stage_hist_{stage}_"):
                        specific = f"stage_hist_{stage}_{col[len('stage_hist_'):]}"
                        if specific in ctx_row.columns:
                            ctx_row[col] = ctx_row[specific]
            seconds = float(predictor.predict_seconds(ctx_row))
            small = serving.get("status") == "EXPERIMENTAL_SMALL_SAMPLE"
            per_stage[stage] = {
                "stage": stage,
                "estimate_seconds": round(seconds, 3),
                "available": True,
                "method": "ml_stage_model",
                "availability": AvailabilityState.ML_PREDICTED,
                "availability_reason": (
                    f"Stage ML prediction ({serving['model_kind']}, trained on "
                    f"{serving['n_train']} eligible observations"
                    + (
                        "; SMALL-SAMPLE stage — honest high-variance limitation "
                        "disclosed, model remains an experimental candidate"
                        if small
                        else ""
                    )
                    + "; experimental candidate, not promoted)."
                ),
                "model_kind": serving.get("model_kind"),
                "model_status": serving.get("status"),
                "n_training_observations": serving.get("n_train"),
                "prior_positive_count": n_prior,
                "prior_loader_positive_count": n_prior_loader,
                "policy_version": STAGE_INTEL_VERSION,                    "note": (
                        "ML stage prediction — a model output, never an actual "
                        "measurement. One of the five additive components of "
                        "the served total ETA."
                    ),
            }

    return StageIntelResult(per_stage=per_stage)


def summarize_stage_intel(per_stage: dict[str, dict]) -> dict:
    """Compact counts for the response header/tests."""
    states = [s.get("availability") for s in per_stage.values()]
    return {
        "ml_predicted": states.count(AvailabilityState.ML_PREDICTED),
        "ml_unavailable": states.count(AvailabilityState.ML_UNAVAILABLE),
        "contextual_statistical": states.count(AvailabilityState.CONTEXTUAL_STATISTICAL),
        "unavailable": states.count(AvailabilityState.UNAVAILABLE),
        "policy_version": STAGE_INTEL_VERSION,
    }
