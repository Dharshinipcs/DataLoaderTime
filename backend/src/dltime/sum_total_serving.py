"""SUM-OF-FIVE-STAGES total serving — the authoritative served ETA.

FINAL PRODUCT DECISION (approved reversal of the AD-32 stage-aware serving;
evidence trail: EXP-036 clean holdout rejected the sum on accuracy, and the
product owner explicitly mandated the sum anyway for transparency/auditability
— see EXPERIMENT_LOG.md EXP-038):

    TOTAL (served/displayed)
      = Staging prediction
      + Prevalidation prediction
      + Transformation prediction
      + Pre-load prediction
      + Data Loading prediction

Properties (all mandatory):
  * DETERMINISTIC — no learned total model drives the served number; the
    displayed total is the exact arithmetic sum of the five DISPLAYED
    operational stage values (same rounded values, same order, every time).
  * APPROVALS EXCLUDED — datamart_approval / dataloading_approval remain
    contextual statistical estimates and NEVER enter the total.
  * NO FLOOR / NO HIDDEN ADJUSTMENT — the retired AD-34 physical-floor rule,
    the stage-aware total model, and the direct total model no longer touch
    the served ETA. Both retired estimators are still evaluated per request
    and returned as clearly-labelled reference fields (rollback/traceability
    evidence; never displayed as the ETA).
  * FAIL-CLOSED — if any operational stage lacks an ML prediction, no total
    is produced at all (no fabrication, no partial sum).

Leakage contract is UNCHANGED: every stage prediction is produced from
pre-execution context + strictly-prior stage-specific history by the existing
serving stage models; nothing post-execution enters any stage or the sum.

The stage-aware total artifacts/files (stage_aware_total.py,
stage_aware_serving.py, stage-aware-total-ad34.pkl) are KEPT UNCHANGED for
traceability and rollback evidence; they are no longer the serving total.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# 2026-09-30: version label reflects the v2 sum policy. This module is now
# the REFERENCE/rollback path (the v2 authoritative sum lives in
# dltime.stage_v2.sum_total_v2 and includes available approval estimates).
SUM_TOTAL_MODEL_VERSION = "sum-of-available-stages-v2"
SUM_TOTAL_ESTIMATOR = "deterministic_sum_of_operational_stage_predictions"

# The served total policy, verbatim for API consumers and docs.
SUM_TOTAL_POLICY = (
    "TOTAL = staging + prevalidation + transformation + pre_load + data_loading "
    "+ available approval estimates (deterministic sum of the five operational "
    "ML stage predictions; approvals enter when their contextual statistical "
    "estimates are available; no floor, no learned adjustment, no correction "
    "factor)."
)


class SumTotalError(RuntimeError):
    """Raised when the deterministic sum cannot be produced honestly."""


def sum_total_from_stage_intel(stage_intelligence: dict[str, Any]) -> dict[str, Any]:
    """Compute the authoritative served total from the Seven-Stage view.

    Sums ONLY available ML stage predictions (method == 'ml_stage_model')
    for the five operational stages, in the canonical display order, using
    the EXACT estimate values the response displays. Raises SumTotalError
    when any operational stage is unavailable (fail-closed; never a partial
    or padded sum).
    """
    from dltime.stage_intel import ML_STAGES

    stages = {s.get("stage"): s for s in (stage_intelligence or {}).get("stages", [])}

    values: dict[str, float] = {}
    unavailable: list[str] = []
    for stage in ML_STAGES:
        entry = stages.get(stage)
        if (
            entry is None
            or not entry.get("available")
            or entry.get("method") != "ml_stage_model"
            or entry.get("estimate_seconds") is None
        ):
            unavailable.append(stage)
            continue
        v = float(entry["estimate_seconds"])
        if not np.isfinite(v):
            unavailable.append(stage)
            continue
        values[stage] = v

    approvals_excluded = [
        {"stage": s.get("stage"), "estimate_seconds": s.get("estimate_seconds")}
        for name, s in stages.items()
        if name not in ML_STAGES
    ]

    if unavailable:
        raise SumTotalError(
            "Total unavailable: operational stage(s) without an ML prediction: "
            f"{sorted(unavailable)}. The sum of five stage predictions is "
            "refused rather than fabricated."
        )

    # Deterministic order = the canonical ML stage order (display order).
    total = float(sum(values[stage] for stage in ML_STAGES))
    return {
        "model_version": SUM_TOTAL_MODEL_VERSION,
        "estimator": SUM_TOTAL_ESTIMATOR,
        "sum_available": True,
        "ml_stage_sum_seconds": total,
        "ml_stage_values_seconds": values,
        "ml_stages_summed": list(ML_STAGES),
        "approvals_excluded": approvals_excluded,
        "policy": SUM_TOTAL_POLICY,
    }


def sum_total_explanation(
    stage_intelligence: dict[str, Any], total_seconds: float
) -> dict[str, Any]:
    """Per-stage explanation of the served total (Task 08 infrastructure reuse).

    The served total is a deterministic sum, so the honest explanation IS the
    decomposition: each applicable operational stage prediction is a literal
    additive component of the displayed total (exact in LINEAR seconds), and
    each available approval estimate is included as a statistical component
    (user final spec 2026-09-30: approvals ARE part of the total). No SHAP is
    run for the total: there is no learned total model to attribute. The
    stage models' own SHAP infrastructure (explain.py) remains available for
    stage-model diagnostics.

    Additivity contract for consumers: base_value_seconds +
    sum_contributions_seconds == model_output_seconds exactly (0.0 error).
    """
    from dltime.stage_intel import ML_STAGES

    labels = {
        "staging": "Staging prediction",
        "prevalidation": "Prevalidation prediction",
        "transformation": "Transformation prediction",
        "pre_load": "Pre-load prediction",
        "data_loading": "Data Loading prediction",
        "datamart_approval": "Datamart Approval estimate",
        "dataloading_approval": "Dataloading Approval estimate",
    }

    stages_list = (stage_intelligence or {}).get("stages", [])
    stages = {s.get("stage"): s for s in stages_list if isinstance(s, dict)}
    applicable = stage_intelligence.get("applicable_stages")
    # Applicable set explicitly present -> the new contract (approvals that
    # are applicable ARE included components). Absent -> legacy payload:
    # operational stages only, approvals rendered as explicit non-components
    # (the pre-2026-09-30 behaviour).
    legacy_mode = applicable is None
    applicable_set = set(applicable) if applicable is not None else set(ML_STAGES)

    values: dict[str, float] = {}
    for stage in ML_STAGES:
        if stage not in applicable_set:
            continue
        entry = stages.get(stage)
        if (
            entry is not None
            and entry.get("available")
            and entry.get("estimate_seconds") is not None
        ):
            v = float(entry["estimate_seconds"])
            if np.isfinite(v):
                values[stage] = v

    approval_values: dict[str, float] = {}
    non_components: list[dict[str, Any]] = []
    for name in ("datamart_approval", "dataloading_approval"):
        entry = stages.get(name)
        has_value = (
            entry is not None
            and entry.get("available")
            and entry.get("estimate_seconds") is not None
            and np.isfinite(float(entry["estimate_seconds"]))
        )
        # 2026-10-02 (user final spec): only LOADER-tier approval evidence (the
        # loader's OWN approval history) joins the total; broader-context
        # estimates render as explicit non-components.
        loader_tier_only = entry is not None and entry.get("included_in_total") is not False
        if name in applicable_set and has_value and loader_tier_only:
            approval_values[name] = float(entry["estimate_seconds"])
        elif has_value and (legacy_mode or not loader_tier_only):
            non_components.append(
                {
                    "feature": f"pred_stage_{name}_sec",
                    "display_name": labels.get(name, name),
                    "value": float(entry["estimate_seconds"]),
                    "included_in_total": False,
                    "contribution_seconds": 0.0,
                    "direction": (
                        "not part of the total (broader-context history, not this loader's own approvals)"
                        if not legacy_mode
                        else "not part of the total"
                    ),
                    "contribution_rank": 0,
                }
            )

    if not values:
        raise SumTotalError(
            "Explanation unavailable: no applicable operational stage prediction "
            "to decompose."
        )

    component_total = float(sum(values.values()) + sum(approval_values.values()))
    total = float(total_seconds)
    # Defensive: the explanation must describe the SAME total that is served.
    if abs(component_total - total) > 1e-6:
        raise SumTotalError("Explanation total disagrees with the served total.")

    def _component(stage: str, value: float, rank: int) -> dict[str, Any]:
        is_approval = stage in ("datamart_approval", "dataloading_approval")
        return {
            "feature": f"pred_stage_{stage}_sec",
            "display_name": labels.get(stage, stage),
            "value": value,
            "included_in_total": True,
            "contribution_seconds": round(value, 6),
            "direction": (
                "approval component (historical median statistic)"
                if is_approval
                else "component of the total"
            ),
            "contribution_rank": rank,
        }

    merged = dict(values)
    merged.update(approval_values)
    ranked = sorted(merged.items(), key=lambda kv: -abs(kv[1]))
    top = [_component(s, v, i) for i, (s, v) in enumerate(ranked, start=1)]

    sum_text = " + ".join(f"{labels[s]} {v:g}s" for s, v in ranked)
    human = (
        f"The predicted total is the exact sum of the applicable stage "
        f"predictions plus the approval estimates backed by this loader's own "
        f"approval history: {sum_text} = {total:g}s. Operational stages are ML "
        f"model outputs; approvals are historical-median statistics."
    )

    return {
        "method": "deterministic_stage_sum",
        "model_version": SUM_TOTAL_MODEL_VERSION,
        "estimator": SUM_TOTAL_ESTIMATOR,
        "output_space": (
            "seconds: the total is exactly the sum of the applicable operational "
            "stage predictions plus the available approval estimates (linear "
            "additivity of the displayed values; no learned total model and no "
            "log-space attribution is involved)"
        ),
        "base_value_seconds": 0.0,
        "sum_contributions_seconds": round(total, 6),
        "model_output_seconds": round(total, 6),
        "model_output_log1p": float(np.log1p(total)),
        "additivity_abs_error": 0.0,
        "interpretation": (
            "Each entry below is one of the applicable stage predictions or "
            "approval estimates that sum exactly to the total — actual model / "
            "statistical outputs, not causal effects and not guarantees."
        ),
        "top_features": top,
        "features": top + non_components,
        "human_readable_summary": human,
    }
