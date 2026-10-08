"""Contextual hierarchical approval-stage estimation (AD-31, prompt §6).

The two approval stages (Datamart Approval, Dataloading Approval) are sparse
by construction (12 and 6 valid execution-level observations on the corrected
source, EXP-028 audit). The locked decision: NO standalone ML regression
models for these stages — a contextual hierarchical STATISTICAL estimator
over legitimate pre-execution context and strictly-prior historical
observations instead.

Evidence ladder (progressively broadens; the narrowest tier with at least
MIN_TIER_OBSERVATIONS positive strictly-prior observations wins):

    CONTEXT_GROUP     same (sprint, connection, loader, dataset) context
    LOADER_DATASET    same (loader, dataset)
    LOADER_CONNECTION same (loader, connection)
    LOADER            same loader
    COMPOSITE         same (dataset, sprint, loader)  [identity-order fallback]
    GLOBAL            all prior observations

Estimator: median (robust to the heavy right tail). NEVER a hardcoded
constant: an estimate exists only when a tier has evidence; otherwise the
result is unavailable with the exact reason (INSUFFICIENT_SAMPLE).

Leakage contract (AD-11/AD-31): only STRICTLY PRIOR training-eligible
executions contribute, ordered by (Start_Time, LDR_Execution_Id) exactly like
every other historical aggregate. The context row itself can never contribute
(it carries no approval observation and is not training-eligible).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

APPROVAL_ESTIMATOR_VERSION = "approval-hier-v1"

# 2026-09-30 (user final spec): approvals are sparse by construction — a
# loader with 4 executions of which 2 carry approvals MUST get its
# loader-specific statistical value. 2026-10-03 (user update): a SINGLE
# recorded approval timing for the loader is shown as that timing itself
# (median of one observation = the observation); from 2 recorded timings on,
# the robust median statistic is used. GLOBAL is the last resort and also
# accepts a single observation so an approval estimate exists whenever ANY
# historical approval exists (but a GLOBAL estimate is display-hidden —
# only the loader's own history is ever shown or summed).
MIN_TIER_OBSERVATIONS = 1
MIN_GLOBAL_OBSERVATIONS = 1


class ApprovalTier:
    CONTEXT_GROUP = "CONTEXT_GROUP"
    LOADER_DATASET = "LOADER_DATASET"
    LOADER_CONNECTION = "LOADER_CONNECTION"
    LOADER = "LOADER"
    COMPOSITE = "COMPOSITE"
    GLOBAL = "GLOBAL"
    UNAVAILABLE = "UNAVAILABLE"


# Ladder order: narrowest -> broadest (§6).
TIER_LADDER: tuple[str, ...] = (
    ApprovalTier.CONTEXT_GROUP,
    ApprovalTier.LOADER_DATASET,
    ApprovalTier.LOADER_CONNECTION,
    ApprovalTier.LOADER,
    ApprovalTier.COMPOSITE,
    ApprovalTier.GLOBAL,
)

_TIER_LABELS = {
    ApprovalTier.CONTEXT_GROUP: "Sprint + Connection + Loader + Dataset historical median",
    ApprovalTier.LOADER_DATASET: "Loader + Dataset historical median",
    ApprovalTier.LOADER_CONNECTION: "Loader + Connection historical median",
    ApprovalTier.LOADER: "Loader historical median",
    ApprovalTier.COMPOSITE: "Dataset + Sprint + Loader historical median",
    ApprovalTier.GLOBAL: "Global historical median",
}


@dataclass
class ApprovalEstimate:
    stage: str
    estimate_seconds: float | None
    tier: str
    tier_label: str | None
    n_evidence: int
    available: bool
    reason: str | None = None

    def to_payload(self) -> dict:
        # 2026-10-03 (user final spec): ONLY approval evidence that comes from
        # THIS loader's own recorded executions is shown and counted. Every
        # loader-matched tier (CONTEXT_GROUP / LOADER_DATASET /
        # LOADER_CONNECTION / LOADER / COMPOSITE all require the loader to
        # match, so they ARE this loader's own history) is included in the
        # total. A broad fallback (GLOBAL — other loaders' history) and an
        # unavailable estimate are hidden from the prediction result entirely:
        # no approval section is displayed when this loader has no recorded
        # approval history of its own.
        loader_own_history = self.tier != ApprovalTier.GLOBAL
        return {
            "stage": self.stage,
            "estimate_seconds": round(self.estimate_seconds, 3) if self.estimate_seconds is not None else None,
            "method": "contextual_hierarchical_statistical",
            "evidence_tier": self.tier,
            "evidence_tier_label": self.tier_label,
            "n_evidence": self.n_evidence,
            "available": self.available,
            "included_in_total": bool(self.available and loader_own_history),
            "total_exclusion_reason": (
                None
                if loader_own_history
                else (
                    "estimate comes from other loaders' history (GLOBAL fallback), "
                    "not this loader's own approval history — hidden from the result"
                    if self.available
                    else "no approval estimate available"
                )
            ),
            "reason": self.reason,
            "estimator_version": APPROVAL_ESTIMATOR_VERSION + "+loader-tier-total",
            "note": (
                "Contextual statistical estimate (median; a single recorded "
                "timing is used as itself) from strictly-prior executions of "
                "this loader — NOT an ML prediction. Included in the total and "
                "displayed ONLY when the evidence comes from this loader's own "
                "approval history; the GLOBAL fallback is disclosed in the "
                "audit payload but hidden from the result per the 2026-10-03 "
                "user final spec."
            ),
        }


def _prior_values(
    ordered: pd.DataFrame,
    value_col: str,
    ctx_pos: int,
    mask_extra: pd.Series | None = None,
) -> pd.Series:
    usable = ordered["is_training_eligible"].fillna(False).astype(bool)
    mask = (ordered.index < ctx_pos) & usable & (ordered[value_col] > 0)
    if mask_extra is not None:
        mask = mask & mask_extra
    return ordered.loc[mask, value_col]


def estimate_approval_stage(
    combined_executions: pd.DataFrame,
    context_execution_id,
    stage: str,
    *,
    value_column: str | None = None,
    min_tier_observations: int = MIN_TIER_OBSERVATIONS,
) -> ApprovalEstimate:
    """Hierarchical contextual estimate for one approval stage.

    combined_executions MUST already contain the context row appended
    (no approval observation, is_training_eligible=False), in the identical
    contract used by the stage/history engines. The frame is ordered here by
    (Start_Time, LDR_Execution_Id) — deterministic, strictly-prior only.
    """
    from dltime.stage_targets import stage_target_column

    col = value_column or stage_target_column(stage)
    if col not in combined_executions.columns:
        raise ValueError(f"Approval stage '{stage}' target column missing: {col}")
    for required in ("Start_Time", "LDR_Execution_Id", "is_training_eligible"):
        if required not in combined_executions.columns:
            raise ValueError(f"Approval estimation requires column '{required}'.")

    df = combined_executions.copy()
    df["_ts"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    if df["_ts"].isna().any():
        raise ValueError("Every execution needs a Start_Time for chronological approval history.")
    ordered = df.sort_values(["_ts", "LDR_Execution_Id"], kind="mergesort").reset_index(drop=True)

    hits = ordered.index[ordered["LDR_Execution_Id"] == context_execution_id]
    if len(hits) != 1:
        raise ValueError(f"Context row {context_execution_id!r} not found exactly once.")
    ctx_pos = int(hits[0])
    ctx = ordered.iloc[ctx_pos]

    def same(col_name: str) -> pd.Series:
        ctx_val = ctx[col_name]
        if pd.isna(ctx_val):
            return pd.Series(False, index=ordered.index)
        return ordered[col_name].astype("string") == str(ctx_val)

    sprint_m, conn_m, loader_m, dataset_m = same("Sprint"), same("execution_connection"), same("Loader_Name"), same("Dataset_display_name")

    tier_masks = {
        ApprovalTier.CONTEXT_GROUP: sprint_m & conn_m & loader_m & dataset_m,
        ApprovalTier.LOADER_DATASET: loader_m & dataset_m,
        ApprovalTier.LOADER_CONNECTION: loader_m & conn_m,
        ApprovalTier.LOADER: loader_m,
        ApprovalTier.COMPOSITE: dataset_m & sprint_m & loader_m,
        ApprovalTier.GLOBAL: pd.Series(True, index=ordered.index),
    }

    missing_evidence: list[str] = []
    for tier in TIER_LADDER:
        vals = _prior_values(ordered, col, ctx_pos, tier_masks[tier])
        n = int(len(vals))
        # GLOBAL is the final fallback: accept a single observation so an
        # estimate exists whenever ANY historical approval timing exists.
        needed = (
            min(min_tier_observations, MIN_GLOBAL_OBSERVATIONS)
            if tier == ApprovalTier.GLOBAL
            else min_tier_observations
        )
        if n >= needed:
            return ApprovalEstimate(
                stage=stage,
                estimate_seconds=float(vals.median()),
                tier=tier,
                tier_label=_TIER_LABELS[tier],
                n_evidence=n,
                available=True,
            )
        missing_evidence.append(f"{tier}={n}")

    return ApprovalEstimate(
        stage=stage,
        estimate_seconds=None,
        tier=ApprovalTier.UNAVAILABLE,
        tier_label=None,
        n_evidence=0,
        available=False,
        reason=(
            f"Insufficient prior evidence at every contextual tier "
            f"({', '.join(missing_evidence)}); minimum {min_tier_observations} "
            "strictly-prior observations required."
        ),
    )


def estimate_all_approvals(
    combined_executions: pd.DataFrame,
    context_execution_id,
    approval_stages: tuple[str, ...],
) -> dict[str, dict]:
    """Payload dict for every approval stage (in the given order)."""
    return {
        stage: estimate_approval_stage(combined_executions, context_execution_id, stage).to_payload()
        for stage in approval_stages
    }
