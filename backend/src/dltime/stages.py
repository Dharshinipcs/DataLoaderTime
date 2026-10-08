"""Stage intelligence (Task 06, supplementary only) — CORRECTED DATA EDITION.

AD-31 (prompt §24) SUPERSEDES the old prior-median stage architecture and the
pre_load/data_loading alias: the corrected Loader_Executions.xlsx (2026-09-21,
EXP-028) carries REAL independent observation fields for staging and pre_load,
and pre_load/data_loading durations are verified DIFFERENT (only 4/36
co-observed rows identical). The locked stage inventory is now:

    FIVE ML stages      staging, prevalidation, transformation, pre_load,
                        data_loading  (stage targets: dltime.stage_targets)
    TWO approval stages datamart_approval, dataloading_approval
                        (contextual hierarchical STATISTICAL estimation only,
                        dltime.approval_estimator — never called ML stages)

Stage durations are POST_EXECUTION observations. They may be used for
training targets, strictly-prior historical features of LATER executions,
analytics and operational visibility — never as the current execution's
inputs, and never summed into the total ETA (AD-19 hard rule unchanged).

Chronology (AD-11 semantics, reused): for a context at time T, ONLY earlier
training-eligible executions contribute prior stage statistics; ordering is
Start_Time + LDR_Execution_Id (deterministic tie-break); the context row
itself carries NA stages and is_training_eligible=False so it can never enter
its own (or any) aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from dltime.historical import ChronologyError


class StageModelability(StrEnum):
    MODELABLE = "MODELABLE"
    ANALYTICS_ONLY = "ANALYTICS_ONLY"
    TOO_SPARSE = "TOO_SPARSE"
    NOT_OBSERVED = "NOT_OBSERVED"


@dataclass(frozen=True)
class StageDefinition:
    """One execution-level stage duration and where it comes from (AD-31)."""

    name: str
    target_column: str  # execution-level stage target column (stage_targets)
    kind: str  # "ml_stage" | "contextual_statistical"
    description: str = ""


# The corrected stage inventory (AD-31, EXP-028). Targets are built by
# dltime.stage_targets.build_stage_targets (unit conversion + span derivation
# + consistency validation). There is NO alias: pre_load and data_loading are
# independent stages on the corrected source.
STAGE_DEFINITIONS: tuple[StageDefinition, ...] = tuple(
    StageDefinition(
        spec.name,
        f"stage_{spec.name}_sec",
        spec.kind,
        spec.description,
    )
    for spec in __import__("dltime.stage_targets", fromlist=["STAGE_FIELD_SPECS"]).STAGE_FIELD_SPECS
)


@dataclass(frozen=True)
class StageConfig:
    """Evidence thresholds (configurable; documented in AD-19)."""

    modelable_min_observations: int = 30
    analytics_min_observations: int = 6
    # minimum strictly-prior positive observations before ANY estimate is made
    min_prior_for_estimate: int = 3
    high_cv_warning: float = 2.0
    low_evidence_warning: int = 10


def stage_duration_series(executions: pd.DataFrame, stage: StageDefinition) -> pd.Series:
    """Execution-level stage duration (seconds); NA where not observed.

    Reads the stage_targets column; raises a clear error when the stage
    targets have not been attached to the frame yet.
    """
    if stage.target_column not in executions.columns:
        raise ChronologyError(
            f"Stage '{stage.name}' target column missing: {stage.target_column}. "
            "Attach stage targets first (dltime.stage_targets.build_stage_targets)."
        )
    return pd.to_numeric(executions[stage.target_column], errors="coerce")


def classify_stage(positive_count: int, config: StageConfig) -> StageModelability:
    if positive_count <= 0:
        return StageModelability.NOT_OBSERVED
    if positive_count >= config.modelable_min_observations:
        return StageModelability.MODELABLE
    if positive_count >= config.analytics_min_observations:
        return StageModelability.ANALYTICS_ONLY
    return StageModelability.TOO_SPARSE


# ---------------------------------------------------------------- analytics


@dataclass
class StageAnalyticsResult:
    stages: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {
            "stages": self.stages,
            "config": self.config,
            "note": (
                "Stage analytics are supplementary operational intelligence. "
                "Stage estimates are NEVER summed into the total execution-time prediction."
            ),
        }


def build_stage_analytics(
    executions: pd.DataFrame,
    config: StageConfig | None = None,
) -> StageAnalyticsResult:
    """Dataset-wide stage statistics over reconstructed executions (real data)."""
    config = config or StageConfig()
    total = len(executions)
    eligible = executions["is_training_eligible"].fillna(False).astype(bool)
    rows: list[dict] = []
    for stage in STAGE_DEFINITIONS:
        v = stage_duration_series(executions, stage)
        pos = v[v > 0]
        n_pos = int(len(pos))
        cv = float(pos.std(ddof=1) / pos.mean()) if n_pos >= 2 and pos.mean() > 0 else None
        row = {
            "stage": stage.name,
            "source": stage.kind,
            "description": stage.description,
            "observation_count": n_pos,
            "non_positive_count": int((v <= 0).sum()),
            "coverage": round(n_pos / total, 4) if total else 0.0,
            "eligible_positive_count": int((pos.index.to_series().map(eligible)).sum()),
            "median_sec": round(float(pos.median()), 3) if n_pos else None,
            "p90_sec": round(float(pos.quantile(0.9)), 3) if n_pos else None,
            "max_sec": round(float(pos.max()), 3) if n_pos else None,
            "cv": round(cv, 3) if cv is not None else None,
            "modelability": classify_stage(n_pos, config).value,
        }
        rows.append(row)
    rows.sort(key=lambda r: -r["observation_count"])
    return StageAnalyticsResult(
        stages=rows,
        config={
            "modelable_min_observations": config.modelable_min_observations,
            "analytics_min_observations": config.analytics_min_observations,
        },
    )


# ------------------------------------------------- chronological stage history


@dataclass
class StageHistoryResult:
    """Prior-only stage history for ONE prediction context."""

    per_stage: dict[str, dict] = field(default_factory=dict)
    ordering: str = "Start_Time, tie-break LDR_Execution_Id (deterministic)"
    prior_filter: str = "training_eligible"


def build_stage_history_for_context(
    combined_executions: pd.DataFrame,
    context_execution_id,
    config: StageConfig | None = None,
) -> StageHistoryResult:
    """Prior-only stage statistics for a single prediction context row.

    combined_executions MUST already contain the context row appended
    (target-less, stages NA, is_training_eligible=False — the exact row built
    by PredictionService.build_context_row). Streaming prefix over the
    deterministic chronological order; the context and every later execution
    can never contribute (AD-11 semantics reused for stages).
    """
    config = config or StageConfig()
    df = combined_executions.copy()
    if "Start_Time" not in df.columns or "LDR_Execution_Id" not in df.columns:
        raise ChronologyError("Chronological ordering requires Start_Time and LDR_Execution_Id.")
    df["_start_ts"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    if df["_start_ts"].isna().any():
        raise ChronologyError("Every execution needs a Start_Time for chronological stage history.")
    ordered = df.sort_values(["_start_ts", "LDR_Execution_Id"], kind="mergesort").reset_index(drop=True)

    ctx_positions = ordered.index[ordered["LDR_Execution_Id"] == context_execution_id]
    if len(ctx_positions) != 1:
        raise ChronologyError(f"Context row {context_execution_id!r} not found exactly once.")
    ctx_pos = int(ctx_positions[0])

    usable = ordered["is_training_eligible"].fillna(False).astype(bool)

    per_stage: dict[str, dict] = {}
    for stage in STAGE_DEFINITIONS:
        v = stage_duration_series(ordered, stage)
        # strict prefix: strictly earlier + eligible + positive stage value
        prior_mask = (ordered.index < ctx_pos) & usable & (v > 0)
        prior_values = v[prior_mask]
        n_prior = int(len(prior_values))
        loader = ordered.loc[ctx_pos, "Loader_Name"]
        same_loader = ordered["Loader_Name"].astype("string") == pd.Series(
            [None if pd.isna(loader) else str(loader)], dtype="string"
        )[0]
        prior_loader_values = v[prior_mask & same_loader]

        estimate: float | None = None
        estimator: str | None = None
        warnings: list[str] = []
        modelability = classify_stage(int((v.dropna() > 0).sum()), config)
        if modelability is not StageModelability.MODELABLE:
            # Evidence-based gate (AD-19): ONLY stages classified MODELABLE on
            # the real dataset receive an estimate. ANALYTICS_ONLY / TOO_SPARSE
            # stages must explicitly report unavailability — never a fragile
            # median from a handful of observations.
            evidence = n_prior
            warnings.append(
                f"No estimate: stage is {modelability.value} on the dataset "
                f"(not enough historical evidence for a stable estimator)."
            )
        elif n_prior >= config.min_prior_for_estimate:
            if len(prior_loader_values) >= config.min_prior_for_estimate:
                estimate = float(prior_loader_values.median())
                estimator = "prior loader median (training-eligible, strictly earlier)"
                evidence = int(len(prior_loader_values))
            else:
                estimate = float(prior_values.median())
                estimator = "prior global median (training-eligible, strictly earlier)"
                evidence = n_prior
            if evidence < config.low_evidence_warning:
                warnings.append(
                    f"Low evidence: only {evidence} prior positive observation(s) for this stage."
                )
        else:
            evidence = n_prior
            warnings.append(
                "No estimate: fewer than "
                f"{config.min_prior_for_estimate} prior positive observations."
            )

        per_stage[stage.name] = {
            "stage": stage.name,
            "prior_positive_count": n_prior,
            "evidence_count": evidence,
            "estimate_seconds": round(estimate, 3) if estimate is not None else None,
            "estimator": estimator,
            "available": estimate is not None,
            "warnings": warnings,
        }

    return StageHistoryResult(per_stage=per_stage)


def stage_context_is_na(combined_executions: pd.DataFrame, context_execution_id) -> bool:
    """Guard: the context row itself must carry no stage observation."""
    row = combined_executions[combined_executions["LDR_Execution_Id"] == context_execution_id]
    if len(row) != 1:
        return False
    for stage in STAGE_DEFINITIONS:
        v = stage_duration_series(row, stage)
        value = v.iloc[0]
        if pd.notna(value):
            return False
    return True
