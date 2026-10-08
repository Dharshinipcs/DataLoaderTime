"""Training eligibility layer (Part F).

Separates three tiers, each with explicit, auditable reasons:

    valid_execution      structurally usable execution record
                         (complete canonical identity, known status)
    target_valid         valid execution AND target exists, finite, positive
    training_eligible    target_valid AND status passes the explicit,
                         documented training-status policy

The status policy is NOT invented silently. The default policy below is the
evidence-backed v1 position (EXPERIMENT_LOG.md EXP-003): completed executions
count outcomes; in-flight/interrupted ones do not. It is data-driven evidence
only after revalidation, and remains configurable.

Extreme long durations are NEVER removed here (locked rule: do not discard
extremes merely for being extreme).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dltime.resolution import ResolutionStatus


class ExecutionStatus:
    """Known statuses in the real data (validated, EXPERIMENT_LOG.md EXP-001)."""

    STOPPED = "STOPPED"
    SUCCEED = "SUCCEED"
    COMPLETED_WITH_WARNING = "COMPLETED_WITH_WARNING"
    COMPLETED_WITH_ERROR = "COMPLETED_WITH_ERROR"
    COMPLETED_WITH_ERROR_AND_WARNING = "COMPLETED_WITH_ERROR_AND_WARNING"
    KILLED = "KILLED"
    ERROR = "ERROR"


COMPLETED_STATUSES = frozenset(
    {
        ExecutionStatus.SUCCEED,
        ExecutionStatus.COMPLETED_WITH_WARNING,
        ExecutionStatus.COMPLETED_WITH_ERROR,
        ExecutionStatus.COMPLETED_WITH_ERROR_AND_WARNING,
        ExecutionStatus.ERROR,
    }
)
INTERRUPTED_STATUSES = frozenset({ExecutionStatus.STOPPED, ExecutionStatus.KILLED})

# v1 default training-status policy (documented, configurable, evidence-backed).
DEFAULT_TRAINING_STATUSES: frozenset[str] = COMPLETED_STATUSES

# AD-31 corrected-data regime (EXP-028 audit): identity completeness on the
# corrected source means (Loader_Name, Sprint). Dataset_display_name is
# populated on only 36/759 raw rows; requiring it would strand the foundation
# at 10 eligible executions. Documented adaptation — never a silent relaxation.
CORRECTED_DATA_IDENTITY_COLUMNS: tuple[str, ...] = ("Loader_Name", "Sprint")


@dataclass
class EligibilityConfig:
    """Configurable eligibility policy (no hard-coded magic thresholds)."""

    training_statuses: frozenset[str] = DEFAULT_TRAINING_STATUSES
    require_resolved_connection: bool = False  # identity completeness is required instead
    require_positive_target: bool = True
    require_finite_target: bool = True
    require_complete_identity: bool = True
    # Which identity columns must be present for a valid execution. The legacy
    # regime (AD-31-corrected: full canonical identity incl. dataset) is the
    # default; the corrected-data regime (AD-31, EXP-028) narrows this to
    # (Loader_Name, Sprint) because Dataset_display_name is populated on only
    # 36/759 raw rows of the corrected source — a documented data-reality
    # adaptation, never a silent relaxation.
    required_identity_columns: tuple[str, ...] = (
        "Loader_Name",
        "Sprint",
        "Dataset_display_name",
    )


@dataclass
class EligibilityResult:
    """Per-execution eligibility with explicit machine-readable reasons."""

    frame: pd.DataFrame  # executions with eligibility columns
    counts: dict[str, int] = field(default_factory=dict)
    reason_counts: dict[str, int] = field(default_factory=dict)


def apply_eligibility(
    executions: pd.DataFrame,
    config: EligibilityConfig | None = None,
    *,
    resolution_status_col: str = "resolution_status",
) -> EligibilityResult:
    """Annotate reconstructed executions with the three eligibility tiers.

    Adds columns:
        is_valid_execution     structural validity (identity + known status)
        is_target_valid        valid execution AND usable target
        is_training_eligible   target_valid AND status policy AND config gates
        eligibility_reasons    ';'-joined refusal reasons ('' when eligible)

    Expects:
        resolution_status column (from dltime.resolution.annotate_raw_rows,
        aggregated to execution level) when present.
    """
    cfg = config or EligibilityConfig()
    df = executions.copy()

    reasons: list[str] = [[] for _ in range(len(df))]

    # --- identity completeness ------------------------------------------------
    identity_labels = {
        "Loader_Name": "missing_loader_name",
        "Sprint": "missing_sprint",
        "Dataset_display_name": "missing_dataset",
    }
    identity_cols = {
        col: identity_labels[col]
        for col in cfg.required_identity_columns
        if col in identity_labels
    }
    for col, reason in identity_cols.items():
        if col in df.columns:
            missing = df[col].isna() | (df[col].astype("string").str.strip().eq(""))
            for i in df.index[missing]:
                reasons[df.index.get_loc(i)].append(reason)

    # --- known status ----------------------------------------------------------
    if "LDR_Status" in df.columns:
        unknown_status = ~df["LDR_Status"].isin(COMPLETED_STATUSES | INTERRUPTED_STATUSES)
        for i in df.index[unknown_status]:
            reasons[df.index.get_loc(i)].append("unknown_status")

    # --- connection resolution (recorded, only gating when configured) ---------
    if resolution_status_col in df.columns and cfg.require_resolved_connection:
        unres = df[resolution_status_col] != ResolutionStatus.RESOLVED.value
        for i in df.index[unres]:
            reasons[df.index.get_loc(i)].append("connection_unresolved")

    valid = pd.Series([len(r) == 0 for r in reasons], index=df.index)

    # --- target validity ---------------------------------------------------------
    # EXP-032/AD-31: `timetaken_in_sec` on the reconstructed executions frame is
    # the AUTHORITATIVE total (total_timetaken_in_min*60, fallback Start/End
    # span) — the corrupted legacy column of the same name never survives
    # reconstruction (reconstruct.py never extracts it). The gate therefore
    # validates authoritative values; column naming is historical only.
    if "timetaken_in_sec" in df.columns:
        t = pd.to_numeric(df["timetaken_in_sec"], errors="coerce")
        missing_t = t.isna()
        nonfinite_t = ~missing_t & ~np.isfinite(t.fillna(0.0))
        nonpos_t = ~missing_t & (t <= 0)
        for i in df.index[missing_t]:
            reasons[df.index.get_loc(i)].append("missing_target")
        if cfg.require_finite_target:
            for i in df.index[nonfinite_t]:
                reasons[df.index.get_loc(i)].append("nonfinite_target")
        if cfg.require_positive_target:
            for i in df.index[nonpos_t]:
                reasons[df.index.get_loc(i)].append("nonpositive_target")
    else:
        missing_t = pd.Series(True, index=df.index)
        nonfinite_t = pd.Series(False, index=df.index)
        nonpos_t = pd.Series(False, index=df.index)
        for i in df.index:
            reasons[df.index.get_loc(i)].append("missing_target_column")

    target_ok = ~missing_t & ~nonfinite_t & ~nonpos_t
    target_valid = valid & target_ok

    # --- training-status policy --------------------------------------------------
    status_ok = pd.Series(True, index=df.index)
    if "LDR_Status" in df.columns:
        status_ok = df["LDR_Status"].isin(cfg.training_statuses)
    for i in df.index[~status_ok]:
        s = df.at[i, "LDR_Status"]
        label = "status_not_training_eligible" + (f":{s}" if pd.notna(s) else "")
        reasons[df.index.get_loc(i)].append(label)

    training_eligible = target_valid & status_ok

    df["is_valid_execution"] = valid
    df["is_target_valid"] = target_valid
    df["is_training_eligible"] = training_eligible
    df["eligibility_reasons"] = [";".join(r) for r in reasons]

    counts = {
        "valid_executions": int(valid.sum()),
        "target_valid": int(target_valid.sum()),
        "training_eligible": int(training_eligible.sum()),
        "total_executions": int(len(df)),
    }
    reason_counts: dict[str, int] = {}
    for r in reasons:
        for item in r:
            key = item.split(":")[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1

    return EligibilityResult(frame=df, counts=counts, reason_counts=reason_counts)
