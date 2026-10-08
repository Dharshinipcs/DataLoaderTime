"""End-to-end data foundation pipeline (Parts B-F of Task 01).

Loads the real sources, reconstructs executions, resolves connections,
validates the target and applies the eligibility tiers. Produces a single
report object consumed by tests, state files and (later) the API layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from dltime.eligibility import EligibilityConfig, apply_eligibility
from dltime.reconstruct import reconstruct_executions
from dltime.resolution import ConnectionResolver, ResolutionStatus, annotate_raw_rows
from dltime.sources import SourceBundle, load_all_sources


@dataclass
class FoundationReport:
    """Aggregated facts of the data foundation (single source of truth for tests)."""

    sources: dict[str, dict[str, object]] = field(default_factory=dict)
    reconstruction: dict[str, object] = field(default_factory=dict)
    resolution_raw_rows: dict[str, int] = field(default_factory=dict)
    resolution_execution_keys: dict[str, int] = field(default_factory=dict)
    resolution_executions: dict[str, int] = field(default_factory=dict)
    resolution_execution_connections: dict[str, int] = field(default_factory=dict)
    target: dict[str, object] = field(default_factory=dict)
    eligibility: dict[str, object] = field(default_factory=dict)
    stage_target_audit: dict[str, object] = field(default_factory=dict)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)


def _exec_resolution_from_rows(raw_annotated: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-row resolution to execution level (AD-04, approved rule).

    For each execution, collect all DISTINCT successfully resolved component
    connections:
        0 distinct connections   -> UNRESOLVED (unless any component is AMBIGUOUS)
        exactly 1 distinct       -> RESOLVED (execution_connection = that connection)
        >1 distinct connections  -> AMBIGUOUS (conflicting evidence; never choose)
    A component-level AMBIGUOUS also escalates the execution to AMBIGUOUS.
    """

    def agg(rows: pd.DataFrame) -> pd.Series:
        conns = set(rows["resolved_connection"].dropna())
        if len(conns) == 1:
            status = ResolutionStatus.RESOLVED.value
        elif len(conns) > 1:
            status = ResolutionStatus.AMBIGUOUS.value
        elif (rows["resolution_status"] == ResolutionStatus.AMBIGUOUS.value).any():
            status = ResolutionStatus.AMBIGUOUS.value
        else:
            status = ResolutionStatus.UNRESOLVED.value
        connection = next(iter(conns)) if len(conns) == 1 else None
        return pd.Series(
            {
                "execution_resolution_status": status,
                "execution_connection": connection,
            }
        )

    out = (
        raw_annotated.groupby("LDR_Execution_Id", sort=True)[
            ["resolution_status", "resolved_connection"]
        ]
        .apply(agg)
        .reset_index()
    )
    out["execution_connection"] = out["execution_connection"].astype("string")
    return out


def build_foundation(bundle: SourceBundle | None = None) -> FoundationReport:
    """Run the complete Task-01 data foundation over the real source files."""
    bundle = bundle or load_all_sources()
    report = FoundationReport(sources=dict(bundle.provenance))

    # --- Part C: execution reconstruction ------------------------------------
    executions, components, diag = reconstruct_executions(bundle.executions)
    report.reconstruction = diag
    report.frames["executions"] = executions
    report.frames["components"] = components

    # --- Part D: connection resolution ----------------------------------------
    resolver = ConnectionResolver(
        bundle.loader_definitions_detail, bundle.loader_definitions_summary
    )
    raw_annotated = annotate_raw_rows(bundle.executions, resolver)
    report.frames["raw_annotated"] = raw_annotated

    report.resolution_raw_rows = (
        raw_annotated["resolution_status"].value_counts().to_dict()
    )
    key_counts = (
        raw_annotated.groupby(["Dataset_display_name", "Sprint", "Loader_Name"], dropna=False)[
            "resolution_status"
        ]
        .first()
        .value_counts()
        .to_dict()
    )
    report.resolution_execution_keys = {str(k): int(v) for k, v in key_counts.items()}

    exec_res = _exec_resolution_from_rows(raw_annotated)
    executions = executions.merge(exec_res, on="LDR_Execution_Id", how="left")
    report.frames["executions"] = executions
    report.resolution_executions = (
        executions["execution_resolution_status"].value_counts().to_dict()
    )
    report.resolution_execution_connections = {
        "resolved_with_connection": int(executions["execution_connection"].notna().sum()),
        "distinct_connections": int(executions["execution_connection"].dropna().nunique()),
    }

    # --- Part E: target validation facts ---------------------------------------
    # EXP-032/AD-31 semantics: executions["timetaken_in_sec"] is NOT the raw
    # legacy column (corrupted: agrees with the Start/End span only 15.1%).
    # Reconstruction derives it from the authoritative total_timetaken_in_min*60
    # (fallback: Start/End span; never the legacy field) and labels each value
    # via target_source. Overview duration statistics must therefore be both
    # authoritative in VALUE and truthful in LABEL — the legacy field is
    # reported only as counts for transparency, never as a duration statistic.
    t = pd.to_numeric(executions["timetaken_in_sec"], errors="coerce")
    source_counts = (
        executions["target_source"].value_counts().to_dict()
        if "target_source" in executions.columns
        else {}
    )
    span_diff = pd.to_numeric(executions.get("target_span_rel_diff"), errors="coerce")
    report.target = {
        "definition": (
            "authoritative total seconds = total_timetaken_in_min * 60 "
            "(fallback: Start_Time->End_Time span); legacy timetaken_in_sec is "
            "corrupted and never used (EXP-032/AD-31/AD-34)"
        ),
        "non_null": int(t.notna().sum()),
        "positive": int((t > 0).sum()),
        "zero": int((t == 0).sum()),
        "negative": int((t < 0).sum()),
        "median": float(t.median()),
        "mean": float(t.mean()),
        "p95": float(t.quantile(0.95)),
        "max": float(t.max()),
        "target_source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "span_consistency": (
            {
                "checked": int(span_diff.notna().sum()),
                "within_5pct": int((span_diff <= 0.05).sum()),
            }
            if span_diff.notna().any()
            else None
        ),
    }

    # --- Part F: eligibility -----------------------------------------------------
    # AD-31 corrected-data regime (EXP-028): the corrected Loader_Executions.xlsx
    # populates Dataset_display_name on only 36/759 raw rows, so the legacy
    # full-canonical-identity gate would strand the foundation at 10 eligible
    # executions. The documented corrected regime requires (Loader_Name, Sprint)
    # identity completeness — the same policy the stage models use — keeping
    # every other gate (status policy, positive finite target) unchanged.
    from dltime.eligibility import CORRECTED_DATA_IDENTITY_COLUMNS

    elig = apply_eligibility(
        executions,
        EligibilityConfig(required_identity_columns=CORRECTED_DATA_IDENTITY_COLUMNS),
    )

    # --- AD-31: attach execution-level stage targets as the single source of
    # truth for every stage consumer (analytics, stage models, serving intel).
    from dltime.stage_targets import build_stage_targets

    stage_build = build_stage_targets(elig.frame, components)
    report.stage_target_audit = stage_build.audit
    report.frames["executions_eligible"] = stage_build.frame
    report.eligibility = {
        "counts": elig.counts,
        "reason_counts": elig.reason_counts,
        "status_distribution": executions["LDR_Status"].value_counts().to_dict(),
        "status_x_eligible": {
            str(status): {
                "total": int(len(g)),
                "eligible": int(g["is_training_eligible"].sum()),
            }
            for status, g in elig.frame.groupby("LDR_Status")
        },
    }
    return report
