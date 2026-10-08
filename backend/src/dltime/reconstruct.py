"""Execution-level reconstruction from raw component rows (locked rule AD-03).

Raw rows in Loader_Executions.xlsx are NOT executions:
    792 raw rows -> 543 unique LDR_Execution_Id values
    74 executions span multiple raw rows (one workflow execution loading
    several datasets), maximum 9 raw rows for one execution.

Per-column policy (from the completed investigation, revalidated by tests):

    EXECUTION_LEVEL_STRICT   constant within every execution; a real conflict
                             raises ReconstructionError instead of guessing.
    COUNT_DISTINCT           dataset dimension of the execution (execution-level
                             derived count) plus the component rows kept aside.
    AGG_SUM                  record/error counters: summed per component row;
                             the sum is stored, never replacing the per-row parts.
    COMPONENT_DETAIL         stage timing/observation fields; kept per component
                             row only. A conservative data-driven widening rule
                             also promotes a component field to execution level
                             when it is single-valued across all rows of an
                             execution (documented deterministic rule).

Conflicts inside one execution are recorded via ReconstructionError /
conflict_details, never silently resolved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class ReconstructionError(ValueError):
    """Raised when raw rows cannot be reconstructed without guessing."""


class ColumnPolicy(str, Enum):
    EXECUTION_LEVEL_STRICT = "EXECUTION_LEVEL_STRICT"
    COUNT_DISTINCT = "COUNT_DISTINCT"
    AGG_SUM = "AGG_SUM"
    COMPONENT_DETAIL = "COMPONENT_DETAIL"
    IGNORE = "IGNORE"  # spreadsheet-export artifact; dropped, counted in diagnostics


class FieldRole(str, Enum):
    EXECUTION_LEVEL = "EXECUTION_LEVEL"
    COMPONENT_LEVEL = "COMPONENT_LEVEL"
    DERIVED_EXECUTION = "DERIVED_EXECUTION"
    TARGET = "TARGET"
    IDENTIFIER = "IDENTIFIER"


# Column -> policy. Adding a new source column requires an explicit decision here.
COLUMN_POLICIES: dict[str, ColumnPolicy] = {
    # identity / structural
    "LDR_Execution_Id": ColumnPolicy.EXECUTION_LEVEL_STRICT,  # grouping key
    "ldr_workflow_execution_id": ColumnPolicy.EXECUTION_LEVEL_STRICT,
    # 2026-09-29 source replacement: Connection_Name is part of the 4-key
    # identity (Sprint + Connection + Loader + Dataset) and is constant per
    # execution; blank on real rows where the export omitted it.
    "Connection_Name": ColumnPolicy.EXECUTION_LEVEL_STRICT,
    "Loader_Name": ColumnPolicy.EXECUTION_LEVEL_STRICT,
    "Sprint": ColumnPolicy.EXECUTION_LEVEL_STRICT,
    "LDR_Status": ColumnPolicy.EXECUTION_LEVEL_STRICT,
    "Start_Time": ColumnPolicy.EXECUTION_LEVEL_STRICT,
    "End_Time": ColumnPolicy.EXECUTION_LEVEL_STRICT,
    "Time_Taken": ColumnPolicy.EXECUTION_LEVEL_STRICT,
    "timetaken_in_sec": ColumnPolicy.EXECUTION_LEVEL_STRICT,  # legacy TARGET (superseded: see AD-31)
    # Corrected dataset (2026-09-21 replacement, EXP-028): authoritative total
    # duration is total_timetaken_in_min (validated against Start/End span;
    # the legacy timetaken_in_sec column is corrupt in the corrected file and
    # is re-derived per the locked AD-31 target-derivation rule).
    "total_timetaken_in_min": ColumnPolicy.EXECUTION_LEVEL_STRICT,
    # Corrected-dataset stage observation fields (each stage now carries its
    # OWN fields; pre_load is an independent stage — EXP-028 evidence).
    "staging_start_time": ColumnPolicy.COMPONENT_DETAIL,
    "staging_end_time": ColumnPolicy.COMPONENT_DETAIL,
    "staging_timetaken_in_min": ColumnPolicy.COMPONENT_DETAIL,
    "prevalidation_timetaken_in_min": ColumnPolicy.COMPONENT_DETAIL,
    "transformation_timetaken_in_min": ColumnPolicy.COMPONENT_DETAIL,
    "pre_load_timetaken_in_min": ColumnPolicy.COMPONENT_DETAIL,
    "data_loading_timetaken_in_min": ColumnPolicy.COMPONENT_DETAIL,
    "datamart_approval_timetaken_in_min": ColumnPolicy.COMPONENT_DETAIL,
    "dataloading_approval_timetaken_in_min": ColumnPolicy.COMPONENT_DETAIL,
    # Stray unnamed spreadsheet-export columns (PostgreSQL interval text +
    # duplicated seconds on 5 rows; EXP-028 audit). Dropped, never read.
    "Unnamed: 42": ColumnPolicy.IGNORE,
    "Unnamed: 43": ColumnPolicy.IGNORE,
    # dataset dimension
    "Dataset_display_name": ColumnPolicy.COUNT_DISTINCT,
    # record/error counters (component rows sum to the execution)
    "Total_Records": ColumnPolicy.AGG_SUM,
    "Success_Records": ColumnPolicy.AGG_SUM,
    "Error_Records": ColumnPolicy.AGG_SUM,
    "Prevalidation_Errors": ColumnPolicy.AGG_SUM,
    "Format_Errors": ColumnPolicy.AGG_SUM,
    "Constraint_Errors": ColumnPolicy.AGG_SUM,
    "Application_Errors": ColumnPolicy.AGG_SUM,
    "Cascade_Errors": ColumnPolicy.AGG_SUM,
    # stage observation fields -> component detail (widen when single-valued)
    "prevalidation_start_time": ColumnPolicy.COMPONENT_DETAIL,
    "prevalidation_end_time": ColumnPolicy.COMPONENT_DETAIL,
    "prevaliadtion_timetaken": ColumnPolicy.COMPONENT_DETAIL,
    "prevaliadtion_timetaken_in_sec": ColumnPolicy.COMPONENT_DETAIL,
    "transformation_start_time": ColumnPolicy.COMPONENT_DETAIL,
    "transformation_end_time": ColumnPolicy.COMPONENT_DETAIL,
    "transformation_timetaken": ColumnPolicy.COMPONENT_DETAIL,
    "transformation_timetaken_in_sec": ColumnPolicy.COMPONENT_DETAIL,
    "pre_load_start_time": ColumnPolicy.COMPONENT_DETAIL,
    "pre_load_end_time": ColumnPolicy.COMPONENT_DETAIL,
    "data_loading_timetaken": ColumnPolicy.COMPONENT_DETAIL,
    "data_loading_timetaken_in_sec": ColumnPolicy.COMPONENT_DETAIL,
    "data_loading_start_time": ColumnPolicy.COMPONENT_DETAIL,
    "data_loading_end_time": ColumnPolicy.COMPONENT_DETAIL,
    "data_loading_timetaken_1": ColumnPolicy.COMPONENT_DETAIL,
    "data_loading_timetaken_in_sec_1": ColumnPolicy.COMPONENT_DETAIL,
    "datamart_approval_timetaken": ColumnPolicy.COMPONENT_DETAIL,
    "datamart_approval_timetaken_in_sec": ColumnPolicy.COMPONENT_DETAIL,
    "dataloading_approval_timetaken": ColumnPolicy.COMPONENT_DETAIL,
    "dataloading_approval_timetaken_in_sec": ColumnPolicy.COMPONENT_DETAIL,
}


def default_policy_for(column: str) -> ColumnPolicy:
    """Fail-closed: unknown columns must be classified explicitly."""
    raise ReconstructionError(
        f"Column '{column}' has no reconstruction policy. "
        "Add an explicit entry to COLUMN_POLICIES before reconstructing."
    )


@dataclass
class ComponentRow:
    """One raw row retained as component/stage detail of its execution."""

    execution_id: object
    dataset: object
    record_counts: dict[str, float | int | None] = field(default_factory=dict)
    stage_observations: dict[str, object] = field(default_factory=dict)


@dataclass
class ReconstructedExecution:
    """ONE logical execution (one LDR_Execution_Id)."""

    execution_id: object
    # execution-level fields (strictly constant within the execution)
    loader_name: object
    sprint: object
    ldr_status: object
    start_time: pd.Timestamp | None
    end_time: pd.Timestamp | None
    time_taken: object
    timetaken_in_sec: float | None  # the prediction target (Part E)
    ldr_workflow_execution_id: object
    # dataset dimension
    n_datasets: int
    datasets: tuple[object, ...]
    # summed component counters
    record_sums: dict[str, float | int]
    # component/stage detail
    component_rows: tuple[ComponentRow, ...]
    n_component_rows: int
    # diagnostics
    n_target_rows: int  # component rows carrying a non-null target
    widened_component_fields: tuple[str, ...] = ()


def _strict_take(group: pd.DataFrame, column: str, execution_id: object) -> object:
    values = group[column].dropna()
    if values.empty:
        return None
    uniq = pd.unique(values)
    if len(uniq) > 1:
        raise ReconstructionError(
            f"Execution {execution_id}: column '{column}' has conflicting values "
            f"{list(map(repr, uniq[:5]))} across component rows; refusing to guess."
        )
    return uniq[0]


def _sum_counts(group: pd.DataFrame, columns: list[str]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for col in columns:
        s = pd.to_numeric(group[col], errors="coerce")
        out[col] = float(s.sum(min_count=1)) if s.notna().any() else None
    return out


def reconstruct_executions(raw_executions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reconstruct one logical execution per LDR_Execution_Id.

    Args:
        raw_executions: raw rows of Loader_Executions.xlsx (unmodified source).

    Returns:
        (executions_df, components_df, diagnostics)

        executions_df  one row per LDR_Execution_Id: execution-level fields,
                       dataset counts, summed counters, widened stage fields.
        components_df  one row per raw component row (row index + execution id
                       + dataset + summed-safe counters + stage observations).
        diagnostics    counts used by tests and PROJECT_STATE.md.
    """
    # --- validate input shape (fail closed on surprises) ---------------------
    unknown = [c for c in raw_executions.columns if c not in COLUMN_POLICIES]
    if unknown:
        raise ReconstructionError(
            f"Unclassified columns in execution source: {unknown}. "
            "Extend COLUMN_POLICIES explicitly."
        )
    if "LDR_Execution_Id" not in raw_executions.columns:
        raise ReconstructionError("Column 'LDR_Execution_Id' is required.")
    if raw_executions["LDR_Execution_Id"].isna().any():
        raise ReconstructionError("Null LDR_Execution_Id values cannot be grouped into executions.")

    count_cols = [c for c, p in COLUMN_POLICIES.items() if p is ColumnPolicy.AGG_SUM]
    stage_cols = [c for c, p in COLUMN_POLICIES.items() if p is ColumnPolicy.COMPONENT_DETAIL]
    identity_cols = ["Loader_Name", "Sprint"]
    # Present-column subset: the corrected source replaced Time_Taken with
    # total_timetaken_in_min; strict extraction applies to whatever exists.
    strict_extra_cols = [
        c
        for c in ("LDR_Status", "ldr_workflow_execution_id", "total_timetaken_in_min")
        if c in raw_executions.columns
    ]
    # Start/End only: legacy Time_Taken is superseded by AD-31 derivation.
    time_cols = [c for c in ("Start_Time", "End_Time") if c in raw_executions.columns]
    ignored_cols = [c for c, p in COLUMN_POLICIES.items() if p is ColumnPolicy.IGNORE]
    present_ignored = [c for c in ignored_cols if c in raw_executions.columns]

    exec_rows: list[dict] = []
    comp_rows: list[dict] = []
    widened: dict[str, int] = {}
    max_rows_per_exec = 0
    multi_row_execs = 0

    for execution_id, group in raw_executions.groupby("LDR_Execution_Id", sort=True):
        group = group.reset_index(drop=True)
        n_rows = len(group)
        max_rows_per_exec = max(max_rows_per_exec, n_rows)
        if n_rows > 1:
            multi_row_execs += 1

        # --- execution-level strict fields -----------------------------------
        # Legacy Time_Taken (PostgreSQL interval text) is superseded by the
        # AD-31 target derivation on the corrected source (EXP-028 audit):
        # it is a per-component reporting artifact in this revision and is not
        # strictly constant across component rows. It is no longer extracted
        # as an execution-level strict field and never feeds the target.
        strict: dict[str, object] = {}
        for col in identity_cols + time_cols + strict_extra_cols:
            if col not in group.columns:
                strict[col] = None
                continue
            strict[col] = _strict_take(group, col, execution_id)
        # Corrected source (EXP-028 audit): the legacy timetaken_in_sec column
        # is BOTH sparse and inconsistent (per-component conflicts, timestamp
        # text mixed in, median ratio-to-span 2.26). It is not extracted as an
        # execution-level field and never feeds the target (AD-31).
        target = None

        # --- dataset dimension ------------------------------------------------
        datasets = tuple(pd.unique(group["Dataset_display_name"].dropna()))
        # Representative dataset for identity joins/history grouping:
        # the FIRST dataset in raw component-row order (deterministic, documented).
        representative_dataset = datasets[0] if datasets else None

        # --- component rows ---------------------------------------------------
        counts = group[count_cols].apply(pd.to_numeric, errors="coerce")
        for i, row in group.iterrows():
            comp: dict[str, object] = {
                "LDR_Execution_Id": execution_id,
                "component_index": int(i),
                "dataset": row.get("Dataset_display_name"),
            }
            for col in count_cols:
                v = counts.loc[i, col]
                comp[col] = None if pd.isna(v) else float(v)
            for col in stage_cols:
                comp[col] = row[col] if col in group.columns else None
            comp_rows.append(comp)

        # --- stage widening (conservative, deterministic, documented) ---------
        exec_widened: list[str] = []
        stage_exec_values: dict[str, object] = {}
        for col in stage_cols:
            if col not in group.columns:
                stage_exec_values[col] = None  # column absent from this source revision
                continue
            vals = group[col].dropna()
            uniq = pd.unique(vals)
            if len(uniq) == 1:
                stage_exec_values[col] = uniq[0]
                if n_rows > 1:
                    widened[col] = widened.get(col, 0) + 1
                    exec_widened.append(col)

        # ---- AD-31 target derivation (corrected dataset, EXP-028) ----------
        # The corrected file's legacy timetaken_in_sec column is corrupted
        # (173/759 values, timestamp text mixed in; median ratio to the
        # Start/End span 2.26). The authoritative supplied duration is
        # total_timetaken_in_min, which agrees with the Start/End span
        # (median rel err 0.22%). Locked derivation rule (AD-31):
        #   1. supplied total_timetaken_in_min x 60 when valid (finite, > 0)
        #   2. else End_Time - Start_Time when a valid positive span exists
        #   3. else missing (never fabricated; no legacy fallback — the legacy
        #      column failed the EXP-028 consistency audit end-to-end)
        start_time = strict.get("Start_Time")
        end_time = strict.get("End_Time")
        total_min = (
            pd.to_numeric(pd.Series([strict.get("total_timetaken_in_min")]), errors="coerce").iloc[0]
            if "total_timetaken_in_min" in strict
            else float("nan")
        )
        span_sec = np.nan
        if start_time is not None and end_time is not None:
            delta = (pd.Timestamp(end_time) - pd.Timestamp(start_time)).total_seconds()
            if np.isfinite(delta) and delta > 0:
                span_sec = float(delta)
        target_source = "unavailable"
        target_value: float | None = None
        target_consistency: float | None = None
        if pd.notna(total_min) and np.isfinite(total_min) and total_min > 0:
            target_value = float(total_min) * 60.0
            target_source = "total_timetaken_in_min_x60"
            if np.isfinite(span_sec):
                target_consistency = (
                    abs(target_value - span_sec) / span_sec if span_sec > 0 else None
                )
        elif np.isfinite(span_sec):
            target_value = float(span_sec)
            target_source = "start_end_span_derived"

        rec = {
            "LDR_Execution_Id": execution_id,
            **strict,
            "timetaken_in_sec": target_value,
            "target_source": target_source,
            "target_span_rel_diff": target_consistency,
            "Dataset_display_name": representative_dataset,
            "n_datasets": len(datasets),
            "datasets": ";".join(str(d) for d in datasets),
            **{f"exec_{c}": v for c, v in stage_exec_values.items()},
            "n_component_rows": n_rows,
            "n_target_rows": int(group["timetaken_in_sec"].notna().sum()),
        }
        for col in count_cols:
            rec[col] = None if counts[col].isna().all() else float(counts[col].sum(min_count=1))
        exec_rows.append(rec)

    executions_df = pd.DataFrame(exec_rows)
    components_df = pd.DataFrame(comp_rows)

    diagnostics = {
        "raw_rows_in": int(len(raw_executions)),
        "executions_out": int(len(executions_df)),
        "component_rows_out": int(len(components_df)),
        "single_row_execs": int((executions_df["n_component_rows"] == 1).sum()),
        "multi_row_execs": int(multi_row_execs),
        "max_rows_per_exec": int(max_rows_per_exec),
        "rows_in_multi_row_execs": int(
            executions_df.loc[executions_df["n_component_rows"] > 1, "n_component_rows"].sum()
        ),
        "widened_fields_top": dict(
            sorted(widened.items(), key=lambda kv: -kv[1])[:12]
        ),
        "executions_with_target": int(executions_df["timetaken_in_sec"].notna().sum()),
    }
    return executions_df, components_df, diagnostics
