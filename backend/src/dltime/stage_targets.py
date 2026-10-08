"""Stage target construction from the corrected execution source (AD-31, EXP-028).

The corrected Loader_Executions.xlsx (2026-09-21 replacement) carries REAL,
independent per-stage observation fields — including staging and pre_load,
which the previous source revision did not (EXP-015/EXP-022 are superseded).

Locked derivation rules (prompt §7, AD-31):
  1. If a valid supplied duration field exists, use it (unit-normalised to
     seconds: minute fields are multiplied by 60).
  2. If the duration field is missing but valid start/end timestamps exist,
     duration = end_time - start_time (converted to the correct unit).
  3. When both exist, the supplied duration is NEVER blindly overwritten; the
     relative difference is recorded and flagged when materially different
     (> CONSISTENCY_TOLERANCE).
  4. If neither exists, the stage target is MISSING — never fabricated and
     never imputed for training convenience.

Multi-component executions (one execution loading several datasets): a stage
may run per component row. The execution-level stage target is the SUM of the
positive per-component durations (the total time that stage consumed within
the execution). Single-component executions are unaffected (sum == value).
Spans aggregate as the union window (min start -> max end) across components.

Approval stages (datamart/dataloading) are sparse by construction and are
estimated by the contextual hierarchical estimator (dltime.approval_estimator),
never by an ML regression model (locked decision, AD-31).

Every stage target column is POST_EXECUTION (leakage registry) and may enter
TRAINING TARGETS only — never a prediction input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class StageTargetError(ValueError):
    """Raised when stage targets cannot be constructed without guessing."""


CONSISTENCY_TOLERANCE = 0.10  # |supplied - span| / span above this is a recorded discrepancy


@dataclass(frozen=True)
class StageFieldSpec:
    """One stage's observation fields and the derivation chain (locked order)."""

    name: str
    supplied_duration_columns: tuple[tuple[str, str], ...]  # (column, unit) in priority order
    start_column: str | None
    end_column: str | None
    kind: str  # "ml_stage" | "contextual_statistical"
    description: str = ""


MIN = "min"
SEC = "sec"

# Locked stage inventory for the corrected source (prompt §5/§6; EXP-028 audit).
# Supplied-duration priority follows the corrected file's actual coverage
# (recorded in the audit), not the older dataset's column spellings.
STAGE_FIELD_SPECS: tuple[StageFieldSpec, ...] = (
    StageFieldSpec(
        "staging",
        (("staging_timetaken_in_min", MIN),),
        "staging_start_time",
        "staging_end_time",
        "ml_stage",
        "New in the corrected source: staging is a real observed phase (716/759 raw rows).",
    ),
    StageFieldSpec(
        "prevalidation",
        (
            ("prevalidation_timetaken_in_min", MIN),
            ("prevaliadtion_timetaken_in_sec", SEC),  # legacy spelling preserved in source
        ),
        "prevalidation_start_time",
        "prevalidation_end_time",
        "ml_stage",
        "Supplied duration preferred (min field, widest coverage); legacy sec spelling kept as secondary.",
    ),
    StageFieldSpec(
        "transformation",
        (
            ("transformation_timetaken_in_min", MIN),
            ("transformation_timetaken_in_sec", SEC),
        ),
        "transformation_start_time",
        "transformation_end_time",
        "ml_stage",
        "Supplied duration preferred (min field, widest coverage).",
    ),
    StageFieldSpec(
        "pre_load",
        (("pre_load_timetaken_in_min", MIN),),
        "pre_load_start_time",
        "pre_load_end_time",
        "ml_stage",
        "REAL independent stage in the corrected source (own fields; EXP-028). The old duplicate-of treatment is removed.",
    ),
    StageFieldSpec(
        "data_loading",
        (
            ("data_loading_timetaken_in_min", MIN),
            ("data_loading_timetaken_in_sec_1", SEC),
        ),
        "data_loading_start_time",
        "data_loading_end_time",
        "ml_stage",
        "Distinct from pre_load (EXP-028: only 4/36 co-observed values identical).",
    ),
    StageFieldSpec(
        "datamart_approval",
        (
            # 2026-09-30: the corrected export carries approval durations in
            # the _in_sec fields (21 real datamart observations); the _in_min
            # spelling is empty on this source. Priority: min (if ever
            # present) -> sec (actual coverage).
            ("datamart_approval_timetaken_in_min", MIN),
            ("datamart_approval_timetaken_in_sec", SEC),
        ),
        None,
        None,
        "contextual_statistical",
        "Sparse approval stage: contextual hierarchical statistical estimation only (no ML model).",
    ),
    StageFieldSpec(
        "dataloading_approval",
        (
            ("dataloading_approval_timetaken_in_min", MIN),
            ("dataloading_approval_timetaken_in_sec", SEC),  # actual coverage on the corrected export (6 real)
        ),
        None,
        None,
        "contextual_statistical",
        "Sparse approval stage: contextual hierarchical statistical estimation only (no ML model).",
    ),
)

ML_STAGE_NAMES: tuple[str, ...] = tuple(s.name for s in STAGE_FIELD_SPECS if s.kind == "ml_stage")
APPROVAL_STAGE_NAMES: tuple[str, ...] = tuple(
    s.name for s in STAGE_FIELD_SPECS if s.kind == "contextual_statistical"
)
ALL_STAGE_NAMES: tuple[str, ...] = tuple(s.name for s in STAGE_FIELD_SPECS)


def stage_target_column(stage: str) -> str:
    return f"stage_{stage}_sec"


@dataclass
class StageTargetBuild:
    """Execution-level stage targets + per-stage quality provenance."""

    frame: pd.DataFrame  # executions frame + stage_<name>_sec (+ provenance) columns
    audit: dict[str, dict] = field(default_factory=dict)


def _to_seconds(series: pd.Series, unit: str) -> pd.Series:
    v = pd.to_numeric(series, errors="coerce").astype(float)
    if unit == MIN:
        return v * 60.0
    return v


def _span_seconds(start: pd.Series, end: pd.Series) -> pd.Series:
    s = pd.to_datetime(start, errors="coerce")
    e = pd.to_datetime(end, errors="coerce")
    span = (e - s).dt.total_seconds()
    return span.where(span.notna() & (span >= 0))  # negative spans invalid -> NA


def _component_stage_seconds(
    components: pd.DataFrame, spec: StageFieldSpec
) -> tuple[pd.Series, pd.Series]:
    """Per-component-row supplied-duration (seconds) and span (seconds) series."""
    supplied = pd.Series(np.nan, index=components.index, dtype=float)
    used_col: pd.Series = pd.Series(pd.NA, index=components.index, dtype="object")
    for col, unit in spec.supplied_duration_columns:
        if col not in components.columns:
            continue
        v = _to_seconds(components[col], unit)
        take = supplied.isna() & v.notna() & (v > 0)  # first valid priority wins; <=0 invalid
        supplied = supplied.where(~take, v)
        used_col = used_col.where(~take, col)
    span = (
        _span_seconds(components[spec.start_column], components[spec.end_column])
        if spec.start_column and spec.start_column in components.columns and spec.end_column in components.columns
        else pd.Series(np.nan, index=components.index, dtype=float)
    )
    return supplied, span


def build_stage_targets(
    executions: pd.DataFrame, components: pd.DataFrame
) -> StageTargetBuild:
    """Attach execution-level stage target columns to the executions frame.

    Requires the frames produced by reconstruct_executions (components carry
    the per-raw-row stage observation fields).
    """
    if "LDR_Execution_Id" not in executions.columns or "LDR_Execution_Id" not in components.columns:
        raise StageTargetError("Stage targets require LDR_Execution_Id on both frames.")

    comp = components.copy()
    out = executions.copy()
    audit: dict[str, dict] = {}

    for spec in STAGE_FIELD_SPECS:
        supplied, span = _component_stage_seconds(comp, spec)
        comp["_supplied"] = supplied
        comp["_span"] = span

        # ---- consistency validation (where both exist on one component row) --
        both = supplied.notna() & span.notna()
        rel_diff = ((supplied - span).abs() / span).where(both & (span > 0))
        n_both = int(both.sum())
        n_discrepant = int((rel_diff > CONSISTENCY_TOLERANCE).sum())

        grouped = comp.groupby("LDR_Execution_Id", sort=True)
        sum_supplied = grouped["_supplied"].sum(min_count=1)
        if spec.start_column and spec.start_column in comp.columns:
            s_start = pd.to_datetime(comp[spec.start_column], errors="coerce")
            s_end = pd.to_datetime(comp[spec.end_column], errors="coerce")
            union = (
                pd.DataFrame({"id": comp["LDR_Execution_Id"], "s": s_start, "e": s_end})
                .groupby("id")
                .agg(s=("s", "min"), e=("e", "max"))
            )
            union_span = (union["e"] - union["s"]).dt.total_seconds()
            union_span = union_span.where((union["s"].notna()) & (union["e"].notna()))
        else:
            union_span = pd.Series(np.nan, index=sum_supplied.index, dtype=float)

        # ---- execution-level derivation chain (AD-31) ------------------------
        # NOTE: grouped results are indexed by LDR_Execution_Id; the executions
        # frame uses a RangeIndex, so everything is mapped through the id column.
        target = sum_supplied.where(sum_supplied > 0)  # invalid/nonpositive -> NA
        source = pd.Series("supplied_duration", index=target.index, dtype="object")
        source = source.where(target.notna(), "missing")
        fallback = union_span.where(union_span > 0)
        take_fb = target.isna() & fallback.notna()
        target = target.where(~take_fb, fallback)
        source = source.where(~take_fb, "span_derived")

        # execution-level consistency (supplied vs union span) where both exist
        exec_rel = ((target - union_span).abs() / union_span).where(
            target.notna() & union_span.notna() & (union_span > 0)
        )
        n_exec_discrepant = int((exec_rel > CONSISTENCY_TOLERANCE).sum())

        col = stage_target_column(spec.name)
        ids = out["LDR_Execution_Id"]
        out[col] = pd.to_numeric(ids.map(target), errors="coerce")
        out[f"stage_{spec.name}_source"] = ids.map(source).fillna("missing")
        out[f"stage_{spec.name}_n_components"] = ids.map(
            grouped["_supplied"].apply(lambda s: int(s.notna().sum()))
        ).fillna(0).astype(int)

        pos = out[col]
        n_valid = int((pos > 0).sum())
        n_eligible = int(((pos > 0) & out["is_training_eligible"].fillna(False).astype(bool)).sum()) \
            if "is_training_eligible" in out.columns else None
        audit[spec.name] = {
            "kind": spec.kind,
            "supplied_duration_columns": [c for c, _ in spec.supplied_duration_columns],
            "component_rows_with_supplied": int(supplied.notna().sum()),
            "component_rows_with_span": int(span.notna().sum()),
            "component_consistency_checked": n_both,
            "component_discrepant_over_tolerance": n_discrepant,
            "executions_with_supplied": int((out[f"stage_{spec.name}_source"] == "supplied_duration").sum()),
            "executions_span_derived": int((out[f"stage_{spec.name}_source"] == "span_derived").sum()),
            "executions_missing": int((out[f"stage_{spec.name}_source"] == "missing").sum()),
            "executions_valid_target": n_valid,
            "executions_zero_or_negative": int(((pos.notna()) & (pos <= 0)).sum()),
            "executions_discrepant_over_tolerance": n_exec_discrepant,
            "training_eligible_with_valid_target": n_eligible,
            "median_sec": round(float(pos[pos > 0].median()), 3) if n_valid else None,
            "p90_sec": round(float(pos[pos > 0].quantile(0.9)), 3) if n_valid else None,
            "max_sec": round(float(pos[pos > 0].max()), 3) if n_valid else None,
            "unique_loaders": int(out.loc[pos > 0, "Loader_Name"].nunique()) if n_valid else 0,
            "unique_datasets": int(out.loc[pos > 0, "Dataset_display_name"].nunique()) if n_valid else 0,
            "unique_connections": (
                int(out.loc[pos > 0, "execution_connection"].nunique())
                if n_valid and "execution_connection" in out.columns
                else 0
            ),
            "description": spec.description,
        }
        comp.drop(columns=["_supplied", "_span"], inplace=True, errors="ignore")

    return StageTargetBuild(frame=out, audit=audit)


def stage_data_audit(build: StageTargetBuild) -> dict:
    """The §8 dataset audit over all seven stage components (EXP-028 record)."""
    return {
        "stages": build.audit,
        "derivation_rule": {
            "priority": ["supplied_duration_unit_normalised", "start_end_span_derived"],
            "consistency_tolerance": CONSISTENCY_TOLERANCE,
            "fabrication_policy": "missing targets stay missing; never imputed",
            "aggregation": "execution target = sum of positive per-component durations; spans union window",
        },
    }
