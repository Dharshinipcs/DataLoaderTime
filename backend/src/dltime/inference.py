"""Pre-execution inference service (Task 04).

Takes a CLEAN pre-execution prediction request (identity + configuration the
caller legitimately knows before execution starts), constructs the exact
Task 02 leakage-safe feature vector through the existing feature-assembly
path, applies the configured history-eligibility policy (AD-16, approved
configurable default min_prior_loader_executions=3) and — only when
eligibility passes — predicts total execution time with the persisted
Task 03 candidate artifact (total-et-v3-candidate, log1p -> expm1).

Response states (explicit; no silent fallbacks):
    PREDICTED             authoritative ETA (history sufficient)
    INSUFFICIENT_HISTORY  no ETA (AD-16 gate; no cold-start fabrication)
    UNRESOLVED_IDENTITY   canonical identity has no definition match
    AMBIGUOUS_IDENTITY    canonical identity matches >1 connection (never chosen)
    INVALID_INPUT         request violates the pre-execution-only contract

TOTAL POLICY (final product decision, EXP-038): the served ETA is the
EXACT DETERMINISTIC SUM of the five operational ML stage predictions
(staging + prevalidation + transformation + pre_load + data_loading).
No floor, no learned adjustment, no correction factor; the two contextual
approval estimates are NEVER part of the total. The retired stage-aware
total model and the direct total model are still evaluated per request
and returned as clearly-labelled reference/rollback evidence only.

Task 10.1: Connection is a REQUIRED request field (AD-18 revised). It stays
validation-only: the AD-04 definition-based resolution remains authoritative,
and a supplied connection that disagrees with it is INVALID_INPUT — user
input never overrides resolution.

Leakage boundary (AD-07/AD-11): the request itself contains only
pre-execution information. Historical features for the requested context are
computed by appending a synthetic context row to the real execution frame and
running the UNMODIFIED Task 02 chronological engine, so only strictly-earlier
training-eligible executions can contribute. The synthetic row carries
target=<NA> and is_training_eligible=False — it can never enter its own (or
any) historical aggregate. Post-execution request fields are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import pandas as pd

from dltime.artifact import SERVING_MODEL_VERSION, predict_with_artifact
from dltime.sum_total_serving import (
    SumTotalError,
    sum_total_explanation,
    sum_total_from_stage_intel,
)
from dltime.feature_assembly import TrainingFrame
from dltime.feature_registry import FeatureRegistry
from dltime.history_eligibility import (
    HistoryEligibility,
    HistoryEligibilityConfig,
    HistoryEligibilityEngine,
)
from dltime.pipeline import FoundationReport
from dltime.resolution import ConnectionResolver, ResolutionStatus
from dltime.risk import assess_risk, insufficient_history_risk, training_range_profile

# Task 11: optional append-only prediction-event persistence. Imported
# lazily by _persist_event so a store problem can never affect prediction.

# Post-execution / derived / target fields that must NEVER appear in a request.
FORBIDDEN_REQUEST_FIELDS: frozenset[str] = frozenset(
    {
        "timetaken_in_sec",
        "Time_Taken",
        "End_Time",
        "LDR_Status",
        "Total_Records",
        "Success_Records",
        "Error_Records",
        "Prevalidation_Errors",
        "Format_Errors",
        "Constraint_Errors",
        "Application_Errors",
        "Cascade_Errors",
        "hist_loader_prior_count",
        "hist_loader_prior_mean_sec",
        "hist_prev_loader_duration_sec",
        "hist_time_since_prev_loader_sec",
        "similarity_score",
        "shap_values",
        "prediction_residual",
    }
)


class PredictionState:
    """Explicit prediction response states (Task 04)."""

    PREDICTED = "PREDICTED"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    UNRESOLVED_IDENTITY = "UNRESOLVED_IDENTITY"
    AMBIGUOUS_IDENTITY = "AMBIGUOUS_IDENTITY"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True)
class PredictionRequest:
    """Clean pre-execution prediction request (no derived/post-execution data).

    Connection is REQUIRED (Task 10.1): the caller must state the connection
    they believe applies; the backend validates it against the authoritative
    AD-04 definition resolution (which is never overridden by input).
    """

    loader_name: str
    sprint: str
    dataset: str
    connection: str  # required; must agree with definition resolution
    workflow_execution_id: str | None = None  # optional caller reference (LINKAGE ONLY)
    planned_start_time: str | None = None  # optional ISO-8601; else deterministic default
    # ---- AD-31 legitimate pre-execution configuration (prompt §2/§21) -------
    # Optional user-supplied configuration describing THIS execution. When
    # present it overrides the definition-derived aggregates for the context
    # row only (historical aggregates are untouched). All default to None ->
    # definition-derived values are used unchanged.
    preval_trans_display_name: str | None = None  # legacy combined field (audit echo)
    prevalidation_display_name: str | None = None  # EXP-032 Part F: PREVALIDATION component
    transformation_display_name: str | None = None  # EXP-032 Part F: TRANSFORMATION component
    loader_type: str | None = None
    calling_type: str | None = None
    execution_order: float | None = None
    prevalidation_count: float | None = None
    transformation_count: float | None = None
    # 2026-10-02: per-component pre-execution rows matching the real detail
    # source (loader_definitions): each row is ONE component of THIS execution
    # (display name + calling_type PREVALIDATION/TRANSFORMATION + component
    # type + execution order). This is the full-fidelity replacement for the
    # single-count fields above; the legacy fields stay accepted for backward
    # compatibility. 2026-10-03: component TYPE (crossreference | dataobject |
    # dataextract | dataset | query | table) rides on every row — a legitimate
    # pre-execution field the models must be able to learn from on retrain.
    preval_trans_components: tuple["PrevalTransComponent", ...] = ()


# AD-31: request fields that may override definition-derived configuration on
# the context row only (legitimate pre-execution information the user knows).
_CONFIG_OVERRIDE_FIELDS = (
    "preval_trans_display_name",
    "prevalidation_display_name",
    "transformation_display_name",
    "loader_type",
    "calling_type",
    "execution_order",
    "prevalidation_count",
    "transformation_count",
    "preval_trans_components",
)


def _clean_components(components: Any) -> tuple[PrevalTransComponent, ...]:
    """Normalize the per-component rows (blank display names dropped).

    Accepts PrevalTransComponent instances or plain dicts/objects with the
    four fields (display name, calling type, component type, execution order);
    invalid rows are dropped, never fabricated.
    """
    if not components:
        return ()
    out: list[PrevalTransComponent] = []
    for c in components:
        if c is None:
            continue

        def _order_of(raw: Any) -> float | None:
            if isinstance(raw, str):
                try:
                    return float(raw)
                except ValueError:
                    return None
            return raw

        if isinstance(c, PrevalTransComponent):
            out.append(
                PrevalTransComponent(
                    _clean_str(c.display_name),
                    _clean_str(c.calling_type),
                    _order_of(c.execution_order),
                    _clean_str(getattr(c, "component_type", None)),
                )
            )
            continue
        if isinstance(c, dict):
            out.append(
                PrevalTransComponent(
                    _clean_str(c.get("display_name")),
                    _clean_str(c.get("calling_type")),
                    _order_of(c.get("execution_order")),
                    _clean_str(c.get("component_type", c.get("type"))),
                )
            )
            continue
        out.append(
            PrevalTransComponent(
                _clean_str(getattr(c, "display_name", None)),
                _clean_str(getattr(c, "calling_type", None)),
                _order_of(getattr(c, "execution_order", None)),
                _clean_str(getattr(c, "component_type", getattr(c, "type", None))),
            )
        )
    return tuple(c for c in out if c.display_name is not None or c.calling_type is not None)


@dataclass(frozen=True)
class PrevalTransComponent:
    """One per-component pre-execution row (loader_definitions detail shape).

    The user enters each PREVALIDATION / TRANSFORMATION component of THIS
    execution exactly as it appears in the pre-execution dataset: its display
    name (the source's single ``preval_trans_display_name`` column), its
    calling type, its component type (crossreference, dataobject, dataextract,
    dataset, query or table) and its execution order.
    """

    display_name: str
    calling_type: str
    execution_order: float | None = None
    component_type: str | None = None


def _apply_context_config_overrides(
    frame: pd.DataFrame, context_execution_id, overrides: dict[str, Any]
) -> pd.DataFrame:
    """Apply user-supplied pre-execution configuration to the CONTEXT row only.

    Overrides (AD-31) map request fields onto the definition-derived feature
    columns, preserving each feature's aggregation semantics for a single
    supplied component (one preval/trans row of this execution):

      - calling_type        -> def_dominant_calling_type (majority of one)
      - execution_order     -> def_execution_order_max / _mean (=_std stays NA:
                               std of one value is undefined, never fabricated)
      - type                -> def_type_nunique = 1 (exactly one type present)
      - prevalidation_count -> def_prevalidation_count (summary max of one)
      - transformation_count-> def_transformation_count (summary max of one)

    EXP-032 (Part F/G): the source's ``preval_trans_display_name`` is a
    COMPONENT label (DO/CR/QUERY/…); the operation is identified by the row's
    ``calling_type`` (PREVALIDATION vs TRANSFORMATION) — the detail table is
    per-component and one execution config routinely contains both kinds.
    Therefore the studio collects SEPARATE component display names:

      - prevalidation_display_name  -> def_preval_name_nunique evidence:
            the PREVALIDATION-component display name for this execution;
            recorded as audit context AND as a def_preval_name_component
            marker column (value 1.0 = exactly one PREVALIDATION display
            name supplied) so inference can distinguish A/B/C/D states.
      - transformation_display_name -> the TRANSFORMATION-component display
            name; identical treatment (def_trans_name_component marker).

    The two overrides NEVER corrupt each other: each writes only its own
    marker column; no shared aggregate is overwritten by either.
    The legacy combined ``preval_trans_display_name`` remains an accepted
    audit echo only (never a model feature).
    """
    applied = False
    mask = frame["LDR_Execution_Id"] == context_execution_id

    def _set(col: str, value: Any) -> None:
        nonlocal applied
        if value is None:
            return
        if col not in frame.columns:
            frame[col] = pd.NA
        frame.loc[mask, col] = value
        applied = True

    calling = overrides.get("calling_type")
    if calling is not None:
        _set("def_dominant_calling_type", calling)
    order = overrides.get("execution_order")
    if order is not None:
        _set("def_execution_order_max", float(order))
        _set("def_execution_order_mean", float(order))
    if overrides.get("loader_type") is not None:
        _set("def_type_nunique", 1.0)
    _set("def_prevalidation_count", overrides.get("prevalidation_count"))
    _set("def_transformation_count", overrides.get("transformation_count"))

    # 2026-10-02: full-fidelity per-component rows. When supplied they are the
    # authoritative configuration of THIS execution: the SAME aggregates as
    # the training-time detail aggregation are re-derived for the context row
    # (per-calling-type counts, pooled order max/mean/std, component count,
    # type nunique, dominant calling type), and per-calling-type display-name
    # markers replace the single-name markers. Historical rows are untouched.
    components = overrides.get("preval_trans_components") or ()
    if components:
        _apply_component_aggregates(_set, components)
    # EXP-032 Part F/G: independent component display-name markers. Each
    # writes ONLY its own column — prevalidation and transformation cannot
    # corrupt each other (verified by the A/B/C/D inference probes).
    _set("def_preval_name_component", overrides.get("prevalidation_display_name"))
    _set("def_trans_name_component", overrides.get("transformation_display_name"))
    if applied and "def_match" in frame.columns:
        frame.loc[mask, "def_match"] = True
    return frame


def _apply_component_aggregates(_set, components: tuple["PrevalTransComponent", ...]) -> None:
    """Re-derive the training-time detail aggregates from per-component rows.

    EXACT definition_features.py semantics for the supplied rows (so the context
    row carries values the models were actually trained on):
      - def_execution_order_max/_mean: pooled max/mean over ALL rows;
        _std: ddof=1 over all rows, NA unless >= 2 rows;
      - def_component_count: number of component rows;
      - def_type_nunique: unique non-blank 'type' values (1 when all rows share
        one type — DataObject in the real source);
      - def_dominant_calling_type: majority calling_type, ties -> PREVALIDATION;
      - def_prevalidation_count / def_transformation_count: PER-TYPE counts;
      - def_preval_name_component / def_trans_name_component: the FIRST
        (lowest execution order, ties in entry order) display name per type.
    """
    orders = [float(c.execution_order) for c in components if c.execution_order is not None]
    if orders:
        _set("def_execution_order_max", float(max(orders)))
        _set("def_execution_order_mean", float(sum(orders) / len(orders)))
        _set("def_execution_order_std", float(np.std(orders, ddof=1)) if len(orders) >= 2 else None)
    _set("def_component_count", float(len(components)))
    # 2026-10-03: the per-row component TYPE is now a first-class pre-execution
    # input. def_type_nunique is the TRUE count of distinct supplied types
    # (legacy single-row fallback: 1.0 when no types are supplied), and each
    # canonical type's count is recorded so the stage models can learn from
    # the type mix on the next retrain (features join the training frame;
    # never hardcoded into any prediction).
    types = [
        str(c.component_type).strip().lower()
        for c in components
        if c.component_type is not None and str(c.component_type).strip()
    ]
    if types:
        _set("def_type_nunique", float(len(set(types))))
        for t in ("crossreference", "dataobject", "dataextract", "dataset", "query", "table"):
            n_t = sum(1 for x in types if x == t)
            if n_t:
                _set(f"def_type_{t}_count", float(n_t))
    else:
        _set("def_type_nunique", 1.0)
    # 2026-10-04 (user spec): execution_order is PER SECTION — prevalidation
    # rows are ordered 1..n among themselves, transformation rows 1..m among
    # themselves (two independent sequences). Per-section order stats and
    # per-section type variety are served so the models see the same feature
    # contract they are (re)trained on.
    prevals = [c for c in components if str(c.calling_type or "").upper() == "PREVALIDATION"]
    trans = [c for c in components if str(c.calling_type or "").upper() == "TRANSFORMATION"]
    for prefix, sec in (("preval", prevals), ("trans", trans)):
        sec_orders = [
            float(c.execution_order) for c in sec if c.execution_order is not None
        ]
        if sec_orders:
            _set(f"def_{prefix}_order_max", float(max(sec_orders)))
            _set(f"def_{prefix}_order_mean", float(sum(sec_orders) / len(sec_orders)))
            _set(
                f"def_{prefix}_order_std",
                float(np.std(sec_orders, ddof=1)) if len(sec_orders) >= 2 else None,
            )
        sec_types = [
            str(c.component_type).strip().lower()
            for c in sec
            if c.component_type is not None and str(c.component_type).strip()
        ]
        _set(f"def_{prefix}_type_nunique", float(len(set(sec_types))) if sec_types else None)
    n_prev, n_trans = len(prevals), len(trans)
    if n_prev and n_trans:
        dominant = "PREVALIDATION" if n_prev >= n_trans else "TRANSFORMATION"
    elif n_prev:
        dominant = "PREVALIDATION"
    elif n_trans:
        dominant = "TRANSFORMATION"
    else:
        dominant = None
    if dominant is not None:
        _set("def_dominant_calling_type", dominant)
    if n_prev:
        _set("def_prevalidation_count", float(n_prev))
        _set(
            "def_preval_name_component",
            (sorted(prevals, key=lambda c: (c.execution_order if c.execution_order is not None else float("inf"),))[0]).display_name,
        )
    if n_trans:
        _set("def_transformation_count", float(n_trans))
        _set(
            "def_trans_name_component",
            (sorted(trans, key=lambda c: (c.execution_order if c.execution_order is not None else float("inf"),))[0].display_name),
        )


def human_readable_duration(seconds: float | None) -> str | None:
    """Format seconds as a compact human-readable duration."""
    if seconds is None or not np.isfinite(seconds):
        return None
    total = int(round(float(seconds)))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _attach_context_stage_history(
    combined: pd.DataFrame,
    context_execution_id,
    ml_stages: tuple[str, ...],
    *,
    stage_specific: bool = False,
    history_stage: str = "staging",
) -> pd.DataFrame:
    """Compute the context row's stage_hist_* features strictly-prior.

    EXP-032 (Part J correction): the previous implementation looped over
    stages writing the SAME generic ``stage_hist_*`` columns, so the LAST
    stage's history silently overwrote every earlier stage's — every stage
    model received data_loading's history. Now:

    - ``stage_specific=True`` additionally writes one full set per stage as
      ``stage_hist_<stage>_*`` so each stage model receives its OWN history
      (matching its per-stage training frame exactly);
    - the GENERIC ``stage_hist_*`` columns ALWAYS carry ``history_stage``
      (default "staging") history — the pinned stage-aware total contract is
      derived from the staging frame, so its base features must carry staging
      history at serving exactly as at build time (never the last loop
      iteration's stage).

    Aggregation semantics are IDENTICAL to the per-stage training frame
    builder (stage_models.build_stage_frame): true per-group masks for
    loader / loader+conn / loader+dataset / composite(D,S,L) + prev
    same-loader value; std with ddof=1 only from n>=2; strictly-earlier
    training-eligible rows with positive stage target only.
    """
    from dltime.stage_targets import stage_target_column

    df = combined.copy()
    df["_ts"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    ordered = df.sort_values(["_ts", "LDR_Execution_Id"], kind="mergesort").reset_index(drop=True)
    hits = ordered.index[ordered["LDR_Execution_Id"] == context_execution_id]
    if len(hits) != 1:
        return combined  # context missing: caller handles
    ctx_pos = int(hits[0])
    usable = ordered["is_training_eligible"].fillna(False).astype(bool)
    ctx = ordered.iloc[ctx_pos]
    loader_str = None if pd.isna(ctx["Loader_Name"]) else str(ctx["Loader_Name"])
    conn_series = (
        ordered["execution_connection"].astype("string")
        if "execution_connection" in ordered.columns
        else pd.Series(pd.NA, index=ordered.index, dtype="string")
    )
    ctx_conn = ctx.get("execution_connection")
    ctx_conn_str = None if pd.isna(ctx_conn) else str(ctx_conn)
    ds_series = ordered["Dataset_display_name"].astype("string")
    ctx_ds = ctx.get("Dataset_display_name")
    ctx_ds_str = None if pd.isna(ctx_ds) else str(ctx_ds)
    sprint_series = ordered["Sprint"].astype("string")
    ctx_sprint = ctx.get("Sprint")
    ctx_sprint_str = None if pd.isna(ctx_sprint) else str(ctx_sprint)

    loader_str_col = ordered["Loader_Name"].astype("string")
    same_loader = loader_str_col == pd.Series([loader_str], dtype="string")[0]
    same_loader_conn = same_loader & (conn_series == pd.Series([ctx_conn_str], dtype="string")[0])
    same_loader_ds = same_loader & (ds_series == pd.Series([ctx_ds_str], dtype="string")[0])
    same_composite = (
        same_loader_ds
        & (sprint_series == pd.Series([ctx_sprint_str], dtype="string")[0])
    )

    def _write(prefix: str, vals_by_mask: list[tuple[str, pd.Series, pd.Series]]) -> None:
        ctx_idx = ordered.index[[ctx_pos]][0]
        for name, mask, v in vals_by_mask:
            vv = v[mask]
            n = int(len(vv))
            ordered.loc[ctx_idx, f"{prefix}_{name}_prior_count"] = n
            ordered.loc[ctx_idx, f"{prefix}_{name}_prior_mean_sec"] = float(vv.mean()) if n else np.nan
            ordered.loc[ctx_idx, f"{prefix}_{name}_prior_median_sec"] = float(vv.median()) if n else np.nan
            ordered.loc[ctx_idx, f"{prefix}_{name}_prior_std_sec"] = float(vv.std(ddof=1)) if n >= 2 else np.nan

    for stage in ml_stages:
        col = stage_target_column(stage)
        if col not in ordered.columns:
            continue
        v = pd.to_numeric(ordered[col], errors="coerce")
        prior = (ordered.index < ctx_pos) & usable & (v > 0)
        groups = [
            ("global", prior, v),
            ("loader", prior & same_loader, v),
            ("loader_conn", prior & same_loader_conn, v),
            ("loader_dataset", prior & same_loader_ds, v),
            ("composite", prior & same_composite, v),
        ]
        lvals = v[prior & same_loader]
        if stage_specific:
            _write(f"stage_hist_{stage}", groups)
            ctx_idx = ordered.index[[ctx_pos]][0]
            ordered.loc[ctx_idx, f"stage_hist_{stage}_prev_stage_sec"] = (
                float(lvals.iloc[-1]) if len(lvals) else np.nan
            )
        if stage == history_stage:
            # generic contract columns: ALWAYS the contract stage's history
            _write("stage_hist", groups)
            ctx_idx = ordered.index[[ctx_pos]][0]
            ordered.loc[ctx_idx, "stage_hist_prev_stage_sec"] = (
                float(lvals.iloc[-1]) if len(lvals) else np.nan
            )

    return ordered


def _stage_aware_feature_vector(
    ctx: pd.DataFrame, stage_intel: dict[str, Any]
) -> pd.DataFrame:
    """Build the 36-feature stage-aware vector for the context row.

    RETIRED from serving (EXP-038): the stage-aware total model no longer
    produces the ETA. This vector is still assembled so the retired model's
    per-request output can be returned as reference evidence only. Same
    construction as AD-32: 29 context features (def_* + stage_hist_*;
    identical semantics to the stage-frame builder) + 7 stage-derived
    features read from the ALREADY-COMPUTED seven-stage view. Actual stage
    durations never enter; the context row carries NA targets by construction.
    """
    from dltime.stage_aware_serving import feature_contract

    expected = feature_contract()
    base = [c for c in expected if c.startswith(("def_", "stage_hist_"))]
    X = ctx[base].copy()
    for col in X.columns:
        if X[col].dtype == "string" or isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].astype("string")
        else:
            # def_dominant_calling_type is the contract's single categorical
            # feature (string dtype upstream); keep string values verbatim,
            # coerce everything else numerically.
            if col == "def_dominant_calling_type":
                v = X[col].iloc[0]
                X[col] = X[col].astype("string") if pd.notna(v) else X[col]
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    for stage_entry in stage_intel.get("stages", []):
        X[f"pred_stage_{stage_entry.get('stage')}_sec"] = pd.to_numeric(
            pd.Series([stage_entry.get("estimate_seconds")], index=X.index),
            errors="coerce",
        ).astype(float)
    if list(X.columns) != expected:
        missing = sorted(set(expected) - set(X.columns))
        raise RuntimeError(
            f"Stage-aware feature vector drift: missing columns {missing[:5]}"
        )
    return X[expected]
    """Format seconds as a compact human-readable duration."""
    if seconds is None or not np.isfinite(seconds):
        return None
    total = int(round(float(seconds)))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _load_stage_aware_reference_artifact() -> dict[str, Any] | None:
    """Load the RETIRED stage-aware total artifact (reference evidence only).

    The stage-aware total model no longer serves the ETA (EXP-038 sum policy);
    its per-request output is still evaluated and returned as
    ``stage_aware_reference_seconds`` for traceability/rollback comparison.
    Any load failure yields None — the reference is optional evidence and can
    never affect the served sum.
    """
    try:
        from dltime.stage_aware_serving import load_stage_aware_artifact

        return load_stage_aware_artifact()
    except Exception:
        return None


@dataclass
class PredictionResult:
    """Prediction response with explicit state and UI-ready metadata."""

    state: str
    reason: str | None = None
    predicted_seconds: float | None = None
    predicted_seconds_human: str | None = None
    predicted_log1p: float | None = None
    model_version: str | None = None
    model_status: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)
    history: dict[str, Any] = field(default_factory=dict)
    loader_case: dict[str, Any] | None = None  # the THREE gate cases (2026-10-02)
    features: dict[str, Any] | None = None
    stage_intelligence: dict[str, Any] | None = None  # supplementary only (Task 06)
    uncertainty: dict[str, Any] | None = None  # Task 07; PREDICTED only
    explanation: dict[str, Any] | None = None  # Task 08; PREDICTED only; never alters the ETA
    evidence_explanation: dict[str, Any] | None = None  # evidence-v1 sentences; PREDICTED only; never alters the ETA
    plain_explanation: str | None = None  # plain-explain-v1: ONE short plain-language reason; PREDICTED only
    risk: dict[str, Any] | None = None  # Task 09; interpretation only; never alters the ETA
    warnings: list[str] = field(default_factory=list)
    serving_role: str | None = None  # sum policy: AUTHORITATIVE_TOTAL (sum of five stages)
    direct_reference_seconds: float | None = None  # retired direct model (reference only)
    stage_aware_reference_seconds: float | None = None  # retired stage-aware model (reference only)
    physical_floor_seconds: float | None = None  # retired AD-34 floor (always None now)
    floor_applied: bool = False  # retired AD-34 floor (always False now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reason": self.reason,
            "predicted_seconds": self.predicted_seconds,
            "predicted_seconds_human": self.predicted_seconds_human,
            "predicted_log1p": self.predicted_log1p,
            "model_version": self.model_version,
            "model_status": self.model_status,
            "inputs": self.inputs,
            "identity": self.identity,
            "history": self.history,
            "loader_case": self.loader_case,
            "features": self.features,
            "stage_intelligence": self.stage_intelligence,
            "uncertainty": self.uncertainty,
            "explanation": self.explanation,
            "evidence_explanation": self.evidence_explanation,
            "plain_explanation": self.plain_explanation,
            "risk": self.risk,
            "warnings": list(self.warnings),
            "serving_role": self.serving_role,
            "direct_reference_seconds": self.direct_reference_seconds,
            "stage_aware_reference_seconds": self.stage_aware_reference_seconds,
            "physical_floor_seconds": self.physical_floor_seconds,
            "floor_applied": self.floor_applied,
        }


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def validate_request(request: PredictionRequest) -> tuple[dict[str, Any] | None, str | None]:
    """Structural/semantic validation of a prediction request.

    Returns (clean_fields, error_reason). clean_fields is None on error.
    Post-execution fields cannot be carried by PredictionRequest, but a
    defense-in-depth check keeps the contract explicit.
    """
    present = {f.name for f in fields(PredictionRequest)}
    leaked = present & FORBIDDEN_REQUEST_FIELDS
    if leaked:
        return None, f"post_execution_fields_rejected: {sorted(leaked)}"

    loader = _clean_str(request.loader_name)
    sprint = _clean_str(request.sprint)
    dataset = _clean_str(request.dataset)
    connection = _clean_str(request.connection)
    if loader is None:
        return None, "missing_required_field: loader_name"
    if sprint is None:
        return None, "missing_required_field: sprint"
    if dataset is None:
        return None, "missing_required_field: dataset"
    if connection is None:
        return None, "missing_required_field: connection"  # Task 10.1: compulsory

    planned = request.planned_start_time
    if planned is not None:
        ts = pd.to_datetime(planned, errors="coerce")
        if pd.isna(ts):
            return None, "unparseable_planned_start_time"
        # Normalize to NAIVE UTC before the value reaches the context row:
        # tz-aware strings (e.g. the browser's toISOString output) mixed into
        # the naive Start_Time column coerce to NaT during chronological
        # parsing and would 500 the request. UTC input is converted to naive
        # UTC; naive input passes through unchanged. Chronology safety is
        # preserved — the timestamp still sorts against recorded history.
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        planned = ts.isoformat(sep=" ")

    return {
        "loader_name": loader,
        "sprint": sprint,
        "dataset": dataset,
        "connection": connection,
        "workflow_execution_id": _clean_str(request.workflow_execution_id),
        "planned_start_time": planned,
        "preval_trans_display_name": _clean_str(request.preval_trans_display_name),
        "preval_trans_components": _clean_components(getattr(request, "preval_trans_components", ())) or None,
        "prevalidation_display_name": _clean_str(request.prevalidation_display_name),
        "transformation_display_name": _clean_str(request.transformation_display_name),
        "loader_type": _clean_str(request.loader_type),
        "calling_type": _clean_str(request.calling_type),
        "execution_order": request.execution_order,
        "prevalidation_count": request.prevalidation_count,
        "transformation_count": request.transformation_count,
        # Per-component echo for the identity audit block (list of dicts).
        "preval_trans_components_echo": (
            [
                {
                    "display_name": c.display_name,
                    "calling_type": c.calling_type,
                    "component_type": c.component_type,
                    "execution_order": c.execution_order,
                }
                for c in _clean_components(getattr(request, "preval_trans_components", ()))
            ]
            or None
        ),
    }, None


class PredictionService:
    """Pre-execution prediction over the persisted Task 03 candidate artifact."""

    def __init__(
        self,
        *,
        training_frame: TrainingFrame,
        artifact: dict,
        foundation: FoundationReport,
        resolver: ConnectionResolver,
        definitions_detail: pd.DataFrame,
        definitions_summary: pd.DataFrame,
        eligibility_config: HistoryEligibilityConfig | None = None,
    ) -> None:
        self.tf = training_frame
        self.artifact = artifact
        self.foundation = foundation
        self.resolver = resolver
        self.definitions_detail = definitions_detail
        self.definitions_summary = definitions_summary
        self.eligibility_config = eligibility_config or HistoryEligibilityConfig()
        self._engine = HistoryEligibilityEngine(self.eligibility_config)
        # Task 07: versioned calibration for the uncertainty layer. Bound to
        # the authoritative total policy; built once on first use if not yet
        # persisted.
        self._calibration: dict[str, Any] | None = None
        # Task 08: cached TreeExplainer (explanatory only; lazily created).
        self._explainer_svc: Any | None = None
        # Task 09: OOD profile over the artifact's training configuration range
        # (built once; config features only).
        self._ood_profile_cache: dict[str, Any] | None | bool = None
        # Task 09: dataset-wide MODELABLE stage names (built once).
        self._modelable_stages_cache: set[str] | None = None
        # Sum-total policy (EXP-038): the calibration for the served ETA is
        # the dedicated sum-total residual calibration (EXP-036 clean
        # protocol). The stage-aware artifact is no longer loaded/served.
        # AD-31: loaded ML stage-model predictors (built once).
        self._stage_models_cache: dict[str, object] | None = None
        # Serving-switch snapshot the cache was built from (2026-10-06):
        # promotion/rollback flip the persisted switch file; the serving path
        # MUST follow it automatically (Prediction Studio uses whatever is
        # approved and serving) instead of serving a stale in-memory model.
        self._stage_models_switch_key: Any = None
        self._direct_calibration: dict[str, Any] | None = None
        self._sum_total_calibration: dict[str, Any] | None = None
        self._combined_frame: pd.DataFrame | None = None
        self._context_with_stage_history: pd.DataFrame | None = None

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_sources(
        cls,
        *,
        artifact: dict | None = None,
        eligibility_config: HistoryEligibilityConfig | None = None,
    ) -> "PredictionService":
        """Build the full service from the real source files."""
        from dltime.feature_assembly import build_training_frame
        from dltime.pipeline import build_foundation
        from dltime.resolution import ConnectionResolver
        from dltime.sources import load_all_sources

        bundle = load_all_sources()
        foundation = build_foundation(bundle)
        tf = build_training_frame(bundle=bundle)
        if artifact is None:
            from dltime.artifact import (
                ArtifactError,
                build_serving_artifact,
                load_serving_artifact,
                save_serving_artifact,
            )

            try:
                artifact = load_serving_artifact()
            except ArtifactError:
                # Artifact not persisted yet (first run on a fresh checkout):
                # build the deterministic candidate on the recorded training
                # window once and persist it. This is NOT retraining — it is
                # the same approved configuration fitted identically.
                artifact = build_serving_artifact(tf)
                save_serving_artifact(artifact)
        resolver = ConnectionResolver(
            bundle.loader_definitions_detail, bundle.loader_definitions_summary
        )
        return cls(
            training_frame=tf,
            artifact=artifact,
            foundation=foundation,
            resolver=resolver,
            definitions_detail=bundle.loader_definitions_detail,
            definitions_summary=bundle.loader_definitions_summary,
            eligibility_config=eligibility_config,
        )

    # ------------------------------------------------------- context assembly
    def _synthetic_execution_id(self) -> Any:
        ids = self.foundation.frames["executions"]["LDR_Execution_Id"]
        if pd.api.types.is_integer_dtype(ids) or pd.api.types.is_float_dtype(ids):
            return int(pd.to_numeric(ids).max()) + 1
        return "__PREDICTION_REQUEST__"

    def _default_start_time(self) -> pd.Timestamp:
        starts = pd.to_datetime(
            self.foundation.frames["executions"]["Start_Time"], errors="coerce"
        )
        return starts.max() + pd.Timedelta(days=1)

    def build_context_row(
        self,
        *,
        loader_name: str,
        sprint: str,
        dataset: str,
        resolution_status: ResolutionStatus,
        resolved_connection: str | None,
        start_time: pd.Timestamp,
        execution_id: Any,
        workflow_execution_id: str | None,
        config_overrides: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Append the synthetic pre-execution context row and run the UNMODIFIED
        Task 02 feature path (definition features + chronological history).

        The context row carries target=<NA> and is_training_eligible=False so
        it can never contribute to its own (or any) historical aggregate
        (AD-11). Only strictly-earlier training-eligible executions feed the
        historical features, exactly as in training.

        config_overrides (AD-31): user-supplied legitimate pre-execution
        configuration for THIS execution. Applied AFTER the definition join so
        the caller's real values take precedence over definition-derived
        aggregates for the context row only; historical rows are untouched.
        """
        executions = self.foundation.frames["executions_eligible"]
        row: dict[str, Any] = {c: pd.NA for c in executions.columns}
        row.update(
            {
                "LDR_Execution_Id": execution_id,
                "ldr_workflow_execution_id": workflow_execution_id,
                "Loader_Name": loader_name,
                "Sprint": sprint,
                "Dataset_display_name": dataset,
                "datasets": dataset,
                "n_datasets": 1,
                "n_component_rows": 1,
                "n_target_rows": 0,
                "LDR_Status": pd.NA,  # unknown before execution; never fabricated
                "Start_Time": start_time,
                "End_Time": pd.NA,
                "Time_Taken": pd.NA,
                "timetaken_in_sec": pd.NA,  # TARGET unknown pre-execution
                "execution_resolution_status": resolution_status.value,
                "execution_connection": (
                    resolved_connection if resolved_connection is not None else pd.NA
                ),
                "is_valid_execution": False,
                "is_target_valid": False,
                "is_training_eligible": False,
                "eligibility_reasons": "prediction_request_context",
            }
        )

        combined = pd.concat(
            [executions, pd.DataFrame([row])], ignore_index=True
        )

        # Exact Task 02 path: definition features then chronological history.
        from dltime.definition_features import attach_definition_features
        from dltime.historical import build_historical_features

        with_def = attach_definition_features(
            combined, self.definitions_detail, self.definitions_summary
        )
        if config_overrides:
            with_def = _apply_context_config_overrides(with_def, execution_id, config_overrides)
        hist = build_historical_features(with_def, prior_filter="training_eligible")
        full = hist.frame
        self._last_combined_frame = full
        ctx = full[full["LDR_Execution_Id"] == execution_id]
        if len(ctx) != 1:
            raise RuntimeError(
                f"Context row resolution failed ({len(ctx)} matches for {execution_id!r})."
            )
        return ctx

    def _context_feature_vector(self, ctx: pd.DataFrame) -> pd.DataFrame:
        """One-row feature matrix with the exact artifact feature order/dtypes."""
        feature_names = list(self.artifact["metadata"]["feature_names_in_order"])
        X = ctx[feature_names].copy()
        categorical = set(self.artifact["metadata"]["categorical_features"])
        for col in feature_names:
            if col in categorical:
                X[col] = X[col].astype("string")
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        return X

    # ---------------------------------------------------------------- helpers
    def _reported_features(self, X: pd.DataFrame, ctx: pd.DataFrame) -> dict[str, Any]:
        """Feature report: the fitted 44-feature artifact contract
        (2026-10-04 spec) in the artifact's saved order. Audit-marker columns
        (def_preval_name_component / def_trans_name_component, EXP-032 Part G
        display-name overrides) are appended only when present in the context
        frame; they are audit context and never reach the served model."""
        reported: dict[str, Any] = {}
        for name in X.columns:
            val = X.iloc[0][name]
            reported[name] = (
                self._num(val)
                if not isinstance(val, str)
                else val
            )
        for marker in ("def_preval_name_component", "def_trans_name_component"):
            if marker in ctx.columns:
                val = ctx[marker].iloc[0]
                reported[marker] = None if pd.isna(val) else str(val)
        return reported

    # ---------------------------------------------------------------- predict
    def predict(self, request: PredictionRequest) -> PredictionResult:
        clean, error = validate_request(request)
        inputs = {
            "loader_name": _clean_str(request.loader_name),
            "sprint": _clean_str(request.sprint),
            "dataset": _clean_str(request.dataset),
            "connection": _clean_str(request.connection),
            "workflow_execution_id": _clean_str(request.workflow_execution_id),
            "planned_start_time": request.planned_start_time,
        }
        if error is not None:
            result = PredictionResult(
                state=PredictionState.INVALID_INPUT, reason=error, inputs=inputs
            )
            self._persist_event(result)
            return result

        # ---- canonical identity resolution (locked rules; never silently chosen)
        status, resolved_connection = self.resolver.resolve(
            clean["dataset"], clean["sprint"], clean["loader_name"]
        )
        identity = {
            "dataset": clean["dataset"],
            "sprint": clean["sprint"],
            "loader_name": clean["loader_name"],
            "resolution_status": status.value,
            "resolved_connection": resolved_connection,
            # Prompt 3 (§14/§15): the caller's workflow reference is persisted
            # in the identity block as LINKAGE METADATA ONLY (exact
            # workflow-reference matching at ingestion; never a model feature,
            # never explained, never exposed as a prediction input).
            "ldr_workflow_execution_id": clean["workflow_execution_id"],
            # AD-31: echo of the caller-supplied pre-execution configuration
            # (audit context only; historical aggregates remain untouched).
            "supplied_configuration": {
                k: clean[k]
                for k in (
                    "preval_trans_display_name",
                    "prevalidation_display_name",
                    "transformation_display_name",
                    "loader_type",
                    "calling_type",
                    "execution_order",
                    "prevalidation_count",
                    "transformation_count",
                    "preval_trans_components_echo",
                )
                if clean.get(k) is not None
            },
            # AD-38 §3/§7: per-field provenance for the configuration inputs.
            # "user_supplied" = the caller sent this pre-execution value;
            # "definition_derived" = the field was left blank and the registered
            # canonical definition supplied it. Nothing is ever fetched from a
            # user input the user did not give: the derived path is disclosed so
            # a blank field is never silently presented as user-asserted data.
            "configuration_provenance": {
                k: ("user_supplied" if clean.get(k) is not None else "definition_derived")
                for k in (
                    "preval_trans_display_name",
                    "prevalidation_display_name",
                    "transformation_display_name",
                    "loader_type",
                    "calling_type",
                    "execution_order",
                    "prevalidation_count",
                    "transformation_count",
                )
            },
        }
        if status is ResolutionStatus.UNRESOLVED:
            # 2026-10-03 (user final spec): a COMPLETELY new loader — no
            # definition match at all — is treated EXACTLY like a
            # registered-but-never-run loader: the same short "New loader"
            # refusal (INSUFFICIENT_HISTORY, no total fabricated), with
            # post-execution entry available so real executions build this
            # loader's history toward the trust threshold. The attempt is
            # still persisted with the full pre-execution inputs for the
            # audit trail. (AMBIGUOUS below STAYS a hard refusal: the system
            # never silently chooses between conflicting definitions.)
            from dltime.stage_v2 import assess_loader_case

            threshold = self.eligibility_config.min_prior_loader_executions
            loader_case = assess_loader_case(
                self.foundation.frames["executions_eligible"],
                clean["loader_name"],
                min_prior=threshold,
            )
            result = PredictionResult(
                state=PredictionState.INSUFFICIENT_HISTORY,
                reason=(
                    "new loader: no definition match for Dataset+Sprint+Loader — "
                    "treated as a brand-new loader; history accumulates from "
                    "saved actuals"
                ),
                inputs=inputs,
                identity=identity,
                warnings=[
                    "identity_not_registered: no definition match — treated as a "
                    "new loader; post-execution records build its history"
                ],
            )
            result.loader_case = loader_case
            result.history = {
                "prior_loader_executions": loader_case.get("prior_count", 0),
                "eligibility_policy": {"min_prior_loader_executions": threshold},
            }
            self._persist_event(result)
            return result
        if status is ResolutionStatus.AMBIGUOUS:
            result = PredictionResult(
                state=PredictionState.AMBIGUOUS_IDENTITY,
                reason="identity matches multiple definition connections; never silently chosen",
                inputs=inputs,
                identity=identity,
            )
            self._persist_event(result)
            return result
        if clean["connection"] is not None and clean["connection"] != resolved_connection:
            result = PredictionResult(
                state=PredictionState.INVALID_INPUT,
                reason=(
                    "provided_connection_disagrees_with_definition_resolution: "
                    f"{clean['connection']!r} != {resolved_connection!r}"
                ),
                inputs=inputs,
                identity=identity,
            )
            # Module 8 hardening: this attempt IS audited like every other
            # state (Task 11: prediction events persist for ALL five states).
            self._persist_event(result)
            return result

        # ---- historical intelligence + configuration features (exact Task 02 path)
        start_time = (
            pd.to_datetime(clean["planned_start_time"])
            if clean["planned_start_time"] is not None
            else self._default_start_time()
        )
        ctx = self.build_context_row(
            loader_name=clean["loader_name"],
            sprint=clean["sprint"],
            dataset=clean["dataset"],
            resolution_status=status,
            resolved_connection=resolved_connection,
            start_time=start_time,
            execution_id=self._synthetic_execution_id(),
            workflow_execution_id=clean["workflow_execution_id"],
            config_overrides={
                k: clean[k] for k in _CONFIG_OVERRIDE_FIELDS if clean.get(k) is not None
            },
        )
        self._combined_frame = self._last_combined_frame
        # User final spec (2026-09-30): the applicable stage set for THIS
        # execution comes from the user's own pre-execution entries. Kept on
        # the engine for the stage-intel/total filtering below.
        self._last_config_overrides = {
            k: clean[k] for k in _CONFIG_OVERRIDE_FIELDS if clean.get(k) is not None
        }
        # Per-component rows (when supplied) are the authoritative applicable-
        # stage signal for THIS execution — kept separately from the scalar
        # overrides so determine_applicable_stages can read them directly.
        if clean.get("preval_trans_components"):
            self._last_preval_trans_components = clean["preval_trans_components"]
        else:
            self._last_preval_trans_components = ()

        # ---- history eligibility (AD-16, configurable)
        eligibility: HistoryEligibility = self._engine.evaluate(
            hist_loader_prior_count=ctx["hist_loader_prior_count"].iloc[0],
            hist_loader_conn_prior_count=ctx["hist_loader_conn_prior_count"].iloc[0],
            hist_time_since_prev_loader_sec=ctx["hist_time_since_prev_loader_sec"].iloc[0],
        )
        history = {
            "prior_loader_executions": eligibility.evidence["prior_loader_executions"],
            "prior_loader_conn_executions": eligibility.evidence[
                "prior_loader_conn_executions"
            ],
            "time_since_prev_loader_sec": eligibility.evidence[
                "time_since_prev_loader_sec"
            ],
            "prior_loader_mean_sec": self._num(
                ctx["hist_loader_prior_mean_sec"].iloc[0]
            ),
            "prior_loader_median_sec": self._num(
                ctx["hist_loader_prior_median_sec"].iloc[0]
            ),
            "prior_loader_std_sec": self._num(
                ctx["hist_loader_prior_std_sec"].iloc[0]
            ),
            "previous_loader_duration_sec": self._num(
                ctx["hist_prev_loader_duration_sec"].iloc[0]
            ),
            "global_prior_count": self._num(ctx["hist_global_prior_count"].iloc[0]),
            "planned_start_time": str(start_time),
            "eligibility_policy": {
                "min_prior_loader_executions": self.eligibility_config.min_prior_loader_executions,
                "source": "AD-16 (approved configurable default; Task 03 evidence EXP-012)",
            },
        }

        warnings: list[str] = []
        if not bool(ctx["def_match"].iloc[0]):
            warnings.append(
                "No definition/configuration records matched the identity; "
                "configuration features are missing for this context."
            )
        if (history["prior_loader_conn_executions"] or 0) == 0:
            warnings.append(
                "No prior loader+connection history; connection-aware historical "
                "features are missing for this context."
            )

        # v2 main-goal policy (2026-09-29): when the stage-v2 bundle is the
        # serving model set, the LOADER THRESHOLD GATE (assess_loader_case,
        # strictly REAL training-eligible priors, threshold 3) is the
        # authoritative availability gate — it supersedes the legacy AD-16
        # history gate, whose prior counts were computed on the frame that now
        # also carries synthetic coverage rows. The v2 gate runs further below
        # and refuses UNKNOWN_LOADER / BELOW_THRESHOLD requests with the
        # per-stage indicative estimates attached, so the legacy early-return
        # here must NOT fire in that regime.
        if not eligibility.sufficient and not self._v2_stage_gate_active():
            result = PredictionResult(
                state=PredictionState.INSUFFICIENT_HISTORY,
                reason=eligibility.reason,
                inputs=inputs,
                identity=identity,
                history=history,
                risk=insufficient_history_risk(eligibility.reason),
                warnings=warnings,
            )
            self._persist_event(result)
            return result

        # ---- authoritative ETA: EXACT SUM of the stage predictions
        # (v2 main-goal policy, 2026-09-29). The loader threshold gate decides
        # the THREE availability cases FIRST: an unknown loader or a loader
        # below the trust threshold gets honest per-stage estimates but NO
        # total (never fabricated). Under the v2 regime the authoritative sum
        # is sum_total_v2: ALL trainable ML stages (staging now included when
        # trained) + the contextual statistical approval estimates, per the
        # user contract "total = stage predictions + approvals". The legacy
        # five-stage sum (approvals excluded) is retired from serving and kept
        # only as a reference/rollback path.
        from dltime.stage_v2 import SUM_VERSION, assess_loader_case, sum_total_v2

        # The gate runs FIRST (it needs only the real-row eligibility mask, so
        # it stays correct even on a malformed frame). Stage intelligence is
        # computed next: always for trusted loaders, and best-effort for
        # gate-blocked ones so the refusal can carry indicative per-stage
        # estimates (if even that fails — e.g. a frame without eligibility
        # evidence — the request fails CLOSED with the gate refusal, never an
        # ETA).
        # The configured per-service threshold flows into the gate (test- and
        # deployment-overridable via HistoryEligibilityConfig).
        loader_case = assess_loader_case(
            self._combined_frame,
            request.loader_name,
            min_prior=self.eligibility_config.min_prior_loader_executions,
        )
        stage_intel = None
        try:
            stage_intel = self.stage_intelligence_from_context(ctx, self._combined_frame)
        except Exception as exc:
            if loader_case.get("total_available"):
                raise
            stage_intel = None
            warnings.append(f"Stage intelligence unavailable on this context: {str(exc)[:140]}")
        sum_total = None
        refusal_reason = ""
        if loader_case.get("total_available"):
            sum_total = sum_total_v2(stage_intel, loader_case)
            if not sum_total.get("sum_available"):
                refusal_reason = sum_total.get("reason", "")
        else:
            # Gate blocked (UNKNOWN_LOADER / BELOW_THRESHOLD): per-stage ML
            # estimates (when computable) are attached to the refusal so the
            # user sees the indicative values AND the exact threshold message.
            refusal_reason = loader_case["message"]
            sum_total = (
                sum_total_v2(stage_intel, loader_case)
                if stage_intel is not None
                else {"sum_version": SUM_VERSION, "sum_available": False,
                      "reason": refusal_reason, "total_seconds": None}
            )
        if sum_total is None or not sum_total.get("sum_available"):
            reason = refusal_reason or (sum_total or {}).get("reason", "total unavailable")
            result = PredictionResult(
                state=PredictionState.INSUFFICIENT_HISTORY,
                reason=reason,
                inputs=inputs,
                identity=identity,
                history=history,
                risk=insufficient_history_risk(reason),
                warnings=warnings,
            )
            result.stage_intelligence = stage_intel
            result.loader_case = loader_case
            self._persist_event(result)
            return result
        predicted_seconds = float(
            sum_total["ml_stage_sum_seconds"]
            if "ml_stage_sum_seconds" in sum_total
            else sum_total["total_seconds"]
        )
        meta: dict[str, Any] = {
            "model_version": sum_total.get(
                "model_version", sum_total.get("sum_version", SUM_VERSION)
            ),
            "status": "DETERMINISTIC_POLICY",
            "role": "AUTHORITATIVE_TOTAL",
            "estimator": sum_total.get("estimator", sum_total.get("sum_version", SUM_VERSION)),
            "loader_case": loader_case,
        }
        # Reference estimators (traceability/rollback evidence only): the
        # retired stage-aware total and the direct total model are still
        # evaluated for the same context, clearly labelled, NEVER displayed
        # as the ETA, and never summed into it.
        X_ref = _stage_aware_feature_vector(
            self._context_with_stage_history if self._context_with_stage_history is not None else ctx,
            getattr(self, "_stage_intel_unfiltered", None) or stage_intel,
        )
        from dltime.stage_aware_serving import (  # retired serving path; kept as evidence
            predict_stage_aware_seconds,
        )

        try:
            stage_aware_reference = predict_stage_aware_seconds(
                _load_stage_aware_reference_artifact(), X_ref
            )
        except Exception:
            stage_aware_reference = None
        # Direct-total reference model (rollback only): kept for comparison
        # evidence; never the served ETA.
        X = X_ref
        # Contract feature vector (the SERVED artifact's exact feature
        # contract) — this is what the response and the persisted audit
        # event report as ``features`` (AD-32 pin: the audit carries the
        # artifact contract, never the retired stage-aware reference
        # vector's columns).
        X_direct = self._context_feature_vector(ctx)
        direct_total_reference = predict_with_artifact(self.artifact, X_direct)
        # Task 07: calibrated uncertainty interval (layer AROUND the point ETA;
        # the point prediction above is untouched). Never on other states.
        try:
            uncertainty = self._uncertainty_section(
                predicted_seconds=predicted_seconds,
                history=history,
            )
        except Exception as exc:
            uncertainty = None
            warnings.append(f"Uncertainty unavailable: {exc}")
        # Task 08: explanation of the ACTUAL served total — under the sum
        # policy the honest explanation IS the decomposition (per-stage
        # additive components + explicit non-components); the stage-aware
        # SHAP explainer is no longer on the serving path.
        try:
            explanation = self._explanation_section(X, stage_intel, predicted_seconds, ctx=ctx)
        except Exception as exc:
            # v2 reality: the legacy five-stage decomposition raises when only
            # four stages serve (staging has no training data on the replaced
            # export). The explanation is supplementary — the point ETA, its
            # stage decomposition in stage_intelligence, and the loader gate
            # stay fully intact, so degrade soft instead of failing the request.
            explanation = None
            warnings.append(
                f"Explanation unavailable: {str(exc)[:160]}"
            )
        # Task 09: operational risk (interpretation layer; failure never
        # removes or changes the point ETA, uncertainty, or explanation).
        try:
            risk = self._risk_section(
                predicted_seconds=predicted_seconds,
                uncertainty=uncertainty,
                history=history,
                X=X,
                stage_intelligence=stage_intel,
            )
        except Exception as exc:
            risk = None
            warnings.append(f"Risk assessment unavailable: {exc}")
        # Evidence-sentence explainability (user final spec, 2026-10-01):
        # Layer 1 per-stage sentences + Layer 2 what-ifs, deterministic
        # templates over real numbers. Supplementary — never alters the ETA.
        try:
            from dltime.explain_evidence import build_evidence_explanation

            _unfiltered = getattr(self, "_stage_intel_unfiltered", None) or {}
            unfiltered_estimates = {
                s.get("stage"): s.get("estimate_seconds")
                for s in (_unfiltered.get("stages") or [])
                if s.get("method") == "ml_stage_model"
            }
            evidence_explanation = build_evidence_explanation(
                stage_intel=stage_intel,
                unfiltered_stage_estimates=unfiltered_estimates,
                history=history,
                ctx_stage_history=getattr(self, "_last_ctx_stage_history", {}) or {},
                total_seconds=predicted_seconds,
                applicable=(stage_intel or {}).get("applicable_stages") or [],
                not_applicable=(stage_intel or {}).get("not_applicable_stages") or [],
            )
        except Exception as exc:
            evidence_explanation = None
            warnings.append(f"Evidence explanation unavailable: {str(exc)[:140]}")
        # plain-explain-v1 (2026-10-05 user spec): ONE short plain-language
        # reason (≤2 sentences, non-technical) for the total. Supplementary
        # only — never alters the ETA.
        try:
            from dltime.explain_evidence import build_plain_explanation

            plain_explanation = build_plain_explanation(
                stage_intel=stage_intel,
                history=history,
                ctx_stage_history=getattr(self, "_last_ctx_stage_history", {}) or {},
                total_seconds=predicted_seconds,
                uncertainty=uncertainty,
            )
        except Exception as exc:
            plain_explanation = None
            warnings.append(f"Plain explanation unavailable: {str(exc)[:140]}")
        result = PredictionResult(
            state=PredictionState.PREDICTED,
            predicted_seconds=float(predicted_seconds),
            predicted_seconds_human=human_readable_duration(predicted_seconds),
            predicted_log1p=float(np.log1p(predicted_seconds)),
            model_version=meta["model_version"],
            model_status=meta["status"],
            serving_role=meta.get("role"),
            direct_reference_seconds=(
                float(direct_total_reference) if direct_total_reference is not None else None
            ),
            stage_aware_reference_seconds=(
                float(stage_aware_reference) if stage_aware_reference is not None else None
            ),
            physical_floor_seconds=None,
            floor_applied=False,
            inputs=inputs,
            identity=identity,
            history=history,
            features=self._reported_features(X_direct, ctx),
            stage_intelligence=stage_intel,
            uncertainty=uncertainty,
            explanation=explanation,
            evidence_explanation=evidence_explanation,
            plain_explanation=plain_explanation,
            risk=risk,
            warnings=warnings,
        )
        self._persist_event(result)
        return result

    # ------------------------------------------------ monitoring (Task 11)
    def shap_explanation_for_request(self, request: "PredictionRequest") -> dict[str, Any]:
        """Technical drill-down (2026-10-06 user spec): recompute the served
        prediction for the SAME pre-execution inputs and return the real
        per-prediction SHAP attribution layer (explanation.shap_explanation).

        READ-ONLY: the recomputation's monitoring event is suppressed, so no
        duplicate prediction row is persisted and nothing (state, ETA, serving
        path) is altered. Non-PREDICTED states explain nothing — fail closed.
        """
        original = self._persist_event
        self._persist_event = lambda result: None  # type: ignore[method-assign]
        try:
            result = self.predict(request)
        finally:
            self._persist_event = original  # type: ignore[method-assign]
        if result.state != PredictionState.PREDICTED:
            return {
                "available": False,
                "state": result.state,
                "reason": result.reason or "prediction withheld — nothing to explain",
            }
        shap_layer = ((result.explanation or {}).get("shap_explanation")) or None
        if not shap_layer or not shap_layer.get("available"):
            return {
                "available": False,
                "state": result.state,
                "reason": "SHAP layer unavailable for this prediction",
            }
        return shap_layer

    def _persist_event(self, result: "PredictionResult") -> None:
        """Append the prediction attempt to the monitoring store (ALL states).

        Isolated exactly like SHAP/risk: any store failure appends a warning
        to the ALREADY-DECIDED result and never changes its state, ETA, or
        any other output. The synthetic context execution id is deliberately
        NOT persisted as a linkage reference — it is an internal identifier,
        not a real-world execution reference (see AD-24: linkage is exact
        workflow-reference only; pre-execution no real LDR_Execution_Id
        exists).
        """
        try:
            from dltime.feedback import finalize_event_id, prediction_event_from_result
            from dltime.store import MonitoringStore

            event = prediction_event_from_result(
                result,
                planned_start_time=(result.inputs or {}).get("planned_start_time"),
            )
            event = finalize_event_id(event)
            MonitoringStore().append_prediction_event(event)
            # AD-33 §24 / AD-37 §16/§26: only SYSTEM-LEVEL event signals become
            # persistent Alert Center items (new-loader discovery, withheld
            # prediction, HIGH_RISK pattern). Per-prediction OOD/low-confidence
            # signals stay on the Prediction Result as transient risk — they
            # must not flood the Alert Center (documented policy).
        except Exception as exc:  # pragma: no cover - persistence isolation
            try:
                result.warnings.append(f"Prediction monitoring unavailable: {exc}")
            except Exception:
                pass

    # ---------------------------------------------------- stage intelligence
    def stage_intelligence(self, request: PredictionRequest) -> dict[str, Any]:
        """SEVEN-STAGE execution view for one prediction context (AD-31).

        Five ML stage models + two contextual approval estimates, built on the
        SAME identity/eligibility boundary as predict(): only RESOLVED
        identities receive the full view; anything else returns
        available=False with the exact reason. The five operational stage
        predictions ARE the authoritative total's exact additive components
        (sum policy, EXP-038); approval estimates are contextual and never
        summed into the total.
        """
        from dltime.stage_intel import build_stage_intel_for_context, summarize_stage_intel

        clean, error = validate_request(request)
        if error is not None:
            return {"available": False, "reason": error, "stages": []}
        status, resolved_connection = self.resolver.resolve(
            clean["dataset"], clean["sprint"], clean["loader_name"]
        )
        if status is not ResolutionStatus.RESOLVED:
            return {
                "available": False,
                "reason": f"identity_{status.value.lower()}",
                "stages": [],
            }

        start_time = (
            pd.to_datetime(clean["planned_start_time"])
            if clean["planned_start_time"] is not None
            else self._default_start_time()
        )
        execution_id = self._synthetic_execution_id()
        ctx = self.build_context_row(
            loader_name=clean["loader_name"],
            sprint=clean["sprint"],
            dataset=clean["dataset"],
            resolution_status=status,
            resolved_connection=resolved_connection,
            start_time=start_time,
            execution_id=execution_id,
            workflow_execution_id=clean["workflow_execution_id"],
            config_overrides={
                k: clean[k] for k in _CONFIG_OVERRIDE_FIELDS if clean.get(k) is not None
            },
        )
        # Attach the AD-31 stage-target columns to the SAME combined frame the
        # context row lives in, so strictly-prior approval evidence and the
        # stage-intel engine see one consistent chronological frame. The
        # foundation frame already carries the stage-target columns (pipeline
        # Part F); the appended context row gets NaN for all of them via the
        # concat alignment — exactly the stage-NA context guard contract.
        combined = pd.concat(
            [self.foundation.frames["executions_eligible"], ctx], ignore_index=True
        )

        # ---- context row stage-history features (strictly prior only) -------
        from dltime.stage_intel import ML_STAGES

        combined = _attach_context_stage_history(
            combined, execution_id, ML_STAGES, stage_specific=True
        )
        intel = build_stage_intel_for_context(
            combined, execution_id, stage_models=self._stage_models()
        )
        summary = summarize_stage_intel(intel.per_stage)
        return {
            "available": True,
            "note": (
                "Seven-Stage Execution View: five ML stage predictions + two "
                "contextual statistical approval estimates. The five operational "
                "stage predictions sum EXACTLY to the served total ETA (sum "
                "policy, EXP-038); the two approval estimates are contextual "
                "and are NOT part of the total. ML stage outputs are model "
                "predictions, never actuals."
            ),
            "ordering": "Start_Time, tie-break LDR_Execution_Id (deterministic)",
            "prior_filter": "training_eligible",
            "policy_version": summary.get("policy_version"),
            "availability_summary": summary,
            "stages": [dict(v) for v in intel.per_stage.values()],
        }

    def _v2_stage_gate_active(self) -> bool:
        """True when the v2 stage bundle is the serving model set.

        In that regime the v2 loader threshold gate (assess_loader_case over
        strictly REAL training-eligible priors) is the authoritative
        availability gate and supersedes the legacy AD-16 history gate.
        """
        try:
            from dltime.stage_v2 import load_stage_v2_bundle

            v2 = load_stage_v2_bundle()
            return bool(v2 is not None and getattr(v2, "predictors", None))
        except Exception:
            return False

    def _stage_models(self) -> dict[str, object]:
        """Loaded stage-model predictors for the serving path.

        2026-09-29 (main-goal v2): the stage-v2 bundle (trained on the FILLED
        dataset, per-stage model comparison) is preferred when present and is
        used ALL-OR-NOTHING — never merged with the legacy predictors, so a
        stage the v2 data cannot train (staging) is honestly UNAVAILABLE
        instead of silently served by the stale legacy model.

        2026-10-06 (promotion/rollback liveness): when a serving override is
        active (a promoted candidate bundle or per-stage candidate), the
        switch-file state is re-checked on every call and the predictors are
        reloaded when it changed — the promoted model starts serving
        IMMEDIATELY (no process restart), and a rollback restores the
        previous serving set the same way. With no override the v2-bundle
        preference above applies unchanged.
        """
        from dltime import retraining_serving as _rs

        try:
            switch_key: Any = (
                _rs.active_version(),
                tuple(
                    sorted(
                        (str(stage), str(ov.get("artifact_path", "")))
                        for stage, ov in (_rs.active_stage_overrides() or {}).items()
                    )
                ),
            )
        except Exception:
            switch_key = (None, ())
        override_active = bool(switch_key[0]) or bool(switch_key[1])
        if (
            self._stage_models_cache is None
            or self._stage_models_switch_key != switch_key
        ):
            v2 = None
            if not override_active:
                try:
                    from dltime.stage_v2 import load_stage_v2_bundle

                    v2 = load_stage_v2_bundle()
                except Exception:
                    v2 = None
            if v2 is not None and getattr(v2, "predictors", None):
                self._stage_models_cache = dict(v2.predictors)
            else:
                from dltime.retraining_serving import load_stage_serving_models_switched

                self._stage_models_cache = load_stage_serving_models_switched()
            self._stage_models_switch_key = switch_key
        return self._stage_models_cache

    # ------------------------------------------------------- uncertainty (T07)
    def _total_calibration_metadata(self) -> dict[str, Any]:
        """Calibration bound to the AUTHORITATIVE total policy (built lazily).

        Sum-total policy (EXP-038): the served ETA is the deterministic sum
        of the five operational stage predictions, so the uncertainty layer
        uses the dedicated sum-total residual calibration artifact
        (unc-log1p-conformal-v1-sum-of-five-stages), built on the EXP-036
        clean leakage-safe holdout. Fail-soft with the exact reason surfaced
        as a warning; never fabricates evidence.
        """
        if self._calibration is None:
            from dltime.sum_total_calibration import (
                SumTotalCalibrationError,
                load_sum_total_calibration,
            )

            try:
                self._calibration = load_sum_total_calibration()
            except SumTotalCalibrationError:
                from dltime.uncertainty import ensure_calibration

                self._calibration = ensure_calibration(self.tf, self.artifact)
        return self._calibration

    def calibration_metadata(self) -> dict[str, Any]:
        """Versioned calibration for the DIRECT reference artifact (rollback path).

        Kept for the reference/rollback model's own diagnostics; the served
        uncertainty uses _total_calibration_metadata (stage-aware binding).
        """
        if self._direct_calibration is None:
            from dltime.uncertainty import ensure_calibration

            self._direct_calibration = ensure_calibration(self.tf, self.artifact)
        return self._direct_calibration

    def _uncertainty_section(
        self, *, predicted_seconds: float, history: dict[str, Any]
    ) -> dict[str, Any]:
        """Task 07 calibrated uncertainty block (PREDICTED path only)."""
        from dltime.uncertainty import uncertainty_section

        calib = self._total_calibration_metadata()
        return uncertainty_section(
            predicted_seconds=predicted_seconds,
            calibration_metadata=calib,
            prior_loader_executions=int(history.get("prior_loader_executions") or 0),
            min_required_history=self.eligibility_config.min_prior_loader_executions,
            evidence={
                "n_calibration_rows": calib["n_calibration_rows"],
                "model_version": calib.get("model_version"),
            },
        )

    # ----------------------------------------------------- explainability (T08)
    def _explanation_section(
        self,
        X: pd.DataFrame,
        stage_intel: dict[str, Any],
        predicted_seconds: float,
        *,
        ctx: Any = None,
    ) -> dict[str, Any]:
        """Task 08 explanation block for the ACTUAL served total (sum policy).

        Two layers, both from existing serving objects (no second feature
        pipeline, no fabricated evidence):
          1. real per-stage SHAP/linear attributions (Task 08 machinery on the
             five stage models' OWN outputs) — the human-readable "why";
          2. the exact five-stage decomposition of the served total
             (sum_total_explanation) — the deterministic arithmetic contract.
        """
        decomposition = sum_total_explanation(stage_intel, predicted_seconds)
        shap_layer: dict[str, Any] | None = None
        try:
            from dltime.stage_shap_explanation import stage_shap_explanation

            ctx_hist = getattr(self, "_context_with_stage_history", None)
            if ctx_hist is not None and len(ctx_hist) == 1:
                if not hasattr(self, "_stage_shap_cache"):
                    self._stage_shap_cache = {}
                shap_layer = stage_shap_explanation(
                    self._stage_models(),
                    ctx_hist,
                    ctx_hist["LDR_Execution_Id"].iloc[0],
                    stage_intel,
                    cache=self._stage_shap_cache,
                )
        except Exception:
            shap_layer = None  # explanatory-only: the decomposition still stands
        explanation = dict(decomposition)
        if shap_layer:
            explanation["shap_explanation"] = {
                "available": True,
                "policy_version": shap_layer.get("policy_version"),
                "method": shap_layer.get("method"),
                "stages_explained": shap_layer.get("stages_explained"),
                "stages_skipped": shap_layer.get("stages_skipped"),
                "top_contributors": shap_layer.get("top_contributors"),
                "human_readable_summary": shap_layer.get("human_readable_summary"),
            }
            explanation["human_readable_summary"] = shap_layer["human_readable_summary"]
        return explanation

    # ------------------------------------------------------- risk (T09)
    def _ood_profile(self) -> dict[str, Any] | None:
        """Config-range profile over the stage population's training window.

        Sum-total policy (EXP-038): the served total is the five-stage sum,
        whose stage models train on the stage-frame population, so that
        population's train window defines the configuration range (identical
        profile source as the retired stage-aware serving used).
        """
        if self._ood_profile_cache is None:
            from dltime.modeling import add_chronological_rank, chronological_holdout_split

            frame = self._stage_frame_for_profile()
            config_features = (
                [
                    f
                    for f in frame.columns
                    if f.startswith("def_")
                    and f != "def_match"
                    and pd.api.types.is_numeric_dtype(frame.get(f))
                ]
                if frame is not None
                else []
            )
            if config_features:
                ordered = add_chronological_rank(frame)
                train, _test, _info = chronological_holdout_split(ordered)
                self._ood_profile_cache = training_range_profile(
                    ordered.loc[train.index], config_features
                )
            else:
                ordered = add_chronological_rank(
                    self.tf.frame[self.tf.frame["is_training_eligible"].fillna(False)]
                )
                train, _test, _info = chronological_holdout_split(ordered)
                meta_direct = self.artifact["metadata"]
                config_features = [
                    f for f in meta_direct["numeric_features"] if f.startswith("def_")
                ]
                X_train = self.tf.X.loc[train.index]
                self._ood_profile_cache = training_range_profile(X_train, config_features)
        return self._ood_profile_cache

    def _stage_frame_for_profile(self) -> pd.DataFrame | None:
        """Stage-frame population for the OOD profile (cached helper)."""
        try:
            from dltime.stage_models import build_stage_frame
            from dltime.sources import load_all_sources

            frame, _feats, _diag = build_stage_frame(
                stage="staging", bundle=load_all_sources()
            )
            return frame[
                frame["LDR_Status"].isin(
                    {
                        "SUCCEED",
                        "COMPLETED_WITH_WARNING",
                        "COMPLETED_WITH_ERROR",
                        "COMPLETED_WITH_ERROR_AND_WARNING",
                        "ERROR",
                    }
                )
                & frame["Loader_Name"].notna()
                & frame["Sprint"].notna()
            ]
        except Exception:  # pragma: no cover - profile is best-effort
            return None

    def _modelable_stages(self) -> set[str]:
        """Dataset-wide MODELABLE stage names (built once; Task 06 classifier).

        AD-31: the foundation frame already carries the stage-target columns
        (single source of truth, pipeline.py Part F).
        """
        if self._modelable_stages_cache is None:
            from dltime.stages import build_stage_analytics

            analytics = build_stage_analytics(self.foundation.frames["executions_eligible"])
            self._modelable_stages_cache = {
                row["stage"] for row in analytics.to_payload()["stages"]
                if row.get("modelability") == "MODELABLE"
            }
        return self._modelable_stages_cache

    def _risk_section(
        self,
        *,
        predicted_seconds: float,
        uncertainty: dict[str, Any] | None,
        history: dict[str, Any],
        X: pd.DataFrame,
        stage_intelligence: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Task 09 operational risk block (PREDICTED path only).

        2026-10-03: the reliability label was removed from the system. The only
        consumer of that label was the LOW_CONFIDENCE risk signal, which is now
        driven DIRECTLY by the calibrated interval width relative to the point
        ETA (width factor >= 0.472, the p90-anchored criterion). No label named
        reliability exists or is exposed anywhere.
        """
        unc = uncertainty or {}
        reliability_label = None
        width = unc.get("interval_width_seconds")
        if (
            width is not None
            and predicted_seconds
            and float(predicted_seconds) > 0
            and (float(width) / float(predicted_seconds)) >= 0.472
        ):
            reliability_label = "LOW"  # internal criterion only — never surfaced as a label
        return assess_risk(
            predicted_seconds=predicted_seconds,
            reliability_label=reliability_label,
            history=history,
            X=X,
            ood_profile=self._ood_profile(),
            stage_intelligence=stage_intelligence,
            modelable_stages=self._modelable_stages(),
        ).to_payload()

    def stage_intelligence_from_context(
        self, ctx: pd.DataFrame, combined: pd.DataFrame | None = None
    ) -> dict[str, Any]:
        """Seven-stage view from an ALREADY-BUILT context row (AD-32 reuse path).

        Same engine as stage_intelligence() but without re-resolving the
        identity or rebuilding the context row — predict() and
        stage_intelligence() now share ONE context computation per request.
        Also stashes the stage-history-enriched context row so the stage-aware
        feature builder reads the EXACT row the view was computed from.
        """
        from dltime.stage_intel import ML_STAGES, build_stage_intel_for_context, summarize_stage_intel

        execution_id = ctx["LDR_Execution_Id"].iloc[0]
        combined_frame = combined if combined is not None else self._combined_frame
        if combined_frame is None:
            return {"available": False, "reason": "context frame unavailable", "stages": []}
        combined_with_hist = _attach_context_stage_history(
            combined_frame, execution_id, ML_STAGES, stage_specific=True
        )
        ctx_hist = combined_with_hist[
            combined_with_hist["LDR_Execution_Id"] == execution_id
        ]
        if len(ctx_hist) == 1:
            self._context_with_stage_history = ctx_hist
            # Evidence-sentence explainability (evidence-v1): stash the SAME
            # per-stage strictly-prior loader history the models themselves
            # consumed, so every sentence quotes real, verifiable numbers.
            def _ctx_num(col: str) -> float | None:
                if col not in ctx_hist.columns:
                    return None
                v = ctx_hist.iloc[0][col]
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    return None
                return f if f == f else None  # NaN -> None

            self._last_ctx_stage_history = {
                stage: {
                    "prior_count": _ctx_num(f"stage_hist_{stage}_loader_prior_count"),
                    "prior_median": _ctx_num(f"stage_hist_{stage}_loader_prior_median_sec"),
                }
                for stage in ML_STAGES
            }
        intel = build_stage_intel_for_context(
            combined_with_hist, execution_id, stage_models=self._stage_models()
        )
        summary = summarize_stage_intel(intel.per_stage)
        # Full (unfiltered) stage payload — internal consumers only (the
        # retired stage-aware reference feature builder needs every stage
        # column it was trained with). Never surfaced in the response.
        self._stage_intel_unfiltered = {
            "stages": [dict(v) for v in intel.per_stage.values()]
        }
        # User final spec (2026-09-30): only APPLICABLE stages are surfaced —
        # the stage set this execution will actually run, determined from the
        # user's own pre-execution entries (prevalidation/transformation only
        # when the user supplied them; staging, pre-load, data loading and the
        # approvals are always part of the execution).
        from dltime.stage_intel import (
            determine_applicable_stages,
            filter_stage_intel_to_applicable,
        )

        intel = filter_stage_intel_to_applicable(
            {"stages": [dict(v) for v in intel.per_stage.values()]},
            determine_applicable_stages(getattr(self, "_last_config_overrides", None)),
        )
        _n_applicable = len(intel.get("applicable_stages", []))
        return {
            "available": True,
            "note": (
                f"Applicable-Stage Execution View ({_n_applicable} stages for this "
                "execution): ML stage predictions + contextual statistical "
                "approval estimates. The total is the EXACT sum of the applicable "
                "stage predictions plus the approval estimates when available. "
                "ML stage outputs are model predictions, never actuals."
            ),
            "ordering": "Start_Time, tie-break LDR_Execution_Id (deterministic)",
            "prior_filter": "training_eligible",
            "policy_version": summary.get("policy_version"),
            "availability_summary": summary,
            "stages": intel.get("stages", []),
            "applicable_stages": intel.get("applicable_stages"),
            "not_applicable_stages": intel.get("not_applicable_stages"),
        }

    @staticmethod
    def _num(value: Any) -> float | None:
        if value is None:
            return None
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return f if np.isfinite(f) else None
