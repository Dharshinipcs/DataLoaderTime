"""Loader Intelligence + adaptive history evidence + cold-start context (Task 10, AD-23).

A read-only INTELLIGENCE layer answering "what do we know about this loader
and its historical behavior?" — never a second prediction path. The ETA gate
remains exactly AD-16 (min 3 prior eligible executions, configurable); a
loader below the gate still receives NO authoritative ETA from this module
or anywhere else.

History-evidence levels (thresholds from EXP-019 investigation, provisional
and documented; they classify EVIDENCE, they do not move the ETA gate):
    NO_HISTORY   0 eligible executions      (194 of 229 definition loaders)
    EMERGING     1-2 eligible executions    (45 loaders) — counts and recency
                 only; NO "typical duration" (a 1-2-sample statistic would
                 mislead)
    SUFFICIENT   3-9 eligible executions    (32 loaders) — median/mean/CV
                 shown; matches the AD-16 ETA-eligible population
    STRONG       >=10 eligible executions   (11 loaders) — full statistics
                 (10+ is the top bin used consistently since EXP-012)

Cold-start contextual intelligence: for an insufficient-history loader, the
ONLY contextual evidence shown is same-DATASET history from OTHER loaders,
and only when it genuinely exists (>=3 eligible observations). EXP-019 shows
this context is usually absent (median 0 rows; 11% of weak loaders have >=3;
7/177 never-executed loaders' datasets overlap executed ones), so the response
explicitly reports "no useful contextual evidence" rather than inventing any.
Contextual figures are labeled CONTEXTUAL everywhere and are NEVER an ETA.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

LOADER_INTEL_VERSION = "loader-intel-v1"

# Evidence-level thresholds (EXP-019; provisional, documented).
EMERGING_MIN = 1      # any eligible execution
SUFFICIENT_MIN = 3    # == AD-16 ETA gate (kept aligned by design, not coincidence)
STRONG_MIN = 10       # top history bin used in EXP-012/EXP-016
TREND_MIN = 6         # stability evidence: 19 loaders; median old/new ratio 1.04

MIN_TYPICAL_SAMPLE = 3  # below this, "typical duration" is withheld (honesty rule)


class EvidenceLevel:
    NO_HISTORY = "NO_HISTORY"
    EMERGING = "EMERGING"
    SUFFICIENT = "SUFFICIENT"
    STRONG = "STRONG"


def evidence_level_for(n: int) -> str:
    if n <= 0:
        return EvidenceLevel.NO_HISTORY
    if n < SUFFICIENT_MIN:
        return EvidenceLevel.EMERGING
    if n < STRONG_MIN:
        return EvidenceLevel.SUFFICIENT
    return EvidenceLevel.STRONG


def _round(v: float | None, nd: int = 3) -> float | None:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return round(float(v), nd)


def _cv_label(cv: float | None) -> str:
    """Presentation label for variability (documented display thresholds:
    Low <= 0.25, Moderate <= 0.75, High above; classification of evidence,
    not a model output)."""
    if cv is None:
        return "—"
    if cv <= 0.25:
        return "Low"
    if cv <= 0.75:
        return "Moderate"
    return "High"


def _stage_rows(pos: pd.DataFrame) -> dict[str, Any]:
    """Per-stage historical intelligence from the reconstructed stage columns
    (pipeline.py Part F; the same source the stage models train on).
    Aggregates only — never a prediction."""
    display = {
        "staging": "Staging",
        "prevalidation": "Prevalidation",
        "transformation": "Transformation",
        "pre_load": "Pre-load",
        "data_loading": "Data Loading",
    }
    rows: list[dict[str, Any]] = []
    n_total = int(len(pos))
    for stage, label in display.items():
        col = f"stage_{stage}_sec"
        if col not in pos.columns:
            rows.append(
                {
                    "stage": stage,
                    "display": label,
                    "available": False,
                    "reason": "stage duration is not recorded for this data",
                }
            )
            continue
        s = pd.to_numeric(pos[col], errors="coerce").dropna()
        s = s[s > 0]
        if s.empty:
            rows.append(
                {
                    "stage": stage,
                    "display": label,
                    "available": False,
                    "reason": "no positive stage durations on record",
                }
            )
            continue
        cv = float(s.std(ddof=1) / s.mean()) if len(s) >= 2 and s.mean() > 0 else None
        rows.append(
            {
                "stage": stage,
                "display": label,
                "available": True,
                "observations": int(len(s)),
                "coverage_pct": _round(100.0 * len(s) / n_total, 1) if n_total else None,
                "median_sec": _round(float(s.median())),
                "typical_sec": _round(float(s.median())),
                "p75_sec": _round(float(s.quantile(0.75))),
                "cv": _round(cv),
                "variability": _cv_label(cv),
            }
        )
    return {"rows": rows}


def _behavior_section(eligible: pd.DataFrame) -> dict[str, Any]:
    """Execution-time behavior: chronological series, distribution summary,
    and recent-vs-overall median. Aggregates of recorded actuals only.
    Duration source: AD-34 — total_timetaken_in_min x 60 is authoritative
    (matches the Start/End span on 757/757 rows); the legacy seconds field
    is a fallback where minutes are missing."""
    if "total_timetaken_in_min" in eligible.columns:
        mins = pd.to_numeric(eligible["total_timetaken_in_min"], errors="coerce") * 60.0
        legacy = pd.to_numeric(eligible.get("timetaken_in_sec"), errors="coerce")
        dur = mins.fillna(legacy)
    else:
        dur = pd.to_numeric(eligible.get("timetaken_in_sec"), errors="coerce")
    mask = dur.notna() & (dur > 0)
    pos = eligible[mask].assign(_dur=dur[mask]).sort_values(
        ["Start_Time", "LDR_Execution_Id"], kind="mergesort"
    )
    n = int(len(pos))
    sec: dict[str, Any] = {"n": n}
    if n == 0:
        return sec
    s = pos["_dur"].astype(float)
    sec["series"] = [
        {
            "start_time": str(t),
            "seconds": _round(float(v)),
            "status": str(st) if pd.notna(st) else None,
        }
        for t, v, st in zip(
            pos["Start_Time"], pos["_dur"], pos["LDR_Status"], strict=False
        )
    ]
    if n >= 2:
        bins_edge = [0, 5, 15, 30, 60, 120, 300, float("inf")]
        labels = ["<5s", "5–15s", "15–30s", "30–60s", "1–2m", "2–5m", "5m+"]
        counts = []
        for lo, hi in zip(bins_edge[:-1], bins_edge[1:], strict=False):
            counts.append(int(((s > lo) & (s <= hi)).sum()))
        sec["distribution"] = {
            "bins": [{"label": l, "count": c} for l, c in zip(labels, counts, strict=False)],
            "min_sec": _round(float(s.min())),
            "max_sec": _round(float(s.max())),
        }
    if n >= MIN_TYPICAL_SAMPLE:
        cv = float(s.std(ddof=1) / s.mean()) if s.mean() > 0 else None
        recent = s.iloc[max(0, n - 5):]
        sec["distribution"].update(
            {
                "median_sec": _round(float(s.median())),
                "p25_sec": _round(float(s.quantile(0.25))),
                "p75_sec": _round(float(s.quantile(0.75))),
                "variability": _cv_label(cv),
                "cv": _round(cv),
            }
        )
        sec["recent_median_sec"] = _round(float(recent.median()))
        sec["median_sec"] = _round(float(s.median()))
    return sec


def _configuration_section(
    definitions_detail: pd.DataFrame,
    definitions_summary: pd.DataFrame | None,
    loader_name: str,
) -> dict[str, Any]:
    """The registered configuration for this loader, read directly from the
    definition sources (the same sources the model's configuration features
    are built from). Read-only presentation — never user input."""
    key = pd.Series([loader_name], dtype="string")[0]
    det = definitions_detail[
        definitions_detail["loader_display_name"].astype("string") == key
    ]
    summ = (
        definitions_summary[
            definitions_summary["loader_display_name"].astype("string") == key
        ]
        if definitions_summary is not None
        else det.iloc[0:0]
    )
    if det.empty and summ.empty:
        return {
            "available": False,
            "reason": "no registered definition rows match this loader",
        }

    names = (
        sorted(set(det["preval_trans_display_name"].dropna().astype(str)))[:3]
        if "preval_trans_display_name" in det.columns
        else []
    )

    def _top(df: pd.DataFrame, col: str) -> str | None:
        if col not in df.columns:
            return None
        vals = df[col].dropna()
        if vals.empty:
            return None
        vals = vals.astype(str)
        return str(vals.value_counts().index[0])

    def _max(df: pd.DataFrame, col: str) -> str | None:
        if col not in df.columns:
            return None
        nums = pd.to_numeric(df[col], errors="coerce").dropna()
        if nums.empty:
            return None
        f = float(nums.max())
        return str(int(f)) if f == int(f) else f"{f:g}"

    rows = [
        {
            "label": "Prevalidation / Transformation",
            "value": ", ".join(names) if names else "—",
        },
        {"label": "Type", "value": _top(det, "type") or "—"},
        {"label": "Calling Type", "value": _top(det, "calling_type") or "—"},
        {"label": "Execution Order", "value": _max(det, "execution_order") or "—"},
        {
            "label": "Prevalidation Count",
            "value": _max(summ, "prevalidation_count") or _max(det, "prevalidation_count") or "—",
        },
        {
            "label": "Transformation Count",
            "value": _max(summ, "transformation_count") or _max(det, "transformation_count") or "—",
        },
    ]
    return {
        "available": True,
        "rows": rows,
        "basis": "registered definition rows for this loader (latest state)",
    }


def build_loader_intel(
    *,
    loader_name: str,
    frame: pd.DataFrame,
    definitions_detail: pd.DataFrame,
    definitions_summary: pd.DataFrame | None = None,
    min_eta_history: int = 3,
) -> dict[str, Any]:
    """Assemble everything known about one loader from real data (read-only).

    `frame` is the reconstructed execution-level frame (543 rows) with
    is_training_eligible, execution_connection, execution_resolution_status,
    Dataset_display_name, Sprint, Start_Time, timetaken_in_sec.
    """
    tgt = "timetaken_in_sec"
    known_in_definitions = bool(
        (definitions_detail["loader_display_name"].astype("string") == loader_name).any()
    )
    exec_rows = frame[frame["Loader_Name"].astype("string") == pd.Series([loader_name], dtype="string")[0]]
    eligible = exec_rows[exec_rows["is_training_eligible"].fillna(False)]
    pos = eligible[eligible[tgt].notna() & (eligible[tgt] > 0)].sort_values(
        ["Start_Time", "LDR_Execution_Id"], kind="mergesort"
    )
    n = int(len(pos))

    level = evidence_level_for(n)
    eta_eligible = n >= min_eta_history

    payload: dict[str, Any] = {
        "loader_name": loader_name,
        "loader_intel_version": LOADER_INTEL_VERSION,
        "known": known_in_definitions or n > 0,
        "known_in_definitions": known_in_definitions,
        "evidence_level": level,
        "eta_eligible": eta_eligible,
        "eta_gate": {
            "min_prior_executions": min_eta_history,
            "source": "AD-16 (approved configurable default; EXP-012 evidence)",
            "satisfied": eta_eligible,
            "explanation": (
                f"{n} eligible prior execution(s) recorded; the authoritative ETA "
                f"gate requires at least {min_eta_history}."
                if not eta_eligible and n > 0
                else (
                    f"No eligible executions recorded; the authoritative ETA gate "
                    f"requires at least {min_eta_history}."
                    if n == 0
                    else f"{n} eligible executions satisfy the gate of {min_eta_history}."
                )
            ),
        },
        "history": _history_section(pos, tgt, n, level),
        "context": _context_section(pos, exec_rows),
        "definition_context": _definition_section(definitions_detail, loader_name),
        "cold_start": _cold_start_section(pos, frame, definitions_detail, loader_name, min_eta_history, tgt),
        "behavior": _behavior_section(eligible),
        "stages": _stage_rows(pos),
        "configuration": _configuration_section(
            definitions_detail, definitions_summary, loader_name
        ),
        "limitations": [],
    }

    if n == 1:
        payload["limitations"].append(
            "Only one eligible execution exists; no typical-duration or variability figure is shown."
        )
    if n == 2:
        payload["limitations"].append(
            "Only two eligible executions exist; no typical-duration or variability "
            "figure is shown (not statistically meaningful)."
        )
    if known_in_definitions is False and n == 0:
        payload["limitations"].append(
            "Loader is unknown to the registered definitions and has no execution history."
        )
    return payload


def _history_section(pos: pd.DataFrame, tgt: str, n: int, level: str) -> dict[str, Any]:
    sec: dict[str, Any] = {
        "eligible_execution_count": n,
        "first_execution": str(pos["Start_Time"].iloc[0]) if n else None,
        "last_execution": str(pos["Start_Time"].iloc[-1]) if n else None,
        "show_typical": n >= MIN_TYPICAL_SAMPLE,
    }
    if n >= MIN_TYPICAL_SAMPLE:
        s = pos[tgt].astype(float)
        cv = float(s.std(ddof=1) / s.mean()) if n >= 2 and s.mean() > 0 else None
        sec.update(
            {
                "median_sec": _round(s.median()),
                "mean_sec": _round(s.mean()),
                "cv": _round(cv),
                "p90_sec": _round(s.quantile(0.9)),
                "max_sec": _round(s.max()),
            }
        )
    if n >= TREND_MIN:
        h = n // 2
        old_m = float(pos[tgt].iloc[:h].median())
        new_m = float(pos[tgt].iloc[h:].median())
        if old_m > 0:
            ratio = new_m / old_m
            trend = (
                "stable"
                if 0.5 <= ratio <= 2.0
                else ("increasing" if ratio > 2.0 else "decreasing")
            )
            sec["trend"] = {
                "basis": "recent-half vs older-half median (documented EXP-019 basis; n>=6 only)",
                "ratio": _round(ratio),
                "label": trend,
            }
    return sec


def _context_section(pos: pd.DataFrame, exec_rows: pd.DataFrame) -> dict[str, Any]:
    datasets = sorted(set(pos["Dataset_display_name"].dropna().astype(str)))
    conns = sorted(set(exec_rows["execution_connection"].dropna().astype(str)))
    sprints = sorted(set(pos["Sprint"].dropna().astype(str)))
    statuses = exec_rows["execution_resolution_status"].value_counts().to_dict()
    return {
        "datasets": datasets,
        "connections": conns,
        "sprints": sprints,
        "connection_resolution": {str(k): int(v) for k, v in statuses.items()},
    }


def _definition_section(definitions_detail: pd.DataFrame, loader_name: str) -> dict[str, Any]:
    rows = definitions_detail[
        definitions_detail["loader_display_name"].astype("string") == pd.Series([loader_name], dtype="string")[0]
    ]
    if rows.empty:
        return {"matched": False}
    return {
        "matched": True,
        "definition_rows": int(len(rows)),
        "datasets": sorted(set(rows["datasetname"].dropna().astype(str))),
        "sprints": sorted(set(rows["sprint_name"].dropna().astype(str))),
        "connections": sorted(set(rows["ldr_connection_name"].dropna().astype(str))),
    }


def _cold_start_section(
    pos: pd.DataFrame,
    frame: pd.DataFrame,
    definitions_detail: pd.DataFrame,
    loader_name: str,
    min_eta_history: int,
    tgt: str,
) -> dict[str, Any]:
    """Contextual (NEVER predictive) evidence for an insufficient-history loader."""
    n = int(len(pos))
    if n >= min_eta_history:
        return {
            "applicable": False,
            "reason": "loader already satisfies the ETA history gate",
            "contextual_intelligence": None,
        }
    eligible = frame[frame["is_training_eligible"].fillna(False)]
    pos_all = eligible[eligible[tgt].notna() & (eligible[tgt] > 0)]

    own_datasets = set(pos["Dataset_display_name"].dropna().astype(str))
    if not own_datasets:
        drows = definitions_detail[
            definitions_detail["loader_display_name"].astype("string") == pd.Series([loader_name], dtype="string")[0]
        ]
        own_datasets = set(drows["datasetname"].dropna().astype(str))

    ctx = pos_all[
        pos_all["Dataset_display_name"].astype("string").isin(pd.Series(sorted(own_datasets), dtype="string"))
        & (pos_all["Loader_Name"].astype("string") != pd.Series([loader_name], dtype="string")[0])
    ] if own_datasets else pos_all.iloc[0:0]

    # Connection context (Prompt 2, EXP-023): resolved connections are a
    # stronger infrastructure signal than dataset membership — an execution
    # runs ON a connection. EXP-023: 58 of 184 gate-failing training rows have
    # >=3 same-connection other-loader eligible rows (25 distinct loaders).
    conn_col = "execution_connection"
    own_conns = set(exec_rows_for_loader(frame, loader_name)[conn_col].dropna().astype(str))
    if not own_conns:
        drows = definitions_detail[
            definitions_detail["loader_display_name"].astype("string") == pd.Series([loader_name], dtype="string")[0]
        ]
        own_conns = set(drows["ldr_connection_name"].dropna().astype(str))
    ctx_conn = pos_all[
        pos_all[conn_col].astype("string").isin(pd.Series(sorted(own_conns), dtype="string"))
        & (pos_all["Loader_Name"].astype("string") != pd.Series([loader_name], dtype="string")[0])
    ] if own_conns else pos_all.iloc[0:0]

    n_ctx = int(len(ctx))
    n_ctx_conn = int(len(ctx_conn))
    section: dict[str, Any] = {
        "applicable": True,
        "eta_produced": False,
        "statement": (
            "This loader does not meet the authoritative ETA history gate. No ETA is "
            "predicted. Any contextual evidence below is descriptive only and is NOT "
            "a prediction."
            if own_datasets or n_ctx > 0 or n_ctx_conn > 0
            else "This loader does not meet the authoritative ETA history gate, and no "
            "useful contextual evidence exists in the recorded data. No ETA is predicted."
        ),
    }
    contextual: dict[str, Any] | None = None
    if own_datasets and n_ctx >= MIN_TYPICAL_SAMPLE:
        s = ctx[tgt].astype(float)
        contextual = {
            "kind": "CONTEXTUAL_SAME_DATASET",
            "basis": (
                f"eligible executions on this loader's dataset(s) {sorted(own_datasets)} "
                f"by other loaders"
            ),
            "observation_count": n_ctx,
            "distinct_loaders": int(ctx["Loader_Name"].nunique()),
            "median_sec": _round(s.median()),
            "p90_sec": _round(s.quantile(0.9)),
            "disclaimer": (
                "Descriptive context about similar workloads — explicitly NOT a "
                "prediction for this loader and NEVER used as its ETA."
            ),
        }
    if own_conns and n_ctx_conn >= MIN_TYPICAL_SAMPLE:
        s = ctx_conn[tgt].astype(float)
        conn_ctx = {
            "kind": "CONTEXTUAL_SAME_CONNECTION",
            "basis": (
                f"eligible executions on this loader's resolved connection(s) "
                f"{sorted(own_conns)} by other loaders"
            ),
            "observation_count": n_ctx_conn,
            "distinct_loaders": int(ctx_conn["Loader_Name"].nunique()),
            "median_sec": _round(s.median()),
            "p90_sec": _round(s.quantile(0.9)),
            "disclaimer": (
                "Descriptive context about the same database/infrastructure — "
                "explicitly NOT a prediction for this loader and NEVER used as its ETA."
            ),
        }
        # Prefer the most specific context when both exist (dataset is workload-
        # specific; connection is infrastructure). Report dataset first, keep
        # connection as an additional block for transparency.
        if contextual is None:
            contextual = conn_ctx
        else:
            section["contextual_intelligence_connection"] = conn_ctx
    section["contextual_intelligence"] = contextual
    if contextual is None:
        section["context_note"] = (
            f"Same-dataset context from other loaders: {n_ctx} observation(s); "
            f"same-connection context: {n_ctx_conn} observation(s) — below the "
            "3-observation honesty threshold, so no contextual statistic is shown."
        )
    return section


def exec_rows_for_loader(frame: pd.DataFrame, loader_name: str) -> pd.DataFrame:
    """All execution rows (any status) for one loader name."""
    return frame[frame["Loader_Name"].astype("string") == pd.Series([loader_name], dtype="string")[0]]
