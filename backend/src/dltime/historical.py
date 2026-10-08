"""Chronological historical features (Task 02, AD-11).

For a current execution E, ONLY executions strictly earlier than E may
contribute. Ordering is deterministic: Start_Time, then LDR_Execution_Id as
tie-break (documented; no random ordering). The current execution itself and
all future executions can never enter its historical aggregates.

Groups (implemented where identity support exists):
    GLOBAL      prior count/mean/median/std over ALL prior executions
    LOADER      prior count/mean/median/std for the same Loader_Name
    CONNECTION  prior count/mean/median/std for same (Loader_Name, connection)
    DATASET     prior count/mean/median/std for same (Loader_Name, dataset)
    COMPOSITE   prior count/mean/median/std for same (dataset, sprint, loader)
    PREVIOUS    previous loader duration, time since previous loader execution

No prior history -> explicit <NA> (never a fabricated value).
Only training-eligible prior executions contribute (documented policy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class ChronologyError(ValueError):
    """Raised when chronological constraints cannot be satisfied."""


HIST_FEATURE_NAMES: tuple[str, ...] = (
    # GLOBAL
    "hist_global_prior_count",
    "hist_global_prior_mean_sec",
    "hist_global_prior_median_sec",
    "hist_global_prior_std_sec",
    # LOADER
    "hist_loader_prior_count",
    "hist_loader_prior_mean_sec",
    "hist_loader_prior_median_sec",
    "hist_loader_prior_std_sec",
    # CONNECTION
    "hist_loader_conn_prior_count",
    "hist_loader_conn_prior_mean_sec",
    "hist_loader_conn_prior_median_sec",
    "hist_loader_conn_prior_std_sec",
    # DATASET
    "hist_loader_dataset_prior_count",
    "hist_loader_dataset_prior_mean_sec",
    "hist_loader_dataset_prior_median_sec",
    "hist_loader_dataset_prior_std_sec",
    # COMPOSITE (canonical identity minus connection)
    "hist_composite_prior_count",
    "hist_composite_prior_mean_sec",
    "hist_composite_prior_median_sec",
    "hist_composite_prior_std_sec",
    # PREVIOUS
    "hist_prev_loader_duration_sec",
    "hist_time_since_prev_loader_sec",
)

HISTORICAL_ONLY = frozenset(HIST_FEATURE_NAMES)


def _sorted_chronologically(executions: pd.DataFrame) -> pd.DataFrame:
    """Deterministic chronological order: Start_Time, then LDR_Execution_Id."""
    df = executions.copy()
    if "Start_Time" not in df.columns:
        raise ChronologyError("Start_Time is required for chronological ordering.")
    df["_start_ts"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    if df["_start_ts"].isna().any():
        raise ChronologyError(
            "Every execution needs a Start_Time for chronological feature construction."
        )
    return df.sort_values(["_start_ts", "LDR_Execution_Id"], kind="mergesort").reset_index(drop=True)


def _prior_stats(values: pd.Series) -> tuple[int, float, float, float]:
    """(count, mean, median, std) of prior target values; NaN stats when empty."""
    n = int(len(values))
    if n == 0:
        return 0, np.nan, np.nan, np.nan
    mean = float(values.mean())
    median = float(values.median())
    std = float(values.std(ddof=1)) if n >= 2 else np.nan
    return n, mean, median, std


@dataclass
class HistoryFeatureResult:
    frame: pd.DataFrame
    diagnostics: dict = field(default_factory=dict)


def build_historical_features(
    executions: pd.DataFrame,
    *,
    prior_filter: str = "training_eligible",
    connection_col: str = "execution_connection",
) -> HistoryFeatureResult:
    """Compute prior-only historical features for every execution.

    Args:
        executions: execution-level frame (reconstructed; one row per
            LDR_Execution_Id) with Start_Time, Loader_Name, Sprint,
            Dataset_display_name, timetaken_in_sec and eligibility columns.
        prior_filter: which prior executions may contribute history:
            "training_eligible" (default; only trustworthy evidence) or "all".
        connection_col: execution-level resolved connection column.

    Returns:
        HistoryFeatureResult with the original frame plus HIST_FEATURE_NAMES
        columns and diagnostics about coverage.
    """
    if "LDR_Execution_Id" not in executions.columns:
        raise ChronologyError("LDR_Execution_Id is required.")
    if "timetaken_in_sec" not in executions.columns:
        raise ChronologyError("timetaken_in_sec is required to build history.")
    if prior_filter not in ("training_eligible", "all"):
        raise ChronologyError(f"Unknown prior_filter '{prior_filter}'")

    df = _sorted_chronologically(executions)
    target = pd.to_numeric(df["timetaken_in_sec"], errors="coerce")

    if prior_filter == "training_eligible":
        if "is_training_eligible" not in df.columns:
            raise ChronologyError(
                "prior_filter='training_eligible' requires is_training_eligible; "
                "run the eligibility layer first or pass prior_filter='all'."
            )
        usable = target.notna() & df["is_training_eligible"].fillna(False).astype(bool)
    else:
        usable = target.notna()

    loader = df["Loader_Name"].astype("string")
    sprint = df["Sprint"].astype("string")
    dataset = df["Dataset_display_name"].astype("string")
    conn = df[connection_col].astype("string") if connection_col in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")

    out = pd.DataFrame(index=df.index)
    rows: list[dict[str, object]] = []
    covered = {g: 0 for g in ("GLOBAL", "LOADER", "CONNECTION", "DATASET", "COMPOSITE")}
    prev_available = 0
    ts_prev_valid = 0

    # Streaming prefix computation over the deterministic chronological order.
    # Each execution sees ONLY the executions before it in this order.
    seen_global: list[float] = []
    seen_by_loader: dict[str, list[float]] = {}
    seen_by_loader_conn: dict[tuple[str, str], list[float]] = {}
    seen_by_loader_dataset: dict[tuple[str, str], list[float]] = {}
    seen_by_composite: dict[tuple[str, str, str], list[float]] = {}
    last_loader_time: dict[str, pd.Timestamp] = {}

    for i in range(len(df)):
        L = loader.iat[i]
        S = sprint.iat[i]
        D = dataset.iat[i]
        C = conn.iat[i]
        t = target.iat[i]
        start = df["_start_ts"].iat[i]

        row: dict[str, object] = {}

        # ---- GLOBAL ----
        g_n, g_mean, g_med, g_std = _prior_stats(pd.Series(seen_global, dtype="float64"))
        row["hist_global_prior_count"] = g_n
        row["hist_global_prior_mean_sec"] = g_mean
        row["hist_global_prior_median_sec"] = g_med
        row["hist_global_prior_std_sec"] = g_std

        # ---- LOADER ----
        key_l = str(L) if pd.notna(L) else None
        lv = seen_by_loader.get(key_l, [])
        l_n, l_mean, l_med, l_std = _prior_stats(pd.Series(lv, dtype="float64"))
        row["hist_loader_prior_count"] = l_n
        row["hist_loader_prior_mean_sec"] = l_mean
        row["hist_loader_prior_median_sec"] = l_med
        row["hist_loader_prior_std_sec"] = l_std

        # ---- CONNECTION (Loader + resolved connection) ----
        key_lc = (key_l, str(C)) if key_l is not None and pd.notna(C) else None
        cv = seen_by_loader_conn.get(key_lc, []) if key_lc else []
        c_n, c_mean, c_med, c_std = _prior_stats(pd.Series(cv, dtype="float64"))
        row["hist_loader_conn_prior_count"] = c_n
        row["hist_loader_conn_prior_mean_sec"] = c_mean
        row["hist_loader_conn_prior_median_sec"] = c_med
        row["hist_loader_conn_prior_std_sec"] = c_std

        # ---- DATASET (Loader + dataset) ----
        key_ld = (key_l, str(D)) if key_l is not None and pd.notna(D) else None
        dv = seen_by_loader_dataset.get(key_ld, []) if key_ld else []
        d_n, d_mean, d_med, d_std = _prior_stats(pd.Series(dv, dtype="float64"))
        row["hist_loader_dataset_prior_count"] = d_n
        row["hist_loader_dataset_prior_mean_sec"] = d_mean
        row["hist_loader_dataset_prior_median_sec"] = d_med
        row["hist_loader_dataset_prior_std_sec"] = d_std

        # ---- COMPOSITE (dataset, sprint, loader) ----
        key_comp = (
            (str(D), str(S), str(L))
            if all(pd.notna(x) for x in (D, S, L))
            else None
        )
        compv = seen_by_composite.get(key_comp, []) if key_comp else []
        comp_n, comp_mean, comp_med, comp_std = _prior_stats(pd.Series(compv, dtype="float64"))
        row["hist_composite_prior_count"] = comp_n
        row["hist_composite_prior_mean_sec"] = comp_mean
        row["hist_composite_prior_median_sec"] = comp_med
        row["hist_composite_prior_std_sec"] = comp_std

        # ---- PREVIOUS ----
        prev_ts = last_loader_time.get(key_l) if key_l else None
        if lv:
            row["hist_prev_loader_duration_sec"] = float(lv[-1])
            prev_available += 1
        else:
            row["hist_prev_loader_duration_sec"] = np.nan
        if prev_ts is not None:
            delta = (start - prev_ts).total_seconds()
            row["hist_time_since_prev_loader_sec"] = max(float(delta), 0.0)  # non-negative
            ts_prev_valid += 1
        else:
            row["hist_time_since_prev_loader_sec"] = np.nan

        # coverage diagnostics (per group: at least one prior observation)
        if g_n > 0:
            covered["GLOBAL"] += 1
        if l_n > 0:
            covered["LOADER"] += 1
        if c_n > 0:
            covered["CONNECTION"] += 1
        if d_n > 0:
            covered["DATASET"] += 1
        if comp_n > 0:
            covered["COMPOSITE"] += 1

        rows.append(row)

        # ---- update "seen" state AFTER processing row i (strictly-prior rule) ----
        if bool(usable.iat[i]):
            seen_global.append(float(t))
            if key_l is not None:
                seen_by_loader.setdefault(key_l, []).append(float(t))
                if pd.notna(C):
                    seen_by_loader_conn.setdefault((key_l, str(C)), []).append(float(t))
                if pd.notna(D):
                    seen_by_loader_dataset.setdefault((key_l, str(D)), []).append(float(t))
            if key_comp is not None:
                seen_by_composite.setdefault(key_comp, []).append(float(t))
            last_loader_time[key_l] = start

    hist = pd.DataFrame(rows, index=df.index)
    result = pd.concat([df.drop(columns=["_start_ts"]), hist], axis=1)

    diagnostics = {
        "prior_filter": prior_filter,
        "usable_prior_executions": int(usable.sum()),
        "coverage": covered,
        "executions_with_prev_loader_duration": prev_available,
        "executions_with_valid_time_since_prev": ts_prev_valid,
        "first_execution_has_no_history": bool(rows[0]["hist_global_prior_count"] == 0),
    }
    return HistoryFeatureResult(frame=result, diagnostics=diagnostics)
