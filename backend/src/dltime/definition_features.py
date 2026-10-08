"""Pre-execution configuration features (Task 02, AD-13).

Derived exclusively from the pre-execution definition sources (Detail +
Summary), matched to executions by canonical identity. Multi-dataset
executions are handled explicitly: configuration features aggregate over ALL
datasets in the execution's `datasets` set (a multi-dataset workflow
execution legitimately loads through several dataset configurations), plus a
representative-dataset join variant.

No outcome information is used. Unmatched executions get <NA> (never
fabricated). A `def_match` diagnostic records whether any definition
evidence exists for the execution identity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DefinitionFeatureError(ValueError):
    """Raised when definition sources cannot support configuration features."""


FEATURE_NAMES: tuple[str, ...] = (
    "def_prevalidation_count",
    "def_transformation_count",
    "def_execution_order_max",
    "def_execution_order_mean",
    "def_execution_order_std",
    "def_component_count",
    "def_type_nunique",
    "def_dominant_calling_type",
    # 2026-10-04 (user spec): execution_order is PER SECTION — prevalidation
    # rows are ordered 1..n among themselves and transformation rows 1..m
    # among themselves (two independent sequences). The pooled stats above are
    # kept for continuity; the per-section stats below carry the real signal.
    "def_preval_order_max",
    "def_preval_order_mean",
    "def_preval_order_std",
    "def_trans_order_max",
    "def_trans_order_mean",
    "def_trans_order_std",
    "def_preval_type_nunique",
    "def_trans_type_nunique",
    "def_type_crossreference_count",
    "def_type_dataobject_count",
    "def_type_dataextract_count",
    "def_type_dataset_count",
    "def_type_query_count",
    "def_type_table_count",
)

CATEGORICAL_FEATURES = frozenset({"def_dominant_calling_type"})

_CANONICAL_TYPES = ("crossreference", "dataobject", "dataextract", "dataset", "query", "table")


def _type_norm(value) -> str | None:
    s = _norm(value)
    return s.strip().lower() if s else None


def _section_order_stats(order: pd.Series) -> dict[str, float]:
    """max/mean/std(ddof=1) of one section's execution_order sequence."""
    out: dict[str, float] = {}
    if not order.empty:
        out["max"] = float(order.max())
        out["mean"] = float(order.mean())
        out["std"] = float(order.std(ddof=1)) if len(order) >= 2 else np.nan
    return out


def _norm(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    return s if s else None


def _dominant(values: pd.Series):
    """Most frequent calling_type; None when values missing or tied."""
    vals = values.dropna()
    if vals.empty:
        return None
    counts = vals.value_counts()
    top = counts[counts == counts.max()].index
    if len(top) != 1:
        return None  # tie -> no silent choice
    return str(top[0])


def _mean_of_max(values: pd.Series) -> float:
    """Mean over distinct datasets of (max prevalidation/transformation count
    within that dataset). Aggregates the summary source across an execution's
    dataset set without double counting rows."""
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return np.nan
    return float(vals.max())  # identity-level max (summary rows are per dataset)


def build_identity_level_definition_features(
    definitions_detail: pd.DataFrame,
    definitions_summary: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (dataset, sprint, loader) identity key with config features."""
    required = {"datasetname", "sprint_name", "loader_display_name"}
    for name, table in (("detail", definitions_detail), ("summary", definitions_summary)):
        missing = required - set(table.columns)
        if missing:
            raise DefinitionFeatureError(f"Definition {name} missing columns: {sorted(missing)}")

    det = definitions_detail.copy()
    det["_d"] = det["datasetname"].map(_norm)
    det["_s"] = det["sprint_name"].map(_norm)
    det["_l"] = det["loader_display_name"].map(_norm)
    det = det.dropna(subset=["_d", "_s", "_l"])

    summ = definitions_summary.copy()
    summ["_d"] = summ["datasetname"].map(_norm)
    summ["_s"] = summ["sprint_name"].map(_norm)
    summ["_l"] = summ["loader_display_name"].map(_norm)
    summ = summ.dropna(subset=["_d", "_s", "_l"])

    if not summ.empty:
        summary_agg = (
            summ.groupby(["_d", "_s", "_l"])
            .agg(
                def_prevalidation_count=("prevalidation_count", "max"),
                def_transformation_count=("transformation_count", "max"),
            )
            .reset_index()
        )
    else:
        summary_agg = pd.DataFrame(
            columns=["_d", "_s", "_l", "def_prevalidation_count", "def_transformation_count"]
        )

    if not det.empty:
        det["_order"] = pd.to_numeric(det["execution_order"], errors="coerce")
        det["_type"] = det["type"].map(_type_norm)
        det["_calling"] = det["calling_type"].map(lambda v: (_norm(v) or "").upper())

        def _detail_agg(g: pd.DataFrame) -> pd.Series:
            prev = g[g["_calling"] == "PREVALIDATION"]
            trans = g[g["_calling"] == "TRANSFORMATION"]
            row = {
                "def_execution_order_max": g["_order"].max(),
                "def_execution_order_mean": g["_order"].mean(),
                "def_execution_order_std": (
                    g["_order"].std(ddof=1) if len(g) >= 2 else np.nan
                ),
                "def_component_count": float(len(g)),
                "def_type_nunique": g["_type"].dropna().nunique(),
                "def_dominant_calling_type": _dominant(g["calling_type"]),
            }
            for prefix, sec in (("preval", prev), ("trans", trans)):
                stats = _section_order_stats(sec["_order"].dropna())
                row[f"def_{prefix}_order_max"] = stats.get("max", np.nan)
                row[f"def_{prefix}_order_mean"] = stats.get("mean", np.nan)
                row[f"def_{prefix}_order_std"] = stats.get("std", np.nan)
                row[f"def_{prefix}_type_nunique"] = sec["_type"].dropna().nunique()
            for t in _CANONICAL_TYPES:
                row[f"def_type_{t}_count"] = float((g["_type"] == t).sum())
            return pd.Series(row)

        detail_agg = (
            det.groupby(["_d", "_s", "_l"])
            .apply(_detail_agg, include_groups=False)
            .reset_index()
        )
    else:
        detail_agg = pd.DataFrame(
            columns=[
                "_d", "_s", "_l", *FEATURE_NAMES,
            ]
        )

    merged = summary_agg.merge(detail_agg, on=["_d", "_s", "_l"], how="outer")
    for col in FEATURE_NAMES:
        if col not in merged.columns:
            merged[col] = np.nan
    return merged[["_d", "_s", "_l", *FEATURE_NAMES]]


def _split_datasets(datasets_value) -> list[str]:
    if datasets_value is None or (isinstance(datasets_value, float) and pd.isna(datasets_value)):
        return []
    s = str(datasets_value).strip()
    if not s:
        return []
    return [p.strip() for p in s.split(";") if p.strip()]


def build_execution_level_definition_features(
    executions: pd.DataFrame,
    definitions_detail: pd.DataFrame,
    definitions_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Configuration features at EXECUTION level.

    For each execution:
      - identity-variant features from the representative dataset triple
        (Dataset_display_name, Sprint, Loader_Name);
      - set-variant features aggregating over ALL datasets of the execution
        (counts summed across distinct datasets, order stats pooled, types
        unioned, calling_type majority vote across datasets).
    Adds `def_match` (any definition evidence found).
    """
    identity_feats = build_identity_level_definition_features(
        definitions_detail, definitions_summary
    )

    det = definitions_detail.copy()
    det["_d"] = det["datasetname"].map(_norm)
    det["_s"] = det["sprint_name"].map(_norm)
    det["_l"] = det["loader_display_name"].map(_norm)
    det["_order"] = pd.to_numeric(det["execution_order"], errors="coerce")
    det = det.dropna(subset=["_s", "_l"])

    summ = definitions_summary.copy()
    summ["_d"] = summ["datasetname"].map(_norm)
    summ["_s"] = summ["sprint_name"].map(_norm)
    summ["_l"] = summ["loader_display_name"].map(_norm)
    summ = summ.dropna(subset=["_s", "_l"])

    # index pools by (sprint, loader) for set-variant aggregation
    det_by_sl = {k: g for k, g in det.groupby(["_s", "_l"])}
    summ_by_sl = {k: g for k, g in summ.groupby(["_s", "_l"])}

    rows: list[dict] = []
    for r in executions.itertuples(index=False):
        sprint = _norm(getattr(r, "Sprint"))
        loader = _norm(getattr(r, "Loader_Name"))
        repr_ds = _norm(getattr(r, "Dataset_display_name"))
        dataset_set = _split_datasets(getattr(r, "datasets", None)) or (
            [repr_ds] if repr_ds else []
        )

        row = {name: np.nan for name in FEATURE_NAMES}
        matched = False

        # identity variant (representative dataset)
        if repr_ds is not None and sprint is not None and loader is not None:
            hit = identity_feats[
                (identity_feats["_d"] == repr_ds)
                & (identity_feats["_s"] == sprint)
                & (identity_feats["_l"] == loader)
            ]
            if len(hit) == 1:
                h = hit.iloc[0]
                for name in FEATURE_NAMES:
                    v = h[name]
                    row[name] = v
                matched = True

        # set variant over all datasets of the execution
        if sprint is not None and loader is not None:
            sl_key = (sprint, loader)
            det_pool = det_by_sl.get(sl_key)
            summ_pool = summ_by_sl.get(sl_key)
            ds_set = set(dataset_set)
            if det_pool is not None:
                pool = det_pool[det_pool["_d"].isin(ds_set)].copy()
                if not pool.empty:
                    matched = True
                    order = pool["_order"].dropna()
                    row["def_execution_order_max"] = float(order.max()) if not order.empty else np.nan
                    row["def_execution_order_mean"] = float(order.mean()) if not order.empty else np.nan
                    row["def_execution_order_std"] = (
                        float(order.std(ddof=1)) if len(order) >= 2 else np.nan
                    )
                    row["def_component_count"] = float(len(pool))
                    pool_types = pool["type"].map(_type_norm).dropna()
                    row["def_type_nunique"] = float(pool_types.nunique())
                    row["def_dominant_calling_type"] = _dominant(pool["calling_type"])
                    prev = pool[pool["calling_type"].map(lambda v: (_norm(v) or "").upper()) == "PREVALIDATION"]
                    trans = pool[pool["calling_type"].map(lambda v: (_norm(v) or "").upper()) == "TRANSFORMATION"]
                    for prefix, sec in (("preval", prev), ("trans", trans)):
                        stats = _section_order_stats(sec["_order"].dropna())
                        row[f"def_{prefix}_order_max"] = stats.get("max", np.nan)
                        row[f"def_{prefix}_order_mean"] = stats.get("mean", np.nan)
                        row[f"def_{prefix}_order_std"] = stats.get("std", np.nan)
                        row[f"def_{prefix}_type_nunique"] = float(
                            sec["type"].map(_type_norm).dropna().nunique()
                        )
                    for t in _CANONICAL_TYPES:
                        row[f"def_type_{t}_count"] = float((pool_types == t).sum())
            if summ_pool is not None:
                pool = summ_pool[summ_pool["_d"].isin(ds_set)]
                if not pool.empty:
                    matched = True
                    row["def_prevalidation_count"] = float(
                        pd.to_numeric(pool["prevalidation_count"], errors="coerce").max()
                    )
                    row["def_transformation_count"] = float(
                        pd.to_numeric(pool["transformation_count"], errors="coerce").max()
                    )

        row["def_match"] = matched
        rows.append(row)

    return pd.DataFrame(rows, index=executions.index)


def attach_definition_features(
    executions: pd.DataFrame,
    definitions_detail: pd.DataFrame,
    definitions_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Attach configuration features to executions (execution-level variant)."""
    feats = build_execution_level_definition_features(
        executions, definitions_detail, definitions_summary
    )
    out = pd.concat([executions.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    return out
