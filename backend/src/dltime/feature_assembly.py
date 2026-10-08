"""Training feature-frame assembly (Task 02, Parts G/H).

Produces the execution-level ML feature frame where every feature is
legitimately available BEFORE the current execution starts:

    PRE-EXECUTION INPUT (definitions) -> pre-execution config features
    PRIOR EXECUTIONS ONLY             -> historical features (AD-11)
    TARGET                            -> timetaken_in_sec (+ log1p prep, Part H)

The frame separates identifiers / pre-execution features / historical
features / target / analytical-only fields and must pass the leakage
validator before it is returned. No model training happens here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dltime.definition_features import (
    FEATURE_NAMES as DEF_FEATURES,
    CATEGORICAL_FEATURES,
    attach_definition_features,
)
from dltime.feature_registry import DEFAULT_FEATURE_REGISTRY, FeatureRegistry
from dltime.historical import HIST_FEATURE_NAMES, build_historical_features
from dltime.history_eligibility import (
    HistoryEligibilityEngine,
    summarize_prior_history_distribution,
)
from dltime.leakage import guard_frame
from dltime.pipeline import build_foundation
from dltime.schemas import DEFAULT_REGISTRY


class FeatureAssemblyError(ValueError):
    """Raised when the feature frame cannot be assembled safely."""


# ---------------- column blocks ----------------
IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "LDR_Execution_Id",
    "ldr_workflow_execution_id",
    "Loader_Name",
    "Sprint",
    "Dataset_display_name",
)
TARGET_COLUMN = "timetaken_in_sec"
ANALYTICAL_ONLY_COLUMNS: tuple[str, ...] = (
    "LDR_Status",
    "Start_Time",
    "End_Time",
    "n_datasets",
    "datasets",
    "n_component_rows",
    "n_target_rows",
    "Total_Records",
    "Success_Records",
    "Error_Records",
    "Prevalidation_Errors",
    "Format_Errors",
    "Constraint_Errors",
    "Application_Errors",
    "Cascade_Errors",
    "execution_resolution_status",
    "execution_connection",
    "is_valid_execution",
    "is_target_valid",
    "is_training_eligible",
    "eligibility_reasons",
    "def_match",
)

def _group_features(prefixes: tuple[str, ...]) -> list[str]:
    return [n for n in HIST_FEATURE_NAMES if any(n.startswith(p) for p in prefixes)]


PREDICTION_FEATURE_GROUPS: dict[str, list[str]] = {
    "PRE_EXECUTION_CONFIG": list(DEF_FEATURES),
    "HISTORICAL_GLOBAL": _group_features(("hist_global",)),
    "HISTORICAL_LOADER": _group_features(("hist_loader_prior",)),
    "HISTORICAL_LOADER_CONNECTION": _group_features(("hist_loader_conn",)),
    "HISTORICAL_LOADER_DATASET": _group_features(("hist_loader_dataset",)),
    "HISTORICAL_COMPOSITE": _group_features(("hist_composite",)),
    "HISTORICAL_PREVIOUS": _group_features(("hist_prev", "hist_time_since")),
}


def sync_engineered_fields(registry=DEFAULT_FEATURE_REGISTRY, field_registry=DEFAULT_REGISTRY) -> None:
    """Register engineered features in the leakage field registry.

    Keeps the Task-01 leakage validator authoritative: every engineered
    feature becomes classifiable exactly like raw fields.
    """
    from dltime.schemas import FieldClassification

    for spec in registry.specs.values():
        if spec.name not in field_registry:
            field_registry.register(
                FieldClassification(
                    spec.name,
                    spec.category,
                    note=f"engineered feature (group={spec.group})",
                    source_role="ENGINEERED",
                )
            )


class TargetTransformer:
    """Target transformation interface (Part H).

    Two modes kept available for model-training evaluation (no choice locked):
        "raw"    -> identity
        "log1p"  -> log1p(y); inverse via expm1
    """

    def __init__(self, mode: str = "log1p") -> None:
        if mode not in ("raw", "log1p"):
            raise FeatureAssemblyError(f"Unknown target mode '{mode}'")
        self.mode = mode

    def transform(self, y: pd.Series) -> np.ndarray:
        y = pd.to_numeric(y, errors="raise").astype(float)
        if (y <= 0).any():
            raise FeatureAssemblyError("Target must be positive for log1p transformation.")
        return np.log1p(y.to_numpy()) if self.mode == "log1p" else y.to_numpy()

    def inverse_transform(self, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.asarray(y_pred, dtype=float)
        return np.expm1(y_pred) if self.mode == "log1p" else y_pred


@dataclass
class TrainingFrame:
    """Assembled, leakage-validated execution-level feature frame."""

    frame: pd.DataFrame
    X: pd.DataFrame  # prediction-time features only (pre-execution + historical)
    y_raw: pd.Series
    y_log1p: pd.Series
    identifier_columns: list[str]
    feature_columns: dict[str, list[str]]
    analytical_columns: list[str]
    target_column: str
    diagnostics: dict = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return [n for cols in self.feature_columns.values() for n in cols]


def build_training_frame(
    *,
    bundle=None,
    prior_filter: str = "training_eligible",
    registry: FeatureRegistry = DEFAULT_FEATURE_REGISTRY,
    validate: bool = True,
) -> TrainingFrame:
    """Assemble the leakage-safe training feature frame from the real sources."""
    sync_engineered_fields(registry)
    return _assemble(
        foundation=build_foundation(bundle),
        prior_filter=prior_filter,
        registry=registry,
        validate=validate,
    )


def _assemble(
    *,
    foundation,
    prior_filter: str,
    registry: FeatureRegistry,
    validate: bool,
) -> TrainingFrame:
    from dltime.sources import load_all_sources

    bundle = load_all_sources()

    # ---- pre-execution configuration features (canonical identity join) ----
    executions = attach_definition_features(
        foundation.frames["executions_eligible"],
        bundle.loader_definitions_detail,
        bundle.loader_definitions_summary,
    )

    # ---- chronological historical features (prior-only) ----
    hist = build_historical_features(executions, prior_filter=prior_filter)
    full = hist.frame

    # ---- training subset: Task-01 eligibility AND feature availability ----
    train = full[full["is_training_eligible"].fillna(False)].copy()

    feature_columns = {g: list(cols) for g, cols in PREDICTION_FEATURE_GROUPS.items()}
    all_features = [n for cols in feature_columns.values() for n in cols]

    X = train[all_features].copy()
    # Categorical config feature: keep as string; model encoding is a Task 03 decision.
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("string")

    y_raw = pd.to_numeric(train[TARGET_COLUMN], errors="coerce")
    if y_raw.isna().any():
        raise FeatureAssemblyError("Training rows must carry a numeric target.")
    if (y_raw <= 0).any():
        raise FeatureAssemblyError("Training target must be positive.")

    y_log1p = pd.Series(np.log1p(y_raw.to_numpy()), index=y_raw.index)

    if validate:
        # Registry metadata must exist and classify every feature.
        for name in all_features:
            registry.get(name)
        # Authoritative Task-01 leakage boundary on the feature set...
        registry.validate_prediction_features(all_features, context="training-frame")
        # ...and on the concrete frame (no post-execution columns alongside).
        guard_frame(X, all_features, context="training-frame")
        # Chronology sanity: first execution chronologically must have zero priors.
        ordered = full.sort_values(["Start_Time", "LDR_Execution_Id"], kind="mergesort")
        first = ordered.iloc[0]
        if first["hist_global_prior_count"] != 0:
            raise FeatureAssemblyError("Chronology violation: first execution has prior history.")

    diagnostics = {
        "reconstructed_executions": int(len(full)),
        "training_rows": int(len(train)),
        "executions_with_at_least_one_prior": int(
            (train["hist_global_prior_count"] > 0).sum()
        ),
        "executions_with_prev_loader_duration": int(
            train["hist_prev_loader_duration_sec"].notna().sum()
        ),
        "executions_with_valid_time_since_prev": (
            int(
                (
                    train["hist_time_since_prev_loader_sec"].notna()
                    & (train["hist_time_since_prev_loader_sec"] >= 0)
                ).sum()
            )
            if "hist_time_since_prev_loader_sec" in train.columns
            else 0
        ),
        "resolved_connection_rows": int(train["execution_connection"].notna().sum()),
        "definition_feature_rows": int(train["def_match"].sum()),
        "prior_loader_history_distribution_all": summarize_prior_history_distribution(
            full["hist_loader_prior_count"]
        ),
        "prior_loader_history_distribution_training": summarize_prior_history_distribution(
            train["hist_loader_prior_count"]
        ),
        "feature_counts_by_group": {g: len(c) for g, c in feature_columns.items()},
        "history_diagnostics": hist.diagnostics,
        "resolution": foundation.resolution_executions,
    }

    return TrainingFrame(
        frame=full,
        X=X,
        y_raw=y_raw,
        y_log1p=y_log1p,
        identifier_columns=list(IDENTIFIER_COLUMNS),
        feature_columns=feature_columns,
        analytical_columns=[
            c for c in ANALYTICAL_ONLY_COLUMNS if c in full.columns
        ],
        target_column=TARGET_COLUMN,
        diagnostics=diagnostics,
    )


def evaluate_history_eligibility(training_frame: TrainingFrame, engine: HistoryEligibilityEngine | None = None):
    """Evaluate configurable history eligibility for every training row."""
    engine = engine or HistoryEligibilityEngine()
    X = training_frame.X
    results = []
    for count, conn_count, ts_prev in zip(
        X["hist_loader_prior_count"],
        X["hist_loader_conn_prior_count"],
        X["hist_time_since_prev_loader_sec"],
    ):
        results.append(engine.evaluate(
            hist_loader_prior_count=count,
            hist_loader_conn_prior_count=conn_count,
            hist_time_since_prev_loader_sec=ts_prev,
        ))
    reasons: dict[str, int] = {}
    for r in results:
        reasons[r.reason] = reasons.get(r.reason, 0) + 1
    return {
        "sufficient": sum(r.sufficient for r in results),
        "insufficient": sum(not r.sufficient for r in results),
        "reason_counts": reasons,
        "config": engine.config,
    }
