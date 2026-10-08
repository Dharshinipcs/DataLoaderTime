"""Machine-readable feature registry (Task 02, Part E).

Every production feature is declared with source fields, group, availability
category, calculation logic, historical flag, grouping identity, prediction
validity and missing-value behavior. The leakage validator consumes the
registry: a feature not registered here (or registered as non-predictable)
cannot enter a prediction-time feature set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dltime.leakage import validate_feature_set
from dltime.schemas import DEFAULT_REGISTRY, LeakageCategory, RegistryError


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    category: str  # LeakageCategory value
    source_fields: tuple[str, ...]
    logic: str
    is_historical: bool
    grouping_identity: str  # "" when not identity-grouped
    valid_for_prediction: bool
    missing_behavior: str


@dataclass
class FeatureRegistry:
    specs: dict[str, FeatureSpec] = field(default_factory=dict)

    def register(self, spec: FeatureSpec) -> None:
        if spec.name in self.specs:
            raise ValueError(f"Feature '{spec.name}' already registered")
        if spec.category not in {LeakageCategory.PRE_EXECUTION, LeakageCategory.DERIVED_FROM_PRE_EXECUTION,
                                 LeakageCategory.DERIVED_FROM_POST_EXECUTION}:
            raise ValueError(f"Feature '{spec.name}' has non-predictable category {spec.category}")
        if spec.is_historical and spec.category != LeakageCategory.DERIVED_FROM_POST_EXECUTION:
            raise ValueError(
                f"Feature '{spec.name}': historical features must be DERIVED_FROM_POST_EXECUTION"
            )
        self.specs[spec.name] = spec

    def names(self, *, group: str | None = None, prediction_only: bool = True) -> list[str]:
        return [
            s.name
            for s in self.specs.values()
            if (group is None or s.group == group) and (s.valid_for_prediction or not prediction_only)
        ]

    def specs_by_group(self) -> dict[str, list[FeatureSpec]]:
        out: dict[str, list[FeatureSpec]] = {}
        for s in self.specs.values():
            out.setdefault(s.group, []).append(s)
        return out

    def validate_prediction_features(
        self, names: list[str], *, context: str = "", field_registry=None
    ) -> None:
        """Authoritative pre-execution boundary check (delegates to leakage.py)."""
        validate_feature_set(
            names,
            registry=field_registry if field_registry is not None else DEFAULT_REGISTRY,
            context=context or "feature-registry",
        )

    def get(self, name: str) -> FeatureSpec:
        if name not in self.specs:
            raise RegistryError(f"Feature '{name}' is not registered.")
        return self.specs[name]


def _hist(group: str, prefix: str, fields: tuple[str, ...], identity: str) -> list[FeatureSpec]:
    return [
        FeatureSpec(
            name=f"hist_{prefix}_prior_{stat}",
            group=f"HISTORICAL_{group}",
            category=LeakageCategory.DERIVED_FROM_POST_EXECUTION,
            source_fields=fields,
            logic=f"prior-only {stat} over strictly earlier executions (AD-11)",
            is_historical=True,
            grouping_identity=identity,
            valid_for_prediction=True,
            missing_behavior="<NA> when no prior execution (never fabricated)",
        )
        for stat in ("count", "mean_sec", "median_sec", "std_sec")
    ]


def build_default_feature_registry() -> FeatureRegistry:
    reg = FeatureRegistry()

    # ---------------- historical features (AD-11) ----------------
    for spec in _hist("GLOBAL", "global", ("timetaken_in_sec of prior executions",), "none"):
        reg.register(spec)
    for spec in _hist("LOADER", "loader", ("timetaken_in_sec of prior executions",), "Loader_Name"):
        reg.register(spec)
    for spec in _hist("LOADER_CONNECTION", "loader_conn", ("timetaken_in_sec of prior executions", "execution_connection"), "Loader_Name+execution_connection"):
        reg.register(spec)
    for spec in _hist("LOADER_DATASET", "loader_dataset", ("timetaken_in_sec of prior executions", "Dataset_display_name"), "Loader_Name+Dataset_display_name"):
        reg.register(spec)
    for spec in _hist("COMPOSITE", "composite", ("timetaken_in_sec of prior executions",), "Dataset+Sprint+Loader"):
        reg.register(spec)

    reg.register(
        FeatureSpec(
            name="hist_prev_loader_duration_sec",
            group="HISTORICAL_PREVIOUS",
            category=LeakageCategory.DERIVED_FROM_POST_EXECUTION,
            source_fields=("timetaken_in_sec of strictly prior same-loader execution",),
            logic="duration of the most recent strictly prior eligible loader execution (AD-11)",
            is_historical=True,
            grouping_identity="Loader_Name",
            valid_for_prediction=True,
            missing_behavior="<NA> when no prior loader execution",
        )
    )
    reg.register(
        FeatureSpec(
            name="hist_time_since_prev_loader_sec",
            group="HISTORICAL_PREVIOUS",
            category=LeakageCategory.DERIVED_FROM_POST_EXECUTION,
            source_fields=("Start_Time of strictly prior same-loader execution",),
            logic="Start_Time(current) - Start_Time(prev), clipped to >= 0",
            is_historical=True,
            grouping_identity="Loader_Name",
            valid_for_prediction=True,
            missing_behavior="<NA> when no prior loader execution",
        )
    )

    # ---------------- pre-execution configuration features (AD-13) ----------------
    def_features = [
        ("def_prevalidation_count", ("LOADER_DEFINATIONS_1.prevalidation_count",), "max prevalidation_count per identity", False),
        ("def_transformation_count", ("LOADER_DEFINATIONS_1.transformation_count",), "max transformation_count per identity", False),
        ("def_execution_order_max", ("LOADER_DEFINATIONS.execution_order",), "max execution_order per identity", False),
        ("def_execution_order_mean", ("LOADER_DEFINATIONS.execution_order",), "mean execution_order per identity", False),
        ("def_execution_order_std", ("LOADER_DEFINATIONS.execution_order",), "std (ddof=1) of execution_order per identity", False),
        ("def_component_count", ("LOADER_DEFINATIONS.execution_order",), "count of definition components per identity", False),
        ("def_type_nunique", ("LOADER_DEFINATIONS.type",), "nunique component types per identity", False),
        # 2026-10-04 (user spec): execution_order is PER SECTION — prevalidation
        # rows ordered 1..n among themselves, transformation rows 1..m among
        # themselves. Per-section order stats + per-section/per-type counts.
        ("def_preval_order_max", ("LOADER_DEFINATIONS.execution_order",), "max prevalidation-section execution_order (independent 1..n sequence)", False),
        ("def_preval_order_mean", ("LOADER_DEFINATIONS.execution_order",), "mean prevalidation-section execution_order", False),
        ("def_preval_order_std", ("LOADER_DEFINATIONS.execution_order",), "std (ddof=1) prevalidation-section execution_order", False),
        ("def_trans_order_max", ("LOADER_DEFINATIONS.execution_order",), "max transformation-section execution_order (independent 1..m sequence)", False),
        ("def_trans_order_mean", ("LOADER_DEFINATIONS.execution_order",), "mean transformation-section execution_order", False),
        ("def_trans_order_std", ("LOADER_DEFINATIONS.execution_order",), "std (ddof=1) transformation-section execution_order", False),
        ("def_preval_type_nunique", ("LOADER_DEFINATIONS.type",), "nunique types among prevalidation components", False),
        ("def_trans_type_nunique", ("LOADER_DEFINATIONS.type",), "nunique types among transformation components", False),
        ("def_type_crossreference_count", ("LOADER_DEFINATIONS.type",), "count of crossreference-type components", False),
        ("def_type_dataobject_count", ("LOADER_DEFINATIONS.type",), "count of dataobject-type components", False),
        ("def_type_dataextract_count", ("LOADER_DEFINATIONS.type",), "count of dataextract-type components", False),
        ("def_type_dataset_count", ("LOADER_DEFINATIONS.type",), "count of dataset-type components", False),
        ("def_type_query_count", ("LOADER_DEFINATIONS.type",), "count of query-type components", False),
        ("def_type_table_count", ("LOADER_DEFINATIONS.type",), "count of table-type components", False),
    ]
    for name, fields, logic, _ in def_features:
        reg.register(
            FeatureSpec(
                name=name,
                group="PRE_EXECUTION_CONFIG",
                category=LeakageCategory.DERIVED_FROM_PRE_EXECUTION,
                source_fields=fields,
                logic=logic,
                is_historical=False,
                grouping_identity="Dataset+Sprint+Loader",
                valid_for_prediction=True,
                missing_behavior="<NA> when identity has no definition records",
            )
        )
    reg.register(
        FeatureSpec(
            name="def_dominant_calling_type",
            group="PRE_EXECUTION_CONFIG",
            category=LeakageCategory.PRE_EXECUTION,
            source_fields=("LOADER_DEFINATIONS.calling_type",),
            logic="most frequent calling_type per identity; None on tie (no silent choice)",
            is_historical=False,
            grouping_identity="Dataset+Sprint+Loader",
            valid_for_prediction=True,
            missing_behavior="<NA> when identity has no definition records or tie",
        )
    )

    return reg


DEFAULT_FEATURE_REGISTRY: FeatureRegistry = build_default_feature_registry()
