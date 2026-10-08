"""Part G: leakage classification and boundary enforcement tests."""

from __future__ import annotations

import pandas as pd
import pytest

from dltime.leakage import LeakageViolation, guard_frame, validate_feature_set
from dltime.schemas import (
    ALL_CATEGORIES,
    DEFAULT_REGISTRY,
    LeakageCategory,
    RegistryError,
)


def test_every_raw_execution_column_is_classified(bundle):
    for col in bundle.executions.columns:
        assert col in DEFAULT_REGISTRY, f"unclassified execution column: {col}"
        assert DEFAULT_REGISTRY.category_of(col) in ALL_CATEGORIES


def test_every_definition_column_is_classified(bundle):
    for col in bundle.loader_definitions_detail.columns:
        assert col in DEFAULT_REGISTRY
    for col in bundle.loader_definitions_summary.columns:
        assert col in DEFAULT_REGISTRY


def test_target_and_post_execution_fields_are_forbidden():
    for name in ("timetaken_in_sec", "Time_Taken", "End_Time", "LDR_Status", "Start_Time"):
        with pytest.raises(LeakageViolation):
            validate_feature_set([name], context="unit-test")


def test_record_error_counters_are_forbidden():
    for name in ("Total_Records", "Success_Records", "Error_Records",
                 "Prevalidation_Errors", "Format_Errors"):
        with pytest.raises(LeakageViolation):
            validate_feature_set([name], context="unit-test")


def test_pre_execution_definition_fields_are_allowed():
    names = ["datasetname", "sprint_name", "loader_display_name", "ldr_connection_name",
             "prevalidation_count", "transformation_count", "execution_order", "type",
             "calling_type"]
    # must not raise
    validate_feature_set(names, context="unit-test")


def test_unclassified_field_rejected():
    with pytest.raises(LeakageViolation):
        validate_feature_set(["brand_new_mystery_field"], context="unit-test")


def test_registry_conflict_detection():
    from dltime.schemas import FieldClassification, FieldRegistry

    reg = FieldRegistry()
    reg.register(FieldClassification("f1", LeakageCategory.PRE_EXECUTION))
    with pytest.raises(ValueError):
        reg.register(FieldClassification("f1", LeakageCategory.POST_EXECUTION))


def test_guard_frame_blocks_target_column_in_feature_frame():
    df = pd.DataFrame(
        {
            "prevalidation_count": [3, 4],
            "transformation_count": [1, 2],
            "timetaken_in_sec": [10.0, 20.0],  # target sitting next to features
        }
    )
    with pytest.raises(LeakageViolation):
        guard_frame(df, ["prevalidation_count", "transformation_count"], context="unit-test")


def test_guard_frame_missing_feature_column():
    df = pd.DataFrame({"prevalidation_count": [3]})
    with pytest.raises(KeyError):
        guard_frame(df, ["prevalidation_count", "transformation_count"], context="unit-test")


def test_registry_error_is_raised_for_unknown_field():
    with pytest.raises(RegistryError):
        DEFAULT_REGISTRY.category_of("no_such_field_anywhere")
