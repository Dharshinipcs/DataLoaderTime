"""Leakage boundary enforcement (locked rule, ARCHITECTURE_DECISIONS.md AD-07).

A prediction made before an execution starts may only use:
  - PRE_EXECUTION fields of the current execution,
  - DERIVED_FROM_PRE_EXECUTION fields,
  - historical information computed exclusively from PRIOR executions
    (chronology enforced in the historical-features layer, next task).

The guards here make crossing the boundary an explicit error rather than an
accident in later feature engineering.
"""

from __future__ import annotations

from collections.abc import Iterable

from dltime.schemas import (
    ALLOWED_FOR_PREDICTION,
    DEFAULT_REGISTRY,
    LeakageCategory,
    RegistryError,
)


class LeakageViolation(AssertionError):
    """Raised when a feature set crosses the pre-execution boundary."""


def validate_feature_set(
    feature_names: Iterable[str],
    *,
    registry=None,
    context: str = "",
    allow_target: bool = False,
) -> None:
    """Assert that every feature is allowed in a prediction-time feature set.

    Args:
        feature_names: names of the columns feeding the model.
        registry: FieldRegistry to consult (defaults to the default registry).
        context: label included in error messages (e.g. "training matrix").
        allow_target: must stay False for prediction-time validation; provided
            only so evaluation harnesses can assert the guard fires correctly.

    Raises:
        LeakageViolation: if any field is TARGET, POST_EXECUTION, IDENTIFIER,
            ANALYTICAL_ONLY, or unclassified.
    """
    if allow_target:
        raise LeakageViolation(
            "allow_target=True is a test-only escape hatch and must never be used "
            "for a real prediction-time feature set."
        )
    # The shared DEFAULT_REGISTRY is the single authoritative classification
    # table (raw fields + engineered features synced from the feature registry).
    reg = registry if registry is not None else DEFAULT_REGISTRY
    ctx = f" ({context})" if context else ""
    for name in feature_names:
        if name not in reg:
            raise LeakageViolation(
                f"Unclassified field '{name}'{ctx} cannot enter a prediction feature set "
                "(register it in dltime.schemas first)."
            )
        category = reg.category_of(name)
        if category not in ALLOWED_FOR_PREDICTION:
            raise LeakageViolation(
                f"Leakage: '{name}' is {category}{ctx} and must not be used to "
                "predict the current execution."
            )


def guard_frame(
    df,
    feature_names: Iterable[str],
    *,
    context: str = "",
    registry=None,
) -> None:
    """Validate a concrete training/prediction frame.

    Fails if any feature column is missing from the frame, or if any extra
    frame column carries post-execution information that could be mistaken
    for a feature.
    """
    names = list(feature_names)
    validate_feature_set(names, registry=registry, context=context)
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise KeyError(f"Feature columns missing from frame{(' (' + context + ')') if context else ''}: {missing}")
    banned_present = [
        c
        for c in df.columns
        if c in {"timetaken_in_sec", "Time_Taken", "End_Time", "LDR_Status", "Start_Time"}
    ]
    if banned_present:
        raise LeakageViolation(
            f"Frame{(' (' + context + ')') if context else ''} carries post-execution/target "
            f"columns alongside features: {banned_present}. Separate the target from "
            "features before calling the model."
        )
