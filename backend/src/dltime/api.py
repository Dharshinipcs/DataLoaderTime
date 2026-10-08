"""FastAPI prediction endpoint (Task 04).

Pre-execution-only contract: the request carries identity/configuration
information a caller legitimately knows BEFORE execution starts. The backend
constructs all historical/configuration/derived features automatically
(Task 02 path). Post-execution fields cannot be expressed in the request
schema; anything resembling post-execution data is rejected as INVALID_INPUT.

States (PredictionResult.to_dict): PREDICTED / INSUFFICIENT_HISTORY /
UNRESOLVED_IDENTITY / AMBIGUOUS_IDENTITY / INVALID_INPUT.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from dltime import __version__, notifications
from dltime.pipeline import build_foundation
from dltime.store import utc_now_iso
from dltime.inference import (
    FORBIDDEN_REQUEST_FIELDS,
    PrevalTransComponent,
    PredictionRequest,
    PredictionService,
    PredictionState,
)


class PrevalTransComponentBody(BaseModel):
    """One per-component pre-execution row (real detail-source shape)."""

    model_config = {"extra": "forbid"}

    display_name: str = Field(min_length=1, description="Component display name (e.g. Preval_1_employeeName, tf)")
    calling_type: str = Field(min_length=1, description="PREVALIDATION or TRANSFORMATION (the component's calling type)")
    # 2026-10-03: the component's TYPE — a full pre-execution field the stage
    # models learn from on retrain (crossreference | dataobject | dataextract |
    # dataset | query | table). Accepted under the source's own key "type".
    component_type: str | None = Field(
        default=None,
        alias="type",
        description="Component type: crossreference, dataobject, dataextract, dataset, query or table",
    )
    execution_order: float | None = Field(default=None, description="Execution order of this component")

    @field_validator("display_name", "calling_type")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class PredictionRequestBody(BaseModel):
    """Clean pre-execution prediction request body (no derived/post-execution data)."""

    model_config = {"extra": "forbid"}  # unknown fields cannot smuggle leakage

    loader_name: str = Field(min_length=1, description="Loader display name")
    sprint: str = Field(min_length=1, description="Sprint name")
    dataset: str = Field(min_length=1, description="Dataset display name")
    # Task 10.1: Connection is now REQUIRED. It remains validation-only — the
    # AD-04 definition-based resolution is authoritative and is never
    # overridden; disagreement returns INVALID_INPUT.
    connection: str = Field(
        min_length=1,
        description="Required; must agree with definition-based resolution (never overrides it)",
    )
    workflow_execution_id: str | None = Field(
        default=None,
        description="LINKAGE ONLY: caller-side execution reference for monitoring. Never a model feature and never shown as a prediction input.",
    )
    planned_start_time: str | None = Field(
        default=None,
        description="Optional evaluation timestamp (ISO-8601). NOT a source/model feature; defaults to latest recorded start + 1 day so historical features stay chronological.",
    )
    # ---- AD-31 legitimate pre-execution configuration (prompt §2/§21) -------
    # Optional user-supplied pre-execution configuration. When supplied, these
    # values describe THIS execution's real configuration and take precedence
    # over definition-derived aggregates for the context row (they never touch
    # historical aggregates). None -> definition-derived values are used.
    preval_trans_display_name: str | None = Field(
        default=None, description="(Legacy combined) prevalidation/transformation display name — audit echo only"
    )
    # EXP-032 Part F/G: the source detail table is PER-COMPONENT with
    # calling_type separating PREVALIDATION from TRANSFORMATION rows; one
    # execution config routinely contains both. Independent fields required.
    prevalidation_display_name: str | None = Field(
        default=None, description="Prevalidation component display name (pre-execution definition field)"
    )
    transformation_display_name: str | None = Field(
        default=None, description="Transformation component display name (pre-execution definition field)"
    )
    loader_type: str | None = Field(
        default=None, description="Loader configuration 'type' (pre-execution definition field)"
    )
    calling_type: str | None = Field(
        default=None, description="Calling type (pre-execution definition field)"
    )
    execution_order: float | None = Field(
        default=None, description="Execution order of this component (pre-execution definition field)"
    )
    prevalidation_count: float | None = Field(
        default=None, description="Prevalidation count for this execution (pre-execution summary field)"
    )
    transformation_count: float | None = Field(
        default=None, description="Transformation count for this execution (pre-execution summary field)"
    )
    # 2026-10-02: per-component rows exactly matching the real pre-execution
    # detail source (one row per component: display name + calling type +
    # execution order). Supersedes the scalar display-name/count fields when
    # supplied; each row re-derives the training-time detail aggregates for
    # the context row only.
    preval_trans_components: list[PrevalTransComponentBody] | None = Field(
        default=None,
        description="Per-component pre-execution rows: display_name, calling_type (PREVALIDATION|TRANSFORMATION), execution_order",
    )

    @field_validator("loader_name", "sprint", "dataset")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("connection")
    @classmethod
    def _connection_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("workflow_execution_id")
    @classmethod
    def _blank_to_none(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v if v else None


class ActualOutcomeBody(BaseModel):
    """Actual execution outcome (POST-execution data; Task 11 + AD-32 §26).

    Fields mirror the corrected ``train/data/Loader_Executions.xlsx`` schema:
    the fields the Loader team genuinely has when an execution finishes.
    Stage durations use the source's MINUTES semantics for the ``_in_min``
    fields; the authoritative total is ``total_timetaken_in_min`` (AD-34:
    EXP-032 proved ``total_min x 60`` matches the Start/End span on 757/757
    rows while legacy ``timetaken_in_sec`` agrees only 15.1%).
    ``actual_total_seconds`` remains as a backward-compat fallback for older
    API clients; when both are sent the minutes field wins and a quality
    warning is raised on a >5s mismatch.

    Post-execution values live ONLY in the monitoring/feedback path — they
    can never enter the pre-execution prediction feature assembly (the
    /predict schema has no such fields).
    """

    model_config = {"extra": "forbid"}

    loader_name: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    sprint: str = Field(min_length=1)
    connection: str = Field(min_length=1)
    workflow_execution_id: str | None = None
    ldr_execution_id: str | None = None
    start_time: str = Field(min_length=1)
    end_time: str | None = None
    # AD-34: authoritative total field is the source's total_timetaken_in_min
    actual_total_timetaken_in_min: float | None = Field(default=None, ge=0)
    actual_total_seconds: float | None = None  # backward-compat fallback
    # AD-38 §18: True when the UI derived the minutes from Start/End (the user
    # did not calculate it by hand). Audit context only — never a feature.
    actual_total_derived_from_timestamps: bool = False
    status: str | None = None
    # ---- XLSX-faithful record/error counts (post-execution) -------------
    total_records: int | None = Field(default=None, ge=0)
    success_records: int | None = Field(default=None, ge=0)
    error_records: int | None = Field(default=None, ge=0)
    prevalidation_errors: int | None = Field(default=None, ge=0)
    format_errors: int | None = Field(default=None, ge=0)
    constraint_errors: int | None = Field(default=None, ge=0)
    application_errors: int | None = Field(default=None, ge=0)
    cascade_errors: int | None = Field(default=None, ge=0)
    # ---- XLSX-faithful stage durations (MINUTES, source semantics) ------
    staging_timetaken_in_min: float | None = Field(default=None, ge=0)
    prevalidation_timetaken_in_min: float | None = Field(default=None, ge=0)
    transformation_timetaken_in_min: float | None = Field(default=None, ge=0)
    pre_load_timetaken_in_min: float | None = Field(default=None, ge=0)
    data_loading_timetaken_in_min: float | None = Field(default=None, ge=0)
    datamart_approval_timetaken_in_min: float | None = Field(default=None, ge=0)
    dataloading_approval_timetaken_in_min: float | None = Field(default=None, ge=0)
    # Free-form stage actuals in SECONDS (legacy contract; explicit fields win)
    stage_actuals: dict[str, float] = Field(default_factory=dict)

    @field_validator("loader_name", "dataset", "sprint", "connection")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class ActualIngestResponseBody(BaseModel):
    """Result of ingesting one actual outcome + attempted linkage."""

    outcome_id: str
    linkage_status: Literal["MATCHED", "UNMATCHED", "AMBIGUOUS"]
    linkage_method: str | None = None
    # 2026-10-05 (user request): plain reason WHY linkage succeeded or
    # failed, so the UI can explain an UNMATCHED manual entry
    # (no equal-identity prediction event vs all already linked).
    linkage_reason: str | None = None
    prediction_event_id: str | None = None
    feedback_id: str | None = None
    total_error: dict[str, Any] | None = None
    stage_errors: dict[str, Any] | None = None
    training_eligible: bool = False
    training_eligibility_reason: str | None = None
    duplicate: bool = False
    warnings: list[str] = Field(default_factory=list)


class PredictionResponseBody(BaseModel):
    """Explicit-state prediction response (UI-ready metadata)."""

    state: Literal[
        "PREDICTED",
        "INSUFFICIENT_HISTORY",
        "UNRESOLVED_IDENTITY",
        "AMBIGUOUS_IDENTITY",
        "INVALID_INPUT",
    ]
    reason: str | None = None
    predicted_seconds: float | None = None
    predicted_seconds_human: str | None = None
    predicted_log1p: float | None = None
    model_version: str | None = None
    model_status: str | None = None
    serving_role: str | None = Field(
        default=None,
        description=(
            "Serving role of the total policy: AUTHORITATIVE_TOTAL (the "
            "deterministic sum of the five operational stage predictions)."
        ),
    )
    direct_reference_seconds: float | None = Field(
        default=None,
        description=(
            "The retired direct-total reference model's ETA for the same "
            "context — comparison/rollback evidence only, never the primary ETA."
        ),
    )
    stage_aware_reference_seconds: float | None = Field(
        default=None,
        description=(
            "The RETIRED stage-aware total model's ETA for the same context — "
            "traceability/rollback evidence only, never the primary ETA."
        ),
    )
    physical_floor_seconds: float | None = Field(
        default=None,
        description=(
            "RETIRED with the AD-34 floor rule: always None under the sum "
            "policy — the total is the exact five-stage sum with no clamping."
        ),
    )
    floor_applied: bool = Field(
        default=False,
        description=(
            "RETIRED with the AD-34 floor rule: always False under the sum "
            "policy — the total is never clamped or adjusted."
        ),
    )
    inputs: dict[str, Any] = Field(default_factory=dict)
    identity: dict[str, Any] = Field(default_factory=dict)
    history: dict[str, Any] = Field(default_factory=dict)
    loader_case: dict[str, Any] | None = Field(
        default=None,
        description=(
            "The THREE loader-history cases (gate): UNKNOWN_LOADER (new loader, "
            "0 priors), BELOW_THRESHOLD (priors under the trust threshold), "
            "KNOWN_LOADER_TRUSTED (predictions + total served)."
        ),
    )
    features: dict[str, Any] | None = None
    stage_intelligence: dict[str, Any] | None = Field(
        default=None,
        description="Supplementary stage intelligence (PREDICTED only; never summed into the total ETA)",
    )
    uncertainty: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Calibrated uncertainty interval around the point ETA "
            "(PREDICTED only; never present for non-prediction states)"
        ),
    )
    explanation: dict[str, Any] | None = Field(
        default=None,
        description=(
            "SHAP model-contribution explanation of the log-scale ETA "
            "(PREDICTED only; explanatory enrichment — never alters the ETA)"
        ),
    )
    evidence_explanation: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Evidence-sentence explainability (evidence-v1): deterministic "
            "plain-English sentences over real, verifiable numbers — no AI/LLM. "
            "PREDICTED only; explanatory enrichment — never alters the ETA."
        ),
    )
    plain_explanation: str | None = Field(
        default=None,
        description=(
            "plain-explain-v1: ONE short plain-language reason (max 2 sentences, "
            "non-technical) for this prediction. PREDICTED only; never alters the ETA."
        ),
    )
    risk: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Deterministic operational risk classification (PREDICTED path: "
            "NORMAL/WARNING/HIGH_RISK/LOW_CONFIDENCE/OUT_OF_DISTRIBUTION; "
            "INSUFFICIENT_HISTORY state carries an opaque INSUFFICIENT_HISTORY "
            "label). Interpretation only — never alters the ETA; not a "
            "probability of failure."
        ),
    )
    warnings: list[str] = Field(default_factory=list)


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """Session-wide PredictionService (sources, artifact, eligibility config)."""
    return PredictionService.from_sources()


@lru_cache(maxsize=1)
def get_monitoring_store():
    """Process-wide append-only monitoring store (backend/store)."""
    from dltime.store import MonitoringStore

    return MonitoringStore()


@lru_cache(maxsize=1)
def get_model_registry():
    """Module 7 governance registries (experiments + model lifecycle).

    First use seeds the governed history with the REAL recorded evidence
    (EXP-012/EXP-023/EXP-025) — historical experiment results are never
    rewritten; seeding is append-only and skipped when records exist.
    Tests must patch MODEL_SEEDING to False (isolated registries only).
    """
    from dltime.model_lab import ExperimentRegistry, ModelRegistry, _JsonlStore

    store = _JsonlStore()
    experiments = ExperimentRegistry(store)
    if MODEL_SEEDING and not experiments.list():
        _seed_experiments(experiments)
    return _LabRegistries(experiments=experiments, models=ModelRegistry(store))


MODEL_SEEDING = True


def _seed_experiments(experiments) -> None:
    """Register the project's REAL recorded experiment evidence (EXP-012,
    EXP-023, EXP-025). Values are transcribed from EXPERIMENT_LOG.md — no
    fabricated results."""
    et_params = {"n_estimators": 200, "min_samples_leaf": 2, "max_features": 1.0, "random_state": 42, "n_jobs": -1}
    chrono = {
        "type": "expanding-window chronological CV + final chronological holdout",
        "config": {"holdout_fraction": 0.25, "min_holdout": 25, "n_folds": 5},
    }
    feats30 = "pre-execution+historical-v30"
    experiments.register(
        {
            "experiment_id": "EXP-LAB-012",
            "kind": "model-candidate-comparison",
            "model_family": "extra_trees_candidate",
            "target_transformation": "log1p",
            "hyperparameters": et_params,
            "feature_set_version": feats30,
            "feature_names": [],
            "training_population": {"rows": 350, "eligible": 350, "train_rows": 262, "holdout_rows": 88},
            "validation_method": chrono,
            "metrics": {
                "pooled_oof": {"mae": 364.6, "median_ae": 5.16, "rmse": 1711.4, "within_20pct": 16.0, "within_2x": 48.6},
                "holdout": {"mae": 696.7, "median_ae": 5.74},
                "served_population": {"n": 80, "median_ae": 2.41, "mae": 70.7, "within_2x": 56.2},
            },
            "slices": {"known_loader_medae": 3.81, "unseen_loader_medae": 15.1, "history_5_9_medae": 1.3},
            "artifact_reference": "backend/artifacts/total-et-v3-candidate.pkl",
            "status": "EVALUATED",
            "decision": "SELECTED_CANDIDATE",
            "rationale": "Task 03 evidence: best chronological stability; beats median baseline 2.4x MedAE on the served population (EXP-012).",
        }
    )
    experiments.register(
        {
            "experiment_id": "EXP-LAB-023",
            "kind": "model-candidate-comparison",
            "model_family": "hist_gradient_boosting",
            "target_transformation": "log1p",
            "hyperparameters": {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 10, "random_state": 42},
            "feature_set_version": feats30,
            "feature_names": [],
            "training_population": {"rows": 350, "train_rows": 262, "holdout_rows": 88},
            "validation_method": chrono,
            "metrics": {
                "pooled_oof": {"mae": 491.2, "median_ae": 15.96, "rmse": 1850.7, "within_2x": 31.4},
                "holdout": {"mae": 733.8, "median_ae": 2.95},
            },
            "slices": {},
            "status": "EVALUATED",
            "decision": "REJECTED",
            "rationale": "Prompt 2 evidence: fold MedAE instability (3.0-194.1) despite glossy holdout MedAE; rejected for chronological instability (EXP-023).",
        }
    )
    experiments.register(
        {
            "experiment_id": "EXP-LAB-025-FEATURE",
            "kind": "feature-experiment",
            "model_family": "extra_trees_candidate",
            "target_transformation": "log1p",
            "hyperparameters": et_params,
            "feature_set_version": "pre-execution+historical-v31+preval_name_nunique",
            "feature_names": ["def_preval_name_nunique"],
            "training_population": {"rows": 350, "train_rows": 262, "holdout_rows": 88},
            "validation_method": chrono,
            "metrics": {
                "pooled_oof": {"mae": 363.26, "median_ae": 4.69, "rmse": 1708.5, "within_20pct": 20.6, "within_2x": 49.1},
                "holdout": {"mae": 705.19, "median_ae": 5.31},
                "served_population": {"n": 80, "median_ae": 2.64, "mae": 61.05},
            },
            "slices": {
                "unseen_loader_medae": 12.16,
                "note": "coverage 129/350 rows; full baseline-vs-candidate table in EXP-025",
            },
            "status": "EVALUATED",
            "decision": None,
            "rationale": "Prompt 4 (§9) required experiment: def_preval_name_nunique improves pooled MedAE 5.16->4.69 but DEGRADES the served population MedAE 2.41->2.64; adoption is a human governance decision (not adopted in Prompt 4).",
        }
    )


class _LabRegistries:
    def __init__(self, experiments, models) -> None:
        self.experiments = experiments
        self.models = models


@lru_cache(maxsize=1)
def get_lab_context() -> dict:
    """Drift/reference context: training-window reference frame + current window.

    Reference = the model's chronological training window (Task 03/04 split,
    never future production data). Current = ALL executions strictly AFTER
    the reference window end (the chronological holdout period — model-unseen
    by construction), not a rolling time window: a rolling 90-day window
    would overlap the reference period on this dataset.
    """
    import pandas as pd

    from dltime.artifact import load_serving_artifact
    from dltime.modeling import add_chronological_rank, chronological_holdout_split

    tf = build_training_frame_bundle()
    eligible = tf.frame[tf.frame["is_training_eligible"].fillna(False)]
    ordered = add_chronological_rank(eligible)
    train, _test, split_info = chronological_holdout_split(ordered)
    reference = tf.frame.loc[train.index]
    # Current monitoring window: the chronological holdout AFTER the training
    # window — the latest recorded executions (model-unseen), not a
    # rolling 90-day window (which would overlap the reference period here).
    reference_end = pd.to_datetime(split_info["train_last_start"])
    current = tf.frame[pd.to_datetime(tf.frame["Start_Time"]) > reference_end]
    artifact = load_serving_artifact()
    feature_names = list(artifact["metadata"]["feature_names_in_order"])
    categorical = list(artifact["metadata"].get("categorical_features", []))
    from dltime.store import MonitoringStore

    return {
        "reference_frame": reference,
        "current_frame": current,
        "reference_window": {
            "first_start": split_info["train_first_start"],
            "last_start": split_info["train_last_start"],
            "rows": int(len(reference)),
        },
        "current_window": {
            "first_start": str(pd.to_datetime(current["Start_Time"]).min()) if len(current) else None,
            "last_start": str(pd.to_datetime(current["Start_Time"]).max()) if len(current) else None,
            "definition": "executions after the model training window (chronological holdout period)",
            "rows": int(len(current)),
        },
        "feature_names": feature_names,
        "categorical_features": categorical,
        "monitoring_store": MonitoringStore(),
    }


def build_training_frame_bundle():
    """Training frame for lab context (cached source path, same as service)."""
    from dltime.feature_assembly import build_training_frame as btf

    return btf(bundle=_load_bundle())


@lru_cache(maxsize=1)
def _load_bundle():
    from dltime.sources import load_all_sources

    return load_all_sources()


@lru_cache(maxsize=1)
def _global_explainer_state() -> tuple[Any, str, list[str] | None]:
    """Cached TreeExplainer for the global summary endpoint (built once).

    AD-32: global explanations describe the AUTHORITATIVE stage-aware total
    model when its artifact is available (encoded-column contract carried so
    the holdout rows can be encoded consistently); otherwise the direct
    reference model.
    """
    from dltime.explain import get_or_build_explainer
    from dltime.stage_aware_serving import artifact_exists, load_stage_aware_artifact

    if artifact_exists():
        sa = load_stage_aware_artifact()
        explainer, _cache = get_or_build_explainer(sa, None)
        return (
            explainer,
            sa["metadata"]["model_version"],
            list(sa["enc_columns"]),
        )
    artifact = get_prediction_service().artifact
    explainer, _cache = get_or_build_explainer(artifact, None)
    return explainer, artifact["metadata"]["model_version"], None


def create_app() -> FastAPI:
    app = FastAPI(
        title="DataZap Loader Execution-Time Prediction API",
        description=(
            "Pre-execution total-execution-time prediction for data loaders. "
            "Requests contain ONLY legitimate pre-execution information; "
            "historical/configuration features are constructed by the backend."
        ),
        version=__version__,
    )

    # Task 05: allow the local Prediction Studio dev server (Vite) to call the
    # API directly. Minimum integration change; no behavior change for /predict.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.post(
        "/predict",
        response_model=PredictionResponseBody,
        response_model_exclude_none=False,
        responses={422: {"description": "Request body violates the schema (extra/blank fields)"}},
    )
    def predict(body: PredictionRequestBody) -> PredictionResponseBody:
        # Defense-in-depth: reject any field that mirrors post-execution data,
        # even though the typed schema cannot carry it.
        smuggled = FORBIDDEN_REQUEST_FIELDS & set(body.model_dump().keys())
        request = PredictionRequest(
            loader_name=body.loader_name,
            sprint=body.sprint,
            dataset=body.dataset,
            connection=body.connection,
            workflow_execution_id=body.workflow_execution_id,
            planned_start_time=body.planned_start_time,
            preval_trans_display_name=body.preval_trans_display_name,
            prevalidation_display_name=body.prevalidation_display_name,
            transformation_display_name=body.transformation_display_name,
            loader_type=body.loader_type,
            calling_type=body.calling_type,
            execution_order=body.execution_order,
            prevalidation_count=body.prevalidation_count,
            transformation_count=body.transformation_count,
            preval_trans_components=(
                [
                    PrevalTransComponent(
                        display_name=c.display_name,
                        calling_type=c.calling_type,
                        component_type=c.component_type,
                        execution_order=c.execution_order,
                    )
                    for c in body.preval_trans_components
                ]
                if body.preval_trans_components
                else ()
            ),
        )
        result = get_prediction_service().predict(request)
        if smuggled:
            result = type(result)(
                state=PredictionState.INVALID_INPUT,
                reason=f"post_execution_fields_rejected: {sorted(smuggled)}",
                inputs=result.inputs,
                identity=result.identity,
                history=result.history,
                warnings=result.warnings,
            )
        return PredictionResponseBody(**result.to_dict())

    @app.post("/explain/prediction")
    def explain_prediction(body: PredictionRequestBody) -> dict:
        """Technical drill-down (2026-10-06 user spec): SHAP feature
        importance for ONE prediction. Recomputes the served prediction for
        the same pre-execution inputs and returns the real per-prediction
        SHAP layer (explanation.shap_explanation). READ-ONLY: no monitoring
        event is persisted and nothing about serving changes."""
        request = PredictionRequest(
            loader_name=body.loader_name,
            sprint=body.sprint,
            dataset=body.dataset,
            connection=body.connection,
            workflow_execution_id=body.workflow_execution_id,
            planned_start_time=body.planned_start_time,
            preval_trans_display_name=body.preval_trans_display_name,
            prevalidation_display_name=body.prevalidation_display_name,
            transformation_display_name=body.transformation_display_name,
            loader_type=body.loader_type,
            calling_type=body.calling_type,
            execution_order=body.execution_order,
            prevalidation_count=body.prevalidation_count,
            transformation_count=body.transformation_count,
            preval_trans_components=(
                [
                    PrevalTransComponent(
                        display_name=c.display_name,
                        calling_type=c.calling_type,
                        component_type=c.component_type,
                        execution_order=c.execution_order,
                    )
                    for c in body.preval_trans_components
                ]
                if body.preval_trans_components
                else ()
            ),
        )
        try:
            return get_prediction_service().shap_explanation_for_request(request)
        except Exception as exc:
            return {"available": False, "reason": f"explanation failed: {str(exc)[:200]}"}

    @app.get("/stages/analytics")
    def stage_analytics() -> dict:
        """Dataset-wide stage analytics (supplementary; real reconstructed data)."""
        from dltime.stages import build_stage_analytics

        service = get_prediction_service()
        executions = service.foundation.frames["executions_eligible"]
        # AD-31: the foundation frame already carries the stage-target columns
        # (single source of truth, pipeline.py Part F).
        result = build_stage_analytics(executions)
        return result.to_payload()

    @app.get("/explanation/global")
    def global_explanation() -> dict:
        """Lightweight GLOBAL stage-model attribution summary (Task 08; supplementary).

        Computed over the final chronological holdout — the exact rows
        the serving artifact never trained on — so no retraining and no
        train-row self-explanation. Explanatory only; never affects /predict.
        Under the sum-total policy the served total is a deterministic sum,
        so the global view describes the stage-aware total's attribution on
        the same holdout as supplementary stage-model diagnostics (clearly
        labelled; NOT an explanation of the served arithmetic — the served
        total's per-prediction explanation is the exact stage decomposition).
        """
        from dltime.modeling import add_chronological_rank, chronological_holdout_split
        from dltime.stage_aware_serving import (
            artifact_exists,
            global_explanation_stage_aware,
            load_stage_aware_artifact,
            stage_aware_holdout_matrix,
        )

        service = get_prediction_service()
        if artifact_exists():
            sa = load_stage_aware_artifact()
            X_holdout, _y = stage_aware_holdout_matrix(sa)
            explainer, _mv, _enc = _global_explainer_state()
            payload = global_explanation_stage_aware(sa, X_holdout, explainer)
            payload["applies_to_served_total"] = False
            payload["note"] = (
                "Supplementary stage-model attribution on the retired stage-aware "
                "total's holdout. The served total is the deterministic sum of the "
                "five operational stage predictions; its per-prediction "
                "explanation is the exact stage decomposition."
            )
            return payload

        from dltime.explain import global_explanation

        ordered = add_chronological_rank(
            service.tf.frame[service.tf.frame["is_training_eligible"].fillna(False)]
        )
        _train, test, _info = chronological_holdout_split(ordered)
        X_holdout = service.tf.X.loc[test.index]
        explainer, _mv, _enc = _global_explainer_state()
        return global_explanation(
            service.artifact,
            X_holdout,
            explainer,
            subset_description=(
                "Task 03 final chronological holdout (artifact training "
                "window excluded; documented in EXP-012/EXP-016)"
            ),
        )

    @app.get("/loaders/{loader_name}")
    def loader_intel(loader_name: str) -> dict:
        """Loader Intelligence (Task 10; read-only intelligence, never prediction).

        Answers "what do we know about this loader?" — history evidence,
        context, and honest cold-start context. The ETA gate stays AD-16 and
        lives in POST /predict; this endpoint never produces an ETA.
        """
        from dltime.loader_intel import build_loader_intel

        service = get_prediction_service()
        return build_loader_intel(
            loader_name=loader_name,
            frame=service.foundation.frames["executions_eligible"],
            definitions_detail=service.definitions_detail,
            definitions_summary=service.definitions_summary,
            min_eta_history=service.eligibility_config.min_prior_loader_executions,
        )

    @app.post("/feedback/actual", response_model=ActualIngestResponseBody)
    def ingest_actual_outcome(body: ActualOutcomeBody) -> ActualIngestResponseBody:
        """Record one actual execution outcome and attempt linkage (Task 11).

        Post-execution ingestion only. Idempotent for identical content;
        conflicting content for the same execution reference is recorded as
        a conflict, never silently overwritten.
        """
        import hashlib

        from dltime.feedback import (
            FEEDBACK_SCHEMA_VERSION,
            build_actual_outcome,
            build_feedback_record,
            compute_stage_errors,
            link_actual_to_predictions,
        )

        # AD-34: authoritative total is total_timetaken_in_min x 60 (EXP-032:
        # matches the physical Start/End span 757/757; legacy seconds field is
        # corrupted). Backward-compat seconds fallback for old API clients.
        total_min = body.actual_total_timetaken_in_min
        total_seconds: float | None
        quality_warnings: list[str] = []
        if total_min is not None:
            total_seconds = float(total_min) * 60.0
            if body.actual_total_seconds is not None and abs(total_seconds - float(body.actual_total_seconds)) > 5.0:
                quality_warnings.append(
                    "total_duration_source_mismatch: total_timetaken_in_min x 60 "
                    "differs from supplied actual_total_seconds by >5s; minutes field wins (AD-34)"
                )
        elif body.actual_total_seconds is not None:
            total_seconds = float(body.actual_total_seconds)
            quality_warnings.append(
                "total_duration_from_legacy_seconds_field: prefer total_timetaken_in_min (AD-34)"
            )
        else:
            raise HTTPException(status_code=422, detail="missing total: provide actual_total_timetaken_in_min")

        record, err = build_actual_outcome(
            loader_name=body.loader_name,
            dataset=body.dataset,
            sprint=body.sprint,
            connection=body.connection,
            start_time=body.start_time,
            end_time=body.end_time,
            ldr_execution_id=body.ldr_execution_id,
            workflow_execution_id=body.workflow_execution_id,
            actual_total_seconds=total_seconds,
            status=body.status,
            stage_actuals=body.stage_actuals,
            record_counts={
                "total_records": body.total_records,
                "success_records": body.success_records,
                "error_records": body.error_records,
                "prevalidation_errors": body.prevalidation_errors,
                "format_errors": body.format_errors,
                "constraint_errors": body.constraint_errors,
                "application_errors": body.application_errors,
                "cascade_errors": body.cascade_errors,
            },
            stage_minutes={
                "staging": body.staging_timetaken_in_min,
                "prevalidation": body.prevalidation_timetaken_in_min,
                "transformation": body.transformation_timetaken_in_min,
                "pre_load": body.pre_load_timetaken_in_min,
                "data_loading": body.data_loading_timetaken_in_min,
                "datamart_approval": body.datamart_approval_timetaken_in_min,
                "dataloading_approval": body.dataloading_approval_timetaken_in_min,
            },
        )
        if record is None:
            raise HTTPException(status_code=422, detail=err)
        record.setdefault("data_quality_warnings", []).extend(quality_warnings)
        # AD-38 §18: disclose duration derivation provenance when the UI
        # derived the total from Start/End instead of the user typing it.
        if body.actual_total_derived_from_timestamps:
            record.setdefault("duration_provenance", {})["total"] = (
                "derived_from_start_end_timestamps"
            )
        # Source-faithful derivation: minutes x 60 -> canonical SECONDS stage
        # map (identical semantics to the stage-target source policy). The
        # explicit seconds map (legacy stage_actuals) wins only where the
        # minutes field was not supplied.
        stage_seconds: dict[str, float] = dict(record.get("stage_actuals") or {})
        for stage, minutes in (record.get("stage_minutes_reported") or {}).items():
            if minutes is not None and stage not in stage_seconds:
                stage_seconds[stage] = float(minutes) * 60.0
        record["stage_actuals"] = stage_seconds

        store = get_monitoring_store()
        outcomes, _w = store.read_actual_outcomes()

        # idempotency / conflict protection (PART J)
        conflicts = [
            o
            for o in outcomes
            if o["outcome_id"] == record["outcome_id"] and o["outcome_hash"] != record["outcome_hash"]
        ]
        # outcome_id is content-derived, so a different hash under the same id
        # is impossible; real conflict check: same execution reference, different content
        ref_conflicts = [
            o
            for o in outcomes
            if o.get("execution_ref") == record.get("execution_ref")
            and o.get("outcome_hash") != record["outcome_hash"]
            and (o.get("execution_ref") or {}).get("workflow_execution_id")
        ]
        duplicates = [o for o in outcomes if o.get("outcome_hash") == record["outcome_hash"]]
        if ref_conflicts and not duplicates:
            raise HTTPException(
                status_code=409,
                detail="conflicting_actual_outcome: an actual with the same execution reference but different content already exists; not overwritten",
            )
        if duplicates:
            existing = duplicates[0]
            feedback_records, _fw = store.read_feedback_records()
            prior = [
                f
                for f in feedback_records
                if f.get("outcome_id") == existing.get("outcome_id")
            ]
            if prior:
                f0 = prior[0]
                return ActualIngestResponseBody(
                    outcome_id=existing["outcome_id"],
                    linkage_status=f0.get("linkage", {}).get("status", "UNMATCHED"),
                    linkage_method=f0.get("linkage", {}).get("method"),
                    linkage_reason=(
                        f0.get("linkage", {}).get("evidence", {}).get("reason")
                        if f0.get("linkage", {}).get("status") != "MATCHED"
                        else "linked to the prediction event recorded before this run"
                    ),
                    prediction_event_id=f0.get("prediction_event_id"),
                    feedback_id=f0.get("feedback_id"),
                    total_error=f0.get("total_error"),
                    stage_errors=f0.get("stage_errors"),
                    training_eligible=bool(existing.get("training_eligible")),
                    training_eligibility_reason=existing.get("training_eligibility_reason"),
                    duplicate=True,
                )

        store.append_actual_outcome(record)

        # 2026-10-05 fixed-scenario policy: outcome-derived alerts removed.
        # The Alert Center carries ONLY the three fixed scenarios (retraining
        # needed, stage-model promotion suggestion, drift detected) — see
        # dltime.notifications.

        # -------- linkage against persisted prediction events
        events, _ew = store.read_prediction_events()
        feedback_records, _fw = store.read_feedback_records()
        linkage = link_actual_to_predictions(record, events, feedback_records)
        linked_event = None
        stage_errors: dict = {}
        if linkage["linkage_status"] == "MATCHED":
            linked_id = linkage["prediction_event_ids"][0]
            linked_event = next((e for e in events if e.get("event_id") == linked_id), None)
            if linked_event is not None:
                si = linked_event.get("stage_intelligence") or {}
                stage_errors = compute_stage_errors(
                    si.get("stages"), record.get("stage_actuals") or {}
                )
        fb = build_feedback_record(
            actual=record,
            linkage=linkage,
            predicted_event=linked_event,
            stage_errors=stage_errors,
            data_quality_warnings=record.get("data_quality_warnings") or [],
        )
        store.append_feedback_record(fb)
        _ = FEEDBACK_SCHEMA_VERSION, hashlib  # (schema carried inside records)
        return ActualIngestResponseBody(
            outcome_id=record["outcome_id"],
            linkage_status=linkage["linkage_status"],
            linkage_method=linkage.get("linkage_method"),
            linkage_reason=(
                linkage.get("evidence", {}).get("reason")
                if linkage.get("linkage_status") != "MATCHED"
                else "linked to the prediction event recorded before this run"
            ),
            prediction_event_id=(linkage.get("prediction_event_ids") or [None])[0],
            feedback_id=fb["feedback_id"],
            total_error=fb.get("total_error"),
            stage_errors=fb.get("stage_errors") or None,
            training_eligible=bool(record.get("training_eligible")),
            training_eligibility_reason=record.get("training_eligibility_reason"),
            duplicate=False,
        )

    @app.get("/loaders")
    def loaders_directory() -> dict:
        """Loader directory rollup for the Loader Intelligence browse UI.

        One row per loader with executed history: execution/status counts,
        median duration, last seen, plus operational prediction accuracy from
        matched monitoring feedback (read-only; never a prediction).
        """
        import pandas as pd

        service = get_prediction_service()
        frame = service.foundation.frames["executions_eligible"]
        store = get_monitoring_store()
        events, _we = store.read_prediction_events()
        outcomes, _wo = store.read_actual_outcomes()
        feedback, _wf = store.read_feedback_records()
        matched = {
            r.get("identity", {}).get("loader_name", "").strip().lower(): r
            for r in feedback
            if r.get("linkage", {}).get("status") == "MATCHED" and r.get("total_error")
        }
        rows: list[dict] = []
        for name, grp in frame.groupby("Loader_Name", dropna=True):
            status = grp["LDR_Status"]
            # 2026-09-29 replacement: no total_timetaken_in_min column — the
            # execution duration is the AD-31 span (End - Start); legacy
            # timetaken_in_sec is span-consistent and usable as a fallback.
            if "total_timetaken_in_min" in grp.columns:
                dur = pd.to_numeric(grp["total_timetaken_in_min"], errors="coerce") * 60.0
            elif "timetaken_in_sec" in grp.columns:
                dur = pd.to_numeric(grp["timetaken_in_sec"], errors="coerce")
            else:
                dur = (
                    pd.to_datetime(grp["End_Time"], errors="coerce")
                    - pd.to_datetime(grp["Start_Time"], errors="coerce")
                ).dt.total_seconds()
            fb = matched.get(str(name).strip().lower())
            rows.append(
                {
                    "loader_name": str(name),
                    "executions": int(len(grp)),
                    "succeed": int((status == "SUCCEED").sum()),
                    "completed_with_warning": int(
                        status.isin(["COMPLETED_WITH_WARNING", "COMPLETED_WITH_ERROR_AND_WARNING"]).sum()
                    ),
                    "failed": int(status.isin(["ERROR", "STOPPED", "KILLED", "COMPLETED_WITH_ERROR"]).sum()),
                    "median_duration_sec": float(dur.median()) if dur.notna().any() else None,
                    "last_seen": str(grp["Start_Time"].max()) if "Start_Time" in grp else None,
                    "sprints": sorted(set(grp["Sprint"].dropna().astype(str)))[:6],
                    "datasets": sorted(set(grp["Dataset_display_name"].dropna().astype(str)))[:6],
                    "connections": sorted(set(grp["execution_connection"].dropna().astype(str)))[:6],
                    "prediction_count": int(
                        sum(
                            1
                            for e in events
                            if (e.get("request") or {}).get("loader_name", "").strip().lower()
                            == str(name).strip().lower()
                        )
                    ),
                    "matched_feedback": {
                        "absolute_error_seconds": fb["total_error"].get("absolute_error_seconds"),
                        "absolute_percentage_error": fb["total_error"].get("absolute_percentage_error"),
                    }
                    if fb
                    else None,
                }
            )
        rows.sort(key=lambda r: (-r["executions"], r["loader_name"]))
        return {"loaders": rows}

    @app.get("/monitoring/summary")
    def monitoring_summary(loader: str | None = None) -> dict:
        """Operational prediction-vs-actual aggregate (Task 11; NOT Task 03)."""
        from dltime.feedback import monitoring_summary as aggregate

        store = get_monitoring_store()
        events, w1 = store.read_prediction_events()
        outcomes, w2 = store.read_actual_outcomes()
        feedback, w3 = store.read_feedback_records()
        summary = aggregate(events, outcomes, feedback, loader_filter=loader)
        summary["store_warnings"] = [w.message for w in (w1 + w2 + w3)][:20]
        # AD-33 §34: served stage-estimate distributions (summary, no verdict).
        from dltime.model_lab import stage_prediction_distributions

        summary["stage_prediction_distributions"] = stage_prediction_distributions(events)
        return summary

    @app.get("/monitoring/events")
    def monitoring_events(
        limit: int = 50,
        state: str | None = None,
    ) -> dict:
        """Recent persisted prediction events (audit view; newest last)."""
        store = get_monitoring_store()
        events, warnings = store.read_prediction_events()
        if state:
            events = [e for e in events if e.get("event_state") == state]
        events = events[-max(1, min(limit, 500)) :]
        return {
            "count": len(events),
            "store_warnings": [w.message for w in warnings][:20],
            "events": events,
        }

    @app.get("/monitoring/feedback")
    def monitoring_feedback(limit: int = 50) -> dict:
        """Recent feedback/link records with errors (audit view)."""
        store = get_monitoring_store()
        feedback, warnings = store.read_feedback_records()
        feedback = feedback[-max(1, min(limit, 500)) :]
        return {
            "count": len(feedback),
            "store_warnings": [w.message for w in warnings][:20],
            "feedback": feedback,
        }

    # -------------------------------------------------- notifications (AD-33)
    @app.get("/notifications")
    def notifications_list() -> dict:
        """Persisted notification feed (newest first) + unread count."""
        return notifications.list_notifications(get_monitoring_store())

    @app.post("/notifications/evaluate")
    def notifications_evaluate() -> dict:
        """Evaluate system-level signals into the notification feed (§24).

        Reuses the SAME drift/readiness computations the dashboards run — no
        new thresholds. Idempotent per open notification key.
        """
        from dltime.model_lab import feature_drift, population_drift
        from dltime.model_lab import retraining_readiness as _rr

        store = get_monitoring_store()
        outcomes, _o = store.read_actual_outcomes()
        feedback, _f = store.read_feedback_records()
        matched = [f for f in feedback if f.get("linkage", {}).get("status") == "MATCHED"]
        lab_ctx = get_lab_context()
        fd = feature_drift(
            lab_ctx["reference_frame"],
            lab_ctx["current_frame"],
            feature_names=lab_ctx["feature_names"],
            categorical_features=lab_ctx["categorical_features"],
        )
        pd_ = population_drift(lab_ctx["reference_frame"], lab_ctx["current_frame"])
        readiness = _rr(
            registry=get_model_registry().models,
            matched_feedback=matched,
            feature_drift_report={"overall_status": fd["overall_status"]},
            population_drift_report={"overall_status": pd_["overall_status"]},
            eligible_actual_count=sum(1 for o in outcomes if o.get("training_eligible")),
        )
        records = notifications.evaluate_system_notifications(
            drift_feature_status=fd["overall_status"],
            drift_population_status=pd_["overall_status"],
            readiness=readiness,
            registry=get_model_registry(),
        )
        fresh = notifications.append_notifications(store, records)
        return {
            **notifications.list_notifications(store),
            "evaluated_alerts": len(records),
            "new_notifications": len(fresh),
        }

    @app.post("/notifications/{notification_id}/read")
    def notifications_mark_read(notification_id: str) -> dict:
        return notifications.mark_notification_read(get_monitoring_store(), notification_id)

    @app.post("/notifications/read-all")
    def notifications_mark_all_read() -> dict:
        return notifications.mark_all_notifications_read(get_monitoring_store())

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @lru_cache(maxsize=1)
    def _overview_static() -> dict:
        """Static-per-process Overview facts (sources + model artifact).

        Presentation only: every number is derived from the actual source
        reconstruction / eligibility / model artifact / monitoring store.
        No fabricated metrics; absent evidence is returned as null and the
        UI must render truthful empty states.
        """
        report = build_foundation(bundle=_load_bundle())
        eligible = report.frames["executions_eligible"]
        # Sum-total policy (EXP-038): the authoritative serving identity is
        # the deterministic five-stage sum; the stage-aware total model is
        # retired (reference/traceability evidence only).
        from dltime.sum_total_calibration import load_sum_total_calibration
        from dltime.sum_total_serving import SUM_TOTAL_ESTIMATOR, SUM_TOTAL_MODEL_VERSION

        from dltime.uncertainty import CALIBRATION_VERSION

        calib_rows = 0
        calib_version = CALIBRATION_VERSION
        try:
            calib_meta = load_sum_total_calibration()
            calib_rows = int(calib_meta.get("n_calibration_rows") or 0)
            calib_version = calib_meta.get("calibration_version", CALIBRATION_VERSION)
        except Exception:
            calib_rows = 0
        meta = {
            "model_version": SUM_TOTAL_MODEL_VERSION,
            "model_status": "DETERMINISTIC_POLICY",
            "feature_set_version": (
                "five operational stage predictions (each its own stage-specific "
                "feature pipeline); total = exact sum"
            ),
            "role": "AUTHORITATIVE_TOTAL",
            "estimator": SUM_TOTAL_ESTIMATOR,
        }
        counts = {str(k): int(v) for k, v in dict(report.eligibility.get("counts", {})).items()}
        return {
            "data": {
                "total_loaders": int(report.frames["executions"]["Loader_Name"].nunique()),
                "total_executions": int(len(report.frames["executions"])),
                "training_eligible": counts.get(
                    "training_eligible", int(len(eligible))
                ),
                # EXP-032/AD-34 (EXP-033): authoritative duration statistics.
                # Value source: total_timetaken_in_min*60 (span fallback) — the
                # legacy timetaken_in_sec is corrupted and never surfaces here.
                "median_duration_sec": report.target["median"],
                "p95_duration_sec": report.target["p95"],
                "duration_stat_definition": report.target.get("definition"),
                "duration_target_source_counts": report.target.get("target_source_counts"),
                "duration_span_consistency": report.target.get("span_consistency"),
            },
            "quality": {
                # Real reconstruction/eligibility facts (Part D/F of the
                # foundation) — shown truthfully on the Data & Sources page.
                "resolution": {
                    str(k): int(v)
                    for k, v in dict(report.resolution_executions).items()
                },
                "eligibility_counts": {
                    str(k): int(v)
                    for k, v in dict(report.eligibility.get("counts", {})).items()
                },
            },
            "sources": [
                {
                    # Real per-file provenance recorded at load time (the
                    # sha256 digest is recorded but intentionally not surfaced
                    # in the product UI; row counts are shown per source).
                    "name": str((prov or {}).get("file") or name),
                    "role": str((prov or {}).get("role") or ""),
                    "rows": int((prov or {}).get("rows", 0) or 0),
                    "columns": int((prov or {}).get("columns", 0) or 0),
                }
                for name, prov in report.sources.items()
            ],
            "model": {
                "model_version": meta["model_version"],
                "model_status": meta.get("model_status") or meta.get("status"),
                "feature_version": meta.get("feature_set_version"),
                "feature_count": 5,
                "serving_role": meta.get("role", "AUTHORITATIVE_TOTAL"),
                "estimator": meta.get("estimator"),
                "calibration": {
                    "version": calib_version,
                    # Measured held-out coverage lives in the experiment log
                    # (EXP-016/EXP-030 evaluation), NOT in the machine-readable
                    # artifact (which stores conformal targets only). Coverage
                    # is exposed as null; the UI shows calibration rows.
                    "coverage_90": None,
                    "rows": calib_rows,
                },
            },
        }

    @lru_cache(maxsize=1)
    def _overview_drift_feedback(feedback_count: int, matched_count: int) -> dict:
        """Frame-based drift statuses for a given feedback activity level.

        Drift over the real reference/current frames is the expensive part;
        cache it per activity level so idle dashboard refreshes stay cheap
        while newly linked outcomes force a fresh evaluation.
        """
        from dltime.model_lab import feature_drift, population_drift

        lab_ctx = get_lab_context()
        fd = feature_drift(
            lab_ctx["reference_frame"],
            lab_ctx["current_frame"],
            feature_names=lab_ctx["feature_names"],
            categorical_features=lab_ctx["categorical_features"],
        )
        pd_ = population_drift(lab_ctx["reference_frame"], lab_ctx["current_frame"])
        # Performance drift is outcome-gated, so it is computed fresh by the
        # caller from the live matched feedback; frame drift is cached here.
        return {
            "feature_status": fd["overall_status"],
            "population_status": pd_["overall_status"],
            "performance_status": "NOT_ASSESSED",
        }

    @app.get("/overview")
    def overview() -> dict:
        """Overview dashboard facts (real data; no fabricated metrics)."""
        static = _overview_static()
        store = get_monitoring_store()
        events, _ = store.read_prediction_events()
        feedback, _ = store.read_feedback_records()
        matched = [f for f in feedback if f.get("linkage", {}).get("status") == "MATCHED"]
        outcomes, _w = store.read_actual_outcomes()
        from dltime.model_lab import prediction_performance_drift, retraining_readiness

        # Drift evaluation is the expensive part; reuse the cached frames for
        # the current feedback activity level (fresh feedback list each request).
        drift = _overview_drift_feedback(len(feedback), len(matched))
        readiness = retraining_readiness(
            registry=get_model_registry().models,
            matched_feedback=matched,
            feature_drift_report={"overall_status": drift["feature_status"]},
            population_drift_report={"overall_status": drift["population_status"]},
            eligible_actual_count=sum(1 for o in outcomes if o.get("training_eligible")),
        )
        perf = prediction_performance_drift(matched)
        drift = {**drift, "performance_status": perf["status"]}
        recent: list[dict[str, Any]] = []
        for ev in reversed(events[-8:]):
            pred = ev.get("prediction") or None
            recent.append(
                {
                    "created_at": ev.get("created_at"),
                    "event_state": ev.get("event_state"),
                    "loader_name": (ev.get("identity") or {}).get("loader_name"),
                    "dataset": (ev.get("identity") or {}).get("dataset"),
                    "sprint": (ev.get("identity") or {}).get("sprint"),
                    "predicted_seconds": (pred or {}).get("predicted_total_seconds"),
                    "risk_label": (pred or {}).get("risk_label"),
                    "workflow_execution_id": (ev.get("identity") or {}).get(
                        "ldr_workflow_execution_id"
                    ),
                }
            )
        return {
            "generated_at": utc_now_iso(),
            "data": static["data"],
            "quality": static["quality"],
            "sources": static["sources"],
            "model": static["model"],
            "monitoring": {
                "prediction_attempts": len(events),
                "state_counts": {
                    s: sum(1 for e in events if e.get("event_state") == s)
                    for s in sorted({str(e.get("event_state")) for e in events})
                },
                "actual_outcomes": len(outcomes),
                "matched_outcomes": len(matched),
            },
            "retraining": {
                "readiness_state": readiness["readiness_state"],
                "automatic_retraining": readiness["automatic_retraining"],
            },
            "drift": drift,
            "recent_events": recent,
        }

    # Module 7 (Prompt 4): Model Laboratory / model registry / drift /
    # retraining readiness — read-only inspection plus explicitly human-gated
    # promote/rollback/reject actions (api_lab.py).
    from dltime.api_lab import wire_model_lab

    wire_model_lab(app, get_registry=get_model_registry, get_lab_context=get_lab_context)

    return app


app = create_app()
