"""Model Laboratory + model lifecycle (Module 7 / Prompt 4; AD-28).

Governance layer ONLY: every state change (promotion, rollback, retraining
decision) is a HUMAN-GATED explicit action. Nothing here retrains
automatically from predictions, and nothing here promotes a candidate
automatically — the locked Task 04/13 distinction between an evaluated
candidate and a production model is enforced by state, not by convention.

Layers (all local, JSONL/JSON persistence under ``backend/artifacts/registry/``):
  - ExperimentRegistry  append-only experiment records (EXP-0xx lineage)
  - ModelRegistry       model versions + lifecycle states + promotion history
  - drift               PSI / KS / categorical-proportion monitoring with an
                        explicit reference baseline and small-sample guard
  - retraining readiness assessment (signals only; human must approve)

Compatibility gates before promotion (§16): feature-registry, preprocessing,
inference, uncertainty-calibration, and SHAP compatibility are verified
against the LIVE artifact contract; a candidate that cannot serve with the
existing uncertainty/explanation layers is refused.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dltime.config import PROJECT_ROOT

REGISTRY_DIR: Path = PROJECT_ROOT / "backend" / "artifacts" / "registry"
EXPERIMENTS_FILE = "experiments.jsonl"
MODELS_FILE = "models.jsonl"
EVENTS_FILE = "lifecycle_events.jsonl"
REGISTRY_SCHEMA_VERSION = "t13-v1"

LAB_VERSION = "model-lab-v1"

# ---------------------------------------------------------------- lifecycle


class ModelState:
    EXPERIMENTAL = "EXPERIMENTAL"
    CANDIDATE_EVALUATED = "CANDIDATE_EVALUATED"
    PROMOTED = "PROMOTED"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


ALL_MODEL_STATES = {
    ModelState.EXPERIMENTAL,
    ModelState.CANDIDATE_EVALUATED,
    ModelState.PROMOTED,
    ModelState.RETIRED,
    ModelState.REJECTED,
}

# Legal transitions (§18: no contradictory states; single promoted model).
LEGAL_TRANSITIONS: dict[str, set[str]] = {
    ModelState.EXPERIMENTAL: {ModelState.CANDIDATE_EVALUATED, ModelState.REJECTED, ModelState.RETIRED},
    ModelState.CANDIDATE_EVALUATED: {ModelState.PROMOTED, ModelState.REJECTED, ModelState.RETIRED},
    ModelState.PROMOTED: {ModelState.RETIRED},  # demoted only via a replacement promotion or rollback
    ModelState.RETIRED: {ModelState.PROMOTED},  # rollback can reactivate a retired previous model
    ModelState.REJECTED: set(),
}


class LabError(RuntimeError):
    """Model Laboratory governance failure (refused action)."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


# ---------------------------------------------------------------- persisted stores


class _JsonlStore:
    """Minimal append-only JSONL store (same conventions as monitoring store)."""

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = Path(directory) if directory is not None else REGISTRY_DIR
        self._lock = threading.Lock()

    def _path(self, name: str) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir / name

    def append(self, name: str, record: dict[str, Any]) -> None:
        line = _canonical(record)
        with self._lock:
            with self._path(name).open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()

    def read(self, name: str) -> list[dict[str, Any]]:
        path = self._path(name)
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # fail-closed on read: corrupt lines are skipped
                if isinstance(rec, dict) and rec.get("schema_version") == REGISTRY_SCHEMA_VERSION:
                    out.append(rec)
        return out

    def rewrite(self, name: str, records: list[dict[str, Any]]) -> None:
        """Rewrite used ONLY for the mutable model-registry head file (the
        experiment/event files stay strictly append-only)."""
        with self._lock:
            path = self._path(name)
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(_canonical(rec) + "\n")
            tmp.replace(path)


class ExperimentRegistry:
    """Append-only experiment record store (never overwritten)."""

    REQUIRED_FIELDS = (
        "experiment_id",
        "created_at",
        "model_family",
        "target_transformation",
        "hyperparameters",
        "feature_set_version",
        "feature_names",
        "training_population",
        "validation_method",
        "metrics",
        "status",
    )

    def __init__(self, store: _JsonlStore | None = None) -> None:
        self.store = store or _JsonlStore()

    def register(self, record: dict[str, Any]) -> dict[str, Any]:
        rec = dict(record)
        rec.setdefault("schema_version", REGISTRY_SCHEMA_VERSION)
        rec.setdefault("created_at", utc_now_iso())
        rec.setdefault("status", "EVALUATED")
        rec.setdefault("decision", None)
        rec.setdefault("rationale", None)
        missing = [f for f in self.REQUIRED_FIELDS if rec.get(f) is None]
        if missing:
            raise LabError(f"Experiment record missing required fields: {missing}")
        if any(e.get("experiment_id") == rec["experiment_id"] for e in self.list()):
            raise LabError(f"Experiment '{rec['experiment_id']}' already registered (append-only).")
        self.store.append(EXPERIMENTS_FILE, rec)
        return rec

    def list(self) -> list[dict[str, Any]]:
        return self.store.read(EXPERIMENTS_FILE)

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        for rec in self.list():
            if rec.get("experiment_id") == experiment_id:
                return rec
        return None


@dataclass
class ModelRecord:
    model_id: str
    model_family: str
    artifact_path: str
    status: str
    feature_set_version: str
    feature_count: int
    target_mode: str
    training_population: dict[str, Any]
    validation_population: dict[str, Any]
    experiment_id: str | None = None
    parent_model_id: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    promoted_at: str | None = None
    retired_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            **self.__dict__.copy(),
        }


class ModelRegistry:
    """Model versions + lifecycle + exactly one promoted (active) model.

    A JSON head file carries the current records; every state change also
    appends an immutable lifecycle event so the full history survives
    rewrites (§20 lineage).
    """

    def __init__(self, store: _JsonlStore | None = None) -> None:
        self.store = store or _JsonlStore()
        self.events = LifecycleLog(self.store)

    # ------------------------------------------------------------- primitives
    def _read_models(self) -> list[dict[str, Any]]:
        return self.store.read(MODELS_FILE)

    def list(self) -> list[dict[str, Any]]:
        return self._read_models()

    def get(self, model_id: str) -> dict[str, Any] | None:
        for rec in self._read_models():
            if rec.get("model_id") == model_id:
                return rec
        return None

    def promoted_model(self) -> dict[str, Any] | None:
        """The single promoted (active) model, if any."""
        promoted = [r for r in self._read_models() if r.get("status") == ModelState.PROMOTED]
        if len(promoted) > 1:  # impossible by construction; defensive
            raise LabError("Registry invariant violated: more than one PROMOTED model.")
        return promoted[0] if promoted else None

    def register(self, record: ModelRecord | dict[str, Any]) -> dict[str, Any]:
        rec = record.to_dict() if isinstance(record, ModelRecord) else dict(record)
        rec.setdefault("schema_version", REGISTRY_SCHEMA_VERSION)
        if rec.get("status") not in ALL_MODEL_STATES:
            raise LabError(f"Invalid model state {rec.get('status')!r}.")
        if not rec.get("artifact_path"):
            raise LabError("Model record must reference an artifact path.")
        models = self._read_models()
        if any(m.get("model_id") == rec["model_id"] for m in models):
            raise LabError(f"Model version '{rec['model_id']}' already registered (unique).")
        if rec.get("status") == ModelState.PROMOTED:
            raise LabError("register() cannot promote; use promote() (human-gated action).")
        models.append(rec)
        self.store.rewrite(MODELS_FILE, models)
        self.events.append(
            event_type="REGISTERED",
            model_id=rec["model_id"],
            payload={"status": rec.get("status"), "experiment_id": rec.get("experiment_id")},
        )
        return rec

    def _set_state(self, model_id: str, new_state: str, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        models = self._read_models()
        target = None
        for rec in models:
            if rec.get("model_id") == model_id:
                target = rec
                break
        if target is None:
            raise LabError(f"Unknown model '{model_id}'.")
        old_state = target.get("status")
        if new_state not in LEGAL_TRANSITIONS.get(old_state, set()):
            raise LabError(f"Illegal transition {old_state} -> {new_state} for '{model_id}'.")
        target["status"] = new_state
        if extra:
            target.update(extra)
        if new_state == ModelState.PROMOTED:
            target["promoted_at"] = utc_now_iso()
            target.pop("retired_at", None)
        if new_state == ModelState.RETIRED:
            target["retired_at"] = utc_now_iso()
        self.store.rewrite(MODELS_FILE, models)
        self.events.append(
            event_type=new_state,
            model_id=model_id,
            payload={"from": old_state, "to": new_state, **(extra or {})},
        )
        return target

    # ------------------------------------------------------- promotion (gated)
    def promote(
        self,
        model_id: str,
        *,
        compatibility: dict[str, Any] | None = None,
        approved_by: str,
        reason: str,
    ) -> dict[str, Any]:
        """Human-gated promotion. Refuses when gates fail; records everything."""
        rec = self.get(model_id)
        if rec is None:
            raise LabError(f"Unknown model '{model_id}'.")
        if rec.get("status") != ModelState.CANDIDATE_EVALUATED:
            raise LabError(
                f"Only CANDIDATE_EVALUATED models can be promoted; '{model_id}' is {rec.get('status')!r}."
            )
        comp = compatibility or {}
        failures = [k for k, v in comp.items() if v is False]
        if failures:
            raise LabError(f"Promotion blocked: compatibility gate(s) failed: {failures}")
        if not Path(rec.get("artifact_path", "")).exists():
            raise LabError(f"Promotion blocked: artifact missing at {rec.get('artifact_path')!r}.")
        previous = self.promoted_model()
        models = self._read_models()
        if previous is not None and previous.get("model_id") != model_id:
            for m in models:
                if m.get("model_id") == previous["model_id"]:
                    m["status"] = ModelState.RETIRED
                    m["retired_at"] = utc_now_iso()
                    self.events.append(
                        event_type=ModelState.RETIRED,
                        model_id=m["model_id"],
                        payload={"from": ModelState.PROMOTED, "to": ModelState.RETIRED, "reason": "superseded by promotion"},
                    )
            self.store.rewrite(MODELS_FILE, models)  # persist retirement before promoting the new head
        target = self._set_state(
            model_id,
            ModelState.PROMOTED,
            extra={
                "promotion": {
                    "previous_model_id": previous.get("model_id") if previous else None,
                    "approved_by": approved_by,
                    "reason": reason,
                    "compatibility": comp,
                    "at": utc_now_iso(),
                }
            },
        )
        self.events.append(
            event_type="PROMOTION_APPROVED",
            model_id=model_id,
            payload={"previous": previous.get("model_id") if previous else None, "approved_by": approved_by, "reason": reason},
        )
        return target

    def reject(self, model_id: str, *, reason: str, approved_by: str = "local-operator") -> dict[str, Any]:
        rec = self.get(model_id)
        if rec is None:
            raise LabError(f"Unknown model '{model_id}'.")
        return self._set_state(model_id, ModelState.REJECTED, extra={"reject_reason": reason, "rejected_by": approved_by})

    # ---------------------------------------------------------- rollback (gated)
    def rollback(self, *, approved_by: str, reason: str) -> dict[str, Any]:
        """Restore the most recent previously-promoted, still-present model."""
        current = self.promoted_model()
        if current is None:
            raise LabError("No promoted model to roll back from.")
        history = [
            e
            for e in self.events.list()
            if e.get("event_type") == "PROMOTION_APPROVED" and e.get("model_id") != current["model_id"]
        ]
        if not history:
            raise LabError("No previous promoted model found in lifecycle history.")
        prev_id = history[-1]["model_id"]
        prev = self.get(prev_id)
        if prev is None:
            raise LabError(f"Previous model '{prev_id}' is no longer registered.")
        artifact_path = Path(prev.get("artifact_path", ""))
        if not artifact_path.exists():
            raise LabError(f"Rollback blocked: previous artifact missing at {artifact_path!r}.")
        comp = prev.get("compatibility") or prev.get("promotion", {}).get("compatibility") or {}
        failures = [k for k, v in comp.items() if v is False]
        if failures:
            raise LabError(f"Rollback blocked: previous model fails compatibility gates {failures}.")
        # demote current, reactivate previous
        self.events.append(
            event_type=ModelState.RETIRED,
            model_id=current["model_id"],
            payload={"from": ModelState.PROMOTED, "to": ModelState.RETIRED, "reason": f"rollback: {reason}"},
        )
        models = self._read_models()
        for m in models:
            if m.get("model_id") == current["model_id"]:
                m["status"] = ModelState.RETIRED
                m["retired_at"] = utc_now_iso()
        self.store.rewrite(MODELS_FILE, models)
        restored = self._set_state(
            prev_id,
            ModelState.PROMOTED,
            extra={
                "promotion": {
                    "previous_model_id": current["model_id"],
                    "approved_by": approved_by,
                    "reason": f"ROLLBACK: {reason}",
                    "at": utc_now_iso(),
                }
            },
        )
        self.events.append(
            event_type="ROLLBACK",
            model_id=prev_id,
            payload={"from_model": current["model_id"], "approved_by": approved_by, "reason": reason},
        )
        return restored


class LifecycleLog:
    """Append-only governance event log (promotion/rollback/registration)."""

    def __init__(self, store: _JsonlStore) -> None:
        self.store = store

    def append(self, *, event_type: str, model_id: str, payload: dict[str, Any] | None = None) -> None:
        self.store.append(
            EVENTS_FILE,
            {
                "schema_version": REGISTRY_SCHEMA_VERSION,
                "event_type": event_type,
                "model_id": model_id,
                "payload": payload or {},
                "at": utc_now_iso(),
            },
        )

    def list(self) -> list[dict[str, Any]]:
        return self.store.read(EVENTS_FILE)


# ---------------------------------------------------------------- compatibility


def compatibility_report(
    artifact_path: str | Path,
    *,
    # 2026-10-04 44-feature contract (22 def + 22 hist). Callers may still pin
    # a narrower serving contract explicitly, but the default must match what
    # build_training_frame() / build_serving_artifact() actually produce.
    expected_feature_count: int = 44,
    expected_calibration_version: str = "unc-log1p-conformal-v1",
) -> dict[str, Any]:
    """§16 promotion-safety verification against the LIVE serving contract.

    Every check returns True/False (never raises) so a promotion refusal can
    list all failing gates. Forbidden/post-execution features are refused.
    """
    import joblib

    from dltime.feature_assembly import sync_engineered_fields
    from dltime.leakage import validate_feature_set
    from dltime.uncertainty import load_calibration_artifact

    path = Path(artifact_path)
    report: dict[str, Any] = {
        "artifact_exists": path.exists(),
        "feature_registry_compatible": False,
        "preprocessing_compatible": False,
        "inference_compatible": False,
        "uncertainty_calibration_compatible": False,
        "explanation_compatible": False,
        "expected_feature_count": expected_feature_count,
        "expected_model_version_ok": False,
        "no_forbidden_feature": False,
        "no_post_execution_feature": False,
        "forbidden_found": [],
        "feature_count": None,
        "model_version": None,
        "checks_detail": {},
    }
    if not path.exists():
        return report
    try:
        art = joblib.load(path)
        meta = art.get("metadata", {})
        features = list(meta.get("feature_names_in_order", []))
        report["feature_count"] = len(features)
        report["model_version"] = meta.get("model_version")
        report["expected_model_version_ok"] = bool(
            meta.get("model_version") and meta.get("status") in ("CANDIDATE_EVALUATED", "EXPERIMENTAL")
        )

        # feature registry + leakage boundary (fresh registry state for engineered names)
        sync_engineered_fields()
        try:
            validate_feature_set(features, context="promotion-gate")
            report["feature_registry_compatible"] = True
        except Exception as exc:
            report["checks_detail"]["feature_registry"] = str(exc)

        banned_tokens = ("timetaken", "time_taken", "end_time", "ldr_status", "workflow_execution_id",
                         "ldr_execution_id", "actual", "residual", "error_records", "total_records")
        forbidden = [f for f in features if any(tok in str(f).lower() for tok in banned_tokens)]
        report["forbidden_found"] = forbidden
        report["no_forbidden_feature"] = not forbidden
        report["no_post_execution_feature"] = not forbidden

        # inference: exact feature-order check + fitted preprocessor presence
        pre = art.get("preprocessor")
        names_in = getattr(pre, "feature_names_in_", None)
        pre_names = [str(x) for x in names_in] if names_in is not None else []
        report["preprocessing_compatible"] = bool(
            pre is not None and pre_names == features and art.get("transformer") is not None
        )
        est = art.get("estimator")
        report["inference_compatible"] = bool(
            callable(getattr(est, "predict", None)) and len(features) == expected_feature_count
        )
        if len(features) != expected_feature_count:
            report["checks_detail"]["feature_count"] = (
                f"artifact carries {len(features)} features; serving contract expects {expected_feature_count}"
            )

        # uncertainty calibration compatibility (version + feature count)
        try:
            cal = load_calibration_artifact(
                meta.get("model_version"), len(features)
            )
            report["uncertainty_calibration_compatible"] = bool(cal)
            report["checks_detail"]["calibration_version"] = getattr(cal, "version", None)
        except Exception as exc:
            report["checks_detail"]["calibration"] = str(exc)

        # explanation compatibility: TreeExplainer can be constructed
        try:
            import shap

            explainer = shap.TreeExplainer(est)
            report["explanation_compatible"] = explainer is not None
        except Exception as exc:
            report["checks_detail"]["explanation"] = str(exc)
    except Exception as exc:  # corrupt artifact etc.
        report["checks_detail"]["load"] = str(exc)
    return report


# ---------------------------------------------------------------- drift (§21-25)

DRIFT_METHOD_NUMERIC = "KS"  # scipy.stats.ks_2samp — explainable, sample-size aware
DRIFT_METHOD_CATEGORICAL = "PROPORTION_SHIFT"
# Heuristic thresholds (documented as project-defined heuristics, NOT truths):
#   KS >= 0.15 warning / >= 0.30 significant (small-sample guarded below)
#   PSI  >= 0.10 warning / >= 0.25 significant (standard practice bins=10)
DRIFT_WARN_KS = 0.15
DRIFT_SIG_KS = 0.30
DRIFT_WARN_PSI = 0.10
DRIFT_SIG_PSI = 0.25
MIN_WINDOW_ROWS = 20  # below this: INSUFFICIENT_SAMPLE (never a drift verdict)
MIN_REF_ROWS = 30

# Candidate-evaluation bar (§13): the candidate must be at least comparable on
# the SAME chronological protocol before a human review is warranted — strictly
# comparable-or-better means ratio <= 1.0 counts as better; up to 1.10 is still
# sent to review as 'broadly comparable' with the tradeoff explained.
CANDIDATE_EVALUATION_MIN_RATIO = 1.10


def psi(expected: list[float], actual: list[float], bins: int = 10) -> float:
    """Population Stability Index over quantile-binned reference distribution."""
    import numpy as np

    e = np.asarray([v for v in expected if v is not None and np.isfinite(v)], dtype=float)
    a = np.asarray([v for v in actual if v is not None and np.isfinite(v)], dtype=float)
    if e.size == 0 or a.size == 0:
        return float("nan")
    qs = np.quantile(e, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    qs = np.unique(qs)
    if qs.size < 3:
        return 0.0  # degenerate reference (single value) — no bin structure
    e_counts = np.histogram(e, bins=qs)[0] / e.size
    a_counts = np.histogram(a, bins=qs)[0] / a.size
    eps = 1e-6
    val = float(np.sum((a_counts - e_counts) * np.log((a_counts + eps) / (e_counts + eps))))
    return val


def ks_statistic(expected: list[float], actual: list[float]) -> tuple[float, float]:
    """Two-sample KS statistic + p-value (scipy already in the dependency tree)."""
    import numpy as np
    from scipy.stats import ks_2samp

    e = np.asarray([v for v in expected if v is not None and np.isfinite(v)], dtype=float)
    a = np.asarray([v for v in actual if v is not None and np.isfinite(v)], dtype=float)
    if e.size == 0 or a.size == 0:
        return float("nan"), float("nan")
    res = ks_2samp(e, a)
    return float(res.statistic), float(res.pvalue)


def _status_from_statistic(stat: float, warn: float, sig: float) -> str:
    import math

    if stat is None or isinstance(stat, float) and math.isnan(stat):
        return "INSUFFICIENT_SAMPLE"
    if stat >= sig:
        return "SIGNIFICANT_DRIFT"
    if stat >= warn:
        return "WARNING"
    return "NO_MEANINGFUL_DRIFT"


def categorical_shift(expected: list[Any], actual: list[Any]) -> dict[str, Any]:
    """Max category-proportion change + total-variation distance."""
    import math
    from collections import Counter

    e = Counter(str(v) for v in expected if v is not None)
    a = Counter(str(v) for v in actual if v is not None)
    if not e or not a:
        return {"statistic": float("nan"), "max_proportion_change": float("nan"), "status": "INSUFFICIENT_SAMPLE"}
    keys = set(e) | set(a)
    te, ta = sum(e.values()), sum(a.values())
    max_chg = max(abs(a.get(k, 0) / ta - e.get(k, 0) / te) for k in keys)
    tv = 0.5 * sum(abs(a.get(k, 0) / ta - e.get(k, 0) / te) for k in keys)
    stat = float(tv)
    status = (
        "SIGNIFICANT_DRIFT" if stat >= 0.30
        else "WARNING" if stat >= 0.15
        else "NO_MEANINGFUL_DRIFT"
    )
    if any(isinstance(x, float) and math.isnan(x) for x in (stat,)):
        status = "INSUFFICIENT_SAMPLE"
    return {"statistic": round(stat, 4), "max_proportion_change": round(max_chg, 4), "status": status, "method": DRIFT_METHOD_CATEGORICAL}


def _status_and_details(stat: float, method: str) -> dict[str, Any]:
    if method == DRIFT_METHOD_NUMERIC:
        status = _status_from_statistic(stat, DRIFT_WARN_KS, DRIFT_SIG_KS)
    else:
        status = _status_from_statistic(stat, DRIFT_WARN_PSI, DRIFT_SIG_PSI)
    return {"status": status}


def feature_drift(
    reference_frame,
    current_frame,
    *,
    feature_names: list[str] | None = None,
    categorical_features: list[str] | None = None,
    max_features_reported: int = 12,
) -> dict[str, Any]:
    """A. Feature/data drift for pre-execution features (§21-A, §22-25)."""
    import numpy as np
    import pandas as pd

    cat = set(categorical_features or [])
    if feature_names is None:
        feature_names = [c for c in reference_frame.columns if c in current_frame.columns]
    n_ref, n_cur = len(reference_frame), len(current_frame)
    out: dict[str, Any] = {
        "method_numeric": DRIFT_METHOD_NUMERIC,
        "method_categorical": DRIFT_METHOD_CATEGORICAL,
        "reference_rows": int(n_ref),
        "current_rows": int(n_cur),
        "min_window_rows": MIN_WINDOW_ROWS,
        "min_reference_rows": MIN_REF_ROWS,
        "thresholds": {
            "ks_warning": DRIFT_WARN_KS,
            "ks_significant": DRIFT_SIG_KS,
            "note": "project-defined heuristics, not universal truths",
        },
        "features": {},
        "overall_status": "NO_MEANINGFUL_DRIFT",
    }
    if n_cur < MIN_WINDOW_ROWS or n_ref < MIN_REF_ROWS:
        out["overall_status"] = "INSUFFICIENT_SAMPLE"
        out["note"] = (
            f"Monitoring window has {n_cur} rows (minimum {MIN_WINDOW_ROWS}); "
            f"reference has {n_ref} (minimum {MIN_REF_ROWS}). No drift verdict is issued."
        )
        return out

    worst = "NO_MEANINGFUL_DRIFT"
    ranking: list[tuple[float, str]] = []
    for name in feature_names[:]:
        if name not in reference_frame.columns or name not in current_frame.columns:
            continue
        ref = reference_frame[name]
        cur = current_frame[name]
        if name in cat:
            res = categorical_shift(list(ref.dropna()), list(cur.dropna()))
        else:
            stat, pval = ks_statistic(list(pd.to_numeric(ref, errors="coerce")), list(pd.to_numeric(cur, errors="coerce")))
            res = {
                "statistic": None if stat != stat else round(stat, 4),
                "p_value": None if pval != pval else round(pval, 6),
                "method": DRIFT_METHOD_NUMERIC,
                **_status_and_details(stat, DRIFT_METHOD_NUMERIC),
            }
            if stat == stat:
                ranking.append((stat, name))
        out["features"][name] = res
        order = {"NO_MEANINGFUL_DRIFT": 0, "WARNING": 1, "SIGNIFICANT_DRIFT": 2, "INSUFFICIENT_SAMPLE": 0}
        if order.get(res.get("status", ""), 0) > order.get(worst, 0):
            worst = res["status"]
    ranking.sort(reverse=True)
    out["top_shifted_features"] = [n for _s, n in ranking[:max_features_reported]]
    out["overall_status"] = worst
    return out


def population_drift(reference_frame, current_frame) -> dict[str, Any]:
    """B. Population drift: composition of loaders/datasets/connections/history."""
    import pandas as pd

    out: dict[str, Any] = {"dimensions": {}}
    if len(current_frame) < MIN_WINDOW_ROWS or len(reference_frame) < MIN_REF_ROWS:
        out["overall_status"] = "INSUFFICIENT_SAMPLE"
        return out
    dims: dict[str, tuple[str, bool]] = {
        "loader": ("Loader_Name", True),
        "dataset": ("Dataset_display_name", True),
        "sprint": ("Sprint", True),
        "connection": ("resolved_connection", True),
    }
    worst = "NO_MEANINGFUL_DRIFT"
    for name, (col, _c) in dims.items():
        if col not in reference_frame.columns or col not in current_frame.columns:
            continue
        res = categorical_shift(
            list(reference_frame[col].astype(str)), list(current_frame[col].astype(str))
        )
        out["dimensions"][name] = res
        if res["status"] in ("WARNING", "SIGNIFICANT_DRIFT"):
            worst = "WARNING" if worst == "NO_MEANINGFUL_DRIFT" else worst
            if res["status"] == "SIGNIFICANT_DRIFT":
                worst = "SIGNIFICANT_DRIFT"
    # history availability shift (numeric prior count)
    hist_cols = [c for c in ("hist_loader_prior_count",) if c in reference_frame.columns and c in current_frame.columns]
    for c in hist_cols:
        stat, _p = ks_statistic(
            list(pd.to_numeric(reference_frame[c], errors="coerce")),
            list(pd.to_numeric(current_frame[c], errors="coerce")),
        )
        out["dimensions"]["history_availability"] = {
            "statistic": None if stat != stat else round(stat, 4),
            "method": DRIFT_METHOD_NUMERIC,
            **_status_and_details(stat, DRIFT_METHOD_NUMERIC),
        }
        if out["dimensions"]["history_availability"]["status"] in ("WARNING", "SIGNIFICANT_DRIFT"):
            if worst == "NO_MEANINGFUL_DRIFT":
                worst = "WARNING"
    out["overall_status"] = worst
    return out


def prediction_performance_drift(
    matched_feedback: list[dict[str, Any]],
    *,
    min_n: int = MIN_WINDOW_ROWS,
) -> dict[str, Any]:
    """C. Prediction-performance drift from matched feedback (§21-C).

    Requires linked actual outcomes; without them, no drift is claimed.
    """
    errs = [f["total_error"]["absolute_error_seconds"] for f in matched_feedback if f.get("total_error")]
    signed = [f["total_error"]["signed_error_seconds"] for f in matched_feedback if f.get("total_error")]
    pcts = [
        f["total_error"]["absolute_percentage_error"]
        for f in matched_feedback
        if f.get("total_error") and f["total_error"].get("absolute_percentage_error") is not None
    ]
    out: dict[str, Any] = {
        "sample_count": len(errs),
        "min_required": min_n,
        "has_actual_outcomes": bool(errs),
        "available": len(errs) >= min_n,
        # Zero matched outcomes is a distinct state from a merely small window
        # (Module 8 §13): no actuals exist at all, so performance drift is not
        # merely under-sampled — it is unassessable until outcomes are linked.
        "status": "INSUFFICIENT_OUTCOMES" if not errs else "INSUFFICIENT_SAMPLE",
    }
    if not errs:
        out["note"] = "No linked actual outcomes yet — prediction-performance drift cannot be assessed."
        return out
    import numpy as np

    def med(v):
        s = sorted(v)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0

    out.update(
        {
            "mae_seconds": round(sum(errs) / len(errs), 4),
            "median_absolute_error_seconds": round(med(errs), 4),
            "rmse_seconds": round(float(np.sqrt(np.mean(np.square(errs)))), 4),
            "mean_signed_error_seconds": round(sum(signed) / len(signed), 4),
            "mape_percent": round(sum(pcts) / len(pcts), 4) if pcts else None,
        }
    )
    if len(errs) >= min_n:
        out["status"] = "MONITORED"
        out["note"] = (
            "Operational error summary for the current window. Compare against the "
            "chronological evaluation baseline before drawing conclusions."
        )
    return out


# ------------------------------------------------------- stage prediction distributions


def stage_prediction_distributions(
    prediction_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Stage-prediction distribution summary across persisted events (AD-33 §34).

    Monitors the SERVED stage estimates (five ML stages + two statistical
    approvals) as recorded in prediction events. This is a distribution
    summary, not a drift verdict: with no reference window for stages there
    is deliberately no drift claim — the view exposes the observed spread so
    a human can compare against stage chronological baselines.
    """
    per_stage: dict[str, list[float]] = {}
    method_by_stage: dict[str, str] = {}
    for ev in prediction_events:
        si = (
            ev.get("stage_intelligence")
            or (ev.get("prediction") or {}).get("stage_intelligence")
            or {}
        )
        for s in si.get("stages") or []:
            stage = s.get("stage")
            val = s.get("estimate_seconds")
            if stage is None or val is None:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            per_stage.setdefault(str(stage), []).append(v)
            if s.get("method"):
                method_by_stage.setdefault(str(stage), str(s.get("method")))

    def _summary(vals: list[float]) -> dict[str, Any] | None:
        if not vals:
            return None
        srt = sorted(vals)
        n = len(srt)
        med = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2.0
        return {
            "n": n,
            "mean_seconds": round(sum(vals) / n, 4),
            "median_seconds": round(med, 4),
            "min_seconds": round(srt[0], 4),
            "max_seconds": round(srt[-1], 4),
        }

    return {
        "event_count": len(prediction_events),
        "stages": {
            stage: {**(summary or {}), "method": method_by_stage.get(stage)}
            for stage, vals in sorted(per_stage.items())
            if (summary := _summary(vals)) is not None
        },
        "note": (
            "Distribution of served stage estimates across prediction events. "
            "Summary only — no stage-level drift verdict without a reference window."
        ),
    }


# ------------------------------------------------------- retraining readiness (§26)


def retraining_readiness(
    *,
    registry: ModelRegistry,
    matched_feedback: list[dict[str, Any]],
    feature_drift_report: dict[str, Any] | None = None,
    population_drift_report: dict[str, Any] | None = None,
    eligible_actual_count: int = 0,
    min_new_eligible: int = 30,
) -> dict[str, Any]:
    """Structured readiness state — SIGNALS ONLY, never launches training.

    States: NOT_READY / READY_FOR_REVIEW / RETRAINING_RECOMMENDED_FOR_REVIEW.
    A human must explicitly approve any retraining; nothing here trains.
    """
    active = registry.promoted_model()
    signals: dict[str, Any] = {
        "eligible_new_actuals": {
            "count": int(eligible_actual_count),
            "threshold": min_new_eligible,
            "met": eligible_actual_count >= min_new_eligible,
        },
        "performance_degradation": {"assessed": False, "met": False},
        "feature_drift": {"status": (feature_drift_report or {}).get("overall_status", "NOT_ASSESSED")},
        "population_drift": {"status": (population_drift_report or {}).get("overall_status", "NOT_ASSESSED")},
    }
    n = len([f for f in matched_feedback if f.get("total_error")])
    if n >= MIN_WINDOW_ROWS and active:
        # degradation = operational MedAE clearly above the chronological baseline
        errs = sorted(f["total_error"]["absolute_error_seconds"] for f in matched_feedback if f.get("total_error"))
        med = errs[n // 2]
        baseline = ((active.get("metrics") or {}).get("served_population") or {}).get("median_ae")
        if baseline is None:
            baseline = ((active.get("metrics") or {}).get("holdout") or {}).get("median_ae")
        if baseline is not None:
            signals["performance_degradation"] = {
                "assessed": True,
                "met": bool(med > 2.0 * float(baseline)),
                "window_medae": med,
                "baseline_medae": baseline,
                "note": "heuristic 2x baseline MedAE flag; a human decides, not the signal",
            }
    sig_drift = (
        signals["feature_drift"]["status"] == "SIGNIFICANT_DRIFT"
        or signals["population_drift"]["status"] == "SIGNIFICANT_DRIFT"
    )
    if signals["performance_degradation"].get("met") or sig_drift:
        state = "RETRAINING_RECOMMENDED_FOR_REVIEW"
    elif signals["eligible_new_actuals"]["met"]:
        state = "READY_FOR_REVIEW"
    else:
        state = "NOT_READY"
    return {
        "readiness_state": state,
        "human_gate_required": True,
        "automatic_retraining": "OFF",
        "signals": signals,
        "note": (
            "Signals only. No training is launched by this system; a human must "
            "explicitly approve any retraining (governance contract, AD-28)."
        ),
    }


# ------------------------------------------------- composed drift overview (§30)

# Human-readable feature-group names for the locked drift view. Raw engineered
# feature names stay available behind the details toggle, never as the primary
# presentation (locked plan §6/§28).
_FEATURE_GROUP_LABELS: list[tuple[str, tuple[str, ...]]] = [
    # Specific groups first — first match wins.
    ("Execution Order", ("execution_order",)),
    ("Prevalidation Count", ("prevalidation_count", "preval_count")),
    ("Transformation Count", ("transformation_count", "transform_count")),
    ("Historical duration context", ("stage_hist_", "hist_", "prior_")),
    ("Loader configuration", ("preval_display_name", "transform_display_name", "type", "calling_type")),
]


def _feature_display_label(name: str) -> str:
    """Group raw engineered feature names into a small set of human labels.

    Deterministic; first matching group wins, otherwise a de-jargoned version
    of the raw name (underscores spaced, technical prefixes trimmed).
    """
    low = name.lower()
    for label, tokens in _FEATURE_GROUP_LABELS:
        if any(t in low for t in tokens):
            return label
    cleaned = name.replace("stage_hist_", "").replace("pred_stage_", "").replace("_sec", "")
    return cleaned.replace("_", " ").strip().title()


def _worst_status(statuses: list[str]) -> str:
    """Worst non-insufficient status; ALL-insufficient stays insufficient (no verdict)."""
    order = {"NO_MEANINGFUL_DRIFT": 0, "WARNING": 1, "SIGNIFICANT_DRIFT": 2}
    decisive = [s for s in statuses if s != "INSUFFICIENT_SAMPLE"]
    if not decisive:
        return "INSUFFICIENT_SAMPLE"
    worst = "NO_MEANINGFUL_DRIFT"
    for s in decisive:
        if order.get(s, 0) > order.get(worst, 0):
            worst = s
    return worst


def _model_health_from_status(status: str | None) -> str:
    if status == "SIGNIFICANT_DRIFT":
        return "DEGRADATION_DETECTED"
    if status == "WARNING":
        return "PERFORMANCE_CHANGING"
    if status in ("INSUFFICIENT_SAMPLE", "INSUFFICIENT_OUTCOMES"):
        return "INSUFFICIENT_ACTUAL_OUTCOMES"
    return "PERFORMING_NORMALLY"


def _data_health_from_status(status: str | None) -> str:
    if status == "SIGNIFICANT_DRIFT":
        return "DRIFT_DETECTED"
    if status == "WARNING":
        return "WATCH"
    if status == "INSUFFICIENT_SAMPLE":
        return "INSUFFICIENT_DATA"
    return "STABLE"


def _median(values: list[float]) -> float:
    """True median (mean of the two middle values for even n) — the
    same convention the monitoring metrics use (feedback.py)."""
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


# Locked plan §37: no stage-health claim without enough evidence. MIN_WINDOW_ROWS
# applies to the OUTCOME sample per stage (same policy constant as input drift).
def stage_model_health(
    matched_feedback: list[dict[str, Any]],
    *,
    selection_experiments: list[dict[str, Any]] | None = None,
    min_n: int = MIN_WINDOW_ROWS,
    serving_epochs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Five-stage model health from LINKED ACTUAL stage outcomes (§14-15).

    Baselines come ONLY from the persisted SELECTION experiment records
    (chronological-holdout MedAE per served stage kind) — never invented.
    A stage with fewer linked outcomes than the monitoring-policy minimum is
    'INSUFFICIENT_EVIDENCE' regardless of how its error compares.
    """
    stage_names = ["Staging", "Prevalidation", "Transformation", "Pre-load", "Data Loading"]
    # registry key -> (display name, SELECTION experiment id suffix)
    registry_keys: dict[str, tuple[str, str]] = {
        "staging": ("Staging", "STAGING"),
        "prevalidation": ("Prevalidation", "PREVALIDATION"),
        "transformation": ("Transformation", "TRANSFORMATION"),
        "pre_load": ("Pre-load", "PRE_LOAD"),
        "data_loading": ("Data Loading", "DATA_LOADING"),
    }
    # ---- baselines from the most recent SELECTION experiment per stage
    baseline_medae: dict[str, float] = {}
    for rec in selection_experiments or []:
        eid = str(rec.get("experiment_id", ""))
        for key, (_label, suffix) in registry_keys.items():
            if not eid.startswith("EXP-STAGE-" + suffix + "-") or not eid.endswith("SELECTION"):
                continue
            kind_metrics = rec.get("metrics") or {}
            vals = [m.get("median_ae") for m in kind_metrics.values() if isinstance(m, dict) and m.get("median_ae") is not None]
            if vals:
                baseline_medae[key] = _median(vals)  # conservative: median across kinds
    # ---- recent per-stage absolute errors from linked feedback
    # (timestamped copies power the live post-promotion error below)
    errs_by_stage: dict[str, list[float]] = {}
    timed_errs_by_stage: dict[str, list[tuple[str, float]]] = {}
    n_outcomes = 0
    for f in matched_feedback:
        se = f.get("stage_errors")
        if not se:
            continue
        n_outcomes += 1
        created = str(f.get("created_at") or "")
        for key, e in se.items():
            try:
                v = float((e or {}).get("absolute_error_seconds"))
            except (TypeError, ValueError):
                continue
            errs_by_stage.setdefault(str(key), []).append(v)
            if created:
                timed_errs_by_stage.setdefault(str(key), []).append((created, v))

    stages_out: dict[str, Any] = {}
    worst = "INSUFFICIENT_EVIDENCE"  # neutral floor: never upgraded without evidence
    severity = {"PERFORMING_NORMALLY": 0, "INSUFFICIENT_EVIDENCE": 0, "PERFORMANCE_CHANGING": 1, "DEGRADATION_DETECTED": 2}
    for key, (label, _suffix) in registry_keys.items():
        errs = sorted(errs_by_stage.get(key, []))
        n = len(errs)
        rec: dict[str, Any] = {
            "stage": label,
            "recent_outcomes": n,
            "min_required": min_n,
        }
        if n < min_n or key not in baseline_medae:
            rec["status"] = "INSUFFICIENT_EVIDENCE"
            rec["note"] = (
                "Not enough linked actual outcomes for this stage yet — no health "
                "claim is made (monitoring policy minimum is {n}).".format(n=min_n)
            )
        else:
            med = _median(errs)
            baseline = baseline_medae[key]
            ratio = med / baseline if baseline > 0 else (float("inf") if med > 0 else 0.0)
            # Same documented heuristic family as total-level readiness (2x flag).
            status = "DEGRADATION_DETECTED" if ratio >= 2.0 else "PERFORMANCE_CHANGING" if ratio >= 1.5 else "PERFORMING_NORMALLY"
            rec.update(
                {
                    "status": status,
                    "baseline_medae_seconds": round(baseline, 4),
                    "recent_medae_seconds": round(med, 4),
                    "baseline_source": "stage selection experiment (chronological holdout)",
                    "note": "Recent median absolute stage error compared against the stage's chronological-holdout baseline.",
                }
            )
        # ---- live post-promotion error: matched outcomes recorded
        # AFTER this stage's serving epoch (the moment the now-serving
        # model was activated). Shown from the first outcome (n >= 1);
        # the health badge above stays policy-gated at MIN_WINDOW_ROWS.
        epoch = (serving_epochs or {}).get(key) or {}
        since = epoch.get("since")
        post: dict[str, Any] | None = None
        if since:
            scoped = [e for ts, e in timed_errs_by_stage.get(key, []) if ts >= str(since)]
            if scoped:
                post = {
                    "model_id": epoch.get("model_id"),
                    "promoted_at": str(since),
                    "outcomes": len(scoped),
                    "median_absolute_error_seconds": round(_median(scoped), 4),
                    "mean_absolute_error_seconds": round(float(sum(scoped) / len(scoped)), 4),
                    "source": epoch.get("source"),
                }
        rec["post_promotion"] = post
        stages_out[key] = rec
        if severity.get(rec["status"], 0) > severity.get(worst, 0):
            worst = rec["status"]
    # ---- total-row live error: same promotion-scoped filter over
    # the TOTAL error, since the LATEST serving epoch (the current
    # five-stage composition was established at the last promotion).
    total_post: dict[str, Any] | None = None
    sinces = [str(e.get("since")) for e in (serving_epochs or {}).values() if e.get("since")]
    if sinces:
        latest = max(sinces)
        total_errs: list[float] = []
        for f in matched_feedback:
            te = f.get("total_error")
            created = str(f.get("created_at") or "")
            if not te or created < latest:
                continue
            try:
                v = float(te.get("absolute_error_seconds"))
            except (TypeError, ValueError):
                continue
            if v == v and v not in (float("inf"), float("-inf")):
                total_errs.append(v)
        if total_errs:
            total_post = {
                "promoted_at": latest,
                "outcomes": len(total_errs),
                "median_absolute_error_seconds": round(_median(total_errs), 4),
                "mean_absolute_error_seconds": round(float(sum(total_errs) / len(total_errs)), 4),
                "source": "latest serving epoch",
            }
    return {
        "stages": stages_out,
        "overall_status": worst,
        "outcomes_with_stage_errors": n_outcomes,
        "total_post_promotion": total_post,
        "note": (
            "Stage health compares recent linked-actual stage errors with each "
            "stage's chronological-holdout baseline. Stages without enough "
            "outcomes are never assigned a health claim."
        ),
    }


def prediction_drift_summary(
    prediction_events: list[dict[str, Any]],
    *,
    ref_first_created: str | None = None,
    ref_last_created: str | None = None,
) -> dict[str, Any]:
    """Prediction drift (§16): are SERVED predictions themselves shifting?

    Compares the distribution of predicted totals served DURING the reference
    (model training) period with the recent window. A distribution change is
    a signal, never by itself a retraining trigger.
    """
    import math

    def _vals(events: list[dict[str, Any]]) -> list[float]:
        out: list[float] = []
        for ev in events:
            p = ev.get("prediction") or ev
            v = p.get("predicted_total_seconds")
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                out.append(v)
        return out

    def _within(ev: dict[str, Any], lo: str | None, hi: str | None) -> bool:
        ts = ev.get("created_at") or ""
        if not ts:
            return False
        if lo and ts < lo:
            return False
        if hi and ts >= hi:
            return False
        return True

    ref_events = [e for e in prediction_events if _within(e, ref_first_created, ref_last_created)]
    cur_events = [e for e in prediction_events if not _within(e, ref_first_created, ref_last_created)]
    if ref_last_created is None:
        # No reference window known: everything recent, split by recency half.
        ordered = sorted((e for e in prediction_events if e.get("created_at")), key=lambda e: e["created_at"])
        half = len(ordered) // 2
        ref_events, cur_events = ordered[:half], ordered[half:]
    ref_vals, cur_vals = _vals(ref_events), _vals(cur_events)
    stat, _p = ks_statistic(ref_vals, cur_vals)
    status = _status_from_statistic(stat, DRIFT_WARN_KS, DRIFT_SIG_KS)
    if not ref_vals or not cur_vals:
        status = "INSUFFICIENT_SAMPLE"
    rng = lambda v: ({"min_seconds": round(min(v), 2), "max_seconds": round(max(v), 2)} if v else None)
    return {
        "reference_predictions": {"count": len(ref_vals), **(rng(ref_vals) or {})},
        "recent_predictions": {"count": len(cur_vals), **(rng(cur_vals) or {})},
        "statistic": None if stat != stat else round(stat, 4),
        "method": DRIFT_METHOD_NUMERIC,
        "status": status,
        "note": (
            "Compares the range of predicted execution times between the reference "
            "period and recent predictions. A change can simply reflect different "
            "workloads and is one signal among several."
        ),
    }


def drift_overview(
    *,
    registry: ModelRegistry,
    matched_feedback: list[dict[str, Any]],
    prediction_events: list[dict[str, Any]],
    feature_drift_report: dict[str, Any] | None = None,
    population_drift_report: dict[str, Any] | None = None,
    performance_report: dict[str, Any] | None = None,
    selection_experiments: list[dict[str, Any]] | None = None,
    reference_window: dict[str, Any] | None = None,
    eligible_actual_count: int = 0,
    serving_epochs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compose the locked Drift & Retraining page payload from EXISTING signals.

    Nothing here computes a new statistic: it reuses feature/population drift,
    prediction-performance drift, readiness signals, stage baselines from the
    experiment registry and the model lifecycle. Every headline carries the
    evidence behind it (§19/§31 'Why?') and nothing retrains or promotes.
    """
    fd = feature_drift_report or {}
    pd_ = population_drift_report or {}
    perf = performance_report or {}
    events = [e for e in prediction_events if e.get("event_state") == "PREDICTED"]

    data_status = _worst_status([fd.get("overall_status", "NO_MEANINGFUL_DRIFT"), pd_.get("overall_status", "NO_MEANINGFUL_DRIFT")])
    model_health = _model_health_from_status(perf.get("status"))
    stage_health = stage_model_health(
        matched_feedback,
        selection_experiments=selection_experiments,
        serving_epochs=serving_epochs,
    )

    # ---- newest retraining candidate (fallback for the
    # "After retraining" column after a page reload): the
    # newest EXP-RETRAIN-* experiment record, whose metrics
    # already hold per-stage candidate MedAEs.
    latest_candidate: dict[str, Any] | None = None
    newest = None
    for rec in selection_experiments or []:
        if not str(rec.get("experiment_id", "")).startswith("EXP-RETRAIN-"):
            continue
        if newest is None or str(rec.get("created_at", "")) > str(newest.get("created_at", "")):
            newest = rec
    if newest is not None:
        latest_candidate = {
            "experiment_id": newest.get("experiment_id"),
            "created_at": newest.get("created_at"),
            "per_stage": {
                str(k): {"median_ae": (m or {}).get("median_ae"), "n": (m or {}).get("n")}
                for k, m in (newest.get("metrics") or {}).items()
                if isinstance(m, dict)
            },
        }
    pred_drift = prediction_drift_summary(
        events,
        ref_first_created=(reference_window or {}).get("first_created"),
        ref_last_created=(reference_window or {}).get("last_created"),
    )

    readiness = retraining_readiness(
        registry=getattr(registry, "models", registry),
        matched_feedback=matched_feedback,
        feature_drift_report=fd,
        population_drift_report=pd_,
        eligible_actual_count=eligible_actual_count,
    )
    signals = readiness["signals"]

    # ---- data drift feature table (grouped labels, detail behind toggles)
    feature_rows: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for name in fd.get("top_shifted_features") or []:
        res = (fd.get("features") or {}).get(name) or {}
        label = _feature_display_label(name)
        row = seen.setdefault(label, {"feature": label, "status": res.get("status", "NO_MEANINGFUL_DRIFT"), "statistic": res.get("statistic"), "method": res.get("method"), "raw_features": []})
        row["raw_features"].append(name)
        if row["status"] == "NO_MEANINGFUL_DRIFT":
            row["status"] = res.get("status", row["status"])
    for row in seen.values():
        row["monitored_inputs"] = len(row.get("raw_features") or [])
        row.pop("raw_features", None)  # raw engineered names stay behind the details toggle
    feature_rows = list(seen.values())[:8]

    # ---- assessment narrative + combined verdict (§5, §17, §18)
    data_label = {
        "STABLE": "stable",
        "WATCH": "showing early changes worth watching",
        "DRIFT_DETECTED": "materially different from the reference data",
        "INSUFFICIENT_DATA": "not yet comparable against the reference",
    }[data_health_from(data_status)]
    perf_label = {
        "PERFORMING_NORMALLY": "tracking actual execution times well",
        "PERFORMANCE_CHANGING": "showing early error changes worth watching",
        "DEGRADATION_DETECTED": "measurably less accurate than its baseline",
        "INSUFFICIENT_ACTUAL_OUTCOMES": "not yet assessable — too few matched actual outcomes",
    }[model_health]
    degraded = model_health == "DEGRADATION_DETECTED" or stage_health["overall_status"] == "DEGRADATION_DETECTED"
    changing = model_health == "PERFORMANCE_CHANGING" or stage_health["overall_status"] == "PERFORMANCE_CHANGING"
    if data_status == "SIGNIFICANT_DRIFT" and degraded:
        verdict, tone = "RETRAINING_REVIEW_RECOMMENDED", "review"
        narrative = (
            "Retraining review recommended. Recent executions differ from the reference "
            "data and prediction error has increased. There is enough new eligible "
            "evidence to evaluate a new model — a human decides."
        )
    elif data_status == "SIGNIFICANT_DRIFT":
        verdict, tone = "CONTINUE_MONITORING", "watch"
        narrative = (
            "Data drift detected — model performance remains " + perf_label + ". Continue "
            "monitoring before considering retraining."
        )
    elif degraded:
        verdict, tone = "RETRAINING_REVIEW_RECOMMENDED", "review"
        narrative = (
            "Retraining review recommended. Prediction error has increased against "
            "chronological baselines even though input distributions look broadly "
            "stable. There is enough evidence to evaluate a new model — a human decides."
        )
    elif verdict_is_review_ready(readiness) or changing:
        verdict, tone = "CONTINUE_MONITORING", "watch"
        narrative = (
            "The system is collecting new evidence (" + str(eligible_actual_count) + " new eligible "
            "outcomes). Inputs are " + data_label + " and the model is " + perf_label + ". No retraining review is needed yet."
        )
    else:
        verdict, tone = "NO_RETRAINING_NEEDED", "healthy"
        narrative = (
            "The system is currently healthy. Inputs are " + data_label + " and the model is "
            + perf_label + ". No retraining review is currently needed."
        )

    evidence: list[dict[str, Any]] = [
        {
            "id": "new_eligible_outcomes",
            "label": f"{eligible_actual_count} new eligible execution outcomes since the current model",
            "met": eligible_actual_count > 0,
            "detail": f"Threshold for review-readiness: {signals['eligible_new_actuals']['threshold']}.",
        },
        {
            "id": "data_drift",
            "label": {
                "NO_MEANINGFUL_DRIFT": "Input distributions broadly stable",
                "WARNING": "Some input distributions changed enough to watch",
                "SIGNIFICANT_DRIFT": "Input distributions differ materially from the reference",
                "INSUFFICIENT_SAMPLE": "Not enough recent executions for an input comparison",
            }[data_status],
            "met": data_status in ("WARNING", "SIGNIFICANT_DRIFT"),
            "detail": f"Compared over {fd.get('current_rows', '?')} recent vs {fd.get('reference_rows', '?')} reference rows.",
        },
        {
            "id": "prediction_drift",
            "label": {
                "NO_MEANINGFUL_DRIFT": "Predicted time ranges consistent with the reference period",
                "WARNING": "Predicted time ranges are shifting",
                "SIGNIFICANT_DRIFT": "Predicted time ranges differ materially from the reference period",
                "INSUFFICIENT_SAMPLE": "Not enough predictions yet for a range comparison",
            }[pred_drift["status"]],
            "met": pred_drift["status"] in ("WARNING", "SIGNIFICANT_DRIFT"),
            "detail": pred_drift["note"],
        },
        {
            "id": "model_performance",
            "label": {
                "PERFORMING_NORMALLY": "Prediction error is within its baseline band",
                "PERFORMANCE_CHANGING": "Prediction error is moving outside its usual band",
                "DEGRADATION_DETECTED": "Recent prediction error increased against the baseline",
                "INSUFFICIENT_ACTUAL_OUTCOMES": "Not enough matched actual outcomes to assess error",
            }[model_health],
            "met": model_health in ("PERFORMANCE_CHANGING", "DEGRADATION_DETECTED"),
            "detail": perf.get("note") or "Error compared against the model's chronological-holdout baseline.",
        },
        {
            "id": "stage_degradation",
            "label": {
                "PERFORMING_NORMALLY": "All five stage models are performing normally",
                "PERFORMANCE_CHANGING": "At least one stage model shows early error changes",
                "DEGRADATION_DETECTED": "At least one stage model shows performance degradation",
                "INSUFFICIENT_EVIDENCE": "Stage models need more matched outcomes before assessment",
            }[stage_health["overall_status"]],
            "met": stage_health["overall_status"] in ("PERFORMANCE_CHANGING", "DEGRADATION_DETECTED"),
            "detail": stage_health["note"],
        },
    ]

    return {
        "overview_version": "drift-overview-v1",
        "headline": {
            "data_health": data_health_from(data_status),
            "model_health": model_health,
            "new_eligible_outcomes": int(eligible_actual_count),
            "action": verdict,
        },
        "assessment": {"verdict": verdict, "tone": tone, "narrative": narrative, "evidence": evidence},
        "data_drift": {
            "overall_status": data_status,
            "feature_rows": feature_rows,
            "population_dimensions": [
                {"dimension": k.replace("_", " ").capitalize(), "status": (v or {}).get("status", "NO_MEANINGFUL_DRIFT")}
                for k, v in (pd_.get("dimensions") or {}).items()
            ],
            "reference_window": (reference_window or {}),
        },
        "prediction_drift": pred_drift,
        "model_performance": {
            "overall": perf,
            "stage_health": stage_health,
            "serving_epoch": serving_epochs or {},
        },
        "retraining": {
            "readiness_state": readiness["readiness_state"],
            "human_gate_required": True,
            "automatic_retraining": "OFF",
            "signals": signals,
            "latest_candidate": latest_candidate,
        },
        "notes": [
            "Drift means a material, persistent distribution change — new executions alone never trigger it.",
            "No training is launched automatically; a human approves any retraining or model promotion.",
        ],
    }


def data_health_from(status: str) -> str:
    """Public mapping used by drift_overview (kept as a function for tests)."""
    return _data_health_from_status(status)


def verdict_is_review_ready(readiness: dict[str, Any]) -> bool:
    """True when the readiness state is READY_FOR_REVIEW (enough new evidence,
    no degradation) — used to phrase the 'collecting evidence' narrative."""
    return readiness.get("readiness_state") == "READY_FOR_REVIEW"
