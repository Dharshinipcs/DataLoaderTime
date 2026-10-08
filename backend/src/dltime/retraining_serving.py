"""Serving-path switch for promoted candidate bundles.

DEFAULT (locked EXP-038 policy): the served ETA is the deterministic sum of
the five operational stage models loaded from the standard artifact
(artifacts/stage-models-ad34.pkl). This module adds explicit, persisted
overrides — never automatic:

  - Whole-bundle switch: after a HUMAN-approved promotion of a candidate
    BUNDLE, stage serving loads that bundle instead.
  - Per-stage switch: after a HUMAN-approved promotion of a single-stage
    candidate, ONLY that stage's model is swapped; every other stage keeps
    serving its current model from the standard artifact (or its own
    override).

A rollback clears the override(s) and serving returns to the standard
artifact. There is no automatic switch anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dltime.config import PROJECT_ROOT

SWITCH_FILE = PROJECT_ROOT / "backend" / "artifacts" / "registry" / "serving_switch.json"


def _read_switch() -> dict[str, Any] | None:
    if not SWITCH_FILE.exists():
        return None
    try:
        data = json.loads(SWITCH_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data.get("active") else None
    except Exception:
        return None  # fail-closed: a corrupt switch never redirects serving


def switch_record() -> dict[str, Any] | None:
    """The full persisted switch payload (whole-bundle + per-stage overrides),
    or None when no override is active."""
    return _read_switch()


def active_version() -> str | None:
    """The candidate bundle version serving is switched to, if any."""
    sw = _read_switch()
    return str(sw["candidate_version"]) if sw else None


def active_stage_overrides() -> dict[str, dict[str, Any]]:
    """Per-stage serving overrides {stage: {candidate_version, artifact_path,
    activated_at}} — empty when none are active."""
    sw = _read_switch()
    if not sw:
        return {}
    ov = sw.get("stage_overrides") or {}
    return {str(k): dict(v) for k, v in ov.items() if isinstance(v, dict)}


def active_artifact_path() -> Path | None:
    sw = _read_switch()
    if not sw:
        return None
    p = Path(sw.get("artifact_path", ""))
    return p if p.exists() else None


def activate(candidate_version: str, *, artifact_path: str | None = None) -> dict[str, Any]:
    """Persist the whole-bundle switch to a promoted candidate bundle
    (explicit action only)."""
    from dltime.model_lab import utc_now_iso

    path = artifact_path or str(
        PROJECT_ROOT / "backend" / "artifacts" / f"stage-models-{candidate_version}.pkl"
    )
    if not Path(path).exists():
        from dltime.model_lab import LabError

        raise LabError(f"Cannot activate serving switch: artifact missing at {path!r}.")
    SWITCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "active": True,
        "candidate_version": candidate_version,
        "artifact_path": path,
        "activated_at": utc_now_iso(),
    }
    SWITCH_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def activate_stage(
    stage: str, candidate_version: str, *, artifact_path: str | None = None
) -> dict[str, Any]:
    """Persist a PER-STAGE serving override (explicit action only).

    The stage's model comes from the single-stage candidate artifact; every
    other stage keeps serving its current model. An active whole-bundle
    switch is cleared first so the two mechanisms never conflict.
    """
    from dltime.model_lab import LabError, utc_now_iso

    path = artifact_path or str(
        PROJECT_ROOT / "backend" / "artifacts" / f"stage-models-{candidate_version}.pkl"
    )
    if not Path(path).exists():
        raise LabError(f"Cannot activate stage switch: artifact missing at {path!r}.")
    SWITCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = _read_switch() or {"active": True}
    # A whole-bundle switch and per-stage overrides are mutually exclusive:
    # activating a stage override drops the bundle switch (fail-closed: the
    # standard artifact + explicit stage overrides fully describe serving).
    payload.pop("candidate_version", None)
    payload.pop("artifact_path", None)
    payload.pop("activated_at", None)
    payload["active"] = True
    overrides = dict(payload.get("stage_overrides") or {})
    overrides[str(stage)] = {
        "candidate_version": candidate_version,
        "artifact_path": path,
        "activated_at": utc_now_iso(),
    }
    payload["stage_overrides"] = overrides
    SWITCH_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def reset(stages: list[str] | None = None) -> None:
    """Clear override(s) (rollback path): serving returns to the standard
    artifact. With ``stages`` only those per-stage overrides are cleared;
    without it the whole switch (bundle + all stage overrides) is cleared."""
    if stages:
        sw = _read_switch()
        if not sw:
            return
        overrides = dict(sw.get("stage_overrides") or {})
        changed = False
        for stage in stages:
            if str(stage) in overrides:
                overrides.pop(str(stage))
                changed = True
        if not changed:
            return
        sw["stage_overrides"] = overrides
        if not overrides and not sw.get("candidate_version"):
            SWITCH_FILE.unlink(missing_ok=True)
            return
        SWITCH_FILE.write_text(json.dumps(sw, indent=2), encoding="utf-8")
        return
    if SWITCH_FILE.exists():
        SWITCH_FILE.unlink()


def load_stage_serving_models_switched(bundle=None) -> dict[str, Any]:
    """Serving entry point honoring the switch (never silently: file-backed).

    Resolution order:
      1. Whole-bundle override active -> serve that bundle's models.
      2. Per-stage overrides -> the standard artifact's models with the
         overridden stages replaced by their single-stage candidate models.
      3. No override -> the standard artifact (or the supplied bundle).
    """
    from dltime.stage_serving import load_stage_serving_models

    sw = _read_switch()
    if sw and sw.get("candidate_version"):
        path = active_artifact_path()
        if path is not None:
            import joblib

            blob = joblib.load(path)
            models = blob.get("models")
            if models:
                return models
        return load_stage_serving_models(bundle=bundle)

    overrides = active_stage_overrides()
    if not overrides:
        return load_stage_serving_models(bundle=bundle)

    import joblib

    models = load_stage_serving_models(bundle=bundle)
    merged = dict(models or {})
    for stage, ov in overrides.items():
        p = Path(ov.get("artifact_path", ""))
        if not p.exists():
            continue  # fail-closed: keep the standard model for this stage
        try:
            blob = joblib.load(p)
            stage_model = (blob.get("models") or {}).get(stage)
            if stage_model is not None:
                merged[stage] = stage_model
        except Exception:
            continue  # fail-closed: keep the standard model for this stage
    return merged
