"""Operational risk engine (Task 09, AD-22).

An independent, deterministic INTERPRETATION layer — never a second
prediction model. The point ETA (Task 04), the calibrated uncertainty
(Task 07), SHAP explanation (Task 08), history eligibility (AD-16), and
stage intelligence (Task 06) are consumed read-only; risk never modifies
any of them, and a risk failure never fails or alters the prediction.

Selected strategy (EXP-018 evidence): a transparent rule HIERARCHY with
machine-readable reasons. A weighted additive score was REJECTED: its
inputs overlap (interval width ↔ history strength), so weights would
double-count; a hierarchy makes precedence explicit and auditable.

Risk is an OPERATIONAL CLASSIFICATION, not a probability of failure and
not model confidence. Labels (PREDICTED path):
    LOW_CONFIDENCE   calibrated interval is very wide (weak evidence)
    OUT_OF_DISTRIBUTION  configuration beyond the model's observed
                     training configuration range (robust training-range
                     fence, config features only — EXP-018: no real
                     holdout contexts flagged, so a flag is meaningful)
    HIGH_RISK        prediction is unprecedented against the loader's own
                     history AND an independent second warning exists
    WARNING          a single evidence-backed operational concern
    NORMAL           no operational concern raised

Precedence (documented, deterministic): LOW_CONFIDENCE > OOD > HIGH_RISK >
WARNING > NORMAL. LOW_CONFIDENCE first so weak evidence is never dressed
up as a stronger signal; HIGH_RISK requires corroboration so a merely
large-but-consistent ETA is never dramatized.

INSUFFICIENT_HISTORY is a prediction STATE (AD-16): it carries no ETA and
receives only the opaque label INSUFFICIENT_HISTORY with the state reason
— no fabricated score. UNRESOLVED/AMBIGUOUS/INVALID receive NO risk at all.

Signal hygiene (no double counting): interval width and prior-count
signals are NOT reused here (already integral to the interval signal);
stage intelligence contributes only "stage estimate unavailable for a
modelable stage" (low stage evidence), never stage magnitudes, and stage
estimates are never summed. All thresholds carry EXP-018 evidence and are
configurable; defaults marked provisional where evidence is thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

RISK_POLICY_VERSION = "risk-hierarchy-v1"

# Config features defining the observed training configuration range.
# Only def_* (pre-execution definition) features are used: input
# unusualness must come from the INPUT configuration, not from historical
# aggregates (whose skewed scales make range ratios meaningless — the
# initial all-feature probe produced a spurious 29,605-unit excess).
OOD_FENCE_EXCESS = 0.5  # robust-IQR units beyond the training [p1, p99] fence
OOD_NOTE = "provisional threshold; flagged 0 of the real holdout contexts at 0.5 (EXP-018 evaluation)"

# ETA vs the loader's own prior median (EXP-018 holdout calibration: p90=3.12,
# max 3.89, share >5x = 0.000 — 5x is unprecedented in the real data).
ETA_VS_MEDIAN_WARN = 3.0   # ~10% flag rate at 3x on real holdout
ETA_VS_MEDIAN_HIGH = 5.0   # unprecedented; also requires a second signal

# LOW_CONFIDENCE maps directly from the interval-width criterion (single
# source of truth; no competing confidence system).


class RiskLabel:
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    HIGH_RISK = "HIGH_RISK"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"


@dataclass
class RiskResult:
    label: str
    reasons: list[dict[str, str]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    policy_version: str = RISK_POLICY_VERSION

    def to_payload(self) -> dict[str, Any]:
        # Prompt 3 (§12): the interpretation disclaimer travels with every
        # risk payload — risk is an operational ATTENTION level derived from
        # transparent signals, never a probability of failure and never a
        # model-confidence statement. No calibrated failure-probability model
        # exists in this project and none is permitted by the locked scope.
        return {
            "label": self.label,
            "reasons": list(self.reasons),
            "evidence": self.evidence,
            "policy_version": self.policy_version,
            "interpretation": (
                "An operational attention level derived from transparent "
                "signals — not a probability of failure and not a statement "
                "about model confidence."
            ),
        }


def insufficient_history_risk(reason: str | None) -> dict[str, Any]:
    """Opaque risk for the INSUFFICIENT_HISTORY prediction state (no ETA exists)."""
    return {
        "label": RiskLabel.INSUFFICIENT_HISTORY,
        "reasons": [
            {
                "code": "INSUFFICIENT_HISTORY",
                "message": (
                    "No operational risk assessment is produced: the loader does "
                    "not meet the minimum prior-history policy, so no ETA exists "
                    "to assess."
                ),
                "evidence": reason or "AD-16 minimum prior-history not met",
            }
        ],
        "evidence": {},
        "policy_version": RISK_POLICY_VERSION,
        "interpretation": (
            "An operational attention level derived from transparent signals — "
            "not a probability of failure and not a statement about model "
            "confidence."
        ),
    }


def training_range_profile(X_train: pd.DataFrame, config_features: list[str]) -> dict[str, Any]:
    """Robust configuration-range profile from the artifact's training window."""
    q01 = X_train[config_features].quantile(0.01)
    q99 = X_train[config_features].quantile(0.99)
    iqr = (X_train[config_features].quantile(0.75) - X_train[config_features].quantile(0.25)).replace(0, np.nan)
    return {
        "config_features": list(config_features),
        "q01": q01.to_dict(),
        "q99": q99.to_dict(),
        "iqr": iqr.to_dict(),
        "n_training_rows": int(len(X_train)),
    }


def config_range_excess(X: pd.DataFrame, profile: dict[str, Any]) -> tuple[float, str | None]:
    """Robust-IQR units by which the context exceeds the training [p1,p99] fence.

    Returns (excess, worst_feature); (0.0, None) when inside the fence or when
    no finite comparison exists (e.g. all-config-missing contexts, which the
    inference layer already flags via its own definition-match warning).
    """
    feats = profile["config_features"]
    if X[feats].notna().sum(axis=1).iloc[0] == 0:
        return 0.0, None
    q01, q99, iqr = profile["q01"], profile["q99"], profile["iqr"]
    lo = (pd.Series(q01) - X[feats].iloc[0]) / pd.Series(iqr)
    hi = (X[feats].iloc[0] - pd.Series(q99)) / pd.Series(iqr)
    both = pd.concat([lo, hi])
    both = both.replace([np.inf, -np.inf], np.nan).dropna()
    if both.empty:
        return 0.0, None
    worst = both.idxmax()
    return float(max(both.max(), 0.0)), (worst if both.max() > 0 else None)


def assess_risk(
    *,
    predicted_seconds: float,
    reliability_label: str | None,
    history: dict[str, Any],
    X: pd.DataFrame,
    ood_profile: dict[str, Any] | None,
    stage_intelligence: dict[str, Any] | None,
    modelable_stages: set[str] | None = None,
) -> RiskResult:
    """Deterministic risk classification for one PREDICTED response.

    Pure function of already-computed prediction-time evidence; identical
    inputs always yield identical output (no randomness, no time dependence).
    """
    reasons: list[dict[str, str]] = []
    evidence: dict[str, Any] = {
        "predicted_seconds": round(float(predicted_seconds), 3),
        "prior_loader_executions": history.get("prior_loader_executions"),
    }

    # ---- signal 1: calibrated interval width (2026-10-03: replaces the
    # removed reliability label — same p90-anchored criterion, no label). A
    # very wide interval means the evidence behind the ETA range is weak.
    if reliability_label == "LOW":
        reasons.append(
            {
                "code": "WIDE_INTERVAL",
                "message": (
                    "The calibrated prediction interval is very wide relative to "
                    "the point estimate: the evidence behind the ETA range is weak."
                ),
                "evidence": "interval width exceeds 0.472x the point estimate (p90-anchored criterion)",
            }
        )

    # ---- signal 2: configuration beyond the observed training range ------
    ood_excess, ood_feature = (0.0, None)
    if ood_profile is not None:
        ood_excess, ood_feature = config_range_excess(X, ood_profile)
        evidence["config_range_excess"] = round(ood_excess, 3)
        if ood_excess > OOD_FENCE_EXCESS and ood_feature is not None:
            reasons.append(
                {
                    "code": "CONFIG_OUT_OF_TRAINING_RANGE",
                    "message": (
                        "The input configuration differs substantially from the "
                        "configurations observed in the model's training window."
                    ),
                    "evidence": f"{ood_feature} is {ood_excess:.1f} robust-IQR units beyond the training range (fence {OOD_FENCE_EXCESS}; {OOD_NOTE})",
                }
            )

    # ---- signal 3: prediction vs the loader's own history -----------------
    med = history.get("prior_loader_median_sec")
    eta_vs_median = None
    if med is not None and float(med) > 0:
        eta_vs_median = float(predicted_seconds) / float(med)
        evidence["eta_vs_prior_loader_median"] = round(eta_vs_median, 3)
        if eta_vs_median > ETA_VS_MEDIAN_HIGH:
            reasons.append(
                {
                    "code": "ETA_FAR_ABOVE_LOADER_HISTORY",
                    "message": (
                        "The predicted duration is far above this loader's own "
                        "historical median — an unprecedented relationship in "
                        "the observed data."
                    ),
                    "evidence": f"ETA is {eta_vs_median:.1f}x the prior loader median (threshold {ETA_VS_MEDIAN_HIGH}x)",
                }
            )
        elif eta_vs_median > ETA_VS_MEDIAN_WARN:
            reasons.append(
                {
                    "code": "ETA_ABOVE_LOADER_HISTORY",
                    "message": (
                        "The predicted duration is well above this loader's own "
                        "historical median."
                    ),
                    "evidence": f"ETA is {eta_vs_median:.1f}x the prior loader median (threshold {ETA_VS_MEDIAN_WARN}x)",
                }
            )

    # ---- signal 4: low stage evidence (availability only, never sums) -----
    low_stage = _low_stage_evidence(stage_intelligence, modelable_stages)
    if low_stage is not None:
        reasons.append(low_stage)

    # ---- deterministic precedence: LOW_CONF > OOD > HIGH > WARNING > NORMAL
    codes = {r["code"] for r in reasons}
    if reliability_label == "LOW":
        label = RiskLabel.LOW_CONFIDENCE
    elif "CONFIG_OUT_OF_TRAINING_RANGE" in codes:
        label = RiskLabel.OUT_OF_DISTRIBUTION
    elif "ETA_FAR_ABOVE_LOADER_HISTORY" in codes and len(codes) >= 2:
        # HIGH_RISK requires corroboration by an independent second signal —
        # a large-but-consistent ETA alone is never HIGH_RISK.
        label = RiskLabel.HIGH_RISK
        reasons.append(
            {
                "code": "HIGH_RISK_CORROBORATED",
                "message": (
                    "HIGH_RISK requires the unprecedented-history signal plus at "
                    "least one independent concern; both are present."
                ),
                "evidence": f"signals: {sorted(c for c in codes if c != 'ETA_FAR_ABOVE_LOADER_HISTORY')}",
            }
        )
    elif codes:
        label = RiskLabel.WARNING
    else:
        label = RiskLabel.NORMAL

    return RiskResult(label=label, reasons=reasons, evidence=evidence)


def _low_stage_evidence(
    stage_intelligence: dict[str, Any] | None, modelable_stages: set[str] | None
) -> dict[str, str] | None:
    """One low-evidence reason when a dataset-wide MODELABLE stage lacks enough
    prior evidence for THIS context.

    Uses availability/evidence only — never stage duration magnitudes, and
    stage estimates are never summed. Analytics-only/sparse stages being
    unavailable is expected and NOT a risk signal.
    """
    if not stage_intelligence or not stage_intelligence.get("available"):
        return None
    if not modelable_stages:
        return None
    stages = stage_intelligence.get("stages") or []
    low = [
        s.get("stage")
        for s in stages
        if (not s.get("available")) and s.get("stage") in modelable_stages
    ]
    if not low:
        return None
    return {
        "code": "LOW_STAGE_EVIDENCE",
        "message": (
            "One or more stages with sufficient dataset-wide history lack enough "
            "prior evidence for this context; stage composition visibility is "
            "reduced for this prediction."
        ),
        "evidence": f"modelable stages unavailable for this context: {sorted(str(s) for s in low)}",
    }
