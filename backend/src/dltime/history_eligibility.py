"""History eligibility (Task 02, AD-12).

Decides whether a prediction context has enough direct history to support an
authoritative ETA. Data-driven and configurable: NO hard-coded universal
threshold. The v1 candidate values below are placeholders for model-validation
evidence (PENDING in AD-12) — they are defaults that Task 03/04 experiments
must justify, adjust, or replace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class HistoryEligibilityConfig:
    """Candidate thresholds — PENDING validation evidence (AD-12).

    min_prior_loader_executions: direct loader priors required.
    min_prior_loader_conn_executions: required when relying on the tighter
        Loader+Connection history (falls back to loader-level evidence when 0).
    """

    min_prior_loader_executions: int = 3
    min_prior_loader_conn_executions: int = 0
    require_nonnegative_time_since_prev: bool = True


@dataclass
class HistoryEligibility:
    sufficient: bool
    reason: str
    evidence: dict = field(default_factory=dict)


class HistoryEligibilityEngine:
    """Evaluates history sufficiency for prediction contexts from real priors."""

    def __init__(self, config: HistoryEligibilityConfig | None = None) -> None:
        self.config = config or HistoryEligibilityConfig()

    def evaluate(
        self,
        *,
        hist_loader_prior_count,
        hist_loader_conn_prior_count=0,
        hist_time_since_prev_loader_sec=None,
    ) -> HistoryEligibility:
        """Decide sufficiency from computed prior-history features."""
        cfg = self.config
        loader_count = 0 if pd.isna(hist_loader_prior_count) else int(hist_loader_prior_count)
        conn_count = 0 if pd.isna(hist_loader_conn_prior_count) else int(hist_loader_conn_prior_count)

        evidence = {
            "prior_loader_executions": loader_count,
            "prior_loader_conn_executions": conn_count,
            "time_since_prev_loader_sec": (
                None if hist_time_since_prev_loader_sec is None or pd.isna(hist_time_since_prev_loader_sec)
                else float(hist_time_since_prev_loader_sec)
            ),
        }

        if loader_count == 0:
            return HistoryEligibility(
                sufficient=False,
                reason="NO_DIRECT_HISTORY",
                evidence=evidence,
            )
        if loader_count < cfg.min_prior_loader_executions:
            return HistoryEligibility(
                sufficient=False,
                reason="INSUFFICIENT_LOADER_HISTORY",
                evidence=evidence,
            )
        if (
            cfg.min_prior_loader_conn_executions > 0
            and conn_count < cfg.min_prior_loader_conn_executions
        ):
            return HistoryEligibility(
                sufficient=False,
                reason="INSUFFICIENT_LOADER_CONNECTION_HISTORY",
                evidence=evidence,
            )
        if (
            cfg.require_nonnegative_time_since_prev
            and hist_time_since_prev_loader_sec is not None
            and not pd.isna(hist_time_since_prev_loader_sec)
            and hist_time_since_prev_loader_sec < 0
        ):
            return HistoryEligibility(
                sufficient=False,
                reason="INVALID_TIME_SINCE_PREVIOUS",
                evidence=evidence,
            )
        return HistoryEligibility(
            sufficient=True,
            reason="SUFFICIENT_DIRECT_HISTORY",
            evidence=evidence,
        )


def summarize_prior_history_distribution(counts: pd.Series) -> dict:
    """Describe the actual prior-loader-history distribution (AD-12 evidence)."""
    c = pd.to_numeric(counts, errors="coerce")
    return {
        "min": int(c.min()),
        "p25": float(c.quantile(0.25)),
        "median": float(c.median()),
        "p75": float(c.quantile(0.75)),
        "p90": float(c.quantile(0.90)),
        "max": int(c.max()),
        "mean": float(c.mean()),
        "zero_history": int((c == 0).sum()),
        "lt_3": int((c < 3).sum()),
        "lt_5": int((c < 5).sum()),
        "ge_3": int((c >= 3).sum()),
        "ge_5": int((c >= 5).sum()),
        "ge_10": int((c >= 10).sum()),
    }
