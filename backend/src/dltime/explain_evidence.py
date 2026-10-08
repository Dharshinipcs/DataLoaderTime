"""Evidence-sentence explainability (user final spec, 2026-10-01).

Deterministic template code — pandas numbers + f-strings only. No AI, no LLM,
no new dependency. Three layers:

  Layer 1 — Evidence sentences: one plain-English sentence per applicable ML
            stage, built from THIS prediction vs the loader's OWN strictly-
            prior history (the same stage_hist_* numbers the model itself
            consumed). Every number is verifiable on the Loader Intelligence
            page.

  Layer 2 — What-if counterfactuals: for prevalidation/transformation stages
            the user did NOT enter, show the stage's own model estimate and
            how to include it (re-scoring with/without the entry is exactly
            what the applicable-stage engine already does).

  Layer 3 — The existing SHAP ranked contributions stay untouched as the
            technical backing ("show the math" for auditors).

Version: evidence-v1 (bump when template wording/semantics change).
"""

from __future__ import annotations

from typing import Any

EVIDENCE_VERSION = "evidence-v1"

_STAGE_DISPLAY = {
    "staging": "Staging",
    "prevalidation": "Prevalidation",
    "transformation": "Transformation",
    "pre_load": "Pre-load",
    "data_loading": "Data loading",
    "datamart_approval": "Datamart approval",
    "dataloading_approval": "Data-loading approval",
}


def _fmt(seconds: float | None) -> str:
    """Compact human formatting, deterministic."""
    if seconds is None:
        return "n/a"
    s = float(seconds)
    if s < 10:
        return f"{s:.1f}s"
    total = int(round(s))
    if total < 60:
        return f"{total}s"
    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s" if sec else f"{minutes}m"
    hours, rem = divmod(minutes, 60)
    return f"{hours}h {rem:02d}m" if rem else f"{hours}h"


def _stage_history_lines(
    stage: str,
    prediction: float,
    prior_count: float | None,
    prior_median: float | None,
    loader_priors: int | None,
) -> str | None:
    """Layer 1 sentence for one ML stage — None when there is nothing honest
    to say (missing history fields)."""
    display = _STAGE_DISPLAY.get(stage, stage)
    n = None if prior_count is None else int(prior_count) if prior_count == prior_count else None
    med = float(prior_median) if prior_median is not None and prior_median == prior_median else None

    if not n:  # 0 or missing
        if loader_priors:
            return (
                f"{display}: your loader has no recorded {display.lower()} history yet, "
                f"so the model learned this stage from similar executions of other loaders "
                f"and predicts {_fmt(prediction)}."
            )
        return (
            f"{display}: no history exists for this loader — the prediction "
            f"({_fmt(prediction)}) comes from similar executions across all loaders."
        )

    if med is None or med <= 0:
        return (
            f"{display}: your loader ran this stage in {n} past execution(s); "
            f"this run is predicted at {_fmt(prediction)}."
        )

    delta = prediction - med
    pct = abs(delta) / med * 100.0
    band = max(0.5, 0.10 * med)  # within 10% (or half a second) reads as "in line"
    if abs(delta) <= band:
        return (
            f"{display}: your loader ran this stage {n} time(s) with a typical time of "
            f"{_fmt(med)} — this run is predicted at {_fmt(prediction)}, right in line "
            f"with your usual."
        )
    direction = "lower" if delta < 0 else "higher"
    return (
        f"{display}: your loader ran this stage {n} time(s) with a typical time of "
        f"{_fmt(med)} — this run is predicted at {_fmt(prediction)}, {pct:.0f}% {direction} "
        f"than your usual, driven by this run's inputs (records, counts, configuration)."
    )


def build_evidence_explanation(
    *,
    stage_intel: dict[str, Any] | None,
    unfiltered_stage_estimates: dict[str, float | None],
    history: dict[str, Any] | None,
    ctx_stage_history: dict[str, dict[str, float | None]],
    total_seconds: float | None,
    applicable: list[str],
    not_applicable: list[str],
) -> dict[str, Any]:
    """Build the full evidence payload. Pure function of the given numbers."""
    stages = (stage_intel or {}).get("stages", []) or []
    loader_priors = None
    if history:
        raw = history.get("prior_loader_executions")
        loader_priors = int(raw) if isinstance(raw, (int, float)) else None

    sentences: list[str] = []
    for entry in stages:
        stage = entry.get("stage")
        if entry.get("method") != "ml_stage_model" or not entry.get("available"):
            continue
        pred = entry.get("estimate_seconds")
        if pred is None:
            continue
        hist = ctx_stage_history.get(stage, {})
        line = _stage_history_lines(
            stage,
            float(pred),
            hist.get("prior_count"),
            hist.get("prior_median"),
            loader_priors,
        )
        if line:
            sentences.append(line)

    # Approval stages: one honest statistical line each (tier + n already in
    # the entry; the sentence translates it). 2026-10-02: only LOADER-tier
    # evidence (the loader's OWN approval history) is included in the total,
    # so each sentence states explicitly whether the estimate is in the total
    # or disclosed-but-excluded (broader-context history).
    approval_lines: list[str] = []
    for entry in stages:
        if entry.get("method") != "contextual_hierarchical_statistical":
            continue
        if not entry.get("available") or entry.get("estimate_seconds") is None:
            continue
        display = _STAGE_DISPLAY.get(entry.get("stage", ""), entry.get("stage"))
        tier = (entry.get("evidence_tier_label") or entry.get("evidence_tier") or "").lower()
        n = entry.get("n_evidence")
        base = f"{display}: estimated at {_fmt(entry.get('estimate_seconds'))}"
        in_total = entry.get("included_in_total")
        if in_total is False:
            # 2026-10-03 (user final spec): a broad-context (GLOBAL fallback)
            # approval estimate is hidden from the prediction result entirely —
            # no approval sentence is emitted when the evidence is not this
            # loader's own approval history.
            continue
        if tier and n:
            approval_lines.append(f"{base} as the historical median of {n} matching past approval(s) ({tier}), included in the total.")
        else:
            approval_lines.append(f"{base} from the historical median of matching past approvals.")
    sentences.extend(approval_lines)

    # Layer 2 — what-if counterfactuals for the optional stages the user did
    # not enter (not applicable). The estimate is that stage's own ML model
    # output for this context (already computed on the unfiltered intel).
    what_ifs: list[str] = []
    for stage in not_applicable:
        if stage not in ("prevalidation", "transformation"):
            continue
        est = unfiltered_stage_estimates.get(stage)
        if est is None:
            continue
        display = _STAGE_DISPLAY.get(stage, stage)
        what_ifs.append(
            f"If this execution also runs {display.lower()}, add roughly {_fmt(est)}: "
            f"enter its display name or count in the form and the total will include it."
        )

    n_ml = sum(1 for s in sentences if s and not s.startswith(tuple(_STAGE_DISPLAY.get(a, a) + ":" for a in [])))
    summary = (
        f"Total {_fmt(total_seconds)} = the ML stage predictions + the statistical "
        f"approval estimates above, each compared against this loader's own history."
    )
    return {
        "available": bool(sentences or what_ifs),
        "evidence_version": EVIDENCE_VERSION,
        "summary": summary,
        "sentences": sentences,
        "what_ifs": what_ifs,
        "ml_sentence_count": len(sentences) - len(approval_lines),
        "approval_sentence_count": len(approval_lines),
    }


PLAIN_EXPLAIN_VERSION = "plain-explain-v2"


def build_plain_explanation(
    *,
    stage_intel: dict[str, Any] | None,
    history: dict[str, Any] | None,
    ctx_stage_history: dict[str, dict[str, float | None]],
    total_seconds: float | None,
    uncertainty: dict[str, Any] | None = None,
) -> str | None:
    """Short plain-language reason (max 3 sentences) for this prediction.

    plain-explain-v2 (2026-10-06 user spec): the ESSENCE of why this ETA,
    assembled from the explainability layers that already run for every
    prediction — no model names, no feature lists:
      1. biggest driver vs the loader's own typical time for that step
         (per-stage prediction vs that stage's recorded history);
      2. the calibrated interval (how far off similar past predictions
         landed — the real time most likely falls in this range);
      3. which pipeline steps do NOT apply to this run (why the total can
         look short).
    Deterministic templates over real numbers only. Supplementary — never
    alters the ETA.
    """
    stages = (stage_intel or {}).get("stages", []) or []
    ml = [
        (str(e.get("stage")), float(e.get("estimate_seconds")))
        for e in stages
        if e.get("method") == "ml_stage_model" and e.get("available") and e.get("estimate_seconds") is not None
    ]
    if not ml:
        return None
    raw = (history or {}).get("prior_loader_executions")
    loader_priors = int(raw) if isinstance(raw, (int, float)) else None
    if not loader_priors:
        return (
            "Your loader has little recorded history yet, so this prediction is based "
            f"mostly on how similar loaders usually run — about {_fmt(total_seconds)} in total."
        )

    total = _fmt(total_seconds)
    lines: list[str] = []
    # Sentence 1 — the essence: the biggest driver compared against the
    # loader's own typical time for that step (per-stage prediction layer
    # vs that stage's recorded history).
    top_stage, _top_sec = max(ml, key=lambda t: t[1])
    top_display = _STAGE_DISPLAY.get(top_stage, top_stage).lower()

    # Largest deviation from the loader's own typical time for a step.
    dev: tuple[str, float, float, float] | None = None  # (stage, seconds, median, pct)
    for stage, sec in ml:
        hist = ctx_stage_history.get(stage) or {}
        n = hist.get("prior_count")
        med = hist.get("prior_median")
        if not n or med is None or med != med or med <= 0:
            continue
        med = float(med)
        pct = abs(sec - med) / med * 100.0
        if pct > 15 and (dev is None or pct > dev[3]):
            dev = (stage, sec, med, pct)

    if dev is not None:
        d_stage, d_sec, d_med, d_pct = dev
        d_display = _STAGE_DISPLAY.get(d_stage, d_stage).lower()
        d_dir = "longer" if d_sec > d_med else "shorter"
        if d_stage == top_stage:
            lines.append(
                f"Most of the expected time comes from the {d_display} step, which this run's "
                f"inputs make about {d_pct:.0f}% {d_dir} than your loader's usual {_fmt(d_med)}."
            )
        else:
            lines.append(
                f"This run's {d_display} step looks about {d_pct:.0f}% {d_dir} than your loader's "
                f"usual {_fmt(d_med)}, and the {top_display} step is the biggest part of the job."
            )
    else:
        lines.append(
            f"This run is set up much like your loader's past executions, with the {top_display} "
            f"step as the biggest part of the job."
        )

    # Sentence 2 — the calibrated interval (Task 07 layer): how far off the
    # model landed on comparable past runs, i.e. how much to trust the point.
    unc = uncertainty or {}
    if unc.get("available") is not False and unc.get("upper_seconds") is not None:
        upper = float(unc["upper_seconds"])
        lower_raw = unc.get("lower_seconds")
        lower = float(lower_raw) if isinstance(lower_raw, (int, float)) else 0.0
        width = unc.get("interval_width_seconds")
        wide = width is not None and total_seconds and float(width) / max(float(total_seconds), 1e-9) >= 1.5
        if lower <= 0:
            lines.append(
                f"On similar past runs this model's misses stayed within about {_fmt(upper)}, "
                + (
                    "so treat the range as realistic rather than the exact number."
                    if wide
                    else "so the actual time should land close to the prediction."
                )
            )
        else:
            lines.append(
                f"On similar past runs this model's misses stayed within about {_fmt(lower)}–{_fmt(upper)}, "
                + (
                    "so treat the range as realistic rather than the exact number."
                    if wide
                    else "so the actual time should land close to the prediction."
                )
            )

    # Sentence 3 — pipeline steps that do NOT apply to this run (why the
    # total can look shorter than a full run).
    not_applicable = [
        str(e.get("stage"))
        for e in stages
        if e.get("method") == "ml_stage_model" and not e.get("available")
    ]
    if not_applicable:
        names = " and ".join(_STAGE_DISPLAY.get(s, s).lower() for s in not_applicable)
        verb = "is" if len(not_applicable) == 1 else "are"
        pronoun = "it adds" if len(not_applicable) == 1 else "they add"
        lines.append(
            f"{names} {verb} not part of this run's configuration, so {pronoun} nothing to the total."
        )

    return " ".join(lines[:3])
