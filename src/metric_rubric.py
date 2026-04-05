"""Condition B: per-metric gloss text for single-criterion judge prompts."""

METRIC_GLOSS_DEFAULTS = {
    "accuracy": (
        "Factual and logical soundness of claims in the answer, relative to the question. "
        "Reward correct, well-grounded statements; penalize clear errors, internal contradictions, "
        "or strong claims with no support when the task calls for care. "
        "If the question is subjective or underspecified, judge plausibility and absence of obvious falsehoods, "
        "not your own preferred opinion."
    ),
    "relevance": (
        "How tightly the content addresses **what was asked**: the stated topic, constraints, and intent. "
        "Penalize topic drift, padding unrelated to the question, or answering a different question. "
        "Completeness of *coverage* is secondary here—ignore missing depth if the material stays on-target "
        "unless the question explicitly demands that depth."
    ),
    "completeness": (
        "Whether the answer covers the **substantive parts** the question demands: explicit sub-questions, "
        "listed items, constraints (e.g. format, length limits), and reasonable follow-ups implied by the wording. "
        "Reward addressing each important piece; penalize skipping a major requested element or only hand-waving "
        "where specifics were expected. "
        "Do not require extra topics the question did not suggest; brevity is fine if nothing material is missing."
    ),
}


def gloss_for_metric(metric_name: str) -> str:
    key = (metric_name or "").strip().lower()
    if not key:
        raise ValueError("Metric name is empty after trim.")
    if key not in METRIC_GLOSS_DEFAULTS:
        allowed = ", ".join(sorted(METRIC_GLOSS_DEFAULTS.keys()))
        raise ValueError(
            f"Unknown metric {metric_name!r} (normalized: {key!r}). "
            f"No gloss is defined — add it to METRIC_GLOSS_DEFAULTS in metric_rubric.py. "
            f"Currently allowed: {allowed}"
        )
    return METRIC_GLOSS_DEFAULTS[key]
