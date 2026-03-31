"""Condition B: per-metric gloss text for single-criterion judge prompts."""

METRIC_GLOSS_DEFAULTS = {
    "accuracy": "Factual correctness: no errors, contradictions, or unsupported claims.",
    "helpfulness": "Usefulness and clarity for someone asking the question; solves the user’s need.",
    "relevance": "How directly and completely the answer targets the question asked.",
    "completeness": "Coverage of the important points implied by the question.",
    "clarity": "Clear structure and wording; easy to understand.",
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
