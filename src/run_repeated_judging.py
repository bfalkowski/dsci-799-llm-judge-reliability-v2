import json
import os
import uuid
import time
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from opentelemetry import trace

from constants import JUDGE_MODEL
from judge import call_judge, extract_json_from_text, is_claude_model
from otel_setup import setup_tracer, get_trace_context
from utils import ENCODING, REPO_ROOT

# ---------- CONFIG ----------
INPUT_PATH = REPO_ROOT / "data" / "mt_bench_subset.json"

JUDGE_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "judge_prompt.txt"

REPEATS = 5
TEMPERATURE = 0.0

# Judging conditions (Phase 1 metadata); filenames use short slug
CONDITION_FILENAME_SLUG = {
    "generic_overall": "gen",
    "metric_rubric": "metric",
    "per_item_custom": "custom",
}
DEFAULT_CONDITION_NAME = "generic_overall"
# Metadata score range; must match judge_prompt / JUDGE_RESPONSE_SCHEMA
DEFAULT_SCORE_MIN = 0
DEFAULT_SCORE_MAX = 100

# ----------------------------
logger = logging.getLogger(__name__)


def _condition_slug(name: str) -> str:
    return CONDITION_FILENAME_SLUG.get(name, name.replace("_", "")[:12])

def load_judge_prompt() -> str:
    """Load judge prompt ..."""
    return JUDGE_PROMPT_PATH.read_text(encoding=ENCODING).strip()


def load_dataset(path: Path):
    with path.open("r", encoding=ENCODING) as f:
        return json.load(f)


def run_experiment(
    judge_model=None,
    repeats=None,
    input_path=None,
    max_items=None,
    temperature=None,
    condition_name=None,
    metric_name=None,
    score_min=None,
    score_max=None,
    dataset_id=None,
):
    """
    Run repeated judging. Accepts optional overrides; otherwise uses env/defaults.
    max_items: limit to first N items (for quick tests).
    temperature: sampling temperature; default from env TEMPERATURE or TEMPERATURE constant (0.0).
    condition_name: generic_overall | metric_rubric | per_item_custom (env CONDITION_NAME).
    metric_name: for B2 metric-rubric runs; null in JSONL when unused.
    score_min, score_max: declared scale in output metadata (defaults 0–100).
    dataset_id: logical dataset id; default = input file stem (e.g. mt_bench_subset).
    Returns path to output file on success.
    """
    load_dotenv(REPO_ROOT / ".env")
    model = judge_model or os.environ.get("JUDGE_MODEL", JUDGE_MODEL)
    k = repeats if repeats is not None else int(os.environ.get("REPEATS", REPEATS))
    temp = (
        float(temperature)
        if temperature is not None
        else float(os.environ.get("TEMPERATURE", str(TEMPERATURE)))
    )
    data_path = Path(input_path) if input_path else INPUT_PATH
    cond = (condition_name or os.environ.get("CONDITION_NAME") or DEFAULT_CONDITION_NAME).strip()
    metric = (metric_name or os.environ.get("METRIC_NAME") or "").strip() or None
    smin = int(score_min) if score_min is not None else int(os.environ.get("SCORE_MIN", str(DEFAULT_SCORE_MIN)))
    smax = int(score_max) if score_max is not None else int(os.environ.get("SCORE_MAX", str(DEFAULT_SCORE_MAX)))
    dset_id = (dataset_id or os.environ.get("DATASET_ID") or data_path.stem).strip()

    if is_claude_model(model):
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env for Claude models.")
    else:
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env for OpenAI models.")

    tracer = setup_tracer()
    dataset = load_dataset(data_path)
    if max_items is not None:
        dataset = dataset[:max_items]
    judge_template = load_judge_prompt()

    execution_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    t_tag = str(temp).replace(".", "p") if "." in str(temp) else str(temp)
    cond_slug = _condition_slug(cond)
    output_path = (
        REPO_ROOT
        / "results"
        / f"mtbench_judge-{model.replace('/', '_')}_cond-{cond_slug}_K{k}_t{t_tag}_{timestamp}.jsonl"
    )
    print(f"Execution ID: {execution_id}")

    with tracer.start_as_current_span("judge_execution") as exec_span:
        exec_span.set_attribute("execution_id", execution_id)
        exec_span.set_attribute("gen_ai.request.model", model)
        exec_span.set_attribute("item_count", len(dataset))
        exec_span.set_attribute("repeats", k)
        exec_span.set_attribute("condition_name", cond)
        exec_span.set_attribute("dataset_id", dset_id)
        exec_span.set_attribute("score_min", smin)
        exec_span.set_attribute("score_max", smax)
        if metric:
            exec_span.set_attribute("metric_name", metric)

        with output_path.open("w", encoding="utf-8") as out_file:

            for item in dataset:
                item_id = item["item_id"]
                question = item["question"]
                response = item["response"]
                raw_instr = (item.get("judge_instructions") or "").strip()
                if cond == "generic_overall":
                    raw_instr = ""
                item_specific_rubric = (
                    f"Item-specific judge instructions:\n{raw_instr}\n\n"
                    if raw_instr
                    else ""
                )

                for idx in range(k):
                    prompt = judge_template.format(
                        question=question,
                        response=response,
                        item_specific_rubric=item_specific_rubric,
                    )

                    with tracer.start_as_current_span("judge_evaluate") as span:
                        span.set_attribute("item_id", str(item_id))
                        span.set_attribute("repeat_idx", idx)
                        span.set_attribute("gen_ai.request.model", model)
                        span.set_attribute("gen_ai.request.temperature", temp)

                        start_time = time.time()
                        logger.info(f"Judging item {item_id} | Repeat {idx}")
                        raw_output, input_tokens, output_tokens = call_judge(
                            prompt,
                            model,
                            system_content="You are an evaluator. Output JSON only.",
                            temperature=temp,
                        )
                        latency = int((time.time() - start_time) * 1000)

                        try:
                            parsed = json.loads(raw_output)
                        except json.JSONDecodeError:
                            parsed = extract_json_from_text(raw_output)
                        sc = parsed.get("score") if parsed else None
                        if parsed and isinstance(sc, int) and smin <= sc <= smax:
                            score = int(sc)
                            justification = str(parsed.get("justification", ""))
                            span.set_attribute("gen_ai.response.score", score)
                            span.set_status(trace.Status(trace.StatusCode.OK))
                        else:
                            print(f"⚠️ Failed to parse JSON for item {item_id}, repeat {idx}")
                            score = None
                            justification = f"PARSE_ERROR: Malformed or invalid judge output"
                            span.set_status(trace.Status(trace.StatusCode.ERROR, justification))

                        span.set_attribute("gen_ai.usage.input_tokens", input_tokens or 0)
                        span.set_attribute("gen_ai.usage.output_tokens", output_tokens or 0)
                        span.set_attribute("latency_ms", latency)
                        trace_id, span_id = get_trace_context()

                    result = {
                        "execution_id": execution_id,
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "item_id": item_id,
                        "idx": idx,
                        "condition_name": cond,
                        "metric_name": metric,
                        "dataset_id": dset_id,
                        "score_min": smin,
                        "score_max": smax,
                        "temperature": temp,
                        "judge_instructions": raw_instr if raw_instr else None,
                        "judge_model": model,
                        "score": score,
                        "justification": justification,
                        "latency_ms": latency,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "span_status": "ok" if score is not None else "error",
                        "span_status_message": None if score is not None else justification,
                        "created_at": datetime.utcnow().isoformat() + "Z"
                    }
                    out_file.write(json.dumps(result) + "\n")
                    print(f"Item {item_id} | Repeat {idx} | Score: {score}")

    print(f"\nDone.... {output_path}")
    return str(output_path)


def main():
    run_experiment()


if __name__ == "__main__":
    main()