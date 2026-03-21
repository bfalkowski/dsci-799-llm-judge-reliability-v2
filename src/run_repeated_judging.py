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

# ----------------------------
logger = logging.getLogger(__name__)

def load_judge_prompt() -> str:
    """Load judge prompt ..."""
    return JUDGE_PROMPT_PATH.read_text(encoding=ENCODING).strip()


def load_dataset(path: Path):
    with path.open("r", encoding=ENCODING) as f:
        return json.load(f)


def run_experiment(judge_model=None, repeats=None, input_path=None, max_items=None):
    """
    Run repeated judging. Accepts optional overrides; otherwise uses env/defaults.
    max_items: limit to first N items (for quick tests).
    Returns path to output file on success.
    """
    load_dotenv(REPO_ROOT / ".env")
    model = judge_model or os.environ.get("JUDGE_MODEL", JUDGE_MODEL)
    k = repeats if repeats is not None else int(os.environ.get("REPEATS", REPEATS))
    data_path = Path(input_path) if input_path else INPUT_PATH

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
    output_path = REPO_ROOT / "results" / f"mtbench_judge-{model.replace('/', '_')}_K{k}_t{TEMPERATURE}_{timestamp}.jsonl"
    print(f"Execution ID: {execution_id}")

    with tracer.start_as_current_span("judge_execution") as exec_span:
        exec_span.set_attribute("execution_id", execution_id)
        exec_span.set_attribute("gen_ai.request.model", model)
        exec_span.set_attribute("item_count", len(dataset))
        exec_span.set_attribute("repeats", k)

        with output_path.open("w", encoding="utf-8") as out_file:

            for item in dataset:
                item_id = item["item_id"]
                question = item["question"]
                response = item["response"]

                for idx in range(k):
                    prompt = judge_template.format(question=question, response=response)

                    with tracer.start_as_current_span("judge_evaluate") as span:
                        span.set_attribute("item_id", str(item_id))
                        span.set_attribute("repeat_idx", idx)
                        span.set_attribute("gen_ai.request.model", model)
                        span.set_attribute("gen_ai.request.temperature", TEMPERATURE)

                        start_time = time.time()
                        logger.info(f"Judging item {item_id} | Repeat {idx}")
                        raw_output, input_tokens, output_tokens = call_judge(
                            prompt, model, system_content="You are an evaluator. Output JSON only."
                        )
                        latency = int((time.time() - start_time) * 1000)

                        try:
                            parsed = json.loads(raw_output)
                        except json.JSONDecodeError:
                            parsed = extract_json_from_text(raw_output)
                        if parsed and isinstance(parsed.get("score"), int) and 1 <= parsed.get("score", 0) <= 10:
                            score = int(parsed["score"])
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