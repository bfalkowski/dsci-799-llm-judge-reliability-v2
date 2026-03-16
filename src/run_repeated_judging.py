import json
import uuid
import time
import logging
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from opentelemetry import trace

from constants import JUDGE_MODEL
from otel_setup import setup_tracer, get_trace_context
from utils import ENCODING, REPO_ROOT

# ---------- CONFIG ----------
INPUT_PATH = REPO_ROOT / "data" / "mt_bench_subset.json"

JUDGE_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "judge_prompt.txt"

REPEATS = 5
TEMPERATURE = 0.0  

#OUTPUT_PATH = REPO_ROOT / "results" / "exp01_repeated_judging.jsonl"

timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
OUTPUT_PATH = REPO_ROOT / "results" / f"mtbench_judge-{JUDGE_MODEL}_K{REPEATS}_t{TEMPERATURE}_{timestamp}.jsonl"

# ----------------------------
logger = logging.getLogger(__name__)

def load_judge_prompt() -> str:
    """Load judge prompt ..."""
    return JUDGE_PROMPT_PATH.read_text(encoding=ENCODING).strip()


def load_dataset(path: Path):
    with path.open("r", encoding=ENCODING) as f:
        return json.load(f)


def main():
    load_dotenv(REPO_ROOT / ".env")
    client = OpenAI()
    tracer = setup_tracer()

    dataset = load_dataset(INPUT_PATH)
    judge_template = load_judge_prompt()

    execution_id = str(uuid.uuid4())
    print(f"Execution ID: {execution_id}")

    with tracer.start_as_current_span("judge_execution") as exec_span:
        exec_span.set_attribute("execution_id", execution_id)
        exec_span.set_attribute("gen_ai.request.model", JUDGE_MODEL)
        exec_span.set_attribute("item_count", len(dataset))
        exec_span.set_attribute("repeats", REPEATS)

        with OUTPUT_PATH.open("w", encoding="utf-8") as out_file:

            for item in dataset:
                item_id = item["item_id"]
                question = item["question"]
                response = item["response"]

                for idx in range(REPEATS):

                    prompt = judge_template.format(
                        question=question,
                        response=response
                    )

                    with tracer.start_as_current_span("judge_evaluate") as span:
                        span.set_attribute("item_id", str(item_id))
                        span.set_attribute("repeat_idx", idx)
                        span.set_attribute("gen_ai.request.model", JUDGE_MODEL)
                        span.set_attribute("gen_ai.request.temperature", TEMPERATURE)

                        start_time = time.time()

                        logger.info(f"Judging item {item_id} | Repeat {idx} | Prompt: {prompt}")
                        completion = client.chat.completions.create(
                            model=JUDGE_MODEL,
                            temperature=TEMPERATURE,
                            messages=[
                                {"role": "system", "content": "You are an evaluator. Output JSON only."},
                                {"role": "user", "content": prompt}
                            ]
                        )

                        latency = int((time.time() - start_time) * 1000)
                        raw_output = completion.choices[0].message.content.strip()

                        # OTEL: token usage (for reliability metrics)
                        usage = completion.usage
                        input_tokens = usage.prompt_tokens if usage else None
                        output_tokens = usage.completion_tokens if usage else None

                        try:
                            parsed = json.loads(raw_output)
                            score = int(parsed["score"])
                            justification = parsed["justification"]
                            span.set_attribute("gen_ai.response.score", score)
                            span.set_status(trace.Status(trace.StatusCode.OK))
                        except Exception as e:
                            print(f"⚠️ Failed to parse JSON for item {item_id}, repeat {idx}")
                            print(raw_output)
                            score = None
                            justification = f"PARSE_ERROR: {str(e)}"
                            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))

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
                        "judge_model": JUDGE_MODEL,
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

    print(f"\nDone.... {OUTPUT_PATH}")


if __name__ == "__main__":
    main()