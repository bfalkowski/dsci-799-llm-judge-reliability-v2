import json
import uuid
import time
import logging
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from constants import JUDGE_MODEL

# ---------- CONFIG ----------
REPOROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = REPOROOT / "data" / "mt_bench_subset.json"

JUDGE_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "judge_prompt.txt"

REPEATS = 1
TEMPERATURE = 0.0  

#OUTPUT_PATH = REPOROOT / "results" / "exp01_repeated_judging.jsonl"

timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
OUTPUT_PATH = Path(f"results/mtbench_judge-{JUDGE_MODEL}_K{REPEATS}_t{TEMPERATURE}_{timestamp}.jsonl")

# ----------------------------
logger = logging.getLogger(__name__)

def load_judge_prompt() -> str:
    """Load judge prompt ..."""
    return JUDGE_PROMPT_PATH.read_text(encoding="utf-8").strip()


def load_dataset(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    load_dotenv(REPOROOT / ".env")
    client = OpenAI()

    dataset = load_dataset(INPUT_PATH)
    judge_template = load_judge_prompt()
    

    execution_id = str(uuid.uuid4())
    print(f"Execution ID: {execution_id}")

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

                start_time = time.time()

                #Sends a request to the model and returns a completion.
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

                try:
                    parsed = json.loads(raw_output)
                    score = int(parsed["score"])
                    justification = parsed["justification"]
                except Exception as e:
                    print(f"⚠️ Failed to parse JSON for item {item_id}, repeat {idx}")
                    print(raw_output)
                    score = None
                    justification = f"PARSE_ERROR: {str(e)}"

                result = {
                    "execution_id": execution_id,

                    "item_id": item_id,
                    "idx": idx,
                    "judge_model": JUDGE_MODEL,
                    "score": score,
                    "justification": justification,

                    "latency_ms": latency,
                    "created_at": datetime.utcnow().isoformat() + "Z"
                }

                out_file.write(json.dumps(result) + "\n")
                print(f"Item {item_id} | Repeat {idx} | Score: {score}")

    print(f"\nDone.... {OUTPUT_PATH}")


if __name__ == "__main__":
    main()