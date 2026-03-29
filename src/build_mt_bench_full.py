"""
Build mt_bench_full.json from raw MT-Bench data.

Sources:
  data/raw/mt_bench/question.jsonl
  data/raw/mt_bench/reference_answer/gpt-4.jsonl

Each item includes judge_instructions (empty string) for per-item custom judge text.
"""

import json
from pathlib import Path

from utils import ENCODING, REPO_ROOT

QUESTIONS_PATH = REPO_ROOT / "data" / "raw" / "mt_bench" / "question.jsonl"
REFERENCES_PATH = REPO_ROOT / "data" / "raw" / "mt_bench" / "reference_answer" / "gpt-4.jsonl"
OUTPUT_PATH = REPO_ROOT / "data" / "mt_bench_full.json"


def main():
    questions = {}
    with QUESTIONS_PATH.open("r", encoding=ENCODING) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj["question_id"]
            turns = obj.get("turns", [])
            questions[qid] = "\n".join(turns) if turns and isinstance(turns[0], str) else "\n".join(str(t) for t in turns)

    result = []
    with REFERENCES_PATH.open("r", encoding=ENCODING) as f:
        for line in f:
            if not line.strip():
                continue
            ref = json.loads(line)
            qid = ref["question_id"]
            if qid not in questions:
                continue
            turns = ref.get("choices", [{}])[0].get("turns", [])
            response = "\n".join(turns) if turns and isinstance(turns[0], str) else "\n".join(str(t) for t in turns)
            result.append({
                "item_id": str(qid),
                "question": questions[qid],
                "response": response,
                "judge_instructions": "",
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding=ENCODING) as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {len(result)} items to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
