Do not commit raw datasets. Document sources here.

## Judge-ready datasets (`mt_bench*.json`)

JSON array of objects:

- `item_id` (string)
- `question` (string)
- `response` (string)
- `judge_instructions` (string, optional) — per-item instructions injected into the judge prompt when non-empty (condition C / custom rubric). Omit or `""` for generic judging only.

Build full set from raw MT-Bench: `cd src && python build_mt_bench_full.py` → `mt_bench_full.json` (requires `data/raw/mt_bench/...`).
