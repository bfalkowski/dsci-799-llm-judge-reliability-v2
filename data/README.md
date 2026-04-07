Do not commit raw datasets. Document sources here.

## Judge-ready datasets (`mt_bench*.json`)

JSON array of objects:

- `item_id` (string)
- `question` (string)
- `response` (string)
- `judge_instructions` (string, optional) — per-item instructions injected into the judge prompt when non-empty (condition C / custom rubric). Omit or `""` for generic judging only.

Build full set from raw MT-Bench: `cd src && python build_mt_bench_full.py` → `mt_bench_full.json` (requires `data/raw/mt_bench/...`).

## Sample OpenAI line-item costs (`samples/openai_cost_line_items_*.csv`)

For **Reliability × economics** “Export $ OpenAI (line items)”, upload a CSV with columns `usage_date_utc`, `model`, `usage_type`, `cost_usd` (one row per model × usage; sums roll up per model). See `samples/openai_cost_line_items_2026-04-05.csv`.
