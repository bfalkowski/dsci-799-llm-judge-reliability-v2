Do not commit raw datasets. Document sources here.

## MT-Bench

- **Raw** (gitignored): `raw/mt_bench/question.jsonl`, `raw/mt_bench/reference_answer/gpt-4.jsonl`
- **Subset** (5 items): `mt_bench_subset.json` – manual slice for quick tests
- **Full** (30 items): `mt_bench_full.json` – built from raw via `python src/build_mt_bench_full.py`
