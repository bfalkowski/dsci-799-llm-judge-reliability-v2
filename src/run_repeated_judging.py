import json
import os
import sys
import uuid
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Set, Tuple

from dotenv import load_dotenv

from opentelemetry import trace

from constants import JUDGE_MODEL
from judge import call_judge, extract_json_from_text, is_claude_model
from metric_rubric import gloss_for_metric
from otel_setup import setup_tracer, get_trace_context
from utils import ENCODING, REPO_ROOT, load_jsonl

# ---------- CONFIG ----------
INPUT_PATH = REPO_ROOT / "data" / "mt_bench_subset.json"

JUDGE_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "judge_prompt.txt"
JUDGE_PROMPT_METRIC_PATH = Path(__file__).resolve().parent / "prompts" / "judge_prompt_metric.txt"

REPEATS = 5
TEMPERATURE = 0.0

# Default metrics when METRIC_NAMES env / metric_names arg absent (condition B)
DEFAULT_METRICS_RUBRIC = ["accuracy", "relevance", "completeness"]

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


def load_judge_metric_prompt() -> str:
    """Load single-metric (condition B) judge prompt template."""
    return JUDGE_PROMPT_METRIC_PATH.read_text(encoding=ENCODING).strip()


def load_dataset(path: Path):
    with path.open("r", encoding=ENCODING) as f:
        return json.load(f)


def _judgment_identity(row: dict) -> Tuple:
    """Stable key for one judgment slot (must match loop order / resume)."""
    cond = str(row.get("condition_name") or "")
    jm = str(row.get("judge_model") or "")
    iid = str(row.get("item_id") or "")
    idx = int(row["idx"])
    if cond == "metric_rubric":
        m = str(row.get("metric_name") or "")
        return ("metric_rubric", jm, iid, idx, m)
    return (cond, jm, iid, idx)


def _prepare_resume(
    path: Path,
    *,
    cond: str,
    dset_id: str,
    k: int,
    temp: float,
    smin: int,
    smax: int,
    models_to_run: list,
    metrics_list: list,
    expected_rows: int,
) -> Tuple[Set[Tuple], str, bool, int]:
    """
    Load existing JSONL; validate it matches this run config; return judgment keys already present.
    """
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"Resume file is empty: {path}")

    first = rows[0]
    eid = str(first.get("execution_id") or "").strip()
    if not eid:
        raise ValueError("Resume file has no execution_id on first row.")

    def _bad(field: str, got, want) -> None:
        raise ValueError(
            f"Resume file metadata mismatch ({field}): file has {got!r}, this run uses {want!r}. "
            "Use the same dataset, condition, K, temperature, and judge list as the original run."
        )

    if str(first.get("condition_name") or "") != cond:
        _bad("condition_name", first.get("condition_name"), cond)
    if str(first.get("dataset_id") or "").strip() != str(dset_id).strip():
        _bad("dataset_id", first.get("dataset_id"), dset_id)
    if float(first.get("temperature", -99999.0)) != float(temp):
        _bad("temperature", first.get("temperature"), temp)
    if int(first.get("score_min", -1)) != int(smin):
        _bad("score_min", first.get("score_min"), smin)
    if int(first.get("score_max", -1)) != int(smax):
        _bad("score_max", first.get("score_max"), smax)

    model_set = {str(m) for m in models_to_run}
    keys_in_order: list = []
    for r in rows:
        if str(r.get("execution_id") or "").strip() != eid:
            raise ValueError(
                "Resume file contains multiple execution_id values; use a single-run JSONL only."
            )
        if str(r.get("judge_model") or "") not in model_set:
            raise ValueError(
                f"Resume row uses judge_model={r.get('judge_model')!r} not in this run's model list."
            )
        idx_i = int(r["idx"])
        if idx_i < 0 or idx_i >= k:
            raise ValueError(f"Resume file has invalid idx={idx_i} for K={k}.")
        if cond == "metric_rubric":
            mn = str(r.get("metric_name") or "")
            if mn not in metrics_list:
                raise ValueError(
                    f"Resume file metric_name={mn!r} not in this run's metrics_list={metrics_list}."
                )
        keys_in_order.append(_judgment_identity(r))

    if len(keys_in_order) != len(set(keys_in_order)):
        raise ValueError(
            "Resume file has duplicate judgment keys (same judge, item, idx, metric). "
            "Remove duplicates or use a clean file."
        )

    if len(rows) >= expected_rows:
        raise ValueError(
            f"Resume file already has {len(rows)} rows (expected {expected_rows} total); nothing to append."
        )

    return set(keys_in_order), eid, bool(first.get("multi_judge_run")), len(rows)


def _ensure_api_keys_for_models(models: list) -> None:
    """Require OpenAI and/or Anthropic keys if any selected judge needs them."""
    need_oai = any(not is_claude_model(str(m)) for m in models)
    need_ant = any(is_claude_model(str(m)) for m in models)
    if need_oai and not (os.environ.get("OPENAI_API_KEY") or "").strip():
        raise RuntimeError(
            "OPENAI_API_KEY is not set in .env but at least one judge model is an OpenAI model."
        )
    if need_ant and not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set in .env but at least one judge model is a Claude model."
        )


def run_experiment(
    judge_model=None,
    judge_models=None,
    repeats=None,
    input_path=None,
    max_items=None,
    temperature=None,
    condition_name=None,
    metric_name=None,
    metric_names=None,
    score_min=None,
    score_max=None,
    dataset_id=None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    resume_path: Optional[str] = None,
):
    """
    Run repeated judging. Accepts optional overrides; otherwise uses env/defaults.
    max_items: limit to first N items (for quick tests).
    temperature: sampling temperature; default from env TEMPERATURE or TEMPERATURE constant (0.0).
    condition_name: generic_overall | metric_rubric | per_item_custom (env CONDITION_NAME).
    metric_name: legacy single metric; use metric_names when possible.
    metric_names: for condition B — list of metric labels; one judge call per (item, repeat, metric).
    score_min, score_max: declared scale in output metadata (defaults 0–100).
    dataset_id: logical dataset id; default = input file stem (e.g. mt_bench_subset).
    judge_models: if a non-empty list, run each model in sequence and append every row to **one** JSONL
        (each line has ``judge_model`` set). ``judge_model`` is ignored when this is set.
    progress_callback: if set, invoked as ``(completed_count, expected_total)`` after each written row
        and once before work begins with ``(already_on_disk, expected_total)`` when resuming, else
        ``(0, expected_total)``.
    **Repeat schedule:** for each judge model, API calls use **round-robin** over repeats (all items at
    ``idx=0``, then all at ``idx=1``, …) so the same prompt is not sent **K** times consecutively.
    Rows still record the correct ``idx`` per judgment.
    resume_path: if set, append **missing** judgments to this JSONL only (same ``execution_id``,
        metadata must match). Skips already-present (judge, item, idx[, metric]) slots.
    On success returns a dict with output_path, expected_rows, written_rows, execution_id,
        resumed (bool), skipped_existing (int), session_new_rows (int).
    Raises RuntimeError if the JSONL row count or parseable records do not match the expected total.
    """
    load_dotenv(REPO_ROOT / ".env")
    k = repeats if repeats is not None else int(os.environ.get("REPEATS", REPEATS))
    temp = (
        float(temperature)
        if temperature is not None
        else float(os.environ.get("TEMPERATURE", str(TEMPERATURE)))
    )
    data_path = Path(input_path) if input_path else INPUT_PATH
    cond = (condition_name or os.environ.get("CONDITION_NAME") or DEFAULT_CONDITION_NAME).strip()
    legacy_metric = (metric_name or os.environ.get("METRIC_NAME") or "").strip() or None
    smin = int(score_min) if score_min is not None else int(os.environ.get("SCORE_MIN", str(DEFAULT_SCORE_MIN)))
    smax = int(score_max) if score_max is not None else int(os.environ.get("SCORE_MAX", str(DEFAULT_SCORE_MAX)))
    dset_id = (dataset_id or os.environ.get("DATASET_ID") or data_path.stem).strip()

    if judge_models is not None and len(judge_models) > 0:
        models_to_run = [str(m).strip() for m in judge_models if str(m).strip()]
        if not models_to_run:
            raise ValueError("judge_models produced an empty list after filtering blanks.")
    else:
        models_to_run = [judge_model or os.environ.get("JUDGE_MODEL", JUDGE_MODEL)]

    _ensure_api_keys_for_models(models_to_run)

    tracer = setup_tracer()
    dataset = load_dataset(data_path)
    if max_items is not None:
        dataset = dataset[:max_items]
    judge_template = load_judge_prompt()
    metric_template = load_judge_metric_prompt()

    metrics_list: list = []
    if cond == "metric_rubric":
        if metric_names is not None:
            metrics_list = [str(m).strip() for m in metric_names if str(m).strip()]
        elif legacy_metric:
            metrics_list = [legacy_metric]
        else:
            raw = (os.environ.get("METRIC_NAMES") or "").strip()
            if raw:
                metrics_list = [x.strip() for x in raw.split(",") if x.strip()]
            else:
                metrics_list = list(DEFAULT_METRICS_RUBRIC)
        if not metrics_list:
            raise ValueError(
                "condition metric_rubric requires at least one metric name "
                "(set metric_names, METRIC_NAMES in .env, or METRIC_NAME for a single metric)."
            )

    n_items = len(dataset)
    n_models = len(models_to_run)
    if cond == "metric_rubric":
        expected_rows = n_items * k * len(metrics_list) * n_models
    else:
        expected_rows = n_items * k * n_models
    if expected_rows <= 0:
        raise ValueError(
            "No judgments to run (empty dataset or zero repeats). "
            f"items={n_items}, K={k}, models={n_models}."
        )

    done_keys: Set[Tuple] = set()
    initial_line_count = 0
    resumed = False
    if resume_path:
        resumed = True
        resume_p = Path(resume_path)
        if not resume_p.is_absolute():
            resume_p = (REPO_ROOT / "results" / resume_p.name).resolve()
        if not resume_p.is_file():
            raise FileNotFoundError(f"Resume file not found: {resume_p}")
        done_keys, execution_id, multi_judge, initial_line_count = _prepare_resume(
            resume_p,
            cond=cond,
            dset_id=dset_id,
            k=k,
            temp=temp,
            smin=smin,
            smax=smax,
            models_to_run=models_to_run,
            metrics_list=metrics_list,
            expected_rows=expected_rows,
        )
        if (len(models_to_run) > 1) != multi_judge:
            raise ValueError(
                "Resume file multi_judge_run does not match this run's number of judge models."
            )
        output_path = resume_p
        print(f"Resuming execution_id={execution_id} — {initial_line_count} rows on disk, appending…")
    else:
        execution_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        t_tag = str(temp).replace(".", "p") if "." in str(temp) else str(temp)
        cond_slug = _condition_slug(cond)
        _fname_model = (
            models_to_run[0].replace("/", "_")
            if len(models_to_run) == 1
            else f"multi{len(models_to_run)}judges"
        )
        output_path = REPO_ROOT / "results" / f"mtbench_judge-{_fname_model}_cond-{cond_slug}_K{k}_t{t_tag}_{timestamp}.jsonl"
        multi_judge = len(models_to_run) > 1
        print(f"Execution ID: {execution_id}")

    file_mode = "a" if resumed else "w"
    skipped_existing = len(done_keys)
    session_new_rows = 0

    with tracer.start_as_current_span("judge_execution") as exec_span:
        exec_span.set_attribute("execution_id", execution_id)
        exec_span.set_attribute("gen_ai.request.model", ",".join(models_to_run))
        exec_span.set_attribute("item_count", len(dataset))
        exec_span.set_attribute("repeats", k)
        exec_span.set_attribute("condition_name", cond)
        exec_span.set_attribute("dataset_id", dset_id)
        exec_span.set_attribute("score_min", smin)
        exec_span.set_attribute("score_max", smax)
        exec_span.set_attribute("multi_judge_run", multi_judge)
        if cond == "metric_rubric":
            exec_span.set_attribute("metric_names", ",".join(metrics_list))
        elif legacy_metric:
            exec_span.set_attribute("metric_name", legacy_metric)
        exec_span.set_attribute("expected_output_rows", expected_rows)
        exec_span.set_attribute("resume_appended_from", initial_line_count if resumed else 0)

        def _write_judge_row(
            out_f,
            *,
            judge_model_used: str,
            item_id,
            idx,
            m_metric,
            raw_instr_value,
            prompt_text,
            log_label: str,
            progress_position: int,
        ):
            nonlocal session_new_rows
            with tracer.start_as_current_span("judge_evaluate") as span:
                span.set_attribute("item_id", str(item_id))
                span.set_attribute("repeat_idx", idx)
                span.set_attribute("gen_ai.request.model", judge_model_used)
                span.set_attribute("gen_ai.request.temperature", temp)
                if m_metric is not None:
                    span.set_attribute("metric_name", str(m_metric))

                start_time = time.time()
                logger.info(log_label)
                raw_output, input_tokens, output_tokens = call_judge(
                    prompt_text,
                    judge_model_used,
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
                    print(f"⚠️ Failed to parse JSON for {log_label}")
                    score = None
                    justification = "PARSE_ERROR: Malformed or invalid judge output"
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
                "metric_name": m_metric,
                "dataset_id": dset_id,
                "score_min": smin,
                "score_max": smax,
                "temperature": temp,
                "judge_instructions": raw_instr_value if raw_instr_value else None,
                "judge_model": judge_model_used,
                "multi_judge_run": multi_judge,
                "score": score,
                "justification": justification,
                "latency_ms": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "span_status": "ok" if score is not None else "error",
                "span_status_message": None if score is not None else justification,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            session_new_rows += 1
            if progress_callback:
                progress_callback(progress_position, expected_rows)
            print(f"{log_label} | Score: {score}")

        try:
            with output_path.open(file_mode, encoding="utf-8") as out_file:
                if progress_callback:
                    progress_callback(initial_line_count, expected_rows)
                processed = 0
                # Round-robin repeats: for each repeat index, hit all items (then next idx).
                for model in models_to_run:
                    if cond == "metric_rubric":
                        for idx in range(k):
                            for item in dataset:
                                item_id = item["item_id"]
                                question = item["question"]
                                response = item["response"]
                                for m in metrics_list:
                                    processed += 1
                                    jk = ("metric_rubric", str(model), str(item_id), idx, str(m))
                                    if jk in done_keys:
                                        if progress_callback:
                                            progress_callback(processed, expected_rows)
                                        continue
                                    gloss = gloss_for_metric(m)
                                    prompt = metric_template.format(
                                        metric_name=m,
                                        metric_gloss=gloss,
                                        question=question,
                                        response=response,
                                    )
                                    _write_judge_row(
                                        out_file,
                                        judge_model_used=model,
                                        item_id=item_id,
                                        idx=idx,
                                        m_metric=m,
                                        raw_instr_value="",
                                        prompt_text=prompt,
                                        log_label=f"{model} | Item {item_id} | Metric {m} | R{idx}",
                                        progress_position=processed,
                                    )
                    else:
                        for idx in range(k):
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

                                processed += 1
                                jk = (cond, str(model), str(item_id), idx)
                                if jk in done_keys:
                                    if progress_callback:
                                        progress_callback(processed, expected_rows)
                                    continue

                                prompt = judge_template.format(
                                    question=question,
                                    response=response,
                                    item_specific_rubric=item_specific_rubric,
                                )
                                _write_judge_row(
                                    out_file,
                                    judge_model_used=model,
                                    item_id=item_id,
                                    idx=idx,
                                    m_metric=None,
                                    raw_instr_value=raw_instr,
                                    prompt_text=prompt,
                                    log_label=f"{model} | Item {item_id} | R{idx}",
                                    progress_position=processed,
                                )
        except Exception as e:
            partial_n = 0
            if output_path.is_file():
                try:
                    partial_n = len(load_jsonl(output_path))
                except Exception:
                    pass
            el = str(e).lower()
            quota_hint = ""
            if (
                "usage limits" in el
                or "regain access" in el
                or "insufficient_quota" in el
                or "billing_hard_limit" in el
                or ("exceeded your" in el and "quota" in el)
            ):
                quota_hint = (
                    "Your provider reported a usage or billing cap (this is not a bug in the dashboard). "
                )
            raise RuntimeError(
                f"{quota_hint}"
                f"Run stopped early after writing {partial_n} complete row(s) to:\n"
                f"  {output_path}\n"
                f"(full run needs {expected_rows} rows). When access is back, use "
                f"Resume partial JSONL with the same dataset, condition, K, temperature, and judges, "
                f"and select that file.\n\n"
                f"Details: {e}"
            ) from e

    written_rows = len(load_jsonl(output_path))
    if written_rows != expected_rows:
        raise RuntimeError(
            f"JSONL validation failed: expected {expected_rows} records, "
            f"found {written_rows} parseable lines in {output_path} "
            f"(execution_id={execution_id})."
        )
    if initial_line_count + session_new_rows != written_rows:
        raise RuntimeError(
            f"Row count mismatch after resume: had {initial_line_count} + wrote {session_new_rows} "
            f"but file has {written_rows} (execution_id={execution_id})."
        )

    print(f"\nDone.... {output_path} ({written_rows} rows)")
    if resumed:
        print(f"  (resumed: skipped {skipped_existing} existing slots, new API rows {session_new_rows})")
    return {
        "output_path": str(output_path),
        "expected_rows": expected_rows,
        "written_rows": written_rows,
        "execution_id": execution_id,
        "resumed": resumed,
        "skipped_existing": skipped_existing,
        "session_new_rows": session_new_rows,
    }


def main():
    resume = sys.argv[1] if len(sys.argv) > 1 else None
    r = run_experiment(resume_path=resume)
    print(r["output_path"])
    if r.get("resumed"):
        print(
            f"Resumed: +{r['session_new_rows']} new rows "
            f"({r['skipped_existing']} slots already on disk)."
        )


if __name__ == "__main__":
    main()