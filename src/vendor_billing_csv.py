"""
Parse vendor billing / usage CSV exports for pairing with judge JSONL runs.

Supported shapes (detected by header row):
- OpenAI project daily cost: amount_value, amount_currency
- OpenAI line-item costs (dashboard-style): model, usage_type, cost_usd (optional usage_date_utc)
- OpenAI completions usage: num_model_requests, input_tokens, output_tokens, model
- Anthropic API tokens (hourly): model_version, usage_input_tokens_no_cache, usage_output_tokens
- Anthropic API cost: model, token_type, cost_usd
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class VendorBillingSummary:
    openai_total_usd: Optional[float] = None
    anthropic_total_usd: Optional[float] = None
    openai_by_model: dict = field(default_factory=dict)  # model_id -> {requests, input, output}
    openai_line_cost_by_model: dict = field(default_factory=dict)  # export model id -> USD (all usage types)
    openai_line_items: List[dict] = field(default_factory=list)  # raw rows for display
    anthropic_tokens_by_model: dict = field(default_factory=dict)  # api id -> {in, out}
    anthropic_cost_by_model: dict = field(default_factory=dict)  # display name -> usd
    source_files: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _sniff_kind(fieldnames: Optional[List[str]], sample_filename: str) -> str:
    if not fieldnames:
        return "unknown"
    fn = [x.strip().lower() for x in fieldnames if x]
    s = set(fn)
    name_l = sample_filename.lower()

    if "amount_value" in s and "amount_currency" in s:
        return "openai_project_cost"
    if (
        "cost_usd" in s
        and "model" in s
        and "usage_type" in s
        and "list_price_usd" not in s
    ):
        return "openai_line_item_cost"
    if "num_model_requests" in s and "input_tokens" in s and "output_tokens" in s:
        return "openai_completions_usage"
    if "usage_input_tokens_no_cache" in s and "model_version" in s and "usage_output_tokens" in s:
        return "anthropic_tokens"
    if "cost_usd" in s and "token_type" in s and "list_price_usd" in s:
        return "anthropic_cost"
    return "unknown"


def parse_csv_text(text: str, filename: str = "") -> Tuple[str, VendorBillingSummary]:
    """Parse one CSV string; returns (kind, partial summary to merge)."""
    partial = VendorBillingSummary(source_files=[filename] if filename else [])
    fio = io.StringIO(text)
    reader = csv.DictReader(fio)
    fieldnames = reader.fieldnames
    kind = _sniff_kind(fieldnames, filename)
    rows = list(reader)
    if not rows and kind == "unknown":
        partial.notes.append(f"{filename or 'file'}: could not detect CSV type (no rows or unknown headers).")
        return "unknown", partial

    if kind == "openai_project_cost":
        total = 0.0
        for row in rows:
            try:
                total += float(row.get("amount_value") or 0)
            except (TypeError, ValueError):
                continue
        partial.openai_total_usd = total if total > 0 else partial.openai_total_usd
        return kind, partial

    if kind == "openai_completions_usage":
        by_m: dict = {}
        for row in rows:
            m = str(row.get("model") or "").strip()
            if not m:
                continue
            try:
                req = int(float(row.get("num_model_requests") or 0))
                inp = int(float(row.get("input_tokens") or 0))
                out = int(float(row.get("output_tokens") or 0))
            except (TypeError, ValueError):
                continue
            if m not in by_m:
                by_m[m] = {"requests": 0, "input": 0, "output": 0}
            by_m[m]["requests"] += req
            by_m[m]["input"] += inp
            by_m[m]["output"] += out
        partial.openai_by_model = by_m
        return kind, partial

    if kind == "openai_line_item_cost":
        by_m: dict = {}
        total = 0.0
        items: List[dict] = []
        for row in rows:
            m = str(row.get("model") or "").strip()
            if not m:
                continue
            try:
                c = float(row.get("cost_usd") or 0)
            except (TypeError, ValueError):
                continue
            total += c
            by_m[m] = by_m.get(m, 0.0) + c
            items.append({
                "usage_date_utc": (row.get("usage_date_utc") or "").strip(),
                "model": m,
                "usage_type": str(row.get("usage_type") or "").strip(),
                "cost_usd": round(c, 6),
            })
        partial.openai_line_cost_by_model = by_m
        partial.openai_line_items = items
        partial.openai_total_usd = total if total > 0 else None
        return kind, partial

    if kind == "anthropic_tokens":
        by_m: dict = {}
        for row in rows:
            m = str(row.get("model_version") or "").strip()
            if not m:
                continue
            try:
                inp = int(float(row.get("usage_input_tokens_no_cache") or 0))
                incr = sum(
                    int(float(row.get(k) or 0))
                    for k in (
                        "usage_input_tokens_cache_write_5m",
                        "usage_input_tokens_cache_write_1h",
                        "usage_input_tokens_cache_read",
                    )
                )
                inp += incr
                out = int(float(row.get("usage_output_tokens") or 0))
            except (TypeError, ValueError):
                continue
            if m not in by_m:
                by_m[m] = {"in": 0, "out": 0}
            by_m[m]["in"] += inp
            by_m[m]["out"] += out
        partial.anthropic_tokens_by_model = by_m
        return kind, partial

    if kind == "anthropic_cost":
        total = 0.0
        by_disp: dict = {}
        for row in rows:
            try:
                c = float(row.get("cost_usd") or 0)
            except (TypeError, ValueError):
                continue
            total += c
            disp = str(row.get("model") or "Anthropic").strip() or "Anthropic"
            by_disp[disp] = by_disp.get(disp, 0.0) + c
        partial.anthropic_total_usd = total if total > 0 else None
        partial.anthropic_cost_by_model = by_disp
        return kind, partial

    partial.notes.append(f"{filename or 'file'}: unhandled kind **{kind}** with {len(rows)} row(s).")
    return kind, partial


def merge_summaries(parts: List[Tuple[str, VendorBillingSummary]]) -> VendorBillingSummary:
    out = VendorBillingSummary()
    for kind, p in parts:
        out.source_files.extend(p.source_files)
        out.notes.extend(p.notes)
        # Invoice totals: first non-null wins to avoid double-counting duplicate uploads.
        if p.openai_total_usd is not None:
            if out.openai_total_usd is None:
                out.openai_total_usd = p.openai_total_usd
            else:
                out.notes.append("Ignored extra OpenAI total (already set; use one project or line-item cost file).")
        if p.anthropic_total_usd is not None:
            if out.anthropic_total_usd is None:
                out.anthropic_total_usd = p.anthropic_total_usd
            else:
                out.notes.append("Ignored extra Anthropic cost-total aggregate (already set).")
        for m, v in p.openai_by_model.items():
            if m not in out.openai_by_model:
                out.openai_by_model[m] = {"requests": 0, "input": 0, "output": 0}
            for k in ("requests", "input", "output"):
                out.openai_by_model[m][k] += v.get(k, 0)
        for m, v in p.openai_line_cost_by_model.items():
            out.openai_line_cost_by_model[m] = out.openai_line_cost_by_model.get(m, 0.0) + float(v)
        out.openai_line_items.extend(p.openai_line_items)
        for m, v in p.anthropic_tokens_by_model.items():
            if m not in out.anthropic_tokens_by_model:
                out.anthropic_tokens_by_model[m] = {"in": 0, "out": 0}
            out.anthropic_tokens_by_model[m]["in"] += v["in"]
            out.anthropic_tokens_by_model[m]["out"] += v["out"]
        for m, v in p.anthropic_cost_by_model.items():
            out.anthropic_cost_by_model[m] = out.anthropic_cost_by_model.get(m, 0.0) + v

    if out.openai_line_cost_by_model:
        line_sum = sum(float(x) for x in out.openai_line_cost_by_model.values())
        if line_sum > 0:
            prev = out.openai_total_usd
            out.openai_total_usd = line_sum
            if prev is not None and abs(float(prev) - line_sum) > 0.02:
                out.notes.append(
                    f"OpenAI: line-item costs sum to **{line_sum:.2f}**; project-level row was **{float(prev):.2f}** — "
                    "**Apply** uses the line-item sum."
                )
    return out


def parse_uploaded_files(uploaded: List[object]) -> VendorBillingSummary:
    """
    Each item is a Streamlit UploadedFile-like object with .name and .getvalue() or .read().
    """
    pieces: List[Tuple[str, VendorBillingSummary]] = []
    for uf in uploaded:
        name = getattr(uf, "name", "") or ""
        try:
            raw = uf.getvalue()
        except AttributeError:
            raw = uf.read()
        if isinstance(raw, str):
            text = raw
        else:
            text = raw.decode("utf-8-sig", errors="replace")
        kind, partial = parse_csv_text(text, filename=name)
        pieces.append((kind, partial))
    return merge_summaries(pieces)
