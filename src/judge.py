"""
LLM-as-a-judge: call judge API (OpenAI or Anthropic) for structured JSON output.
Supports OpenAI (gpt-*) and Anthropic (claude-*). Returns raw content and token usage.
Raises RuntimeError if the selected provider's API key is not set.
"""

import json
import os
import sys
from typing import Optional

JUDGE_TEMPERATURE = 0.0

# JSON schema for structured output: { score: int 0-100, justification: str }
JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Quality score 0-100"},
        "justification": {"type": "string", "description": "Short explanation"},
    },
    "required": ["score", "justification"],
    "additionalProperties": False,
}

RUBRIC_GENERATOR_SYSTEM = (
    "You write evaluation rubrics for another LLM judge. Output plain text only — no JSON, no markdown fences."
)


def build_rubric_generator_user_prompt(question: str, response: str) -> str:
    q = (question or "").strip()
    r = (response or "").strip()
    return f"""You are designing per-item judge instructions for another model that will score candidate answers on a scale of 0 (worst) to 100 (best).

Benchmark question:
{q}

Reference (gold) response:
{r}

Write concise judge instructions (prose; bullets ok) that the judge will follow when scoring *any* answer to this question. The instructions MUST:
1. State clearly that the final output must be a single integer from 0 to 100.
2. Use an explicit add-points and deduct-points style: say what earns credit (and roughly how much if helpful) and what should reduce the score.
3. Tie criteria to this specific task (correctness, completeness, relevance to the question).

Output ONLY the instructions text — no title line like "Instructions:", no preamble."""


def call_text_model(
    user_prompt: str,
    model: str,
    system_content: str = RUBRIC_GENERATOR_SYSTEM,
    temperature: float = 0.2,
    max_tokens: int = 2048,
):
    """
    Plain-text completion (no JSON schema). For rubric generation, etc.
    Returns (text, input_tokens, output_tokens).
    """
    t = float(temperature)
    if is_claude_model(model):
        return _call_anthropic_text(user_prompt, model, system_content, temperature=t, max_tokens=max_tokens)
    return _call_openai_text(user_prompt, model, system_content, temperature=t, max_tokens=max_tokens)


def _call_openai_text(prompt: str, model: str, system_content: str, temperature: float, max_tokens: int):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env at the repo root, or export it before running."
        )
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package is not installed. Run: pip install openai")

    print(f"[judge] OpenAI text call ({model})...", file=sys.stderr)
    client = OpenAI(api_key=api_key.strip())
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = ""
    if resp.choices and resp.choices[0].message.content:
        content = resp.choices[0].message.content.strip()
    if not content:
        raise RuntimeError("API returned empty text response.")
    usage = resp.usage
    input_tokens = usage.prompt_tokens if usage else None
    output_tokens = usage.completion_tokens if usage else None
    return content, input_tokens, output_tokens


def _call_anthropic_text(prompt: str, model: str, system_content: str, temperature: float, max_tokens: int):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env at the repo root, or export it before running."
        )
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package is not installed. Run: pip install anthropic")

    print(f"[judge] Anthropic text call ({model})...", file=sys.stderr)
    client = anthropic.Anthropic(api_key=api_key.strip())
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_content,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "thinking": {"type": "disabled"},
    }
    resp = client.messages.create(**kwargs)
    content = ""
    if resp.content:
        for block in resp.content:
            if hasattr(block, "text") and block.text:
                content = block.text.strip()
                break
    if not content:
        raise RuntimeError("API returned empty text response.")
    usage = getattr(resp, "usage", None)
    input_tokens = usage.input_tokens if usage else None
    output_tokens = usage.output_tokens if usage else None
    return content, input_tokens, output_tokens


def is_claude_model(model_id):
    """True if model uses Anthropic API (claude-*)."""
    return model_id and str(model_id).strip().lower().startswith("claude")


def call_judge(
    prompt: str,
    model: str,
    system_content: str = "You are an evaluator. Output JSON only.",
    temperature: Optional[float] = None,
):
    """
    Call judge LLM. Routes to OpenAI or Anthropic based on model id.
    Returns (raw_content_str, input_tokens, output_tokens).
    Raises RuntimeError if API key not set or on failure.
    """
    t = JUDGE_TEMPERATURE if temperature is None else temperature
    if is_claude_model(model):
        return _call_anthropic(prompt, model, system_content, temperature=t)
    return _call_openai(prompt, model, system_content, temperature=t)


def _call_openai(prompt: str, model: str, system_content: str, temperature: float):
    """Call OpenAI judge. Requires OPENAI_API_KEY."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env at the repo root, or export it before running."
        )
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package is not installed. Run: pip install openai")

    print(f"[judge] Calling OpenAI ({model})...", file=sys.stderr)
    client = OpenAI(api_key=api_key.strip())
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=500,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "judge_output",
                "strict": True,
                "schema": JUDGE_RESPONSE_SCHEMA,
            },
        },
    )
    content = ""
    if resp.choices and resp.choices[0].message.content:
        content = resp.choices[0].message.content.strip()
    print(f"[judge] OpenAI response received ({len(content)} chars)", file=sys.stderr)
    if not content:
        raise RuntimeError("Judge API returned empty response.")

    usage = resp.usage
    input_tokens = usage.prompt_tokens if usage else None
    output_tokens = usage.completion_tokens if usage else None
    return content, input_tokens, output_tokens


def extract_json_from_text(text: str):
    """Extract JSON object from text, handling markdown code blocks and surrounding text."""
    import re
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for sep in ("```json", "```"):
        if sep in text:
            parts = text.split(sep, 1)
            if len(parts) > 1:
                rest = parts[1].split("```", 1)[0].strip()
                try:
                    return json.loads(rest)
                except json.JSONDecodeError:
                    pass
    # Try to find {...} in text
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _call_anthropic(prompt: str, model: str, system_content: str, temperature: float):
    """Call Anthropic Claude judge. Requires ANTHROPIC_API_KEY."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env at the repo root, or export it before running."
        )
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package is not installed. Run: pip install anthropic")

    print(f"[judge] Calling Anthropic Claude ({model})...", file=sys.stderr)
    client = anthropic.Anthropic(api_key=api_key.strip())

    # Use prompt-only (structured output API varies by SDK version)
    # Disable extended thinking for simpler, faster responses (no thinking blocks)
    kwargs = {
        "model": model,
        "max_tokens": 500,
        "system": f"{system_content}\n\nRespond with a JSON object: {{\"score\": <0-100>, \"justification\": \"<short explanation>\"}}",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "thinking": {"type": "disabled"},
    }

    resp = client.messages.create(**kwargs)

    content = ""
    if resp.content:
        for block in resp.content:
            if hasattr(block, "text") and block.text:
                content = block.text.strip()
                break
    print(f"[judge] Anthropic response received ({len(content)} chars)", file=sys.stderr)
    if not content:
        raise RuntimeError("Judge API returned empty response.")

    usage = getattr(resp, "usage", None)
    input_tokens = usage.input_tokens if usage else None
    output_tokens = usage.output_tokens if usage else None

    return content, input_tokens, output_tokens
