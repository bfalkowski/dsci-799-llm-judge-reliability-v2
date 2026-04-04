"""
LLM-as-a-judge: call judge API (OpenAI or Anthropic) for structured JSON output.
Supports OpenAI (gpt-*) and Anthropic (claude-*). Returns raw content and token usage.
call_judge retries transient failures (429, 5xx, timeouts) with exponential backoff (see JUDGE_MAX_RETRIES).
Raises RuntimeError if the selected provider's API key is not set.
"""

import json
import os
import random
import sys
import time
from typing import Optional

JUDGE_TEMPERATURE = 0.0

# Transient API failures (rate limits, 5xx, timeouts): retry with exponential backoff.
JUDGE_MAX_RETRIES_DEFAULT = 5
JUDGE_RETRY_BASE_SEC = 1.0


def _max_judge_retries() -> int:
    raw = (os.environ.get("JUDGE_MAX_RETRIES") or "").strip()
    if not raw:
        return JUDGE_MAX_RETRIES_DEFAULT
    try:
        return max(0, int(raw))
    except ValueError:
        return JUDGE_MAX_RETRIES_DEFAULT

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


def _retryable_judge_error(exc: BaseException) -> bool:
    """True for rate limits, server errors, and network timeouts — safe to retry once or more."""
    if not isinstance(exc, Exception):
        return False

    try:
        from openai import APIConnectionError, APITimeoutError, RateLimitError
        from openai import BadRequestError, AuthenticationError

        if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
            return True
        if isinstance(exc, (BadRequestError, AuthenticationError)):
            return False
        try:
            from openai import PermissionDeniedError

            if isinstance(exc, PermissionDeniedError):
                return False
        except ImportError:
            pass
        try:
            from openai import InternalServerError as OAIInternalServerError

            if isinstance(exc, OAIInternalServerError):
                return True
        except ImportError:
            pass
        try:
            from openai import APIStatusError

            if isinstance(exc, APIStatusError):
                c = getattr(exc, "status_code", None)
                if c == 429 or (c is not None and 500 <= c < 600):
                    return True
                return False
        except ImportError:
            pass
    except ImportError:
        pass

    try:
        import anthropic

        if isinstance(exc, anthropic.APIConnectionError):
            return True
        if isinstance(exc, anthropic.RateLimitError):
            return True
        if hasattr(anthropic, "InternalServerError") and isinstance(
            exc, anthropic.InternalServerError
        ):
            return True
        c = getattr(exc, "status_code", None)
        if c == 429 or (c is not None and 500 <= c < 600):
            return True
    except ImportError:
        pass

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    msg = str(exc).lower()
    if "rate limit" in msg or "too many requests" in msg:
        return True
    if "timeout" in msg or "timed out" in msg:
        return True
    if "connection reset" in msg or "connection aborted" in msg:
        return True
    if "overloaded" in msg or "503" in msg:
        return True
    c = getattr(exc, "status_code", None)
    if c == 429 or (c is not None and 500 <= c < 600):
        return True
    return False


def _call_judge_once(
    prompt: str,
    model: str,
    system_content: str,
    temperature: float,
):
    if is_claude_model(model):
        return _call_anthropic(prompt, model, system_content, temperature=temperature)
    return _call_openai(prompt, model, system_content, temperature=temperature)


def call_judge(
    prompt: str,
    model: str,
    system_content: str = "You are an evaluator. Output JSON only.",
    temperature: Optional[float] = None,
):
    """
    Call judge LLM. Routes to OpenAI or Anthropic based on model id.
    Returns (raw_content_str, input_tokens, output_tokens).
    Retries transient errors (429, 5xx, timeouts) with backoff; attempts = 1 + JUDGE_MAX_RETRIES (default 5).
    Raises RuntimeError if API key not set or on non-retryable failure.
    """
    t = JUDGE_TEMPERATURE if temperature is None else temperature
    max_retries = _max_judge_retries()
    attempts = max_retries + 1
    last_exc: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            return _call_judge_once(prompt, model, system_content, t)
        except Exception as e:
            last_exc = e
            if attempt >= max_retries or not _retryable_judge_error(e):
                raise
            delay = JUDGE_RETRY_BASE_SEC * (2**attempt) + random.uniform(0, 0.35)
            print(
                f"[judge] transient error (attempt {attempt + 1}/{attempts}): {e!s}; "
                f"retrying in {delay:.1f}s…",
                file=sys.stderr,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _openai_response_format_not_supported(err: BaseException) -> bool:
    """True if the API rejected structured output mode for this model (e.g. base gpt-4)."""
    msg = str(err).lower()
    code = getattr(err, "status_code", None)
    if code != 400:
        return False
    return (
        "json_schema" in msg
        or "response_format" in msg
        or "structured" in msg
        or "invalid parameter" in msg
    )


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
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    base_kw = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=500,
    )
    schema_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "judge_output",
            "strict": True,
            "schema": JUDGE_RESPONSE_SCHEMA,
        },
    }

    def _complete(response_format: Optional[dict]):
        kw = dict(base_kw)
        if response_format is not None:
            kw["response_format"] = response_format
        return client.chat.completions.create(**kw)

    try:
        resp = _complete(schema_format)
    except Exception as e:
        if not _openai_response_format_not_supported(e):
            raise
        print(
            f"[judge] json_schema not supported for {model}; retrying with json_object mode…",
            file=sys.stderr,
        )
        try:
            resp = _complete({"type": "json_object"})
        except Exception as e2:
            if not _openai_response_format_not_supported(e2):
                raise
            print(
                f"[judge] json_object mode not supported for {model}; plain completion + parse…",
                file=sys.stderr,
            )
            resp = _complete(None)

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
