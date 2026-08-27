"""Minimal Ollama chat client (stdlib only) with JSON-schema output support."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from pipeline.config import OLLAMA_URL


class OllamaError(RuntimeError):
    pass


def chat(model: str, messages: list[dict], schema: dict | None = None,
         temperature: float = 0.2, num_ctx: int = 32768, timeout: int = 600) -> str:
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        # num_predict caps pathological generation loops: a truncated response
        # is salvageable, a hung request is not.
        "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": 4096},
    }
    if schema is not None:
        payload["format"] = schema

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        # Socket timeouts raise bare OSError/TimeoutError, not URLError; all
        # of them must become OllamaError so callers' fallbacks engage.
        raise OllamaError(f"Ollama request failed ({OLLAMA_URL}): {e}") from e

    content = body.get("message", {}).get("content", "")
    if not content:
        raise OllamaError(f"empty response from {model}")
    return content


def _repair_json(content: str) -> str:
    """Deterministic repairs for the model's common JSON mistakes. Invalid
    escapes dominate: OCR source text contains stray backslashes the model
    echoes into strings, and it reproduces them identically on retry."""
    import re

    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
    # Escape any backslash not starting a legal JSON escape.
    content = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", content)
    # Strip trailing commas before } or ].
    content = re.sub(r",\s*([}\]])", r"\1", content)
    return content


def _salvage_segments(content: str) -> dict | None:
    """Truncated generation cannot be prevented by grammar constraints; when
    the tail is cut off, salvage every COMPLETE segment object."""
    import re

    objs = []
    for m in re.finditer(r'\{\s*"type"\s*:\s*"(\w+)"\s*,\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
                         content):
        try:
            objs.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    return {"segments": objs} if objs else None


def chat_json(model: str, messages: list[dict], schema: dict,
              temperature: float = 0.2, retries: int = 2, **kw) -> dict:
    last_err: Exception | None = None
    last_content = ""
    for attempt in range(retries + 1):
        content = chat(model, messages, schema=schema, temperature=temperature, **kw)
        last_content = content
        for candidate in (content, _repair_json(content)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                last_err = e
        salvaged = _salvage_segments(_repair_json(content))
        if salvaged:
            return salvaged
        messages = messages + [
            {"role": "assistant", "content": content[:2000]},
            {"role": "user", "content": "That was not valid JSON. Return ONLY the JSON object, "
                                        "with all backslashes double-escaped."},
        ]
    from pathlib import Path
    dump = Path("/tmp/ollama_bad_json.txt")
    dump.write_text(last_content)
    raise OllamaError(f"invalid JSON after {retries + 1} attempts: {last_err} (raw -> {dump})")


def available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3):
            return True
    except OSError:
        return False
