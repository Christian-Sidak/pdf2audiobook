"""LLM judge (local Qwen via Ollama) with a disk cache.

Used only for cases deterministic checks cannot decide: is an unmatched
rewrite sentence a hallucination, or a legitimate rewording?
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline.config import JUDGE_MODEL
from pipeline.ollama_client import available, chat_json

CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "judge"

VERDICT_SCHEMA = {
    "type": "object",
    "required": ["verdict", "reason"],
    "properties": {
        "verdict": {"type": "string", "enum": ["faithful", "hallucination", "content_loss"]},
        "reason": {"type": "string"},
        "offending_text": {"type": "string"},
    },
}

JUDGE_PROMPT = """You are auditing an audiobook narration rewrite of a scholarly book. The rewrite is allowed to: verbalize numbers as words, expand abbreviations, drop citations (ibid., page refs, bracketed numbers), and smooth phrasing for the ear. It is NOT allowed to add facts or drop substantive content.

SOURCE PASSAGE:
{source}

REWRITE PASSAGE:
{rewrite}

QUESTION: {question}

Return JSON: {{"verdict": "faithful" | "hallucination" | "content_loss", "reason": "...", "offending_text": "..."}}"""


def judge_available() -> bool:
    return available()


def judge(source: str, rewrite: str, question: str, model: str = JUDGE_MODEL) -> dict:
    key = hashlib.sha256(f"{model}|{source}|{rewrite}|{question}".encode()).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    prompt = JUDGE_PROMPT.format(source=source[:4000], rewrite=rewrite[:4000], question=question)
    result = chat_json(model, [{"role": "user", "content": prompt}],
                       VERDICT_SCHEMA, temperature=0.0)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(result))
    return result
