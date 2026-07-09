"""
LLM-driven solver for ConvFinQA turns.

The solver:
  - Formats the document into markdown
  - Sends document + prior turn history + current question to an LLM backend
  - Receives structured JSON (SolverResponse)
  - Optionally evaluates arithmetic deterministically via safe_eval()
"""

from __future__ import annotations

import json
import os

import requests
from google import genai
from google.genai import types

from .calculator import safe_eval
from .format import format_document
from .logger import get_logger
from .models import ConvFinQARecord
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .schemas import SolverResponse

logger = get_logger(__name__)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GOOGLE_API_KEY. Create a .env file with GOOGLE_API_KEY=... "
                "(it is already gitignored) or export it in your shell."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _call_gemini(prompt: str) -> SolverResponse:
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    client = _get_client()
    # Note: `generate_content` doesn't have an explicit system/user role here.
    # We pass the system rules + the user prompt as separate content parts.
    response = client.models.generate_content(
        model=model,
        contents=[SYSTEM_PROMPT, prompt],
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=SolverResponse,
        ),
    )

    if response.parsed is not None and isinstance(response.parsed, SolverResponse):
        return response.parsed

    # Fallback if parsed isn't available for any reason.
    return SolverResponse.model_validate_json(response.text)


def _call_ollama(prompt: str) -> SolverResponse:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "gemma4:latest")

    url = f"{base_url}/api/chat"
    # `format="json"` requests a JSON-only completion. Some local models still return
    # JSON values with stringified nulls/numbers, so we normalize before validation.
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    raw = resp.json()
    content = raw.get("message", {}).get("content", "")
    try:
        obj = json.loads(content)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Ollama returned invalid JSON: {e}") from e

    # Normalize common local-model quirks so Pydantic validation is stable across models.
    if isinstance(obj, dict):
        ans = obj.get("answer")
        if ans in ("null", "", "None"):
            obj["answer"] = None
        elif isinstance(ans, str):
            try:
                obj["answer"] = float(ans)
            except Exception:
                pass

        expr = obj.get("expression")
        if expr in ("null", "", "None"):
            obj["expression"] = None

        ip = obj.get("is_percentage")
        if isinstance(ip, str):
            obj["is_percentage"] = ip.strip().lower() in {"true", "1", "yes"}

    return SolverResponse.model_validate(obj)


def _choose_backend() -> str:
    # Prefer Ollama if configured (local, no API key / quota dependency),
    # otherwise fall back to Gemini.
    if os.getenv("OLLAMA_MODEL") or os.getenv("OLLAMA_BASE_URL"):
        return "ollama"
    return "gemini"


def answer_turn(
    record: ConvFinQARecord,
    history: list[tuple[str, float]],
    question: str,
) -> tuple[float, str]:
    """Answer one conversational turn as (numeric_answer, reasoning)."""
    doc_text = format_document(record.doc)
    prompt = build_user_prompt(doc_text, history, question)

    backend = _choose_backend()

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            if backend == "ollama":
                parsed = _call_ollama(prompt)
            else:
                parsed = _call_gemini(prompt)

            numeric_answer: float | None = None
            if parsed.expression:
                try:
                    # Prefer deterministic arithmetic to avoid model math mistakes and keep eval repeatable.
                    numeric_answer = safe_eval(parsed.expression)
                except Exception as e:  # noqa: BLE001
                    logger.warning("safe_eval failed: %s", e)
                    if parsed.answer is not None:
                        numeric_answer = float(parsed.answer)
                    else:
                        raise
            elif parsed.answer is not None:
                numeric_answer = float(parsed.answer)

            if numeric_answer is None:
                raise ValueError("Model returned neither expression nor answer.")

            return numeric_answer, parsed.reasoning
        except Exception as e:  # noqa: BLE001
            last_error = e
            logger.warning("Solver attempt %d failed: %s", attempt + 1, e)

    raise RuntimeError(f"Solver failed after retries: {last_error}")

