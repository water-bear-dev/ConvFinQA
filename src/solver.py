"""
LLM-driven solver for ConvFinQA turns.

The solver:
  - Formats the document into markdown
  - Routes each turn to single-call or two-step retrieve-then-compute
  - Sends document + prior turn history + current question to an LLM backend
  - Receives structured JSON (SolverResponse / RetrievalResponse)
  - Optionally evaluates arithmetic deterministically via safe_eval()
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, TypeVar

import requests
from google import genai
from google.genai import types
from pydantic import BaseModel

from .calculator import (
    expression_has_unsupported_calls,
    safe_eval,
)
from .format import format_document
from .history import TurnState, normalize_history
from .logger import get_logger
from .models import ConvFinQARecord
from .postprocess import finalize_answer
from .prompts import (
    COMPUTE_SYSTEM_PROMPT,
    RETRIEVE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_compute_prompt,
    build_retrieve_prompt,
    build_user_prompt,
)
from .router import is_reask_question, route_turn
from .schemas import CellRetrieval, RetrievalResponse, SolverResponse

logger = get_logger(__name__)

_client: genai.Client | None = None
T = TypeVar("T", bound=BaseModel)

_NUM_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?")

_HISTORY_MISMATCH_THRESHOLD = 0.05


@dataclass(frozen=True)
class AnswerResult:
    answer: float
    reasoning: str
    route: str = "single"
    retried: bool = False
    metric: str | None = None


def _parse_numeric_answer(text: str) -> float | None:
    """
    Best-effort parse of a numeric answer from local-model strings.

    Handles cases like:
      - \"$4 million\" -> 4000000.0
      - \"(1,234)\" -> -1234.0
      - \"23.1\" -> 23.1
      - sentences containing a number -> first number
    """
    s = text.strip()
    if not s:
        return None

    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1].strip()

    s_lower = s.lower()
    multiplier = 1.0
    if "billion" in s_lower:
        multiplier = 1_000_000_000.0
    elif "million" in s_lower:
        multiplier = 1_000_000.0
    elif "thousand" in s_lower:
        multiplier = 1_000.0

    s_clean = s.replace("$", "").replace("%", "").strip()

    m = _NUM_RE.search(s_clean)
    if not m:
        return None

    num_str = m.group(0).replace(",", "")
    try:
        val = float(num_str) * multiplier
    except Exception:  # noqa: BLE001
        return None

    if negative:
        val = -val
    return val


def _normalize_solver_obj(obj: dict[str, Any]) -> None:
    if "reasoning" not in obj:
        obj["reasoning"] = None

    ans = obj.get("answer")
    if ans in ("null", "", "None"):
        obj["answer"] = None
    elif isinstance(ans, str):
        parsed = _parse_numeric_answer(ans)
        obj["answer"] = parsed if parsed is not None else ans

    expr = obj.get("expression")
    if expr in ("null", "", "None"):
        obj["expression"] = None

    ip = obj.get("is_percentage")
    if isinstance(ip, str):
        obj["is_percentage"] = ip.strip().lower() in {"true", "1", "yes"}


def _normalize_retrieval_obj(obj: dict[str, Any]) -> None:
    if "reasoning" not in obj:
        obj["reasoning"] = None
    if "active_metric" not in obj:
        obj["active_metric"] = None
    cells = obj.get("cells")
    if cells is None:
        obj["cells"] = []
    elif isinstance(cells, list):
        for cell in cells:
            if isinstance(cell, dict) and isinstance(cell.get("value"), str):
                parsed = _parse_numeric_answer(cell["value"])
                if parsed is not None:
                    cell["value"] = parsed


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


def _choose_backend() -> str:
    if os.getenv("OLLAMA_MODEL") or os.getenv("OLLAMA_BASE_URL"):
        return "ollama"
    return "gemini"


def _call_gemini(
    system_prompt: str, user_prompt: str, schema: type[T]
) -> T:
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    client = _get_client()
    response = client.models.generate_content(
        model=model,
        contents=[system_prompt, user_prompt],
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    if response.parsed is not None and isinstance(response.parsed, schema):
        return response.parsed

    return schema.model_validate_json(response.text)


def _call_ollama(
    system_prompt: str, user_prompt: str, schema: type[T]
) -> T:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "gemma4:latest")

    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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

    if isinstance(obj, dict):
        if schema is SolverResponse:
            _normalize_solver_obj(obj)
        elif schema is RetrievalResponse:
            _normalize_retrieval_obj(obj)

    return schema.model_validate(obj)


def _call_llm(
    system_prompt: str, user_prompt: str, schema: type[T]
) -> T:
    backend = _choose_backend()
    if backend == "ollama":
        return _call_ollama(system_prompt, user_prompt, schema)
    return _call_gemini(system_prompt, user_prompt, schema)


def _extract_numeric_from_response(parsed: SolverResponse) -> float:
    numeric_answer: float | None = None
    if parsed.expression:
        try:
            if expression_has_unsupported_calls(parsed.expression):
                raise ValueError("unsupported expression functions")
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
    return numeric_answer


def _cells_to_dicts(cells: list[CellRetrieval]) -> list[dict[str, Any]]:
    return [cell.model_dump() for cell in cells]


def _history_mismatch(
    history: list[TurnState],
    fresh_cells: list[CellRetrieval],
    question: str,
) -> tuple[bool, str | None]:
    """Detect if retrieved values disagree with prior history answers."""
    if not history or not fresh_cells:
        return False, None

    fresh_value = fresh_cells[0].value
    candidates = [t.answer for t in history] if is_reask_question(question) else [history[-1].answer]

    for prior_answer in reversed(candidates):
        denom = max(abs(prior_answer), 1.0)
        if abs(fresh_value - prior_answer) / denom > _HISTORY_MISMATCH_THRESHOLD:
            note = (
                f"Prior answer {prior_answer} may be wrong. "
                f"Use document value {fresh_value} from "
                f"row '{fresh_cells[0].row_label}', column '{fresh_cells[0].column_header}'."
            )
            return True, note
    return False, None


def retrieve_cells(
    record: ConvFinQARecord,
    history: list[TurnState],
    question: str,
) -> RetrievalResponse:
    """Retrieve relevant table cells for the current question."""
    doc_text = format_document(record.doc)
    prompt = build_retrieve_prompt(doc_text, history, question)
    return _call_llm(RETRIEVE_SYSTEM_PROMPT, prompt, RetrievalResponse)


def compute_answer(
    history: list[TurnState],
    question: str,
    cells: list[CellRetrieval],
    active_metric: str | None = None,
    correction_note: str | None = None,
) -> SolverResponse:
    """Compute the numeric answer from retrieved cells and prior turns."""
    prompt = build_compute_prompt(
        _cells_to_dicts(cells),
        history,
        question,
        active_metric=active_metric,
        correction_note=correction_note,
    )
    return _call_llm(COMPUTE_SYSTEM_PROMPT, prompt, SolverResponse)


def _solve_single(
    record: ConvFinQARecord,
    history: list[TurnState],
    question: str,
) -> tuple[float, str, bool]:
    doc_text = format_document(record.doc)
    prompt = build_user_prompt(doc_text, history, question)
    parsed = _call_llm(SYSTEM_PROMPT, prompt, SolverResponse)
    answer = _extract_numeric_from_response(parsed)
    answer = finalize_answer(answer, question, parsed.is_percentage)
    return answer, parsed.reasoning or "", False


def _solve_two_step(
    record: ConvFinQARecord,
    history: list[TurnState],
    question: str,
) -> tuple[float, str, str | None, bool]:
    retrieval = retrieve_cells(record, history, question)
    cells = retrieval.cells
    active_metric = retrieval.active_metric

    mismatch, correction_note = _history_mismatch(history, cells, question)
    retried = False

    parsed = compute_answer(
        history,
        question,
        cells,
        active_metric=active_metric,
        correction_note=correction_note,
    )
    try:
        answer = _extract_numeric_from_response(parsed)
    except Exception:
        if mismatch and correction_note:
            retried = True
            parsed = compute_answer(
                history,
                question,
                cells,
                active_metric=active_metric,
                correction_note=correction_note + " Recompute using retrieved values only.",
            )
            answer = _extract_numeric_from_response(parsed)
        else:
            raise

    answer = finalize_answer(answer, question, parsed.is_percentage)
    metric = active_metric or (cells[0].metric if cells else None)
    return answer, parsed.reasoning or retrieval.reasoning or "", metric, retried


def answer_turn(
    record: ConvFinQARecord,
    history: list[TurnState] | list[tuple[str, float]],
    question: str,
    turn_idx: int = 0,
) -> AnswerResult:
    """Answer one conversational turn."""
    hist = normalize_history(history)
    has_type2 = record.features.has_type2_question
    route = route_turn(turn_idx, question, has_type2)

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            if route == "two_step":
                answer, reasoning, metric, retried = _solve_two_step(
                    record, hist, question
                )
                return AnswerResult(
                    answer=answer,
                    reasoning=reasoning,
                    route="two_step",
                    retried=retried,
                    metric=metric,
                )

            answer, reasoning, retried = _solve_single(record, hist, question)
            return AnswerResult(
                answer=answer,
                reasoning=reasoning,
                route="single",
                retried=retried,
                metric=hist[-1].metric if hist else None,
            )
        except Exception as e:  # noqa: BLE001
            last_error = e
            logger.warning("Solver attempt %d failed: %s", attempt + 1, e)

    return AnswerResult(
        answer=0.0,
        reasoning=f"ERROR: {last_error}",
        route=route,
        retried=False,
    )
