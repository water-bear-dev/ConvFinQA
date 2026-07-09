"""
Evaluation harness for ConvFinQA execution accuracy (Exe Acc).

Replays scripted conv_questions in order, maintaining conversation history
the same way the CLI chat loop would (including wrong prior answers).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .data import DatasetLoader
from .history import TurnState
from .solver import answer_turn


def answers_match(pred: float, gold: float, tol: float = 1e-2) -> bool:
    """
    Compare predicted vs gold numeric answer with relative tolerance.

    Uses absolute tolerance when gold is 0 to avoid division by zero.
    """
    if gold == 0:
        return abs(pred) < tol
    return abs(pred - gold) / max(abs(gold), 1) < tol


def infer_error_tag(
    turn_idx: int,
    question: str,
    has_type2: bool,
    correct: bool,
    reasoning: str | None,
) -> str | None:
    """Heuristic error category for incorrect turns."""
    if correct:
        return None
    if reasoning and reasoning.startswith("ERROR:"):
        return "parse"

    q = question.lower()
    if turn_idx == 0:
        return "lookup"
    if has_type2 and turn_idx >= 2:
        return "type2"
    if any(w in q for w in ("percent", "percentage", "ratio", "relation", "proportion")):
        return "percentage"
    if any(
        w in q
        for w in ("difference", "change", "increase", "decrease", "subtract", "sum", "total")
    ):
        return "math"
    if any(w in q for w in ("what about", "and what", "that", "it in", "and in")):
        return "coreference"
    return "other"


@dataclass(frozen=True)
class EvalSummary:
    split: str
    limit: int
    total_records: int
    total_turns: int
    correct_turns: int
    acc: float
    acc_turn01: float
    acc_turn2plus: float
    acc_type1: float
    acc_type2: float
    acc_by_route: dict[str, float]
    retry_count: int
    error_tag_counts: dict[str, int]
    failures: list[dict[str, Any]]
    results: list[dict[str, Any]]


def evaluate_split(
    split: str,
    limit: int = 50,
    dataset_path: str = "data/convfinqa_dataset.json",
    failure_limit: int = 10,
    output_path: str | None = None,
) -> EvalSummary:
    """
    Evaluate the solver on the first `limit` conversations in a split.

    Returns a compact summary plus a small sample of failures for inspection.
    """
    loader = DatasetLoader(dataset_path)
    records = loader.iter_split(split)[:limit]

    total_turns = 0
    correct_turns = 0

    t01_total = t01_correct = 0
    t2p_total = t2p_correct = 0
    type1_total = type1_correct = 0
    type2_total = type2_correct = 0

    route_totals: dict[str, int] = {}
    route_correct: dict[str, int] = {}
    retry_count = 0
    error_tag_counts: dict[str, int] = {}

    failures: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for record in records:
        history: list[TurnState] = []

        for turn_idx, (q, gold_raw) in enumerate(
            zip(record.dialogue.conv_questions, record.dialogue.executed_answers)
        ):
            result = answer_turn(record, history, q, turn_idx=turn_idx)
            pred = result.answer
            reasoning = result.reasoning

            history.append(
                TurnState(question=q, answer=pred, metric=result.metric)
            )

            if result.retried:
                retry_count += 1

            try:
                gold = float(gold_raw)
                comparable = True
            except Exception:  # noqa: BLE001
                gold = 0.0
                comparable = False

            is_correct = comparable and answers_match(pred, gold)
            error_tag = infer_error_tag(
                turn_idx, q, record.features.has_type2_question, is_correct, reasoning
            )

            total_turns += 1
            if is_correct:
                correct_turns += 1

            route_totals[result.route] = route_totals.get(result.route, 0) + 1
            if is_correct:
                route_correct[result.route] = route_correct.get(result.route, 0) + 1

            if error_tag:
                error_tag_counts[error_tag] = error_tag_counts.get(error_tag, 0) + 1

            row = {
                "record_id": record.id,
                "turn": turn_idx,
                "question": q,
                "gold_raw": gold_raw,
                "gold": gold if comparable else None,
                "pred": pred,
                "reasoning": reasoning,
                "correct": is_correct,
                "has_type2": record.features.has_type2_question,
                "route": result.route,
                "retried": result.retried,
                "error_tag": error_tag,
            }
            results.append(row)

            if turn_idx <= 1:
                t01_total += 1
                if is_correct:
                    t01_correct += 1
            else:
                t2p_total += 1
                if is_correct:
                    t2p_correct += 1

            if record.features.has_type2_question:
                type2_total += 1
                if is_correct:
                    type2_correct += 1
            else:
                type1_total += 1
                if is_correct:
                    type1_correct += 1

            if (not is_correct) and len(failures) < failure_limit:
                failures.append(
                    {
                        "record_id": record.id,
                        "turn": turn_idx,
                        "question": q,
                        "gold_raw": gold_raw,
                        "pred": pred,
                        "reasoning": reasoning,
                        "has_type2": record.features.has_type2_question,
                        "gold_castable": comparable,
                        "route": result.route,
                        "error_tag": error_tag,
                    }
                )

    acc = correct_turns / total_turns if total_turns else 0.0
    acc_turn01 = t01_correct / t01_total if t01_total else 0.0
    acc_turn2plus = t2p_correct / t2p_total if t2p_total else 0.0
    acc_type1 = type1_correct / type1_total if type1_total else 0.0
    acc_type2 = type2_correct / type2_total if type2_total else 0.0
    acc_by_route = {
        route: route_correct.get(route, 0) / total
        for route, total in route_totals.items()
    }

    summary = EvalSummary(
        split=split,
        limit=limit,
        total_records=len(records),
        total_turns=total_turns,
        correct_turns=correct_turns,
        acc=acc,
        acc_turn01=acc_turn01,
        acc_turn2plus=acc_turn2plus,
        acc_type1=acc_type1,
        acc_type2=acc_type2,
        acc_by_route=acc_by_route,
        retry_count=retry_count,
        error_tag_counts=error_tag_counts,
        failures=failures,
        results=results,
    )

    if output_path:
        payload = {
            "summary": {
                "split": summary.split,
                "limit": summary.limit,
                "total_records": summary.total_records,
                "total_turns": summary.total_turns,
                "correct_turns": summary.correct_turns,
                "acc": summary.acc,
                "acc_turn01": summary.acc_turn01,
                "acc_turn2plus": summary.acc_turn2plus,
                "acc_type1": summary.acc_type1,
                "acc_type2": summary.acc_type2,
                "acc_by_route": summary.acc_by_route,
                "retry_count": summary.retry_count,
                "error_tag_counts": summary.error_tag_counts,
            },
            "failures": summary.failures,
            "results": summary.results,
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)

    return summary


def compare_eval_results(
    baseline_path: str, candidate_path: str
) -> dict[str, Any]:
    """Compare two eval JSON artifacts and return deltas."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(candidate_path) as f:
        candidate = json.load(f)

    b_sum = baseline["summary"]
    c_sum = candidate["summary"]

    b_rows = {
        (r["record_id"], r["turn"]): r for r in baseline.get("results", [])
    }
    c_rows = {
        (r["record_id"], r["turn"]): r for r in candidate.get("results", [])
    }

    fixed: list[dict[str, Any]] = []
    broken: list[dict[str, Any]] = []
    for key, brow in b_rows.items():
        crow = c_rows.get(key)
        if not crow:
            continue
        if not brow.get("correct") and crow.get("correct"):
            fixed.append(crow)
        elif brow.get("correct") and not crow.get("correct"):
            broken.append(crow)

    metrics = [
        "acc",
        "acc_turn01",
        "acc_turn2plus",
        "acc_type1",
        "acc_type2",
    ]
    deltas = {
        m: c_sum.get(m, 0) - b_sum.get(m, 0) for m in metrics
    }

    return {
        "baseline": b_sum,
        "candidate": c_sum,
        "deltas": deltas,
        "fixed_count": len(fixed),
        "broken_count": len(broken),
        "fixed_sample": fixed[:10],
        "broken_sample": broken[:10],
    }
