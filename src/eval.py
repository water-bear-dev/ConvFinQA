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
from .solver import answer_turn


def answers_match(pred: float, gold: float, tol: float = 1e-2) -> bool:
    """
    Compare predicted vs gold numeric answer with relative tolerance.

    Uses absolute tolerance when gold is 0 to avoid division by zero.
    """
    if gold == 0:
        return abs(pred) < tol
    return abs(pred - gold) / max(abs(gold), 1) < tol


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
    failures: list[dict[str, Any]]  # small sample of incorrect turns (for report/examples)
    results: list[dict[str, Any]]  # full per-turn trace (one row per dialogue turn)


def evaluate_split(
    split: str,
    limit: int = 50,
    dataset_path: str = "data/convfinqa_dataset.json",
    failure_limit: int = 10,
    output_path: str | None = None,  # when set, writes summary+results to JSON for analysis
) -> EvalSummary:
    """
    Evaluate the solver on the first `limit` conversations in a split.

    Returns a compact summary plus a small sample of failures for inspection.
    """
    loader = DatasetLoader(dataset_path)
    records = loader.iter_split(split)[:limit]

    total_turns = 0
    correct_turns = 0

    # Breakdowns
    t01_total = t01_correct = 0
    t2p_total = t2p_correct = 0
    type1_total = type1_correct = 0
    type2_total = type2_correct = 0

    failures: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for record in records:
        history: list[tuple[str, float]] = []

        for turn_idx, (q, gold_raw) in enumerate(
            zip(record.dialogue.conv_questions, record.dialogue.executed_answers)
        ):
            pred, reasoning = answer_turn(record, history, q)
            # Important: we append predictions even when wrong. This mirrors real chat
            # usage where mistakes compound across turns, and matches ConvFinQA's
            # conversational evaluation setting.
            history.append((q, pred))

            try:
                gold = float(gold_raw)
                comparable = True
            except Exception:  # noqa: BLE001
                gold = 0.0
                comparable = False

            is_correct = comparable and answers_match(pred, gold)

            total_turns += 1
            if is_correct:
                correct_turns += 1

            results.append(
                {
                    "record_id": record.id,
                    "turn": turn_idx,
                    "question": q,
                    "gold_raw": gold_raw,
                    "gold": gold if comparable else None,
                    "pred": pred,
                    "reasoning": reasoning,
                    "correct": is_correct,
                    "has_type2": record.features.has_type2_question,
                }
            )

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
                    }
                )

    acc = correct_turns / total_turns if total_turns else 0.0
    acc_turn01 = t01_correct / t01_total if t01_total else 0.0
    acc_turn2plus = t2p_correct / t2p_total if t2p_total else 0.0
    acc_type1 = type1_correct / type1_total if type1_total else 0.0
    acc_type2 = type2_correct / type2_total if type2_total else 0.0

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
        failures=failures,
        results=results,
    )

    if output_path:
        # Keep the on-disk artifact self-contained: summary for quick reading,
        # failures for error analysis, and results for deeper inspection.
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
            },
            "failures": summary.failures,
            "results": summary.results,
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)

    return summary

