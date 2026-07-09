"""
Safe evaluation of simple arithmetic expressions.

This is used to execute LLM-provided expressions deterministically, avoiding
model arithmetic errors while preventing arbitrary code execution.
"""

from __future__ import annotations

import ast
import operator
from typing import Callable

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def safe_eval(expr: str) -> float:
    """Evaluate a restricted arithmetic expression into a float."""
    node = ast.parse(expr, mode="eval").body
    return _eval_node(node)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)

    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _BIN_OPS[type(node.op)](left, right)

    raise ValueError(f"Unsupported expression: {ast.dump(node, include_attributes=False)}")

