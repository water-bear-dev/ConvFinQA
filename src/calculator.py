"""
Safe evaluation of simple arithmetic expressions.

This is used to execute LLM-provided expressions deterministically, avoiding
model arithmetic errors while preventing arbitrary code execution.
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Callable

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_UNSUPPORTED_CALLS = frozenset({"min", "max", "sum", "abs", "round"})


def expression_has_unsupported_calls(expr: str) -> bool:
    """Return True if the expression contains unsupported function calls."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return True

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _UNSUPPORTED_CALLS:
                return True
            return True
    return False


def normalize_expression(expr: str) -> str:
    """
    Normalize a model-provided expression for safe_eval.

    Strips leading '=', currency/percent symbols, and thousand-separator commas.
    Raises ValueError if unsupported function calls are present.
    """
    s = expr.strip().lstrip("=").strip()
    s = s.replace("$", "").replace("%", "")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    if expression_has_unsupported_calls(s):
        raise ValueError(f"Unsupported expression functions: {expr!r}")
    return s


def safe_eval(expr: str) -> float:
    """Evaluate a restricted arithmetic expression into a float."""
    normalized = normalize_expression(expr)
    node = ast.parse(normalized, mode="eval").body
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
