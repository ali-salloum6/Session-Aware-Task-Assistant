from __future__ import annotations

import ast
import operator

_OPS: dict[type, object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

MAX_EXPR_CHARS = 200


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))  # type: ignore[operator]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))  # type: ignore[operator]
    raise ValueError("unsupported or unsafe expression")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. 2*(3+4). No eval()."""
    if not isinstance(expression, str):
        return "calculator error: expression must be a string, e.g. '2*(3+4)'"
    expr = expression.strip()
    if not expr:
        return "calculator error: empty expression; pass e.g. '2*(3+4)'"
    if len(expr) > MAX_EXPR_CHARS:
        return (
            f"calculator error: expression too long "
            f"({len(expr)} chars, max {MAX_EXPR_CHARS})"
        )
    try:
        return str(_eval(ast.parse(expr, mode="eval")))
    except SyntaxError:
        return (
            f"calculator error: invalid syntax in {expr!r}; "
            "use a plain arithmetic expression like '480*3'"
        )
    except ZeroDivisionError:
        return "calculator error: division by zero"
    except Exception as exc:  # noqa: BLE001 — actionable tool error
        return (
            f"calculator error: {exc}. "
            "Only + - * / ** // % and numbers are allowed."
        )
