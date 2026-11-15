"""Row model shared by every output format.

Rows are plain ``dict`` objects in insertion order, which is what all three
writers iterate.  This module owns the single definition of:

* how a path is resolved against a record,
* how the parsed condition AST is evaluated,
* how a select field (plain, function, expression, aggregation) becomes an
  output value, and
* how values are rendered for cells and for JSON.

Keeping rendering and ordering here is what makes the formats agree: the table
and CSV writers never re-invent value formatting, and the JSON writer
serialises the same dict with the same key order.
"""

from __future__ import annotations

import json
import math
from typing import Any

from jonq.ast import (
    AndCondition,
    BetweenCondition,
    Condition,
    ExprType,
    InCondition,
    NotCondition,
    OrCondition,
    PathType,
)
from jonq.parser import parse_expression, parse_path
from jonq.streaming.plan import PlanField

MISSING = object()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_path(record: Any, path: str) -> Any:
    """Resolve a dotted/indexed ``path`` against ``record``.

    Returns :data:`MISSING` when any segment is absent.  A nested list stops
    resolution (streaming does not flatten arrays); the caller decides whether
    that is an error.  The function never raises for type mismatches.
    """
    if path in ("", ".", None):
        return record
    if path == "*":
        return record
    current = record
    for element in parse_path(path).elements:
        current = _step(current, element)
        if current is MISSING:
            return MISSING
    return current


def _step(value: Any, element) -> Any:
    if element.type == PathType.FIELD:
        if isinstance(value, dict):
            return value.get(element.value, MISSING)
        return MISSING
    if element.type == PathType.ARRAY_INDEX:
        if isinstance(value, list):
            try:
                index = int(element.value)
                return value[index] if -len(value) <= index < len(value) else MISSING
            except (TypeError, ValueError):
                return MISSING
        return MISSING
    if element.type == PathType.ARRAY:
        if isinstance(value, dict):
            return value.get(element.value, MISSING)
        if element.value in ("", "."):
            return value
        return MISSING
    return MISSING


def path_exists(record: Any, path: str) -> bool:
    return resolve_path(record, path) is not MISSING


# ---------------------------------------------------------------------------
# AST evaluation
# ---------------------------------------------------------------------------

def evaluate_condition(condition_ast, record: Any) -> bool:
    """Evaluate a parsed condition AST against one record."""
    if condition_ast is None:
        return True
    return bool(_eval(condition_ast, record))


def _eval(node, record: Any) -> Any:
    if node is None:
        return None
    if isinstance(node, AndCondition):
        return bool(_eval(node.left, record)) and bool(_eval(node.right, record))
    if isinstance(node, OrCondition):
        return bool(_eval(node.left, record)) or bool(_eval(node.right, record))
    if isinstance(node, NotCondition):
        return not bool(_eval(node.condition, record))
    if isinstance(node, BetweenCondition):
        value = resolve_path(record, node.field)
        low = _literal(node.low)
        high = _literal(node.high)
        if value is MISSING or value is None or low is None or high is None:
            return False
        try:
            return low <= value <= high
        except TypeError:
            return False
    if isinstance(node, InCondition):
        value = resolve_path(record, node.field)
        if value is MISSING:
            value = None
        for candidate in node.values:
            if _values_equal(value, _literal(candidate)):
                return True
        return False
    if isinstance(node, Condition):
        return _eval_expression(node.expr, record)
    return _eval_expression(node, record)


def _eval_expression(expr, record: Any) -> Any:
    if expr is None:
        return None
    if not hasattr(expr, "type"):
        return expr

    kind = expr.type
    if kind == ExprType.FIELD:
        path = expr.value
        if path in ("", ".", "*"):
            return record
        value = resolve_path(record, path)
        return None if value is MISSING else value
    if kind == ExprType.LITERAL:
        return _literal(expr.value)
    if kind == ExprType.OPERATION:
        return _eval_operation(expr.value, expr.args, record)
    if kind == ExprType.BINARY_CONDITION:
        return _eval_binary(expr.value, expr.args, record)
    if kind == ExprType.FUNCTION:
        return _apply_function(expr.value, expr.args, record)
    if kind == ExprType.AGGREGATION:
        value = _eval_expression(expr.args[0], record) if expr.args else record
        return _aggregate_single(expr.value, value)
    if kind == ExprType.CASE:
        return _eval_case(expr, record)
    return expr.value


def _literal(token):
    if isinstance(token, Expression_LITERAL_TYPES):
        return token
    text = str(token).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        inner = text[1:-1]
        if text[0] == '"':
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return inner
        return inner
    if text in ("null", "None"):
        return None
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


Expression_LITERAL_TYPES = (int, float, bool, type(None), list, dict)


def _eval_operation(op: str, args, record: Any) -> Any:
    left = _eval_expression(args[0], record)
    right = _eval_expression(args[1], record)
    if op == "||":
        return left if left not in (None, "", []) else right
    if left is None or right is None:
        return None
    try:
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op == "%":
            return left % right
        if op == "^":
            return left**right
    except (TypeError, ZeroDivisionError, ValueError):
        return None
    return None


def _eval_binary(op: str, args, record: Any) -> bool:
    left = _eval_expression(args[0], record)
    right = _eval_expression(args[1], record)
    if op == "contains":
        if left is None:
            return False
        return str(right) in _to_text(left)
    if op == "like":
        return _like(left, "" if right is None else str(right))
    if op == "==":
        return _values_equal(left, right)
    if op == "!=":
        return not _values_equal(left, right)
    if left is None or right is None:
        return False
    try:
        if op == ">":
            return left > right
        if op == "<":
            return left < right
        if op == ">=":
            return left >= right
        if op == "<=":
            return left <= right
    except TypeError:
        return False
    return False


def _values_equal(left, right) -> bool:
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    return left == right


def _like(value: Any, pattern: str) -> bool:
    if value is None:
        return False
    text = str(value)
    if pattern.startswith("%") and pattern.endswith("%"):
        return pattern[1:-1] in text
    if pattern.startswith("%"):
        return text.endswith(pattern[1:])
    if pattern.endswith("%"):
        return text.startswith(pattern[:-1])
    return pattern in text


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _eval_case(expr, record: Any) -> Any:
    for condition_text, value_text in expr.args or ():
        try:
            from jonq.query_parser import parse_condition_string

            condition_ast = parse_condition_string(condition_text)
        except ValueError:
            condition_ast = None
        if evaluate_condition(condition_ast, record):
            try:
                return _eval_expression(parse_expression(value_text), record)
            except ValueError:
                return value_text
    if expr.value:
        try:
            return _eval_expression(parse_expression(expr.value), record)
        except ValueError:
            return expr.value
    return None


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_record(record: Any, plan) -> dict:
    """Turn one raw record into an output row according to ``plan.fields``."""
    if plan.is_select_all:
        if isinstance(record, dict):
            return dict(record)
        return {"value": record}
    row: dict[str, Any] = {}
    for field in plan.fields:
        row[field.alias] = _field_value(field, record)
    return row


def _field_value(field: PlanField, record: Any) -> Any:
    if field.kind == "field":
        if field.path in ("*", None):
            return record
        value = resolve_path(record, field.path)
        return None if value is MISSING else value
    if field.kind == "function":
        value = resolve_path(record, field.param)
        if value is MISSING:
            return None
        return _apply_function_name(field.func, value)
    if field.kind == "expression":
        try:
            return _eval_expression(parse_expression(field.expression or ""), record)
        except ValueError:
            return None
    if field.kind == "count_distinct":
        value = resolve_path(record, field.param)
        return 0 if value is MISSING else 1
    if field.kind == "aggregation":
        if field.func == "count" and field.param == "*":
            return 1
        value = resolve_path(record, field.param)
        if value is MISSING:
            value = None
        return _aggregate_single(field.func, value)
    return None


def _aggregate_single(func: str, value: Any):
    if func == "count":
        if isinstance(value, list):
            return len(value)
        return 1
    if func in ("sum", "avg", "min", "max"):
        numbers = _numbers_in(value)
        if func == "sum":
            return sum(numbers)
        if func == "avg":
            return sum(numbers) / len(numbers) if numbers else None
        if not numbers:
            return None
        return max(numbers) if func == "max" else min(numbers)
    return value


def _numbers_in(value: Any) -> list:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [value]
    if isinstance(value, list):
        numbers = []
        for item in value:
            numbers.extend(_numbers_in(item))
        return numbers
    return []


def _apply_function(func: str | None, args, record: Any) -> Any:
    if func == "coalesce":
        for arg in args or ():
            value = _eval_expression(arg, record)
            if value is not None:
                return value
        return None
    if func == "if_null" and args:
        value = _eval_expression(args[0], record)
        if value is None and len(args) > 1:
            return _eval_expression(args[1], record)
        return value
    value = _eval_expression(args[0], record) if args else record
    return _apply_function_name(func, value)


def _apply_function_name(func: str | None, value: Any) -> Any:
    if func is None or value is None:
        return value
    try:
        if func == "upper":
            return value.upper() if isinstance(value, str) else value
        if func == "lower":
            return value.lower() if isinstance(value, str) else value
        if func == "length":
            return len(value) if hasattr(value, "__len__") else 0
        if func == "round":
            return round(value)
        if func == "abs":
            return abs(value)
        if func == "ceil":
            return math.ceil(value)
        if func == "floor":
            return math.floor(value)
        if func in ("int", "to_number"):
            return int(value)
        if func == "float":
            return float(value)
        if func in ("str", "string", "to_string"):
            return str(value)
        if func == "trim":
            return value.strip() if isinstance(value, str) else value
        if func == "ltrim":
            return value.lstrip() if isinstance(value, str) else value
        if func == "rtrim":
            return value.rstrip() if isinstance(value, str) else value
        if func == "reverse":
            return list(reversed(value))
        if func == "sort":
            return sorted(value)
        if func == "unique":
            return sorted(set(value))
        if func == "flatten":
            return _flatten(value)
        if func == "not_null":
            return value
        if func == "keys":
            return list(value.keys()) if isinstance(value, dict) else list(range(len(value)))
        if func == "values":
            return list(value.values()) if isinstance(value, dict) else list(value)
        if func == "type":
            return _json_type(value)
        if func == "tojson":
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if func == "fromjson":
            return json.loads(value)
        if func == "to_entries":
            return [{"key": k, "value": v} for k, v in value.items()]
        if func == "from_entries":
            return {entry["key"]: entry["value"] for entry in value}
    except (TypeError, ValueError, AttributeError):
        return None
    return value


def _flatten(value: Any) -> list:
    result = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, list):
                result.extend(_flatten(item))
            else:
                result.append(item)
    else:
        result.append(value)
    return result


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def cell_text(value: Any) -> str:
    """Render one value for the table and CSV writers.

    Both writers call this, so a value containing a comma, newline or quote is
    escaped identically (CSV quoting then happens in ``csv.writer`` on this
    text).  Nested structures use compact JSON.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return json_number(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def json_number(value):
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return "null"
        if value.is_integer():
            return str(int(value))
    return str(value)


def merge_headers(rows) -> list[str]:
    headers: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                headers.append(key)
    return headers
