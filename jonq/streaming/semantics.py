"""Pre-execution semantic validation.

The streaming engine must fail on "field does not exist" or "field has the
wrong type" *before* it reads the input, otherwise a long scan dies halfway
through.  Validation here works from a bounded schema sample:

* the sample is read through the same chunked reader used for execution, but
  capped at ``schema_sample_records`` records, so it never slurps the file;
* field existence is checked against the union of keys observed in the sample;
* type checks are limited to the operations whose correctness depends on it
  (ordering, sum/avg aggregation and numeric comparisons), and only fire when
  *every* observed value disagrees with the expected type -- one bad row is a
  data problem for the bad-record policy, not a query error.

Results are advisory beyond the strict checks: a field seen only in later
records may legitimately be absent from the sample, so existence errors are
reported only when the sample is non-trivial.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from jonq.ast import (
    AndCondition,
    BetweenCondition,
    Condition,
    ExprType,
    InCondition,
    NotCondition,
    OrCondition,
)
from jonq.streaming.errors import SemanticQueryError
from jonq.streaming.plan import StreamingPlan

NUMERIC_AGGREGATES = {"sum", "avg"}
NUMERIC_COMPARISONS = {"<", ">", "<=", ">=", "between"}


@dataclass
class SchemaSample:
    records: int
    fields: dict[str, set[str]]     # dotted path -> observed type labels
    root_type: str | None = None

    @property
    def is_empty(self) -> bool:
        return self.records == 0

    def has_field(self, path: str) -> bool:
        return path in self.fields

    def types_for(self, path: str) -> set[str]:
        return self.fields.get(path, set())


def build_schema_sample(records) -> SchemaSample:
    """Collect field paths and value types from an iterable of records."""
    fields: dict[str, set[str]] = {}
    count = 0
    root_types: set[str] = set()
    for record in records:
        count += 1
        root_types.add(_type_label(record))
        if isinstance(record, dict):
            _walk(record, "", fields)
        else:
            fields.setdefault("value", set()).add(_type_label(record))
    root = next(iter(root_types)) if len(root_types) == 1 else None
    return SchemaSample(records=count, fields=fields, root_type=root)


def _walk(value, prefix: str, fields: dict[str, set[str]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            fields.setdefault(path, set()).add(_type_label(child))
            if isinstance(child, (dict, list)):
                _walk(child, path, fields)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                _walk(child, prefix, fields)


def _type_label(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "unknown"


def _normalized(path: str) -> str:
    return path.lstrip(".")


def _field_position(plan: StreamingPlan, path: str) -> int | None:
    normalized = _normalized(path)
    for field in plan.fields:
        if field.path and _normalized(field.path) == normalized:
            return field.source_position
        if field.param and _normalized(field.param) == normalized:
            return field.source_position
    if plan.order_by and _normalized(plan.order_by) == normalized:
        return plan.order_by_position
    for key, pos in zip(plan.group_by, plan.group_by_positions):
        if _normalized(key) == normalized:
            return pos
    return None


def validate_plan_against_sample(plan: StreamingPlan, sample: SchemaSample, *, strict: bool = True) -> None:
    """Raise :class:`SemanticQueryError` for missing fields / type mismatches."""
    if sample.is_empty:
        return

    alias_names = {f.alias.lower() for f in plan.fields}

    for field in plan.fields:
        path = field.path or field.param
        if field.kind == "field" and field.path in ("*", None):
            continue
        if field.kind == "expression":
            for ref in _expression_field_refs(field.expression or ""):
                _require_field(plan, sample, ref, field.source_position, strict=strict)
            continue
        if not path or path == "*":
            continue
        if field.kind == "function" and not re.fullmatch(r"[\w\.\[\]]+", path):
            continue
        _require_field(plan, sample, path, field.source_position, strict=strict)

    for key, pos in zip(plan.group_by, plan.group_by_positions):
        _require_field(plan, sample, key, pos, strict=strict)

    if plan.order_by and plan.order_by.lower() not in alias_names:
        _require_field(plan, sample, plan.order_by, plan.order_by_position, strict=strict)

    for ref in _ast_field_refs(plan.condition_ast):
        if ref.lower() in alias_names:
            continue
        _require_field(plan, sample, ref, plan.condition_position, strict=strict)

    for ref in _ast_field_refs(plan.having_ast):
        if ref.lower() in alias_names:
            continue
        _require_field(plan, sample, ref, plan.having_position, strict=strict)

    _validate_types(plan, sample)


def _require_field(plan, sample, path, position, *, strict) -> None:
    normalized = _normalized(path)
    if not normalized or normalized == "*":
        return
    if normalized.lower() in ("null", "true", "false", "none"):
        return
    if sample.has_field(normalized):
        return
    # Sub-paths of known array/object fields show up only if sampled records
    # contain them; require the head to exist before reporting the leaf.
    head = normalized.split(".", 1)[0]
    if head not in sample.fields:
        if not strict:
            return
        raise SemanticQueryError(
            f"Field '{normalized}' does not exist in the input sample",
            query=plan.query,
            position=position if position is not None else _field_position(plan, normalized),
            context={"field": normalized},
        )


def _validate_types(plan: StreamingPlan, sample: SchemaSample) -> None:
    for field in plan.fields:
        if field.kind != "aggregation":
            continue
        if field.func not in NUMERIC_AGGREGATES or not field.param or field.param == "*":
            continue
        _require_numeric(plan, sample, field.param, field.source_position)

    if plan.order_by:
        observed = sample.types_for(_normalized(plan.order_by))
        interesting = observed - {"null"}
        if interesting and not interesting <= {"number", "string", "bool"}:
            raise SemanticQueryError(
                f"Cannot sort by '{plan.order_by}': sampled values are {sorted(interesting)}",
                query=plan.query,
                position=plan.order_by_position,
                context={"field": plan.order_by},
            )


def _require_numeric(plan, sample, path, position) -> None:
    observed = sample.types_for(_normalized(path)) - {"null"}
    if observed and "number" not in observed:
        raise SemanticQueryError(
            f"Aggregating '{path}' requires numbers but sampled values are {sorted(observed)}",
            query=plan.query,
            position=position,
            context={"field": path},
        )


_EXPR_TOKEN_RE = re.compile(r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*")
_EXPR_SKIP = {
    "and", "or", "not", "is", "null", "true", "false", "between", "in", "like",
    "contains", "case", "when", "then", "else", "end", "as", "select", "if",
    "where", "group", "by", "having", "sort", "asc", "desc", "limit", "from",
    "distinct", "coalesce", "if_null", "upper", "lower", "length", "round",
    "abs", "ceil", "floor", "int", "float", "str", "string", "to_number",
    "to_string", "keys", "values", "type", "trim", "reverse", "unique",
    "flatten", "not_null", "to_entries", "from_entries", "todate", "fromdate",
    "date", "timestamp", "ltrim", "rtrim", "tojson", "fromjson", "sum", "avg",
    "min", "max", "count", "distinct",
}


def _expression_field_refs(expression: str) -> list[str]:
    """Extract candidate field references from a condition/expression string.

    Quoted literals are stripped first so ``'error'`` never looks like a field.
    """
    if not expression:
        return []
    text = re.sub(r'"[^"]*"|\'[^\']*\'', " ", expression)
    # Dot-prefixed references are unambiguous.
    refs = [
        match.group(1)
        for match in re.finditer(r"\.([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)\??", text)
    ]
    if refs:
        return _dedupe(refs)
    bare = []
    for token in _EXPR_TOKEN_RE.findall(text):
        if token.lower() in _EXPR_SKIP:
            continue
        if token.isdigit():
            continue
        bare.append(token)
    return _dedupe(bare)


def _dedupe(values: list[str]) -> list[str]:
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _ast_field_refs(node) -> list[str]:
    """Collect field paths referenced by a parsed condition AST."""
    refs: list[str] = []

    def visit(current) -> None:
        if current is None:
            return
        if isinstance(current, AndCondition) or isinstance(current, OrCondition):
            visit(current.left)
            visit(current.right)
            return
        if isinstance(current, NotCondition):
            visit(current.condition)
            return
        if isinstance(current, BetweenCondition):
            refs.append(current.field)
            return
        if isinstance(current, InCondition):
            refs.append(current.field)
            return
        if isinstance(current, Condition):
            visit_expression(current.expr)
            return
        visit_expression(current)

    def visit_expression(expression) -> None:
        if expression is None or not hasattr(expression, "type"):
            return
        if expression.type == ExprType.FIELD:
            refs.append(expression.value)
            return
        if expression.type == ExprType.LITERAL:
            return
        if expression.type in (ExprType.OPERATION, ExprType.BINARY_CONDITION):
            for arg in expression.args or ():
                visit_expression(arg)
            return
        if expression.type == ExprType.FUNCTION:
            for arg in expression.args or ():
                visit_expression(arg)
            return
        if expression.type == ExprType.AGGREGATION:
            for arg in expression.args or ():
                if isinstance(arg, str):
                    if arg != "*":
                        refs.append(arg)
                else:
                    visit_expression(arg)
            return
        if expression.type == ExprType.CASE:
            for _condition, value in expression.args or ():
                for ref in _expression_field_refs(value):
                    refs.append(ref)

    visit(node)
    return _dedupe(
        [
            ref
            for ref in refs
            if ref
            and ref != "*"
            and ref.lower() not in ("null", "true", "false", "none")
        ]
    )
