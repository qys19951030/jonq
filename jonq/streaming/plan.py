"""Position-aware query compilation for the streaming engine.

The legacy parser (:mod:`jonq.query_parser`) works on bare token strings and
reports errors by token index.  The streaming engine needs character offsets so
syntax errors can say *where* the query went wrong, and it needs to reject
semantic errors (unknown fields, type mismatches) before it reads any data.

The plan produced here is intentionally narrow: it only describes the pieces
the bounded-memory engine implements directly -- projection, filtering,
grouping, ordering, distinct and limit.  Anything outside that subset raises a
:class:`~jonq.streaming.errors.StructuredQueryError` instead of silently
falling back, so behaviour is never ambiguous.

``compile_plan`` reuses the legacy parser for the heavy lifting and adds a
positional token map on top.  That keeps the accepted syntax identical to the
jq-backed path while giving the streaming engine offsets for its diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jonq.query_parser import (
    parse_condition_string,
    parse_query,
    tokenize_query,
)
from jonq.streaming.errors import StructuredQueryError
from jonq.tokenizer import tokenize_with_positions


# Field kinds the streaming plan understands, mirroring the legacy tuples.
FIELD_KINDS = {"field", "aggregation", "count_distinct", "function", "expression"}

_CLAUSE_KEYWORDS = {"if", "where", "group", "having", "sort", "limit", "from"}
_STOP_KEYWORDS = {
    "if": {"group", "having", "sort", "limit", "from"},
    "where": {"group", "having", "sort", "limit", "from"},
    "having": {"sort", "limit", "from"},
    "group": {"having", "sort", "limit", "from"},
    "sort": {"limit", "from"},
}


def clause_texts(query: str) -> dict[str, str]:
    """Map clause keyword -> raw source text of that clause.

    Clause boundaries come from the positional token stream, so quoted
    keywords inside literals and nested parentheses are handled naturally.
    """
    tokens = tokenize_with_positions(query)
    depth = 0
    boundaries: list[tuple[int, str]] = []
    for index, (value, position) in enumerate(tokens):
        lowered = value.lower()
        if value in ("(", "["):
            depth += 1
        elif value in (")", "]"):
            depth -= 1
        elif depth == 0 and lowered in _CLAUSE_KEYWORDS:
            if lowered == "from" and index > 0 and tokens[index - 1][0].lower() == "from":
                continue
            boundaries.append((index, lowered))

    clauses: dict[str, str] = {}
    for order, (token_index, keyword) in enumerate(boundaries):
        if keyword in clauses:
            continue
        # The clause body starts after the keyword (and after 'by' for group).
        body_start_token = token_index + 1
        if keyword == "group" and body_start_token < len(tokens) and tokens[body_start_token][0].lower() == "by":
            body_start_token += 1
        if body_start_token >= len(tokens):
            continue
        start = tokens[body_start_token][1]
        end = len(query)
        stops = _STOP_KEYWORDS.get(keyword, set())
        for later_index, later_keyword in boundaries[order + 1 :]:
            if later_keyword in stops:
                end = tokens[later_index][1]
                break
        clauses[keyword] = query[start:end].strip()
    return clauses


@dataclass(frozen=True)
class PlanField:
    kind: str
    alias: str
    path: str | None = None
    func: str | None = None
    param: str | None = None
    expression: str | None = None
    source_position: int | None = None

    @property
    def is_aggregate(self) -> bool:
        return self.kind in ("aggregation", "count_distinct")


@dataclass(frozen=True)
class StreamingPlan:
    query: str
    fields: tuple[PlanField, ...]
    condition: str | None = None
    condition_position: int | None = None
    condition_ast: Any = None
    condition_text: str | None = None
    group_by: tuple[str, ...] = ()
    group_by_positions: tuple[int | None, ...] = ()
    having: str | None = None
    having_position: int | None = None
    having_ast: Any = None
    having_text: str | None = None
    order_by: str | None = None
    order_by_position: int | None = None
    sort_direction: str = "asc"
    limit: str | None = None
    from_path: str | None = None
    from_position: int | None = None
    distinct: bool = False

    @property
    def has_aggregates(self) -> bool:
        return any(f.is_aggregate for f in self.fields)

    @property
    def is_grouped(self) -> bool:
        return bool(self.group_by)

    @property
    def needs_ordering(self) -> bool:
        return bool(self.order_by)

    @property
    def is_select_all(self) -> bool:
        return (
            len(self.fields) == 1
            and self.fields[0].kind == "field"
            and self.fields[0].path == "*"
        )


def _position_map(query: str) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {}
    for value, pos in tokenize_with_positions(query):
        positions.setdefault(value.lower(), []).append(pos)
    return positions


def _first_position(positions: dict[str, list[int]], value: str) -> int | None:
    hits = positions.get(str(value).lower())
    return hits[0] if hits else None


def _plan_fields(raw_fields, positions) -> tuple[PlanField, ...]:
    used: dict[str, int] = {}
    planned = []
    for raw in raw_fields:
        kind = raw[0]
        if kind == "field":
            _, path, alias = raw
            field = PlanField(kind="field", alias=alias, path=path)
        elif kind == "aggregation":
            _, func, param, alias = raw
            field = PlanField(kind="aggregation", alias=alias, func=func, param=param)
        elif kind == "count_distinct":
            _, param, alias = raw
            field = PlanField(kind="count_distinct", alias=alias, param=param)
        elif kind == "function":
            _, func, param, alias = raw
            field = PlanField(kind="function", alias=alias, func=func, param=param)
        elif kind == "expression":
            _, expression, alias = raw
            field = PlanField(kind="expression", alias=alias, expression=expression)
        else:  # pragma: no cover - guarded by _check_supported
            raise StructuredQueryError(f"Unsupported field kind '{kind}'")

        # Locate the alias token when present, else the first remaining token
        # matching the field's own text.
        candidates = positions.get(field.alias.lower(), [])
        cursor = used.get(field.alias.lower(), 0)
        pos = None
        if cursor < len(candidates):
            pos = candidates[cursor]
        elif field.path:
            hits = positions.get(field.path.lower(), [])
            if hits:
                pos = hits[min(cursor, len(hits) - 1)]
        used[field.alias.lower()] = cursor + 1
        planned.append(
            PlanField(
                kind=field.kind,
                alias=field.alias,
                path=field.path,
                func=field.func,
                param=field.param,
                expression=field.expression,
                source_position=pos,
            )
        )
    return tuple(planned)


def _check_supported(plan: StreamingPlan) -> None:
    if plan.from_path and "[]" in plan.from_path:
        raise StructuredQueryError(
            "The streaming engine does not support array-flattening 'from' paths; "
            "use the default execution mode for this query",
            query=plan.query,
            position=plan.from_position,
            kind="unsupported",
        )
    for field in plan.fields:
        if field.kind == "field" and field.path and "[]" in field.path:
            raise StructuredQueryError(
                f"Streaming does not support array-flattening field '{field.path}'",
                query=plan.query,
                position=field.source_position,
                kind="unsupported",
            )
    if plan.distinct and plan.has_aggregates:
        raise StructuredQueryError(
            "DISTINCT with aggregates is not supported in streaming mode",
            query=plan.query,
            position=_first_position(_position_map(plan.query), "distinct"),
            kind="unsupported",
        )


def compile_plan(query: str) -> StreamingPlan:
    """Parse ``query`` into a :class:`StreamingPlan` with error positions."""
    if not isinstance(query, str):
        raise StructuredQueryError("Query must be a string", kind="syntax")
    try:
        positions = _position_map(query)
        tokens = tokenize_query(query)
    except ValueError as exc:
        raise _syntax_error_from_message(query, str(exc)) from exc

    try:
        (
            fields,
            condition,
            group_by,
            having,
            order_by,
            sort_direction,
            limit,
            from_path,
            distinct,
        ) = parse_query(tokens)
    except ValueError as exc:
        raise _syntax_error_from_message(query, str(exc)) from exc

    texts = clause_texts(query)
    condition_ast = _build_condition_ast(texts.get("if") or texts.get("where"))
    having_ast = _build_condition_ast(texts.get("having"))

    plan = StreamingPlan(
        query=query,
        fields=_plan_fields(fields, positions),
        condition=condition,
        condition_position=_first_position(positions, "if") or _first_position(positions, "where"),
        condition_ast=condition_ast,
        condition_text=texts.get("if") or texts.get("where"),
        group_by=tuple(group_by or ()),
        group_by_positions=tuple(
            _first_position(positions, g) for g in (group_by or ())
        ),
        having=having,
        having_position=_first_position(positions, "having"),
        having_ast=having_ast,
        having_text=texts.get("having"),
        order_by=order_by,
        order_by_position=_first_position(positions, order_by) if order_by else None,
        sort_direction=sort_direction,
        limit=str(limit) if limit is not None else None,
        from_path=from_path,
        from_position=_first_position(positions, from_path) if from_path else None,
        distinct=distinct,
    )
    _check_supported(plan)
    return plan


def _build_condition_ast(text):
    if not text:
        return None
    try:
        return parse_condition_string(text)
    except ValueError:
        return None


def _syntax_error_from_message(query: str, message: str) -> StructuredQueryError:
    """Attach a character offset to a legacy parser ``ValueError`` message."""
    position = _locate_message(query, message)
    return StructuredQueryError(message, query=query, position=position, kind="syntax")


def _locate_message(query: str, message: str) -> int | None:
    if "position " in message:
        tail = message.rsplit("position ", 1)[-1].strip().rstrip(".")
        head = tail.split()[0] if tail else ""
        if head.isdigit():
            token_index = int(head)
            try:
                tokens = tokenize_with_positions(query)
                if token_index < len(tokens):
                    return tokens[token_index][1]
            except ValueError:
                return None

    # Messages without a token index still get a useful anchor.
    lowered = message.lower()
    if "unbalanced parentheses" in lowered:
        depth = 0
        last_open = None
        for index, char in enumerate(query):
            if char == "(":
                depth += 1
                last_open = index
            elif char == ")":
                depth -= 1
                if depth < 0:
                    return index
        if depth > 0 and last_open is not None:
            return last_open
        return None

    anchors = {
        "expected path after 'from'": "from",
        "expected 'by' after 'group'": "group",
        "having clause can only be used with group by": "having",
        "expected number after 'limit'": "limit",
        "expected field list after 'select'": "select",
        "expected alias after 'as'": "as",
        "expected field name after comma": ",",
    }
    for prefix, anchor in anchors.items():
        if lowered.startswith(prefix):
            try:
                tokens = tokenize_with_positions(query)
            except ValueError:
                return None
            for value, pos in tokens:
                if value.lower() == anchor:
                    return pos
            return None

    bad_function = "unknown function '"
    if lowered.startswith(bad_function):
        name = message[len(bad_function):].rstrip("'")
        try:
            tokens = tokenize_with_positions(query)
        except ValueError:
            return None
        for value, pos in tokens:
            if value.lower() == name.lower():
                return pos
        return None

    return None


def plan_field_names(plan: StreamingPlan) -> tuple[str, ...]:
    return tuple(f.alias for f in plan.fields)


def references_alias(plan: StreamingPlan, name: str) -> bool:
    return any(f.alias.lower() == name.lower() for f in plan.fields)
