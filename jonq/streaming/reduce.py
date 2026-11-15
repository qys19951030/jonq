"""Reduce stage: turn a projected row stream into the final ordered rows.

Four strategies, all bounded by chunk size (plus the number of distinct
groups/keys, which a global group-by or sort necessarily keeps):

``plain``
    Pass rows through, applying ``limit`` with a counter.
``ordered``
    External stable merge sort: chunk-sorted spill files merged by key, with
    identical keys keeping input order in both directions (descending spill
    runs are written descending, so stability survives the merge).
``grouped``
    Chunk-local aggregation merged across chunks.  Partial states are
    associative, so counts/sums/minima/maxima equal a one-shot computation;
    ``avg`` is derived from merged ``sum``/``count``.
``distinct``
    Sort by the full row, then drop adjacent duplicates -- matching the legacy
    ``unique`` semantics (sorted, deduplicated).
"""

from __future__ import annotations

import json
from typing import Iterator

from jonq.streaming.rows import MISSING, resolve_path
from jonq.streaming.sortagg import (
    AggregateState,
    ExternalSorter,
    GroupedAggregator,
    sortable_key,
)


def _row_of(item):
    return item.row if hasattr(item, "row") else item


def _record_of(item):
    return item.record if hasattr(item, "record") else item


# ---------------------------------------------------------------------------
# Plain / limit
# ---------------------------------------------------------------------------

def reduce_plain(rows: Iterator[dict], plan, stats) -> Iterator[dict]:
    limit = int(plan.limit) if plan.limit else None
    emitted = 0
    for item in rows:
        row = _row_of(item)
        if limit is not None and emitted >= limit:
            stats.records_limited_out += 1
            continue
        emitted += 1
        yield row


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def reduce_ordered(rows, plan, config, workspace, stats) -> Iterator[dict]:
    sorter = ExternalSorter(
        workspace,
        chunk_size=config.chunk_size,
        fan_in=config.merge_fan_in,
        label="sort",
        descending=plan.sort_direction == "desc",
    )
    for item in rows:
        row = _row_of(item)
        value = _row_key(row, _record_of(item), plan.order_by)
        sorter.add(value, row)

    limit = int(plan.limit) if plan.limit else None
    emitted = 0
    for row in sorter.finish():
        if limit is not None and emitted >= limit:
            stats.records_limited_out += 1
            continue
        emitted += 1
        yield row
    _record_spill(stats, sorter)


def _record_spill(stats, sorter) -> None:
    stats.spill_chunk_count = max(stats.spill_chunk_count, len(sorter.spill_files))
    stats.spill_bytes = max(stats.spill_bytes, sorter.spill_bytes)
    stats.merge_passes = max(stats.merge_passes, sorter.merge_passes)


def _row_key(row, record, path):
    if row is not None and path in row:
        return row[path]
    value = resolve_path(record if record is not None else row, path)
    if value is MISSING and row is not None:
        value = resolve_path(row, path)
    return None if value is MISSING else value


# ---------------------------------------------------------------------------
# Distinct
# ---------------------------------------------------------------------------

def reduce_distinct(rows, plan, config, workspace, stats) -> Iterator[dict]:
    sorter = ExternalSorter(
        workspace,
        chunk_size=config.chunk_size,
        fan_in=config.merge_fan_in,
        label="distinct",
    )
    for item in rows:
        row = _row_of(item)
        sorter.add(_row_signature(row), row)

    previous = MISSING
    deduped = []
    for row in sorter.finish():
        key = _row_signature(row)
        if key == previous:
            stats.records_deduplicated += 1
            continue
        previous = key
        deduped.append(row)
    _record_spill(stats, sorter)

    # Deduplication yields rows sorted by value (the legacy `unique`
    # semantics).  A query-level sort/limit then runs over that stream so the
    # final order and truncation match the whole-file path.
    iterator = iter(deduped)
    if plan.needs_ordering:
        yield from reduce_ordered(iterator, plan, config, workspace, stats)
        return
    yield from reduce_plain(iterator, plan, stats)


def _row_signature(row: dict):
    return tuple(sortable_key(value) for value in row.values())


# ---------------------------------------------------------------------------
# Aggregate state helpers
# ---------------------------------------------------------------------------

def _aggregate_input(row, record, field):
    if field.kind == "count_distinct":
        value = resolve_path(record, field.param) if record is not None else MISSING
        return None if value is MISSING else value
    if field.func == "count" and field.param == "*":
        return 1
    value = row.get(field.alias)
    if value is None and record is not None and field.param:
        resolved = resolve_path(record, field.param)
        if resolved is not MISSING:
            value = resolved
    return value


def _accumulate(state: AggregateState, field, value) -> None:
    func = field.func if field.kind == "aggregation" else "count_distinct"
    if func == "count":
        if isinstance(value, list):
            state.count += len(value)
        elif value is not None:
            state.count += 1
        return
    if func == "count_distinct":
        if value is not None:
            state.count += 1
            state.distinct.add(_freeze(value))
        return
    for number in _numbers(value):
        state.count += 1
        state.sum += number
        if state.min is None or number < state.min:
            state.min = number
        if state.max is None or number > state.max:
            state.max = number


def _numbers(value):
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [value]
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_numbers(item))
        return out
    return []


def _finalize(field, state: AggregateState):
    func = field.func if field.kind == "aggregation" else "count_distinct"
    if func == "count":
        return state.count
    if func == "count_distinct":
        return len(state.distinct)
    if func == "sum":
        return _normalize_number(state.sum)
    if func == "avg":
        if not state.count:
            return None
        return _normalize_number(state.sum / state.count)
    if func == "min":
        return state.min
    if func == "max":
        return state.max
    return None


def _normalize_number(value):
    """Render whole floats as ints, matching jq's number formatting."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _finalize_empty(field):
    if field.func in ("count", "sum"):
        return 0
    return None


def _freeze(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return value


# ---------------------------------------------------------------------------
# Global aggregation (no group by)
# ---------------------------------------------------------------------------

def reduce_global_aggregate(rows, plan, config, stats) -> Iterator[dict]:
    aggregate_fields = [f for f in plan.fields if f.kind in ("aggregation", "count_distinct")]
    states = {field.alias: AggregateState() for field in aggregate_fields}
    first_row = None
    first_record = None
    saw_row = False

    for item in rows:
        saw_row = True
        if first_row is None:
            first_row = _row_of(item)
            first_record = _record_of(item)
        for field in aggregate_fields:
            _accumulate(
                states[field.alias],
                field,
                _aggregate_input(_row_of(item), _record_of(item), field),
            )

    output = {}
    for field in plan.fields:
        if field.kind in ("aggregation", "count_distinct"):
            output[field.alias] = _finalize(field, states[field.alias])
        elif not saw_row:
            output[field.alias] = None
        else:
            output[field.alias] = _row_key(first_row, first_record, field.path or field.alias)

    if not saw_row:
        for field in aggregate_fields:
            output[field.alias] = (
                0 if field.kind == "count_distinct" else _finalize_empty(field)
            )
    yield output


# ---------------------------------------------------------------------------
# Grouped aggregation
# ---------------------------------------------------------------------------

def reduce_grouped(rows, plan, config, workspace, stats) -> Iterator[dict]:
    aggregate_fields = [f for f in plan.fields if f.kind in ("aggregation", "count_distinct")]
    aggregator = GroupedAggregator(workspace, chunk_size=config.chunk_size)

    for item in rows:
        row = _row_of(item)
        record = _record_of(item)
        key_values = tuple(_row_key(row, record, key) for key in plan.group_by)
        states = aggregator.get(key_values, first_row=row, first_record=record)
        for field in aggregate_fields:
            state = states.get(field.alias)
            if state is None:
                state = states[field.alias] = AggregateState()
            _accumulate(
                state,
                field,
                _aggregate_input(row, record, field),
            )
        aggregator.maybe_spill()

    stats.spill_chunk_count = max(stats.spill_chunk_count, len(aggregator.spill_files))
    stats.spill_bytes = max(stats.spill_bytes, aggregator.spill_bytes)
    entries = aggregator.finish()
    stats.groups_formed = len(entries)
    entries.sort(
        key=lambda entry: tuple(sortable_key(value) for value in entry["key_values"])
    )

    output_rows = []
    for entry in entries:
        row = {}
        first_row = entry["first_row"] or {}
        for field in plan.fields:
            if field.kind in ("aggregation", "count_distinct"):
                state = entry["states"].get(field.alias, AggregateState())
                row[field.alias] = _finalize(field, state)
            else:
                # Grouped projection of a non-aggregate field uses the group's
                # first record, which is also the projected row for that field.
                row[field.alias] = first_row.get(field.alias)
        output_rows.append(row)

    if plan.having_ast is not None:
        from jonq.streaming.rows import evaluate_condition

        kept = []
        for row in output_rows:
            if evaluate_condition(plan.having_ast, row):
                kept.append(row)
            else:
                stats.records_filtered_by_having += 1
        output_rows = kept

    iterator = iter(output_rows)
    if plan.needs_ordering:
        yield from reduce_ordered(
            iterator, plan, config, workspace, stats
        )
        return
    yield from reduce_plain(iterator, plan, stats)
