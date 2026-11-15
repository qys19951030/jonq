"""Memory-bounded sort, grouping and aggregation over record chunks.

Sorting never holds the full result set: each chunk is sorted in memory and
written to a spill file; the spill files are then merged with a heap, and equal
keys keep their original arrival order (stable).  Stability is encoded in the
sort key as a per-record sequence number, so it survives the merge.

Grouping aggregates chunk-by-chunk and merges partial states across chunks.
Partial states are associative for ``count``/``sum``/``min``/``max``; ``avg``
keeps ``(sum, count)`` so merged averages match an ungrouped computation.  Float
sums are compared with a configurable tolerance rather than for exact equality.

All intermediate state is bounded by the number of live groups/keys, which is
the unavoidable cost of a global sort or group-by; chunk size bounds the rest.
"""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterator

from jonq.streaming.spill import TempWorkspace


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=True)


def _loads(text: str):
    return json.loads(text)


SORTABLE_TYPES = (int, float, str, bool, type(None))


def sortable_key(value):
    """Map a value to a key usable across mixed types without raising.

    The key is a 3-tuple so values of different JSON types still form a total
    order.  The tuple is JSON round-trippable (as a list) which is why spilled
    entries store it directly instead of the original value.
    """
    if value is None:
        return (0, 0.0, "")
    if isinstance(value, bool):
        return (1, float(value), "")
    if isinstance(value, (int, float)):
        number = float(value)
        if number != number:  # NaN has no ordering; sort it first.
            return (2, float("-inf"), "")
        return (2, number, "")
    if isinstance(value, str):
        return (3, 0.0, value)
    return (4, 0.0, _dumps(value))


def _key_from_stored(stored) -> tuple:
    """Rebuild a sortable key tuple from its JSON representation."""
    if isinstance(stored, (list, tuple)) and len(stored) == 3:
        kind, number, text = stored
        return (kind, float(number), text)
    return sortable_key(stored)


@dataclass(frozen=True)
class SortEntry:
    key: tuple
    sequence: int
    payload: dict

    def as_record(self) -> dict:
        return {"k": list(self.key), "s": self.sequence, "r": self.payload}


def _entry_from_record(record: dict) -> "SortEntry":
    return SortEntry(_key_from_stored(record["k"]), record["s"], record["r"])


class ExternalSorter:
    """Sorts rows by a key, spilling stable chunk files and merging them.

    ``descending`` must be fixed at construction: each spill run is written in
    the requested order.  A merge can only stream a run from its head, so a
    descending merge needs descending runs, not ascending runs read backwards.
    """

    def __init__(
        self,
        workspace: TempWorkspace,
        *,
        chunk_size: int,
        fan_in: int = 64,
        label: str = "sort",
        descending: bool = False,
    ):
        self.workspace = workspace
        self.chunk_size = chunk_size
        self.fan_in = max(2, fan_in)
        self.label = label
        self.descending = descending
        self.spill_files: list[str] = []
        self.spill_bytes = 0
        self.merge_passes = 0
        self._buffer: list[tuple[tuple, int, dict]] = []
        self._sequence = 0

    def add(self, key, row: dict) -> None:
        self._buffer.append((sortable_key(key), self._sequence, row))
        self._sequence += 1
        if len(self._buffer) >= self.chunk_size:
            self._flush()

    def _ordered_buffer(self):
        self._buffer.sort(key=lambda item: (item[0], item[1]))
        if self.descending:
            return _reverse_by_key_stable(self._buffer)
        return self._buffer

    def _flush(self) -> None:
        if not self._buffer:
            return
        ordered = self._ordered_buffer()
        path = self.workspace.new_file(prefix=f"{self.label}_spill")
        written = 0
        with open(path, "w", encoding="utf-8") as handle:
            for key, sequence, row in ordered:
                line = _dumps({"k": list(key), "s": sequence, "r": row})
                handle.write(line)
                handle.write("\n")
                written += len(line.encode("utf-8")) + 1
        self.spill_files.append(path)
        self.spill_bytes += written
        self._buffer = []

    def finish(self, *, descending: bool | None = None) -> Iterator[dict]:
        """Yield sorted rows, merging spill files in bounded fan-in passes."""
        if descending is not None and descending != self.descending:
            raise ValueError("descending must be fixed when the sorter is created")
        if self._buffer and not self.spill_files:
            for _key, _sequence, row in self._ordered_buffer():
                yield row
            self._buffer = []
            return
        if self._buffer:
            self._flush()

        while len(self.spill_files) > self.fan_in:
            self.spill_files = self._merge_pass(self.spill_files)
            self.merge_passes += 1

        for entry in _merge_files(self.spill_files, descending=self.descending):
            yield entry.payload

    def _merge_pass(self, files: list[str]) -> list[str]:
        merged_files: list[str] = []
        for start in range(0, len(files), self.fan_in):
            group = files[start : start + self.fan_in]
            path = self.workspace.new_file(prefix=f"{self.label}_merge")
            with open(path, "w", encoding="utf-8") as out:
                # Intermediate files must keep the sorter's direction: the next
                # pass reads them from the head.
                for entry in _merge_files(group, descending=self.descending):
                    line = _dumps(entry.as_record())
                    out.write(line)
                    out.write("\n")
            merged_files.append(path)
        return merged_files


def _reverse_by_key_stable(buffer):
    """Reverse key order but keep arrival order inside each equal-key run."""
    groups = []
    index = 0
    while index < len(buffer):
        key = buffer[index][0]
        end = index
        while end < len(buffer) and buffer[end][0] == key:
            end += 1
        groups.append(buffer[index:end])
        index = end
    ordered = []
    for group in reversed(groups):
        ordered.extend(group)
    return ordered


class _Descending:
    """Wraps a sort key so ordering is reversed while ties still compare equal."""

    __slots__ = ("key",)

    def __init__(self, key):
        self.key = key

    def __eq__(self, other):
        return self.key == other.key

    def __lt__(self, other):
        return self.key > other.key

    def __hash__(self):
        return hash(self.key)


def _merge_files(files: list[str], *, descending: bool = False) -> Iterator[SortEntry]:
    if not files:
        return
    heap: list[tuple] = []
    handles = []

    def heap_key(entry: SortEntry, index: int):
        primary = entry.key
        if descending:
            primary = _Descending(primary)
        # Sequence stays ascending inside a key group so equal keys keep arrival
        # order in both directions.
        return (primary, entry.sequence, index)

    try:
        for index, path in enumerate(files):
            handle = open(path, "r", encoding="utf-8")
            handles.append(handle)
            line = handle.readline()
            if line.strip():
                entry = _entry_from_record(_loads(line))
                heapq.heappush(heap, (*heap_key(entry, index), index, entry))
        while heap:
            _, _, _, index, entry = heapq.heappop(heap)
            yield entry
            handle = handles[index]
            line = handle.readline()
            if line.strip():
                next_entry = _entry_from_record(_loads(line))
                heapq.heappush(heap, (*heap_key(next_entry, index), index, next_entry))
    finally:
        for handle in handles:
            handle.close()


# ---------------------------------------------------------------------------
# Grouped aggregation
# ---------------------------------------------------------------------------

@dataclass
class AggregateState:
    count: int = 0
    sum: float = 0.0
    min: Any = None
    max: Any = None
    distinct: set = field(default_factory=set)

    def merge(self, other: "AggregateState") -> None:
        self.count += other.count
        self.sum += other.sum
        if other.min is not None and (self.min is None or other.min < self.min):
            self.min = other.min
        if other.max is not None and (self.max is None or other.max > self.max):
            self.max = other.max
        self.distinct |= other.distinct

    def to_record(self) -> dict:
        distinct_values = None
        if self.distinct:
            distinct_values = [_freeze_key(value) for value in self.distinct]
        return {
            "count": self.count,
            "sum": self.sum,
            "min": self.min,
            "max": self.max,
            "distinct": distinct_values,
        }

    @classmethod
    def from_record(cls, record) -> "AggregateState":
        if isinstance(record, AggregateState):
            return record
        values = record.get("distinct")
        state = cls(
            count=record.get("count", 0),
            sum=record.get("sum", 0.0),
            min=record.get("min"),
            max=record.get("max"),
            distinct=set(values or ()),
        )
        return state


def _freeze_key(value):
    """Make a value JSON-serialisable while keeping it hashable/comparable."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _dumps(value)


class GroupedAggregator:
    """Chunk-local group aggregation with spill-and-merge across chunks.

    Partial states are written to a spill file whenever the in-memory group map
    reaches ``chunk_size`` entries (or the memory budget requires it); the spill
    files are merged into one state map at the end.  Because the partial states
    are associative, the merged result equals a one-pass computation.
    """

    def __init__(self, workspace: TempWorkspace, *, chunk_size: int, label: str = "group"):
        self.workspace = workspace
        self.chunk_size = max(1, chunk_size)
        self.label = label
        self._groups: dict[tuple, tuple] = {}
        self.spill_files: list[str] = []
        self.spill_bytes = 0
        self.chunks_flushed = 0

    def _frozen_key(self, key_values):
        return tuple(sortable_key(value) for value in key_values)

    def get(self, key_values, first_row=None, first_record=None):
        """Return the mutable state dict for a group, creating it if needed.

        ``first_row``/``first_record`` are kept so projection of non-aggregate
        fields can use the first record of the group, matching the legacy
        ``.[0].field`` semantics even after a spill.
        """
        frozen = self._frozen_key(key_values)
        entry = self._groups.get(frozen)
        if entry is None:
            entry = {
                "key_values": list(key_values),
                "states": {},
                "first_row": first_row,
                "first_record": first_record,
            }
            self._groups[frozen] = entry
        elif entry["first_row"] is None and first_row is not None:
            entry["first_row"] = first_row
            entry["first_record"] = first_record
        return entry["states"]

    def maybe_spill(self) -> None:
        if len(self._groups) >= self.chunk_size:
            self.spill()

    def spill(self) -> None:
        if not self._groups:
            return
        path = self.workspace.new_file(prefix=f"{self.label}_spill")
        written = 0
        with open(path, "w", encoding="utf-8") as handle:
            for entry in self._groups.values():
                record = {
                    "k": entry["key_values"],
                    "s": {
                        alias: state.to_record()
                        for alias, state in entry["states"].items()
                    },
                    "r": entry["first_row"],
                }
                line = _dumps(record)
                handle.write(line)
                handle.write("\n")
                written += len(line.encode("utf-8")) + 1
        self.spill_files.append(path)
        self.spill_bytes += written
        self.chunks_flushed += 1
        self._groups = {}

    def finish(self):
        """Merge every chunk's partial groups into the final entry list."""
        merged: dict[tuple, dict] = {}
        pending = list(self._groups.values()) if self._groups else []

        def merge_entry(key_values, states, first_row):
            frozen = self._frozen_key(key_values)
            existing = merged.get(frozen)
            if existing is None:
                existing = {
                    "key_values": list(key_values),
                    "states": {
                        alias: AggregateState.from_record(record)
                        for alias, record in states.items()
                    },
                    "first_row": first_row,
                }
                merged[frozen] = existing
                return
            if existing["first_row"] is None and first_row is not None:
                existing["first_row"] = first_row
            target_states = existing["states"]
            for alias, record in states.items():
                incoming = (
                    record
                    if isinstance(record, AggregateState)
                    else AggregateState.from_record(record)
                )
                target = target_states.get(alias)
                if target is None:
                    target_states[alias] = incoming
                else:
                    target.merge(incoming)

        for entry in pending:
            merge_entry(entry["key_values"], entry["states"], entry["first_row"])

        for path in self.spill_files:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = _loads(line)
                    merge_entry(record["k"], record["s"], record.get("r"))

        return list(merged.values())


def floats_equal(left: float, right: float, tolerance: float) -> bool:
    if left == right:
        return True
    if math.isinf(left) or math.isinf(right) or math.isnan(left) or math.isnan(right):
        return False
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))
