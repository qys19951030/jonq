"""Scan stage: chunked read -> bad-record policy -> filter -> project.

The scanner owns the input-side counters so the engine can merge them into the
reported statistics after the writer drains the row stream.  It never holds
more than one reader batch at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from jonq.streaming.config import StreamingConfig
from jonq.streaming.readers import InputSource, open_input
from jonq.streaming.rows import evaluate_condition, project_record


@dataclass
class ProjectedRow:
    """A projected output row plus the raw record it came from.

    Group-by and grouping-level projection need values from the raw record that
    may not be part of the selected projection (e.g. ``select count(*) group by
    city`` still needs ``city``).  Keeping the raw record on the stream costs
    one reference per in-flight row, not a copy.
    """

    row: dict
    record: object


class Scanner:
    def __init__(self, sources: list[InputSource], plan, config: StreamingConfig):
        self.sources = sources
        self.plan = plan
        self.config = config
        self.records_read = 0
        self.records_skipped = 0
        self.bytes_read = 0
        self.chunk_count = 0
        self.records_filtered = 0
        self.records_selected = 0
        self.skipped_records: list[dict] = []
        self.finished = False

    def __iter__(self) -> Iterator[dict]:
        for batch in self._batches():
            self.chunk_count += 1
            self.bytes_read += batch.bytes_read
            for bad in batch.bad:
                self.records_skipped += 1
                self.skipped_records.append(
                    {
                        "source": bad.source,
                        "line": bad.line_number,
                        "record": bad.record_index,
                        "reason": bad.reason,
                    }
                )
            for envelope in batch.records:
                self.records_read += 1
                if not evaluate_condition(self.plan.condition_ast, envelope.value):
                    self.records_filtered += 1
                    continue
                self.records_selected += 1
                yield ProjectedRow(
                    row=project_record(envelope.value, self.plan),
                    record=envelope.value,
                )
        self.finished = True

    def _batches(self):
        for source in self.sources:
            yield from source.reader.iter_batches()

    def apply_to(self, stats) -> None:
        stats.records_read += self.records_read
        stats.records_skipped += self.records_skipped
        stats.bytes_read += self.bytes_read
        stats.chunk_count += self.chunk_count
        stats.records_filtered += self.records_filtered
        stats.records_selected += self.records_selected


def build_scanner(path, config: StreamingConfig, plan, *, from_segments=None) -> Scanner:
    sources = open_input(path, config, from_segments=from_segments)
    reader_config = config.resolved_inputs()
    return Scanner(sources, plan, reader_config)
