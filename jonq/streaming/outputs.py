"""Output writers that share one row model.

All formats consume the same ordered ``dict`` rows and the same header list, so
they cannot disagree about row count or field order.  Value text comes from
:func:`jonq.streaming.rows.cell_text`, which means a value containing a comma,
newline or quote is escaped the same way by the table and CSV writers; the JSON
writer serialises the same dict so its escaping is JSON's, and its field order
is the dict insertion order (identical to the header list).

When the query uses ``select *`` the header set is only known after rows are
seen.  For those queries the writers spool projected rows to a temporary file
and replay them once headers are fixed, keeping memory bounded.
"""

from __future__ import annotations

import csv
import io
import json
import os
from typing import Iterator

from jonq.streaming.rows import cell_text, merge_headers
from jonq.streaming.spill import TempWorkspace

FORMATS = ("json", "jsonl", "csv", "table")


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class OutputSink:
    """A text destination: a file path or an in-memory buffer."""

    def __init__(self, path: str | None = None):
        self.path = path
        self._buffer = io.StringIO() if path is None else None
        self._handle = None
        self.closed = False

    def write(self, text: str) -> None:
        if self.path is None:
            self._buffer.write(text)
            return
        if self._handle is None:
            self._handle = open(self.path, "w", encoding="utf-8", newline="")
        self._handle.write(text)

    def close(self) -> str | None:
        if self.closed:
            return None
        self.closed = True
        if self._handle is not None:
            self._handle.close()
            return None
        return self._buffer.getvalue()

    @property
    def text(self) -> str:
        if self._buffer is None:
            return ""
        return self._buffer.getvalue()


def _iter_spooled(rows: Iterator[dict], workspace: TempWorkspace, headers_out: list[str]):
    """Spool rows, collect the union of keys in first-seen order, then replay.

    Replayed rows are normalised to the full header list (missing keys become
    ``None``) so every format sees the same field set *and* field order, even
    when input records have heterogeneous keys under ``select *``.
    """
    path = workspace.new_file(prefix="rows")
    ordered: list[str] = []
    seen: set[str] = set()
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
            handle.write(_dumps(row))
            handle.write("\n")

    def replay():
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    yield {key: row.get(key) for key in ordered}

    headers_out.extend(ordered)
    return replay()


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_json(rows: Iterator[dict], sink: OutputSink, *, single: bool = False, indent: bool = False) -> int:
    count = 0
    if not single:
        sink.write("[")
    for row in rows:
        if count:
            sink.write(", ")
        sink.write(json.dumps(row, ensure_ascii=False, indent=2 if indent else None, separators=None if indent else (",", ":")))
        count += 1
    if not single:
        sink.write("]")
    return count


def write_jsonl(rows: Iterator[dict], sink: OutputSink) -> int:
    count = 0
    for row in rows:
        if count:
            sink.write("\n")
        sink.write(_dumps(row))
        count += 1
    return count


def write_csv(rows: Iterator[dict], sink: OutputSink, *, headers: list[str] | None = None) -> int:
    return _write_delimited(rows, sink, headers=headers)


def _write_delimited(
    rows: Iterator[dict],
    sink: OutputSink,
    *,
    headers: list[str] | None,
) -> int:
    """Write CSV from one row stream.

    Headers come from the plan when known; otherwise the first row fixes them
    and later rows are written in that order (extra keys are appended to the
    header only when the caller pre-spooled, so streaming stays consistent).
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=",", quoting=csv.QUOTE_MINIMAL)

    if headers is None:
        first = next(rows, None)
        if first is None:
            return 0
        headers = list(first)
        writer.writerow(headers)
        writer.writerow([cell_text(first.get(key)) for key in headers])
        count = 1
    else:
        writer.writerow(headers)
        count = 0

    for row in rows:
        writer.writerow([cell_text(row.get(key)) for key in headers])
        count += 1

    sink.write(buffer.getvalue())
    return count


def write_table(
    rows: Iterator[dict],
    sink: OutputSink,
    *,
    headers: list[str] | None = None,
    workspace: TempWorkspace | None = None,
) -> int:
    """Render an aligned table without holding every row in memory.

    Rows are spooled to a temporary file, then replayed twice: once to measure
    column widths and once to write.  Memory therefore tracks the column set
    and widest cell, not the row count.  Long cells are never truncated -- a
    wide ``max_width`` disables the terminal-width shrinking that would
    otherwise cut them.
    """
    return _write_table_spooled(rows, sink, headers=headers, workspace=workspace)


def _write_table_spooled(rows, sink, *, headers, workspace) -> int:
    if workspace is None:
        import shutil
        import tempfile

        temp_dir = tempfile.mkdtemp(prefix="jonq_table_")
        try:
            return _write_table_from_dir(rows, sink, headers, temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    return _write_table_from_dir(rows, sink, headers, workspace.path)


def _write_table_from_dir(rows, sink, headers, directory) -> int:

    spool_path = os.path.join(directory, "table_rows.jsonl")
    ordered: list[str] = []
    seen: set[str] = set()
    count = 0
    with open(spool_path, "w", encoding="utf-8") as handle:
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
            handle.write(_dumps(row))
            handle.write("\n")
            count += 1

    if headers is None:
        headers = ordered
    if not headers:
        return 0

    widths = [len(str(header)) for header in headers]

    def iter_rows():
        with open(spool_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    for row in iter_rows():
        for index, header in enumerate(headers):
            width = len(_table_cell(row.get(header)))
            if width > widths[index]:
                widths[index] = width

    from jonq.table_utils import _build_row_line, _Colors

    colors = _Colors(False)
    lines = [
        f"{colors.BOLD}{colors.CYAN}{_build_row_line(headers, widths)}{colors.NC}",
        "|".join("-" * (width + 2) for width in widths),
    ]
    for row in iter_rows():
        lines.append(_build_row_line([_table_cell(row.get(h)) for h in headers], widths))
    sink.write("\n".join(lines))
    return count


def _table_cell(value):
    """Escape line breaks so a multi-line value stays on one table row."""
    text = cell_text(value)
    if "\n" in text or "\r" in text or "\t" in text:
        return text.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    return text


def write_rows(
    rows: Iterator[dict],
    sink: OutputSink,
    *,
    output_format: str,
    headers: list[str] | None = None,
    workspace: TempWorkspace | None = None,
    single_object: bool = False,
    indent: bool = False,
) -> int:
    """Dispatch to the format writer, spooling when headers are unknown."""
    if output_format not in FORMATS:
        raise ValueError(f"Unsupported output format '{output_format}'")

    if headers is None:
        # ``select *`` leaves the header set unknown until rows are seen.  Spool
        # them once for *every* format so the union of keys (first-seen order)
        # is applied consistently -- JSON must not silently emit per-row key
        # sets that disagree with the CSV/table header.
        if workspace is None:
            # Without a workspace the header set cannot be two-passed; fall back
            # to materialising, which is correct but less bounded.
            materialized_rows = list(rows)
            headers = merge_headers(materialized_rows)
            rows = iter(
                {header: row.get(header) for header in headers}
                for row in materialized_rows
            )
        else:
            collected: list[str] = []
            rows = _iter_spooled(rows, workspace, collected)
            headers = collected

    if output_format == "json":
        return write_json(rows, sink, single=single_object, indent=indent)
    if output_format == "jsonl":
        return write_jsonl(rows, sink)
    if output_format == "csv":
        return write_csv(rows, sink, headers=headers)
    return write_table(rows, sink, headers=headers, workspace=workspace)
