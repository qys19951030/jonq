"""Chunked input readers for the streaming engine.

Three input shapes are supported; the shape is detected unless the caller pins
it through :class:`~jonq.streaming.config.StreamingConfig`:

``array``
    One JSON document whose root is an array.  Elements are decoded from a
    bounded text buffer, so peak memory tracks the largest single record rather
    than the whole file.
``jsonl``
    One JSON value per line.  Lines are read incrementally and line numbers are
    preserved so a bad record can be reported precisely.
``manifest``
    A JSON array (or plain newline list) of file paths.  Duplicate paths are
    removed with a count, and each referenced file is read as an array, a JSONL
    stream, or a single object.

Readers yield :class:`RecordEnvelope` objects carrying the source label, line
number and encoded size needed for bad-record reporting, memory accounting and
statistics.  Malformed records are surfaced as :class:`BadRecord` items so the
caller can apply the configured policy; with ``error`` the reader raises, with
``skip`` the engine counts and moves on.  For a root array the parser resyncs to
the next top-level element after a bad one, so skipping never drops a valid
record; if the remaining document cannot be resynchronised the error is raised.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterator

from jonq.streaming.config import StreamingConfig
from jonq.streaming.errors import BadRecordError, ManifestError, MemoryLimitError, PreflightError

_READ_SIZE = 64 * 1024
_JSONL_SUFFIXES = (".jsonl", ".ndjson")


@dataclass
class RecordEnvelope:
    value: object
    source: str
    line_number: int | None = None
    record_index: int | None = None
    encoded_size: int = 0
    raw_text: str | None = None


@dataclass
class BadRecord:
    source: str
    reason: str
    line_number: int | None = None
    record_index: int | None = None
    raw_text: str | None = None

    def to_error(self) -> BadRecordError:
        return BadRecordError(
            f"Could not parse record in {self.source}",
            source=self.source,
            line_number=self.line_number,
            record_index=self.record_index,
            record_text=self.raw_text,
            reason=self.reason,
        )


@dataclass
class RecordBatch:
    """A chunk's worth of reader output: good records plus bad ones."""

    records: list[RecordEnvelope] = field(default_factory=list)
    bad: list[BadRecord] = field(default_factory=list)
    bytes_read: int = 0
    record_bytes: int = 0
    chunk_number: int = 0


def _size_of(value) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def detect_input_kind(path: str) -> str:
    """Return ``array``, ``jsonl`` or ``single`` for one input file."""
    if path.lower().endswith(_JSONL_SUFFIXES):
        return "jsonl"
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        while True:
            char = handle.read(1)
            if not char:
                return "jsonl"
            if char.isspace():
                continue
            return "array" if char == "[" else "single"


def _is_manifest_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith((".manifest", ".manifest.json", ".paths", ".list"))


def _read_manifest_paths(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    if not text.strip():
        raise ManifestError(f"Manifest '{path}' is empty.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [line.strip() for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        data = data.get("files") or data.get("paths") or data.get("inputs")
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ManifestError(
            f"Manifest '{path}' must contain a list of file paths."
        )
    if not data:
        raise ManifestError(f"Manifest '{path}' lists no files.")
    return data


def resolve_manifest(paths, *, manifest_dir: str | None = None) -> tuple[list[str], int]:
    """Resolve manifest entries relative to the manifest, dropping duplicates.

    Returns ``(resolved_paths, duplicates_removed)`` preserving first-seen order.
    """
    resolved: list[str] = []
    seen: set[str] = set()
    duplicates = 0
    for entry in paths:
        if not os.path.isabs(entry) and manifest_dir:
            candidate = os.path.normpath(os.path.join(manifest_dir, entry))
        else:
            candidate = os.path.normpath(entry)
        key = os.path.realpath(candidate)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        resolved.append(candidate)
    return resolved, duplicates


class _StreamingArrayParser:
    """Incrementally decode a root JSON array from a text file.

    The parser keeps only the current buffer window plus the partially decoded
    value.  ``next_record`` returns ``(value, line_number, encoded_size)`` or
    ``None`` at the closing bracket.
    """

    def __init__(self, path: str):
        self.path = path
        self.handle = open(path, "r", encoding="utf-8")
        self.buffer = ""
        self.offset = 0          # characters consumed from the file
        self.consumed = 0        # characters consumed from self.buffer
        self.eof = False
        self.state = "start"
        self.decoder = json.JSONDecoder()
        self.line = 1

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()

    def _fill(self) -> bool:
        if self.eof:
            return False
        chunk = self.handle.read(_READ_SIZE)
        if not chunk:
            self.eof = True
            return False
        self.buffer += chunk
        return True

    def _compact(self) -> None:
        if self.consumed > _READ_SIZE * 4:
            self.offset += self.consumed
            self.buffer = self.buffer[self.consumed :]
            self.consumed = 0

    def _peek(self, index: int) -> str:
        while self.consumed + index >= len(self.buffer):
            if not self._fill():
                return ""
        return self.buffer[self.consumed + index]

    def _advance(self, count: int = 1) -> str:
        text = self.buffer[self.consumed : self.consumed + count]
        self.consumed += count
        self.line += text.count("\n")
        self._compact()
        return text

    def _skip_ws(self) -> None:
        while True:
            char = self._peek(0)
            if not char:
                return
            if not char.isspace():
                return
            self._advance(1)

    def _ensure(self, minimum: int) -> None:
        while len(self.buffer) - self.consumed < minimum and self._fill():
            pass

    def next_record(self):
        if self.state == "start":
            self._skip_ws()
            char = self._peek(0)
            if char != "[":
                raise BadRecordError(
                    f"Expected a JSON array at the root of '{self.path}'",
                    source=self.path,
                    line_number=1,
                    reason="root is not an array",
                )
            self._advance(1)
            self.state = "between"

        self._skip_ws()
        char = self._peek(0)
        if not char:
            if self.state == "between":
                # Allow a trailing comma-free close; missing ] is tolerated like
                # lenient JSON readers, but an open element is an error.
                raise BadRecordError(
                    f"Unexpected end of input in '{self.path}'",
                    source=self.path,
                    line_number=self.line,
                    reason="truncated JSON array",
                )
            return None

        if char == ",":
            if self.state != "value":
                raise BadRecordError(
                    f"Unexpected comma in '{self.path}'",
                    source=self.path,
                    line_number=self.line,
                    reason="misplaced comma",
                )
            self._advance(1)
            self._skip_ws()
            char = self._peek(0)
            if char == "]":
                raise BadRecordError(
                    f"Trailing comma in '{self.path}'",
                    source=self.path,
                    line_number=self.line,
                    reason="trailing comma",
                )
        elif char == "]":
            self._advance(1)
            self.state = "done"
            return None

        start_line = self.line
        start_offset = self.offset + self.consumed
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.consumed)
                break
            except json.JSONDecodeError as exc:
                # Grow the window unless the decoder is stuck at the very start
                # of a value and there is more input to read.
                if self._fill():
                    continue
                reason = f"{exc.msg} at line {start_line}, column {exc.colno}"
                raise _ArrayElementError(
                    reason=reason,
                    line_number=start_line,
                    start_offset=start_offset,
                    raw_text=self.buffer[self.consumed : self.consumed + 160],
                ) from exc

        raw = self.buffer[self.consumed : end]
        self._advance(end - self.consumed)
        self.state = "value"
        return value, start_line, len(raw.encode("utf-8")), raw

    def resync_after_error(self) -> bool:
        """Skip to the next top-level comma or closing bracket.

        Returns ``False`` when the remaining document cannot be resynchronised
        (i.e. we hit EOF), in which case no further records should be read.
        """
        depth = 0
        in_string = False
        escape = False
        while True:
            char = self._peek(0)
            if not char:
                return False
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                self._advance(1)
                continue
            if char == '"':
                in_string = True
            elif char in "[{":
                depth += 1
            elif char in "]}":
                if depth == 0:
                    self._advance(1)
                    self.state = "value"
                    return True
                depth -= 1
            elif char == "," and depth == 0:
                self._advance(1)
                self.state = "value"
                return True
            self._advance(1)

    # -- nested-array support for `from <path>` ------------------------------

    def _read_key(self) -> str:
        self._skip_ws()
        while True:
            try:
                key, end = self.decoder.raw_decode(self.buffer, self.consumed)
                self._advance(end - self.consumed)
                return key
            except json.JSONDecodeError as exc:
                if self._fill():
                    continue
                raise BadRecordError(
                    f"Could not read an object key in '{self.path}'",
                    source=self.path,
                    line_number=self.line,
                    reason=exc.msg,
                ) from exc

    def _skip_value(self) -> None:
        """Consume one JSON value without buffering it entirely."""
        while True:
            char = self._peek(0)
            if not char:
                raise BadRecordError(
                    f"Unexpected end of input in '{self.path}'",
                    source=self.path,
                    line_number=self.line,
                    reason="truncated JSON value",
                )
            if char not in "{[\"" and (char.isdigit() or char == "-"):
                # Primitive: consume until a structural delimiter.
                while True:
                    char = self._peek(0)
                    if not char or char in ",}] \t\r\n":
                        return
                    self._advance(1)
            break

        char = self._peek(0)
        if char == '"':
            self._advance(1)
            escape = False
            while True:
                char = self._peek(0)
                if not char:
                    raise BadRecordError(
                        f"Unterminated string in '{self.path}'",
                        source=self.path,
                        line_number=self.line,
                        reason="unterminated string",
                    )
                self._advance(1)
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    return
        elif char in "{[":
            depth = 0
            in_string = False
            escape = False
            while True:
                char = self._peek(0)
                if not char:
                    raise BadRecordError(
                        f"Unexpected end of input in '{self.path}'",
                        source=self.path,
                        line_number=self.line,
                        reason="truncated JSON value",
                    )
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                elif char == '"':
                    in_string = True
                elif char in "{[":
                    depth += 1
                elif char in "}]":
                    depth -= 1
                    if depth == 0:
                        self._advance(1)
                        return
                self._advance(1)
        else:
            while True:
                char = self._peek(0)
                if not char or char in ",}] \t\r\n":
                    return
                self._advance(1)

    def enter_object_path(self, segments: list[str]) -> None:
        """Descend a root object to ``segments`` and enter its array.

        The scan is incremental: sibling values are skipped character by
        character without being fully buffered, so a huge unrelated field does
        not inflate memory.
        """
        self._skip_ws()
        char = self._peek(0)
        if char != "{":
            raise BadRecordError(
                f"'from' requires an object at the root of '{self.path}'",
                source=self.path,
                line_number=self.line,
                reason="root is not an object",
            )
        self._advance(1)

        depth = 0
        while depth < len(segments):
            segment = segments[depth]
            last = depth == len(segments) - 1

            # Search the keys of the current object level for ``segment``.
            while True:
                self._skip_ws()
                char = self._peek(0)
                if not char or char == "}":
                    self._raise_missing_path(segments)
                key = self._read_key()
                self._skip_ws()
                if self._peek(0) != ":":
                    raise BadRecordError(
                        f"Expected ':' after key '{key}' in '{self.path}'",
                        source=self.path,
                        line_number=self.line,
                        reason="malformed object",
                    )
                self._advance(1)
                self._skip_ws()

                if key == segment:
                    break

                self._skip_value()
                self._skip_ws()
                sep = self._peek(0)
                if sep == ",":
                    self._advance(1)
                    continue
                if not sep or sep == "}":
                    self._raise_missing_path(segments)
                raise BadRecordError(
                    f"Malformed object in '{self.path}'",
                    source=self.path,
                    line_number=self.line,
                    reason="expected ',' or '}'",
                )

            char = self._peek(0)
            if last:
                if char != "[":
                    raise BadRecordError(
                        f"'from {'.'.join(segments)}' does not point at an array in '{self.path}'",
                        source=self.path,
                        line_number=self.line,
                        reason="from path is not an array",
                    )
                self._advance(1)
                self.state = "between"
                return
            if char != "{":
                self._raise_missing_path(segments)
            self._advance(1)
            depth += 1

    def _raise_missing_path(self, segments) -> None:
        raise BadRecordError(
            f"'from {'.'.join(segments)}' was not found in '{self.path}'",
            source=self.path,
            line_number=self.line,
            reason="from path is missing",
        )


class _ArrayElementError(Exception):
    def __init__(self, *, reason, line_number, start_offset, raw_text):
        super().__init__(reason)
        self.reason = reason
        self.line_number = line_number
        self.start_offset = start_offset
        self.raw_text = raw_text


class _ChunkedReader:
    """Base class: pull envelopes, turning array errors into bad records."""

    def __init__(self, path: str, config: StreamingConfig):
        self.path = path
        self.config = config
        self.bad_records: list[BadRecord] = []

    def _envelope(self, value, *, line_number, record_index, encoded_size, raw_text=None):
        size = encoded_size or _size_of(value)
        if size > self.config.memory_limit:
            raise MemoryLimitError(
                f"A single record in '{self.path}' needs {size} bytes, "
                f"above memory_limit={self.config.memory_limit}",
                context={"source": self.path, "record_bytes": size},
            )
        return RecordEnvelope(
            value=value,
            source=self.path,
            line_number=line_number,
            record_index=record_index,
            encoded_size=size,
            raw_text=raw_text,
        )


class ArrayReader(_ChunkedReader):
    def __init__(self, path: str, config: StreamingConfig, *, from_segments: list[str] | None = None):
        super().__init__(path, config)
        self.from_segments = list(from_segments or [])

    def iter_batches(self) -> Iterator[RecordBatch]:
        parser = _StreamingArrayParser(self.path)
        if self.from_segments:
            parser.enter_object_path(self.from_segments)
        batch = RecordBatch()
        index = 0
        try:
            while True:
                try:
                    item = parser.next_record()
                except _ArrayElementError as exc:
                    bad = BadRecord(
                        source=self.path,
                        reason=exc.reason,
                        line_number=exc.line_number,
                        record_index=index,
                        raw_text=exc.raw_text,
                    )
                    batch.bad.append(bad)
                    if not self.config.skips_bad_records:
                        raise bad.to_error() from exc
                    if not parser.resync_after_error():
                        break
                    index += 1
                    continue
                if item is None:
                    break
                value, line_number, encoded_size, raw = item
                try:
                    envelope = self._envelope(
                        value,
                        line_number=line_number,
                        record_index=index,
                        encoded_size=encoded_size,
                        raw_text=raw,
                    )
                except MemoryLimitError:
                    raise
                batch.records.append(envelope)
                batch.bytes_read += encoded_size
                batch.record_bytes += encoded_size
                index += 1
                if len(batch.records) + len(batch.bad) >= self.config.chunk_size:
                    batch.chunk_number = index // self.config.chunk_size
                    yield batch
                    batch = RecordBatch()
            if batch.records or batch.bad:
                yield batch
        finally:
            parser.close()


class JsonlReader(_ChunkedReader):
    def iter_batches(self) -> Iterator[RecordBatch]:
        batch = RecordBatch()
        index = 0
        with open(self.path, "r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                size = len(line.encode("utf-8"))
                batch.bytes_read += size
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    bad = BadRecord(
                        source=self.path,
                        reason=f"{exc.msg} at line {line_number}, column {exc.colno}",
                        line_number=line_number,
                        record_index=index,
                        raw_text=line,
                    )
                    batch.bad.append(bad)
                    if not self.config.skips_bad_records:
                        raise bad.to_error() from exc
                    index += 1
                    if len(batch.records) + len(batch.bad) >= self.config.chunk_size:
                        yield batch
                        batch = RecordBatch()
                    continue
                batch.records.append(
                    self._envelope(
                        value,
                        line_number=line_number,
                        record_index=index,
                        encoded_size=size,
                        raw_text=line,
                    )
                )
                batch.record_bytes += size
                index += 1
                if len(batch.records) + len(batch.bad) >= self.config.chunk_size:
                    yield batch
                    batch = RecordBatch()
        if batch.records or batch.bad:
            yield batch


class SingleValueReader(_ChunkedReader):
    """Reads a document whose root is a single object/scalar as one record."""

    def iter_batches(self) -> Iterator[RecordBatch]:
        with open(self.path, "r", encoding="utf-8") as handle:
            text = handle.read()
        batch = RecordBatch()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            batch.bad.append(
                BadRecord(
                    source=self.path,
                    reason=f"{exc.msg} at line {exc.lineno}, column {exc.colno}",
                    line_number=exc.lineno,
                    raw_text=text[:160],
                )
            )
            if not self.config.skips_bad_records:
                raise batch.bad[0].to_error() from exc
            batch.bytes_read = len(text.encode("utf-8"))
            yield batch
            return
        size = len(text.encode("utf-8"))
        batch.records.append(
            self._envelope(value, line_number=1, record_index=0, encoded_size=size, raw_text=text[:160])
        )
        batch.bytes_read = size
        batch.record_bytes = size
        yield batch


@dataclass
class InputSource:
    """A resolved input plus the bookkeeping the engine reports."""

    path: str
    kind: str
    reader: _ChunkedReader
    duplicates_removed: int = 0


def open_single_source(path: str, config: StreamingConfig, *, from_segments: list[str] | None = None) -> InputSource:
    if from_segments:
        # A `from <path>` query descends into the document, so the root object
        # is walked incrementally rather than read as a single record.
        return InputSource(
            path=path,
            kind="array",
            reader=ArrayReader(path, config, from_segments=from_segments),
        )
    kind = config.input_kind if config.input_kind in ("array", "jsonl") else detect_input_kind(path)
    if kind == "jsonl":
        return InputSource(path=path, kind="jsonl", reader=JsonlReader(path, config))
    if kind == "single":
        return InputSource(path=path, kind="single", reader=SingleValueReader(path, config))
    return InputSource(
        path=path,
        kind="array",
        reader=ArrayReader(path, config, from_segments=from_segments),
    )


def open_input(path: str | None, config: StreamingConfig, *, from_segments: list[str] | None = None) -> list[InputSource]:
    """Resolve ``path`` into one or more readable sources.

    A manifest (explicit ``input_kind='manifest'``, a ``.manifest``-style
    suffix, or ``config.manifest_paths`` when ``path`` is ``None``) expands into
    one source per file with duplicates removed.
    """
    if path is None:
        if not config.manifest_paths:
            raise PreflightError("No input source given.")
        resolved, duplicates = resolve_manifest(config.manifest_paths)
        return _sources_for_paths(resolved, duplicates, config, from_segments)

    explicit_manifest = config.input_kind == "manifest"
    if explicit_manifest or _is_manifest_path(path):
        entries = _read_manifest_paths(path)
        resolved, duplicates = resolve_manifest(
            entries, manifest_dir=os.path.dirname(os.path.abspath(path))
        )
        return _sources_for_paths(resolved, duplicates, config, from_segments)

    if config.manifest_paths:
        resolved, duplicates = resolve_manifest(config.manifest_paths)
        return _sources_for_paths(resolved, duplicates, config, from_segments)

    return [open_single_source(path, config, from_segments=from_segments)]


def _sources_for_paths(resolved, duplicates, config, from_segments):
    sources = []
    for index, entry in enumerate(resolved):
        if not os.path.isfile(entry):
            raise PreflightError(f"Manifest entry does not exist: '{entry}'")
        source = open_single_source(entry, config, from_segments=from_segments)
        if index == 0:
            source.duplicates_removed = duplicates
        sources.append(source)
    return sources


def iter_batches(sources: list[InputSource]) -> Iterator[RecordBatch]:
    for source in sources:
        yield from source.reader.iter_batches()
