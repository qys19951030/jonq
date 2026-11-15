"""Errors raised by the bounded-memory execution engine.

Every error carries a ``phase`` so callers can distinguish preflight problems
(bad configuration, missing input, invalid query) from runtime data problems
(a malformed record).  Query errors carry a character offset plus a snippet of
the query around it; bad-record errors carry a source label and line number.
"""

from __future__ import annotations


class StreamingError(Exception):
    """Base class for all errors raised by the streaming engine."""

    phase = "runtime"
    exit_code = 1

    def __init__(self, message, *, phase=None, location=None, context=None):
        super().__init__(message)
        if phase is not None:
            self.phase = phase
        self.location = location
        self.context = dict(context or {})

    def format_error(self) -> str:
        parts = [f"Error: {self.args[0]}"]
        if self.location is not None:
            parts.append(f"  at {self.location}")
        for key, value in self.context.items():
            parts.append(f"  {key}: {value}")
        return "\n".join(parts)


class PreflightError(StreamingError):
    """A problem detected before any query work starts."""

    phase = "preflight"


class MemoryLimitError(PreflightError):
    """The configured memory limit cannot hold a single input record."""

    phase = "memory"


def _line_and_column(query: str, position: int) -> tuple[int, int]:
    prefix = query[: max(0, position)]
    line = prefix.count("\n") + 1
    column = len(prefix.rsplit("\n", 1)[-1]) + 1
    return line, column


def _snippet_around(query: str, position: int, *, width: int = 20) -> str:
    if not query:
        return ""
    start = max(0, position - width)
    end = min(len(query), position + width)
    before = query[start:position]
    after = query[position:end]
    snippet = before + "|" + after
    if start > 0:
        snippet = "…" + snippet
    if end < len(query):
        snippet = snippet + "…"
    return snippet.replace("\n", "\\n")


class StructuredQueryError(StreamingError):
    """A query is invalid; carries position info when available."""

    phase = "query"

    def __init__(
        self,
        message,
        *,
        query=None,
        position=None,
        snippet=None,
        kind="syntax",
        context=None,
    ):
        super().__init__(message, phase="query", context=context)
        self.query = query
        self.position = position
        self.kind = kind
        if position is not None and query is not None:
            self.line, self.column = _line_and_column(query, position)
            self.snippet = snippet if snippet is not None else _snippet_around(query, position)
            self.location = f"line {self.line}, column {self.column} (offset {position})"
        elif position is not None:
            self.line = None
            self.column = None
            self.snippet = snippet
            self.location = f"offset {position}"
        else:
            self.line = None
            self.column = None
            self.snippet = snippet
            self.location = None

    def format_error(self) -> str:
        parts = [f"Error: {self.args[0]}"]
        if self.location is not None:
            parts.append(f"  at {self.location}")
        if self.snippet:
            parts.append(f"  near: {self.snippet}")
        for key, value in self.context.items():
            parts.append(f"  {key}: {value}")
        return "\n".join(parts)


class SemanticQueryError(StructuredQueryError):
    """A query references a field that does not exist or has the wrong type."""

    def __init__(self, message, **kwargs):
        kwargs.setdefault("kind", "semantic")
        super().__init__(message, **kwargs)


class BadRecordError(StreamingError):
    """An input record could not be parsed or decoded."""

    phase = "input"

    def __init__(
        self,
        message,
        *,
        source=None,
        line_number=None,
        record_index=None,
        record_text=None,
        reason=None,
        context=None,
    ):
        ctx = dict(context or {})
        if reason:
            ctx.setdefault("reason", reason)
        super().__init__(message, phase="input", context=ctx)
        self.source = source
        self.line_number = line_number
        self.record_index = record_index
        self.record_text = record_text
        self.reason = reason
        if source is not None and line_number is not None:
            self.location = f"{source}:{line_number}"
        elif source is not None and record_index is not None:
            self.location = f"{source} (record {record_index})"

    def format_error(self) -> str:
        parts = [f"Error: {self.args[0]}"]
        if self.location is not None:
            parts.append(f"  at {self.location}")
        if self.record_text is not None:
            preview = self.record_text.strip()
            if len(preview) > 120:
                preview = preview[:117] + "…"
            parts.append(f"  record: {preview}")
        if self.reason:
            parts.append(f"  reason: {self.reason}")
        for key, value in self.context.items():
            if key == "reason":
                continue
            parts.append(f"  {key}: {value}")
        return "\n".join(parts)


class OutputError(StreamingError):
    """The requested output destination cannot be written."""

    phase = "preflight"


class ManifestError(PreflightError):
    """An input manifest is missing, empty, or references unusable paths."""

    phase = "preflight"
