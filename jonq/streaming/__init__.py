"""Bounded-memory streaming execution engine for jonq.

The legacy execution path (:func:`jonq.api.execute`) reads a whole JSON
document into memory and hands it to jq.  This package adds an opt-in engine
that keeps memory bounded regardless of input size:

* chunked readers for a root JSON array, JSON-lines streams, and manifests of
  JSON files,
* external merge sort that spills chunk files and preserves input order for
  equal keys,
* chunk-local aggregation merged across partitions,
* one row model shared by the table, CSV and JSON writers,
* configurable bad-record policies, and
* execution statistics that can be recomputed from the chunk/merge process.

Nothing in this package changes the behaviour of the legacy path; callers opt
in through :func:`jonq.streaming.engine.execute_streaming`.
"""

from jonq.streaming.config import StreamingConfig, parse_size
from jonq.streaming.errors import (
    BadRecordError,
    MemoryLimitError,
    PreflightError,
    StreamingError,
    StructuredQueryError,
)
from jonq.streaming.stats import QueryStats

__all__ = [
    "StreamingConfig",
    "parse_size",
    "StreamingError",
    "PreflightError",
    "MemoryLimitError",
    "StructuredQueryError",
    "BadRecordError",
    "QueryStats",
    "execute_streaming",
    "stream_query",
    "StreamingResult",
]


def __getattr__(name):
    # ``engine`` imports the rest of the package, so importing it eagerly here
    # would make ``jonq.streaming.config`` depend on the engine.  Resolve the
    # public entry points lazily instead.
    if name in ("execute_streaming", "stream_query", "StreamingResult"):
        from jonq.streaming import engine

        return getattr(engine, name)
    raise AttributeError(f"module 'jonq.streaming' has no attribute {name!r}")
