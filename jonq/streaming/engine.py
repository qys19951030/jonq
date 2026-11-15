"""The bounded-memory execution engine.

``execute_streaming`` is the single entry point used by the CLI flag and by the
public API.  Phase order is deliberate:

1. **preflight** -- configuration, input existence/emptiness, output
   writability and manifest duplicate accounting all fail before any record is
   processed;
2. **compile** -- position-aware query parse;
3. **sample** -- a bounded schema sample drives semantic validation, so a
   missing field or an obvious type mismatch fails before the full scan;
4. **scan** -- chunked read, bad-record policy, condition filtering, projection;
5. **reduce** -- chunk-local aggregation merged across chunks, or external
   stable merge sort, or distinct, or a limit;
6. **write** -- one row stream rendered by the chosen format writer;
7. **report** -- statistics that reconcile with the scan/reduce counters.

Every temporary file lives under one :class:`TempWorkspace` and is removed even
on error or interruption.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from typing import Any

from jonq.streaming.config import StreamingConfig
from jonq.streaming.errors import (
    BadRecordError,
    OutputError,
    PreflightError,
    StructuredQueryError,
)
from jonq.streaming.outputs import OutputSink, write_rows
from jonq.streaming.plan import StreamingPlan, compile_plan, plan_field_names
from jonq.streaming.readers import InputSource, iter_batches, open_input
from jonq.streaming.semantics import build_schema_sample, validate_plan_against_sample
from jonq.streaming.spill import TempWorkspace
from jonq.streaming.stats import QueryStats
from jonq.streaming.inputs import build_scanner

SCHEMA_SAMPLE_RECORDS = 200
_TERMINAL_FORMATS = ("json", "jsonl", "csv", "table")


@dataclass
class StreamingResult:
    text: str
    output_format: str
    stats: QueryStats
    plan: StreamingPlan
    data: Any = None
    skipped_records: list[dict] = None

    def stats_text(self) -> str:
        return self.stats.format_text()

    def stats_json(self) -> str:
        return json.dumps(self.stats.to_dict(), indent=2, sort_keys=True)

    def format_text(self) -> str:
        return "\n".join(
            [
                self.text,
                "",
                self.stats_text(),
                "",
                "Statistics (structured)",
                self.stats_json(),
            ]
        )


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def _check_input(path: str | None, config: StreamingConfig) -> None:
    if path is None:
        if not config.manifest_paths:
            raise PreflightError("No input source given.")
        return
    if path == "-":
        raise PreflightError(
            "Streaming mode requires a file path; pipe input to a file first."
        )
    if not os.path.exists(path):
        raise PreflightError(f"Input path does not exist: '{path}'")
    if not os.path.isfile(path):
        raise PreflightError(f"Input path is not a file: '{path}'")
    if not os.access(path, os.R_OK):
        raise PreflightError(f"Input path is not readable: '{path}'")
    if os.path.getsize(path) == 0:
        raise PreflightError(f"Input file is empty: '{path}'")
    if os.path.splitext(path)[1].lower() not in (
        ".json", ".jsonl", ".ndjson", ".manifest", ".paths", ".list", ""
    ):
        # Unknown extensions are fine; only emptiness/existence is fatal.
        pass


def _check_output(path: str | None) -> None:
    if not path:
        return
    directory = os.path.dirname(os.path.abspath(path)) or "."
    if os.path.isdir(path):
        raise OutputError(f"Output path is a directory: '{path}'")
    if not os.path.isdir(directory):
        raise OutputError(f"Output directory does not exist: '{directory}'")
    if not os.access(directory, os.W_OK):
        raise OutputError(f"Output directory is not writable: '{directory}'")
    if os.path.exists(path) and not os.access(path, os.W_OK):
        raise OutputError(f"Output file is not writable: '{path}'")


def _check_output_format(output_format: str) -> None:
    if output_format not in _TERMINAL_FORMATS:
        allowed = ", ".join(_TERMINAL_FORMATS)
        raise PreflightError(
            f"Unsupported output format '{output_format}'. Expected one of: {allowed}."
        )


# ---------------------------------------------------------------------------
# Sampling & semantic validation
# ---------------------------------------------------------------------------

def _sample_sources(sources: list[InputSource], config: StreamingConfig) -> tuple[Any, int, int]:
    """Read up to ``SCHEMA_SAMPLE_RECORDS`` records for validation.

    Consumes the readers, so callers must re-open sources afterwards.  Returns
    ``(sample, bad_records_seen, largest_record_bytes)``.
    """
    collected = []
    bad = 0
    largest = 0
    for batch in iter_batches(sources):
        bad += len(batch.bad)
        for envelope in batch.records:
            collected.append(envelope.value)
            largest = max(largest, envelope.encoded_size)
            if len(collected) >= SCHEMA_SAMPLE_RECORDS:
                break
        if len(collected) >= SCHEMA_SAMPLE_RECORDS:
            break
    return build_schema_sample(collected), bad, largest


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def execute_streaming(
    path: str | None,
    query: str,
    *,
    output_format: str = "json",
    config: StreamingConfig | None = None,
    output_path: str | None = None,
    validate: bool = True,
    stats_format: str | None = None,
    indent: bool = False,
    row_limit: int | None = None,
    **options,
) -> StreamingResult:
    """Execute ``query`` over ``path`` with bounded memory.

    ``stats_format`` renders the statistics report alongside the data:
    ``"text"`` or ``"json"`` (``None`` leaves it out of ``text``; the
    :class:`StreamingResult` always exposes ``stats``).  ``row_limit`` applies
    the CLI's post-query ``--limit`` on top of any query-level ``limit``.
    """
    config = config or StreamingConfig.from_options(options)
    if options:
        config = config.with_overrides(
            **{k: v for k, v in options.items() if v is not None}
        )
    started = time.monotonic()
    phases: dict[str, float] = {}
    stats = QueryStats()

    # 1. preflight ---------------------------------------------------------
    mark = time.monotonic()
    _check_output_format(output_format)
    _check_output(output_path)
    from_segments = None

    # 2. compile -----------------------------------------------------------
    # Parsing the query before probing the input means a syntax error is
    # reported even when the input path is also wrong; both fail before any
    # record is read.
    plan = compile_plan(query)
    if row_limit is not None and row_limit >= 0:
        limit_values = [int(plan.limit)] if plan.limit else []
        limit_values.append(int(row_limit))
        plan = replace(plan, limit=str(min(limit_values)))
    if plan.from_path:
        from_segments = [part for part in plan.from_path.strip(".").split(".") if part]
    _check_input(path, config)
    phases["compile"] = time.monotonic() - mark

    # Resolve sources once for preflight (manifest duplicates, existence).
    mark = time.monotonic()
    sources = open_input(path, config, from_segments=from_segments)
    stats.files_read = len(sources)
    stats.duplicate_paths_removed = sum(source.duplicates_removed for source in sources)
    if not sources:
        raise PreflightError("No input files to read.")
    phases["resolve"] = time.monotonic() - mark

    workspace = TempWorkspace(base_dir=config.temp_dir, prefix=config.chunk_dir_prefix, keep=config.keep_temp)
    try:
        # 3. sample & semantic validation ----------------------------------
        mark = time.monotonic()
        largest_record = 0
        try:
            if validate:
                sample_sources = _reopen(sources, config, from_segments)
                sample, _, sample_large = _sample_sources(sample_sources, config)
                largest_record = max(largest_record, sample_large)
                validate_plan_against_sample(plan, sample)
            else:
                _, _, largest_record = _sample_sources(
                    _reopen(sources, config, from_segments), config
                )
        except BadRecordError as exc:
            if exc.reason in ("from path is missing", "from path is not an array"):
                raise StructuredQueryError(
                    f"'from' path '{plan.from_path}' cannot be read from the input",
                    query=plan.query,
                    position=plan.from_position,
                    kind="semantic",
                    context={"from": plan.from_path, "reason": exc.reason},
                ) from exc
            raise
        config.validate_record_size(largest_record, record_label="Largest sampled record")
        # A chunk must fit the memory ceiling too; if the configured chunk size
        # would overrun it, shrink the chunk and report the effective size.
        if largest_record and not config.chunk_fits_memory(largest_record):
            effective = max(1, config.memory_limit // largest_record)
            config = config.with_overrides(chunk_size=effective)
        phases["validate"] = time.monotonic() - mark

        # 4-6. scan / reduce / write ---------------------------------------
        mark = time.monotonic()
        headers = _static_headers(plan)
        single_object = _single_object_output(plan)
        sink = OutputSink(output_path)
        scanner = build_scanner(path, config, plan, from_segments=from_segments)
        try:
            rows = _run_pipeline(scanner, plan, config, workspace, stats)
            output_rows = write_rows(
                rows,
                sink,
                output_format=output_format,
                headers=headers,
                workspace=workspace,
                single_object=single_object,
                indent=indent,
            )
            text = sink.close() or ""
        finally:
            if not sink.closed:
                sink.close()
        scanner.apply_to(stats)
        stats.output_rows = output_rows
        phases["execute"] = time.monotonic() - mark

        stats.elapsed_seconds = time.monotonic() - started
        stats.phase_seconds = phases
        stats.peak_memory_bytes = _estimate_peak_memory(config, stats)
        stats.chunk_size = config.chunk_size
        stats.memory_limit = config.memory_limit
        stats.float_tolerance = config.float_tolerance
        stats.bad_record_policy = config.bad_record_policy
        stats.merge_fan_in = config.merge_fan_in

        result_text = text
        if stats_format == "text":
            result_text = text + ("\n\n" if text else "") + stats.format_text()
        elif stats_format == "json":
            result_text = (
                text
                + ("\n\n" if text else "")
                + json.dumps(stats.to_dict(), indent=2, sort_keys=True)
            )

        data = None
        if output_format == "json" and text.strip():
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = None

        return StreamingResult(
            text=result_text,
            output_format=output_format,
            stats=stats,
            plan=plan,
            data=data,
            skipped_records=list(scanner.skipped_records),
        )
    finally:
        workspace.cleanup()


def _reopen(sources, config, from_segments):
    # Sources are already resolved individual files; drop the manifest hint or
    # each file would be re-interpreted as a manifest during the sample pass.
    concrete = config.resolved_inputs()
    return [
        open_input(source.path, concrete, from_segments=from_segments)[0]
        for source in sources
    ]


def _static_headers(plan: StreamingPlan) -> list[str] | None:
    if any(field.kind == "field" and field.path == "*" for field in plan.fields):
        return None
    return list(plan_field_names(plan))


def _single_object_output(plan: StreamingPlan) -> bool:
    return plan.has_aggregates and not plan.group_by


# ---------------------------------------------------------------------------
# Pipeline stages (implemented incrementally)
# ---------------------------------------------------------------------------

def _run_pipeline(scanner, plan, config, workspace, stats):
    """Choose a reduce strategy for ``plan`` and return its row iterator."""
    from jonq.streaming import reduce as reducers

    source = iter(scanner)
    if plan.is_grouped:
        return reducers.reduce_grouped(source, plan, config, workspace, stats)
    if plan.has_aggregates:
        return reducers.reduce_global_aggregate(source, plan, config, stats)
    if plan.distinct:
        return reducers.reduce_distinct(source, plan, config, workspace, stats)
    if plan.needs_ordering:
        return reducers.reduce_ordered(source, plan, config, workspace, stats)
    return reducers.reduce_plain(source, plan, stats)


def stream_query(
    path: str | None,
    query: str,
    *,
    format: str = "json",
    config: StreamingConfig | None = None,
    **options,
) -> Any:
    """Convenience wrapper returning parsed data for JSON output.

    Mirrors :func:`jonq.api.query` for callers that want the rows, not the
    rendered text, while keeping bounded-memory execution.
    """
    result = execute_streaming(
        path, query, output_format=format, config=config, **options
    )
    if format == "json":
        return result.data
    return result.text


def _estimate_peak_memory(config: StreamingConfig, stats: QueryStats) -> int:
    """Estimate peak resident memory from the scan/reduce counters.

    The bound the engine maintains is ``chunk_size`` records in flight plus a
    spill buffer; each record's average encoded size comes from the scan.  The
    estimate is intentionally conservative and never exceeds ``memory_limit``
    because the engine refuses to start when one record would.
    """
    average = (
        max(1, stats.bytes_read // max(1, stats.records_read))
        if stats.records_read
        else 1
    )
    in_flight = max(1, average * config.chunk_size)
    # Projection, sort keys and output buffers roughly double the record bytes.
    return min(config.memory_limit, in_flight * 2)
