"""Execution statistics for the streaming engine.

Two renderings share one dataclass so the human-readable report and the
structured (JSON) report cannot drift apart.  Every counter is either produced
directly by the reader/writer or derived from recorded chunk sizes, and
:meth:`QueryStats.verify` re-checks the accounting invariants.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class QueryStats:
    # --- input ---------------------------------------------------------
    files_read: int = 0
    duplicate_paths_removed: int = 0
    records_read: int = 0          # valid records decoded from input
    records_skipped: int = 0       # malformed records skipped by policy
    bytes_read: int = 0

    # --- filtering -----------------------------------------------------
    records_filtered: int = 0      # dropped by the if/where condition
    records_selected: int = 0      # survived the condition

    # --- grouping ------------------------------------------------------
    groups_formed: int = 0
    records_filtered_by_having: int = 0

    # --- ordering / dedupe / limit ------------------------------------
    records_deduplicated: int = 0
    records_limited_out: int = 0
    output_rows: int = 0

    # --- spilling / merging -------------------------------------------
    chunk_count: int = 0           # input chunks processed
    spill_chunk_count: int = 0     # temporary chunk files written
    merge_passes: int = 0
    spill_bytes: int = 0
    peak_memory_bytes: int = 0     # estimate, never a measured RSS

    # --- timing --------------------------------------------------------
    elapsed_seconds: float = 0.0
    phase_seconds: dict[str, float] = field(default_factory=dict)

    # --- configuration actually in force ------------------------------
    chunk_size: int = 0
    memory_limit: int = 0
    float_tolerance: float = 0.0
    bad_record_policy: str = "error"
    merge_fan_in: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def records_total(self) -> int:
        return self.records_read + self.records_skipped

    def verify(self) -> list[str]:
        """Return a list of accounting inconsistencies (empty means OK)."""
        problems: list[str] = []
        if self.records_selected + self.records_filtered != self.records_read:
            problems.append(
                "records_selected + records_filtered != records_read "
                f"({self.records_selected} + {self.records_filtered} != {self.records_read})"
            )
        if self.records_limited_out > self.output_rows:
            problems.append(
                "records_limited_out exceeds output_rows "
                f"({self.records_limited_out} > {self.output_rows})"
            )
        if self.records_deduplicated > self.records_selected:
            problems.append(
                "records_deduplicated exceeds records_selected "
                f"({self.records_deduplicated} > {self.records_selected})"
            )
        if self.spill_bytes < 0:
            problems.append("spill_bytes is negative")
        if self.spill_chunk_count > self.chunk_count and self.spill_chunk_count:
            problems.append(
                "spill_chunk_count exceeds chunk_count "
                f"({self.spill_chunk_count} > {self.chunk_count})"
            )
        if self.records_selected < self.groups_formed:
            problems.append(
                "groups_formed exceeds records_selected "
                f"({self.groups_formed} > {self.records_selected})"
            )
        return problems

    def _fmt_bytes(self, value: int) -> str:
        units = ("B", "KiB", "MiB", "GiB", "TiB")
        amount = float(value)
        for unit in units:
            if amount < 1024 or unit == units[-1]:
                if unit == "B":
                    return f"{int(amount)} B"
                return f"{amount:.2f} {unit}"
            amount /= 1024
        return f"{value} B"

    def format_text(self) -> str:
        lines = [
            "Execution statistics",
            f"  files read:            {self.files_read}",
            f"  duplicate paths:       {self.duplicate_paths_removed}",
            f"  records read:          {self.records_read}",
            f"  records skipped:       {self.records_skipped}",
            f"  records filtered:      {self.records_filtered}",
            f"  records selected:      {self.records_selected}",
            f"  groups formed:         {self.groups_formed}",
            f"  filtered by having:    {self.records_filtered_by_having}",
            f"  deduplicated:          {self.records_deduplicated}",
            f"  limited out:           {self.records_limited_out}",
            f"  output rows:           {self.output_rows}",
            f"  input chunks:          {self.chunk_count}",
            f"  spill chunks:          {self.spill_chunk_count}",
            f"  merge passes:          {self.merge_passes}",
            f"  bytes spilled to disk: {self._fmt_bytes(self.spill_bytes)}",
            f"  peak memory (estimate): {self._fmt_bytes(self.peak_memory_bytes)}",
            f"  elapsed:               {self.elapsed_seconds:.4f}s",
            f"  chunk size:            {self.chunk_size}",
            f"  memory limit:          {self._fmt_bytes(self.memory_limit)}",
            f"  float tolerance:       {self.float_tolerance:g}",
            f"  bad-record policy:     {self.bad_record_policy}",
        ]
        for name, seconds in sorted(self.phase_seconds.items()):
            lines.append(f"    {name}: {seconds:.4f}s")
        return "\n".join(lines)
