"""Configuration objects for the bounded-memory execution engine.

Chunk size and memory ceiling are never hard-coded at call sites: they come
from :class:`StreamingConfig`, which can be built from CLI options, a mapping,
or environment variables.  Sizes accept plain byte counts or human suffixes
(``"64MB"``, ``"1GiB"``, ``4096``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_MEMORY_LIMIT = 64 * 1024 * 1024
DEFAULT_FLOAT_TOLERANCE = 1e-9
DEFAULT_MERGE_FAN_IN = 64
DEFAULT_CHUNK_DIR_PREFIX = "jonq_stream_"
MIN_MEMORY_LIMIT = 1024

SIZE_ENV_CHUNK = "JONQ_STREAM_CHUNK_SIZE"
SIZE_ENV_MEMORY = "JONQ_STREAM_MEMORY_LIMIT"

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?)?\s*$", re.IGNORECASE)
_UNIT_FACTORS = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "kib": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "mib": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
    "gib": 1024**3,
    "t": 1024**4,
    "tb": 1024**4,
    "tib": 1024**4,
}


class ConfigError(ValueError):
    """Raised when streaming configuration is invalid."""


def parse_size(value, *, name: str = "size") -> int:
    """Parse ``512``, ``"64MB"`` or ``"1GiB"`` into a byte count."""
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a byte count, not a boolean")
    if isinstance(value, int):
        if value <= 0:
            raise ConfigError(f"{name} must be a positive byte count, got {value}")
        return value
    if isinstance(value, float):
        if value <= 0:
            raise ConfigError(f"{name} must be a positive byte count, got {value}")
        return int(value)
    if not isinstance(value, str):
        raise ConfigError(f"{name} must be an integer or a size string, got {type(value).__name__}")
    match = _SIZE_RE.match(value)
    if not match:
        raise ConfigError(
            f"{name} must look like '1024', '64MB', or '1GiB', got {value!r}"
        )
    number = float(match.group(1))
    unit = (match.group(2) or "").lower()
    result = int(number * _UNIT_FACTORS[unit])
    if result <= 0:
        raise ConfigError(f"{name} must be a positive byte count, got {value!r}")
    return result


def _coerce_chunk_size(value, *, name: str = "chunk_size") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class StreamingConfig:
    """Chunking, spilling and bad-record settings for the streaming engine."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    memory_limit: int = DEFAULT_MEMORY_LIMIT
    float_tolerance: float = DEFAULT_FLOAT_TOLERANCE
    merge_fan_in: int = DEFAULT_MERGE_FAN_IN
    bad_record_policy: str = "error"  # "error" | "skip"
    temp_dir: str | None = None
    keep_temp: bool = False
    input_kind: str | None = None  # None = auto, or "array"/"jsonl"/"manifest"
    manifest_paths: tuple[str, ...] = ()
    chunk_dir_prefix: str = DEFAULT_CHUNK_DIR_PREFIX

    def __post_init__(self):
        object.__setattr__(
            self, "chunk_size", _coerce_chunk_size(self.chunk_size)
        )
        object.__setattr__(self, "memory_limit", parse_size(self.memory_limit, name="memory_limit"))
        if self.memory_limit < MIN_MEMORY_LIMIT:
            raise ConfigError(
                f"memory_limit must be at least {MIN_MEMORY_LIMIT} bytes, "
                f"got {self.memory_limit}"
            )
        if self.float_tolerance < 0:
            raise ConfigError("float_tolerance must be non-negative")
        object.__setattr__(self, "merge_fan_in", _coerce_chunk_size(self.merge_fan_in, name="merge_fan_in"))
        if self.bad_record_policy not in ("error", "skip"):
            raise ConfigError("bad_record_policy must be 'error' or 'skip'")
        if self.input_kind not in (None, "array", "jsonl", "manifest"):
            raise ConfigError("input_kind must be one of array, jsonl, manifest or None")
        object.__setattr__(self, "manifest_paths", tuple(self.manifest_paths or ()))

    @classmethod
    def from_options(cls, options: dict | None = None, **overrides) -> "StreamingConfig":
        """Build a config from a mapping (e.g. CLI options) plus overrides.

        Unknown keys are ignored so the CLI can pass its whole option dict.
        """
        options = dict(options or {})
        merged = {}
        for key in (
            "chunk_size",
            "memory_limit",
            "float_tolerance",
            "merge_fan_in",
            "bad_record_policy",
            "temp_dir",
            "keep_temp",
            "input_kind",
            "manifest_paths",
            "chunk_dir_prefix",
        ):
            if key in options and options[key] is not None:
                merged[key] = options[key]
        merged.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**merged)

    @classmethod
    def from_env(cls, env: dict | None = None, **overrides) -> "StreamingConfig":
        env = os.environ if env is None else env
        base: dict = {}
        if env.get(SIZE_ENV_CHUNK):
            base["chunk_size"] = int(env[SIZE_ENV_CHUNK])
        if env.get(SIZE_ENV_MEMORY):
            base["memory_limit"] = parse_size(env[SIZE_ENV_MEMORY], name=SIZE_ENV_MEMORY)
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)

    def with_overrides(self, **overrides) -> "StreamingConfig":
        return replace(
            self,
            **{k: v for k, v in overrides.items() if v is not None},
        )

    def resolved_inputs(self) -> "StreamingConfig":
        """Config for reading individual files once a manifest is expanded."""
        from dataclasses import replace as _replace

        return _replace(self, input_kind=None, manifest_paths=())

    @property
    def skips_bad_records(self) -> bool:
        return self.bad_record_policy == "skip"

    def effective_chunk_bytes(self, average_record_bytes: int) -> int:
        """Bytes one chunk may occupy once average record size is known."""
        return max(1, average_record_bytes) * self.chunk_size

    def chunk_fits_memory(self, average_record_bytes: int) -> bool:
        return self.effective_chunk_bytes(average_record_bytes) <= self.memory_limit

    def validate_record_size(self, record_size: int, *, record_label: str = "record") -> None:
        """Fail before execution when a single record cannot fit in memory."""
        if record_size > self.memory_limit:
            from jonq.streaming.errors import MemoryLimitError

            raise MemoryLimitError(
                f"{record_label} needs {record_size} bytes but memory_limit is "
                f"{self.memory_limit} bytes",
                context={"memory_limit": self.memory_limit, "record_bytes": record_size},
            )

    def to_dict(self) -> dict:
        return {
            "chunk_size": self.chunk_size,
            "memory_limit": self.memory_limit,
            "float_tolerance": self.float_tolerance,
            "merge_fan_in": self.merge_fan_in,
            "bad_record_policy": self.bad_record_policy,
            "temp_dir": self.temp_dir,
            "keep_temp": self.keep_temp,
            "input_kind": self.input_kind,
            "manifest_paths": list(self.manifest_paths),
        }
