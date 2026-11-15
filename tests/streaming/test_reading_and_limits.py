"""Chunked input, memory bounds, and preflight validation."""

from __future__ import annotations

import json
import os

import pytest

from jonq.streaming.config import ConfigError, StreamingConfig, parse_size
from jonq.streaming.engine import execute_streaming
from jonq.streaming.errors import (
    ManifestError,
    MemoryLimitError,
    OutputError,
    PreflightError,
)
from jonq.streaming.readers import iter_batches, open_input


class TestConfig:
    def test_parse_size_units(self):
        assert parse_size(512) == 512
        assert parse_size("1KB") == 1024
        assert parse_size("64MB") == 64 * 1024**2
        assert parse_size("1GiB") == 1024**3

    def test_parse_size_rejects_garbage(self):
        with pytest.raises(ConfigError):
            parse_size("not-a-size")
        with pytest.raises(ConfigError):
            parse_size(0)
        with pytest.raises(ConfigError):
            parse_size(-5)

    def test_config_rejects_bad_chunk_size(self):
        with pytest.raises(ConfigError):
            StreamingConfig(chunk_size=0)
        with pytest.raises(ConfigError):
            StreamingConfig(chunk_size="10")

    def test_config_rejects_tiny_memory_limit(self):
        with pytest.raises(ConfigError):
            StreamingConfig(memory_limit=3)

    def test_config_rejects_unknown_policy(self):
        with pytest.raises(ConfigError):
            StreamingConfig(bad_record_policy="explode")

    def test_config_is_not_hardcoded(self):
        config = StreamingConfig(chunk_size=7, memory_limit="2MB")
        assert config.chunk_size == 7
        assert config.memory_limit == 2 * 1024**2


class TestChunkedReading:
    def test_array_is_split_into_chunks(self, write_json):
        path = write_json("n.json", [{"i": i} for i in range(9)])
        config = StreamingConfig(chunk_size=4)
        batches = list(iter_batches(open_input(path, config)))
        assert [len(batch.records) for batch in batches] == [4, 4, 1]
        seen = [envelope.value["i"] for batch in batches for envelope in batch.records]
        assert seen == list(range(9))

    def test_chunk_size_controls_batch_boundaries(self, write_json):
        path = write_json("n.json", [{"i": i} for i in range(10)])
        for chunk_size in (1, 3, 10, 50):
            batches = list(
                iter_batches(open_input(path, StreamingConfig(chunk_size=chunk_size)))
            )
            assert all(len(batch.records) <= chunk_size for batch in batches)
            total = sum(len(batch.records) for batch in batches)
            assert total == 10

    def test_jsonl_line_numbers_are_preserved(self, write_text):
        path = write_text("n.jsonl", '{"i":0}\n\n{"i":1}\n{"i":2}\n')
        batches = list(
            iter_batches(open_input(path, StreamingConfig(chunk_size=10)))
        )
        lines = [envelope.line_number for batch in batches for envelope in batch.records]
        assert lines == [1, 3, 4]

    def test_bytes_read_matches_input_size(self, write_json):
        data = [{"i": i, "text": "x" * 10} for i in range(5)]
        path = write_json("n.json", data)
        config = StreamingConfig(chunk_size=2)
        batches = list(iter_batches(open_input(path, config)))
        # Each record's encoded size is counted; the sum must be positive and
        # bounded by the file size plus separators.
        total = sum(batch.bytes_read for batch in batches)
        assert 0 < total <= os.path.getsize(path) + 16 * len(data)


class TestMemoryLimit:
    def test_single_record_above_limit_fails_before_execution(self, write_json):
        path = write_json("big.json", [{"payload": "x" * 5000}])
        with pytest.raises(MemoryLimitError) as excinfo:
            execute_streaming(
                path, "select payload", config=StreamingConfig(memory_limit=1024)
            )
        assert "memory_limit" in str(excinfo.value)

    def test_limit_large_enough_succeeds(self, write_json):
        path = write_json("ok.json", [{"payload": "x" * 100} for _ in range(5)])
        result = execute_streaming(
            path, "select payload", config=StreamingConfig(memory_limit=64 * 1024)
        )
        assert result.stats.output_rows == 5

    def test_peak_memory_never_exceeds_limit(self, write_json):
        path = write_json("n.json", [{"i": i, "pad": "y" * 40} for i in range(50)])
        config = StreamingConfig(chunk_size=5, memory_limit=16 * 1024)
        result = execute_streaming(path, "select i sort i", config=config)
        assert result.stats.peak_memory_bytes <= config.memory_limit

    def test_peak_memory_grows_with_chunk_size(self, write_json):
        path = write_json("n.json", [{"i": i, "pad": "y" * 40} for i in range(60)])
        small = execute_streaming(
            path, "select i", config=StreamingConfig(chunk_size=2)
        )
        large = execute_streaming(
            path, "select i", config=StreamingConfig(chunk_size=40)
        )
        assert large.stats.peak_memory_bytes >= small.stats.peak_memory_bytes

    def test_chunk_shrinks_to_fit_memory_limit(self, write_json):
        # chunk_size * record size would exceed the ceiling, so the engine must
        # reduce the effective chunk size instead of overrunning it.
        path = write_json("n.json", [{"i": i, "pad": "x" * 100} for i in range(300)])
        config = StreamingConfig(chunk_size=200, memory_limit=4096)
        result = execute_streaming(path, "select i sort i", config=config)
        assert result.stats.output_rows == 300
        assert result.stats.peak_memory_bytes <= config.memory_limit
        # More chunks than the configured 2 are needed once the chunk shrinks.
        assert result.stats.chunk_count > 2

    def _write_stream(self, path, count):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("[")
            for index in range(count):
                if index:
                    handle.write(",")
                handle.write(
                    json.dumps({"id": index, "g": index % 10, "v": "x" * 40})
                )
            handle.write("]")

    def test_real_peak_memory_is_bounded_by_input_size(self, tmp_path):
        import tracemalloc

        small = tmp_path / "small.json"
        large = tmp_path / "large.json"
        self._write_stream(small, 4000)
        self._write_stream(large, 16000)
        config = StreamingConfig(chunk_size=200)

        def peak_for(path):
            tracemalloc.start()
            execute_streaming(
                str(path),
                "select g, count(*) as n, sum(id) as s group by g sort n desc",
                config=config,
            )
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return peak

        small_peak = peak_for(small)
        large_peak = peak_for(large)
        # 4x the records must not mean 4x the peak memory; chunked execution
        # keeps the growth far below linear.
        assert large_peak < small_peak * 2


class TestPreflight:
    def test_missing_input_path(self, tmp_path):
        with pytest.raises(PreflightError, match="does not exist"):
            execute_streaming(str(tmp_path / "nope.json"), "select id")

    def test_input_is_a_directory(self, tmp_path):
        with pytest.raises(PreflightError, match="not a file"):
            execute_streaming(str(tmp_path), "select id")

    def test_empty_input_file(self, write_text):
        path = write_text("empty.json", "")
        with pytest.raises(PreflightError, match="empty"):
            execute_streaming(path, "select id")

    def test_output_directory_missing(self, write_json, tmp_path):
        path = write_json("n.json", [{"i": 1}])
        with pytest.raises(OutputError, match="Output directory"):
            execute_streaming(
                path, "select i", output_path=str(tmp_path / "missing" / "out.json")
            )

    def test_output_format_rejected(self, write_json):
        path = write_json("n.json", [{"i": 1}])
        with pytest.raises(PreflightError, match="Unsupported output format"):
            execute_streaming(path, "select i", output_format="yaml")

    def test_manifest_missing_entry(self, write_json, tmp_path):
        first = write_json("a.json", [{"i": 1}])
        manifest = tmp_path / "m.manifest"
        manifest.write_text(json.dumps([first, str(tmp_path / "gone.json")]))
        with pytest.raises(PreflightError, match="does not exist"):
            execute_streaming(str(manifest), "select i")

    def test_manifest_empty(self, write_text):
        path = write_text("m.manifest", "[]")
        with pytest.raises(ManifestError, match="lists no files"):
            execute_streaming(path, "select i")

    def test_manifest_duplicates_detected_and_counted(self, write_json, tmp_path):
        first = write_json("a.json", [{"i": 1}])
        second = write_json("b.json", [{"i": 2}])
        manifest = tmp_path / "m.manifest"
        manifest.write_text(json.dumps([first, second, first, second, first]))
        result = execute_streaming(str(manifest), "select i")
        assert result.stats.duplicate_paths_removed == 3
        assert result.stats.files_read == 2
        assert result.stats.output_rows == 2

    def test_manifest_relative_paths_resolve_against_manifest(self, tmp_path):
        (tmp_path / "a.json").write_text(json.dumps([{"i": 1}]))
        manifest = tmp_path / "m.manifest"
        manifest.write_text(json.dumps(["a.json"]))
        result = execute_streaming(str(manifest), "select i")
        assert result.stats.files_read == 1
        assert json.loads(result.text) == [{"i": 1}]
