"""Statistics: two renderings agree and the numbers recompute from the process."""

from __future__ import annotations

import json

import pytest

from jonq.streaming.config import StreamingConfig
from jonq.streaming.engine import execute_streaming
from jonq.streaming.stats import QueryStats


@pytest.fixture
def sample_file(write_json):
    return write_json(
        "s.json",
        [{"id": i, "g": i % 3, "v": i * 1.5} for i in range(40)],
    )


class TestTwoRenderingsAgree:
    def test_text_and_json_report_same_numbers(self, sample_file):
        config = StreamingConfig(chunk_size=5)
        result = execute_streaming(
            sample_file,
            "select g, count(*) as n, sum(v) as total group by g",
            config=config,
        )
        text = result.stats.format_text()
        data = json.loads(json.dumps(result.stats.to_dict()))

        for key in (
            "records_read",
            "records_skipped",
            "records_filtered",
            "output_rows",
            "chunk_count",
            "spill_chunk_count",
            "spill_bytes",
            "peak_memory_bytes",
        ):
            assert str(data[key]) in text
        assert f"{data['elapsed_seconds']:.4f}s" in text

    def test_stats_render_in_output_when_requested(self, sample_file):
        result = execute_streaming(
            sample_file,
            "select id",
            config=StreamingConfig(chunk_size=4),
            stats_format="text",
        )
        assert "Execution statistics" in result.text
        data_result = execute_streaming(
            sample_file,
            "select id",
            config=StreamingConfig(chunk_size=4),
            stats_format="json",
        )
        stats_part = data_result.text.rsplit("\n\n", 1)[-1]
        parsed = json.loads(stats_part)
        assert parsed["output_rows"] == 40


class TestNumbersRecompute:
    def test_scan_accounting(self, sample_file):
        result = execute_streaming(
            sample_file, "select id if id >= 10", config=StreamingConfig(chunk_size=6)
        )
        stats = result.stats
        assert stats.records_selected + stats.records_filtered == stats.records_read
        assert stats.records_filtered == 10
        assert stats.records_selected == 30
        assert stats.output_rows == 30

    def test_chunk_count_matches_batches(self, sample_file):
        result = execute_streaming(
            sample_file, "select id", config=StreamingConfig(chunk_size=8)
        )
        # 40 records, chunk size 8 -> 5 chunks.
        assert result.stats.chunk_count == 5

    def test_spill_bytes_match_spill_files(self, write_json, tmp_path):
        path = write_json("s.json", [{"k": i % 7, "i": i} for i in range(35)])
        config = StreamingConfig(chunk_size=5, temp_dir=str(tmp_path), keep_temp=True)
        result = execute_streaming(path, "select k, i sort k", config=config)
        try:
            assert result.stats.spill_chunk_count >= 1
            spill_files = _spill_files(tmp_path)
            assert spill_files
            actual_bytes = sum(
                len(line.encode("utf-8")) + 1
                for spill in spill_files
                for line in _read_lines(spill)
            )
            assert result.stats.spill_bytes == actual_bytes
        finally:
            import shutil
            import os

            for name in os.listdir(tmp_path):
                candidate = os.path.join(tmp_path, name)
                if os.path.isdir(candidate):
                    shutil.rmtree(candidate, ignore_errors=True)

    def test_limit_accounting(self, sample_file):
        result = execute_streaming(
            sample_file, "select id limit 5", config=StreamingConfig(chunk_size=10)
        )
        assert result.stats.output_rows == 5
        assert result.stats.records_limited_out == 35

    def test_groups_formed_matches_output(self, sample_file):
        result = execute_streaming(
            sample_file,
            "select g, count(*) as n group by g",
            config=StreamingConfig(chunk_size=4),
        )
        assert result.stats.groups_formed == result.stats.output_rows == 3

    def test_verify_reports_no_problems(self, sample_file):
        result = execute_streaming(
            sample_file,
            "select g, count(*) as n group by g sort n desc",
            config=StreamingConfig(chunk_size=3),
        )
        assert result.stats.verify() == []

    def test_verify_detects_inconsistent_counters(self):
        stats = QueryStats(records_read=10, records_selected=5, records_filtered=3)
        assert any("records_read" in problem for problem in stats.verify())

    def test_stats_include_effective_configuration(self, sample_file):
        config = StreamingConfig(
            chunk_size=9, memory_limit="1MB", float_tolerance=1e-6,
            bad_record_policy="skip",
        )
        result = execute_streaming(sample_file, "select id", config=config)
        stats = result.stats
        assert stats.chunk_size == 9
        assert stats.memory_limit == 1024**2
        assert stats.float_tolerance == 1e-6
        assert stats.bad_record_policy == "skip"

    def test_phase_timings_present(self, sample_file):
        result = execute_streaming(sample_file, "select id")
        assert "compile" in result.stats.phase_seconds
        assert "execute" in result.stats.phase_seconds
        assert result.stats.elapsed_seconds >= 0


def _spill_files(directory):
    import os

    found = []
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if "spill" in name:
                found.append(os.path.join(root, name))
    return found


def _read_lines(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


class TestSkippedAccounting:
    def test_skip_counted_in_both_reports(self, write_text):
        path = write_text("s.jsonl", '{"a":1}\n{oops}\n{"a":2}\n')
        config = StreamingConfig(chunk_size=10, bad_record_policy="skip")
        result = execute_streaming(path, "select a", config=config, stats_format="text")
        assert "records skipped:       1" in result.text
        assert result.stats.records_skipped == 1
        assert result.stats.records_total == 3
