"""Bad-record policies: skipping, counting, and error reporting."""

from __future__ import annotations

import json

import pytest

from jonq.streaming.config import StreamingConfig
from jonq.streaming.engine import execute_streaming
from jonq.streaming.errors import BadRecordError


@pytest.fixture
def broken_jsonl(write_text):
    return write_text(
        "data.jsonl",
        '{"id":1}\n'
        "{not json}\n"
        '{"id":2}\n'
        "\n"
        '{"id":3,"note":"ok"}\n'
        '{"id":4,}\n',
    )


class TestErrorPolicy:
    def test_default_policy_raises(self, broken_jsonl):
        with pytest.raises(BadRecordError) as excinfo:
            execute_streaming(broken_jsonl, "select id")
        error = excinfo.value
        assert error.line_number == 2
        assert error.reason
        assert error.record_text is not None

    def test_error_reports_source_and_line(self, broken_jsonl):
        try:
            execute_streaming(broken_jsonl, "select id")
        except BadRecordError as exc:
            assert exc.location.startswith(broken_jsonl)
            assert exc.line_number == 2
            assert "line 2" in exc.format_error()
        else:
            pytest.fail("expected BadRecordError")


class TestSkipPolicy:
    def test_skips_and_counts(self, broken_jsonl):
        config = StreamingConfig(chunk_size=2, bad_record_policy="skip")
        result = execute_streaming(broken_jsonl, "select id", config=config)
        assert result.stats.records_skipped == 2
        assert result.stats.records_read == 3
        assert json.loads(result.text) == [{"id": 1}, {"id": 2}, {"id": 3}]

    def test_skip_records_expose_line_and_reason(self, broken_jsonl):
        config = StreamingConfig(chunk_size=10, bad_record_policy="skip")
        result = execute_streaming(broken_jsonl, "select id", config=config)
        assert [record["line"] for record in result.skipped_records] == [2, 6]
        assert all(record["reason"] for record in result.skipped_records)

    def test_skip_count_consistent_across_formats(self, broken_jsonl):
        config = StreamingConfig(chunk_size=2, bad_record_policy="skip")
        counts = {}
        row_counts = {}
        for fmt in ("json", "csv", "table"):
            result = execute_streaming(
                broken_jsonl, "select id", output_format=fmt, config=config
            )
            counts[fmt] = result.stats.records_skipped
            row_counts[fmt] = result.stats.output_rows
        assert set(counts.values()) == {2}
        assert set(row_counts.values()) == {3}

    def test_skip_policy_image_is_part_of_stats(self, broken_jsonl):
        config = StreamingConfig(chunk_size=2, bad_record_policy="skip")
        result = execute_streaming(
            broken_jsonl, "select id", config=config, stats_format="json"
        )
        stats = json.loads(result.text.split("\n\n", 1)[1])
        assert stats["records_skipped"] == 2

    def test_array_element_errors_are_skippable(self, write_text):
        path = write_text("arr.json", '[{"id":1},{"id":oops},{"id":3}]')
        config = StreamingConfig(chunk_size=10, bad_record_policy="skip")
        result = execute_streaming(path, "select id", config=config)
        assert json.loads(result.text) == [{"id": 1}, {"id": 3}]
        assert result.stats.records_skipped == 1

    def test_array_error_policy_floor(self, write_text):
        path = write_text("arr.json", '[{"id":1},{"id":oops}]')
        with pytest.raises(BadRecordError):
            execute_streaming(path, "select id", config=StreamingConfig())


class TestBadRecordsWithSort:
    def test_sort_with_skips(self, write_text):
        path = write_text(
            "s.jsonl",
            '{"id":3}\n'
            "{oops}\n"
            '{"id":1}\n'
            '{"id":2}\n',
        )
        config = StreamingConfig(chunk_size=2, bad_record_policy="skip")
        result = execute_streaming(path, "select id sort id", config=config)
        assert json.loads(result.text) == [{"id": 1}, {"id": 2}, {"id": 3}]
        assert result.stats.records_skipped == 1

    def test_grouping_with_skips_uses_only_valid_records(self, write_text):
        path = write_text(
            "s.jsonl",
            '{"g":"a"}\n'
            "{oops}\n"
            '{"g":"a"}\n'
            '{"g":"b"}\n',
        )
        config = StreamingConfig(chunk_size=2, bad_record_policy="skip")
        result = execute_streaming(
            path, "select g, count(*) as n group by g", config=config
        )
        assert json.loads(result.text) == [
            {"g": "a", "n": 2},
            {"g": "b", "n": 1},
        ]
        assert result.stats.records_skipped == 1
