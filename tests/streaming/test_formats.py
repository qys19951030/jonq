"""Cross-format agreement: row count, field order, escaping."""

from __future__ import annotations

import csv
import io
import json

import pytest

from jonq.streaming.config import StreamingConfig
from jonq.streaming.engine import execute_streaming
from jonq.streaming.rows import cell_text, merge_headers


@pytest.fixture
def messy_rows(write_json):
    return write_json(
        "messy.json",
        [
            {"name": "Plain", "note": "no separators", "n": 1},
            {"name": "Comma, Value", "note": 'has "quotes"', "n": 2},
            {"name": "Line\nBreak", "note": "tab\there", "n": 3},
            {"name": "Trailing ", "note": " leading", "n": 4},
        ],
    )


def parse_csv(text):
    return list(csv.reader(io.StringIO(text)))


class TestSameRowsAndOrder:
    @pytest.mark.parametrize(
        "query",
        [
            "select name, note, n",
            "select n, name",
            "select name, note, n sort n desc",
            "select name, note, n limit 2",
        ],
    )
    def test_row_count_matches(self, messy_rows, query):
        config = StreamingConfig(chunk_size=2)
        json_result = execute_streaming(
            messy_rows, query, output_format="json", config=config
        )
        csv_result = execute_streaming(
            messy_rows, query, output_format="csv", config=config
        )
        table_result = execute_streaming(
            messy_rows, query, output_format="table", config=config
        )
        jsonl_result = execute_streaming(
            messy_rows, query, output_format="jsonl", config=config
        )
        assert json_result.stats.output_rows == csv_result.stats.output_rows
        assert json_result.stats.output_rows == table_result.stats.output_rows
        assert json_result.stats.output_rows == jsonl_result.stats.output_rows
        csv_body = parse_csv(csv_result.text)
        assert len(csv_body) - 1 == json_result.stats.output_rows
        table_lines = [
            line for line in table_result.text.splitlines() if line.startswith(" ")
        ]
        # Header + one line per row.
        assert len(table_lines) == json_result.stats.output_rows + 1

    def test_field_order_matches_json(self, messy_rows):
        config = StreamingConfig(chunk_size=2)
        query = "select note, name, n"
        json_result = execute_streaming(
            messy_rows, query, output_format="json", config=config
        )
        csv_result = execute_streaming(
            messy_rows, query, output_format="csv", config=config
        )
        table_result = execute_streaming(
            messy_rows, query, output_format="table", config=config
        )
        json_keys = list(json.loads(json_result.text)[0].keys())
        csv_header = parse_csv(csv_result.text)[0]
        table_header = [
            cell.strip()
            for cell in table_result.text.splitlines()[0].strip().strip("|").split("|")
        ]
        assert json_keys == csv_header == table_header

    def test_select_star_header_order(self, write_text):
        # Records with differing key order: the union of keys in first-seen
        # order must win in every format, and rows are padded so all formats
        # agree on the field set.
        path = write_text(
            "s.json",
            '[{"b":1,"a":2},{"a":3,"c":4}]',
        )
        config = StreamingConfig(chunk_size=2)
        json_result = execute_streaming(path, "select *", output_format="json", config=config)
        csv_result = execute_streaming(path, "select *", output_format="csv", config=config)
        table_result = execute_streaming(path, "select *", output_format="table", config=config)
        rows = json.loads(json_result.text)
        keys = list(rows[0].keys())
        assert keys == ["b", "a", "c"]
        assert rows[1] == {"b": None, "a": 3, "c": 4}
        assert parse_csv(csv_result.text)[0] == keys
        table_header = [
            cell.strip()
            for cell in table_result.text.splitlines()[0].strip().strip("|").split("|")
        ]
        assert table_header == keys


class TestEscaping:
    def test_csv_quote_comma_newline(self, messy_rows):
        config = StreamingConfig(chunk_size=10)
        result = execute_streaming(
            messy_rows, "select name, note, n", output_format="csv", config=config
        )
        rows = parse_csv(result.text)
        assert rows[0] == ["name", "note", "n"]
        assert rows[1] == ["Plain", "no separators", "1"]
        assert rows[2] == ["Comma, Value", 'has "quotes"', "2"]
        # A newline inside a cell stays inside the quoted field, not as a row break.
        assert rows[3] == ["Line\nBreak", "tab\there", "3"]

    def test_json_escapes_match_values(self, messy_rows):
        config = StreamingConfig(chunk_size=10)
        result = execute_streaming(
            messy_rows, "select name, note", output_format="json", config=config
        )
        rows = json.loads(result.text)
        assert rows[1]["name"] == "Comma, Value"
        assert rows[1]["note"] == 'has "quotes"'
        assert rows[2]["name"] == "Line\nBreak"

    def test_table_does_not_truncate_or_split_rows(self, messy_rows):
        config = StreamingConfig(chunk_size=10)
        result = execute_streaming(
            messy_rows, "select name, note", output_format="table", config=config
        )
        lines = result.text.splitlines()
        body = [line for line in lines if line.startswith(" ")]
        assert len(body) == 5  # header + 4 rows
        # The newline inside a cell is escaped, so it cannot break the row.
        assert any("Line\\nBreak" in line for line in body)
        assert any('has "quotes"' in line for line in body)

    def test_cell_text_is_shared_source(self):
        assert cell_text("a,b") == "a,b"
        assert cell_text(True) == "true"
        assert cell_text(None) == "null"
        assert cell_text({"a": 1}) == '{"a":1}'

    def test_table_and_csv_agree_on_cell_text(self, write_json):
        path = write_json("s.json", [{"v": "a,b"}, {"v": 'q"z'}])
        config = StreamingConfig(chunk_size=10)
        table_result = execute_streaming(
            path, "select v", output_format="table", config=config
        )
        csv_result = execute_streaming(
            path, "select v", output_format="csv", config=config
        )
        for value in ("a,b", 'q"z'):
            assert value in table_result.text
        csv_rows = parse_csv(csv_result.text)
        assert ["a,b"] in csv_rows
        assert ['q"z'] in csv_rows

    def test_jsonl_parses_back_to_same_rows(self, messy_rows):
        config = StreamingConfig(chunk_size=2)
        json_result = execute_streaming(
            messy_rows, "select name, note, n", output_format="json", config=config
        )
        jsonl_result = execute_streaming(
            messy_rows, "select name, note, n", output_format="jsonl", config=config
        )
        jsonl_rows = [
            json.loads(line) for line in jsonl_result.text.splitlines() if line.strip()
        ]
        assert jsonl_rows == json.loads(json_result.text)


class TestEmptyResults:
    @pytest.mark.parametrize("fmt", ["json", "jsonl", "csv", "table"])
    def test_empty_input_still_renders_header(self, write_json, fmt):
        path = write_json("e.json", [])
        result = execute_streaming(
            path, "select name, age", output_format=fmt, config=StreamingConfig()
        )
        assert result.stats.output_rows == 0
        if fmt == "json":
            assert result.text == "[]"
        if fmt == "csv":
            assert parse_csv(result.text)[0] == ["name", "age"]

    @pytest.mark.parametrize("fmt", ["json", "csv", "table"])
    def test_no_matching_rows_keeps_row_count_zero(self, write_json, fmt):
        path = write_json("s.json", [{"a": 1}, {"a": 2}])
        result = execute_streaming(
            path, "select a if a > 100", output_format=fmt, config=StreamingConfig()
        )
        assert result.stats.output_rows == 0
        if fmt == "json":
            assert json.loads(result.text) == []


class TestHeaderHelpers:
    def test_merge_headers_first_seen(self):
        assert merge_headers([{"b": 1, "a": 2}, {"a": 3, "c": 4}]) == ["b", "a", "c"]

    def test_csv_with_unknown_headers_spools(self, write_text):
        path = write_text("s.json", '[{"z":1,"a":2}]')
        result = execute_streaming(
            path, "select *", output_format="csv", config=StreamingConfig(chunk_size=1)
        )
        assert parse_csv(result.text)[0] == ["z", "a"]
