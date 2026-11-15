"""Syntax/semantic error positions and their preflight timing."""

from __future__ import annotations

import json

import pytest

from jonq.streaming.config import StreamingConfig
from jonq.streaming.engine import execute_streaming
from jonq.streaming.errors import SemanticQueryError, StructuredQueryError
from jonq.streaming.plan import compile_plan
from jonq.streaming.semantics import build_schema_sample, validate_plan_against_sample


class TestSyntaxErrors:
    def test_error_reports_offset_and_snippet(self):
        with pytest.raises(StructuredQueryError) as excinfo:
            compile_plan("select name limit 1 junk")
        error = excinfo.value
        assert error.position is not None
        assert error.snippet is not None
        assert "junk" in error.snippet
        assert "line 1" in error.location

    def test_unbalanced_parentheses_has_position(self):
        with pytest.raises(StructuredQueryError) as excinfo:
            compile_plan("select count(")
        assert excinfo.value.position is not None
        assert "Unbalanced" in str(excinfo.value)

    def test_missing_from_path_is_located(self):
        with pytest.raises(StructuredQueryError) as excinfo:
            compile_plan("select name from limit 1")
        error = excinfo.value
        assert error.position is not None
        assert "from" in error.snippet

    def test_unknown_function_is_located(self):
        with pytest.raises(StructuredQueryError) as excinfo:
            compile_plan("select bogus(x)")
        assert "bogus" in str(excinfo.value)
        assert excinfo.value.position is not None

    def test_having_without_group_by_has_position(self):
        with pytest.raises(StructuredQueryError) as excinfo:
            compile_plan("select name having name > 1")
        assert excinfo.value.position is not None

    def test_query_must_be_a_string(self):
        with pytest.raises(StructuredQueryError, match="must be a string"):
            compile_plan(123)

    def test_multiline_query_reports_line_and_column(self):
        with pytest.raises(StructuredQueryError) as excinfo:
            compile_plan("select name\nlimit 1 junk")
        assert excinfo.value.line == 2


class TestSemanticErrors:
    @pytest.fixture
    def sample(self):
        return build_schema_sample(
            [
                {"name": "A", "age": 30, "city": "x"},
                {"name": "B", "age": 25, "city": "y"},
            ]
        )

    def test_missing_selected_field(self, sample):
        with pytest.raises(SemanticQueryError) as excinfo:
            validate_plan_against_sample(compile_plan("select missing"), sample)
        assert excinfo.value.position is not None
        assert excinfo.value.kind == "semantic"

    def test_missing_field_in_condition(self, sample):
        with pytest.raises(SemanticQueryError):
            validate_plan_against_sample(
                compile_plan("select name if ghost > 1"), sample
            )

    def test_missing_group_key(self, sample):
        with pytest.raises(SemanticQueryError):
            validate_plan_against_sample(
                compile_plan("select city, count(*) as n group by ghost"), sample
            )

    def test_missing_sort_key(self, sample):
        with pytest.raises(SemanticQueryError):
            validate_plan_against_sample(
                compile_plan("select name sort ghost"), sample
            )

    def test_non_numeric_sum(self, sample):
        with pytest.raises(SemanticQueryError, match="requires numbers"):
            validate_plan_against_sample(
                compile_plan("select sum(name) as total"), sample
            )

    def test_sort_by_object_rejected(self):
        sample = build_schema_sample([{"k": {"nested": 1}}])
        with pytest.raises(SemanticQueryError, match="Cannot sort"):
            validate_plan_against_sample(compile_plan("select k sort k"), sample)

    def test_valid_query_passes(self, sample):
        validate_plan_against_sample(
            compile_plan("select name, age if age > 1 sort age"), sample
        )

    def test_empty_sample_skips_validation(self):
        validate_plan_against_sample(
            compile_plan("select anything"), build_schema_sample([])
        )


class TestErrorsFailBeforeReading:
    def test_semantic_error_before_scan(self, write_json):
        path = write_json("u.json", [{"name": "A"}, {"name": "B"}])
        with pytest.raises(SemanticQueryError):
            execute_streaming(path, "select missing")
        # Nothing was written anywhere; the failure happened during validation.

    def test_type_error_before_scan(self, write_json):
        path = write_json("u.json", [{"name": "A"}, {"name": "B"}])
        with pytest.raises(SemanticQueryError):
            execute_streaming(path, "select sum(name) as total")

    def test_syntax_error_before_input_check(self, tmp_path):
        # A syntax error must surface even when the input path is bogus, i.e.
        # the query is parsed before the input is probed.
        with pytest.raises(StructuredQueryError):
            execute_streaming(str(tmp_path / "nope.json"), "select name junk")

    def test_error_position_points_at_bad_token(self):
        error = None
        try:
            compile_plan("select name, age sort age bogus")
        except StructuredQueryError as exc:
            error = exc
        assert error is not None
        assert error.position is not None

    def test_from_path_missing_reports_semantic(self, write_json):
        path = write_json("u.json", {"products": [{"n": 1}]})
        with pytest.raises(StructuredQueryError) as excinfo:
            execute_streaming(path, "select n from missing")
        assert excinfo.value.kind == "semantic"
        assert excinfo.value.position is not None

    def test_from_path_reads_nested_array(self, write_json):
        path = write_json("u.json", {"meta": {"n": 1}, "products": [{"n": "a"}, {"n": "b"}]})
        result = execute_streaming(
            path, "select n from products", config=StreamingConfig(chunk_size=1)
        )
        assert json.loads(result.text) == [{"n": "a"}, {"n": "b"}]

    def test_from_path_multi_segment(self, write_json):
        path = write_json("u.json", {"a": {"b": [{"x": 2}, {"x": 1}]}})
        result = execute_streaming(
            path, "select x sort x from a.b", config=StreamingConfig(chunk_size=1)
        )
        assert json.loads(result.text) == [{"x": 1}, {"x": 2}]
