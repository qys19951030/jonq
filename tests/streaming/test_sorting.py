"""External merge sort: correctness, stability, chunk independence."""

from __future__ import annotations

import json
import random

import pytest

from jonq.streaming.config import StreamingConfig
from jonq.streaming.engine import execute_streaming
from jonq.streaming.sortagg import ExternalSorter
from jonq.streaming.spill import TempWorkspace


def run(path, query, *, fmt="json", config=None):
    return execute_streaming(
        path, query, output_format=fmt, config=config or StreamingConfig(chunk_size=3)
    )


def rows_of(result):
    return json.loads(result.text)


class TestSortMatchesFullSort:
    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 100])
    def test_ascending_matches_python_sort(self, write_json, chunk_size):
        data = [{"k": v, "tag": i} for i, v in enumerate([5, 3, 9, 1, 7, 3, 8, 2])]
        path = write_json("s.json", data)
        result = run(path, "select k, tag sort k", config=StreamingConfig(chunk_size=chunk_size))
        expected = sorted(data, key=lambda row: row["k"])
        assert rows_of(result) == expected

    @pytest.mark.parametrize("chunk_size", [1, 2, 5, 100])
    def test_descending_matches_python_sort(self, write_json, chunk_size):
        data = [{"k": v, "tag": i} for i, v in enumerate([5, 3, 9, 1, 7, 3, 8, 2])]
        path = write_json("s.json", data)
        result = run(
            path, "select k, tag sort k desc", config=StreamingConfig(chunk_size=chunk_size)
        )
        expected = sorted(data, key=lambda row: row["k"], reverse=True)
        assert rows_of(result) == expected

    def test_result_independent_of_chunk_size(self, write_json):
        random.seed(7)
        data = [{"k": random.randint(0, 20), "tag": i} for i in range(200)]
        path = write_json("s.json", data)
        outputs = []
        for chunk_size in (1, 4, 17, 1000):
            result = run(
                path, "select k, tag sort k", config=StreamingConfig(chunk_size=chunk_size)
            )
            outputs.append(rows_of(result))
        assert all(output == outputs[0] for output in outputs)


class TestStability:
    def test_equal_keys_keep_input_order_ascending(self, write_json):
        data = [
            {"k": 1, "tag": "a"},
            {"k": 2, "tag": "b"},
            {"k": 1, "tag": "c"},
            {"k": 2, "tag": "d"},
            {"k": 1, "tag": "e"},
        ]
        path = write_json("s.json", data)
        result = run(path, "select k, tag sort k", config=StreamingConfig(chunk_size=2))
        assert rows_of(result) == [
            {"k": 1, "tag": "a"},
            {"k": 1, "tag": "c"},
            {"k": 1, "tag": "e"},
            {"k": 2, "tag": "b"},
            {"k": 2, "tag": "d"},
        ]

    def test_equal_keys_keep_input_order_descending(self, write_json):
        data = [
            {"k": 1, "tag": "a"},
            {"k": 2, "tag": "b"},
            {"k": 1, "tag": "c"},
            {"k": 2, "tag": "d"},
        ]
        path = write_json("s.json", data)
        result = run(
            path, "select k, tag sort k desc", config=StreamingConfig(chunk_size=2)
        )
        assert rows_of(result) == [
            {"k": 2, "tag": "b"},
            {"k": 2, "tag": "d"},
            {"k": 1, "tag": "a"},
            {"k": 1, "tag": "c"},
        ]

    def test_stability_across_many_spill_files(self, write_json):
        data = [{"k": 1, "tag": i} for i in range(40)]
        path = write_json("s.json", data)
        result = run(path, "select k, tag sort k", config=StreamingConfig(chunk_size=3))
        assert [row["tag"] for row in rows_of(result)] == list(range(40))

    @pytest.mark.parametrize("fan_in", [2, 3, 7, 100])
    def test_multi_pass_merge_preserves_direction_and_stability(self, write_json, fan_in):
        data = [{"k": i % 11, "seq": i} for i in range(300)]
        path = write_json("s.json", data)
        config = StreamingConfig(chunk_size=6, merge_fan_in=fan_in)

        ascending = rows_of(run(path, "select k, seq sort k", config=config))
        assert ascending == sorted(
            ({"k": row["k"], "seq": row["seq"]} for row in data),
            key=lambda row: row["k"],
        )

        descending = rows_of(run(path, "select k, seq sort k desc", config=config))
        expected = []
        for key in sorted({row["k"] for row in data}, reverse=True):
            expected.extend(
                {"k": row["k"], "seq": row["seq"]} for row in data if row["k"] == key
            )
        assert descending == expected

    def test_sorter_direct_stability(self):
        workspace = TempWorkspace()
        try:
            sorter = ExternalSorter(workspace, chunk_size=2, fan_in=2)
            for i, key in enumerate([2, 1, 2, 1, 2, 1, 2, 1]):
                sorter.add(key, {"key": key, "tag": i})
            out = list(sorter.finish())
            assert [row["tag"] for row in out if row["key"] == 1] == [1, 3, 5, 7]
            assert [row["tag"] for row in out if row["key"] == 2] == [0, 2, 4, 6]
        finally:
            workspace.cleanup()


class TestSortSpilling:
    def test_spill_files_are_created(self, write_json):
        path = write_json("s.json", [{"k": i % 5, "i": i} for i in range(30)])
        result = run(path, "select k, i sort k", config=StreamingConfig(chunk_size=4))
        assert result.stats.spill_chunk_count >= 1
        assert result.stats.spill_bytes > 0

    def test_no_spill_for_single_chunk(self, write_json):
        path = write_json("s.json", [{"k": i} for i in range(5)])
        result = run(path, "select k sort k", config=StreamingConfig(chunk_size=100))
        assert result.stats.output_rows == 5

    def test_limit_applies_after_sort(self, write_json):
        data = [{"k": v} for v in (5, 1, 4, 2, 3)]
        path = write_json("s.json", data)
        result = run(
            path, "select k sort k limit 2", config=StreamingConfig(chunk_size=2)
        )
        assert rows_of(result) == [{"k": 1}, {"k": 2}]
        assert result.stats.output_rows == 2

    def test_mixed_types_do_not_crash(self, write_json):
        data = [{"k": 1}, {"k": "a"}, {"k": None}, {"k": True}, {"k": 2}]
        path = write_json("s.json", data)
        result = run(path, "select k sort k", config=StreamingConfig(chunk_size=2))
        assert result.stats.output_rows == 5

    def test_sort_by_field_not_selected(self, write_json):
        data = [{"name": "b", "age": 2}, {"name": "a", "age": 1}, {"name": "c", "age": 3}]
        path = write_json("s.json", data, )
        result = run(path, "select name, age sort age", config=StreamingConfig(chunk_size=2))
        assert rows_of(result) == [
            {"name": "a", "age": 1},
            {"name": "b", "age": 2},
            {"name": "c", "age": 3},
        ]

    def test_sort_on_aggregate_alias(self, write_json):
        data = [
            {"city": "A", "v": 1},
            {"city": "B", "v": 10},
            {"city": "A", "v": 2},
        ]
        path = write_json("s.json", data)
        result = run(
            path,
            "select city, count(*) as n group by city sort n desc",
            config=StreamingConfig(chunk_size=2),
        )
        assert rows_of(result) == [{"city": "A", "n": 2}, {"city": "B", "n": 1}]
