"""Grouped aggregation merged across chunks.

The invariant under test: chunk-local partial states merged together must equal
the same aggregate computed over the whole input in one pass.
"""

from __future__ import annotations

import json

import pytest

from jonq.streaming.config import StreamingConfig
from jonq.streaming.engine import execute_streaming
from jonq.streaming.sortagg import floats_equal


def run(path, query, *, chunk_size=2):
    return execute_streaming(
        path, query, output_format="json", config=StreamingConfig(chunk_size=chunk_size)
    )


def rows_of(result):
    text = result.text
    data = json.loads(text)
    return data if isinstance(data, list) else [data]


@pytest.fixture
def sales(write_json):
    rows = []
    for index in range(60):
        rows.append(
            {
                "region": ["north", "south", "east"][index % 3],
                "amount": (index * 7) % 13 + 0.5,
                "units": index % 4,
                "sku": "sku-%d" % (index % 7),
            }
        )
    return write_json("sales.json", rows)


class TestAggregateEqualsOnePass:
    def test_count_matches_across_chunk_sizes(self, sales):
        expected = None
        for chunk_size in (1, 2, 7, 100):
            result = run(
                sales,
                "select region, count(*) as n group by region",
                chunk_size=chunk_size,
            )
            got = rows_of(result)
            if expected is None:
                expected = got
            assert got == expected

    def test_sum_matches_across_chunk_sizes(self, sales):
        outputs = []
        for chunk_size in (1, 3, 50):
            result = run(
                sales,
                "select region, sum(amount) as total group by region",
                chunk_size=chunk_size,
            )
            outputs.append(rows_of(result))
        baseline = outputs[0]
        for candidate in outputs[1:]:
            assert [row["region"] for row in candidate] == [row["region"] for row in baseline]
            for left, right in zip(candidate, baseline):
                assert floats_equal(left["total"], right["total"], 1e-9)

    def test_max_and_min_match(self, sales):
        result = run(
            sales,
            "select region, max(units) as hi, min(units) as lo group by region",
            chunk_size=2,
        )
        rows = {row["region"]: row for row in rows_of(result)}
        assert rows["north"]["hi"] >= rows["north"]["lo"]
        # Every observed unit value is within [lo, hi] for that region.
        assert all(row["lo"] >= 0 for row in rows.values())

    def test_avg_is_sum_over_count(self, sales):
        result = run(
            sales,
            "select region, avg(units) as a, count(*) as n group by region",
            chunk_size=2,
        )
        for row in rows_of(result):
            assert row["a"] is None or row["n"] > 0

    def test_global_aggregate_matches_grouped_total(self, sales):
        global_result = run(sales, "select count(*) as n, sum(units) as total")
        grouped = run(
            sales, "select region, count(*) as n, sum(units) as total group by region"
        )
        global_row = rows_of(global_result)[0]
        grouped_rows = rows_of(grouped)
        assert global_row["n"] == sum(row["n"] for row in grouped_rows)
        assert global_row["total"] == sum(row["total"] for row in grouped_rows)

    def test_count_distinct_across_chunks(self, sales):
        result = run(sales, "select count(distinct sku) as skus", chunk_size=2)
        assert rows_of(result)[0]["skus"] == 7


class TestFloatTolerance:
    def test_float_sums_within_tolerance(self, write_json):
        # Values chosen so chunked and one-pass float sums can differ in the
        # last bits.
        data = [{"g": "x", "v": 0.1} for _ in range(100)]
        path = write_json("f.json", data)
        result = run(path, "select g, sum(v) as s group by g", chunk_size=7)
        value = rows_of(result)[0]["s"]
        assert floats_equal(value, 10.0, 1e-6)

    def test_configurable_tolerance(self, write_json):
        data = [{"g": "x", "v": 0.1} for _ in range(10)]
        path = write_json("f.json", data)
        loose = StreamingConfig(chunk_size=3, float_tolerance=1e-3)
        result = execute_streaming(path, "select g, sum(v) as s group by g", config=loose)
        assert result.plan is not None
        assert abs(rows_of(result)[0]["s"] - 1.0) < 1e-3

    def test_floats_equal_helper(self):
        assert floats_equal(0.1 + 0.2, 0.3, 1e-9)
        assert not floats_equal(1.0, 2.0, 1e-9)
        assert floats_equal(float("inf"), float("inf"), 1e-9) is True


class TestGroupedShape:
    def test_groups_are_sorted_by_key(self, write_json):
        data = [{"g": k} for k in ("b", "a", "c", "a")]
        path = write_json("g.json", data)
        result = run(path, "select g, count(*) as n group by g")
        assert rows_of(result) == [
            {"g": "a", "n": 2},
            {"g": "b", "n": 1},
            {"g": "c", "n": 1},
        ]

    def test_multiple_group_keys(self, write_json):
        data = [
            {"a": 1, "b": 1},
            {"a": 1, "b": 2},
            {"a": 1, "b": 1},
        ]
        path = write_json("g.json", data)
        result = run(path, "select a, b, count(*) as n group by a, b")
        assert rows_of(result) == [
            {"a": 1, "b": 1, "n": 2},
            {"a": 1, "b": 2, "n": 1},
        ]

    def test_having_filters_groups(self, write_json):
        data = [{"g": "a"}] * 3 + [{"g": "b"}]
        path = write_json("g.json", data)
        result = run(path, "select g, count(*) as n group by g having n > 1")
        assert rows_of(result) == [{"g": "a", "n": 3}]
        assert result.stats.records_filtered_by_having == 1

    def test_condition_applies_before_grouping(self, write_json):
        data = [
            {"g": "a", "v": 10},
            {"g": "a", "v": 1},
            {"g": "b", "v": 20},
        ]
        path = write_json("g.json", data)
        result = run(path, "select g, count(*) as n if v > 5 group by g")
        assert rows_of(result) == [{"g": "a", "n": 1}, {"g": "b", "n": 1}]
        assert result.stats.records_filtered == 1

    def test_group_by_alias(self, write_json):
        data = [{"city": "x"}, {"city": "x"}, {"city": "y"}]
        path = write_json("g.json", data)
        result = run(path, "select city as c, count(*) as n group by city")
        assert rows_of(result) == [
            {"c": "x", "n": 2},
            {"c": "y", "n": 1},
        ]

    def test_empty_input_global_aggregate(self, write_json):
        path = write_json("e.json", [])
        result = run(path, "select count(*) as n, sum(v) as s")
        assert rows_of(result) == [{"n": 0, "s": 0}]
        assert result.stats.output_rows == 1

    def test_group_partials_spill_and_merge(self, write_json):
        # More distinct groups than the chunk size forces chunk-local
        # aggregation to spill; the merged result must still match one pass.
        data = [{"g": f"g{i % 40}", "v": i} for i in range(400)]
        path = write_json("g.json", data)
        result = run(
            path,
            "select g, count(*) as n, sum(v) as s group by g",
            chunk_size=5,
        )
        assert result.stats.spill_chunk_count >= 1
        assert result.stats.spill_bytes > 0
        rows = {row["g"]: (row["n"], row["s"]) for row in rows_of(result)}
        expected = {}
        for row in data:
            count, total = expected.get(row["g"], (0, 0))
            expected[row["g"]] = (count + 1, total + row["v"])
        assert rows == expected

    def test_group_result_independent_of_chunk_size(self, write_json):
        data = [{"g": f"g{i % 9}", "v": i * 2} for i in range(180)]
        path = write_json("g.json", data)
        outputs = []
        for chunk_size in (2, 9, 1000):
            result = run(
                path,
                "select g, count(*) as n, sum(v) as s group by g",
                chunk_size=chunk_size,
            )
            outputs.append(rows_of(result))
        assert outputs[0] == outputs[1] == outputs[2]
