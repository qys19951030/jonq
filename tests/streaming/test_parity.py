"""Parity with the legacy whole-file path.

For queries the bounded engine supports, results must match ``jonq.api.query``
on the same fixture data.  These tests pin that agreement so the new engine
cannot drift from the jq-backed one.
"""

from __future__ import annotations

import json

import pytest

from jonq.api import query as legacy_query
from jonq.streaming.config import StreamingConfig
from jonq.streaming.engine import execute_streaming

FIXTURE = "tests/json_test_files/complex.json"


def stream_rows(path, q, chunk_size=2):
    result = execute_streaming(
        path, q, output_format="json", config=StreamingConfig(chunk_size=chunk_size)
    )
    data = json.loads(result.text)
    return data if isinstance(data, list) else [data]


def legacy_rows(path, q):
    data = legacy_query(path, q, validate=False)
    return data if isinstance(data, list) else [data]


def test_fixture_exists():
    import os

    assert os.path.isfile(FIXTURE)


@pytest.fixture
def fixture_path():
    import os

    return os.path.abspath(FIXTURE)


class TestSimpleParity:
    def test_select_star(self, fixture_path):
        assert stream_rows(fixture_path, "select *") == legacy_rows(fixture_path, "select *")

    def test_filter(self, fixture_path):
        # Discover a usable numeric field from the fixture at runtime.
        sample = legacy_rows(fixture_path, "select *")[0]
        numeric = [
            key for key, value in sample.items() if isinstance(value, (int, float))
        ]
        if not numeric:
            pytest.skip("fixture has no numeric field")
        field = numeric[0]
        q = f"select * if {field} >= 1"
        assert stream_rows(fixture_path, q) == legacy_rows(fixture_path, q)


class TestRepositoryFixtures:
    """Run the same queries over the checked-in array fixtures."""

    @pytest.mark.parametrize(
        "fixture",
        ["all.json", "simple.json", "smoke.json"],
    )
    def test_repository_array_fixtures(self, fixture):
        import os

        path = os.path.abspath(os.path.join("tests/json_test_files", fixture))
        if not os.path.isfile(path):
            pytest.skip(f"{fixture} not present")

        # smoke.json has no `id`; derive a common field from the first record.
        first = legacy_rows(path, "select *")[0]
        sortable = [
            key for key, value in first.items() if isinstance(value, (int, float, str))
        ]
        queries = ["select *", "select name", "select distinct name"]
        if sortable:
            field = sortable[0]
            queries.extend([f"select {field} sort {field}", f"select name if {field} != null"])
        for query in queries:
            assert stream_rows(path, query) == legacy_rows(path, query), (fixture, query)

    def test_nested_fixture_with_from_path(self):
        import os

        path = os.path.abspath("tests/json_test_files/nested.json")
        if not os.path.isfile(path):
            pytest.skip("nested.json not present")
        streamed = stream_rows(path, "select name, profile.age as age")
        legacy = legacy_rows(path, "select name, profile.age as age")
        assert streamed == legacy


class TestUsersParity:
    @pytest.fixture
    def users(self, write_json):
        data = [
            {"city": "NY", "age": 30, "team": "a", "name": "A"},
            {"city": "NY", "age": 20, "team": "b", "name": "B"},
            {"city": "LA", "age": 40, "team": "a", "name": "C"},
            {"city": "LA", "age": 35, "team": "a", "name": "C"},
            {"city": "SF", "age": 25, "team": "b", "name": "D"},
        ]
        return write_json("u.json", data)

    @pytest.mark.parametrize(
        "q",
        [
            "select name, age",
            "select name, age if age > 25",
            "select distinct city",
            "select distinct city sort city desc",
            "select distinct city limit 2",
            "select name, age limit 2",
        ],
    )
    def test_non_aggregate_queries(self, users, q):
        assert stream_rows(users, q) == legacy_rows(users, q)

    @pytest.mark.parametrize(
        "q",
        [
            "select city, count(*) as n group by city",
            "select city, sum(age) as total group by city",
            "select count(*) as n",
            "select city, count(*) as n group by city having n > 1",
        ],
    )
    def test_aggregate_queries(self, users, q):
        assert stream_rows(users, q) == legacy_rows(users, q)

    def test_sort_ascending_is_identical(self, users):
        q = "select name, age sort age"
        assert stream_rows(users, q) == legacy_rows(users, q)

    def test_sort_descending_on_distinct_keys_is_identical(self, users):
        q = "select name, age sort age desc"
        streamed = stream_rows(users, q)
        legacy = legacy_rows(users, q)
        # Descending order matches; stability inside equal keys is a documented
        # improvement over the legacy reverse, and this fixture has unique ages.
        assert streamed == legacy

    def test_grouped_sort_with_tied_keys_is_stable(self, users):
        # The legacy `sort_by(...) | reverse` reverses tied groups too; the
        # bounded engine keeps input order inside a tie, which the requirements
        # mandate.  Ordering by key is therefore compared as a multiset of keys
        # plus monotonicity rather than exact row order.
        q = "select city, avg(age) as a group by city sort a desc"
        streamed = stream_rows(users, q)
        values = [row["a"] for row in streamed]
        assert values == sorted(values, reverse=True)
        assert sorted(row["city"] for row in streamed) == sorted(
            row["city"] for row in legacy_rows(users, q)
        )
