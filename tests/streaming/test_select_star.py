"""Select *: union-of-keys projection, spooling, and per-format agreement."""

from __future__ import annotations

import json


from jonq.streaming.config import StreamingConfig
from jonq.streaming.engine import execute_streaming


def test_select_star_returns_full_records(write_json):
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    path = write_json("s.json", data)
    result = execute_streaming(path, "select *", config=StreamingConfig(chunk_size=1))
    assert json.loads(result.text) == data


def test_select_star_with_condition(write_json):
    data = [{"a": 1}, {"a": 5}, {"a": 9}]
    path = write_json("s.json", data)
    result = execute_streaming(
        path, "select * if a > 2", config=StreamingConfig(chunk_size=1)
    )
    assert json.loads(result.text) == [{"a": 5}, {"a": 9}]


def test_select_star_with_sort(write_json):
    data = [{"a": 3}, {"a": 1}, {"a": 2}]
    path = write_json("s.json", data)
    result = execute_streaming(
        path, "select * sort a", config=StreamingConfig(chunk_size=2)
    )
    assert json.loads(result.text) == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_select_star_heterogeneous_keys_union(write_json):
    data = [{"a": 1}, {"b": 2}, {"a": 3, "c": 4}]
    path = write_json("s.json", data)
    result = execute_streaming(
        path, "select *", output_format="json", config=StreamingConfig(chunk_size=2)
    )
    rows = json.loads(result.text)
    assert list(rows[0].keys()) == ["a", "b", "c"]
    assert rows[0] == {"a": 1, "b": None, "c": None}
    assert rows[1] == {"a": None, "b": 2, "c": None}
    assert rows[2] == {"a": 3, "b": None, "c": 4}


def test_select_star_scalar_root_uses_value_key(write_text):
    path = write_text("s.json", "[1, 2, 3]")
    result = execute_streaming(path, "select *", config=StreamingConfig(chunk_size=1))
    assert json.loads(result.text) == [
        {"value": 1},
        {"value": 2},
        {"value": 3},
    ]


def test_select_star_jsonl_with_sort_and_limit(write_json):
    data = [{"k": i % 3, "i": i} for i in range(9)]
    path = write_json("s.json", data)
    result = execute_streaming(
        path,
        "select * sort k limit 4",
        config=StreamingConfig(chunk_size=2),
    )
    rows = json.loads(result.text)
    assert [row["k"] for row in rows] == [0, 0, 0, 1]
