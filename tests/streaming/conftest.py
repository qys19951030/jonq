"""Shared fixtures and helpers for the streaming engine tests.

Everything here uses repository fixtures and pytest's ``tmp_path``; nothing
touches the network.
"""

from __future__ import annotations

import io
import json

import pytest

from jonq.streaming.config import StreamingConfig
from jonq.streaming.engine import execute_streaming


@pytest.fixture
def write_json(tmp_path):
    def _write(name, data):
        path = tmp_path / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    return _write


@pytest.fixture
def write_text(tmp_path):
    def _write(name, text):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    return _write


@pytest.fixture
def users_file(write_json):
    return write_json(
        "users.json",
        [
            {"id": 1, "name": "Alice", "age": 30, "city": "New York", "team": "core"},
            {"id": 2, "name": "Bob", "age": 25, "city": "Chicago", "team": "core"},
            {"id": 3, "name": "Cara", "age": 41, "city": "New York", "team": "infra"},
            {"id": 4, "name": "Dan", "age": 19, "city": "Austin", "team": "infra"},
            {"id": 5, "name": "Eve", "age": 35, "city": "Chicago", "team": "data"},
            {"id": 6, "name": "Finn", "age": 30, "city": "Austin", "team": "data"},
        ],
    )


@pytest.fixture
def default_config():
    return StreamingConfig(chunk_size=2)


def run(path, query, *, fmt="json", config=None, **kwargs):
    return execute_streaming(
        path,
        query,
        output_format=fmt,
        config=config or StreamingConfig(chunk_size=2),
        **kwargs,
    )


def load_rows(result):
    if result.output_format in ("json", "jsonl"):
        return json.loads(result.text) if result.output_format == "json" else [
            json.loads(line) for line in result.text.splitlines() if line.strip()
        ]
    raise AssertionError("load_rows only handles json/jsonl")


def csv_rows(text, delimiter=","):
    import csv

    return list(csv.reader(io.StringIO(text), delimiter=delimiter))
