"""Temporary spill cleanup and CLI integration for the bounded engine."""

from __future__ import annotations

import json
import os

import pytest

from jonq.streaming.config import StreamingConfig
from jonq.streaming.engine import execute_streaming
from jonq.streaming.spill import TempWorkspace
from jonq.streaming.errors import BadRecordError, StructuredQueryError


@pytest.fixture
def many_rows(write_json):
    return write_json("s.json", [{"k": i % 5, "i": i} for i in range(60)])


class TestTempCleanup:
    def _spill_dirs(self, directory):
        return [
            name
            for name in os.listdir(directory)
            if name.startswith("jonq_stream_")
        ]

    def test_workspace_removed_after_success(self, many_rows, tmp_path):
        config = StreamingConfig(chunk_size=4, temp_dir=str(tmp_path))
        execute_streaming(many_rows, "select k, i sort k", config=config)
        assert self._spill_dirs(tmp_path) == []

    def test_workspace_removed_after_error(self, many_rows, tmp_path):
        config = StreamingConfig(chunk_size=4, temp_dir=str(tmp_path))
        with pytest.raises(StructuredQueryError):
            execute_streaming(many_rows, "select k sort k junk", config=config)
        assert self._spill_dirs(tmp_path) == []

    def test_workspace_removed_after_bad_record(self, write_text, tmp_path):
        path = write_text("s.jsonl", '{"a":1}\n{oops}\n')
        config = StreamingConfig(chunk_size=4, temp_dir=str(tmp_path))
        with pytest.raises(BadRecordError):
            execute_streaming(path, "select a", config=config)
        assert self._spill_dirs(tmp_path) == []

    def test_keep_temp_preserves_files(self, many_rows, tmp_path):
        config = StreamingConfig(chunk_size=4, temp_dir=str(tmp_path), keep_temp=True)
        try:
            result = execute_streaming(many_rows, "select k, i sort k", config=config)
            assert result.stats.output_rows == 60
            assert os.listdir(tmp_path)
        finally:
            import shutil

            for name in os.listdir(tmp_path):
                shutil.rmtree(os.path.join(tmp_path, name), ignore_errors=True)

    def test_workspace_context_manager_cleans(self):
        workspace = TempWorkspace()
        path = workspace.new_file()
        with open(path, "w") as handle:
            handle.write("x")
        directory = workspace.path
        workspace.cleanup()
        assert not os.path.exists(directory)

    def test_workspace_cleanup_is_idempotent(self):
        workspace = TempWorkspace()
        workspace.cleanup()
        workspace.cleanup()  # second call must not raise


class TestCliBounded:
    def test_bounded_flag_matches_default_output(self, users_file, capsys):
        from unittest.mock import patch

        from jonq.main import main

        with patch("sys.argv", ["jonq", users_file, "select name, age sort age"]):
            main()
        default = capsys.readouterr().out.strip()

        with patch(
            "sys.argv",
            ["jonq", users_file, "select name, age sort age", "--bounded", "--chunk-size", "2"],
        ):
            main()
        bounded = capsys.readouterr().out.strip()

        assert json.loads(default) == json.loads(bounded)

    def test_bounded_stats_go_to_stderr(self, users_file, capsys):
        from unittest.mock import patch

        from jonq.main import main

        with patch(
            "sys.argv",
            ["jonq", users_file, "select name", "--bounded", "--stats", "text"],
        ):
            main()
        captured = capsys.readouterr()
        assert json.loads(captured.out)
        assert "Execution statistics" in captured.err

    def test_bounded_rejects_stdin(self, capsys):
        from unittest.mock import patch

        from jonq.main import main

        with patch("sys.argv", ["jonq", "-", "select name", "--bounded"]):
            with patch("sys.stdin.isatty", return_value=False):
                with pytest.raises(SystemExit) as excinfo:
                    main()
        assert excinfo.value.code == 1

    def test_bounded_rejects_glob(self, capsys):
        from unittest.mock import patch

        from jonq.main import main

        with patch("sys.argv", ["jonq", "*.json", "select name", "--bounded"]):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 1

    def test_bounded_rejects_raw_output(self, users_file, capsys):
        from unittest.mock import patch

        from jonq.main import main

        with patch("sys.argv", ["jonq", users_file, "select name", "--bounded", "-r"]):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 1
        assert "--raw" in capsys.readouterr().err

    def test_bounded_rejects_unsupported_format(self, users_file, capsys):
        from unittest.mock import patch

        from jonq.main import main

        with patch(
            "sys.argv", ["jonq", users_file, "select name", "--bounded", "-f", "yaml"]
        ):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 1

    def test_bounded_limit_option_applies(self, users_file, capsys):
        from unittest.mock import patch

        from jonq.main import main

        with patch(
            "sys.argv",
            ["jonq", users_file, "select name sort name", "--bounded", "-n", "2"],
        ):
            main()
        assert json.loads(capsys.readouterr().out) == [{"name": "Alice"}, {"name": "Bob"}]

    def test_bounded_syntax_error_shows_position(self, users_file, capsys):
        from unittest.mock import patch

        from jonq.main import main

        with patch(
            "sys.argv", ["jonq", users_file, "select name limit 1 junk", "--bounded"]
        ):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "offset" in captured.out
        assert "junk" in captured.out

    def test_bounded_semantic_error_shows_field(self, users_file, capsys):
        from unittest.mock import patch

        from jonq.main import main

        with patch("sys.argv", ["jonq", users_file, "select ghost", "--bounded"]):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "ghost" in captured.out
        assert "does not exist" in captured.out

    def test_bounded_manifest_flag(self, tmp_path, capsys):
        from unittest.mock import patch

        from jonq.main import main

        first = tmp_path / "a.json"
        first.write_text(json.dumps([{"i": 1}]))
        second = tmp_path / "b.json"
        second.write_text(json.dumps([{"i": 2}]))
        manifest = tmp_path / "m.manifest"
        manifest.write_text(json.dumps([str(first), str(second), str(first)]))

        with patch(
            "sys.argv",
            ["jonq", str(manifest), "select i sort i", "--bounded", "--manifest"],
        ):
            main()
        captured = capsys.readouterr()
        assert json.loads(captured.out) == [{"i": 1}, {"i": 2}]

    def test_bounded_bad_record_skip_reports_to_stderr(self, write_text, capsys):
        from unittest.mock import patch

        from jonq.main import main

        path = write_text("s.jsonl", '{"a":1}\n{oops}\n')
        with patch(
            "sys.argv",
            ["jonq", path, "select a", "--bounded", "--bad-records", "skip"],
        ):
            main()
        captured = capsys.readouterr()
        assert json.loads(captured.out) == [{"a": 1}]
        assert "Skipped record" in captured.err
