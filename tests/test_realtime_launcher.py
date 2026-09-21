from pathlib import Path
import re
import subprocess
from unittest.mock import Mock

from realtime_launcher import launch_realtime


def test_launch_realtime_reports_running_child(tmp_path):
    process = Mock(pid=1234)
    process.wait.side_effect = subprocess.TimeoutExpired(
        cmd=["python", "realtime.py"],
        timeout=1.0,
    )
    popen = Mock(return_value=process)

    result = launch_realtime(
        Path("realtime.py"), 0, tmp_path, tmp_path, popen=popen
    )

    assert result.started is True
    assert result.pid == 1234
    assert result.error is None


def test_launch_realtime_returns_tail_for_immediate_failure(tmp_path):
    process = Mock(pid=1234)
    process.wait.return_value = 1

    def fake_popen(*args, **kwargs):
        kwargs["stdout"].write(b"x" * 5000 + b"camera failed")
        kwargs["stdout"].flush()
        return process

    result = launch_realtime(
        Path("realtime.py"), 9, tmp_path, tmp_path, popen=fake_popen
    )

    assert result.started is False
    assert result.error.endswith("camera failed")
    assert len(result.error.encode("utf-8")) <= 4096


def test_launch_realtime_includes_exit_code_when_tail_is_empty(tmp_path):
    process = Mock(pid=2222)
    process.wait.return_value = 3

    def fake_popen(*args, **kwargs):
        kwargs["stdout"].flush()
        return process

    result = launch_realtime(
        Path("realtime.py"), 0, tmp_path, tmp_path, popen=fake_popen
    )

    assert result.started is False
    assert result.pid == 2222
    assert "exit code 3" in result.error
    assert "Log was empty" in result.error


def test_launch_realtime_returns_actionable_error_when_tail_unreadable(tmp_path, monkeypatch):
    process = Mock(pid=3333)
    process.wait.return_value = 5

    def fake_popen(*args, **kwargs):
        kwargs["stdout"].write(b"startup output")
        kwargs["stdout"].flush()
        return process

    import realtime_launcher
    monkeypatch.setattr(
        realtime_launcher,
        "_read_tail",
        Mock(side_effect=OSError("tail read denied")),
    )

    result = launch_realtime(
        Path("realtime.py"), 0, tmp_path, tmp_path, popen=fake_popen
    )

    assert result.started is False
    assert result.pid == 3333
    assert "exit code 5" in result.error
    assert "could not read startup log tail" in result.error
    assert "tail read denied" in result.error


def test_launch_realtime_returns_error_when_popen_raises_oserror(tmp_path):
    def failing_popen(*args, **kwargs):
        raise OSError("spawn failed")

    result = launch_realtime(
        Path("realtime.py"),
        0,
        tmp_path,
        tmp_path,
        popen=failing_popen,
    )

    assert result.started is False
    assert result.pid is None
    assert result.error == "spawn failed"


def test_launch_realtime_returns_error_when_log_dir_creation_fails(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"

    def fail_mkdir(self, *args, **kwargs):
        if self == log_dir:
            raise OSError("mkdir failed")
        return None

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)

    result = launch_realtime(
        Path("realtime.py"),
        0,
        tmp_path,
        log_dir,
    )

    assert result.started is False
    assert result.pid is None
    assert result.error == "mkdir failed"


def test_launch_realtime_log_filename_has_utc_prefix_and_unique_suffix(tmp_path):
    process = Mock(pid=9876)
    process.wait.side_effect = subprocess.TimeoutExpired(
        cmd=["python", "realtime.py"],
        timeout=1.0,
    )
    popen = Mock(return_value=process)

    first = launch_realtime(
        Path("realtime.py"),
        0,
        tmp_path,
        tmp_path,
        popen=popen,
    )
    second = launch_realtime(
        Path("realtime.py"),
        0,
        tmp_path,
        tmp_path,
        popen=popen,
    )

    pattern = re.compile(r"^realtime-\d{8}T\d{6}\d{6}Z-[0-9a-f]{32}\.log$")
    assert pattern.match(first.log_path.name)
    assert pattern.match(second.log_path.name)
    assert first.log_path.name != second.log_path.name
