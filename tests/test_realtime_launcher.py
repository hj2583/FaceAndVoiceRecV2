from pathlib import Path
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
