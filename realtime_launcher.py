from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import sys
import uuid


@dataclass(frozen=True)
class LaunchResult:
    started: bool
    pid: int | None
    error: str | None
    log_path: Path


def _read_tail(path: Path, max_bytes: int = 4096) -> str:
    with path.open("rb") as log_file:
        log_file.seek(0, os.SEEK_END)
        size = log_file.tell()
        log_file.seek(max(0, size - max_bytes))
        return log_file.read(max_bytes).decode("utf-8", errors="replace").strip()


def launch_realtime(
    script: Path,
    camera: int,
    cwd: Path,
    log_dir: Path,
    startup_timeout: float = 1.0,
    popen=subprocess.Popen,
) -> LaunchResult:
    log_path: Path | None = None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    unique = uuid.uuid4().hex
    log_path = Path(log_dir) / f"realtime-{stamp}-{unique}.log"

    command = [sys.executable, str(script), "--camera", str(camera)]
    popen_kwargs = {
        "cwd": str(cwd),
        "stdout": None,
        "stderr": subprocess.STDOUT,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("wb") as output:
            popen_kwargs["stdout"] = output
            process = popen(command, **popen_kwargs)
            try:
                process.wait(timeout=startup_timeout)
            except subprocess.TimeoutExpired:
                return LaunchResult(True, process.pid, None, log_path)
    except OSError as exc:
        fallback_log_path = log_path if log_path is not None else Path(log_dir) / "realtime-launch.log"
        return LaunchResult(False, None, str(exc), fallback_log_path)

    error = _read_tail(log_path)
    if not error:
        error = "Realtime process exited during startup. See log for details."
    return LaunchResult(False, process.pid, error, log_path)
