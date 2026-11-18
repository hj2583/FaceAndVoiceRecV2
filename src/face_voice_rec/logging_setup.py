"""Application logging: a general log, an errors-only log, and the console."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from face_voice_rec import config

_FORMAT = "%(asctime)s %(levelname)s %(module)s.%(funcName)s:%(lineno)d | %(message)s"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 5
_HANDLER_FLAG = "_face_voice_rec_handler"


def _file_handler(path: Path, level: int) -> logging.Handler:
    handler = RotatingFileHandler(
        path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
        delay=True,
    )
    handler.setLevel(level)
    return handler


def _console_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    return handler


def setup_logging() -> None:
    root = logging.getLogger()
    # Streamlit reruns the script on every interaction.
    if any(getattr(handler, _HANDLER_FLAG, False) for handler in root.handlers):
        return

    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT)
    for handler in (
        _file_handler(log_dir / "app.log", logging.INFO),
        _file_handler(log_dir / "errors.log", logging.ERROR),
        _console_handler(),
    ):
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_FLAG, True)
        root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
