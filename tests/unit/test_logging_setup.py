import logging
import re

import pytest

from face_voice_rec import config, logging_setup


@pytest.fixture
def root_logger(monkeypatch, tmp_path):
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    level_before = root.level
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
    yield root
    for handler in list(root.handlers):
        if handler not in handlers_before:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level_before)


def test_setup_logging_is_idempotent(root_logger):
    before = len(root_logger.handlers)

    logging_setup.setup_logging()
    after_first = len(root_logger.handlers)
    logging_setup.setup_logging()

    assert after_first == before + 3
    assert len(root_logger.handlers) == after_first


def test_app_log_gets_info_and_errors_log_gets_only_errors(root_logger, tmp_path):
    logging_setup.setup_logging()
    log = logging.getLogger("face_voice_rec.logging_demo")

    log.info("info line")
    log.warning("warning line")
    log.error("error line")

    app_text = (tmp_path / "logs" / "app.log").read_text(encoding="utf-8")
    errors_text = (tmp_path / "logs" / "errors.log").read_text(encoding="utf-8")
    assert "info line" in app_text
    assert "warning line" in app_text
    assert "error line" in app_text
    assert "error line" in errors_text
    assert "info line" not in errors_text
    assert "warning line" not in errors_text


def test_log_line_shows_when_level_and_where(root_logger, tmp_path):
    logging_setup.setup_logging()

    logging.getLogger("face_voice_rec.logging_demo").info("hello")

    last_line = (tmp_path / "logs" / "app.log").read_text(encoding="utf-8").splitlines()[-1]
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} INFO "
        r"test_logging_setup\.test_log_line_shows_when_level_and_where:\d+ \| hello",
        last_line,
    )
