# SPDX-License-Identifier: Apache-2.0

import logging
import re

from src.api.logging_config import (
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    configure_logging,
)


def test_logging_format_matches_requested_pattern() -> None:
    log_record = logging.LogRecord(
        name="src.api.main",
        level=logging.INFO,
        pathname="main.py",
        lineno=42,
        msg="Layer 3 started",
        args=(),
        exc_info=None,
        func="create_app",
    )
    log_record.threadName = "api-worker"
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    formatted_message = formatter.format(log_record)

    assert re.fullmatch(
        r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}\.\d{3} "
        r"\[api-worker\] INFO  src\.api\.main\.create_app - Layer 3 started",
        formatted_message,
    )


def test_configure_logging_applies_format_to_uvicorn_handlers(monkeypatch) -> None:
    logger_names = ("uvicorn", "uvicorn.error", "uvicorn.access")
    loggers = {name: logging.getLogger(name) for name in logger_names}
    root_logger = logging.getLogger()
    for logger in loggers.values():
        monkeypatch.setattr(logger, "handlers", [logging.StreamHandler()])

    basic_config: dict[str, object] = {}
    monkeypatch.setattr(
        logging,
        "basicConfig",
        lambda **kwargs: basic_config.update(kwargs),
    )

    def get_test_logger(name: str | None = None) -> logging.Logger:
        return root_logger if name is None else loggers[name]

    monkeypatch.setattr(logging, "getLogger", get_test_logger)

    configure_logging()

    assert basic_config["level"] == logging.INFO
    assert basic_config["format"] == LOG_FORMAT
    assert basic_config["datefmt"] == LOG_DATE_FORMAT
    for logger in loggers.values():
        assert all(handler.formatter._fmt == LOG_FORMAT for handler in logger.handlers)
        assert all(
            handler.formatter.datefmt == LOG_DATE_FORMAT for handler in logger.handlers
        )
