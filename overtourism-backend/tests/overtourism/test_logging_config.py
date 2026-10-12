# SPDX-License-Identifier: Apache-2.0

import logging
import re

from overtourism.backend.utils.logging_config import LOG_DATE_FORMAT, LOG_FORMAT


def test_logging_format_matches_requested_pattern() -> None:
    log_record = logging.LogRecord(
        name="overtourism.backend.main",
        level=logging.INFO,
        pathname="main.py",
        lineno=42,
        msg="Backend started",
        args=(),
        exc_info=None,
        func="create_app",
    )
    log_record.threadName = "api-worker"
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    formatted_message = formatter.format(log_record)

    assert re.fullmatch(
        r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}\.\d{3} "
        r"\[api-worker\] INFO  overtourism\.backend\.main\.create_app - Backend started",
        formatted_message,
    )
