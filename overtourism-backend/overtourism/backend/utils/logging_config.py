# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d [%(threadName)s] "
    "%(levelname)-5s %(name)s.%(funcName)s - %(message)s"
)
LOG_DATE_FORMAT = "%d-%m-%Y %H:%M:%S"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[logging.StreamHandler()],
    )

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        for handler in logging.getLogger(logger_name).handlers:
            handler.setFormatter(formatter)
