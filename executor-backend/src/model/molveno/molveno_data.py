"""Data files of the Molveno model: location, download from the platform, startup check.

The Molveno model reads the presence statistics produced by the Molveno data-preparation
pipeline (``data_preparation/model/molveno``, logged to the DigitalHub platform as
dataitems with ``--upload``), from ``MOLVENO_DATA_DIR`` or ``<executor-backend>/database/molveno``.
Download and startup rules: :mod:`src.model.common.model_data`.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from src.model.common.model_data import ModelData

DATA_DIR_ENV = "MOLVENO_DATA_DIR"
PRESENCE_STATS_FILE = "molveno_presence_stats.parquet"
MONTHLY_STATS_FILE = "molveno_presence_stats_monthly.parquet"
WEATHER_FREQ_FILE = "molveno_weather_freq.parquet"
DATA_FILES: tuple[str, ...] = (PRESENCE_STATS_FILE, MONTHLY_STATS_FILE, WEATHER_FREQ_FILE)


def _validate(folder: Path) -> None:
    from .molveno_presence_stats import load_tables

    load_tables(folder)


MOLVENO_DATA = ModelData(model="molveno", files=DATA_FILES, env_var=DATA_DIR_ENV, validate=_validate)


def molveno_data_dir() -> Path:
    """Folder of the Molveno data files (``MOLVENO_DATA_DIR`` or the default)."""
    return MOLVENO_DATA.data_dir()


def download_molveno_data(overwrite: bool = False) -> Path:
    """Download the Molveno dataitems from the platform (see :meth:`ModelData.download`)."""
    return MOLVENO_DATA.download(overwrite=overwrite)


def ensure_molveno_data() -> Path:
    """Startup step (see :meth:`ModelData.ensure`)."""
    return MOLVENO_DATA.ensure()
