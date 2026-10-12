"""Data files of the Fazzon model: location, download from the platform, startup check.

The Fazzon model reads the presence statistics produced by the Fazzon data-preparation
pipeline (``data_preparation/model/fazzon``, logged to the DigitalHub platform as
dataitems with ``--upload``), from ``FAZZON_DATA_DIR`` or ``<executor-backend>/database/fazzon``.
Download and startup rules: :mod:`src.model.common.model_data`.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from src.model.common.model_data import ModelData, standalone_mode

__all__ = [
    "DATA_DIR_ENV",
    "DATA_FILES",
    "DEFAULT_DATA_DIR",
    "MODAL_SHIFT_FILE",
    "PRESENCE_STATS_FILE",
    "download_fazzon_data",
    "ensure_fazzon_data",
    "fazzon_data_dir",
    "standalone_mode",
]

DATA_DIR_ENV = "FAZZON_DATA_DIR"
PRESENCE_STATS_FILE = "fazzon_presence_stats.parquet"
MODAL_SHIFT_FILE = "fazzon_modal_shift.parquet"
DATA_FILES: tuple[str, ...] = (PRESENCE_STATS_FILE, MODAL_SHIFT_FILE)


def _validate(folder: Path) -> None:
    from .fazzon_presence_stats import behaviour_params, load_presence_stats

    load_presence_stats(folder)
    behaviour_params(folder)


FAZZON_DATA = ModelData(
    model="fazzon", files=DATA_FILES, env_var=DATA_DIR_ENV, validate=_validate
)
DEFAULT_DATA_DIR = FAZZON_DATA.default_dir


def fazzon_data_dir() -> Path:
    """Folder of the Fazzon data files (``FAZZON_DATA_DIR`` or the default)."""
    return FAZZON_DATA.data_dir()


def download_fazzon_data(overwrite: bool = False) -> Path:
    """Download the Fazzon dataitems from the platform (see :meth:`ModelData.download`)."""
    return FAZZON_DATA.download(overwrite=overwrite)


def ensure_fazzon_data() -> Path:
    """Startup step (see :meth:`ModelData.ensure`)."""
    return FAZZON_DATA.ensure()
