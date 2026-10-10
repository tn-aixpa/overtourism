"""Data files of the Fazzon model: location, download from the platform, startup check.

The Fazzon model reads the presence statistics produced by the Fazzon data-preparation
pipeline (``data_preparation/model/fazzon``, logged to the DigitalHub platform as
dataitems with ``--upload``).  They live in the folder returned by
:func:`fazzon_data_dir`:

* ``FAZZON_DATA_DIR`` environment variable, if set;
* otherwise ``<executor-backend>/database/fazzon``.

As for the indicator data of the CRUD API (``download_index_data``), the same
environment variables select where the data comes from:

* ``DT_OVERTURISM_STANDALONE_MODE`` (default ``true``): in standalone mode nothing is
  downloaded and the files must already be in the folder;
* ``PROJECT_NAME`` (default ``overtourism``): the platform project of the dataitems;
  platform credentials come from the environment or a ``dhcli login``.

Unlike the indicator data, any failure raises: a model without its data does not start.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR_ENV = "FAZZON_DATA_DIR"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "database" / "fazzon"
PRESENCE_STATS_FILE = "fazzon_presence_stats.parquet"
DATA_FILES: tuple[str, ...] = (PRESENCE_STATS_FILE,)


def fazzon_data_dir() -> Path:
    """Folder of the Fazzon data files (``FAZZON_DATA_DIR`` or the default)."""
    return Path(os.environ.get(DATA_DIR_ENV) or DEFAULT_DATA_DIR)


def standalone_mode() -> bool:
    return os.getenv("DT_OVERTURISM_STANDALONE_MODE", "true").lower() == "true"


def _missing_files(folder: Path) -> list[str]:
    return [name for name in DATA_FILES if not (folder / name).exists()]


def download_fazzon_data(overwrite: bool = False) -> Path:
    """Download the Fazzon dataitems from the platform into :func:`fazzon_data_dir`.

    Without *overwrite*, nothing is downloaded if all the files are already there.
    The files are downloaded to a temporary folder and checked (they must load as
    model statistics); only then they replace the current ones.  Any failure raises
    and leaves the current files untouched.
    """
    target = fazzon_data_dir()
    if not overwrite and not _missing_files(target):
        logger.info("Fazzon data already in %s: download skipped", target)
        return target

    import digitalhub as dh  # type: ignore[import-untyped]

    from .fazzon_presence_stats import load_presence_stats

    project_name = os.getenv("PROJECT_NAME", "overtourism")
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".fazzon_download_") as tmp:
        tmp_dir = Path(tmp)
        try:
            project = dh.get_project(project_name)
            for name in DATA_FILES:
                logger.info("Downloading dataitem %s from project %s", name, project_name)
                project.get_dataitem(name).download(str(tmp_dir), overwrite=True)
        except Exception as e:
            raise RuntimeError(
                f"Fazzon data download from platform project {project_name!r} failed: {e}"
            ) from e
        missing = _missing_files(tmp_dir)
        if missing:
            raise RuntimeError(f"Fazzon data download: files not received {missing}")
        load_presence_stats(tmp_dir)  # raises if the new files are not usable
        for name in DATA_FILES:
            shutil.move(str(tmp_dir / name), str(target / name))
    logger.info("Fazzon data downloaded to %s", target)
    return target


def ensure_fazzon_data() -> Path:
    """Startup step: download the data unless in standalone mode, then check it is there."""
    if not standalone_mode():
        download_fazzon_data()
    target = fazzon_data_dir()
    missing = _missing_files(target)
    if missing:
        raise FileNotFoundError(
            f"Fazzon data files missing in {target}: {missing} "
            f"(set {DATA_DIR_ENV}, or DT_OVERTURISM_STANDALONE_MODE=false to download them)"
        )
    return target
