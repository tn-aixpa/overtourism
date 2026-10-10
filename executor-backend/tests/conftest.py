# SPDX-License-Identifier: Apache-2.0
"""Test data for the Fazzon model.

The Fazzon model has no built-in statistics: it reads them from the folder in
``FAZZON_DATA_DIR``.  The variable is set here, before any test module imports the
model, to a temporary folder holding the statistics in the pipeline's layout (2025
calibration values).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

SEASON_STATS = pd.DataFrame(
    {
        "cluster_name": ["june", "july", "august", "september"],
        "freq_rel": [3 / 82, 31 / 82, 31 / 82, 17 / 82],
        "mean_car_visitors": [600.0, 498.1, 767.4, 224.9],
        "std_car_visitors": [311.4, 153.4, 293.1, 118.9],
        "mean_other_visitors": [269.5, 223.8, 344.8, 101.0],
        "std_other_visitors": [134.8, 111.9, 172.4, 50.5],
    }
)


def write_fazzon_stats(folder: Path, season=SEASON_STATS) -> Path:
    """Write the Fazzon statistics file (pipeline layout) into *folder*."""
    folder.mkdir(parents=True, exist_ok=True)
    season.assign(calibration="parking_2025").to_parquet(
        folder / "fazzon_presence_season_stats.parquet"
    )
    return folder


FAZZON_TEST_DATA_DIR = write_fazzon_stats(Path(tempfile.mkdtemp(prefix="fazzon_test_data_")))
os.environ["FAZZON_DATA_DIR"] = str(FAZZON_TEST_DATA_DIR)
os.environ["DT_OVERTURISM_STANDALONE_MODE"] = "true"
