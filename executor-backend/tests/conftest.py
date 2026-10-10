# SPDX-License-Identifier: Apache-2.0
"""Test data for the Fazzon model.

The Fazzon model has no built-in statistics: it reads them from the folder in
``FAZZON_DATA_DIR``.  The variable is set here, before any test module imports the
model, to a temporary folder holding statistics in the pipeline's layout (2025
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
        "freq_rel": [3 / 81, 31 / 81, 30 / 81, 17 / 81],
        "mean_car_visitors": [600.0, 498.1, 790.4, 224.9],
        "std_car_visitors": [311.4, 153.4, 268.0, 118.9],
        "mean_other_visitors": [269.5, 223.8, 355.1, 101.0],
        "std_other_visitors": [134.8, 111.9, 177.6, 50.5],
    }
)
DAY_TYPE_STATS = pd.DataFrame(
    {
        "season": ["june", "june", "july", "july", "august", "august", "september", "september"],
        "day_type": ["peak", "base"] * 4,
        "freq_rel": [2 / 7, 5 / 7, 1 / 7, 6 / 7, 2 / 7, 5 / 7, 2 / 7, 5 / 7],
        "mean_car_visitors": [0.0, 0.0, 157.1, -26.2, 112.0, -44.8, 0.0, 0.0],
        "mean_other_visitors": [0.0, 0.0, 70.6, -11.8, 50.3, -20.1, 0.0, 0.0],
    }
)


def write_fazzon_stats(folder: Path, season=SEASON_STATS, day_type=DAY_TYPE_STATS) -> Path:
    """Write the two Fazzon statistics files (pipeline layout) into *folder*."""
    folder.mkdir(parents=True, exist_ok=True)
    season.assign(calibration="parking_2025").to_parquet(
        folder / "fazzon_presence_season_stats.parquet"
    )
    day_type.assign(calibration="parking_2025").to_parquet(
        folder / "fazzon_presence_day_type_stats.parquet"
    )
    return folder


FAZZON_TEST_DATA_DIR = write_fazzon_stats(Path(tempfile.mkdtemp(prefix="fazzon_test_data_")))
os.environ["FAZZON_DATA_DIR"] = str(FAZZON_TEST_DATA_DIR)
os.environ["DT_OVERTURISM_STANDALONE_MODE"] = "true"
