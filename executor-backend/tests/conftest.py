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

_SEASONS = ["june", "july", "august", "september"]
_WEATHERS = ["good", "cool", "rainy"]
_SEASON_FREQ = {"june": 3 / 82, "july": 31 / 82, "august": 31 / 82, "september": 17 / 82}
_WEATHER_FREQ = {"good": 0.62, "cool": 0.2, "rainy": 0.18}
_FACTOR = {"good": 1.0, "cool": 0.73, "rainy": 0.6}
_LEVEL = {"june": 678.1, "july": 611.7, "august": 897.3, "september": 258.3}  # good-day car visitors
_CV = {"june": 0.36, "july": 0.27, "august": 0.3, "september": 0.59}

PRESENCE_STATS = pd.DataFrame(
    [
        {
            "season": s,
            "weather": w,
            "season_freq_rel": _SEASON_FREQ[s],
            "weather_freq_rel": _WEATHER_FREQ[w],
            "mean_car_visitors": _LEVEL[s] * _FACTOR[w],
            "std_car_visitors": _LEVEL[s] * _FACTOR[w] * _CV[s],
            "mean_other_visitors": _LEVEL[s] * _FACTOR[w] * 0.449,
            "std_other_visitors": _LEVEL[s] * _FACTOR[w] * 0.449 * 0.5,
        }
        for s in _SEASONS
        for w in _WEATHERS
    ]
)


def write_fazzon_stats(folder: Path, stats=PRESENCE_STATS) -> Path:
    """Write the Fazzon presence statistics file (pipeline layout) into *folder*."""
    folder.mkdir(parents=True, exist_ok=True)
    stats.assign(calibration="parking_2025").to_parquet(folder / "fazzon_presence_stats.parquet")
    return folder


FAZZON_TEST_DATA_DIR = write_fazzon_stats(Path(tempfile.mkdtemp(prefix="fazzon_test_data_")))
os.environ["FAZZON_DATA_DIR"] = str(FAZZON_TEST_DATA_DIR)
os.environ["DT_OVERTURISM_STANDALONE_MODE"] = "true"
