"""Presence statistics for the Fazzon (Lago dei Caprioli, Val di Sole) overtourism model.

Two presence variables are modelled:

``pv_visitors_car``
    Daily visitors arriving by private car.

``pv_visitors_other``
    Daily visitors arriving by non-car modes (shuttle bus, on foot, bicycle).

Both are conditioned on the season only (no day-type split: decision D5 in
``data_preparation/model/fazzon/ASSUMPTIONS.md``; no weather CV yet).

Data source
-----------
The season table is the reference dataset ``fazzon_presence_season_stats`` produced
by the Fazzon data-preparation pipeline (``data_preparation/model/fazzon``); its
calibration and assumptions are listed in that folder's ``ASSUMPTIONS.md`` (A1: 2.83
persons per car; A3: non-car share 0.31; A4: non-car SD 50 % of the mean).  It is
read from the Fazzon data folder (see :mod:`src.model.fazzon.fazzon_data`) every time
:func:`presence_inputs` is called, i.e. every time a model is built: there are no
built-in values, and a missing or malformed file is an error.
"""

# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

import pandas as pd
from scipy import stats

from .fazzon_data import SEASON_STATS_FILE, fazzon_data_dir

logger = logging.getLogger(__name__)

SEASONS: tuple[str, ...] = ("june", "july", "august", "september")
SEASON_STATS_COLUMNS: tuple[str, ...] = (
    "freq_rel",
    "mean_car_visitors",
    "std_car_visitors",
    "mean_other_visitors",
    "std_other_visitors",
)


# ---------------------------------------------------------------------------
# Loading the reference dataset
# ---------------------------------------------------------------------------


def load_presence_stats(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Return the season statistics read from *data_dir*, indexed by season.

    *data_dir* defaults to :func:`~src.model.fazzon.fazzon_data.fazzon_data_dir`.
    A missing file, columns or seasons raise an error.
    """
    data_dir = Path(data_dir) if data_dir is not None else fazzon_data_dir()
    path = data_dir / SEASON_STATS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Fazzon presence statistics not found: {path}")

    df = pd.read_parquet(path)
    lacking = sorted({"cluster_name", *SEASON_STATS_COLUMNS} - set(df.columns))
    if lacking:
        raise ValueError(f"Fazzon season statistics in {data_dir}: missing columns {lacking}")
    df = df.set_index("cluster_name")[list(SEASON_STATS_COLUMNS)]
    if set(df.index) != set(SEASONS):
        raise ValueError(
            f"Fazzon season statistics: seasons {sorted(df.index)}, expected {sorted(SEASONS)}"
        )
    logger.info("Fazzon presence statistics loaded from %s", data_dir)
    return df.loc[list(SEASONS)]


# ---------------------------------------------------------------------------
# Conditional distribution factories
# ---------------------------------------------------------------------------


def make_presence_factories(yr_season_stats: pd.DataFrame):
    """Return ``(car_visitors_stats, other_visitors_stats)`` closures for the given stats.

    Both factories take the season and return a **Gamma** distribution with the
    season's mean μ and variance σ²::

        k = μ² / σ²   (shape)
        θ = σ² / μ    (scale)

    Gamma is naturally supported on (0, ∞) with a smooth density at 0.  The mean is
    floored at 1.0 before computing the Gamma parameters.
    """

    def _gamma(mean: float, var: float):
        mean = max(mean, 1.0)
        k = mean**2 / var
        theta = var / mean
        return stats.gamma(a=k, scale=theta)

    def car_visitors_stats(season: str):
        mean = float(yr_season_stats.loc[season, "mean_car_visitors"])
        var = float(yr_season_stats.loc[season, "std_car_visitors"]) ** 2
        return _gamma(mean, var)

    def other_visitors_stats(season: str):
        mean = float(yr_season_stats.loc[season, "mean_other_visitors"])
        var = float(yr_season_stats.loc[season, "std_other_visitors"]) ** 2
        return _gamma(mean, var)

    return car_visitors_stats, other_visitors_stats


def presence_inputs(data_dir: str | Path | None = None):
    """Return ``(season_weights, car_visitors_stats, other_visitors_stats)``.

    Reads the reference dataset (see :func:`load_presence_stats`); called by
    ``FazzonModel.default_inputs`` each time a model is built, so a model built after
    a data download uses the new statistics.
    """
    season_stats = load_presence_stats(data_dir)
    season_weights = {s: float(season_stats.loc[s, "freq_rel"]) for s in season_stats.index}
    car_stats, other_stats = make_presence_factories(season_stats)
    return season_weights, car_stats, other_stats
