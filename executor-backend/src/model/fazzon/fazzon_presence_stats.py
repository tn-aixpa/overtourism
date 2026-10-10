"""Presence statistics for the Fazzon (Lago dei Caprioli, Val di Sole) overtourism model.

Two presence variables are modelled:

``pv_visitors_car``
    Daily visitors arriving by private car.

``pv_visitors_other``
    Daily visitors arriving by non-car modes (shuttle bus, on foot, bicycle).

Both are conditioned on the season and on the weather class (good / cool / rainy, D7 in
``data_preparation/model/fazzon/ASSUMPTIONS.md``); there is no day-type split (D5).
The data has ``P(weather | season)`` (climatology, A20); the model uses one weather
distribution for all seasons, their weighted average (D13).

Data source
-----------
The table is the reference dataset ``fazzon_presence_stats`` produced by the Fazzon
data-preparation pipeline (``data_preparation/model/fazzon``): one row per (season,
weather) cell.  Its calibration and assumptions are listed in that folder's
``ASSUMPTIONS.md`` (A1: 2.83 persons per car; A3: non-car share 0.31; A4: non-car SD
50 % of the mean; A14: season weights; A19–A21: weather classes, frequencies and
effect).  It is read from the Fazzon data folder (see
:mod:`src.model.fazzon.fazzon_data`) every time :func:`presence_inputs` is called,
i.e. every time a model is built: there are no built-in values, and a missing or
malformed file is an error.
"""

# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

import pandas as pd
from scipy import stats

from .fazzon_data import PRESENCE_STATS_FILE, fazzon_data_dir

logger = logging.getLogger(__name__)

SEASONS: tuple[str, ...] = ("june", "july", "august", "september")
WEATHERS: tuple[str, ...] = ("good", "cool", "rainy")
PRESENCE_STATS_COLUMNS: tuple[str, ...] = (
    "season_freq_rel",
    "weather_freq_rel",
    "mean_car_visitors",
    "std_car_visitors",
    "mean_other_visitors",
    "std_other_visitors",
)


# ---------------------------------------------------------------------------
# Loading the reference dataset
# ---------------------------------------------------------------------------


def load_presence_stats(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Return the presence statistics read from *data_dir*, indexed by (season, weather).

    *data_dir* defaults to :func:`~src.model.fazzon.fazzon_data.fazzon_data_dir`.
    A missing file, columns or (season, weather) cells raise an error.
    """
    data_dir = Path(data_dir) if data_dir is not None else fazzon_data_dir()
    path = data_dir / PRESENCE_STATS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Fazzon presence statistics not found: {path}")

    df = pd.read_parquet(path)
    lacking = sorted({"season", "weather", *PRESENCE_STATS_COLUMNS} - set(df.columns))
    if lacking:
        raise ValueError(f"Fazzon presence statistics in {data_dir}: missing columns {lacking}")
    df = df.set_index(["season", "weather"])[list(PRESENCE_STATS_COLUMNS)]
    expected = {(s, w) for s in SEASONS for w in WEATHERS}
    if set(df.index) != expected:
        raise ValueError(
            f"Fazzon presence statistics: (season, weather) cells {sorted(df.index)}, "
            f"expected {sorted(expected)}"
        )
    logger.info("Fazzon presence statistics loaded from %s", data_dir)
    order = pd.MultiIndex.from_tuples(
        [(s, w) for s in SEASONS for w in WEATHERS], names=["season", "weather"]
    )
    return df.loc[order]


# ---------------------------------------------------------------------------
# Conditional distribution factories
# ---------------------------------------------------------------------------


def make_presence_factories(stats_table: pd.DataFrame):
    """Return ``(car_visitors_stats, other_visitors_stats)`` closures for the given stats.

    Both factories take the season and the weather class and return a **Gamma**
    distribution with the cell's mean μ and variance σ²::

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

    def car_visitors_stats(season: str, weather: str):
        mean = float(stats_table.loc[(season, weather), "mean_car_visitors"])
        var = float(stats_table.loc[(season, weather), "std_car_visitors"]) ** 2
        return _gamma(mean, var)

    def other_visitors_stats(season: str, weather: str):
        mean = float(stats_table.loc[(season, weather), "mean_other_visitors"])
        var = float(stats_table.loc[(season, weather), "std_other_visitors"]) ** 2
        return _gamma(mean, var)

    return car_visitors_stats, other_visitors_stats


def presence_inputs(data_dir: str | Path | None = None):
    """Return ``(season_weights, weather_weights, car_visitors_stats, other_visitors_stats)``.

    * ``season_weights``: ``{season: P(season)}`` (A14);
    * ``weather_weights``: ``{weather: sum_s P(s) P(weather | s)}`` (A20, D13): the weather
      category does not depend on the season in the model, because the modelling library
      cannot fix the season (or the weather) of a season-dependent category in a scenario.

    Reads the reference dataset (see :func:`load_presence_stats`); called by
    ``FazzonModel.default_inputs`` each time a model is built, so a model built after
    a data download uses the new statistics.
    """
    table = load_presence_stats(data_dir)
    season_weights = {
        s: float(table.loc[(s, WEATHERS[0]), "season_freq_rel"]) for s in SEASONS
    }
    weather_weights = {
        w: sum(season_weights[s] * float(table.loc[(s, w), "weather_freq_rel"]) for s in SEASONS)
        for w in WEATHERS
    }
    total = sum(weather_weights.values())
    weather_weights = {w: p / total for w, p in weather_weights.items()}
    car_stats, other_stats = make_presence_factories(table)
    return season_weights, weather_weights, car_stats, other_stats
