"""Presence statistics for the Fazzon (Lago dei Caprioli, Val di Sole) overtourism model.

Two presence variables are modelled:

``pv_visitors_car``
    Daily visitors arriving by private car.

``pv_visitors_other``
    Daily visitors arriving by non-car modes (shuttle bus, on foot, bicycle).

Additive decomposition::

    mean_total = season_mean + day_type_correction(season, day_type)
    var_total  = season_var(season)

Unlike Portofino, there is no weather CV.

Data source
-----------
The season and day-type tables are the reference datasets
``fazzon_presence_season_stats`` and ``fazzon_presence_day_type_stats`` produced by
the Fazzon data-preparation pipeline (``data_preparation/model/fazzon``), which also
documents the calibration and its assumptions (2025 parking tickets × 2.83 persons per
car; non-car visitors from the 2022 EETRA modal share 0.31, ASSUMPTION).  They are
read from the Fazzon data folder (see :mod:`src.model.fazzon.fazzon_data`) every time
:func:`presence_inputs` is called, i.e. every time a model is built: there are no
built-in values, and a missing or malformed file is an error.
"""

# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

import pandas as pd
from scipy import stats

from .fazzon_data import DAY_TYPE_STATS_FILE, SEASON_STATS_FILE, fazzon_data_dir

logger = logging.getLogger(__name__)

SEASONS: tuple[str, ...] = ("june", "july", "august", "september")
DAY_TYPES: tuple[str, ...] = ("peak", "base")
SEASON_STATS_COLUMNS: tuple[str, ...] = (
    "freq_rel",
    "mean_car_visitors",
    "std_car_visitors",
    "mean_other_visitors",
    "std_other_visitors",
)
DAY_TYPE_STATS_COLUMNS: tuple[str, ...] = (
    "freq_rel",
    "mean_car_visitors",
    "mean_other_visitors",
)

# day_type weights: peak=2/7, base=5/7.
# The August bucket definition (peak=Mon+Tue, 2 days) gives the natural 2/7
# peak weight.  July has peak=1/7 but 2/7 is used as a cross-season central
# estimate consistent with the calendar share of a 2-day peak bucket.
# ASSUMPTION — adjust if season weights are required to vary across seasons.
day_type: dict[str, float] = {"peak": 2.0 / 7.0, "base": 5.0 / 7.0}  # ASSUMPTION


# ---------------------------------------------------------------------------
# Loading the reference datasets
# ---------------------------------------------------------------------------


def load_presence_stats(
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(season_stats, day_type_stats)`` read from *data_dir*.

    *data_dir* defaults to :func:`~src.model.fazzon.fazzon_data.fazzon_data_dir`.
    ``season_stats`` is indexed by season, ``day_type_stats`` by ``(season, day_type)``.
    Missing files, columns, seasons or day types raise an error.
    """
    data_dir = Path(data_dir) if data_dir is not None else fazzon_data_dir()
    files = [data_dir / SEASON_STATS_FILE, data_dir / DAY_TYPE_STATS_FILE]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Fazzon presence statistics not found: {missing}")

    season_df = pd.read_parquet(files[0])
    day_df = pd.read_parquet(files[1])
    for name, df, cols in (
        ("season", season_df, ["cluster_name", *SEASON_STATS_COLUMNS]),
        ("day-type", day_df, ["season", "day_type", *DAY_TYPE_STATS_COLUMNS]),
    ):
        lacking = sorted(set(cols) - set(df.columns))
        if lacking:
            raise ValueError(
                f"Fazzon {name} statistics in {data_dir}: missing columns {lacking}"
            )
    season_df = season_df.set_index("cluster_name")[list(SEASON_STATS_COLUMNS)]
    day_df = day_df.set_index(["season", "day_type"])[list(DAY_TYPE_STATS_COLUMNS)]

    if set(season_df.index) != set(SEASONS):
        raise ValueError(
            f"Fazzon season statistics: seasons {sorted(season_df.index)}, "
            f"expected {sorted(SEASONS)}"
        )
    expected_cells = {(s, d) for s in SEASONS for d in DAY_TYPES}
    if set(day_df.index) != expected_cells:
        raise ValueError(
            "Fazzon day-type statistics: (season, day_type) cells differ from the expected ones"
        )
    logger.info("Fazzon presence statistics loaded from %s", data_dir)
    order = pd.MultiIndex.from_tuples(
        [(s, d) for s in SEASONS for d in DAY_TYPES], names=["season", "day_type"]
    )
    return season_df.loc[list(SEASONS)], day_df.loc[order]


# ---------------------------------------------------------------------------
# Conditional distribution factories
# ---------------------------------------------------------------------------


def make_presence_factories(
    yr_season_stats: pd.DataFrame,
    yr_day_type_stats: pd.DataFrame,
):
    """Return ``(car_visitors_stats, other_visitors_stats)`` closures for the given stats.

    Both factories use the supplied season and day-type DataFrames for their
    cell means, and the per-season variance stored in *yr_season_stats*.

    There is no weather CV for Fazzon.  The variance decomposition is therefore
    simpler than Portofino::

        var_total = season_var(season)

    The day-type correction shifts only the **mean**; its variance is already
    captured inside the per-season variance.

    The distributions are **Gamma**, parameterised by the combined mean μ and
    variance σ²::

        k = μ² / σ²   (shape)
        θ = σ² / μ    (scale)

    Gamma is naturally supported on (0, ∞) with a smooth density at 0.  The
    mean is floored at 1.0 before computing Gamma parameters to guard against
    negative cell means (e.g. a small-n June base-day combination).
    """

    def _gamma(mean: float, var: float):
        mean = max(mean, 1.0)  # guard against rare low-season base-day negatives
        k = mean**2 / var
        theta = var / mean
        return stats.gamma(a=k, scale=theta)

    def car_visitors_stats(day_type: str, season: str):
        mean = float(yr_season_stats.loc[season, "mean_car_visitors"])
        var = float(yr_season_stats.loc[season, "std_car_visitors"]) ** 2
        mean += float(yr_day_type_stats.loc[(season, day_type), "mean_car_visitors"])
        return _gamma(mean, var)

    def other_visitors_stats(day_type: str, season: str):
        mean = float(yr_season_stats.loc[season, "mean_other_visitors"])
        var = float(yr_season_stats.loc[season, "std_other_visitors"]) ** 2
        mean += float(yr_day_type_stats.loc[(season, day_type), "mean_other_visitors"])
        return _gamma(mean, var)

    return car_visitors_stats, other_visitors_stats


def presence_inputs(data_dir: str | Path | None = None):
    """Return ``(season_weights, day_type_weights, car_visitors_stats, other_visitors_stats)``.

    Reads the reference datasets (see :func:`load_presence_stats`); called by
    ``FazzonModel.default_inputs`` each time a model is built, so a model built after
    a data download uses the new statistics.
    """
    season_stats, day_type_stats = load_presence_stats(data_dir)
    season_weights = {s: float(season_stats.loc[s, "freq_rel"]) for s in season_stats.index}
    car_stats, other_stats = make_presence_factories(season_stats, day_type_stats)
    return season_weights, dict(day_type), car_stats, other_stats
