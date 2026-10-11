import numpy as np
import pandas as pd

from model.molveno.config import load_config
from model.molveno.reference_datasets import (
    REFERENCE_PERIOD,
    SEASONS,
    WEEKDAYS,
    assign_seasons,
    build_molveno_daily,
    build_season_stats,
    build_weekday_stats,
)
from model.molveno.standardize import standardize_vodafone


def test_config_lists_all_sources():
    assert {"vodafone_attendences", "meteo_bulletin"} <= set(load_config().sources)


def test_standardize_vodafone_keeps_molveno_tourists_and_visitors(tmp_path):
    rows = [
        ("2023-07-01", "TN_MKT_AL_3", "27", "ITA", "TOURIST", 100),
        ("2023-07-01", "TN_MKT_AL_3", "27", "DEU", "TOURIST", 20),
        ("2023-07-01", "TN_MKT_AL_3", "27", "ITA", "VISITOR", 50),
        ("2023-07-01", "TN_MKT_AL_3", "27", "ITA", "INHABITANT", 900),
        ("2023-07-01", "TN_MKT_AL_3", "8", "ITA", "TOURIST", 999),
        ("2023-07-01", "AL_2", "27", "ITA", "TOURIST", 999),
    ]
    path = tmp_path / "vodafone.csv"
    pd.DataFrame(rows, columns=["date", "locType", "locId", "userCountry", "userProfile", "value"]).to_csv(path, index=False)
    out = standardize_vodafone(path)
    assert out.to_dict("records") == [{"date": pd.Timestamp("2023-07-01"), "tourists": 120, "excursionists": 50}]


def _vodafone(seed=0):
    """Synthetic presences around the reference period: 4 demand levels plus a weekly pattern."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(REFERENCE_PERIOD[0] - pd.Timedelta(days=10), REFERENCE_PERIOD[1] + pd.Timedelta(days=10))
    level = rng.choice([1000.0, 2000.0, 3000.0, 4000.0], size=len(dates))
    weekly = np.where(dates.dayofweek >= 5, 500.0, 0.0)
    return pd.DataFrame({
        "date": dates,
        "tourists": level + rng.normal(0, 50, len(dates)),
        "excursionists": level * 1.2 + weekly + rng.normal(0, 50, len(dates)),
    })


def test_tourists_are_taken_from_the_following_day():
    vodafone = _vodafone()
    daily = build_molveno_daily(vodafone).set_index("date")
    day = REFERENCE_PERIOD[0]
    assert daily.loc[day, "tourists"] == vodafone.set_index("date").loc[day + pd.Timedelta(days=1), "tourists"]


def test_seasons_are_ordered_by_tourists_and_weights_sum_to_one():
    daily = build_molveno_daily(_vodafone())
    daily["season"] = assign_seasons(daily)
    stats = build_season_stats(daily)
    assert tuple(stats["season"]) == SEASONS
    assert stats["mean_tourists"].is_monotonic_increasing
    assert np.isclose(stats["freq_rel"].sum(), 1.0)
    assert stats["days"].sum() == (REFERENCE_PERIOD[1] - REFERENCE_PERIOD[0]).days + 1


def test_weekday_deviation_shows_the_weekend():
    stats = build_weekday_stats(build_molveno_daily(_vodafone())).set_index("weekday")
    assert tuple(stats.index) == WEEKDAYS
    assert stats.loc["sunday", "mean_excursionists"] > stats.loc["wednesday", "mean_excursionists"] + 300
