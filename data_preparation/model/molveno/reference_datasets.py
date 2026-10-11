# SPDX-License-Identifier: Apache-2.0
"""
STEP 2 - Molveno reference datasets.

Input : <normalized>/molveno/ tables (step 1)
Output: <final>/molveno/
  - molveno_daily: one row per day, presences, season and weekday deviation of the reference days
  - molveno_season_stats: per season, its weight and the mean / SD of tourists and excursionists
  - molveno_weekday_stats: per weekday, mean / SD of the deviation from the weekly trend

Every choice is an entry of ASSUMPTIONS.md (D*).
"""

import logging

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

REFERENCE_PERIOD = (pd.Timestamp("2023-07-01"), pd.Timestamp("2023-09-30"))  # D1
TOURIST_DATE_SHIFT = 1  # D2: tourists of a day = Vodafone tourists dated the following day
SEASONS = ("low", "mid", "high", "very high")  # D3: by increasing mean tourists
KMEANS_SEED = 0  # D3
TREND_WINDOW = 7  # D4: centred moving average, days
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
PRESENCES = ("tourists", "excursionists")


def build_molveno_daily(vodafone: pd.DataFrame) -> pd.DataFrame:
    """Presences per day (D2 applied) with the weekly trend and the deviation from it (D4)."""
    daily = vodafone.set_index("date").asfreq("D")
    daily["tourists"] = daily["tourists"].shift(-TOURIST_DATE_SHIFT)
    trend = daily[list(PRESENCES)].rolling(TREND_WINDOW, center=True).mean()
    for p in PRESENCES:
        daily[f"{p}_trend"] = trend[p]
        daily[f"{p}_deviation"] = daily[p] - trend[p]
    daily["weekday"] = [WEEKDAYS[d] for d in daily.index.dayofweek]
    daily["reference"] = (daily.index >= REFERENCE_PERIOD[0]) & (daily.index <= REFERENCE_PERIOD[1])
    return daily.reset_index()


def _reference_days(daily: pd.DataFrame) -> pd.DataFrame:
    ref = daily[daily["reference"]]
    expected = (REFERENCE_PERIOD[1] - REFERENCE_PERIOD[0]).days + 1
    cols = [*PRESENCES, *(f"{p}_deviation" for p in PRESENCES)]
    if len(ref) != expected or ref[cols].isna().any().any():
        raise ValueError(f"Reference period {REFERENCE_PERIOD}: missing days or presences")
    return ref


def assign_seasons(daily: pd.DataFrame) -> pd.Series:
    """D3: k-means with 4 clusters on the standardized tourists and excursionists of the
    reference days; clusters named by increasing mean tourists."""
    ref = _reference_days(daily)
    x = StandardScaler().fit_transform(ref[list(PRESENCES)])
    labels = KMeans(len(SEASONS), n_init=10, random_state=KMEANS_SEED).fit_predict(x)
    order = ref.groupby(labels)["tourists"].mean().sort_values().index
    names = dict(zip(order, SEASONS))
    return pd.Series([names[k] for k in labels], index=ref.index, name="season")


def _mean_std(df: pd.DataFrame, by: str, suffix: str = "") -> pd.DataFrame:
    g = df.groupby(by)
    out = pd.DataFrame({f"{stat}_{p}": getattr(g[f"{p}{suffix}"], stat)() for p in PRESENCES for stat in ("mean", "std")})
    return out[[f"{stat}_{p}" for p in PRESENCES for stat in ("mean", "std")]]


def build_season_stats(daily: pd.DataFrame) -> pd.DataFrame:
    ref = daily[daily["season"].notna()]
    out = _mean_std(ref, "season")
    out.insert(0, "freq_rel", ref["season"].value_counts(normalize=True))
    out.insert(1, "days", ref["season"].value_counts())
    return out.loc[list(SEASONS)].rename_axis("season").reset_index()


def build_weekday_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """D4: per weekday, mean and SD of the deviation from the centred weekly moving average."""
    out = _mean_std(_reference_days(daily), "weekday", suffix="_deviation")
    return out.loc[list(WEEKDAYS)].rename_axis("weekday").reset_index()


def build_all(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    daily = build_molveno_daily(tables["vodafone_presences"])
    daily["season"] = assign_seasons(daily)
    weather = tables["meteo_bulletin"].set_index("date")["weather_desc"]
    daily["weather_desc"] = daily["date"].map(weather)
    seasons = build_season_stats(daily)
    logger.info("Seasons (days): %s", dict(zip(seasons["season"], seasons["days"])))
    return {
        "molveno_daily": daily,
        "molveno_season_stats": seasons,
        "molveno_weekday_stats": build_weekday_stats(daily),
    }

