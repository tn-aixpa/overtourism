# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pandas as pd


def get_dataframe(name: str, type: str = "json") -> pd.DataFrame:
    path = Path("..") / "testdata" / "overtourism-data" / name
    match type:
        case "json":
            path = path.with_suffix(".json")
            return pd.read_json(path)
        case "csv":
            path = path.with_suffix(".csv")
            return pd.read_csv(path)
        case "parquet":
            path = path.with_suffix(".parquet")
            return pd.read_parquet(path)
        case _:
            raise NotImplementedError(f"Unsupported type: {type}")


season_stats = get_dataframe("season_stats", "parquet")

weather_stats = pd.DataFrame(
    [
        [0.15, 140.30952381, 463.13601314, -773.23809524, 1187.05786413],
        [0.20, 128.32142857, 319.20470369, 31.10714286, 824.51775728],
        [0.65, 72.86263736, 406.09163233, 282.91208791, 839.91419686],
    ],
    columns=[
        "freq_rel",
        "mean_tourists",
        "std_tourists",
        "mean_excursionists",
        "std_excursionists",
    ],
    index=["bad", "unsettled", "good"],
)

weekday_stats = pd.DataFrame(
    [
        [-362.65934066, 112.47000414, -391.15384615, 791.84158997],
        [-265.34065934, 122.13648742, -352.27472527, 937.65181712],
        [-147.92307692, 247.5534754, -465.62637363, 733.9652083],
        [92.23076923, 233.81638923, -380.35164835, 779.58869239],
        [678.05494505, 149.44551186, -232.91208791, 500.50732948],
        [320.46938776, 204.41996226, 590.98979592, 797.28902804],
        [-278.93406593, 193.01797188, 1240.27472527, 1149.59709071],
    ],
    columns=[
        "mean_tourists",
        "std_tourists",
        "mean_excursionists",
        "std_excursionists",
    ],
    index=[
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ],
)

season = {s: season_stats.loc[s, "freq_rel"] for s in season_stats.index}
weather = {w: weather_stats.loc[w, "freq_rel"] for w in weather_stats.index}
weekday = weekday_stats.index


def tourist_presences_stats(weekday, season, weather):
    # Season
    mean = season_stats.loc[season, "mean_tourists"]
    std2 = season_stats.loc[season, "std_tourists"] ** 2
    # Weather
    mean += weather_stats.loc[weather, "mean_tourists"]
    std2 += weather_stats.loc[weather, "std_tourists"] ** 2
    # Weekdays
    mean += weekday_stats.loc[weekday, "mean_tourists"]
    std2 += weekday_stats.loc[weekday, "std_tourists"] ** 2
    # Finalize and return
    return {"mean": mean, "std": std2 ** (1 / 2)}


def excursionist_presences_stats(weekday, season, weather):
    # Season
    mean = season_stats.loc[season, "mean_excursionists"]
    std2 = season_stats.loc[season, "std_excursionists"] ** 2
    # Weather
    mean += weather_stats.loc[weather, "mean_excursionists"]
    std2 += weather_stats.loc[weather, "std_excursionists"] ** 2
    # Weekdays
    mean += weekday_stats.loc[weekday, "mean_excursionists"]
    std2 += weekday_stats.loc[weekday, "std_excursionists"] ** 2
    # Finalize and return
    return {"mean": mean, "std": std2 ** (1 / 2)}
