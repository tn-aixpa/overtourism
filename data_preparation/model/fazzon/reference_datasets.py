# SPDX-License-Identifier: Apache-2.0
"""
STEP 2 - Fazzon reference datasets.

Input : <normalized>/fazzon/ tables (step 1)
Output: <final>/fazzon/
  - fazzon_daily: one row per day, measured Fazzon data + access regime flags + predictors
  - fazzon_presence_season_stats, fazzon_presence_day_type_stats: the presence statistics of
    the Fazzon model (executor-backend/src/model/fazzon/fazzon_presence_stats.py), computed
    from the data instead of hard-coded

`fazzon_daily` holds measured values only; assumptions (occupancy, non-car share, ...) are
applied in the statistics, where they are documented.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ access regimes
# 2025: parking at 5 euro/day until full, no booking. Greenbus pilot 28/06-20/07 (1 euro,
#       wider loop through Fucine and Ossana).
# 2026: regulated access 04/07-31/08 (traffic light 9-19, booking, 20 euro/day, 10 euro after
#       12:00, 5 euro motorbikes, free after 17:00); Lakebus shuttle 04/07-30/08 every 30 min
#       from Pellizzano Parco Sama.
SHUTTLE_PERIODS = {
    "greenbus_pilot": (pd.Timestamp("2025-06-28"), pd.Timestamp("2025-07-20")),
    "lakebus": (pd.Timestamp("2026-07-04"), pd.Timestamp("2026-08-30")),
}
REGULATED_PARKING = (pd.Timestamp("2026-07-04"), pd.Timestamp("2026-08-31"))
SHUTTLE_NOT_RUNNING = pd.to_datetime(["2026-08-21"])  # bad weather all day, bus not active
# Events that make the car count not comparable ("5days": >2000 people, extra meadows used
# as parking, cars not counted).
EVENT_DAYS = pd.to_datetime(["2026-07-02", "2026-07-03"])

# ------------------------------------------------------------------ predictors
ROAD_UP_VALLEY_DIR = 2  # SS42 km increase downhill (Ossana -> Pellizzano -> Mezzana)
ROAD_MORNING_HOURS = range(7, 12)
ROAD_TOTAL_STATIONS = ("pellizzano", "pracorno", "celledizzo", "tonale", "revo")


def _road_predictors(hourly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    cars = daily.assign(cars=daily["cars"].where(daily["valid"]))
    by_dir = cars.pivot_table(index="date", columns=["station", "direction"], values="cars", dropna=False)
    out = pd.DataFrame(index=by_dir.index)
    for station in ROAD_TOTAL_STATIONS:
        out[f"strada_{station}_totale"] = by_dir[station].sum(axis=1, min_count=2)  # NaN if a direction is missing
    up = hourly[(hourly["station"] == "pellizzano") & (hourly["direction"] == ROAD_UP_VALLEY_DIR)
                & hourly["hour"].isin(ROAD_MORNING_HOURS)]
    morning_up = up.groupby("date")["cars"].sum()
    out["strada_pellizzano_salita_mattina"] = morning_up.where(by_dir[("pellizzano", ROAD_UP_VALLEY_DIR)].notna())
    return out


def _vodafone_predictors(vodafone: pd.DataFrame) -> pd.DataFrame:
    wide = vodafone.pivot(index="date", columns="area", values="presences")
    return wide.add_prefix("vf_pres_")


def _bike_predictors(bikes: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    near = set(stations.loc[(stations["kind"] == "bike") & (stations["role"] == "near_fazzon"), "code"])
    return pd.DataFrame({
        "bici_near_fazzon": bikes[bikes["sensor"].isin(near)].groupby("date")["bikes_peds"].sum(),
        "bici_valley_total": bikes.groupby("date")["bikes_peds"].sum(),
    })


def _tgc_predictors(tgc: pd.DataFrame) -> pd.DataFrame:
    tonale = tgc[tgc["service"].str.contains("Tonale", case=False, na=False)]
    return tonale.groupby("date")["entrants"].sum().rename("tgc_accessi_tonale").to_frame()


# ------------------------------------------------------------------ fazzon_daily
def _fazzon_counts(parking: pd.DataFrame, parking_2025: pd.DataFrame, greenbus: pd.DataFrame) -> pd.DataFrame:
    fz = parking.set_index("date")
    # the 2025-only sheet has days missing from the combined one (late September)
    old = parking_2025.set_index("date")["cars"]
    both = old.index.intersection(fz.index)
    mismatch = (fz.loc[both, "cars_full"] != old.loc[both]).sum()
    if mismatch:
        raise ValueError(f"{mismatch} days with different 2025 car counts in the two parking sheets")
    extra = old.index.difference(fz.index)
    fz = pd.concat([fz, pd.DataFrame({"cars_full": old.loc[extra]})]).sort_index()

    fz["cars"] = fz["cars_full"] + fz["cars_half"].fillna(0)
    valid_bus = greenbus[~greenbus["other_area"]].set_index("date")["tickets"]
    fz["greenbus_tickets"] = valid_bus
    return fz


def build_fazzon_daily(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per day from 2023-01-01 to the last day with any data."""
    fz = _fazzon_counts(tables["fazzon_parking"], tables["fazzon_parking_2025"], tables["fazzon_greenbus_2025"])
    meteo = tables["meteo_station"].set_index("date")
    bulletin = tables["meteo_bulletin"].set_index("date")
    parts = [
        fz[["cars_full", "cars_half", "cars", "moto_persons", "bus_up", "bus_down", "greenbus_tickets", "note"]],
        meteo[["tmin", "tmax", "rain_mm", "quality"]].rename(columns={"quality": "meteo_quality"}),
        bulletin.rename(columns={c: f"{c}_bollettino" for c in ("tmin", "tmax", "rain_mm")}),
        _road_predictors(tables["road_counts_hourly"], tables["road_counts_daily"]),
        _vodafone_predictors(tables["vodafone_presences"]),
        _bike_predictors(tables["bike_counts"], tables["stations"]),
        _tgc_predictors(tables["tgc_accessi"]),
    ]
    start = pd.Timestamp("2023-01-01")
    end = max(p.index.max() for p in parts)
    daily = pd.DataFrame(index=pd.date_range(start, end, freq="D", name="date"))
    for p in parts:
        daily = daily.join(p)

    d = daily.index.to_series()
    daily["shuttle"] = None
    for name, (first, last) in SHUTTLE_PERIODS.items():
        daily.loc[d.between(first, last) & ~d.isin(SHUTTLE_NOT_RUNNING), "shuttle"] = name
    daily["regulated_parking"] = d.between(*REGULATED_PARKING)
    daily["event_day"] = d.isin(EVENT_DAYS)
    daily["year"] = d.dt.year
    daily["month"] = d.dt.month
    daily["dow"] = d.dt.day_name().str.lower()
    daily["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    return daily.reset_index()


# ------------------------------------------------------------------ presence statistics
# Calibration of the current model (fazzon_presence_stats.py), reproduced from the data.
PERSONS_PER_CAR = 2.83  # NetMobility 2021 field study
NON_CAR_SHARE = 0.31  # 2022 EETRA survey; ASSUMPTION for the non-car visitors
OTHER_STD_REL = 0.50  # ASSUMPTION: relative SD of non-car visitors
CALIBRATION_YEAR = 2025
# 28/08/2025 (27 cars) is left out of the model calibration: the hard-coded statistics
# (August n=30, 81 days) are reproduced only without it. Reason not documented.
CALIBRATION_EXCLUDED_DAYS = pd.to_datetime(["2025-08-28"])
SEASONS = {6: "june", 7: "july", 8: "august", 9: "september"}
# Days counted as "peak" per season; the others are "base". June and September are an
# ASSUMPTION (conventional weekend, zero correction: too few days to calibrate).
DAY_TYPE_PEAK = {
    "june": ["saturday", "sunday"],
    "july": ["sunday"],
    "august": ["monday", "tuesday"],
    "september": ["saturday", "sunday"],
}
CALIBRATED_DAY_TYPE_SEASONS = ("july", "august")


def build_presence_stats(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Season and day-type statistics in the layout of fazzon_presence_stats.py.

    Season: mean and SD of daily cars x PERSONS_PER_CAR (the SD is the plain per-season SD).
    Non-car visitors = car visitors x NON_CAR_SHARE / (1 - NON_CAR_SHARE), SD = OTHER_STD_REL x mean.
    Day type: peak correction = mean(peak days) - season mean; base correction chosen so that the
    frequency-weighted mean correction is zero.
    """
    other_ratio = NON_CAR_SHARE / (1.0 - NON_CAR_SHARE)
    cal = daily[(daily["year"] == CALIBRATION_YEAR) & daily["cars"].notna()
                & ~daily["date"].isin(CALIBRATION_EXCLUDED_DAYS)].copy()
    cal["season"] = cal["month"].map(SEASONS)
    cal = cal.dropna(subset=["season"])

    season_rows, day_rows = [], []
    for season, g in cal.groupby("season", sort=False):
        mean_cars, sd_cars = g["cars"].mean(), g["cars"].std()
        mean_car = mean_cars * PERSONS_PER_CAR
        season_rows.append({
            "cluster_name": season,
            "freq_rel": len(g) / len(cal),
            "mean_car_visitors": mean_car,
            "std_car_visitors": sd_cars * PERSONS_PER_CAR,
            "mean_other_visitors": mean_car * other_ratio,
            "std_other_visitors": mean_car * other_ratio * OTHER_STD_REL,
            "n_days": len(g),
        })
        peak_days = DAY_TYPE_PEAK[season]
        f_peak = len(peak_days) / 7.0
        if season in CALIBRATED_DAY_TYPE_SEASONS:
            peak_corr = (g.loc[g["dow"].isin(peak_days), "cars"].mean() - mean_cars) * PERSONS_PER_CAR
        else:
            peak_corr = 0.0
        base_corr = -peak_corr * f_peak / (1.0 - f_peak)
        for day_type, freq, corr in (("peak", f_peak, peak_corr), ("base", 1.0 - f_peak, base_corr)):
            day_rows.append({
                "season": season, "day_type": day_type, "freq_rel": freq,
                "mean_car_visitors": corr, "mean_other_visitors": corr * other_ratio,
                "days": ",".join(peak_days if day_type == "peak" else
                                 [d for d in ("monday", "tuesday", "wednesday", "thursday", "friday",
                                              "saturday", "sunday") if d not in peak_days]),
            })

    order = [s for s in SEASONS.values() if s in set(cal["season"])]
    season_stats = pd.DataFrame(season_rows).set_index("cluster_name").loc[order].reset_index()
    day_type_stats = pd.DataFrame(day_rows)
    day_type_stats["season"] = pd.Categorical(day_type_stats["season"], order, ordered=True)
    day_type_stats = day_type_stats.sort_values(["season", "day_type"], ascending=[True, False])
    day_type_stats["season"] = day_type_stats["season"].astype(str)
    for df in (season_stats, day_type_stats):
        df.insert(0, "calibration", f"parking_{CALIBRATION_YEAR}")
    return season_stats, day_type_stats.reset_index(drop=True)


def build_all(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    daily = build_fazzon_daily(tables)
    season_stats, day_type_stats = build_presence_stats(daily)
    _check(daily, season_stats, day_type_stats)
    return {
        "fazzon_daily": daily,
        "fazzon_presence_season_stats": season_stats,
        "fazzon_presence_day_type_stats": day_type_stats,
    }


def _check(daily: pd.DataFrame, season_stats: pd.DataFrame, day_type_stats: pd.DataFrame) -> None:
    if daily["date"].duplicated().any():
        raise ValueError("fazzon_daily: duplicated dates")
    if not np.isclose(season_stats["freq_rel"].sum(), 1.0):
        raise ValueError("season freq_rel does not sum to 1")
    weighted = (day_type_stats["freq_rel"] * day_type_stats["mean_car_visitors"]).groupby(day_type_stats["season"]).sum()
    if not np.allclose(weighted, 0.0):
        raise ValueError(f"day-type corrections are not zero-mean: {weighted.to_dict()}")
    for year in (2025, 2026):
        n = daily.loc[daily["year"] == year, "cars"].notna().sum()
        logger.info("fazzon_daily: %d days with Fazzon car counts in %d", n, year)
