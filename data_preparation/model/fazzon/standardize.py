# SPDX-License-Identifier: Apache-2.0
"""
STEP 1 - Standardization of the Fazzon sources.

Input : raw files (step 0)
Output: <normalized>/fazzon/<table>.parquet, one clean table per source

Only cleaning and documented data-quality rules here (types, names, units, invalid
readings flagged); no modelling choices, which belong to step 2.
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from model.fazzon.config import STATIONS_FILE

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ Fazzon parking / bus
# 2025 and 2026 columns of the operator sheet "N° persone", by position (row 4 onwards).
_PARKING_COLUMNS = [
    "date_2025", "cars_2025", "_p_auto", "_p_auto_corr", "bus_up_2025", "moto_persons_2025", "_p_tot",
    "_p_tot_corr", "date_2026", "cars_full_2026", "_p_full", "cars_half_2026", "_p_half", "bus_up_2026",
    "moto_persons_2026", "_p_tot_2026", "note_2026", "note2_2026",
]

# The Greenbus 2025 pilot ran 28/06-20/07/2025; the later rows of the summary refer to
# another area (confirmed by the operator) and are flagged, not dropped.
GREENBUS_LAST_DAY = pd.Timestamp("2025-07-20")


def standardize_parking(path) -> pd.DataFrame:
    """Operator spreadsheet, 2025 and 2026 side by side -> one row per day.

    `bus_up`: "N° persone bus" (people going up); `bus_down`: people coming down, written
    as text ("301 (discesa)") in the note column from 01/08/2026. The sheet's own person
    totals are not read (inconsistent formulas, OPEN_QUESTIONS.md point 7).
    """
    raw = pd.read_excel(path, header=None, skiprows=3).iloc[:, : len(_PARKING_COLUMNS)]
    raw.columns = _PARKING_COLUMNS
    raw = raw.dropna(subset=["date_2025"])

    def num(col):
        return pd.to_numeric(raw[col], errors="coerce")

    note = raw["note_2026"].astype("string")
    is_bus_down = note.str.contains("discesa", na=False)
    y25 = pd.DataFrame({
        "date": pd.to_datetime(raw["date_2025"]),
        "cars_full": num("cars_2025"),
        "cars_half": np.nan,
        "moto_persons": num("moto_persons_2025"),
        "bus_up": num("bus_up_2025"),
        "bus_down": np.nan,
        "note": pd.NA,
    })
    notes_2026 = raw["note2_2026"].astype("string").fillna(note.where(~is_bus_down))
    # non-numeric car counts (e.g. "5days") are events: keep them in the note
    cars_text = raw["cars_full_2026"].where(pd.to_numeric(raw["cars_full_2026"], errors="coerce").isna())
    notes_2026 = notes_2026.fillna(cars_text.astype("string"))
    y26 = pd.DataFrame({
        "date": pd.to_datetime(raw["date_2026"]),
        "cars_full": num("cars_full_2026"),
        "cars_half": num("cars_half_2026"),
        "moto_persons": num("moto_persons_2026"),
        "bus_up": num("bus_up_2026"),
        "bus_down": note.str.extract(r"(\d+)")[0].astype(float).where(is_bus_down),
        "note": notes_2026,
    })
    out = pd.concat([y25, y26], ignore_index=True)
    out["note"] = out["note"].astype("string")
    return out.sort_values("date").reset_index(drop=True)


def standardize_parking_2025(path) -> pd.DataFrame:
    """2025-only parking sheet ("Parcheggio"): date, cars."""
    raw = pd.read_excel(path, sheet_name="Parcheggio")
    out = pd.DataFrame({"date": pd.to_datetime(raw["DATA"]), "cars": pd.to_numeric(raw["N° auto"], errors="coerce")})
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def standardize_greenbus_2025(path) -> pd.DataFrame:
    """Greenbus 2025 summary: tickets sold per day, `other_area` for the rows after the pilot."""
    raw = pd.read_excel(path, header=None)
    rows = []
    for date_col, value_col in ((1, 2), (4, 5)):  # the sheet has two side-by-side blocks
        block = raw.iloc[:, [date_col, value_col]].copy()
        block.columns = ["date", "tickets"]
        rows.append(block)
    out = pd.concat(rows)
    out["date"] = pd.to_datetime(out["date"], format="ISO8601", errors="coerce")  # header cells -> NaT
    out["tickets"] = pd.to_numeric(out["tickets"], errors="coerce")
    out = out.dropna(subset=["date", "tickets"]).sort_values("date").reset_index(drop=True)
    out["other_area"] = out["date"] > GREENBUS_LAST_DAY
    return out


# ------------------------------------------------------------------ road counts
ROAD_CAR_CLASSES = (2, 3)  # autovetture e monovolumi, also with trailer
ROAD_MIN_DAILY_CARS = 50  # below this per direction the sensor is considered off


def standardize_road_counts(path, stations: pd.DataFrame) -> pd.DataFrame:
    """PAT hourly counts (class x speed band rows) -> cars per station / date / direction / hour.

    Streamed with pyarrow: the source is ~2.3 GB / 69 M rows.
    """
    convert = pacsv.ConvertOptions(
        column_types={"PUNTO": pa.string(), "DATA": pa.string(), "ORA": pa.string(),
                      "DIREZIONE": pa.int8(), "CLASSE": pa.int8(), "N_VEICOLI": pa.int32()},
        include_columns=["PUNTO", "DATA", "ORA", "DIREZIONE", "CLASSE", "N_VEICOLI"],
    )
    keys = ["PUNTO", "DATA", "DIREZIONE", "ORA"]
    classes = pa.array(ROAD_CAR_CLASSES, type=pa.int8())
    partials = []
    with pacsv.open_csv(path, convert_options=convert, read_options=pacsv.ReadOptions(block_size=64 << 20)) as reader:
        for batch in reader:
            table = pa.Table.from_batches([batch])
            table = table.filter(pc.is_in(table["CLASSE"], value_set=classes))
            partials.append(table.group_by(keys).aggregate([("N_VEICOLI", "sum")]))
    agg = pa.concat_tables(partials).group_by(keys).aggregate([("N_VEICOLI_sum", "sum")]).to_pandas()

    names = stations.loc[stations["kind"] == "road"].set_index("code")["name"]
    out = pd.DataFrame({
        "station_code": agg["PUNTO"],
        "station": agg["PUNTO"].map(names),
        "date": pd.to_datetime(agg["DATA"], format="%d/%m/%Y"),
        "direction": agg["DIREZIONE"].astype("int8"),
        "hour": agg["ORA"].astype("int8"),
        "cars": agg["N_VEICOLI_sum_sum"].astype("int32"),
    })
    unknown = out["station"].isna()
    if unknown.any():
        raise ValueError(f"road stations not in {STATIONS_FILE.name}: {sorted(out.loc[unknown, 'station_code'].unique())}")
    return out.sort_values(["station", "date", "direction", "hour"]).reset_index(drop=True)


def road_counts_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Cars per station / date / direction, with `valid`.

    Valid = at least 23 hours present and at least ROAD_MIN_DAILY_CARS cars: several sensors
    keep reporting zeros for months (Dimaro from mid-2024, Tonale in 2022, Croviana direction 2
    most of the time); those zeros are missing data, not empty roads.
    """
    daily = (
        hourly.groupby(["station_code", "station", "date", "direction"])
        .agg(cars=("cars", "sum"), hours=("hour", "nunique"))
        .reset_index()
    )
    daily["valid"] = (daily["hours"] >= 23) & (daily["cars"] >= ROAD_MIN_DAILY_CARS)
    return daily


# ------------------------------------------------------------------ weather
def standardize_meteo_station(files: dict[str, list[Path]]) -> pd.DataFrame:
    """Meteotrentino yearly JSON -> date, tmin, tmax, rain_mm, quality code (as delivered;
    the codes are not documented: 145 from 2024, 1 in 2023, 81 on 4 days of 2023)."""
    def load(measure):
        return pd.concat([pd.DataFrame(json.loads(Path(f).read_text())["dati"]) for f in files[measure]],
                         ignore_index=True)

    temp = load("Temperature").rename(columns={"Min": "tmin", "Max": "tmax", "QualityMax": "quality"})
    rain = load("Precipitazioni").rename(columns={"Valore": "rain_mm"})
    out = temp[["giorno", "tmin", "tmax", "quality"]].merge(rain[["giorno", "rain_mm"]], on="giorno", how="outer")
    out["date"] = pd.to_datetime(out["giorno"])  # ISO yyyy-mm-dd (rain is listed by day, then month)
    return out[["date", "tmin", "tmax", "rain_mm", "quality"]].sort_values("date").reset_index(drop=True)


def standardize_meteo_bulletin(path, comune: str = "MALE'") -> pd.DataFrame:
    """Meteotrentino bulletin per comune (rounded values, text description)."""
    raw = pd.read_csv(path)
    raw = raw[raw["comune"] == comune]
    return pd.DataFrame({
        "date": pd.to_datetime(raw["data"]).dt.normalize(),
        "weather_desc": raw["meteo"],
        "tmin": raw["tmin"],
        "tmax": raw["tmax"],
        "rain_mm": raw["pioggiamm"],
    }).sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------------ Vodafone
# Val di Sole areas in vodafone_aree.csv
VODAFONE_LOCATIONS = {
    "LAGO_CAPR": "poi_caprioli",
    "8": "comune_ossana_pellizzano",
    "3": "comune_vermiglio",
    "104": "apt_valdisole",
    "22": "provincia_trentino",
}
VODAFONE_PROFILES = ("TOURIST", "VISITOR")  # excludes INHABITANT / COMMUTER


def _read_vodafone(path, columns) -> pd.DataFrame:
    table = pacsv.read_csv(
        path,
        convert_options=pacsv.ConvertOptions(
            column_types={"locId": pa.string(), "date": pa.string()}, include_columns=columns,
        ),
    )
    table = table.filter(pc.is_in(table["locId"], value_set=pa.array(list(VODAFONE_LOCATIONS))))
    return table.to_pandas()


def standardize_vodafone(path_attendences, path_str) -> pd.DataFrame:
    """Daily presences per Val di Sole area: Italian visitors (``userCountry == "ITA"``) plus the
    pre-aggregated foreign total (``*_STR``) rather than single nationalities, which are masked
    to 0 below the privacy threshold of 15."""
    att = _read_vodafone(path_attendences, ["date", "locId", "userCountry", "userProfile", "value"])
    att_str = _read_vodafone(path_str, ["date", "locId", "userProfile", "value"])
    ita = (att[(att["userCountry"] == "ITA") & att["userProfile"].isin(VODAFONE_PROFILES)]
           .groupby(["date", "locId"])["value"].sum().rename("presences_ita"))
    foreign = (att_str[att_str["userProfile"].isin(VODAFONE_PROFILES)]
               .groupby(["date", "locId"])["value"].sum().rename("presences_foreign"))
    out = pd.concat([ita, foreign], axis=1).fillna(0).reset_index()
    out["presences"] = out["presences_ita"] + out["presences_foreign"]
    out["date"] = pd.to_datetime(out["date"])
    out["area"] = out["locId"].map(VODAFONE_LOCATIONS)
    return out.rename(columns={"locId": "loc_id"})[
        ["date", "loc_id", "area", "presences_ita", "presences_foreign", "presences"]
    ].sort_values(["area", "date"]).reset_index(drop=True)


# ------------------------------------------------------------------ bikes, Guest Card
def standardize_bike_counts(path) -> pd.DataFrame:
    """Hourly bike/pedestrian counters -> per sensor and day. cl8 + cl10 = bikes and pedestrians;
    cl0 (motorbikes) kept separately."""
    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["data"]).dt.normalize()
    out = (raw.groupby(["date", "sensore"])[["cl0", "cl8", "cl10", "nonrilevato"]].sum().reset_index()
           .rename(columns={"sensore": "sensor", "cl0": "motorbikes", "nonrilevato": "not_detected"}))
    out["bikes_peds"] = out["cl8"] + out["cl10"]
    return out


VALDISOLE_PLACES_RE = r"(?i)OSSANA|PELLIZZANO|VERMIGLIO|DIMARO|MALE|RABBI|PEIO|COMMEZZADURA|CALDES|MEZZANA|FOLGARIDA"


def standardize_tgc_accessi(path) -> pd.DataFrame:
    """Guest Card entries in Val di Sole -> entrants per day, place and service."""
    raw = pd.read_csv(path, sep=";", encoding="utf-8", encoding_errors="replace",
                      usecols=["Data Marcata", "NomeServizio", "Entranti", "Luogo"])
    raw = raw[raw["Luogo"].astype(str).str.contains(VALDISOLE_PLACES_RE, regex=True)]
    raw["date"] = pd.to_datetime(raw["Data Marcata"], format="%d/%m/%Y %H:%M", errors="coerce").dt.normalize()
    out = (raw.dropna(subset=["date"]).groupby(["date", "Luogo", "NomeServizio"])["Entranti"].sum().reset_index()
           .rename(columns={"Luogo": "place", "NomeServizio": "service", "Entranti": "entrants"}))
    out["place"] = out["place"].map(lambda s: re.sub(r"\s+", " ", str(s)).strip())
    return out


# ------------------------------------------------------------------ all
def standardize_all(paths: dict[str, Path], meteo_files: dict[str, list[Path]]) -> dict[str, pd.DataFrame]:
    stations = pd.read_csv(STATIONS_FILE, dtype={"code": str})
    tables = {"stations": stations}
    logger.info("Parking and bus...")
    tables["fazzon_parking"] = standardize_parking(paths["parking"])
    tables["fazzon_parking_2025"] = standardize_parking_2025(paths["parking_2025"])
    tables["fazzon_greenbus_2025"] = standardize_greenbus_2025(paths["greenbus_2025"])
    logger.info("Road counts (streaming %s)...", Path(paths["road_counts"]).name)
    tables["road_counts_hourly"] = standardize_road_counts(paths["road_counts"], stations)
    tables["road_counts_daily"] = road_counts_daily(tables["road_counts_hourly"])
    logger.info("Weather...")
    tables["meteo_station"] = standardize_meteo_station(meteo_files)
    tables["meteo_bulletin"] = standardize_meteo_bulletin(paths["meteo_bulletin"])
    logger.info("Vodafone...")
    tables["vodafone_presences"] = standardize_vodafone(paths["vodafone_attendences"], paths["vodafone_attendences_str"])
    logger.info("Bike counters and Guest Card...")
    tables["bike_counts"] = standardize_bike_counts(paths["bike_counts"])
    tables["tgc_accessi"] = standardize_tgc_accessi(paths["tgc_accessi"])
    return tables
