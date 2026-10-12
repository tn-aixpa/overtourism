# SPDX-License-Identifier: Apache-2.0
"""
STEP 1 - Standardization of the Molveno sources.

Input : raw files (step 0, model/common/sources.py)
Output: <normalized>/molveno/<table>.parquet, one clean table per source

Only cleaning here; modelling choices belong to step 2 (IDs refer to ASSUMPTIONS.md).
"""

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from model.common.weather import standardize_meteo_station

logger = logging.getLogger(__name__)

# vodafone_aree.csv: 27;comune;TN_MKT_AL_3;Molveno
VODAFONE_LOC_ID = "27"
VODAFONE_LOC_TYPE = "TN_MKT_AL_3"
VODAFONE_PROFILES = {"TOURIST": "tourists", "VISITOR": "excursionists"}


def _read_vodafone(path, columns) -> pd.DataFrame:
    table = pacsv.read_csv(
        path,
        convert_options=pacsv.ConvertOptions(
            column_types={"locId": pa.string(), "date": pa.string()}, include_columns=columns,
        ),
    )
    keep = pc.and_(pc.equal(table["locId"], VODAFONE_LOC_ID), pc.equal(table["locType"], VODAFONE_LOC_TYPE))
    raw = table.filter(keep).to_pandas()
    return raw[raw["userProfile"].isin(list(VODAFONE_PROFILES))]


def _per_day(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.pivot_table(index="date", columns="userProfile", values="value", aggfunc="sum")


def standardize_vodafone(path_attendences, path_str) -> pd.DataFrame:
    """D13: daily presences in the comune of Molveno per profile: Italians (``userCountry ==
    "ITA"``) plus the aggregated foreign total (``*_STR``), since single foreign countries are
    masked to 0 below the privacy threshold of 15."""
    att = _read_vodafone(path_attendences, ["date", "locType", "locId", "userCountry", "userProfile", "value"])
    foreign = _per_day(_read_vodafone(path_str, ["date", "locType", "locId", "userProfile", "value"]))
    ita = _per_day(att[att["userCountry"] == "ITA"])
    out = ita.add(foreign.reindex(ita.index).fillna(0), fill_value=0)  # no STR row = no foreigners
    out = out.rename(columns=VODAFONE_PROFILES).reset_index()
    out.columns.name = None
    out["date"] = pd.to_datetime(out["date"])
    return out[["date", *VODAFONE_PROFILES.values()]].sort_values("date").reset_index(drop=True)


def standardize_all(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    logger.info("Vodafone...")
    tables = {"vodafone_presences": standardize_vodafone(paths["vodafone_attendences"], paths["vodafone_attendences_str"])}
    logger.info("Weather...")
    tables["meteo_station"] = standardize_meteo_station(paths["meteo_station"])
    return tables
