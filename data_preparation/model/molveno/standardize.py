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

from model.common.weather import standardize_meteo_bulletin

logger = logging.getLogger(__name__)

# vodafone_aree.csv: 27;comune;TN_MKT_AL_3;Molveno
VODAFONE_LOC_ID = "27"
VODAFONE_LOC_TYPE = "TN_MKT_AL_3"
VODAFONE_PROFILES = {"TOURIST": "tourists", "VISITOR": "excursionists"}  # D5
METEO_BULLETIN_COMUNE = "MOLVENO"


def standardize_vodafone(path) -> pd.DataFrame:
    """D5: daily presences in the comune of Molveno per profile, all user countries summed
    (single countries are masked to 0 below the privacy threshold)."""
    table = pacsv.read_csv(
        path,
        convert_options=pacsv.ConvertOptions(
            column_types={"locId": pa.string(), "date": pa.string()},
            include_columns=["date", "locType", "locId", "userProfile", "value"],
        ),
    )
    table = table.filter(pc.and_(pc.equal(table["locId"], VODAFONE_LOC_ID), pc.equal(table["locType"], VODAFONE_LOC_TYPE)))
    raw = table.to_pandas()
    raw = raw[raw["userProfile"].isin(list(VODAFONE_PROFILES))]
    out = raw.pivot_table(index="date", columns="userProfile", values="value", aggfunc="sum")
    out = out.rename(columns=VODAFONE_PROFILES).reset_index()
    out.columns.name = None
    out["date"] = pd.to_datetime(out["date"])
    return out[["date", *VODAFONE_PROFILES.values()]].sort_values("date").reset_index(drop=True)


def standardize_all(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    logger.info("Vodafone...")
    tables = {"vodafone_presences": standardize_vodafone(paths["vodafone_attendences"])}
    logger.info("Weather...")
    tables["meteo_bulletin"] = standardize_meteo_bulletin(paths["meteo_bulletin"], METEO_BULLETIN_COMUNE)
    return tables
