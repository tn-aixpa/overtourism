# SPDX-License-Identifier: Apache-2.0
"""
STEP 0 - Download.

Two entry points, one per pipeline:

    download_raw_base_data()      base build   -> Output/data/raw_data/  (+ Output/mapping/)
    download_update_raw_data(cfg) yearly update -> Output/data/raw_data/  (+ Output/mapping/)

Data are saved exactly as downloaded: no renaming, no normalization.
The reference files (json mappings + geojson) go to Output/mapping/ and are downloaded in BOTH
cases, by the same function (`download_mappings`).
"""

import io
import logging
from pathlib import Path

from utils.config import (
    LOCAL_SOURCE_DIR,
    MAPPING_DIR,
    RAW_DIR,
    REFERENCES,
    TYPE_FORMAT,
    setup_logging,
)
from utils.datasets import (
    COVISITS_RAW_FILE,
    PRESENZE_BASE_YEARS,
    PRESENZE_FOLDER,
    PRESENZE_MEASURES,
    PRESENZE_TAGS,
)
from utils.io import ensure_dir, raw_path, save_computed_dfs

logger = logging.getLogger(__name__)

S3_DATA = [
    "Annuario-TavXIII-per-comune-csv.csv",
    "strutture_annuario_2024.ods",
    "popolazione_2020_2024.parquet",
    "vodafone_attendences.parquet",
    "vodafone_attendences_new.csv",  # completes the presences up to the end of the base window (2024-12-31)
    COVISITS_RAW_FILE,  # Vodafone OD co-visits (flows between areas)
    # ISPAT arrivals / presences (alb, extralb) of the base years; later years come with the updates
    *(
        f"{PRESENZE_FOLDER}/{measure}_{tag}_{year}.csv"
        for year in PRESENZE_BASE_YEARS
        for tag in PRESENZE_TAGS.values()
        for measure in PRESENZE_MEASURES
    ),
    *(
        f"Energia/{folder}/{filename}"
        for folder in ("ee_prelievi", "gas_prelievi")
        for filename in ("49_3.csv", "55_1.csv", "56_1.csv", "93_1.csv", "96.csv")
    ),
]


# ------------------------------------------------------------------ low level
def fetch_bytes(key: str, *, local_dir=None, save_to=None) -> io.BytesIO:
    """Content of `key` (relative to the data-lake prefix, or to `local_dir` if given).
    If `save_to` is set, a copy is stored there under the file's base name."""
    if local_dir is not None:
        buffer = io.BytesIO((Path(local_dir) / key).read_bytes())
    else:
        from utils.remote import get_s3

        buffer = get_s3(key)
    if save_to is not None:
        save_to = ensure_dir(save_to)
        (save_to / Path(key).name).write_bytes(buffer.getvalue())
    return buffer


# ------------------------------------------------------------------ shared
def download_mappings(mapping_dir=MAPPING_DIR, local_dir=None) -> None:
    """Downloads the reference files (json mappings + geojson, `references` in settings.yaml)
    into `mapping_dir`. Used by the base build AND by the update."""
    mapping_dir = ensure_dir(mapping_dir)
    for key in REFERENCES.values():
        logger.info("Downloading mapping %s...", key)
        fetch_bytes(key, local_dir=local_dir, save_to=mapping_dir)
    logger.info("Mappings saved in %s", mapping_dir)


# ------------------------------------------------------------------ base build
def download_raw_base_data(
    out_dir=RAW_DIR, type_format=TYPE_FORMAT, mapping_dir=MAPPING_DIR
):
    """Base build: the historical raw files + the mappings."""
    from utils.remote import get_dataframe

    out_dir = ensure_dir(out_dir)

    for name in S3_DATA:
        logger.info("Downloading object %s from S3...", name)
        # The energy folders contain identically named category files; keep their
        # folder names in raw_data so electricity and gas are never overwritten.
        fetch_bytes(name, save_to=raw_path(name, out_dir).parent)

    download_mappings(mapping_dir)
    logger.info("Raw data saved in %s, mappings in %s", out_dir, mapping_dir)


# ------------------------------------------------------------------ update
def download_update_raw_data(
    config, out_dir=RAW_DIR, mapping_dir=MAPPING_DIR, local_dir=LOCAL_SOURCE_DIR
):
    """Update: the source files listed in the update config + the mappings.
    `local_dir` (settings.yaml: update.local_source_dir) reads them from a folder instead of S3.
    """
    from utils.update.spec import load_config

    config = load_config(config)
    out_dir = ensure_dir(out_dir)

    for dataset in config.datasets:
        for src in config.sources[dataset]:
            logger.info("[%s] downloading %s...", dataset, src.file)
            fetch_bytes(
                src.file, local_dir=local_dir, save_to=raw_path(src.file, out_dir).parent
            )

    download_mappings(mapping_dir, local_dir=local_dir)
    logger.info("Update raw data saved in %s, mappings in %s", out_dir, mapping_dir)


if __name__ == "__main__":
    setup_logging()
    download_raw_base_data()
