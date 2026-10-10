# SPDX-License-Identifier: Apache-2.0
"""
STEP 0 - Raw sources of the Fazzon pipeline.

Input : data lake (settings.yaml platform.s3_prefix + fazzon.yaml keys), or fazzon.yaml local_source_dir;
        Meteotrentino open-data service for the measured weather
Output: paths of the raw files, as they arrive (no renaming of the content, no normalization)

With a local source dir the files are read in place (no copy: the road counts alone are ~2.3 GB).
From the data lake they are streamed to <raw>/fazzon/ and reused on the next run unless `refresh`.
"""

import datetime as dt
import logging
import urllib.request
from pathlib import Path

from model.fazzon.config import FAZZON_RAW_DIR, FazzonConfig, Source
from utils.io import ensure_dir

logger = logging.getLogger(__name__)

METEO_SERVICE = (
    "https://dati.meteotrentino.it/service.asmx/"
    "datiStoriciCaratteristiciSerie{measure}UnArea?codice={code}&anno={year}"
)
METEO_MEASURES = ("Temperature", "Precipitazioni")


def _download_s3(source: Source, dest: Path) -> None:
    """Streams one object to `dest` (download_file does not hold it in memory)."""
    from utils.config import S3_PREFIX
    from utils.remote import init_s3

    _, bucket = init_s3()
    tmp = dest.with_suffix(dest.suffix + ".part")
    bucket.download_file(S3_PREFIX + source.key, str(tmp))
    tmp.rename(dest)


def resolve_sources(config: FazzonConfig, refresh: bool = False, raw_dir=FAZZON_RAW_DIR) -> dict[str, Path]:
    """Local path of every source in fazzon.yaml, downloading it from the data lake if needed."""
    paths = {}
    for name, source in config.sources.items():
        if config.local_source_dir is not None:
            path = config.local_source_dir / source.local
            if not path.exists():
                raise FileNotFoundError(f"[{name}] {path} not found (fazzon.yaml local_source_dir)")
        else:
            path = ensure_dir(raw_dir) / Path(source.key).name
            if refresh or not path.exists():
                logger.info("[%s] downloading %s from the data lake...", name, source.key)
                _download_s3(source, path)
        paths[name] = path
    return paths


def fetch_meteo_station(config: FazzonConfig, refresh: bool = False, raw_dir=FAZZON_RAW_DIR) -> dict[str, list[Path]]:
    """Yearly JSON files of the Meteotrentino station, cached in <raw>/fazzon/meteo/.

    The current year is always downloaded again (new days); past years only if missing or `refresh`.
    """
    meteo_dir = ensure_dir(Path(raw_dir) / "meteo")
    this_year = dt.date.today().year
    files: dict[str, list[Path]] = {m: [] for m in METEO_MEASURES}
    for measure in METEO_MEASURES:
        for year in range(config.meteo_first_year, this_year + 1):
            path = meteo_dir / f"{config.meteo_station}_{measure}_{year}.json"
            if refresh or year == this_year or not path.exists():
                url = METEO_SERVICE.format(measure=measure, code=config.meteo_station, year=year)
                logger.info("Downloading %s %s %d from Meteotrentino...", config.meteo_station, measure, year)
                with urllib.request.urlopen(url, timeout=60) as resp:
                    path.write_bytes(resp.read())
            files[measure].append(path)
    return files
