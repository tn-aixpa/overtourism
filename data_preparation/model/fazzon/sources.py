# SPDX-License-Identifier: Apache-2.0
"""
STEP 0 - Raw sources of the Fazzon pipeline.

Input : data lake (settings.yaml platform.s3_prefix + fazzon.yaml keys), or fazzon.yaml local_source_dir
Output: paths of the raw files, as they arrive (no renaming of the content, no normalization)

As in the main pipeline, only the data lake is read: external sources (e.g. the Meteotrentino
weather) are fetched by the scripts in data_download/ and uploaded to the data lake.

With a local source dir the files are read in place (no copy: the road counts alone are ~2.3 GB).
From the data lake they are streamed to <raw>/fazzon/ and reused on the next run unless `refresh`.
"""

import logging
from pathlib import Path

from model.fazzon.config import FAZZON_RAW_DIR, FazzonConfig, Source
from utils.io import ensure_dir

logger = logging.getLogger(__name__)


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

