# SPDX-License-Identifier: Apache-2.0
"""
Sources of the model pipelines (model/<name>/): configuration and step 0.

Each pipeline lists its sources in a YAML file (``<name>.yaml``):

    local_source_dir: data     # read the sources from this folder (null = data lake)
    sources:
      <name>:
        key: <object key on the data lake, relative to settings.yaml platform.s3_prefix>
        local: <path relative to local_source_dir>

As in the main pipeline, only the data lake is read: external sources are fetched by the
scripts in data_download/ and uploaded to the data lake. With a local source dir the files are
read in place (no copy); from the data lake they are streamed to the pipeline's raw folder and
reused on the next run unless `refresh`.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from model.common.io import ensure_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Source:
    name: str
    key: str  # data-lake key, relative to the S3 prefix
    local: str  # path relative to the local source dir


@dataclass(frozen=True)
class SourcesConfig:
    sources: dict[str, Source]
    local_source_dir: Path | None


def load_sources_config(path, local_source_dir=None) -> SourcesConfig:
    """Reads the YAML; `local_source_dir` (e.g. from the command line) overrides its value."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sources = {
        name: Source(name=name, key=spec["key"], local=spec["local"])
        for name, spec in (raw.get("sources") or {}).items()
    }
    if local_source_dir is not None:
        return SourcesConfig(sources=sources, local_source_dir=Path(local_source_dir).expanduser().resolve())
    local = raw.get("local_source_dir")
    local_dir = None
    if local:
        local_dir = Path(local).expanduser()
        if not local_dir.is_absolute():
            local_dir = path.parent / local_dir
    return SourcesConfig(sources=sources, local_source_dir=local_dir)


def _download_s3(source: Source, dest: Path) -> None:
    """Streams one object to `dest` (download_file does not hold it in memory)."""
    from model.common.config import S3_PREFIX
    from model.common.remote import init_s3

    _, bucket = init_s3()
    tmp = dest.with_suffix(dest.suffix + ".part")
    bucket.download_file(S3_PREFIX + source.key, str(tmp))
    tmp.rename(dest)


def resolve_sources(config: SourcesConfig, raw_dir: Path, refresh: bool = False) -> dict[str, Path]:
    """Local path of every source, downloading it from the data lake if needed."""
    paths = {}
    for name, source in config.sources.items():
        if config.local_source_dir is not None:
            path = config.local_source_dir / source.local
            if not path.exists():
                raise FileNotFoundError(f"[{name}] {path} not found (local_source_dir)")
        else:
            path = ensure_dir(raw_dir) / Path(source.key).name
            if refresh or not path.exists():
                logger.info("[%s] downloading %s from the data lake...", name, source.key)
                _download_s3(source, path)
        paths[name] = path
    return paths
