# SPDX-License-Identifier: Apache-2.0
"""Configuration of the Fazzon pipeline: fazzon.yaml (sources) + settings.yaml (directories, platform)."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from utils.config import FINAL_DIR, NORMALIZED_DIR, RAW_DIR

FAZZON_DIR = Path(__file__).resolve().parent
CONFIG_FILE = FAZZON_DIR / "fazzon.yaml"
STATIONS_FILE = FAZZON_DIR / "reference" / "stations.csv"

FAZZON_RAW_DIR = RAW_DIR / "fazzon"
FAZZON_NORMALIZED_DIR = NORMALIZED_DIR / "fazzon"
FAZZON_FINAL_DIR = FINAL_DIR / "fazzon"


@dataclass(frozen=True)
class Source:
    name: str
    key: str  # data-lake key, relative to the S3 prefix
    local: str  # path relative to the local source dir


@dataclass(frozen=True)
class FazzonConfig:
    sources: dict[str, Source]
    local_source_dir: Path | None


def load_config(path=CONFIG_FILE) -> FazzonConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sources = {
        name: Source(name=name, key=spec["key"], local=spec["local"])
        for name, spec in (raw.get("sources") or {}).items()
    }
    local = raw.get("local_source_dir")
    local_dir = None
    if local:
        local_dir = Path(local).expanduser()
        if not local_dir.is_absolute():
            local_dir = path.parent / local_dir
    return FazzonConfig(sources=sources, local_source_dir=local_dir)
