import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.model.fazzon import fazzon_data
from src.model.fazzon.fazzon_presence_stats import (
    load_presence_stats,
    make_presence_factories,
    presence_inputs,
)
from tests.conftest import SEASON_STATS, write_fazzon_stats


def test_reads_reference_dataset_from_data_dir():
    season = load_presence_stats()
    assert list(season.index) == ["june", "july", "august", "september"]
    assert season.loc["august", "mean_car_visitors"] == 767.4
    car, other = make_presence_factories(season)
    assert car("august").mean() == pytest.approx(767.4)
    assert other("july").std() == pytest.approx(111.9)


def test_presence_inputs_reads_files_at_every_call(tmp_path, monkeypatch):
    season = SEASON_STATS.copy()
    season.loc[season["cluster_name"] == "july", "mean_car_visitors"] = 1000.0
    write_fazzon_stats(tmp_path, season=season)
    monkeypatch.setenv(fazzon_data.DATA_DIR_ENV, str(tmp_path))
    weights, car, _ = presence_inputs()
    assert weights["july"] == pytest.approx(31 / 82)
    assert car("july").mean() == pytest.approx(1000.0)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_presence_stats(tmp_path)


def test_missing_season_raises(tmp_path):
    write_fazzon_stats(tmp_path, season=SEASON_STATS[SEASON_STATS["cluster_name"] != "june"])
    with pytest.raises(ValueError, match="seasons"):
        load_presence_stats(tmp_path)


def test_missing_column_raises(tmp_path):
    write_fazzon_stats(tmp_path, season=SEASON_STATS.drop(columns="mean_other_visitors"))
    with pytest.raises(ValueError, match="missing columns"):
        load_presence_stats(tmp_path)


# ---------------------------------------------------------------- download / startup


def _fake_digitalhub(monkeypatch, source_dir=None, error=None):
    """Replace the digitalhub client: dataitems are copied from *source_dir*, or *error* is raised."""

    def get_dataitem(name):
        def download(dest, overwrite=False):
            if error is not None:
                raise error
            (Path(dest) / name).write_bytes((source_dir / name).read_bytes())

        return SimpleNamespace(download=download)

    monkeypatch.setitem(
        sys.modules,
        "digitalhub",
        SimpleNamespace(get_project=lambda name: SimpleNamespace(get_dataitem=get_dataitem)),
    )


def test_download_replaces_files(tmp_path, monkeypatch):
    remote = SEASON_STATS.copy()
    remote["mean_car_visitors"] += 1.0
    source = write_fazzon_stats(tmp_path / "remote", season=remote)
    target = write_fazzon_stats(tmp_path / "local")
    monkeypatch.setenv(fazzon_data.DATA_DIR_ENV, str(target))
    _fake_digitalhub(monkeypatch, source_dir=source)

    assert fazzon_data.download_fazzon_data(overwrite=True) == target
    season = load_presence_stats(target)
    assert season.loc["august", "mean_car_visitors"] == pytest.approx(768.4)


def test_download_skipped_when_files_present(tmp_path, monkeypatch):
    target = write_fazzon_stats(tmp_path / "local")
    monkeypatch.setenv(fazzon_data.DATA_DIR_ENV, str(target))
    _fake_digitalhub(monkeypatch, error=AssertionError("must not download"))
    assert fazzon_data.download_fazzon_data() == target


def test_failed_download_raises_and_keeps_files(tmp_path, monkeypatch):
    target = write_fazzon_stats(tmp_path / "local")
    before = (target / fazzon_data.SEASON_STATS_FILE).read_bytes()
    monkeypatch.setenv(fazzon_data.DATA_DIR_ENV, str(target))
    _fake_digitalhub(monkeypatch, error=OSError("platform unreachable"))
    with pytest.raises(RuntimeError, match="platform unreachable"):
        fazzon_data.download_fazzon_data(overwrite=True)
    assert (target / fazzon_data.SEASON_STATS_FILE).read_bytes() == before


def test_unusable_download_raises_and_keeps_files(tmp_path, monkeypatch):
    source = write_fazzon_stats(tmp_path / "remote", season=SEASON_STATS.drop(columns="freq_rel"))
    target = write_fazzon_stats(tmp_path / "local")
    before = (target / fazzon_data.SEASON_STATS_FILE).read_bytes()
    monkeypatch.setenv(fazzon_data.DATA_DIR_ENV, str(target))
    _fake_digitalhub(monkeypatch, source_dir=source)
    with pytest.raises(ValueError, match="missing columns"):
        fazzon_data.download_fazzon_data(overwrite=True)
    assert (target / fazzon_data.SEASON_STATS_FILE).read_bytes() == before


def test_startup_standalone_requires_files(tmp_path, monkeypatch):
    monkeypatch.setenv(fazzon_data.DATA_DIR_ENV, str(tmp_path / "empty"))
    monkeypatch.setenv("DT_OVERTURISM_STANDALONE_MODE", "true")
    with pytest.raises(FileNotFoundError, match="missing"):
        fazzon_data.ensure_fazzon_data()


def test_startup_downloads_when_not_standalone(tmp_path, monkeypatch):
    source = write_fazzon_stats(tmp_path / "remote")
    monkeypatch.setenv(fazzon_data.DATA_DIR_ENV, str(tmp_path / "local"))
    monkeypatch.setenv("DT_OVERTURISM_STANDALONE_MODE", "false")
    _fake_digitalhub(monkeypatch, source_dir=source)
    assert fazzon_data.ensure_fazzon_data() == tmp_path / "local"
