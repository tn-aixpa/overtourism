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
from tests.conftest import PRESENCE_STATS, write_fazzon_stats


def test_reads_reference_dataset_from_data_dir():
    table = load_presence_stats()
    assert table.index.names == ["season", "weather"]
    assert len(table) == 12
    car, other = make_presence_factories(table)
    assert car(season="august", weather="rainy").mean() == pytest.approx(897.3 * 0.6)
    assert other(season="july", weather="good").std() == pytest.approx(611.7 * 0.449 * 0.5)


def test_presence_inputs_reads_files_at_every_call(tmp_path, monkeypatch):
    stats = PRESENCE_STATS.copy()
    stats.loc[stats["season"] == "july", "mean_car_visitors"] = 1000.0
    write_fazzon_stats(tmp_path, stats=stats)
    monkeypatch.setenv(fazzon_data.DATA_DIR_ENV, str(tmp_path))
    seasons, weather, car, _ = presence_inputs()
    assert seasons["july"] == pytest.approx(31 / 82)
    assert weather == pytest.approx({"good": 0.62, "cool": 0.2, "rainy": 0.18})
    assert car(season="july", weather="cool").mean() == pytest.approx(1000.0)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_presence_stats(tmp_path)


def test_missing_cell_raises(tmp_path):
    write_fazzon_stats(tmp_path, stats=PRESENCE_STATS[PRESENCE_STATS["season"] != "june"])
    with pytest.raises(ValueError, match="cells"):
        load_presence_stats(tmp_path)


def test_missing_column_raises(tmp_path):
    write_fazzon_stats(tmp_path, stats=PRESENCE_STATS.drop(columns="weather_freq_rel"))
    with pytest.raises(ValueError, match="missing columns"):
        load_presence_stats(tmp_path)


def test_weather_weights_average_the_seasons(tmp_path, monkeypatch):
    stats = PRESENCE_STATS.copy()
    july = stats["season"] == "july"
    stats.loc[july, "weather_freq_rel"] = stats.loc[july, "weather"].map({"good": 0.5, "cool": 0.3, "rainy": 0.2})
    write_fazzon_stats(tmp_path, stats=stats)
    seasons, weather, _, _ = presence_inputs(tmp_path)
    expected_good = seasons["july"] * 0.5 + (1 - seasons["july"]) * 0.62
    assert weather["good"] == pytest.approx(expected_good)
    assert sum(weather.values()) == pytest.approx(1.0)


def test_season_and_weather_can_be_fixed():
    from src.model.fazzon.fazzon_backend import FazzonBackend

    backend = FazzonBackend()
    for overrides in ({"weather": "rainy"}, {"season": "august"}, {"season": "august", "weather": "good"}):
        output = backend.evaluate(overrides)
        assert len(output.samples_x) > 0


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
    remote = PRESENCE_STATS.copy()
    remote["mean_car_visitors"] += 1.0
    source = write_fazzon_stats(tmp_path / "remote", stats=remote)
    target = write_fazzon_stats(tmp_path / "local")
    monkeypatch.setenv(fazzon_data.DATA_DIR_ENV, str(target))
    _fake_digitalhub(monkeypatch, source_dir=source)

    assert fazzon_data.download_fazzon_data(overwrite=True) == target
    table = load_presence_stats(target)
    assert table.loc[("august", "good"), "mean_car_visitors"] == pytest.approx(898.3)


def test_download_skipped_when_files_present(tmp_path, monkeypatch):
    target = write_fazzon_stats(tmp_path / "local")
    monkeypatch.setenv(fazzon_data.DATA_DIR_ENV, str(target))
    _fake_digitalhub(monkeypatch, error=AssertionError("must not download"))
    assert fazzon_data.download_fazzon_data() == target


def test_failed_download_raises_and_keeps_files(tmp_path, monkeypatch):
    target = write_fazzon_stats(tmp_path / "local")
    before = (target / fazzon_data.PRESENCE_STATS_FILE).read_bytes()
    monkeypatch.setenv(fazzon_data.DATA_DIR_ENV, str(target))
    _fake_digitalhub(monkeypatch, error=OSError("platform unreachable"))
    with pytest.raises(RuntimeError, match="platform unreachable"):
        fazzon_data.download_fazzon_data(overwrite=True)
    assert (target / fazzon_data.PRESENCE_STATS_FILE).read_bytes() == before


def test_unusable_download_raises_and_keeps_files(tmp_path, monkeypatch):
    source = write_fazzon_stats(tmp_path / "remote", stats=PRESENCE_STATS.drop(columns="season_freq_rel"))
    target = write_fazzon_stats(tmp_path / "local")
    before = (target / fazzon_data.PRESENCE_STATS_FILE).read_bytes()
    monkeypatch.setenv(fazzon_data.DATA_DIR_ENV, str(target))
    _fake_digitalhub(monkeypatch, source_dir=source)
    with pytest.raises(ValueError, match="missing columns"):
        fazzon_data.download_fazzon_data(overwrite=True)
    assert (target / fazzon_data.PRESENCE_STATS_FILE).read_bytes() == before


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
