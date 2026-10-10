import pandas as pd
import pytest

from src.model.fazzon.fazzon_presence_stats import (
    DATA_DIR_ENV,
    DAY_TYPE_STATS_FILE,
    DEFAULT_DAY_TYPE_STATS,
    DEFAULT_SEASON_STATS,
    SEASON_STATS_FILE,
    load_presence_stats,
    make_presence_factories,
)


def _write_reference(path, season=DEFAULT_SEASON_STATS, day_type=DEFAULT_DAY_TYPE_STATS):
    # same layout as the pipeline output: flat columns plus extra metadata columns
    season.reset_index().assign(calibration="parking_2025", n_days=1).to_parquet(path / SEASON_STATS_FILE)
    day_type.reset_index().assign(calibration="parking_2025").to_parquet(path / DAY_TYPE_STATS_FILE)


def test_defaults_without_data_dir(monkeypatch):
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    season, day_type = load_presence_stats()
    assert season is DEFAULT_SEASON_STATS
    assert day_type is DEFAULT_DAY_TYPE_STATS


def test_reads_reference_datasets_from_env_dir(tmp_path, monkeypatch):
    season_in = DEFAULT_SEASON_STATS.copy()
    season_in.loc["august", "mean_car_visitors"] = 800.0
    _write_reference(tmp_path, season=season_in)
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    season, day_type = load_presence_stats()
    assert list(season.columns) == list(DEFAULT_SEASON_STATS.columns)
    assert season.loc["august", "mean_car_visitors"] == 800.0
    pd.testing.assert_frame_equal(day_type, DEFAULT_DAY_TYPE_STATS, check_dtype=False)
    car, _ = make_presence_factories(season, day_type)
    expected = 800.0 + DEFAULT_DAY_TYPE_STATS.loc[("august", "peak"), "mean_car_visitors"]
    assert car("peak", "august").mean() == pytest.approx(expected)


def test_missing_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_presence_stats(tmp_path)


def test_missing_season_raises(tmp_path):
    _write_reference(tmp_path, season=DEFAULT_SEASON_STATS.drop(index="june"))
    with pytest.raises(ValueError, match="seasons"):
        load_presence_stats(tmp_path)
