"""Molveno presence statistics: loading from the pipeline's files, lookups, period, backend."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import molveno_tables, write_molveno_stats

from src.model.molveno import molveno_data
from src.model.molveno.molveno_backend import MolvenoBackend
from src.model.molveno.molveno_presence_stats import (
    DAY_INDEX_NAME,
    ESTIMATOR_INDEX_NAME,
    ESTIMATOR_ML,
    ESTIMATOR_MONTHLY,
    load_tables,
    make_presence_factories,
    summer_day,
    weather_weights,
)


def _kwargs(day, weekday="wednesday", weather="good", estimator=ESTIMATOR_ML):
    return {DAY_INDEX_NAME: day, "weekday": weekday, "weather": weather, ESTIMATOR_INDEX_NAME: estimator}


def test_load_tables_reads_the_pipeline_files():
    tables = load_tables()
    assert tables.n_days == 122
    assert tables.day_label[0] == "06-01" and tables.day_label[-1] == "09-30"
    assert tables.day_month[61] == 8


def test_missing_file_raises(tmp_path):
    folder = write_molveno_stats(tmp_path)
    (folder / molveno_data.WEATHER_FREQ_FILE).unlink()
    with pytest.raises(FileNotFoundError):
        load_tables(folder)


def test_missing_cell_raises(tmp_path):
    tables = molveno_tables()
    daily = tables["molveno_presence_stats"]
    tables["molveno_presence_stats"] = daily[~((daily["summer_day"] == 10) & (daily["weather"] == "rainy"))]
    with pytest.raises(ValueError, match="cells missing"):
        load_tables(write_molveno_stats(tmp_path, tables))


def test_summer_day_is_the_day_the_value_falls_in():
    assert [summer_day(v, 122) for v in (0.0, 0.99, 61.5, 121.9, 122.0, -1.0)] == [0, 0, 61, 121, 121, 0]


def test_factories_look_up_the_day_or_the_month():
    tables = load_tables()
    tourists, excursionists = make_presence_factories(tables)
    daily = tables.daily[(75, "sunday", "rainy")]
    assert tourists(**_kwargs(75.4, "sunday", "rainy")).mean() == pytest.approx(daily[0], rel=1e-3)
    assert excursionists(**_kwargs(75.4, "sunday", "rainy")).mean() == pytest.approx(daily[2], rel=1e-3)
    monthly = tables.monthly[(8, "sunday", "rainy")]
    got = excursionists(**_kwargs(75.4, "sunday", "rainy", ESTIMATOR_MONTHLY)).mean()
    assert got == pytest.approx(monthly[2], rel=1e-3)
    assert monthly[2] != pytest.approx(daily[2], rel=1e-3)


def test_presences_are_not_negative():
    tourists, _ = make_presence_factories(load_tables())
    assert tourists(**_kwargs(0.0)).ppf(0.0) >= 0.0


def test_weather_weights_follow_the_months_of_the_period():
    tables = load_tables()
    whole, june = weather_weights(tables), weather_weights(tables, 0, 30)
    assert sum(whole.values()) == pytest.approx(1.0)
    assert june == pytest.approx(tables.weather_freq[6])
    assert weather_weights(tables, 5, 5) == pytest.approx(tables.weather_freq[6])  # empty: one day


# ---------------------------------------------------------------- backend


@pytest.fixture(scope="module")
def backend():
    return MolvenoBackend()


def test_schema_has_the_period_but_not_the_estimator(backend):
    schema = {m.name: m for m in backend.parameter_schema()}
    assert schema[DAY_INDEX_NAME].kind == "distribution"
    assert schema[DAY_INDEX_NAME].default_range == (0.0, 122.0)
    assert set(schema["weather"].support) == {"good", "cool", "rainy"}
    assert ESTIMATOR_INDEX_NAME not in schema
    assert "season" not in schema


def test_estimator_is_a_backend_choice(backend):
    monthly = MolvenoBackend(presence_estimator=ESTIMATOR_MONTHLY)
    period = {DAY_INDEX_NAME: (61.0, 86.0), "weekday": "sunday", "weather": "good"}
    ml_sample = np.mean(backend.evaluate(period).samples_y)
    assert np.mean(monthly.evaluate(period).samples_y) != pytest.approx(ml_sample, rel=1e-3)
    with pytest.raises(ValueError, match="Unknown presence estimator"):
        MolvenoBackend(presence_estimator="nope")


def test_default_scenario_fixes_the_estimator_and_weights_the_weather(backend):
    from civic_digital_twins.dt_model import Scenario

    inp = backend.model.inputs
    scenario = backend._with_period(Scenario(backend.model, overrides={}))
    assert scenario.overrides[inp.cv_presence_estimator] == ESTIMATOR_ML
    assert scenario.overrides[inp.cv_weather] == pytest.approx(weather_weights(backend._tables))


def test_period_restricts_the_presence_samples(backend):
    june = backend.evaluate({DAY_INDEX_NAME: (0.0, 15.0)})
    august = backend.evaluate({DAY_INDEX_NAME: (61.0, 86.0)})
    assert np.mean(august.samples_x) > np.mean(june.samples_x) + 1500


def test_pinned_weather_is_kept(backend):
    rainy = backend.evaluate({"weather": "rainy"})
    good = backend.evaluate({"weather": "good"})
    assert np.mean(rainy.samples_y) < np.mean(good.samples_y) - 800


def test_empty_period_is_one_day(backend):
    out = backend.evaluate({DAY_INDEX_NAME: (61.0, 61.0)})
    assert len(out.samples_x) > 0


def test_startup_standalone_requires_files(tmp_path, monkeypatch):
    monkeypatch.setenv(molveno_data.DATA_DIR_ENV, str(tmp_path / "empty"))
    monkeypatch.setenv("DT_OVERTURISM_STANDALONE_MODE", "true")
    with pytest.raises(FileNotFoundError, match="molveno data files missing"):
        molveno_data.ensure_molveno_data()
