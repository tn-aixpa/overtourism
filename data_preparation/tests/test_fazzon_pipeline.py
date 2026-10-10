import numpy as np
import pandas as pd
import pytest

from model.fazzon.config import load_config
from model.fazzon.reference_datasets import (
    PERSONS_PER_CAR,
    build_fazzon_daily,
    build_presence_stats,
)
from model.fazzon.standardize import (
    road_counts_daily,
    standardize_greenbus_2025,
    standardize_parking,
    standardize_road_counts,
)


def test_config_lists_all_sources():
    config = load_config()
    assert {"parking", "road_counts", "vodafone_attendences", "meteo_station", "meteo_bulletin"} <= set(config.sources)


def _parking_sheet(tmp_path):
    header = [[None] * 18, [None] * 18, ["DATA"] + [None] * 17]
    rows = [
        # 2025 block (8 cols) | 2026 block (8 cols) | note | note2
        ["2025-07-02", 99, None, None, 53, 0, None, None,
         "2026-07-01", 47, None, None, None, 0, 0, None, None, None],
        ["2025-07-03", 176, None, None, 38, 0, None, None,
         "2026-07-02", "5days", None, None, None, None, None, None, "5days - evento", None],
        ["2025-08-12", 422, None, None, None, 0, None, None,
         "2026-08-11", 73, None, 77, None, 428, 16, None, "487 (discesa)", "Brutto tempo"],
    ]
    path = tmp_path / "parking.xlsx"
    pd.DataFrame(header + rows).to_excel(path, header=False, index=False)
    return path


def test_standardize_parking_reads_both_years_and_bus_down(tmp_path):
    out = standardize_parking(_parking_sheet(tmp_path)).set_index("date")
    assert out.loc["2025-07-02", "cars_full"] == 99
    assert out.loc["2025-07-02", "bus_up"] == 53
    assert np.isnan(out.loc["2026-07-02", "cars_full"])  # "5days" is not a count
    assert out.loc["2026-07-02", "note"] == "5days - evento"
    assert out.loc["2026-08-11", "cars_half"] == 77
    assert out.loc["2026-08-11", "bus_down"] == 487
    assert out.loc["2026-08-11", "note"] == "Brutto tempo"


def test_greenbus_rows_after_pilot_are_flagged(tmp_path):
    sheet = pd.DataFrame([
        [None, "RIEPILOGO", None, None, None, None],
        [None, "Data", "Ticket Venduti", None, "Data", "Ticket Venduti"],
        [None, "2025-07-20", 90, None, None, None],
        [None, "2025-08-06", 90, None, None, None],
    ])
    path = tmp_path / "bus.xlsx"
    sheet.to_excel(path, header=False, index=False)
    out = standardize_greenbus_2025(path).set_index("date")
    assert not out.loc["2025-07-20", "other_area"]
    assert out.loc["2025-08-06", "other_area"]


def test_road_counts_keep_cars_only_and_flag_dead_sensors(tmp_path):
    lines = ['"PUNTO","DATA","ORA","DIREZIONE","CLASSE","VELOCITA","N_VEICOLI"']
    for hour in range(24):
        lines.append(f'"110","01/08/2025","{hour:02d}",2,2,5,10')   # cars
        lines.append(f'"110","01/08/2025","{hour:02d}",2,2,6,5')    # cars, another speed band
        lines.append(f'"110","01/08/2025","{hour:02d}",2,9,5,1')    # bus: excluded
        lines.append(f'"110","01/08/2025","{hour:02d}",1,2,5,0')    # direction 1 reads zero
    path = tmp_path / "road.csv"
    path.write_text("\n".join(lines))
    stations = pd.DataFrame({"kind": ["road"], "code": ["110"], "name": ["pellizzano"]})
    hourly = standardize_road_counts(path, stations)
    assert set(hourly["station"]) == {"pellizzano"}
    assert hourly.loc[hourly["direction"] == 2, "cars"].eq(15).all()
    daily = road_counts_daily(hourly).set_index("direction")
    assert daily.loc[2, "cars"] == 24 * 15 and daily.loc[2, "valid"]
    assert not daily.loc[1, "valid"]


def _tables_for_daily(cars_by_date: dict[str, float]) -> dict[str, pd.DataFrame]:
    dates = pd.to_datetime(list(cars_by_date))
    parking = pd.DataFrame({"date": dates, "cars_full": list(cars_by_date.values()), "cars_half": np.nan,
                            "moto_persons": 0.0, "bus_up": np.nan, "bus_down": np.nan, "note": pd.NA})
    empty_dates = pd.DatetimeIndex([], name="date")
    hourly = pd.DataFrame({"station": pd.Series(dtype=str), "date": empty_dates, "direction": pd.Series(dtype=int),
                           "hour": pd.Series(dtype=int), "cars": pd.Series(dtype=int)})
    daily_road = pd.DataFrame({
        "station": [s for s in ("pellizzano", "pracorno", "celledizzo", "tonale", "revo") for _ in (1, 2)],
        "date": dates[0], "direction": [1, 2] * 5, "cars": 1000, "valid": True,
    })
    return {
        "fazzon_parking": parking,
        "fazzon_parking_2025": parking[["date"]].assign(cars=parking["cars_full"]),
        "fazzon_greenbus_2025": pd.DataFrame({"date": dates[:1], "tickets": [9.0], "other_area": [False]}),
        "meteo_station": pd.DataFrame({"date": dates, "tmin": 10.0, "tmax": 25.0, "rain_mm": 0.0, "quality": 145}),
        "meteo_bulletin": pd.DataFrame({"date": dates, "weather_desc": "Sereno", "tmin": 10, "tmax": 25, "rain_mm": 0}),
        "road_counts_hourly": hourly,
        "road_counts_daily": daily_road,
        "vodafone_presences": pd.DataFrame({"date": dates, "area": "apt_valdisole", "presences": 100.0}),
        "bike_counts": pd.DataFrame({"date": dates, "sensor": "TFXV3R06", "bikes_peds": 50}),
        "stations": pd.DataFrame({"kind": ["bike"], "code": ["TFXV3R06"], "role": ["near_fazzon"]}),
        "tgc_accessi": pd.DataFrame({"date": dates, "service": "Passo Tonale", "entrants": 3}),
    }


def test_presence_stats_per_season():
    july = {str(d.date()): (300.0 if d.dayofweek == 6 else 100.0) for d in pd.date_range("2025-07-01", "2025-07-31")}
    august = {str(d.date()): 200.0 for d in pd.date_range("2025-08-01", "2025-08-31")}
    august["2025-08-28"] = 27.0  # D4: bad-weather days are part of the calibration
    daily = build_fazzon_daily(_tables_for_daily({**july, **august}))
    stats = build_presence_stats(daily).set_index("cluster_name")
    assert list(stats.index) == ["july", "august"]
    assert stats.loc["august", "n_days"] == 31
    assert stats.loc["august", "mean_car_visitors"] == pytest.approx(np.mean(list(august.values())) * PERSONS_PER_CAR)
    assert stats["freq_rel"].sum() == pytest.approx(1.0)
    assert "day_type" not in stats.columns


def test_parking_price_per_day():
    days = ["2025-07-10", "2026-06-30", "2026-07-10", "2026-08-21", "2026-09-05"]
    daily = build_fazzon_daily(_tables_for_daily({d: 100.0 for d in days})).set_index("date")
    price = daily.loc[pd.to_datetime(days), ["parking_price", "parking_price_afternoon", "parking_price_assumed"]]
    assert price["parking_price"].tolist() == [5.0, 5.0, 20.0, 5.0, 5.0]
    assert price["parking_price_afternoon"].tolist() == [5.0, 5.0, 10.0, 5.0, 5.0]
    assert price["parking_price_assumed"].tolist() == [False, True, False, False, True]
    assert np.isnan(daily.loc["2026-07-11", "parking_price"])  # no parking data that day
