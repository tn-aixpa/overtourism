# SPDX-License-Identifier: Apache-2.0
"""
STEP 2b - Modal-shift estimation (D8).

Input : fazzon_daily, fazzon_presence_stats (step 2)
Output: <final>/fazzon/fazzon_modal_shift (one row per parameter and occupancy variant, with a
        bootstrap 90 % interval) and fazzon_modal_shift_daily (the 2026 days used)

Behaviour model (every element is an entry of ASSUMPTIONS.md):

* Cars at the 2025 price (5 euro) on a 2026 day: the 2025 level for its season and weather class
  (fazzon_presence_stats, A21) times a 2026 year effect, estimated on the 2026 days at 5 euro
  without the access rules (A22).
* Car response to the price (A23): cars(p) = cars(5) * exp(-beta * (p - 5)), p = average price
  paid per car (full-day and afternoon tickets, A24). With the 2026 rules the whole package
  (price, booking, traffic light, shuttle) is attributed to the price (A25).
* People who no longer come by car = cars(5) * occupancy(5) - cars(p) * occupancy(p) split among
  shuttle (bus up, measured), walk-up (bus down - bus up, A6, from 01/08/2026), motorbike and
  elsewhere (the rest). Occupancy: the operator's values (A2) or the constant 2.83 (A1); the split
  depends strongly on it, so both are reported.

Uncertainty: bootstrap over days (the regime days and, separately, the days of the year effect).
"""

import logging

import numpy as np
import pandas as pd

from model.fazzon.reference_datasets import PERSONS_PER_CAR, PRICE_2025, SEASONS

logger = logging.getLogger(__name__)

OCCUPANCY_VARIANTS = {  # (persons per car at 5 euro, with the 2026 rules)
    "operator": (2.5, 3.5),  # A2
    "constant": (PERSONS_PER_CAR, PERSONS_PER_CAR),  # A1
}
N_BOOTSTRAP = 2000
INTERVAL = (0.05, 0.95)
SEED = 0


def modal_shift_days(daily: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """2026 days with car counts, the cars expected in 2025 conditions and the price paid.

    ``role``: ``year_effect`` (5 euro, no access rules, no event; A22) or ``regime`` (access
    rules and shuttle running; D2, D3 excluded).
    """
    expected = (stats.set_index(["season", "weather"])["mean_car_visitors"] / PERSONS_PER_CAR)
    f = daily[(daily["year"] == 2026) & daily["cars"].notna() & daily["month"].isin(list(SEASONS))].copy()
    f = f.dropna(subset=["weather"])
    f["season"] = f["month"].map(SEASONS)
    f["cars_2025_conditions"] = [expected[(s, w)] for s, w in zip(f["season"], f["weather"])]
    half = f["cars_half"].fillna(0)
    f["price_paid"] = (f["cars_full"] * f["parking_price"] + half * f["parking_price_afternoon"]) / f["cars"]  # A24
    f["walk_up"] = (f["bus_down"] - f["bus_up"]).clip(lower=0)  # A6
    f["role"] = None
    f.loc[~f["regulated_parking"] & ~f["event_day"] & (f["parking_price"] == PRICE_2025), "role"] = "year_effect"
    f.loc[f["regulated_parking"] & f["shuttle"].notna() & ~f["event_day"], "role"] = "regime"
    cols = ["date", "season", "weather", "role", "cars", "cars_2025_conditions", "price_paid",
            "bus_up", "bus_down", "walk_up", "moto_persons", "parking_price_assumed"]
    return f.dropna(subset=["role"])[cols].reset_index(drop=True)


def _estimate(year: pd.DataFrame, regime: pd.DataFrame) -> dict[str, float]:
    """Point estimates on one (re)sample of days."""
    year_effect = year["cars"].sum() / year["cars_2025_conditions"].sum()
    cars5 = regime["cars_2025_conditions"] * year_effect
    response = regime["cars"].sum() / cars5.sum()
    price = (regime["price_paid"] * regime["cars"]).sum() / regime["cars"].sum()
    out = {
        "year_effect": year_effect,
        "car_response": response,
        "price_paid": price,
        "beta_per_euro": -np.log(response) / (price - PRICE_2025),
    }
    with_walk = regime[regime["bus_down"].notna()]  # walk-up known from 01/08/2026
    for name, (occ5, occ_regime) in OCCUPANCY_VARIANTS.items():
        for sub, suffix in ((regime, "all_days"), (with_walk, "with_walk_up")):
            lost = (sub["cars_2025_conditions"] * year_effect * occ5 - sub["cars"] * occ_regime).sum()
            shuttle = sub["bus_up"].sum() / lost
            moto = sub["moto_persons"].fillna(0).sum() / lost
            out[f"{name}__{suffix}__lost_per_day"] = lost / len(sub)
            out[f"{name}__{suffix}__share_shuttle"] = shuttle
            out[f"{name}__{suffix}__share_moto"] = moto
            if suffix == "with_walk_up":
                walk = sub["walk_up"].sum() / lost
                out[f"{name}__{suffix}__share_walk_up"] = walk
                out[f"{name}__{suffix}__share_elsewhere"] = 1.0 - shuttle - walk - moto
    return out


def estimate_modal_shift(daily: pd.DataFrame, stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(estimates, days)``: one row per parameter with the point estimate and the
    bootstrap interval, and the days used."""
    days = modal_shift_days(daily, stats)
    year = days[days["role"] == "year_effect"]
    regime = days[days["role"] == "regime"]
    point = _estimate(year, regime)

    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(N_BOOTSTRAP):
        y = year.iloc[rng.integers(0, len(year), len(year))]
        r = regime.iloc[rng.integers(0, len(regime), len(regime))]
        boot.append(_estimate(y, r))
    boot = pd.DataFrame(boot)

    rows = []
    for key, value in point.items():
        variant, sample, parameter = key.split("__") if "__" in key else ("", "", key)
        lo, hi = boot[key].quantile(list(INTERVAL))
        rows.append({"parameter": parameter, "occupancy": variant, "days": sample,
                     "estimate": value, "low": lo, "high": hi})
    estimates = pd.DataFrame(rows)
    estimates["n_year_effect_days"] = len(year)
    estimates["n_regime_days"] = len(regime)
    logger.info("Modal shift: car response %.2f (%.2f-%.2f), beta %.3f per euro",
                point["car_response"], *boot["car_response"].quantile(list(INTERVAL)), point["beta_per_euro"])
    return estimates, days
