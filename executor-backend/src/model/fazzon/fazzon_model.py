"""Fazzon (Lago dei Caprioli, Val di Sole) overtourism model definition.

The model uses the ``@define`` / ``compute()`` contract throughout — no
``legacy=True`` escape hatches.  Each concern sub-model declares its I/O
contract via ``@inputs`` / ``@outputs`` dataclasses and implements a single
``compute()`` method.  The root :class:`FazzonModel` wires all sub-models
inside its own ``compute()`` and exposes scenario-accessible indexes as
plain attributes set there.

Sub-models:

:class:`ParkingModel`
    Simultaneous car occupancy vs. policy cap (≈ 150 simultaneous spaces).

:class:`RoadModel`
    Daily vehicle passages (cars + shuttle buses) vs. effective road capacity
    under semaphore control.  Computes a **road cascade factor** (in [0, 1])
    and exposes it via ``Outputs.cascade_road`` for downstream sub-models.

:class:`FoodModel`
    Peak simultaneous diners (after road cascade) vs. total seat capacity
    (4 venues, 520 seats).  Takes ``cascade_road`` from :class:`RoadModel`.

:class:`FazzonModel`
    Root model — owns the season CV, two PVs, all ``i_*`` defaults, and wires the
    three sub-models.  Call :meth:`FazzonModel.default_inputs` to obtain a
    ready-to-use :class:`~FazzonModel.Inputs` instance.

Every default value refers to an entry of the Fazzon assumptions register,
``data_preparation/model/fazzon/ASSUMPTIONS.md`` (IDs ``A*`` / ``D*``).  The
lakeside constraint was dropped (D6).

Context variables
-----------------
cv_season   : june / july / august / september   (from 2025 parking data; A14)

Presence variables
------------------
pv_visitors_car   : daily car-mode visitors (calibrated; 2025 parking data; A1)
pv_visitors_other : daily non-car visitors  (ASSUMPTION A3, A4)

No day-type CV (D5); no weather CV yet (D7, planned).
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from civic_digital_twins.dt_model import (
    CategoricalIndex,
    ConditionalDistributionIndex,
    DistributionIndex,
    GenericIndex,
    Index,
    Model,
    define,
    graph,
    inputs,
    outputs,
)
from scipy import stats

from .fazzon_presence_stats import presence_inputs

# ---------------------------------------------------------------------------
# Constraint
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Constraint:
    """Named pairing of a usage formula index and a capacity index."""

    name: str
    usage: Index
    capacity: Index


# ---------------------------------------------------------------------------
# ParkingModel
# ---------------------------------------------------------------------------


@define("Parking")
class ParkingModel(Model):
    """Concern sub-model — simultaneous parking occupancy.

    Converts daily car-mode visitor count to a simultaneous occupancy estimate
    using the observed occupancy ratio (dwell_time / operating_window = 0.555),
    then compares against the policy cap.

    Formula::

        pv_cars = pv_visitors_car / i_xa_persons_per_car
        i_u_parking = pv_cars * i_occupancy_ratio
    """

    @inputs
    class Inputs:
        """Contractual inputs of :class:`ParkingModel`."""

        pv_visitors_car: ConditionalDistributionIndex
        i_xa_persons_per_car: Index
        i_occupancy_ratio: Index
        i_c_parking: DistributionIndex

    @outputs
    class Outputs:
        """Contractual outputs of :class:`ParkingModel`."""

        i_u_parking: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute simultaneous parking occupancy from daily car-mode visitors."""
        i_u_parking = Index(
            "parking usage",
            inputs.pv_visitors_car
            / inputs.i_xa_persons_per_car
            * inputs.i_occupancy_ratio,
        )
        self.constraint = Constraint(
            name="parking", usage=i_u_parking, capacity=inputs.i_c_parking
        )
        return ParkingModel.Outputs(i_u_parking=i_u_parking)


# ---------------------------------------------------------------------------
# RoadModel
# ---------------------------------------------------------------------------


@define("Road")
class RoadModel(Model):
    """Concern sub-model — daily vehicle passages on the Fazzon access road.

    The access road (4.3 km, single-lane with semaphore under shuttle
    operation) carries both private cars and shuttle buses.  The daily vehicle
    passage count is compared to an effective road capacity.

    A **road cascade factor** is computed here and exposed via
    ``Outputs.cascade_road`` for downstream use in :class:`FoodModel`::

        cascade_road = min(1, i_c_road / i_u_road)

    When the road is not saturated the factor is 1.0; when saturated it
    represents the fraction of intended visitors that can physically reach
    the lake.

    Formula::

        road_veh_cars = (pv_visitors_car + pv_visitors_other)
                        * i_car_mode_share / i_xa_persons_per_car
                        * i_xo_road_trips
        i_u_road = road_veh_cars + i_shuttle_daily_trips
    """

    @inputs
    class Inputs:
        """Contractual inputs of :class:`RoadModel`."""

        pv_visitors_car: ConditionalDistributionIndex
        pv_visitors_other: ConditionalDistributionIndex
        i_car_mode_share: Index
        i_xa_persons_per_car: Index
        i_xo_road_trips: Index
        i_shuttle_daily_trips: Index
        i_c_road: DistributionIndex

    @outputs
    class Outputs:
        """Contractual outputs of :class:`RoadModel`."""

        i_u_road: Index
        cascade_road: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute road vehicle demand, cascade factor, and road constraint."""
        _road_vehicle_demand = (
            (inputs.pv_visitors_car + inputs.pv_visitors_other)
            * inputs.i_car_mode_share
            / inputs.i_xa_persons_per_car
            * inputs.i_xo_road_trips
            + inputs.i_shuttle_daily_trips
        )
        i_u_road = Index("road usage", _road_vehicle_demand)

        # Cascade: fraction of visitors reaching the lake when road is saturated.
        # 1e-10 guard: piecewise evaluates all branches eagerly in numpy; the
        # ratio branch is only used when demand > capacity (> 0), so the
        # denominator epsilon only matters at demand≈0 where the condition
        # is False anyway — it prevents a RuntimeWarning there.
        cascade_road = Index(
            "road cascade factor",
            graph.piecewise(
                (
                    inputs.i_c_road / (_road_vehicle_demand + 1e-10),
                    _road_vehicle_demand > inputs.i_c_road,
                ),
                (1.0, True),
            ),
        )

        self.constraint = Constraint(
            name="road", usage=i_u_road, capacity=inputs.i_c_road
        )
        return RoadModel.Outputs(i_u_road=i_u_road, cascade_road=cascade_road)


# ---------------------------------------------------------------------------
# FoodModel
# ---------------------------------------------------------------------------


@define("Food")
class FoodModel(Model):
    """Concern sub-model — daily restaurant demand at the 4 lake venues.

    Models peak simultaneous diners as a fraction of visitors reaching the
    lake (after road cascade), divided by the daily seat turnover.  The
    constraint is peak simultaneous demand ≤ total seats (520).

    Formula::

        pv_visitors_at_lake = (pv_visitors_car + pv_visitors_other) * cascade_road
        i_u_food = pv_visitors_at_lake * i_u_food_fraction / i_xo_food_sittings
    """

    @inputs
    class Inputs:
        """Contractual inputs of :class:`FoodModel`."""

        pv_visitors_car: ConditionalDistributionIndex
        pv_visitors_other: ConditionalDistributionIndex
        cascade_road: Index
        i_u_food_fraction: Index
        i_xo_food_sittings: Index
        i_c_seats: Index

    @outputs
    class Outputs:
        """Contractual outputs of :class:`FoodModel`."""

        i_u_food: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute peak simultaneous diners after road cascade."""
        pv_visitors_at_lake = (
            inputs.pv_visitors_car + inputs.pv_visitors_other
        ) * inputs.cascade_road
        i_u_food = Index(
            "food usage",
            pv_visitors_at_lake * inputs.i_u_food_fraction / inputs.i_xo_food_sittings,
        )
        self.constraint = Constraint(
            name="food", usage=i_u_food, capacity=inputs.i_c_seats
        )
        return FoodModel.Outputs(i_u_food=i_u_food)


# ---------------------------------------------------------------------------
# FazzonModel
# ---------------------------------------------------------------------------


@define("Fazzon")
class FazzonModel(Model):
    """Root overtourism model for Fazzon — wires the three concern sub-models.

    Instantiate via :meth:`default_inputs`::

        model = FazzonModel(inputs=FazzonModel.default_inputs())

    Callers can override any parameter via :class:`~dt_model.Scenario`
    overrides targeting the scalar indexes exposed as instance attributes
    (e.g. ``model.i_c_parking``, ``model.i_shuttle_daily_trips``).  These
    attributes are aliased from ``model.inputs.*`` inside ``compute()`` and
    refer to the same index object, so scenario overrides work identically
    whether you target ``model.i_c_parking`` or ``model.inputs.i_c_parking``.

    Context variables
    -----------------
    cv_season   : june / july / august / september   (prior: proportional to observed days, A14)

    Presence variables
    ------------------
    pv_visitors_car   : daily car-mode visitors (axis x on the sustainability field)
    pv_visitors_other : daily non-car visitors  (axis y on the sustainability field)

    The assumptions (``A*`` IDs) are listed, with their status, in
    ``data_preparation/model/fazzon/ASSUMPTIONS.md``.
    """

    @inputs
    class Inputs:
        """All domain parameters of :class:`FazzonModel`."""

        # Context variables
        cv_season: CategoricalIndex
        # Presence distributions
        pv_visitors_car: ConditionalDistributionIndex
        pv_visitors_other: ConditionalDistributionIndex
        # Conversion parameters
        i_xa_persons_per_car: Index
        i_occupancy_ratio: Index
        i_xo_road_trips: Index
        i_car_mode_share: Index
        # Shuttle
        i_shuttle_daily_trips: Index
        # Food parameters
        i_u_food_fraction: Index
        i_xo_food_sittings: Index
        # Capacity parameters
        i_c_parking: DistributionIndex
        i_c_road: DistributionIndex
        i_c_seats: Index

    @outputs
    class Outputs:
        """Contractual outputs of :class:`FazzonModel`."""

        usage_indexes: list[GenericIndex]

    @classmethod
    def default_inputs(cls) -> FazzonModel.Inputs:
        """Return default domain inputs for the 2025 regulated season.

        All defaults are calibrated from 2025 Fazzon parking ticket data and
        the 2022 EETRA modal-split survey.  Values marked ASSUMPTION have not
        been confirmed by domain experts.

        Override individual fields with :func:`dataclasses.replace` or supply
        a :class:`~dt_model.Scenario` at evaluation time.

        The presence statistics are read from the Fazzon data folder at every call
        (see :mod:`src.model.fazzon.fazzon_data`).
        """
        season, car_visitors_stats, other_visitors_stats = presence_inputs()
        cv_season = CategoricalIndex("season", {s: season[s] for s in season})  # A14

        pv_visitors_car = ConditionalDistributionIndex(
            "visitors_car",
            [cv_season],
            car_visitors_stats,
        )
        pv_visitors_other = ConditionalDistributionIndex(
            "visitors_other",
            [cv_season],
            other_visitors_stats,
        )

        return cls.Inputs(
            cv_season=cv_season,
            pv_visitors_car=pv_visitors_car,
            pv_visitors_other=pv_visitors_other,
            # Conversion parameters
            i_xa_persons_per_car=Index("persons per car", 2.83),  # A1: NetMobility 2021
            i_occupancy_ratio=Index(
                "parking occupancy ratio", 0.555
            ),  # A8: 4.44h/8h, 2021 calibration
            i_xo_road_trips=Index(
                "road round-trip multiplier", 2.0
            ),  # A9: in + out = 2 passages
            i_car_mode_share=Index(
                "car mode share", 0.69
            ),  # A3: 1 - 0.31 (2022 EETRA); to be replaced (D8); scenario-overridable
            # Shuttle
            i_shuttle_daily_trips=Index("shuttle daily vehicle passages", 0.0),
            # 0 = no shuttle (2025 baseline: the Greenbus pilot only ran 28/06-20/07, D1).
            # Override in scenarios: 32 = 2 buses × 8 A/R trips × 2 directions.
            # Food
            i_u_food_fraction=Index(
                "restaurant usage fraction", 0.55
            ),  # A10: >50% (EETRA 2022)
            i_xo_food_sittings=Index(
                "food seat turnover", 2.0
            ),  # A11: ~2 sittings/seat/day
            # Capacities
            # A12 — Parking: policy cap is 150 simultaneous cars.  Triangular captures
            # uncertainty about whitelist slots (30 reserved), enforcement, and
            # peak-hour variability.  Mode=200 = physical upper bound.
            i_c_parking=DistributionIndex(
                "parking capacity (simultaneous cars)",
                stats.triang,
                {"loc": 150.0, "scale": 100.0, "c": 0.5},  # mode=200, range [150, 250]
            ),
            # A13 — Road: effective daily vehicle capacity under semaphore control.
            # Peak observed 2025: 422 cars × 2 trips = 844 passages; road not saturated.
            i_c_road=DistributionIndex(
                "road capacity (vehicle passages/day)",
                stats.uniform,
                {
                    "loc": 800.0,
                    "scale": 600.0,
                },  # ASSUMPTION uniform[800, 1400] veh passages/day
            ),
            # A15 — Restaurant seats: 4 venues, 520 total (from data — no uncertainty).
            i_c_seats=Index("restaurant seats", 520.0),
        )

    def compute(self, inputs: Inputs) -> Outputs:
        """Wire the three concern sub-models from flat inputs."""
        parking = ParkingModel(
            inputs=ParkingModel.Inputs(
                pv_visitors_car=inputs.pv_visitors_car,
                i_xa_persons_per_car=inputs.i_xa_persons_per_car,
                i_occupancy_ratio=inputs.i_occupancy_ratio,
                i_c_parking=inputs.i_c_parking,
            )
        )
        road = RoadModel(
            inputs=RoadModel.Inputs(
                pv_visitors_car=inputs.pv_visitors_car,
                pv_visitors_other=inputs.pv_visitors_other,
                i_car_mode_share=inputs.i_car_mode_share,
                i_xa_persons_per_car=inputs.i_xa_persons_per_car,
                i_xo_road_trips=inputs.i_xo_road_trips,
                i_shuttle_daily_trips=inputs.i_shuttle_daily_trips,
                i_c_road=inputs.i_c_road,
            )
        )
        food = FoodModel(
            inputs=FoodModel.Inputs(
                pv_visitors_car=inputs.pv_visitors_car,
                pv_visitors_other=inputs.pv_visitors_other,
                cascade_road=road.outputs.cascade_road,
                i_u_food_fraction=inputs.i_u_food_fraction,
                i_xo_food_sittings=inputs.i_xo_food_sittings,
                i_c_seats=inputs.i_c_seats,
            )
        )
        self.constraints = [
            parking.constraint,
            road.constraint,
            food.constraint,
        ]

        # Expose scenario-accessible indexes as direct attributes.
        # These are identity-aliases of the corresponding inputs fields:
        # scenario overrides work identically on either name.
        self.i_c_parking = inputs.i_c_parking
        self.i_c_road = inputs.i_c_road
        self.i_c_seats = inputs.i_c_seats
        self.i_car_mode_share = inputs.i_car_mode_share
        self.i_shuttle_daily_trips = inputs.i_shuttle_daily_trips
        # Cascade factor is computed inside RoadModel; expose for inspection.
        self.i_cascade_road = road.outputs.cascade_road

        return FazzonModel.Outputs(usage_indexes=[c.usage for c in self.constraints])
