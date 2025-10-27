# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from civic_digital_twins.dt_model import (
    CategoricalContextVariable,
    Constraint,
    PresenceVariable,
    SymIndex,
    UniformCategoricalContextVariable,
)
from civic_digital_twins.dt_model.internal.sympyke import Eq, Piecewise, Symbol
from civic_digital_twins.dt_model.model.abstract_model import AbstractModel
from civic_digital_twins.dt_model.symbols.index import (
    ConstIndex,
    LognormDistIndex,
    TriangDistIndex,
    UniformDistIndex,
)

# Comment below if using platform integration
from overtourism.model.presence_stats_local import (
    excursionist_presences_stats,
    season,
    tourist_presences_stats,
    weather,
    weekday,
)

# Uncomment below if using platform integration
# from overtourism.model.presence_stats import excursionist_presences_stats, get_frequencies, tourist_presences_stats
# season, weather, weekday = get_frequencies()

CV_weekday = UniformCategoricalContextVariable("weekday", [Symbol(v) for v in weekday])
CV_season = CategoricalContextVariable(
    "season", {Symbol(v): season[v] for v in season.keys()}
)
CV_weather = CategoricalContextVariable(
    "weather", {Symbol(v): weather[v] for v in weather.keys()}
)

# Presence variables

TOURISTS = "Turisti"
EXCURSIONISTS = "Escursionisti"

PV_tourists = PresenceVariable(
    TOURISTS, [CV_weekday, CV_season, CV_weather], tourist_presences_stats
)
PV_excursionists = PresenceVariable(
    EXCURSIONISTS, [CV_weekday, CV_season, CV_weather], excursionist_presences_stats
)

# Capacity indexes

I_C_parking = UniformDistIndex("available_parking_spaces", 350.0, 100.0)
I_C_beach = UniformDistIndex("available_beach_seats", 6000.0, 1000.0)
I_C_accommodation = LognormDistIndex("available_beds", s=0.125, loc=0.0, scale=5000.0)
I_C_food = TriangDistIndex("available_restaurant_seats", loc=2400.0, scale=800.0, c=0.5)

# Usage indexes

I_U_tourists_parking = ConstIndex(
    "tourists_parking_percentage", 2.0
)  # "Percentuale di turisti che usano i parcheggi"
I_U_excursionists_parking = ConstIndex("excursionists_parking_percentage", 80.0)

# I_U_tourists_beach = SymIndex(
#     # "Percentuale di turisti che vanno in spiaggia",
#     "tourists_beach_percentage",
#     Piecewise(
#         (15.0, Eq(CV_weather, Symbol("bad"))),
#         (50.0, True),
#     ),
#     cvs=[CV_weather],
# )
I_U_tourists_beach = ConstIndex(
    # "Percentuale di turisti che vanno in spiaggia",
    "tourists_beach_percentage",
    50.0,
)
# I_U_excursionists_beach = SymIndex(
#     # "Percentuale di escursionisti che vanno in spiaggia",
#     "excursionists_beach_percentage",
#     Piecewise(
#         (35.0, Eq(CV_weather, Symbol("bad"))),
#         (80.0, True),
#     ),
#     cvs=[CV_weather],
# )
I_U_excursionists_beach = ConstIndex(
    # "Percentuale di escursionisti che vanno in spiaggia",
    "excursionists_beach_percentage",
    80.0,
)

I_U_tourists_accommodation = ConstIndex("tourists_accommodation_percentage", 90.0)

I_U_tourists_food = ConstIndex("tourists_restaurant_percentage", 20.0)
I_U_excursionists_food = SymIndex(
    "excursionists_restaurant_percentage",
    # "Percentuale di escursionisti che usano i ristoranti",
    Piecewise(
        (80.0, Eq(CV_weather, Symbol("bad"))),
        (40.0, True),
    ),
    cvs=[CV_weather],
)

# Conversion indexes

I_Xa_tourists_per_vehicle = ConstIndex("tourists_per_vehicle_average", 2.5)
I_Xa_excursionists_per_vehicle = ConstIndex("excursionists_per_vehicle_average", 2.5)
I_Xo_tourists_parking = ConstIndex("tourists_parking_turnover", 1.05)
I_Xo_excursionists_parking = ConstIndex("excursionists_parking_turnover", 3.50)

I_Xo_tourists_beach = UniformDistIndex("tourists_beach_turnover", loc=1.0, scale=2.0)
I_Xo_excursionists_beach = ConstIndex("excursionists_beach_turnover", 1.05)

I_Xa_tourists_accommodation = ConstIndex(
    "tourists_accommodation_allocation_factor", 85.0
)

I_Xa_visitors_food = ConstIndex("visitors_food_allocation_factor", 90.0)
I_Xo_visitors_food = ConstIndex("visitors_food_turnover", 2.0)

# Presence indexes

I_P_tourists_reduction_factor = ConstIndex("tourists_reduction_factor", 100.0)
I_P_excursionists_reduction_factor = ConstIndex("excursionists_reduction_factor", 100.0)

I_P_tourists_saturation_level = ConstIndex("tourists_saturation_level", 10000.0)
I_P_excursionists_saturation_level = ConstIndex(
    "excursionists_saturation_level", 10000.0
)

# Constraints
C_parking = Constraint(
    usage=(
        PV_tourists.node
        * I_U_tourists_parking.node
        / 100.0
        / (I_Xa_tourists_per_vehicle.node * I_Xo_tourists_parking.node)
        + PV_excursionists.node
        * I_U_excursionists_parking.node
        / 100.0
        / (I_Xa_excursionists_per_vehicle.node * I_Xo_excursionists_parking.node)
    ),
    capacity=I_C_parking,
    name="parcheggi",
)

C_beach = Constraint(
    usage=(
        PV_tourists.node * I_U_tourists_beach.node / 100.0 / I_Xo_tourists_beach.node
        + PV_excursionists.node
        * I_U_excursionists_beach.node
        / 100.0
        / I_Xo_excursionists_beach.node
    ),
    capacity=I_C_beach,
    name="spiaggia",
)

C_accommodation = Constraint(
    usage=(
        PV_tourists.node
        * (I_U_tourists_accommodation.node / 100.0)
        / (I_Xa_tourists_accommodation.node / 100.0)
    ),
    capacity=I_C_accommodation,
    name="alberghi",
)

C_food = Constraint(
    usage=(
        (
            PV_tourists.node * I_U_tourists_food.node / 100.0
            + PV_excursionists.node * I_U_excursionists_food.node / 100.0
        )
        / (I_Xa_visitors_food.node / 100.0 * I_Xo_visitors_food.node)
    ),
    capacity=I_C_food,
    name="ristoranti",
)


# Constant strings to label the models
BASE_MODEL = "Scenario di partenza"

# Base model
M_Base = AbstractModel(
    BASE_MODEL,
    [CV_weekday, CV_season, CV_weather],
    [PV_tourists, PV_excursionists],
    [
        I_U_tourists_parking,
        I_U_excursionists_parking,
        I_U_tourists_beach,
        I_U_excursionists_beach,
        I_U_tourists_accommodation,
        I_U_tourists_food,
        I_U_excursionists_food,
        I_Xa_tourists_per_vehicle,
        I_Xa_excursionists_per_vehicle,
        I_Xa_tourists_accommodation,
        I_Xo_tourists_parking,
        I_Xo_excursionists_parking,
        I_Xo_tourists_beach,
        I_Xo_excursionists_beach,
        I_Xa_visitors_food,
        I_Xo_visitors_food,
        I_P_tourists_reduction_factor,
        I_P_excursionists_reduction_factor,
        I_P_tourists_saturation_level,
        I_P_excursionists_saturation_level,
    ],
    [I_C_parking, I_C_beach, I_C_accommodation, I_C_food],
    [C_parking, C_beach, C_accommodation, C_food],
)


# Base situation
S_Base = {}

# Good weather situation
S_Good_Weather = {CV_weather: [Symbol("good")]}

# Bad weather situation
S_Bad_Weather = {CV_weather: [Symbol("bad")]}

# High season situation
S_High_Season = {CV_season: [Symbol("high"), Symbol("very high")]}

# Low season situation
S_Low_Season = {CV_season: [Symbol("low"), Symbol("mid")]}

# Weekend situation
S_Weekend_Days = {CV_weekday: [Symbol("saturday"), Symbol("sunday")]}

# Working days situation
S_Working_Days = {
    CV_weekday: [
        Symbol("monday"),
        Symbol("tuesday"),
        Symbol("wednesday"),
        Symbol("thursday"),
        Symbol("friday"),
    ]
}
