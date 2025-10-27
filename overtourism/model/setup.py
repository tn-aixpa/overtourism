# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np

from overtourism.dt_studio.manager.config.classes import (
    Grid,
    ModelOutput,
    Sampler,
    Situation,
    StoreConfig,
)
from overtourism.dt_studio.manager.problem.manager import ProblemManager
from overtourism.dt_studio.viewer.models import ViewSettings
from overtourism.dt_studio.viewer.viewer import ModelViewer
from overtourism.model.kpis import build_output
from overtourism.model.model import (
    C_accommodation,
    C_beach,
    C_food,
    C_parking,
    I_P_excursionists_reduction_factor,
    I_P_excursionists_saturation_level,
    I_P_tourists_reduction_factor,
    I_P_tourists_saturation_level,
    M_Base,
    PV_excursionists,
    PV_tourists,
    S_Bad_Weather,
    S_Base,
    S_Good_Weather,
    S_High_Season,
    S_Low_Season,
    S_Weekend_Days,
    S_Working_Days,
)

# Grid
(t_max, e_max) = (10000, 10000)
(t_sample, e_sample) = (100, 100)
tourists_line = np.linspace(0, t_max, t_sample + 1)
excursionists_line = np.linspace(0, e_max, e_sample + 1)
x, y = np.meshgrid(tourists_line, excursionists_line)
grid = {PV_tourists: tourists_line, PV_excursionists: excursionists_line}
target_presence_samples = 1200

grid = Grid(grid, x, y, t_max, e_max)

t_sample_dict = {
    "p_var": PV_tourists,
    "reduction_index_name": I_P_tourists_reduction_factor.name,
    "saturation_index_name": I_P_tourists_saturation_level.name,
    "target_presence_samples": target_presence_samples,
}
e_sample_dict = {
    "p_var": PV_excursionists,
    "reduction_index_name": I_P_excursionists_reduction_factor.name,
    "saturation_index_name": I_P_excursionists_saturation_level.name,
    "target_presence_samples": target_presence_samples,
}


# Add model to manager
sampler = Sampler([e_sample_dict, t_sample_dict])
BASE_CONTEXT = "Condizioni medie di riferimento"
situations = [
    Situation(None, "Condizioni medie di riferimento", S_Base),
    Situation("good_weather", "Meteo > Bel tempo", S_Good_Weather),
    Situation("bad_weather", "Meteo > Cattivo tempo", S_Bad_Weather),
    Situation("high_season", "Stagione > Alta", S_High_Season),
    Situation("low_season", "Stagione > Bassa", S_Low_Season),
    Situation("weekend_days", "Giorni settimana > Fine settimana", S_Weekend_Days),
    Situation("working_days", "Giorni settimana > Giorni lavorativi", S_Working_Days),
]
store_conf = StoreConfig(
    "local", {"folder": Path(__file__).parent.parent / "data" / "problems"}
)
problem_manager = ProblemManager(
    M_Base,
    build_output,
    ModelOutput,
    sampler,
    situations,
    grid,
    store_conf,
)


path = str(Path(__file__).parent / "indexes.yaml")
viewer = ModelViewer(path)
name_to_linestyle = {
    C_parking.name: "solid",
    C_accommodation.name: "dash",
    C_food.name: "dashdot",
    C_beach.name: "dot",
}
view = ViewSettings(
    x=PV_tourists.name, y=PV_excursionists.name, name_to_linestyle=name_to_linestyle
)
