# SPDX-License-Identifier: Apache-2.0

"""
Test configuration and fixtures for dt_studio tests.

This module provides pytest fixtures and configuration specifically
for the dt_studio test suite.
"""

import os
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import required modules for real model setup
from overtourism.dt_studio.manager.config.classes import (
    Grid,
    ModelOutput,
    Sampler,
    Situation,
    StoreConfig,
)
from overtourism.model.setup import (
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
    build_output,
)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@pytest.fixture
def real_model_config(tmp_path):
    """
    Provide real model configuration to eliminate code duplication.

    This fixture creates the complete real model setup (Grid, Sampler, Situations)
    that was previously duplicated across multiple test files.
    """
    # Grid configuration
    (t_max, e_max) = (10000, 10000)
    (t_sample, e_sample) = (100, 100)
    tourists_line = np.linspace(0, t_max, t_sample + 1)
    excursionists_line = np.linspace(0, e_max, e_sample + 1)
    x, y = np.meshgrid(tourists_line, excursionists_line)
    grid = {PV_tourists: tourists_line, PV_excursionists: excursionists_line}
    target_presence_samples = 1200

    grid_obj = Grid(grid, x, y, t_max, e_max)

    # Sampler configuration
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
    sampler_obj = Sampler([e_sample_dict, t_sample_dict])

    # Situations configuration
    situations_list = [
        Situation(None, "Condizioni medie di riferimento", S_Base),
        Situation("good_weather", "Meteo > Bel tempo", S_Good_Weather),
        Situation("bad_weather", "Meteo > Cattivo tempo", S_Bad_Weather),
        Situation("high_season", "Stagione > Alta", S_High_Season),
        Situation("low_season", "Stagione > Bassa", S_Low_Season),
        Situation("weekend_days", "Giorni settimana > Fine settimana", S_Weekend_Days),
        Situation(
            "working_days", "Giorni settimana > Giorni lavorativi", S_Working_Days
        ),
    ]

    # Store configuration pointing to temporary directory
    export_folder = tmp_path / "test_problems"
    export_folder.mkdir(exist_ok=True)
    store_config_obj = StoreConfig("local", {"folder": export_folder})

    return {
        "abstract_model": M_Base,
        "kpi_function": build_output,
        "output_data": ModelOutput,
        "grid": grid_obj,
        "sampler": sampler_obj,
        "situations": situations_list,
        "store_config": store_config_obj,
        "export_folder": export_folder,
    }


@pytest.fixture
def problem_manager_deps(real_model_config):
    """Dependencies for ProblemManager tests."""
    return {
        "abstract_model": real_model_config["abstract_model"],
        "kpi_function": real_model_config["kpi_function"],
        "output_data": real_model_config["output_data"],
        "sampler": real_model_config["sampler"],
        "situations": real_model_config["situations"],
        "grid": real_model_config["grid"],
        "store": real_model_config["store_config"],
    }


@pytest.fixture
def scenario_manager_deps(real_model_config):
    """Dependencies for ScenarioManager tests."""
    return {
        "abstract_model": real_model_config["abstract_model"],
        "kpi_function": real_model_config["kpi_function"],
        "output_data": real_model_config["output_data"],
        "sampler": real_model_config["sampler"],
        "situations": real_model_config["situations"],
        "grid": real_model_config["grid"],
    }


@pytest.fixture
def mock_io_builder():
    """Mock io_builder for tests that need store operations."""
    with patch("overtourism.dt_studio.manager.problem.manager.io_builder") as mock:
        mock_store = Mock()
        mock.create.return_value = mock_store
        yield mock_store


@pytest.fixture
def viewer_mocks():
    """Common mocks for viewer tests."""
    with (
        patch("overtourism.dt_studio.viewer.viewer.load_yaml") as mock_load_yaml,
        patch(
            "overtourism.dt_studio.viewer.viewer.build_indexes_from_config"
        ) as mock_build_indexes,
    ):
        mock_load_yaml.return_value = {
            "indexes": [
                {
                    "index_id": "test_index",
                    "group": "test_group",
                    "index_type": "constant",
                }
            ]
        }
        mock_build_indexes.return_value = [Mock()]

        yield {"load_yaml": mock_load_yaml, "build_indexes": mock_build_indexes}
