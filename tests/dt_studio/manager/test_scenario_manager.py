# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from overtourism.dt_studio.manager.scenario.manager import ScenarioManager
from overtourism.dt_studio.manager.utils.exception import (
    ConfigurationError,
    ScenarioAlreadyExists,
    ScenarioDoesNotExist,
)
from overtourism.model.setup import PV_excursionists, PV_tourists, S_Good_Weather


class TestScenarioManagerRealConfig:
    """Test suite for ScenarioManager using real configuration from setup.py."""

    @pytest.fixture(autouse=True)
    def setup_method(self, real_model_config):
        """Set up test fixtures with real configuration from conftest fixture."""
        self.problem_id = "test_problem"

        # Get all configuration from the shared fixture
        self.grid = real_model_config["grid"]
        self.sampler = real_model_config["sampler"]
        self.situations = real_model_config["situations"]
        self.abstract_model = real_model_config["abstract_model"]
        self.output_function = real_model_config["kpi_function"]
        self.output_data = real_model_config["output_data"]

        # Real metadata mock with proper structure
        self.metadata = Mock()
        self.metadata.name = "Test Problem"
        self.metadata.description = "Test problem description"
        self.metadata.created = "2023-01-01T00:00:00"
        self.metadata.updated = "2023-01-01T00:00:00"

        self.scenario_manager = ScenarioManager(
            problem_id=self.problem_id,
            abstract_model=self.abstract_model,
            output_function=self.output_function,
            output_data=self.output_data,
            sampler=self.sampler,
            situations=self.situations,
            grid=self.grid,
            metadata=self.metadata,
        )

        yield

    def test_init_with_real_config(self):
        """Test ScenarioManager initialization with real configuration."""
        assert self.scenario_manager.problem_id == self.problem_id
        assert self.scenario_manager.abstract_model == self.abstract_model
        assert self.scenario_manager.output_function == self.output_function
        assert isinstance(self.scenario_manager.instantiated_models, dict)
        assert len(self.scenario_manager.instantiated_models) == 0
        assert self.scenario_manager.sampler == self.sampler
        assert self.scenario_manager.situations == self.situations
        assert self.scenario_manager.grid == self.grid

    def test_add_scenario_no_base_model(self):
        """Test adding scenario without base model raises error."""
        self.scenario_manager.abstract_model = None

        with pytest.raises(ConfigurationError):
            self.scenario_manager.add_scenario("test_scenario")

    @patch("overtourism.dt_studio.manager.scenario.manager.InstantiatedModel")
    @patch("overtourism.dt_studio.manager.scenario.manager.get_diff")
    @patch("overtourism.dt_studio.manager.scenario.manager.ScenarioMetadata")
    @patch("overtourism.dt_studio.manager.scenario.manager.ScenarioState")
    @patch("overtourism.dt_studio.manager.scenario.manager.get_timestamp")
    def test_add_scenario_success_with_real_config(
        self, mock_timestamp, mock_state, mock_metadata, mock_diff, mock_model
    ):
        """Test successfully adding a scenario with real configuration."""
        scenario_id = "test_scenario"
        mock_timestamp.return_value = "2023-01-01T00:00:00"
        mock_diff.return_value = {"index1": "value1"}

        self.scenario_manager.add_scenario(scenario_id)

        assert scenario_id in self.scenario_manager.instantiated_models
        # Verify InstantiatedModel was called with the real M_Base model
        # The get_diff() returns an empty dict by default, so expect that
        mock_model.assert_called_once_with(self.abstract_model, scenario_id, {})
        mock_state.assert_called_once()

    def test_add_scenario_already_exists(self):
        """Test adding scenario that already exists raises error."""
        scenario_id = "test_scenario"

        # Mock an existing scenario
        mock_scenario_state = Mock()
        mock_scenario_state.scenario_id = scenario_id
        self.scenario_manager.instantiated_models[scenario_id] = mock_scenario_state

        with pytest.raises(ScenarioAlreadyExists):
            self.scenario_manager.add_scenario(scenario_id)

    def test_update_scenario_not_exists(self):
        """Test updating non-existent scenario raises error."""
        with pytest.raises(ScenarioDoesNotExist):
            self.scenario_manager.update_scenario("nonexistent", {})

    @patch("overtourism.dt_studio.manager.scenario.manager.InstantiatedModel")
    @patch("overtourism.dt_studio.manager.scenario.manager.get_diff")
    @patch("overtourism.dt_studio.manager.scenario.manager.ScenarioMetadata")
    @patch("overtourism.dt_studio.manager.scenario.manager.ScenarioState")
    @patch("overtourism.dt_studio.manager.scenario.manager.get_timestamp")
    def test_update_scenario_success(
        self, mock_timestamp, mock_state, mock_metadata, mock_diff, mock_model
    ):
        """Test successfully updating a scenario."""
        scenario_id = "test_scenario"
        values = {"param1": "value1"}

        # Add existing scenario
        mock_existing = Mock()
        mock_existing.metadata.name = "Test"
        mock_existing.metadata.description = "Test desc"
        mock_existing.metadata.created = "2023-01-01"
        self.scenario_manager.instantiated_models[scenario_id] = mock_existing

        mock_timestamp.return_value = "2023-01-02T00:00:00"
        mock_diff.return_value = {"index1": "updated"}

        self.scenario_manager.update_scenario(scenario_id, values)

        mock_model.assert_called_once_with(self.abstract_model, scenario_id, values)
        mock_state.assert_called_once()

    def test_delete_scenario_success(self):
        """Test successfully removing a scenario."""
        scenario_id = "test_scenario"
        self.scenario_manager.instantiated_models[scenario_id] = Mock()

        self.scenario_manager.delete_scenario(scenario_id)

        assert scenario_id not in self.scenario_manager.instantiated_models

    def test_delete_scenario_not_exists(self):
        """Test removing non-existent scenario raises error."""
        with pytest.raises(ScenarioDoesNotExist):
            self.scenario_manager.delete_scenario("nonexistent")

    @patch("overtourism.dt_studio.manager.scenario.manager.scenario_values")
    def test_export_scenario(self, mock_scenario_values):
        """Test exporting a scenario."""
        scenario_id = "test_scenario"
        mock_state = Mock()
        mock_state.model.get_values.return_value = {"param": "value"}
        mock_state.metadata = Mock()
        self.scenario_manager.instantiated_models[scenario_id] = mock_state

        expected_result = Mock()
        mock_scenario_values.return_value = expected_result

        result = self.scenario_manager.export_scenario(scenario_id)

        assert result == expected_result
        mock_scenario_values.assert_called_once_with(
            scenario_id=scenario_id,
            values={"param": "value"},
            metadata=mock_state.metadata,
        )

    def test_import_scenario_already_exists(self):
        """Test importing scenario that already exists is skipped."""
        scenario_data = Mock()
        scenario_data.scenario_id = "existing_scenario"

        # Mock existing scenario
        self.scenario_manager.instantiated_models["existing_scenario"] = Mock()

        # Should not raise error, just skip - patch evaluate_scenario to avoid configuration issues
        with patch.object(self.scenario_manager, "evaluate_scenario"):
            self.scenario_manager.import_scenario(scenario_data, {})

    def test_evaluate_scenario_no_grid(self):
        """Test evaluating scenario without grid raises error."""
        self.scenario_manager.grid = None

        with pytest.raises(ConfigurationError):
            self.scenario_manager.evaluate_scenario("test_scenario")

    def test_evaluate_scenario_no_output_data(self):
        """Test evaluating scenario without output data raises error."""
        self.scenario_manager.output_data = None

        with pytest.raises(ConfigurationError):
            self.scenario_manager.evaluate_scenario("test_scenario")

    def test_get_scenario_names(self):
        """Test getting list of scenario names."""
        scenarios = ["scenario1", "scenario2", "scenario3"]
        for s in scenarios:
            self.scenario_manager.instantiated_models[s] = Mock()

        result = self.scenario_manager.get_scenario_names()

        assert set(result) == set(scenarios)

    def test_get_scenario_number(self):
        """Test getting number of scenarios."""
        scenarios = ["scenario1", "scenario2", "scenario3"]
        for s in scenarios:
            self.scenario_manager.instantiated_models[s] = Mock()

        result = self.scenario_manager.get_scenario_number()

        assert result == 3

    def test_get_scenario_state(self):
        """Test getting scenario state."""
        scenario_id = "test_scenario"
        mock_state = Mock()
        self.scenario_manager.instantiated_models[scenario_id] = mock_state

        result = self.scenario_manager.get_scenario_state(scenario_id)

        assert result == mock_state

    def test_update_problem_name(self):
        """Test updating problem name."""
        new_name = "Updated Problem Name"

        with patch(
            "overtourism.dt_studio.manager.scenario.manager.get_timestamp"
        ) as mock_timestamp:
            mock_timestamp.return_value = "2023-01-01T00:00:00"

            self.scenario_manager.update_problem_name(new_name)

            assert self.scenario_manager.metadata.name == new_name
            assert self.scenario_manager.metadata.updated == "2023-01-01T00:00:00"

    def test_update_problem_description(self):
        """Test updating problem description."""
        new_description = "Updated Problem Description"

        with patch(
            "overtourism.dt_studio.manager.scenario.manager.get_timestamp"
        ) as mock_timestamp:
            mock_timestamp.return_value = "2023-01-01T00:00:00"

            self.scenario_manager.update_problem_description(new_description)

            assert self.scenario_manager.metadata.description == new_description
            assert self.scenario_manager.metadata.updated == "2023-01-01T00:00:00"

    def test_update_scenario_name(self):
        """Test updating scenario name."""
        scenario_id = "test_scenario"
        new_name = "Updated Scenario Name"

        mock_state = Mock()
        self.scenario_manager.instantiated_models[scenario_id] = mock_state

        with patch(
            "overtourism.dt_studio.manager.scenario.manager.get_timestamp"
        ) as mock_timestamp:
            mock_timestamp.return_value = "2023-01-01T00:00:00"

            self.scenario_manager.update_scenario_name(scenario_id, new_name)

            assert mock_state.metadata.name == new_name
            assert mock_state.metadata.updated == "2023-01-01T00:00:00"

    def test_update_scenario_description(self):
        """Test updating scenario description."""
        scenario_id = "test_scenario"
        new_description = "Updated Scenario Description"

        mock_state = Mock()
        self.scenario_manager.instantiated_models[scenario_id] = mock_state

        with patch(
            "overtourism.dt_studio.manager.scenario.manager.get_timestamp"
        ) as mock_timestamp:
            mock_timestamp.return_value = "2023-01-01T00:00:00"

            self.scenario_manager.update_scenario_description(
                scenario_id, new_description
            )

            assert mock_state.metadata.description == new_description
            assert mock_state.metadata.updated == "2023-01-01T00:00:00"

    def test_filter_situation_found(self):
        """Test filtering situation when it exists."""
        situation_name = "good_weather"

        result_dict, result_name = self.scenario_manager._filter_situation(
            situation_name
        )

        assert result_dict == S_Good_Weather
        assert result_name == situation_name

    def test_filter_situation_not_found(self):
        """Test filtering situation when it doesn't exist."""
        situation_name = "nonexistent_situation"

        result_dict, result_name = self.scenario_manager._filter_situation(
            situation_name
        )

        assert result_dict == {}
        assert result_name is None

    def test_real_situations_integration(self):
        """Test that real situations from setup.py are properly integrated."""
        # Verify all 7 situations are loaded
        assert len(self.scenario_manager.situations) == 7

        situation_names = [s.name for s in self.scenario_manager.situations]
        expected_names = [
            None,
            "good_weather",
            "bad_weather",
            "high_season",
            "low_season",
            "weekend_days",
            "working_days",
        ]

        for expected in expected_names:
            assert expected in situation_names

        # Test filtering with real situation names
        result_dict, result_name = self.scenario_manager._filter_situation(
            "good_weather"
        )
        assert result_dict == S_Good_Weather
        assert result_name == "good_weather"

    def test_real_grid_configuration_validation(self):
        """Test that the real grid configuration is properly set up."""
        assert self.scenario_manager.grid is not None
        assert hasattr(self.scenario_manager.grid, "grid")
        assert hasattr(self.scenario_manager.grid, "x")
        assert hasattr(self.scenario_manager.grid, "y")
        assert hasattr(self.scenario_manager.grid, "x_max")
        assert hasattr(self.scenario_manager.grid, "y_max")

        # Verify grid contains expected presence variables
        assert PV_tourists in self.scenario_manager.grid.grid
        assert PV_excursionists in self.scenario_manager.grid.grid

    def test_real_sampler_configuration_validation(self):
        """Test that the real sampler configuration is properly set up."""
        assert self.scenario_manager.sampler is not None
        assert hasattr(self.scenario_manager.sampler, "sampling_dicts")
        sampling_dicts = self.scenario_manager.sampler.sampling_dicts
        assert len(sampling_dicts) == 2  # tourists and excursionists

        # Verify presence variables are configured
        p_vars = [d["p_var"] for d in sampling_dicts]
        assert PV_tourists in p_vars
        assert PV_excursionists in p_vars
