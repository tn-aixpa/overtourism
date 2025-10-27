# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from overtourism.dt_studio.manager.problem.manager import ProblemManager
from overtourism.dt_studio.manager.scenario.manager import ScenarioManager
from overtourism.dt_studio.viewer.viewer import ModelViewer
from overtourism.model.setup import M_Base, build_output


class TestDTStudioIntegration:
    """Integration tests for the dt_studio module."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up integration test fixtures."""
        # Use real M_Base model instance directly (it's already instantiated)
        self.abstract_model = M_Base

        # Use real build_output function instead of mock
        self.kpi_function = build_output

        # Keep these as mocks since they're configuration objects
        self.mock_output_data = Mock()
        self.mock_sampler = Mock()
        self.mock_situations = []
        self.mock_grid = Mock()
        self.mock_store_config = Mock()
        self.mock_store_config.store_type = "local"
        self.mock_store_config.config = {}

        yield

    @patch("overtourism.dt_studio.manager.problem.manager.io_builder")
    def test_problem_workflow(self, mock_io_builder):
        """Test complete problem management workflow."""
        mock_store = Mock()
        mock_io_builder.create.return_value = mock_store

        # Create problem manager
        problem_manager = ProblemManager(
            abstract_model=self.abstract_model,
            kpi_function=self.kpi_function,
            output_data=self.mock_output_data,
            sampler=self.mock_sampler,
            situations=self.mock_situations,
            grid=self.mock_grid,
            store=self.mock_store_config,
        )

        # Test workflow: add -> get -> export -> delete
        problem_id = "integration_test_problem"

        # Add problem
        problem_manager.add_problem(
            problem_id=problem_id,
            name="Integration Test Problem",
            description="A problem for integration testing",
        )

        # Verify problem exists
        assert problem_id in problem_manager.problems
        problem = problem_manager.get_problem(problem_id)
        assert isinstance(problem, ScenarioManager)

        # Add scenario to problem
        scenario_id = "test_scenario"
        problem.add_scenario(scenario_id, values={}, name="Test Scenario")

        # Verify scenario exists
        assert scenario_id in problem.instantiated_models

        # Export problem (mock the export dependencies)
        with patch(
            "overtourism.dt_studio.manager.problem.manager.ProblemValues"
        ) as mock_problem_values:
            mock_instance = Mock()
            mock_instance.to_dict.return_value = {"test": "data"}
            mock_problem_values.return_value = mock_instance

            problem_manager.export_problem(problem_id)
            mock_store.export_problem.assert_called_once()

        # Delete problem
        problem_manager.delete_problem(problem_id)
        assert problem_id not in problem_manager.problems

    @patch("overtourism.dt_studio.viewer.viewer.load_yaml")
    @patch("overtourism.dt_studio.viewer.viewer.build_indexes_from_config")
    def test_viewer_workflow(self, mock_build_indexes, mock_load_yaml):
        """Test complete viewer workflow."""
        # Mock configuration
        config = {
            "indexes": [
                {
                    "index_id": "capacity",
                    "group": "constraints",
                    "index_type": "constant",
                }
            ]
        }
        mock_load_yaml.return_value = config

        mock_index = Mock()
        mock_index.group = "constraints"
        mock_index.index_id = "capacity"
        mock_index.index_type = "constant"
        mock_build_indexes.return_value = [mock_index]

        # Create viewer
        viewer = ModelViewer("/test/config.yaml")

        # Test groups functionality
        groups = viewer.get_groups()
        assert isinstance(groups, list)
        assert len(groups) > 0

        # Test visualization preparation
        mock_data = Mock()
        mock_data.sample_x = [1, 2, 3]
        mock_data.sample_y = [4, 5, 6]
        mock_data.uncertainty = [0.1, 0.2, 0.3]
        mock_data.constraint_curves = {}
        mock_data.x_max = 10
        mock_data.y_max = 10

        # Test uncertainty extraction
        uncertainty = viewer._get_uncertainty(mock_data)
        assert uncertainty == [0.1, 0.2, 0.3]

        # Test widgets configuration
        vals = {"capacity": 100}
        mock_index.copy.return_value = mock_index
        mock_index.to_dict.return_value = {"id": "capacity", "type": "constant"}

        widgets = viewer.get_widgets(vals)
        assert isinstance(widgets, dict)

    @patch("overtourism.dt_studio.manager.problem.manager.io_builder")
    def test_scenario_management_workflow(self, mock_io_builder):
        """Test scenario management within a problem."""
        mock_store = Mock()
        mock_io_builder.create.return_value = mock_store

        # Create problem manager and add a problem
        problem_manager = ProblemManager(
            abstract_model=self.abstract_model,
            kpi_function=self.kpi_function,
            output_data=self.mock_output_data,
            sampler=self.mock_sampler,
            situations=self.mock_situations,
            grid=self.mock_grid,
            store=self.mock_store_config,
        )

        problem_id = "scenario_test_problem"
        problem_manager.add_problem(problem_id)
        scenario_manager = problem_manager.get_problem(problem_id)

        # Test multiple scenarios
        scenarios = ["baseline", "scenario_a", "scenario_b"]
        for scenario_id in scenarios:
            scenario_manager.add_scenario(
                scenario_id=scenario_id,
                values={f"param_{scenario_id}": 1.0},
                name=f"Scenario {scenario_id}",
                description=f"Description for {scenario_id}",
            )

        # Verify all scenarios exist
        assert scenario_manager.get_scenario_number() == 3
        scenario_names = scenario_manager.get_scenario_names()
        for scenario_id in scenarios:
            assert scenario_id in scenario_names

        # Test scenario updates
        scenario_manager.update_scenario_name("baseline", "Updated Baseline")
        baseline_state = scenario_manager.get_scenario_state("baseline")
        assert baseline_state.metadata.name == "Updated Baseline"

        # Test scenario removal
        scenario_manager.delete_scenario("scenario_b")
        assert scenario_manager.get_scenario_number() == 2
        assert "scenario_b" not in scenario_manager.get_scenario_names()

    def test_error_handling_workflow(self):
        """Test error handling across the dt_studio components."""
        with patch("overtourism.dt_studio.manager.problem.manager.io_builder"):
            problem_manager = ProblemManager(
                abstract_model=self.abstract_model,
                kpi_function=self.kpi_function,
                output_data=self.mock_output_data,
                sampler=self.mock_sampler,
                situations=self.mock_situations,
                grid=self.mock_grid,
                store=self.mock_store_config,
            )

            # Test getting non-existent problem
            with pytest.raises(KeyError):
                problem_manager.get_problem("nonexistent")

            # Add a problem and test scenario errors
            problem_id = "error_test_problem"
            problem_manager.add_problem(problem_id)
            scenario_manager = problem_manager.get_problem(problem_id)

            # Test scenario does not exist error
            from overtourism.dt_studio.manager.utils.exception import (
                ScenarioDoesNotExist,
            )

            with pytest.raises(ScenarioDoesNotExist):
                scenario_manager.delete_scenario("nonexistent_scenario")

            # Test scenario already exists error
            scenario_id = "test_scenario"
            scenario_manager.add_scenario(scenario_id)

            from overtourism.dt_studio.manager.utils.exception import (
                ScenarioAlreadyExists,
            )

            with pytest.raises(ScenarioAlreadyExists):
                scenario_manager.add_scenario(scenario_id)  # Try to add again

    @patch("overtourism.dt_studio.viewer.viewer.load_yaml")
    @patch("overtourism.dt_studio.viewer.viewer.build_indexes_from_config")
    @patch("overtourism.dt_studio.viewer.viewer.bidimensional_figure")
    def test_visualization_integration(
        self, mock_bidimensional_figure, mock_build_indexes, mock_load_yaml
    ):
        """Test integration between problem management and visualization."""
        # Setup viewer
        config = {"indexes": []}
        mock_load_yaml.return_value = config
        mock_build_indexes.return_value = []

        viewer = ModelViewer("/test/config.yaml")

        # Mock visualization data (simulating output from scenario evaluation)
        mock_data = Mock()
        mock_data.sample_x = [10, 20, 30]
        mock_data.sample_y = [15, 25, 35]
        mock_data.constraint_curves = {}
        mock_data.x_max = 50
        mock_data.y_max = 50

        mock_view = Mock()
        mock_view.view_type = "Bidimensionale"
        mock_view.constraint = None

        # Mock the figure creation
        mock_figure = Mock()
        mock_bidimensional_figure.return_value = mock_figure

        # Patch the uncertainty method
        with patch.object(viewer, "_get_uncertainty", return_value=[0.1, 0.2, 0.3]):
            result = viewer.viz(mock_data, mock_view, "Integration Test")

            assert result == mock_figure
            mock_bidimensional_figure.assert_called_once()


class TestDTStudioRealConfigIntegration:
    """Integration tests using real model configuration from conftest."""

    @pytest.fixture(autouse=True)
    def setup_method(self, real_model_config):
        """Set up integration test fixtures with real configuration."""
        # Use real configuration from the shared fixture
        self.real_config = real_model_config

        # Extract real components
        self.abstract_model = real_model_config["abstract_model"]
        self.kpi_function = real_model_config["kpi_function"]
        self.output_data = real_model_config["output_data"]
        self.sampler = real_model_config["sampler"]
        self.situations = real_model_config["situations"]
        self.grid = real_model_config["grid"]
        self.store_config = real_model_config["store_config"]

    @patch("overtourism.dt_studio.manager.problem.manager.io_builder")
    def test_problem_workflow_with_real_config(self, mock_io_builder):
        """Test complete problem management workflow with real configuration."""
        mock_store = Mock()
        mock_io_builder.create.return_value = mock_store

        # Create problem manager with real configuration
        problem_manager = ProblemManager(
            abstract_model=self.abstract_model,
            kpi_function=self.kpi_function,
            output_data=self.output_data,
            sampler=self.sampler,
            situations=self.situations,
            grid=self.grid,
            store=self.store_config,
        )

        # Test workflow with real data
        problem_id = "real_config_integration_test"

        # Add problem
        problem_manager.add_problem(
            problem_id=problem_id,
            name="Real Config Integration Test",
            description="Integration test using real model configuration",
        )

        # Verify problem was created with real configuration
        assert problem_id in problem_manager.problems
        problem = problem_manager.get_problem(problem_id)
        assert isinstance(problem, ScenarioManager)

        # Verify the scenario manager has the real configuration
        assert problem.abstract_model == self.abstract_model
        assert problem.output_function == self.kpi_function
        assert problem.sampler == self.sampler
        assert problem.situations == self.situations
        assert problem.grid == self.grid

        # Test that we can work with real situations
        scenario_id = "real_scenario"
        values = {}  # Empty values for base scenario

        problem.add_scenario(scenario_id, values=values, name="Real Scenario Test")

        # Verify scenario was created
        assert scenario_id in problem.instantiated_models

        # Verify the scenario has access to real situations
        assert len(problem.situations) == 7  # Real number of situations from setup.py

        # Test that the real sampler has correct configuration
        assert len(self.sampler.sampling_dicts) == 2  # Tourists and excursionists

        # Test that the real grid has correct dimensions
        assert self.grid.x_max == 10000
        assert self.grid.y_max == 10000
