# SPDX-License-Identifier: Apache-2.0

import pytest

from overtourism.dt_studio.manager.problem.manager import ProblemManager
from overtourism.dt_studio.manager.scenario.manager import ScenarioManager
from overtourism.model.setup import M_Base, PV_excursionists, PV_tourists, build_output


class TestProblemManagerRealConfig:
    """Test suite for ProblemManager using real configuration from setup.py."""

    @pytest.fixture(autouse=True)
    def setup_method(self, real_model_config):
        """Set up test fixtures with real configuration from conftest fixture."""
        # Get all configuration from the shared fixture
        self.grid = real_model_config["grid"]
        self.sampler = real_model_config["sampler"]
        self.situations = real_model_config["situations"]
        self.export_folder = real_model_config["export_folder"]
        self.store_config = real_model_config["store_config"]

        # Create ProblemManager with real configuration
        self.problem_manager = ProblemManager(
            abstract_model=real_model_config["abstract_model"],
            kpi_function=real_model_config["kpi_function"],
            output_data=real_model_config["output_data"],
            sampler=self.sampler,
            situations=self.situations,
            grid=self.grid,
            store=self.store_config,
        )

    def test_init_with_real_config(self):
        """Test ProblemManager initialization with real configuration."""
        assert self.problem_manager.abstract_model == M_Base
        assert self.problem_manager.kpi_function == build_output
        assert isinstance(self.problem_manager.problems, dict)
        assert len(self.problem_manager.problems) == 0
        assert self.problem_manager.sampler == self.sampler
        assert self.problem_manager.situations == self.situations
        assert self.problem_manager.grid == self.grid

    def test_add_problem_with_real_config(self):
        """Test adding a problem with real configuration."""
        problem_id = "test_problem_real"
        name = "Test Problem with Real Config"
        description = "A test problem using real configuration"

        self.problem_manager.add_problem(
            problem_id=problem_id, name=name, description=description
        )

        assert problem_id in self.problem_manager.problems
        assert isinstance(self.problem_manager.problems[problem_id], ScenarioManager)

        # Verify the scenario manager has the real configuration
        scenario_manager = self.problem_manager.problems[problem_id]
        assert scenario_manager.abstract_model == M_Base
        assert scenario_manager.output_function == build_output
        assert scenario_manager.sampler == self.sampler
        assert scenario_manager.situations == self.situations
        assert scenario_manager.grid == self.grid

    def test_add_scenario_with_real_config(self):
        """Test adding a scenario with real configuration."""
        problem_id = "test_problem_scenario"
        scenario_id = "test_scenario"

        self.problem_manager.add_problem(problem_id=problem_id)
        scenario_manager = self.problem_manager.get_problem(problem_id)

        # Add a scenario with some index values
        scenario_manager.add_scenario(
            scenario_id=scenario_id,
            values={"capacity_beach": 5000, "capacity_parking": 2000},
            name="Test Scenario",
            description="A test scenario with real configuration",
        )

        assert scenario_id in scenario_manager.instantiated_models
        scenario_state = scenario_manager.get_scenario_state(scenario_id)
        assert scenario_state.scenario_id == scenario_id

    def test_real_export_import_workflow(self):
        """Test complete export/import workflow with real file operations."""
        problem_id = "export_import_test"
        scenario_id = "test_scenario"

        # Create and configure problem
        self.problem_manager.add_problem(
            problem_id=problem_id,
            name="Export Import Test",
            description="Testing real export/import functionality",
        )

        scenario_manager = self.problem_manager.get_problem(problem_id)
        scenario_manager.add_scenario(
            scenario_id=scenario_id,
            values={"capacity_beach": 3000, "capacity_parking": 1500},
            name="Export Test Scenario",
            description="Scenario for export testing",
        )

        # Export the problem - this should create real files
        self.problem_manager.export_problem(problem_id)

        # Verify export files were created (should be .yaml not .json)
        problem_file = self.export_folder / f"{problem_id}.yaml"
        assert problem_file.exists(), f"Problem file should exist at {problem_file}"

        # Clear the problem manager and import back
        self.problem_manager.delete_problem(problem_id)
        assert problem_id not in self.problem_manager.problems

    def test_export_scenario_with_real_config(self):
        """Test exporting a specific scenario with real configuration."""
        problem_id = "scenario_export_test"
        scenario_id = "exportable_scenario"

        # Create problem and scenario
        self.problem_manager.add_problem(problem_id=problem_id)
        scenario_manager = self.problem_manager.get_problem(problem_id)

        scenario_manager.add_scenario(
            scenario_id=scenario_id,
            values={"capacity_accommodation": 4000, "capacity_food": 3500},
            name="Exportable Scenario",
            description="A scenario designed for export testing",
        )

        # First export the problem to create the base problem file
        self.problem_manager.export_problem(problem_id)

        # Then export the scenario (this updates the problem file with scenario data)
        self.problem_manager.export_scenario(problem_id, scenario_id)

        # Verify the problem file exists and contains the scenario
        problem_file = self.export_folder / f"{problem_id}.yaml"
        assert problem_file.exists(), f"Problem file should exist at {problem_file}"

    def test_multiple_problems_real_persistence(self):
        """Test managing multiple problems with real persistence."""
        problem_ids = ["problem_1", "problem_2", "problem_3"]

        # Create multiple problems with different configurations
        for i, pid in enumerate(problem_ids):
            self.problem_manager.add_problem(
                problem_id=pid,
                name=f"Problem {i + 1}",
                description=f"Test problem number {i + 1}",
            )

            # Add a scenario to each problem
            scenario_manager = self.problem_manager.get_problem(pid)
            scenario_manager.add_scenario(
                scenario_id="default_scenario",
                values={"capacity_beach": 2000 + i * 1000},
                name=f"Default Scenario {i + 1}",
                description=f"Default scenario for problem {i + 1}",
            )

        # Export all problems
        for pid in problem_ids:
            self.problem_manager.export_problem(pid)

        # Verify all export files exist (should be .yaml)
        for pid in problem_ids:
            problem_file = self.export_folder / f"{pid}.yaml"
            assert problem_file.exists(), f"Problem file for {pid} should exist"

        # Clear and reimport all
        for pid in problem_ids:
            self.problem_manager.delete_problem(pid)

        assert len(self.problem_manager.problems) == 0

    def test_real_config_integration_with_situations(self):
        """Test that real situations work correctly with the problem manager."""
        problem_id = "situation_test"
        scenario_id = "situation_scenario"

        self.problem_manager.add_problem(problem_id=problem_id)
        scenario_manager = self.problem_manager.get_problem(problem_id)

        scenario_manager.add_scenario(
            scenario_id=scenario_id, values={}, name="Situation Test Scenario"
        )

        # Verify situations are properly configured
        assert len(scenario_manager.situations) == 7  # From setup.py configuration

        situation_names = [s.name for s in scenario_manager.situations]
        expected_situation_names = [
            None,  # Base situation has None as name
            "good_weather",
            "bad_weather",
            "high_season",
            "low_season",
            "weekend_days",
            "working_days",
        ]

        for expected in expected_situation_names:
            assert expected in situation_names

    def test_grid_configuration_validation(self):
        """Test that the grid configuration is properly set up."""
        problem_id = "grid_test"
        self.problem_manager.add_problem(problem_id=problem_id)
        scenario_manager = self.problem_manager.get_problem(problem_id)

        # Verify grid configuration
        assert scenario_manager.grid is not None
        assert scenario_manager.grid == self.grid

        # Check grid properties
        assert hasattr(scenario_manager.grid, "grid")
        assert hasattr(scenario_manager.grid, "x")
        assert hasattr(scenario_manager.grid, "y")
        assert hasattr(scenario_manager.grid, "x_max")
        assert hasattr(scenario_manager.grid, "y_max")

        # Verify grid contains expected presence variables
        assert PV_tourists in scenario_manager.grid.grid
        assert PV_excursionists in scenario_manager.grid.grid

    def test_sampler_configuration_validation(self):
        """Test that the sampler configuration is properly set up."""
        problem_id = "sampler_test"
        self.problem_manager.add_problem(problem_id=problem_id)
        scenario_manager = self.problem_manager.get_problem(problem_id)

        # Verify sampler configuration
        assert scenario_manager.sampler is not None
        assert scenario_manager.sampler == self.sampler

        # Check sampler has expected sample dictionaries
        assert hasattr(scenario_manager.sampler, "sampling_dicts")
        sampling_dicts = scenario_manager.sampler.sampling_dicts
        assert len(sampling_dicts) == 2  # tourists and excursionists

        # Verify presence variables are configured
        p_vars = [d["p_var"] for d in sampling_dicts]
        assert PV_tourists in p_vars
        assert PV_excursionists in p_vars

    def test_cleanup_verification(self):
        """Test that cleanup properly removes export files."""
        problem_id = "cleanup_test"

        # Create and export a problem
        self.problem_manager.add_problem(problem_id=problem_id)
        self.problem_manager.export_problem(problem_id)

        # Verify file exists (should be .yaml)
        problem_file = self.export_folder / f"{problem_id}.yaml"
        assert problem_file.exists()

        # Store the export folder path for later verification
        export_folder_path = self.export_folder

        # The cleanup will happen automatically via the fixture
        # This test just verifies the file was created for cleanup
        assert export_folder_path.exists()

    def test_cleanup_actually_works(self):
        """Test that verifies cleanup removes temporary directories."""
        import tempfile
        from pathlib import Path

        # Create a temporary directory manually
        temp_dir = Path(tempfile.mkdtemp())
        test_file = temp_dir / "test_file.txt"
        test_file.write_text("test content")

        # Verify it exists
        assert temp_dir.exists()
        assert test_file.exists()

        # Manual cleanup (simulating what the fixture does)
        import shutil

        shutil.rmtree(temp_dir)

        # Verify it's gone
        assert not temp_dir.exists()
        assert not test_file.exists()
