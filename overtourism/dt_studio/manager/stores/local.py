# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing
from pathlib import Path

from ..io.utils import get_glob, load_problem, save_problem
from .base import Store

if typing.TYPE_CHECKING:
    from ..io.classes import ProblemValues, ScenarioValues


class LocalIOStore(Store):
    """Local filesystem implementation of the IOStore interface.

    Provides methods for storing and retrieving problem and scenario data
    using the local filesystem. Data is stored in YAML format.

    Parameters
    ----------
    folder : str
        Path to the directory where files will be stored

    Attributes
    ----------
    folder : Path
        Pathlib object representing the storage directory
    """

    def __init__(self, folder: str) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def export_problem(self, problem_id: str, problem_data: dict) -> None:
        """Save problem data to a YAML file.

        Parameters
        ----------
        problem_id : str
            Unique identifier for the problem
        problem_data : dict
            Problem configuration data to save
        """
        path = self.folder / f"{problem_id}.yaml"
        save_problem(problem_data, path)

    def import_problem(self, problem_id: str) -> ProblemValues:
        """Load problem data from a YAML file.

        Parameters
        ----------
        problem_id : str
            Identifier of the problem to load

        Returns
        -------
        ProblemValues
            Loaded problem configuration
        """
        path = self.folder / problem_id
        return load_problem(path)

    def list_problems(self) -> list[str]:
        """Get list of all problem files in storage directory.

        Returns
        -------
        list[str]
            List of problem file paths
        """
        return get_glob(self.folder)

    def delete_problem(self, problem_id: str) -> None:
        """
        Delete a problem file from the storage directory.

        Parameters
        ----------
        problem_id : str
            Identifier of the problem to delete
        """
        path = self.folder / f"{problem_id}.yaml"
        if path.exists():
            path.unlink()

    def export_scenario(
        self,
        problem_id: str,
        scenario_id: str,
        scenario_data: ScenarioValues,
    ) -> None:
        """Save scenario data within its parent problem file.

        Updates or adds the scenario to the problem configuration and
        saves the updated problem file.

        Parameters
        ----------
        problem_id : str
            Identifier of the parent problem
        scenario_id : str
            Identifier of the scenario to save
        scenario_data : ScenarioValues
            Scenario configuration to save
        """
        problem = self.import_problem(f"{problem_id}.yaml")
        problem.scenarios = [
            s for s in problem.scenarios if s.scenario_id != scenario_id
        ]
        problem.scenarios.append(scenario_data)
        self.export_problem(problem_id, problem.to_dict())

    def import_scenario(self, problem_id: str, scenario_id: str) -> ScenarioValues:
        """Load scenario data from its parent problem file.

        Parameters
        ----------
        problem_id : str
            Identifier of the parent problem
        scenario_id : str
            Identifier of the scenario to load

        Returns
        -------
        ScenarioValues
            Loaded scenario configuration
        """
        problem = self.import_problem(f"{problem_id}.yaml")
        return next(s for s in problem.scenarios if s.scenario_id == scenario_id)

    def delete_scenario(self, problem_id: str, scenario_id: str) -> None:
        """
        Delete a scenario from its parent problem file.

        Parameters
        ----------
        problem_id : str
            Identifier of the parent problem
        scenario_id : str
            Identifier of the scenario to delete
        """
        problem = self.import_problem(f"{problem_id}.yaml")
        problem.scenarios = [
            s for s in problem.scenarios if s.scenario_id != scenario_id
        ]
        self.export_problem(problem_id, problem.to_dict())
