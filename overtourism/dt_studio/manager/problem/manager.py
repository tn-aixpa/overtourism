# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing

from ..io.classes import ProblemValues
from ..io.utils import prepare_values
from ..scenario.manager import ScenarioManager
from ..stores.builder import io_builder
from .metadata import default_problem_metadata

if typing.TYPE_CHECKING:
    from typing import Callable

    from civic_digital_twins.dt_model.model.abstract_model import AbstractModel

    from ..config.classes import Grid, ModelOutput, Sampler, Situation, StoreConfig
    from ..stores.base import Store


class ProblemManager:
    """
    Class to manage problems.
    """

    def __init__(
        self,
        abstract_model: AbstractModel,
        kpi_function: Callable,
        output_data: ModelOutput,
        sampler: Sampler,
        situations: list[Situation],
        grid: Grid,
        store: StoreConfig,
    ) -> None:
        self.abstract_model = abstract_model
        self.kpi_function = kpi_function
        self.output_data = output_data
        self.sampler = sampler
        self.situations = situations
        self.grid = grid
        self.store: Store = io_builder.create(store.store_type, **store.config)

        self.problems: dict[str, ScenarioManager] = {}

    ##############################
    # Problems
    ##############################

    def add_problem(
        self,
        problem_id: str,
        name: str | None = None,
        description: str | None = None,
        created: str | None = None,
        updated: str | None = None,
        editable_indexes: list[str] | None = None,
    ) -> None:
        """
        Add new problem to the manager.

        Parameters
        ----------
        problem_id : str
            Problem id.
        problem_name : str
            Problem name.
        problem_description : str
            Problem description.
        created : str
            Problem creation date.
        updated : str
            Problem update date.
        """
        metadata = default_problem_metadata(
            problem_id,
            name,
            description,
            created,
            updated,
            editable_indexes,
        )
        self.problems[problem_id] = ScenarioManager(
            problem_id,
            self.abstract_model,
            self.kpi_function,
            self.output_data,
            self.sampler,
            self.situations,
            self.grid,
            metadata=metadata,
        )

    def get_problem(self, problem_id: str) -> ScenarioManager:
        """
        Retrieve problem from the manager by id.

        Parameters
        ----------
        problem_id : str
            Problem id.

        Returns
        -------
        ScenarioManager
        """
        return self.problems[problem_id]

    def delete_problem(self, problem_id: str) -> None:
        """
        Delete problem from the manager by id.

        Parameters
        ----------
        problem_id : str
            Problem id.
        """
        self.problems.pop(problem_id, None)
        self.store.delete_problem(problem_id)

    def export_problem(
        self,
        problem_id: str,
        editable_indexes: list[str] | None = None,
    ) -> None:
        """
        Export problem from the manager by id.

        Parameters
        ----------
        problem_id : str
            Problem id.
        """
        problem = self.get_problem(problem_id)
        scenarios = [
            problem.export_scenario(model) for model in problem.instantiated_models
        ]
        if editable_indexes is not None:
            problem.metadata.editable_indexes = editable_indexes
        data = ProblemValues(
            problem_id=problem.problem_id,
            scenarios=scenarios,
            metadata=problem.metadata,
        )
        self.store.export_problem(problem_id, data.to_dict())

    def import_problem(self, problem_id: str) -> None:
        """
        Import problem from id.

        Parameters
        ----------
        problem_id : str
            Problem id.

        Returns
        -------
        str
        """
        # Add new problem
        problem_data = self.store.import_problem(problem_id)
        self.add_problem(
            problem_id=problem_data.problem_id,
            name=problem_data.metadata.name,
            description=problem_data.metadata.description,
            created=problem_data.metadata.created,
            updated=problem_data.metadata.updated,
            editable_indexes=problem_data.metadata.editable_indexes,
        )

        # Import scenarios
        for scenario in problem_data.scenarios:
            values = prepare_values(scenario)
            self.problems[problem_data.problem_id].import_scenario(scenario, values)

    def import_problems(self, problem_ids: list[str] | None = None) -> None:
        """
        Import multiple problems. If problem_id is None, all problems are imported.

        Parameters
        ----------
        problem_id : list[str]
            List of problem ids.
        """
        if problem_ids is None:
            problem_ids = self.store.list_problems()

        for p in problem_ids:
            self.import_problem(p)

    def export_scenario(self, problem_id: str, scenario_id: str) -> None:
        data = self.get_problem(problem_id).export_scenario(scenario_id)
        self.store.export_scenario(problem_id, scenario_id, data)

    def delete_scenario(self, problem_id: str, scenario_id: str) -> None:
        self.get_problem(problem_id).delete_scenario(scenario_id)
        self.store.delete_scenario(problem_id, scenario_id)
