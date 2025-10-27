# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing
from typing import Callable

from civic_digital_twins.dt_model.ensemble import Ensemble
from civic_digital_twins.dt_model.model.instantiated_model import InstantiatedModel
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation

from ..indexes.utils import get_diff
from ..io.utils import scenario_values
from ..utils.exception import (
    ConfigurationError,
    ScenarioAlreadyExists,
    ScenarioDoesNotExist,
)
from ..utils.utils import get_timestamp
from .metadata import ScenarioMetadata
from .state import ScenarioState

if typing.TYPE_CHECKING:
    from civic_digital_twins.dt_model.internal.sympyke.symbol import SymbolValue
    from civic_digital_twins.dt_model.model.abstract_model import AbstractModel
    from civic_digital_twins.dt_model.symbols.context_variable import ContextVariable

    from ..config.classes import Grid, ModelOutput, Sampler, Situation
    from ..io.classes import ScenarioValues
    from ..problem.metadata import ProblemMetadata
    from .state import SituationData


class ScenarioManager:
    def __init__(
        self,
        problem_id: str,
        abstract_model: AbstractModel,
        output_function: Callable,
        output_data: ModelOutput,
        sampler: Sampler,
        situations: list[Situation],
        grid: Grid,
        metadata: ProblemMetadata,
    ) -> None:
        self.problem_id = problem_id
        self.abstract_model = abstract_model
        self.output_function = output_function
        self.output_data = output_data
        self.sampler = sampler
        self.situations = situations
        self.grid = grid
        self.metadata = metadata

        self.instantiated_models: dict[str, ScenarioState] = {}

    ##############################
    # Operations
    ##############################

    def add_scenario(
        self,
        scenario_id: str,
        values: dict | None = None,
        name: str | None = None,
        description: str | None = None,
        created: str | None = None,
        updated: str | None = None,
    ) -> None:
        """
        Add a new scenario to the manager.

        Parameters
        ----------
        scenario_id : str
            Unique identifier for the scenario.
        values : dict, optional
            Values to instantiate the scenario with.
        name : str, optional
            Name of the scenario.
        description : str, optional
            Description of the scenario.
        created : str, optional
            Creation timestamp.
        updated : str, optional
            Last updated timestamp.

        Raises
        ------
        ScenarioDoesNotExist
            If no base model is provided.
        ScenarioAlreadyExists
            If a scenario with the same ID already exists.
        """
        # Check on base model
        if self.abstract_model is None:
            raise ConfigurationError("No base model provided")

        # Check on model name
        for scenario_state in self.instantiated_models.values():
            if scenario_state.scenario_id == scenario_id:
                raise ScenarioAlreadyExists(
                    f"Model with name {scenario_id} already exists"
                )

        self.instantiated_models[scenario_id] = self._create_scenario_state(
            scenario_id=scenario_id,
            values=values,
            name=name,
            description=description,
            created=created,
            updated=updated,
        )

    def update_scenario(
        self,
        scenario_id: str,
        values: dict,
    ) -> None:
        """
        Update an existing scenario with new values.

        Parameters
        ----------
        scenario_id : str
            Identifier of the scenario to update.
        values : dict
            New values for the scenario.

        Raises
        ------
        ScenarioDoesNotExist
            If the scenario does not exist.
        """
        # Check scenario exists
        if scenario_id not in self.instantiated_models:
            raise ScenarioDoesNotExist(f"Model with name {scenario_id} does not exist")

        # Update model from base
        base_model = self.instantiated_models[scenario_id]
        model = InstantiatedModel(self.abstract_model, scenario_id, values)

        index_diffs = get_diff(model)
        metadata = ScenarioMetadata(
            name=base_model.metadata.name,
            description=base_model.metadata.description,
            created=base_model.metadata.created,
            updated=get_timestamp(),
            index_diffs=index_diffs,
        )
        self.instantiated_models[scenario_id] = ScenarioState(
            model=model,
            scenario_id=scenario_id,
            metadata=metadata,
        )

    def delete_scenario(
        self,
        scenario_id: str,
    ) -> None:
        """
        Remove a scenario from the manager.

        Parameters
        ----------
        scenario_id : str
            Identifier of the scenario to remove.

        Raises
        ------
        ScenarioDoesNotExist
            If the scenario does not exist.
        """
        try:
            self.instantiated_models.pop(scenario_id)
        except KeyError:
            raise ScenarioDoesNotExist(f"Model with name {scenario_id} does not exist")

    def export_scenario(
        self,
        scenario_id: str,
    ) -> ScenarioValues:
        """
        Export a scenario's values and metadata.

        Parameters
        ----------
        scenario_id : str
            Identifier of the scenario to export.

        Returns
        -------
        ScenarioValues
            Exported scenario data.
        """
        # Export only changed values
        scenario_state = self.get_scenario_state(scenario_id)
        return scenario_values(
            scenario_id=scenario_id,
            values=scenario_state.model.get_values(),
            metadata=scenario_state.metadata,
        )

    def import_scenario(
        self,
        scenario_data: ScenarioValues,
        values: dict,
    ) -> None:
        """
        Import a scenario into the manager.

        Parameters
        ----------
        scenario_data : ScenarioValues
            Scenario data to import.
        values : dict
            Values to instantiate the scenario with.

        Notes
        -----
        If the scenario already exists, the import is skipped.
        """
        try:
            self.add_scenario(
                scenario_id=scenario_data.scenario_id,
                values=values,
                name=scenario_data.metadata.name,
                description=scenario_data.metadata.description,
                created=scenario_data.metadata.created,
                updated=scenario_data.metadata.updated,
            )
            self.evaluate_scenario(scenario_data.scenario_id)
        except ScenarioAlreadyExists:
            pass

    def evaluate_scenario(
        self,
        scenario_id: str,
        situation: str | None = None,
        ensemble_size: int = 20,
    ) -> None:
        """
        Evaluate a scenario for a given situation.

        Parameters
        ----------
        scenario_id : str
            Identifier of the scenario to evaluate.
        situation : str, optional
            Situation ID to evaluate. If None, uses default.
        ensemble_size : int, optional
            Ensemble size for evaluation (default is 20).

        Raises
        ------
        ConfigurationError
            If grid or output data is not provided.
        """
        # Check on grid registration
        if self.grid is None:
            raise ConfigurationError("No grid provided")

        if self.output_data is None:
            raise ConfigurationError("No output data provided")

        # Collect model status
        try:
            scenario_state = self.get_scenario_state(scenario_id)
        except KeyError:
            self.add_scenario(scenario_id)
            scenario_state = self.get_scenario_state(scenario_id)

        # Execute evaluation
        scenario_state = self._execute_evaluation(
            scenario_state, situation, ensemble_size
        )
        self.instantiated_models[scenario_id] = scenario_state

    def _execute_evaluation(
        self,
        scenario_state: ScenarioState,
        situation: str | None = None,
        ensemble_size: int = 20,
    ) -> ScenarioState:
        """
        Execute evaluation for a scenario state and situation.

        Parameters
        ----------
        scenario_state : ScenarioState
            Scenario state to evaluate.
        situation : str, optional
            Situation ID to evaluate.
        ensemble_size : int, optional
            Ensemble size for evaluation.

        Returns
        -------
        ScenarioState
            Updated scenario state with evaluation results.
        """
        # Filter situations
        compute_situation, situation_id = self._filter_situation(situation)

        # Build evaluation
        ensemble = Ensemble(
            scenario_state.model,
            compute_situation,
            cv_ensemble_size=ensemble_size,
        )
        evaluation = Evaluation(scenario_state.model, ensemble)

        # Run evaluation
        scenario_state.is_evaluating = True
        evaluation.evaluate_grid(self.grid.grid)
        scenario_state.is_evaluating = False

        eval_data = self.output_function(
            scenario_state.model,
            evaluation,
            ensemble,
            sampling_dicts=self.sampler.sampling_dicts,
        )

        data = self.output_data(
            x_max=self.grid.x_max,
            y_max=self.grid.y_max,
            **eval_data,
        )

        scenario_state.add_data(situation_id, data)
        scenario_state.metadata.updated = get_timestamp()

        return scenario_state

    def _create_scenario_state(
        self,
        scenario_id: str,
        values: dict | None = None,
        name: str | None = None,
        description: str | None = None,
        created: str | None = None,
        updated: str | None = None,
        index_diffs: dict[str, str] | None = None,
    ) -> ScenarioState:
        """
        Create a new ScenarioState instance.

        Parameters
        ----------
        scenario_id : str
            Identifier for the scenario.
        values : dict, optional
            Values to instantiate the scenario with.
        name : str, optional
            Name of the scenario.
        description : str, optional
            Description of the scenario.
        created : str, optional
            Creation timestamp.
        updated : str, optional
            Last updated timestamp.
        index_diffs : dict[str, str], optional
            Differences in model indices.

        Returns
        -------
        ScenarioState
            The created scenario state.
        """
        # Initialize model
        values = {} if values is None else values
        model = InstantiatedModel(
            self.abstract_model,
            scenario_id,
            values,
        )

        # Initialize metadata
        now_timestamp = get_timestamp()
        name = scenario_id if name is None else name
        description = "" if description is None else description
        created = now_timestamp if created is None else created
        updated = now_timestamp if updated is None else updated
        index_diffs = get_diff(model) if index_diffs is None else index_diffs
        metadata = ScenarioMetadata(
            name=name,
            description=description,
            created=created,
            updated=updated,
            index_diffs=index_diffs,
        )

        # Initialize scenario state
        return ScenarioState(
            model=model,
            scenario_id=scenario_id,
            metadata=metadata,
        )

    ##############################
    # Metadata
    ##############################

    def update_metadata_field(
        self,
        field: str,
        value: str,
        scenario_id: str | None = None,
    ) -> None:
        """
        Update a metadata field for the problem or a specific scenario.

        Parameters
        ----------
        field : str
            Metadata field to update ('name' or 'description').
        value : str
            New value for the field.
        scenario_id : str, optional
            If provided, updates scenario metadata; otherwise, updates problem metadata.
        """
        now_timestamp = get_timestamp()

        if scenario_id is not None:
            scenario_state = self.get_scenario_state(scenario_id)
            setattr(scenario_state.metadata, field, value)
            scenario_state.metadata.updated = now_timestamp
        else:
            setattr(self.metadata, field, value)

        self.metadata.updated = now_timestamp

    def update_problem_name(self, name: str) -> None:
        """Update the problem's name."""
        self.update_metadata_field("name", name)

    def update_problem_description(self, description: str) -> None:
        """Update the problem's description."""
        self.update_metadata_field("description", description)

    def update_scenario_name(self, scenario_id: str, name: str) -> None:
        """Update a scenario's name."""
        self.update_metadata_field("name", name, scenario_id)

    def update_scenario_description(self, scenario_id: str, description: str) -> None:
        """Update a scenario's description."""
        self.update_metadata_field("description", description, scenario_id)

    ##############################
    # Helper getters
    ##############################

    def get_scenario_names(self) -> list[str]:
        """
        Get a list of all scenario IDs.

        Returns
        -------
        list[str]
            List of scenario IDs.
        """
        return list(self.instantiated_models.keys())

    def get_scenario_number(self) -> int:
        """
        Get the number of scenarios managed.

        Returns
        -------
        int
            Number of scenarios.
        """
        return len(self.instantiated_models)

    def get_scenario_state(
        self,
        scenario_id: str,
    ) -> ScenarioState:
        """
        Get the ScenarioState for a given scenario ID.

        Parameters
        ----------
        scenario_id : str
            Identifier of the scenario.

        Returns
        -------
        ScenarioState
            The scenario state.
        """
        return self.instantiated_models[scenario_id]

    def get_scenario_data(
        self,
        scenario_id: str,
        situation: str | None = None,
    ) -> ModelOutput | None:
        """
        Get the output data for a scenario and situation.

        Parameters
        ----------
        scenario_id : str
            Identifier of the scenario.
        situation : str, optional
            Situation ID.

        Returns
        -------
        ModelOutput or None
            Output data for the scenario and situation.
        """
        _, situation_id = self._filter_situation(situation)
        return self.get_scenario_state(scenario_id).get_data(situation_id).data

    def get_all_scenario_data(
        self,
        scenario_id: str,
    ) -> dict[str, SituationData]:
        """
        Get all situation data for a scenario.

        Parameters
        ----------
        scenario_id : str
            Identifier of the scenario.

        Returns
        -------
        dict[str, SituationData]
            All situation data for the scenario.
        """
        return self.get_scenario_state(scenario_id).get_all_data()

    ##############################
    # Session
    ##############################

    def _temp_state_from_scenario(
        self,
        scenario_id: str,
        values: dict,
    ) -> ScenarioState:
        """
        Create a temporary scenario state from an existing scenario and new values.

        Parameters
        ----------
        scenario_id : str
            Identifier of the scenario.
        values : dict
            Values to instantiate the temporary scenario.

        Returns
        -------
        ScenarioState
            The temporary scenario state.
        """
        origin = self.get_scenario_state(scenario_id)
        now_timestamp = get_timestamp()
        return self._create_scenario_state(
            scenario_id=scenario_id,
            values=values,
            name=origin.metadata.name,
            description=origin.metadata.description,
            created=now_timestamp,
            updated=now_timestamp,
        )

    def evaluate_session_scenario(
        self,
        scenario_id: str,
        values: dict,
        situation: str | None = None,
        ensemble_size: int = 20,
        evaluate: bool = False,
    ) -> ScenarioState:
        """
        Evaluate a temporary scenario session.

        Parameters
        ----------
        scenario_id : str
            Identifier of the scenario.
        values : dict
            Values to instantiate the temporary scenario.
        situation : str, optional
            Situation ID to evaluate.
        ensemble_size : int, optional
            Ensemble size for evaluation.
        evaluate : bool, optional
            If True, perform evaluation.

        Returns
        -------
        ScenarioState
            The evaluated or prepared scenario state.
        """
        state = self._temp_state_from_scenario(scenario_id, values)
        if evaluate:
            return self._execute_evaluation(state, situation, ensemble_size)
        state.situations = self.get_all_scenario_data(scenario_id)
        return state

    ##############################
    # Helpers
    ##############################

    def _filter_situation(
        self,
        situation: str | None,
    ) -> tuple[dict[ContextVariable, list[SymbolValue]], str]:
        """
        Filter and retrieve the situation by its ID.

        Parameters
        ----------
        situation : str, optional
            Situation ID to filter.

        Returns
        -------
        tuple[dict[ContextVariable, list[SymbolValue]], str]
            Filtered situation dictionary and its name.
        """
        compute_situation: dict[ContextVariable, list[SymbolValue]] = {}
        for s in self.situations:
            for ctx, sym in s.values.items():
                if s.name == situation:
                    compute_situation[ctx] = sym
                    return compute_situation, s.name
        return compute_situation, None
