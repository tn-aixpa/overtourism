# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing

from civic_digital_twins.dt_model.model.instantiated_model import InstantiatedModel

from overtourism.dt_studio.manager.config.classes import ModelOutput

if typing.TYPE_CHECKING:
    from overtourism.dt_studio.manager.scenario.metadata import ScenarioMetadata


class SituationData:
    """Container for storing situation-specific model outputs.

    A helper class that encapsulates model output data for a specific situation
    within a scenario.

    Parameters
    ----------
    data : ModelOutput | None, optional
        Model output data for the situation, by default None

    Attributes
    ----------
    data : ModelOutput | None
        Stored model output data
    """

    def __init__(self, data: ModelOutput | None = None) -> None:
        self.data = data


class ScenarioState:
    """Manager for scenario state and associated situation data.

    Maintains the state of a scenario including its model, metadata, and
    collection of situation-specific data. Provides methods for data
    management and state copying.

    Parameters
    ----------
    model : InstantiatedModel
        The instantiated model associated with this scenario
    scenario_id : str
        Unique identifier for the scenario
    metadata : ScenarioMetadata
        Metadata associated with the scenario

    Attributes
    ----------
    scenario_id : str
        Unique scenario identifier
    model : InstantiatedModel
        Associated model instance
    metadata : ScenarioMetadata
        Metadata associated with the scenario
    situations : dict[str, SituationData]
        Dictionary mapping situation IDs to their data
    is_evaluating : bool
        Flag indicating if scenario is currently being evaluated
    """

    def __init__(
        self,
        model: InstantiatedModel,
        scenario_id: str,
        metadata: ScenarioMetadata,
    ) -> None:
        self.scenario_id = scenario_id
        self.model = model
        self.metadata = metadata
        self.situations: dict[str, SituationData] = {}
        self.is_evaluating: bool = False

    def add_data(
        self,
        situation_id: str,
        data: ModelOutput | None = None,
    ) -> None:
        """Add or update situation data in the scenario.

        Parameters
        ----------
        situation_id : str
            Unique identifier for the situation
        data : ModelOutput | None, optional
            Model output data for the situation, by default None
        """
        self.situations[situation_id] = SituationData(data)

    def get_data(self, situation_id: str) -> SituationData:
        """Retrieve situation data, creating empty entry if not present.

        Parameters
        ----------
        situation_id : str
            Unique identifier for the situation

        Returns
        -------
        SituationData
            Stored situation data or newly created empty container
        """
        # Create empty situation data if not present
        if situation_id not in self.situations:
            self.situations[situation_id] = SituationData()
        return self.situations[situation_id]

    def get_all_data(self) -> dict[str, SituationData]:
        """Retrieve all situation data.

        Returns
        -------
        dict[str, SituationData]
            Dictionary mapping situation IDs to their data
        """
        return self.situations

    def copy(self) -> ScenarioState:
        """Create a deep copy of the scenario state.

        Returns
        -------
        ScenarioState
            New scenario state instance with copied data
        """
        state = ScenarioState(
            model=self.model,
            scenario_id=self.scenario_id,
            metadata=self.metadata,
        )
        state.situations = self.situations.copy()
        return state
