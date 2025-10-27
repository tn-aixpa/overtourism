# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing

from overtourism.dt_studio.manager.problem.metadata import ProblemMetadata
from overtourism.dt_studio.manager.utils.utils import Dictable

if typing.TYPE_CHECKING:
    from overtourism.dt_studio.manager.scenario.metadata import ScenarioMetadata


class IndexValues(Dictable):
    """Container for model parameter values and their types.

    Stores information about a single model parameter, including its name,
    value (which can be either a constant or distribution parameters),
    and type specification.

    Parameters
    ----------
    index_name : str
        Name of the parameter
    index_value : dict | float
        Value or distribution parameters for the index
    index_type : str
        Type of the index (from IndexType enum)
    """

    def __init__(
        self,
        index_name: str,
        index_value: dict | float,
        index_type: str,
    ) -> None:
        self.index_name = index_name
        self.index_value = index_value
        self.index_type = index_type


class ScenarioValues(Dictable):
    """Container for scenario configuration and its parameters.

    Represents a single scenario configuration including metadata and
    a collection of IndexValues representing model parameters.

    Parameters
    ----------
    scenario_id : str
        Unique identifier for the scenario
    indexes : list[IndexValues]
        List of IndexValues representing model parameters
    metadata : ScenarioMetadata
        Metadata for the scenario
    """

    def __init__(
        self,
        scenario_id: str,
        indexes: list[IndexValues],
        metadata: ScenarioMetadata,
    ) -> None:
        self.scenario_id = scenario_id
        self.indexes = indexes
        self.metadata = metadata


class ProblemValues(Dictable):
    """Container for complete problem definition with multiple scenarios.

    Top-level container representing a complete problem configuration,
    including metadata and a collection of scenarios.

    Parameters
    ----------
    problem_id : str
        Unique identifier for the problem
    metadata : ProblemMetadata
        Metadata for the problem
    scenarios : list[ScenarioValues]
        List of scenario configurations
    """

    def __init__(
        self,
        problem_id: str,
        scenarios: list[ScenarioValues],
        metadata: ProblemMetadata,
    ) -> None:
        self.problem_id = problem_id
        self.scenarios = scenarios
        self.metadata = metadata
