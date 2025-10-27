# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import glob
from pathlib import Path

import yaml
from scipy import stats

from overtourism.dt_studio.manager.problem.metadata import (
    ProblemMetadata,
    default_problem_metadata,
)
from overtourism.dt_studio.manager.scenario.metadata import ScenarioMetadata

from ..indexes.enums import IndexType
from ..scenario.metadata import default_scenario_metadata
from .classes import IndexValues, ProblemValues, ScenarioValues


def scenario_values(
    scenario_id: str,
    values: dict,
    metadata: ScenarioMetadata | None = None,
) -> ScenarioValues:
    """Create a ScenarioValues object from model parameters.

    Processes a dictionary of model values and converts them into a structured
    ScenarioValues object, handling both constant values and statistical distributions.

    Parameters
    ----------
    scenario_id : str
        Unique identifier for the scenario
    values : dict
        Dictionary containing model parameters and their values/distributions
    metadata : ScenarioMetadata
        Metadata for the scenario

    Returns
    -------
    ScenarioValues
        Object containing structured scenario data and processed model parameters

    Notes
    -----
    Supported value types:
    - Numeric constants (int, float)
    - scipy.stats distributions (uniform, lognorm, triang)
    Other value types are ignored during processing.
    """
    indexes = []
    for key, value in values.items():
        # Skip non-numeric and non-distribution values
        if not isinstance(value, (int, float, stats._distn_infrastructure.rv_frozen)):
            continue

        # Process scipy.stats distribution objects
        elif isinstance(value, stats._distn_infrastructure.rv_frozen):
            indexes.append(IndexValues(key, value.kwds, value.dist.name))

        # Process numeric constants
        else:
            indexes.append(IndexValues(key, value, IndexType.CONSTANT.value))

    return ScenarioValues(scenario_id=scenario_id, indexes=indexes, metadata=metadata)


def prepare_values(scenario_data: ScenarioValues) -> dict:
    """
    Prepare indexes values for instantiation.

    Parameters
    ----------
    scenario_data : ScenarioValues
        Class containing scenario data.

    Returns
    -------
    dict
        Dictionary of indexes values

    Notes
    -----
    Supported distribution types:
    - CONSTANT: Direct value assignment
    - UNIFORM: loc, scale parameters
    - LOGNORM: loc, scale, s parameters
    - TRIANG: loc, scale, c parameters
    """
    values = {}
    for val in scenario_data.indexes:
        match val.index_type:
            case IndexType.CONSTANT.value:
                values[val.index_name] = val.index_value
            case IndexType.UNIFORM.value:
                values[val.index_name] = stats.uniform(
                    **{
                        "loc": val.index_value["loc"],
                        "scale": val.index_value["scale"],
                    }
                )
            case IndexType.LOGNORM.value:
                values[val.index_name] = stats.lognorm(
                    **{
                        "loc": val.index_value["loc"],
                        "scale": val.index_value["scale"],
                        "s": val.index_value["s"],
                    }
                )
            case IndexType.TRIANG.value:
                values[val.index_name] = stats.triang(
                    **{
                        "loc": val.index_value["loc"],
                        "scale": val.index_value["scale"],
                        "c": val.index_value["c"],
                    }
                )
    return values


def save_yaml(data: dict, filename: Path) -> None:
    """Save dictionary data to a YAML file.

    Parameters
    ----------
    data : dict
        Data to be saved
    filename : Path
        Path to the output YAML file
    """
    with open(filename, "w") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)


def load_yaml(filename: Path | str) -> dict:
    """Load data from a YAML file.

    Parameters
    ----------
    filename : Path | str
        Path to the YAML file to read

    Returns
    -------
    dict
        Parsed YAML content
    """
    with open(filename, "r") as f:
        return yaml.safe_load(f)


def save_problem(problem: dict, filename: str) -> None:
    """Save a problem definition to a YAML file.

    Parameters
    ----------
    problem : dict
        Problem definition data
    filename : str
        Path where to save the YAML file
    """
    save_yaml(problem, filename)


def get_glob(path: str) -> list[str]:
    """Get list of all files in a directory.

    Parameters
    ----------
    path : str
        Directory path to search

    Returns
    -------
    list[str]
        List of full paths to all files in the directory
    """
    return glob.glob(str(Path(path) / "*"))


def load_problem(filename: str) -> ProblemValues:
    """
    Load nested dictionary into ProblemValues class hierarchy.

    Parameters
    ----------
    filename : str
        Path to the YAML file containing problem data

    Returns
    -------
    ProblemValues
        Initialized problem values object with all nested data
    """
    data = load_yaml(filename)

    problem_id = data["problem_id"]

    # Create metadata object
    try:
        metadata = ProblemMetadata(**data["metadata"])
    except KeyError:
        metadata = default_problem_metadata(problem_id)

    # Process scenarios
    scenarios = []
    if "scenarios" in data:
        for scenario_data in data["scenarios"]:
            scenario_id = scenario_data["scenario_id"]

            # Create scenario metadata
            try:
                scenario_metadata = ScenarioMetadata(**scenario_data["metadata"])
            except KeyError:
                scenario_metadata = default_scenario_metadata(scenario_id)

            # Process indexes
            indexes = []
            if "indexes" in scenario_data:
                for index_data in scenario_data["indexes"]:
                    index = IndexValues(
                        index_name=index_data["index_name"],
                        index_value=index_data["index_value"],
                        index_type=index_data["index_type"],
                    )
                    indexes.append(index)

            # Create scenario with collected data
            scenario = ScenarioValues(
                scenario_id=scenario_id, indexes=indexes, metadata=scenario_metadata
            )
            scenarios.append(scenario)

    # Create and return problem values
    return ProblemValues(problem_id=problem_id, scenarios=scenarios, metadata=metadata)
