# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from civic_digital_twins.dt_model.internal.sympyke.symbol import SymbolValue
    from civic_digital_twins.dt_model.symbols.context_variable import ContextVariable
    from civic_digital_twins.dt_model.symbols.presence_variable import PresenceVariable


class ModelOutput:
    """
    Container for model evaluation results and computed metrics.
    """

    def __init__(
        self,
        x_max: int,
        y_max: int,
        sample_x: list[float],
        sample_y: list[float],
        kpis: dict[str, float],
        uncertainty: list[float],
        uncertainty_by_constraint: dict[str, list[float]],
        constraint_curves: dict[str, list[float]],
        usage: list[float],
        usage_by_constraint: dict[str, list[float]],
        usage_uncertainty: list[float],
        usage_uncertainty_by_constraint: dict[str, list[float]],
        capacity_mean: float,
        capacity_mean_by_constraint: dict[str, float],
    ) -> None:
        self.x_max = x_max
        self.y_max = y_max
        self.sample_x = sample_x
        self.sample_y = sample_y
        self.kpis = kpis
        self.uncertainty = uncertainty
        self.uncertainty_by_constraint = uncertainty_by_constraint
        self.constraint_curves = constraint_curves
        self.usage = usage
        self.usage_by_constraint = usage_by_constraint
        self.usage_uncertainty = usage_uncertainty
        self.usage_uncertainty_by_constraint = usage_uncertainty_by_constraint
        self.capacity_mean = capacity_mean
        self.capacity_mean_by_constraint = capacity_mean_by_constraint

    def to_dict(self) -> dict:
        return self.__dict__


class Sampler:
    """Helper class for data sampling operations.

    Parameters
    ----------
    sampling_function : Callable
        Function to perform the sampling
    sampling_dicts : list[dict]
        Configuration dictionaries for sampling

    Attributes
    ----------
    sampling_function : Callable
        Stored sampling function
    sampling_dicts : list[dict]
        Stored sampling configurations
    """

    def __init__(self, sampling_dicts: list[dict]) -> None:
        self.sampling_dicts = sampling_dicts


class Situation:
    """Container for situation-specific model configurations.

    Parameters
    ----------
    name : str
        Situation identifier
    description : str
        Detailed description of the situation
    values : dict[ContextVariable, list[SymbolValue]]
        Variable values defining the situation

    Attributes
    ----------
    name : str
        Situation identifier
    description : str
        Detailed description
    values : dict[ContextVariable, list[SymbolValue]]
        Variable configurations
    """

    def __init__(
        self,
        name: str,
        description: str,
        values: dict[ContextVariable, list[SymbolValue]],
    ) -> None:
        self.name = name
        self.description = description
        self.values = values


class Grid:
    """Configuration for evaluation grid parameters.

    Parameters
    ----------
    grid : dict[PresenceVariable, np.ndarray]
        Grid values for each presence variable
    x : np.ndarray
        X-axis grid coordinates
    y : np.ndarray
        Y-axis grid coordinates
    x_max : float
        Maximum x-axis value
    y_max : float
        Maximum y-axis value

    Attributes
    ----------
    grid : dict[PresenceVariable, np.ndarray]
        Grid configuration
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    x_max : float
        X-axis limit
    y_max : float
        Y-axis limit
    """

    def __init__(
        self,
        grid: dict[PresenceVariable, np.ndarray],
        x: np.ndarray,
        y: np.ndarray,
        x_max: float,
        y_max: float,
    ) -> None:
        self.grid = grid
        self.x = x
        self.y = y
        self.x_max = x_max
        self.y_max = y_max


class StoreConfig:
    """Configuration for evaluation data storage.

    Parameters
    ----------
    store_type : str
        Type of storage backend to use
    config : dict | None, optional
        Storage-specific configuration options

    Attributes
    ----------
    store_type : str
        Selected storage type
    config : dict | None
        Storage configuration
    """

    def __init__(self, store_type: str, config: dict | None = None) -> None:
        self.store_type = store_type
        self.config = config
