# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import typing
from datetime import datetime
from typing import Any, cast
from uuid import uuid4

from civic_digital_twins.dt_model.symbols.index import Distribution
from overtourism.dt_studio.viewer.indexes.index import (
    VizConstIndex,
    VizIndex,
    VizLognormDistIndex,
    VizTriangDistIndex,
    VizUniformDistIndex,
)
from scipy import stats

if typing.TYPE_CHECKING:
    from overtourism.dt_studio.manager.config.classes import ModelOutput
    from overtourism.dt_studio.manager.problem.manager import ProblemManager
    from overtourism.dt_studio.viewer.viewer import ModelViewer


BASE_ROUTE = "/api/v1"


def load_class(
    module_name: str, path: str, instance_name: str
) -> ProblemManager | ModelViewer:
    """
    Load instance from path.

    Parameters
    ----------
    module_name : str
        Name of module where model is defined.
    path : str
        Path to module.
    instance_name : str
        Name of instance variable.

    Returns
    -------
    ProblemManager
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        raise ImportError(f"Module '{module_name}' not found in path {path}")
    imported_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported_module)
    return getattr(imported_module, instance_name)


def prepare_values_for_eval(values: dict, viewer: ModelViewer) -> dict:
    """
    Prepare values for model evaluation.

    Parameters
    ----------
    values : dict
        Values to prepare.
    viewer : ModelViewer
        Model viewer.

    Returns
    -------
    dict
        Prepared values.
    """
    new_vals = {}
    for k, v in values.items():
        for i in viewer.indexes:
            if i.index_id == k:
                new_vals[k] = format_value(i, v)
    return new_vals


def format_value(idx: VizIndex, value: Any) -> Any:
    if isinstance(idx, VizConstIndex):
        return value
    elif isinstance(idx, VizUniformDistIndex):
        (f, t) = value
        # Prevent division by zero warning and color disappearing from the plot
        # as documented in https://github.com/tn-aixpa/dt-model/issues/10
        diff = max(t - f, 1e-04)
        return cast(Distribution, stats.uniform(loc=f, scale=diff))
    elif isinstance(idx, VizLognormDistIndex):
        return cast(Distribution, stats.lognorm(loc=idx.loc, scale=value, s=idx.s))
    elif isinstance(idx, VizTriangDistIndex):
        (f, t) = value
        # Prevent division by zero warning and color disappearing from the plot
        # as documented in https://github.com/tn-aixpa/dt-model/issues/10
        diff = max(t - f, 1e-04)
        return cast(Distribution, stats.triang(loc=f, scale=diff, c=idx.c))
    return value


def get_id(scenario_id: str, session_id: str) -> str:
    return f"{scenario_id}_{session_id}_{uuid4().hex}"


def get_timestamp() -> str:
    """
    Get the current timestamp timezoned.

    Returns
    -------
    str
        The current timestamp.
    """
    return datetime.now().astimezone().isoformat()


def arrange_data(data: ModelOutput) -> dict:
    d = {}
    d["points"] = {}
    d["points"]["uncertainty"] = []
    d["points"]["uncertainty_by_constraint"] = {}
    for i in data.uncertainty_by_constraint:
        d["points"]["uncertainty_by_constraint"][i] = []

    for i in list(
        zip(
            data.sample_x,
            data.sample_y,
            data.uncertainty,
            data.usage,
            data.usage_uncertainty,
        )
    ):
        d["points"]["uncertainty"].append(
            {
                "tourists": i[0],
                "excursionists": i[1],
                "index": i[2],
                "usage": i[3],
                "usage_uncertainty": i[4],
            }
        )

    for k, v in data.uncertainty_by_constraint.items():
        for i in list(
            zip(
                data.sample_x,
                data.sample_y,
                v,
                data.usage_by_constraint[k],
                data.usage_uncertainty_by_constraint[k],
            )
        ):
            d["points"]["uncertainty_by_constraint"][k].append(
                {
                    "tourists": i[0],
                    "excursionists": i[1],
                    "index": i[2],
                    "usage": i[3],
                    "usage_uncertainty": i[4],
                }
            )

    d["kpis"] = data.kpis
    d["x_max"] = data.x_max
    d["y_max"] = data.y_max
    d["capacity_mean"] = data.capacity_mean
    d["capacity_mean_by_constraint"] = data.capacity_mean_by_constraint
    d["constraint_curves"] = data.constraint_curves
    return d


def get_widget_by_group(viewer: ModelViewer, groups: list[str]) -> list[str]:
    """
    Get widgets by group.

    Parameters
    ----------
    viewer : ModelViewer
        Model viewer.
    groups : list[str]
        List of groups to filter widgets.

    Returns
    -------
    list[str]
        List of widget IDs.
    """
    widgets = []
    for i in viewer.indexes:
        if i.group in groups:
            widgets.append(i.index_id)
    return widgets
