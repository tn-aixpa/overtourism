# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing

import numpy as np

from ..manager.indexes.enums import IndexType
from ..manager.io.utils import load_yaml
from .indexes.utils import build_indexes_from_config
from .models import ViewType
from .plots.bidimensional import bidimensional_figure
from .plots.unidimensional import unidimensional_figure

if typing.TYPE_CHECKING:
    from plotly import graph_objects as go

    from ..manager.config.classes import ModelOutput
    from .models import ViewSettings


class ModelViewer:
    def __init__(self, path: str) -> None:
        config = load_yaml(path)
        self.indexes = build_indexes_from_config(config)
        self.groups = self._build_groups()

    ######################
    # Viz
    ######################

    def viz(
        self,
        data: ModelOutput,
        view: ViewSettings,
        title: str,
    ) -> go.Figure:
        if view.view_type == ViewType.BIDIMENSIONAL.value:
            uncertainty = self._get_uncertainty(
                data,
                view.constraint,
            )
            point_data = {
                "x": data.sample_x,
                "y": data.sample_y,
                "z": uncertainty,
            }
            return bidimensional_figure(
                view,
                point_data,
                data.constraint_curves,
                data.x_max,
                data.y_max,
            )
        elif view.view_type == ViewType.UNIDIMENSIONAL.value:
            capacity_mean = self._get_1d_capacity_capacity_mean(
                data,
                view.constraint,
            )
            sample_t, sample_e, usage, usage_uncertainty = self._get_1d_sample(
                data,
                view.constraint,
            )
            ymax, top = self._get_1d_ymax_top(
                usage,
                capacity_mean,
                sample_t,
                sample_e,
            )
            return unidimensional_figure(
                view,
                usage_uncertainty,
                capacity_mean,
                ymax,
                sample_t,
                sample_e,
                usage,
                top,
                title,
            )
        else:
            raise ValueError(f"Unknown view type: {view.view_type}")

    ######################
    # 1d viz
    ######################

    @staticmethod
    def _get_1d_sample(
        data: ModelOutput,
        constraint: str | None = None,
    ) -> tuple:
        sample_x = data.sample_x
        sample_y = data.sample_y
        if constraint is None:
            usage = data.usage
            usage_uncertainty = data.usage_uncertainty
        else:
            usage = data.usage_by_constraint[constraint]
            usage_uncertainty = data.usage_uncertainty_by_constraint[constraint]
        sample = sorted(
            zip(sample_x, sample_y, usage, usage_uncertainty), key=lambda a: a[2]
        )
        return zip(*sample)

    def _get_1d_capacity_capacity_mean(
        self,
        data: ModelOutput,
        constraint: str | None = None,
    ) -> float:
        if constraint is None:
            return data.capacity_mean
        return data.capacity_mean_by_constraint[constraint]

    @staticmethod
    def _get_1d_ymax_top(
        usage,
        capacity_mean,
        sample_t,
        sample_e,
    ) -> tuple:
        """Prepare data for unidimensional visualization."""
        ymax = int(max(max(usage), capacity_mean) * 1.2)
        top = int(
            1.2
            * max(np.array(sample_e) + np.array(sample_t))
            * max(1, capacity_mean / max(usage))
        )
        return ymax, top

    ######################
    # 2d viz
    ######################

    @staticmethod
    def _get_uncertainty(
        data: ModelOutput,
        constraint: str | None = None,
    ) -> list[float]:
        if constraint is None:
            return data.uncertainty
        return data.uncertainty_by_constraint[constraint]

    @staticmethod
    def _get_point_data(
        uncertainty_matrix: dict,
    ) -> dict:
        """
        Prepare data for uncertainty visualization.
        """
        data = {"x": [], "y": [], "z": []}
        for i in uncertainty_matrix:
            data["x"].append(i["tourists"])
            data["y"].append(i["excursionists"])
            data["z"].append(i["index"])
        return data

    ######################
    # Groups
    ######################

    def get_groups(self) -> list:
        return self.groups

    def _build_groups(self) -> list:
        """
        Build groups from model capacities and constraints
        """
        groups = {}
        for cap_or_constr in self.indexes:
            group_id = (
                cap_or_constr.group if cap_or_constr.group is not None else "general"
            )
            if group_id not in groups:
                groups[group_id] = self._get_group(group_id)
            groups[group_id]["parameters"].append(cap_or_constr)
        return list(groups.values())

    @staticmethod
    def _get_group(group_id: str) -> dict:
        return {
            "id": group_id,
            "label": group_id,
            "parameters": [],
        }

    def get_widgets(self, vals: dict):
        widgets = {}
        for i in self.indexes:
            idx = i.copy()
            if idx.group not in widgets:
                widgets[i.group] = []
            if idx.index_id in vals:
                if idx.index_type == IndexType.CONSTANT.value:
                    idx.v = vals[idx.index_id]
                else:
                    value = vals[idx.index_id].kwds
                    idx.loc = value["loc"]
                    idx.scale = value["scale"]

            widgets[idx.group].append(idx.to_dict())
        return widgets
