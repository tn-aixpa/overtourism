# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum


class ViewSettings:
    """
    Class to include settings for the viewer.

    Attributes
    ----------
    name : str
        The name of the scenario
    x : str
        The name of the x-axis
    y : str
        The name of the y-axis
    view_type : str
        The type of the view (Unidimensional or bidimensional)
    uncertainty : bool
        Whether to plot the uncertainty heatmap
    constraint : str
        The name of the constraint to plot
    presence : bool
        Whether to plot the presence scatter
    name_to_linestyle : dict[str, str]
        A dictionary mapping constraint names to line styles
    """

    def __init__(
        self,
        name: str | None = None,
        x: str | None = None,
        y: str | None = None,
        view_type: str | None = None,
        uncertainty: bool = False,
        constraint: str | None = None,
        presence: bool = False,
        name_to_linestyle: dict[str, str] | None = None,
    ):
        self.name = name
        self.x = x
        self.y = y
        self.view_type = view_type
        self.uncertainty = uncertainty
        self.constraint = constraint
        self.presence = presence
        self.name_to_linestyle = name_to_linestyle


class ViewType(Enum):
    UNIDIMENSIONAL = "Unidimensionale"
    BIDIMENSIONAL = "Bidimensionale"
    UNCERTAINTY = "Incertezza"
