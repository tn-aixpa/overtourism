# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing

import numpy as np
from overtourism.dt_studio.viewer.models import ViewSettings
from plotly import graph_objects as go

from .utils import colorscale

if typing.TYPE_CHECKING:
    from overtourism.dt_studio.viewer.models import ViewSettings


def presence_bars(x, y1, y2, name1, name2) -> list[go.Bar]:
    """Create stacked bar plots for presence visualization.

    Parameters
    ----------
    x : list[float]
        X-axis values for bar positions
    y1 : list[float]
        Values for first presence component
    y2 : list[float]
        Values for second presence component
    name1 : str
        Label for first presence component
    name2 : str
        Label for second presence component

    Returns
    -------
    list[go.Bar]
        List of two Plotly bar traces configured for stacking
    """
    return [
        go.Bar(
            x=x,
            y=y1,
            name=name1,
            marker_color="rgba(169, 169, 169, 0.8)",
            yaxis="y2",
        ),
        go.Bar(
            x=x,
            y=y2,
            name=name2,
            marker_color="rgba(128, 128, 128, 0.8)",
            yaxis="y2",
        ),
    ]


def usage_scatter(x, y, color_values) -> go.Scatter:
    """Create scatter plot for usage visualization.

    Parameters
    ----------
    x : list[float]
        X-axis values for scatter points
    y : list[float]
        Y-axis values representing usage levels
    color_values : list[float]
        List of color values for each point

    Returns
    -------
    go.Scatter
        Plotly scatter trace with black-outlined colored markers
    """
    return go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="Utilizzo",
        marker=dict(
            color=color_values,
            colorscale=colorscale,
            cmin=0,
            cmax=1,
            line=dict(color="black", width=1),
            size=5,
        ),
        yaxis="y3",
    )


def capacity_line(x, y, name) -> go.Scatter:
    """Create the capacity threshold line.

    Parameters
    ----------
    x : list[float]
        X-axis values for line coordinates
    y : list[float]
        Y-axis values for line coordinates (usually constant)
    name : str
        Label for the capacity line in the legend

    Returns
    -------
    go.Scatter
        Plotly line trace with dashed black style
    """
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=name,
        line=dict(
            color="black",
            width=2,
            dash="dash",
        ),
        yaxis="y3",
    )


def get_unidimensional_layout(
    view: ViewSettings, title: str, ymax: int, top: np.ndarray | None
) -> dict:
    """Create layout configuration for unidimensional figure.

    Configures a complex layout with multiple y-axes:
    - Primary y-axis (hidden) spanning full height
    - Right y-axis for presence values
    - Left y-axis for usage/overcrowding levels

    Parameters
    ----------
    view : ViewSettings
        Configuration object with visualization preferences
    title : str
        Base title for the plot
    ymax : int
        Maximum value for usage/overcrowding y-axis
    top : np.ndarray | None
        Optional maximum value for presence y-axis

    Returns
    -------
    dict
        Plotly layout configuration including title, axes, and legend settings
    """
    return dict(
        title=dict(
            text=title,
            font=dict(size=10),
        ),
        plot_bgcolor="white",  # Add white background
        xaxis=dict(
            title=(
                "Giorni (ordinati per livello di affollamento)"
                if view.constraint is None
                else "Giorni (ordinati per livello di utilizzo)"
            ),
            showgrid=False,
            gridcolor="lightgray",
            showline=True,
            linewidth=1,
            linecolor="gray",
        ),
        yaxis=dict(
            showticklabels=False, domain=[0, 1]
        ),  # Make primary axis span full height
        yaxis2=dict(
            title="Presenze",
            side="right",
            range=[0, top] if top is not None else None,
            showticklabels=view.presence,
            overlaying="y",  # Overlay on primary axis
        ),
        yaxis3=dict(
            title=(
                "Livello di sovraffollamento"
                if view.constraint is None
                else f"Livello di utilizzo del sottosistema {view.constraint}"
            ),
            side="left",
            range=[0, ymax],
            tickformat=",.0%" if view.constraint is None else None,
            overlaying="y",  # Overlay on primary axis
        ),
        barmode="stack",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )


def unidimensional_figure(
    view: ViewSettings,
    usage_uncertainty: list,
    capacity_mean: float,
    ymax: int,
    sample_t: list,
    sample_e: list,
    usage: list,
    top: np.ndarray | None,
    title: str,
) -> go.Figure:
    """Create a unidimensional visualization of tourism data with capacity and usage.

    This function generates a complex Plotly figure that can display:
    - An uncertainty heatmap showing the distribution of values
    - Stacked bar charts showing presence data
    - A capacity threshold line
    - A scatter plot showing usage levels with sustainability indicators

    Parameters
    ----------
    view : ViewSettings
        Configuration object controlling visualization options and styles
    capacity : list | None
        2D matrix containing capacity distribution values, or None if not available
    capacity_mean : float
        Mean capacity threshold value
    ymax : int
        Maximum value for y-axis range
    sample_t : list
        First component of presence data
    sample_e : list
        Second component of presence data
    usage : list
        Usage values for scatter plot
    top : np.ndarray | None
        Optional maximum value for presence axis scaling
    title : str
        Title for the plot

    Returns
    -------
    go.Figure
        Plotly figure object containing the complete visualization
    """
    fig = go.Figure()

    # Plot presence bars if enabled
    if view.presence:
        x_values = np.linspace(0, 100, len(sample_t))
        for bar in presence_bars(x_values, sample_e, sample_t, view.x, view.y):
            fig.add_trace(bar)

    # Plot capacity line
    cap_name = (
        "Soglia di sovraffollamento"
        if view.constraint is None
        else "Capacità di carico"
    )
    fig.add_trace(capacity_line([0, 100], [capacity_mean, capacity_mean], cap_name))

    # Add usage scatter plot
    x_values = np.linspace(0, 100, len(usage))
    fig.add_trace(usage_scatter(x_values, usage, usage_uncertainty))

    # Update layout
    fig.update_layout(get_unidimensional_layout(view, title, ymax, top))

    return fig
