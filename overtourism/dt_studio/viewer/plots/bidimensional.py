# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing

from overtourism.dt_studio.viewer.plots.utils import colorscale
from plotly import graph_objects as go

if typing.TYPE_CHECKING:
    from overtourism.dt_studio.viewer.models import ViewSettings


def bidimensional_figure(
    view: ViewSettings,
    data: list[dict[str, float]],
    constraint_curves: dict[str, list[float]],
    x_max: float,
    y_max: float,
) -> go.Figure:
    """Create a visualization of tourism data with uncertainty.

    Parameters
    ----------
    view : ViewSettings
        Configuration object controlling visualization options and styles
    data : list[dict[str, float]]
        List of dictionaries containing data points with uncertainty values
    title : str
        Title for the plot

    Returns
    -------
    go.Figure
        Plotly figure representing the uncertainty visualization
    """
    fig = go.Figure()

    fig.add_trace(get_uncertainty_scatter(**data))

    # Add constraint curves, optionally filtered by the selected constraint
    # Each constraint uses a different line style defined in view settings
    for constraint, modal in constraint_curves.items():
        if view.constraint is None or constraint == view.constraint:
            ls = view.name_to_linestyle[constraint]
            x_modal, y_modal = modal
            fig.add_trace(constraint_lines(x_modal, y_modal, constraint, ls))

    fig.update_layout(get_uncertainty_layout(x_max, y_max))
    return fig


def get_uncertainty_scatter(
    x: list[float],
    y: list[float],
    z: list[float],
) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        customdata=z,
        mode="markers",
        marker=dict(
            color=z,
            colorscale=colorscale,
            reversescale=True,
            size=5,
            line=dict(color="black", width=0.5),  # Color of the contour of the points
        ),
        hovertemplate="<b>contesto:</b> %{customdata}<br>"
        + "<b>turisti"
        + ":</b> %{x}<br>"
        + "<b>escursionisti"
        + ":</b> %{y}<br>"
        + "<extra></extra>",  # This removes the secondary box with trace name
        showlegend=False,
    )


def get_uncertainty_layout(
    x_max: float, y_max: float, x_title: str = "Turisti", y_title: str = "Escursionisti"
) -> dict:
    """Create layout configuration for uncertainty figure.

    Parameters
    ----------
    view : ViewSettings
        Configuration object with visualization preferences
    title : str
        Base title for the plot

    Returns
    -------
    dict
        Plotly layout configuration including title, axes, and legend settings
    """
    return dict(
        plot_bgcolor="white",
        margin=dict(
            t=50, b=130, l=20, r=20
        ),  # Reserve extra space at the bottom (to be kept also when the legend is not shown)
        xaxis=dict(
            title=x_title,
            range=[0, x_max],  # Set the range for the x-axis
            tickformat=".0f",
            showline=True,
            linewidth=1,
            linecolor="grey",
            # scaleanchor="y",
            # scaleratio=1,
            dtick=1000,
            showgrid=False,
        ),
        yaxis=dict(
            # scaleratio=1,
            title=y_title,
            range=[0, y_max],  # Set the range for the y-axis
            tickformat=".0f",
            showline=True,
            linewidth=1,
            linecolor="grey",
            dtick=1000,
            showgrid=False,
        ),
        width=550,  # Width of figure in px
        height=550,  # Height of figure in px
        shapes=[
            # draw an upper border for the plot area
            dict(
                type="line",
                x0=0,
                y0=y_max,
                x1=x_max,
                y1=y_max,
                xref="x",
                yref="y",
                line=dict(
                    color="grey",
                    width=2,
                ),
            ),
            # draw a right border for the plot area
            dict(
                type="line",
                x0=x_max,
                y0=0,
                x1=x_max,
                y1=y_max,
                xref="x",
                yref="y",
                line=dict(
                    color="grey",
                    width=2,
                ),
            ),
        ],
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,  # Adjust this value to position the legend below the graph
            xanchor="center",
            x=0.5,
        ),
    )


def constraint_lines(
    x: list[float],
    y: list[float],
    name: str,
    ls: str,
) -> go.Scatter:
    """Create line plot for constraint curves.

    Parameters
    ----------
    x : list[float]
        X-coordinates of the constraint curve
    y : list[float]
        Y-coordinates of the constraint curve
    name : str
        Name of the constraint for legend
    ls : str
        Line style (dash pattern)

    Returns
    -------
    go.Scatter
        Plotly line trace with specified style
    """
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=name,
        line=dict(color="black", width=2, dash=ls),
        showlegend=True,
    )
