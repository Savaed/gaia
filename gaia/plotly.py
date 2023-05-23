from itertools import zip_longest
from typing import Any, Iterable, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from gaia.data.models import PeriodicEvent, TceLabel


class Margins(TypedDict):
    left: int
    rigth: int
    top: int
    bottom: int


def _to_plotly_margins(margins: Margins) -> dict[str, int]:
    return {margin[0]: px for margin, px in margins.items()}  # type: ignore


def plot_time_series(
    data: dict[str, npt.NDArray[np.float_]] | pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str | None = None,
    hover_data: list[str] | None = None,
    color: str | None = None,
    period_edges: str | None = None,
    margins: Margins | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Create a `plotly` scatter plot.

    Args:
        data (dict[str, npt.NDArray[np.float_]] | pd.DataFrame): Time series to plot
        x (str): Name of X axis data
        y (str): Name of Y axis data
        hover_data (list[str] | None, optional): What to display when hover on data points.
            Defaults to None.
        color (str | None, optional): Specify by what markers should be colorized. Defaults to None.
        period_edges (Iterable[float] | None, optional): Endpoint of observation period.
            Defaults to None.
        margins: (Margins | None, optional): Margins for plot in `px` units.
            If None then `Margins(left=0, rigth=0, bottom=0, top=30)` is used. Defaults to None
        **kwargs (Any): Named parameters than should be passed to `px.scatter()` function

    Returns:
        go.Figure: `plotly.graph_objects.Figure` scatter plot which can be modified futher
    """
    # Extend 'edges' to the series length as they are just a few values (one per observation period)
    values = list(zip_longest(*data.values()))
    df = pd.DataFrame(values, columns=data.keys())
    margins = margins or Margins(left=0, rigth=0, bottom=0, top=30)

    fig = px.scatter(df, x=x, y=y, hover_data=hover_data, color=color, title=title, **kwargs)
    fig.update_traces(marker={"opacity": 0.75, "size": 5})
    fig.update_layout(margin=_to_plotly_margins(margins))

    if color:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", xanchor="left", y=1.02))

    if period_edges:
        edges = df[period_edges].values
        edges = sorted(edges[~np.isnan(edges)])[:-1]  # Do not plot the last period endpoint
        for edge in edges:
            fig.add_vline(x=edge, line_width=2, line_dash="dash", line_color="green")

    return fig


def plot_empty_scatter(margins: Margins | None = None) -> go.Figure:
    fig = px.scatter()
    margins = margins or Margins(left=0, rigth=0, bottom=0, top=30)
    return fig.update_layout(margin=_to_plotly_margins(margins))


def plot_tces_histograms(events: Iterable[PeriodicEvent]) -> tuple[go.Figure, go.Figure]:
    """Create `plotly` histograms for the TCE orbital periods and transit durations.

    Args:
        events (Iterable[PeriodicEvent]): Transits for all TCEs

    Returns:
        tuple [go.Figure, go.Figure]: Orbital period and transit duration histograms, with the
            x-axis on a logarithmic scale
    """
    return (
        px.histogram(
            x=np.log10([event.period for event in events]),
            marginal="box",
            labels={"x": "Orbital period [log10(day)]"},
            height=300,
        ),
        px.histogram(
            x=np.log10([event.duration for event in events]),
            marginal="box",
            labels={"x": "Transit duration [log10(hour)]"},
            height=300,
        ),
    )


def plot_tces_classes_distribution(distribution: dict[TceLabel, int]) -> go.Figure:
    unlabeled_count = distribution[TceLabel.UNKNOWN]
    all_count = sum(distribution.values()) - distribution[TceLabel.FP]
    labels = [
        TceLabel.UNKNOWN.name,
        "KNOWN",
        TceLabel.PC.name,
        TceLabel.FP.name,
        TceLabel.AFP.name,
        TceLabel.NTP.name,
    ]
    parents = ["", "", "KNOWN", "KNOWN", TceLabel.FP.name, TceLabel.FP.name]
    count = [
        unlabeled_count,
        all_count - unlabeled_count,
        distribution[TceLabel.PC],
        distribution[TceLabel.FP],
        distribution[TceLabel.AFP],
        distribution[TceLabel.NTP],
    ]
    fig = go.Figure(go.Sunburst(labels=labels, parents=parents, values=count, branchvalues="total"))
    return fig
