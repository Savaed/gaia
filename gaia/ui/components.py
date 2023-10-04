from typing import Callable, Iterable, NotRequired, Sequence, TypeAlias, TypedDict

import numpy as np
from dash import MATCH, Input, Output, State, callback, dcc, html
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go

from gaia.data.models import AnySeries, IntSeries, Series
from gaia.log import logger
from gaia.plotly import plot_empty_scatter, plot_time_series
from gaia.ui.store import RedisStore, TimeSeriesAIOData


PreprocessingFunction: TypeAlias = Callable[[Series], Series]


class ComponentIds:
    TIME_SERIES_CHART = "time-series-graph"
    TIME_SERIES_PERIODS_DROPDOWN = "time-series-data-periods"
    TARGET_ID_INPUT = "target-id-input"
    TCE_LIST = "tce-list"
    HIGHLIGHT_TCES_BTN = "highlight-tces-on-time-series"
    TIME_SERIES_CHART_LIST = "time-series-graph-list"
    TIME_SERIES_GRAPH_OPTIONS = "time-series-graph-switches"
    TIME_SERIES_AIO_STORE = "internal-store-time-series-aio"
    ADD_TIME_SERIES_GRAPH = "add-or-remove-time-series-graph"
    GLOBAL_STORE = "global-store"
    STELLAR_PARAMETERS = "stellar-parameters-container"


class DropdownOption(TypedDict):
    label: str | list[Component]
    value: str


class _GraphData(TypedDict):
    periods: IntSeries
    time: Series
    series: Series
    tce_tranists: NotRequired[AnySeries]
    period_edges: NotRequired[Series]


ComponentId: TypeAlias = dict[str, str]


def switches(options: Iterable[DropdownOption], id: str | ComponentId) -> dcc.Checklist:
    switch_options = [
        DropdownOption(
            label=[html.Span(className="slider"), html.Small(option["label"])],
            value=option["value"],
        )
        for option in options
    ]
    return dcc.Checklist(
        options=switch_options,
        value=[],
        labelClassName="switch",
        className="time-series-graph-switches",
        id=id,
        labelStyle={"display": "flex"},
    )


def create_component_id(*, type_: str, index: str) -> ComponentId:
    return {"type": type_, "index": index}


class _SubcomponentId(TypedDict):
    component: str
    subcomponent: str
    aio_id: str


class TimeSeriesAIO(html.Div):
    """Dash All-in-One time series graph component.

    This encapsulates all the logic, component layout, and data needed to render a graph and handle
    its updates. Allows to select specific observation periods, highlight TCE transits and the edges
    of the period. It uses (Fake)Redis to store all the data for the visuals, storing the data key
    in a client-side memory store. All callbacks are stateless.

    See also: https://dash.plotly.com/all-in-one-components
    """

    _preprocessing: dict[str, PreprocessingFunction] = {}
    _HIGHLIGHT_TCES = "Highlight TCEs"
    _HIGHLIGHT_PERIOD_EDGES = "Highlight edges"

    class _Ids:
        close = lambda aio_id: _SubcomponentId(  # noqa
            component="TimeSeriesAIO",
            subcomponent="close",
            aio_id=aio_id,
        )

    ids = _Ids

    def __init__(
        self,
        series: Sequence[Series],
        time: Sequence[Series],
        periods_mask: IntSeries,
        tce_transits: AnySeries,
        graph_name: str,
        graph_id: str,
    ) -> None:
        log = logger.bind(graph_id=graph_id, graph_name=graph_name)
        log.info("Initializing time series graph component")

        periods = list(set(periods_mask))

        if preprocessing_function := self._preprocessing.get(graph_id):
            processed_segments = [preprocessing_function(segment) for segment in series]

            # `preprocessing_function` should  always return a 1D array (a flat version of `series`)
            processed_series = np.concatenate(processed_segments)
        else:
            axis = series[0].ndim - 1  # For 1D `series` concatenate axis=0, for 2D axis=1, etc.
            processed_series = np.concatenate(series, axis)

        data = TimeSeriesAIOData(
            graph_id=graph_id,
            graph_name=graph_name,
            series=processed_series,
            time=np.concatenate(time),
            periods_mask=periods_mask,
            tce_transits=tce_transits,
        )
        hash_key = RedisStore.save(data)

        graph = self._create_graph(data, periods, highlight_tces=False, show_edges=False)
        graph.update_layout(margin=dict(l=0, r=0, t=30, b=0))

        header = html.Div(
            [
                html.H2(graph_name),
                html.I(
                    className="fa-solid fa-xmark time-series-remove",
                    id=self.ids.close(graph_id),
                ),
            ],
            className="time-series-graph-header",
        )
        settings = html.Div(
            [
                html.Div(
                    switches(
                        [
                            DropdownOption(label=self._HIGHLIGHT_TCES, value=self._HIGHLIGHT_TCES),
                            DropdownOption(
                                label=self._HIGHLIGHT_PERIOD_EDGES,
                                value=self._HIGHLIGHT_PERIOD_EDGES,
                            ),
                        ],
                        id=create_component_id(
                            type_=ComponentIds.TIME_SERIES_GRAPH_OPTIONS,
                            index=graph_id,
                        ),
                    ),
                    className="time-series-graph-switches",
                ),
                html.Div(
                    [
                        dcc.Dropdown(
                            options=periods,
                            multi=True,
                            value=periods,
                            placeholder="Select observation periods",
                            id=create_component_id(
                                type_=ComponentIds.TIME_SERIES_PERIODS_DROPDOWN,
                                index=graph_id,
                            ),
                        ),
                    ],
                ),
            ],
            className="time-series-graph-settings",
        )
        graph_container = html.Div(
            dcc.Loading(
                dcc.Graph(
                    figure=graph,
                    id=create_component_id(type_=ComponentIds.TIME_SERIES_CHART, index=graph_id),
                ),
            ),
            className="time-series-graph",
        )
        store = dcc.Store(
            id=create_component_id(type_=ComponentIds.TIME_SERIES_AIO_STORE, index=graph_id),
            data=hash_key,
        )

        super().__init__(
            id=graph_id,
            children=html.Div([header, settings, graph_container, store]),
            className="time-series-graph-container",
        )
        log.info("Graph component initialized")

    @classmethod
    def add_preprocessing(cls, key: str, fn: PreprocessingFunction) -> None:
        cls._preprocessing[key] = fn

    @staticmethod
    @callback(
        Output(
            create_component_id(type_=ComponentIds.TIME_SERIES_CHART, index=MATCH),
            "figure",
        ),
        [
            Input(
                create_component_id(
                    type_=ComponentIds.TIME_SERIES_GRAPH_OPTIONS,
                    index=MATCH,
                ),
                "value",
            ),
            Input(
                create_component_id(
                    type_=ComponentIds.TIME_SERIES_PERIODS_DROPDOWN,
                    index=MATCH,
                ),
                "value",
            ),
            State(
                create_component_id(type_=ComponentIds.TIME_SERIES_AIO_STORE, index=MATCH),
                "data",
            ),
        ],
        prevent_initial_call=True,
    )
    def update_graph(
        graph_options: list[str],
        selected_periods: list[str],
        data_key: str,
    ) -> go.Figure:
        logger.info("Updating time series graph")

        if not all([selected_periods, data_key]):
            return plot_empty_scatter()

        highlight_tces = TimeSeriesAIO._HIGHLIGHT_TCES in graph_options
        show_edges = TimeSeriesAIO._HIGHLIGHT_PERIOD_EDGES in graph_options

        try:
            data: TimeSeriesAIOData = RedisStore.load(data_key)
        except KeyError:
            logger.bind(data_key=data_key).warning("Loading data from Redis failed")
            raise PreventUpdate

        return TimeSeriesAIO._create_graph(data, selected_periods, highlight_tces, show_edges)

    @staticmethod
    def _create_graph(
        data: TimeSeriesAIOData,
        selected_periods: list[str],
        highlight_tces: bool,
        show_edges: bool,
        periods_label: str = "period",
    ) -> go.Figure:
        graph_data = TimeSeriesAIO._prepare_time_series_to_plot(data, selected_periods)

        return plot_time_series(
            graph_data,
            x="time",
            y="series",
            color="tce_tranists" if highlight_tces else None,
            period_edges="period_edges" if show_edges else None,
            hover_data=["periods"],
            labels=dict(series=data["graph_name"], periods=periods_label),
        )

    @staticmethod
    def _prepare_time_series_to_plot(
        data: TimeSeriesAIOData,
        selected_periods: list[str],
    ) -> _GraphData:
        logger.bind(periods=selected_periods).debug("Preparing time series data to plot")
        selected_periods_mask = [period in selected_periods for period in data["periods_mask"]]
        period_edges_mask = np.argwhere(np.diff(data["periods_mask"]) != 0).flatten()

        return _GraphData(
            time=data["time"][selected_periods_mask],
            series=data["series"][selected_periods_mask],
            periods=data["periods_mask"][selected_periods_mask],
            tce_tranists=data["tce_transits"][selected_periods_mask],
            period_edges=data["time"][period_edges_mask],
        )
