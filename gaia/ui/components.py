# type: ignore


from typing import Callable, Iterable, TypeAlias, TypedDict

import numpy as np
from dash import MATCH, Input, Output, State, callback, dcc, html
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go

from gaia.data.models import Series
from gaia.ui.store import PeriodicData, RedisStore, TimeSeriesAIOData
from gaia.visualisation.plotly import plot_empty_scatter, plot_time_series


PreprocessingFunction: TypeAlias = Callable[[tuple[PeriodicData, ...]], PeriodicData]


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
    periods: Series
    time: Series
    series: Series
    tce_tranist_highlights: Iterable[str]  # TODO: NotRequired[Iterable[str]] in python 3.11
    period_edges: Iterable[float]  # TODO: NotRequired[Iterable[float]] in python 3.11


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
    )


def create_pattern_matching_id(*, type_: str, index: str) -> ComponentId:
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
        *series: PeriodicData,
        time: PeriodicData,
        period_edges: PeriodicData,
        tce_transit_highlights: PeriodicData,
        name: str,
        id_: str,
    ) -> None:
        periods = list(period_edges)  # Get period names
        # Get period names for each time series value
        period_labels = {period: [period] * len(t) for period, t in time.items()}

        if preprocessing_function := self._preprocessing.get(id_):
            preprocessed_series = preprocessing_function(series)
        else:
            preprocessed_series = series[0]

        data = TimeSeriesAIOData(
            name=name,
            id_=id_,
            time=time,
            series=preprocessed_series,
            period_edges=period_edges,
            periods_labels=period_labels,
            tce_transits=tce_transit_highlights,
        )
        hash_key = RedisStore.save(data)

        graph = self._create_graph(data, periods, highlight_tces=False, show_edges=False)
        graph.update_layout(margin=dict(l=0, r=0, t=30, b=0))

        header = html.Div(
            [
                html.H2(name),
                html.I(
                    className="fa-solid fa-xmark time-series-remove",
                    id=self.ids.close(id_),
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
                        id=create_pattern_matching_id(
                            type_=ComponentIds.TIME_SERIES_GRAPH_OPTIONS,
                            index=id_,
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
                            id=create_pattern_matching_id(
                                type_=ComponentIds.TIME_SERIES_PERIODS_DROPDOWN,
                                index=id_,
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
                    id=create_pattern_matching_id(type_=ComponentIds.TIME_SERIES_CHART, index=id_),
                ),
            ),
            className="time-series-graph",
        )
        store = dcc.Store(
            id=create_pattern_matching_id(type_=ComponentIds.TIME_SERIES_AIO_STORE, index=id_),
            data=hash_key,
        )

        super().__init__(
            id=id_,
            children=html.Div([header, settings, graph_container, store]),
            className="time-series-graph-container",
        )

    @classmethod
    def add_preprocessing(cls, key: str, fn: PreprocessingFunction) -> None:
        cls._preprocessing[key] = fn

    @staticmethod
    @callback(
        Output(
            create_pattern_matching_id(type_=ComponentIds.TIME_SERIES_CHART, index=MATCH),
            "figure",
        ),
        [
            Input(
                create_pattern_matching_id(
                    type_=ComponentIds.TIME_SERIES_GRAPH_OPTIONS,
                    index=MATCH,
                ),
                "value",
            ),
            Input(
                create_pattern_matching_id(
                    type_=ComponentIds.TIME_SERIES_PERIODS_DROPDOWN,
                    index=MATCH,
                ),
                "value",
            ),
            State(
                create_pattern_matching_id(type_=ComponentIds.TIME_SERIES_AIO_STORE, index=MATCH),
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
        if not all([selected_periods, data_key]):
            return plot_empty_scatter()

        highlight_tces = TimeSeriesAIO._HIGHLIGHT_TCES in graph_options
        show_edges = TimeSeriesAIO._HIGHLIGHT_PERIOD_EDGES in graph_options

        try:
            data: TimeSeriesAIOData = RedisStore.load(data_key)
        except KeyError:
            raise PreventUpdate

        return TimeSeriesAIO._create_graph(data, selected_periods, highlight_tces, show_edges)

    @staticmethod
    def _create_graph(
        data: TimeSeriesAIOData,
        selected_periods: list[str],
        highlight_tces: bool,
        show_edges: bool,
    ) -> go.Figure:
        graph_data = TimeSeriesAIO._prepare_time_series_to_plot(data, selected_periods)
        return plot_time_series(
            graph_data,
            x="time",
            y="series",
            color="tce" if highlight_tces else None,
            period_edges="period_edges" if show_edges else None,
            hover_data=["periods"],
            labels=dict(series=data["name"], periods="quarter"),
        )

    @staticmethod
    def _prepare_time_series_to_plot(
        data: TimeSeriesAIOData,
        selected_periods: list[str],
    ) -> _GraphData:
        # 'TimeSeriesAIOData' to '_GraphData' keys mapping
        # TODO: This mapping is probably unnecessary, so refactor at some point
        mapping = zip(
            ("periods", "time", "series", "tce", "period_edges"),
            ("periods_labels", "time", "series", "tce_transits", "period_edges"),
            strict=True,
        )
        graph_data: _GraphData = {}
        for data_key, field in mapping:
            series = [
                values for period, values in data[field].items() if period in selected_periods
            ]
            graph_data[data_key] = np.concatenate(series)
        return graph_data
