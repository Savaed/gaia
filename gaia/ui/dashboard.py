from dataclasses import asdict
from enum import Enum
from typing import Any, Callable, Iterable, TypeAlias, TypeVar

import numpy as np
from dash import ALL, Input, Output, State, callback, ctx, dcc, html
from dash.exceptions import PreventUpdate

from gaia.data.models import TCE, Series, StellarParameters, TceLabel, flatten_dict
from gaia.data.preprocessing import compute_transits
from gaia.data.stores import (
    StellarParametersStore,
    TceStore,
    TimeSeriesStore,
)
from gaia.log import logger
from gaia.plotly import plot_tces_classes_distribution, plot_tces_histograms
from gaia.ui.components import ComponentIds, DropdownOption, TimeSeriesAIO
from gaia.ui.store import AllData, GlobalStore, RedisStore


TStore = TypeVar("TStore", TceStore, TimeSeriesStore, StellarParametersStore)  # type: ignore
STORES: dict[type, TStore] = {}  # type: ignore


def get_key_for_value(value: Any, dct: dict[Any, Any]) -> Any:
    return [k for k, v in dct.items() if value == v][0]


def render_insight(*, icon: str, icon_color: str, header: str, text: str, footer: str) -> html.Div:
    return html.Div(
        [
            html.Div(
                html.I(className=f"{icon} insight-icon"),
                className=f"insight-icon-bg {icon_color}",
            ),
            html.H3(header, className="insight-header"),
            html.H1(text, className="insight-text"),
            html.P(footer, className="text-muted insight-footer"),
        ],
        className="insight",
    )


def render_rows(data: dict[str, Any]) -> list[html.Div]:
    rows: list[html.Div] = []
    for field, value in data.items():
        # Escape Enum <name, value> representation as it cannot be render via Dash (unsafe html).

        if isinstance(value, Enum):
            field_value = value.value
        elif isinstance(value, (int, float)):
            field_value = round(value, 4)
        else:
            field_value = value

        rows.append(
            html.Div(
                [html.H3(f"{field}:"), html.P(field_value, className="text-muted")],
                className="data-row",
            ),
        )
    return rows


def render_stellar_parameters_overview(stellar_params: StellarParameters) -> html.Div:
    params = flatten_dict(asdict(stellar_params))
    return html.Div(
        [
            html.H2("Stellar Parameters"),
            html.Div(
                [
                    html.H3("target"),
                    html.Span(stellar_params.id, className="badge bg-primary"),
                ],
                className="stellar-parameters-overview",
            ),
            html.Div(render_rows(params), className="data-rows"),
        ],
        className="stellar-parameters",
    )


def render_stellar_parameters() -> html.Div:
    @callback(
        Output(ComponentIds.STELLAR_PARAMETERS, "children"),
        Input(ComponentIds.GLOBAL_STORE, "data"),
        prevent_initial_call=True,
    )
    def update(store: GlobalStore) -> html.Div:
        logger.info("Update stellar parameters layout")
        data_key = store["redis_data_key"]

        if not data_key:
            raise PreventUpdate

        try:
            data: AllData = RedisStore.load(data_key)
        except KeyError:
            logger.bind(data_key=data_key).warning("Loadin data from Redis failed")
            raise PreventUpdate

        params = data["stellar_parameters"]
        return render_stellar_parameters_overview(params)

    logger.info("Render initial stellar parameters layout")
    return html.Div(
        html.Div(
            [
                html.H2("Stellar Parameters"),
                render_placeholder(
                    "No stellar parameters available. Please search for target or TCE first",
                    placeholder_class="stellar-parameters-placeholder",
                ),
            ],
            className="stellar-parameters",
        ),
        id=ComponentIds.STELLAR_PARAMETERS,
        className="stellar-parameters-container",
    )


def render_placeholder(text: str, placeholder_class: str) -> html.Div:
    return html.Div(
        [
            html.I(className="fa-regular fa-image"),
            html.P(
                text,
                className="text-muted",
            ),
        ],
        className=placeholder_class,
    )


TCE_LABELS_COLORS = {
    TceLabel.PC: ("success", "bg-success-light"),
    TceLabel.AFP: ("warning", "bg-warning-light"),
    TceLabel.NTP: ("danger", "bg-danger-light"),
    TceLabel.FP: ("warning", "bg-warning-light"),
    TceLabel.UNKNOWN: ("primary", "bg-primary-light"),
}


def render_tce(tce: TCE) -> html.Div:
    tce_dict = flatten_dict(asdict(tce))
    icon_color, icon_background_color = TCE_LABELS_COLORS[tce.label]

    return html.Div(
        [
            html.Details(
                [
                    html.Summary(
                        html.Div(
                            [
                                render_tce_icon(
                                    icon="fa-solid fa-globe",
                                    color=icon_color,
                                    icon_class="tce-overview-icon",
                                    background_class="tce-overview-icon-bg",
                                    background_color=icon_background_color,
                                ),
                                html.Div(
                                    [
                                        html.H3(tce.name or f"TCE {tce.id}"),
                                        html.Small(
                                            f"{tce.target_id}/{tce.id}",
                                            className="text-muted",
                                        ),
                                    ],
                                ),
                                html.Span(tce.label.name, className=f"badge bg-{icon_color}"),
                            ],
                            className="tce-overview",
                        ),
                    ),
                    html.Div(
                        render_rows(tce_dict),
                        className="tce-details data-rows",
                    ),
                ],
                className="details5",
            ),
        ],
        className="tce",
    )


def get_data_stores() -> tuple[TceStore, StellarParametersStore, TimeSeriesStore]:  # type: ignore
    return STORES[TceStore], STORES[StellarParametersStore], STORES[TimeSeriesStore]


@callback(
    Output(ComponentIds.GLOBAL_STORE, "data"),
    [
        Input(ComponentIds.TARGET_ID_INPUT, "value"),
        State(ComponentIds.GLOBAL_STORE, "data"),
    ],
    prevent_initial_call=True,
)
def download_data(id_: str, store: GlobalStore) -> GlobalStore:
    log = logger.bind(id=id_)
    log.info("Downloading data")

    if not (id_ := id_.strip()):
        raise PreventUpdate

    try:
        tce_store, stellar_store, time_series_store = get_data_stores()
    except KeyError as ex:
        logger.error(f"Data store '{ex}' not set")
        raise PreventUpdate

    if id_.isnumeric():
        target_id = int(id_)
        tces = tce_store.get_all_for_target(target_id)
    else:
        tces = [tce_store.get_by_name(id_)]
        target_id = tces[0].target_id

    stellar_params = stellar_store.get(target_id)

    time_series = time_series_store.get(target_id)
    log.info("Data downloaded")

    tce_transists = compute_transits(
        tces,
        np.concatenate([series["time"] for series in time_series]),
    )
    log.info("TCE transits highlights computed")

    data = AllData(
        time_series=time_series,
        tce_transits=tce_transists,
        tces=tces,
        stellar_parameters=stellar_params,
    )
    hash_key = RedisStore.save(data)
    store["redis_data_key"] = hash_key
    log.bind(data_key=hash_key).info("Serialized data saved on Redis")
    return store


def render_tces() -> html.Div:
    @callback(
        Output(ComponentIds.TCE_LIST, "children"),
        Input(ComponentIds.GLOBAL_STORE, "data"),
        prevent_initial_call=True,
    )
    def update(store: GlobalStore) -> list[html.Div]:
        logger.info("Updating TCEs layout")
        data_key = store["redis_data_key"]

        if not data_key:
            raise PreventUpdate

        try:
            data: AllData = RedisStore.load(data_key)
        except KeyError:
            logger.bind(data_key=data_key).warning("Loading data from Redis failed")
            raise PreventUpdate

        tces = data["tces"]
        return [render_tce(tce) for tce in tces]

    logger.info("Render initial TCEs layout")
    return html.Div(
        [
            html.H2("TCEs List"),
            html.Div(
                dcc.Loading(
                    render_placeholder(
                        "No TCEs available. Please search for target or TCE first",
                        placeholder_class="tce-placeholder",
                    ),
                ),
                id=ComponentIds.TCE_LIST,
                className="tces",
            ),
        ],
        className="tces-container",
    )


def render_time_series_graphs_dropdown(
    available_graphs: Iterable[str | DropdownOption],
    id: str | dict[str, str],
) -> html.Div:
    return html.Div(
        html.Details(
            [
                html.Summary(
                    html.Div(html.I(className="fa-solid fa-plus"), className="add-time-series-btn"),
                ),
                dcc.Checklist(
                    options=available_graphs,
                    value=[],
                    className="add-time-series-options",
                    labelClassName="add-time-series-option",
                    id=id,
                ),
            ],
            id="add-time-series-details",
        ),
    )


def render_tce_icon(
    icon: str,
    color: str,
    background_color: str,
    icon_class: str,
    background_class: str,
) -> html.Div:
    return html.Div(
        html.I(className=f"{icon} {color} {icon_class}"),
        className=f"{background_class} {background_color}",
    )


@callback(
    [
        Output(ComponentIds.ADD_TIME_SERIES_GRAPH, "options"),
        Output(ComponentIds.ADD_TIME_SERIES_GRAPH, "style"),
    ],
    [
        Input(ComponentIds.ADD_TIME_SERIES_GRAPH, "value"),
        State(ComponentIds.GLOBAL_STORE, "data"),
    ],
    prevent_initial_call=True,
)
def update_avaialable_graphs_options(
    selected_graphs: list[str],
    store: GlobalStore,
) -> tuple[list[DropdownOption], dict[str, str]]:
    # Return all availables graphs - selected graphs
    new_options = [
        DropdownOption(label=label, value=value)
        for label, value in store["available_graphs"].items()
        if value not in selected_graphs
    ]

    # Hide dropdown if no options are avaialable
    dropdown_visibility = "visible" if new_options else "hidden"
    return new_options, {"visibility": dropdown_visibility}


@callback(
    Output(ComponentIds.ADD_TIME_SERIES_GRAPH, "value"),
    [
        Input(TimeSeriesAIO.ids.close(ALL), "n_clicks"),
        State(ComponentIds.ADD_TIME_SERIES_GRAPH, "value"),
    ],
    prevent_initial_call=True,
)
def update_avaialble_graphs_value(
    _: int,
    currently_selected_graphs: list[str],
) -> list[str]:
    if not any([trigger["value"] for trigger in ctx.triggered]):  # No update, graph just rendered
        raise PreventUpdate

    removed_graph_name = ctx.triggered_id["aio_id"]
    new_values = [value for value in currently_selected_graphs if value != removed_graph_name]
    return new_values


def create_graphs(
    store: GlobalStore,
    selected_graphs_ids: list[str],
    _: list[dict[Any, Any]],
) -> list[TimeSeriesAIO]:
    try:
        data: AllData = RedisStore.load(store["redis_data_key"])
    except Exception:
        raise PreventUpdate

    new_graphs: list[TimeSeriesAIO] = []

    for graph_id in selected_graphs_ids:
        # `graph_id` = "series" or "series1,series2,..."
        graph_data = create_graph_data(store["available_graphs"], data, graph_id)
        new_graphs.append(graph_data)

    return new_graphs


def create_graph_data(
    available_graphs: dict[str, str],
    data: AllData,
    graph_id: str,
) -> TimeSeriesAIO:
    graph_name = get_key_for_value(graph_id, available_graphs)
    series_names = graph_id.split(",")

    series: list[Series] = []
    time_series = data["time_series"]

    for segment in time_series:
        individual_series = tuple(segment[name] for name in series_names)  # type: ignore

        # Stack time series to 2D array if many series should be preporcessed before final plotting.
        if len(individual_series) > 1:
            segment = np.stack(individual_series, axis=0)
        else:
            segment = individual_series[0]

        series.append(segment)  # type: ignore

    periods_mask = np.concatenate(
        [np.repeat(segment["period"], segment["time"].size) for segment in time_series],
    )

    return TimeSeriesAIO(
        series,
        time=[segment["time"] for segment in time_series],
        periods_mask=periods_mask,
        tce_transits=data["tce_transits"],
        graph_id=graph_id,
        graph_name=graph_name,
    )


def add_remove_graphs(
    store: GlobalStore,
    selected_graphs_ids: list[str],
    rendered_graphs: list[dict[Any, Any]],
) -> list[TimeSeriesAIO | dict[Any, Any]]:
    rendered_graphs_ids = {graph["props"]["id"] for graph in rendered_graphs}
    graph_ids_to_add = set(selected_graphs_ids) - rendered_graphs_ids

    if graph_ids_to_add:
        try:
            data: AllData = RedisStore.load(store["redis_data_key"])
        except Exception:
            raise PreventUpdate

        new_graphs: list[TimeSeriesAIO] = []

        for graph_id in graph_ids_to_add:
            new_graph_data = create_graph_data(store["available_graphs"], data, graph_id)
            new_graphs.append(new_graph_data)

        return rendered_graphs + new_graphs  # type: ignore

    # Remove graph
    graph_ids_to_remove = rendered_graphs_ids - set(selected_graphs_ids)
    return [graph for graph in rendered_graphs if graph["props"]["id"] not in graph_ids_to_remove]


UpdateGraphsFunction: TypeAlias = Callable[
    [GlobalStore, list[str], list[dict[Any, Any]]],
    list[TimeSeriesAIO],
]


# Functions to create/modify graphs on add/remove/data download
UPDATE_GRAPHS_HANDLERS: dict[str, UpdateGraphsFunction] = {
    ComponentIds.GLOBAL_STORE: create_graphs,
    ComponentIds.ADD_TIME_SERIES_GRAPH: add_remove_graphs,  # type: ignore
}


@callback(
    Output(ComponentIds.TIME_SERIES_CHART_LIST, "children"),
    [
        Input(ComponentIds.GLOBAL_STORE, "data"),
        Input(ComponentIds.ADD_TIME_SERIES_GRAPH, "value"),
        State(ComponentIds.TIME_SERIES_CHART_LIST, "children"),
    ],
    prevent_initial_call=True,
)
def update_graphs_collection(
    store: GlobalStore,
    selected_graphs: list[str],
    rendered_graphs: list[dict[Any, Any]],
) -> list[TimeSeriesAIO]:
    update_handler = UPDATE_GRAPHS_HANDLERS[ctx.triggered_id]
    return update_handler(store, selected_graphs, rendered_graphs)


def render_tce_labels_distribution(labels_distribution: dict[TceLabel, int]) -> html.Div:
    fig = plot_tces_classes_distribution(labels_distribution)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=180,
        hoverlabel=dict(font_family="poppins, arial, sans-serif"),
    )
    return html.Div(dcc.Graph(figure=fig), className="insight")


def create_dashboard(available_time_series_graphs: dict[str, str]) -> html.Div:
    try:
        tce_store, _, _ = get_data_stores()
    except KeyError as ex:
        logger.error(f"Data store {ex} not set")
        raise PreventUpdate

    insights = html.Div(
        [
            render_insight(
                icon="fa-solid fa-database",
                icon_color="bg-warning",
                header="Data Source",
                text="Kepler",
                footer="The name of the mission the data comes from",
            ),
            render_insight(
                icon="fa-solid fa-globe",
                icon_color="bg-success",
                header="TCEs count",
                text=str(tce_store.tce_count),
                footer="The number of Threshold-Crossing Events",
            ),
            render_insight(
                icon="fa-solid fa-sun",
                icon_color="bg-primary",
                header="Targets Count",
                text=str(len(tce_store.unique_target_ids)),
                footer="The number of unique target stars or binary/multiple systems",
            ),
            render_tce_labels_distribution(tce_store.labels_distribution),
        ],
        className="insights",
    )
    time_series_graphs = html.Div(
        [
            html.Div(
                [
                    html.H2("Time Series"),
                    render_time_series_graphs_dropdown(
                        [
                            DropdownOption(label=label, value=value)
                            for label, value in available_time_series_graphs.items()
                        ],
                        id=ComponentIds.ADD_TIME_SERIES_GRAPH,
                    ),
                ],
                className="time-series-header",
            ),
            html.Div([], id=ComponentIds.TIME_SERIES_CHART_LIST, className="time-series-graphs"),
        ],
        className="time-series-container",
    )

    histogram_figures = plot_tces_histograms([event for _, _, event in tce_store.events])
    for fig in histogram_figures:
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="TCE count",
            font_family="poppins, arial, sans-serif",
        )

    tces_histograms = html.Div(
        [
            html.H2("TCE period and duration", className="tce-dist-header"),
            html.Div(
                [dcc.Graph(figure=fig, className="tce-dist-graph") for fig in histogram_figures],
                className="tce-dist-container",
            ),
        ],
    )

    return html.Div(
        [
            html.Main(
                [
                    html.H1("Dashboard"),
                    dcc.Input(
                        id=ComponentIds.TARGET_ID_INPUT,
                        type="text",
                        className="target-or-tce-search-bar",
                        placeholder="Search target or TCE",
                        value="",
                        debounce=True,
                    ),
                    insights,
                    tces_histograms,
                    time_series_graphs,
                ],
            ),
            html.Div([render_stellar_parameters(), render_tces()], className="right"),
            dcc.Store(
                id=ComponentIds.GLOBAL_STORE,
                storage_type="local",
                data=GlobalStore(redis_data_key="", available_graphs=available_time_series_graphs),
            ),
        ],
        className="app-container",
    )
