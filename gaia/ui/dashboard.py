import gzip
from dataclasses import asdict
from enum import Enum
from typing import Any, Callable, Iterable, TypeAlias

from dash import ALL, Input, Output, State, callback, ctx, dcc, html
from dash.exceptions import PreventUpdate

from gaia.data.models import (
    TCE,
    KeplerStellarParameters,
    KeplerTCE,
    KeplerTimeSeries,
    StellarParameters,
    TceLabel,
)
from gaia.data.sources import StellarParametersSource, TceSource, TimeSeriesSource
from gaia.io import CsvTableReader, TimeSeriesPickleReader
from gaia.ui.components import ComponentIds, DropdownOption, TimeSeriesAIO
from gaia.ui.store import AllData, GlobalStore, RedisStore
from gaia.ui.utils import compute_period_edges, compute_transits, flatten_series, get_key_for_value
from gaia.visualisation.plotly import plot_tces_classes_distribution, plot_tces_histograms


def get_tce_source():
    reader = CsvTableReader(
        source="/home/krzysiek/projects/gaia/data/raw/tables/q1_q17_dr25_tce_merged.csv",
        mapping=dict(
            kepid="target_id",
            tce_plnt_num="tce_id",
            tce_cap_stat="opt_ghost_core_aperture_corr",
            tce_hap_stat="opt_ghost_halo_aperture_corr",
            boot_fap="bootstrap_false_alarm_proba",
            tce_rb_tcount0="rolling_band_fgt",
            tce_prad="radius",
            tcet_period="fitted_period",
            tce_depth="transit_depth",
            tce_time0bk="epoch",
            tce_duration="duration",
            tce_period="period",
            kepler_name="name",
            tce_label="label",
        ),
    )
    return TceSource[KeplerTCE](reader)


def get_series_source():
    reader = TimeSeriesPickleReader[dict[str, KeplerTimeSeries]](
        "/home/krzysiek/projects/gaia/",
        id_path_pattern="test-{id}.gz",
        decompression_fn=gzip.decompress,
    )
    return TimeSeriesSource[KeplerTimeSeries](reader)


def get_params_source():
    reader = CsvTableReader(
        source="/home/krzysiek/projects/gaia/data/raw/tables/q1_q17_dr25_stellar.csv",
        mapping=dict(
            kepid="target_id",
            teff="effective_temperature",
            radius="radius",
            mass="mass",
            dens="density",
            logg="surface_gravity",
            feh="metallicity",
        ),
    )
    return StellarParametersSource[KeplerStellarParameters](reader)


TCE_SOURCE = get_tce_source()
SP_SOURCE = get_params_source()
TS_SOURCE = get_series_source()


def render_insight(*, icon: str, icon_color: str, header: str, text: str, footer: str) -> html.Div:
    return html.Div(
        [
            html.Div(
                html.I(className=f"{icon} insight-icon"), className=f"insight-icon-bg {icon_color}"
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
        # Escape Enum <name, value> representation as it cannot be render via Dash (unsafe html)
        p_value = value.value if isinstance(value, Enum) else value
        h3 = f"{field}:"
        rows.append(
            html.Div([html.H3(h3), html.P(p_value, className="text-muted")], className="data-row")
        )
    return rows


def render_stellar_parameters_overview(stellar_params: StellarParameters) -> html.Div:
    params = {k: v for k, v in asdict(stellar_params).items() if not k.startswith("_")}
    params = dict(sorted(params.items(), key=lambda x: x[0]))

    return html.Div(
        [
            html.H2("Stellar Parameters"),
            html.Div(
                [
                    html.H3("target"),
                    html.Span(stellar_params.target_id, className="badge bg-primary"),
                ],
                className="stellar-parameters-overview",
            ),
            html.Div(render_rows(params), className="data-rows"),
        ],
        className="stellar-parameters",
    )


def render_stellar_parameters():
    @callback(
        Output(ComponentIds.STELLAR_PARAMETERS, "children"),
        Input(ComponentIds.GLOBAL_STORE, "data"),
        prevent_initial_call=True,
    )
    def update(store: GlobalStore):
        data_key = store["redis_data_key"]

        if not data_key:
            raise PreventUpdate

        try:
            data: AllData = RedisStore.load(data_key)
        except Exception:
            raise PreventUpdate

        params = data["stellar_parameters"]
        return render_stellar_parameters_overview(params)

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
    TceLabel.AFP: ("warning", "bg-warning-light"),
    TceLabel.NTP: ("danger", "bg-danger-light"),
    TceLabel.PC: ("success", "bg-success-light"),
    TceLabel.UNKNOWN: ("primary", "bg-primary-light"),
}


def render_tce(tce: TCE):
    tce_dict = {k: v for k, v in asdict(tce).items() if not k.startswith("_")}
    tce_dict = dict(sorted(tce_dict.items(), key=lambda x: x[0]))
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
                                        html.H3(tce.name or f"TCE {tce.tce_id}"),
                                        html.Small(
                                            f"{tce.target_id}/{tce.tce_id}", className="text-muted"
                                        ),
                                    ]
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


TARGET_ID_NORMALIZATION = {"kepler": lambda target_id: f"{target_id:09d}"}


@callback(
    Output(ComponentIds.GLOBAL_STORE, "data"),
    [
        Input(ComponentIds.TARGET_ID_INPUT, "value"),
        State(ComponentIds.GLOBAL_STORE, "data"),
    ],
    prevent_initial_call=True,
)
def download_data(id_: str, store: GlobalStore) -> str:
    id_ = id_.strip()
    if not id_:
        raise PreventUpdate

    if id_.isnumeric():
        target_id = int(id_)
        tces = TCE_SOURCE.get_all_for_target(target_id)
    else:
        tces = [TCE_SOURCE.get_by_name(id_)]
        target_id = tces[0].target_id

    stellar_params = SP_SOURCE.get(target_id)
    normalized_id = TARGET_ID_NORMALIZATION["kepler"](target_id)
    time_series = TS_SOURCE.get(normalized_id)

    # Preprocessing
    period_edges = compute_period_edges(time_series)
    time, flat_series = flatten_series(time_series)
    tce_transists = compute_transits(tces, time)

    data = AllData(
        period_edges=period_edges,
        series=flat_series,
        time=time,
        tce_transits=tce_transists,
        tces=tces,
        stellar_parameters=stellar_params,
    )
    hash_key = RedisStore.save(data)
    store["redis_data_key"] = hash_key
    return store


def render_tces():
    @callback(
        Output(ComponentIds.TCE_LIST, "children"),
        Input(ComponentIds.GLOBAL_STORE, "data"),
        prevent_initial_call=True,
    )
    def update(store: GlobalStore):
        data_key = store["redis_data_key"]

        if not data_key:
            raise PreventUpdate

        try:
            data: AllData = RedisStore.load(data_key)
        except Exception:
            raise PreventUpdate

        tces = data["tces"]
        return [render_tce(tce) for tce in tces]

    return html.Div(
        [
            html.H2("TCEs List"),
            html.Div(
                dcc.Loading(
                    render_placeholder(
                        "No TCEs available. Please search for target or TCE first",
                        placeholder_class="tce-placeholder",
                    )
                ),
                id=ComponentIds.TCE_LIST,
                className="tces",
            ),
        ],
        className="tces-container",
    )


def render_time_series_graphs_dropdown(
    available_graphs: Iterable[str | DropdownOption], id: str | dict[str, str]
) -> html.Div:
    return html.Div(
        html.Details(
            [
                html.Summary(
                    html.Div(html.I(className="fa-solid fa-plus"), className="add-time-series-btn")
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
        )
    )


def render_tce_icon(
    icon: str, color: str, background_color: str, icon_class: str, background_class: str
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
def update_avaialble_graphs_options(
    selected_graphs: list[str], store: GlobalStore
) -> list[DropdownOption]:
    # Return all availables graphs - selected graphs
    new_options = [
        DropdownOption(label=label, value=value)
        for label, value in store["available_graphs"].items()
        if value not in selected_graphs
    ]
    # Hide dropdown if no options are avaialable
    style = {"visibility": "visible" if new_options else "hidden"}
    return new_options, style


@callback(
    Output(ComponentIds.ADD_TIME_SERIES_GRAPH, "value"),
    [
        Input(TimeSeriesAIO.ids.close(ALL), "n_clicks"),
        State(ComponentIds.ADD_TIME_SERIES_GRAPH, "value"),
    ],
    prevent_initial_call=True,
)
def update_avaialble_graphs_value(
    _: int, currently_selected_graphs: list[str]
) -> list[DropdownOption]:
    if not any([trigger["value"] for trigger in ctx.triggered]):  # No update, graph just rendered
        raise PreventUpdate

    removed_graph_name = ctx.triggered_id["aio_id"]
    new_values = [value for value in currently_selected_graphs if value != removed_graph_name]
    return new_values


def create_graphs(
    store: GlobalStore, selected_graphs_ids: list[str], _: list[dict[Any, Any]]
) -> list[TimeSeriesAIO]:
    try:
        data: AllData = RedisStore.load(store["redis_data_key"])
    except Exception:
        raise PreventUpdate

    time_series = data["series"]
    new_graphs: list[TimeSeriesAIO] = []

    for graph_id in selected_graphs_ids:        
        # `graph_id` = "series" or "series1,series2,..."
        graph_name = get_key_for_value(graph_id, store["available_graphs"])

        graph_series = [time_series[series_name] for series_name in graph_id.split(",")]
        new_graphs.append(
            TimeSeriesAIO(
                *graph_series,
                time=data["time"],
                period_edges=data["period_edges"],
                tce_transit_highlights=data["tce_transits"],
                id_=graph_id,
                name=graph_name,
            )
        )

    return new_graphs


def add_remove_graphs(
    store: GlobalStore, selected_graphs_ids: list[str], rendered_graphs: list[dict[Any, Any]]
) -> list[TimeSeriesAIO]:
    rendered_graphs_ids = {graph["props"]["id"] for graph in rendered_graphs}
    graph_ids_to_add = set(selected_graphs_ids) - rendered_graphs_ids

    if graph_ids_to_add:
        try:
            data: AllData = RedisStore.load(store["redis_data_key"])
        except Exception:
            raise PreventUpdate

        time_series = data["series"]
        new_graphs: list[TimeSeriesAIO] = []

        for graph_id in graph_ids_to_add:
            series = [time_series[series_name] for series_name in graph_id.split(",")]
            graph_name = get_key_for_value(graph_id, store["available_graphs"])
            new_graphs.append(
                TimeSeriesAIO(
                    *series,
                    time=data["time"],
                    period_edges=data["period_edges"],
                    tce_transit_highlights=data["tce_transits"],
                    id_=graph_id,
                    name=graph_name,
                )
            )

        return rendered_graphs + new_graphs

    # Remove graph
    graph_ids_to_remove = rendered_graphs_ids - set(selected_graphs_ids)
    return [graph for graph in rendered_graphs if graph["props"]["id"] not in graph_ids_to_remove]


UpdateGraphsFunction: TypeAlias = Callable[
    [GlobalStore, list[str], list[dict[Any, Any]]], list[TimeSeriesAIO]
]


# Functions to create/modify graphs on add/remove/data download
UPDATE_GRAPHS_HANDLERS: dict[str, UpdateGraphsFunction] = {
    ComponentIds.GLOBAL_STORE: create_graphs,
    ComponentIds.ADD_TIME_SERIES_GRAPH: add_remove_graphs,
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
    store: GlobalStore, selected_graphs: list[str], rendered_graphs: list[dict[Any, Any]]
) -> list[TimeSeriesAIO]:
    update_handler = UPDATE_GRAPHS_HANDLERS[ctx.triggered_id]
    return update_handler(store, selected_graphs, rendered_graphs)


def render_tce_labels_distribution(labels_distribution: dict[str, int]) -> html.Div:
    fig = plot_tces_classes_distribution(labels_distribution)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=180,
        hoverlabel=dict(font_family="poppins, arial, sans-serif"),
    )
    return html.Div(dcc.Graph(figure=fig), className="insight")


def create_dashboard(
    tce_source: TceSource[TCE], available_time_series_graphs: dict[str, str], data_origin: str
) -> html.Div:
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
                text=tce_source.tce_count,
                footer="The number of Threshold-Crossing Events",
            ),
            render_insight(
                icon="fa-solid fa-sun",
                icon_color="bg-primary",
                header="Targets Count",
                text=len(tce_source.target_unique_ids),
                footer="The number of unique target stars or binary/multiple systems",
            ),
            render_tce_labels_distribution(TCE_SOURCE.labels_distribution),
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

    histogram_figures = plot_tces_histograms(TCE_SOURCE.events)
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
        ]
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
                ]
            ),
            html.Div([render_stellar_parameters(), render_tces()], className="right"),
            dcc.Store(
                id=ComponentIds.GLOBAL_STORE,
                storage_type="local",
                data=GlobalStore(
                    redis_data_key="",
                    available_graphs=available_time_series_graphs,
                    data_origin=data_origin,
                ),
            ),
        ],
        className="app-container",
    )
