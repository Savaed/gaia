from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from gaia.data.models import TCE, PeriodicTimeSeries, TimeSeries
from gaia.data.preporcessing import phase_fold_time
from gaia.ui.store import PeriodicData


def get_key_for_value(value: Any, dct: dict[Any, Any]) -> Any:
    return [key for key, value_ in dct.items() if value == value_][0]


def compute_period_edges(time_series: PeriodicTimeSeries) -> PeriodicData:
    return {period: [np.nanmax(series["TIME"])] for period, series in time_series.items()}


def flatten_series(
    time_series: dict[str, TimeSeries],
) -> tuple[PeriodicData, dict[str, PeriodicData]]:
    flat_series: dict[str, PeriodicData] = defaultdict(dict)

    for period, series_segment in time_series.items():
        for field, series in series_segment.items():
            flat_series[field][period] = series
    return flat_series.pop("TIME"), flat_series


def compute_transits(tces: list[TCE], time: PeriodicData) -> dict[str, Iterable[str]]:
    transits: dict[str, Iterable[str]] = {}

    for period, time_segment in time.items():
        tce_map = [None] * len(time_segment)

        for tce in tces:
            tce_name = tce.name or tce.tce_id
            folded_time = phase_fold_time(tce.event.period, tce.event.epoch, time_segment)
            tce_map = np.where(np.abs(folded_time) <= tce.event.duration, tce_name, tce_map)

        transits[period] = list(map(str, tce_map))

    return transits
