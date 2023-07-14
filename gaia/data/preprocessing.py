from typing import Iterable, Protocol, TypeAlias

import numpy as np

from gaia.data.models import TCE, AnySeries, PeriodicEvent, Series


def compute_euclidean_distance(series: Series) -> Series:
    """Compute euclidean distance between points in 2D array.

    Euclidean distance (norm 2) is as follow: `sqrt(x^2+y^2)`

    Args:
        series (Series): 2D array of float values

    Raises:
        ValueError: `series` is not 2D

    Returns:
        Series: 1D array of euclidean distance for each pair of points from `series`
    """
    ndim = series.ndim
    if ndim != 2:
        raise ValueError(f"Expected 'series' to be 2D, but got {ndim}D")

    return np.linalg.norm(series, axis=0)  # type: ignore


def normalize_median(series: Series) -> Series:
    """Normalize a 1D array of values by dividing it by its median (ignoring NaN values).

    Args:
        series (Series): 1D array of values

    Raises:
        ValueError: `series` is not 1D

    Returns:
        Series: Normalized values as: `series / median(series)`
    """
    ndim = series.ndim
    if ndim != 1:
        raise ValueError(f"Expected 'series' to be 1D, but got {ndim}D")

    return series / np.nanmedian(series)


def phase_fold_time(time: Series, *, epoch: float, period: float) -> Series:
    """Create a phase-folded time vector.

    Args:
        time (Series): 1D array of time values
        epoch (float): First transit occurrence. This value is mapped to zero
        period (float): Period to fold over

    Raises:
        ValueError: `period` <= 0 OR `time` is not 1D

    Returns:
        Series: 1D numpy array folded around a `period` with time values within
            `[-period / 2, period / 2]`
    """
    if period <= 0:
        raise ValueError(f"Expected 'period' > 0, but got {period=}")
    if time.ndim != 1:
        raise ValueError(f"Expected 'time' to be 1D, but got {time.ndim}D")

    half_period = period / 2
    folded_time = np.mod(time + (half_period - epoch), period)
    folded_time -= half_period
    return folded_time


def compute_transits(
    tces: Iterable[TCE],
    time: Series,
    default: str = "no detected",
) -> AnySeries:
    """Determine whether there is a TCE transit for the time series.

    Args:
        tces(Iterable [TCE]): Sequence of TCEs
        time (Series): 1D array of time values
        default (str, optional): Text for non-transit points. Defaults to "not detected"

    Returns:
        AnySeries: A mask indicating for which time values TCE transit occurs. For transit
        points the name or ID of TCE will be included in the transit mask.
    """
    ndim = time.ndim
    if ndim != 1:
        raise ValueError(f"Expected 'time' to be 1D, but got {ndim}D")

    transits_mask = [default] * len(time)

    for tce in tces:
        tce_name = tce.name or str(tce.id)
        folded_time = phase_fold_time(time, epoch=tce.event.epoch, period=tce.event.period)
        transits_mask = [
            tce_name if np.abs(current_time) <= tce.event.duration else transit_marker
            for current_time, transit_marker in zip(folded_time, transits_mask)
        ]

    return np.array(transits_mask)


MultiSegmentSeries: TypeAlias = list[Series]


def split_arrays(
    time: MultiSegmentSeries,
    series: MultiSegmentSeries,
    gap_with: float = 0.75,
) -> tuple[MultiSegmentSeries, MultiSegmentSeries]:
    """Split time series at gaps.

    Args:
        time (MultiSegmentSeries): A list of 1D arrays of time values
        series (MultiSegmentSeries): A list of 1D arrays of time series features corresponding
        to the `time`
        gap_with (float, optional): Minimum time gap (in units of time) for split. Defaults to 0.75.

    Raises:
        ValueError: `gap_width` <= 0 OR any of `time` or `series` values has dimension != 1

    Returns:
        tuple[MultiSegmentSeries_, MultiSegmentSeries_]: Splitted time and series arrays
    """
    if gap_with <= 0:
        raise ValueError(f"Expected 'gap_width' > 0, but got {gap_with=}")

    if any((t.ndim != 1 for t in time)) or any((s.ndim != 1 for s in series)):
        raise ValueError(
            "Expected all series in 'time' and 'series' be 1D, but at least one is not",
        )

    out_series: MultiSegmentSeries = []
    out_time: MultiSegmentSeries = []
    split_indicies = [np.argwhere(np.diff(t) > gap_with).flatten() + 1 for t in time]

    for time_segment, series_segment, split_index in zip(time, series, split_indicies):
        out_time.extend(np.array_split(time_segment, split_index))
        out_series.extend(np.array_split(series_segment, split_index))

    return out_time, out_series


class EventWidthStrategy(Protocol):
    def __call__(self, period: float, duration: float) -> float:
        ...


class AdjustedPadding:
    """Compute the event time duration as `min(3*duration, abs(weak_secondary_phase), period)`."""

    def __init__(self, secondary_phase: float) -> None:
        self._secondary_phase = secondary_phase

    def __call__(self, period: float, duration: float) -> float:
        return min(3 * duration, abs(self._secondary_phase), period)


def remove_events(
    time: MultiSegmentSeries,
    tce_events: Iterable[PeriodicEvent],
    series: MultiSegmentSeries,
    event_width_calc: EventWidthStrategy,
    include_empty_segments: bool = True,
) -> tuple[MultiSegmentSeries, MultiSegmentSeries]:
    """Remove transit events from a time series.

    Args:
        time (MultiSegmentSeries): A list of 1D arrays of time values
        series (MultiSegmentSeries): A list of 1D arrays of time series features corresponding
        to the `time`
        tce_events (Iterable[PeriodicEvent]): TCE transit events
        event_width_calc (EventWidthStrategy): Callable to compute a width of transits to remove
        include_empty_segments (bool, optional): Whether to include empty segments. Defaults to
        True.

    Raises:
        ValueError: No `tce_events` provided OR lengths of `time` and `series` are different OR any
        of `time` or `series` values has dimension != 1

    Returns:
        tuple[MultiSegmentSeries, MultiSegmentSeries]: `time` and `series` with events removed.
    """
    if not tce_events:
        raise ValueError("No tce events provided")
    if any((t.ndim != 1 for t in time)) or any((s.ndim != 1 for s in series)):
        raise ValueError(
            "Expected all segments in 'time' and 'series' be 1D, but at least one is not",
        )

    out_time: MultiSegmentSeries = []
    out_series: MultiSegmentSeries = []

    for time_segment, series_segment in zip(time, series, strict=True):
        transit_mask = np.ones_like(time_segment)

        for event in tce_events:
            folded_time = phase_fold_time(time_segment, epoch=event.epoch, period=event.period)
            transit_distance = np.abs(folded_time)
            event_width = event_width_calc(event.period, event.duration)
            transit_mask = np.logical_and(transit_mask, transit_distance > 0.5 * event_width)

        if include_empty_segments or transit_mask.any():
            out_time.append(time_segment[transit_mask])
            out_series.append(series_segment[transit_mask])

    return out_time, out_series
