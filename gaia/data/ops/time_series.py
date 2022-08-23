"""Preprocessing functions for time series."""

from typing import Optional, Protocol, Union

import numpy as np
import structlog

from gaia.data.models import TCE


logger = structlog.stdlib.get_logger()

_MultiDataSegments = tuple[list[np.ndarray], ...]
_Segments = Union[list[np.ndarray], np.ndarray]


def wrap_in_list(segments: _Segments) -> Union[list[np.ndarray], np.ndarray]:
    """Wrap numpy array in a list."""
    if isinstance(segments, np.ndarray) and segments.ndim == 1:
        return [segments]
    return segments


def split_arrays(time: _Segments, series: _Segments, gap_with: float = 0.75) -> _MultiDataSegments:
    """Split time series at gaps.

    Accepts either a single time segment and time series segment or a list of such segments.

    Parameters
    ----------
    time : _Segments
        Time of observations. Single numpy array or a list of arrays
    series : _Segments
        A sequence of time series features corresponding to the `time`. Single numpy array or a
        list of arrays
    gap_with : float, optional
        Minimum time gap (in units of time) for split, by default 0.75

    Returns
    -------
    _MultiDataSegments
        Each tuple element is the list of splitted data sequence. The first element is
        `time` followed by `series`.

    Raises
    ------
    ValueError
        `gap_width` is less or equal to zero
    """
    if gap_with <= 0:
        raise ValueError(f"'gap_width' must be grater than zero, but got {gap_with=}'")

    log = logger.bind()
    out_series = []
    out_time = []
    time = wrap_in_list(time)
    series = wrap_in_list(series)
    log.info("Split arrays x and y", num_arrays=len(time), gap_with=gap_with)
    split_indicies = [np.argwhere(np.diff(t) > gap_with).flatten() + 1 for t in time]

    for time_segment, series_segment, split_indx in zip(time, series, split_indicies):
        out_time.extend(np.array_split(time_segment, split_indx))
        out_series.extend(np.array_split(series_segment, split_indx))

    log.info("Arrays splited successfuly", num_splited_arrays=len(out_time))
    return out_time, out_series


def phase_fold_time(
    time: np.ndarray,
    *,
    epoch: float,
    period: float,
    series: Optional[np.ndarray] = None,
    sort: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Create a phase-folded vector.

    Parameters
    ----------
    time : np.ndarray
        1D numpy array of time values
    epoch : float
        First transit occurrence. This value is mapped to zero
    period : float
        Period to fold over
    series: np.ndarray
        A sequence of time series features corresponding to the `time`
    sort : bool, optional
        Whether to sort a folded values by `time` in ascending order, by default False

    Returns
    -------
    Union[np.ndarray, tuple[np.ndarray, np.ndarray]]
        1D numpy arrays with folded around a `period` with time values within
        `[-period / 2, period / 2]` if only `time` is passed.
        A tuple of two 1D numpy arrays with folded around a `period` with time values within
        `[-period / 2, period / 2]` and series values folded in the same way

    Raises
    ------
    ValueError
        `period` is less than or equal to zero
    """
    if period <= 0:
        raise ValueError(f"'period' must be grater than zero, but got {period=}")

    half_period = period / 2
    folded_time = np.mod(time + (half_period - epoch), period)
    folded_time -= half_period
    order = np.argsort(folded_time) if sort else np.arange(time.size)
    return folded_time[order] if series is None else (folded_time[order], series[order])


class EventRemovingWidthStrategy(Protocol):
    """Interface to compute time width for removing events."""

    def __call__(self, duration: float, period: float, secondary_phase: float) -> float:
        ...


class AdjustedPadding:
    """
    Compute the time width for removing events as
    `min(3 * duration, abs(weak secondary phase), period)`.
    """

    def __call__(self, duration: float, period: float, secondary_phase: float) -> float:
        return min(3 * duration, abs(secondary_phase), period)


def remove_events(
    time: _Segments,
    tces: list[TCE],
    series: _Segments,
    compute_width_strategy: EventRemovingWidthStrategy,
    include_empty_segments: bool = True,
) -> _MultiDataSegments:
    """Remove events from a time series.

    Accepts either a single time segment and time series segment or a list of such segments.

    Parameters
    ----------
    time : _Segments
        A sequence of time values of observations. Single numpy array or a list of arrays
    events : list[PeriodicEvent]
        A list of TCEs. Each of them must contain PeriodicEvent object and weak secondary phase
        attribute
    series: _Segments
        A sequence of time series features corresponding to the `time`. Single numpy array or
        a list of arrays
    compute_width_strategy: EventRemovingWidthStrategy
        Implemenattaion of EventRemovingWidthStrategy interface. Specifies how to compute time
        width for events removing
    include_empty_segments : bool, optional
        Whether to include empty segments in the output, by default True

    Returns
    -------
    _MultiDataSegments
        Each tuple element is the list of numpy arrays with events removed.
        The first element is `time` followed by `series`.
    """
    log = logger.bind()
    log.info("Removing events from time series", width_strategy=compute_width_strategy)
    if not tces:
        log.warning("No events provided. Return unchanged 'time', 'series'")
        return time, series

    time = wrap_in_list(time)
    series = wrap_in_list(series)
    out_time = []
    out_series = []

    for time_segment, series_segment in zip(time, series):
        mask = np.ones_like(time_segment)

        for tce in tces:
            event = tce.event
            folded_time = phase_fold_time(time_segment, epoch=event.epoch, period=event.period)
            transit_dist = np.abs(folded_time)
            rm_width = compute_width_strategy(event.duration, event.period, tce.secondary_max_phase)
            mask = np.logical_and(mask, transit_dist > 0.5 * rm_width)

        if include_empty_segments or mask.any():
            out_time.append(time_segment[mask])
            out_series.append(series_segment[mask])

    log.info(f"{len(tces)} events {'masked' if include_empty_segments else 'removed'}")
    return out_time, out_series


def interpolate_masked_spline(
    time: _Segments, masked_time: _Segments, masked_splines: _Segments
) -> list[np.ndarray]:
    """Linearly interpolate spline values across masked points.

    Accepts either a single time segment and time series segment or a list of such segments.

    Parameters
    ----------
    time : _Segments
        A sequence of time values of observations. Single numpy array or a list of arrays
    masked_time : _Segments
        A sequence of time values of observations with some values missing (masked). Single numpy
        array or a list of arrays
    masked_splines : tuple[_Segments]
        Masked spline values corresponding to `masked_time`. Each tuple element is a single
        numpy array or a list of arrays

    Returns
    -------
    list[np.ndarray]
        The list of masked splines with missing points linearly interpolated.
    """
    log = logger.bind()
    log.info("Linearly interpolate masked spline")
    time = wrap_in_list(time)
    masked_time = wrap_in_list(masked_time)
    interpolations = []
    segments = zip(time, masked_time, masked_splines)

    for time_segment, masked_time_segment, spline_segment in segments:
        if masked_time_segment.size:
            interpolation_segment = np.interp(time_segment, masked_time_segment, spline_segment)
            interpolations.append(interpolation_segment)
        else:
            log.warning("Empty masked time segment. Interpolate with NaNs")
            interpolations.append(np.array([np.nan] * len(time_segment)))

    return interpolations
