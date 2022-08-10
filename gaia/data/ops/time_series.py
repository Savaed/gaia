"""Preprocessing functions for time series."""

from collections import defaultdict
from typing import Union

import numpy as np

from gaia.data.models import PeriodicEvent


_MultiDataSegments = tuple[list[np.ndarray], ...]
_Segments = Union[list[np.ndarray], np.ndarray]


def wrap_in_list(segments: _Segments) -> _Segments:
    """Wraps numpy array in a list."""
    if isinstance(segments, np.ndarray) and segments.ndim == 1:
        return [segments]
    return segments


def split_arrays(time: _Segments, *series: _Segments, gap_with: float = 0.75) -> _MultiDataSegments:
    """Split time series tables at gaps.

    Accepts either a single time segment and time series segment or a list of such segments.

    Parameters
    ----------
    time : _Segments
        Time of observations. Single numpy array or a list of arrays
    *series : _Segments
        Time series associated with the `time` parameter. Single numpy array or a list of arrays

    gap_with : float, optional
        Minimum time gap (in units of time) for split, by default 0.75

    Returns
    -------
    _MultiDataSegments
        The tuple of the list of numpy arrays. The tuple is of the size of `series` size.
        Each tuple element is the list of splitted data sequence. The first element is
        `all_time` followed by `series` elements.

    Raises
    ------
    ValueError
        `gap_width` is less or equal to zero
    """
    if gap_with <= 0:
        raise ValueError(f"'gap_width' must be grater than zero, but got {gap_with=}'")

    time = wrap_in_list(time)
    out_ts = defaultdict(lambda: [])
    out_time = []
    split_indicies = [np.argwhere(np.diff(t) > gap_with).flatten() + 1 for t in time]

    for time_segment, split_indx in zip(time, split_indicies):
        out_time.extend(np.array_split(time_segment, split_indx))

    for series_indx, time_series in enumerate(series):
        time_series = wrap_in_list(time_series)

        for split_indx, time_series_segment in zip(split_indicies, time_series):
            out_ts[series_indx].extend(np.array_split(time_series_segment, split_indx))

    return out_time, *out_ts.values()


def phase_fold_time(
    time: np.ndarray, epoch: float, period: float, sort: bool = False
) -> np.ndarray:
    """Create a phase-folded time vector.

    Parameters
    ----------
    time : np.ndarray
        1D numpy array of time values
    epoch : float
        First transit occurrence. This value is mapped to zero
    period : float
        Period to fold over
    sort : bool, optional
        Whether to sort a folded time in ascending order, by default False

    Returns
    -------
    np.ndarray
        1D numpy array with time folded around a `period` with all values within
        `[-period / 2, period / 2]`

    Raises
    ------
    ValueError
        `period` is less than or equal to zero
    """
    if period <= 0:
        raise ValueError(f"'period' must be grater than zero, but got {period=}")

    half_period = period / 2
    result = np.mod(time + (half_period - epoch), period)
    result -= half_period
    return np.sort(result) if sort else result


def remove_events(
    all_time: _Segments,
    events: list[PeriodicEvent],
    width_factor: float = 1.0,
    include_empty_segments: bool = True,
    **series: _Segments,
) -> _MultiDataSegments:
    """Remove events from a time series.

    Accepts either a single time segment and time series segment or a list of such segments.

    Parameters
    ----------
    time : _Segments
        A sequence of time values of observations. Single numpy array or a list of arrays
    events : list[PeriodicEvent]
        A list of events to remove
    width_factor : float, optional
        A fractional multiplier of the duration of each event to remove, by default 1.0
    include_empty_segments : bool, optional
        Whether to include empty segments in the output, by default True

    Returns
    -------
    _MultiDataSegments
        The tuple of the list of numpy arrays. The tuple is of the size of `series` size + 1.
        Each tuple element is the list of numpy arrays with events removed.
        The first element is `all_time` followed by `series` elements.
    """
    if not events:
        return all_time, *series.values()

    all_time = wrap_in_list(all_time)

    out_time = []
    out_series = defaultdict(lambda: [])

    for segment_num, time_segment in enumerate(all_time):
        mask = np.ones_like(time_segment)
        for event in events:
            transit_dist = np.abs(phase_fold_time(time_segment, event.epoch, event.period))
            mask = np.logical_and(mask, transit_dist > 0.5 * width_factor * event.duration)

        if include_empty_segments or mask.any():
            out_time.append(time_segment[mask])

            for series_ind, ts in enumerate(series.values()):
                ts = wrap_in_list(ts)
                series_masked_segment = ts[segment_num][mask]
                out_series[series_ind].append(series_masked_segment)

    return out_time, *out_series.values()


def interpolate_masked_spline(
    all_time: _Segments, all_masked_time: _Segments, *all_masked_splines: _Segments
) -> _MultiDataSegments:
    """Linearly interpolate spline values across masked points.

    Accepts either a single time segment and time series segment or a list of such segments.

    Parameters
    ----------
    all_time : _Segments
        A sequence of time values of observations. Single numpy array or a list of arrays
    all_masked_time : _Segments
        A sequence of time values of observations with some values missing (masked). Single numpy
        array or a list of arrays
    *all_masked_splines : tuple[_Segments]
        Masked spline values corresponding to `all_masked_time`. Each tuple element is a single
        numpy array or a list of arrays

    Returns
    -------
    _MultiDataSegments
        The tuple of the list of numpy arrays. The tuple is of the size of `all_masked_splines`
        size. Each tuple element is the list of masked splines with missing points linearly
        interpolated. The first element is `all_time` followed by `series` elements.
    """
    interp_splines = defaultdict(lambda: [])

    all_time = wrap_in_list(all_time)
    all_masked_time = wrap_in_list(all_masked_time)

    for spline_indx, all_masked_spline in enumerate(all_masked_splines):
        all_masked_spline = wrap_in_list(all_masked_spline)
        for time, masked_time, masked_spline in zip(all_time, all_masked_time, all_masked_spline):
            if masked_time.size:
                interp_splines[spline_indx].append(np.interp(time, masked_time, masked_spline))
            else:
                interp_splines[spline_indx].append(np.array([np.nan] * len(time)))

    out_splines = tuple(list(spline_values) for spline_values in interp_splines.values())
    return out_splines
