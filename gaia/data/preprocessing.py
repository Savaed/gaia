from dataclasses import dataclass
from typing import Iterable, Protocol, TypeAlias

import numpy as np

from gaia.data.models import TCE, AnySeries, PeriodicEvent, Series


@dataclass
class InvalidDimensionError(Exception):
    """Raised when the tensor (mostly vectors or matrices) has invalid dimension(s)."""

    required_dim: int
    actual_dim: int
    invalid_parameters_name: str | None = None

    def __str__(self) -> str:
        return f"Expected {self.invalid_parameters_name or 'data'} to be {self.required_dim}D, but got {self.actual_dim}D"  # noqa


def compute_euclidean_distance(series: Series) -> Series:
    """Compute euclidean distance between points in 2D array.

    Euclidean distance (norm 2) is as follow: `sqrt(x^2+y^2)`

    Args:
        series (Series): 2D array of float values

    Raises:
        InvalidDimensionError: `series` is not 2D

    Returns:
        Series: 1D array of euclidean distance for each pair of points from `series`
    """
    ndim = series.ndim
    if ndim != 2:
        raise InvalidDimensionError(required_dim=2, actual_dim=ndim)

    return np.linalg.norm(series, axis=0)  # type: ignore


def normalize_median(series: Series) -> Series:
    """Normalize a 1D array of values by dividing it by its median (ignoring NaN values).

    Args:
        series (Series): 1D array of values

    Raises:
        InvalidDimensionError: `series` is not 1D

    Returns:
        Series: Normalized values as: `series / median(series)`
    """
    ndim = series.ndim
    if ndim != 1:
        raise InvalidDimensionError(required_dim=1, actual_dim=ndim)

    return series / np.nanmedian(series)


def phase_fold_time(time: Series, *, epoch: float, period: float) -> Series:
    """Create a phase-folded time vector.

    Args:
        time (Series): 1D array of time values
        epoch (float): First transit occurrence. This value is mapped to zero
        period (float): Period to fold over

    Raises:
        ValueError: `period` <= 0
        InvalidDimensionError: `time` is not 1D

    Returns:
        Series: 1D numpy array folded around a `period` with time values within
        `[-period / 2, period / 2]`
    """
    if period <= 0:
        raise ValueError(f"Expected 'period' > 0, but got {period=}")
    if time.ndim != 1:
        raise InvalidDimensionError(required_dim=1, actual_dim=time.ndim)

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

    Raises:
        InvalidDimensionError: `time` is not 1D

    Returns:
        AnySeries: A mask indicating for which time values TCE transit occurs. For transit
        points the name or ID of TCE will be included in the transit mask.
    """
    ndim = time.ndim
    if ndim != 1:
        raise InvalidDimensionError(required_dim=1, actual_dim=ndim, invalid_parameters_name="time")

    transits_mask = [default] * len(time)

    for tce in tces:
        tce_name = tce.name or str(tce.id)
        folded_time = phase_fold_time(time, epoch=tce.event.epoch, period=tce.event.period)
        transits_mask = [
            tce_name if np.abs(current_time) <= tce.event.duration else transit_marker
            for current_time, transit_marker in zip(folded_time, transits_mask)
        ]

    return np.array(transits_mask)


IterableOfSeries: TypeAlias = Iterable[Series]
ListOfSeries: TypeAlias = list[Series]


def _check_series_dimension(
    series: IterableOfSeries,
    series_name: str | None = None,
    required_dim: int = 1,
) -> None:
    for i, segment in enumerate(series):
        if segment.ndim != required_dim:
            name = f"{series_name or 'data'}[{i}]"
            raise InvalidDimensionError(
                required_dim=required_dim,
                actual_dim=segment.ndim,
                invalid_parameters_name=name,
            )


def split_arrays(
    time: IterableOfSeries,
    series: IterableOfSeries,
    gap_with: float = 0.75,
) -> tuple[ListOfSeries, ListOfSeries]:
    """Split time series at gaps.

    Args:
        time (IterableOfSeries): An iterable of 1D arrays of time values
        series (IterableOfSeries): An iterable of 1D arrays of time series features corresponding
        to the `time`
        gap_with (float, optional): Minimum time gap (in units of time) for split. Defaults to 0.75.

    Raises:
        ValueError: `gap_width` <= 0 OR `time` and `series` lengths are different
        InvalidDimensionError: Any of `time` or `series` values has dimension != 1

        # TODO: test do tego ^^^

    Returns:
        tuple[ListOfSeries, ListOfSeries]: Splitted time and series arrays
    """
    if gap_with <= 0:
        raise ValueError(f"Expected 'gap_width' > 0, but got {gap_with=}")
    _check_series_dimension(time, "time")
    _check_series_dimension(series, "series")

    out_series: ListOfSeries = []
    out_time: ListOfSeries = []
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
    time: IterableOfSeries,
    tce_events: Iterable[PeriodicEvent],
    series: IterableOfSeries,
    event_width_calc: EventWidthStrategy,
    include_empty_segments: bool = True,
) -> tuple[ListOfSeries, ListOfSeries]:
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
    _check_series_dimension(time, "time")
    _check_series_dimension(series, "series")

    out_time: ListOfSeries = []
    out_series: ListOfSeries = []

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
