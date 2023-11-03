from collections.abc import Callable, Collection, Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
import scipy.stats
from scipy.interpolate import LSQUnivariateSpline

from gaia.data.models import (
    TCE,
    AnySeries,
    IterableOfSeries,
    ListOfSeries,
    PeriodicEvent,
    Series,
)
from gaia.stats import BooleanArray, bic, diffs, robust_mean


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
    """Create a phase-folded time vector ommiting any NaN values.

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
    time: ListOfSeries,
    series: ListOfSeries,
    gap: float = 0.75,
) -> tuple[ListOfSeries, ListOfSeries]:
    """Split time series at gaps.

    If the time is not sorted in ascending order, it will be sorted internally.

    Args:
        time (IterableOfSeries): An iterable of 1D arrays of time values
        series (IterableOfSeries): An iterable of 1D arrays of time series features corresponding
        to the `time`
        gap (float, optional): Minimum time gap (in units of time) for split. Defaults to 0.75.

    Raises:
        ValueError: `gap` <= 0 OR `time` and `series` lengths are different
        InvalidDimensionError: Any of `time` or `series` values has dimension != 1

    Returns:
        tuple[ListOfSeries, ListOfSeries]: Splitted time and series arrays
    """
    if gap <= 0:
        raise ValueError(f"Expected 'gap' > 0, but got {gap=}")

    time_diff = [np.diff(t) for t in time]

    if any(np.any(d < 0) for d in time_diff):
        for i, (t, s) in enumerate(zip(time, series)):
            sort_indices = np.argsort(t)
            time[i] = t[sort_indices]
            series[i] = s[sort_indices]

    _check_series_dimension(time, "time")
    _check_series_dimension(series, "series")

    out_series: ListOfSeries = []
    out_time: ListOfSeries = []
    split_indicies = [np.argwhere(np.diff(t) > gap).flatten() + 1 for t in time]

    for time_segment, series_segment, split_index in zip(time, series, split_indicies):
        out_time.extend(np.array_split(time_segment, split_index))
        out_series.extend(np.array_split(series_segment, split_index))

    return out_time, out_series


class EventWidthStrategy(Protocol):
    def __call__(self, period: float, duration: float) -> float:
        ...


class AdjustedPadding:
    """Compute the event time duration as `min(3*duration, abs(secondary_phase), period)`."""

    def __init__(self, secondary_phase: float) -> None:
        self._secondary_phase = secondary_phase

    def __call__(self, period: float, duration: float) -> float:
        return min(3 * duration, abs(self._secondary_phase), period)


class SecondaryTransitAdjustedPadding:
    """Compute the event time duration to remove for secondary transit views.

    It uses a following rule:
    `if duration < abs(secondary_phase): 3*duration else: abs(secondary_phase) + duration`
    """

    def __init__(self, secondary_phase: float) -> None:
        self._secondary_phase = secondary_phase

    def __call__(self, period: float, duration: float) -> float:
        if duration < abs(self._secondary_phase):
            return 3 * duration

        return abs(self._secondary_phase) + duration


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


def interpolate_masked_spline(
    time: IterableOfSeries,
    masked_time: IterableOfSeries,
    masked_splines: IterableOfSeries,
) -> ListOfSeries:
    """Linearly interpolate spline values across masked (missing) points.

    If any of the `masked_time` segments are empty, they will be interpolated with `np.nan` values.

    Args:
        time (IterableOfSeries): A sequence of 1D arrays of time values at which to evaluate the
        interpolated values
        masked_time (IterableOfSeries): A sequence of 1D arrays of time values with some values
        missing (masked)
        masked_splines (IterableOfSeries): Masked spline values corresponding to `masked_time`

    Raises:
        ValueError: Lengths of `time`, `masked_time` and `masked_splines` are different
        InvalidDimensionError: Any of `time`, `masked_time` or `masked_splines` values has
        dimension != 1

    Returns:
        ListOfSeries: The list of masked splines (as 1D numpy arrays) with missing points linearly
        interpolated.
    """
    _check_series_dimension(time, "time")
    _check_series_dimension(masked_time, "masked_time")
    _check_series_dimension(masked_splines, "masked_splines")

    interpolations: ListOfSeries = []
    segments = zip(time, masked_time, masked_splines, strict=True)

    for time_segment, masked_time_segment, spline_segment in segments:
        if masked_time_segment.any():
            interpolation_segment = np.interp(time_segment, masked_time_segment, spline_segment)
            interpolations.append(interpolation_segment)
        else:
            interpolations.append(np.array([np.nan] * len(time_segment)))

    return interpolations


def _validate_create_bins_inputs(
    *,
    x: Series,
    y: Series,
    num_bins: int,
    bin_width: float,
    x_min: float,
    x_max: float,
) -> None:
    """Validate inputs of `bin_aggregate` function. Raise `ValueError` or `InvalidDimensionError`"""
    if any(np.diff(x) < 0):
        raise ValueError("'x' values must be sorted in ascending order")

    x_len = len(x)
    y_len = len(y)

    if x_len != y_len:
        raise ValueError(f"Expected 'len(x)' and 'len(y)' be the same, but got {x_len=}, {y_len=}")

    if num_bins < 2:
        raise ValueError(f"Expected 'num_bins' be at least 2, but got {num_bins}")

    if x_len < 2:
        raise ValueError(f"Expected 'len(x)' be at least 2, but got {x_len}")

    # Even if the x_max is passed as None it defaults to x.max() so this also checks if
    # x_min < x.max()
    if x_min >= x_max:
        raise ValueError(f"Expected 'x_min' be less than 'x_max', but got {x_min=}, {x_max=}")

    max_bin_width = x_max - x_min
    if not 0 < bin_width < max_bin_width:
        raise ValueError(
            f"Expected 'bin_width' be in the range (0, {max_bin_width}), but got {bin_width=}",
        )


class BinFunction(Protocol):
    def __call__(
        self,
        x: Series,
        y: Series,
        *,
        num_bins: int,
        bin_width: float | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
    ) -> tuple[Series]:
        ...


def create_bins(
    x: Series,
    y: Series,
    *,
    num_bins: int,
    bin_width: float | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
) -> tuple[Series]:
    """Assign the y-values to the uniform intervals (bins) created along the x-axis.

    The interval `[x_min, x_max)` is divided into `num_bins` uniformly spaced intervals of width
    `bin_width` and y-values corresponding with each interval are assigned to those bins. Any empty
    bins will be returned as an empty 1D numpy array.

    Args:
        x (Series): 1D numpy array of x-values sorted in ascending order. Must have at least 2
            elements, and all elements cannot be the same value
        y (Series): 1D numpy array with the same length as `x`
        num_bins (int): The number of intervals to divide the x-axis into. Must be at least 2
        bin_width (float | None, optional): The width of each bin on the x-axis. Must be positive
            and less than x_max - x_min. If `None` passed it is computed as `(x_max - x_min) /
            num_bins` to cover all x-values. Defaults to None.
        x_min (float | None, optional): The inclusive leftmost value to consider on the x-axis.
            Must be less than or equal to the largest value of `x`. If `None` passed it is computed
            as `min(x)`. Defaults to None.
        x_max (float | None, optional): The exclusive rightmost value to consider on the x-axis.
            Must be greater than `x_min`. If `None` passed it is computed as `max(x)`. Defaults
            to None.

    Note:
        The bins dividing is unstable due to the precision of floating-point numbers and may lead to
        different results when using x_min, x_max e.g.:
        - `x_min=0.6`, `x_max=1.6`, (value range x=1.0), num_bins=5 -> count_bins=[2,2,2,2,1])
        - `x_min=0.5`, `x_max=1.5`, (value range x=1.0), num_bins=5 -> count_bins=[2,2,2,2,2])

    Raises:
        ValueError:
        - `x` is not sorted in ascending order,
        - length of `x` and `y` mismatches,
        - length of `x` is less than 2,
        - `num_bins` is less than 2,
        - `x_min` is greater than or equal to `x_max`,
        - `x_min` is greater than the largest value of `x`,
        - `bin_width` not satisfy a condition 0 < bin_width < x_max - x_min.

        InvalidDimensionError: `x` or `y` has dimension != 1

    Returns:
        tuple[Series]: A tuple of 1D numpy arrays containing the y-values of evenly spaced bins
    """
    if x.ndim != 1:
        raise InvalidDimensionError(required_dim=1, actual_dim=x.ndim, invalid_parameters_name="x")
    if y.ndim != 1:
        raise InvalidDimensionError(required_dim=1, actual_dim=y.ndim, invalid_parameters_name="y")

    if x_min is None:
        x_min = x.min()

    if x_max is None:
        x_max = x.max()

    if bin_width is None:
        # By default `bin width` will be large enough to cover all x-values with respect to
        # `num_bins` without any bins gap or overlapping.
        bin_width = abs(x_max - x_min) / num_bins

    _validate_create_bins_inputs(
        x=x,
        y=y,
        num_bins=num_bins,
        bin_width=bin_width,
        x_min=x_min,
        x_max=x_max,
    )

    # If `bin_width` is close to `abs(x_max - x_min)`, then `bin_spacing`
    # is very small, leading to slower operation.
    bin_spacing = (x_max - x_min - bin_width) / (num_bins - 1)
    left_bin_edges = np.arange(x_min, x_max, bin_spacing)
    right_bin_edges = np.arange(x_min + bin_width, x_max + 10e-06, bin_spacing)
    bin_edges = zip(left_bin_edges, right_bin_edges)

    # NOTE: This mask is unstable due to the precision of floating-point numbers and may lead to
    # different results when using x_min, x_max e.g.:
    #   x=0.6-1.6, (value range x=1.0), num_bins=5 -> count_bins=[2,2,2,2,1])
    #   x=0.5-1.5, (value range x=1.0), num_bins=5 -> count_bins=[2,2,2,2,2])
    bins_masks = [np.logical_and(x >= left, x < right) for left, right in bin_edges]
    bins_y = tuple([y[mask] if mask.any() else np.array([]) for mask in bins_masks])
    return bins_y  # type: ignore


Boundaries: TypeAlias = tuple[float, float]


class TimeBoundariesFunction(Protocol):
    def __call__(self, period: float, duration: float) -> Boundaries:
        ...


def compute_local_view_time_boundaries(
    period: float,
    duration: float,
    num_durations: float = 2.5,
) -> Boundaries:
    """Compute a time range for a local time series view as specified in the `ExoMiner` paper.

    Time min is computed as `max(-period / 2, -duration * num_durations)`.
    Time max is computed as `max(period / 2, duration * num_durations)`.

    Args:
        period (float): TCE period
        duration (float): TCE transit duration
        num_durations (float, optional): Number of event transit durations to consider in
            calculation. Defaults to 2.5

    Note:
        For details see the original paper: https://arxiv.org/pdf/2111.10009.pdf

    Raises:
        ValueError: `period`/`duration`/`num_durations` < 0

    Returns:
        Boundaries: Minimum and maximum time for a local view
    """
    if period < 0:
        raise ValueError(f"Expected 'period' >= 0, but got {period=}")
    if duration < 0:
        raise ValueError(f"Expected 'duration' >= 0, but got {duration=}")
    if num_durations < 0:
        raise ValueError(f"Expected 'num_durations' >= 0, but got {num_durations=}")
    time_min = max(-period / 2, -duration * num_durations)
    time_max = min(period / 2, duration * num_durations)
    return time_min, time_max


def compute_global_view_time_boundaries(period: float, duration: float) -> Boundaries:
    """Compute a time range for a global time series view as specified in the `ExoMiner` paper.

    Time min is computed as `-period / 2`. Time max is computed as `period / 2`.

    Args:
        period (float): TCE period
        duration (float): Unused. Only for `TimeBoundariesFunction` compatibility

    Note:
        For details see the original paper: https://arxiv.org/pdf/2111.10009.pdf

    Raises:
        ValueError: `period` < 0

    Returns:
        Boundaries: Minimum and maximum time for a global view
    """
    if period < 0:
        raise ValueError(f"Expected 'period' >= 0, but got {period=}")
    return -period / 2, period / 2


class BinWidthFunction(Protocol):
    def __call__(self, period: float, duration: float) -> float:
        ...


def compute_local_view_bin_width(
    period: float,
    duration: float,
    bin_width_factor: float = 0.16,
) -> float:
    """Compute a bin width for a local time series view as specified in the `ExoMiner` paper.

    Bin width is computed as `duration * bin_width_factor`.

    Args:
        period (float): Unused. Only for `BinWidthFunction` compatibility
        duration (float): TCE transit duration
        bin_width_factor (float, optional): Bin width factor. Larger factor results with larger
            bins. Defaults to 0.16.

    Note:
        For details see the original paper: https://arxiv.org/pdf/2111.10009.pdf

    Raises:
        ValueError: `duration` or `bin_width_factor` < 0

    Returns:
        float: Bin width for a local view
    """
    if duration < 0:
        raise ValueError(f"Expected 'duration' >= 0, but got {duration=}")
    if bin_width_factor < 0:
        raise ValueError(f"Expected 'bin_width_factor' >= 0, but got {bin_width_factor=}")
    return duration * bin_width_factor


def compute_global_view_bin_width(
    period: float,
    duration: float,
    num_bins: int = 301,
    bin_width_factor: float = 0.16,
) -> float:
    """Compute a bin width for a global time series view as specified in the `ExoMiner` paper.

    Bin width is computed as `max(period / num_bins, bin_width_factor * duration)`.

    Args:
        period (float): TCE period
        duration (float): TCE transit duration
        num_bins (int, optional): Number of bins to include in a view. Defaults to 301
        bin_width_factor (float, optional): Bin width factor. Larger factor results in larger
            bins. Defaults to 0.16.

    Note:
        For details see the original paper: https://arxiv.org/pdf/2111.10009.pdf

    Raises:
        ValueError: `period`/`duration`/`bin_width_factor` < 0 OR `num_bins` < 2

    Returns:
        float: Bin width for a global view
    """
    if period < 0:
        raise ValueError(f"Expected 'period' >= 0, but got {period=}")
    if duration < 0:
        raise ValueError(f"Expected 'duration' >= 0, but got {duration=}")
    if bin_width_factor < 0:
        raise ValueError(f"Expected 'bin_width_factor' >= 0, but got {bin_width_factor=}")
    if num_bins < 2:
        raise ValueError(f"Expected 'num_bins' >= 2, but got {num_bins=}")

    return max(period / num_bins, bin_width_factor * duration)


AggregateFunction: TypeAlias = Callable[[Series], float]


class ViewGenerator:
    """Time series view generator."""

    def __init__(
        self,
        folded_time: Series,
        series: Series,
        bin_func: BinFunction,
        aggregate_func: AggregateFunction,
        default: float | AggregateFunction = 0.0,
    ) -> None:
        self._folded_time = folded_time
        self._series = series
        self._bin_func = bin_func
        self._aggregate_func = aggregate_func
        self._empty_bin_handler = default if callable(default) else lambda _: default  # type: ignore # noqa

    def generate(self, num_bins: int, time_min_max: Boundaries, bin_width: float) -> Series:
        """Generate a time series view by binning and aggregating y-values.

        Args:
            num_bins (int): The number of bins to divide the time series into. Must be at least 2
            time_min_max (Boundaries): `[min_time, max_time)` to consider on the folded time axis
            bin_width (float): The width of each bin on the folded time axis. Must be greater than 0

        Raises:
            ValueError: `num_bins` < 2 OR `bin_width` <= 0

        Returns:
            Series: Time series view with a length equal to the `num_bins`, created by binning and
                aggregating the y-values.
        """
        if num_bins < 2:
            raise ValueError(f"Expected 'num_bins' be at least 2, but got {num_bins}")
        if bin_width <= 0:
            raise ValueError(f"Expected 'bin_width' to be a positive float, but got {bin_width}")

        time_min, time_max = time_min_max
        bins = self._bin_func(
            self._folded_time,
            self._series,
            num_bins=num_bins,
            bin_width=bin_width,
            x_min=time_min,
            x_max=time_max,
        )
        view = [
            self._aggregate_func(bin) if bin.any() else self._empty_bin_handler(self._series)
            for bin in bins
        ]
        return np.array(view)


@dataclass
class InsufficientPointsError(Exception):
    """Raised when insufficient points are available for spline fitting."""

    available_points: int
    num_min_points: int


class SplineError(Exception):
    """Raised when an error occurs in the underlying spline-fitting implementation."""


class Spline:
    def __init__(
        self,
        knots_spacing: Collection[float],
        k: int = 3,
        max_iter: int = 5,
        sigma_cut: float = 3.0,
        penalty_coeff: float = 1.0,
    ) -> None:
        self._knots_spacing = knots_spacing
        self._k = k
        self._max_iter = max_iter
        self._sigma_cut = sigma_cut
        self._penalty_coeff = penalty_coeff

    def fit(
        self,
        x: IterableOfSeries,
        y: IterableOfSeries,
    ) -> tuple[ListOfSeries, list[BooleanArray]]:
        """Calculate the best fit spline for each value segment, omitting the nan values.

        The spline is fitted using an iterative process of removing outliers that can cause the
        spline be "pulled" by points with extreme values. In each iteration, the spline is fitted,
        and if they are any points where the absolute deviation from the median (MAD) of the
        outliers is at least 3 * sigma (where sigma is an estimate of the standard deviation of the
        residuals) these points are removed, and the spline is re-fitted.

        Args:
            x (IterableOfSeries): X-axis values segments
            y (IterableOfSeries): Y-axis values segments

        Raises:
            ValueError: Any of following cases:
              - knots spacing is an empty numpy array,
              - max iteration < 1,
              - spline degree not in range of [1, 5],
              - sigma cut <= 0,
              - x or y are empty lists,
              - All x and/or y segments are empty arrays.

            SplineError: Segments cannot be fit

        Returns:
            tuple[ListOfSeries, list[BooleanArray]]: Best fitted b-splines for each time segment and
                masks which indicate which points were used to fit a spline
        """
        self._validate_fit_parameters(x, y)

        best_bic = np.inf
        best_spline: ListOfSeries = []
        sqrt_2 = 1.4142135623730951
        y_diffs = diffs(y, sqrt_2)
        sigma = scipy.stats.median_abs_deviation(y_diffs, scale="normal", nan_policy="omit")

        for spacing in self._knots_spacing:
            num_free_params = 0
            num_points = 0
            ssr = 0
            spline_mask: list[BooleanArray] = []
            spline: ListOfSeries = []
            is_invalid_knots = False

            for xi, yi in zip(x, y):
                # Add an empty array if there is no x-values and skip this segment.
                if not xi.any():  # pragma: no cover
                    spline.append(np.array([]))
                    continue

                current_knots = np.arange(np.nanmin(xi) + spacing, np.nanmax(xi), spacing)

                try:
                    spline_piece, mask = self._fit_segment(xi, yi, knots=current_knots)
                except InsufficientPointsError:
                    # After removing outliers there are less poinst than neccesery to fit a spline
                    # or the entire x segement is to small.
                    spline.append(np.repeat(np.nan, yi.size))
                    spline_mask.append(np.zeros_like(yi, dtype=bool))
                    continue
                except SplineError:
                    # Current knots spacing led to the internal spline error. Skip this spacing.
                    is_invalid_knots = True
                    continue

                spline.append(spline_piece)
                spline_mask.append(mask)

                # Number of free parameters = number of knots + degree of spline - 1
                num_free_params += len(current_knots) + self._k - 1

                # Accumulate the number of points used to fit a spline and the squared residuals.
                num_points += np.count_nonzero(mask)
                ssr += np.sum((yi[mask] - spline_piece[mask]) ** 2)

            if is_invalid_knots or not num_points:
                # self.log.warning("Skipping current knots spacing", spacing=spacing)
                continue

            current_bic = bic(
                k=num_free_params,
                n=num_points,
                sigma=sigma,
                ssr=ssr,
                penalty_coeff=self._penalty_coeff,
            )

            if current_bic < best_bic or not best_spline:
                best_bic = current_bic
                best_spline = spline

        if not best_spline:
            raise SplineError("Spline fitting failes for all time series segments")

        return best_spline, spline_mask

    def _validate_fit_parameters(self, x: IterableOfSeries, y: IterableOfSeries) -> None:
        """Raise `ValueError` is any of `self.fit()` parameters is invalid."""
        if len(self._knots_spacing) == 0:
            raise ValueError("Knots spacing cannot be an empty array")
        if self._max_iter < 1:
            raise ValueError(f"Expected 'max_iter' to be at least 1, but got {self._max_iter}")
        if not 1 <= self._k <= 5:
            raise ValueError(
                f"Degree of a spline 'k' must be in the range [1, 5], but got {self._k}",
            )
        if self._sigma_cut <= 0:
            raise ValueError(f"Expected 'sigma_cut' to be > 0, but got {self._sigma_cut}")
        if not x or not y:
            raise ValueError("No values provided to 'x' or 'y' parameter")

        is_all_x_empty = not any(xi.any() for xi in x)
        is_all_y_empty = not any(yi.any() for yi in y)
        if is_all_x_empty or is_all_y_empty:
            raise ValueError("All segments in 'x' or 'y' are empty")

    def _fit_segment(self, x: Series, y: Series, knots: Series) -> tuple[Series, BooleanArray]:
        """Fit 1D segment of values."""
        x_len = len(x)
        if x_len <= self._k:
            raise InsufficientPointsError(available_points=x_len, num_min_points=self._k + 1)

        # Values of the best fitting spline evaluated at the time segment
        spline = np.array([])
        mask = np.isfinite(y)  # Try to fit all finite points

        for _ in range(self._max_iter):
            if spline.any():
                # Choose finite points where the absolute deviation from the median residual is less
                # than outlier_cut*sigma, where sigma is a robust estimate of the standard deviation
                # of the residuals from the previous spline.
                residuals = y - spline
                _, _, new_mask = robust_mean(residuals, sigma_cut=self._sigma_cut)

                if np.array_equal(new_mask, mask):
                    break  # Spline converged

                mask = new_mask

            available_points = np.count_nonzero(mask)
            if available_points <= self._k:
                raise InsufficientPointsError(available_points, num_min_points=self._k + 1)

            try:
                spline = LSQUnivariateSpline(x[mask], y[mask], k=self._k, t=knots)(x)
            except ValueError:
                # Occasionally, knot spacing leads to the choice of incorrect knots.
                # Raise SplainError and then skip current knots spacing.
                raise SplineError("Specified knots led to the internal spline error")

        return spline, mask


def flatten_time_series(
    all_time: IterableOfSeries,
    all_values: IterableOfSeries,
    events: Iterable[PeriodicEvent],
    gap: float,
    spline: Spline,
    include_empty_segments: bool = True,
) -> Series:
    """Remove time series low-frequency variability.

    Args:
        all_time (IterableOfSeries): Time value segments
        all_values (IterableOfSeries): Segments of the observed values corresponding to `all_time`
        events_with_secondary (Iterable[PeriodicEvent]): TCE transit events
        gap (floating): Minimum width of the gap in time units at which the time series is divided
        into smaller parts
        spline (Spline): A spline object to use in time series fitting

    Returns:
        Series: Flattened time series with low-frequency variability removed.
    """
    time, raw_values = split_arrays(list(all_time), list(all_values), gap)
    masked_time = deepcopy(time)
    masked_values = deepcopy(raw_values)

    for event in events:
        masked_time, masked_values = remove_events(
            masked_time,
            [event],
            masked_values,
            AdjustedPadding(event.secondary_phase),  # type: ignore # TODO: Make this dynamic
            include_empty_segments=include_empty_segments,
        )

    linearly_interpolated_spline = interpolate_masked_spline(time, masked_time, masked_values)
    best_fitted_splines, _ = spline.fit(time, linearly_interpolated_spline)
    concatenated_raw_values = np.concatenate(raw_values)
    concatenated_spline = np.concatenate(best_fitted_splines)

    return concatenated_raw_values / concatenated_spline
