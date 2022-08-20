"""Utilities for generating phase and transit views for time series values."""

from enum import Enum, auto
from typing import Any, Callable, Optional, Union

import numpy as np


ScalarOrArray = Union[np.ndarray, float, int]


class TimeSeriesViewType(Enum):
    """The type of time series view."""

    LOCAL = auto()
    """Transit view which provides more duration-specific time series insight."""

    GLOBAL = auto()
    """Phase view which provides more period-specific time series insight."""


def norm_median_min(values: ScalarOrArray) -> ScalarOrArray:
    """Normalize values using min-median normalization.

    Normalization is as follow: `(x - median(x)) / abs(min(x))`

    Parameters
    ----------
    values : ScalarOrArray
        Values to normalize

    Returns
    -------
    ScalarOrArray
        Normalized values
    """
    return (values - np.median(values)) / np.abs(np.min(values))


def norm_median_std(values: ScalarOrArray, stddev: Optional[ScalarOrArray] = None) -> ScalarOrArray:
    """Normalize values using median-standard deviation normalization.

    Normalization is as follow: `(x - median(x)) / std`

    Parameters
    ----------
    values : ScalarOrArray
        Values to normalize
    std : Optional[ScalarOrArray], optional
        Standard deviation to use in normalization. If None, the standard deviation will be computed
        from `values` along the 0 axis, by default None

    Returns
    -------
    ScalarOrArray
        Normalized values
    """
    if stddev is None:
        stddev = np.std(values, axis=0)

    return (values - np.median(values)) / stddev


class ViewGenerator:
    """Provides methods for generating time series views of the event."""

    def __init__(
        self,
        time: np.ndarray,
        series: np.ndarray,
        period: float,
        duration: float,
        bin_aggr_fn: Callable,
        empty_bin_handler: Callable[[ScalarOrArray], ScalarOrArray],
    ) -> None:
        """Initialize a ViewGenerator object.

        Parameters
        ----------
        time : np.ndarray
            Phase folded time of observations sorted in the asecnding order
        series : np.ndarray
            A sequence of phase folded time series features corresponding to the `time`
        period : float
            Period of the event
        duration : float
            Duration of event transit
        bin_aggr_fn : Callable
            Callable to discretise and aggregate values into bins
        empty_bin_handler : Callable[[ScalarOrArray], ScalarOrArray]
            Callable to handle empty bins. e.g. `np.median`

        Notes
        -----
        Parameter `x` must be sorted in ascending order.
        """
        self.time = time
        self.series = series
        self.period = period
        self.duration = duration
        self.bin_aggr_fn = bin_aggr_fn
        self.empty_bin_handler = empty_bin_handler

        self._view_params = {
            TimeSeriesViewType.GLOBAL: {
                "bin_width": lambda num_bins, bin_width_factor: max(
                    self.period / num_bins, bin_width_factor * self.duration
                ),
                "t_min": lambda: -self.period / 2,
                "t_max": lambda: self.period / 2,
            },
            TimeSeriesViewType.LOCAL: {
                "bin_width": lambda bin_width_factor: self.duration * bin_width_factor,
                "t_min": lambda num_durations: max(-period / 2, -duration * num_durations),
                "t_max": lambda num_durations: max(period / 2, -duration * num_durations),
            },
        }

    def _generate_view(
        self,
        num_bins: int,
        bin_width: float,
        t_min: float,
        t_max: float,
        norm_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    ) -> np.ndarray:
        """Generate view for the event."""
        view, bin_counts = self.bin_aggr_fn(
            self.time, self.series, num_bins=num_bins, bin_width=bin_width, x_min=t_min, x_max=t_max
        )
        view = np.where(bin_counts > 0, view, self.empty_bin_handler(self.series))
        return norm_fn(view) if norm_fn else view

    def generate_view(
        self,
        kind: TimeSeriesViewType,
        *,
        num_bins: int,
        bin_width_factor: float = 0.16,
        norm_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate a view of phase-folded time series for the specific event.

        Parameters
        ----------
        kind : TimeSeriesViewType
            Type of a view. For example transit (local) or phase (global) view
        num_bins : int
            Number of bins to use in aggregation
        bin_width_factor : float, optional
            Fractional transit duration used to compute event removing width, by default 0.16
        norm_fn : Optional[Callable[[ScalarOrArray], ScalarOrArray]], optional
            Normalization function. If None, normalization is not performed, by default None

        See also
        --------
        bin_aggregate, phase_fold_time

        Returns
        -------
        np.ndarray
            A view of the time series created for the specific event
        """
        num_durations = kwargs.pop("num_durations", 2.5)  # Currently used only for a local view
        view_params = self._get_view_params(kind, num_bins, bin_width_factor, num_durations)
        return self._generate_view(num_bins=num_bins, norm_fn=norm_fn, **view_params)

    def _get_view_params(
        self, kind: TimeSeriesViewType, num_bins: int, bin_width_factor: float, num_durations: int
    ) -> dict[str, Any]:
        """Get view-specific parameters."""
        # TODO: Probably to refactor in some free time.
        # Those params are hard-coded and cannot be extended in easy way.

        bin_width = (
            max(self.period / num_bins, bin_width_factor * self.duration)
            if kind is TimeSeriesViewType.GLOBAL
            else self.duration * bin_width_factor
        )
        time_max = (
            self.period / 2
            if kind is TimeSeriesViewType.GLOBAL
            else min(self.period / 2, self.duration * num_durations)
        )
        time_min = (
            -self.period / 2
            if kind is TimeSeriesViewType.GLOBAL
            else max(-self.period / 2, -self.duration * num_durations)
        )
        return {"bin_width": bin_width, "t_min": time_min, "t_max": time_max}
