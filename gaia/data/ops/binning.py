"""Preprocessing functions for the discretization and aggregation of time series values."""

from typing import Callable, Optional

import numpy as np


BinAggregateResult = tuple[np.ndarray, list[int]]
AggregateFunction = Callable[[np.ndarray], np.ndarray]


def _validate_bin_aggregate_inputs(
    x: np.ndarray,
    y: np.ndarray,
    num_bins: int,
    bin_width: float,
    x_min: float,
    x_max: float
) -> None:
    """
    Validate inputs of `an aggregate` function. Raise ValueError with a specific message when any
    one of the inputs is invalid.
    """
    if any(np.diff(x) < 0):
        raise ValueError("'x' must be sorted in ascending order")

    x_len = len(x)
    y_len = len(y)

    if x_len != y_len:
        raise ValueError(f"'len(x)' and 'len(y)' must be the same, but got {x_len=}, {y_len=}")

    if num_bins < 2:
        raise ValueError(f"'num_bins' must be at least 2, but got {num_bins=}")

    if x_len < 2:
        raise ValueError(f"The lenght of 'x' must be at least 2, but got {x_len=}")

    if x_min >= x_max:
        raise ValueError(f"'x_min' must be less than 'x_max', but got {x_min=}, {x_max=}")

    if not x_min <= x.max():
        raise ValueError(
            f"'x_min' must be less than or equal to the largest 'x' value, but got {x_min} "
        )

    max_bin_width = x_max - x_min
    if not 0 < bin_width < max_bin_width:
        raise ValueError(
            f"'bin_width' must be in the range (0, {max_bin_width}), but got {bin_width=}"
        )


def bin_aggregate(
    x: np.ndarray,
    y: np.ndarray,
    num_bins: int,
    bin_width: Optional[float] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    aggr_fn: Optional[AggregateFunction] = np.nanmedian,
) -> BinAggregateResult:
    """Aggregate y-values in uniform intervals (bins) along the x-axis.

    The interval [x_min, x_max) is divided into `num_bins` uniformly spaced intervals of width
    `bin_width`. The value computed for each bin is the aggregation of all y-values whose
    corresponding x-value is in the interval. The default aggregation function is `np.nanmedian`.

    Parameters
    ----------
    x : np.ndarray
        1D numpy array of x-coordinates sorted in ascending order. Must have at least 2 elements,
        and all elements cannot be the same value
    y : np.ndarray
        N-dimensional numpy array with the same length as `x`. The 1D array is recommended to use
    num_bins : int
        The number of intervals to divide the x-axis into. Must be at least 2
    bin_width : Optional[float], optional
        The width of each bin on the x-axis. Must be positive, and less than x_max - x_min.
        If `None` passed it is computed as `(x_max - x_min) / num_bins`, by default None
    x_min : Optional[float], optional
        The inclusive leftmost value to consider on the x-axis. Must be less than or equal to the
        largest value of `x`. If `None` passed it is computed as `np.min(x)`, by default None
    x_max : Optional[float], optional
        The exclusive rightmost value to consider on the x-axis. Must be greater than x_min.
        If `None` passed it is computed as `np.max(x)`, by default None
    aggr_fn : Optional[AggregateFunction], optional
        A function that will be called with signature aggr_fn(y, axis=0) to aggregate values within
        each bin, by default `np.nanmedian`

    Returns
    -------
    BinAggregateResult
        results: numpy array of length `num_bins` containing the aggregated y-values of uniformly
            spaced bins on the x-axis.
        bin_counts: 1D numpy array of length `num_bins` indicating the number of y-values in each
            bin.

    Notes
    -----
    Parameter `x` must be sorted in ascending order.

    Raises
    ------
    ValueError
        - `x` is not sorted in ascending order
        - length of `x` and `y` mismatches
        - length of `x` is less than 2
        - `num_bins` is less than 2
        - `x_min` is greater than or equal to `x_max`
        - `x_min` is greater than the largest value of `x`
        - `bin_width` not satify a condition 0 < bin_width < x_max - x_min
    """
    if x_min is None:
        x_min = x.min()

    if x_max is None:
        x_max = x.max()

    if bin_width is None:
        bin_width = abs(x_max - x_min) / num_bins

    if not aggr_fn:
        aggr_fn = np.nanmedian

    _validate_bin_aggregate_inputs(
        x=x, y=y, num_bins=num_bins, bin_width=bin_width, x_min=x_min, x_max=x_max
    )

    bin_spacing = (x_max - x_min - bin_width) / (num_bins - 1)
    left_bin_edges = np.arange(x_min, x_max, bin_spacing)
    rigth_bin_edges = np.arange(x_min + bin_width, x_max + 10e-06, bin_spacing)
    bin_edges = zip(left_bin_edges, rigth_bin_edges)
    bins_masks = [np.logical_and(x >= left, x < rigth) for left, rigth in bin_edges]
    results = np.array([aggr_fn(y[m], axis=0) for m in bins_masks])
    bin_counts = np.count_nonzero(bins_masks, axis=1)
    return results, bin_counts
