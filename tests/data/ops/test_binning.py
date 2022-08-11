"""Unit tests for `gaia.data.ops.binning` module."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from gaia.data.ops.binning import bin_aggregate


@pytest.mark.parametrize(
    "x,y,num_bins,bin_width,x_min,x_max,aggr_fn,expected_error_msg",
    [
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3, 4]),
            2,
            1,
            1,
            3,
            None,
            r"'len\(x\)' and 'len\(y\)' must be the same.*",
        ),
        (
            np.array([1, 3, 2]),
            np.array([1, 2, 3]),
            2,
            1,
            1,
            3,
            None,
            "'x' must be sorted in ascending order",
        ),
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            1,
            1,
            1,
            3,
            None,
            r"'num_bins' must be at least 2,.*",
        ),
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            2,
            -1,
            1,
            3,
            None,
            r"'bin_width' must be in the range.*",
        ),
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            2,
            0,
            1,
            3,
            None,
            r"'bin_width' must be in the range.*",
        ),
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            2,
            100,
            1,
            3,
            None,
            r"'bin_width' must be in the range.*",
        ),
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            2,
            1,
            3,
            1,
            None,
            r"'x_min' must be less than 'x_max'.*",
        ),
    ],
    ids=[
        "len(x)_mismatch_len(y)",
        "x_in_desc_order",
        "num_bin_less_than_2",
        "negative_bin_width",
        "bin_width_is_zero",
        "bin_width_is_to_big",
        "x_min_greater_than_x_max",
    ],
)
def test_bin_aggregate__invalid_inputs(
    x, y, num_bins, bin_width, x_min, x_max, aggr_fn, expected_error_msg
):
    """Test check whether ValueError with proper message is raised when any of inputs is invalid."""
    with pytest.raises(ValueError, match=expected_error_msg):
        bin_aggregate(
            x=x,
            y=y,
            num_bins=num_bins,
            bin_width=bin_width,
            x_min=x_min,
            x_max=x_max,
            aggr_fn=aggr_fn,
        )


@pytest.mark.parametrize(
    "x,y,num_bins,bin_width,x_min,x_max,expected_results,expected_bin_counts",
    [
        (
            np.array([-4, -2, -2, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3]),
            np.array([0, -1, 1, 4, 5, 6, 1, 3, 3, 9, 1, 1, 1, 1, -2]),
            5,
            2,
            -5,
            5,
            [0, 0, 5, 4, 0.4],
            [1, 2, 3, 4, 5],
        ),
        (
            np.array([-4, -2, -2, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3]),
            np.array(
                [
                    [0, 123],
                    [-1, 0],
                    [1, 2],
                    [4, -4],
                    [5, 4],
                    [6, 6],
                    [1, 50],
                    [3, 100],
                    [3, 100],
                    [9, 110],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [-2, 5],
                ]
            ),
            5,
            2,
            -5,
            5,
            [[0, 123], [0, 1], [5, 2], [4, 90], [0.4, 1]],
            [1, 2, 3, 4, 5],
        ),
    ],
    ids=["1D_y", "2D_y"],
)
def test_bin_aggregate__mean_function(
    x, y, num_bins, bin_width, x_min, x_max, expected_results, expected_bin_counts
):
    """Test check whether a correct data is returned with aggregate values as means."""
    results, bin_counts = bin_aggregate(
        x=x,
        y=y,
        num_bins=num_bins,
        bin_width=bin_width,
        x_min=x_min,
        x_max=x_max,
        aggr_fn=np.mean,
    )

    assert_array_almost_equal(results, expected_results)
    assert_array_equal(bin_counts, expected_bin_counts)


@pytest.mark.parametrize(
    "x,y,num_bins,bin_width,x_min,x_max,expected_results,expected_bin_counts",
    [
        (
            np.array([-4, -2, -2, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3]),
            np.array([0, -1, 1, 4, 5, 6, 1, 3, 3, 9, 1, 1, 1, 1, -2]),
            5,
            2,
            -5,
            5,
            [0, 0, 5, 3, 1],
            [1, 2, 3, 4, 5],
        ),
        (
            np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]),
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            5,
            6,
            -7,
            7,
            [3, 4.5, 6.5, 8.5, 10.5],
            [5, 6, 6, 6, 6],
        ),
        (
            np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]),
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            5,
            1,
            -4.5,
            4.5,
            [3, 5, 7, 9, 11],
            [1, 1, 1, 1, 1],
        ),
    ],
    ids=["1D_y", "wide_bins", "narrow_bins"],
)
def test_bin_aggregate__median_function(
    x, y, num_bins, bin_width, x_min, x_max, expected_results, expected_bin_counts
):
    """Test check whether a correct data is returned with aggregate values as means."""
    results, bin_counts = bin_aggregate(
        x=x,
        y=y,
        num_bins=num_bins,
        bin_width=bin_width,
        x_min=x_min,
        x_max=x_max,
        aggr_fn=np.median,
    )

    assert_array_almost_equal(results, expected_results)
    assert_array_equal(bin_counts, expected_bin_counts)
