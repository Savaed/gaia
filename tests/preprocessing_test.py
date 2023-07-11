import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from gaia.data.preprocessing import phase_fold_time, split_arrays


@pytest.fixture(params=["simple", "large_epoch", "negative_epoch", "negative_time"])
def time_to_fold(request):
    """Return data to test time vector folding in form of `(time, expected, period, epoch)`."""
    time = np.arange(0, 2, 0.1)
    period = 1

    data = {
        "simple": (
            time,
            [
                -0.45,
                -0.35,
                -0.25,
                -0.15,
                -0.05,
                0.05,
                0.15,
                0.25,
                0.35,
                0.45,
                -0.45,
                -0.35,
                -0.25,
                -0.15,
                -0.05,
                0.05,
                0.15,
                0.25,
                0.35,
                0.45,
            ],
            period,
            0.45,
        ),
        "large_epoch": (
            time,
            [
                -0.25,
                -0.15,
                -0.05,
                0.05,
                0.15,
                0.25,
                0.35,
                0.45,
                -0.45,
                -0.35,
                -0.25,
                -0.15,
                -0.05,
                0.05,
                0.15,
                0.25,
                0.35,
                0.45,
                -0.45,
                -0.35,
            ],
            period,
            1.25,
        ),
        "negative_epoch": (
            time,
            [
                -0.35,
                -0.25,
                -0.15,
                -0.05,
                0.05,
                0.15,
                0.25,
                0.35,
                0.45,
                -0.45,
                -0.35,
                -0.25,
                -0.15,
                -0.05,
                0.05,
                0.15,
                0.25,
                0.35,
                0.45,
                -0.45,
            ],
            period,
            -1.65,
        ),
        "negative_time": (
            np.arange(-3, -1, 0.1),
            [
                0.45,
                -0.45,
                -0.35,
                -0.25,
                -0.15,
                -0.05,
                0.05,
                0.15,
                0.25,
                0.35,
                0.45,
                -0.45,
                -0.35,
                -0.25,
                -0.15,
                -0.05,
                0.05,
                0.15,
                0.25,
                0.35,
            ],
            period,
            0.55,
        ),
    }
    return data[request.param]


def test_phase_fold_time__return_correct_data(time_to_fold):
    """Test that the returned time is properly folded over an event phase."""
    time, expected, period, epoch = time_to_fold
    actual = phase_fold_time(time, epoch=epoch, period=period)
    assert_array_almost_equal(actual, expected)


@pytest.mark.parametrize(
    "time,period",
    [(np.arange(10), -1), (np.arange(10).reshape((2, 5)), 1)],
    ids=["good_time_bad_period", "good_period_bad_time"],
)
def test_phase_fold_time__invalid_parameters(time, period):
    """Test that `ValueError` is raised when any of the parameters are invalid."""
    with pytest.raises(ValueError):
        phase_fold_time(time, epoch=1, period=period)


@pytest.mark.parametrize(
    "time,series,expected",
    [
        ([np.array([])], [np.array([])], ([], [])),
        ([], [], ([], [])),
        (
            [np.array([0.1, 0.2, 0.3, 1.1, 1.2])],
            [np.array([1, 2, 3, 4, 5])],
            (
                [np.array([0.1, 0.2, 0.3]), np.array([1.1, 1.2])],
                [np.array([1, 2, 3]), np.array([4, 5])],
            ),
        ),
        (
            [np.array([0.1, 0.2, 0.3, 1.1, 1.2]), np.array([1.3, 1.4])],
            [np.array([1, 2, 3, 4, 5]), np.array([6, 7])],
            (
                [np.array([0.1, 0.2, 0.3]), np.array([1.1, 1.2]), np.array([1.3, 1.4])],
                [np.array([1, 2, 3]), np.array([4, 5]), np.array([6, 7])],
            ),
        ),
    ],
    ids=["empty_segments", "empty_series", "single_segment", "multiple_segments"],
)
def test_split_arrays__return_correct_data(time, series, expected):
    """Test that time and time series features are properly split."""
    actual_time, actual_series = split_arrays(time, series)
    expected_time, expected_series = expected
    assert all([np.array_equal(left, right) for left, right in zip(actual_time, expected_time)])
    assert all([np.array_equal(left, right) for left, right in zip(actual_series, expected_series)])


@pytest.mark.parametrize("gap_width", [-1.0, 0.0])
def test_split_arrays__invalid_gap_width(gap_width):
    """Test that `ValueError` is raised when `gap_width` < 0."""
    series = [np.array([1, 2, 3])]
    with pytest.raises(ValueError, match="'gap_width' > 0"):
        split_arrays(series, series, gap_with=gap_width)


@pytest.mark.parametrize(
    "time,series",
    [
        ([np.arange(10).reshape((2, 5))], [np.arange(10)]),
        ([np.arange(10)], [np.arange(10).reshape((2, 5))]),
    ],
    ids=[
        "2D_series_1D_time",
        "1D_time_2D_series",
    ],
)
def test_split_arrays__invalid_time_or_series_dimensions(time, series):
    """
    Test that `ValueError` is raised when any of individual `time` or `series` values has
    dimension != 1.
    """
    with pytest.raises(ValueError, match="Expected all series in 'time' and 'series' be 1D"):
        split_arrays(time, series)
