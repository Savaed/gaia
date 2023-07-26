import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from gaia.data.models import TCE, PeriodicEvent, TceLabel
from gaia.data.preprocessing import (
    AdjustedPadding,
    InvalidDimensionError,
    bin_aggregate,
    compute_euclidean_distance,
    compute_transits,
    interpolate_masked_spline,
    normalize_median,
    phase_fold_time,
    remove_events,
    split_arrays,
)
from tests.conftest import assert_iterable_of_arrays_almost_equal, assert_iterable_of_arrays_equal


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


@pytest.mark.parametrize("period", [-1, 0])
def test_phase_fold_time__invalid_parameters(period):
    """Test that `ValueError` is raised when any of the parameters are invalid."""
    with pytest.raises(ValueError):
        phase_fold_time(np.array([1, 2]), epoch=1, period=period)


def test_phase_fold_time__invalid_data_dimension():
    """Test that `InvalidDimensionError` is raised when `time` dimension != 1."""
    time = np.arange(10.0).reshape((2, 5))
    with pytest.raises(InvalidDimensionError):
        phase_fold_time(time, epoch=1, period=1)


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
    assert_iterable_of_arrays_equal(actual_time, expected_time)
    assert_iterable_of_arrays_equal(actual_series, expected_series)


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
    with pytest.raises(InvalidDimensionError):
        split_arrays(time, series)


@pytest.mark.parametrize(
    "X,Y,expected",
    [
        (
            np.array([1, 2, 3, 4, 5]),
            np.array([6, 7, 8, 9, 10]),
            np.array([6.08276253, 7.28010989, 8.54400375, 9.8488578, 11.18033989]),
        ),
        (
            np.array([1, 2, 3, np.nan, 5]),
            np.array([6, np.nan, 8, 9, 10]),
            np.array([6.08276253, np.nan, 8.54400375, np.nan, 11.18033989]),
        ),
    ],
)
def test_compute_euclidean_distance__return_correct_distance(X, Y, expected):
    """Test that the Euclidean distance between an array of 2D points is calculated correctly."""
    points = np.array([X, Y])
    actual = compute_euclidean_distance(points)
    assert_array_almost_equal(actual, expected)


def test_compute_euclidean_distance__invalid_data_dimension():
    """Test that `InvalidDimensionError` is raised when an array of points has dimension != 2."""
    points = np.array([1, 2, 3, 4, 5])
    with pytest.raises(InvalidDimensionError):
        compute_euclidean_distance(points)


@pytest.mark.parametrize(
    "series,expected",
    [
        (
            np.array([1, 2, 3, 4, 5]),
            np.array([0.33333333, 0.66666667, 1.0, 1.33333333, 1.66666667]),
        ),
        (
            np.array([1, 2, np.nan, 4, 5]),
            np.array([0.33333333, 0.66666667, np.nan, 1.33333333, 1.66666667]),
        ),
    ],
)
def test_normalize_median__return_correct_data(series, expected):
    """Test that data is correctly normalized: `normalized = series / nanmedian(series)`."""
    actual = normalize_median(series)
    assert_array_almost_equal(actual, expected)


def test_normalize_median__invalid_data_dimension():
    """Test that `InvalidDimensionError` is raised when an input array has dimension != 1."""
    points = np.array([1, 2, 3, 4, 5, 6]).reshape((2, 3))
    with pytest.raises(InvalidDimensionError):
        normalize_median(points)


@pytest.mark.parametrize(
    "tce,time,expected",
    [
        (
            TCE(
                id=1,
                target_id=2,
                name=None,
                label=TceLabel.PC,
                event=PeriodicEvent(epoch=1, duration=1, period=4),
            ),
            np.arange(10.0),
            np.array(["1", "1", "1", "no detected", "1", "1", "1", "no detected", "1", "1"]),
        ),
        (
            TCE(
                id=1,
                target_id=2,
                name="tce-name",
                label=TceLabel.PC,
                event=PeriodicEvent(epoch=1, duration=1, period=4),
            ),
            np.arange(10.0),
            np.array(
                [
                    "tce-name",
                    "tce-name",
                    "tce-name",
                    "no detected",
                    "tce-name",
                    "tce-name",
                    "tce-name",
                    "no detected",
                    "tce-name",
                    "tce-name",
                ],
            ),
        ),
    ],
    ids=["tce_id", "tce_name"],
)
def test_compute_transits__compute_correctly(tce, time, expected):
    """Test that TCE transits are correctly computed."""
    actual = compute_transits([tce], time)
    assert_array_equal(actual, expected)


def test_compute_transits__invalid_time_dimension():
    """Test that `InvalidDimensionError` is raised when `time` has dimension != 1D."""
    tce = TCE(id=1, target_id=1, name="tce", label=TceLabel.PC, event=PeriodicEvent(1, 1, 1))
    time = np.arange(10.0).reshape((2, 5))
    with pytest.raises(InvalidDimensionError):
        compute_transits([tce], time)


@pytest.mark.parametrize(
    "period,duration,secondary_phase,expected",
    [
        (1, 0.5, 1.2, 1),
        (1, 0.3, 1.2, 0.9),
        (1, 0.5, 0.3, 0.3),
        (1, 0.5, -0.3, 0.3),
    ],
)
def test_adjasted_padding__case(period, duration, secondary_phase, expected):
    """Test that event width is correctly computed."""
    adjusted_padding = AdjustedPadding(secondary_phase)
    actual = adjusted_padding(period, duration)
    assert actual == pytest.approx(expected)


def three_duration(_, duration):  # Implementation of `EventWidthStrategy` for testing.
    return 3 * duration


@pytest.mark.parametrize(
    "all_time,series,events,expected_time,expected_series",
    [
        (
            [np.arange(20)],
            [10 * np.arange(20)],
            [PeriodicEvent(period=4, duration=1, epoch=3)],
            [np.array([1, 5, 9, 13, 17])],
            [np.array([10, 50, 90, 130, 170])],
        ),
        (
            [np.arange(20)],
            [10 * np.arange(20)],
            [
                PeriodicEvent(period=4, duration=1, epoch=3),
                PeriodicEvent(period=7, duration=1, epoch=6),
            ],
            [np.array([1, 9, 17])],
            [np.array([10, 90, 170])],
        ),
        (
            [np.arange(10), np.arange(10, 20)],
            [np.arange(0, 100, 10), np.arange(100, 200, 10)],
            [
                PeriodicEvent(period=4, duration=1, epoch=3),
                PeriodicEvent(period=7, duration=1, epoch=6),
            ],
            [np.array([1, 9]), np.array([17])],
            [np.array([10, 90]), np.array([170])],
        ),
    ],
    ids=["one_segment_one_event", "one_segment_two_events", "two_segments_two_events"],
)
def test_remove_events__remove_events_correctly(
    all_time,
    series,
    events,
    expected_time,
    expected_series,
):
    """Test check whether events are properly removed with only one time series feature."""
    actual_time, actual_series = remove_events(all_time, events, series, three_duration)
    assert_iterable_of_arrays_equal(actual_time, expected_time)
    assert_iterable_of_arrays_equal(actual_series, expected_series)


@pytest.mark.parametrize(
    "include_empty,all_time,series,events,expected_time,expected_series",
    [
        (
            True,
            [np.arange(5), np.arange(10, 20)],
            [np.arange(0, 50, 10), np.arange(100, 200, 10)],
            [PeriodicEvent(period=10, duration=2, epoch=2.5)],
            [np.array([]), np.array([16, 17, 18, 19])],
            [np.array([]), np.array([160, 170, 180, 190])],
        ),
        (
            False,
            [np.arange(5), np.arange(10, 20)],
            [np.arange(0, 50, 10), np.arange(100, 200, 10)],
            [PeriodicEvent(period=10, duration=2, epoch=2.5)],
            [np.array([16, 17, 18, 19])],
            [np.array([160, 170, 180, 190])],
        ),
    ],
    ids=["include_empty_segments", "exclude_empty_segments"],
)
def test_remove_events__handle_empty_segments(
    include_empty,
    all_time,
    series,
    events,
    expected_time,
    expected_series,
):
    """Test that empty segments are returned if `include_empty_segments=True`."""
    result_time, result_series = remove_events(
        all_time,
        events,
        series,
        three_duration,
        include_empty_segments=include_empty,
    )
    assert_iterable_of_arrays_equal(result_time, expected_time)
    assert_iterable_of_arrays_equal(result_series, expected_series)


@pytest.mark.parametrize(
    "time,series,events",
    [
        ([np.arange(5)], [np.arange(5)], []),
        ([np.arange(5), np.arange(5)], [np.arange(5)], [PeriodicEvent(1, 1, 1)]),
    ],
    ids=["no_events_provided", "different_time_and_series_lenghts"],
)
def test_remove_events__invalid_inputs(time, series, events):
    """Test that `ValueError` is raised when any of inputs is invalid."""
    with pytest.raises(ValueError):
        remove_events(time, events, series, three_duration)


def test_remove_events__invalid_data_dimension():
    """Test that `InvalidDimensionError` is raised when any of inputs is invalid."""
    time = [np.arange(10.0).reshape((2, 5))]
    series = [np.arange(5.0)]
    events = [PeriodicEvent(1, 1, 1)]
    with pytest.raises(InvalidDimensionError):
        remove_events(time, events, series, three_duration)


@pytest.mark.parametrize(
    "time,masked_time,masked_splines",
    [
        (
            [np.array([1, 2])],
            [np.array([1, 2])],
            [np.array([1, 2]), np.array([1, 2])],
        ),
        (
            [np.array([1, 2])],
            [np.array([1, 2]), np.array([1, 2])],
            [np.array([1, 2])],
        ),
        (
            [np.array([1, 2]), np.array([1, 2])],
            [np.array([1, 2])],
            [np.array([1, 2])],
        ),
    ],
    ids=["invalid_masked_splines", "invalid_masked_time", "invalid_time"],
)
def test_interpolate_masked_spline__inputs_lenghts_mismatch(time, masked_time, masked_splines):
    """Test that `ValueError` is raised when lengths of `time`, `masked_time` or `masked_splines`
    are different.
    """
    with pytest.raises(ValueError):
        interpolate_masked_spline(time, masked_time, masked_splines)


@pytest.mark.parametrize(
    "time,masked_time,masked_splines",
    [
        (
            [np.array([1, 2])],
            [np.array([1, 2])],
            [np.array([1, 2]).reshape((2, 1))],
        ),
        (
            [np.array([1, 2])],
            [np.array([1, 2]).reshape((2, 1))],
            [np.array([1, 2])],
        ),
        (
            [np.array([1, 2]).reshape((2, 1))],
            [np.array([1, 2])],
            [np.array([1, 2])],
        ),
    ],
    ids=["invalid_masked_splines", "invalid_masked_time", "invalid_time"],
)
def test_interpolate_masked_spline__invalid_data_dimension(time, masked_time, masked_splines):
    """Test that `InvalidDimensionError` is raised when dimension of `time`, `masked_time` or
    `masked_splines` != 1.
    """
    with pytest.raises(InvalidDimensionError):
        interpolate_masked_spline(time, masked_time, masked_splines)


@pytest.mark.parametrize(
    "time,masked_time,masked_splines,expected",
    [
        ([], [], [], []),
        (
            [np.linspace(0, 1, 21)],
            [np.linspace(0, 1, 20)],
            [np.sin(np.linspace(0, 1, 20))],
            [np.sin(np.linspace(0, 1, 21))],
        ),
        (
            [np.linspace(0, 1, 21)],
            [np.linspace(0, 1, 20)],
            [2 * np.linspace(0, 1, 20)],
            [2 * np.linspace(0, 1, 21)],
        ),
        (
            [np.linspace(0, 1, 21), np.linspace(1, 2, 21)],
            [np.linspace(0, 1, 20), np.linspace(1, 2, 20)],
            [np.sin(np.linspace(0, 1, 20)), np.sin(np.linspace(1, 2, 20))],
            [np.sin(np.linspace(0, 1, 21)), np.sin(np.linspace(1, 2, 21))],
        ),
        (
            [np.linspace(0, 1, 21), np.linspace(1, 2, 21)],
            [np.linspace(0, 1, 20), np.array([])],
            [np.sin(np.linspace(0, 1, 20)), np.array([])],
            [np.sin(np.linspace(0, 1, 21)), np.array([np.nan] * 21)],
        ),
    ],
    ids=["all_inputs_empty", "sin", "linear", "2_segments_sin", "empty_segment"],
)
def test_interpolate_masked_spline__interpolate_correctly(
    time,
    masked_time,
    masked_splines,
    expected,
):
    """Test that masked spline is correctly linearly interpolate."""
    actual = interpolate_masked_spline(time, masked_time, masked_splines)
    assert_iterable_of_arrays_almost_equal(actual, expected, relative_tolerance=0.001)


@pytest.mark.parametrize(
    "x,y,num_bins,bin_width,x_min,x_max",
    [
        (np.arange(10.0), np.arange(10.0), 1, 1, 0, 9),
        (np.arange(10.0), np.arange(10.0), 2, 1, 9, 0),
        (np.arange(10.0), np.arange(10.0), 2, 1, 11, None),
        (np.arange(10.0), np.arange(10.0), 2, 11, 0, 9),
        (np.arange(10.0), np.arange(10.0), 2, 0, 0, 9),
        (np.array([1]), np.array([1]), 2, 11, 0, 9),
        (np.arange(10.0)[::-1], np.arange(10.0), 2, 1, 0, 9),
        (np.arange(10.0), np.arange(20.0), 2, 1, 0, 9),
    ],
    ids=[
        "num_bins<2",
        "x_min>x_max",
        "x_min>x.max()",  # In current implementation this is the same case as `x_min>x_max`.
        "bin_width_to_big",
        "bin_width_equals_0",
        "len(x)<2",
        "x_not_sorted_asc",
        "len(x)!=len(y)",
    ],
)
def test_bin_aggregate__invalid_input(x, y, num_bins, bin_width, x_min, x_max):
    """Test that `ValueError` is raised when any of inputs is invalid."""
    with pytest.raises(ValueError):
        bin_aggregate(x, y, num_bins=num_bins, bin_width=bin_width, x_min=x_min, x_max=x_max)


@pytest.mark.parametrize(
    "x,y",
    [
        (np.arange(10).reshape((2, 5)), np.arange(10)),
        (np.arange(10), np.arange(10).reshape((2, 5))),
    ],
)
def test_bin_aggregate__invalid_data_dimension(x, y):
    """Test that `InvalidDimensionError` is raised when any of inputs is invalid."""
    with pytest.raises(InvalidDimensionError):
        bin_aggregate(x, y, num_bins=5)


@pytest.mark.parametrize(
    "x,y,num_bins,bin_width,expected_aggr,expected_bin_counts",
    [
        (
            np.arange(0, 1.1, 0.1),
            np.arange(0, 1.1, 0.1),
            10,
            None,
            np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ),
        (
            np.arange(0, 2.1, 0.1),
            np.arange(0, 2.1, 0.1),
            10,
            None,
            np.array([0.05, 0.25, 0.45, 0.65, 0.85, 1.05, 1.25, 1.45, 1.65, 1.85]),
            np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
        ),
        (
            np.arange(0, 2, 0.1),
            np.arange(0, 2, 0.1),
            10,
            None,
            np.array([0.05, 0.25, 0.45, 0.65, 0.85, 1.05, 1.25, 1.45, 1.65, 1.8]),
            np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 1]),
        ),
        (
            np.arange(0, 1.1, 0.1),
            np.arange(0, 1.1, 0.1),
            4,
            0.2,
            np.array([0.05, 0.35, 0.65, 0.85]),
            np.array([2, 2, 2, 2]),
        ),
        (
            np.arange(0, 1.1, 0.1),
            np.arange(0, 1.1, 0.1),
            4,
            0.3,
            np.array([0.1, 0.4, 0.6, 0.8]),
            np.array([3, 3, 3, 3]),
        ),
    ],
    ids=[
        "single_value_bins",
        "multiple_values_bins",
        "unequal_bins",
        "bins_with_gaps",
        "overlaying_bins",
    ],
)
def test_bin_aggregate__aggregate_correctly(
    x,
    y,
    num_bins,
    bin_width,
    expected_aggr,
    expected_bin_counts,
):
    """Test that the data is binnarize and aggregate correctly."""
    actual_result, actual_bin_counts = bin_aggregate(
        x,
        y,
        num_bins=num_bins,
        bin_width=bin_width,
        aggr_func=np.median,
    )
    assert_array_almost_equal(actual_result, expected_aggr)
    assert_array_equal(actual_bin_counts, expected_bin_counts)


@pytest.mark.parametrize(
    "x,y,num_bins,x_min,x_max,expected_aggr,expected_bin_counts",
    [
        (
            np.arange(0, 2.1, 0.1),
            np.arange(0, 2.1, 0.1),
            5,
            1,
            None,
            np.array([1.05, 1.25, 1.45, 1.65, 1.85]),
            np.array([2, 2, 2, 2, 2]),
        ),
        (
            np.arange(0, 2.1, 0.1),
            np.arange(0, 2.1, 0.1),
            5,
            None,
            1,
            np.array([0.05, 0.25, 0.45, 0.65, 0.85]),
            np.array([2, 2, 2, 2, 2]),
        ),
        (
            np.arange(0, 2.1, 0.1),
            np.arange(0, 2.1, 0.1),
            5,
            0.5,
            1.5,
            np.array([0.55, 0.75, 0.95, 1.15, 1.35]),
            np.array([2, 2, 2, 2, 2]),
        ),
    ],
    ids=["min", "max", "both"],
)
def test_bin_aggregate__respect_x_min_max_boundaries(
    x,
    y,
    num_bins,
    x_min,
    x_max,
    expected_aggr,
    expected_bin_counts,
):
    """Test that the data is binnarize and aggregate correctly with min and max boundires set."""
    actual_result, actual_bin_counts = bin_aggregate(
        x,
        y,
        num_bins=num_bins,
        aggr_func=np.median,
        x_min=x_min,
        x_max=x_max,
    )
    assert_array_almost_equal(actual_result, expected_aggr)
    assert_array_equal(actual_bin_counts, expected_bin_counts)


def test_bin_aggregate__default_value_for_empty_bins():
    """Test that default value is used as aggregation for empty bins."""
    x = np.arange(0, 1, 0.1)
    y = np.arange(0, 1, 0.1)
    num_bins = 4
    default = 1.0
    x_min = 0.6
    expected_aggr = np.array([0.6, 0.7, 0.8, 1.0])
    expected_bin_counts = np.array([1, 1, 1, 0])
    actual_result, actual_bin_counts = bin_aggregate(
        x,
        y,
        num_bins=num_bins,
        x_min=x_min,
        default=default,
    )
    assert_array_almost_equal(actual_result, expected_aggr)
    assert_array_equal(actual_bin_counts, expected_bin_counts)


@pytest.mark.parametrize(
    "aggr_func,x,y,num_bins,expected_aggr,expected_bin_counts",
    [
        (
            None,
            np.arange(0, 2.1, 0.1),
            np.arange(0, 2.1, 0.1),
            10,
            np.array([0.05, 0.25, 0.45, 0.65, 0.85, 1.05, 1.25, 1.45, 1.65, 1.85]),
            np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
        ),
        (
            np.max,
            np.arange(0, 2.1, 0.1),
            np.arange(0, 2.1, 0.1),
            10,
            np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]),
            np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
        ),
        (
            np.min,
            np.arange(0, 2.1, 0.1),
            np.arange(0, 2.1, 0.1),
            10,
            np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]),
            np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
        ),
        (
            np.mean,
            np.arange(0, 2.1, 0.1),
            np.arange(0, 2.1, 0.1),
            2,
            np.array([0.45, 1.45]),
            np.array([10, 10]),
        ),
        (
            lambda y: np.mean(y) ** 2,
            np.arange(0, 2.1, 0.1),
            np.arange(0, 2.1, 0.1),
            2,
            np.array([0.2025, 2.1025]),
            np.array([10, 10]),
        ),
    ],
    ids=["default_nanmedian", "max", "min", "mean", "custom"],
)
def test_bin_aggregate__different_aggregation_function(
    aggr_func,
    x,
    y,
    num_bins,
    expected_aggr,
    expected_bin_counts,
):
    """Test that various aggregate functions works propperly."""
    actual_result, actual_bin_counts = bin_aggregate(x, y, num_bins=num_bins, aggr_func=aggr_func)
    assert_array_almost_equal(actual_result, expected_aggr)
    assert_array_equal(actual_bin_counts, expected_bin_counts)
