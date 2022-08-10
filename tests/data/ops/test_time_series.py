"""Unit tests for `gaia.data.ops.time_series.py` module."""


from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from gaia.data.models import PeriodicEvent
from gaia.data.ops.time_series import _Segments, phase_fold_time, remove_events, split_arrays


@pytest.fixture(
    name="single_segment",
    params=(
        (
            np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0]),
            [np.array([0.1, 0.2, 0.3]), np.array([0.8, 0.9, 1.0])],
        ),
        (
            np.array([0.1]),
            [np.array([0.1])],
        ),
    ),
    ids=["one_element", "multi_elements"],
)
def fixture_single_segment(request):
    time, expected_time = request.param
    all_time_series = np.ones_like(time)
    expected_time_series = [np.ones_like(t) for t in expected_time]
    return time, all_time_series, expected_time, expected_time_series


@pytest.fixture(name="multi_segment")
def fixture_multi_segment(single_segment):
    def _fixture_multi_segment(n_segments):
        time, all_time_series, expected_time, expected_time_series = single_segment
        return (
            n_segments * [time] if n_segments > 1 else time,
            n_segments * [all_time_series] if n_segments > 1 else all_time_series,
            n_segments * expected_time,
            n_segments * expected_time_series,
        )

    return _fixture_multi_segment


@dataclass
class SplitArraysTestCase:
    n_features: int
    n_segments: int


@pytest.fixture(
    name="arrays_to_split",
    params=[
        SplitArraysTestCase(n_features=1, n_segments=1),
        SplitArraysTestCase(n_features=1, n_segments=2),
        SplitArraysTestCase(n_features=1, n_segments=3),
        SplitArraysTestCase(n_features=2, n_segments=1),
        SplitArraysTestCase(n_features=2, n_segments=2),
        SplitArraysTestCase(n_features=2, n_segments=3),
    ],
    ids=[
        "1_feature-1_segment",
        "1_feature-2_segments",
        "1_feature-3_segments",
        "2_feature-1_segment",
        "2_feature-2_segments",
        "2_feature-3_segments",
    ],
)
def fixture_arrays_to_split(request, multi_segment):
    case: SplitArraysTestCase = request.param
    time, all_time_series, expected_time, expected_time_series = multi_segment(case.n_segments)
    features = tuple(all_time_series for _ in range(case.n_features))
    return time, features, expected_time, expected_time_series


def test_split_arrays__return_correct_data(arrays_to_split):
    """
    Test check whether time and time series features are properly split or different input
    configurations.
    """
    all_time, all_features, expected_time, expected_features = arrays_to_split

    result_time, *result_ts = split_arrays(all_time, *all_features, gap_with=0.49)

    assert_array_equal(result_time, expected_time)
    for ts in result_ts:
        assert_array_equal(ts, expected_features)


@pytest.mark.parametrize("gap_width", [-1.0, 0.0])
def test_split_arrays__invalid_gap_width(gap_width: float):
    """Test check whether ValueError is raise when `gap_width` is less than `0`."""
    with pytest.raises(ValueError):
        split_arrays(np.arange(10), np.arange(10), gap_with=gap_width)


@pytest.mark.parametrize("period", [-1.0, 0.0])
def test_phase_fold_time__invalid_period(period: float):
    """Test check whether ValueError is raised when `period` is not a positive number."""
    with pytest.raises(ValueError):
        phase_fold_time(np.arange(10), epoch=1, period=period)


@dataclass
class PhaseFoldTestCase:
    data_type: str


@pytest.fixture(
    name="time_to_fold",
    params=[
        PhaseFoldTestCase(data_type="simple"),
        PhaseFoldTestCase(data_type="large_epoch"),
        PhaseFoldTestCase(data_type="negative_epoch"),
        PhaseFoldTestCase(data_type="negative_time"),
    ],
    ids=[
        "simple",
        "negative_epoch",
        "large_epoch",
        "negative_time",
    ],
)
def fixture_time_to_fold(request):
    case: PhaseFoldTestCase = request.param
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

    return data[case.data_type]


def test_phase_fold_time__return_correct_data(time_to_fold):
    """Test check whether the returned time is properly folded over an event phase."""
    time, expected, period, epoch = time_to_fold
    result = phase_fold_time(time, epoch, period)
    assert_array_almost_equal(result, expected)


@pytest.mark.parametrize("time_to_fold", [PhaseFoldTestCase(data_type="simple")], indirect=True)
def test_phase_fold_time__sort_folded_time(time_to_fold):
    """Test check whether a folded time is properly sort in ascending order."""
    time, expected, period, epoch = time_to_fold
    expected.sort()

    result = phase_fold_time(time, epoch, period, sort=True)

    assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    "all_time,all_ts,events,expected_time,expected_ts",
    [
        (
            np.arange(20),
            10 * np.arange(20),
            [PeriodicEvent(period=4, duration=1.5, epoch=3.5)],
            [np.array([1, 2, 5, 6, 9, 10, 13, 14, 17, 18])],
            [np.array([10, 20, 50, 60, 90, 100, 130, 140, 170, 180])],
        ),
        (
            np.arange(20),
            10 * np.arange(20),
            [
                PeriodicEvent(period=4, duration=1.5, epoch=3.5),
                PeriodicEvent(period=7, duration=1.5, epoch=6.5),
            ],
            [np.array([1, 2, 5, 9, 10, 17, 18])],
            [np.array([10, 20, 50, 90, 100, 170, 180])],
        ),
        (
            [np.arange(10), np.arange(10, 20)],
            [np.arange(0, 100, 10), np.arange(100, 200, 10)],
            [
                PeriodicEvent(period=4, duration=1.5, epoch=3.5),
                PeriodicEvent(period=7, duration=1.5, epoch=6.5),
            ],
            [np.array([1, 2, 5, 9]), np.array([10, 17, 18])],
            [np.array([10, 20, 50, 90]), np.array([100, 170, 180])],
        ),
    ],
    ids=["one_segment_one_event", "one_segment_two_events", "two_segments_two_events"],
)
def test_remove_events__one_feature(all_time, all_ts, events, expected_time, expected_ts):
    """Test check whether events are properly removed with only one time series feature."""
    result_time, result_ts = remove_events(all_time, events, f1=all_ts)

    for result, expetced in zip(result_time, expected_time):
        assert_array_equal(result, expetced)

    for result, expetced in zip(result_ts, expected_ts):
        assert_array_equal(result, expetced)


@pytest.mark.parametrize(
    "all_time,all_ts,events,expected_time,expected_ts",
    [
        (
            np.arange(20),
            10 * np.arange(20),
            [PeriodicEvent(period=4, duration=1.5, epoch=3.5)],
            [np.array([1, 2, 5, 6, 9, 10, 13, 14, 17, 18])],
            [np.array([10, 20, 50, 60, 90, 100, 130, 140, 170, 180])],
        ),
        (
            np.arange(20),
            10 * np.arange(20),
            [
                PeriodicEvent(period=4, duration=1.5, epoch=3.5),
                PeriodicEvent(period=7, duration=1.5, epoch=6.5),
            ],
            [np.array([1, 2, 5, 9, 10, 17, 18])],
            [np.array([10, 20, 50, 90, 100, 170, 180])],
        ),
        (
            [np.arange(10), np.arange(10, 20)],
            [np.arange(0, 100, 10), np.arange(100, 200, 10)],
            [
                PeriodicEvent(period=4, duration=1.5, epoch=3.5),
                PeriodicEvent(period=7, duration=1.5, epoch=6.5),
            ],
            [np.array([1, 2, 5, 9]), np.array([10, 17, 18])],
            [np.array([10, 20, 50, 90]), np.array([100, 170, 180])],
        ),
    ],
    ids=["one_segment_one_event", "one_segment_two_events", "two_segments_two_events"],
)
def test_remove_events__two_features(all_time, all_ts, events, expected_time, expected_ts):
    """Test check whether events are properly removed with more than one time series feature."""
    result_time, *result_ts = remove_events(all_time, events, f1=all_ts, f2=all_ts)

    for result, expetced in zip(result_time, expected_time):
        assert_array_equal(result, expetced)

    for result_feature in result_ts:
        for result, expetced in zip(result_feature, expected_ts):
            assert_array_equal(result, expetced)


@pytest.fixture(name="empty_segment")
def fixture_empty_segment(request):
    include_empty = request.param

    if include_empty:
        return (
            [np.arange(5), np.arange(10, 20)],
            [np.arange(0, 50, 10), np.arange(100, 200, 10)],
            [PeriodicEvent(period=10, duration=2, epoch=2.5)],
            [np.array([]), np.array([16, 17, 18, 19])],
            [np.array([]), np.array([160, 170, 180, 190])],
        )

    return (
        [np.arange(5), np.arange(10, 20)],
        [np.arange(0, 50, 10), np.arange(100, 200, 10)],
        [PeriodicEvent(period=10, duration=2, epoch=2.5)],
        [np.array([16, 17, 18, 19])],
        [np.array([160, 170, 180, 190])],
    )


@pytest.mark.parametrize(
    "include_empty,empty_segment",
    [(True, True), (False, False)],
    indirect=["empty_segment"],
    ids=["include", "exclude"],
)
def test_remove_events__handle_empty_segments(include_empty, empty_segment):
    """
    Test check whether empty segments are included in the return value if
    `include_empty_segments` is true. Otherwise, it should not be included.
    """
    all_time, all_ts, events, exp_time, exp_ts = empty_segment
    result_time, result_ts = remove_events(
        all_time, events, f1=all_ts, include_empty_segments=include_empty, width_factor=3
    )

    for result, expetced in zip(result_time, exp_time):
        assert_array_equal(result, expetced)

    for result, expetced in zip(result_ts, exp_ts):
        assert_array_equal(result, expetced)




