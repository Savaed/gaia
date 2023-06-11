import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from gaia.data.preprocessing import phase_fold_time


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
