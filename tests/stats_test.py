import numpy as np
import pytest
from numpy.testing import assert_array_equal

from gaia.stats import bic, diffs


def test_bic__compute_correct_bic():
    """Test that Bayesian information criterion is computed correctly."""
    expected = 2.8378770664113455
    actual = bic(1, 1, 1, 1, 1)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "k,n,ssr,sigma",
    [
        (-1, 1, 2, 3),
        (1, 0, 2, 3),
        (1, 2, -1, 3),
        (1, 2, 3, -1),
    ],
)
def test_bic__invalid_inputs(k, n, ssr, sigma):
    """Test that `ValueError` is raised when any of inputs is invalid."""
    with pytest.raises(ValueError):
        bic(k, n, sigma, ssr)


@pytest.mark.parametrize(
    "y,scale_coeff,expected",
    [
        ([np.array([1, 1, 1])], 1.0, np.array([0, 0])),
        ([np.array([1, 2, 3])], 1.0, np.array([1, 1])),
        ([np.array([3, 2, 1])], 1.0, np.array([-1, -1])),
        ([np.array([-3, -2, -1])], 1.0, np.array([1, 1])),
        ([np.array([1, 2, 3])], 2, np.array([0.5, 0.5])),
    ],
)
def test_diffs__compute_correct_values(y, scale_coeff, expected):
    """Test that diffs (`y[i+1] - y[i]`) are computed correctly."""
    actual = diffs(y, scale_coeff)
    assert_array_equal(actual, expected)


def test_diffs__scale_coeff_equal_zero():
    """Test that `ValueError` is raised when `scale_coeff=0`."""
    with pytest.raises(ValueError):
        diffs([np.array([1, 2, 3])], 0)
