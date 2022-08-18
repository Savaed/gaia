# pylint: skip-file

"""Unit tests for `gaia.data.ops.fitting` module."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from gaia.data.ops.fitting import KeplerSpline


CUBIC_FN = lambda x: (x - 5) ** 3 + 2 * (x - 5) ** 2 + 10


@pytest.fixture(name="make_interp_data")
def fixture_make_interp_data():
    def _make_interp_data(insert_outliers, fn_to_interpolate):
        x = [np.arange(0, 10, 0.1) for _ in range(2)]
        y = [fn_to_interpolate(xi) for xi in x]

        if insert_outliers:
            outliers_indices = [25, 50, 75]
            for i in range(len(x)):
                y[i][outliers_indices] = [9, -3, 6]
        else:
            outliers_indices = []

        return x, y, outliers_indices

    return _make_interp_data


@pytest.fixture(name="interp_data")
def fixture_interp_data(request, make_interp_data):
    insert_outliers, fn_to_interpolate = request.param
    return make_interp_data(insert_outliers, fn_to_interpolate)


@pytest.fixture(name="knots_spacing")
def fixture_knots_spacing():
    return np.logspace(np.log10(0.5), np.log10(1), num=20)


class TestKeplerSpline:
    @pytest.mark.parametrize(
        "interp_data",
        [
            (False, np.sin),
            (True, np.sin),
            (False, lambda x: x * x),
            (True, lambda x: x * x),
            (False, CUBIC_FN),
            (True, CUBIC_FN),
        ],
        indirect=["interp_data"],
        ids=[
            "sin_without_outliers",
            "sin_with_outliers",
            "quadratic_without_outliers",
            "quadratic_with_outliers",
            "cubic_without_outliers",
            "cubic_with_outliers",
        ],
    )
    def test_interpolation_correctness(self, interp_data, knots_spacing):
        """
        Test check whether a very close interpolated data and a correct
        outlier mask are returned for specific function interpolation.
        """
        x, y, outliers_indices = interp_data

        # This mask indicates points that should be closely interpolated. Some other points may be
        # excluded in the iterative process of removing outliers after interpolation.
        real_mask = np.ones_like(x, dtype=bool)
        for i in range(len(x)):
            real_mask[i][outliers_indices] = False

        result = KeplerSpline(x, y).fit(knots_spacing)

        # Expect a close fit beyond the outliers for each segment
        for result_i, yi, mask in zip(result, y, real_mask):
            assert_array_almost_equal(result_i[mask], yi[mask], decimal=2)

    @pytest.mark.parametrize(
        "x,k",
        [
            ([np.array([1, 2])], 2),
            ([np.array([1])], 2),
            ([np.array([1, 2, 3])], 3),
            ([np.array([1, 2, 3, 4])], 4),
        ],
        ids=[
            "2x_k=2",
            "1x_k=2",
            "3x_k=3",
            "4x_k=4",
        ],
    )
    def test_not_enough_data_points(self, x: int, k: int, knots_spacing: np.ndarray):
        """
        Test check whether the output is all NaNs when the
        is not enough points to fit one of the segments.

        Min number of data points is `spline_degree + 1`.
        """
        y = np.ones_like(x)
        expected = [np.array([np.nan] * x[0].size)]

        result = KeplerSpline(x, y).fit(knots_spacing, k=k)

        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "k,maxiter",
        [(-1, 2), (6, 2), (2, 0), (12, -3)],
        ids=[
            "k_to_low_maxiter_ok",
            "k_to_high_maxiter_ok",
            "k_ok_maxiter_bad",
            "k_bad_maxiter_bad",
        ],
    )
    def test_invalid_inputs(self, k: int, maxiter: int, knots_spacing: np.ndarray):
        """
        Test check whether ValueError is raised when a spline
        degree (`k`) is not 1 <= k <=5 or `maxiter` is less than 1.
        """
        x = [np.arange(10)]
        y = np.ones_like(x)

        with pytest.raises(ValueError):
            KeplerSpline(x, y).fit(knots_spacing, k=k, maxiter=maxiter)
