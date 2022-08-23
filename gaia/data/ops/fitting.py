"""Functions for computing normalization splines for Kepler light curves."""

import math
from asyncio.log import logger
from dataclasses import dataclass

import numpy as np
import structlog
from scipy.interpolate import LSQUnivariateSpline
from scipy.stats import median_abs_deviation

from gaia.stats import bic, diffs, robust_mean


logger = structlog.stdlib.get_logger()


@dataclass
class InsufficientPointsError(Exception):
    """Raised when insufficient points are available for spline fitting."""

    available_points: int
    num_min_points: int


class SplineError(Exception):
    """Raised when an error occurs in the underlying spline-fitting implementation."""


class KeplerSpline:
    """B-spline for Kepler time series values."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self._x = x
        self._y = y

    def fit(
        self,
        knots_spacing: np.ndarray,
        k: int = 3,
        maxiter: int = 5,
        sigma_cut: int = 3,
        penalty_coeff: float = 1.0,
    ) -> list[np.ndarray]:
        """Calculate the best fit spline curve for each time series segment.

        The spline is fitted using an iterative process of removing outliers that can cause the
        spline be "pulled" by points with extreme values. In each iteration, the spline is fitted,
        and if they are any points where the absolute deviation from the median of the outliers is
        at least 3 * sigma (where sigma is an estimate of the standard deviation of the residuals)
        these points are removed, and the spline is re-fitted.

        Parameters
        ----------
        knots_spacing : np.ndarray
            Sequence of spaces between internal knots (breakpoints) in the spline
        k : int, optional
            The degree of a spline, by default 3
        maxiter : int, optional
            Maximum re-fitting interactions number, by default 5
        sigma_cut : int, optional
            The maximum number of standard deviations from the median spline residual before
            a point is considered an outlier, by default 3
        penalty_coeff : float, optional
            Coefficient of penalty term in Bayesian Information Criterium, by default 1.0

        Returns
        -------
        list[np.ndarray]
            Best fitted b-splines for each time segment
        """
        log = logger.bind()
        log.info(
            "Fit b-splines to time series",
            knots_spacing=knots_spacing,
            k=k,
            maxiter=maxiter,
            sigma_cut=sigma_cut,
            penalty_coeff=penalty_coeff,
        )

        if maxiter < 1:
            raise ValueError(f"'maxiter' must be at least 1, but got {maxiter=}")

        if not 1 <= k <= 5:
            raise ValueError(f"Degree of a spline 'k' must be in the range [1, 5], but got {k=}")

        best_bic = np.inf
        best_spline = None

        y_diffs = diffs(self._y, math.sqrt(2))

        if not y_diffs.size:
            log.warning("No y-values provided. Return NaN values array with the same shape")
            return [np.array([np.nan] * len(y)) for y in self._y]

        sigma = median_abs_deviation(y_diffs, scale="normal")

        for iter_num, spacing in enumerate(knots_spacing):
            num_free_params = 0
            num_points = 0
            ssr = 0
            light_curve_mask = []
            spline = []
            bad_knots = False

            for segment_num, (x, y) in enumerate(zip(self._x, self._y)):
                log = log.bind(iter_i=iter_num, segment_i=segment_num)

                if not x.size:
                    log.warning("Time segment is empty. Skip this segment")
                    spline.append(np.array([]))
                    continue

                x_min = x.min()
                x_max = x.max()
                current_knots = np.arange(x_min + spacing, x_max, spacing)

                try:
                    spline_piece, mask = self._fit_segment(
                        x, y, knots=current_knots, k=k, maxiter=maxiter, sigma_cut=sigma_cut
                    )
                except InsufficientPointsError as ex:
                    # InsufficientPointsError raised when after removing outliers there are less
                    #  poinst than neccesery to fit
                    log.warning("Cannot fit time segment", ex_msg=ex)
                    spline.append(np.array([np.nan] * len(y)))
                    light_curve_mask.append(np.zeros_like(y, dtype=bool))
                    continue
                except SplineError:
                    # Current knots spacing led to the internal spline error. Skip this spacing
                    log.warning("Knots spacing led to the internal spline error", spacing=spacing)
                    bad_knots = True
                    break

                spline.append(spline_piece)
                light_curve_mask.append(mask)

                # Number of free parameters: number of knots + degree of spline - 1
                num_free_params += len(current_knots) + k - 1

                # Accumulate the number of points and the squared residuals.
                num_points += np.count_nonzero(mask)
                ssr += np.sum((y[mask] - spline_piece[mask]) ** 2)

            log = log.try_unbind("segment_i")
            if bad_knots:
                log.debug("Skipping current knots spacing", spacing=spacing)
                continue

            current_bic = bic(
                k=num_free_params, n=num_points, sigma=sigma, ssr=ssr, penalty_coeff=penalty_coeff
            )

            if current_bic < best_bic or best_spline is None:
                best_bic = current_bic
                best_spline = spline
                log.debug("Found new best-fit spline", bic=best_bic)

        if best_spline is None:
            # If cannot fit the spline return NaNs for all time segments
            log.error("Cannot fit spline. Return NaN values for all time segments")
            best_spline = [np.array([np.nan] * y.size) for y in self._y]

        return best_spline

    def _fit_segment(
        self,
        x: np.ndarray,
        y: np.ndarray,
        knots: np.ndarray,
        k: int = 3,
        maxiter: int = 5,
        sigma_cut: float = 3.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit 1D segment of time series values."""
        x_len = len(x)
        if x_len <= k:
            raise InsufficientPointsError(available_points=x_len, num_min_points=k + 1)

        # Values of the best fitting spline evaluated at the time segment.
        spline = None

        # Mask indicating the points used to fit the spline.
        mask = None

        for _ in range(maxiter):
            if spline is None:
                mask = np.ones_like(x, dtype=bool)  # Try to fit all points.
            else:
                # Choose points where the absolute deviation from the median residual is less than
                # outlier_cut*sigma, where sigma is a robust estimate of the standard deviation of
                # the residuals from the previous spline.
                residuals = y - spline
                _, _, new_mask = robust_mean(residuals, sigma_cut=sigma_cut)

                if np.array_equal(new_mask, mask):
                    break  # Spline converged.

                mask = new_mask

            available_points = np.count_nonzero(mask)
            if available_points <= k:
                raise InsufficientPointsError(available_points, num_min_points=k + 1)

            try:
                spline = LSQUnivariateSpline(x[mask], y[mask], k=k, t=knots)(x)
            except ValueError as ex:
                # Occasionally, knot spacing leads to the choice of incorrect knots.
                # Raise SplainError and then skip current knots space.
                msg = f"Specified knots led to the internal spline error. {ex}"
                raise SplineError(msg) from ex

        return spline, mask
