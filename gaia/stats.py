"""Functions for computing statistics."""

from typing import SupportsFloat

import numpy as np


def _compensate_robust_mean(
    y: np.ndarray, absdev: np.ndarray, sigma: float, cut: float
) -> tuple[float, np.ndarray]:
    """Compensate the estimate of sigma due to trimming away outliers."""
    # Identify outliers using estimate of the standard deviation of y.
    mask = absdev <= cut * sigma

    # Recompute the standard deviation, using the sample standard deviation of non-outlier points.
    sigma = np.std(y[mask])

    # Compensate the estimate of sigma due to trimming away outliers. The following formula is an
    # approximation, see http://w.astro.berkeley.edu/~johnjohn/idlprocs/robust_mean.pro.
    sc = np.max([cut, 1.0])
    if sc <= 4.5:
        sigma /= -0.15405 + 0.90723 * sc - 0.23584 * sc**2 + 0.020142 * sc**3

    return sigma, mask


def robust_mean(y: np.ndarray, sigma_cut: float) -> tuple[float, float, np.ndarray]:
    """Computes a robust mean estimate in the presence of outliers.

    Parameters
    ----------
    y : np.ndarray
        1D numpy array. Assumed to be normally distributed with outliers
    sigma_cut : float
        Points more than this number of standard deviations from the median are ignored

    Returns
    -------
    tuple[float, float, np.ndarray]
        A robust estimate of the mean of y.
        The standard deviation of the mean.
        Boolean array with the same length as y. Values corresponding to outliers in y are False.
        All other values are True.
    """

    # Makke a robust estimate of the standard deviation of y, assuming y is normally distributed.
    # The conversion factor of 0.67449 takes the median absolute deviation (MAD) to the standard
    # deviation of a normal distribution. See the link belowe for mor info.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html?highlight=scipy%20stats%20norm
    absdev = np.abs(y - np.median(y))
    sigma = np.median(absdev) / 0.67449

    # If the previous estimate of the standard deviation using the MAD is zero, fall back to a
    # robust estimate using the mean absolute deviation. This estimator has a different conversion
    # factor of 1.253. See, e.g. https://www.mathworks.com/help/stats/mad.html.
    if np.isclose(sigma, 0.0):
        sigma = 1.253 * np.mean(absdev)

    sigma, _ = _compensate_robust_mean(y, absdev, sigma, sigma_cut)
    sigma, mask = _compensate_robust_mean(y, absdev, sigma, sigma_cut)

    # Final estimate is the sample mean with outliers removed.
    mean = np.mean(y[mask])
    mean_stddev = sigma / np.sqrt(len(y) - 1.0)

    return mean, mean_stddev, mask


def diffs(y: np.ndarray, scale_coeff: float) -> np.ndarray:
    """Compute diffs of y-values.

    Parameters
    ----------
    y : np.ndarray
        Values for which diffs will be compute
    scale_coeff : float
        Scaling coefficient

    Returns
    -------
    np.ndarray
        Diffs of y-values
    """
    # Compute the assumed standard deviation of Gaussian white noise about the spline model.
    # Assume that each flux value f[i] is a Gaussian random variable f[i] ~ N(s[i], sigma^2),
    # where s is the value of the true spline model and sigma is the constant standard deviation
    # for all flux values.Moreover, we assume that s[i] ~= s[i+1].
    # Therefore, (f[i+1] - f[i]) / sqrt(2) ~ N(0, sigma^2).
    scaled_diffs = [np.diff(yi) / scale_coeff for yi in y]
    return np.concatenate(scaled_diffs) if scaled_diffs else np.array([])


def bic(k: int, n: int, sigma: float, ssr: float, penalty_coeff: float) -> float:
    """Calculate Bayesian information criterion (BIC).

    Models with lower BIC are usually preferred.

    Parameters
    ----------
    k : int
        Numbers of free parameters in a model
    n : int
        Sample size, the number of elements in the sample
    sigma : float
        Standard deviation
    ssr : float
        The sum of the squares of residuals
    penalty_coeff : float
        Penalty factor for the number of parameters. The lower the coefficient, the better the fit
        and the greater the chance of overfitting the model

    Returns
    -------
    float
        Bayesian information criterion for the given parameters
    """
    # The following term is -2*ln(L), where L is the likelihood of the data given the model, under
    # the assumption that the model errors are iid Gaussian with mean 0 and std dev `sigma`.
    likelihood = n * np.log(2 * np.pi * sigma**2) + ssr / sigma**2
    penalty = k * np.log(n)
    return likelihood + penalty_coeff * penalty


def euclidian_distance(x: SupportsFloat, y: SupportsFloat) -> SupportsFloat:
    """Compute the Euclidean distance between two points or two sequences of points

    Euclidean distance is defined as follow: `dist = sqrt(x^2 + y^2)`

    Parameters
    ----------
    x : SupportsFloat
        One point or a sequence of points
    y : SupportsFloat
        Second point or a sequence of points

    Returns
    -------
    SupportsFloat
        Euclidean distance between `x` and `y`. This is a float or a numpy array of float values
    """
    return np.sqrt(x * x + y * y)
