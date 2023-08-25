import numpy as np

from gaia.data.models import BooleanArray, IterableOfSeries, Series


def _compensate_robust_mean(
    y: Series,
    absdev: Series,
    sigma_cut: float,
    cut: float,
) -> tuple[float, BooleanArray]:
    """Compensate the estimate of standard deviation due to trimming away outliers."""
    # Identify outliers using estimate of the standard deviation of y.
    mask = absdev <= cut * sigma_cut

    # Recompute the standard deviation, using the sample standard deviation of non-outlier points.
    sigma_cut = np.std(y[mask])

    # Compensate the estimate of sigma due to trimming away outliers. The following formula is an
    # approximation, see http://w.astro.berkeley.edu/~johnjohn/idlprocs/robust_mean.pro.
    max_sigma_cut = np.max([cut, 1.0])
    if max_sigma_cut <= 4.5:
        sigma_cut /= (
            -0.15405
            + 0.90723 * max_sigma_cut
            - 0.23584 * max_sigma_cut**2
            + 0.020142 * max_sigma_cut**3
        )
    return sigma_cut, mask


def robust_mean(y: Series, sigma_cut: float) -> tuple[float, float, BooleanArray]:
    """Computes a robust mean estimate in the presence of outliers.

    Args:
        y (Series): Values for which mean will be compute. Assumed to be normally distributed with
            outliers
        sigma_cut (float): Points more than this number of standard deviations from the median are
            ignored

    Returns:
        tuple[float, float, BooleanArray]: A tuple of:
          - A robust estimate of the mean of y.
          - The standard deviation of the mean.
          - A mask: values corresponding to outliers in y are `False`. All other values are `True`.
    """
    # Make a robust estimate of the standard deviation of y, assuming y is normally distributed.
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


def diffs(y: IterableOfSeries, scale_coeff: float = 1.0) -> Series:
    """Compute diffs of y-values.

    Args:
        y (Series): Values for which diffs will be compute
        scale_coeff (float): Scaling coefficient. Defaults to 1.0

    Returns:
        Series: Diffs of y-values
    """
    # Compute the assumed standard deviation of Gaussian white noise about the spline model.
    # Assume that each flux value f[i] is a Gaussian random variable f[i] ~ N(s[i], sigma^2),
    # where s is the value of the true spline model and sigma is the constant standard deviation
    # for all flux values. Moreover, we assume that s[i] ~= s[i+1].
    # Therefore, (f[i+1] - f[i]) / sqrt(2) ~ N(0, sigma^2).
    if scale_coeff == 0:
        raise ValueError("'scale_coeff' cannot be 0")

    scaled_diffs = [np.diff(yi) / scale_coeff for yi in y]
    return np.concatenate(scaled_diffs) if scaled_diffs else np.array([])


def bic(k: int, n: int, sigma: float, ssr: float, penalty_coeff: float = 1.0) -> float:
    """Calculate Bayesian information criterion (BIC).

    Args:
        k (int): Number of free parameters in a model. Must be >= 0
        n (int): Sample size, the number of elements in the sample. Must be > 0
        sigma (float): Standard deviation. Must be >= 0
        ssr (float): The sum of the squares of residuals. Must be >= 0
        penalty_coeff (float): Penalty factor for the number of parameters. The lower the
            coefficient, the better the fit and the greater the chance of overfitting the model.
            Defaults to 1.0.

    Returns:
        float: Bayesian information criterion (BIC) for the given parameters
    """
    if sigma < 0:
        raise ValueError("Standard deviation 'sigma' cannot be < 0")
    if n <= 0:
        raise ValueError("Sample size 'n' cannot be <= 0")
    if ssr < 0:
        raise ValueError("Sum of the squares of residuals 'ssr' cannot be < 0")
    if k < 0:
        raise ValueError("Number of free parameters 'k' cannot be < 0")
    # The following term is -2*ln(L), where L is the likelihood of the data given the model, under
    # the assumption that the model errors are iid Gaussian with mean 0 and std dev `sigma`.
    normalized_sigma = sigma + 1e-6
    likelihood = n * np.log(2 * np.pi * normalized_sigma**2) + ssr / normalized_sigma**2
    penalty = k * np.log(n)
    return likelihood + penalty_coeff * penalty  # type: ignore
