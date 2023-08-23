import numpy as np

from gaia.data.models import IterableOfSeries, Series


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

    Models with lower BIC are usually preferred.

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
