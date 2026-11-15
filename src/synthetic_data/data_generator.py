import math
from logging import getLogger

import numpy as np
import scipy

logger = getLogger()


def apply_ar1_along_time(eps: np.ndarray, phi: float) -> np.ndarray:
    """Turn i.i.d.-in-time standard Gaussian series into a stationary AR(1).

    z_0 = eps_0, z_t = phi * z_{t-1} + sqrt(1 - phi^2) * eps_t, so each time
    step keeps the marginal distribution and the inter-member correlation of eps.

    Args:
        eps (np.ndarray): zero-mean, unit-variance Gaussian samples. Shape
            (n_members, n_times); axis = member, time. One time step = 1 month.
        phi (float): lag-1 autocorrelation per time step, 0 <= phi < 1.

    Returns:
        np.ndarray: AR(1) samples with the same shape as eps.
    """
    assert eps.ndim == 2  # axis = member, time
    assert 0.0 <= phi < 1.0
    if phi == 0.0:
        return eps
    b = math.sqrt(1.0 - phi**2)
    z = np.empty_like(eps)
    z[:, 0] = eps[:, 0]
    for t in range(1, eps.shape[1]):
        z[:, t] = phi * z[:, t - 1] + b * eps[:, t]
    return z


def generate_correlated_gaussian(
    *,
    n_ensembles: int,
    data_length: int,
    sigma: float,
    correlation: float,
    seed: int = 42,
    phi: float = 0.0,
    **kwargs,
) -> np.ndarray:
    #
    assert n_ensembles > 0 and data_length > 0 and sigma > 0.0
    assert -1.0 < correlation < 1.0

    rng = np.random.default_rng(seed)

    # Diagnoal components (variaces) are equal to 1
    # Non-diagonal components (correlations) are equal to `correlation`
    cor_matrix = np.ones(n_ensembles) * correlation + np.identity(n_ensembles) * (
        1 - correlation
    )
    assert cor_matrix.shape == (n_ensembles, n_ensembles)

    means = np.zeros(n_ensembles)

    mvn = rng.multivariate_normal(mean=means, cov=cor_matrix, size=data_length)
    mvn = mvn.transpose()
    assert mvn.shape == (n_ensembles, data_length)
    # phi == 0 keeps the original draws bitwise.
    if phi != 0.0:
        mvn = apply_ar1_along_time(mvn, phi)

    return mvn * sigma


def generate_correlated_uniform(
    *,
    n_ensembles: int,
    data_length: int,
    low: float,
    high: float,
    correlation: float,
    seed: int = 42,
    phi: float = 0.0,
    **kwargs,
) -> np.ndarray:
    #
    assert n_ensembles > 0 and data_length > 0
    assert low < high
    assert -1.0 < correlation < 1.0

    rng = np.random.default_rng(seed)

    # Diagnoal components (variaces) are equal to 1
    # Non-diagonal components (correlations) are equal to `correlation`
    cor_matrix = np.ones(n_ensembles) * correlation + np.identity(n_ensembles) * (
        1 - correlation
    )
    assert cor_matrix.shape == (n_ensembles, n_ensembles)

    means = np.zeros(n_ensembles)

    mvn = rng.multivariate_normal(mean=means, cov=cor_matrix, size=data_length)
    mvn = mvn.transpose()
    assert mvn.shape == (n_ensembles, data_length)
    # AR(1) in Gaussian space, before the copula, so the uniform marginal is kept.
    if phi != 0.0:
        mvn = apply_ar1_along_time(mvn, phi)

    uniforms = scipy.stats.norm.cdf(mvn)
    # value range is [0, 1]

    return uniforms * (high - low) + low


def generate_lognormal_corr(N, dim, rho_Y, mu, sigma, seed, phi: float = 0.0, **kwargs):
    rng = np.random.default_rng(seed)

    rho_X = (1.0 / (sigma**2)) * np.log(rho_Y * (np.exp(sigma**2) - 1.0) + 1.0)

    R_X = np.full((dim, dim), rho_X)
    np.fill_diagonal(R_X, 1.0)

    mean_vec = np.full(dim, mu)
    cov_mat = sigma**2 * R_X

    X = rng.multivariate_normal(mean=mean_vec, cov=cov_mat, size=N)

    # AR(1) in standardized Gaussian space, so the lognormal marginal and rho_X
    # stay as in the i.i.d. case. Skipped at phi == 0 to keep the draws bitwise.
    if phi != 0.0:
        z = apply_ar1_along_time(((X - mu) / sigma).transpose(), phi).transpose()
        X = mu + sigma * z

    Y = np.exp(X)

    return Y
