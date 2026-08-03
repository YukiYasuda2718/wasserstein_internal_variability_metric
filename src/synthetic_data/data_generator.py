from logging import getLogger

import numpy as np
import scipy

logger = getLogger()


def generate_correlated_gaussian(
    *,
    n_ensembles: int,
    data_length: int,
    sigma: float,
    correlation: float,
    seed: int = 42,
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

    return mvn * sigma


def generate_correlated_uniform(
    *,
    n_ensembles: int,
    data_length: int,
    low: float,
    high: float,
    correlation: float,
    seed: int = 42,
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

    uniforms = scipy.stats.norm.cdf(mvn)
    # value range is [0, 1]

    return uniforms * (high - low) + low


def generate_lognormal_corr(N, dim, rho_Y, mu, sigma, seed, **kwargs):
    rng = np.random.default_rng(seed)

    rho_X = (1.0 / (sigma**2)) * np.log(rho_Y * (np.exp(sigma**2) - 1.0) + 1.0)

    R_X = np.full((dim, dim), rho_X)
    np.fill_diagonal(R_X, 1.0)

    mean_vec = np.full(dim, mu)
    cov_mat = sigma**2 * R_X

    X = rng.multivariate_normal(mean=mean_vec, cov=cov_mat, size=N)

    Y = np.exp(X)

    return Y
