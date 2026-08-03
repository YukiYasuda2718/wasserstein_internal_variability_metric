import math
from logging import getLogger

import numpy as np
import ot

from .histogram_helper import calc_optimal_bins_knuth19

logger = getLogger()


def calc_gamma_w_using_emd2_1d(samples: np.ndarray, p: int) -> float:
    assert p in (1, 2)
    assert samples.ndim == 2
    # axis = ensemble, time

    representations = np.mean(samples, axis=0)  # take mean along ensemble
    total_rep = np.mean(representations)

    _samps = samples.flatten()

    w_int = ot.emd2_1d(_samps, representations, metric="minkowski", p=p)

    if p == 1:
        w_ext = np.mean(np.abs(representations - total_rep))
        den = w_int + w_ext
        return np.nan if den == 0 else w_int / den
    elif p == 2:
        w_ext = np.mean((representations - total_rep) ** 2)
        den = w_int + w_ext
        return np.nan if den == 0 else math.sqrt(w_int) / math.sqrt(den)
    else:
        raise ValueError()


def calc_gamma_w_using_sort(
    samples: np.ndarray, p: int, denom: str = "int+ext"
) -> float:
    assert p in (1, 2)
    assert samples.ndim == 2
    assert denom in ["int+ext", "tot"]
    # axis = ensemble, time

    representations = np.mean(samples, axis=0)  # take mean along ensemble
    total_rep = np.mean(representations)

    _samps = samples.flatten()
    _reps = np.tile(representations[None, :], reps=(samples.shape[0], 1)).flatten()
    assert _reps.shape == _samps.shape

    diffs = np.sort(_samps) - np.sort(_reps)

    if p == 1:
        w_int = np.mean(np.abs(diffs))
        w_ext = np.mean(np.abs(representations - total_rep))
        w_tot = np.mean(np.abs(_samps - total_rep))
        den = w_int + w_ext if denom == "int+ext" else w_tot
        return np.nan if den == 0 else w_int / den
    elif p == 2:

        w_int = np.mean((diffs) ** 2)
        w_ext = np.mean((representations - total_rep) ** 2)
        w_tot = np.mean((_samps - total_rep) ** 2)
        den = w_int + w_ext if denom == "int+ext" else w_tot
        return np.nan if den == 0 else math.sqrt(w_int) / math.sqrt(den)
    else:
        raise ValueError()


def calc_gamma_with_computing_aves_inside(
    samples: np.ndarray,
    max_nbins: int = 500,
    return_nbins: bool = False,
    n_bins: int = None,
    use_logscale: bool = False,
) -> float:
    """This is a wrapper of the method "calc_gamma." Unlike calc_gamma, the necessary inputs, such as averages, are calculated inside.

    Args:
        samples (np.ndarray): raw data
        max_nbins (int, optional): maximum number of bins for applying the Knuth 2019 method. Defaults to 200.

    Returns:
        float: informtion theory based metric
    """
    #
    assert isinstance(samples, np.ndarray) and samples.ndim == 2
    # axis = ensemble, time

    n_ens = samples.shape[0]  # num of ensemble members

    averages = np.mean(samples, axis=0)  # average along ens dim
    _samps = samples.flatten()
    value_range = (np.min(_samps), np.max(_samps))

    if n_bins is None:
        n_bins = calc_optimal_bins_knuth19(
            _samps, value_range, max_nbins=max_nbins, use_logscale=use_logscale
        )

    g = calc_gamma(
        samples=_samps,
        averages=averages,
        n_repeats_for_ave=n_ens,
        value_range=value_range,
        n_bins=n_bins,
    )

    if return_nbins:
        return g, n_bins
    else:
        return g


def calc_gamma(
    samples: np.ndarray,
    averages: np.ndarray,
    n_repeats_for_ave: int,
    value_range: tuple[float, float],
    n_bins: int,
) -> float:
    """Calculate information theory based metric proposed by Sane et al. (2024).

    See Eq (8) in Sane et al. (2024), "Internal Versus Forced Variability Metrics for General Circulation Models Using Information Theory"

    Args:
        samples (np.ndarray): raw data
        averages (np.ndarray): ensemble average of raw data
        n_repeats_for_ave (int): this is typically number of ensemble members
        value_range (tuple[float, float]): value range for samples and averages
        n_bins (int): number of bins for making histograms of samples and averages

    Returns:
        float: gamma. information theory based metric
    """
    #
    assert isinstance(samples, np.ndarray) and samples.ndim == 1
    assert isinstance(averages, np.ndarray) and averages.ndim == 1
    assert value_range[0] < value_range[1]
    assert n_bins > 0 and isinstance(n_bins, int)

    _aves = np.tile(averages[None, :], reps=(n_repeats_for_ave, 1)).flatten()
    value_2d_range = [list(value_range), list(value_range)]

    assert _aves.shape == samples.shape

    probs_xy = np.histogram2d(
        x=samples, y=_aves, bins=[n_bins, n_bins], range=value_2d_range, density=False
    )[0]

    probs_xy = probs_xy / np.sum(probs_xy)

    # Marginalized along axis=1 (along y axis), then add axis=1 for broadcasting
    probs_x = np.sum(probs_xy, axis=1)[:, None]

    # Marginalized along axis=0 (along x axis), then add axis=0 for broadcasting
    probs_y = np.sum(probs_xy, axis=0)[None, :]

    mask_xy = probs_xy > 0.0
    mask_x = probs_x > 0.0

    mutual_info = np.sum(
        probs_xy[mask_xy] * np.log2(probs_xy[mask_xy] / (probs_x * probs_y)[mask_xy])
    )
    entropy = -1.0 * np.sum(probs_x[mask_x] * np.log2(probs_x[mask_x]))

    assert (
        mutual_info >= 0.0 and entropy >= 0.0
    ), f"{mutual_info=}, {entropy=}, {n_bins=}, {value_range=}"

    gamma = 1.0 - mutual_info / entropy

    return gamma


def calc_gamma_std(samples: np.ndarray) -> float:
    """Calculate variace based metric in Sane et al. (2024).

    See Eq (8) in Sane et al. (2024), "Internal Versus Forced Variability Metrics for General Circulation Models Using Information Theory"

    Args:
        samples (np.ndarray): 2D raw data. Axes are (ensemble, time).

    Returns:
        float: gamma_std. variance based metric.
    """
    #
    assert isinstance(samples, np.ndarray) and samples.ndim == 2
    # axis = ensemble, time

    ens_means = np.mean(samples, axis=0, keepdims=True)
    diffs_int = samples - ens_means
    sigma2_int = np.mean(diffs_int**2)

    means = np.mean(ens_means, axis=1, keepdims=True)  # time mean
    diffs_ext = ens_means - means
    sigma2_ext = np.mean(diffs_ext**2)

    gamma = np.sqrt(sigma2_int) / np.sqrt(sigma2_int + sigma2_ext)

    return gamma
