from logging import getLogger

import numpy as np
import ot
import scipy.special

logger = getLogger()


def calc_optimal_bins_knuth19(
    data: np.ndarray,
    value_range: tuple[float, float],
    max_nbins: int,
    use_logscale: bool = False,
) -> int:
    #
    assert isinstance(data, np.ndarray), "Not np array."
    assert data.ndim == 1, "Not 1D array."
    assert value_range[0] < value_range[1]
    assert max_nbins >= 3 and isinstance(max_nbins, int)

    n_data = len(data)

    if use_logscale:
        raise NotImplementedError()
    else:
        list_nbins = np.arange(2, max_nbins)

    log_ps = np.full(shape=(len(list_nbins),), fill_value=np.nan)

    for i, n_bins in enumerate(list_nbins):
        freqs = np.histogram(
            data,
            bins=n_bins,
            range=value_range,
        )[0]

        # See Eq (32) in Knuth (2019), "Optimal data-based binning for histograms and histogram-based probability density models"

        part1 = n_data * np.log(n_bins) + scipy.special.gammaln(n_bins / 2.0)
        part2 = -n_bins * scipy.special.gammaln(0.5) - scipy.special.gammaln(
            n_data + n_bins / 2.0
        )
        part3 = np.sum(scipy.special.gammaln(freqs + 0.5))
        log_ps[i] = part1 + part2 + part3

    ret = int(list_nbins[int(np.nanargmax(log_ps))])

    return ret
