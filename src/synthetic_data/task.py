import copy
import sys
from logging import getLogger

import numpy as np
from src.metrics.metrics import (
    calc_gamma,
    calc_gamma_std,
    calc_gamma_w_using_emd2_1d,
    calc_gamma_w_using_sort,
)
from src.metrics.sane2024 import infofromy, optbins
from src.metrics.histogram_helper import calc_optimal_bins_knuth19
from src.synthetic_data.data_generator import (
    generate_correlated_gaussian,
    generate_correlated_uniform,
    generate_lognormal_corr,
)

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = getLogger(__name__)


def calc_results_for_normal_dist(
    n_ens: int,
    n_times: int,
    is_outlier: bool,
    max_nbins: int,
    sigma: float = 1.0,
    seed: int = 42,
    phi: float = 0.0,
) -> dict[str, list]:
    #

    results: dict[str, list] = {
        "gamma_std": [],
        "gamma_w1": [],
        "gamma_w2": [],
        "gamma_w3": [],
        "gamma_w4": [],
        "gamma_w5": [],
        "gamma_sane": [],
        "gamma_sane_original": [],
        "gamma_sane_original_with_fixed_params": [],
        "corr": [],
    }

    all_corrs = sorted(
        set(
            list(np.linspace(0.0, 0.009, 10, dtype=np.float64))
            + list(np.linspace(0.01, 0.99, 99, dtype=np.float64))
            + list(np.linspace(0.99, 0.999, 10, dtype=np.float64))
        )
    )

    for corr in all_corrs:
        config = {
            "n_ensembles": n_ens,
            "data_length": n_times,
            "sigma": sigma,
            "correlation": corr,
            "seed": seed,
            "phi": phi,
        }
        results["corr"].append(corr)

        samples = generate_correlated_gaussian(**config)
        assert samples.shape == (n_ens, n_times)

        if is_outlier:
            # Set 50 outliers if n_times == 10,000 following Sane et al. 2024
            samples[:, : int(n_times * 50.0 / 10_000)] = 5.0

        g_std = calc_gamma_std(samples)
        results["gamma_std"].append(g_std)

        g1 = calc_gamma_w_using_emd2_1d(samples, p=1)
        results["gamma_w1"].append(g1)
        g2 = calc_gamma_w_using_emd2_1d(samples, p=2)
        results["gamma_w2"].append(g2)
        g3 = calc_gamma_w_using_sort(samples, p=1)
        results["gamma_w3"].append(g3)
        g4 = calc_gamma_w_using_sort(samples, p=2)
        results["gamma_w4"].append(g4)
        g5 = calc_gamma_w_using_sort(samples, p=1, denom="tot")
        results["gamma_w5"].append(g5)

        _samps = samples.flatten()
        value_range = (np.min(_samps), np.max(_samps))
        n_bins = calc_optimal_bins_knuth19(_samps, value_range, max_nbins=max_nbins)
        averages = np.mean(samples, axis=0)

        if n_bins == max_nbins:
            logger.debug(
                f"Max bins reached (normal): {n_ens=}, {n_times=}, {is_outlier=}, {sigma=}, {max_nbins=}, {seed=}, {corr=},"
            )

        g_sane = calc_gamma(
            _samps,
            averages,
            n_repeats_for_ave=config["n_ensembles"],
            value_range=value_range,
            n_bins=n_bins,
        )
        results["gamma_sane"].append(g_sane)

        _samps = samples.flatten()
        n_bins = optbins(data=_samps, maxM=500)
        range_x = np.max(_samps) - np.min(_samps)
        g_sane_original, _ = infofromy(y=samples, M=n_ens, range_x=range_x, bins=n_bins)
        results["gamma_sane_original"].append(g_sane_original)

        range_x = 10.0
        n_bins = 60  # fixed params from Sane et al. (2024)
        g_sane_original, _ = infofromy(y=samples, M=n_ens, range_x=range_x, bins=n_bins)
        results["gamma_sane_original_with_fixed_params"].append(g_sane_original)

    return results


def calc_results_for_uniform_dist(
    n_ens: int,
    n_times: int,
    is_outlier: bool,
    max_nbins: int,
    high: float = 1.0,
    low: float = -1.0,
    seed: int = 42,
    phi: float = 0.0,
) -> dict[str, list]:
    #

    results: dict[str, list] = {
        "gamma_std": [],
        "gamma_w1": [],
        "gamma_w2": [],
        "gamma_w3": [],
        "gamma_w4": [],
        "gamma_w5": [],
        "gamma_sane": [],
        "gamma_sane_original": [],
        "gamma_sane_original_with_fixed_params": [],
        "corr": [],
    }

    all_corrs = sorted(
        set(
            list(np.linspace(0.0, 0.009, 10, dtype=np.float64))
            + list(np.linspace(0.01, 0.99, 99, dtype=np.float64))
            + list(np.linspace(0.99, 0.999, 10, dtype=np.float64))
        )
    )

    for corr in all_corrs:
        config = {
            "n_ensembles": n_ens,
            "data_length": n_times,
            "low": low,
            "high": high,
            "correlation": corr,
            "seed": seed,
            "phi": phi,
        }
        results["corr"].append(corr)

        samples = generate_correlated_uniform(**config)
        assert samples.shape == (n_ens, n_times)

        if is_outlier:
            # Set 50 outliers if n_times == 10,000 following Sane et al. 2024
            samples[:, : int(n_times * 50.0 / 10_000)] = 1.5

        g_std = calc_gamma_std(samples)
        results["gamma_std"].append(g_std)

        g1 = calc_gamma_w_using_emd2_1d(samples, p=1)
        results["gamma_w1"].append(g1)
        g2 = calc_gamma_w_using_emd2_1d(samples, p=2)
        results["gamma_w2"].append(g2)
        g3 = calc_gamma_w_using_sort(samples, p=1)
        results["gamma_w3"].append(g3)
        g4 = calc_gamma_w_using_sort(samples, p=2)
        results["gamma_w4"].append(g4)
        g5 = calc_gamma_w_using_sort(samples, p=1, denom="tot")
        results["gamma_w5"].append(g5)

        _samps = samples.flatten()
        value_range = (np.min(_samps), np.max(_samps))
        n_bins = calc_optimal_bins_knuth19(_samps, value_range, max_nbins=max_nbins)
        averages = np.mean(samples, axis=0)

        if n_bins == max_nbins:
            logger.debug(
                f"Max bins reached (uniform): {n_ens=}, {n_times=}, {is_outlier=}, {high=}, {low=}, {max_nbins=}, {seed=}, {corr=},"
            )

        g_sane = calc_gamma(
            _samps,
            averages,
            n_repeats_for_ave=config["n_ensembles"],
            value_range=value_range,
            n_bins=n_bins,
        )
        results["gamma_sane"].append(g_sane)

        _samps = samples.flatten()
        n_bins = optbins(data=_samps, maxM=500)
        range_x = np.max(_samps) - np.min(_samps)
        g_sane_original, _ = infofromy(y=samples, M=n_ens, range_x=range_x, bins=n_bins)
        results["gamma_sane_original"].append(g_sane_original)

        range_x = 2.0
        n_bins = 60  # fixed params from Sane et al. (2024)
        g_sane_original, _ = infofromy(y=samples, M=n_ens, range_x=range_x, bins=n_bins)
        results["gamma_sane_original_with_fixed_params"].append(g_sane_original)

    return results


def calc_results_for_lognormal_dist(
    n_ens: int,
    n_times: int,
    is_outlier: bool,
    max_nbins: int,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: int = 42,
    phi: float = 0.0,
) -> dict[str, list]:
    #

    results: dict[str, list] = {
        "gamma_std": [],
        "gamma_w1": [],
        "gamma_w2": [],
        "gamma_w3": [],
        "gamma_w4": [],
        "gamma_w5": [],
        "gamma_sane": [],
        "gamma_sane_original": [],
        "gamma_sane_original_with_fixed_params": [],
        "corr": [],
    }

    all_corrs = sorted(
        set(
            list(np.linspace(0.0, 0.009, 10, dtype=np.float64))
            + list(np.linspace(0.01, 0.99, 99, dtype=np.float64))
            + list(np.linspace(0.99, 0.999, 10, dtype=np.float64))
        )
    )

    for corr in all_corrs:
        config = {
            "dim": n_ens,
            "N": n_times,
            "mu": mu,
            "sigma": sigma,
            "rho_Y": corr,
            "seed": seed,
            "phi": phi,
        }
        results["corr"].append(corr)

        samples = generate_lognormal_corr(**config).transpose()
        assert samples.shape == (n_ens, n_times), f"{samples.shape=}"

        if is_outlier:
            # 16.561 = (1.0 - 25.0 / 10_000)*100 percentile point for log-normal dist.
            samples[:, : int(n_times * 50.0 / 10_000)] = 16.561

        g_std = calc_gamma_std(samples)
        results["gamma_std"].append(g_std)

        g1 = calc_gamma_w_using_emd2_1d(samples, p=1)
        results["gamma_w1"].append(g1)
        g2 = calc_gamma_w_using_emd2_1d(samples, p=2)
        results["gamma_w2"].append(g2)
        g3 = calc_gamma_w_using_sort(samples, p=1)
        results["gamma_w3"].append(g3)
        g4 = calc_gamma_w_using_sort(samples, p=2)
        results["gamma_w4"].append(g4)
        g5 = calc_gamma_w_using_sort(samples, p=1, denom="tot")
        results["gamma_w5"].append(g5)

        _samps = samples.flatten()
        value_range = (np.min(_samps), np.max(_samps))
        n_bins = calc_optimal_bins_knuth19(
            _samps, value_range, max_nbins=max_nbins, use_logscale=False
        )
        averages = np.mean(samples, axis=0)

        if n_bins == max_nbins:
            logger.debug(
                f"Max bins reached (lognormal): {n_ens=}, {n_times=}, {is_outlier=}, {mu=}, {sigma=}, {max_nbins=}, {seed=}, {corr=},"
            )

        g_sane = calc_gamma(
            _samps,
            averages,
            n_repeats_for_ave=config["dim"],
            value_range=value_range,
            n_bins=n_bins,
        )
        results["gamma_sane"].append(g_sane)

        _samps = samples.flatten()
        n_bins = optbins(data=_samps, maxM=500)
        range_x = np.max(_samps) - np.min(_samps)
        g_sane_original, _ = infofromy(y=samples, M=n_ens, range_x=range_x, bins=n_bins)
        results["gamma_sane_original"].append(g_sane_original)

        range_x = 20.0  # this param is somewhat arbitrary, set to include outliers (== 16.561)
        n_bins = 60  # fixed param from Sane et al. (2024)
        g_sane_original, _ = infofromy(y=samples, M=n_ens, range_x=range_x, bins=n_bins)
        results["gamma_sane_original_with_fixed_params"].append(g_sane_original)

    return results
