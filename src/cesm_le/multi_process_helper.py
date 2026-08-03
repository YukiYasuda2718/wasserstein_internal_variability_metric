import copy
import sys
from itertools import product
from multiprocessing import Process, Queue, cpu_count
from typing import Callable, Literal, Tuple

import numpy as np
import xarray as xr

from src.metrics.metrics import (
    calc_gamma_std,
    calc_gamma_w_using_emd2_1d,
    calc_gamma_w_using_sort,
    calc_gamma_with_computing_aves_inside,
)
from src.metrics.sane2024 import optbins, infofromy


if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def _worker(job_queue, result_queue):
    while True:
        try:
            job = job_queue.get()
            if job is None:  # termination signal
                break

            ilon, ilat, samples, max_nbins, ave_nbins, use_logscale = job
            assert samples.ndim == 2  # ensemble and time
            n_ens, n_times = samples.shape

            g_std = calc_gamma_std(samples)
            g_w = calc_gamma_w_using_emd2_1d(samples, p=1)
            g_w_sort = calc_gamma_w_using_sort(samples, p=1, denom="int+ext")
            g_w_sort_tot = calc_gamma_w_using_sort(samples, p=1, denom="tot")
            g_sane, nbins = calc_gamma_with_computing_aves_inside(
                samples,
                return_nbins=True,
                max_nbins=max_nbins,
                use_logscale=use_logscale,
            )
            g_sane_with_ave_bin = calc_gamma_with_computing_aves_inside(
                samples, n_bins=ave_nbins
            )

            _samps = samples.flatten()
            n_bins = optbins(data=_samps, maxM=500)
            range_x = np.max(_samps) - np.min(_samps)
            g_sane_original, _ = infofromy(
                y=samples, M=n_ens, range_x=range_x, bins=n_bins
            )

            result_queue.put(
                (
                    ilon,
                    ilat,
                    g_std,
                    g_w,
                    g_w_sort,
                    g_sane,
                    nbins,
                    g_sane_with_ave_bin,
                    g_sane_original,
                    g_w_sort_tot,
                )
            )

        except Exception as e:
            print(f"Error in worker process: {e}")
            # Must match the 9-element normal payload above, otherwise the
            # collector fails to unpack and aborts the whole run.
            result_queue.put((None, None, None, None, None, None, None, None, None))


def parallel_process_dataset_with_process_worker(
    ds: xr.Dataset,
    ens_start: int,
    ens_end: int,
    var_name: str,
    max_nbins: int,
    ave_nbins: int,
    use_logscale: bool,
    n_processes=None,
    remove_annual_cycle: Literal[
        "remove_climatology", "remove_running_mean", "remove_annual_harmonic", "none"
    ] = "none",
):
    if n_processes is None:
        n_processes = cpu_count()

    print("Reading data.")
    assert ds[var_name].dims == ("member_id", "time", "lat", "lon")

    if var_name == "TREFHT":
        all_data = (ds[var_name] - 273.15).data
    else:
        all_data = (ds[var_name]).data
    nlon, nlat, ntime, nens = (
        ds.sizes["lon"],
        ds.sizes["lat"],
        ds.sizes["time"],
        ds.sizes["member_id"],
    )

    print("Prepare calculation")
    # Initialize with NaN so that grid points that fail in a worker remain
    # clearly missing (NaN) instead of a misleading 0. nbins is integer, so it
    # uses -1 as its missing sentinel.
    g_std = np.full(shape=(nlon, nlat), fill_value=np.nan, dtype=np.float32)
    g_w = np.full(shape=(nlon, nlat), fill_value=np.nan, dtype=np.float32)
    g_w_sort = np.full(shape=(nlon, nlat), fill_value=np.nan, dtype=np.float32)
    g_sane = np.full(shape=(nlon, nlat), fill_value=np.nan, dtype=np.float32)
    g_sane_original = np.full(shape=(nlon, nlat), fill_value=np.nan, dtype=np.float32)
    nbins = np.full(shape=(nlon, nlat), fill_value=-1, dtype=np.int32)
    g_sane_ave_nbin = np.full(shape=(nlon, nlat), fill_value=np.nan, dtype=np.float32)
    g_w_sort_tot = np.full(shape=(nlon, nlat), fill_value=np.nan, dtype=np.float32)

    job_queue = Queue()
    result_queue = Queue()

    processes = []
    for _ in range(n_processes):
        p = Process(target=_worker, args=(job_queue, result_queue))
        p.start()
        processes.append(p)

    print(f"All data shape = {all_data.shape}")

    assert all_data.shape == (nens, ntime, nlat, nlon)

    if remove_annual_cycle == "remove_climatology":
        print("Removing monthly climatology (deseasonalizing)")
        # Subtract the monthly climatology to remove the seasonal cycle.
        # The climatology is averaged over both ensemble members and years for
        # each calendar month, so it is shared across all members. As a result
        # it removes the seasonal cycle only from the forced (ensemble-mean)
        # component and leaves the internal residual (member minus ensemble mean)
        # unchanged for any member subset. Only a per-calendar-month constant is
        # removed, so the forced trend is retained.
        assert (
            ntime % 12 == 0
        ), f"Monthly data covering whole years is assumed: {ntime=}"
        n_year = ntime // 12
        reshaped = all_data.reshape(nens, n_year, 12, nlat, nlon)
        monthly_clim = reshaped.mean(
            axis=(0, 1), keepdims=True
        )  # (1, 1, 12, nlat, nlon)
        all_data = (reshaped - monthly_clim).reshape(nens, ntime, nlat, nlon)
    elif remove_annual_cycle == "remove_running_mean":
        print(
            "Removing annual cycle via a 13-month centered running mean (per member, low-pass filter)"
        )
        # The 13-month centered running mean is a low-pass filter: it averages
        # away the annual cycle (period ~12 months) per member, so the
        # running mean itself (not the raw data minus it) is the
        # deseasonalized output. Subtracting the running mean from the raw
        # data would instead isolate the annual cycle and any sub-annual
        # noise, which is the opposite of removing it.
        # As a side effect of the low-pass filtering, sub-annual internal
        # variability is smoothed away too, unlike the climatology option.
        # The 13-month centered window has no valid mean for the first/last
        # 6 months, so those time steps are dropped (ntime shrinks by 12).
        kernel = np.ones(13) / 13.0
        all_data[:, 6:-6, ...] = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="valid"),
            axis=1,
            arr=copy.deepcopy(all_data),
        )
    elif remove_annual_cycle == "remove_annual_harmonic":
        print(
            "Removing annual cycle by notching the fundamental annual frequency "
            "(1 cycle/year) via FFT (per member), after linear detrending"
        )
        # Applied independently per member (not shared across the ensemble,
        # unlike remove_climatology). Only the fundamental annual harmonic is
        # removed; higher harmonics of the (possibly non-sinusoidal) seasonal
        # shape are left untouched, and so is every other frequency
        # (interannual variability, sub-annual internal noise). This avoids
        # the over-smoothing side effect of remove_running_mean, at the cost
        # of leaving any non-sinusoidal part of the seasonal cycle in place.
        # With ntime % 12 == 0, the annual frequency (1/12 cycles/month)
        # falls exactly on rfft bin index n_year = ntime // 12.
        # The DFT implicitly assumes the record is periodic over its full
        # length, so a mismatch between the start and end values (e.g. from
        # a trend) leaks energy into every bin, including the annual one.
        # Removing the per-member, per-grid-point linear trend beforehand
        # (least squares) and adding it back afterwards greatly reduces this
        # leakage; only nonlinear low-frequency variability can still leak.
        assert (
            ntime % 12 == 0
        ), f"Monthly data covering whole years is assumed: {ntime=}"
        n_year = ntime // 12

        t = np.arange(ntime, dtype=all_data.dtype).reshape(1, ntime, 1, 1)
        t_mean = t.mean()
        y_mean = all_data.mean(axis=1, keepdims=True)  # (nens, 1, nlat, nlon)
        slope = ((t - t_mean) * (all_data - y_mean)).sum(axis=1, keepdims=True) / (
            (t - t_mean) ** 2
        ).sum()  # (nens, 1, nlat, nlon)
        intercept = y_mean - slope * t_mean  # (nens, 1, nlat, nlon)
        trend = intercept + slope * t  # (nens, ntime, nlat, nlon)

        spectrum = np.fft.rfft(all_data - trend, axis=1)
        spectrum[:, n_year, ...] = 0.0
        all_data = np.fft.irfft(spectrum, n=ntime, axis=1) + trend
    elif remove_annual_cycle == "none":
        pass
    else:
        raise ValueError(f"Unknown {remove_annual_cycle=}")

    total_jobs = nlon * nlat
    for ilon, ilat in tqdm(
        product(range(nlon), range(nlat)), desc="Submitting jobs", total=total_jobs
    ):
        samples = all_data[ens_start:ens_end, :, ilat, ilon]
        job_queue.put((ilon, ilat, samples, max_nbins, ave_nbins, use_logscale))

    for _ in range(n_processes):
        job_queue.put(None)  # add the signal of termination.

    for _ in tqdm(range(total_jobs), desc="Collecting results"):
        (
            ilon,
            ilat,
            g_std_val,
            g_w_val,
            g_w_sort_val,
            g_sane_val,
            nbins_val,
            g_sane_ave_nbin_val,
            g_sane_original_val,
            g_w_sort_tot_val,
        ) = result_queue.get()
        if ilon is not None:
            g_std[ilon, ilat] = g_std_val
            g_w[ilon, ilat] = g_w_val
            g_w_sort[ilon, ilat] = g_w_sort_val
            g_sane[ilon, ilat] = g_sane_val
            g_sane_original[ilon, ilat] = g_sane_original_val
            nbins[ilon, ilat] = nbins_val
            g_sane_ave_nbin[ilon, ilat] = g_sane_ave_nbin_val
            g_w_sort_tot[ilon, ilat] = g_w_sort_tot_val

    for p in processes:
        p.join()

    remaining_results = []
    while not result_queue.empty():
        remaining_results.append(result_queue.get())

    if remaining_results:
        print(
            f"Warning: Found {len(remaining_results)} unprocessed results in the queue!"
        )
        for (
            ilon,
            ilat,
            g_std_val,
            g_w_val,
            g_w_sort_val,
            g_sane_val,
            nbins_val,
            g_sane_ave_nbin_val,
            g_sane_original_val,
            g_w_sort_tot_val,
        ) in remaining_results:
            if ilon is not None:
                g_std[ilon, ilat] = g_std_val
                g_w[ilon, ilat] = g_w_val
                g_w_sort[ilon, ilat] = g_w_sort_val
                g_sane[ilon, ilat] = g_sane_val
                g_sane_original[ilon, ilat] = g_sane_original_val
                nbins[ilon, ilat] = nbins_val
                g_sane_ave_nbin[ilon, ilat] = g_sane_ave_nbin_val
                g_w_sort_tot[ilon, ilat] = g_w_sort_tot_val

    return (
        g_std,
        g_w,
        g_w_sort,
        g_sane,
        nbins,
        g_sane_ave_nbin,
        g_sane_original,
        g_w_sort_tot,
    )
