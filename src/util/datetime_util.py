import cftime
import xarray as xr


def get_target_datetimes(lst_dt: list[cftime.DatetimeNoLeap], year: int, month: int):
    target_dt = []
    for _dt in lst_dt:
        dt = _dt.values.item()
        if dt.year == year and dt.month == month:
            target_dt.append(dt)
    return target_dt


def save_time_averaged_data(
    ds: xr.DataArray, target_dt: list[cftime.DatetimeNoLeap], file_path: str
):
    ds.sel(time=target_dt).mean(dim="time").to_netcdf(file_path)
