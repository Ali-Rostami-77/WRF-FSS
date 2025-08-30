import os
import glob
import xarray as xr
import pandas as pd

def process_era5_6h(input_dir, output_dir, time_dim="valid_time"):
    """
    Read hourly ERA5 files (having variable `total_precip` and coordinate `valid_time`),
    compute 6-hour sums (labelled by the LAST time in each 6h window), and write
    one NetCDF per 6-hour bin. The output variable is named 'tp-6h'.
    """
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {input_dir!r}")

    # Open all files (combined by coordinates)
    ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)
    if "total_precip" not in ds.variables:
        raise KeyError("Input files must contain variable 'total_precip'")

    da = ds["total_precip"]

    # If there's a singleton 'number' (ensemble) dimension, drop it
    if "number" in da.dims and da.sizes["number"] == 1:
        da = da.isel(number=0)

    # Make sure the time dimension exists and is named as expected
    if time_dim not in da.dims:
        # try to rename 'time' -> 'valid_time' if that matches your data layout
        if "time" in da.dims:
            da = da.rename({"time": time_dim})
        else:
            raise KeyError(f"Could not find time dimension '{time_dim}' or 'time' in data variables.")

    # Detect latitude / longitude names (prefer 'latitude'/'longitude')
    lat_name = None
    lon_name = None
    for candidate in ("latitude", "lat"):
        if candidate in da.dims or candidate in ds.coords or candidate in ds.variables:
            lat_name = candidate
            break
    for candidate in ("longitude", "lon", "long"):
        if candidate in da.dims or candidate in ds.coords or candidate in ds.variables:
            lon_name = candidate
            break
    if lat_name is None or lon_name is None:
        raise KeyError("Couldn't detect latitude/longitude coordinate names (looked for 'latitude/lat' and 'longitude/lon').")

    # --- Resample to 6-hour sums ---
    # label='right' makes the resampled timestamp equal to the end of the 6-hour bin
    da6h = da.resample(**{time_dim: "6H"}, label="right", closed="right").sum()

    print(f"Input timesteps: {da.sizes[time_dim]}")
    print(f"Output timesteps (6-hour accumulations): {da6h.sizes[time_dim]}")

    # Write each 6-hour step into a separate NetCDF file (single valid_time per file)
    for i, t in enumerate(da6h[time_dim].values):
        # select the 2D (lat,lon) slice for this 6-hour bin, then re-add a length-1 time dim
        da_2d = da6h.isel({time_dim: i})                 # dims -> (latitude, longitude)
        da_1t = da_2d.expand_dims({time_dim: [t]})       # dims -> (valid_time=1, latitude, longitude)

        # name the variable 'tp-6h' as requested
        da_1t.name = "tp-6h"
        ds_out = da_1t.to_dataset()

        # ensure coords are copied and named properly (they should be already)
        ds_out = ds_out.assign_coords({
            lat_name: ds[lat_name].values,
            lon_name: ds[lon_name].values,
            time_dim: [pd.Timestamp(t)]
        })

        # Add useful metadata
        ds_out["tp-6h"].attrs["long_name"] = "6-hour total precipitation"
        # preserve unit if present in original
        orig_units = ds["total_precip"].attrs.get("units") if "total_precip" in ds else None
        if orig_units:
            ds_out["tp-6h"].attrs["units"] = orig_units
        ds_out.attrs["description"] = "ERA5 6-hour accumulated total_precip (summed from hourly files)"

        # Generate filename using the end-time of the 6h window (YYYYMMDD_HHMM)
        timestamp = pd.Timestamp(t).strftime("%Y%m%d_%H%M")
        out_name = f"ERA5_tp6h_{timestamp}.nc"
        out_path = os.path.join(output_dir, out_name)

        ds_out.to_netcdf(out_path)
        print(f"Written: {out_path}")

    ds.close()
    print("✅ Finished: Each output file contains exactly ONE timestep (valid_time).")


if __name__ == "__main__":
    input_directory = r"E:/case/"
    output_directory = r"E:/case/"
    process_era5_6h(input_directory, output_directory)
