# Another gridded data suitable for validation is ERA5 dataset.
# This code converst convective and large-scale rain rate to mm/hr,
# then aggregate them into hourly rain rates files 

import xarray as xr
import pandas as pd
import os

#  Input and Output Paths 
input_path = "path to NetCDF files/rates.nc"
output_dir = "path to desired directory/hourly_split/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

#  Load the dataset 
ds = xr.open_dataset(input_path)

# Convert units from kg/m²/s to mm/hr 
crr_mmhr = ds['crr'] * 3600  # Convective rain rate
lsrr_mmhr = ds['lsrr'] * 3600  # Large-scale rain rate

# Combine them into total precipitation rate (with new variable total_precip)
total_precip = crr_mmhr + lsrr_mmhr
total_precip.name = "total_precip"  # Name for the variable

#  Create a new dataset with all three 
ds_converted = xr.Dataset({
    "crr_mmhr": crr_mmhr,
    "lsrr_mmhr": lsrr_mmhr,
    "total_precip": total_precip
})

# === Loop through each timestep and save as hourly NetCDF ===
for i in range(ds.dims['valid_time']):
    hourly_ds = ds_converted.isel(valid_time=i)

    # Extract timestamp
    timestamp = pd.to_datetime(hourly_ds.valid_time.values)
    time_str = timestamp.strftime('%Y%m%d_%H%M')  # e.g., 20230131_0600

    # Create output filename
    output_filename = f"ERA5_precip_{time_str}.nc"
    output_path = os.path.join(output_dir, output_filename)

    # Add single time value to make sure it has a time dimension
    hourly_ds = hourly_ds.expand_dims('valid_time')

    # Save to NetCDF
    hourly_ds.to_netcdf(output_path)

    print(f"Saved: {output_path}")
