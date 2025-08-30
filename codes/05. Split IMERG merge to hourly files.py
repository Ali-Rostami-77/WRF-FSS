# For precipitation validation the need of a gridded dataset is necessary
# Since IMERG GPM V07 data is in 30 minutes intervals, this code makes new hourly rain rate files.

import xarray as xr
from pathlib import Path
import pandas as pd

# Input and output paths
input_path = "Path to the directory of GPM files"
output_dir = "Path to output file"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Open the NetCDF file
ds = xr.open_dataset(input_path)

# Here one can skip the number of NetCDF files in hour in order to matches the spin-up time.
start_time = pd.to_datetime(ds.time[0].values)
end_skip_time = start_time + pd.Timedelta(hours=0)

# Select all data after the DESIRED hours
ds_subset = ds.sel(time=slice(end_skip_time, None))

# Function to generate filename with full timestamp
def generate_filename(time):
    return f"IMERG_precip_{time.strftime('%Y%m%d_%H%M')}.nc"

# Loop through each hour and save as separate NetCDF files
# Spliting and making new files for each hour is useful for validation
for i in range(len(ds_subset.time)):
    # Select single hour
    hour_data = ds_subset.isel(time=i)
    
    # Get the actual timestamp
    timestamp = pd.to_datetime(hour_data.time.values)
    
    # Create output filename with full date and hour
    output_path = Path(output_dir) / generate_filename(timestamp)
    
    # Save to new NetCDF file
    hour_data.to_netcdf(output_path)
    print(f"Saved {timestamp} to {output_path}")

# Close the dataset
ds.close()