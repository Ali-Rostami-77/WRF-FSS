# This code generates 6-hourly precipitation data from IMERG GPM dataset.

import xarray as xr
import numpy as np
import glob
import os
import pandas as pd

# --- Load IMERG files ---
imerg_files = sorted(glob.glob("Path to GPM dataset directory/*.nc4")) # the format is NetCDF4

# Calculate how many 6-hour periods we have (12 half-hourly files per 6 hours)
num_6hour_periods = len(imerg_files) // 12

for period in range(num_6hour_periods):
    # Get the 12 files for this 6-hour period
    start_idx = period * 12
    end_idx = start_idx + 12
    period_files = imerg_files[start_idx:end_idx]
    
    six_hour_rain_list = []
    lat_values = None
    lon_values = None
    final_time = None   # will hold the last file's timestamp
    
    # Process each pair of files to get hourly accumulation
    for i in range(0, len(period_files), 2):
        if i+1 >= len(period_files):
            break  # Skip if we don't have a complete pair
            
        ds1 = xr.open_dataset(period_files[i])
        ds2 = xr.open_dataset(period_files[i+1])

        # Extract and squeeze the rain rate data
        rain1 = ds1["precipitation"].squeeze()
        rain2 = ds2["precipitation"].squeeze()

        # Ensure same dimension ordering (lat: 150, lon: 200)
        rain1 = rain1.transpose('lat', 'lon') if 'lat' in rain1.dims else rain1
        rain2 = rain2.transpose('lat', 'lon') if 'lat' in rain2.dims else rain2

        # Store lat/lon values from first file
        if lat_values is None:
            lat_values = rain1.lat.values
            lon_values = rain1.lon.values
            # Verify dimensions
            if len(lat_values) != 150 or len(lon_values) != 200:
                raise ValueError(f"Unexpected dimensions: lat={len(lat_values)}, lon={len(lon_values)}")

        # Sum the two half-hour values to get 1-hour accumulated rain
        hourly_rain = (rain1 + rain2)  

        six_hour_rain_list.append(hourly_rain.values)

        # Keep updating the "final_time" with the second file's time
        final_time = pd.to_datetime(ds2.time.values[0])

        ds1.close()
        ds2.close()
    
    # Sum all hourly accumulations to get 6-hour total
    if six_hour_rain_list:
        six_hour_total = np.sum(six_hour_rain_list, axis=0)
        
        # Verify final array dimensions (should be 150x200)
        if six_hour_total.shape != (150, 200):
            raise ValueError(f"Unexpected final shape: {six_hour_total.shape}. Expected (150, 200)")
        
        # Create xarray DataArray with proper dimensions
        six_hour_da = xr.DataArray(
            data=np.expand_dims(six_hour_total, axis=0),  # Add time dimension
            dims=["time", "lat", "lon"],
            coords={
                "time": [final_time],   # use the last file's timestamp
                "lat": lat_values,
                "lon": lon_values
            },
            name="precipitation"
        )
        
        # Convert to Dataset
        six_hour_ds = six_hour_da.to_dataset()
        
        # Add metadata
        six_hour_ds.attrs["description"] = "6-hour accumulated IMERG precipitation"
        six_hour_ds.attrs["units"] = "mm"
        six_hour_ds.attrs["time_coverage_start"] = str(pd.to_datetime(xr.open_dataset(period_files[0]).time.values[0]))
        six_hour_ds.attrs["time_coverage_end"] = str(final_time)
        
        # Create output directory if it doesn't exist
        output_dir = "Path to your destination"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp (based on final_time)
        timestamp = final_time.strftime("%Y%m%d_%H%M")
        output_path = os.path.join(output_dir, f"imerg_6hour_{timestamp}.nc")
        
        # Save to NetCDF
        six_hour_ds.to_netcdf(output_path)
        print(f"✅ Saved 6-hour IMERG precipitation: {output_path} (time={final_time})")

print("\nProcessing complete!")
