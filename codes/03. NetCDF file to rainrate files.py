# This code does spliting, It make hourly rainrate files from NetCDF file of precipitation.
# Making hourly files helps performing validations better and hourly

import xarray as xr
import numpy as np
from datetime import datetime

def create_hourly_rainrate_files(input_file, output_prefix):
    """
    Create hourly rainrate files where:
    - First hour (13:00) is kept as original
    - Subsequent hours show difference from previous hour
    
    Args:
        input_file: Path to the spun-up WRF file
        output_prefix: Prefix for output files
    """
    ds = xr.open_dataset(input_file)
    
    # Get time information
    times = ds['XTIME'].values
    num_times = len(times)
    
    if num_times < 1:
        print("Error: No time steps found")
        return
    
    # Process each time step
    for i in range(num_times):
        # Create filename timestamp
        try:
            # Get time as numpy datetime64
            current_time = ds['XTIME'][i].values
            time_str = np.datetime_as_string(current_time, unit='h')
            time_str = time_str.replace('-','').replace('T','_')[:13]
        except:
            # Fallback method if numpy conversion fails
            start_date = datetime.strptime(ds.attrs['START_DATE'], '%Y-%m-%d_%H:%M:%S')
            minutes = int(ds['XTIME'][i].values % (24*60))  # Handle large values
            current_time = start_date + timedelta(minutes=minutes)
            time_str = current_time.strftime('%Y%m%d_%H00')
        
        # For first time step (13:00), keep original values
        # This should done because spin-up code makes hourly rates for the first time step
        if i == 0:
            hour_ds = xr.Dataset({
                'RAINNC': ds['RAINNC'][i],
                'RAINC': ds['RAINC'][i],
                'XLAT': ds['XLAT'][i],
                'XLONG': ds['XLONG'][i],
                'XTIME': ds['XTIME'][i]
            })
        else:
            # For subsequent hours, calculate difference from previous hour
            rainnc_diff = ds['RAINNC'][i] - ds['RAINNC'][i-1]
            rainc_diff = ds['RAINC'][i] - ds['RAINC'][i-1]
            
            hour_ds = xr.Dataset({
                'RAINNC': rainnc_diff,
                'RAINC': rainc_diff,
                'XLAT': ds['XLAT'][i],
                'XLONG': ds['XLONG'][i],
                'XTIME': ds['XTIME'][i]
            })
        
        # Copy attributes
        for var in ['RAINNC', 'RAINC', 'XLAT', 'XLONG', 'XTIME']:
            hour_ds[var].attrs = ds[var].attrs
        
        # Save file
        output_file = f"{output_prefix}{time_str}.nc"
        hour_ds.to_netcdf(output_file)
        print(f"Created: {output_file}")
    
    ds.close()

# Should place the paths 
# Remember to make the 'hourly_split-D?' directory first
if __name__ == "__main__":
    input_file = "Path to NetCDF file/merged_wrf_precip-D?.nc"
    output_prefix = "Path to new rain rate directory /hourly_split-D?/wrf_rainrate_"
    
    create_hourly_rainrate_files(input_file, output_prefix)