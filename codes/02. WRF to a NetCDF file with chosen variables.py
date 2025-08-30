# this code converts spun-up WRF outputs into one NetCDF file with chosen variables 
# The purpose of generating this code is to reduce reading and processing time for a chosen variable and time

import xarray as xr
import numpy as np
import pandas as pd
import glob
import os

def merge_wrfout_files(input_dir, output_file, pattern='wrfout*'):
    """
    Merge multiple WRF output files into a single NetCDF file, keeping only specified variables.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing WRF output files
    output_file : str
        Path for the output NetCDF file
    pattern : str
        File pattern to match WRF output files (default: 'wrfout*')
    """
    # List all WRF output files
    file_list = sorted(glob.glob(os.path.join(input_dir, pattern)))
    
    if not file_list:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(file_list)} WRF output files to process")
    
    # Here is the Varables one needs to keep
    vars_to_keep = ['XLAT', 'XLONG', 'XTIME', 'RAINNC', 'RAINC']
    
    # Open first file to get dimensions and attributes
    with xr.open_dataset(file_list[0]) as ds:
        # Check if all required variables exist
        missing_vars = [var for var in vars_to_keep if var not in ds]
        if missing_vars:
            print(f"Error: Missing required variables: {missing_vars}")
            return
        
        # Get global attributes
        global_attrs = ds.attrs

    # Variable like time should be considered well, becausing it is changing along each file    
    # Process files one by one and concatenate along time dimension
    datasets = []
    for i, file in enumerate(file_list):
        print(f"Processing file {i+1}/{len(file_list)}: {os.path.basename(file)}")
        
        with xr.open_dataset(file) as ds:
            # Select only the variables we want to keep
            ds_subset = ds[vars_to_keep]
            
            # Add to our list
            datasets.append(ds_subset)
    
    # Concatenate all datasets along time dimension
    print("Concatenating files...")
    merged_ds = xr.concat(datasets, dim='Time')
    
    # Add back global attributes
    merged_ds.attrs = global_attrs
    
    # Update attributes to reflect merging
    merged_ds.attrs['history'] = f"Merged {len(file_list)} WRF output files on {pd.Timestamp.now().isoformat()}"
    merged_ds.attrs['source_files'] = ', '.join([os.path.basename(f) for f in file_list])
    
    # Save to output file
    print(f"Saving merged data to {output_file}")
    encoding = {
        'RAINC': {'zlib': True, 'complevel': 4},
        'RAINNC': {'zlib': True, 'complevel': 4},
        'XLAT': {'zlib': True, 'complevel': 4},
        'XLONG': {'zlib': True, 'complevel': 4},
        'XTIME': {'zlib': True, 'complevel': 4}
    }
    
    merged_ds.to_netcdf(output_file, encoding=encoding)
    print("Done!")

if __name__ == '__main__':
    # Here the Path of input WRF directory for a single domain and destination path added
    input_directory = 'Path to WRF directory'
    output_filename = 'Path to destination/merged_wrf_precip-D?.nc'
    
    merge_wrfout_files(input_directory, output_filename)