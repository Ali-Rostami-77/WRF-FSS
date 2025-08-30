# This code is generated in order to reduce the spin-up time from WRF Model Output.
# It changes all variables that has ACCUMULATED characteristics and preserves the others variables.

import os
import numpy as np
from netCDF4 import Dataset

# Configuration 
# Suitable for WRF directory which Contains hourly outputs
input_dir = 'Path of WRF directory of a single domain' 
output_dir = 'Path of desired new files'
spinup_time = '12:00'  # UTC (will be converted to URL-encoded format)

# Convert spinup_time to URL-encoded format (e.g., "12:00" → "12%3A00%3A00") because the format of WRF are like this
spinup_time_encoded = spinup_time.replace(':', '%3A') + '%3A00'

# Create output directory (in case of not creating the path already)
os.makedirs(output_dir, exist_ok=True)

# Get sorted list of WRF files
wrf_files = sorted([f for f in os.listdir(input_dir) if f.startswith('wrfout')])

# Find the FIRST spin-up file (12:00 UTC in URL-encoded format)
# This is important because all reduction of accumulated values should be done to the first spin-up file
spinup_file = None
for f in wrf_files:
    if spinup_time_encoded in f:
        spinup_file = f
        break

if not spinup_file:
    raise ValueError(f"No file found with time {spinup_time} (encoded as {spinup_time_encoded})")

# Open spin-up file to get reference values
with Dataset(os.path.join(input_dir, spinup_file), 'r') as ds:
    # List of accumulated variables to adjust (RAINC, RAINNC, etc.)
    # This list is optional, one can change or add other variables.
    accum_vars = ['ACGRDFLX', 'ACSNOM', 'RAINC', 'RAINSH', 'RAINNC', 
                 'SNOWNC', 'GRAUPELNC', 'HAILNC']
    
    # Store spin-up values (subtract these from later files)
    spinup_values = {}
    for var in accum_vars:
        if var in ds.variables:
            spinup_values[var] = ds.variables[var][:].copy()

# Process ALL files AFTER the FIRST spin-up file (including later 12:00 files)
process_files = False
for f in wrf_files:
    if f == spinup_file:
        process_files = True  # Start processing from the next file
        continue  # Skip the first spin-up file (but adjust later ones)
    
    if process_files:
        print(f"\nAdjusting {f}...")
        
        with Dataset(os.path.join(input_dir, f), 'r') as src:
            # Create a new adjusted file in output_dir
            with Dataset(os.path.join(output_dir, f), 'w') as dst:
                # 1. Copy global attributes (metadata)
                dst.setncatts(src.__dict__)
                
                # 2. Copy dimensions (time, south_north, west_east, etc.)
                for dim_name, dim in src.dimensions.items():
                    dst.createDimension(dim_name, len(dim) if not dim.isunlimited() else None)
                
                # 3. Copy ALL variables
                for var_name, var in src.variables.items():
                    # Create variable in output file (same shape/dtype)
                    dst_var = dst.createVariable(var_name, var.datatype, var.dimensions)
                    
                    # Copy variable attributes (units, description, etc.)
                    dst_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                    
                    # Here is a verification line, shows the amounts that are reducted
                    # If it's an accumulated variable, subtract spin-up values
                    if var_name in accum_vars:
                        original_values = var[:]
                        adjusted_values = original_values - spinup_values[var_name]
                        dst_var[:] = adjusted_values
                        
                        # Print changes (for verification)
                        print(f"  {var_name}:")
                        print(f"    Original max: {np.nanmax(original_values):.2f}")
                        print(f"    Adjusted max: {np.nanmax(adjusted_values):.2f}")
                        print(f"    Spin-up subtracted: {np.nanmax(spinup_values[var_name]):.2f}")
                    else:
                        dst_var[:] = var[:]  # Copy unchanged

print(f"\nDone! Spin-up adjusted files saved in: {output_dir}")