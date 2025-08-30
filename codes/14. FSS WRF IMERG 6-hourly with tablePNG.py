# This code performs the fuzzy or neighbourhood validation for precipitation the FSS method.
# Observation dataset is IMERG GPM final and Model dataset is WRF

import xarray as xr
import numpy as np
from scipy.interpolate import griddata
from scipy.signal import convolve2d
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

def calculate_fss(obs, model, thresholds, window_sizes):
    """Calculate Fractions Skill Score (FSS) between observed and model data"""
    fss_results = {}
    
    if obs.shape != model.shape:
        raise ValueError(f"Shape mismatch: obs {obs.shape} vs model {model.shape}")
    
    for threshold in thresholds:
        fss_results[threshold] = {}
        obs_binary = (obs >= threshold).astype(int)
        model_binary = (model >= threshold).astype(int)
        
        for window in window_sizes:
            kernel = np.ones((window, window)) / (window**2)
            obs_frac = convolve2d(obs_binary, kernel, mode='same', boundary='fill', fillvalue=0)
            model_frac = convolve2d(model_binary, kernel, mode='same', boundary='fill', fillvalue=0)
            
            mse = np.mean((obs_frac - model_frac)**2)
            mse_ref = np.mean(obs_frac**2) + np.mean(model_frac**2)
            fss = 1 - mse/mse_ref if mse_ref > 0 else np.nan
            fss_results[threshold][window] = fss
    
    return fss_results

def standardize_dims(ds):
    """Ensure consistent (time, lat, lon) dimension ordering"""
    if 'lat' in ds.dims and 'lon' in ds.dims:
        if 'time' in ds.dims:
            return ds.transpose('time', 'lat', 'lon')
        else:
            return ds.transpose('lat', 'lon')
    elif 'latitude' in ds.dims and 'longitude' in ds.dims:
        rename_dict = {'latitude': 'lat', 'longitude': 'lon'}
        ds = ds.rename(rename_dict)
        if 'time' in ds.dims:
            return ds.transpose('time', 'lat', 'lon')
        else:
            return ds.transpose('lat', 'lon')
    else:
        raise ValueError("Cannot identify lat/lon dimensions")

def interpolate_wrf_to_imerg_lowres(wrf_ds, target_lat, target_lon):
    """Interpolate low-res WRF to IMERG grid"""
    points = np.column_stack([wrf_ds.XLONG.values.ravel(), wrf_ds.XLAT.values.ravel()])
    values = wrf_ds.RAINNC_6hr.values.ravel()   # use your variable name
    
    lat_grid, lon_grid = np.meshgrid(target_lat, target_lon, indexing='ij')
    wrf_interp = griddata(points, values, (lon_grid, lat_grid), method='linear', fill_value=0)
    return wrf_interp

def create_fss_table(fss_results, output_path, title_suffix=""):
    """Create a PNG table of FSS results with horizontal color bar"""
    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(211)
    ax.axis('off')
    
    thresholds = sorted(fss_results.keys())
    window_sizes = sorted(next(iter(fss_results.values())).keys())
    
    columns = ['Threshold (mm/6hr)'] + [f'{w}x{w}' for w in window_sizes]
    cell_text = []
    
    for thresh in thresholds:
        row = [f"{thresh:.1f}"] + [f"{fss_results[thresh][w]:.3f}" for w in window_sizes]
        cell_text.append(row)
    
    table = ax.table(cellText=cell_text,
                    colLabels=columns,
                    loc='center',
                    cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    all_values = []
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', weight='bold')
        elif j == 0:
            cell.set_facecolor('#f1f1f1')
        elif i > 0:
            val = float(cell.get_text().get_text())
            all_values.append(val)
            cell.set_facecolor(plt.cm.viridis(val))
    
    cax = plt.subplot(212)
    cax.axis('off')
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=cax, orientation='horizontal', pad=0.2)
    cbar.set_label('FSS Score (0 = Worst, 1 = Best)', labelpad=10)
    
    title = 'Fractions Skill Score (FSS) Verification Results'
    if title_suffix:
        title += f" - {title_suffix}"
    plt.suptitle(title, fontsize=14, y=0.95)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

def process_single_file(imerg_file, wrf_file, area_bounds, thresholds, window_sizes, output_dir):
    """Process FSS verification for a single IMERG + WRF file"""
    try:
        imerg_ds = standardize_dims(xr.open_dataset(imerg_file))
        wrf_ds = xr.open_dataset(wrf_file)
        
        min_lon, max_lon, min_lat, max_lat = area_bounds
        
        imerg_subset = imerg_ds.sel(
            lon=slice(min_lon, max_lon),
            lat=slice(min_lat, max_lat))
        
        wrf_mask = (
            (wrf_ds.XLONG >= min_lon) & 
            (wrf_ds.XLONG <= max_lon) & 
            (wrf_ds.XLAT >= min_lat) & 
            (wrf_ds.XLAT <= max_lat)
        )
        wrf_subset = wrf_ds.where(wrf_mask, drop=True)
        
        # Interpolate WRF to IMERG grid
        wrf_interp = interpolate_wrf_to_imerg_lowres(
            wrf_subset,
            imerg_subset.lat.values,
            imerg_subset.lon.values)
        
        # Extract variables
        imerg_precip = imerg_subset.precipitation.values   # variable name
        # --- FIX SHAPES ---
        imerg_precip = np.squeeze(imerg_precip)  # remove time=1 dimension if present
        wrf_interp   = np.squeeze(wrf_interp)    # just in case
        # ------------------
        
        fss_results = calculate_fss(imerg_precip, wrf_interp, thresholds, window_sizes)
        
        timestamp = "SingleFile"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        table_path = output_dir / f"fss_verification_{timestamp}.png"
        
        create_fss_table(fss_results, table_path, title_suffix="6-hour Accumulation")
        
        imerg_ds.close()
        wrf_ds.close()
        
        return fss_results, str(table_path)
    
    except Exception as e:
        print(f"Error processing single file: {str(e)}")
        return None, None

# ===== Configuration =====
area_bounds = (53.5, 62.5, 26.3, 30) # (min lon, max lon, min lat, max lat)
thresholds = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]   # mm/6hr
window_sizes = [3, 5, 7, 9, 11]
output_dir = "Path to desired directory"

# File paths (replace with your own)
imerg_file = "Path to IMERG Precipitation data"
wrf_file   = "Path to WRF precipitation data"

if __name__ == "__main__":
    print("Starting single file FSS verification...")
    fss_results, table_path = process_single_file(
        imerg_file, wrf_file, area_bounds, thresholds, window_sizes, output_dir)
    
    if fss_results:
        print("\n=== FSS Results (Single File) ===")
        for threshold, scores in fss_results.items():
            print(f"\nThreshold: {threshold} mm/6hr")
            for window, fss in scores.items():
                print(f"  Window {window}x{window}: FSS = {fss:.3f}")
        print(f"\nResults table saved at: {table_path}")
