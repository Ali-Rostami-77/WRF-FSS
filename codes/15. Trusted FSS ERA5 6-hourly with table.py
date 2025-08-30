# This code performs the fuzzy or neighbourhood validation for precipitation the FSS method.
# Observation dataset is ERA5 and Model dataset is WRF

import xarray as xr
import numpy as np
from scipy.interpolate import griddata
from scipy.signal import convolve2d
from pathlib import Path
import matplotlib.pyplot as plt
import traceback

def calculate_fss(obs, model, thresholds, window_sizes):
    """Calculate Fractions Skill Score (FSS) between observed and model data"""
    fss_results = {}

    obs = np.squeeze(obs)
    model = np.squeeze(model)

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
    """Ensure consistent (lat, lon) dimension ordering for ERA5 data"""
    if 'latitude' in ds.dims and 'longitude' in ds.dims:
        return ds.rename({'latitude':'lat', 'longitude':'lon'})
    elif 'lat' in ds.dims and 'lon' in ds.dims:
        return ds.transpose('lat', 'lon')
    else:
        raise ValueError("Cannot identify lat/lon dimensions")

def downscale_wrf_to_era5(wrf_ds, era5_lats, era5_lons):
    """Interpolate WRF data to match ERA5 grid points"""
    wrf_lons = wrf_ds.XLONG.values
    wrf_lats = wrf_ds.XLAT.values
    wrf_precip = wrf_ds.RAINNC.values + wrf_ds.RAINC.values

    lon_grid, lat_grid = np.meshgrid(era5_lons, era5_lats)

    wrf_on_era5 = griddata(
        points=np.column_stack([wrf_lons.ravel(), wrf_lats.ravel()]),
        values=wrf_precip.ravel(),
        xi=(lon_grid, lat_grid),
        method='linear',
        fill_value=0
    )

    return wrf_on_era5

def create_fss_table(fss_results, output_path, title_suffix=""):
    """Create a PNG table of FSS results"""
    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(211)
    ax.axis('off')

    thresholds = sorted(fss_results.keys())
    window_sizes = sorted(next(iter(fss_results.values())).keys())

    columns = ['Threshold (mm/hr)'] + [f'{w}x{w}' for w in window_sizes]
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

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', weight='bold')
        elif j == 0:
            cell.set_facecolor('#f1f1f1')
        elif i > 0:
            val = float(cell.get_text().get_text())
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

def process_single_fss(era5_path, wrf_path, area_bounds, thresholds, window_sizes, output_dir):
    """Process FSS verification for a single ERA5 + WRF pair"""
    try:
        era5_ds = standardize_dims(xr.open_dataset(era5_path))
        wrf_ds = xr.open_dataset(wrf_path)

        if 'time' in era5_ds.dims:
            era5_ds = era5_ds.isel(time=0)
        elif 'valid_time' in era5_ds.dims:
            era5_ds = era5_ds.isel(valid_time=0)

        if 'expver' in era5_ds.dims:
            era5_ds = era5_ds.isel(expver=0)

        era5_precip = era5_ds['tp-6h']

        min_lon, max_lon, min_lat, max_lat = area_bounds

        lat_descending = era5_ds.lat[0] > era5_ds.lat[-1]
        if lat_descending:
            era5_subset = era5_ds.sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat))
        else:
            era5_subset = era5_ds.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))

        era5_subset = era5_subset.squeeze()

        wrf_ds['RAINNC'] = wrf_ds['RAINNC'].astype(np.float32)
        wrf_ds['RAINC'] = wrf_ds['RAINC'].astype(np.float32)

        wrf_mask = (
            (wrf_ds.XLONG >= min_lon) &
            (wrf_ds.XLONG <= max_lon) &
            (wrf_ds.XLAT >= min_lat) &
            (wrf_ds.XLAT <= max_lat)
        )
        wrf_subset = wrf_ds.where(wrf_mask, drop=True)

        wrf_on_era5 = downscale_wrf_to_era5(
            wrf_subset,
            era5_subset.lat.values,
            era5_subset.lon.values
        )

        fss_results = calculate_fss(
            era5_subset['tp-6h'].values,
            wrf_on_era5,
            thresholds,
            window_sizes
            )


        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        table_path = output_dir / "fss_verification_single.png"
        create_fss_table(fss_results, table_path, title_suffix="Single File Validation")

        era5_ds.close()
        wrf_ds.close()

        return fss_results, str(table_path)

    except Exception as e:
        print(f"Error processing {era5_path}: {str(e)}")
        traceback.print_exc()
        return None, None

# ===== Configuration =====
area_bounds = (52.5, 58, 26.3, 29.5)
thresholds = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]
window_sizes = [3, 5, 7, 9, 11]
output_dir = "Path to desired directory"

# File paths
era5_file = "Path to ERA5 precipitation data"
wrf_file = "Path to WRF precipitation data"

if __name__ == "__main__":
    print("Starting single file FSS verification (ERA5 = observation, WRF = model)...")
    fss_results, table_path = process_single_fss(
        era5_file, wrf_file, area_bounds, thresholds, window_sizes, output_dir)

    if fss_results:
        print("\n=== FSS Results (Single File) ===")
        for threshold, scores in fss_results.items():
            print(f"\nThreshold: {threshold} mm/hr")
            for window, fss in scores.items():
                print(f"  Window {window}x{window}: FSS = {fss:.3f}")
        print(f"\nResults table saved at: {table_path}")
