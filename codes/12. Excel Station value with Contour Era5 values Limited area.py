# This code generates precipitation pattern of ERA5 + synoptic stations' values

import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os

# --- 1. Load Iran Shapefile ---
shapefile_path = "C:/BEST_SHAPE_FILE/gadm41_IRN_1.shp"
iran_shape = gpd.read_file(shapefile_path)

# --- 2. Load ERA5 Precipitation Data ---
era5_path = "E:/case/ERA5_tp6h_20220727_0000.nc"
try:
    ds = xr.open_dataset(era5_path)
    print("\n=== Dataset Info ===")
    print(ds)
    
    # Debug print dimensions and variables
    print("\n=== Dimensions ===")
    print(ds.dims)
    print("\n=== Variables ===")
    print(list(ds.variables.keys()))
    
    # Debug print to verify ERA5 time
    print("\nERA5 time:", ds['valid_time'].values)

    # Get precipitation data
    if 'tp-6h' in ds:
        precip = ds['tp-6h']
    else:
        raise ValueError("Could not find precipitation variable in ERA5 file")
    
    # Debug precipitation dimensions
    print("\n=== Precipitation Dimensions ===")
    print(f"Original precipitation dimensions: {precip.dims}")
    print(f"Original precipitation shape: {precip.shape}")
    
    # Handle multi-dimensional precipitation data
    if len(precip.dims) > 2:
        print("\nWarning: Precipitation data has more than 2 dimensions")
        # Try to find time dimension
        time_dims = [dim for dim in precip.dims if 'time' in dim.lower()]
        if time_dims:
            print(f"Found time dimension(s): {time_dims}")
            # Use the first time dimension found
            precip = precip.isel({time_dims[0]: 0})
            print(f"Selected first time step from dimension '{time_dims[0]}'")
        else:
            print("No time dimension found - selecting first element along first dimension")
            precip = precip.isel({precip.dims[0]: 0})
        
        print(f"New precipitation dimensions: {precip.dims}")
        print(f"New precipitation shape: {precip.shape}")
    
    # Get coordinates
    lat = ds['latitude']
    lon = ds['longitude']
    print(f"\nLatitude shape: {lat.shape}")
    print(f"Longitude shape: {lon.shape}")

except Exception as e:
    print(f"\nError loading ERA5 data: {str(e)}")
    raise

# --- 3. Load Station Data ---
station_path = "E:/Real/2022-07-26-Case 18-24.xlsx"
try:
    stations = pd.read_excel(station_path)
    stations['date_time'] = pd.to_datetime(stations['date_time'])
    
    # Find matching time in ERA5 data
    era5_time = pd.to_datetime(ds['valid_time'].values[0])
    print("\nERA5 reference time:", era5_time)

    # Find stations within ±3 hours of ERA5 time
    time_diff = np.abs(stations['date_time'] - era5_time)
    current_stations = stations[time_diff <= pd.Timedelta(hours=3)]

    if len(current_stations) == 0:
        print(f"Warning: No station data found near {era5_time}")
        current_stations = stations

    # Convert to GeoDataFrame
    stations_gdf = gpd.GeoDataFrame(
        current_stations,
        geometry=gpd.points_from_xy(current_stations.longitude, current_stations.latitude),
        crs="EPSG:4326"
    )
except Exception as e:
    print(f"\nError loading station data: {str(e)}")
    raise

# --- 4. Define Area of Interest ---
min_lat, max_lat = 26.3, 29.5
min_lon, max_lon = 53, 60

# --- 5. Plot Setup ---
fig = plt.figure(figsize=(12, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# --- 6. Plot Precipitation ---
precip_levels = [0, 0.5, 2, 4, 8, 16, 32, 64, 128]
colors = ["#ffffff", "#cceeff", "#99ccff", "#3399ff", "#0066ff",
          "#0000ff", "#00cc00", "#009900", "#ffcc00", "#ff9900",
          "#ff0000", "#990000"]
cmap = ListedColormap(colors)
norm = BoundaryNorm(precip_levels, cmap.N)

try:
    # Plot precipitation as contours instead of heatmap
    print("\n=== Final Plotting Data ===")
    print(f"Longitude shape: {lon.shape}")
    print(f"Latitude shape: {lat.shape}")
    print(f"Precipitation shape: {precip.shape}")
    
    # Use contourf to plot filled contours
    contour = ax.contourf(lon, lat, precip,
                         levels=precip_levels,
                         cmap=cmap, norm=norm,
                         transform=ccrs.PlateCarree())

    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
    cbar.set_label('6-hour Precipitation (mm)', fontsize=12)
    cbar.set_ticks(precip_levels)

except Exception as e:
    print(f"\nError plotting precipitation: {str(e)}")
    raise

# --- 7. Plot Map Features ---
iran_shape.boundary.plot(ax=ax, color="gray", linewidth=1, transform=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5)

# --- 8. Plot Station Data ---
for x, y, label in zip(stations_gdf.geometry.x, stations_gdf.geometry.y, stations_gdf['rrr']):
    ax.plot(x, y, 'o', color='black', markersize=8, transform=ccrs.PlateCarree())
    ax.text(x + 0.15, y + 0.1, f"{label:.1f}", 
            color='black', ha='center', va='center',
            fontsize=9, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1),
            transform=ccrs.PlateCarree())

# --- 9. Finalize Map ---
ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

# Add gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}

# Format time for title
plot_time = pd.to_datetime(era5_time).strftime('%Y-%m-%d %H:%M')
ax.set_title(f"ERA5 6-Hour Precipitation for {plot_time}", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# --- Precipitation Statistics ---
print("\n=== Precipitation Statistics ===")
print(f"Global min precipitation: {float(precip.min()):.2f} mm")
print(f"Global max precipitation: {float(precip.max()):.2f} mm")

# Calculate stats for study area
lon_2d, lat_2d = np.meshgrid(lon, lat)
mask = (lat_2d >= min_lat) & (lat_2d <= max_lat) & (lon_2d >= min_lon) & (lon_2d <= max_lon)
area_precip = precip.where(mask)

print(f"\n--- Within study area ({min_lon}-{max_lon}E, {min_lat}-{max_lat}N) ---")
print(f"Min precipitation: {float(area_precip.min()):.2f} mm")
print(f"Max precipitation: {float(area_precip.max()):.2f} mm")
print(f"Mean precipitation: {float(area_precip.mean()):.2f} mm")

# Add North arrow
ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction',
            ha='center', va='center', fontsize=15,
            bbox=dict(boxstyle='circle,pad=0.2', fc='white', ec='black'))
ax.arrow(0.95, 0.78, 0, 0.1, transform=ax.transAxes,
         head_width=0.02, head_length=0.02, fc='black', ec='black')

plt.tight_layout()

# --- 10. Save High-Resolution Output ---
output_dir = "E:/ERA5/2022-07-26/6-hourly/"
os.makedirs(output_dir, exist_ok=True)

# Generate output filename with timestamp
output_filename = f"ERA5_6hr_precip_{plot_time.replace(' ', '_').replace(':', '')}.png"
output_path = os.path.join(output_dir, output_filename)

# Save figure with 600 dpi resolution
plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
print(f"\nSaved high-resolution plot to: {output_path}")

plt.show()