# This code generates Precipitaion Pattern from GPM IMERG + Stations for a limited area

import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# --- 1. Load Iran Shapefile ---
shapefile_path = "Path to shapefile.shp"
iran_shape = gpd.read_file(shapefile_path)

# --- 2. Load IMERG GPM Precipitation Data ---
nc_path = "Path to GPM 6 hourly precipitation data.nc"
ds = xr.open_dataset(nc_path)

# Debug print to verify original IMERG time
print("Original IMERG time:", ds['time'].values)

# Adjust IMERG time forward by ? hours to match station data (in case of time difference)
ds['time'] = pd.to_datetime(ds['time'].values) + pd.Timedelta(hours=0)
print("Adjusted IMERG time:", ds['time'].values)

# Get precipitation data
precip = ds['precipitation']
lat = ds['lat']
lon = ds['lon']

# --- 3. Load Station Data (Reference for correct time) ---
station_path = "Path to synoptic stations' precipitation data.xlsx"
stations = pd.read_excel(station_path)
stations['date_time'] = pd.to_datetime(stations['date_time'])

# Find matching time in IMERG data
matching_time = stations['date_time'].iloc[0]  
print("Station reference time:", matching_time)

# Convert IMERG times to pandas Timestamps for comparison
imerg_times = pd.to_datetime(ds['time'].values)

# Find closest IMERG time to station time
time_diff = np.abs(imerg_times - matching_time)
closest_time_idx = np.argmin(time_diff)
selected_time = imerg_times[closest_time_idx]

# Select precipitation for this time
precip = ds['precipitation'].isel(time=closest_time_idx)
print("Selected IMERG time:", selected_time)

# Filter stations for this time
current_stations = stations[stations['date_time'] == selected_time]
if len(current_stations) == 0:
    print(f"Warning: No station data found for {selected_time}")
    current_stations = stations

# Convert to GeoDataFrame
stations_gdf = gpd.GeoDataFrame(
    current_stations,
    geometry=gpd.points_from_xy(current_stations.longitude, current_stations.latitude),
    crs="EPSG:4326"
)

# --- 4. Define Area of Interest --- for different areas use different values
min_lat, max_lat = 26, 29.6
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

# Plot precipitation as contours instead of heatmap because of 0.1 degree resolution for better view
contour = ax.contourf(lon, lat, precip,
                     levels=precip_levels,
                     cmap=cmap, norm=norm,
                     transform=ccrs.PlateCarree())

# Add colorbar
cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.02, shrink=0.5)
cbar.set_label('Precipitation (mm)', fontsize=12)
cbar.set_ticks(precip_levels)

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

# Add latitude and longitude gridlines and labels
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False    # Turn off top labels
gl.right_labels = False  # Turn off right labels
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}

# Format time for title
plot_time = selected_time.strftime('%Y-%m-%d %H:%M')
ax.set_title(f"IMERG GPM Precipitation for {plot_time}", fontsize=14)
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
output_path = "Path to save file.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"\nMap saved to: {output_path}")
plt.show()