# This code generate Precipitation pattern from wrf output + station rainfall for a limited area.
# For a specific area of interest, my study comes up with 3 different CU schemes for 3 domains.
# Therefore, this code generates 9 subplots into the main plot.

import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. Load Iran Shapefile ---
shapefile_path = "C:/BEST_SHAPE_FILE/gadm41_IRN_1.shp"
iran_shape = gpd.read_file(shapefile_path)

# --- 2. Input Files Setup ---
wrf_files = {
    "KF": [
        "E:/KF/2023-04-12/6hr_accum-D1/wrf_rain_accum_6hr_20230412_1800.nc",
        "E:/KF/2023-04-12/6hr_accum-D2/wrf_rain_accum_6hr_20230412_1800.nc",
        "E:/KF/2023-04-12/6hr_accum-D3/wrf_rain_accum_6hr_20230412_1800.nc"
    ],
    "BMJ": [
        "E:/BMJ/2023-04-12/6hr_accum-D1/wrf_rain_accum_6hr_20230412_1800.nc",
        "E:/BMJ/2023-04-12/6hr_accum-D2/wrf_rain_accum_6hr_20230412_1800.nc",
        "E:/BMJ/2023-04-12/6hr_accum-D3/wrf_rain_accum_6hr_20230412_1800.nc"
    ],
    "GF": [
        "E:/GF/2023-04-12/6hr_accum-D1/wrf_rain_accum_6hr_20230412_1800.nc",
        "E:/GF/2023-04-12/6hr_accum-D2/wrf_rain_accum_6hr_20230412_1800.nc",
        "E:/GF/2023-04-12/6hr_accum-D3/wrf_rain_accum_6hr_20230412_1800.nc"
    ]
}

# --- 3. Load Station Data ---
station_path = "E:/Real/2023-04-12-Case 12-18.xlsx"
stations = pd.read_excel(station_path)

stations['date_time'] = pd.to_datetime(stations['date_time'])
matching_time = pd.to_datetime("2023-04-12 18:00:00")
current_stations = stations[stations['date_time'] == matching_time]
if len(current_stations) == 0:
    print("Warning: No station data found for this time, using all stations instead.")
    current_stations = stations

stations_gdf = gpd.GeoDataFrame(
    current_stations,
    geometry=gpd.points_from_xy(current_stations.longitude, current_stations.latitude),
    crs="EPSG:4326"
)

# --- 4. Area of Interest ---
min_lat, max_lat = 35, 39.5
min_lon, max_lon = 44, 48

# --- 5. Colormap Setup ---
precip_levels = [0, 0.5, 2, 4, 8, 16, 32, 64, 128]
colors = ["#ffffff", "#cceeff", "#99ccff", "#3399ff", "#0066ff",
          "#0000ff", "#00cc00", "#009900", "#ffcc00", "#ff9900",
          "#ff0000", "#990000"]
cmap = ListedColormap(colors)
norm = BoundaryNorm(precip_levels, cmap.N)

# --- 6. Figure Setup ---
fig, axes = plt.subplots(3, 3, figsize=(18, 14),
                         subplot_kw={'projection': ccrs.PlateCarree()})

# --- 7. Loop Through Files ---
schemes = list(wrf_files.keys())
domains = ["D1", "D2", "D3"]

for i, scheme in enumerate(schemes):       # KF, BMJ, GF rows
    for j, wrf_file in enumerate(wrf_files[scheme]):   # D1, D2, D3 cols
        ax = axes[i, j]

        # Load NetCDF
        ds = xr.open_dataset(wrf_file)
        precip = ds['RAINNC_6hr'] + ds['RAINC_6hr']
        lat = ds['XLAT']
        lon = ds['XLONG']

        # Plot precipitation: D1 & D2 = contourf, D3 = pcolormesh
        if j < 2:
            mesh = ax.contourf(lon, lat, precip,
                               levels=precip_levels, cmap=cmap, norm=norm,
                               transform=ccrs.PlateCarree(), extend="max")
        else:
            mesh = ax.pcolormesh(lon, lat, precip,
                                 cmap=cmap, norm=norm,
                                 shading='auto', transform=ccrs.PlateCarree())

        # Add shapefile & features
        iran_shape.boundary.plot(ax=ax, color="gray", linewidth=1, transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5)

        # Plot stations
        for x, y, label in zip(stations_gdf.geometry.x, stations_gdf.geometry.y, stations_gdf['rrr']):
            ax.plot(x, y, 'o', color='black', markersize=5, transform=ccrs.PlateCarree())
            ax.text(x + 0.01, y + 0.01, f"{label:.1f}",
                    color='black', ha='right', va='bottom',
                    fontsize=8, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=1),
                    transform=ccrs.PlateCarree())

        # Set extent
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

        # Titles
        ax.set_title(f"{scheme} - {domains[j]}", fontsize=14)

# --- 8. Shared Colorbar Below ---
# Different cropping needs subplot adjusting
# Use tight_layout first to compact plots
plt.tight_layout(rect=[0.15, 0.2, 0.85, 0.96]) #(left, vertical compact,right, above)

# Then add colorbar manually below
cbar_ax = fig.add_axes([0.25, 0.15, 0.5, 0.02])  # [left, bottom, width, height]
cbar = fig.colorbar(mesh, cax=cbar_ax, orientation='horizontal')
cbar.set_label("6-hour Precipitation (mm)", fontsize=12)
cbar.set_ticks(precip_levels)

# --- 9. Finalize ---
fig.suptitle("WRF rainfall map with synoptic stations for 2023-04-12 18:00:00",
             fontsize=16, fontweight="bold")

plt.savefig("E:/ERA5/2023-04-12/WRF-2023-04-12-18.PNG", dpi=600, bbox_inches="tight")
plt.show()
