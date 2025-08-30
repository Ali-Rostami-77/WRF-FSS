# This code plots the rainfall from synoptic stations with elevation on background.
# It uses Iran's shapefile, world's elevation map and dataset of synoptic stations. 

import geopandas as gpd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Load Iran Shapefile (Gray) ---
shapefile_path = "path to shapefile.shp"
iran_shape = gpd.read_file(shapefile_path)

# --- 2. Load Iran Elevation GeoTIFF ---
elevation_tif_path = "path to elevation map.tiff"
elevation_data = rasterio.open(elevation_tif_path)

# --- 3. Load Rainfall Data ---
rainfall_xlsx_path = "path to rainfall dataset.xlsx"
rainfall_df = pd.read_excel(rainfall_xlsx_path)

# Convert to GeoDataFrame
rainfall_gdf = gpd.GeoDataFrame(
    rainfall_df,
    geometry=gpd.points_from_xy(rainfall_df.longitude, rainfall_df.latitude),
    crs="EPSG:4326"
)

# Get unique date_time values (assuming you want to plot one time period)
# the 'date_time' is the name of the column in which the date and time stores, it may be different in other dataset.
selected_time = rainfall_df['date_time'].iloc[0]  # Takes first time in dataset

# Filter data for the selected time
current_rainfall = rainfall_gdf[rainfall_gdf['date_time'] == selected_time]

# --- 4. Define Crop Coordinates ---
# for different area use different cropping
min_lat, max_lat = 35.2, 36.2
min_lon, max_lon = 50.5, 52.4


# --- Plotting ---
fig, ax = plt.subplots(figsize=(12, 10))

# Plot elevation background
elevation_plot = show(
    elevation_data,
    ax=ax,
    cmap="terrain",
    vmin=0,
    vmax=5600
)

# Add colorbar for elevation
cbar = plt.colorbar(elevation_plot.get_images()[0], ax=ax, orientation='vertical', shrink=0.55)
cbar.set_label('Elevation (meters)', fontsize=12)

# Plot Iran borders
iran_shape.boundary.plot(
    ax=ax,
    color="gray",
    linewidth=2,
)

# Plot rainfall stations with values (adjusted position)
# 'rrr' is the column of 6-hour rainfall
for x, y, label in zip(current_rainfall.geometry.x, current_rainfall.geometry.y, current_rainfall['rrr']):
    # Plot the black dot
    ax.plot(x, y, 'o', color='black', markersize=5)
    
    # Plot the rainfall value slightly above the dot
    ax.text(x + 0.01, y + 0.01,
            f"{label:.1f}", 
            color='black', 
            ha='right', 
            va='bottom',
            fontsize=9, 
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2))

# Crop to desired area
ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)

# Add North arrow
ax.annotate('N', xy=(0.95, 0.96), xycoords='axes fraction',
            ha='center', va='center', fontsize=15,
            bbox=dict(boxstyle='circle,pad=0.2', fc='white', ec='black'))
ax.arrow(0.95, 0.79, 0, 0.1, transform=ax.transAxes,
         head_width=0.02, head_length=0.02, fc='black', ec='black')

# Add title with date and time
title_date = pd.to_datetime(selected_time).strftime('%Y-%m-%d %H:%M')
ax.set_title(f"6-hourly rainfall (mm) for {title_date}", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
# --- Save the plot with 60 dpi resolution ---
output_path = "path of saving the plot.png"  # Change to your desired path
plt.savefig(output_path, dpi=600, bbox_inches='tight')
plt.show()
print(f"Map saved to: {output_path}")