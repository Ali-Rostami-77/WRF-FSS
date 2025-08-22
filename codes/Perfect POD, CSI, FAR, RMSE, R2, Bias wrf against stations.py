import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, r2_score
import os
from datetime import datetime
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde

# Configuration
WRF_FILE = "/media/My Passport/Spin-up/GF/2023-01-31/6hr_accum-D3/wrf_rain_accum_6hr_20230131_1800.nc"  # Example file
STATION_FILE = "/media/09102022065/My Thesis/Observation Data/data manipulation/2023-01-31-Case 12-18.xlsx"
OUTPUT_DIR = "/media/My Passport/verification_results13222/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thresholds for categorical verification (mm)
THRESHOLDS = [0.1, 1, 5, 10, 20,50]

def load_station_data(wrf_time):
    """Load and filter station data for specific WRF time"""
    # Read the Excel file - skip the first row if it contains headers
    df = pd.read_excel(STATION_FILE, header=0)
    
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
    
    # Convert to datetime and filter for matching time
    df['date_time'] = pd.to_datetime(df['date_time'])
    station_df = df[df['date_time'] == wrf_time].copy()
    
    # Check if we have data
    if len(station_df) == 0:
        raise ValueError(f"No station data found for {wrf_time}")
    
    # Return relevant columns (adjust based on your actual Excel structure)
    return station_df[['station_name', 'latitude', 'longitude', 'date_time', 'rrr']]

def extract_wrf_precip(wrf_file, stations):
    """Extract WRF precipitation at station locations"""
    results = []
    
    with xr.open_dataset(wrf_file) as ds:
        # Get WRF time from XTIME (already in datetime format)
        wrf_time = pd.to_datetime(ds.XTIME.values)
        
        for _, station in stations.iterrows():
            # Find nearest grid point
            dist = np.sqrt((ds.XLAT - station['latitude'])**2 + 
                        (ds.XLONG - station['longitude'])**2)
            i, j = np.unravel_index(dist.argmin(), dist.shape)
            
            # Get WRF precipitation (using RAINNC_6hr)
            wrf_precip = float(ds['RAINNC_6hr'][i,j].values)
            
            results.append({
                'Time': wrf_time,
                'Station': station['station_name'],
                'Observed': station['rrr'],
                'WRF': wrf_precip,
                'Latitude': station['latitude'],
                'Longitude': station['longitude']
            })
    
    return pd.DataFrame(results)

def calculate_categorical_metrics(verif_df, thresholds):
    """Calculate categorical verification metrics"""
    metrics = []
    
    for thresh in thresholds:
        # Create binary arrays
        obs_binary = (verif_df['Observed'] >= thresh).astype(int)
        wrf_binary = (verif_df['WRF'] >= thresh).astype(int)
        
        # Handle cases where all values are same
        unique_classes = np.unique(np.concatenate([obs_binary, wrf_binary]))
        if len(unique_classes) == 1:
            # All predictions and observations are the same
            if unique_classes[0] == 1:  # All hits (rain)
                tp = len(obs_binary)
                fn = fp = tn = 0
            else:  # All correct negatives (no rain)
                tn = len(obs_binary)
                tp = fn = fp = 0
        else:
            # Normal case with both classes present
            tn, fp, fn, tp = confusion_matrix(obs_binary, wrf_binary, labels=[0,1]).ravel()
        
        # Calculate metrics (with edge case handling)
        pod = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        far = fp / (tp + fp) if (tp + fp) > 0 else np.nan
        csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else np.nan
        
        metrics.append({
            'Threshold': thresh,
            'POD': pod,
            'FAR': far,
            'CSI': csi,
            'Hits': tp,
            'Misses': fn,
            'False_Alarms': fp,
            'Correct_Negatives': tn
        })
    
    return pd.DataFrame(metrics)

def calculate_continuous_metrics(verif_df):
    """Calculate continuous verification metrics"""
    obs = verif_df['Observed'].values
    wrf = verif_df['WRF'].values
    
    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((wrf - obs) ** 2))
    
    # Bias (Mean Error)
    bias = np.mean(wrf - obs)
    
    # Mean Bias (same as Bias for precipitation)
    mean_bias = bias
    
    # R-squared (Coefficient of Determination)
    if np.var(obs) == 0:
        r_squared = np.nan
    else:
        r_squared = r2_score(obs, wrf)
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(wrf - obs))
    
    # Correlation coefficient
    correlation = np.corrcoef(obs, wrf)[0, 1] if len(obs) > 1 else np.nan
    
    return {
        'RMSE': rmse,
        'Bias': bias,
        'Mean_Bias': mean_bias,
        'R_squared': r_squared,
        'MAE': mae,
        'Correlation': correlation,
        'Number_of_Stations': len(verif_df)
    }

def plot_performance_diagram(metrics_df, output_path):
    """Create performance diagram"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate success ratio
    metrics_df['SR'] = 1 - metrics_df['FAR']
    
    # Create CSI contours (with division by zero handling)
    pod_grid = np.linspace(0, 1, 100)
    sr_grid = np.linspace(0, 1, 100)
    POD, SR = np.meshgrid(pod_grid, sr_grid)
    
    # Handle division by zero in CSI calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        CSI = (POD * SR) / (POD + SR - POD * SR)
        CSI = np.nan_to_num(CSI, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Plot contours
    cs = ax.contour(SR, POD, CSI, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 
                   colors='gray', linestyles='dashed')
    ax.clabel(cs, inline=True, fontsize=10)
    
    # Plot bias lines
    for bias in [0.5, 1, 2]:
        x = np.linspace(0, 1, 100)
        y = bias * x
        y[y > 1] = 1
        ax.plot(x, y, color='gray', alpha=0.5, linestyle='dotted')
        ax.text(x[-10], y[-10], f'B={bias}', ha='right', va='bottom')
    
    # Plot metrics
    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics_df)))
    for idx, row in metrics_df.iterrows():
        ax.scatter(row['SR'], row['POD'], color=colors[idx], s=100,
                  label=f"{row['Threshold']} mm")
        ax.text(row['SR']+0.02, row['POD'], f"{row['Threshold']}", 
                ha='left', va='center')
    
    # Format plot
    ax.set_xlabel('Success Ratio (1 - FAR)')
    ax.set_ylabel('Probability of Detection (POD)')
    ax.set_title(f'WRF Performance\n{metrics_df.iloc[0]["Time"].strftime("%Y-%m-%d %H:%M")}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(title='Threshold (mm)', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_scatter_comparison(verif_df, output_path):
    """Create scatter plot of WRF vs Observed"""
    plt.figure(figsize=(10, 8))
    
    # Calculate density
    x = verif_df['Observed']
    y = verif_df['WRF']
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    
    # Sort by density
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    # Create plot
    sc = plt.scatter(x, y, c=z, s=50, cmap='viridis', alpha=0.7, edgecolor='k', linewidth=0.5)
    plt.colorbar(sc, label='Point Density')
    
    # Add 1:1 line
    max_val = max(x.max(), y.max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    
    # Calculate and add statistics
    corr = np.corrcoef(x, y)[0,1]
    rmse = np.sqrt(np.mean((y - x) ** 2))
    bias = np.mean(y - x)
    r_squared = r2_score(x, y) if np.var(x) > 0 else np.nan
    
    stats_text = f"Correlation: {corr:.2f}\nRMSE: {rmse:.2f} mm\nBias: {bias:.2f} mm\nR²: {r_squared:.2f}"
    
    plt.text(0.05, 0.95, stats_text, 
             transform=plt.gca().transAxes, ha='left', va='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Format plot
    plt.xlabel('Observed Precipitation (mm)')
    plt.ylabel('WRF Precipitation (mm)')
    plt.title(f'WRF vs Observed\n{verif_df.iloc[0]["Time"].strftime("%Y-%m-%d %H:%M")}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    print(f"Processing WRF file: {WRF_FILE}")
    
    # Get WRF time from filename
    filename = os.path.basename(WRF_FILE)
    # Extract the datetime parts from your filename format "wrf_rain_accum_6hr_20220726_1800.nc"
    date_str = filename.split('_')[-2]  # Gets "20220726"
    time_str = filename.split('_')[-1].split('.')[0]  # Gets "1800"
    wrf_time = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
    
    # Step 1: Load matching station data
    print("Loading station observations...")
    station_df = load_station_data(wrf_time)
    
    # Step 2: Extract WRF values at stations
    print("Matching WRF to stations...")
    verif_df = extract_wrf_precip(WRF_FILE, station_df)
    
    # Save matched data
    matched_path = os.path.join(OUTPUT_DIR, f"matched_{wrf_time.strftime('%Y%m%d%H')}.csv")
    verif_df.to_csv(matched_path, index=False)
    print(f"Saved matched data to: {matched_path}")
    
    # Step 3: Calculate metrics
    print("Calculating verification metrics...")
    
    # Calculate categorical metrics
    categorical_metrics = calculate_categorical_metrics(verif_df, THRESHOLDS)
    categorical_metrics['Time'] = wrf_time
    
    # Calculate continuous metrics
    continuous_metrics = calculate_continuous_metrics(verif_df)
    continuous_metrics['Time'] = wrf_time
    
    # Save categorical metrics
    metrics_path = os.path.join(OUTPUT_DIR, f"categorical_metrics_{wrf_time.strftime('%Y%m%d%H')}.csv")
    categorical_metrics.to_csv(metrics_path, index=False)
    print(f"Saved categorical metrics to: {metrics_path}")
    
    # Save continuous metrics
    continuous_df = pd.DataFrame([continuous_metrics])
    continuous_path = os.path.join(OUTPUT_DIR, f"continuous_metrics_{wrf_time.strftime('%Y%m%d%H')}.csv")
    continuous_df.to_csv(continuous_path, index=False)
    print(f"Saved continuous metrics to: {continuous_path}")
    
    # Print continuous metrics summary
    print("\nContinuous Verification Metrics:")
    print(f"RMSE: {continuous_metrics['RMSE']:.2f} mm")
    print(f"Bias: {continuous_metrics['Bias']:.2f} mm")
    print(f"Mean Bias: {continuous_metrics['Mean_Bias']:.2f} mm")
    print(f"R²: {continuous_metrics['R_squared']:.2f}")
    print(f"MAE: {continuous_metrics['MAE']:.2f} mm")
    print(f"Correlation: {continuous_metrics['Correlation']:.2f}")
    print(f"Number of Stations: {continuous_metrics['Number_of_Stations']}")
    
    # Step 4: Create plots
    print("Generating verification plots...")
    
    # Performance diagram
    perf_path = os.path.join(OUTPUT_DIR, f"performance_{wrf_time.strftime('%Y%m%d%H')}.png")
    plot_performance_diagram(categorical_metrics, perf_path)
    print(f"Created performance diagram: {perf_path}")
    
    # Scatter plot
    scatter_path = os.path.join(OUTPUT_DIR, f"scatter_{wrf_time.strftime('%Y%m%d%H')}.png")
    plot_scatter_comparison(verif_df, scatter_path)
    print(f"Created scatter plot: {scatter_path}")
    
    print("\nVerification complete!")

if __name__ == "__main__":
    main()