import os
import re
import glob
import numpy as np
import pandas as pd
from scipy import stats
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, **kwargs): return iterable

import config

def parse_cp_files():
    """
    Stage 1: Scans the Cp folders and extracts Geometry + TSR -> Cp.
    """
    print("\n[Stage 1] Parsing Aerodynamic Datasets (Cp)...")
    parsed_data = []
    
    for folder in config.CP_FOLDERS:
        folder_path = os.path.join(config.DATA_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found. Skipping.")
            continue
            
        # Get all files within the current folder
        files = glob.glob(os.path.join(folder_path, "*"))
        
        for filepath in tqdm(files, desc=f"Processing {folder}"):
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue
                
            # If a file is empty or too short, skip it
            if len(lines) < 3:
                continue

            header_idx = next((i for i, line in enumerate(lines) if 'Circ' in line or 'Twist' in line), 0)
            sim_names = lines[header_idx].strip().split(';')
            
            for line in lines[header_idx + 2:]:
                vals = line.strip().split(';')
                if len(vals) < 2: continue
                    
                for i, sim_name in enumerate(sim_names):
                    match = re.search(r'NACA_?\s*00(\d{2}).*?Circ(\d+).*?Sol([\d.]+)', sim_name)
                    if match:
                        thickness = float(match.group(1))
                        twist = float(match.group(2))
                        solidity = float(match.group(3).rstrip('.'))
                        
                        tsr_idx, cp_idx = 2 * i, 2 * i + 1
                        if cp_idx < len(vals):
                            try:
                                tsr_val = float(vals[tsr_idx].strip())
                                cp_val = float(vals[cp_idx].strip())
                                parsed_data.append([thickness, twist, solidity, tsr_val, cp_val])
                            except ValueError:
                                pass

    df = pd.DataFrame(parsed_data, columns=['thickness', 'twist', 'solidity', 'tsr', 'cp'])
    return df

def extract_steady_state_trf(filepath):
    """Calculates TRF from the final 25% of time-series data."""
    try:
        df_time = pd.read_csv(filepath, sep=';', skiprows=2, on_bad_lines='skip')
        if len(df_time.columns) < 2:
            df_time = pd.read_csv(filepath, sep=',', skiprows=2, on_bad_lines='skip')
        if not any(c for c in df_time.columns if 'Torque' in c or 'Moment' in c):
            df_time = pd.read_csv(filepath, sep=';', skiprows=0, on_bad_lines='skip')
    except Exception:
        return np.nan

    torque_col_list = [col for col in df_time.columns if 'Torque' in col or 'Moment' in col]
    if not torque_col_list: return np.nan
        
    torque_col = torque_col_list[0]
    df_time[torque_col] = pd.to_numeric(df_time[torque_col], errors='coerce')
    df_time = df_time.dropna(subset=[torque_col])

    if df_time.empty: return np.nan

    cutoff_idx = int(len(df_time) * 0.75)
    steady_df = df_time.iloc[cutoff_idx:]
    
    if steady_df.empty: return np.nan
        
    T_max, T_min, T_avg = steady_df[torque_col].max(), steady_df[torque_col].min(), steady_df[torque_col].mean()
    
    if T_avg == 0: return np.nan
    return (T_max - T_min) / abs(T_avg)

def parse_azim_folders():
    """
    Stage 1: Scans the Azim folders and calculates TRF.
    """
    print("\n[Stage 1] Parsing Structural Datasets (TRF)...")
    trf_data = []
    
    for folder in config.AZIM_FOLDERS:
        folder_path = os.path.join(config.DATA_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found.")
            continue
            
        files = glob.glob(os.path.join(folder_path, "*"))
        for file in tqdm(files, desc=f"Processing {folder}"):
            match = re.search(r'NACA_?\s*00(\d{2}).*?Circ(\d+).*?Sol([\d.]+)', os.path.basename(file))
            if match:
                thickness, twist, solidity = float(match.group(1)), float(match.group(2)), float(match.group(3).rstrip('.'))
                trf = extract_steady_state_trf(file)
                if not np.isnan(trf):
                    trf_data.append([thickness, twist, solidity, trf])
                    
    df = pd.DataFrame(trf_data, columns=['thickness', 'twist', 'solidity', 'trf'])
    return df

def clean_dataset(df, target_col, min_val, max_val=None):
    """
    Stage 2: Applies physical bounds and Boundary-Preserving Z-score outlier detection.
    """
    initial_len = len(df)
    
    # 1. Drop NaNs and Duplicates based on the target
    df = df.drop_duplicates().dropna(subset=[target_col])
    
    # 2. Physical Bounds Filter
    if max_val is not None:
        df = df[(df[target_col] >= min_val) & (df[target_col] <= max_val)]
    else:
        df = df[df[target_col] >= min_val]
        
    # 3. Boundary-Preserving Z-Score Statistical Anomaly Filter
    if len(df) > 0:
        mean_val = df[target_col].mean()
        std_val = df[target_col].std()
        
        if std_val > 0:
            z_scores = np.abs((df[target_col] - mean_val) / std_val)
            # Use threshold from config if available, otherwise default to 3.0
            threshold = getattr(config, 'Z_SCORE_THRESHOLD', 3.0)
            is_valid_z = z_scores < threshold
            
            # HARD OVERRIDE: Never drop the absolute minimums and maximums (crucial for Pareto!)
            is_min = df[target_col] == df[target_col].min()
            is_max = df[target_col] == df[target_col].max()

            # DIAGNOSTIC (FIX): This override can force-keep a min/max row even
            # when it's a severe z-score outlier (e.g. a parser artifact). That
            # single row can skew scaler_y (mean/std used to normalize the
            # target) and sits in sparse input space, so the GP naturally
            # assigns it high sigma -- one likely source of the long
            # uncertainty whiskers on the regression plot. We still keep the
            # boundary rows (Pareto needs them), but now flag them so it's
            # visible when they're worth a manual sanity check.
            forced_outliers = (is_min | is_max) & ~is_valid_z
            if forced_outliers.any():
                print(f" -> ⚠️  {target_col.upper()}: {forced_outliers.sum()} boundary row(s) "
                      f"(min/max) forced past the z-score filter -- verify these aren't parser artifacts:")
                cols_to_show = [c for c in ['thickness', 'twist', 'solidity', 'tsr', target_col] if c in df.columns]
                print(df.loc[forced_outliers, cols_to_show].to_string(index=False))

            # Keep row if it's statistically valid OR if it's an extreme boundary condition
            df = df[is_valid_z | is_min | is_max]
    
    dropped = initial_len - len(df)
    print(f" -> Cleaned {target_col.upper()}: Kept {len(df)} rows. Dropped {dropped} anomalies/outliers.")
    return df

def load_and_clean_all():
    """Master function to execute parsing, caching, and cleaning."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    cp_cache = os.path.join(config.DATA_DIR, "parsed_cp_cache.csv")
    trf_cache = os.path.join(config.DATA_DIR, "parsed_trf_cache.csv")
    
    # 1. Load or Parse Aerodynamic (Cp) Data
    if os.path.exists(cp_cache):
        print(f"\n[Stage 1] ⚡ Loading FAST cached Aerodynamic Datasets (Cp) from {cp_cache}...")
        df_cp = pd.read_csv(cp_cache)
    else:
        df_cp = parse_cp_files()
        if not df_cp.empty:
            df_cp.to_csv(cp_cache, index=False)
            print(f" -> 💾 Saved parsed Cp data to cache: {cp_cache}")
        
    # 2. Load or Parse Structural (TRF) Data
    if os.path.exists(trf_cache):
        print(f"\n[Stage 1] ⚡ Loading FAST cached Structural Datasets (TRF) from {trf_cache}...")
        df_trf = pd.read_csv(trf_cache)
    else:
        df_trf = parse_azim_folders()
        if not df_trf.empty:
            df_trf.to_csv(trf_cache, index=False)
            print(f" -> 💾 Saved parsed TRF data to cache: {trf_cache}")
        
    # 3. Clean Datasets
    # Applied fresh every run so changes to config physical bounds take effect instantly
    df_cp_clean = pd.DataFrame()
    df_trf_clean = pd.DataFrame()
    
    if not df_cp.empty:
        df_cp_clean = clean_dataset(df_cp, 'cp', config.CP_MIN, config.CP_MAX)
    if not df_trf.empty:
        # Assuming config.TRF_MAX doesn't exist by default, setting it to None
        df_trf_clean = clean_dataset(df_trf, 'trf', config.TRF_MIN, getattr(config, 'TRF_MAX', None))
    
    return df_cp_clean, df_trf_clean

if __name__ == "__main__":
    # Test execution block
    print("Testing Data Loader (Phase 1)...")
    cp_data, trf_data = load_and_clean_all()
    print("\nPhase 1 Complete.")