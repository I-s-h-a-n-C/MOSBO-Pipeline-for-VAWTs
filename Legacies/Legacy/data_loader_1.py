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
    Stage 1: Scans the Cp CSVs and extracts Geometry + TSR -> Cp.
    """
    print("\n[Stage 1] Parsing Aerodynamic Datasets (Cp)...")
    parsed_data = []
    
    for filename in config.CP_FILES:
        filepath = os.path.join(config.DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Skipping.")
            continue
            
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
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
    Stage 2: Applies physical bounds and Z-score outlier detection.
    """
    initial_len = len(df)
    
    # 1. Drop NaNs and Duplicates
    df = df.drop_duplicates().dropna()
    
    # 2. Physical Bounds Filter
    if max_val is not None:
        df = df[(df[target_col] >= min_val) & (df[target_col] <= max_val)]
    else:
        df = df[df[target_col] > min_val]
        
    # 3. Z-Score Statistical Anomaly Filter
    # Only filter numeric columns to prevent errors
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < config.Z_SCORE_THRESHOLD).all(axis=1)]
    
    dropped = initial_len - len(df)
    print(f" -> Cleaned {target_col.upper()}: Kept {len(df)} rows. Dropped {dropped} anomalies/outliers.")
    return df

def load_and_clean_all():
    """Master function to execute parsing and cleaning."""
    df_cp = parse_cp_files()
    df_cp_clean = clean_dataset(df_cp, 'cp', config.CP_MIN, config.CP_MAX)
    
    df_trf = parse_azim_folders()
    df_trf_clean = clean_dataset(df_trf, 'trf', config.TRF_MIN)
    
    return df_cp_clean, df_trf_clean

if __name__ == "__main__":
    # Test execution block
    print("Testing Data Loader (Phase 1)...")
    cp_data, trf_data = load_and_clean_all()
    print("\nPhase 1 Complete.")