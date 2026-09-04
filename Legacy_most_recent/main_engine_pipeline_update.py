import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import glob
import matplotlib as mpl # Added for modern colormap handling

# Import the MOSBO 2.0 pipeline modules
import config
import data_loader
import surrogates
import hybrid_optimizer

def generate_regression_figures(builder, df, target_col, inputs):
    """
    Stage 14: Publication Figures (Actual vs Predicted & Residuals).
    Generates high-quality plots for academic review.
    """
    print(f" -> Generating Actual vs Predicted plot for {builder.target_name}...")
    
    # 1. Align data cleaning with the surrogate bounds
    df_plot = df.copy()
    if target_col.lower() == 'cp':
        df_plot = df_plot[df_plot['cp'] >= -0.20].reset_index(drop=True)
        
    # 2. AGGREGATION FIX: Eliminate horizontal "random dot" streaks.
    # The CFD data contains multiple oscillating snapshots for identical input geometries.
    # The surrogate predicts the mean, so plotting raw oscillating actuals creates horizontal noise.
    # We aggregate exact duplicates to their mean target value to show true predictive performance.
    df_agg = df_plot.groupby(inputs, as_index=False)[target_col].mean()
    
    X = df_agg[inputs].values
    y_actual = df_agg[target_col].values
    y_pred, y_sigma = builder.predict(X)
    
    # Setup Figure with 2 subplots (Scatter + Residual Histogram)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Actual vs Predicted
    # Separate error bars from markers to prevent muddy gray smudging
    ax1.errorbar(y_actual, y_pred, yerr=1.96*y_sigma, fmt='none', alpha=0.25, 
                 ecolor='gray', elinewidth=1, capsize=0, zorder=1, label='95% CI')
    
    # Crisp scatter dots on top
    ax1.scatter(y_actual, y_pred, c='dodgerblue', alpha=0.75, edgecolor='black', 
                linewidth=0.8, s=40, zorder=2, label='Predicted Mean')
    
    # 1:1 Identity Line with dynamic bounds
    min_val, max_val = min(y_actual), max(y_actual)
    margin = (max_val - min_val) * 0.05
    ax1.plot([min_val - margin, max_val + margin], 
             [min_val - margin, max_val + margin], 'k--', lw=2, zorder=3, label='Perfect Model')
    
    ax1.set_xlim(min_val - margin, max_val + margin)
    ax1.set_ylim(min_val - margin, max_val + margin)
    
    ax1.set_title(f"{builder.target_name}: Actual vs. Predicted", fontsize=14, fontweight='bold')
    ax1.set_xlabel(f"Actual {builder.target_name} (CFD Mean)", fontsize=12)
    ax1.set_ylabel(f"Predicted {builder.target_name} (GP Surrogate)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # Subplot 2: Residual Distribution
    residuals = y_actual - y_pred
    ax2.hist(residuals, bins=50, color='coral', edgecolor='black', alpha=0.8)
    ax2.axvline(0, color='k', linestyle='dashed', linewidth=2)
    ax2.set_yscale('log') # Added Log Scale to visualize outlier tails
    ax2.set_title(f"{builder.target_name}: Residual Distribution (Log Scale)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Residual Error (Actual - Predicted)", fontsize=12)
    ax2.set_ylabel("Frequency (Log)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Save the figure
    os.makedirs("figures", exist_ok=True)
    filepath = os.path.join("figures", f"{builder.target_name}_Regression_Analysis.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"    Saved: {filepath}")

def generate_ard_sensitivity_figure(builder):
    """
    Stage 14: ARD Feature Sensitivity Plot (GP Hyperparameter Analysis).
    """
    if builder.model_type != 'gp':
        print(f" -> Skipping ARD plot for {builder.target_name} (Not a GP model).")
        return

    length_scales, relevance_pct = builder.get_feature_sensitivities()
    if length_scales is None:
        print(f" -> [Warning] Could not extract ARD length scales for {builder.target_name}.")
        return

    features = [f.capitalize() for f in builder.input_cols]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Raw Length Scales (Lower = More Sensitive)
    bars1 = ax1.barh(features, length_scales, color='mediumseagreen', edgecolor='black', alpha=0.85)
    ax1.set_xlabel(r"Learned Length Scale ($\ell_i$) $\rightarrow$ [Smaller = More Sensitive]", fontsize=11)
    ax1.set_title(f"{builder.target_name}: Raw ARD Length Scales", fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5, axis='x')
    
    for bar in bars1:
        width = bar.get_width()
        ax1.text(width + (max(length_scales)*0.02), bar.get_y() + bar.get_height()/2, 
                 f"{width:.3f}", va='center', ha='left', fontsize=10, fontweight='bold')

    # 2. Normalized Relative Importance (1 / Length Scale)
    bars2 = ax2.barh(features, relevance_pct, color='royalblue', edgecolor='black', alpha=0.85)
    ax2.set_xlabel("Relative Sensitivity Weight (%)", fontsize=11)
    ax2.set_title(f"{builder.target_name}: Feature Importance Score", fontsize=13, fontweight='bold')
    ax2.set_xlim(0, max(relevance_pct) * 1.18)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='x')
    
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + 1.0, bar.get_y() + bar.get_height()/2, 
                 f"{width:.1f}%", va='center', ha='left', fontsize=10, fontweight='bold')

    plt.suptitle(f"Gaussian Process ARD Sensitivity Analysis ({builder.target_name})", fontsize=15, fontweight='bold', y=1.02)
    
    os.makedirs("figures", exist_ok=True)
    filepath = os.path.join("figures", f"{builder.target_name}_ARD_Sensitivity.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {filepath}")

def extract_raw_pareto_front(df_cp, df_trf):
    """Calculates non-dominated Pareto front from raw pre-optimization CFD dataset."""
    merged = pd.merge(df_cp, df_trf, on=['thickness', 'twist', 'solidity'], how='inner')
    if merged.empty:
        return None, None
    
    pts = merged[['trf', 'cp']].values
    is_pareto = np.ones(pts.shape[0], dtype=bool)
    for i, p in enumerate(pts):
        if is_pareto[i]:
            # A point dominates p if it has lower or equal TRF AND higher or equal Cp
            dominated = (pts[:, 0] <= p[0]) & (pts[:, 1] >= p[1]) & ((pts[:, 0] < p[0]) | (pts[:, 1] > p[1]))
            if np.any(dominated):
                is_pareto[i] = False

    df_raw_pareto = merged[is_pareto].sort_values(by='cp', ascending=False).reset_index(drop=True)
    return merged, df_raw_pareto

def generate_raw_pareto_figure(df_merged, df_raw_pareto):
    """
    Stage 14, Plot 0: Pre-Optimization Baseline Pareto Front showing all CFD simulation runs.
    Plots open blue circle scatter points and a red Pareto frontier overlay.
    """
    print(" -> Generating Pre-Optimization Raw CFD Baseline Pareto Front Plot...")
    
    if df_merged is None or df_merged.empty or df_raw_pareto is None or df_raw_pareto.empty:
        print("    [Warning] Insufficient raw CFD data to plot raw Pareto front.")
        return

    plt.figure(figsize=(10, 7))
    
    # 1. Plot all raw CFD simulation points
    plt.scatter(df_merged['trf'], df_merged['cp'], 
                facecolors='none', edgecolors='blue', linewidths=1.2, s=35, 
                alpha=0.6, label='Raw CFD Simulations')
    
    # 2. Sort Pareto points ascending by TRF for clean line connectivity
    df_raw_sorted = df_raw_pareto.sort_values(by='trf', ascending=True)
    
    # 3. Red non-dominated Pareto frontier line tracing the boundary
    plt.plot(df_raw_sorted['trf'], df_raw_sorted['cp'], 'r-', lw=2.5, 
             label='Baseline Non-Dominated Pareto Front')
    plt.scatter(df_raw_sorted['trf'], df_raw_sorted['cp'], 
                color='red', s=40, zorder=5, label='Pareto Optimal Runs')
    
    plt.title("Pre-Optimization Baseline Pareto Front (Raw CFD Dataset)", fontsize=15, fontweight='bold')
    plt.xlabel(r"Torque Ripple Factor (TRF) $\leftarrow$ Lower is Better", fontsize=12)
    plt.ylabel(r"Power Coefficient ($C_p$) $\rightarrow$ Higher is Better", fontsize=12)
    
    # Dynamic y-axis scaling
    all_cp = df_merged['cp'].values
    cp_min, cp_max = np.min(all_cp), np.max(all_cp)
    cp_margin = max((cp_max - cp_min) * 0.10, 0.005)
    plt.ylim(cp_min - cp_margin, cp_max + cp_margin)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    
    os.makedirs("figures", exist_ok=True)
    filepath = os.path.join("figures", "Pareto_Front_Pre_Optimization_Raw.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"    Saved: {filepath}")

def generate_optimized_pareto_figure(df_pareto):
    """
    Stage 14, Plot 1: Plots the Risk-Aware NSGA-II Optimized Pareto Front colored by prediction uncertainty.
    """
    print(" -> Generating Uncertainty-Colored NSGA-II Pareto Front Plot...")
    
    plt.figure(figsize=(10, 7))
    
    scatter = plt.scatter(df_pareto['Pred_TRF'], df_pareto['Pred_Cp'], 
                          c=df_pareto['Sigma_Cp'], cmap='viridis', 
                          s=80, alpha=0.9, edgecolor='k', label='Post-NSGA-II Optimized Pareto')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label(r'Prediction Uncertainty ($\sigma_{C_p}$)', fontsize=12)
    
    # Highlight extreme designs
    min_trf_idx = df_pareto['Pred_TRF'].idxmin()
    max_cp_idx = df_pareto['Pred_Cp'].idxmax()
    
    plt.scatter(df_pareto.loc[min_trf_idx, 'Pred_TRF'], df_pareto.loc[min_trf_idx, 'Pred_Cp'], 
                facecolors='none', edgecolors='red', s=250, lw=2, label='Min Vibration (TRF)')
                
    plt.scatter(df_pareto.loc[max_cp_idx, 'Pred_TRF'], df_pareto.loc[max_cp_idx, 'Pred_Cp'], 
                facecolors='none', edgecolors='blue', s=250, lw=2, label='Max Efficiency ($C_p$)')
    
    plt.title("Risk-Aware NSGA-II Pareto Front", fontsize=15, fontweight='bold')
    plt.xlabel(r"Torque Ripple Factor (TRF) $\leftarrow$ Lower is Better", fontsize=12)
    plt.ylabel(r"Power Coefficient ($C_p$) $\rightarrow$ Higher is Better", fontsize=12)
    
    # Dynamic y-axis scaling based strictly on optimal points
    all_cp = df_pareto['Pred_Cp'].values
    cp_min, cp_max = np.min(all_cp), np.max(all_cp)
    cp_margin = max((cp_max - cp_min) * 0.10, 0.005)
    plt.ylim(cp_min - cp_margin, cp_max + cp_margin)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    
    os.makedirs("figures", exist_ok=True)
    filepath = os.path.join("figures", "Pareto_Front_Optimized.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"    Saved: {filepath}")

def generate_pareto_comparison_figure(df_pareto, df_raw_pareto=None):
    """
    Stage 14, Plot 2: Plots Pre-NSGA-II Baseline (Raw CFD) vs Post-NSGA-II Optimized Pareto Front.
    """
    print(" -> Generating Pre vs Post NSGA-II Dual Pareto Front Comparison Plot...")
    
    plt.figure(figsize=(10, 7))
    
    # 1. Pre-NSGA-II Pareto Front (Baseline Raw CFD)
    if df_raw_pareto is not None and not df_raw_pareto.empty:
        plt.plot(df_raw_pareto['trf'], df_raw_pareto['cp'], 'o--', color='tab:gray', 
                 linewidth=1.5, markersize=7, alpha=0.75, label='Pre-NSGA-II Baseline Pareto (Raw CFD)')
    
    # 2. Post-NSGA-II Pareto Front (Optimized)
    plt.plot(df_pareto['Pred_TRF'], df_pareto['Pred_Cp'], 's-', color='tab:blue', 
             linewidth=2, markersize=6, alpha=0.9, label='Post-NSGA-II Optimized Pareto')
    
    # Highlight extreme designs
    min_trf_idx = df_pareto['Pred_TRF'].idxmin()
    max_cp_idx = df_pareto['Pred_Cp'].idxmax()
    
    plt.scatter(df_pareto.loc[min_trf_idx, 'Pred_TRF'], df_pareto.loc[min_trf_idx, 'Pred_Cp'], 
                facecolors='none', edgecolors='red', s=250, lw=2, label='Min Vibration (TRF)')
                
    plt.scatter(df_pareto.loc[max_cp_idx, 'Pred_TRF'], df_pareto.loc[max_cp_idx, 'Pred_Cp'], 
                facecolors='none', edgecolors='blue', s=250, lw=2, label='Max Efficiency ($C_p$)')
    
    plt.title("Pre-Optimization vs Risk-Aware NSGA-II Pareto Front", fontsize=15, fontweight='bold')
    plt.xlabel(r"Torque Ripple Factor (TRF) $\leftarrow$ Lower is Better", fontsize=12)
    plt.ylabel(r"Power Coefficient ($C_p$) $\rightarrow$ Higher is Better", fontsize=12)
    
    # Combined dynamic y-axis scaling
    if df_raw_pareto is not None and not df_raw_pareto.empty:
        all_cp = np.concatenate([df_pareto['Pred_Cp'].values, df_raw_pareto['cp'].values])
    else:
        all_cp = df_pareto['Pred_Cp'].values
        
    cp_min, cp_max = np.min(all_cp), np.max(all_cp)
    cp_margin = max((cp_max - cp_min) * 0.10, 0.005)
    plt.ylim(cp_min - cp_margin, cp_max + cp_margin)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    
    os.makedirs("figures", exist_ok=True)
    filepath = os.path.join("figures", "Pareto_Front_Comparison.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"    Saved: {filepath}")

def generate_historical_pareto_figure(current_df, history_dir):
    """
    Stage 14, Plot 3: Scans historical runs and plots their Pareto fronts as a background 
    to visually track optimization improvements over successive pipeline executions.
    """
    print(" -> Generating Historical Pareto Evolution Plot...")
    
    plt.figure(figsize=(10, 7))
    
    # 1. Load, sort, and plot all historical runs
    # String sorting automatically orders the YYYYMMDD_HHMMSS timestamps chronologically
    history_files = sorted(glob.glob(os.path.join(history_dir, "*.csv")))
    
    all_valid_cp = []
    
    for i, file in enumerate(history_files):
        try:
            hist_df = pd.read_csv(file)
            if 'Pred_TRF' in hist_df.columns and 'Pred_Cp' in hist_df.columns:
                
                # Filter out early initialization noise (very low Cp outputs)
                valid_idx = hist_df['Pred_Cp'] >= 0.35
                if not valid_idx.any():
                    continue
                    
                all_valid_cp.extend(hist_df.loc[valid_idx, 'Pred_Cp'].tolist())
                
                # Plot historical runs identically with fixed 25% opacity and size 80
                if 'Sigma_Cp' in hist_df.columns:
                    plt.scatter(hist_df.loc[valid_idx, 'Pred_TRF'], hist_df.loc[valid_idx, 'Pred_Cp'], 
                                c=hist_df.loc[valid_idx, 'Sigma_Cp'], cmap='viridis', 
                                s=80, alpha=0.25, edgecolor='none')
                else:
                    # Fallback if historical run doesn't have Sigma_Cp
                    plt.scatter(hist_df.loc[valid_idx, 'Pred_TRF'], hist_df.loc[valid_idx, 'Pred_Cp'], 
                                color='gray', s=80, alpha=0.25, edgecolor='none')
                    
        except Exception:
            pass # Ignore corrupted or unrelated CSVs
            
    # 2. Plot the current latest run matching the historical plot style
    valid_idx_curr = current_df['Pred_Cp'] >= 0.20
    all_valid_cp.extend(current_df.loc[valid_idx_curr, 'Pred_Cp'].tolist())
    
    scatter = plt.scatter(current_df.loc[valid_idx_curr, 'Pred_TRF'], current_df.loc[valid_idx_curr, 'Pred_Cp'], 
                          c=current_df.loc[valid_idx_curr, 'Sigma_Cp'], cmap='viridis', 
                          s=80, alpha=0.25, edgecolor='none')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label(r'Prediction Uncertainty ($\sigma_{C_p}$)', fontsize=12)
             
    plt.title("Evolution of Optimal VAWT Designs Over Successive Runs", fontsize=15, fontweight='bold')
    plt.xlabel(r"Torque Ripple Factor (TRF) $\leftarrow$ Lower is Better", fontsize=12)
    plt.ylabel(r"Power Coefficient ($C_p$) $\rightarrow$ Higher is Better", fontsize=12)
    
    # Set dynamic y-axis bound based strictly on valid points
    if all_valid_cp:
        cp_min, cp_max = np.min(all_valid_cp), np.max(all_valid_cp)
        cp_margin = max((cp_max - cp_min) * 0.10, 0.005)
        plt.ylim(cp_min - cp_margin, cp_max + cp_margin)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    os.makedirs("figures", exist_ok=True)
    filepath = os.path.join("figures", "Pareto_Historical_Evolution.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"    Saved: {filepath}")

def generate_hypervolume_figure(hv_history):
    """
    Stage 14, Plot 5: Plots Hypervolume convergence history across NSGA-II generations
    to prove optimizer health, progression, and algorithmic stability to reviewers.
    """
    print(" -> Generating Hypervolume Convergence Plot...")
    if not hv_history:
        print("    [Warning] No hypervolume history available to plot.")
        return

    plt.figure(figsize=(9, 6))
    generations = np.arange(1, len(hv_history) + 1)
    
    plt.plot(generations, hv_history, color='purple', lw=2.5, label='Pareto Front Hypervolume')
    plt.fill_between(generations, hv_history, color='purple', alpha=0.15)
    
    plt.title("NSGA-II Optimization Convergence (Hypervolume Indicator)", fontsize=14, fontweight='bold')
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Hypervolume (HV)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    
    os.makedirs("figures", exist_ok=True)
    filepath = os.path.join("figures", "Hypervolume_Convergence.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"    Saved: {filepath}")

def generate_parallel_coordinates_figure(df_pareto):
    """
    Stage 14, Plot 4: Parallel Coordinates Plot.
    Visualizes the high-dimensional mapping from design variables to objectives.
    """
    print(" -> Generating Parallel Coordinates Plot (Design Space)...")
    
    # Define the sequence of axes to plot (Inputs -> Outputs)
    cols = ['Thickness', 'Twist', 'Solidity', 'TSR', 'Pred_TRF', 'Pred_Cp']
    display_labels = ['Thickness', 'Twist', 'Solidity', 'TSR', 'Predicted TRF', 'Predicted Cp']
    df_plot = df_pareto[cols].copy()
    
    # Min-Max normalize each column to [0, 1] so they can share a single Y-axis
    min_vals = df_plot.min()
    max_vals = df_plot.max()
    df_norm = (df_plot - min_vals) / (max_vals - min_vals + 1e-9)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create a colormap based on the primary objective (Pred_Cp)
    cmap = plt.colormaps['viridis']
    norm = plt.Normalize(df_plot['Pred_Cp'].min(), df_plot['Pred_Cp'].max())
    
    # Plot each Pareto-optimal design as a continuous line across the axes
    x = range(len(cols))
    for i in range(len(df_norm)):
        y = df_norm.iloc[i].values
        color = cmap(norm(df_plot.iloc[i]['Pred_Cp']))
        ax.plot(x, y, color=color, alpha=0.6, linewidth=1.5)
        
    # Draw vertical black lines to represent each axis
    for i in x:
        ax.axvline(i, color='black', linestyle='-', linewidth=1.2, alpha=0.5)
        
    # Format the X-axis with the variable names
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=12, fontweight='bold')
    
    # Hide the default normalized Y-axis, we will add absolute labels instead
    ax.set_yticks([])
    
    # Add absolute min/max text labels at the top and bottom of each vertical axis
    for i, col in enumerate(cols):
        ax.text(i, 1.02, f"{max_vals[col]:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(i, -0.02, f"{min_vals[col]:.3f}", ha='center', va='top', fontsize=10, fontweight='bold')
        
    # Add the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r'Predicted Efficiency ($C_p$)', fontsize=12, fontweight='bold')
    
    plt.title("Parallel Coordinates: High-Dimensional Pareto Design Space", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    os.makedirs("figures", exist_ok=True)
    filepath = os.path.join("figures", "Parallel_Coordinates_Design_Space.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"    Saved: {filepath}")

def generate_contour_heatmaps(cp_builder, trf_builder, df_cp, fixed_twist=0.0, fixed_tsr=2.5):
    """
    Stage 14: Generates side-by-side 2D Contour Heatmaps for Cp and TRF 
    across NACA Thickness vs Solidity. Remaining variables are held constant.
    """
    print(" -> Generating 2D Surrogate Contour Heatmaps...")
    
    # 1. Generate a dense 2D meshgrid for Thickness and Solidity
    thickness_range = np.linspace(df_cp['thickness'].min(), df_cp['thickness'].max(), 100)
    solidity_range = np.linspace(df_cp['solidity'].min(), df_cp['solidity'].max(), 100)
    T, S = np.meshgrid(thickness_range, solidity_range)
    
    grid_shape = T.shape
    T_flat = T.ravel()
    S_flat = S.ravel()
    
    # 2. Fix non-plotted variables to constant scalars
    twist_flat = np.full_like(T_flat, fixed_twist)
    tsr_flat = np.full_like(T_flat, fixed_tsr)
    
    # 3. Stack inputs into expected feature order for each surrogate
    X_cp_grid = np.column_stack([T_flat, twist_flat, S_flat, tsr_flat])  # [thickness, twist, solidity, tsr]
    X_trf_grid = np.column_stack([T_flat, twist_flat, S_flat])           # [thickness, twist, solidity]
    
    # 4. Predict values using trained surrogate models
    cp_pred, _ = cp_builder.predict(X_cp_grid)
    trf_pred, _ = trf_builder.predict(X_trf_grid)
    
    Z_cp = cp_pred.reshape(grid_shape)
    Z_trf = trf_pred.reshape(grid_shape)
    
    # 5. Render side-by-side contour plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Aerodynamic Power Coefficient (Cp) Contour
    c1 = ax1.contourf(T, S, Z_cp, levels=30, cmap='viridis')
    fig.colorbar(c1, ax=ax1, label=r'Power Coefficient ($C_p$)')
    ax1.set_title(f"Aerodynamic Efficiency ($C_p$)\nFixed Twist={fixed_twist}°, TSR={fixed_tsr}", fontsize=12, fontweight='bold')
    ax1.set_xlabel("NACA Profile Thickness", fontsize=11)
    ax1.set_ylabel("Solidity", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Structural Vibration (TRF) Contour
    c2 = ax2.contourf(T, S, Z_trf, levels=30, cmap='plasma')
    fig.colorbar(c2, ax=ax2, label='Torque Ripple Factor (TRF)')
    ax2.set_title(f"Structural Vibration (TRF)\nFixed Twist={fixed_twist}°", fontsize=12, fontweight='bold')
    ax2.set_xlabel("NACA Profile Thickness", fontsize=11)
    ax2.set_ylabel("Solidity", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Save the figure
    os.makedirs("figures", exist_ok=True)
    filepath = os.path.join("figures", "Surrogate_Contour_Heatmaps.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"    Saved: {filepath}")

def virtual_wind_tunnel(cp_surrogate, trf_surrogate):
    """
    Stage 13: Interactive loop allowing engineers to query the GP model directly,
    complete with 95% Confidence Intervals.
    """
    while True:
        print("\n" + "="*60)
        print("          🌪️  MOSBO 2.0 VIRTUAL WIND TUNNEL  🌪️          ")
        print("="*60)
        print("Type 'exit' to quit at any prompt.")
        
        try:
            thick_in = input("1. Enter Max Thickness (e.g., 15 for 0015): ").strip().lower()
            if thick_in == 'exit': break
            thickness = float(thick_in)
            
            twist_in = input("2. Enter Twist Angle (Degrees): ").strip().lower()
            if twist_in == 'exit': break
            twist = float(twist_in)
            
            sol_in = input("3. Enter Solidity (e.g., 0.15 - 0.45): ").strip().lower()
            if sol_in == 'exit': break
            solidity = float(sol_in)
            
            tsr_in = input("4. Enter Tip Speed Ratio (TSR): ").strip().lower()
            if tsr_in == 'exit': break
            tsr = float(tsr_in)
            
            # Formatting for prediction
            x_cp = np.array([[thickness, twist, solidity, tsr]])
            x_trf = np.array([[thickness, twist, solidity]])
            
            # Predict with Uncertainty
            cp_val, cp_sig = cp_surrogate.predict(x_cp)
            trf_val, trf_sig = trf_surrogate.predict(x_trf)
            
            # Extract scalars
            cp, s_cp = cp_val[0], cp_sig[0]
            trf, s_trf = trf_val[0], trf_sig[0]
            
            # Confidence Logic
            cp_conf = "High" if s_cp < 0.015 else ("Medium" if s_cp < 0.03 else "Low (Needs CFD Validation)")
            
            print("\n" + "-"*40)
            print("🚀 PREDICTION RESULTS")
            print("-"*40)
            print(f"Predicted Cp  : {cp:.4f} ± {s_cp:.4f}")
            print(f" └ 95% CI     : [{cp - 1.96*s_cp:.4f} to {cp + 1.96*s_cp:.4f}]")
            print(f" └ Confidence : {cp_conf}")
            print("")
            print(f"Predicted TRF : {trf:.4f} ± {s_trf:.4f}")
            print(f" └ 95% CI     : [{max(0, trf - 1.96*s_trf):.4f} to {trf + 1.96*s_trf:.4f}]")
            print("-"*40)
            
        except ValueError:
            print("\n[Error] Invalid numerical input. Please try again.")

def main():
    print("="*60)
    print("     MOSBO 2.0 : Multi-Objective Surrogate Optimization")
    print("="*60)
    
    # 1. Ingestion & Cleaning
    df_cp, df_trf = data_loader.load_and_clean_all()
    if df_cp.empty or df_trf.empty:
        print("\n[Error] Failed to load sufficient data. Please check data files and config.py bounds.")
        sys.exit(1)
        
    # 2. Gaussian Process Training & Validation
    cp_builder, trf_builder = surrogates.build_surrogates(df_cp, df_trf)
    
    # 3. Generate Diagnostics & Publication Figures
    print("\n[Stage 14] Generating Publication Figures...")
    generate_regression_figures(cp_builder, df_cp, 'cp', ['thickness', 'twist', 'solidity', 'tsr'])
    generate_regression_figures(trf_builder, df_trf, 'trf', ['thickness', 'twist', 'solidity'])
    generate_contour_heatmaps(cp_builder, trf_builder, df_cp, fixed_twist=df_cp['twist'].mean(), fixed_tsr=df_cp['tsr'].mean())
    generate_ard_sensitivity_figure(cp_builder)
    generate_ard_sensitivity_figure(trf_builder)
    
    # 4. Risk-Aware Multi-Objective Optimization
    df_pareto, hv_history = hybrid_optimizer.run_optimization(cp_builder, trf_builder, df_cp)
    
    # 5. Export and Plot Pareto Fronts
    if df_pareto is not None:
        os.makedirs("output", exist_ok=True)
        
        # --- Historical Archiving Logic ---
        history_dir = os.path.join("output", "history")
        os.makedirs(history_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        hist_path = os.path.join(history_dir, f"Pareto_{timestamp}.csv")
        df_pareto.to_csv(hist_path, index=False)
        print(f"\n -> 💾 Pareto front archived to history: {hist_path}")
        # ---------------------------------------
        
        pareto_path = os.path.join("output", "Optimal_Pareto_Designs.csv")
        df_pareto.to_csv(pareto_path, index=False)
        print(f" -> 💾 Latest Pareto front exported to: {pareto_path}")
        
        df_merged_raw, df_raw_pareto = extract_raw_pareto_front(df_cp, df_trf)
        generate_raw_pareto_figure(df_merged_raw, df_raw_pareto)
        generate_optimized_pareto_figure(df_pareto)
        generate_pareto_comparison_figure(df_pareto, df_raw_pareto)
        generate_historical_pareto_figure(df_pareto, history_dir)
        generate_parallel_coordinates_figure(df_pareto)
        generate_hypervolume_figure(hv_history)
    
    # 6. Enter Interactive Mode
    virtual_wind_tunnel(cp_builder, trf_builder)

if __name__ == "__main__":
    # Wrap in a try-except to catch manual keyboard interrupts cleanly
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user. Exiting MOSBO 2.0 gracefully.")
        sys.exit(0)