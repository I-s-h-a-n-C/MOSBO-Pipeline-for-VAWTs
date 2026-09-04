import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the MOSBO 2.0 pipeline modules
import config
import data_loader
import surrogates
import optimizer

def generate_regression_figures(builder, df, target_col, inputs):
    """
    Stage 14: Publication Figures (Actual vs Predicted & Residuals).
    Generates high-quality plots for academic review.
    """
    print(f" -> Generating Actual vs Predicted plot for {builder.target_name}...")
    
    X = df[inputs].values
    y_actual = df[target_col].values
    y_pred, y_sigma = builder.predict(X)
    
    # Setup Figure with 2 subplots (Scatter + Residual Histogram)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Actual vs Predicted
    ax1.errorbar(y_actual, y_pred, yerr=1.96*y_sigma, fmt='o', alpha=0.4, 
                 ecolor='lightgray', elinewidth=1, capsize=0, label='Predictions (95% CI)')
    ax1.scatter(y_actual, y_pred, c='blue', alpha=0.6, edgecolor='k', s=40)
    
    # 1:1 Identity Line
    min_val, max_val = min(y_actual), max(y_actual)
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Model')
    
    ax1.set_title(f"{builder.target_name}: Actual vs. Predicted", fontsize=14, fontweight='bold')
    ax1.set_xlabel(f"Actual {builder.target_name} (CFD)", fontsize=12)
    ax1.set_ylabel(f"Predicted {builder.target_name} (GP Surrogate)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Subplot 2: Residual Distribution
    residuals = y_actual - y_pred
    ax2.hist(residuals, bins=30, color='coral', edgecolor='black', alpha=0.8)
    ax2.axvline(0, color='k', linestyle='dashed', linewidth=2)
    ax2.set_title(f"{builder.target_name}: Residual Distribution", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Residual Error (Actual - Predicted)", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Save the figure
    os.makedirs("figures", exist_ok=True)
    filepath = os.path.join("figures", f"{builder.target_name}_Regression_Analysis.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
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
    
    # 1. Plot all raw CFD simulation points (open blue circles as in reference image)
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
    
    # Set upper bound for Cp axis
    plt.ylim(top=0.4)
    
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
    
    # Set y-axis upper bound to 0.4
    plt.ylim(top=0.4)
    
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
    
    # Set y-axis upper bound to 0.4
    plt.ylim(top=0.4)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    
    os.makedirs("figures", exist_ok=True)
    filepath = os.path.join("figures", "Pareto_Front_Comparison.png")
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
    
    # 4. Risk-Aware Multi-Objective Optimization
    df_pareto = optimizer.run_optimization(cp_builder, trf_builder, df_cp)
    
    # 5. Export and Plot Pareto Fronts
    if df_pareto is not None:
        os.makedirs("output", exist_ok=True)
        pareto_path = os.path.join("output", "Optimal_Pareto_Designs.csv")
        df_pareto.to_csv(pareto_path, index=False)
        print(f"\n -> 💾 Pareto front exported to: {pareto_path}")
        
        df_merged_raw, df_raw_pareto = extract_raw_pareto_front(df_cp, df_trf)
        generate_raw_pareto_figure(df_merged_raw, df_raw_pareto)
        generate_optimized_pareto_figure(df_pareto)
        generate_pareto_comparison_figure(df_pareto, df_raw_pareto)
    
    # 6. Enter Interactive Mode
    virtual_wind_tunnel(cp_builder, trf_builder)

if __name__ == "__main__":
    # Wrap in a try-except to catch manual keyboard interrupts cleanly
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user. Exiting MOSBO 2.0 gracefully.")
        sys.exit(0)