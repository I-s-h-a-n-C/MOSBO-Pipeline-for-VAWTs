import os
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

import config


class SurrogateBuilder:
    """
    Handles Stage 3-8 of the MOSBO Pipeline:
    - Stage 3: Train/Test Splitting
    - Stage 4: Feature Scaling (StandardScaler)
    - Stage 5: Automated Kernel Search & Hyperparameter Tuning
    - Stage 6: Hold-out Test Set Evaluation & Diagnostics
    - Stage 7: K-Fold Cross Validation
    - Stage 8: Uncertainty Quantification & Model Persistence
    """
    def __init__(self, target_name, input_cols):
        self.target_name = target_name.upper()
        self.input_cols = input_cols
        
        # Feature and target scalers for Stage 4
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.model = None
        self.best_kernel_name = None
        self.training_time_sec = 0.0

    def prepare_data(self, df, target_col):
        """Stage 3 & 4: Data Partitioning and Standardization."""
        print(f"\n==================================================")
        print(f" 📊 [{self.target_name}] Stage 3 & 4: Data Preparation")
        print(f"==================================================")
        print(f" -> Target Variable     : '{target_col}'")
        print(f" -> Feature Columns ({len(self.input_cols)}): {self.input_cols}")
        print(f" -> Total Samples Input : {len(df)}")
        
        X = df[self.input_cols].values
        y = df[target_col].values.reshape(-1, 1)
        
        # Stage 3: Train / Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.TRAIN_TEST_SPLIT, 
            random_state=config.RANDOM_STATE
        )
        print(f" -> Split Ratio         : {(1.0 - config.TRAIN_TEST_SPLIT)*100:.0f}% Train / {config.TRAIN_TEST_SPLIT*100:.0f}% Test")
        print(f" -> Raw Train Shape     : {X_train.shape}")
        print(f" -> Raw Test Shape      : {X_test.shape}")
        
        # Cap large dataset size to avoid O(N^3) Gaussian Process matrix inversion overhead
        MAX_GP_SAMPLES = 2500
        if len(X_train) > MAX_GP_SAMPLES:
            np.random.seed(config.RANDOM_STATE)
            sub_idx = np.random.choice(len(X_train), size=MAX_GP_SAMPLES, replace=False)
            X_train = X_train[sub_idx]
            y_train = y_train[sub_idx]
            print(f" -> ⚡ Subsampled training set to {MAX_GP_SAMPLES} points to optimize matrix inversion speed.")

        # Stage 4: Scale Data (Essential for GP numerical stability)
        print(f" -> Fitting StandardScaler on features and targets...")
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train).flatten()
        
        X_test_scaled = self.scaler_X.transform(X_test)
        y_test_scaled = self.scaler_y.transform(y_test).flatten()
        
        print(f" -> Target Mean (Train) : {self.scaler_y.mean_[0]:.4f} ± {self.scaler_y.scale_[0]:.4f}")
        print(f" -> Data preparation complete.")
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X, y

    def train_with_kernel_search(self, X_train_scaled, y_train_scaled):
        """Stage 5: Train GP and automatically select the best kernel."""
        start_time = time.time()
        print(f"[{self.target_name}] Running Automated Kernel Search...")
        
        # Increased length scale upper bound 100x (from 1e3 to 1e5) for k2__length_scale hyperparameter flexibility
        bounds = (1e-3, 1e5)
        dims = X_train_scaled.shape[1]
        
        # Candidate Kernels
        kernels = {
            "RBF": C(1.0) * RBF(length_scale=[1.0] * dims, length_scale_bounds=bounds),
            "Matern 3/2": C(1.0) * Matern(length_scale=[1.0] * dims, length_scale_bounds=bounds, nu=1.5),
            "Matern 5/2": C(1.0) * Matern(length_scale=[1.0] * dims, length_scale_bounds=bounds, nu=2.5),
            "Rational Quadratic": C(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=bounds)
        }
        
        best_score = -np.inf
        best_model = None
        kernel_results = {}
        
        # Internal hold-out validation set for hyperparameter comparison
        X_t, X_val, y_t, y_val = train_test_split(
            X_train_scaled, y_train_scaled, test_size=0.2, random_state=config.RANDOM_STATE
        )
        print(f" -> Evaluating {len(kernels)} candidate kernels on hold-out validation split ({len(X_val)} samples)...")
        
        for name, kernel in tqdm(kernels.items(), desc=f" [{self.target_name}] Searching Kernels", leave=True):
            k_start = time.time()
            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=2, 
                alpha=1e-3, random_state=config.RANDOM_STATE
            )
            gp.fit(X_t, y_t)
            score = gp.score(X_val, y_val)
            k_duration = time.time() - k_start
            
            kernel_results[name] = (score, k_duration)
            print(f"    ├─ {name:<20}: Validation R² = {score:.4f}  (Time: {k_duration:.2f}s)")
            
            if score > best_score:
                best_score = score
                best_model = gp
                self.best_kernel_name = name

        print(f" -> 🏆 Selected Kernel: '{self.best_kernel_name}' with Val R² = {best_score:.4f}")
        print(f" -> Retraining selected model on entire training dataset ({len(X_train_scaled)} samples)...")
        
        self.model = best_model
        # Freeze optimizer for downstream predictions to avoid re-optimizing length-scales
        self.model.optimizer = None
        self.model.fit(X_train_scaled, y_train_scaled)
        
        self.training_time_sec = time.time() - start_time
        print(f" -> Final Fitted Kernel: {self.model.kernel_}")
        print(f" -> Stage 5 Completed in {self.training_time_sec:.2f} seconds.")

    def evaluate(self, X_test_scaled, y_test_scaled):
        """Stage 6: Test set validation and metric reporting."""
        print(f"\n==================================================")
        print(f" 📈 [{self.target_name}] Stage 6: Validation Diagnostics")
        print(f"==================================================")
        
        y_pred_scaled = self.model.predict(X_test_scaled)
        
        # Transform back to original physical scales
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = self.scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        exp_var = explained_variance_score(y_true, y_pred)
        max_err = np.max(np.abs(y_true - y_pred))
        
        print(f" -> Metric Summary (Physical Units):")
        print(f"    ├─ Coefficient of Determination (R²) : {r2:.6f}")
        print(f"    ├─ Root Mean Squared Error (RMSE)    : {rmse:.6f}")
        print(f"    ├─ Mean Absolute Error (MAE)         : {mae:.6f}")
        print(f"    ├─ Explained Variance Score          : {exp_var:.6f}")
        print(f"    └─ Maximum Single Error              : {max_err:.6f}")
        
        # Simple quality check warning
        if r2 < 0.85:
            print(f" ⚠️ Warning: R² score is below 0.85. Consider adding more CFD training points.")
        else:
            print(f" ✅ Model passed validation accuracy benchmarks.")
            
        return r2, rmse, mae

    def cross_validate(self, X_scaled, y_scaled):
        """Stage 7: K-Fold Cross Validation for generalized accuracy check."""
        print(f"\n==================================================")
        print(f" 🔄 [{self.target_name}] Stage 7: {config.CV_FOLDS}-Fold Cross Validation")
        print(f"==================================================")
        
        MAX_CV_SAMPLES = 2000
        if len(X_scaled) > MAX_CV_SAMPLES:
            np.random.seed(config.RANDOM_STATE)
            cv_idx = np.random.choice(len(X_scaled), size=MAX_CV_SAMPLES, replace=False)
            X_cv, y_cv = X_scaled[cv_idx], y_scaled[cv_idx]
            print(f" -> Capped cross-validation sample set to {MAX_CV_SAMPLES} points for speed.")
        else:
            X_cv, y_cv = X_scaled, y_scaled

        kf = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        r2_scores = []
        rmse_scores = []
        
        folds_list = list(kf.split(X_cv))
        for fold_idx, (train_idx, val_idx) in enumerate(tqdm(folds_list, desc=f" [{self.target_name}] CV Progress", leave=True)):
            X_f_train, X_f_val = X_cv[train_idx], X_cv[val_idx]
            y_f_train, y_f_val = y_cv[train_idx], y_cv[val_idx]
            
            gp = GaussianProcessRegressor(
                kernel=self.model.kernel_, optimizer=None, 
                alpha=1e-3, random_state=config.RANDOM_STATE
            )
            gp.fit(X_f_train, y_f_train)
            
            y_f_pred = gp.predict(X_f_val)
            r2 = r2_score(y_f_val, y_f_pred)
            rmse = np.sqrt(mean_squared_error(y_f_val, y_f_pred))
            
            r2_scores.append(r2)
            rmse_scores.append(rmse)
            print(f"    ├─ Fold {fold_idx + 1}/{config.CV_FOLDS}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
            
        mean_r2, std_r2 = np.mean(r2_scores), np.std(r2_scores)
        mean_rmse, std_rmse = np.mean(rmse_scores), np.std(rmse_scores)
        
        print(f" -> Cross-Validation Summary:")
        print(f"    ├─ Mean R² Score : {mean_r2:.4f} ± {std_r2:.4f}")
        print(f"    └─ Mean RMSE     : {mean_rmse:.4f} ± {std_rmse:.4f}")
        
        return mean_r2, std_r2

    def predict(self, X_new):
        """
        Stage 8: Uncertainty Quantification.
        Returns physical unscaled predictions (mean) AND standard deviations (sigma).
        """
        X_new_scaled = self.scaler_X.transform(X_new)
        y_pred_scaled, sigma_scaled = self.model.predict(X_new_scaled, return_std=True)
        
        # Inverse transform mean prediction
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Inverse transform standard deviation using target scaler standard deviation
        y_sigma = sigma_scaled * self.scaler_y.scale_[0]
        
        return y_pred, y_sigma

    def save_model(self, folder_path=config.BASE_DIR):
        """Save trained model, scalers, and feature metadata using joblib."""
        os.makedirs(folder_path, exist_ok=True)
        filename = os.path.join(folder_path, f"{self.target_name}_gp_model.pkl")
        
        save_dict = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'features': self.input_cols,
            'best_kernel': self.best_kernel_name
        }
        joblib.dump(save_dict, filename)
        print(f" -> 💾 Saved surrogate model artifact to: {filename}")

    def load_model(self, filepath):
        """Load pre-trained model and scalers from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        checkpoint = joblib.load(filepath)
        self.model = checkpoint['model']
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        self.input_cols = checkpoint.get('features', self.input_cols)
        self.best_kernel_name = checkpoint.get('best_kernel', 'Loaded')
        print(f" -> 📂 Successfully loaded model '{self.target_name}' from {filepath}")


def build_surrogates(df_cp, df_trf):
    """
    Master orchestrator for constructing both Aerodynamic (Cp) 
    and Structural (TRF) Gaussian Process surrogate models.
    """
    print("\n" + "="*60)
    print(" 🚀 STARTING SURROGATE MODELING PIPELINE (Stages 3-8)")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # 1. Aerodynamic Power Coefficient (Cp) Surrogate
    # -------------------------------------------------------------------------
    print("\n[STEP 1/2] Building Aerodynamic Surrogate (Cp)...")
    cp_builder = SurrogateBuilder('Cp', ['thickness', 'twist', 'solidity', 'tsr'])
    X_train_c, X_test_c, y_train_c, y_test_c, X_full_c, y_full_c = cp_builder.prepare_data(df_cp, 'cp')
    cp_builder.train_with_kernel_search(X_train_c, y_train_c)
    cp_builder.evaluate(X_test_c, y_test_c)
    
    # Perform Cross-Validation on Full Scaled Dataset
    X_full_scaled_c = cp_builder.scaler_X.transform(X_full_c)
    y_full_scaled_c = cp_builder.scaler_y.transform(y_full_c).flatten()
    cp_builder.cross_validate(X_full_scaled_c, y_full_scaled_c)
    cp_builder.save_model()

    print("\n" + "-"*50)
    
    # -------------------------------------------------------------------------
    # 2. Structural Torque Ripple Factor (TRF) Surrogate
    # -------------------------------------------------------------------------
    print("\n[STEP 2/2] Building Structural Surrogate (TRF)...")
    trf_builder = SurrogateBuilder('TRF', ['thickness', 'twist', 'solidity'])
    X_train_t, X_test_t, y_train_t, y_test_t, X_full_t, y_full_t = trf_builder.prepare_data(df_trf, 'trf')
    trf_builder.train_with_kernel_search(X_train_t, y_train_t)
    trf_builder.evaluate(X_test_t, y_test_t)
    
    # Perform Cross-Validation on Full Scaled Dataset
    X_full_scaled_t = trf_builder.scaler_X.transform(X_full_t)
    y_full_scaled_t = trf_builder.scaler_y.transform(y_full_t).flatten()
    trf_builder.cross_validate(X_full_scaled_t, y_full_scaled_t)
    trf_builder.save_model()
    
    print("\n" + "="*60)
    print(" ✅ ALL SURROGATE MODELS TRAINED & VALIDATED SUCCESSFULLY")
    print("="*60)
    
    return cp_builder, trf_builder


if __name__ == "__main__":
    print("Run main.py to execute the full MOSBO 2.0 optimization pipeline.")