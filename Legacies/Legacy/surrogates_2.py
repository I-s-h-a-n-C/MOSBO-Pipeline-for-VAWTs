import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

import config

class SurrogateBuilder:
    """
    Handles Stage 3-8: Splitting, Scaling, Kernel Search, CV, and Uncertainty generation.
    """
    def __init__(self, target_name, input_cols):
        self.target_name = target_name.upper()
        self.input_cols = input_cols
        
        # Scalers for Stage 4
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.model = None
        self.best_kernel_name = None

    def prepare_data(self, df, target_col):
        """Stage 3 & 4: Train/Test Split and StandardScaler"""
        print(f"\n[{self.target_name}] Preparing Data...")
        
        X = df[self.input_cols].values
        y = df[target_col].values.reshape(-1, 1)
        
        # Stage 3: Train / Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.TRAIN_TEST_SPLIT, 
            random_state=config.RANDOM_STATE
        )
        
        # Stage 4: Scale Data (Crucial for GP stability)
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train).flatten()
        
        X_test_scaled = self.scaler_X.transform(X_test)
        y_test_scaled = self.scaler_y.transform(y_test).flatten()
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X, y

    def train_with_kernel_search(self, X_train_scaled, y_train_scaled):
        """Stage 5: Train GP and automatically select the best kernel."""
        print(f"[{self.target_name}] Running Automated Kernel Search...")
        
        # Base bounds for the length scales
        bounds = (1e-3, 1e3)
        dims = X_train_scaled.shape[1]
        
        # Candidate Kernels
        kernels = {
            "RBF": C(1.0) * RBF(length_scale=[1.0] * dims, length_scale_bounds=bounds),
            "Matern 3/2": C(1.0) * Matern(length_scale=[1.0] * dims, length_scale_bounds=bounds, nu=1.5),
            "Matern 5/2": C(1.0) * Matern(length_scale=[1.0] * dims, length_scale_bounds=bounds, nu=2.5),
            "Rational Quadratic": C(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0)
        }
        
        best_score = -np.inf
        best_model = None
        
        # Simple Hold-out validation for kernel search to save time
        X_t, X_val, y_t, y_val = train_test_split(
            X_train_scaled, y_train_scaled, test_size=0.2, random_state=config.RANDOM_STATE
        )
        
        # Wrap candidate kernel iteration in a tqdm progress bar
        for name, kernel in tqdm(kernels.items(), desc=f"[{self.target_name}] Kernel Search", leave=True):
            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5, 
                alpha=1e-3, random_state=config.RANDOM_STATE
            )
            gp.fit(X_t, y_t)
            score = gp.score(X_val, y_val)
            
            if score > best_score:
                best_score = score
                best_model = gp
                self.best_kernel_name = name
                
        print(f"[{self.target_name}] Best Kernel Selected: {self.best_kernel_name} (Val R2: {best_score:.4f})")
        
        # Retrain best model on full training set
        self.model = best_model
        self.model.fit(X_train_scaled, y_train_scaled)

    def evaluate(self, X_test_scaled, y_test_scaled):
        """Stage 6: Validation (R2, RMSE, MAE)"""
        y_pred_scaled = self.model.predict(X_test_scaled)
        
        # Inverse transform to get physical error metrics
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = self.scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        print(f"\n[{self.target_name}] Stage 6 Validation Metrics:")
        print(f" -> R²   = {r2:.4f}")
        print(f" -> RMSE = {rmse:.4f}")
        print(f" -> MAE  = {mae:.4f}")
        return r2, rmse, mae

    def cross_validate(self, X_scaled, y_scaled):
        """Stage 7: 5-Fold Cross Validation"""
        print(f"\n[{self.target_name}] Running {config.CV_FOLDS}-Fold Cross Validation...")
        kf = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        
        r2_scores = []
        folds_list = list(kf.split(X_scaled))
        for fold, (train_idx, val_idx) in enumerate(tqdm(folds_list, desc=f"[{self.target_name}] CV Folds", leave=True)):
            X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
            y_fold_train, y_fold_val = y_scaled[train_idx], y_scaled[val_idx]
            
            gp = GaussianProcessRegressor(
                kernel=self.model.kernel_, n_restarts_optimizer=2, 
                alpha=1e-3, random_state=config.RANDOM_STATE
            )
            gp.fit(X_fold_train, y_fold_train)
            score = gp.score(X_fold_val, y_fold_val)
            r2_scores.append(score)
            
        mean_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        print(f"[{self.target_name}] CV Result: Mean R² = {mean_r2:.4f} ± {std_r2:.4f}")
        return mean_r2, std_r2

    def predict(self, X_new):
        """
        Stage 8: Uncertainty Quantification.
        Returns physical unscaled predictions AND unscaled standard deviations (sigma).
        """
        X_new_scaled = self.scaler_X.transform(X_new)
        y_pred_scaled, sigma_scaled = self.model.predict(X_new_scaled, return_std=True)
        
        # Inverse transform mean
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Inverse transform standard deviation (Scale by the scaler's standard deviation factor)
        # scaler_y.scale_[0] holds the standard deviation of the training targets.
        y_sigma = sigma_scaled * self.scaler_y.scale_[0]
        
        return y_pred, y_sigma
        
    def save_model(self, folder_path=config.BASE_DIR):
        """Persist the model and scalers for future use without retraining."""
        filename = os.path.join(folder_path, f"{self.target_name}_gp_model.pkl")
        joblib.dump({
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'features': self.input_cols
        }, filename)
        print(f"[{self.target_name}] Model saved to {filename}")

def build_surrogates(df_cp, df_trf):
    """Master orchestrator to build both Cp and TRF models."""
    
    # 1. Aerodynamic Surrogate (Cp)
    cp_builder = SurrogateBuilder('Cp', ['thickness', 'twist', 'solidity', 'tsr'])
    X_train_c, X_test_c, y_train_c, y_test_c, X_full_c, y_full_c = cp_builder.prepare_data(df_cp, 'cp')
    cp_builder.train_with_kernel_search(X_train_c, y_train_c)
    cp_builder.evaluate(X_test_c, y_test_c)
    
    # Full dataset scaled for CV
    X_full_scaled_c = cp_builder.scaler_X.transform(X_full_c)
    y_full_scaled_c = cp_builder.scaler_y.transform(y_full_c).flatten()
    cp_builder.cross_validate(X_full_scaled_c, y_full_scaled_c)
    cp_builder.save_model()

    print("\n" + "="*50)
    
    # 2. Structural Surrogate (TRF)
    trf_builder = SurrogateBuilder('TRF', ['thickness', 'twist', 'solidity'])
    X_train_t, X_test_t, y_train_t, y_test_t, X_full_t, y_full_t = trf_builder.prepare_data(df_trf, 'trf')
    trf_builder.train_with_kernel_search(X_train_t, y_train_t)
    trf_builder.evaluate(X_test_t, y_test_t)
    
    X_full_scaled_t = trf_builder.scaler_X.transform(X_full_t)
    y_full_scaled_t = trf_builder.scaler_y.transform(y_full_t).flatten()
    trf_builder.cross_validate(X_full_scaled_t, y_full_scaled_t)
    trf_builder.save_model()
    
    return cp_builder, trf_builder

if __name__ == "__main__":
    # Note: To test this file standalone, you would import load_and_clean_all from data_loader
    print("Run main.py to execute the full pipeline.")