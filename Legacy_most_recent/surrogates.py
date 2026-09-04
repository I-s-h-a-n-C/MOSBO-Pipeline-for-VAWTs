import os
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C, WhiteKernel
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
    - Stage 5: Model Training (Automated GP Kernel Search)
    - Stage 6: Hold-out Test Set Evaluation & Diagnostics
    - Stage 7: K-Fold Cross Validation
    - Stage 8: Uncertainty Quantification & Model Persistence
    """
    def __init__(self, target_name, input_cols, model_type='gp'):
        self.target_name = target_name.upper()
        self.input_cols = input_cols
        self.model_type = model_type.lower()
        
        # Feature and target scalers for Stage 4
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.model = None
        self.best_kernel_name = None
        self.training_time_sec = 0.0

    def _cluster_and_sample(self, X, y, max_samples, context_msg=""):
        """
        Intelligently downsamples dataset by clustering feature-space and performance-space, 
        ensuring extreme physical boundaries and diverse geometries are preserved.
        """
        if len(X) <= max_samples:
            return X, y
            
        print(f" -> 🧩 Clustering {len(X)} points to extract {max_samples} diverse boundary cases {context_msg}...")
        
        # Standardize locally so features and target have equal weighting in distance calculations
        temp_X = StandardScaler().fit_transform(X)
        temp_y = StandardScaler().fit_transform(y.reshape(-1, 1))
        features_for_clustering = np.hstack([temp_X, temp_y])
        
        # Create clusters (max 50 for speed and fine-grained diversity)
        num_clusters = min(50, max_samples // 10)
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=config.RANDOM_STATE, n_init='auto')
        cluster_labels = kmeans.fit_predict(features_for_clustering)
        
        selected_idx = []
        samples_per_cluster = max_samples // num_clusters
        
        np.random.seed(config.RANDOM_STATE)
        
        # Sample evenly from each cluster to preserve design diversity
        for i in range(num_clusters):
            cluster_idx = np.where(cluster_labels == i)[0]
            if len(cluster_idx) > 0:
                if len(cluster_idx) <= samples_per_cluster:
                    selected_idx.extend(cluster_idx)
                else:
                    selected_idx.extend(np.random.choice(cluster_idx, size=samples_per_cluster, replace=False))
                    
        # Fill any mathematical shortfall due to uneven cluster sizes
        selected_idx = list(set(selected_idx))
        shortfall = max_samples - len(selected_idx)
        if shortfall > 0:
            remaining_idx = list(set(range(len(X))) - set(selected_idx))
            if remaining_idx:
                selected_idx.extend(np.random.choice(remaining_idx, size=min(shortfall, len(remaining_idx)), replace=False))
                
        # HARD GUARANTEE: Force include the absolute extremum values (Peak Cp / Min TRF)
        y_flat = y.flatten()
        min_idx = np.argmin(y_flat)
        max_idx = np.argmax(y_flat)
        if min_idx not in selected_idx:
            selected_idx[0] = min_idx
        if max_idx not in selected_idx:
            selected_idx[1] = max_idx
            
        selected_idx = np.array(selected_idx)
        return X[selected_idx], y[selected_idx]

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
        if self.model_type == 'gp':
            MAX_GP_SAMPLES = 10000 
            X_train, y_train = self._cluster_and_sample(X_train, y_train, MAX_GP_SAMPLES, context_msg="(Final Training)")
        else:
            print(f" -> [Random Forest] Utilizing full training dataset: {len(X_train)} samples.")

        # Stage 4: Scale Data (Essential for GP numerical stability and uniform RF variance)
        print(f" -> Fitting StandardScaler on features and targets...")
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train).flatten()
        
        X_test_scaled = self.scaler_X.transform(X_test)
        y_test_scaled = self.scaler_y.transform(y_test).flatten()
        
        print(f" -> Target Mean (Train) : {self.scaler_y.mean_[0]:.4f} ± {self.scaler_y.scale_[0]:.4f}")
        print(f" -> Data preparation complete.")
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X, y

    def get_feature_sensitivities(self):
        """
        Extracts ARD length scales from the fitted GP kernel and calculates 
        relative feature relevance weights (1 / length_scale).
        """
        if self.model_type != 'gp' or self.model is None:
            return None, None

        def _find_length_scale(kernel):
            # Recursively traverse Sum or Product kernels
            if hasattr(kernel, "k1"):
                ls = _find_length_scale(kernel.k1)
                if ls is not None: 
                    return ls
                return _find_length_scale(kernel.k2)
            elif hasattr(kernel, "length_scale"):
                return np.atleast_1d(kernel.length_scale)
            return None

        length_scales = _find_length_scale(self.model.kernel_)
        
        if length_scales is None or len(length_scales) != len(self.input_cols):
            return None, None

        # Inverse Length Scale = Sensitivity Weight
        inverse_ls = 1.0 / length_scales
        # Normalize to percentage importance
        relevance_pct = (inverse_ls / np.sum(inverse_ls)) * 100.0
        
        return length_scales, relevance_pct

    def train_random_forest(self, X_train_scaled, y_train_scaled):
        """Trains a Random Forest Regressor using hyperparameter constraints to prevent overfitting."""
        start_time = time.time()
        print(f"[{self.target_name}] Training Random Forest Regressor on {len(X_train_scaled)} points...")
        
        # Added hyperparameter constraints (max_depth & min_samples_leaf) to prevent massive overfitting
        self.model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=15,             # Prevents trees from growing infinitely deep
            min_samples_leaf=2,       # Forces generalized groupings at the leaves
            random_state=config.RANDOM_STATE, 
            n_jobs=-1                 # Uses all available CPU cores
        )
        self.model.fit(X_train_scaled, y_train_scaled)
        
        self.best_kernel_name = "RandomForest"
        self.training_time_sec = time.time() - start_time
        print(f" -> Stage 5 Completed in {self.training_time_sec:.2f} seconds.")

    def train_with_kernel_search(self, X_train_scaled, y_train_scaled):
        """Stage 5: Train GP and automatically select the best kernel."""
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        
        start_time = time.time()
        print(f"[{self.target_name}] Running Automated Kernel Search...")
        
        bounds = (1e-2, 1e2)
        dims = X_train_scaled.shape[1]
        
        kernels = {
            "RBF": C(1.0) * RBF(length_scale=[1.0] * dims, length_scale_bounds=bounds),
            "Matern 3/2": C(1.0) * Matern(length_scale=[1.0] * dims, length_scale_bounds=bounds, nu=1.5),
            "Matern 5/2": C(1.0) * Matern(length_scale=[1.0] * dims, length_scale_bounds=bounds, nu=2.5),
            "Rational Quadratic": C(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=bounds),
            "Matern 5/2 + Noise": C(1.0) * Matern(length_scale=[1.0] * dims, length_scale_bounds=bounds, nu=2.5) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-7, 1e-2))
        }
        
        best_score = -np.inf
        best_kernel_name = None
        kernel_results = {}
        
        SEARCH_SAMPLES = 1500
        X_search, y_search = self._cluster_and_sample(X_train_scaled, y_train_scaled, SEARCH_SAMPLES, context_msg="(Kernel Search)")

        kf = KFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
        print(f" -> Evaluating {len(kernels)} candidate kernels using 3-Fold CV on {len(X_search)} subset...")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            
            for name, kernel in tqdm(kernels.items(), desc=f" [{self.target_name}] Searching Kernels", leave=True):
                k_start = time.time()
                cv_scores = []
                
                for train_idx, val_idx in kf.split(X_search):
                    X_tr, X_va = X_search[train_idx], X_search[val_idx]
                    y_tr, y_va = y_search[train_idx], y_search[val_idx]
                    
                    gp = GaussianProcessRegressor(
                        kernel=clone(kernel), n_restarts_optimizer=3, 
                        alpha=1e-4, random_state=config.RANDOM_STATE
                    )
                    gp.fit(X_tr, y_tr)
                    cv_scores.append(gp.score(X_va, y_va))
                    
                mean_score = np.mean(cv_scores)
                k_duration = time.time() - k_start
                
                kernel_results[name] = (mean_score, k_duration)
                print(f"    ├─ {name:<20}: Mean CV R² = {mean_score:.4f}  (Time: {k_duration:.2f}s)")
                
                if mean_score > best_score:
                    best_score = mean_score
                    self.best_kernel_name = name

            print(f" -> 🏆 Selected Kernel: '{self.best_kernel_name}' with CV R² = {best_score:.4f}")
            print(f" -> Retraining selected model on entire training dataset ({len(X_train_scaled)} samples)...")
            
            self.model = GaussianProcessRegressor(
                kernel=clone(kernels[self.best_kernel_name]), n_restarts_optimizer=25, 
                alpha=1e-3, random_state=config.RANDOM_STATE
            )
            
            self.model.fit(X_train_scaled, y_train_scaled)
        
        self.model.optimizer = None
        
        self.training_time_sec = time.time() - start_time
        print(f" -> Final Fitted Kernel: {self.model.kernel_}")
        print(f" -> Stage 5 Completed in {self.training_time_sec:.2f} seconds.")

    def evaluate(self, X_test_scaled, y_test_scaled):
        """Stage 6: Test set validation and metric reporting."""
        print(f"\n==================================================")
        print(f" 📈 [{self.target_name}] Stage 6: Validation Diagnostics")
        print(f"==================================================")
        
        # Use our own predict method to handle both RF and GP outputs
        if self.model_type == 'gp':
            y_pred_scaled = self.model.predict(X_test_scaled)
        elif self.model_type == 'rf':
            y_pred_scaled = self.model.predict(X_test_scaled)
            
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
        
        if self.model_type == 'gp':
            MAX_CV_SAMPLES = 2000
            X_cv, y_cv = self._cluster_and_sample(X_scaled, y_scaled, MAX_CV_SAMPLES, context_msg="(Cross-Validation)")
        else:
            # Random Forest scales perfectly, skip downsampling for CV
            X_cv, y_cv = X_scaled, y_scaled

        kf = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        r2_scores = []
        rmse_scores = []
        
        folds_list = list(kf.split(X_cv))
        for fold_idx, (train_idx, val_idx) in enumerate(tqdm(folds_list, desc=f" [{self.target_name}] CV Progress", leave=True)):
            X_f_train, X_f_val = X_cv[train_idx], X_cv[val_idx]
            y_f_train, y_f_val = y_cv[train_idx], y_cv[val_idx]
            
            if self.model_type == 'gp':
                model_clone = GaussianProcessRegressor(
                    kernel=self.model.kernel_, optimizer=None, 
                    alpha=1e-3, random_state=config.RANDOM_STATE
                )
            elif self.model_type == 'rf':
                model_clone = RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=15, 
                    min_samples_leaf=4, 
                    random_state=config.RANDOM_STATE, 
                    n_jobs=-1
                )
                
            model_clone.fit(X_f_train, y_f_train)
            
            y_f_pred = model_clone.predict(X_f_val)
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
        Calculates GP standard deviation natively, and RF standard deviation across decision trees.
        """
        X_new_scaled = self.scaler_X.transform(X_new)
        
        if self.model_type == 'gp':
            y_pred_scaled, sigma_scaled = self.model.predict(X_new_scaled, return_std=True)
        elif self.model_type == 'rf':
            # Extract predictions from all individual trees to calculate variance/uncertainty
            all_tree_preds = np.stack([tree.predict(X_new_scaled) for tree in self.model.estimators_])
            y_pred_scaled = all_tree_preds.mean(axis=0)
            sigma_scaled = all_tree_preds.std(axis=0)
        
        # Inverse transform mean prediction
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Inverse transform standard deviation using target scaler standard deviation
        y_sigma = sigma_scaled * self.scaler_y.scale_[0]
        
        return y_pred, y_sigma

    def save_model(self, folder_path=config.BASE_DIR):
        """Save trained model, scalers, and feature metadata using joblib."""
        os.makedirs(folder_path, exist_ok=True)
        filename = os.path.join(folder_path, f"{self.target_name}_model.pkl")
        
        save_dict = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'features': self.input_cols,
            'best_kernel': self.best_kernel_name,
            'model_type': self.model_type
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
        self.model_type = checkpoint.get('model_type', 'gp')
        print(f" -> 📂 Successfully loaded model '{self.target_name}' from {filepath}")


def build_surrogates(df_cp, df_trf):
    """
    Master orchestrator for constructing both Aerodynamic (Cp) 
    and Structural (TRF) surrogate models.
    """
    print("\n" + "="*60)
    print(" 🚀 STARTING SURROGATE MODELING PIPELINE (Stages 3-8)")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # 1. Aerodynamic Power Coefficient (Cp) Surrogate - GPR
    # -------------------------------------------------------------------------
    print("\n[STEP 1/2] Building Aerodynamic Surrogate (Cp)...")
    cp_builder = SurrogateBuilder('Cp', ['thickness', 'twist', 'solidity', 'tsr'], model_type='gp')
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
    # 2. Structural Torque Ripple Factor (TRF) Surrogate - GAUSSIAN PROCESS
    # -------------------------------------------------------------------------
    print("\n[STEP 2/2] Building Structural Surrogate (TRF)...")
    trf_builder = SurrogateBuilder('TRF', ['thickness', 'twist', 'solidity'], model_type='gp')
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