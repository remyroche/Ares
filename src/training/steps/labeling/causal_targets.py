"""
Causal Targets Module

Implements Double Machine Learning (DML) for computing causal targets
including treatment effects and causal residuals.

Key Features:
1. Double Machine Learning for causal effect estimation
2. CATE (Conditional Average Treatment Effect) computation
3. Causal residual targets for Chaser system
4. Treatment effect heterogeneity analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
from scipy.stats import norm

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

class CausalTargetComputer:
    """
    Computes causal targets using Double Machine Learning.
    
    Implements DML to estimate treatment effects and create
    causal residual targets for the Chaser system.
    """
    
    def __init__(
        self,
        treatment_model: str = "ridge",
        outcome_model: str = "ridge",
        n_folds: int = 3,
        random_state: int = 42,
        verbose: bool = True,
        checkpoint_manager=None,
        symbol: str = "UNKNOWN",
        cate_config: Optional[Dict[str, Any]] = None,
        subsample_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Causal Target Computer.
        
        Args:
            treatment_model: Model for treatment prediction
            outcome_model: Model for outcome prediction
            n_folds: Number of cross-validation folds
            random_state: Random seed
            verbose: Whether to print progress information
            checkpoint_manager: Optional checkpoint manager for saving progress
            symbol: Trading symbol for checkpoint naming
            cate_config: Configuration for CATE model (model_type, params)
            subsample_config: Configuration for subsampling (threshold, method)
        """
        self.treatment_model = treatment_model
        self.outcome_model = outcome_model
        self.n_folds = n_folds
        self.random_state = random_state
        self.verbose = verbose
        self.checkpoint_manager = checkpoint_manager
        self.symbol = symbol
        
        # CATE Configuration
        self.cate_config = cate_config or {}
        self.cate_model_type = self.cate_config.get('model_type', 'random_forest')
        self.cate_params = self.cate_config.get('params', {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_leaf': 10,
            'n_jobs': -1
        })

        # Subsample Configuration
        self.subsample_config = subsample_config or {}
        self.large_dataset_threshold = self.subsample_config.get('threshold', 50000)
        self.subsample_method = self.subsample_config.get('method', 'adaptive') # Default to adaptive

        # Storage for models and results
        self.treatment_models_ = {}
        self.outcome_models_ = {}
        self.causal_effects_ = {}
        self.cate_estimates_ = None
        self.residual_targets_ = {}
        self.refutation_scores_ = None
        self.causal_effect_frame_ = None
        self.subsample_indices_ = None  # Store subsample indices for alignment
        
    def _get_model(self, model_type: str, model_name: str):
        """
        Get model instance based on name.
        
        Args:
            model_type: Type of model ("treatment" or "outcome")
            model_name: Model name
            
        Returns:
            Model instance
        """
        models = {
            "linear": Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', LinearRegression())
            ]),
            "ridge": Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', Ridge(alpha=1.0))
            ]),
            "lasso": Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', Lasso(alpha=1.0))
            ]),
            "random_forest": RandomForestRegressor(
                n_estimators=50, random_state=self.random_state  # Reduced from 100
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=50, random_state=self.random_state  # Reduced from 100
            )
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return models[model_name]
    
    def _validate_and_clean_data(
        self, 
        data: pd.DataFrame, 
        data_name: str = "data"
    ) -> pd.DataFrame:
        """
        Validate and clean data for infinity and extreme values before sklearn processing.
        
        Args:
            data: DataFrame to validate and clean
            data_name: Name of the data for logging
            
        Returns:
            Cleaned DataFrame
        """
        try:
            if self.verbose:
                tprint_info(f"🔍 Validating {data_name} for infinity and extreme values...")
            
            data_clean = data.copy()
            
            # Check for infinity values
            inf_counts = {}
            for col in data_clean.select_dtypes(include=[np.number]).columns:
                inf_count = np.isinf(data_clean[col]).sum()
                if inf_count > 0:
                    inf_counts[col] = inf_count
            
            if inf_counts:
                total_inf = sum(inf_counts.values())
                if self.verbose:
                    tprint_warning(f"⚠️ Found {total_inf} infinity values in {data_name}, replacing with NaN")
                
                # Replace infinity with NaN
                for col, count in inf_counts.items():
                    data_clean[col] = data_clean[col].replace([np.inf, -np.inf], np.nan)
            
            # Check for extremely large values
            float64_max = np.finfo(np.float64).max / 1000  # Safety margin
            large_counts = {}
            
            for col in data_clean.select_dtypes(include=[np.number]).columns:
                large_count = (np.abs(data_clean[col]) > float64_max).sum()
                if large_count > 0:
                    large_counts[col] = large_count
            
            if large_counts:
                total_large = sum(large_counts.values())
                if self.verbose:
                    tprint_warning(f"⚠️ Found {total_large} extremely large values in {data_name}, clipping")
                
                # Clip extreme values
                for col in large_counts.keys():
                    data_clean[col] = data_clean[col].clip(lower=-float64_max, upper=float64_max)
            
            # Handle NaN values that resulted from infinity replacement
            nan_counts = data_clean.isna().sum()
            if nan_counts.any():
                total_nan = nan_counts.sum()
                if self.verbose:
                    tprint_info(f"   📝 Found {total_nan} NaN values in {data_name} (will be handled by sklearn imputers)")
            
            return data_clean
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Data validation failed for {data_name}: {e}")
            return data
    
    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize memory usage by downcasting numeric columns.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Memory-optimized DataFrame
        """
        try:
            # Make a copy to avoid modifying original
            df_optimized = df.copy()
            
            # Downcast float columns
            for col in df_optimized.select_dtypes(include=['float64']).columns:
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
            
            # Downcast integer columns
            for col in df_optimized.select_dtypes(include=['int64']).columns:
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
            
            if self.verbose:
                original_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
                optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2  # MB
                reduction = (original_memory - optimized_memory) / original_memory * 100
                tprint_info(f"   Memory optimized: {original_memory:.1f}MB → {optimized_memory:.1f}MB ({reduction:.1f}% reduction)")
            
            return df_optimized
        except Exception as e:
            if self.verbose:
                tprint_warning(f"   Memory optimization failed: {e}")
            return df
    
    def compute_dml_causal_effects(
        self,
        X: pd.DataFrame,
        treatments: Union[pd.Series, np.ndarray],
        outcomes: Union[pd.Series, np.ndarray],
        confounders: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Compute causal effects using Double Machine Learning.
        
        Args:
            X: Feature matrix
            treatments: Treatment variable(s)
            outcomes: Outcome variable(s)
            confounders: Confounding variables
            
        Returns:
            Dictionary with causal effect estimates
        """
        try:
            if self.verbose:
                tprint_info("🧮 Computing DML Causal Effects...")
            
            # Convert to numpy arrays for downstream math while keeping indices for confounders
            treatments_array = np.asarray(treatments)
            outcomes_array = np.asarray(outcomes)
            
            if treatments_array.ndim == 1:
                treatments_array = treatments_array.reshape(-1, 1)
            if outcomes_array.ndim == 1:
                outcomes_array = outcomes_array.reshape(-1, 1)
            
            # Use X as confounders if not provided, ensuring only numeric columns
            if confounders is None:
                confounders = X.select_dtypes(include=[np.number])
            else:
                confounders = pd.concat(
                    [
                        X.select_dtypes(include=[np.number]),
                        confounders.select_dtypes(include=[np.number])
                    ],
                    axis=1
                )
            
            # Ensure we have some numeric features
            if confounders.empty:
                raise ValueError("No numeric features available as confounders")
            
            # Apply memory optimization
            confounders = self._optimize_memory(confounders)
            
            # Validate and clean confounder data for sklearn compatibility
            confounders = self._validate_and_clean_data(confounders, "confounder data")
            
            n_samples = len(treatments_array)
            treatment_width = treatments_array.shape[1]
            outcome_width = outcomes_array.shape[1]
            
            # Adaptive subsampling for large datasets
            if n_samples > self.large_dataset_threshold:
                if self.verbose:
                    tprint_info(f"   Large dataset detected ({n_samples:,} samples), applying {self.subsample_method} subsampling...")
                
                sample_indices = None
                
                if self.subsample_method == 'adaptive':
                    # Adaptive strategy: check outcome variance to decide stratification
                    outcome_std = np.std(outcomes_array)
                    treatment_std = np.std(treatments_array[:, 0])

                    # If high variance, we need more representative sampling
                    # We stratify by both treatment and outcome if possible
                    if outcome_std > 1.0 or treatment_std > 1.0: # Arbitrary high variance check
                         if self.verbose: tprint_info("   High variance detected - using dual stratification")
                         # Bin both
                         t_bins = pd.qcut(treatments_array[:, 0], q=5, labels=False, duplicates='drop')
                         y_bins = pd.qcut(outcomes_array.ravel(), q=5, labels=False, duplicates='drop')
                         strata = t_bins * 5 + y_bins # 25 strata
                         n_strata = 25
                    else:
                         # Standard treatment stratification
                         strata = pd.qcut(treatments_array[:, 0], q=10, labels=False, duplicates='drop')
                         n_strata = 10

                    # Sample within each stratum
                    target_size = self.large_dataset_threshold
                    sample_size_per_stratum = max(10, target_size // n_strata)

                    sample_indices = []
                    for s in range(n_strata):
                        s_indices = np.where(strata == s)[0]
                        if len(s_indices) > 0:
                            n_sample = min(len(s_indices), sample_size_per_stratum)
                            s_sampled = np.random.choice(s_indices, size=n_sample, replace=False)
                            sample_indices.extend(s_sampled)

                else:
                    # Default: Stratified by treatment (legacy)
                    treatment_quantiles = pd.qcut(treatments_array[:, 0], q=10, labels=False, duplicates='drop')
                    sample_size_per_stratum = min(self.large_dataset_threshold // 10, n_samples // 10)
                    sample_indices = []

                    for stratum in range(10):
                        stratum_indices = np.where(treatment_quantiles == stratum)[0]
                        if len(stratum_indices) > 0:
                            n_sample = min(len(stratum_indices), sample_size_per_stratum)
                            sampled_indices = np.random.choice(stratum_indices, size=n_sample, replace=False)
                            sample_indices.extend(sampled_indices)
                
                # Convert to arrays and shuffle
                sample_indices = np.array(sample_indices)
                np.random.shuffle(sample_indices)
                
                # Store subsample indices for later alignment
                self.subsample_indices_ = sample_indices
                
                # Apply sampling
                treatments_array = treatments_array[sample_indices]
                outcomes_array = outcomes_array[sample_indices]
                confounders = confounders.iloc[sample_indices]
                
                n_samples = len(treatments_array)
                if self.verbose:
                    tprint_info(f"   Subsampled to {n_samples:,} samples")
            
            # Initialize cross-validation
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            
            # Initialize arrays for predictions
            treatment_pred = np.zeros((n_samples, treatment_width), dtype=float)
            outcome_pred = np.zeros((n_samples, outcome_width), dtype=float)

            def _reshape_prediction(prediction: Any, expected_width: int) -> np.ndarray:
                pred_array = np.asarray(prediction)
                if pred_array.ndim == 1:
                    pred_array = pred_array.reshape(-1, 1)
                if pred_array.shape[1] != expected_width:
                    raise ValueError(
                        f"Prediction width mismatch: expected {expected_width}, got {pred_array.shape[1]}"
                    )
                return pred_array
            
            # Cross-validation for out-of-fold predictions
            for fold, (train_idx, val_idx) in enumerate(kf.split(confounders)):
                if self.verbose:
                    tprint_info(f"   Processing fold {fold + 1}/{self.n_folds}...")
                
                # Split data
                X_train, X_val = confounders.iloc[train_idx], confounders.iloc[val_idx]
                
                T_train, T_val = treatments_array[train_idx], treatments_array[val_idx]
                Y_train, Y_val = outcomes_array[train_idx], outcomes_array[val_idx]
                
                # Fit treatment model
                treatment_model = self._get_model("treatment", self.treatment_model)
                
                # Handle multi-output training
                if T_train.shape[1] > 1:
                    treatment_model.fit(X_train, T_train)
                else:
                    treatment_model.fit(X_train, T_train.ravel())
                
                treatment_pred[val_idx] = _reshape_prediction(
                    treatment_model.predict(X_val),
                    treatment_width
                )
                
                # Fit outcome model
                outcome_model = self._get_model("outcome", self.outcome_model)
                
                # Handle multi-output training (unlikely for outcomes but safe)
                if Y_train.shape[1] > 1:
                    outcome_model.fit(X_train, Y_train)
                else:
                    outcome_model.fit(X_train, Y_train.ravel())

                outcome_pred[val_idx] = _reshape_prediction(
                    outcome_model.predict(X_val),
                    outcome_width
                )
                
                # Store models for last fold
                if fold == self.n_folds - 1:
                    self.treatment_models_["final"] = treatment_model
                    self.outcome_models_["final"] = outcome_model
            
            # Compute residuals
            treatment_residuals = treatments_array - treatment_pred
            outcome_residuals = outcomes_array - outcome_pred
            
            # Estimate causal effect using residuals
            causal_effect_model = LinearRegression()
            causal_effect_model.fit(treatment_residuals, outcome_residuals.ravel())
            
            causal_effect = causal_effect_model.coef_[0]
            causal_effect_se = np.sqrt(np.var(outcome_residuals - causal_effect * treatment_residuals) / 
                                      np.var(treatment_residuals) / n_samples)
            ci_radius = 1.96 * causal_effect_se
            effect_ci_low = causal_effect - ci_radius
            effect_ci_high = causal_effect + ci_radius
            
            # Compute R-squared (used as simple refutation proxy)
            y_pred = causal_effect * treatment_residuals
            r_squared = 1 - np.var(outcome_residuals - y_pred) / np.var(outcome_residuals)
            refutation_score = float(np.clip(r_squared, 0.0, 1.0))
            
            # Store results
            self.causal_effects_ = {
                "causal_effect": causal_effect,
                "standard_error": causal_effect_se,
                "r_squared": r_squared,
                "treatment_residuals": treatment_residuals,
                "outcome_residuals": outcome_residuals,
                "treatment_predictions": treatment_pred,
                "outcome_predictions": outcome_pred,
                "subsample_indices": self.subsample_indices_  # Store for alignment
            }
            # Create Layer 4-ready causal effect frame
            self.causal_effect_frame_ = pd.DataFrame({
                "causal_effect_estimate": np.full(len(X), causal_effect),
                "causal_effect_ci_low": np.full(len(X), effect_ci_low),
                "causal_effect_ci_high": np.full(len(X), effect_ci_high),
                "causal_refutation_score": np.full(len(X), refutation_score)
            }, index=X.index)
            self.refutation_scores_ = self.causal_effect_frame_["causal_refutation_score"]
            
            if self.verbose:
                tprint_success("✅ DML Causal Effects Computed:")
                tprint_info(f"   - Causal effect: {causal_effect:.6f}")
                tprint_info(f"   - Standard error: {causal_effect_se:.6f}")
                tprint_info(f"   - R-squared: {r_squared:.4f}")
                tprint_info(f"   - Treatment model: {self.treatment_model}")
                tprint_info(f"   - Outcome model: {self.outcome_model}")
            
            # Save checkpoint after DML effects computation
            if self.checkpoint_manager:
                try:
                    checkpoint_data = {
                        'causal_effects': self.causal_effects_,
                        'causal_effect_frame': self.causal_effect_frame_,
                        'treatment_models': self.treatment_models_,
                        'outcome_models': self.outcome_models_
                    }
                    self.checkpoint_manager.save_checkpoint('dml_effects_computed', checkpoint_data, self.symbol)
                    if self.verbose:
                        tprint_info("   💾 Saved checkpoint: dml_effects_computed")
                except Exception as e:
                    if self.verbose:
                        tprint_warning(f"   ⚠️ Failed to save dml_effects_computed checkpoint: {e}")
            
            return self.causal_effects_
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ DML causal effect computation failed: {e}")
            raise
    
    def compute_cate(
        self,
        X: pd.DataFrame,
        treatments: Union[pd.Series, np.ndarray],
        outcomes: Union[pd.Series, np.ndarray],
        confounders: Optional[pd.DataFrame] = None,
        heterogeneity_features: Optional[List[str]] = None
    ) -> pd.Series:
        """
        Compute Conditional Average Treatment Effects (CATE).
        
        Args:
            X: Feature matrix
            treatments: Treatment variable
            outcomes: Outcome variable
            confounders: Confounding variables
            heterogeneity_features: Features for effect heterogeneity
            
        Returns:
            CATE estimates
        """
        try:
            if self.verbose:
                tprint_info("🎯 Computing CATE...")
            
            # Use computed causal effects if available
            if not self.causal_effects_:
                self.compute_dml_causal_effects(X, treatments, outcomes, confounders)
            
            # Select heterogeneity features
            if heterogeneity_features is None:
                heterogeneity_features = X.columns.tolist()
            
            heterogeneity_data = X[heterogeneity_features].select_dtypes(include=[np.number])
            
            if heterogeneity_data.empty:
                tprint_warning("   No numeric features available for CATE, using X instead")
                heterogeneity_data = X.select_dtypes(include=[np.number])
            
            # Validate and clean heterogeneity data for sklearn compatibility
            heterogeneity_data = self._validate_and_clean_data(heterogeneity_data, "heterogeneity data")
            
            # Use treatment residuals for CATE estimation
            treatment_residuals = self.causal_effects_["treatment_residuals"].ravel()
            outcome_residuals = self.causal_effects_["outcome_residuals"].ravel()
            
            # Handle subsample indices if present
            if self.subsample_indices_ is not None:
                # Use only the subsampled heterogeneity data that matches the residuals
                heterogeneity_data = heterogeneity_data.iloc[self.subsample_indices_]
                if self.verbose:
                    tprint_info(f"   Using subsampled heterogeneity data: {len(heterogeneity_data)} samples")
            else:
                if self.verbose:
                    tprint_info(f"   Using full heterogeneity data: {len(heterogeneity_data)} samples")
            
            # Initialize CATE model based on config
            if self.cate_model_type == 'xgboost':
                try:
                    from xgboost import XGBRegressor
                    cate_model = XGBRegressor(
                        random_state=self.random_state,
                        **self.cate_params
                    )
                except ImportError:
                    tprint_warning("   ⚠️ XGBoost not available, falling back to RandomForest")
                    cate_model = RandomForestRegressor(
                        random_state=self.random_state,
                        **{k: v for k, v in self.cate_params.items() if k in ['n_estimators', 'max_depth', 'min_samples_leaf', 'n_jobs']}
                    )
            else:
                # Default to RandomForest
                # Filter params to ensure compatibility if they were meant for another model
                rf_params = {k: v for k, v in self.cate_params.items() if k in ['n_estimators', 'max_depth', 'min_samples_leaf', 'n_jobs', 'max_features']}
                cate_model = RandomForestRegressor(
                    random_state=self.random_state,
                    **rf_params
                )
            
            # Predict treatment effects using heterogeneity features
            # This is a simplified CATE estimation (Residual-on-Residual)
            treatment_effects = outcome_residuals / (treatment_residuals + 1e-8)
            
            # Filter extreme values (outliers in ratio)
            valid_effects = np.abs(treatment_effects) < 10  # Filter extreme values
            X_valid = heterogeneity_data[valid_effects]
            effects_valid = treatment_effects[valid_effects]
            
            if len(X_valid) > 50: # Ensure enough samples for training
                if self.verbose:
                    tprint_info(f"   Fitting {type(cate_model).__name__} for CATE ({len(X_valid)} samples)...")
                cate_model.fit(X_valid, effects_valid)
                cate_estimates = cate_model.predict(heterogeneity_data)
            else:
                if self.verbose:
                    tprint_warning("   ⚠️ Too few valid effects for CATE model, using constant effect")
                # Fallback to constant effect
                cate_estimates = np.full(len(heterogeneity_data), self.causal_effects_["causal_effect"])
            
            # Expand CATE estimates back to original data size if subsampling was used
            if self.subsample_indices_ is not None:
                # Create full-size array with default values
                full_cate_estimates = np.full(len(X), self.causal_effects_["causal_effect"])
                # Fill in the computed values at the subsample positions
                full_cate_estimates[self.subsample_indices_] = cate_estimates
                cate_estimates = full_cate_estimates
            
            cate_series = pd.Series(cate_estimates, index=X.index)
            self.cate_estimates_ = cate_series
            
            if self.verbose:
                tprint_success("✅ CATE Computed:")
                tprint_info(f"   - Mean CATE: {cate_estimates.mean():.6f}")
                tprint_info(f"   - Std CATE: {cate_estimates.std():.6f}")
                tprint_info(f"   - Range: [{cate_estimates.min():.6f}, {cate_estimates.max():.6f}]")
            
            # Save checkpoint after CATE computation
            if self.checkpoint_manager:
                try:
                    checkpoint_data = {
                        'cate_estimates': self.cate_estimates_,
                        'causal_effects': self.causal_effects_
                    }
                    self.checkpoint_manager.save_checkpoint('cate_computed', checkpoint_data, self.symbol)
                    if self.verbose:
                        tprint_info("   💾 Saved checkpoint: cate_computed")
                except Exception as e:
                    if self.verbose:
                        tprint_warning(f"   ⚠️ Failed to save cate_computed checkpoint: {e}")
            
            return cate_series
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ CATE computation failed: {e}")
            return pd.Series(0, index=X.index)
    
    def compute_causal_residuals(
        self,
        X: pd.DataFrame,
        treatments: Union[pd.Series, np.ndarray],
        outcomes: Union[pd.Series, np.ndarray],
        confounders: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Compute causal residuals for Chaser targeting.
        
        Args:
            X: Feature matrix
            treatments: Treatment variable
            outcomes: Outcome variable
            confounders: Confounding variables
            
        Returns:
            Causal residual targets
        """
        try:
            if self.verbose:
                tprint_info("🎯 Computing Causal Residuals for Chaser...")
            
            # Compute causal effects if not available
            if not self.causal_effects_:
                self.compute_dml_causal_effects(X, treatments, outcomes, confounders)
            
            # Get predictions from outcome model
            outcome_predictions = self.causal_effects_["outcome_predictions"].ravel()
            
            # Compute causal residuals
            if isinstance(outcomes, pd.Series):
                outcomes_values = outcomes.values
            else:
                outcomes_values = outcomes.ravel()
            
            # Validate input data for infinity and extreme values
            if self.verbose:
                tprint_info("🔍 Validating outcome data for residual computation...")
            
            # Check for infinity values
            outcomes_inf = np.isinf(outcomes_values).sum()
            predictions_inf = np.isinf(outcome_predictions).sum()
            
            if outcomes_inf > 0:
                if self.verbose:
                    tprint_warning(f"⚠️ Found {outcomes_inf} infinity values in outcomes, replacing with NaN")
                outcomes_values = np.where(np.isinf(outcomes_values), np.nan, outcomes_values)
            
            if predictions_inf > 0:
                if self.verbose:
                    tprint_warning(f"⚠️ Found {predictions_inf} infinity values in predictions, replacing with NaN")
                outcome_predictions = np.where(np.isinf(outcome_predictions), np.nan, outcome_predictions)
            
            # Check for extremely large values
            float64_max = np.finfo(np.float64).max / 1000
            outcomes_large = (np.abs(outcomes_values) > float64_max).sum()
            predictions_large = (np.abs(outcome_predictions) > float64_max).sum()
            
            if outcomes_large > 0:
                if self.verbose:
                    tprint_warning(f"⚠️ Found {outcomes_large} extremely large values in outcomes, clipping")
                outcomes_values = np.clip(outcomes_values, -float64_max, float64_max)
            
            if predictions_large > 0:
                if self.verbose:
                    tprint_warning(f"⚠️ Found {predictions_large} extremely large values in predictions, clipping")
                outcome_predictions = np.clip(outcome_predictions, -float64_max, float64_max)
            
            # Compute residuals with validation
            if self.subsample_indices_ is not None:
                # Handle subsampled case: align outcomes with predictions
                if isinstance(outcomes, pd.Series):
                    outcomes_values = outcomes.iloc[self.subsample_indices_].values
                else:
                    outcomes_values = outcomes[self.subsample_indices_].ravel()
                
                causal_residuals = outcomes_values - outcome_predictions
                
                # Expand residuals back to original size
                full_residuals = np.zeros(len(X))
                full_residuals[self.subsample_indices_] = causal_residuals
                causal_residuals = full_residuals
            else:
                # Full dataset case
                if isinstance(outcomes, pd.Series):
                    outcomes_values = outcomes.values
                else:
                    outcomes_values = outcomes.ravel()
                
                causal_residuals = outcomes_values - outcome_predictions
            
            # Handle any NaN values that resulted from infinity replacement
            nan_mask = np.isnan(causal_residuals)
            if nan_mask.any():
                if self.verbose:
                    tprint_warning(f"⚠️ Found {nan_mask.sum()} NaN residuals after computation, setting to 0")
                causal_residuals = np.where(nan_mask, 0, causal_residuals)
            
            # Convert to Series
            residual_series = pd.Series(causal_residuals, index=X.index)
            self.residual_targets_ = residual_series
            
            if self.verbose:
                tprint_success("✅ Causal Residuals Computed:")
                tprint_info(f"   - Mean residual: {causal_residuals.mean():.6f}")
                tprint_info(f"   - Std residual: {causal_residuals.std():.6f}")
                tprint_info(f"   - Range: [{causal_residuals.min():.6f}, {causal_residuals.max():.6f}]")
            
            return residual_series
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Causal residual computation failed: {e}")
            return pd.Series(0, index=X.index)
    
    def analyze_treatment_effect_heterogeneity(
        self,
        X: pd.DataFrame,
        cate_estimates: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Analyze treatment effect heterogeneity.
        
        Args:
            X: Feature matrix
            cate_estimates: CATE estimates
            
        Returns:
            Heterogeneity analysis results
        """
        try:
            if self.verbose:
                tprint_info("📊 Analyzing Treatment Effect Heterogeneity...")
            
            if cate_estimates is None:
                if self.cate_estimates_ is not None and isinstance(self.cate_estimates_, pd.Series):
                    cate_estimates = self.cate_estimates_
                else:
                    return {"error": "No valid CATE estimates available"}
            
            # Basic statistics
            heterogeneity_stats = {
                "mean_cate": cate_estimates.mean(),
                "std_cate": cate_estimates.std(),
                "min_cate": cate_estimates.min(),
                "max_cate": cate_estimates.max(),
                "range_cate": cate_estimates.max() - cate_estimates.min(),
                "positive_effects": (cate_estimates > 0).mean(),
                "negative_effects": (cate_estimates < 0).mean()
            }
            
            # Feature heterogeneity analysis
            feature_heterogeneity = {}
            
            for feature in X.columns:
                if X[feature].dtype in ['int64', 'float64']:
                    # Correlation with CATE
                    correlation = X[feature].corr(cate_estimates)
                    feature_heterogeneity[feature] = {
                        "correlation": correlation,
                        "abs_correlation": abs(correlation)
                    }
            
            # Sort features by absolute correlation
            sorted_features = sorted(
                feature_heterogeneity.items(),
                key=lambda x: x[1]["abs_correlation"],
                reverse=True
            )
            
            heterogeneity_analysis = {
                "overall_stats": heterogeneity_stats,
                "feature_heterogeneity": dict(sorted_features[:10]),  # Top 10 features
                "top_heterogeneous_features": [feat for feat, stats in sorted_features[:5]]
            }
            
            if self.verbose:
                tprint_success("✅ Heterogeneity Analysis Complete:")
                tprint_info(f"   - Mean CATE: {heterogeneity_stats['mean_cate']:.6f}")
                tprint_info(f"   - CATE std: {heterogeneity_stats['std_cate']:.6f}")
                tprint_info(f"   - Positive effects: {heterogeneity_stats['positive_effects']:.2%}")
                tprint_info(f"   - Top heterogeneous feature: {sorted_features[0][0] if sorted_features else 'None'}")
            
            return heterogeneity_analysis
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Heterogeneity analysis failed: {e}")
            return {"error": str(e)}
    
    def create_chaser_targets(
        self,
        X: pd.DataFrame,
        treatments: Union[pd.Series, np.ndarray],
        outcomes: Union[pd.Series, np.ndarray],
        confounders: Optional[pd.DataFrame] = None,
        include_cate: bool = True
    ) -> Dict[str, pd.Series]:
        """
        Create complete set of targets for Chaser system.
        
        Args:
            X: Feature matrix
            treatments: Treatment variable
            outcomes: Outcome variable
            confounders: Confounding variables
            include_cate: Whether to include CATE estimates
            
        Returns:
            Dictionary of Chaser targets
        """
        try:
            if self.verbose:
                tprint_info("🎯 Creating Chaser Targets...")
            
            chaser_targets = {}
            
            # Primary causal residuals
            causal_residuals = self.compute_causal_residuals(X, treatments, outcomes, confounders)
            chaser_targets["causal_residuals"] = causal_residuals
            
            # CATE estimates (optional)
            if include_cate:
                cate_estimates = self.compute_cate(X, treatments, outcomes, confounders)
                chaser_targets["cate_estimates"] = cate_estimates
            
            # Treatment effect heterogeneity score
            if self.cate_estimates_ is not None and isinstance(self.cate_estimates_, pd.Series):
                heterogeneity_score = (self.cate_estimates_ - self.cate_estimates_.mean()) / (self.cate_estimates_.std() + 1e-8)
                chaser_targets["heterogeneity_score"] = heterogeneity_score
            
            # Treatment residuals (for additional analysis)
            if self.causal_effects_:
                treatment_residuals_data = self.causal_effects_["treatment_residuals"].ravel()
                
                if self.subsample_indices_ is not None:
                    # Expand treatment residuals back to original size
                    full_treatment_residuals = np.zeros(len(X))
                    full_treatment_residuals[self.subsample_indices_] = treatment_residuals_data
                    treatment_residuals_data = full_treatment_residuals
                
                treatment_residuals = pd.Series(treatment_residuals_data, index=X.index)
                chaser_targets["treatment_residuals"] = treatment_residuals
            
            # Add Layer 4-ready causal effect frame columns
            if self.causal_effect_frame_ is not None:
                for col in self.causal_effect_frame_.columns:
                    chaser_targets[col] = self.causal_effect_frame_[col]
            
            if self.verbose:
                tprint_success("✅ Chaser Targets Created:")
                for target_name, target_data in chaser_targets.items():
                    tprint_info(f"   - {target_name}: mean={target_data.mean():.6f}, std={target_data.std():.6f}")
            
            return chaser_targets
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Chaser target creation failed: {e}")
            return {}
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of causal target computation.
        
        Returns:
            Summary dictionary
        """
        summary = {
            "treatment_model": self.treatment_model,
            "outcome_model": self.outcome_model,
            "n_folds": self.n_folds,
            "has_causal_effects": len(self.causal_effects_) > 0,
            "has_cate_estimates": len(self.cate_estimates_) > 0,
            "has_residual_targets": len(self.residual_targets_) > 0
        }
        
        if self.causal_effects_:
            summary["causal_effect"] = self.causal_effects_["causal_effect"]
            summary["causal_effect_se"] = self.causal_effects_["standard_error"]
            summary["r_squared"] = self.causal_effects_["r_squared"]
        
        return summary

# Convenience functions
def quick_causal_targets(
    X: pd.DataFrame,
    treatments: Union[pd.Series, np.ndarray],
    outcomes: Union[pd.Series, np.ndarray],
    **kwargs
) -> Dict[str, pd.Series]:
    """
    Quick causal target computation.
    
    Args:
        X: Feature matrix
        treatments: Treatment variable
        outcomes: Outcome variable
        **kwargs: Additional parameters
        
    Returns:
        Dictionary of causal targets
    """
    computer = CausalTargetComputer(**kwargs)
    return computer.create_chaser_targets(X, treatments, outcomes)

def compute_dml_effects(
    X: pd.DataFrame,
    treatments: Union[pd.Series, np.ndarray],
    outcomes: Union[pd.Series, np.ndarray],
    **kwargs
) -> Dict[str, Any]:
    """
    Quick DML causal effect computation.
    
    Args:
        X: Feature matrix
        treatments: Treatment variable
        outcomes: Outcome variable
        **kwargs: Additional parameters
        
    Returns:
        Causal effect estimates
    """
    computer = CausalTargetComputer(**kwargs)
    return computer.compute_dml_causal_effects(X, treatments, outcomes)

def create_residual_targets(
    X: pd.DataFrame,
    outcomes: Union[pd.Series, np.ndarray],
    predictions: Union[pd.Series, np.ndarray],
    **kwargs
) -> pd.Series:
    """
    Create simple residual targets.
    
    Args:
        X: Feature matrix
        outcomes: True outcomes
        predictions: Predicted outcomes
        **kwargs: Additional parameters
        
    Returns:
        Residual targets
    """
    if isinstance(outcomes, pd.Series):
        outcomes_values = outcomes.values
    else:
        outcomes_values = outcomes.ravel()
    
    if isinstance(predictions, pd.Series):
        predictions_values = predictions.values
    else:
        predictions_values = predictions.ravel()
    
    residuals = outcomes_values - predictions_values
    return pd.Series(residuals, index=X.index)
