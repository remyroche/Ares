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
        treatment_model: str = "random_forest",
        outcome_model: str = "random_forest",
        n_folds: int = 5,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize Causal Target Computer.
        
        Args:
            treatment_model: Model for treatment prediction
            outcome_model: Model for outcome prediction
            n_folds: Number of cross-validation folds
            random_state: Random seed
            verbose: Whether to print progress information
        """
        self.treatment_model = treatment_model
        self.outcome_model = outcome_model
        self.n_folds = n_folds
        self.random_state = random_state
        self.verbose = verbose
        
        # Storage for models and results
        self.treatment_models_ = {}
        self.outcome_models_ = {}
        self.causal_effects_ = {}
        self.cate_estimates_ = None
        self.residual_targets_ = {}
        self.refutation_scores_ = None
        self.causal_effect_frame_ = None
        
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
            "linear": LinearRegression(),
            "ridge": Ridge(alpha=1.0),
            "lasso": Lasso(alpha=1.0),
            "random_forest": RandomForestRegressor(
                n_estimators=100, random_state=self.random_state
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100, random_state=self.random_state
            )
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return models[model_name]
    
    def get_guarded_confounders(
        self,
        X: pd.DataFrame,
        treatment_cols: List[str],
        correlation_threshold: float = 0.85
    ) -> pd.DataFrame:
        """
        Enhanced 2026 Causal Filter:
        1. Removes Treatment Leakage (T is not in W).
        2. Implements Temporal Discipline (Pre-Treatment check).
        3. Handles Multi-collinearity (Denoising W).
        """
        # 1. Start with all numeric
        potential_w = X.select_dtypes(include=[np.number])

        # 2. EXCLUDE the treatment itself and any direct descendants
        potential_w = potential_w.drop(columns=treatment_cols, errors='ignore')

        # 3. TEMPORAL DISCIPLINE: Only keep features that are ex-ante (historical)
        # We look for 'lag', 'rolling', or '_w' (rolling window) to identify historical context
        valid_confounders = [
            c for c in potential_w.columns
            if 'lag' in c.lower() or 'rolling' in c.lower() or '_w' in c.lower()
        ]

        # If filtering removed everything, fall back to potential_w (with warning)
        # to prevent DML failure, but log it.
        if not valid_confounders:
            if self.verbose:
                tprint_warning("   ⚠️ Guarded selection removed all features! Falling back to all numeric features (potential leakage).")
            return potential_w

        # 4. DENOISING: Remove redundant confounders
        w_subset = potential_w[valid_confounders]
        corr_matrix = w_subset.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

        final_w_cols = [c for c in valid_confounders if c not in to_drop]

        if self.verbose:
            tprint_info(f"   🛡️ Guarded Confounders: Selected {len(final_w_cols)} features (dropped {len(to_drop)} redundant)")

        return X[final_w_cols]

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
            
            # Determine treatment columns to exclude BEFORE converting to numpy
            treatment_cols = []
            if isinstance(treatments, pd.Series):
                if treatments.name:
                    treatment_cols = [str(treatments.name)]
            elif isinstance(treatments, pd.DataFrame):
                treatment_cols = [str(c) for c in treatments.columns]

            # Convert to numpy arrays for downstream math while keeping indices for confounders
            treatments_array = np.asarray(treatments)
            outcomes_array = np.asarray(outcomes)
            
            if treatments_array.ndim == 1:
                treatments_array = treatments_array.reshape(-1, 1)
            if outcomes_array.ndim == 1:
                outcomes_array = outcomes_array.reshape(-1, 1)
            
            # Use X as confounders if not provided, using guarded selection
            if confounders is None:
                confounders = self.get_guarded_confounders(X, treatment_cols)
            else:
                confounders = pd.concat(
                    [
                        self.get_guarded_confounders(X, treatment_cols),
                        confounders.select_dtypes(include=[np.number])
                    ],
                    axis=1
                )
                # Deduplicate columns if explicit confounders overlap with X
                confounders = confounders.loc[:, ~confounders.columns.duplicated()]
            
            # Ensure we have some numeric features
            if confounders.empty:
                raise ValueError("No numeric features available as confounders")
            
            n_samples = len(treatments_array)
            treatment_width = treatments_array.shape[1]
            outcome_width = outcomes_array.shape[1]
            
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
                "outcome_predictions": outcome_pred
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
            
            # Use treatment residuals for CATE estimation
            treatment_residuals = self.causal_effects_["treatment_residuals"].ravel()
            outcome_residuals = self.causal_effects_["outcome_residuals"].ravel()
            
            # Fit heterogeneity model
            cate_model = RandomForestRegressor(
                n_estimators=100, random_state=self.random_state
            )
            
            # Predict treatment effects using heterogeneity features
            # This is a simplified CATE estimation
            treatment_effects = outcome_residuals / (treatment_residuals + 1e-8)
            
            # Filter extreme values
            valid_effects = np.abs(treatment_effects) < 10  # Filter extreme values
            X_valid = heterogeneity_data[valid_effects]
            effects_valid = treatment_effects[valid_effects]
            
            if len(X_valid) > 10:
                cate_model.fit(X_valid, effects_valid)
                cate_estimates = cate_model.predict(heterogeneity_data)
            else:
                # Fallback to constant effect
                cate_estimates = np.full(len(X), self.causal_effects_["causal_effect"])
            
            cate_series = pd.Series(cate_estimates, index=X.index)
            self.cate_estimates_ = cate_series
            
            if self.verbose:
                tprint_success("✅ CATE Computed:")
                tprint_info(f"   - Mean CATE: {cate_estimates.mean():.6f}")
                tprint_info(f"   - Std CATE: {cate_estimates.std():.6f}")
                tprint_info(f"   - Range: [{cate_estimates.min():.6f}, {cate_estimates.max():.6f}]")
            
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
            
            causal_residuals = outcomes_values - outcome_predictions
            
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
                treatment_residuals = pd.Series(
                    self.causal_effects_["treatment_residuals"].ravel(),
                    index=X.index
                )
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
