"""
Layer 2.5: The Chaser - Non-Linear Alpha Extraction System

The Chaser is the "High-ROI Muscle" that hunts for non-linear alpha
in the gaps of market physics, operating on causal residuals.

Key Components:
1. Causal Residual Targeting (y~ = y_actual - y_causal_anchor)
2. Non-Causal Feature Selection (technical indicators only)
3. Independent XGBoost + CatBoost Models (fed separately to next layer)
4. Conflict Detection with Causal Anchor
5. Confidence Scoring for Meta-Learner
"""

import time
import numpy as np
import scipy.stats as stats
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import catboost as cb
import warnings
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

class Layer25Chaser:
    """
    Layer 2.5 Chaser: Non-linear alpha extraction from causal residuals.
    
    The Chaser learns only the "unexplained alpha" that the Causal Anchor
    cannot capture, focusing on temporary inefficiencies and anomalies.
    """
    
    def __init__(
        self,
        xgb_params: Optional[Dict] = None,
        cat_params: Optional[Dict] = None,
        confidence_threshold: float = 0.5,
        conflict_threshold: float = 2.0,
        verbose: bool = True
    ):
        """
        Initialize Layer 2.5 Chaser.

        Args:
            xgb_params: XGBoost hyperparameters
            cat_params: CatBoost hyperparameters
            confidence_threshold: Minimum confidence for predictions
            conflict_threshold: Threshold for conflict detection (std deviations)
            verbose: Whether to print progress information
        """
        self.verbose = verbose

        # Default XGBoost parameters
        self.xgb_params = xgb_params or {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }

        # Default CatBoost parameters
        self.cat_params = cat_params or {
            'iterations': 200,
            'depth': 6,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': False,
            'od_type': 'Iter',
            'od_wait': 20
        }

        # Thresholds
        self.confidence_threshold = confidence_threshold
        self.conflict_threshold = conflict_threshold

        # Initialize models
        self.xgb_model = None
        self.cat_model = None
        
        # Initialize enhanced components
        self.regime_liquidity_generator = RegimeLiquidityFeatureGenerator(verbose=verbose)
        self.uncertainty_quantifier = CausalUncertaintyQuantification(verbose=verbose)
        
        # Enhanced feature tracking
        self.regime_features_count_ = 0
        self.liquidity_features_count_ = 0
        self.uncertainty_metrics_ = {}
        
        # Initialize enhanced components
        self.regime_liquidity_generator = RegimeLiquidityFeatureGenerator(verbose=verbose)
        self.uncertainty_quantifier = CausalUncertaintyQuantification(verbose=verbose)
        
        # Enhanced feature tracking
        self.regime_features_count_ = 0
        self.liquidity_features_count_ = 0
        self.uncertainty_metrics_ = {}

        # Training metadata
        self.feature_names = None
        self.training_score = None
        self.cv_scores = None
        self.prediction_std = None
        
    def _validate_inputs(
        self,
        X_non_causal: pd.DataFrame,
        y_residuals: pd.Series,
        handle_outliers: bool = False,
        outlier_method: str = 'iqr',
        outlier_threshold: float = 1.5,
        outlier_handling_strategy: str = 'winsorize'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Validate and prepare input data with optional outlier handling."""
        try:
            # Remove NaN values
            valid_mask = ~(X_non_causal.isna().any(axis=1) | y_residuals.isna())
            X_clean = X_non_causal[valid_mask]
            y_clean = y_residuals[valid_mask]

            if len(X_clean) < 100:
                raise ValueError(f"Insufficient training data: {len(X_clean)} samples")

            # Handle outliers if requested
            if handle_outliers:
                y_clean, outlier_mask = self.detect_and_handle_outliers(
                    y_clean,
                    method=outlier_method,
                    threshold=outlier_threshold,
                    handling_strategy=outlier_handling_strategy
                )
                # Remove outliers if strategy is 'remove'
                if outlier_handling_strategy == 'remove':
                    X_clean = X_clean[~outlier_mask]

            if self.verbose:
                tprint_info(f"🔍 Chaser data validation:")
                nan_removed = len(X_non_causal) - len(X_non_causal[valid_mask])
                final_samples = len(X_clean)
                tprint_info(f"   - Samples: {final_samples} (NaN removed: {nan_removed})")
                if handle_outliers and outlier_handling_strategy == 'remove':
                    outliers_removed = outlier_mask.sum()
                    tprint_info(f"   - Outliers removed: {outliers_removed}")
                tprint_info(f"   - Features: {len(X_clean.columns)}")
                tprint_info(f"   - Target mean: {y_clean.mean():.6f}")
                tprint_info(f"   - Target std: {y_clean.std():.6f}")

            return X_clean, y_clean

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Input validation failed: {e}")
            raise
    
    def fit(
        self,
        X_non_causal: pd.DataFrame,
        y_residuals: pd.Series,
        cv_folds: int = 5,
        early_stopping_rounds: int = 50,
        handle_outliers: bool = False,
        outlier_method: str = 'iqr',
        outlier_threshold: float = 1.5,
        outlier_handling_strategy: str = 'winsorize'
    ) -> Dict[str, Any]:
        """
        Fit the Chaser models independently on causal residuals.

        Args:
            X_non_causal: Non-causal features (technical indicators)
            y_residuals: Causal residuals (y_actual - y_causal_anchor)
            cv_folds: Number of cross-validation folds
            early_stopping_rounds: Early stopping patience
            handle_outliers: Whether to detect and handle outliers in residuals
            outlier_method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            outlier_threshold: Threshold for outlier detection
            outlier_handling_strategy: How to handle outliers ('remove', 'winsorize', 'transform')

        Returns:
            Dictionary with training metrics
        """
        try:
            if self.verbose:
                tprint_info("🚀 Training Layer 2.5 Chaser on Causal Residuals...")

            # Validate inputs
            X_clean, y_clean = self._validate_inputs(
                X_non_causal, y_residuals,
                handle_outliers=handle_outliers,
                outlier_method=outlier_method,
                outlier_threshold=outlier_threshold,
                outlier_handling_strategy=outlier_handling_strategy
            )

            # Store feature names
            self.feature_names = X_clean.columns.tolist()

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)

            # XGBoost training with CV
            if self.verbose:
                tprint_info("   📊 Training XGBoost model...")

            self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
            xgb_cv_scores = cross_val_score(
                self.xgb_model, X_clean, y_clean,
                cv=tscv, scoring='neg_mean_squared_error'
            )
            self.xgb_model.fit(X_clean, y_clean)

            # CatBoost training with CV
            if self.verbose:
                tprint_info("   📊 Training CatBoost model...")

            self.cat_model = cb.CatBoostRegressor(**self.cat_params)
            cat_cv_scores = cross_val_score(
                self.cat_model, X_clean, y_clean,
                cv=tscv, scoring='neg_mean_squared_error'
            )
            self.cat_model.fit(X_clean, y_clean)

            # Calculate training metrics for both models
            xgb_train_pred = self.xgb_model.predict(X_clean)
            cat_train_pred = self.cat_model.predict(X_clean)

            xgb_metrics = {
                'mse': mean_squared_error(y_clean, xgb_train_pred),
                'mae': mean_absolute_error(y_clean, xgb_train_pred),
                'rmse': np.sqrt(mean_squared_error(y_clean, xgb_train_pred)),
                'r2': 1 - (np.var(y_clean - xgb_train_pred) / np.var(y_clean))
            }

            cat_metrics = {
                'mse': mean_squared_error(y_clean, cat_train_pred),
                'mae': mean_absolute_error(y_clean, cat_train_pred),
                'rmse': np.sqrt(mean_squared_error(y_clean, cat_train_pred)),
                'r2': 1 - (np.var(y_clean - cat_train_pred) / np.var(y_clean))
            }

            self.training_score = {
                'xgb': xgb_metrics,
                'cat': cat_metrics
            }

            # Store CV scores
            self.cv_scores = {
                'xgb_cv_mse': -xgb_cv_scores.mean(),
                'cat_cv_mse': -cat_cv_scores.mean(),
                'xgb_cv_std': xgb_cv_scores.std(),
                'cat_cv_std': cat_cv_scores.std()
            }

            # Calculate prediction standard deviation for confidence
            self.prediction_std = np.std(np.column_stack([xgb_train_pred, cat_train_pred]), axis=1).mean()

            if self.verbose:
                tprint_success("✅ Chaser training complete!")
                tprint_info(f"   - XGBoost - Training RMSE: {xgb_metrics['rmse']:.6f}, R²: {xgb_metrics['r2']:.4f}")
                tprint_info(f"   - CatBoost - Training RMSE: {cat_metrics['rmse']:.6f}, R²: {cat_metrics['r2']:.4f}")
                tprint_info(f"   - XGBoost CV RMSE: {np.sqrt(self.cv_scores['xgb_cv_mse']):.6f}")
                tprint_info(f"   - CatBoost CV RMSE: {np.sqrt(self.cv_scores['cat_cv_mse']):.6f}")
                tprint_info(f"   - Prediction std: {self.prediction_std:.6f}")

            return {
                'training_metrics': self.training_score,
                'cv_metrics': self.cv_scores,
                'feature_count': len(self.feature_names),
                'sample_count': len(X_clean)
            }

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Chaser training failed: {e}")
            raise
    
    def predict(
        self,
        X_non_causal: pd.DataFrame,
        return_confidence: bool = True
    ) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], np.ndarray]]:
        """
        Predict residual alpha using both Chaser models independently.

        Args:
            X_non_causal: Non-causal features
            return_confidence: Whether to return confidence scores

        Returns:
            Dictionary with predictions from both models and optionally confidence scores
        """
        try:
            if self.xgb_model is None or self.cat_model is None:
                raise ValueError("Chaser models not fitted. Call fit() first.")

            # Ensure feature order matches training
            if self.feature_names is not None:
                X_aligned = X_non_causal[self.feature_names].fillna(0)
            else:
                X_aligned = X_non_causal.fillna(0)

            # Get predictions from both models independently
            xgb_pred = self.xgb_model.predict(X_aligned)
            cat_pred = self.cat_model.predict(X_aligned)

            predictions = {
                'xgb': xgb_pred,
                'cat': cat_pred
            }

            if not return_confidence:
                return predictions

            # Calculate confidence based on model agreement
            pred_std = np.std(np.column_stack([xgb_pred, cat_pred]), axis=1)
            confidence = 1.0 / (1.0 + pred_std / (self.prediction_std + 1e-8))

            return predictions, confidence

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Chaser prediction failed: {e}")
            raise
    
    def detect_conflict(
        self,
        chaser_predictions: Dict[str, np.ndarray],
        causal_anchor_prediction: np.ndarray,
        chaser_confidence: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Detect conflict between Chaser models and Causal Anchor.

        Args:
            chaser_predictions: Dictionary with predictions from both Chaser models
            causal_anchor_prediction: Causal Anchor baseline predictions
            chaser_confidence: Chaser confidence scores

        Returns:
            Dictionary with conflict metrics for both models
        """
        try:
            conflict_results = {}

            for model_name, chaser_prediction in chaser_predictions.items():
                # Total prediction (Anchor + Chaser)
                total_prediction = causal_anchor_prediction + chaser_prediction

                # Conflict detection: Chaser betting against Anchor
                conflict_direction = np.sign(chaser_prediction) != np.sign(causal_anchor_prediction)
                conflict_magnitude = np.abs(chaser_prediction) / (np.abs(causal_anchor_prediction) + 1e-8)

                # Conflict flag (high confidence + opposite direction)
                conflict_flag = conflict_direction & (chaser_confidence > self.confidence_threshold)

                # Conflict intensity (weighted by confidence and magnitude)
                conflict_intensity = conflict_flag.astype(float) * chaser_confidence * conflict_magnitude

                conflict_results[model_name] = {
                    'conflict_flag': conflict_flag,
                    'conflict_intensity': conflict_intensity,
                    'conflict_direction': conflict_direction.astype(int),
                    'conflict_magnitude': conflict_magnitude,
                    'total_prediction': total_prediction
                }

            return conflict_results

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Conflict detection failed: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Get feature importance from both models.
        
        Returns:
            Dictionary with feature importance from XGBoost and CatBoost
        """
        try:
            if self.xgb_model is None or self.cat_model is None:
                raise ValueError("Models not fitted yet")
            
            importance = {
                'xgb_importance': dict(zip(self.feature_names, self.xgb_model.feature_importances_)),
                'cat_importance': dict(zip(self.feature_names, self.cat_model.get_feature_importance()))
            }
            
            # Average importance
            avg_importance = {}
            for feature in self.feature_names:
                avg_importance[feature] = (
                    importance['xgb_importance'][feature] + 
                    importance['cat_importance'][feature]
                ) / 2.0
            
            importance['avg_importance'] = avg_importance
            
            return importance
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Feature importance extraction failed: {e}")
            raise
    
    def evaluate(
        self,
        X_non_causal: pd.DataFrame,
        y_residuals: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate Chaser models performance on test data.

        Args:
            X_non_causal: Test features
            y_residuals: Test residuals

        Returns:
            Dictionary with evaluation metrics for both models
        """
        try:
            predictions, confidence = self.predict(X_non_causal, return_confidence=True)

            evaluation_results = {}

            for model_name, model_predictions in predictions.items():
                metrics = {
                    'mse': mean_squared_error(y_residuals, model_predictions),
                    'mae': mean_absolute_error(y_residuals, model_predictions),
                    'rmse': np.sqrt(mean_squared_error(y_residuals, model_predictions)),
                    'r2': 1 - (np.var(y_residuals - model_predictions) / np.var(y_residuals))
                }
                evaluation_results[model_name] = metrics

            # Add confidence metrics (shared across models)
            evaluation_results['confidence'] = {
                'mean_confidence': np.mean(confidence),
                'high_confidence_ratio': np.mean(confidence > self.confidence_threshold)
            }

            return evaluation_results

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Chaser evaluation failed: {e}")
            raise

    def optimize_hyperparameters(
        self,
        X_non_causal: pd.DataFrame,
        y_residuals: pd.Series,
        optimization_fraction: float = 0.3,
        n_trials: int = 100,
        timeout: int = 3600,
        cv_folds: int = 3,
        random_state: int = 42
    ) -> Dict[str, Dict[str, Any]]:
        """
        Optimize hyperparameters for both XGBoost and CatBoost using Optuna.

        Args:
            X_non_causal: Non-causal features
            y_residuals: Target residuals
            optimization_fraction: Fraction of data to use for optimization (default 30%)
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            cv_folds: Number of CV folds for evaluation
            random_state: Random state for reproducibility

        Returns:
            Dictionary with optimized parameters for both models
        """
        try:
            if self.verbose:
                tprint_info("🔬 Starting hyperparameter optimization with Optuna...")

            # Use subset of data for optimization (30%)
            if optimization_fraction < 1.0:
                X_opt, _, y_opt, _ = train_test_split(
                    X_non_causal, y_residuals,
                    train_size=optimization_fraction,
                    random_state=random_state,
                    shuffle=False  # Preserve time series order
                )
                if self.verbose:
                    tprint_info(f"   📊 Using {len(X_opt)} samples ({optimization_fraction*100:.0f}%) for optimization")
            else:
                X_opt, y_opt = X_non_causal, y_residuals

            # Time series cross-validation for optimization
            tscv = TimeSeriesSplit(n_splits=cv_folds)

            # Optimize XGBoost
            if self.verbose:
                tprint_info("   🚀 Optimizing XGBoost hyperparameters...")

            def xgb_objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                    'random_state': random_state,
                    'n_jobs': -1,
                }

                model = xgb.XGBRegressor(**params)

                # Time series cross-validation scores
                cv_scores = cross_val_score(
                    model, X_opt, y_opt, cv=tscv,
                    scoring='neg_mean_squared_error'
                )

                # Report intermediate results for pruning
                trial.report(-cv_scores.mean(), step=0)

                # Prune if necessary
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return -cv_scores.mean()  # Return negative MSE (higher is better for maximization)

            xgb_study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=random_state),
                pruner=MedianPruner()
            )

            xgb_study.optimize(xgb_objective, n_trials=n_trials//2, timeout=timeout//2)

            # Optimize CatBoost
            if self.verbose:
                tprint_info("   🚀 Optimizing CatBoost hyperparameters...")

            def cat_objective(trial):
                params = {
                    'iterations': trial.suggest_int('iterations', 50, 500),
                    'depth': trial.suggest_int('depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                    'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'random_seed': random_state,
                    'verbose': False,
                    'od_type': 'Iter',
                    'od_wait': trial.suggest_int('od_wait', 10, 50)
                }

                model = cb.CatBoostRegressor(**params)

                # Time series cross-validation scores
                cv_scores = cross_val_score(
                    model, X_opt, y_opt, cv=tscv,
                    scoring='neg_mean_squared_error'
                )

                # Report intermediate results for pruning
                trial.report(-cv_scores.mean(), step=0)

                # Prune if necessary
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return -cv_scores.mean()  # Return negative MSE (higher is better for maximization)

            cat_study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=random_state),
                pruner=MedianPruner()
            )

            cat_study.optimize(cat_objective, n_trials=n_trials//2, timeout=timeout//2)

            # Store optimized parameters
            optimized_params = {
                'xgb': xgb_study.best_params,
                'cat': cat_study.best_params,
                'optimization_info': {
                    'xgb_best_score': xgb_study.best_value,
                    'cat_best_score': cat_study.best_value,
                    'xgb_trials': len(xgb_study.trials),
                    'cat_trials': len(cat_study.trials),
                    'optimization_fraction': optimization_fraction,
                    'samples_used': len(X_opt)
                }
            }

            # Update instance parameters
            self.xgb_params.update(xgb_study.best_params)
            self.cat_params.update(cat_study.best_params)

            if self.verbose:
                tprint_success("✅ Hyperparameter optimization complete!")
                tprint_info(f"   📊 XGBoost best CV score: {-xgb_study.best_value:.6f}")
                tprint_info(f"   📊 CatBoost best CV score: {-cat_study.best_value:.6f}")
                tprint_info(f"   📊 Total trials: {len(xgb_study.trials) + len(cat_study.trials)}")

            return optimized_params

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Hyperparameter optimization failed: {e}")
            raise

    def detect_and_handle_outliers(
        self,
        y_residuals: pd.Series,
        method: str = 'iqr',
        threshold: float = 1.5,
        handling_strategy: str = 'winsorize',
        contamination: float = 0.05
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Detect and handle outliers in residual data.

        Args:
            y_residuals: Target residuals
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for outlier detection (IQR multiplier or z-score)
            handling_strategy: How to handle outliers ('remove', 'winsorize', 'transform')
            contamination: Expected proportion of outliers (for isolation forest)

        Returns:
            Tuple of (clean_residuals, outlier_mask)
        """
        try:
            if self.verbose:
                tprint_info(f"🔍 Detecting outliers using {method} method...")

            outlier_mask = pd.Series(False, index=y_residuals.index)

            if method == 'iqr':
                # IQR-based outlier detection
                Q1 = y_residuals.quantile(0.25)
                Q3 = y_residuals.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (y_residuals < lower_bound) | (y_residuals > upper_bound)

            elif method == 'zscore':
                # Z-score based outlier detection
                from scipy import stats
                z_scores = np.abs(stats.zscore(y_residuals))
                outlier_mask = z_scores > threshold

            elif method == 'isolation_forest':
                # Isolation Forest for outlier detection
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(
                    contamination=contamination,
                    random_state=42,
                    n_estimators=100
                )
                # Need to reshape for sklearn
                outlier_predictions = iso_forest.fit_predict(y_residuals.values.reshape(-1, 1))
                outlier_mask = outlier_predictions == -1

            else:
                raise ValueError(f"Unknown outlier detection method: {method}")

            n_outliers = outlier_mask.sum()

            if self.verbose:
                tprint_warning(f"⚠️  Detected {n_outliers} outliers ({n_outliers/len(y_residuals)*100:.2f}%) using {method}")

            # Handle outliers based on strategy
            if handling_strategy == 'remove':
                clean_residuals = y_residuals[~outlier_mask]
                if self.verbose:
                    tprint_info(f"   🗑️  Removed {n_outliers} outliers, {len(clean_residuals)} samples remaining")

            elif handling_strategy == 'winsorize':
                # Winsorize to percentile bounds
                lower_bound = y_residuals[~outlier_mask].quantile(0.05)
                upper_bound = y_residuals[~outlier_mask].quantile(0.95)
                clean_residuals = np.clip(y_residuals, lower_bound, upper_bound)
                if self.verbose:
                    tprint_info(f"   ✂️  Winsorized outliers to [{lower_bound:.6f}, {upper_bound:.6f}] range")

            elif handling_strategy == 'transform':
                # Log transformation for positive outliers (assuming residuals can be positive/negative)
                clean_residuals = y_residuals.copy()
                # Apply log transformation to extreme values
                extreme_mask = outlier_mask & (np.abs(y_residuals) > y_residuals.std() * 2)
                if extreme_mask.any():
                    # Use signed log transformation
                    signs = np.sign(clean_residuals[extreme_mask])
                    transformed = np.log1p(np.abs(clean_residuals[extreme_mask]))
                    clean_residuals.loc[extreme_mask] = signs * transformed
                    if self.verbose:
                        tprint_info(f"   🔄 Log-transformed {extreme_mask.sum()} extreme outliers")

            else:
                raise ValueError(f"Unknown outlier handling strategy: {handling_strategy}")

            return clean_residuals, outlier_mask

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Outlier detection/handling failed: {e}")
            raise

# Convenience functions
def create_chaser(
    xgb_params: Optional[Dict] = None,
    cat_params: Optional[Dict] = None,
    **kwargs
) -> Layer25Chaser:
    """
    Create a Chaser instance with default or custom parameters.

    Args:
        xgb_params: XGBoost parameters
        cat_params: CatBoost parameters
        **kwargs: Additional parameters

    Returns:
        Configured Chaser instance
    """
    return Layer25Chaser(
        xgb_params=xgb_params,
        cat_params=cat_params,
        **kwargs
    )

def quick_chaser_fit(
    X_non_causal: pd.DataFrame,
    y_residuals: pd.Series,
    **kwargs
) -> Layer25Chaser:
    """
    Quick fit a Chaser with default parameters.
    
    Args:
        X_non_causal: Non-causal features
        y_residuals: Causal residuals
        **kwargs: Additional parameters
        
    Returns:
        Fitted Chaser instance
    """
    chaser = create_chaser(**kwargs)
    chaser.fit(X_non_causal, y_residuals)
    return chaser
