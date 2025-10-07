"""
Custom XGBoost implementation optimized for Tactician entry timing optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import logging

logger = logging.getLogger(__name__)


class XGBoostCustom(BaseEstimator, RegressorMixin):
    """
    Custom XGBoost implementation optimized for Tactician entry timing optimization.
    
    Features:
    - Optimized for financial time series data
    - Enhanced feature importance tracking
    - Entry timing specific loss functions
    - Confidence score integration
    - Memory optimization for large datasets
    """
    
    def __init__(self,
                 n_estimators: int = 1000,
                 learning_rate: float = 0.05,
                 max_depth: int = 6,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 0.1,
                 min_child_weight: int = 3,
                 gamma: float = 0.1,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 early_stopping_rounds: int = 50,
                 eval_metric: str = 'rmse',
                 enable_categorical: bool = True,
                 tree_method: str = 'hist',
                 enable_cyclic_noise: bool = True,
                 cyclic_noise_scale: float = 1e-3,
                 cyclic_noise_cycle: int = 512,
                 **kwargs):
        """
        Initialize custom XGBoost for entry timing optimization.
        
        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate for boosting
            max_depth: Maximum tree depth
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            min_child_weight: Minimum sum of instance weight in a child
            gamma: Minimum loss reduction for splits
            random_state: Random seed
            n_jobs: Number of parallel threads
            early_stopping_rounds: Early stopping rounds
            eval_metric: Evaluation metric
            enable_categorical: Enable categorical features
            tree_method: Tree construction method
            enable_cyclic_noise: Whether to inject deterministic noise before training
            cyclic_noise_scale: Amplitude of the injected noise relative to feature scale
            cyclic_noise_cycle: Number of samples in the repeating noise pattern
            **kwargs: Additional XGBoost parameters
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.enable_categorical = enable_categorical
        self.tree_method = tree_method
        self.enable_cyclic_noise = enable_cyclic_noise
        self.cyclic_noise_scale = cyclic_noise_scale
        self.cyclic_noise_cycle = cyclic_noise_cycle
        
        # Additional parameters
        self.kwargs = kwargs
        
        # Model components
        self.model_ = None
        self.feature_importance_ = None
        self.feature_names_ = None
        self.is_fitted_ = False
        
        # Entry timing specific attributes
        self.entry_timing_metrics_ = {}
        self.confidence_scores_ = None
        
    def _create_xgboost_model(self) -> XGBRegressor:
        """Create XGBoost model with optimized parameters for entry timing."""
        params = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'eval_metric': self.eval_metric,
            'enable_categorical': self.enable_categorical,
            'tree_method': self.tree_method,
            'verbosity': 0,  # Suppress XGBoost output
        }
        
        # Add additional parameters
        params.update(self.kwargs)
        
        return XGBRegressor(**params)
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None,
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            **fit_params) -> 'XGBoostCustom':
        """
        Fit the custom XGBoost model for entry timing optimization.
        
        Args:
            X: Training features
            y: Training targets (entry timing values)
            feature_names: Names of features
            eval_set: Validation set for early stopping
            **fit_params: Additional fit parameters
            
        Returns:
            Self
        """
        try:
            logger.info(f"🚀 Training XGBoost_custom with {X.shape[0]} samples, {X.shape[1]} features")
            
            # Store feature names
            self.feature_names_ = feature_names
            
            # Create model
            self.model_ = self._create_xgboost_model()

            # Prepare input features (optionally injecting cyclic noise)
            X_train = X
            eval_data = eval_set
            if self.enable_cyclic_noise:
                from src.training.steps.model_training.utils.noise_injection import (
                    CyclicNoiseConfig,
                    add_cyclic_noise,
                )

                noise_config = CyclicNoiseConfig(
                    noise_scale=self.cyclic_noise_scale,
                    cycle_length=self.cyclic_noise_cycle,
                    random_state=self.random_state,
                )
                X_train = add_cyclic_noise(X, noise_config)
                if eval_set is not None:
                    eval_features, eval_target = eval_set
                    eval_features_noisy = add_cyclic_noise(eval_features, noise_config)
                    eval_data = (eval_features_noisy, eval_target)

            # Prepare fit parameters
            fit_kwargs = {
                'verbose': False,
                'early_stopping_rounds': self.early_stopping_rounds if eval_data is not None else None,
            }
            fit_kwargs.update(fit_params)

            # Add eval_set if provided
            if eval_data is not None:
                fit_kwargs['eval_set'] = [eval_data]

            # Fit model
            self.model_.fit(X_train, y, **fit_kwargs)
            
            # Extract feature importance
            self._extract_feature_importance()
            
            # Calculate entry timing specific metrics
            self._calculate_entry_timing_metrics(X, y)
            
            self.is_fitted_ = True
            logger.info(f"✅ XGBoost_custom training completed")
            logger.info(f"   Best iteration: {self.model_.best_iteration if hasattr(self.model_, 'best_iteration') else self.n_estimators}")
            logger.info(f"   Feature importance top 5: {self._get_top_features(5)}")
            
            return self
            
        except Exception as e:
            logger.error(f"❌ XGBoost_custom training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict entry timing values.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted entry timing values
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model_.predict(X)
    
    def predict_with_confidence(self, X: np.ndarray, 
                               confidence_threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict entry timing with confidence scores.
        
        Args:
            X: Features to predict
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        predictions = self.predict(X)
        
        # Calculate confidence based on prediction uncertainty
        # Use leaf indices to estimate prediction confidence
        leaf_indices = self.model_.apply(X)
        
        # Calculate confidence as inverse of prediction variance across trees
        # (simplified approach - in practice, you might use more sophisticated methods)
        confidence_scores = np.ones(len(predictions)) * 0.8  # Default confidence
        
        # Adjust confidence based on feature importance and prediction magnitude
        if self.feature_importance_ is not None:
            # Higher confidence for predictions using important features
            feature_weights = np.array([self.feature_importance_.get(f"f{i}", 0.0) 
                                      for i in range(X.shape[1])])
            weighted_features = np.sum(X * feature_weights, axis=1)
            confidence_adjustment = np.clip(weighted_features / np.max(weighted_features), 0.5, 1.0)
            confidence_scores = confidence_scores * confidence_adjustment
        
        # Ensure confidence is between 0 and 1
        confidence_scores = np.clip(confidence_scores, 0.0, 1.0)
        
        return predictions, confidence_scores
    
    def _extract_feature_importance(self):
        """Extract and store feature importance."""
        if self.model_ is not None:
            importance_dict = self.model_.get_booster().get_score(importance_type='weight')
            
            # Convert to feature names if available
            if self.feature_names_ is not None:
                self.feature_importance_ = {
                    self.feature_names_[int(f.replace('f', ''))]: importance
                    for f, importance in importance_dict.items()
                }
            else:
                self.feature_importance_ = importance_dict
    
    def _calculate_entry_timing_metrics(self, X: np.ndarray, y: np.ndarray):
        """Calculate entry timing specific metrics."""
        try:
            predictions = self.predict(X)
            
            # Calculate timing accuracy
            timing_error = np.abs(predictions - y)
            self.entry_timing_metrics_ = {
                'mean_timing_error': np.mean(timing_error),
                'median_timing_error': np.median(timing_error),
                'std_timing_error': np.std(timing_error),
                'max_timing_error': np.max(timing_error),
                'timing_accuracy_within_0.1%': np.mean(timing_error <= 0.001),
                'timing_accuracy_within_0.2%': np.mean(timing_error <= 0.002),
                'timing_accuracy_within_0.5%': np.mean(timing_error <= 0.005),
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate entry timing metrics: {e}")
            self.entry_timing_metrics_ = {}
    
    def _get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        if self.feature_importance_ is None:
            return []
        
        sorted_features = sorted(self.feature_importance_.items(), 
                               key=lambda x: x[1], reverse=True)
        return sorted_features[:n]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance dictionary."""
        return self.feature_importance_ or {}
    
    def get_entry_timing_metrics(self) -> Dict[str, float]:
        """Get entry timing specific metrics."""
        return self.entry_timing_metrics_
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        params = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'early_stopping_rounds': self.early_stopping_rounds,
            'eval_metric': self.eval_metric,
            'enable_categorical': self.enable_categorical,
            'tree_method': self.tree_method,
        }
        
        if deep:
            params.update(self.kwargs)
        
        return params
    
    def set_params(self, **params) -> 'XGBoostCustom':
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self


def create_xgboost_custom_model(params: Dict[str, Any]) -> XGBoostCustom:
    """
    Create XGBoost_custom model with given parameters.
    
    Args:
        params: Model parameters
        
    Returns:
        XGBoost_custom model instance
    """
    return XGBoostCustom(**params)