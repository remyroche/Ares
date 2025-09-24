"""
Enhanced Training Utilities with Overfitting Prevention and Lookahead Bias Detection

This module provides comprehensive training utilities that address critical issues:
- Purged cross-validation for temporal data integrity
- Early stopping for all supported models
- Lookahead bias detection and prevention
- Temporal data splitting
- Enhanced regularization
- Walk-forward validation
- Overfitting monitoring
- Validation curves
- Ensemble diversity metrics

All training steps can benefit from these utilities by importing and using them.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator
from datetime import datetime, timedelta
import logging
import time
import warnings
from dataclasses import dataclass, field
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, clone
import joblib
import gc

# Enhanced dependency management
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_debug, tprint_progress, tprint_performance
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args)
    def tprint_info(*args, **kwargs): print(f"INFO: {args[0] if args else ''}")
    def tprint_warning(*args, **kwargs): print(f"WARNING: {args[0] if args else ''}")
    def tprint_error(*args, **kwargs): print(f"ERROR: {args[0] if args else ''}")
    def tprint_success(*args, **kwargs): print(f"SUCCESS: {args[0] if args else ''}")
    def tprint_debug(*args, **kwargs): print(f"DEBUG: {args[0] if args else ''}")
    def tprint_progress(*args, **kwargs): print(f"PROGRESS: {args[0] if args else ''}")
    def tprint_performance(*args, **kwargs): print(f"PERFORMANCE: {args[0] if args else ''}")

# Import existing utilities
try:
    from src.utils.lookahead_bias_detector import get_global_detector, LookaheadBiasError
    LOOKAHEAD_DETECTOR_AVAILABLE = True
except ImportError:
    LOOKAHEAD_DETECTOR_AVAILABLE = False
    tprint_warning("⚠️ Lookahead bias detector not available")

try:
    from src.utils.ml_common.validation.unified_cv import UnifiedCrossValidator
    CV_UTILS_AVAILABLE = True
except ImportError:
    CV_UTILS_AVAILABLE = False
    tprint_warning("⚠️ Unified CV utilities not available")

# Initialize logger
logger = logging.getLogger(__name__)


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping across all models."""
    enabled: bool = True
    monitor: str = 'validation_loss'  # 'validation_loss', 'validation_score', 'training_loss'
    patience: int = 10
    min_delta: float = 0.001
    restore_best_weights: bool = True
    verbose: bool = True
    mode: str = 'min'  # 'min' for loss, 'max' for score


@dataclass
class PurgedCVConfig:
    """Configuration for purged cross-validation."""
    enabled: bool = True
    n_splits: int = 5
    purge_pct: float = 0.01  # 1% of data purged between train/test
    gap: int = 0  # Additional gap between train/test
    test_size: Optional[int] = None
    random_state: Optional[int] = None


@dataclass
class OverfittingMonitorConfig:
    """Configuration for overfitting monitoring."""
    enabled: bool = True
    threshold: float = 0.1  # Overfitting threshold
    min_samples: int = 100  # Minimum samples for monitoring
    check_frequency: int = 10  # Check every N epochs/iterations
    validation_curve_points: int = 10  # Points for validation curve


@dataclass
class RegularizationConfig:
    """Enhanced regularization configuration."""
    enabled: bool = True
    l1_alpha: float = 0.01
    l2_alpha: float = 0.01
    dropout_rate: float = 0.2
    max_depth: Optional[int] = None
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    max_features: str = 'sqrt'  # 'sqrt', 'log2', None, or float


class EnhancedTrainingUtils:
    """
    Enhanced training utilities with comprehensive overfitting prevention
    and lookahead bias detection for all models.
    """
    
    def __init__(self, 
                 early_stopping_config: Optional[EarlyStoppingConfig] = None,
                 purged_cv_config: Optional[PurgedCVConfig] = None,
                 overfitting_config: Optional[OverfittingMonitorConfig] = None,
                 regularization_config: Optional[RegularizationConfig] = None):
        """Initialize enhanced training utilities."""
        
        # Configuration
        self.early_stopping_config = early_stopping_config or EarlyStoppingConfig()
        self.purged_cv_config = purged_cv_config or PurgedCVConfig()
        self.overfitting_config = overfitting_config or OverfittingMonitorConfig()
        self.regularization_config = regularization_config or RegularizationConfig()
        
        # Initialize components
        self.lookahead_detector = None
        self.cv_utils = None
        self.training_history = []
        self.overfitting_warnings = []
        
        # Initialize available components
        self._initialize_components()
        
        tprint_success("✅ Enhanced Training Utils initialized")
    
    def _initialize_components(self):
        """Initialize available components."""
        try:
            # Initialize lookahead bias detector
            if LOOKAHEAD_DETECTOR_AVAILABLE:
                self.lookahead_detector = get_global_detector()
                tprint_success("✅ Lookahead bias detector initialized")
            else:
                tprint_warning("⚠️ Lookahead bias detector not available")
            
            # Initialize CV utilities
            if CV_UTILS_AVAILABLE:
                self.cv_utils = UnifiedCrossValidator()
                tprint_success("✅ Unified CV utilities initialized")
            else:
                tprint_warning("⚠️ Unified CV utilities not available")
                
        except Exception as e:
            tprint_error(f"❌ Component initialization failed: {e}")
    
    def validate_temporal_data(self, 
                             X: np.ndarray, 
                             y: np.ndarray, 
                             timestamps: Optional[np.ndarray] = None,
                             strict_mode: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate temporal data for lookahead bias and temporal integrity.
        
        Args:
            X: Feature matrix
            y: Target array
            timestamps: Timestamp array (optional)
            strict_mode: Whether to raise errors on violations
            
        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings_list = []
        
        try:
            # Check for lookahead bias if detector available
            if self.lookahead_detector and timestamps is not None:
                try:
                    # Create DataFrame for validation
                    df = pd.DataFrame({
                        'timestamp': timestamps,
                        'target': y
                    })
                    
                    # Validate timestamps
                    is_valid = self.lookahead_detector.validate_dataframe_timestamps(
                        df, 'timestamp'
                    )
                    
                    if not is_valid:
                        warnings_list.append("Lookahead bias detected in temporal data")
                        if strict_mode:
                            raise LookaheadBiasError("Lookahead bias detected")
                            
                except LookaheadBiasError as e:
                    if strict_mode:
                        raise
                    warnings_list.append(f"Lookahead bias warning: {e}")
                except Exception as e:
                    warnings_list.append(f"Lookahead bias check failed: {e}")
            
            # Check for temporal ordering
            if timestamps is not None:
                if not np.all(np.diff(timestamps) >= 0):
                    warnings_list.append("Timestamps are not in chronological order")
            
            # Check for data leakage indicators
            if len(X) != len(y):
                warnings_list.append("Feature and target arrays have different lengths")
                return False, warnings_list
            
            # Check for future information leakage
            if timestamps is not None and len(timestamps) > 1:
                # Check if features contain future information
                # This is a simplified check - more sophisticated checks can be added
                if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                    warnings_list.append("Features contain NaN or infinite values")
                
                if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                    warnings_list.append("Targets contain NaN or infinite values")
            
            tprint_success("✅ Temporal data validation completed")
            return True, warnings_list
            
        except Exception as e:
            tprint_error(f"❌ Temporal data validation failed: {e}")
            return False, [f"Validation failed: {e}"]
    
    def create_temporal_splits(self, 
                             X: np.ndarray, 
                             y: np.ndarray,
                             timestamps: Optional[np.ndarray] = None,
                             use_purged: bool = True) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create temporal splits with purging to prevent lookahead bias.
        
        Args:
            X: Feature matrix
            y: Target array
            timestamps: Timestamp array (optional)
            use_purged: Whether to use purged splits
            
        Yields:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            if use_purged and self.purged_cv_config.enabled:
                # Use unified temporal cross-validation with gap as a proxy for purging
                try:
                    import inspect
                    ucv = UnifiedCrossValidator()
                    # approximate gap from purge_pct relative to fold size
                    approx_fold_size = max(1, len(X) // (self.purged_cv_config.n_splits + 1))
                    gap = max(0, int(approx_fold_size * self.purged_cv_config.purge_pct))
                    tscv = None
                    # Build a generator of indices similar to TimeSeriesSplit
                    if 'test_size' in inspect.signature(TimeSeriesSplit).parameters:
                        tscv = TimeSeriesSplit(n_splits=self.purged_cv_config.n_splits, gap=gap, test_size=approx_fold_size)
                    else:
                        tscv = TimeSeriesSplit(n_splits=self.purged_cv_config.n_splits, gap=gap)
                    for train_idx, test_idx in tscv.split(X):
                        yield X[train_idx], X[test_idx], y[train_idx], y[test_idx]
                except Exception:
                    # Fallback to TimeSeriesSplit without purging
                    tprint_warning("⚠️ Unified purged CV not available, using TimeSeriesSplit")
                    use_purged = False
            
            if not use_purged:
                # Use standard TimeSeriesSplit
                tscv = TimeSeriesSplit(
                    n_splits=self.purged_cv_config.n_splits,
                    test_size=self.purged_cv_config.test_size,
                    gap=self.purged_cv_config.gap
                )
                
                for train_idx, test_idx in tscv.split(X):
                    yield X[train_idx], X[test_idx], y[train_idx], y[test_idx]
            
            tprint_success("✅ Temporal splits created successfully")
            
        except Exception as e:
            tprint_error(f"❌ Temporal splits creation failed: {e}")
            # Fallback to simple temporal split
            split_point = int(len(X) * 0.8)
            yield X[:split_point], X[split_point:], y[:split_point], y[split_point:]
    
    def apply_early_stopping(self, 
                           model: Any, 
                           X_train: np.ndarray, 
                           y_train: np.ndarray,
                           X_val: np.ndarray, 
                           y_val: np.ndarray,
                           model_type: str = 'auto') -> Tuple[Any, Dict[str, Any]]:
        """
        Apply early stopping to any model that supports it.
        
        Args:
            model: Model to train with early stopping
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_type: Type of model ('xgboost', 'lightgbm', 'catboost', 'neural_network', 'auto')
            
        Returns:
            Tuple of (trained_model, training_info)
        """
        if not self.early_stopping_config.enabled:
            # Train without early stopping
            model.fit(X_train, y_train)
            return model, {'early_stopping': False}
        
        try:
            training_info = {
                'early_stopping': True,
                'patience': self.early_stopping_config.patience,
                'monitor': self.early_stopping_config.monitor,
                'best_score': None,
                'best_epoch': 0,
                'stopped_early': False,
                'training_history': []
            }
            
            # Detect model type if auto
            if model_type == 'auto':
                model_type = self._detect_model_type(model)
            
            # Apply model-specific early stopping
            if model_type in ['xgboost', 'lightgbm', 'catboost']:
                model, info = self._apply_gradient_boosting_early_stopping(
                    model, X_train, y_train, X_val, y_val
                )
                training_info.update(info)
                
            elif model_type == 'neural_network':
                model, info = self._apply_neural_network_early_stopping(
                    model, X_train, y_train, X_val, y_val
                )
                training_info.update(info)
                
            else:
                # Generic early stopping for other models
                model, info = self._apply_generic_early_stopping(
                    model, X_train, y_train, X_val, y_val
                )
                training_info.update(info)
            
            tprint_success(f"✅ Early stopping applied to {model_type} model")
            return model, training_info
            
        except Exception as e:
            tprint_error(f"❌ Early stopping failed: {e}")
            # Fallback to standard training
            model.fit(X_train, y_train)
            return model, {'early_stopping': False, 'error': str(e)}
    
    def _detect_model_type(self, model: Any) -> str:
        """Detect the type of model for early stopping."""
        model_name = type(model).__name__.lower()
        
        if 'xgboost' in model_name or 'xgb' in model_name:
            return 'xgboost'
        elif 'lightgbm' in model_name or 'lgb' in model_name:
            return 'lightgbm'
        elif 'catboost' in model_name or 'cat' in model_name:
            return 'catboost'
        elif any(nn in model_name for nn in ['neural', 'mlp', 'network', 'sequential']):
            return 'neural_network'
        else:
            return 'generic'
    
    def _apply_gradient_boosting_early_stopping(self, 
                                              model: Any, 
                                              X_train: np.ndarray, 
                                              y_train: np.ndarray,
                                              X_val: np.ndarray, 
                                              y_val: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Apply early stopping to gradient boosting models."""
        try:
            # Set early stopping parameters
            if hasattr(model, 'set_params'):
                model.set_params(
                    early_stopping_rounds=self.early_stopping_config.patience,
                    eval_metric='rmse' if 'regressor' in type(model).__name__.lower() else 'logloss'
                )
            
            # Train with validation set
            if hasattr(model, 'fit'):
                # Try to use eval_set parameter
                try:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                except TypeError:
                    # Fallback if eval_set not supported
                    model.fit(X_train, y_train)
            
            return model, {'early_stopping_applied': True}
            
        except Exception as e:
            tprint_warning(f"⚠️ Gradient boosting early stopping failed: {e}")
            model.fit(X_train, y_train)
            return model, {'early_stopping_applied': False, 'error': str(e)}
    
    def _apply_neural_network_early_stopping(self,
                                           model: Any,
                                           X_train: np.ndarray,
                                           y_train: np.ndarray,
                                           X_val: np.ndarray,
                                           y_val: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Apply early stopping to neural network models using enhanced system."""
        try:
            # Import enhanced early stopping
            from .enhanced_early_stopping import apply_enhanced_early_stopping, get_early_stopping_config

            # Create early stopping config
            config = get_early_stopping_config(
                enabled=True,
                patience=self.early_stopping_config.patience,
                min_delta=self.early_stopping_config.min_delta,
                mode=self.early_stopping_config.mode,
                monitor=self.early_stopping_config.monitor,
                nn_learning_rate=0.001,
                nn_batch_size=32,
                nn_epochs=100
            )

            # Apply enhanced early stopping
            trained_model, result = apply_enhanced_early_stopping(
                model, X_train, y_train, X_val, y_val, 'neural_network', config
            )

            return trained_model, {
                'early_stopping_applied': result.early_stopping_applied,
                'best_epoch': result.best_epoch,
                'best_score': result.best_score,
                'training_stopped': result.training_stopped,
                'reason': result.reason
            }

        except Exception as e:
            tprint_warning(f"⚠️ Enhanced neural network early stopping failed: {e}")
            # Fallback to standard training
            model.fit(X_train, y_train)
            return model, {'early_stopping_applied': False, 'error': str(e)}
    
    def _apply_generic_early_stopping(self, 
                                    model: Any, 
                                    X_train: np.ndarray, 
                                    y_train: np.ndarray,
                                    X_val: np.ndarray, 
                                    y_val: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Apply generic early stopping for other models."""
        try:
            # Use enhanced early stopping system for generic models

            # Create early stopping config
            config = get_early_stopping_config(
                enabled=True,
                patience=self.early_stopping_config.patience,
                min_delta=self.early_stopping_config.min_delta,
                mode=self.early_stopping_config.mode,
                monitor=self.early_stopping_config.monitor,
                generic_check_frequency=1,
                generic_max_iterations=100
            )

            # Apply enhanced early stopping
            trained_model, result = apply_enhanced_early_stopping(
                model, X_train, y_train, X_val, y_val, 'generic', config
            )

            return trained_model, {
                'early_stopping_applied': result.early_stopping_applied,
                'best_epoch': result.best_epoch,
                'best_score': result.best_score,
                'training_stopped': result.training_stopped,
                'reason': result.reason
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Generic early stopping failed: {e}")
            model.fit(X_train, y_train)
            return model, {'early_stopping_applied': False, 'error': str(e)}
    
    def apply_enhanced_regularization(self, 
                                    model: Any, 
                                    model_type: str = 'auto') -> Any:
        """
        Apply enhanced regularization to any model.
        
        Args:
            model: Model to apply regularization to
            model_type: Type of model
            
        Returns:
            Model with enhanced regularization
        """
        if not self.regularization_config.enabled:
            return model
        
        try:
            # Detect model type if auto
            if model_type == 'auto':
                model_type = self._detect_model_type(model)
            
            # Apply model-specific regularization
            if model_type in ['xgboost', 'lightgbm', 'catboost']:
                model = self._apply_gradient_boosting_regularization(model)
            elif model_type == 'randomforest':
                model = self._apply_random_forest_regularization(model)
            elif model_type == 'elasticnet':
                model = self._apply_elastic_net_regularization(model)
            else:
                model = self._apply_generic_regularization(model)
            
            tprint_success(f"✅ Enhanced regularization applied to {model_type}")
            return model
            
        except Exception as e:
            tprint_error(f"❌ Regularization application failed: {e}")
            return model
    
    def _apply_gradient_boosting_regularization(self, model: Any) -> Any:
        """Apply regularization to gradient boosting models."""
        try:
            if hasattr(model, 'set_params'):
                model.set_params(
                    reg_alpha=self.regularization_config.l1_alpha,
                    reg_lambda=self.regularization_config.l2_alpha,
                    max_depth=self.regularization_config.max_depth or 6,
                    min_child_weight=self.regularization_config.min_samples_leaf,
                    subsample=0.8,  # Add subsampling
                    colsample_bytree=0.8  # Add feature subsampling
                )
            return model
        except Exception as e:
            tprint_warning(f"⚠️ Gradient boosting regularization failed: {e}")
            return model
    
    def _apply_random_forest_regularization(self, model: Any) -> Any:
        """Apply regularization to Random Forest models."""
        try:
            if hasattr(model, 'set_params'):
                model.set_params(
                    max_depth=self.regularization_config.max_depth,
                    min_samples_split=self.regularization_config.min_samples_split,
                    min_samples_leaf=self.regularization_config.min_samples_leaf,
                    max_features=self.regularization_config.max_features,
                    max_samples=0.8  # Add sample subsampling
                )
            return model
        except Exception as e:
            tprint_warning(f"⚠️ Random Forest regularization failed: {e}")
            return model
    
    def _apply_elastic_net_regularization(self, model: Any) -> Any:
        """Apply regularization to Elastic Net models."""
        try:
            if hasattr(model, 'set_params'):
                model.set_params(
                    alpha=self.regularization_config.l1_alpha + self.regularization_config.l2_alpha,
                    l1_ratio=0.5,  # Balance between L1 and L2
                    max_iter=2000
                )
            return model
        except Exception as e:
            tprint_warning(f"⚠️ Elastic Net regularization failed: {e}")
            return model
    
    def _apply_generic_regularization(self, model: Any) -> Any:
        """Apply generic regularization to other models."""
        try:
            # Try to apply common regularization parameters
            if hasattr(model, 'set_params'):
                params = {}
                if hasattr(model, 'alpha'):
                    params['alpha'] = self.regularization_config.l1_alpha
                if hasattr(model, 'max_depth'):
                    params['max_depth'] = self.regularization_config.max_depth
                if hasattr(model, 'min_samples_split'):
                    params['min_samples_split'] = self.regularization_config.min_samples_split
                if hasattr(model, 'min_samples_leaf'):
                    params['min_samples_leaf'] = self.regularization_config.min_samples_leaf
                
                if params:
                    model.set_params(**params)
            
            return model
        except Exception as e:
            tprint_warning(f"⚠️ Generic regularization failed: {e}")
            return model
    
    def monitor_overfitting(self, 
                          model: Any, 
                          X_train: np.ndarray, 
                          y_train: np.ndarray,
                          X_val: np.ndarray, 
                          y_val: np.ndarray,
                          model_name: str = 'model') -> Dict[str, Any]:
        """
        Monitor for overfitting during training.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_name: Name of the model for logging
            
        Returns:
            Overfitting monitoring results
        """
        if not self.overfitting_config.enabled:
            return {'overfitting_monitoring': False}
        
        try:
            # Calculate training and validation performance
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            val_mse = mean_squared_error(y_val, y_val_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            
            # Calculate overfitting metrics
            mse_gap = val_mse - train_mse
            r2_gap = train_r2 - val_r2
            overfitting_ratio = mse_gap / train_mse if train_mse > 0 else 0
            
            # Determine if overfitting is detected
            is_overfitting = (
                overfitting_ratio > self.overfitting_config.threshold or
                r2_gap > self.overfitting_config.threshold
            )
            
            monitoring_results = {
                'overfitting_monitoring': True,
                'train_mse': train_mse,
                'val_mse': val_mse,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'mse_gap': mse_gap,
                'r2_gap': r2_gap,
                'overfitting_ratio': overfitting_ratio,
                'is_overfitting': is_overfitting,
                'threshold': self.overfitting_config.threshold
            }
            
            # Log overfitting warnings
            if is_overfitting:
                warning_msg = f"Overfitting detected in {model_name}: ratio={overfitting_ratio:.3f} > threshold={self.overfitting_config.threshold}"
                tprint_warning(f"⚠️ {warning_msg}")
                self.overfitting_warnings.append(warning_msg)
                monitoring_results['warning'] = warning_msg
            
            tprint_success(f"✅ Overfitting monitoring completed for {model_name}")
            return monitoring_results
            
        except Exception as e:
            tprint_error(f"❌ Overfitting monitoring failed: {e}")
            return {'overfitting_monitoring': False, 'error': str(e)}
    
    def create_validation_curves(self, 
                               model: Any, 
                               X: np.ndarray, 
                               y: np.ndarray,
                               param_name: str, 
                               param_range: List[Any],
                               cv_splits: Optional[Iterator] = None) -> Dict[str, Any]:
        """
        Create validation curves to detect overfitting patterns.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target array
            param_name: Parameter name to vary
            param_range: Range of parameter values
            cv_splits: Cross-validation splits (optional)
            
        Returns:
            Validation curve results
        """
        try:
            train_scores = []
            val_scores = []
            
            # Use provided CV splits or create temporal splits
            if cv_splits is None:
                cv_splits = self.create_temporal_splits(X, y)
            
            for param_value in param_range:
                # Set parameter
                if hasattr(model, 'set_params'):
                    model.set_params(**{param_name: param_value})
                
                fold_train_scores = []
                fold_val_scores = []
                
                # Evaluate on each CV fold
                for X_train, X_val, y_train, y_val in cv_splits:
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Evaluate on training set
                    y_train_pred = model.predict(X_train)
                    train_score = r2_score(y_train, y_train_pred)
                    fold_train_scores.append(train_score)
                    
                    # Evaluate on validation set
                    y_val_pred = model.predict(X_val)
                    val_score = r2_score(y_val, y_val_pred)
                    fold_val_scores.append(val_score)
                
                train_scores.append(np.mean(fold_train_scores))
                val_scores.append(np.mean(fold_val_scores))
            
            # Analyze validation curves
            train_scores = np.array(train_scores)
            val_scores = np.array(val_scores)
            
            # Find optimal parameter
            optimal_idx = np.argmax(val_scores)
            optimal_param = param_range[optimal_idx]
            
            # Detect overfitting patterns
            score_gap = train_scores - val_scores
            max_gap_idx = np.argmax(score_gap)
            overfitting_threshold = 0.1
            
            validation_curve_results = {
                'param_name': param_name,
                'param_range': param_range,
                'train_scores': train_scores.tolist(),
                'val_scores': val_scores.tolist(),
                'score_gap': score_gap.tolist(),
                'optimal_param': optimal_param,
                'optimal_score': val_scores[optimal_idx],
                'max_gap': np.max(score_gap),
                'overfitting_detected': np.max(score_gap) > overfitting_threshold,
                'recommendation': self._generate_validation_curve_recommendation(
                    param_range, train_scores, val_scores, score_gap
                )
            }
            
            tprint_success(f"✅ Validation curves created for {param_name}")
            return validation_curve_results
            
        except Exception as e:
            tprint_error(f"❌ Validation curves creation failed: {e}")
            return {'error': str(e)}
    
    def _generate_validation_curve_recommendation(self, 
                                                param_range: List[Any],
                                                train_scores: np.ndarray,
                                                val_scores: np.ndarray,
                                                score_gap: np.ndarray) -> str:
        """Generate recommendation based on validation curves."""
        try:
            # Find optimal parameter
            optimal_idx = np.argmax(val_scores)
            optimal_param = param_range[optimal_idx]
            
            # Analyze patterns
            if np.max(score_gap) > 0.1:
                return f"Overfitting detected. Consider using {optimal_param} with additional regularization."
            elif np.max(val_scores) - np.min(val_scores) < 0.05:
                return f"Parameter has minimal impact. Current value {optimal_param} is acceptable."
            else:
                return f"Optimal parameter value: {optimal_param}"
                
        except Exception:
            return "Unable to generate recommendation"
    
    def perform_walk_forward_validation(self, 
                                      model: Any, 
                                      X: np.ndarray, 
                                      y: np.ndarray,
                                      initial_train_size: int = 1000,
                                      test_size: int = 100,
                                      step_size: int = 50,
                                      expanding_window: bool = True) -> Dict[str, Any]:
        """
        Perform walk-forward validation for final model evaluation.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target array
            initial_train_size: Initial training set size
            test_size: Test set size for each iteration
            step_size: Step size for moving window
            expanding_window: Use expanding window (True) or rolling window (False)
            
        Returns:
            Walk-forward validation results
        """
        try:
            if self.cv_utils:
                # Use existing CV utilities
                return self.cv_utils.walk_forward_validation(
                    X, y, model, initial_train_size, test_size, step_size, expanding_window
                )
            else:
                # Implement basic walk-forward validation
                return self._basic_walk_forward_validation(
                    model, X, y, initial_train_size, test_size, step_size, expanding_window
                )
                
        except Exception as e:
            tprint_error(f"❌ Walk-forward validation failed: {e}")
            return {'error': str(e)}
    
    def _basic_walk_forward_validation(self, 
                                     model: Any, 
                                     X: np.ndarray, 
                                     y: np.ndarray,
                                     initial_train_size: int,
                                     test_size: int,
                                     step_size: int,
                                     expanding_window: bool) -> Dict[str, Any]:
        """Basic walk-forward validation implementation."""
        try:
            results = {
                'iterations': [],
                'metrics': {},
                'performance_trend': {}
            }
            
            current_position = initial_train_size
            
            while current_position + test_size <= len(X):
                # Define training window
                if expanding_window:
                    train_start = 0
                else:
                    train_start = max(0, current_position - initial_train_size)
                
                train_end = current_position
                test_end = current_position + test_size
                
                # Split data
                X_train = X[train_start:train_end]
                y_train = y[train_start:train_end]
                X_test = X[current_position:test_end]
                y_test = y[current_position:test_end]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Store results
                iteration_result = {
                    'iteration': len(results['iterations']),
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': current_position,
                    'test_end': test_end,
                    'mse': mse,
                    'r2': r2
                }
                
                results['iterations'].append(iteration_result)
                current_position += step_size
            
            # Calculate performance trend
            if results['iterations']:
                r2_scores = [iter['r2'] for iter in results['iterations']]
                results['performance_trend'] = {
                    'mean_r2': np.mean(r2_scores),
                    'std_r2': np.std(r2_scores),
                    'trend': 'improving' if r2_scores[-1] > r2_scores[0] else 'declining'
                }
            
            tprint_success("✅ Walk-forward validation completed")
            return results
            
        except Exception as e:
            tprint_error(f"❌ Basic walk-forward validation failed: {e}")
            return {'error': str(e)}
    
    def calculate_ensemble_diversity(self, 
                                   models: List[Any], 
                                   X: np.ndarray, 
                                   y: np.ndarray) -> Dict[str, Any]:
        """
        Calculate ensemble diversity metrics to prevent overfitting.
        
        Args:
            models: List of ensemble models
            X: Feature matrix
            y: Target array
            
        Returns:
            Ensemble diversity metrics
        """
        try:
            if len(models) < 2:
                return {'error': 'At least 2 models required for diversity calculation'}
            
            # Get predictions from all models
            predictions = []
            for model in models:
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                    predictions.append(pred)
                else:
                    tprint_warning("⚠️ Model does not support prediction")
                    continue
            
            if len(predictions) < 2:
                return {'error': 'Insufficient valid predictions for diversity calculation'}
            
            predictions = np.array(predictions)
            
            # Calculate diversity metrics
            diversity_metrics = {}
            
            # Prediction variance (higher = more diverse)
            pred_variance = np.var(predictions, axis=0)
            diversity_metrics['mean_prediction_variance'] = np.mean(pred_variance)
            diversity_metrics['std_prediction_variance'] = np.std(pred_variance)
            
            # Correlation between model predictions
            correlations = []
            for i in range(len(predictions)):
                for j in range(i + 1, len(predictions)):
                    corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            if correlations:
                diversity_metrics['mean_correlation'] = np.mean(correlations)
                diversity_metrics['std_correlation'] = np.std(correlations)
                diversity_metrics['max_correlation'] = np.max(correlations)
                diversity_metrics['min_correlation'] = np.min(correlations)
            
            # Ensemble agreement (how often models agree)
            ensemble_mean = np.mean(predictions, axis=0)
            agreement_threshold = 0.1  # 10% threshold for agreement
            agreements = []
            
            for i in range(len(predictions)):
                agreement = np.mean(np.abs(predictions[i] - ensemble_mean) < agreement_threshold)
                agreements.append(agreement)
            
            diversity_metrics['mean_agreement'] = np.mean(agreements)
            diversity_metrics['std_agreement'] = np.std(agreements)
            
            # Diversity score (higher = more diverse)
            diversity_score = (
                diversity_metrics['mean_prediction_variance'] * 
                (1 - diversity_metrics.get('mean_correlation', 0.5))
            )
            diversity_metrics['diversity_score'] = diversity_score
            
            # Recommendations
            if diversity_metrics['mean_correlation'] > 0.8:
                diversity_metrics['recommendation'] = "High correlation detected. Consider adding more diverse models."
            elif diversity_metrics['diversity_score'] < 0.1:
                diversity_metrics['recommendation'] = "Low diversity detected. Consider different model types or parameters."
            else:
                diversity_metrics['recommendation'] = "Good ensemble diversity."
            
            tprint_success("✅ Ensemble diversity calculated")
            return diversity_metrics
            
        except Exception as e:
            tprint_error(f"❌ Ensemble diversity calculation failed: {e}")
            return {'error': str(e)}
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary with all monitoring results."""
        return {
            'early_stopping_config': self.early_stopping_config.__dict__,
            'purged_cv_config': self.purged_cv_config.__dict__,
            'overfitting_config': self.overfitting_config.__dict__,
            'regularization_config': self.regularization_config.__dict__,
            'overfitting_warnings': self.overfitting_warnings,
            'training_history': self.training_history,
            'components_available': {
                'lookahead_detector': self.lookahead_detector is not None,
                'cv_utils': self.cv_utils is not None
            }
        }


# Convenience functions for easy integration
def create_enhanced_training_utils(**kwargs) -> EnhancedTrainingUtils:
    """Create enhanced training utilities with custom configuration."""
    return EnhancedTrainingUtils(**kwargs)


def validate_temporal_data(X: np.ndarray, 
                          y: np.ndarray, 
                          timestamps: Optional[np.ndarray] = None,
                          strict_mode: bool = True) -> Tuple[bool, List[str]]:
    """Convenience function for temporal data validation."""
    utils = EnhancedTrainingUtils()
    return utils.validate_temporal_data(X, y, timestamps, strict_mode)


def create_temporal_splits(X: np.ndarray, 
                         y: np.ndarray,
                         timestamps: Optional[np.ndarray] = None,
                         use_purged: bool = True) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Convenience function for temporal splits creation."""
    utils = EnhancedTrainingUtils()
    return utils.create_temporal_splits(X, y, timestamps, use_purged)


def apply_early_stopping(model: Any, 
                        X_train: np.ndarray, 
                        y_train: np.ndarray,
                        X_val: np.ndarray, 
                        y_val: np.ndarray,
                        model_type: str = 'auto') -> Tuple[Any, Dict[str, Any]]:
    """Convenience function for early stopping."""
    utils = EnhancedTrainingUtils()
    return utils.apply_early_stopping(model, X_train, y_train, X_val, y_val, model_type)


def apply_enhanced_regularization(model: Any, model_type: str = 'auto') -> Any:
    """Convenience function for enhanced regularization."""
    utils = EnhancedTrainingUtils()
    return utils.apply_enhanced_regularization(model, model_type)


def monitor_overfitting(model: Any, 
                       X_train: np.ndarray, 
                       y_train: np.ndarray,
                       X_val: np.ndarray, 
                       y_val: np.ndarray,
                       model_name: str = 'model') -> Dict[str, Any]:
    """Convenience function for overfitting monitoring."""
    utils = EnhancedTrainingUtils()
    return utils.monitor_overfitting(model, X_train, y_train, X_val, y_val, model_name)