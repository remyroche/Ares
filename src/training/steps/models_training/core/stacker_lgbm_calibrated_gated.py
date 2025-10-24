"""
Stacked LightGBM with Calibration and Gating

This module provides an advanced ensemble method that combines:
1. Stacking with LightGBM meta-learner
2. Probability calibration
3. Gating mechanism for dynamic model selection
4. Financial-aware performance optimization

Key Features:
1. Out-of-fold predictions for leakage prevention
2. Isotonic/Platt calibration
3. Volatility-aware gating
4. Regime-based model selection
5. Confidence threshold adaptation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, KFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss
import logging

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, 
    tprint_debug, tprint_data_format, tprint_data_preview, LogLevel
)

from .error_handling import (
    handle_errors, validate_config, validate_data, safe_import,
    MLModelTrainerError, ConfigurationError, DataValidationError, 
    ModelTrainingError, PredictionError, ResourceError
)
from .weighted_loss_framework import (
    WeightedLossManager, WeightedLossConfig, WeightingStrategy
)

# Use safe import with fallback
lgb = safe_import('lightgbm', 'lightgbm')
LIGHTGBM_AVAILABLE = lgb is not None

logger = logging.getLogger(__name__)

class GatingMechanism:
    """Gating mechanism for dynamic model selection based on market conditions."""
    
    def __init__(self, 
                 volatility_window: int = 20,
                 volatility_threshold: float = 0.02,
                 confidence_threshold: float = 0.7,
                 min_threshold: float = 0.5,
                 max_threshold: float = 0.9,
                 regime_aware: bool = True):
        self.volatility_window = volatility_window
        self.volatility_threshold = volatility_threshold
        self.confidence_threshold = confidence_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.regime_aware = regime_aware
        
        # Gating state
        self.volatility_history = []
        self.regime_history = []
        self.gate_weights = None
    
    def calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate rolling volatility."""
        if len(returns) < self.volatility_window:
            return 0.0
        return np.std(returns[-self.volatility_window:])
    
    def update_gate_weights(self, 
                          base_predictions: Dict[str, np.ndarray],
                          volatility: float,
                          regime: Optional[str] = None) -> Dict[str, float]:
        """Update gating weights based on current market conditions."""
        
        # Initialize equal weights
        model_names = list(base_predictions.keys())
        weights = {name: 1.0 / len(model_names) for name in model_names}
        
        # Volatility-based adjustment
        if volatility > self.volatility_threshold:
            # High volatility: prefer more conservative models
            for name in model_names:
                if 'conservative' in name.lower() or 'stable' in name.lower():
                    weights[name] *= 1.2
                elif 'aggressive' in name.lower() or 'momentum' in name.lower():
                    weights[name] *= 0.8
        else:
            # Low volatility: prefer more aggressive models
            for name in model_names:
                if 'aggressive' in name.lower() or 'momentum' in name.lower():
                    weights[name] *= 1.2
                elif 'conservative' in name.lower() or 'stable' in name.lower():
                    weights[name] *= 0.8
        
        # Regime-based adjustment
        if self.regime_aware and regime is not None:
            regime_weights = {
                'low_volatility': {'conservative': 1.3, 'stable': 1.1, 'aggressive': 0.7},
                'medium_volatility': {'conservative': 1.0, 'stable': 1.0, 'aggressive': 1.0},
                'high_volatility': {'conservative': 0.7, 'stable': 0.9, 'aggressive': 1.3}
            }
            
            if regime in regime_weights:
                for name in model_names:
                    for pattern, multiplier in regime_weights[regime].items():
                        if pattern in name.lower():
                            weights[name] *= multiplier
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        
        self.gate_weights = weights
        return weights
    
    def get_gate_weights(self) -> Dict[str, float]:
        """Get current gating weights."""
        return self.gate_weights or {}

class StackerLGBMCalibratedGated(BaseEstimator, ClassifierMixin):
    """Stacked LightGBM with calibration and gating."""
    
    def __init__(self,
                 # Base models configuration
                 base_models: List[Dict[str, Any]] = None,
                 
                 # Meta-learner configuration
                 meta_learner_type: str = "LIGHTGBM",
                 meta_learner_params: Dict[str, Any] = None,
                 
                 # Stacking configuration
                 cv_folds: int = 5,
                 use_features_in_secondary: bool = True,
                 use_proba_as_level1: bool = True,
                 n_jobs: int = -1,
                 
                 # Calibration configuration
                 calibration_method: str = "isotonic",  # "isotonic" or "platt"
                 calibration_cv_folds: int = 3,
                 calibration_test_size: float = 0.2,
                 
                 # Gating configuration
                 enable_gating: bool = True,
                 volatility_window: int = 20,
                 volatility_threshold: float = 0.02,
                 confidence_threshold: float = 0.7,
                 min_threshold: float = 0.5,
                 max_threshold: float = 0.9,
                 regime_aware: bool = True,
                 
                 # Performance optimization
                 enable_parallel_processing: bool = True,
                 memory_efficient: bool = True,
                 validation_split: float = 0.2,
                 
                 # Weighted loss configuration
                 enable_weighted_loss: bool = True,
                 weighted_loss_config: Optional[Dict[str, Any]] = None,
                 
                 # Random state
                 random_state: int = 42):
        """Initialize the stacked ensemble."""
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for StackerLGBMCalibratedGated. Install with: pip install lightgbm")
        
        # Base models
        self.base_models = base_models or []
        
        # Meta-learner
        self.meta_learner_type = meta_learner_type
        self.meta_learner_params = meta_learner_params or {}
        
        # Stacking
        self.cv_folds = cv_folds
        self.use_features_in_secondary = use_features_in_secondary
        self.use_proba_as_level1 = use_proba_as_level1
        self.n_jobs = n_jobs
        
        # Calibration
        self.calibration_method = calibration_method
        self.calibration_cv_folds = calibration_cv_folds
        self.calibration_test_size = calibration_test_size
        
        # Gating
        self.enable_gating = enable_gating
        self.gating_mechanism = GatingMechanism(
            volatility_window=volatility_window,
            volatility_threshold=volatility_threshold,
            confidence_threshold=confidence_threshold,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            regime_aware=regime_aware
        )
        
        # Performance
        self.enable_parallel_processing = enable_parallel_processing
        self.memory_efficient = memory_efficient
        self.validation_split = validation_split
        self.random_state = random_state
        
        # Weighted loss configuration
        self.enable_weighted_loss = enable_weighted_loss
        self.weighted_loss_config = weighted_loss_config or {}
        self.weighted_loss_manager = None
        
        # Model state
        self.base_models_fitted = []
        self.meta_model = None
        self.calibrator = None
        self.is_fitted = False
        self.n_features_in_ = None
        self.classes_ = None
        self.feature_names_ = None
        
        # OOF predictions storage
        self.oof_predictions = {}
        self.oof_probabilities = {}
        self.fold_assignments = None
    
    def _create_base_model(self, model_config: Dict[str, Any]) -> BaseEstimator:
        """Create a base model from configuration."""
        model_type = model_config.get('type', 'LIGHTGBM').upper()
        params = model_config.get('parameters', {})
        
        if model_type == 'LIGHTGBM':
            return lgb.LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                boosting_type='gbdt',
                random_state=self.random_state,
                verbose=-1,
                n_jobs=1,
                **params
            )
        elif model_type == 'CATBOOST':
            try:
                from catboost import CatBoostClassifier
                return CatBoostClassifier(
                    random_seed=self.random_state,
                    verbose=False,
                    thread_count=1,
                    **params
                )
            except ImportError:
                logger.warning("CatBoost not available, falling back to LightGBM")
                return lgb.LGBMClassifier(
                    objective='binary',
                    metric='binary_logloss',
                    boosting_type='gbdt',
                    random_state=self.random_state,
                    verbose=-1,
                    n_jobs=1,
                    **params
                )
        elif model_type == 'XGBOOST':
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    random_state=self.random_state,
                    verbosity=0,
                    n_jobs=1,
                    **params
                )
            except ImportError:
                logger.warning("XGBoost not available, falling back to LightGBM")
                return lgb.LGBMClassifier(
                    objective='binary',
                    metric='binary_logloss',
                    boosting_type='gbdt',
                    random_state=self.random_state,
                    verbose=-1,
                    n_jobs=1,
                    **params
                )
        else:
            # Fallback to LightGBM
            logger.warning(f"Unknown model type {model_type}, falling back to LightGBM")
            return lgb.LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                boosting_type='gbdt',
                random_state=self.random_state,
                verbose=-1,
                n_jobs=1,
                **params
            )
    
    def _create_meta_model(self) -> BaseEstimator:
        """Create meta-learner model."""
        if self.meta_learner_type.upper() == 'LIGHTGBM':
            return lgb.LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                boosting_type='gbdt',
                random_state=self.random_state,
                verbose=-1,
                n_jobs=1,
                **self.meta_learner_params
            )
        else:
            # Fallback to LightGBM
            return lgb.LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                boosting_type='gbdt',
                random_state=self.random_state,
                verbose=-1,
                n_jobs=1,
                **self.meta_learner_params
            )
    
    def _generate_oof_predictions(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
        """Generate out-of-fold predictions for stacking."""
        logger.info(f"Generating OOF predictions with {self.cv_folds} folds")
        
        # Use TimeSeriesSplit for time series data
        cv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        n_samples = X.shape[0]
        oof_predictions = {}
        oof_probabilities = {}
        fold_assignments = np.full(n_samples, -1, dtype=int)
        
        # Initialize OOF containers
        for i, model_config in enumerate(self.base_models):
            model_name = model_config.get('name', f'model_{i}')
            oof_predictions[model_name] = np.zeros(n_samples)
            oof_probabilities[model_name] = np.zeros((n_samples, 2))  # Binary classification
        
        # Generate OOF predictions
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
            fold_assignments[val_idx] = fold_idx
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train each base model
            for i, model_config in enumerate(self.base_models):
                model_name = model_config.get('name', f'model_{i}')
                base_model = self._create_base_model(model_config)
                
                # Train model
                if hasattr(base_model, 'fit'):
                    base_model.fit(X_train, y_train)
                    
                    # Generate predictions
                    val_pred = base_model.predict(X_val)
                    oof_predictions[model_name][val_idx] = val_pred
                    
                    # Generate probabilities if available
                    if hasattr(base_model, 'predict_proba'):
                        val_proba = base_model.predict_proba(X_val)
                        oof_probabilities[model_name][val_idx] = val_proba
                    else:
                        # Convert predictions to probabilities
                        oof_probabilities[model_name][val_idx, 1] = val_pred
                        oof_probabilities[model_name][val_idx, 0] = 1 - val_pred
        
        logger.info("OOF predictions generated successfully")
        return oof_predictions, oof_probabilities, fold_assignments
    
    def _prepare_meta_features(self, X: np.ndarray, oof_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Prepare meta-features for the meta-learner."""
        # Use probabilities as level-1 features
        if self.use_proba_as_level1:
            meta_features = np.column_stack([
                oof_predictions[name][:, 1] for name in oof_predictions.keys()
            ])
        else:
            meta_features = np.column_stack([
                oof_predictions[name] for name in oof_predictions.keys()
            ])
        
        # Add original features if specified
        if self.use_features_in_secondary:
            meta_features = np.hstack([meta_features, X])
        
        return meta_features
    
    def _apply_gating(self, predictions: Dict[str, np.ndarray], 
                     volatility: Optional[float] = None,
                     regime: Optional[str] = None) -> np.ndarray:
        """Apply gating mechanism to combine base model predictions."""
        if not self.enable_gating:
            # Simple average
            return np.mean(list(predictions.values()), axis=0)
        
        # Update gating weights
        gate_weights = self.gating_mechanism.update_gate_weights(
            predictions, volatility, regime
        )
        
        # Apply weighted combination
        weighted_predictions = np.zeros_like(list(predictions.values())[0])
        for model_name, pred in predictions.items():
            weight = gate_weights.get(model_name, 1.0 / len(predictions))
            weighted_predictions += weight * pred
        
        return weighted_predictions
    
    def _calibrate_predictions(self, y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
        """Calibrate probability predictions."""
        if self.calibration_method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(y_prob, y_true)
            return calibrator.predict(y_prob)
        elif self.calibration_method == "platt":
            calibrator = LogisticRegression()
            calibrator.fit(y_prob.reshape(-1, 1), y_true)
            return calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        else:
            logger.warning(f"Unknown calibration method {self.calibration_method}, using raw probabilities")
            return y_prob
    
    @handle_errors(error_type=ModelTrainingError, reraise=True)
    def fit(self, X: np.ndarray, y: np.ndarray, 
            sample_weight: Optional[np.ndarray] = None,
            volatility: Optional[np.ndarray] = None,
            regime: Optional[np.ndarray] = None) -> 'StackerLGBMCalibratedGated':
        """Fit the stacked ensemble."""
        logger.info("Training StackerLGBMCalibratedGated ensemble")
        
        # Validate inputs
        validate_data(X, y, min_samples=10, min_features=1)
        if len(self.base_models) == 0:
            raise ConfigurationError("No base models specified")
        
        # Store feature info
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        
        # Initialize weighted loss manager if enabled
        if self.enable_weighted_loss and self.weighted_loss_manager is not None:
            tprint_info("Initializing weighted loss manager...")
            self.weighted_loss_manager.fit(X, y)
        
        # Generate OOF predictions
        oof_predictions, oof_probabilities, fold_assignments = self._generate_oof_predictions(X, y)
        
        # Store OOF predictions
        self.oof_predictions = oof_predictions
        self.oof_probabilities = oof_probabilities
        self.fold_assignments = fold_assignments
        
        # Prepare meta-features
        meta_features = self._prepare_meta_features(X, oof_probabilities)
        
        # Train meta-learner
        logger.info("Training meta-learner")
        self.meta_model = self._create_meta_model()
        
        # Get sample weights if weighted loss is enabled
        meta_sample_weight = None
        if self.enable_weighted_loss and self.weighted_loss_manager is not None:
            tprint_info("Calculating sample weights for meta-learner...")
            meta_sample_weight = self.weighted_loss_manager.get_sample_weights(meta_features, y)
            tprint_debug(f"Meta-learner sample weight statistics - Mean: {np.mean(meta_sample_weight):.3f}, Std: {np.std(meta_sample_weight):.3f}")
        
        # Use last fold for validation if needed
        if self.validation_split > 0:
            last_fold = fold_assignments == fold_assignments.max()
            train_mask = ~last_fold
            
            X_meta_train = meta_features[train_mask]
            y_train = y[train_mask]
            X_meta_val = meta_features[last_fold]
            y_val = y[last_fold]
            
            # Get sample weights for training and validation
            train_sample_weight = meta_sample_weight[train_mask] if meta_sample_weight is not None else None
            val_sample_weight = meta_sample_weight[last_fold] if meta_sample_weight is not None else None
            
            self.meta_model.fit(
                X_meta_train, y_train,
                eval_set=[(X_meta_val, y_val)],
                sample_weight=train_sample_weight,
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
        else:
            self.meta_model.fit(meta_features, y, sample_weight=meta_sample_weight)
        
        # Train base models on full data
        logger.info("Training base models on full data")
        self.base_models_fitted = []
        for model_config in self.base_models:
            base_model = self._create_base_model(model_config)
            base_model.fit(X, y)
            self.base_models_fitted.append(base_model)
        
        # Calibrate meta-model predictions
        logger.info("Calibrating meta-model predictions")
        meta_predictions = self.meta_model.predict_proba(meta_features)[:, 1]
        self.calibrator = self._calibrate_predictions(y, meta_predictions)
        
        self.is_fitted = True
        logger.info("StackerLGBMCalibratedGated training completed")
        
        return self
    
    @handle_errors(error_type=PredictionError, reraise=True)
    def predict(self, X: np.ndarray, 
                volatility: Optional[np.ndarray] = None,
                regime: Optional[np.ndarray] = None) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise PredictionError("Model must be fitted before prediction")
        
        validate_data(X, min_samples=1, min_features=self.n_features_in_)
        
        # Generate base model predictions
        base_predictions = {}
        base_probabilities = {}
        
        for i, (model_config, base_model) in enumerate(zip(self.base_models, self.base_models_fitted)):
            model_name = model_config.get('name', f'model_{i}')
            
            # Get predictions
            pred = base_model.predict(X)
            base_predictions[model_name] = pred
            
            # Get probabilities if available
            if hasattr(base_model, 'predict_proba'):
                proba = base_model.predict_proba(X)
                base_probabilities[model_name] = proba[:, 1]
            else:
                base_probabilities[model_name] = pred
        
        # Apply gating if enabled
        if self.enable_gating:
            current_volatility = None
            current_regime = None
            
            if volatility is not None and len(volatility) > 0:
                current_volatility = np.mean(volatility[-self.gating_mechanism.volatility_window:])
            
            if regime is not None and len(regime) > 0:
                current_regime = regime[-1] if len(regime) > 0 else None
            
            # Apply gating to base predictions
            gated_predictions = self._apply_gating(
                base_probabilities, current_volatility, current_regime
            )
        else:
            gated_predictions = np.mean(list(base_probabilities.values()), axis=0)
        
        # Prepare meta-features
        meta_features = self._prepare_meta_features(X, {name: pred.reshape(-1, 1) for name, pred in base_probabilities.items()})
        
        # Get meta-model predictions
        meta_predictions = self.meta_model.predict_proba(meta_features)[:, 1]
        
        # Apply calibration
        calibrated_predictions = self._calibrate_predictions(
            np.ones_like(meta_predictions), meta_predictions
        )
        
        # Combine gated base predictions with calibrated meta predictions
        final_predictions = 0.7 * calibrated_predictions + 0.3 * gated_predictions
        
        # Convert to binary predictions
        return (final_predictions > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray,
                     volatility: Optional[np.ndarray] = None,
                     regime: Optional[np.ndarray] = None) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Generate base model predictions
        base_probabilities = {}
        
        for i, (model_config, base_model) in enumerate(zip(self.base_models, self.base_models_fitted)):
            model_name = model_config.get('name', f'model_{i}')
            
            # Get probabilities if available
            if hasattr(base_model, 'predict_proba'):
                proba = base_model.predict_proba(X)
                base_probabilities[model_name] = proba[:, 1]
            else:
                pred = base_model.predict(X)
                base_probabilities[model_name] = pred
        
        # Apply gating if enabled
        if self.enable_gating:
            current_volatility = None
            current_regime = None
            
            if volatility is not None and len(volatility) > 0:
                current_volatility = np.mean(volatility[-self.gating_mechanism.volatility_window:])
            
            if regime is not None and len(regime) > 0:
                current_regime = regime[-1] if len(regime) > 0 else None
            
            # Apply gating to base predictions
            gated_predictions = self._apply_gating(
                base_probabilities, current_volatility, current_regime
            )
        else:
            gated_predictions = np.mean(list(base_probabilities.values()), axis=0)
        
        # Prepare meta-features
        meta_features = self._prepare_meta_features(X, {name: pred.reshape(-1, 1) for name, pred in base_probabilities.items()})
        
        # Get meta-model predictions
        meta_predictions = self.meta_model.predict_proba(meta_features)[:, 1]
        
        # Apply calibration
        calibrated_predictions = self._calibrate_predictions(
            np.ones_like(meta_predictions), meta_predictions
        )
        
        # Combine gated base predictions with calibrated meta predictions
        final_predictions = 0.7 * calibrated_predictions + 0.3 * gated_predictions
        
        # Return as probability matrix
        return np.column_stack([1 - final_predictions, final_predictions])
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from meta-model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.meta_model, 'feature_importances_'):
            return self.meta_model.feature_importances_
        else:
            logger.warning("Meta-model does not support feature importance")
            return np.array([])
    
    def get_gating_weights(self) -> Dict[str, float]:
        """Get current gating weights."""
        return self.gating_mechanism.get_gate_weights()
    
    def get_oof_predictions(self) -> Dict[str, np.ndarray]:
        """Get out-of-fold predictions for analysis."""
        return self.oof_predictions.copy()

# Factory function
def create_stacker_lgbm_calibrated_gated(**kwargs) -> StackerLGBMCalibratedGated:
    """Create StackerLGBMCalibratedGated with default parameters."""
    return StackerLGBMCalibratedGated(**kwargs)