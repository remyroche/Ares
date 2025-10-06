"""
Stacker LGBM Calibrated Meta-Learner

Per-regime meta-learner that combines predictions from A1-A4 models
with calibrated uncertainty estimation and regime-specific calibration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import joblib
import os

logger = logging.getLogger(__name__)


@dataclass
class StackerConfig:
    """Configuration for the stacker meta-learner."""
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 500
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    min_child_samples: int = 20
    objective: str = 'binary'
    metric: str = 'binary_logloss'
    boosting_type: str = 'gbdt'
    num_leaves: int = 31
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    verbose: int = -1
    random_state: int = 42


@dataclass
class CalibrationConfig:
    """Configuration for meta-learner calibration."""
    method: str = 'isotonic'  # 'isotonic' or 'sigmoid'
    cv_folds: int = 5
    enable_venn_abers: bool = True
    confidence_levels: List[float] = None
    per_regime_calibration: bool = True

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.5, 0.6, 0.7, 0.8, 0.9]


class VennAbersCalibration:
    """Venn-Abers calibration for uncertainty estimation."""
    
    def __init__(self, confidence_levels: List[float] = None):
        self.confidence_levels = confidence_levels or [0.5, 0.6, 0.7, 0.8, 0.9]
        self.calibrators = {}
        self.is_fitted = False
    
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> 'VennAbersCalibration':
        """Fit Venn-Abers calibrators."""
        for level in self.confidence_levels:
            # Create binary targets for this confidence level
            y_binary = (y_prob >= level).astype(int)
            
            if len(np.unique(y_binary)) > 1:  # Ensure we have both classes
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(y_prob, y_binary)
                self.calibrators[level] = calibrator
        
        self.is_fitted = True
        return self
    
    def predict_confidence(self, y_prob: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict confidence intervals."""
        if not self.is_fitted:
            raise ValueError("Venn-Abers calibrators must be fitted first")
        
        results = {}
        for level, calibrator in self.calibrators.items():
            calibrated_probs = calibrator.predict(y_prob)
            results[f'confidence_{level}'] = calibrated_probs
        
        return results


class RegimeSpecificCalibrator:
    """Regime-specific calibration for meta-learner."""
    
    def __init__(self, calibration_config: CalibrationConfig):
        self.calibration_config = calibration_config
        self.regime_calibrators = {}
        self.global_calibrator = None
        self.is_fitted = False
    
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray, regimes: np.ndarray) -> 'RegimeSpecificCalibrator':
        """Fit regime-specific calibrators."""
        unique_regimes = np.unique(regimes)
        
        # Fit global calibrator
        if self.calibration_config.method in ['isotonic', 'sigmoid']:
            self.global_calibrator = CalibratedClassifierCV(
                method=self.calibration_config.method,
                cv=self.calibration_config.cv_folds
            )
            # Create dummy model for calibration
            from sklearn.linear_model import LogisticRegression
            dummy_model = LogisticRegression()
            dummy_model.fit(y_prob.reshape(-1, 1), y_true)
            self.global_calibrator.fit(dummy_model, y_true)
        
        # Fit regime-specific calibrators
        if self.calibration_config.per_regime_calibration:
            for regime in unique_regimes:
                regime_mask = regimes == regime
                if np.sum(regime_mask) > 50:  # Minimum samples for regime calibration
                    regime_y_true = y_true[regime_mask]
                    regime_y_prob = y_prob[regime_mask]
                    
                    if len(np.unique(regime_y_true)) > 1:  # Ensure both classes present
                        if self.calibration_config.method in ['isotonic', 'sigmoid']:
                            regime_calibrator = CalibratedClassifierCV(
                                method=self.calibration_config.method,
                                cv=min(self.calibration_config.cv_folds, len(regime_y_true) // 10)
                            )
                            # Create dummy model for calibration
                            dummy_model = LogisticRegression()
                            dummy_model.fit(regime_y_prob.reshape(-1, 1), regime_y_true)
                            regime_calibrator.fit(dummy_model, regime_y_true)
                            self.regime_calibrators[regime] = regime_calibrator
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, y_prob: np.ndarray, regimes: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities."""
        if not self.is_fitted:
            raise ValueError("Regime-specific calibrators must be fitted first")
        
        calibrated_probs = np.zeros_like(y_prob)
        
        for i, (prob, regime) in enumerate(zip(y_prob, regimes)):
            if regime in self.regime_calibrators:
                # Use regime-specific calibrator
                calibrated_probs[i] = self.regime_calibrators[regime].predict_proba([prob])[0, 1]
            elif self.global_calibrator is not None:
                # Use global calibrator
                calibrated_probs[i] = self.global_calibrator.predict_proba([prob])[0, 1]
            else:
                # No calibration
                calibrated_probs[i] = prob
        
        return calibrated_probs


class StackerLGBMCalibrated:
    """Stacker LGBM Calibrated Meta-Learner for per-regime ensemble."""
    
    def __init__(self, 
                 stacker_config: Optional[StackerConfig] = None,
                 calibration_config: Optional[CalibrationConfig] = None):
        self.stacker_config = stacker_config or StackerConfig()
        self.calibration_config = calibration_config or CalibrationConfig()
        
        # Model components
        self.stacker_model = None
        self.regime_calibrator = None
        self.venn_abers = None
        self.feature_names = None
        self.is_fitted = False
        
        logger.info("Initialized Stacker LGBM Calibrated Meta-Learner")
    
    def _prepare_stacking_features(self, model_predictions: Dict[str, np.ndarray], 
                                 uncertainty_estimates: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """Prepare features for stacking from model predictions and uncertainty estimates."""
        features = []
        
        # Add base model predictions
        for model_name, predictions in model_predictions.items():
            features.append(predictions)
        
        # Add uncertainty features
        for model_name, uncertainty in uncertainty_estimates.items():
            if 'margin_stats' in uncertainty:
                margin_stats = uncertainty['margin_stats']
                features.append(np.array([margin_stats['mean_probability']]))
                features.append(np.array([margin_stats['std_probability']]))
                features.append(np.array([margin_stats['confidence_range']]))
            
            if 'confidence_intervals' in uncertainty:
                for level, conf_values in uncertainty['confidence_intervals'].items():
                    features.append(conf_values)
        
        # Stack features
        if features:
            stacked_features = np.column_stack(features)
        else:
            # Fallback: use only base predictions
            stacked_features = np.column_stack(list(model_predictions.values()))
        
        return stacked_features
    
    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights for imbalanced data."""
        unique_classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        class_weights = {}
        for class_label, count in zip(unique_classes, counts):
            class_weights[class_label] = total_samples / (len(unique_classes) * count)
        
        return class_weights
    
    def fit(self, model_predictions: Dict[str, np.ndarray], 
            uncertainty_estimates: Dict[str, Dict[str, Any]],
            y: np.ndarray, 
            regimes: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> 'StackerLGBMCalibrated':
        """Fit the stacker meta-learner."""
        logger.info("Fitting Stacker LGBM Calibrated Meta-Learner...")
        
        # Convert to numpy arrays
        y = np.asarray(y)
        regimes = np.asarray(regimes)
        
        # Ensure binary classification
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(f"Binary classification requires exactly 2 classes, got {len(unique_classes)}")
        
        # Prepare stacking features
        X_stacked = self._prepare_stacking_features(model_predictions, uncertainty_estimates)
        logger.info(f"Stacking features shape: {X_stacked.shape}")
        
        # Compute class weights
        class_weights = self._compute_class_weights(y)
        logger.info(f"Class weights: {class_weights}")
        
        # Create LightGBM stacker model
        self.stacker_model = lgb.LGBMClassifier(
            max_depth=self.stacker_config.max_depth,
            learning_rate=self.stacker_config.learning_rate,
            n_estimators=self.stacker_config.n_estimators,
            subsample=self.stacker_config.subsample,
            colsample_bytree=self.stacker_config.colsample_bytree,
            reg_alpha=self.stacker_config.reg_alpha,
            reg_lambda=self.stacker_config.reg_lambda,
            min_child_samples=self.stacker_config.min_child_samples,
            objective=self.stacker_config.objective,
            metric=self.stacker_config.metric,
            boosting_type=self.stacker_config.boosting_type,
            num_leaves=self.stacker_config.num_leaves,
            feature_fraction=self.stacker_config.feature_fraction,
            bagging_fraction=self.stacker_config.bagging_fraction,
            bagging_freq=self.stacker_config.bagging_freq,
            class_weight=class_weights,
            verbose=self.stacker_config.verbose,
            random_state=self.stacker_config.random_state
        )
        
        # Fit stacker model
        if sample_weight is not None:
            self.stacker_model.fit(X_stacked, y, sample_weight=sample_weight)
        else:
            self.stacker_model.fit(X_stacked, y)
        
        # Get predictions for calibration
        y_prob = self.stacker_model.predict_proba(X_stacked)[:, 1]
        
        # Fit regime-specific calibration
        self.regime_calibrator = RegimeSpecificCalibrator(self.calibration_config)
        self.regime_calibrator.fit(y, y_prob, regimes)
        
        # Fit Venn-Abers calibration
        if self.calibration_config.enable_venn_abers:
            self.venn_abers = VennAbersCalibration(self.calibration_config.confidence_levels)
            self.venn_abers.fit(y, y_prob)
        
        self.is_fitted = True
        logger.info("✅ Stacker LGBM Calibrated Meta-Learner fitted successfully")
        return self
    
    def predict_proba(self, model_predictions: Dict[str, np.ndarray], 
                     uncertainty_estimates: Dict[str, Dict[str, Any]],
                     regimes: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before prediction")
        
        # Prepare stacking features
        X_stacked = self._prepare_stacking_features(model_predictions, uncertainty_estimates)
        
        # Get base predictions
        y_prob = self.stacker_model.predict_proba(X_stacked)[:, 1]
        
        # Apply regime-specific calibration
        if self.regime_calibrator is not None:
            y_prob = self.regime_calibrator.predict_proba(y_prob, regimes)
        
        return y_prob
    
    def predict_uncertainty(self, model_predictions: Dict[str, np.ndarray], 
                           uncertainty_estimates: Dict[str, Dict[str, Any]],
                           regimes: np.ndarray) -> Dict[str, Any]:
        """Predict uncertainty estimates."""
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before prediction")
        
        # Get base predictions
        y_prob = self.predict_proba(model_predictions, uncertainty_estimates, regimes)
        
        # Get Venn-Abers confidence intervals
        uncertainty_results = {
            'probability': y_prob,
            'confidence_intervals': {}
        }
        
        if self.venn_abers is not None:
            confidence_intervals = self.venn_abers.predict_confidence(y_prob)
            uncertainty_results['confidence_intervals'] = confidence_intervals
        
        # Add margin statistics
        uncertainty_results['margin_stats'] = {
            'mean_probability': np.mean(y_prob),
            'std_probability': np.std(y_prob),
            'min_probability': np.min(y_prob),
            'max_probability': np.max(y_prob),
            'confidence_range': np.max(y_prob) - np.min(y_prob)
        }
        
        # Add regime-specific statistics
        unique_regimes = np.unique(regimes)
        regime_stats = {}
        for regime in unique_regimes:
            regime_mask = regimes == regime
            regime_probs = y_prob[regime_mask]
            if len(regime_probs) > 0:
                regime_stats[f'regime_{regime}'] = {
                    'mean_probability': np.mean(regime_probs),
                    'std_probability': np.std(regime_probs),
                    'min_probability': np.min(regime_probs),
                    'max_probability': np.max(regime_probs),
                    'sample_count': len(regime_probs)
                }
        
        uncertainty_results['regime_stats'] = regime_stats
        
        return uncertainty_results
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from the stacker model."""
        if not self.is_fitted or self.stacker_model is None:
            return {}
        
        importance = self.stacker_model.feature_importances_
        
        # Create feature names
        feature_names = []
        for i in range(len(importance)):
            if i < 4:  # Base model predictions
                model_names = ['A1_LightGBM', 'A2_XGBoost', 'A3_FTTransformer', 'A4_CatBoost']
                feature_names.append(f"{model_names[i]}_prediction")
            else:
                feature_names.append(f"uncertainty_feature_{i-4}")
        
        return {
            'importance_scores': importance,
            'feature_names': feature_names,
            'top_features': sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:20]
        }
    
    def get_regime_calibration_info(self) -> Dict[str, Any]:
        """Get regime calibration information."""
        if not self.is_fitted or self.regime_calibrator is None:
            return {}
        
        return {
            'per_regime_calibration': self.calibration_config.per_regime_calibration,
            'calibration_method': self.calibration_config.method,
            'cv_folds': self.calibration_config.cv_folds,
            'regimes_calibrated': list(self.regime_calibrator.regime_calibrators.keys()),
            'global_calibrator_available': self.regime_calibrator.global_calibrator is not None
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'stacker_model': self.stacker_model,
            'regime_calibrator': self.regime_calibrator,
            'venn_abers': self.venn_abers,
            'feature_names': self.feature_names,
            'stacker_config': self.stacker_config,
            'calibration_config': self.calibration_config
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"✅ Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'StackerLGBMCalibrated':
        """Load the model from disk."""
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(
            stacker_config=model_data['stacker_config'],
            calibration_config=model_data['calibration_config']
        )
        
        # Restore state
        instance.stacker_model = model_data['stacker_model']
        instance.regime_calibrator = model_data['regime_calibrator']
        instance.venn_abers = model_data['venn_abers']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = True
        
        logger.info(f"✅ Model loaded from {filepath}")
        return instance


# Factory function for easy model creation
def create_stacker_lgbm_calibrated(stacker_config: Optional[StackerConfig] = None,
                                  calibration_config: Optional[CalibrationConfig] = None) -> StackerLGBMCalibrated:
    """Create a Stacker LGBM Calibrated meta-learner with the specified configurations."""
    return StackerLGBMCalibrated(stacker_config, calibration_config)