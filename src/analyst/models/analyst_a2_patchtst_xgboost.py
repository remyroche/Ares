"""
Analyst A2 Model: PatchTST-Embed + XGBoost

Binary "green light" classification with:
- 300+ features, regime posteriors, cross-TF aggregates
- BCE loss (class-weighted)
- XGBoost with gbtree booster, max_depth=6, eta=0.05, subsample=0.7, colsample_bytree=0.7
- Diversified booster bias compared to LightGBM
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import joblib
import os

logger = logging.getLogger(__name__)


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost model."""
    max_depth: int = 6
    learning_rate: float = 0.05  # eta
    n_estimators: int = 1000
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    min_child_weight: int = 1
    gamma: float = 0.0
    booster: str = 'gbtree'
    objective: str = 'binary:logistic'
    eval_metric: str = 'logloss'
    tree_method: str = 'hist'
    grow_policy: str = 'depthwise'
    max_leaves: int = 0
    max_bin: int = 256
    verbosity: int = 0
    random_state: int = 42


@dataclass
class CalibrationConfig:
    """Configuration for model calibration."""
    method: str = 'isotonic'  # 'isotonic' or 'sigmoid'
    cv_folds: int = 5
    enable_venn_abers: bool = True
    confidence_levels: List[float] = None

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.5, 0.6, 0.7, 0.8, 0.9]


# Import PatchTST embedding from A1 model
from .analyst_a1_patchtst_lightgbm import PatchTSTEmbedding, PatchTSTConfig, VennAbersCalibration


class AnalystA2Model:
    """Analyst A2: PatchTST-Embed + XGBoost with calibration."""
    
    def __init__(self, 
                 patchtst_config: Optional[PatchTSTConfig] = None,
                 xgboost_config: Optional[XGBoostConfig] = None,
                 calibration_config: Optional[CalibrationConfig] = None):
        self.patchtst_config = patchtst_config or PatchTSTConfig()
        self.xgboost_config = xgboost_config or XGBoostConfig()
        self.calibration_config = calibration_config or CalibrationConfig()
        
        # Model components
        self.patchtst_embedding = PatchTSTEmbedding(self.patchtst_config)
        self.xgboost_model = None
        self.calibrated_model = None
        self.venn_abers = None
        self.feature_names = None
        self.is_fitted = False
        
        logger.info("Initialized Analyst A2 Model (PatchTST-Embed + XGBoost)")
    
    def _prepare_features(self, X: np.ndarray, regimes: Optional[np.ndarray] = None) -> np.ndarray:
        """Prepare features with PatchTST embedding."""
        # Get PatchTST embeddings
        patchtst_features = self.patchtst_embedding.transform(X, regimes)
        
        # Combine with original features
        if X.shape[0] == patchtst_features.shape[0]:
            combined_features = np.hstack([X, patchtst_features])
        else:
            # Handle dimension mismatch by padding/truncating
            min_samples = min(X.shape[0], patchtst_features.shape[0])
            combined_features = np.hstack([
                X[:min_samples], 
                patchtst_features[:min_samples]
            ])
        
        return combined_features
    
    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights for imbalanced data."""
        unique_classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        class_weights = {}
        for class_label, count in zip(unique_classes, counts):
            class_weights[class_label] = total_samples / (len(unique_classes) * count)
        
        return class_weights
    
    def _create_sample_weights(self, y: np.ndarray, class_weights: Dict[int, float]) -> np.ndarray:
        """Create sample weights for XGBoost."""
        sample_weights = np.zeros(len(y))
        for class_label, weight in class_weights.items():
            sample_weights[y == class_label] = weight
        return sample_weights
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            regimes: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None) -> 'AnalystA2Model':
        """Fit the Analyst A2 model."""
        logger.info("Fitting Analyst A2 Model...")
        
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
            X = X.values
        
        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Ensure binary classification
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(f"Binary classification requires exactly 2 classes, got {len(unique_classes)}")
        
        # Compute class weights
        class_weights = self._compute_class_weights(y)
        logger.info(f"Class weights: {class_weights}")
        
        # Fit PatchTST embedding
        self.patchtst_embedding.fit(X, y, regimes)
        
        # Prepare features
        X_enhanced = self._prepare_features(X, regimes)
        logger.info(f"Enhanced features shape: {X_enhanced.shape}")
        
        # Create sample weights for XGBoost
        if sample_weight is not None:
            # Combine with class weights
            class_sample_weights = self._create_sample_weights(y, class_weights)
            final_sample_weights = sample_weight * class_sample_weights
        else:
            final_sample_weights = self._create_sample_weights(y, class_weights)
        
        # Create XGBoost model
        self.xgboost_model = xgb.XGBClassifier(
            max_depth=self.xgboost_config.max_depth,
            learning_rate=self.xgboost_config.learning_rate,
            n_estimators=self.xgboost_config.n_estimators,
            subsample=self.xgboost_config.subsample,
            colsample_bytree=self.xgboost_config.colsample_bytree,
            reg_alpha=self.xgboost_config.reg_alpha,
            reg_lambda=self.xgboost_config.reg_lambda,
            min_child_weight=self.xgboost_config.min_child_weight,
            gamma=self.xgboost_config.gamma,
            booster=self.xgboost_config.booster,
            objective=self.xgboost_config.objective,
            eval_metric=self.xgboost_config.eval_metric,
            tree_method=self.xgboost_config.tree_method,
            grow_policy=self.xgboost_config.grow_policy,
            max_leaves=self.xgboost_config.max_leaves,
            max_bin=self.xgboost_config.max_bin,
            verbosity=self.xgboost_config.verbosity,
            random_state=self.xgboost_config.random_state,
            n_jobs=-1
        )
        
        # Fit XGBoost model
        self.xgboost_model.fit(
            X_enhanced, 
            y, 
            sample_weight=final_sample_weights,
            eval_set=[(X_enhanced, y)],
            verbose=False
        )
        
        # Get predictions for calibration
        y_prob = self.xgboost_model.predict_proba(X_enhanced)[:, 1]
        
        # Fit calibration
        if self.calibration_config.method in ['isotonic', 'sigmoid']:
            self.calibrated_model = CalibratedClassifierCV(
                self.xgboost_model,
                method=self.calibration_config.method,
                cv=self.calibration_config.cv_folds
            )
            self.calibrated_model.fit(X_enhanced, y)
        
        # Fit Venn-Abers calibration
        if self.calibration_config.enable_venn_abers:
            self.venn_abers = VennAbersCalibration(self.calibration_config.confidence_levels)
            self.venn_abers.fit(y, y_prob)
        
        self.is_fitted = True
        logger.info("✅ Analyst A2 Model fitted successfully")
        return self
    
    def predict_proba(self, X: np.ndarray, regimes: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare features
        X_enhanced = self._prepare_features(X, regimes)
        
        # Get predictions
        if self.calibrated_model is not None:
            y_prob = self.calibrated_model.predict_proba(X_enhanced)[:, 1]
        else:
            y_prob = self.xgboost_model.predict_proba(X_enhanced)[:, 1]
        
        return y_prob
    
    def predict_uncertainty(self, X: np.ndarray, regimes: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Predict uncertainty estimates."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get base predictions
        y_prob = self.predict_proba(X, regimes)
        
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
        
        return uncertainty_results
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from the model."""
        if not self.is_fitted or self.xgboost_model is None:
            return {}
        
        importance = self.xgboost_model.feature_importances_
        
        # Create feature names
        if self.feature_names is not None:
            # Original features + PatchTST features
            patchtst_feature_names = [f'patchtst_{i}' for i in range(self.patchtst_embedding.patch_embeddings.shape[1])]
            all_feature_names = self.feature_names + patchtst_feature_names
        else:
            all_feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        return {
            'importance_scores': importance,
            'feature_names': all_feature_names,
            'top_features': sorted(zip(all_feature_names, importance), key=lambda x: x[1], reverse=True)[:20]
        }
    
    def get_booster_info(self) -> Dict[str, Any]:
        """Get XGBoost booster information."""
        if not self.is_fitted or self.xgboost_model is None:
            return {}
        
        booster = self.xgboost_model.get_booster()
        
        return {
            'booster_type': self.xgboost_config.booster,
            'num_trees': booster.num_boosted_rounds(),
            'max_depth': self.xgboost_config.max_depth,
            'learning_rate': self.xgboost_config.learning_rate,
            'subsample': self.xgboost_config.subsample,
            'colsample_bytree': self.xgboost_config.colsample_bytree,
            'tree_method': self.xgboost_config.tree_method,
            'grow_policy': self.xgboost_config.grow_policy
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'patchtst_embedding': self.patchtst_embedding,
            'xgboost_model': self.xgboost_model,
            'calibrated_model': self.calibrated_model,
            'venn_abers': self.venn_abers,
            'feature_names': self.feature_names,
            'patchtst_config': self.patchtst_config,
            'xgboost_config': self.xgboost_config,
            'calibration_config': self.calibration_config
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"✅ Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'AnalystA2Model':
        """Load the model from disk."""
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(
            patchtst_config=model_data['patchtst_config'],
            xgboost_config=model_data['xgboost_config'],
            calibration_config=model_data['calibration_config']
        )
        
        # Restore state
        instance.patchtst_embedding = model_data['patchtst_embedding']
        instance.xgboost_model = model_data['xgboost_model']
        instance.calibrated_model = model_data['calibrated_model']
        instance.venn_abers = model_data['venn_abers']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = True
        
        logger.info(f"✅ Model loaded from {filepath}")
        return instance


# Factory function for easy model creation
def create_analyst_a2_model(patchtst_config: Optional[PatchTSTConfig] = None,
                           xgboost_config: Optional[XGBoostConfig] = None,
                           calibration_config: Optional[CalibrationConfig] = None) -> AnalystA2Model:
    """Create an Analyst A2 model with the specified configurations."""
    return AnalystA2Model(patchtst_config, xgboost_config, calibration_config)