"""
Stacker LGBM Calibrated Meta-Learner

This module implements a stacker LGBM with calibration for both analyst and tactician meta-learners.
The stacker combines base model predictions and applies calibration to improve probability estimates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import warnings
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import log_loss, brier_score_loss

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class StackerLGBMCalibratedConfig:
    """Configuration for Stacker LGBM Calibrated meta-learner."""
    # LGBM parameters (updated hyperparameters)
    max_depth: int = 2  # Explicitly constrain depth for calibrated stacker
    num_leaves: int = 4  # Consistent with depth-2 trees (<= 2^depth)
    min_child_samples: int = 800  # 600-1000 range
    lambda_l2: float = 40.0  # Slightly stronger regularization for shallow trees
    feature_fraction: float = 0.7  # 0.6-0.8 range
    learning_rate: float = 0.05
    n_estimators: int = 500

    # Calibration parameters
    calibration_method: str = "isotonic"  # or "sigmoid"
    cv_folds: int = 5

    # Training parameters
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = -1

class StackerLGBMCalibrated(BaseEstimator, RegressorMixin):
    """
    Stacker LGBM Calibrated Meta-Learner.

    This model combines base model predictions using LightGBM and applies
    calibration to improve probability estimates.
    """

    def __init__(self, config: Optional[StackerLGBMCalibratedConfig] = None):
        """Initialize the Stacker LGBM Calibrated model."""
        self.config = config or StackerLGBMCalibratedConfig()

        # Components
        self.lgbm_model = None
        self.calibrated_model = None
        self.scaler = None

        # State
        self.fitted = False
        self.feature_names = None
        self.base_model_names = None

    def _prepare_stacking_features(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Prepare features for stacking from base model predictions."""
        try:
            # Extract prediction arrays
            prediction_arrays = []
            feature_names = []

            for model_name, predictions in base_predictions.items():
                if isinstance(predictions, dict):
                    # If predictions is a dict, extract all values
                    for key, values in predictions.items():
                        if isinstance(values, np.ndarray):
                            prediction_arrays.append(values)
                            feature_names.append(f"{model_name}_{key}")
                elif isinstance(predictions, np.ndarray):
                    # If predictions is a numpy array
                    if predictions.ndim == 1:
                        prediction_arrays.append(predictions.reshape(-1, 1))
                        feature_names.append(f"{model_name}_prediction")
                    else:
                        # Multi-dimensional array
                        for i in range(predictions.shape[1]):
                            prediction_arrays.append(predictions[:, i])
                            feature_names.append(f"{model_name}_pred_{i}")
                else:
                    logger.warning(f"⚠️ Unsupported prediction type for {model_name}: {type(predictions)}")
                    continue

            if not prediction_arrays:
                raise ValueError("No valid base model predictions provided")

            # Stack predictions horizontally
            stacking_features = np.column_stack(prediction_arrays)

            # Store feature names
            self.feature_names = feature_names

            return stacking_features

        except Exception as e:
            logger.error(f"❌ Stacking feature preparation failed: {e}")
            raise

    def _create_meta_features(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create meta-features from base model predictions."""
        try:
            meta_features = []

            # Extract all predictions into a list
            all_predictions = []
            for model_name, predictions in base_predictions.items():
                if isinstance(predictions, dict):
                    for key, values in predictions.items():
                        if isinstance(values, np.ndarray) and values.ndim == 1:
                            all_predictions.append(values)
                elif isinstance(predictions, np.ndarray) and predictions.ndim == 1:
                    all_predictions.append(predictions)

            if not all_predictions:
                return np.array([])

            # Stack all predictions
            pred_matrix = np.column_stack(all_predictions)

            # Create meta-features
            meta_features = []

            # Mean prediction
            meta_features.append(np.mean(pred_matrix, axis=1))

            # Standard deviation of predictions
            meta_features.append(np.std(pred_matrix, axis=1))

            # Min and max predictions
            meta_features.append(np.min(pred_matrix, axis=1))
            meta_features.append(np.max(pred_matrix, axis=1))

            # Range (max - min)
            meta_features.append(np.max(pred_matrix, axis=1) - np.min(pred_matrix, axis=1))

            # Median prediction
            meta_features.append(np.median(pred_matrix, axis=1))

            # Prediction disagreement (variance)
            meta_features.append(np.var(pred_matrix, axis=1))

            # Number of models predicting above threshold
            threshold = 0.5
            meta_features.append(np.sum(pred_matrix > threshold, axis=1))

            # Consensus strength (fraction of models agreeing with majority)
            majority_pred = np.round(np.mean(pred_matrix, axis=1))
            consensus = np.mean(np.abs(pred_matrix - majority_pred.reshape(-1, 1)) < 0.1, axis=1)
            meta_features.append(consensus)

            return np.column_stack(meta_features)

        except Exception as e:
            logger.warning(f"⚠️ Meta-feature creation failed: {e}")
            return np.array([])

    def fit(self, base_predictions: Dict[str, np.ndarray], y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> 'StackerLGBMCalibrated':
        """Fit the Stacker LGBM Calibrated model."""
        try:
            import lightgbm as lgb

            # Store base model names
            self.base_model_names = list(base_predictions.keys())

            # Prepare stacking features
            stacking_features = self._prepare_stacking_features(base_predictions)

            # Create meta-features
            meta_features = self._create_meta_features(base_predictions)

            # Combine stacking features with meta-features
            if meta_features.size > 0:
                X_combined = np.hstack([stacking_features, meta_features])
            else:
                X_combined = stacking_features

            # Scale features
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_combined)

            # Create LightGBM model
            self.lgbm_model = lgb.LGBMRegressor(
                max_depth=self.config.max_depth,
                num_leaves=self.config.num_leaves,
                min_child_samples=self.config.min_child_samples,
                reg_lambda=self.config.lambda_l2,
                feature_fraction=self.config.feature_fraction,
                learning_rate=self.config.learning_rate,
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose
            )

            # Fit LightGBM model
            if sample_weight is not None:
                self.lgbm_model.fit(X_scaled, y, sample_weight=sample_weight)
            else:
                self.lgbm_model.fit(X_scaled, y)

            # Apply calibration
            if self.config.calibration_method in ["isotonic", "sigmoid"]:
                # Get out-of-fold predictions for calibration
                oof_predictions = cross_val_predict(
                    self.lgbm_model, X_scaled, y,
                    cv=self.config.cv_folds, method='predict'
                )

                # Create a dummy classifier for calibration
                from sklearn.base import BaseEstimator, ClassifierMixin

                class DummyClassifier(BaseEstimator, ClassifierMixin):
                    def __init__(self, regressor):
                        self.regressor = regressor

                    def fit(self, X, y):
                        return self

                    def predict_proba(self, X):
                        predictions = self.regressor.predict(X)
                        # Convert regression predictions to probabilities
                        predictions = np.clip(predictions, 0, 1)
                        return np.column_stack([1 - predictions, predictions])

                dummy_classifier = DummyClassifier(self.lgbm_model)

                # Apply calibration
                self.calibrated_model = CalibratedClassifierCV(
                    dummy_classifier,
                    method=self.config.calibration_method,
                    cv=self.config.cv_folds
                )

                # Fit calibration
                self.calibrated_model.fit(X_scaled, y)
            else:
                # No calibration
                self.calibrated_model = None

            self.fitted = True
            logger.info(f"✅ Stacker LGBM Calibrated model fitted with {X_combined.shape[1]} features")

            return self

        except Exception as e:
            logger.error(f"❌ Stacker LGBM Calibrated model fitting failed: {e}")
            raise

    def predict(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Make predictions using the fitted model."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        try:
            # Prepare stacking features
            stacking_features = self._prepare_stacking_features(base_predictions)

            # Create meta-features
            meta_features = self._create_meta_features(base_predictions)

            # Combine features
            if meta_features.size > 0:
                X_combined = np.hstack([stacking_features, meta_features])
            else:
                X_combined = stacking_features

            # Scale features
            X_scaled = self.scaler.transform(X_combined)

            # Make predictions
            if self.calibrated_model is not None:
                # Use calibrated predictions
                proba = self.calibrated_model.predict_proba(X_scaled)
                predictions = proba[:, 1]  # Get positive class probability
            else:
                # Use raw LGBM predictions
                predictions = self.lgbm_model.predict(X_scaled)

            return predictions

        except Exception as e:
            logger.error(f"❌ Stacker LGBM Calibrated model prediction failed: {e}")
            raise

    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Make probability predictions using the fitted model."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        try:
            # Prepare stacking features
            stacking_features = self._prepare_stacking_features(base_predictions)

            # Create meta-features
            meta_features = self._create_meta_features(base_predictions)

            # Combine features
            if meta_features.size > 0:
                X_combined = np.hstack([stacking_features, meta_features])
            else:
                X_combined = stacking_features

            # Scale features
            X_scaled = self.scaler.transform(X_combined)

            # Make probability predictions
            if self.calibrated_model is not None:
                # Use calibrated probabilities
                proba = self.calibrated_model.predict_proba(X_scaled)
            else:
                # Convert regression predictions to probabilities
                predictions = self.lgbm_model.predict(X_scaled)
                predictions = np.clip(predictions, 0, 1)
                proba = np.column_stack([1 - predictions, predictions])

            return proba

        except Exception as e:
            logger.error(f"❌ Stacker LGBM Calibrated model probability prediction failed: {e}")
            raise

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from the LightGBM model."""
        if not self.fitted or self.lgbm_model is None:
            return np.array([])

        try:
            return self.lgbm_model.feature_importances_
        except Exception as e:
            logger.warning(f"⚠️ Could not get feature importance: {e}")
            return np.array([])

    def get_feature_names(self) -> List[str]:
        """Get feature names used in the model."""
        if not self.fitted:
            return []

        return self.feature_names or []

    def evaluate_calibration(self, base_predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate calibration quality."""
        if not self.fitted:
            raise ValueError("Model must be fitted before evaluation")

        try:
            # Get probability predictions
            proba = self.predict_proba(base_predictions)
            proba_positive = proba[:, 1]

            # Calculate calibration metrics
            log_loss_score = log_loss(y_true, proba_positive)
            brier_score = brier_score_loss(y_true, proba_positive)

            return {
                'log_loss': log_loss_score,
                'brier_score': brier_score,
                'calibration_method': self.config.calibration_method
            }

        except Exception as e:
            logger.warning(f"⚠️ Calibration evaluation failed: {e}")
            return {}

# Factory function
def create_stacker_lgbm_calibrated(config: Optional[StackerLGBMCalibratedConfig] = None) -> StackerLGBMCalibrated:
    """Create Stacker LGBM Calibrated model."""
    return StackerLGBMCalibrated(config)
