"""
TCN Classifier Wrapper

This module provides a wrapper to convert TCNRegressor to a classifier
for binary and multiclass classification tasks.

Key Features:
1. Wraps TCNRegressor for classification
2. Configurable threshold for binary classification
3. Sklearn-compatible API
4. Probability prediction support
"""

import numpy as np
from typing import Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array
import logging

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, 
    tprint_debug, tprint_data_format, tprint_data_preview, LogLevel
)

from .error_handling import (
    handle_errors, validate_data,
    MLModelTrainerError, DataValidationError, ModelTrainingError, PredictionError
)

logger = logging.getLogger(__name__)

class TCNClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper to convert TCNRegressor to a classifier."""
    
    def __init__(self, 
                 tcn_regressor,
                 threshold: float = 0.5,
                 probability_method: str = "sigmoid"):  # "sigmoid" or "softmax"
        """
        Initialize TCN classifier wrapper.
        
        Args:
            tcn_regressor: TCNRegressor instance
            threshold: Threshold for binary classification
            probability_method: Method for converting regression outputs to probabilities
        """
        self.tcn_regressor = tcn_regressor
        self.threshold = threshold
        self.probability_method = probability_method
        
        # State
        self.is_fitted = False
        self.n_features_in_ = None
        self.classes_ = None
        self.label_encoder_ = None
    
    @handle_errors(error_type=ModelTrainingError, reraise=True)
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> 'TCNClassifierWrapper':
        """Fit the TCN classifier wrapper."""
        tprint_info(f"Fitting TCN classifier wrapper - X shape: {X.shape}, y shape: {y.shape if hasattr(y, 'shape') else len(y)}")
        
        # Validate inputs
        validate_data(X, y, min_samples=10, min_features=1)
        X, y = check_X_y(X, y)
        tprint_data_format(X, "Input features X", LogLevel.DEBUG)
        tprint_data_format(y, "Target labels y", LogLevel.DEBUG)
        
        # Store feature info
        self.n_features_in_ = X.shape[1]
        
        # Encode labels
        tprint_debug("Encoding labels with LabelEncoder")
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        tprint_data_format(self.classes_, "Encoded classes", LogLevel.DEBUG)
        
        # Fit the underlying regressor
        tprint_info("Fitting underlying TCN regressor")
        self.tcn_regressor.fit(X, y_encoded, **fit_params)
        
        self.is_fitted = True
        tprint_success(f"TCNClassifierWrapper fitted with {len(self.classes_)} classes")
        logger.info(f"TCNClassifierWrapper fitted with {len(self.classes_)} classes")
        
        return self
    
    @handle_errors(error_type=PredictionError, reraise=True)
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            tprint_error("Model must be fitted before prediction")
            raise PredictionError("Model must be fitted before prediction")
        
        tprint_debug(f"Making predictions for {X.shape[0]} samples")
        validate_data(X, min_samples=1, min_features=self.n_features_in_)
        tprint_data_format(X, "Input features for prediction", LogLevel.DEBUG)
        
        # Get regression predictions
        y_reg = self.tcn_regressor.predict(X)
        tprint_data_format(y_reg, "Regression predictions", LogLevel.DEBUG)
        
        # Convert to classification predictions
        if len(self.classes_) == 2:
            # Binary classification
            tprint_debug(f"Binary classification with threshold: {self.threshold}")
            y_pred = (y_reg > self.threshold).astype(int)
        else:
            # Multiclass classification - use argmax
            # For multiclass, we need to reshape the regression output
            if y_reg.ndim == 1:
                # Single output - convert to one-hot and use argmax
                y_pred = np.argmax(self._regression_to_probabilities(y_reg), axis=1)
            else:
                y_pred = np.argmax(y_reg, axis=1)
        
        # Convert back to original labels
        return self.label_encoder_.inverse_transform(y_pred)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_fitted:
            tprint_error("Model must be fitted before prediction")
            raise ValueError("Model must be fitted before prediction")
        
        tprint_debug(f"Making probability predictions for {X.shape[0]} samples")
        
        # Get regression predictions
        y_reg = self.tcn_regressor.predict(X)
        tprint_data_format(y_reg, "Regression predictions for probabilities", LogLevel.DEBUG)
        
        # Convert to probabilities
        probabilities = self._regression_to_probabilities(y_reg)
        tprint_data_format(probabilities, "Probability predictions", LogLevel.DEBUG)
        
        return probabilities
    
    def _regression_to_probabilities(self, y_reg: np.ndarray) -> np.ndarray:
        """Convert regression outputs to probabilities."""
        if len(self.classes_) == 2:
            # Binary classification
            if self.probability_method == "sigmoid":
                # Sigmoid function
                prob_positive = 1 / (1 + np.exp(-y_reg))
                return np.column_stack([1 - prob_positive, prob_positive])
            else:
                # Simple threshold-based probability
                prob_positive = np.where(y_reg > self.threshold, 0.8, 0.2)
                return np.column_stack([1 - prob_positive, prob_positive])
        else:
            # Multiclass classification
            if y_reg.ndim == 1:
                # Single output - convert to one-hot
                if self.probability_method == "softmax":
                    # Softmax function
                    exp_scores = np.exp(y_reg - np.max(y_reg, axis=0, keepdims=True))
                    probabilities = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
                else:
                    # Simple normalization
                    probabilities = np.abs(y_reg)
                    probabilities = probabilities / np.sum(probabilities, axis=0, keepdims=True)
                
                # Reshape to (n_samples, n_classes)
                if probabilities.ndim == 1:
                    # Single sample case
                    probabilities = probabilities.reshape(1, -1)
                
                return probabilities
            else:
                # Multiple outputs - use softmax
                if self.probability_method == "softmax":
                    exp_scores = np.exp(y_reg - np.max(y_reg, axis=1, keepdims=True))
                    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                else:
                    # Simple normalization
                    probabilities = np.abs(y_reg)
                    probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
                
                return probabilities
    
    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for sklearn compatibility."""
        params = {
            'threshold': self.threshold,
            'probability_method': self.probability_method
        }
        
        if deep and hasattr(self.tcn_regressor, 'get_params'):
            params['tcn_regressor'] = self.tcn_regressor.get_params(deep)
        
        return params
    
    def set_params(self, **params) -> 'TCNClassifierWrapper':
        """Set parameters for sklearn compatibility."""
        if 'threshold' in params:
            self.threshold = params.pop('threshold')
        if 'probability_method' in params:
            self.probability_method = params.pop('probability_method')
        
        if hasattr(self.tcn_regressor, 'set_params') and params:
            self.tcn_regressor.set_params(**params)
        
        return self

# Factory function
def create_tcn_classifier(tcn_regressor, **kwargs) -> TCNClassifierWrapper:
    """Create TCN classifier wrapper."""
    return TCNClassifierWrapper(tcn_regressor, **kwargs)