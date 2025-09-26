#!/usr/bin/env python3
"""
Unified Data Processing Pipeline

This module provides a wrapper around the unified data preprocessing system
for both NAS and TAS systems, maintaining backward compatibility.
"""

# Import additional utilities for data splitting and cross-validation
import importlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

if TYPE_CHECKING:  # pragma: no cover - type checking helper
    from src.utils.nas_tas.data_preprocessing import (
        UnifiedDataPreprocessor,
        PreprocessingConfig,
        PreprocessingResult,
        PreprocessingStep,
    )
else:
    UnifiedDataPreprocessor = Any  # type: ignore
    PreprocessingConfig = Any  # type: ignore
    PreprocessingResult = Any  # type: ignore
    PreprocessingStep = Any  # type: ignore

# Import utility modules
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    )
    UTILITY_MODULES_AVAILABLE = True
except ImportError:
    UTILITY_MODULES_AVAILABLE = False
    # Fallback functions
    def tprint(*args, **kwargs):
        print(*args, **kwargs)
    def tprint_info(*args, **kwargs):
        print("INFO:", *args, **kwargs)
    def tprint_warning(*args, **kwargs):
        print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs):
        print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs):
        print("SUCCESS:", *args, **kwargs)

logger = logging.getLogger(__name__)


class DataProcessorConfigurationError(ValueError):
    """Raised when the provided configuration for the data processor is invalid."""


class DataProcessingError(RuntimeError):
    """Raised when the unified data processor fails to process data."""


class DataSplitError(RuntimeError):
    """Raised when the unified data processor fails to split data."""


def _load_data_preprocessing_module():
    return importlib.import_module("src.utils.nas_tas.data_preprocessing")


def _get_preprocessing_config_cls():
    module = _load_data_preprocessing_module()
    return getattr(module, "PreprocessingConfig")


def _get_preprocessor_cls():
    module = _load_data_preprocessing_module()
    return getattr(module, "UnifiedDataPreprocessor")


@dataclass(frozen=True)
class _ValidatedDataProcessorConfig:
    validation_split: float
    max_features: Optional[int]
    allow_non_numeric: bool


class UnifiedDataProcessor:
    """Unified data processing pipeline for both NAS and TAS systems."""

    def __init__(self, config: Dict[str, Any], preprocessor: Optional[Any] = None):
        """Initialize unified data processor."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self._validated_config = self._validate_and_freeze_config(config)

        # Initialize the unified data preprocessor
        if preprocessor is not None:
            self.data_preprocessor = preprocessor
        else:
            preprocessing_config_cls = _get_preprocessing_config_cls()
            preprocessing_config = preprocessing_config_cls()
            preprocessor_cls = _get_preprocessor_cls()
            self.data_preprocessor = preprocessor_cls(preprocessing_config)
        
        # Initialize preprocessing components
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        
        # Processing state
        self.is_fitted = False
        self.feature_names = None
        self.target_encoder = None

    def _validate_and_freeze_config(self, config: Dict[str, Any]) -> _ValidatedDataProcessorConfig:
        """Validate incoming configuration and return an immutable view."""
        validation_split = config.get('validation_split', 0.2)
        if not isinstance(validation_split, (int, float)):
            raise DataProcessorConfigurationError("validation_split must be numeric")
        if not 0.0 < validation_split < 0.5:
            raise DataProcessorConfigurationError("validation_split must be between 0 and 0.5 for stability")

        max_features = config.get('max_features')
        if max_features is not None:
            if not isinstance(max_features, int) or max_features <= 0:
                raise DataProcessorConfigurationError("max_features must be a positive integer when provided")

        allow_non_numeric = bool(config.get('allow_non_numeric', True))

        return _ValidatedDataProcessorConfig(
            validation_split=float(validation_split),
            max_features=max_features,
            allow_non_numeric=allow_non_numeric,
        )
    
    def process_data(self, 
                    X: np.ndarray, 
                    y: np.ndarray,
                    data_type: str = "general",
                    fit: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Process data using unified pipeline."""
        
        tprint_info(f"Processing {data_type} data with shape: {X.shape}")
        
        processing_info = {
            'original_shape': X.shape,
            'data_type': data_type,
            'fit': fit
        }
        
        try:
            # Convert numpy arrays to DataFrame for unified preprocessing
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X)
            else:
                X_df = X.copy()

            if not self._validated_config.allow_non_numeric and not np.issubdtype(X_df.to_numpy().dtype, np.number):
                raise DataProcessingError("Input features must be numeric when non-numeric data is disallowed")

            # Use the unified data preprocessor
            preprocessing_result = self.data_preprocessor.preprocess_data(X_df)

            # Convert back to numpy arrays
            X_processed = preprocessing_result.processed_data.values
            y_processed = y  # Target remains unchanged

            # Update processing info
            processing_info.update({
                'final_shape': X_processed.shape,
                'preprocessing_steps_applied': preprocessing_result.preprocessing_steps_applied,
                'data_quality_improvement': getattr(preprocessing_result, 'data_quality_improvement', None),
                'preprocessing_time': getattr(preprocessing_result, 'preprocessing_time', None),
                'memory_usage': getattr(preprocessing_result, 'memory_usage', None),
                'hardware_acceleration_used': getattr(preprocessing_result, 'hardware_acceleration_used', None),
                'matrix_operations_used': getattr(preprocessing_result, 'matrix_operations_used', None)
            })

            if fit:
                self.is_fitted = True

            tprint_success(f"Data processing completed. Final shape: {X_processed.shape}")

            return X_processed, y_processed, processing_info

        except Exception as e:
            processing_info['error'] = str(e)
            tprint_error(f"Data processing failed: {e}")
            raise DataProcessingError(str(e)) from e
    
    def _handle_missing_values(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Handle missing values in data."""
        # Check for missing values
        if np.isnan(X).any():
            tprint_warning("Missing values detected in features")
            # Fill missing values with median
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        if np.isnan(y).any():
            tprint_warning("Missing values detected in target")
            # For target, we might want to drop rows with missing values
            mask = ~np.isnan(y)
            X = X[mask]
            y = y[mask]
        
        return X, y
    
    def _select_features(self, X: np.ndarray, y: np.ndarray, fit: bool = True) -> np.ndarray:
        """Select features using mutual information or F-test."""
        max_features = self._validated_config.max_features or X.shape[1]

        if X.shape[1] <= max_features:
            return X
        
        try:
            # Determine if classification or regression
            n_unique = len(np.unique(y))
            
            if n_unique <= 10:  # Classification
                score_func = mutual_info_classif
            else:  # Regression
                score_func = f_classif
            
            if fit or self.feature_selector is None:
                self.feature_selector = SelectKBest(score_func=score_func, k=max_features)
                X_selected = self.feature_selector.fit_transform(X, y)
            else:
                X_selected = self.feature_selector.transform(X)
            
            tprint_info(f"Feature selection: {X.shape[1]} -> {X_selected.shape[1]} features")
            return X_selected
            
        except Exception as e:
            tprint_warning(f"Feature selection failed: {e}")
            return X
    
    def _normalize_data(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize data to [0, 1] range."""
        try:
            if fit or self.scaler is None:
                from sklearn.preprocessing import MinMaxScaler
                self.scaler = MinMaxScaler()
                X_normalized = self.scaler.fit_transform(X)
            else:
                X_normalized = self.scaler.transform(X)
            
            return X_normalized
            
        except Exception as e:
            tprint_warning(f"Normalization failed: {e}")
            return X
    
    def _standardize_data(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Standardize data to zero mean and unit variance."""
        try:
            if fit or self.scaler is None:
                self.scaler = StandardScaler()
                X_standardized = self.scaler.fit_transform(X)
            else:
                X_standardized = self.scaler.transform(X)
            
            return X_standardized
            
        except Exception as e:
            tprint_warning(f"Standardization failed: {e}")
            return X
    
    def _handle_outliers(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Handle outliers using IQR method."""
        try:
            # Simple outlier detection using IQR
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find non-outlier rows
            mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)
            
            if np.sum(mask) < len(mask) * 0.8:  # If more than 20% are outliers, be conservative
                tprint_warning("Too many outliers detected, keeping all data")
                return X, y
            
            X_clean = X[mask]
            y_clean = y[mask]
            
            tprint_info(f"Outlier handling: {len(X)} -> {len(X_clean)} samples")
            
            return X_clean, y_clean
            
        except Exception as e:
            tprint_warning(f"Outlier handling failed: {e}")
            return X, y
    
    def split_data(self, 
                   X: np.ndarray, 
                   y: np.ndarray,
                   data_type: str = "general") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/validation sets."""
        
        validation_split = self._validated_config.validation_split

        try:
            # Determine if classification or regression
            n_unique = len(np.unique(y))

            if n_unique <= 10:  # Classification - use stratified split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=validation_split, random_state=42, stratify=y
                )
            else:  # Regression - use random split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=validation_split, random_state=42
                )

            tprint_success(f"Data split: train={X_train.shape[0]}, val={X_val.shape[0]}")

            return X_train, X_val, y_train, y_val

        except Exception as e:
            tprint_error(f"Data splitting failed: {e}")
            raise DataSplitError(str(e)) from e
    
    def get_cross_validation_splits(self, 
                                   X: np.ndarray, 
                                   y: np.ndarray,
                                   n_splits: int = 5,
                                   data_type: str = "general") -> Any:
        """Get cross-validation splits."""
        
        try:
            # Determine if classification or regression
            n_unique = len(np.unique(y))
            
            if n_unique <= 10:  # Classification - use stratified K-fold
                return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            else:  # Regression - use regular K-fold or time series split
                if data_type == "time_series":
                    return TimeSeriesSplit(n_splits=n_splits)
                else:
                    from sklearn.model_selection import KFold
                    return KFold(n_splits=n_splits, shuffle=True, random_state=42)
                    
        except Exception as e:
            tprint_warning(f"Cross-validation setup failed: {e}")
            from sklearn.model_selection import KFold
            return KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data using fitted scaler."""
        if self.scaler is not None:
            try:
                return self.scaler.inverse_transform(X)
            except Exception as e:
                tprint_warning(f"Inverse transform failed: {e}")
        
        return X
    
    def get_feature_names(self) -> Optional[List[str]]:
        """Get feature names if available."""
        if self.feature_selector is not None and hasattr(self.feature_selector, 'get_support'):
            # Get selected feature indices
            selected_indices = self.feature_selector.get_support(indices=True)
            if self.feature_names is not None:
                return [self.feature_names[i] for i in selected_indices]
        
        return self.feature_names
    
    def set_feature_names(self, feature_names: List[str]):
        """Set feature names."""
        self.feature_names = feature_names


__all__ = [
    'UnifiedDataProcessor',
    'DataProcessorConfigurationError',
    'DataProcessingError',
    'DataSplitError',
]
