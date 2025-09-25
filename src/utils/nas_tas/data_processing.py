#!/usr/bin/env python3
"""
Unified Data Processing Pipeline

This module provides a unified data processing pipeline for both NAS and TAS systems,
consolidating data preprocessing, validation, and splitting functionality.

Key Features:
- Unified data preprocessing pipeline
- Feature selection and engineering
- Data validation and quality checks
- Train/validation/test splitting
- Cross-validation support
- Data normalization and standardization
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

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


class UnifiedDataProcessor:
    """Unified data processing pipeline for both NAS and TAS systems."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize unified data processor."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize preprocessing components
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        
        # Processing state
        self.is_fitted = False
        self.feature_names = None
        self.target_encoder = None
    
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
            # Handle missing values
            if self.config.get('handle_missing_values', True):
                X, y = self._handle_missing_values(X, y)
            
            # Feature selection
            if self.config.get('enable_feature_selection', False):
                X = self._select_features(X, y, fit=fit)
            
            # Normalization/standardization
            if self.config.get('normalize_data', False):
                X = self._normalize_data(X, fit=fit)
            elif self.config.get('standardize_data', False):
                X = self._standardize_data(X, fit=fit)
            
            # Outlier detection and handling
            if self.config.get('outlier_detection', False):
                X, y = self._handle_outliers(X, y)
            
            # Update processing info
            processing_info.update({
                'final_shape': X.shape,
                'missing_values_handled': self.config.get('handle_missing_values', True),
                'feature_selection_applied': self.config.get('enable_feature_selection', False),
                'normalization_applied': self.config.get('normalize_data', False),
                'standardization_applied': self.config.get('standardize_data', False),
                'outliers_handled': self.config.get('outlier_detection', False)
            })
            
            if fit:
                self.is_fitted = True
            
            tprint_success(f"Data processing completed. Final shape: {X.shape}")
            
            return X, y, processing_info
            
        except Exception as e:
            tprint_error(f"Data processing failed: {e}")
            processing_info['error'] = str(e)
            return X, y, processing_info
    
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
        max_features = self.config.get('max_features', X.shape[1])
        
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
        
        validation_split = self.config.get('validation_split', 0.2)
        
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
            # Fallback to simple split
            split_idx = int(len(X) * (1 - validation_split))
            return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
    
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