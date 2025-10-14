"""
Robust Feature Scaling

This module provides robust scaling methods and validation for feature comparison.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

class RobustFeatureScaler:
    """
    Robust feature scaler with multiple scaling methods and validation.
    """
    
    def __init__(self, method: str = 'robust', quantile_range: Tuple[float, float] = (25.0, 75.0)):
        """
        Initialize robust feature scaler.
        
        Args:
            method: Scaling method ('standard', 'robust', 'minmax', 'quantile', 'power', 'robust_quantile')
            quantile_range: Quantile range for robust scaling
        """
        self.method = method
        self.quantile_range = quantile_range
        self.scaler = None
        self.scaling_params = {}
        self.is_fitted = False
        
    def _get_scaler(self):
        """Get the appropriate scaler based on method."""
        if self.method == 'standard':
            return StandardScaler()
        elif self.method == 'robust':
            return RobustScaler(quantile_range=self.quantile_range)
        elif self.method == 'minmax':
            return MinMaxScaler()
        elif self.method == 'quantile':
            return QuantileTransformer(output_distribution='normal', random_state=42)
        elif self.method == 'power':
            return PowerTransformer(method='yeo-johnson', standardize=True)
        elif self.method == 'robust_quantile':
            return RobustScaler(quantile_range=self.quantile_range)
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler and transform data.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Scaled DataFrame
        """
        # Handle missing values
        X_clean = X.fillna(X.median())
        
        # Initialize scaler
        self.scaler = self._get_scaler()
        
        # Fit and transform
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Convert back to DataFrame
        X_scaled_df = pd.DataFrame(
            X_scaled,
            columns=X_clean.columns,
            index=X_clean.index
        )
        
        # Store scaling parameters
        self.scaling_params = {
            'method': self.method,
            'quantile_range': self.quantile_range,
            'feature_means': X_clean.mean().to_dict(),
            'feature_stds': X_clean.std().to_dict(),
            'feature_medians': X_clean.median().to_dict(),
            'feature_q25': X_clean.quantile(0.25).to_dict(),
            'feature_q75': X_clean.quantile(0.75).to_dict(),
            'feature_mins': X_clean.min().to_dict(),
            'feature_maxs': X_clean.max().to_dict()
        }
        
        self.is_fitted = True
        
        return X_scaled_df
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted scaler.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Scaled DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        # Handle missing values using stored parameters
        X_clean = X.fillna(pd.Series(self.scaling_params['feature_medians']))
        
        # Transform
        X_scaled = self.scaler.transform(X_clean)
        
        # Convert back to DataFrame
        X_scaled_df = pd.DataFrame(
            X_scaled,
            columns=X_clean.columns,
            index=X_clean.index
        )
        
        return X_scaled_df
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled data.
        
        Args:
            X: Scaled DataFrame
            
        Returns:
            Original scale DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        
        X_original = self.scaler.inverse_transform(X)
        
        return pd.DataFrame(
            X_original,
            columns=X.columns,
            index=X.index
        )
    
    def validate_scaling(self, X_original: pd.DataFrame, X_scaled: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the quality of scaling.
        
        Args:
            X_original: Original data
            X_scaled: Scaled data
            
        Returns:
            Validation metrics
        """
        validation = {
            'scaling_method': self.method,
            'n_features': X_scaled.shape[1],
            'n_samples': X_scaled.shape[0],
            'scaling_quality': {}
        }
        
        # Check for infinite or NaN values
        has_inf = np.isinf(X_scaled).any().any()
        has_nan = X_scaled.isna().any().any()
        
        validation['scaling_quality']['has_infinite'] = has_inf
        validation['scaling_quality']['has_nan'] = has_nan
        
        # Check scaling statistics
        if not has_inf and not has_nan:
            validation['scaling_quality']['mean_abs'] = X_scaled.abs().mean().mean()
            validation['scaling_quality']['std'] = X_scaled.std().mean()
            validation['scaling_quality']['min'] = X_scaled.min().min()
            validation['scaling_quality']['max'] = X_scaled.max().max()
            
            # Check if data is properly centered and scaled
            if self.method in ['standard', 'robust']:
                validation['scaling_quality']['mean_close_to_zero'] = abs(X_scaled.mean().mean()) < 0.01
                validation['scaling_quality']['std_close_to_one'] = abs(X_scaled.std().mean() - 1.0) < 0.1
        
        return validation

class MultiMethodScaler:
    """
    Apply multiple scaling methods and compare their effectiveness.
    """
    
    def __init__(self, methods: List[str] = None):
        """
        Initialize multi-method scaler.
        
        Args:
            methods: List of scaling methods to compare
        """
        if methods is None:
            methods = ['standard', 'robust', 'minmax', 'quantile', 'power']
        
        self.methods = methods
        self.scalers = {}
        self.scaled_data = {}
        self.validation_results = {}
    
    def fit_transform_all(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Apply all scaling methods and return results.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Dictionary with scaled data for each method
        """
        for method in self.methods:
            try:
                scaler = RobustFeatureScaler(method=method)
                X_scaled = scaler.fit_transform(X)
                
                self.scalers[method] = scaler
                self.scaled_data[method] = X_scaled
                
                # Validate scaling
                self.validation_results[method] = scaler.validate_scaling(X, X_scaled)
                
                logger.info(f"Applied {method} scaling successfully")
                
            except Exception as e:
                logger.error(f"Error applying {method} scaling: {e}")
                self.validation_results[method] = {'error': str(e)}
        
        return self.scaled_data
    
    def get_best_scaling_method(self, criteria: str = 'robustness') -> str:
        """
        Get the best scaling method based on criteria.
        
        Args:
            criteria: Criteria for selection ('robustness', 'normality', 'stability')
            
        Returns:
            Best scaling method name
        """
        if not self.validation_results:
            return self.methods[0]
        
        scores = {}
        
        for method, validation in self.validation_results.items():
            if 'error' in validation:
                scores[method] = -1
                continue
            
            if criteria == 'robustness':
                # Prefer methods that handle outliers well
                has_inf = validation['scaling_quality'].get('has_infinite', True)
                has_nan = validation['scaling_quality'].get('has_nan', True)
                score = 0 if (has_inf or has_nan) else 1
                scores[method] = score
                
            elif criteria == 'normality':
                # Prefer methods that produce more normal distributions
                if 'std_close_to_one' in validation['scaling_quality']:
                    score = 1 if validation['scaling_quality']['std_close_to_one'] else 0
                    scores[method] = score
                else:
                    scores[method] = 0
                    
            elif criteria == 'stability':
                # Prefer methods with stable scaling
                mean_abs = validation['scaling_quality'].get('mean_abs', 1)
                score = 1 / (1 + mean_abs)  # Lower mean_abs is better
                scores[method] = score
        
        return max(scores, key=scores.get) if scores else self.methods[0]

def detect_outliers_robust(X: pd.DataFrame, method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers using robust methods.
    
    Args:
        X: Input DataFrame
        method: Outlier detection method ('iqr', 'zscore', 'modified_zscore')
        factor: Factor for outlier detection
        
    Returns:
        Boolean DataFrame indicating outliers
    """
    outlier_mask = pd.DataFrame(False, index=X.index, columns=X.columns)
    
    for col in X.columns:
        if method == 'iqr':
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            outlier_mask[col] = (X[col] < lower_bound) | (X[col] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(X[col].dropna()))
            outlier_mask[col] = z_scores > factor
            
        elif method == 'modified_zscore':
            median = X[col].median()
            mad = np.median(np.abs(X[col] - median))
            modified_z_scores = 0.6745 * (X[col] - median) / mad
            outlier_mask[col] = np.abs(modified_z_scores) > factor
    
    return outlier_mask

def robust_feature_selection(X: pd.DataFrame, y: pd.Series, 
                           method: str = 'variance_threshold', 
                           threshold: float = 0.01) -> List[str]:
    """
    Robust feature selection based on variance and stability.
    
    Args:
        X: Feature matrix
        y: Target variable
        method: Selection method ('variance_threshold', 'correlation_threshold')
        threshold: Threshold for selection
        
    Returns:
        List of selected feature names
    """
    if method == 'variance_threshold':
        # Remove low variance features
        variances = X.var()
        selected_features = variances[variances > threshold].index.tolist()
        
    elif method == 'correlation_threshold':
        # Remove features with low correlation to target
        correlations = X.corrwith(y).abs()
        selected_features = correlations[correlations > threshold].index.tolist()
        
    else:
        selected_features = X.columns.tolist()
    
    return selected_features