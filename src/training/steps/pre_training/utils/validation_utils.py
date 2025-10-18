"""
Validation utilities for pre-training steps.

This module provides comprehensive validation utilities for data quality,
feature validation, and model preparation validation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataQualityValidator:
    """Comprehensive data quality validator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the data quality validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def validate_data_quality(self, df: pd.DataFrame,
                            required_columns: Optional[List[str]] = None,
                            min_rows: int = 100,
                            max_null_ratio: float = 0.5) -> Dict[str, Any]:
        """Validate comprehensive data quality.
        
        Args:
            df: Input DataFrame
            required_columns: List of required columns
            min_rows: Minimum number of rows required
            max_null_ratio: Maximum allowed null ratio
            
        Returns:
            Dictionary containing validation results
        """
        results = {
            'valid': True,
            'issues': [],
            'quality_score': 0.0,
            'recommendations': []
        }
        
        # Basic validation
        if df is None or df.empty:
            results['valid'] = False
            results['issues'].append("DataFrame is None or empty")
            return results
        
        # Row count validation
        if len(df) < min_rows:
            results['valid'] = False
            results['issues'].append(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
        
        # Column validation
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                results['valid'] = False
                results['issues'].append(f"Missing required columns: {missing_cols}")
        
        # Null value validation
        null_ratios = df.isnull().sum() / len(df)
        high_null_cols = null_ratios[null_ratios > max_null_ratio].index.tolist()
        if high_null_cols:
            results['issues'].append(f"Columns with high null ratio: {high_null_cols}")
            results['recommendations'].append("Consider dropping or imputing high null columns")
        
        # Data type validation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            results['issues'].append("No numeric columns found")
            results['recommendations'].append("Ensure data contains numeric features")
        
        # Calculate quality score
        results['quality_score'] = self._calculate_quality_score(df, results)
        
        return results
    
    def _calculate_quality_score(self, df: pd.DataFrame, results: Dict[str, Any]) -> float:
        """Calculate data quality score."""
        score = 1.0
        
        # Penalize for issues
        score -= len(results['issues']) * 0.1
        
        # Penalize for high null ratios
        null_ratios = df.isnull().sum() / len(df)
        avg_null_ratio = null_ratios.mean()
        score -= avg_null_ratio * 0.5
        
        # Bonus for good data characteristics
        if len(df) > 1000:
            score += 0.1
        if len(df.select_dtypes(include=[np.number]).columns) > 5:
            score += 0.1
        
        return max(0.0, min(1.0, score))

class FeatureValidator:
    """Feature-specific validation utilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the feature validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def validate_features(self, df: pd.DataFrame,
                         feature_columns: List[str],
                         target_column: str) -> Dict[str, Any]:
        """Validate feature data quality and relationships.
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature column names
            target_column: Name of target column
            
        Returns:
            Dictionary containing validation results
        """
        results = {
            'valid': True,
            'issues': [],
            'feature_stats': {},
            'correlations': {}
        }
        
        # Check for missing features
        missing_features = set(feature_columns) - set(df.columns)
        if missing_features:
            results['valid'] = False
            results['issues'].append(f"Missing features: {missing_features}")
        
        # Check for target column
        if target_column not in df.columns:
            results['valid'] = False
            results['issues'].append(f"Missing target column: {target_column}")
        
        # Validate each feature
        for col in feature_columns:
            if col in df.columns:
                feature_stats = self._analyze_feature(df[col])
                results['feature_stats'][col] = feature_stats
                
                # Check for problematic values
                if feature_stats['infinite_count'] > 0:
                    results['issues'].append(f"Column {col} has {feature_stats['infinite_count']} infinite values")
                
                if feature_stats['null_count'] > len(df) * 0.5:
                    results['issues'].append(f"Column {col} has {feature_stats['null_count']} null values")
        
        # Calculate correlations with target
        if target_column in df.columns:
            for col in feature_columns:
                if col in df.columns:
                    try:
                        corr = df[col].corr(df[target_column])
                        results['correlations'][col] = corr
                    except Exception as e:
                        results['issues'].append(f"Could not calculate correlation for {col}: {e}")
        
        return results
    
    def _analyze_feature(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze a single feature series."""
        return {
            'null_count': series.isnull().sum(),
            'infinite_count': np.isinf(series).sum(),
            'unique_count': series.nunique(),
            'mean': series.mean() if series.dtype in ['float64', 'int64'] else None,
            'std': series.std() if series.dtype in ['float64', 'int64'] else None,
            'min': series.min() if series.dtype in ['float64', 'int64'] else None,
            'max': series.max() if series.dtype in ['float64', 'int64'] else None
        }

class ModelPreparationValidator:
    """Validator for model preparation steps."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the model preparation validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def validate_model_data(self, X: pd.DataFrame,
                           y: pd.Series,
                           test_size: float = 0.2) -> Dict[str, Any]:
        """Validate data for model training.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Test set size ratio
            
        Returns:
            Dictionary containing validation results
        """
        results = {
            'valid': True,
            'issues': [],
            'data_info': {},
            'recommendations': []
        }
        
        # Check data shapes
        if len(X) != len(y):
            results['valid'] = False
            results['issues'].append(f"Feature matrix length ({len(X)}) != target length ({len(y)})")
        
        # Check for sufficient data
        min_samples = int(1 / test_size) * 10  # At least 10 samples per fold
        if len(X) < min_samples:
            results['valid'] = False
            results['issues'].append(f"Insufficient data: {len(X)} samples, minimum required: {min_samples}")
        
        # Check for missing values
        if X.isnull().any().any():
            results['issues'].append("Feature matrix contains missing values")
            results['recommendations'].append("Consider imputing missing values")
        
        if y.isnull().any():
            results['issues'].append("Target vector contains missing values")
            results['recommendations'].append("Remove or impute missing target values")
        
        # Check for constant features
        constant_features = X.columns[X.nunique() <= 1].tolist()
        if constant_features:
            results['issues'].append(f"Constant features found: {constant_features}")
            results['recommendations'].append("Remove constant features")
        
        # Store data information
        results['data_info'] = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_names': list(X.columns),
            'target_distribution': y.value_counts().to_dict() if hasattr(y, 'value_counts') else None
        }
        
        return results

# Convenience functions
def validate_training_data_comprehensive(df: pd.DataFrame,
                                       feature_columns: List[str],
                                       target_column: str,
                                       config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Comprehensive validation of training data.
    
    Args:
        df: Input DataFrame
        feature_columns: List of feature columns
        target_column: Name of target column
        config: Configuration dictionary
        
    Returns:
        Dictionary containing comprehensive validation results
    """
    validator = DataQualityValidator(config)
    feature_validator = FeatureValidator(config)
    
    # Run all validations
    data_quality = validator.validate_data_quality(df, feature_columns + [target_column])
    feature_validation = feature_validator.validate_features(df, feature_columns, target_column)
    
    return {
        'data_quality': data_quality,
        'feature_validation': feature_validation,
        'overall_valid': data_quality['valid'] and feature_validation['valid'],
        'timestamp': datetime.now()
    }

def validate_model_preparation(X: pd.DataFrame,
                             y: pd.Series,
                             config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate model preparation data.
    
    Args:
        X: Feature matrix
        y: Target vector
        config: Configuration dictionary
        
    Returns:
        Dictionary containing validation results
    """
    validator = ModelPreparationValidator(config)
    return validator.validate_model_data(X, y)
