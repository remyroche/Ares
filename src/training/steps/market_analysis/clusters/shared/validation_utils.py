"""
Comprehensive validation framework for clustering operations.

This module centralizes all validation logic to eliminate duplication
across the clustering codebase and provide consistent validation patterns.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils.math_validation import validate_finite, validate_positive, validate_range
from src.utils.tprint import tprint, tprint_warning, tprint_error


@dataclass
class ValidationResult:
    """Standardized validation result."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0
    
    def get_summary(self) -> str:
        """Get a summary of validation results."""
        if self.is_valid:
            if self.warnings:
                return f"✅ Valid with {len(self.warnings)} warnings"
            else:
                return "✅ Valid"
        else:
            return f"❌ Invalid: {len(self.errors)} errors, {len(self.warnings)} warnings"


class ClusteringValidationUtils:
    """Centralized validation utilities for clustering operations."""
    
    @staticmethod
    def validate_features(features: np.ndarray, 
                         feature_names: Optional[List[str]] = None,
                         min_samples: int = 10,
                         min_features: int = 2,
                         max_nan_ratio: float = 0.1) -> ValidationResult:
        """
        Comprehensive feature validation.
        
        Args:
            features: Feature matrix to validate
            feature_names: Optional feature names
            min_samples: Minimum number of samples required
            min_features: Minimum number of features required
            max_nan_ratio: Maximum allowed ratio of NaN values
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        metadata = {}
        
        try:
            # Basic shape validation
            if features is None or features.size == 0:
                errors.append("Features are empty or None")
                return ValidationResult(False, errors, warnings, metadata)
            
            if features.shape[0] < min_samples:
                warnings.append(f"Very few samples for clustering: {features.shape[0]} < {min_samples}")
            
            if features.shape[1] < min_features:
                errors.append(f"Insufficient features for clustering: {features.shape[1]} < {min_features}")
            
            # Finite value validation
            if not validate_finite(features, "features"):
                errors.append("Features contain non-finite values")
            
            # NaN and infinite value checks
            nan_count = np.isnan(features).sum()
            inf_count = np.isinf(features).sum()
            total_values = features.size
            
            nan_ratio = nan_count / total_values if total_values > 0 else 0
            inf_ratio = inf_count / total_values if total_values > 0 else 0
            
            if nan_ratio > max_nan_ratio:
                errors.append(f"Too many NaN values: {nan_ratio:.2%} > {max_nan_ratio:.2%}")
            elif nan_count > 0:
                warnings.append(f"Features contain {nan_count} NaN values ({nan_ratio:.2%})")
            
            if inf_count > 0:
                errors.append(f"Features contain {inf_count} infinite values ({inf_ratio:.2%})")
            
            # Feature name validation
            if feature_names and len(feature_names) != features.shape[1]:
                warnings.append(f"Feature names length mismatch: {len(feature_names)} vs {features.shape[1]}")
            
            # Variance validation
            feature_vars = np.var(features, axis=0)
            zero_var_features = np.sum(feature_vars == 0)
            if zero_var_features > 0:
                warnings.append(f"{zero_var_features} features have zero variance")
            
            # Range validation
            feature_mins = np.min(features, axis=0)
            feature_maxs = np.max(features, axis=0)
            feature_ranges = feature_maxs - feature_mins
            
            metadata = {
                'shape': features.shape,
                'nan_count': nan_count,
                'inf_count': inf_count,
                'nan_ratio': nan_ratio,
                'inf_ratio': inf_ratio,
                'feature_count': features.shape[1],
                'sample_count': features.shape[0],
                'zero_variance_features': zero_var_features,
                'feature_variance_stats': {
                    'min': float(np.min(feature_vars)),
                    'max': float(np.max(feature_vars)),
                    'mean': float(np.mean(feature_vars)),
                    'std': float(np.std(feature_vars))
                },
                'feature_range_stats': {
                    'min': float(np.min(feature_ranges)),
                    'max': float(np.max(feature_ranges)),
                    'mean': float(np.mean(feature_ranges))
                }
            }
            
            return ValidationResult(len(errors) == 0, errors, warnings, metadata)
            
        except Exception as e:
            return ValidationResult(False, [f"Feature validation error: {e}"], [], {})
    
    @staticmethod
    def validate_clustering_assignments(assignments: np.ndarray,
                                      expected_length: int,
                                      min_clusters: int = 2,
                                      max_clusters: Optional[int] = None) -> ValidationResult:
        """
        Validate clustering assignments.
        
        Args:
            assignments: Cluster assignments array
            expected_length: Expected length of assignments
            min_clusters: Minimum number of clusters required
            max_clusters: Maximum number of clusters allowed
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        metadata = {}
        
        try:
            if assignments is None:
                errors.append("Assignments are None")
                return ValidationResult(False, errors, warnings, metadata)
            
            if len(assignments) != expected_length:
                errors.append(f"Assignment length mismatch: {len(assignments)} vs {expected_length}")
            
            if not np.issubdtype(assignments.dtype, np.integer):
                errors.append("Assignments must be integer type")
            
            unique_clusters = np.unique(assignments)
            n_clusters = len(unique_clusters)
            
            if n_clusters < min_clusters:
                errors.append(f"Too few clusters: {n_clusters} < {min_clusters}")
            
            if max_clusters and n_clusters > max_clusters:
                warnings.append(f"Many clusters: {n_clusters} > {max_clusters}")
            
            cluster_sizes = np.bincount(assignments)
            min_size = cluster_sizes.min()
            max_size = cluster_sizes.max()
            
            if min_size == 0:
                warnings.append("Empty clusters detected")
            
            # Balance validation
            balance_ratio = max_size / min_size if min_size > 0 else float('inf')
            if balance_ratio > 10:
                warnings.append(f"Unbalanced clusters: max/min ratio = {balance_ratio:.1f}")
            
            metadata = {
                'n_clusters': n_clusters,
                'min_cluster_size': int(min_size),
                'max_cluster_size': int(max_size),
                'cluster_balance_ratio': float(balance_ratio),
                'cluster_sizes': cluster_sizes.tolist(),
                'unique_clusters': unique_clusters.tolist()
            }
            
            return ValidationResult(len(errors) == 0, errors, warnings, metadata)
            
        except Exception as e:
            return ValidationResult(False, [f"Assignment validation error: {e}"], [], {})
    
    @staticmethod
    def validate_market_data(market_data: pd.DataFrame,
                           required_columns: List[str] = None,
                           min_rows: int = 1) -> ValidationResult:
        """
        Validate market data structure.
        
        Args:
            market_data: Market data DataFrame
            required_columns: List of required column names
            min_rows: Minimum number of rows required
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        metadata = {}
        
        try:
            if market_data is None or market_data.empty:
                if min_rows > 0:
                    errors.append(f"Market data is empty, need at least {min_rows} rows")
                else:
                    warnings.append("Market data is empty")
                return ValidationResult(len(errors) == 0, errors, warnings, metadata)
            
            if len(market_data) < min_rows:
                errors.append(f"Insufficient rows: {len(market_data)} < {min_rows}")
            
            # Required columns check
            if required_columns:
                missing_columns = set(required_columns) - set(market_data.columns)
                if missing_columns:
                    errors.append(f"Missing required columns: {list(missing_columns)}")
            
            # Data quality checks
            null_counts = market_data.isnull().sum()
            high_null_columns = null_counts[null_counts > len(market_data) * 0.1]
            
            if len(high_null_columns) > 0:
                warnings.append(f"High null percentage in columns: {high_null_columns.to_dict()}")
            
            # Duplicate rows check
            duplicate_rows = market_data.duplicated().sum()
            if duplicate_rows > 0:
                warnings.append(f"{duplicate_rows} duplicate rows found")
            
            metadata = {
                'shape': market_data.shape,
                'columns': list(market_data.columns),
                'null_counts': null_counts.to_dict(),
                'dtypes': market_data.dtypes.to_dict(),
                'duplicate_rows': int(duplicate_rows),
                'high_null_columns': high_null_columns.to_dict()
            }
            
            return ValidationResult(len(errors) == 0, errors, warnings, metadata)
            
        except Exception as e:
            return ValidationResult(False, [f"Market data validation error: {e}"], [], {})
    
    @staticmethod
    def validate_clustering_config(config: Any, 
                                 required_attrs: List[str] = None) -> ValidationResult:
        """
        Validate clustering configuration object.
        
        Args:
            config: Configuration object to validate
            required_attrs: List of required attribute names
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        metadata = {}
        
        try:
            if config is None:
                errors.append("Configuration is None")
                return ValidationResult(False, errors, warnings, metadata)
            
            if required_attrs:
                missing_attrs = []
                for attr in required_attrs:
                    if not hasattr(config, attr):
                        missing_attrs.append(attr)
                
                if missing_attrs:
                    errors.append(f"Missing required config attributes: {missing_attrs}")
            
            # Validate numeric attributes
            numeric_attrs = ['n_clusters', 'max_iter', 'random_state', 'n_components']
            for attr in numeric_attrs:
                if hasattr(config, attr):
                    value = getattr(config, attr)
                    if value is not None:
                        if not validate_finite(value, attr):
                            errors.append(f"Invalid {attr}: {value}")
                        elif attr in ['n_clusters', 'max_iter', 'n_components'] and value <= 0:
                            errors.append(f"{attr} must be positive: {value}")
            
            metadata = {
                'config_type': type(config).__name__,
                'available_attrs': [attr for attr in dir(config) if not attr.startswith('_')],
                'required_attrs': required_attrs or []
            }
            
            return ValidationResult(len(errors) == 0, errors, warnings, metadata)
            
        except Exception as e:
            return ValidationResult(False, [f"Config validation error: {e}"], [], {})
    
    @staticmethod
    def safe_validate_with_logging(validation_func, *args, **kwargs) -> ValidationResult:
        """
        Safely execute validation with proper error logging.
        
        Args:
            validation_func: Validation function to execute
            *args: Positional arguments for validation function
            **kwargs: Keyword arguments for validation function
            
        Returns:
            ValidationResult with validation status and details
        """
        try:
            result = validation_func(*args, **kwargs)
            
            if not result.is_valid:
                for error in result.errors:
                    tprint_error(f"Validation error: {error}")
            
            for warning in result.warnings:
                tprint_warning(f"Validation warning: {warning}")
            
            return result
            
        except Exception as e:
            tprint_error(f"Validation function failed: {e}")
            return ValidationResult(False, [f"Validation function error: {e}"], [], {})
    
    @staticmethod
    def validate_memory_usage(data: Union[np.ndarray, pd.DataFrame], 
                            max_memory_mb: float = 1000.0) -> ValidationResult:
        """
        Validate memory usage of data structures.
        
        Args:
            data: Data structure to validate
            max_memory_mb: Maximum allowed memory usage in MB
            
        Returns:
            ValidationResult with memory validation status
        """
        errors = []
        warnings = []
        metadata = {}
        
        try:
            if data is None:
                return ValidationResult(True, errors, warnings, metadata)
            
            # Calculate memory usage
            if isinstance(data, np.ndarray):
                memory_bytes = data.nbytes
            elif isinstance(data, pd.DataFrame):
                memory_bytes = data.memory_usage(deep=True).sum()
            else:
                warnings.append(f"Unknown data type for memory validation: {type(data)}")
                return ValidationResult(True, errors, warnings, metadata)
            
            memory_mb = memory_bytes / (1024 * 1024)
            
            if memory_mb > max_memory_mb:
                errors.append(f"Memory usage too high: {memory_mb:.1f}MB > {max_memory_mb}MB")
            elif memory_mb > max_memory_mb * 0.8:
                warnings.append(f"High memory usage: {memory_mb:.1f}MB")
            
            metadata = {
                'memory_bytes': memory_bytes,
                'memory_mb': memory_mb,
                'max_memory_mb': max_memory_mb,
                'memory_ratio': memory_mb / max_memory_mb
            }
            
            return ValidationResult(len(errors) == 0, errors, warnings, metadata)
            
        except Exception as e:
            return ValidationResult(False, [f"Memory validation error: {e}"], [], {})


# Convenience functions for common validation patterns
def validate_features_safe(features: np.ndarray, **kwargs) -> bool:
    """Safe feature validation returning boolean."""
    result = ClusteringValidationUtils.validate_features(features, **kwargs)
    return result.is_valid


def validate_assignments_safe(assignments: np.ndarray, expected_length: int, **kwargs) -> bool:
    """Safe assignment validation returning boolean."""
    result = ClusteringValidationUtils.validate_clustering_assignments(assignments, expected_length, **kwargs)
    return result.is_valid


def validate_market_data_safe(market_data: pd.DataFrame, **kwargs) -> bool:
    """Safe market data validation returning boolean."""
    result = ClusteringValidationUtils.validate_market_data(market_data, **kwargs)
    return result.is_valid