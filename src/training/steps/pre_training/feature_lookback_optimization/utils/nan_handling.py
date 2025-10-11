"""
Robust NaN handling utilities for feature lookback optimization.

This module provides safe NaN handling that avoids artificial correlations
and maintains data integrity during optimization.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
from dataclasses import dataclass

from .error_handling import safe_operation, DataValidationError


@dataclass
class AlignmentResult:
    """Result of array alignment with NaN handling."""
    feature_values: np.ndarray
    target_values: np.ndarray
    valid_indices: np.ndarray
    n_valid: int
    n_dropped: int


class SafeNaNHandler:
    """Safe NaN handling that preserves data integrity."""
    
    @staticmethod
    @safe_operation("array alignment with NaN handling", default_value=None)
    def align_arrays_safely(
        feature_values: np.ndarray, 
        target_values: np.ndarray,
        min_valid_samples: int = 10
    ) -> Optional[AlignmentResult]:
        """
        Align arrays safely by removing NaN values without creating artificial correlations.
        
        Args:
            feature_values: Feature array (may contain NaNs)
            target_values: Target array (may contain NaNs)
            min_valid_samples: Minimum number of valid samples required
            
        Returns:
            AlignmentResult with aligned arrays and metadata
            
        Raises:
            DataValidationError: If insufficient valid data
        """
        if len(feature_values) != len(target_values):
            raise DataValidationError(
                f"Array length mismatch: {len(feature_values)} vs {len(target_values)}"
            )
        
        # Create boolean mask for valid (non-NaN) values
        feature_valid = ~np.isnan(feature_values)
        target_valid = ~np.isnan(target_values)
        valid_mask = feature_valid & target_valid
        
        n_valid = np.sum(valid_mask)
        n_dropped = len(feature_values) - n_valid
        
        if n_valid < min_valid_samples:
            raise DataValidationError(
                f"Insufficient valid data: {n_valid} < {min_valid_samples} required "
                f"(dropped {n_dropped} NaN values)"
            )
        
        # Extract valid values
        aligned_features = feature_values[valid_mask]
        aligned_targets = target_values[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        return AlignmentResult(
            feature_values=aligned_features,
            target_values=aligned_targets,
            valid_indices=valid_indices,
            n_valid=n_valid,
            n_dropped=n_dropped
        )
    
    @staticmethod
    @safe_operation("feature array validation", default_value=False)
    def validate_feature_array(
        feature_values: np.ndarray,
        feature_name: str,
        min_variance: float = 1e-8
    ) -> bool:
        """
        Validate feature array for optimization suitability.
        
        Args:
            feature_values: Feature array to validate
            feature_name: Name of feature for error messages
            min_variance: Minimum variance threshold
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            DataValidationError: If feature is invalid
        """
        if len(feature_values) == 0:
            raise DataValidationError(f"Empty feature array for {feature_name}")
        
        # Check for all NaN values
        if np.all(np.isnan(feature_values)):
            raise DataValidationError(f"All NaN values in feature {feature_name}")
        
        # Check for constant values (no variance)
        valid_values = feature_values[~np.isnan(feature_values)]
        if len(valid_values) > 0:
            variance = np.var(valid_values)
            if variance < min_variance:
                raise DataValidationError(
                    f"Feature {feature_name} has insufficient variance: {variance:.2e} < {min_variance:.2e}"
                )
        
        return True
    
    @staticmethod
    @safe_operation("safe array creation", default_value=None)
    def create_safe_array(
        data: Union[pd.Series, np.ndarray, list],
        length: int,
        fill_value: float = np.nan
    ) -> np.ndarray:
        """
        Create a safe numpy array with proper NaN handling.
        
        Args:
            data: Input data
            length: Expected length
            fill_value: Value to use for missing data (default: NaN)
            
        Returns:
            Safe numpy array
        """
        if isinstance(data, pd.Series):
            values = data.values
        elif isinstance(data, np.ndarray):
            values = data
        else:
            values = np.array(data)
        
        # Ensure correct length
        if len(values) < length:
            # Pad with fill_value
            padded = np.full(length, fill_value)
            padded[:len(values)] = values
            return padded
        elif len(values) > length:
            # Truncate
            return values[:length]
        else:
            return values
    
    @staticmethod
    def get_nan_statistics(feature_values: np.ndarray, target_values: np.ndarray) -> dict:
        """Get detailed NaN statistics for debugging."""
        feature_nans = np.sum(np.isnan(feature_values))
        target_nans = np.sum(np.isnan(target_values))
        both_nans = np.sum(np.isnan(feature_values) & np.isnan(target_values))
        either_nans = np.sum(np.isnan(feature_values) | np.isnan(target_values))
        
        return {
            'total_samples': len(feature_values),
            'feature_nans': feature_nans,
            'target_nans': target_nans,
            'both_nans': both_nans,
            'either_nans': either_nans,
            'valid_samples': len(feature_values) - either_nans,
            'feature_nan_rate': feature_nans / len(feature_values),
            'target_nan_rate': target_nans / len(feature_values),
            'valid_rate': (len(feature_values) - either_nans) / len(feature_values)
        }


def safe_correlation_with_nan_handling(
    feature_values: np.ndarray, 
    target_values: np.ndarray,
    method: str = 'pearson',
    min_samples: int = 10
) -> float:
    """
    Calculate correlation with proper NaN handling.
    
    Args:
        feature_values: Feature array
        target_values: Target array  
        method: Correlation method ('pearson', 'spearman')
        min_samples: Minimum samples required
        
    Returns:
        Correlation coefficient (0.0 if insufficient data)
    """
    try:
        handler = SafeNaNHandler()
        alignment = handler.align_arrays_safely(feature_values, target_values, min_samples)
        
        if method == 'pearson':
            return np.corrcoef(alignment.feature_values, alignment.target_values)[0, 1]
        elif method == 'spearman':
            from scipy.stats import spearmanr
            return spearmanr(alignment.feature_values, alignment.target_values)[0]
        else:
            raise ValueError(f"Unknown correlation method: {method}")
            
    except (DataValidationError, ValueError) as e:
        # Log warning but don't fail - return 0.0 for insufficient data
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Correlation calculation failed: {e}")
        return 0.0
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Unexpected error in correlation calculation: {e}")
        return 0.0


def safe_mutual_information_with_nan_handling(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    n_bins: int = 10,
    min_samples: int = 20
) -> float:
    """
    Calculate mutual information with proper NaN handling.
    
    Args:
        feature_values: Feature array
        target_values: Target array
        n_bins: Number of bins for discretization
        min_samples: Minimum samples required
        
    Returns:
        Mutual information value (0.0 if insufficient data)
    """
    try:
        handler = SafeNaNHandler()
        alignment = handler.align_arrays_safely(feature_values, target_values, min_samples)
        
        # Use sklearn for robust MI calculation
        from sklearn.feature_selection import mutual_info_regression
        return mutual_info_regression(
            alignment.feature_values.reshape(-1, 1), 
            alignment.target_values,
            discrete_features=False,
            random_state=42
        )[0]
        
    except (DataValidationError, ImportError) as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Mutual information calculation failed: {e}")
        return 0.0
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Unexpected error in MI calculation: {e}")
        return 0.0