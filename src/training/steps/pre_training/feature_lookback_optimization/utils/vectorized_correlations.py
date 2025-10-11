"""
Vectorized correlation calculations for feature lookback optimization.

This module provides highly optimized, vectorized correlation calculations
that significantly improve performance over loop-based approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

from .error_handling import safe_operation, DataValidationError
from .nan_handling import SafeNaNHandler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class CorrelationResult:
    """Result of vectorized correlation calculation."""
    pearson: float
    spearman: float
    mutual_info: float
    r_squared: float
    n_valid: int
    computation_time: float


class VectorizedCorrelationCalculator:
    """High-performance vectorized correlation calculator."""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.nan_handler = SafeNaNHandler()
        self._gpu_available = self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            import cupy as cp
            return True
        except ImportError:
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                return False
    
    @safe_operation("vectorized correlation calculation", default_value=None)
    def calculate_comprehensive_correlations_vectorized(
        self, 
        feature_values: np.ndarray, 
        target_values: np.ndarray,
        min_samples: int = 10
    ) -> Optional[CorrelationResult]:
        """
        Calculate all correlation metrics using vectorized operations.
        
        Args:
            feature_values: Feature array
            target_values: Target array
            min_samples: Minimum valid samples required
            
        Returns:
            CorrelationResult with all metrics
        """
        import time
        start_time = time.time()
        
        # Align arrays safely
        try:
            alignment = self.nan_handler.align_arrays_safely(
                feature_values, target_values, min_samples
            )
            x = alignment.feature_values
            y = alignment.target_values
            n_valid = alignment.n_valid
        except DataValidationError:
            return None
        
        # Calculate all correlations vectorized
        pearson = self._calculate_pearson_vectorized(x, y)
        spearman = self._calculate_spearman_vectorized(x, y)
        mutual_info = self._calculate_mutual_info_vectorized(x, y)
        r_squared = pearson ** 2
        
        computation_time = time.time() - start_time
        
        return CorrelationResult(
            pearson=pearson,
            spearman=spearman,
            mutual_info=mutual_info,
            r_squared=r_squared,
            n_valid=n_valid,
            computation_time=computation_time
        )
    
    def _calculate_pearson_vectorized(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Pearson correlation using vectorized operations."""
        try:
            # Center the data
            x_centered = x - np.mean(x)
            y_centered = y - np.mean(y)
            
            # Calculate correlation coefficient
            numerator = np.sum(x_centered * y_centered)
            denominator = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
            
            if denominator == 0:
                return 0.0
            
            correlation = numerator / denominator
            return float(np.clip(correlation, -1.0, 1.0))
            
        except (ValueError, ZeroDivisionError, OverflowError):
            return 0.0
    
    def _calculate_spearman_vectorized(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Spearman correlation using vectorized operations."""
        try:
            # Convert to ranks
            x_ranks = self._rank_vectorized(x)
            y_ranks = self._rank_vectorized(y)
            
            # Use Pearson correlation on ranks
            return self._calculate_pearson_vectorized(x_ranks, y_ranks)
            
        except (ValueError, ZeroDivisionError, OverflowError):
            return 0.0
    
    def _rank_vectorized(self, data: np.ndarray) -> np.ndarray:
        """Calculate ranks using vectorized operations."""
        # Handle ties by using average rank
        sorted_indices = np.argsort(data)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(data))
        
        # Handle ties
        unique_values, inverse_indices = np.unique(data, return_inverse=True)
        for value in unique_values:
            mask = data == value
            if np.sum(mask) > 1:  # Ties exist
                ranks[mask] = np.mean(ranks[mask])
        
        return ranks.astype(float)
    
    def _calculate_mutual_info_vectorized(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate mutual information using vectorized operations."""
        try:
            # Use sklearn for robust MI calculation
            from sklearn.feature_selection import mutual_info_regression
            return float(mutual_info_regression(
                x.reshape(-1, 1), 
                y,
                discrete_features=False,
                random_state=42
            )[0])
        except ImportError:
            # Fallback to manual calculation
            return self._calculate_mi_manual_vectorized(x, y)
        except Exception:
            return 0.0
    
    def _calculate_mi_manual_vectorized(self, x: np.ndarray, y: np.ndarray) -> float:
        """Manual vectorized mutual information calculation."""
        try:
            # Discretize data into bins
            n_bins = min(20, int(np.sqrt(len(x))))
            if n_bins < 2:
                return 0.0
            
            # Create bins
            x_bins = np.digitize(x, np.linspace(np.min(x), np.max(x), n_bins))
            y_bins = np.digitize(y, np.linspace(np.min(y), np.max(y), n_bins))
            
            # Calculate joint histogram
            joint_hist, _, _ = np.histogram2d(x_bins, y_bins, bins=n_bins)
            joint_prob = joint_hist / np.sum(joint_hist)
            
            # Calculate marginal probabilities
            x_prob = np.sum(joint_prob, axis=1)
            y_prob = np.sum(joint_prob, axis=0)
            
            # Calculate MI using vectorized operations
            # MI = sum(p(x,y) * log(p(x,y) / (p(x) * p(y))))
            log_ratio = np.log2(joint_prob / (x_prob[:, np.newaxis] * y_prob[np.newaxis, :] + 1e-10))
            mi = np.sum(joint_prob * log_ratio)
            
            return float(np.maximum(0.0, mi))
            
        except (ValueError, ZeroDivisionError, OverflowError):
            return 0.0
    
    @safe_operation("batch correlation calculation", default_value=[])
    def calculate_batch_correlations(
        self, 
        feature_matrix: np.ndarray, 
        target_values: np.ndarray,
        min_samples: int = 10
    ) -> List[CorrelationResult]:
        """
        Calculate correlations for multiple features in batch.
        
        Args:
            feature_matrix: Matrix of features (n_samples, n_features)
            target_values: Target values (n_samples,)
            min_samples: Minimum valid samples required
            
        Returns:
            List of CorrelationResult for each feature
        """
        if feature_matrix.shape[0] != len(target_values):
            raise DataValidationError(
                f"Feature matrix and target length mismatch: {feature_matrix.shape[0]} vs {len(target_values)}"
            )
        
        results = []
        n_features = feature_matrix.shape[1]
        
        # Process features in batches for memory efficiency
        batch_size = min(100, n_features)
        
        for i in range(0, n_features, batch_size):
            end_idx = min(i + batch_size, n_features)
            batch_features = feature_matrix[:, i:end_idx]
            
            for j in range(batch_features.shape[1]):
                feature_idx = i + j
                feature_values = batch_features[:, j]
                
                result = self.calculate_comprehensive_correlations_vectorized(
                    feature_values, target_values, min_samples
                )
                
                if result is not None:
                    results.append(result)
                else:
                    # Create empty result for failed calculation
                    results.append(CorrelationResult(
                        pearson=0.0, spearman=0.0, mutual_info=0.0, 
                        r_squared=0.0, n_valid=0, computation_time=0.0
                    ))
        
        return results
    
    def calculate_correlation_matrix_vectorized(
        self, 
        data: np.ndarray,
        method: str = 'pearson'
    ) -> np.ndarray:
        """
        Calculate correlation matrix using vectorized operations.
        
        Args:
            data: Data matrix (n_samples, n_features)
            method: Correlation method ('pearson', 'spearman')
            
        Returns:
            Correlation matrix (n_features, n_features)
        """
        n_features = data.shape[1]
        corr_matrix = np.eye(n_features)  # Initialize with identity matrix
        
        if method == 'pearson':
            # Use numpy's optimized correlation function
            corr_matrix = np.corrcoef(data.T)
        elif method == 'spearman':
            # Calculate Spearman correlation matrix
            ranks = np.apply_along_axis(self._rank_vectorized, 0, data)
            corr_matrix = np.corrcoef(ranks.T)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Handle NaN values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return corr_matrix
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'gpu_available': self._gpu_available,
            'use_gpu': self.use_gpu and self._gpu_available,
            'nan_handler_stats': self.nan_handler.get_nan_statistics(
                np.array([]), np.array([])
            ) if hasattr(self.nan_handler, 'get_nan_statistics') else {}
        }


# Convenience functions for easy integration
def calculate_correlations_vectorized(
    feature_values: np.ndarray, 
    target_values: np.ndarray,
    min_samples: int = 10
) -> Dict[str, float]:
    """
    Calculate correlations using vectorized operations.
    
    Args:
        feature_values: Feature array
        target_values: Target array
        min_samples: Minimum valid samples required
        
    Returns:
        Dictionary with correlation metrics
    """
    calculator = VectorizedCorrelationCalculator()
    result = calculator.calculate_comprehensive_correlations_vectorized(
        feature_values, target_values, min_samples
    )
    
    if result is None:
        return {'pearson': 0.0, 'spearman': 0.0, 'mutual_info': 0.0, 'r_squared': 0.0}
    
    return {
        'pearson': result.pearson,
        'spearman': result.spearman,
        'mutual_info': result.mutual_info,
        'r_squared': result.r_squared
    }


def calculate_batch_correlations_vectorized(
    feature_matrix: np.ndarray, 
    target_values: np.ndarray,
    min_samples: int = 10
) -> List[Dict[str, float]]:
    """
    Calculate correlations for multiple features in batch.
    
    Args:
        feature_matrix: Matrix of features (n_samples, n_features)
        target_values: Target values (n_samples,)
        min_samples: Minimum valid samples required
        
    Returns:
        List of correlation dictionaries for each feature
    """
    calculator = VectorizedCorrelationCalculator()
    results = calculator.calculate_batch_correlations(
        feature_matrix, target_values, min_samples
    )
    
    return [
        {
            'pearson': r.pearson,
            'spearman': r.spearman,
            'mutual_info': r.mutual_info,
            'r_squared': r.r_squared
        }
        for r in results
    ]