"""
Base Matrix Operations - Core Functions Without Dependencies

This module provides basic matrix operations that don't have circular dependencies.
These are the fundamental matrix utilities that other modules can safely import.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
# Optional imports with fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


def safe_correlation_matrix(data: Union['np.ndarray', 'pd.DataFrame'], method: str = 'pearson') -> 'np.ndarray':
    """
    Safely compute correlation matrix with error handling.
    
    Args:
        data: Input data (numpy array or DataFrame)
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Correlation matrix or identity matrix if computation fails
    """
    try:
        if not NUMPY_AVAILABLE:
            return None
        
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            corr_matrix = data.corr(method=method)
            return corr_matrix.values
        else:
            # For numpy arrays, use numpy's corrcoef
            if method == 'pearson':
                return np.corrcoef(data.T)
            else:
                # For other methods, convert to DataFrame first
                if PANDAS_AVAILABLE:
                    df = pd.DataFrame(data)
                    corr_matrix = df.corr(method=method)
                    return corr_matrix.values
                else:
                    return np.corrcoef(data.T)
    except Exception:
        # Return identity matrix as fallback
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            n = len(data.columns)
        else:
            n = data.shape[1] if len(data.shape) > 1 else 1
        return np.eye(n)


def safe_matrix_rank(matrix: 'np.ndarray', tol: Optional[float] = None) -> int:
    """
    Safely compute matrix rank.
    
    Args:
        matrix: Input matrix
        tol: Tolerance for rank computation
        
    Returns:
        Matrix rank or 0 if computation fails
    """
    try:
        if not NUMPY_AVAILABLE:
            return 0
        return np.linalg.matrix_rank(matrix, tol=tol)
    except Exception:
        return 0


def safe_condition_number(matrix: 'np.ndarray') -> float:
    """
    Safely compute condition number.
    
    Args:
        matrix: Input matrix
        
    Returns:
        Condition number or 0 if computation fails
    """
    try:
        if not NUMPY_AVAILABLE:
            return 0.0
        return np.linalg.cond(matrix)
    except Exception:
        return 0.0


def matrix_normalize(matrix: 'np.ndarray', method: str = 'zscore') -> 'np.ndarray':
    """
    Normalize matrix using specified method.
    
    Args:
        matrix: Input matrix
        method: Normalization method ('zscore', 'minmax', 'robust')
        
    Returns:
        Normalized matrix
    """
    try:
        if not NUMPY_AVAILABLE:
            return matrix
        
        if method == 'zscore':
            mean = np.mean(matrix, axis=0)
            std = np.std(matrix, axis=0)
            std = np.where(std == 0, 1, std)  # Avoid division by zero
            return (matrix - mean) / std
        elif method == 'minmax':
            min_val = np.min(matrix, axis=0)
            max_val = np.max(matrix, axis=0)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)  # Avoid division by zero
            return (matrix - min_val) / range_val
        elif method == 'robust':
            median = np.median(matrix, axis=0)
            mad = np.median(np.abs(matrix - median), axis=0)
            mad = np.where(mad == 0, 1, mad)  # Avoid division by zero
            return (matrix - median) / mad
        else:
            return matrix
    except Exception:
        return matrix


def extract_matrix_features(matrix: 'np.ndarray') -> Dict[str, float]:
    """
    Extract basic features from a matrix.
    
    Args:
        matrix: Input matrix
        
    Returns:
        Dictionary of matrix features
    """
    try:
        if not NUMPY_AVAILABLE:
            return {
                'shape': (0, 0),
                'rank': 0,
                'condition_number': 0.0,
                'determinant': 0.0,
                'trace': 0.0,
                'frobenius_norm': 0.0,
                'spectral_norm': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        features = {
            'shape': matrix.shape,
            'rank': safe_matrix_rank(matrix),
            'condition_number': safe_condition_number(matrix),
            'determinant': np.linalg.det(matrix) if matrix.shape[0] == matrix.shape[1] else 0.0,
            'trace': np.trace(matrix) if matrix.shape[0] == matrix.shape[1] else 0.0,
            'frobenius_norm': np.linalg.norm(matrix, 'fro'),
            'spectral_norm': np.linalg.norm(matrix, 2),
            'mean': np.mean(matrix),
            'std': np.std(matrix),
            'min': np.min(matrix),
            'max': np.max(matrix)
        }
        return features
    except Exception:
        return {
            'shape': (0, 0),
            'rank': 0,
            'condition_number': 0.0,
            'determinant': 0.0,
            'trace': 0.0,
            'frobenius_norm': 0.0,
            'spectral_norm': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def create_fallback_matrix_operations():
    """
    Create a fallback matrix operations object.
    
    Returns:
        Fallback matrix operations object
    """
    class FallbackMatrixOperations:
        def __init__(self):
            self.logger = get_logger(__name__)
        
        def safe_correlation_matrix(self, data, method='pearson'):
            return safe_correlation_matrix(data, method)
        
        def safe_matrix_rank(self, matrix, tol=None):
            return safe_matrix_rank(matrix, tol)
        
        def safe_condition_number(self, matrix):
            return safe_condition_number(matrix)
        
        def matrix_normalize(self, matrix, method='zscore'):
            return matrix_normalize(matrix, method)
        
        def extract_matrix_features(self, matrix):
            return extract_matrix_features(matrix)
    
    return FallbackMatrixOperations()
