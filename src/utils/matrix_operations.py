#!/usr/bin/env python3
"""
Matrix Operations Module

This module provides unified matrix operations for the HMM clustering system,
integrating with existing matrix operation utilities.
"""

import numpy as np
import logging
from typing import Optional, Union, Any

# Try to import existing unified matrix operations
try:
    from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations as _UnifiedMatrixOperations
    UNIFIED_MATRIX_OPS_AVAILABLE = True
except ImportError:
    UNIFIED_MATRIX_OPS_AVAILABLE = False
    _UnifiedMatrixOperations = None

logger = logging.getLogger(__name__)


class UnifiedMatrixOperations:
    """
    Unified matrix operations wrapper that provides compatibility
    with the existing matrix operations system.
    """
    
    def __init__(self):
        """Initialize the unified matrix operations."""
        if UNIFIED_MATRIX_OPS_AVAILABLE:
            self._ops = _UnifiedMatrixOperations()
            logger.info("Using existing unified matrix operations")
        else:
            self._ops = None
            logger.warning("Unified matrix operations not available, using fallback implementations")
    
    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Perform matrix multiplication with optimization.
        
        Args:
            a: First matrix
            b: Second matrix
            
        Returns:
            Result of matrix multiplication
        """
        if self._ops and hasattr(self._ops, 'matrix_multiply'):
            return self._ops.matrix_multiply(a, b)
        else:
            # Fallback to numpy
            return np.dot(a, b)
    
    def matrix_inverse(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute matrix inverse with stability checks.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Inverse of the matrix
        """
        if self._ops and hasattr(self._ops, 'matrix_inverse'):
            return self._ops.matrix_inverse(matrix)
        else:
            # Fallback with stability check
            try:
                return np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse for singular matrices
                logger.warning("Matrix is singular, using pseudo-inverse")
                return np.linalg.pinv(matrix)
    
    def matrix_decomposition(self, matrix: np.ndarray, method: str = 'svd') -> tuple:
        """
        Perform matrix decomposition.
        
        Args:
            matrix: Input matrix
            method: Decomposition method ('svd', 'cholesky', 'eigen')
            
        Returns:
            Decomposition results
        """
        if self._ops and hasattr(self._ops, 'matrix_decomposition'):
            return self._ops.matrix_decomposition(matrix, method)
        else:
            # Fallback implementations
            if method == 'svd':
                return np.linalg.svd(matrix)
            elif method == 'cholesky':
                return np.linalg.cholesky(matrix)
            elif method == 'eigen':
                return np.linalg.eig(matrix)
            else:
                raise ValueError(f"Unknown decomposition method: {method}")
    
    def optimize_array(self, array: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
        """
        Optimize array for memory and performance.
        
        Args:
            array: Input array
            dtype: Target data type
            
        Returns:
            Optimized array
        """
        if self._ops and hasattr(self._ops, 'optimize_array'):
            return self._ops.optimize_array(array, dtype)
        else:
            # Fallback optimization
            if dtype is not None:
                array = array.astype(dtype)
            
            # Ensure C-contiguous for better performance
            if not array.flags['C_CONTIGUOUS']:
                array = np.ascontiguousarray(array)
            
            return array
    
    def create_memory_efficient_array(self, data: Any, dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Create a memory-efficient array.
        
        Args:
            data: Input data
            dtype: Data type for the array
            
        Returns:
            Memory-efficient array
        """
        if self._ops and hasattr(self._ops, 'create_memory_efficient_array'):
            return self._ops.create_memory_efficient_array(data, dtype)
        else:
            # Fallback implementation
            array = np.asarray(data, dtype=dtype)
            return np.ascontiguousarray(array)
    
    def compute_covariance(self, data: np.ndarray, regularization: float = 1e-6) -> np.ndarray:
        """
        Compute covariance matrix with regularization.
        
        Args:
            data: Input data matrix
            regularization: Regularization parameter
            
        Returns:
            Regularized covariance matrix
        """
        if self._ops and hasattr(self._ops, 'compute_covariance'):
            return self._ops.compute_covariance(data, regularization)
        else:
            # Fallback implementation
            cov = np.cov(data.T)
            # Add regularization to diagonal
            cov += regularization * np.eye(cov.shape[0])
            return cov
    
    def is_available(self) -> bool:
        """Check if unified matrix operations are available."""
        return UNIFIED_MATRIX_OPS_AVAILABLE and self._ops is not None


# Create a default instance for backward compatibility
default_matrix_ops = UnifiedMatrixOperations()

# Export commonly used functions
def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiplication wrapper."""
    return default_matrix_ops.matrix_multiply(a, b)

def matrix_inverse(matrix: np.ndarray) -> np.ndarray:
    """Matrix inverse wrapper."""
    return default_matrix_ops.matrix_inverse(matrix)

def optimize_array(array: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Array optimization wrapper."""
    return default_matrix_ops.optimize_array(array, dtype)