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
    
    def calculate_pairwise_similarities(self, feature_vectors: np.ndarray, method: str = 'cosine_with_cv_filtering') -> np.ndarray:
        """
        Calculate pairwise similarities between feature vectors.
        
        Args:
            feature_vectors: Matrix of feature vectors (n_samples, n_features)
            method: Similarity calculation method
            
        Returns:
            Similarity matrix (n_samples, n_samples)
        """
        if self._ops and hasattr(self._ops, 'calculate_pairwise_similarities'):
            return self._ops.calculate_pairwise_similarities(feature_vectors, method)
        else:
            # Fallback implementation
            logger.info(f"🔄 Using fallback pairwise similarity calculation with method: {method}")
            
            if method == 'cosine_with_cv_filtering' or method == 'cosine':
                # Normalize feature vectors for cosine similarity
                norms = np.linalg.norm(feature_vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                normalized_vectors = feature_vectors / norms
                
                # Calculate cosine similarity matrix
                similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
                
                # Ensure diagonal is 1.0 and values are in [0, 1]
                np.fill_diagonal(similarity_matrix, 1.0)
                similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)
                
                return similarity_matrix
            
            elif method == 'euclidean':
                # Calculate Euclidean distance and convert to similarity
                from scipy.spatial.distance import pdist, squareform
                distances = squareform(pdist(feature_vectors, metric='euclidean'))
                # Convert distance to similarity (closer = more similar)
                max_dist = np.max(distances)
                if max_dist > 0:
                    similarity_matrix = 1.0 - (distances / max_dist)
                else:
                    similarity_matrix = np.ones_like(distances)
                return similarity_matrix
            
            else:
                logger.warning(f"Unknown similarity method: {method}, using cosine")
                return self.calculate_pairwise_similarities(feature_vectors, 'cosine')
    
    def apply_cv_filtering(self, similarity_matrix: np.ndarray, cv_values: np.ndarray, max_cv_difference: float = 0.5) -> np.ndarray:
        """
        Apply CV (coefficient of variation) filtering to similarity matrix.
        
        Args:
            similarity_matrix: Input similarity matrix
            cv_values: CV values for each sample
            max_cv_difference: Maximum allowed CV difference for similarity
            
        Returns:
            Filtered similarity matrix
        """
        if self._ops and hasattr(self._ops, 'apply_cv_filtering'):
            return self._ops.apply_cv_filtering(similarity_matrix, cv_values, max_cv_difference)
        else:
            # Fallback implementation
            logger.info(f"🔄 Using fallback CV filtering with max_cv_difference: {max_cv_difference}")
            
            filtered_matrix = similarity_matrix.copy()
            n_samples = len(cv_values)
            
            for i in range(n_samples):
                for j in range(n_samples):
                    if i != j:  # Don't modify diagonal
                        cv_diff = abs(cv_values[i] - cv_values[j])
                        if cv_diff > max_cv_difference:
                            # Reduce similarity for regimes with very different CVs
                            reduction_factor = min(cv_diff / max_cv_difference, 5.0)  # Cap at 5x reduction
                            filtered_matrix[i, j] *= (1.0 / reduction_factor)
                            filtered_matrix[i, j] = max(filtered_matrix[i, j], 0.01)  # Keep minimum similarity
            
            return filtered_matrix
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage and performance.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        if self._ops and hasattr(self._ops, 'optimize_dataframe'):
            return self._ops.optimize_dataframe(df)
        else:
            # FIXED: Enhanced DataFrame optimization
            optimized_df = df.copy()
            
            # Optimize dtypes
            for col in optimized_df.columns:
                if optimized_df[col].dtype == 'object':
                    try:
                        optimized_df[col] = pd.to_numeric(optimized_df[col], errors='ignore')
                    except:
                        pass
                elif optimized_df[col].dtype == 'float64':
                    # Try to downcast to float32 if possible
                    if optimized_df[col].min() >= np.finfo(np.float32).min and optimized_df[col].max() <= np.finfo(np.float32).max:
                        optimized_df[col] = optimized_df[col].astype(np.float32)
                elif optimized_df[col].dtype == 'int64':
                    # Try to downcast to smaller int types
                    if optimized_df[col].min() >= np.iinfo(np.int32).min and optimized_df[col].max() <= np.iinfo(np.int32).max:
                        optimized_df[col] = optimized_df[col].astype(np.int32)
                    elif optimized_df[col].min() >= np.iinfo(np.int16).min and optimized_df[col].max() <= np.iinfo(np.int16).max:
                        optimized_df[col] = optimized_df[col].astype(np.int16)
                    elif optimized_df[col].min() >= np.iinfo(np.int8).min and optimized_df[col].max() <= np.iinfo(np.int8).max:
                        optimized_df[col] = optimized_df[col].astype(np.int8)
            
            return optimized_df
    
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

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame optimization wrapper."""
    return default_matrix_ops.optimize_dataframe(df)