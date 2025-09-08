from ..standardized_parquet_handler import standardized_parquet_handler
"""Comprehensive Validation Framework for Step 7 Enhanced Matrix Operations.

This module provides validation capabilities for input data, matrix operations,
and feature importance results.
"""
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import logging

# Optional dependencies with fallback handling
try:
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

class ComprehensiveValidator:
    """Comprehensive validation framework for step07 operations."""
    
    def __init__(self, logger):
        self.logger = logger
        self.validation_results = {}
        self.validation_rules = {}
    
    def validate_input_data(self, data: Any, data_type: str) -> Tuple[bool, List[str]]:
        """Validate input data based on type."""
        errors = []
        
        if data_type == "dataframe":
            if not PANDAS_AVAILABLE:
                errors.append("Pandas not available for DataFrame validation")
            elif not isinstance(data, pd.DataFrame):
                errors.append("Data is not a pandas DataFrame")
            elif data.empty:
                errors.append("DataFrame is empty")
            elif data.isnull().all().any():
                errors.append("DataFrame has columns with all null values")
        
        elif data_type == "numpy_array":
            if not NUMPY_AVAILABLE:
                errors.append("NumPy not available for array validation")
            elif not isinstance(data, np.ndarray):
                errors.append("Data is not a numpy array")
            elif data.size == 0:
                errors.append("Array is empty")
            elif np.isnan(data).all():
                errors.append("Array contains only NaN values")
        
        elif data_type == "dict":
            if not isinstance(data, dict):
                errors.append("Data is not a dictionary")
            elif not data:
                errors.append("Dictionary is empty")
        
        is_valid = len(errors) == 0
        if not is_valid:
            self.logger.warning(f"⚠️ Input validation failed: {errors}")
        else:
            self.logger.debug(f"✅ Input validation passed for {data_type}")
        
        return is_valid, errors
    
    def validate_matrix_operations(self, matrix: Any, operation_type: str) -> Tuple[bool, List[str]]:
        """Validate matrix operations."""
        errors = []
        
        if not NUMPY_AVAILABLE:
            errors.append("NumPy not available for matrix validation")
            return False, errors
        
        if not isinstance(matrix, np.ndarray):
            errors.append("Matrix is not a numpy array")
            return False, errors
        
        if operation_type == "correlation":
            if not np.allclose(matrix, matrix.T, rtol = 1e-10):
                errors.append("Correlation matrix is not symmetric")
            if not np.all(np.diag(matrix) == 1.0):
                errors.append("Correlation matrix diagonal is not 1.0")
            if np.any(np.abs(matrix) > 1.0):
                errors.append("Correlation matrix has values outside [-1, 1]")
        
        elif operation_type == "covariance":
            if not np.allclose(matrix, matrix.T, rtol = 1e-10):
                errors.append("Covariance matrix is not symmetric")
            if np.any(np.diag(matrix) < 0):
                errors.append("Covariance matrix has negative diagonal values")
        
        elif operation_type == "eigenvalues":
            if not np.allclose(matrix, matrix.T, rtol = 1e-10):
                errors.append("Matrix is not symmetric for eigenvalue computation")
            if np.any(np.iscomplex(matrix)):
                errors.append("Matrix has complex eigenvalues")
        
        is_valid = len(errors) == 0
        if not is_valid:
            self.logger.warning(f"⚠️ Matrix validation failed for {operation_type}: {errors}")
        else:
            self.logger.debug(f"✅ Matrix validation passed for {operation_type}")
        
        return is_valid, errors
    
    def validate_feature_importance(self, importance_dict: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validate feature importance results."""
        errors = []
        
        if not isinstance(importance_dict, dict):
            errors.append("Feature importance is not a dictionary")
        elif not importance_dict:
            errors.append("Feature importance dictionary is empty")
        else:
            values = list(importance_dict.values())
            if NUMPY_AVAILABLE:
                if any(np.isnan(v) for v in values):
                    errors.append("Feature importance contains NaN values")
                if any(np.isinf(v) for v in values):
                    errors.append("Feature importance contains infinite values")
            if any(v < 0 for v in values):
                errors.append("Feature importance contains negative values")
        
        is_valid = len(errors) == 0
        if not is_valid:
            self.logger.warning(f"⚠️ Feature importance validation failed: {errors}")
        else:
            self.logger.debug("✅ Feature importance validation passed")
        
        return is_valid, errors
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        return {
            'validation_results': self.validation_results,
            'validation_rules': self.validation_rules,
            'total_validations': len(self.validation_results)
        }

__all__ = ['ComprehensiveValidator']