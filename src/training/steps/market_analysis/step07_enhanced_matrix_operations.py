"""
Compatibility module for step07_enhanced_matrix_operations imports.

This module provides backwards compatibility for imports that were previously
available in the deleted step07_enhanced_matrix_operations.py files.

All functionality has been moved to ml_commons.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

# Import the actual functionality from the new locations
try:
    from src.utils.ml_common.matrix_operations import (
        get_enhanced_matrix_operations,
        EnhancedMatrixOperations
    )
    FUNCTIONALITY_AVAILABLE = True
except ImportError as e:
    FUNCTIONALITY_AVAILABLE = False
    warnings.warn(f"Step07 functionality not available: {e}")


class Step7EnhancedMatrixOperations:
    """
    Compatibility class for Step7EnhancedMatrixOperations.
    
    This class provides backwards compatibility for the Step7EnhancedMatrixOperations
    that was previously available in step07_enhanced_matrix_operations.py.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced matrix operations."""
        if not FUNCTIONALITY_AVAILABLE:
            raise ImportError("Step07 functionality not available. Please ensure ml_commons is properly installed.")
        
        self.config = config or {}
        self.matrix_ops = get_enhanced_matrix_operations()
        
        warnings.warn(
            "Step7EnhancedMatrixOperations is deprecated. Use get_enhanced_matrix_operations() from src.utils.ml_common.matrix_operations instead.",
            DeprecationWarning,
            stacklevel=2
        )
    
    def matrix_multiply(self, A: Any, B: Any, **kwargs) -> Any:
        """Matrix multiplication using enhanced matrix operations."""
        return self.matrix_ops.matrix_multiply(A, B, **kwargs)
    
    def matrix_inverse(self, A: Any, **kwargs) -> Any:
        """Matrix inverse using enhanced matrix operations."""
        return self.matrix_ops.matrix_inverse(A, **kwargs)
    
    def matrix_decomposition(self, A: Any, **kwargs) -> Any:
        """Matrix decomposition using enhanced matrix operations."""
        return self.matrix_ops.matrix_decomposition(A, **kwargs)


class EnhancedMatrixOperationsStep:
    """
    Compatibility class for EnhancedMatrixOperationsStep.
    
    This class provides backwards compatibility for the EnhancedMatrixOperationsStep
    that was previously available in step07_enhanced_matrix_operations.py.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced matrix operations step."""
        if not FUNCTIONALITY_AVAILABLE:
            raise ImportError("Step07 functionality not available. Please ensure ml_commons is properly installed.")
        
        self.config = config or {}
        self.matrix_ops = get_enhanced_matrix_operations()
        
        warnings.warn(
            "EnhancedMatrixOperationsStep is deprecated. Use get_enhanced_matrix_operations() from src.utils.ml_common.matrix_operations instead.",
            DeprecationWarning,
            stacklevel=2
        )
    
    def execute(self, data: Any, **kwargs) -> Any:
        """Execute matrix operations using enhanced matrix operations."""
        return self.matrix_ops.execute(data, **kwargs)
    
    def process_matrix(self, matrix: Any, **kwargs) -> Any:
        """Process matrix using enhanced matrix operations."""
        return self.matrix_ops.process_matrix(matrix, **kwargs)


def run_step(data: Any, config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
    """
    Compatibility function for run_step.
    
    This function provides backwards compatibility for the run_step function
    that was previously available in step07_enhanced_matrix_operations.py.
    """
    if not FUNCTIONALITY_AVAILABLE:
        raise ImportError("Step07 functionality not available. Please ensure ml_commons is properly installed.")
    
    warnings.warn(
        "run_step is deprecated. Use get_enhanced_matrix_operations() from src.utils.ml_common.matrix_operations instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    matrix_ops = get_enhanced_matrix_operations()
    return matrix_ops.execute(data, config=config, **kwargs)


# Backwards compatibility exports
__all__ = [
    'Step7EnhancedMatrixOperations',
    'EnhancedMatrixOperationsStep',
    'run_step',
    'FUNCTIONALITY_AVAILABLE'
]