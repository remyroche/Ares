from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Union

# Enhanced hardware optimization imports
try:
    from src.utils.hardware.optimization_decorators import (
        auto_optimize, memory_efficient, OptimizationConfig, OptimizationLevel
    )
    from src.utils.hardware.m1_enhanced_gpu_manager import get_enhanced_gpu_manager, GPUOperationType
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    # Create dummy decorators
    def auto_optimize(config=None):
        def decorator(func):
            return func
        return decorator
    def memory_efficient(config=None):
        def decorator(func):
            return func
        return decorator

ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]

@auto_optimize(OptimizationConfig(
    enable_caching=True,
    enable_dtype_optimization=True,
    optimization_level=OptimizationLevel.MAXIMUM
))
def safe_matrix_multiply(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Enhanced matrix multiplication with GPU acceleration and shape checks."""
    # Try GPU acceleration if available
    if HARDWARE_OPTIMIZATION_AVAILABLE:
        try:
            gpu_manager = get_enhanced_gpu_manager()
            if gpu_manager.is_available():
                return gpu_manager.optimize_matrix_operations(
                    a, b, operation_type=GPUOperationType.MATRIX_MULTIPLICATION
                )
        except Exception:
            pass  # Fallback to CPU implementation
    
    # CPU implementation with enhanced optimization
    a_np = np.asarray(a, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64)
    if a_np.ndim < 2:
        a_np = a_np.reshape(1, -1)
    if b_np.ndim < 2:
        b_np = b_np.reshape(-1, 1)
    if a_np.shape[1] != b_np.shape[0]:
        raise ValueError(f"Incompatible shapes for matmul: {a_np.shape} @ {b_np.shape}")
    return a_np @ b_np

@memory_efficient(OptimizationConfig(
    enable_dtype_optimization=True,
    optimization_level=OptimizationLevel.AGGRESSIVE,
    enable_compression=True
))
def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced DataFrame optimization using hardware optimization tools."""
    # Use enhanced hardware optimization if available
    if HARDWARE_OPTIMIZATION_AVAILABLE:
        try:
            from src.utils.hardware.enhanced_caching_system import optimize_dataframe_default
            return optimize_dataframe_default(df)
        except ImportError:
            pass  # Fallback to enhanced implementation
    
    # Enhanced implementation with better optimization
    df_opt = df.copy()
    
    # Optimize float columns
    for col in df_opt.select_dtypes(include=["float", "float64", "float32"]).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast="float")
    
    # Optimize integer columns
    for col in df_opt.select_dtypes(include=["int", "int64", "int32"]).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast="integer")
    
    # Convert object columns to category if beneficial
    for col in df_opt.select_dtypes(include=["object"]).columns:
        if df_opt[col].nunique() / len(df_opt) < 0.5:  # If less than 50% unique values
            df_opt[col] = df_opt[col].astype('category')
    
    return df_opt