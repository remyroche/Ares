from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Union

# Enhanced hardware optimization imports
from src.utils.hardware import (
    optimize_dataframe_default, optimize_numpy_array_default,
    auto_optimize, performance_tracked, memory_optimized,
    MemoryOptimizationLevel
)

ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]

@auto_optimize(optimize_inputs=True, optimize_outputs=True)
@performance_tracked
def safe_matrix_multiply(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """np.matmul with shape checks + upcasting to float64 and enhanced optimization."""
    try:
        # Optimize inputs if they are DataFrames
        if isinstance(a, pd.DataFrame):
            a = optimize_dataframe_default(a)
        if isinstance(b, pd.DataFrame):
            b = optimize_dataframe_default(b)
        
        a_np = np.asarray(a, dtype=np.float64)
        b_np = np.asarray(b, dtype=np.float64)
        
        if a_np.ndim < 2:
            a_np = a_np.reshape(1, -1)
        if b_np.ndim < 2:
            b_np = b_np.reshape(-1, 1)
        if a_np.shape[1] != b_np.shape[0]:
            raise ValueError(f"Incompatible shapes for matmul: {a_np.shape} @ {b_np.shape}")
        
        result = a_np @ b_np
        
        # Optimize output array
        return optimize_numpy_array_default(result)
        
    except Exception as e:
        # Fallback to basic implementation
        a_np = np.asarray(a, dtype=np.float64)
        b_np = np.asarray(b, dtype=np.float64)
        if a_np.ndim < 2:
            a_np = a_np.reshape(1, -1)
        if b_np.ndim < 2:
            b_np = b_np.reshape(-1, 1)
        if a_np.shape[1] != b_np.shape[0]:
            raise ValueError(f"Incompatible shapes for matmul: {a_np.shape} @ {b_np.shape}")
        return a_np @ b_np

@memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
@performance_tracked
def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numerics to reduce memory footprint for matrix ops with enhanced optimization."""
    try:
        # Use enhanced optimization system
        return optimize_dataframe_default(df)
    except Exception as e:
        # Fallback to basic implementation
        df_opt = df.copy()
        for col in df_opt.select_dtypes(include=["float", "float64", "float32"]).columns:
            df_opt[col] = pd.to_numeric(df_opt[col], downcast="float")
        for col in df_opt.select_dtypes(include=["int", "int64", "int32"]).columns:
            df_opt[col] = pd.to_numeric(df_opt[col], downcast="integer")
        return df_opt