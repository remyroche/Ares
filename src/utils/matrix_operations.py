from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Union

ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]

def safe_matrix_multiply(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """np.matmul with shape checks + upcasting to float64."""
    a_np = np.asarray(a, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64)
    if a_np.ndim < 2:
        a_np = a_np.reshape(1, -1)
    if b_np.ndim < 2:
        b_np = b_np.reshape(-1, 1)
    if a_np.shape[1] != b_np.shape[0]:
        raise ValueError(f"Incompatible shapes for matmul: {a_np.shape} @ {b_np.shape}")
    return a_np @ b_np

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numerics to reduce memory footprint for matrix ops."""
    df_opt = df.copy()
    for col in df_opt.select_dtypes(include=["float", "float64", "float32"]).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast="float")
    for col in df_opt.select_dtypes(include=["int", "int64", "int32"]).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast="integer")
    return df_opt