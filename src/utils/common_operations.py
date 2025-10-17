from __future__ import annotations
from typing import Any, Callable
import pandas as pd

def safe_dataframe_operation(operation_func: Callable[..., pd.DataFrame], *args, **kwargs) -> pd.DataFrame:
    """Run a dataframe op with a tiny safety net."""
    if not callable(operation_func):
        raise TypeError("operation_func must be callable")
    df = operation_func(*args, **kwargs)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("operation_func must return a pandas DataFrame")
    return df