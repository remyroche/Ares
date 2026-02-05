import sys
import time
import functools
import random
import pandas as pd
import numpy as np

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    tprint(f"Entering function: retry_with_backoff in utils.py")
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        tprint(f"Max retries reached for {func.__name__}. Raising exception.")
                        raise e
                    sleep = (backoff_in_seconds * 2 ** x +
                             random.uniform(0, 1))
                    tprint(f"Error in {func.__name__}: {e}. Retrying in {sleep:.2f}s...")
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

def tprint(msg: str):
    ts = pd.Timestamp.now('UTC').strftime("%Y-%m-%d %H:%M:%S")
    sys.stdout.write(f"[{ts} UTC] {msg}\n")
    sys.stdout.flush()

class Timer:
    def __init__(self, label: str):
        tprint(f"Entering function: __init__ in utils.py")
        self.label = label
        self.t0 = None
    def __enter__(self):
        tprint(f"Entering function: __enter__ in utils.py")
        self.t0 = time.time()
        tprint(f"START: {self.label}")
        return self
    def __exit__(self, exc_type, exc, tb):
        tprint(f"Entering function: __exit__ in utils.py")
        dt = time.time() - self.t0
        tprint(f"END: {self.label} ({dt:.2f}s)")

def check_inf_nan(df: pd.DataFrame, name: str):
    if df is None:
        return

    if df.empty:
        tprint(f"DataFrame {name} is empty.")
        return

    # Only check numeric columns
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return

    vals = num_df.to_numpy(copy=False)
    bad = ~np.isfinite(vals)

    if bad.any():
        row_mask = bad.any(axis=1)
        col_mask = bad.any(axis=0)

        affected_rows = df.index[row_mask]
        affected_cols = num_df.columns[col_mask]

        if len(affected_rows) > 0:
            row_range = f"{affected_rows[0]} to {affected_rows[-1]}"
        else:
            row_range = "None"

        col_list = list(affected_cols)

        tprint(f"Inf/NaN detected in {name}: Rows [{row_range}], Cols {col_list}")

    # Check for low variation (std < 1e-9)
    try:
        stds = np.nanstd(vals, axis=0)
        low_var_mask = stds < 1e-9
        if low_var_mask.any():
            low_var_cols = num_df.columns[low_var_mask].tolist()
            tprint(f"Low variation detected in {name}: Cols {low_var_cols} (std < 1e-9)")
    except Exception as e:
        tprint(f"Error checking variation in {name}: {e}")

def clean_dataset(X: pd.DataFrame, y=None, sample_weight=None, name="X"):
    """
    Strict cleaning of dataset:
    - Drops columns that are ALL NaN/Inf (within numeric block)
    - Drops rows that have ANY NaN/Inf (within numeric block)
    - Checks for float32 overflow
    - Aligns y and sample_weight
    """
    tprint(f"Cleaning dataset {name}...")

    if X is None or X.empty:
        tprint(f"{name} is empty.")
        return X, y, sample_weight

    check_inf_nan(X, f"{name}_raw")

    # Separate numeric/non-numeric once
    num = X.select_dtypes(include=[np.number])

    if num.empty:
        # No numeric columns, return as is (consistent with not dropping anything if no bad values found)
        return X, y, sample_weight

    vals = num.to_numpy(copy=False)

    # Check for Inf/NaN
    bad = ~np.isfinite(vals)

    # Overflow check only if float64
    if vals.dtype == np.float64:
        bad |= (np.abs(vals) > 3.4028235e38)

    # Drop all-bad numeric columns
    col_all_bad = bad.all(axis=0)
    if col_all_bad.any():
        drop_cols = num.columns[col_all_bad]
        tprint(f"Dropping {len(drop_cols)} columns (all bad): {drop_cols.tolist()}")
        X = X.drop(columns=drop_cols)

        # Update bad mask to reflect dropped columns
        # Slice columns from bad matrix
        bad = bad[:, ~col_all_bad]
        # vals is not needed anymore for row check unless we want to recompute, which we don't.
        # bad matrix already contains boolean info.

    # Drop rows with ANY bad value in remaining numeric columns
    row_bad = bad.any(axis=1)

    if row_bad.any():
        tprint(f"Dropping {row_bad.sum()} rows due to NaN/Inf values.")
        keep_mask = ~row_bad

        # Capture original length for y check
        orig_len = len(X)

        X = X.iloc[keep_mask]

        # Align y
        if y is not None:
            # Check if y is likely aligned with X (same length)
            # We use hasattr check and len check
            if hasattr(y, 'shape') and y.shape[0] == orig_len:
                if isinstance(y, (pd.Series, pd.DataFrame)):
                    y = y.iloc[keep_mask]
                else:
                    y = np.asarray(y)[keep_mask]
            else:
                 # If y was not same length as X originally, we don't touch it?
                 # Or warn? Original code warned.
                 tprint(f"WARNING: y shape {y.shape if hasattr(y, 'shape') else 'unknown'} mismatch with original X {orig_len}. Not aligning y.")

        # Align sample_weight
        if sample_weight is not None:
            if hasattr(sample_weight, 'shape') and sample_weight.shape[0] == orig_len:
                if isinstance(sample_weight, (pd.Series, pd.DataFrame)):
                    sample_weight = sample_weight.iloc[keep_mask]
                else:
                    sample_weight = np.asarray(sample_weight)[keep_mask]
            else:
                 tprint(f"WARNING: sample_weight shape {sample_weight.shape if hasattr(sample_weight, 'shape') else 'unknown'} mismatch with original X {orig_len}. Not aligning weights.")

    return X, y, sample_weight
