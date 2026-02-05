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

    # Ensure we are working with numpy/pandas
    vals = df.values

    # Handle mixed types causing object array
    try:
        is_inf = np.isinf(vals)
        is_nan = np.isnan(vals)
    except TypeError:
        # Likely object array
        try:
             # Try element-wise or assume non-numeric cols are issue
             # Just check numeric columns?
             num_df = df.select_dtypes(include=[np.number])
             if not num_df.empty:
                 vals = num_df.values
                 is_inf = np.isinf(vals)
                 is_nan = np.isnan(vals)
             else:
                 return
        except:
             return

    if is_inf.any() or is_nan.any():
        mask = is_inf | is_nan
        # Rows with any inf/nan
        row_mask = mask.any(axis=1)
        affected_rows = df.index[row_mask]

        # Columns with any inf/nan
        col_mask = mask.any(axis=0)
        affected_cols = df.columns[col_mask]

        if len(affected_rows) > 0:
            row_range = f"{affected_rows[0]} to {affected_rows[-1]}"
        else:
            row_range = "None"

        col_list = list(affected_cols)

        tprint(f"Inf/NaN detected in {name}: Rows [{row_range}], Cols {col_list}")

    # Check for low variation (std < 1e-9)
    # Using numpy std along axis 0, ignoring NaNs to avoid issues with previous check
    try:
        # Only numeric
        num_df = df.select_dtypes(include=[np.number])
        if not num_df.empty:
            vals = num_df.values
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
    - Drops columns that are ALL NaN/Inf
    - Drops rows that have ANY NaN/Inf
    - Checks for float32 overflow
    - Aligns y and sample_weight
    """
    tprint(f"Cleaning dataset {name}...")

    if X is None or X.empty:
        tprint(f"{name} is empty.")
        return X, y, sample_weight

    check_inf_nan(X, f"{name}_raw")

    cols_to_drop = []
    bad_rows_mask = np.zeros(len(X), dtype=bool)

    for col in X.columns:
        vals = X[col].values

        try:
            is_nan = pd.isna(vals)

            if vals.dtype == object:
                is_inf = np.zeros_like(is_nan, dtype=bool)
            else:
                is_inf = np.isinf(vals)

            is_bad = is_nan | is_inf

            # Check for float32 overflow if float64
            if vals.dtype == np.float64:
                 # > 3.4028235e+38
                 is_overflow = np.abs(vals) > 3.4028235e+38
                 is_bad |= is_overflow

            if is_bad.all():
                tprint(f"Feature {col}: ALL values are NaN/Inf. Removing column.")
                cols_to_drop.append(col)
                # Do NOT mark rows as bad if we drop the column
            elif is_bad.any():
                count = is_bad.sum()
                tprint(f"Feature {col}: {count} bad values. Marking rows for removal.")
                bad_rows_mask |= is_bad

        except Exception as e:
            tprint(f"Error checking column {col}: {e}. Marking as bad column.")
            cols_to_drop.append(col)

    if cols_to_drop:
        tprint(f"Dropping {len(cols_to_drop)} columns: {cols_to_drop}")
        X = X.drop(columns=cols_to_drop)

    if bad_rows_mask.any():
        tprint(f"Dropping {bad_rows_mask.sum()} rows due to NaN/Inf values.")
        X = X[~bad_rows_mask]

        # Align y
        if y is not None:
            if hasattr(y, 'shape') and y.shape[0] == len(bad_rows_mask):
                if isinstance(y, (pd.Series, pd.DataFrame)):
                    y = y.iloc[~bad_rows_mask]
                else:
                    y = np.asarray(y)[~bad_rows_mask]
            else:
                tprint(f"WARNING: y shape {y.shape} mismatch with X {len(bad_rows_mask)}. Not aligning y.")

        # Align sample_weight
        if sample_weight is not None:
            if hasattr(sample_weight, 'shape') and sample_weight.shape[0] == len(bad_rows_mask):
                if isinstance(sample_weight, (pd.Series, pd.DataFrame)):
                    sample_weight = sample_weight.iloc[~bad_rows_mask]
                else:
                    sample_weight = np.asarray(sample_weight)[~bad_rows_mask]
            else:
                tprint(f"WARNING: sample_weight shape {sample_weight.shape} mismatch with X {len(bad_rows_mask)}. Not aligning weights.")

    return X, y, sample_weight
