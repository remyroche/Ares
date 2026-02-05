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
    if df is None: return

    # Ensure we are working with numpy/pandas
    vals = df.values
    is_inf = np.isinf(vals)
    is_nan = np.isnan(vals)

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
