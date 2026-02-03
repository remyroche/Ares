import sys
import time
import functools
import random
import pandas as pd

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    tprint(f"Entering function: retry_with_backoff in utils.py")
    def decorator(func):
        tprint(f"Entering function: decorator in utils.py")
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tprint(f"Entering function: wrapper in utils.py")
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise e
                    sleep = (backoff_in_seconds * 2 ** x +
                             random.uniform(0, 1))
                    tprint(f"Error in {func.__name__}: {e}. Retrying in {sleep:.2f}s...")
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

def tprint(msg: str):
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
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
