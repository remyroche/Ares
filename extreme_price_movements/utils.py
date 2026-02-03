import sys
import time
import pandas as pd

def tprint(msg: str):
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    sys.stdout.write(f"[{ts} UTC] {msg}\n")
    sys.stdout.flush()

class Timer:
    def __init__(self, label: str):
        self.label = label
        self.t0 = None
    def __enter__(self):
        self.t0 = time.time()
        tprint(f"START: {self.label}")
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        tprint(f"END: {self.label} ({dt:.2f}s)")
