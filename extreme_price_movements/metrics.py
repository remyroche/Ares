import os
import sys
import pandas as pd
from datetime import datetime, timezone
from .utils import tprint

def tprint(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    sys.stdout.write(f"[{ts} UTC] {msg}\n")
    sys.stdout.flush()

class MetricsLogger:
    def __init__(self, log_dir="logs"):
        tprint(f"Entering function: __init__ in metrics.py")
        tprint(f"Initializing MetricsLogger with log_dir='{log_dir}'")
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        tprint(f"Directory '{self.log_dir}' ensured (created if not exists).")

    def _get_log_path(self, ts: pd.Timestamp) -> str:
        tprint(f"Entering function: _get_log_path in metrics.py")
        date_str = ts.strftime("%Y-%m-%d")
        path = os.path.join(self.log_dir, f"metrics_{date_str}.csv")
        tprint(f"Determined log path: '{path}' for timestamp: {ts}")
        return path

    def log(self, ts_sig: pd.Timestamp, data: dict):
        """
        Logs metrics to a CSV file partitioned by day of ts_sig.
        data: dictionary of metrics
        """
        tprint(f"Entering function: log in metrics.py")
        tprint(f"Logging metrics for ts_sig={ts_sig}. Data keys: {list(data.keys())}")

        # Ensure ts_sig is in data
        row = data.copy()
        row["ts_sig"] = ts_sig.isoformat()
        row["log_ts"] = datetime.now(timezone.utc).isoformat()

        tprint(f"Prepared data row with timestamp info added.")

        path = self._get_log_path(ts_sig)
        df = pd.DataFrame([row])

        exists = os.path.exists(path)
        tprint(f"Writing to file: '{path}'. File exists: {exists}")

        if not exists:
            df.to_csv(path, index=False)
            tprint(f"Created new log file: {path}")
        else:
            df.to_csv(path, mode='a', header=False, index=False)
            tprint(f"Appended to existing log file: {path}")

        # Also tprint summary
        tprint(f"Metrics saved to {path}: {row}")
