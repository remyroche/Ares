import os
import sys
import pandas as pd
from datetime import datetime, timezone

def tprint(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    sys.stdout.write(f"[{ts} UTC] {msg}\n")
    sys.stdout.flush()

class MetricsLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def _get_log_path(self, ts: pd.Timestamp) -> str:
        date_str = ts.strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"metrics_{date_str}.csv")

    def log(self, ts_sig: pd.Timestamp, data: dict):
        """
        Logs metrics to a CSV file partitioned by day of ts_sig.
        data: dictionary of metrics
        """
        # Ensure ts_sig is in data
        row = data.copy()
        row["ts_sig"] = ts_sig.isoformat()
        row["log_ts"] = datetime.now(timezone.utc).isoformat()

        path = self._get_log_path(ts_sig)
        df = pd.DataFrame([row])

        if not os.path.exists(path):
            df.to_csv(path, index=False)
        else:
            df.to_csv(path, mode='a', header=False, index=False)

        # Also tprint summary
        tprint(f"Metrics saved to {path}: {row}")
