import pandas as pd
from datetime import datetime, timezone
from .utils import tprint

def now_utc():
    tprint(f"Entering function: now_utc in time_utils.py")
    return datetime.now(timezone.utc)

def floor_to_hour(dt: datetime) -> datetime:
    """Floors a datetime to the start of the hour."""
    tprint(f"Entering function: floor_to_hour in time_utils.py")
    return dt.replace(minute=0, second=0, microsecond=0)

def get_ts_sig() -> pd.Timestamp:
    """
    Returns the signal timestamp: floor_to_hour(now_utc) - 1h.
    This represents the last fully closed hourly bar.
    """
    tprint(f"Entering function: get_ts_sig in time_utils.py")
    now = now_utc()
    ts_sig = floor_to_hour(now) - pd.Timedelta(hours=1)
    return pd.Timestamp(ts_sig)

def ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures the index is UTC localized."""
    tprint(f"Entering function: ensure_utc in time_utils.py")
    if df.index.tz is None:
        # Assume naive is UTC if not specified, or raise error?
        # User said "Anti-footgun: never do .floor("D") on a tz-naive index."
        # Here we convert naive to UTC assuming it was meant to be UTC.
        return df.tz_localize("UTC")
    else:
        return df.tz_convert("UTC")
