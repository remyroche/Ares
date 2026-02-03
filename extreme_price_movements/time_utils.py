import pandas as pd
from datetime import datetime, timezone

def now_utc():
    return datetime.now(timezone.utc)

def floor_to_hour(dt: datetime) -> datetime:
    """Floors a datetime to the start of the hour."""
    return dt.replace(minute=0, second=0, microsecond=0)

def get_ts_sig() -> pd.Timestamp:
    """
    Returns the signal timestamp: floor_to_hour(now_utc) - 1h.
    This represents the last fully closed hourly bar.
    """
    now = now_utc()
    ts_sig = floor_to_hour(now) - pd.Timedelta(hours=1)
    return pd.Timestamp(ts_sig)

def ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures the index is UTC localized."""
    if df.index.tz is None:
        # Assume naive is UTC if not specified, or raise error?
        # User said "Anti-footgun: never do .floor("D") on a tz-naive index."
        # Here we convert naive to UTC assuming it was meant to be UTC.
        return df.tz_localize("UTC")
    else:
        return df.tz_convert("UTC")
