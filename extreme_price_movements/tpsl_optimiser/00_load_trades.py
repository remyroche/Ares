from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "bucket",
    "confidence",
    "entry_price",
    "exit_price",
    "is_long",
}


@dataclass(frozen=True)
class LoadTradesConfig:
    # Deprecated: fixed floor. Now we use top 20% decile.
    confidence_floor: float = 0.0
    timeframe: str = "15min"
    top_percentile: float = 0.80 # Top 20%


def _ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_trades_for_bucket(trades: pd.DataFrame, bucket: str, cfg: LoadTradesConfig | None = None) -> pd.DataFrame:
    cfg = cfg or LoadTradesConfig()
    _ensure_columns(trades, REQUIRED_COLUMNS)

    out = trades.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]) 
    out = out[out["bucket"].astype(str) == str(bucket)]

    # Calculate threshold for Top 20%
    if not out.empty:
        threshold = out["confidence"].quantile(cfg.top_percentile)
        out = out[out["confidence"] >= threshold]

    out["timestamp_15m"] = out["timestamp"].dt.floor(cfg.timeframe)
    out = out.sort_values("timestamp_15m").reset_index(drop=True)
    return out
