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
    confidence_floor: float = 0.60
    timeframe: str = "15min"


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
    out = out[out["confidence"].astype(float) >= float(cfg.confidence_floor)]
    out["timestamp_15m"] = out["timestamp"].dt.floor(cfg.timeframe)
    out = out.sort_values("timestamp_15m").reset_index(drop=True)
    return out
