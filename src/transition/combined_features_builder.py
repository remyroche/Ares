# src/transition/combined_features_builder.py

from src.utils.logger import system_logger
from typing import Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

REQUIRED_FEATURES = [
    "log_returns",
    "volatility_20",
    "volume_ratio",
    "rsi",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bb_position",
    "bb_width",
    "atr",
    "volatility_regime",
    "volatility_acceleration",
]


@dataclass
class CombinedFeaturesConfig:
    volatility_threshold: float = 0.02


class CombinedFeaturesBuilder:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.logger = system_logger.getChild("CombinedFeaturesBuilder")
        cf = (
            (config or {}).get("combined_features", {})
            if isinstance(config, dict)
            else {}
        )
        self.cfg = CombinedFeaturesConfig(
            volatility_threshold=float(cf.get("volatility_threshold", 0.02)),
        )

    def _rsi(self, close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.replace([np.inf, -np.inf], np.nan).fillna(50.0)

    def _macd(
        self,
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    def _bb(
        self,
        close: pd.Series,
        window: int = 20,
        k: float = 2.0,
    ) -> tuple[pd.Series, pd.Series]:
        sma = close.rolling(window, min_periods=1).mean()
        std = close.rolling(window, min_periods=1).std()
        upper = sma + k * std
        lower = sma - k * std
        width = (upper - lower) / sma.replace(0, np.nan)
        pos = (close - lower) / (upper - lower).replace(0, np.nan)
        return pos.fillna(0.5), width.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window, min_periods=1).mean()
