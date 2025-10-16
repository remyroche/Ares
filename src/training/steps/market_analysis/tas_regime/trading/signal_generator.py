"""Signal generation module for TAS trading engine."""

from __future__ import annotations
import warnings

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

import numpy as np
import pandas as pd

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:

    cp = None

logger = logging.getLogger(__name__)

@dataclass
class SignalConfig:
    """Configuration for the :class:`TradingSignalGenerator`."""

    short_window: int = 5
    long_window: int = 20
    volatility_lookback: int = 10
    min_signal_strength: float = 0.01
    min_regime_confidence: float = 0.5
    capital_fraction_per_trade: float = 0.05
    base_quantity: float = 1.0
    max_signals: int = 5
    allow_short: bool = True
    adapt_to_volatility: bool = True
    min_price: float = 0.5
    min_volume: float = 0.0
    default_symbol: str = "TAS"
    regime_bias: Optional[str] = None

    def __post_init__(self) -> None:
        if self.short_window <= 0 or self.long_window <= 0:
            raise ValueError("Signal windows must be positive")
        if self.short_window > self.long_window:
            logger.debug(
                "Short window larger than long window – swapping values."
            )
            self.short_window, self.long_window = self.long_window, self.short_window
        if not 0 < self.capital_fraction_per_trade <= 1:
            raise ValueError("capital_fraction_per_trade must be within (0, 1]")
        if self.max_signals <= 0:
            raise ValueError("max_signals must be positive")

class TradingSignalGenerator:
    """Generate trading signals for the TAS trading engine."""

    def __init__(self, config: SignalConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        regime_info: Optional[Dict[str, Any]] = None,
        architecture_info: Optional[Dict[str, Any]] = None,
        current_positions: Optional[Dict[str, float]] = None,
        current_capital: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Generate regime-aware trading signals.

        Parameters
        ----------
        market_data:
            Price data containing at least a ``close`` column.
        regime_info:
            Metadata describing the detected market regime.
        architecture_info:
            Optional details about the winning architecture for the regime.
        current_positions:
            Dict of current position sizes by symbol.
        current_capital:
            Current available capital for trading.

        Returns
        -------
        list of dict
            A list of signal dictionaries compatible with :class:`TradingEngine`.
        """

        if market_data is None or market_data.empty:
            self.logger.debug("No market data supplied – skipping signal generation.")
            return []

        close_series = self._extract_series(market_data, "close")
        if close_series is None or close_series.empty:
            self.logger.debug("No close prices available – skipping signal generation.")
            return []

        if len(close_series) < max(self.config.short_window, self.config.long_window):
            self.logger.debug("Insufficient history for moving average calculation.")
            return []

        regime_confidence = 1.0
        if regime_info:
            regime_confidence = float(regime_info.get("confidence", regime_confidence))

        if regime_confidence < self.config.min_regime_confidence:
            self.logger.debug(
                "Regime confidence %.3f below threshold %.3f.",
                regime_confidence,
                self.config.min_regime_confidence,
            )
            return []

        volume_series = self._extract_series(market_data, "volume")
        if volume_series is not None and not volume_series.empty:
            latest_volume = float(volume_series.iloc[-1])
            if latest_volume < self.config.min_volume:
                self.logger.debug(
                    "Volume %.2f below minimum %.2f – skipping signal.",
                    latest_volume,
                    self.config.min_volume,
                )
                return []

        short_ma = close_series.rolling(self.config.short_window).mean().iloc[-1]
        long_ma = close_series.rolling(self.config.long_window).mean().iloc[-1]
        latest_price = float(close_series.iloc[-1])

        if latest_price < self.config.min_price:
            self.logger.debug(
                "Price %.2f below minimum price %.2f – skipping signal.",
                latest_price,
                self.config.min_price,
            )
            return []

        # Signal strength based on moving-average differential.
        signal_strength = 0.0
        if long_ma != 0:
            signal_strength = (short_ma - long_ma) / abs(long_ma)

        if abs(signal_strength) < self.config.min_signal_strength:
            self.logger.debug(
                "Signal strength %.5f below threshold %.5f.",
                signal_strength,
                self.config.min_signal_strength,
            )
            return []

        side = "buy" if signal_strength > 0 else "sell"
        if side == "sell" and not self.config.allow_short:
            self.logger.debug("Short positions disabled – skipping sell signal.")
            return []

        if self.config.regime_bias:
            bias = self.config.regime_bias.lower()
            if bias == "bullish" and side != "buy":
                self.logger.debug("Bullish bias prevents sell signal.")
                return []
            if bias == "bearish" and side != "sell":
                self.logger.debug("Bearish bias prevents buy signal.")
                return []

        symbol = self._infer_symbol(market_data, regime_info)
        current_position = 0.0
        if current_positions and symbol in current_positions:
            current_position = float(current_positions[symbol])

        quantity = self._determine_quantity(
            price=latest_price,
            current_capital=current_capital,
            signal_strength=signal_strength,
            regime_confidence=regime_confidence,
            current_position=current_position,
            close_series=close_series,
        )

        signal: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": latest_price,
            "order_type": "market",
            "confidence": regime_confidence,
            "signal_strength": float(signal_strength),
            "regime_info": regime_info or {},
            "architecture_info": architecture_info or {},
            "current_position": current_position,
        }

        self.logger.debug("Generated signal: %s", signal)
        return [signal][: self.config.max_signals]

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _determine_quantity(
        self,
        *,
        price: float,
        current_capital: float,
        signal_strength: float,
        regime_confidence: float,
        current_position: float,
        close_series: pd.Series,
    ) -> float:
        if price <= 0:
            return self.config.base_quantity

        capital_allocation = current_capital * self.config.capital_fraction_per_trade
        if capital_allocation <= 0:
            capital_allocation = self.config.base_quantity * price

        quantity = max(self.config.base_quantity, capital_allocation / price)

        if self.config.adapt_to_volatility and len(close_series) > 1:
            lookback = min(self.config.volatility_lookback, len(close_series) - 1)
            if lookback > 1:
                volatility = close_series.pct_change().rolling(lookback).std().iloc[-1]
                if np.isfinite(volatility) and volatility > 0:
                    quantity = quantity / (1 + 10 * volatility)

        # Increase size if confidence is high and strength strong.
        scaling = max(0.1, min(2.0, abs(signal_strength) * 10 * regime_confidence))
        quantity *= scaling

        # Prevent position flip oversizing.
        if current_position != 0:
            quantity = max(self.config.base_quantity, quantity - abs(current_position))

        return float(max(self.config.base_quantity, quantity))

    def _extract_series(
        self, market_data: pd.DataFrame, field: str
    ) -> Optional[pd.Series]:
        if field in market_data.columns:
            return market_data[field].astype(float)

        columns = market_data.columns
        if isinstance(columns, pd.MultiIndex):
            matching = [col for col in columns if col[-1] == field]
            if matching:
                # Use the first matching column for simplicity.
                return market_data[matching[0]].astype(float)

        # Fall back to column names that end with the field (e.g., BTC_close).
        for column in columns:
            if str(column).lower().endswith(field.lower()):
                return market_data[column].astype(float)

        return None

    def _infer_symbol(
        self, market_data: pd.DataFrame, regime_info: Optional[Dict[str, Any]]
    ) -> str:
        if regime_info and "symbol" in regime_info:
            return str(regime_info["symbol"])

        columns = market_data.columns
        if isinstance(columns, pd.MultiIndex):
            return str(columns[0][0])

        for column in columns:
            if str(column).lower().endswith("_close"):
                return str(column).rsplit("_", 1)[0]

        return self.config.default_symbol

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _vectorbt_apply_operation(self, data: pd.Series, func,
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)

        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
