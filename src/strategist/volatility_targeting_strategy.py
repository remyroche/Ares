from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from src.utils.error_handler import (
    handle_data_processing_errors,
    handle_errors,
)
from src.utils.logger import system_logger


class VolatilityMethod(Enum):
    """Enumeration of different volatility calculation methods."""

    SIMPLE = "simple"
    EWMA = "ewma"
    GARCH = "garch"
    PARKINSON = "parkinson"
    ADAPTIVE = "adaptive"


@dataclass
class VolatilityTargetingConfig:
    """Configuration for volatility targeting strategy."""

    target_volatility: float = 0.15  # 15% annual target
    volatility_method: VolatilityMethod = VolatilityMethod.EWMA
    lookback_period: int = 20
    max_leverage: float = 3.0
    min_leverage: float = 0.1
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    kelly_enhancement: bool = False
    max_kelly_fraction: float = 0.25


class VolatilityTargetingStrategy:
    """
    Implements volatility targeting for dynamic position sizing and risk management.

    This strategy adjusts position sizes to maintain a consistent level of portfolio volatility,
    helping to stabilize returns and improve risk-adjusted performance.
    """

    def __init__(self, config: VolatilityTargetingConfig | None = None):
        self.config = config or VolatilityTargetingConfig()
        self.logger = system_logger.getChild("VolatilityTargeting")
        self.logger.info(
            f"Initialized volatility targeting strategy with target: {self.config.target_volatility:.1%}",
        )

    @handle_errors(
        exceptions=(ValueError, TypeError),
        default_return=1.0,
        context="calculate_position_multiplier",
    )
    def calculate_position_multiplier(
        self,
        price_data: pd.DataFrame,
        current_volatility: float | None = None,
    ) -> float:
        """
        Calculate the position size multiplier based on current volatility.

        Args:
            price_data: DataFrame with OHLCV data
            current_volatility: Optional pre-calculated volatility

        Returns:
            Position size multiplier
        """
        if current_volatility is None:
            current_volatility = self.calculate_volatility(price_data)

        if current_volatility <= 0 or pd.isna(current_volatility):
            self.logger.warning(
                "Invalid volatility value, returning neutral multiplier",
            )
            return 1.0

        # Basic volatility targeting formula
        multiplier = self.config.target_volatility / current_volatility

        # Apply Kelly enhancement if enabled
        if self.config.kelly_enhancement:
            kelly_factor = self._calculate_kelly_factor(price_data)
            multiplier *= kelly_factor

        # Clip to configured bounds
        multiplier = np.clip(
            multiplier,
            self.config.min_leverage,
            self.config.max_leverage,
        )

        self.logger.debug(f"Calculated position multiplier: {multiplier:.3f}")
        return multiplier

    @handle_data_processing_errors(default_return=0.15, context="calculate_volatility")
    def calculate_volatility(self, price_data: pd.DataFrame) -> float:
        """Calculate volatility based on the configured method."""
        if price_data.empty or "close" not in price_data.columns:
            return self.config.target_volatility

        returns = price_data["close"].pct_change().dropna()

        if len(returns) < 2:
            return self.config.target_volatility

        method = self.config.volatility_method

        if method == VolatilityMethod.SIMPLE:
            return self._calculate_simple_volatility(returns)
        if method == VolatilityMethod.EWMA:
            return self._calculate_ewma_volatility(returns)
        if method == VolatilityMethod.GARCH:
            return self._calculate_garch_volatility(returns)
        if method == VolatilityMethod.PARKINSON:
            return self._calculate_parkinson_volatility(price_data)
        if method == VolatilityMethod.ADAPTIVE:
            return self._calculate_adaptive_volatility(returns, price_data)
        return self._calculate_ewma_volatility(returns)

    def _calculate_simple_volatility(self, returns: pd.Series) -> float:
        """Calculate simple historical volatility."""
        if len(returns) < self.config.lookback_period:
            period = len(returns)
        else:
            period = self.config.lookback_period

        vol = returns.tail(period).std() * np.sqrt(252)
        return vol if not pd.isna(vol) else self.config.target_volatility

    def _calculate_ewma_volatility(self, returns: pd.Series) -> float:
        """Calculate EWMA volatility."""
        ewma_vol = returns.ewm(span=self.config.lookback_period).std().iloc[
            -1
        ] * np.sqrt(252)
        return ewma_vol if not pd.isna(ewma_vol) else self.config.target_volatility

    def _calculate_garch_volatility(self, returns: pd.Series) -> float:
        """Calculate GARCH-like volatility (simplified)."""
        # Simplified GARCH(1,1) approximation
        squared_returns = returns**2
        if len(squared_returns) < self.config.lookback_period:
            period = len(squared_returns)
        else:
            period = self.config.lookback_period

        garch_vol = np.sqrt(squared_returns.tail(period).mean()) * np.sqrt(252)
        return garch_vol if not pd.isna(garch_vol) else self.config.target_volatility

    def _calculate_parkinson_volatility(self, price_data: pd.DataFrame) -> float:
        """Calculate Parkinson volatility using high-low range."""
        if not all(col in price_data.columns for col in ["high", "low"]):
            return self.config.target_volatility

        if len(price_data) < self.config.lookback_period:
            period = len(price_data)
        else:
            period = self.config.lookback_period

        log_hl = np.log(price_data["high"] / price_data["low"])
        parkinson_vol = np.sqrt((log_hl**2).tail(period).mean() * 252 / (4 * np.log(2)))

        return (
            parkinson_vol
            if not pd.isna(parkinson_vol)
            else self.config.target_volatility
        )

    def _calculate_adaptive_volatility(
        self,
        returns: pd.Series,
        price_data: pd.DataFrame,
    ) -> float:
        """Calculate adaptive volatility using multiple methods."""
        # Combine multiple volatility estimates
        simple_vol = self._calculate_simple_volatility(returns)
        ewma_vol = self._calculate_ewma_volatility(returns)

        # Weight based on recent market conditions
        recent_vol = returns.tail(5).std() * np.sqrt(252)

        if recent_vol > simple_vol * 1.5:  # High volatility regime
            weight_ewma = 0.7
            weight_simple = 0.3
        else:  # Normal/low volatility regime
            weight_ewma = 0.5
            weight_simple = 0.5

        adaptive_vol = weight_ewma * ewma_vol + weight_simple * simple_vol
        return (
            adaptive_vol if not pd.isna(adaptive_vol) else self.config.target_volatility
        )

    def _calculate_momentum_factor(self, price_data: pd.DataFrame) -> float:
        """Calculate momentum factor for position sizing adjustment."""
        if len(price_data) < 10:
            return 1.0

        # Calculate short-term momentum
        momentum_period = min(10, len(price_data) - 1)
        price_momentum = price_data["close"].pct_change(momentum_period).iloc[-1]

        if pd.isna(price_momentum):
            return 1.0

        # Reduce exposure during negative momentum
        if price_momentum < -0.05:  # -5% momentum
            return 0.7
        if price_momentum > 0.05:  # +5% momentum
            return 1.1
        return 1.0

    def _calculate_regime_factor(
        self,
        price_data: pd.DataFrame,
        current_volatility: float,
    ) -> float:
        """Calculate regime-based adjustment factor."""
        if len(price_data) < 60:
            return 1.0

        # Calculate long-term volatility for comparison
        returns = price_data["close"].pct_change()
        long_term_vol = returns.tail(60).std() * np.sqrt(252)

        if pd.isna(long_term_vol) or long_term_vol <= 0:
            return 1.0

        # Determine volatility regime
        vol_ratio = current_volatility / long_term_vol

        if vol_ratio > 1.5:  # High volatility regime
            return 0.6  # Reduce exposure significantly
        if vol_ratio > 1.2:  # Elevated volatility
            return 0.8  # Moderate reduction
        if vol_ratio < 0.7:  # Low volatility regime
            return 1.3  # Increase exposure
        return 1.0  # Normal regime

    def _calculate_kelly_factor(self, price_data: pd.DataFrame) -> float:
        """Calculate Kelly criterion enhancement factor with leverage adjustment."""
        if len(price_data) < self.config.lookback_period:
            return 1.0

        returns = price_data["close"].pct_change().dropna()

        if len(returns) < self.config.lookback_period:
            return 1.0

        period_returns = returns.tail(self.config.lookback_period)

        mean_return = period_returns.mean() * 252  # Annualized
        variance = period_returns.var() * 252  # Annualized

        if variance <= 0 or pd.isna(variance):
            return 1.0

        # Basic Kelly fraction
        kelly_fraction = mean_return / variance

        # Adjust for high leverage (leverage increases risk exponentially)
        # For leverage > 10x, we need to be more conservative
        leverage_adjustment = min(1.0, 10.0 / max(1.0, self.config.max_leverage))
        kelly_fraction *= leverage_adjustment

        # Confidence-based adjustment (higher confidence = higher position size)
        confidence = self._get_market_confidence(price_data)
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0 range
        kelly_fraction *= confidence_multiplier

        # Clip to configured bounds
        kelly_fraction = np.clip(kelly_fraction, 0, self.config.max_kelly_fraction)

        # Return as enhancement factor
        return 1.0 + kelly_fraction

    def _get_market_confidence(self, price_data: pd.DataFrame) -> float:
        """Calculate market confidence based on volatility and trend strength."""
        if len(price_data) < 20:
            return 0.5

        # Calculate trend strength using ADX-like measure
        high_low = price_data["high"] - price_data["low"]
        high_close = np.abs(price_data["high"] - price_data["close"].shift(1))
        low_close = np.abs(price_data["low"] - price_data["close"].shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        # Calculate directional movement
        up_move = price_data["high"] - price_data["high"].shift(1)
        down_move = price_data["low"].shift(1) - price_data["low"]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / atr

        # ADX calculation
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean()

        # Volatility-based confidence
        current_vol = self.calculate_volatility(price_data)
        vol_confidence = 1.0 - min(1.0, current_vol / self.config.target_volatility)

        # Trend-based confidence
        trend_confidence = (
            min(1.0, adx.iloc[-1] / 50.0) if not pd.isna(adx.iloc[-1]) else 0.5
        )

        # Combined confidence
        confidence = vol_confidence * 0.4 + trend_confidence * 0.6
        return np.clip(confidence, 0.1, 0.9)  # Ensure reasonable bounds

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generate_portfolio_allocation",
    )
    def generate_portfolio_allocation(
        self,
        assets_data: dict[str, pd.DataFrame],
        base_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Generate volatility-targeted portfolio allocation.

        Args:
            assets_data: Dictionary of asset price DataFrames
            base_weights: Optional base allocation weights

        Returns:
            Dictionary of adjusted allocation weights
        """
        if not assets_data:
            return {}

        # Default equal weights if not provided
        if base_weights is None:
            base_weights = {asset: 1.0 / len(assets_data) for asset in assets_data}

        adjusted_weights = {}
        total_adjustment = 0

        for asset, price_data in assets_data.items():
            base_weight = base_weights.get(asset, 0)

            if base_weight <= 0 or price_data.empty:
                adjusted_weights[asset] = 0
                continue

            # Calculate volatility targeting multiplier for this asset
            multiplier = self.calculate_position_multiplier(price_data)

            # Apply to base weight
            adjusted_weight = base_weight * multiplier
            adjusted_weights[asset] = adjusted_weight
            total_adjustment += adjusted_weight

        # Normalize weights to sum to 1
        if total_adjustment > 0:
            for asset in adjusted_weights:
                adjusted_weights[asset] /= total_adjustment

        self.logger.info(
            f"Generated volatility-targeted allocation: {adjusted_weights}",
        )
        return adjusted_weights

    def get_strategy_stats(self, price_data: pd.DataFrame) -> dict[str, float]:
        """Get strategy statistics and diagnostics."""
        current_vol = self.calculate_volatility(price_data)
        multiplier = self.calculate_position_multiplier(price_data, current_vol)

        stats = {
            "current_volatility": current_vol,
            "target_volatility": self.config.target_volatility,
            "position_multiplier": multiplier,
            "volatility_ratio": current_vol / self.config.target_volatility,
            "method": self.config.volatility_method.value,
            "lookback_period": self.config.lookback_period,
        }

        if len(price_data) >= 30:
            returns = price_data["close"].pct_change().dropna()
            stats.update(
                {
                    "sharpe_ratio": (returns.mean() / returns.std()) * np.sqrt(252)
                    if returns.std() > 0
                    else 0,
                    "max_drawdown": self._calculate_max_drawdown(price_data["close"]),
                    "win_rate": (returns > 0).mean(),
                },
            )

        return stats

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(prices) < 2:
            return 0.0

        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        return drawdown.min() if not drawdown.empty else 0.0


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = VolatilityTargetingConfig(
        target_volatility=0.12,  # 12% target
        volatility_method=VolatilityMethod.EWMA,
        lookback_period=20,
        momentum_filter=True,
        regime_adjustment=True,
    )

    # Initialize strategy
    vol_strategy = VolatilityTargetingStrategy(config)

    # Example with sample data (normally you'd use real market data)
    dates = pd.date_range("2023-01-01", "2024-01-01", freq="D")
    np.random.seed(42)
    prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, len(dates)))

    sample_data = pd.DataFrame(
        {
            "close": prices,
            "high": prices * np.random.uniform(1.0, 1.02, len(dates)),
            "low": prices * np.random.uniform(0.98, 1.0, len(dates)),
            "volume": np.random.randint(1000, 10000, len(dates)),
        },
        index=dates,
    )

    # Calculate position multiplier
    multiplier = vol_strategy.calculate_position_multiplier(sample_data)
    print(f"Position multiplier: {multiplier:.3f}")

    # Get strategy statistics
    stats = vol_strategy.get_strategy_stats(sample_data)
    for key, value in stats.items():
        print(f"{key}: {value}")
