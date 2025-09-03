"""Feature engineering components.

This module contains specialized components for feature engineering
including technical indicators, interactions, and regime-aware features.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.utils.logger import system_logger


class TechnicalIndicatorEngine:
    """Engine for creating technical indicators."""
    
    def __init__(self, lookback_periods: Dict[str, List[int]]):
        """Initialize technical indicator engine.
        
        Args:
            lookback_periods: Dictionary of lookback periods by type
        """
        self.lookback_periods = lookback_periods
        self.logger = system_logger.getChild("TechnicalIndicatorEngine")
        
        # Default periods if not specified
        self.default_periods = {
            "short": [5, 10, 20],
            "medium": [50, 100],
            "long": [200]
        }
        
    def apply_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply technical indicators to data.
        
        Args:
            data: Market data
            
        Returns:
            Data with technical indicators
        """
        # Price-based indicators
        data = self._add_moving_averages(data)
        data = self._add_price_channels(data)
        data = self._add_momentum_indicators(data)
        
        # Volatility indicators
        data = self._add_volatility_indicators(data)
        
        # Volume indicators
        if "volume" in data.columns:
            data = self._add_volume_indicators(data)
        
        # Pattern recognition
        data = self._add_pattern_features(data)
        
        return data
    
    def _add_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add moving average indicators."""
        # Simple moving averages
        all_periods = []
        for period_type in ["short", "medium", "long"]:
            periods = self.lookback_periods.get(
                period_type, 
                self.default_periods[period_type]
            )
            all_periods.extend(periods)
        
        for period in all_periods:
            # SMA
            data[f"feature_sma_{period}"] = data["close"].rolling(period).mean()
            data[f"feature_sma_{period}_ratio"] = data["close"] / data[f"feature_sma_{period}"]
            
            # EMA
            data[f"feature_ema_{period}"] = data["close"].ewm(span=period).mean()
            data[f"feature_ema_{period}_ratio"] = data["close"] / data[f"feature_ema_{period}"]
        
        # Moving average crossovers
        short_periods = self.lookback_periods.get("short", self.default_periods["short"])
        medium_periods = self.lookback_periods.get("medium", self.default_periods["medium"])
        
        if short_periods and medium_periods:
            short_ma = data[f"feature_sma_{short_periods[0]}"]
            long_ma = data[f"feature_sma_{medium_periods[0]}"]
            data["feature_ma_crossover"] = (short_ma > long_ma).astype(int)
            data["feature_ma_spread"] = (short_ma - long_ma) / long_ma
        
        return data
    
    def _add_price_channels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price channel indicators."""
        # Bollinger Bands
        for period in [20, 50]:
            sma = data["close"].rolling(period).mean()
            std = data["close"].rolling(period).std()
            
            data[f"feature_bb_upper_{period}"] = sma + 2 * std
            data[f"feature_bb_lower_{period}"] = sma - 2 * std
            data[f"feature_bb_width_{period}"] = (
                data[f"feature_bb_upper_{period}"] - data[f"feature_bb_lower_{period}"]
            ) / sma
            data[f"feature_bb_position_{period}"] = (
                (data["close"] - data[f"feature_bb_lower_{period}"]) / 
                (data[f"feature_bb_upper_{period}"] - data[f"feature_bb_lower_{period}"])
            )
        
        # Keltner Channels
        if all(col in data.columns for col in ["high", "low"]):
            for period in [20]:
                typical_price = (data["high"] + data["low"] + data["close"]) / 3
                ema = typical_price.ewm(span=period).mean()
                atr = self._calculate_atr(data, period)
                
                data[f"feature_kc_upper_{period}"] = ema + 2 * atr
                data[f"feature_kc_lower_{period}"] = ema - 2 * atr
                data[f"feature_kc_position_{period}"] = (
                    (data["close"] - data[f"feature_kc_lower_{period}"]) /
                    (data[f"feature_kc_upper_{period}"] - data[f"feature_kc_lower_{period}"])
                )
        
        return data
    
    def _add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI
        for period in [14, 21]:
            data[f"feature_rsi_{period}"] = self._calculate_rsi(data["close"], period)
        
        # MACD
        data["feature_macd"] = (
            data["close"].ewm(span=12).mean() - 
            data["close"].ewm(span=26).mean()
        )
        data["feature_macd_signal"] = data["feature_macd"].ewm(span=9).mean()
        data["feature_macd_histogram"] = data["feature_macd"] - data["feature_macd_signal"]
        
        # Rate of Change
        for period in [10, 20]:
            data[f"feature_roc_{period}"] = (
                (data["close"] - data["close"].shift(period)) / 
                data["close"].shift(period) * 100
            )
        
        # Stochastic
        if all(col in data.columns for col in ["high", "low"]):
            for period in [14]:
                lowest_low = data["low"].rolling(period).min()
                highest_high = data["high"].rolling(period).max()
                data[f"feature_stoch_k_{period}"] = (
                    100 * (data["close"] - lowest_low) / (highest_high - lowest_low)
                )
                data[f"feature_stoch_d_{period}"] = (
                    data[f"feature_stoch_k_{period}"].rolling(3).mean()
                )
        
        return data
    
    def _add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        # Historical volatility
        returns = data["close"].pct_change()
        for period in [10, 20, 50]:
            data[f"feature_volatility_{period}"] = returns.rolling(period).std()
            data[f"feature_volatility_ratio_{period}"] = (
                data[f"feature_volatility_{period}"] / 
                data[f"feature_volatility_{period}"].rolling(period).mean()
            )
        
        # ATR
        if all(col in data.columns for col in ["high", "low"]):
            for period in [14, 20]:
                data[f"feature_atr_{period}"] = self._calculate_atr(data, period)
                data[f"feature_atr_ratio_{period}"] = (
                    data[f"feature_atr_{period}"] / data["close"]
                )
        
        # Parkinson volatility
        if all(col in data.columns for col in ["high", "low"]):
            hl_ratio = np.log(data["high"] / data["low"])
            data["feature_parkinson_vol"] = hl_ratio.rolling(20).apply(
                lambda x: np.sqrt(np.mean(x**2) / (4 * np.log(2)))
            )
        
        return data
    
    def _add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        # Volume moving averages
        for period in [10, 20]:
            data[f"feature_volume_sma_{period}"] = data["volume"].rolling(period).mean()
            data[f"feature_volume_ratio_{period}"] = (
                data["volume"] / data[f"feature_volume_sma_{period}"]
            )
        
        # On-Balance Volume
        data["feature_obv"] = (np.sign(data["close"].diff()) * data["volume"]).cumsum()
        data["feature_obv_sma"] = data["feature_obv"].rolling(20).mean()
        
        # Volume-Price Trend
        data["feature_vpt"] = (
            (data["close"].diff() / data["close"].shift()) * data["volume"]
        ).cumsum()
        
        # Money Flow Index
        if all(col in data.columns for col in ["high", "low"]):
            typical_price = (data["high"] + data["low"] + data["close"]) / 3
            money_flow = typical_price * data["volume"]
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
            
            positive_mf = positive_flow.rolling(14).sum()
            negative_mf = negative_flow.rolling(14).sum()
            
            mfi_ratio = positive_mf / negative_mf
            data["feature_mfi"] = 100 - (100 / (1 + mfi_ratio))
        
        return data
    
    def _add_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features."""
        # Price patterns
        data["feature_higher_high"] = (
            (data["high"] > data["high"].shift(1)) & 
            (data["high"].shift(1) > data["high"].shift(2))
        ).astype(int)
        
        data["feature_lower_low"] = (
            (data["low"] < data["low"].shift(1)) & 
            (data["low"].shift(1) < data["low"].shift(2))
        ).astype(int)
        
        # Support/Resistance levels
        for period in [20, 50]:
            data[f"feature_resistance_{period}"] = data["high"].rolling(period).max()
            data[f"feature_support_{period}"] = data["low"].rolling(period).min()
            data[f"feature_sr_position_{period}"] = (
                (data["close"] - data[f"feature_support_{period}"]) /
                (data[f"feature_resistance_{period}"] - data[f"feature_support_{period}"])
            )
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()


class FeatureInteractionEngine:
    """Engine for creating feature interactions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize interaction engine.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config
        self.logger = system_logger.getChild("FeatureInteractionEngine")
        
    async def create_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create feature interactions.
        
        Args:
            data: Data with features
            
        Returns:
            Data with interaction features
        """
        # Price-volume interactions
        data = self._create_price_volume_interactions(data)
        
        # Momentum-volatility interactions
        data = self._create_momentum_volatility_interactions(data)
        
        # Technical indicator interactions
        data = self._create_indicator_interactions(data)
        
        # Cross-timeframe interactions
        data = self._create_cross_timeframe_interactions(data)
        
        return data
    
    def _create_price_volume_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create price-volume interaction features."""
        if "volume" in data.columns:
            # Price change on volume
            if "feature_returns" not in data.columns:
                data["feature_returns"] = data["close"].pct_change()
            
            data["feature_price_volume_interaction"] = (
                data["feature_returns"] * np.log1p(data["volume"])
            )
            
            # Volume-weighted price momentum
            if "feature_volume_ratio_20" in data.columns:
                data["feature_volume_weighted_momentum"] = (
                    data["feature_returns"].rolling(10).mean() * 
                    data["feature_volume_ratio_20"]
                )
        
        return data
    
    def _create_momentum_volatility_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create momentum-volatility interactions."""
        # RSI-volatility interaction
        if "feature_rsi_14" in data.columns and "feature_volatility_20" in data.columns:
            data["feature_rsi_volatility_interaction"] = (
                data["feature_rsi_14"] * data["feature_volatility_20"]
            )
        
        # MACD-ATR interaction
        if "feature_macd" in data.columns and "feature_atr_14" in data.columns:
            data["feature_macd_atr_interaction"] = (
                data["feature_macd"] / (data["feature_atr_14"] + 1e-8)
            )
        
        return data
    
    def _create_indicator_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interactions between technical indicators."""
        # RSI-Stochastic interaction
        if "feature_rsi_14" in data.columns and "feature_stoch_k_14" in data.columns:
            data["feature_rsi_stoch_interaction"] = (
                data["feature_rsi_14"] * data["feature_stoch_k_14"] / 100
            )
        
        # Bollinger Band - Keltner Channel squeeze
        if all(col in data.columns for col in ["feature_bb_width_20", "feature_kc_upper_20"]):
            bb_width = data["feature_bb_width_20"]
            kc_width = (
                data["feature_kc_upper_20"] - data.get("feature_kc_lower_20", 0)
            ) / data["close"]
            data["feature_bb_kc_squeeze"] = bb_width / (kc_width + 1e-8)
        
        # Moving average divergence
        ma_pairs = [
            ("feature_sma_10", "feature_sma_50"),
            ("feature_ema_10", "feature_ema_50"),
            ("feature_sma_20", "feature_sma_100")
        ]
        
        for fast_ma, slow_ma in ma_pairs:
            if fast_ma in data.columns and slow_ma in data.columns:
                interaction_name = f"feature_{fast_ma}_{slow_ma}_divergence"
                data[interaction_name] = (
                    data[fast_ma] - data[slow_ma]
                ) / data[slow_ma]
        
        return data
    
    def _create_cross_timeframe_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create cross-timeframe interactions."""
        # Short vs long term momentum
        if "feature_roc_10" in data.columns and "feature_roc_20" in data.columns:
            data["feature_momentum_divergence"] = (
                data["feature_roc_10"] - data["feature_roc_20"]
            )
        
        # Multi-timeframe volatility ratio
        vol_pairs = [
            ("feature_volatility_10", "feature_volatility_50"),
            ("feature_volatility_20", "feature_volatility_50")
        ]
        
        for short_vol, long_vol in vol_pairs:
            if short_vol in data.columns and long_vol in data.columns:
                interaction_name = f"{short_vol}_{long_vol}_ratio"
                data[interaction_name] = data[short_vol] / (data[long_vol] + 1e-8)
        
        return data


class RegimeAwareFeatureEngine:
    """Engine for creating regime-aware features."""
    
    def __init__(self):
        """Initialize regime-aware feature engine."""
        self.logger = system_logger.getChild("RegimeAwareFeatureEngine")
        
    def create_regime_features(
        self,
        data: pd.DataFrame,
        regime_characteristics: Dict[str, Any]
    ) -> pd.DataFrame:
        """Create regime-aware features.
        
        Args:
            data: Data with regime labels
            regime_characteristics: Characteristics of each regime
            
        Returns:
            Data with regime features
        """
        if "regime_label" not in data.columns:
            return data
        
        # One-hot encode regimes
        regime_dummies = pd.get_dummies(data["regime_label"], prefix="feature_regime")
        data = pd.concat([data, regime_dummies], axis=1)
        
        # Regime transition features
        data = self._add_regime_transition_features(data)
        
        # Regime-specific statistics
        data = self._add_regime_statistics(data, regime_characteristics)
        
        # Regime persistence features
        data = self._add_regime_persistence_features(data)
        
        return data
    
    def _add_regime_transition_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add regime transition features."""
        # Regime change indicator
        data["feature_regime_changed"] = (
            data["regime_label"] != data["regime_label"].shift(1)
        ).astype(int)
        
        # Previous regime
        data["feature_prev_regime"] = data["regime_label"].shift(1)
        
        # Next regime (look-ahead bias, only for analysis)
        # data["feature_next_regime"] = data["regime_label"].shift(-1)
        
        # Time since regime change
        regime_groups = (data["regime_label"] != data["regime_label"].shift()).cumsum()
        data["feature_time_in_regime"] = data.groupby(regime_groups).cumcount()
        
        # Regime duration (backwards looking)
        regime_durations = data.groupby(regime_groups).size()
        data["feature_regime_duration"] = data.groupby(regime_groups).transform(
            lambda x: np.arange(len(x), 0, -1)
        )
        
        return data
    
    def _add_regime_statistics(
        self,
        data: pd.DataFrame,
        regime_characteristics: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add regime-specific statistics."""
        # Map regime characteristics to features
        for regime_key, chars in regime_characteristics.items():
            if isinstance(chars, dict) and regime_key.startswith("regime_"):
                regime_id = int(regime_key.split("_")[1])
                
                # Add regime volatility
                if "volatility_20_mean" in chars:
                    mask = data["regime_label"] == regime_id
                    data.loc[mask, "feature_regime_volatility"] = chars["volatility_20_mean"]
                
                # Add regime return expectation
                if "returns_mean" in chars:
                    mask = data["regime_label"] == regime_id
                    data.loc[mask, "feature_regime_return_expectation"] = chars["returns_mean"]
        
        # Fill NaN values with overall statistics
        if "feature_regime_volatility" in data.columns:
            data["feature_regime_volatility"].fillna(
                data["feature_volatility_20"].mean(), inplace=True
            )
        
        return data
    
    def _add_regime_persistence_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add regime persistence features."""
        # Calculate regime stability score
        window = 20
        regime_changes = data["feature_regime_changed"].rolling(window).sum()
        data["feature_regime_stability"] = 1 - (regime_changes / window)
        
        # Regime concentration
        regime_counts = data["regime_label"].rolling(window).apply(
            lambda x: pd.Series(x).value_counts().iloc[0] / len(x)
        )
        data["feature_regime_concentration"] = regime_counts
        
        return data