# src/tactician/exit_strategy_feature_engineering.py

"""
Exit Strategy Feature Engineering for Tactician.
Creates comprehensive features for trend reversal detection and exit timing,
building on existing step6 features with additional exit-specific features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.decorators import guard_dataframe_nulls, with_tracing_span


class ExitStrategyFeatureEngineering:
    """
    Comprehensive exit strategy feature engineering system.
    Creates features for trend reversal detection and optimal exit timing.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize exit strategy feature engineering.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("ExitStrategyFeatureEngineering")
        
        # Load configuration
        self.exit_config = config.get("exit_strategy", {})
        self.feature_config = self.exit_config.get("feature_engineering", {})
        
        # Timeframe configurations
        self.timeframes = self.feature_config.get("timeframes", ["1m", "5m", "15m"])
        self.lookback_periods = self.feature_config.get("lookback_periods", [10, 20, 50, 100])
        
        # Feature categories
        self.feature_categories = [
            "momentum_reversal",
            "volatility_reversal", 
            "volume_reversal",
            "support_resistance",
            "trend_strength",
            "profit_decay",
            "time_decay",
            "market_regime",
            "entry_timing",
            "exit_timing"
        ]

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=pd.DataFrame(),
        context="exit strategy feature engineering initialization"
    )
    async def initialize(self) -> bool:
        """
        Initialize the exit strategy feature engineering system.

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("🔧 Exit strategy feature engineering system initialized")
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("❌ Invalid exit strategy feature engineering configuration")
                return False
                
            self.logger.info("✅ Exit strategy feature engineering initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Exit strategy feature engineering initialization failed: {e}")
            return False

    def _validate_configuration(self) -> bool:
        """
        Validate configuration parameters.

        Returns:
            bool: True if configuration is valid
        """
        try:
            if not self.timeframes:
                self.logger.error("No timeframes specified")
                return False
                
            if not self.lookback_periods:
                self.logger.error("No lookback periods specified")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    @guard_dataframe_nulls
    @with_tracing_span("exit_strategy_apply_all")
    async def apply_all(self, df: pd.DataFrame, position_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Apply all exit strategy feature engineering.

        Args:
            df: Input dataframe with OHLCV data
            position_context: Optional position context for profit/time features

        Returns:
            pd.DataFrame: DataFrame with exit strategy features added
        """
        try:
            self.logger.info("🚀 Applying exit strategy feature engineering")
            self.logger.info(f"   - Input shape: {df.shape}")
            self.logger.info(f"   - Feature categories: {self.feature_categories}")
            
            # Create copy to avoid modifying original
            result_df = df.copy()
            
            # Apply momentum reversal features
            result_df = await self._apply_momentum_reversal_features(result_df)
            
            # Apply volatility reversal features
            result_df = await self._apply_volatility_reversal_features(result_df)
            
            # Apply volume reversal features
            result_df = await self._apply_volume_reversal_features(result_df)
            
            # Apply support/resistance features
            result_df = await self._apply_support_resistance_features(result_df)
            
            # Apply trend strength features
            result_df = await self._apply_trend_strength_features(result_df)
            
            # Apply profit decay features (if position context provided)
            if position_context:
                result_df = await self._apply_profit_decay_features(result_df, position_context)
            
            # Apply time decay features
            result_df = await self._apply_time_decay_features(result_df)
            
            # Apply market regime features
            result_df = await self._apply_market_regime_features(result_df)
            
            # Apply entry timing features
            result_df = await self._apply_entry_timing_features(result_df)
            
            # Apply exit timing features
            result_df = await self._apply_exit_timing_features(result_df)
            
            # Calculate feature statistics
            features_added = len(result_df.columns) - len(df.columns)
            self.logger.info("✅ Exit strategy feature engineering completed")
            self.logger.info(f"   - Output shape: {result_df.shape}")
            self.logger.info(f"   - Features added: {features_added}")
            self.logger.info(f"   - Total features: {len(result_df.columns)}")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"❌ Exit strategy feature engineering failed: {e}")
            return df

    async def _apply_momentum_reversal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply momentum reversal detection features."""
        try:
            self.logger.info("   ✅ Applied momentum_reversal features")
            
            # RSI divergence features
            for period in [14, 21]:
                df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
                df[f'rsi_divergence_{period}'] = self._calculate_rsi_divergence(df, period)
            
            # MACD reversal features
            df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])
            df['macd_reversal_signal'] = self._calculate_macd_reversal(df)
            
            # Stochastic reversal features
            df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df)
            df['stoch_reversal_signal'] = self._calculate_stochastic_reversal(df)
            
            # Price momentum reversal
            for period in [5, 10, 20]:
                df[f'momentum_reversal_{period}'] = self._calculate_momentum_reversal(df, period)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Momentum reversal features failed: {e}")
            return df

    async def _apply_volatility_reversal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply volatility reversal detection features."""
        try:
            self.logger.info("   ✅ Applied volatility_reversal features")
            
            # ATR-based volatility reversal
            for period in [14, 21]:
                df[f'atr_{period}'] = self._calculate_atr(df, period)
                df[f'atr_reversal_{period}'] = self._calculate_atr_reversal(df, period)
            
            # Bollinger Bands reversal
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df)
            df['bb_reversal_signal'] = self._calculate_bb_reversal(df)
            
            # Volatility contraction/expansion
            df['volatility_regime'] = self._calculate_volatility_regime(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Volatility reversal features failed: {e}")
            return df

    async def _apply_volume_reversal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply volume reversal detection features."""
        try:
            self.logger.info("   ✅ Applied volume_reversal features")
            
            # Volume momentum
            for period in [10, 20]:
                df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
                df[f'volume_momentum_{period}'] = df['volume'] / df[f'volume_sma_{period}']
            
            # Volume price trend
            df['vpt'] = self._calculate_vpt(df)
            df['vpt_reversal'] = self._calculate_vpt_reversal(df)
            
            # Volume rate of change
            df['volume_roc'] = self._calculate_volume_roc(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Volume reversal features failed: {e}")
            return df

    async def _apply_support_resistance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply support/resistance level features."""
        try:
            self.logger.info("   ✅ Applied support_resistance features")
            
            # Dynamic support/resistance levels
            for period in [20, 50]:
                df[f'support_{period}'] = df['low'].rolling(period).min()
                df[f'resistance_{period}'] = df['high'].rolling(period).max()
                df[f'price_to_support_{period}'] = (df['close'] - df[f'support_{period}']) / df[f'support_{period}']
                df[f'price_to_resistance_{period}'] = (df[f'resistance_{period}'] - df['close']) / df['close']
            
            # Pivot points
            df['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
            df['r1'] = 2 * df['pivot_point'] - df['low']
            df['s1'] = 2 * df['pivot_point'] - df['high']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Support/resistance features failed: {e}")
            return df

    async def _apply_trend_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply trend strength measurement features."""
        try:
            self.logger.info("   ✅ Applied trend_strength features")
            
            # ADX trend strength
            df['adx'] = self._calculate_adx(df)
            df['trend_strength'] = self._calculate_trend_strength(df)
            
            # Moving average alignment
            for period in [10, 20, 50]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
            
            df['ma_alignment'] = self._calculate_ma_alignment(df)
            
            # Linear regression slope
            for period in [20, 50]:
                df[f'linreg_slope_{period}'] = self._calculate_linear_regression_slope(df, period)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Trend strength features failed: {e}")
            return df

    async def _apply_profit_decay_features(self, df: pd.DataFrame, position_context: Dict[str, Any]) -> pd.DataFrame:
        """Apply profit decay features based on position context."""
        try:
            self.logger.info("   ✅ Applied profit_decay features")
            
            # Extract position information
            entry_price = position_context.get('entry_price', df['close'].iloc[-1])
            entry_time = position_context.get('entry_time', df.index[-1])
            current_pnl = position_context.get('current_pnl', 0.0)
            
            # Profit decay indicators
            df['profit_decay_rate'] = self._calculate_profit_decay_rate(df, entry_price, current_pnl)
            df['profit_preservation_score'] = self._calculate_profit_preservation_score(df, entry_price)
            
            # Time-based profit decay
            df['time_since_entry'] = (df.index - entry_time).total_seconds() / 60  # minutes
            df['profit_time_decay'] = self._calculate_profit_time_decay(df, current_pnl)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Profit decay features failed: {e}")
            return df

    async def _apply_time_decay_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply time-based decay features."""
        try:
            self.logger.info("   ✅ Applied time_decay features")
            
            # Time-based momentum decay
            for period in [5, 15, 30]:  # minutes
                df[f'time_decay_momentum_{period}'] = self._calculate_time_decay_momentum(df, period)
            
            # Session time features
            df['session_progress'] = self._calculate_session_progress(df)
            df['time_of_day_volatility'] = self._calculate_time_of_day_volatility(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Time decay features failed: {e}")
            return df

    async def _apply_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply market regime detection features."""
        try:
            self.logger.info("   ✅ Applied market_regime features")
            
            # Volatility regime
            df['volatility_regime'] = self._calculate_volatility_regime(df)
            
            # Trend regime
            df['trend_regime'] = self._calculate_trend_regime(df)
            
            # Market structure
            df['market_structure'] = self._calculate_market_structure(df)
            
            # Regime transition probability
            df['regime_transition_prob'] = self._calculate_regime_transition_probability(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Market regime features failed: {e}")
            return df

    async def _apply_entry_timing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply entry timing optimization features."""
        try:
            self.logger.info("   ✅ Applied entry_timing features")
            
            # Multi-timeframe alignment
            for tf in self.timeframes:
                df[f'{tf}_alignment_score'] = self._calculate_timeframe_alignment(df, tf)
            
            # Entry timing signals
            df['optimal_entry_timing'] = self._calculate_optimal_entry_timing(df)
            df['entry_timing_confidence'] = self._calculate_entry_timing_confidence(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Entry timing features failed: {e}")
            return df

    async def _apply_exit_timing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply exit timing optimization features."""
        try:
            self.logger.info("   ✅ Applied exit_timing features")
            
            # Exit urgency indicators
            df['exit_urgency_score'] = self._calculate_exit_urgency_score(df)
            df['reversal_probability'] = self._calculate_reversal_probability(df)
            
            # Optimal exit timing
            df['optimal_exit_timing'] = self._calculate_optimal_exit_timing(df)
            df['exit_timing_confidence'] = self._calculate_exit_timing_confidence(df)
            
            # Risk-adjusted exit signals
            df['risk_adjusted_exit_signal'] = self._calculate_risk_adjusted_exit_signal(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Exit timing features failed: {e}")
            return df

    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_rsi_divergence(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate RSI divergence."""
        rsi = self._calculate_rsi(df['close'], period)
        
        # Price and RSI peaks/troughs
        price_peaks = df['close'].rolling(5, center=True).apply(lambda x: 1 if x.iloc[2] == x.max() else 0)
        rsi_peaks = rsi.rolling(5, center=True).apply(lambda x: 1 if x.iloc[2] == x.max() else 0)
        
        divergence = np.where(
            (price_peaks == 1) & (rsi_peaks == 0),
            -1,  # Bearish divergence
            np.where(
                (price_peaks == 0) & (rsi_peaks == 1),
                1,  # Bullish divergence
                0
            )
        )
        
        return pd.Series(divergence, index=df.index)

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    def _calculate_macd_reversal(self, df: pd.DataFrame) -> pd.Series:
        """Calculate MACD reversal signals."""
        macd, signal, histogram = self._calculate_macd(df['close'])
        
        # MACD crossover signals
        macd_cross_up = (macd > signal) & (macd.shift(1) <= signal.shift(1))
        macd_cross_down = (macd < signal) & (macd.shift(1) >= signal.shift(1))
        
        reversal_signal = np.where(macd_cross_up, 1, np.where(macd_cross_down, -1, 0))
        return pd.Series(reversal_signal, index=df.index)

    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator."""
        lowest_low = df['low'].rolling(k_period).min()
        highest_high = df['high'].rolling(k_period).max()
        
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(d_period).mean()
        
        return k_percent, d_percent

    def _calculate_stochastic_reversal(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Stochastic reversal signals."""
        k, d = self._calculate_stochastic(df)
        
        # Overbought/oversold reversals
        overbought_reversal = (k > 80) & (k.shift(1) <= 80) & (k < k.shift(1))
        oversold_reversal = (k < 20) & (k.shift(1) >= 20) & (k > k.shift(1))
        
        reversal_signal = np.where(overbought_reversal, -1, np.where(oversold_reversal, 1, 0))
        return pd.Series(reversal_signal, index=df.index)

    def _calculate_momentum_reversal(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate momentum reversal signals."""
        momentum = df['close'] / df['close'].shift(period) - 1
        momentum_ma = momentum.rolling(period).mean()
        
        # Momentum reversal when current momentum crosses below/above its MA
        reversal_signal = np.where(
            (momentum < momentum_ma) & (momentum.shift(1) >= momentum_ma.shift(1)),
            -1,  # Bearish reversal
            np.where(
                (momentum > momentum_ma) & (momentum.shift(1) <= momentum_ma.shift(1)),
                1,  # Bullish reversal
                0
            )
        )
        
        return pd.Series(reversal_signal, index=df.index)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        
        return atr

    def _calculate_atr_reversal(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR-based reversal signals."""
        atr = self._calculate_atr(df, period)
        atr_ma = atr.rolling(period).mean()
        
        # Volatility expansion/contraction
        volatility_expansion = atr > atr_ma * 1.2
        volatility_contraction = atr < atr_ma * 0.8
        
        reversal_signal = np.where(volatility_expansion, 1, np.where(volatility_contraction, -1, 0))
        return pd.Series(reversal_signal, index=df.index)

    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band

    def _calculate_bb_reversal(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Bollinger Bands reversal signals."""
        upper, middle, lower = self._calculate_bollinger_bands(df)
        
        # Price touching bands
        touch_upper = df['close'] >= upper * 0.99
        touch_lower = df['close'] <= lower * 1.01
        
        reversal_signal = np.where(touch_upper, -1, np.where(touch_lower, 1, 0))
        return pd.Series(reversal_signal, index=df.index)

    def _calculate_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility regime classification."""
        atr = self._calculate_atr(df, 14)
        atr_ma = atr.rolling(50).mean()
        
        # Classify volatility regimes
        regime = np.where(atr > atr_ma * 1.5, 2,  # High volatility
                         np.where(atr < atr_ma * 0.7, 0,  # Low volatility
                                 1))  # Normal volatility
        
        return pd.Series(regime, index=df.index)

    def _calculate_vpt(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend."""
        price_change = df['close'].pct_change()
        vpt = (price_change * df['volume']).cumsum()
        return vpt

    def _calculate_vpt_reversal(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VPT reversal signals."""
        vpt = self._calculate_vpt(df)
        vpt_ma = vpt.rolling(20).mean()
        
        # VPT crossing its moving average
        reversal_signal = np.where(
            (vpt > vpt_ma) & (vpt.shift(1) <= vpt_ma.shift(1)),
            1,  # Bullish reversal
            np.where(
                (vpt < vpt_ma) & (vpt.shift(1) >= vpt_ma.shift(1)),
                -1,  # Bearish reversal
                0
            )
        )
        
        return pd.Series(reversal_signal, index=df.index)

    def _calculate_volume_roc(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate Volume Rate of Change."""
        volume_roc = df['volume'].pct_change(period)
        return volume_roc

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        # Simplified ADX calculation
        tr = self._calculate_atr(df, 1)  # True Range
        dm_plus = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                          np.maximum(df['high'] - df['high'].shift(1), 0), 0)
        dm_minus = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                           np.maximum(df['low'].shift(1) - df['low'], 0), 0)
        
        di_plus = 100 * pd.Series(dm_plus).rolling(period).mean() / tr.rolling(period).mean()
        di_minus = 100 * pd.Series(dm_minus).rolling(period).mean() / tr.rolling(period).mean()
        
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).rolling(period).mean()
        
        return adx

    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength indicator."""
        adx = self._calculate_adx(df)
        
        # Classify trend strength
        strength = np.where(adx > 25, 2,  # Strong trend
                           np.where(adx > 20, 1,  # Moderate trend
                                   0))  # Weak trend
        
        return pd.Series(strength, index=df.index)

    def _calculate_ma_alignment(self, df: pd.DataFrame) -> pd.Series:
        """Calculate moving average alignment score."""
        sma_10 = df['close'].rolling(10).mean()
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        
        # Count aligned moving averages
        alignment_score = np.where(
            (sma_10 > sma_20) & (sma_20 > sma_50), 3,  # Bullish alignment
            np.where(
                (sma_10 < sma_20) & (sma_20 < sma_50), -3,  # Bearish alignment
                np.where(
                    (sma_10 > sma_20) | (sma_20 > sma_50), 1,  # Mixed bullish
                    -1  # Mixed bearish
                )
            )
        )
        
        return pd.Series(alignment_score, index=df.index)

    def _calculate_linear_regression_slope(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate linear regression slope for trend direction."""
        def rolling_slope(x):
            if len(x) < 2:
                return np.nan
            y = np.arange(len(x))
            slope = np.polyfit(y, x, 1)[0]
            return slope
        
        slope = df['close'].rolling(period).apply(rolling_slope)
        return slope

    def _calculate_profit_decay_rate(self, df: pd.DataFrame, entry_price: float, current_pnl: float) -> pd.Series:
        """Calculate profit decay rate."""
        # Simplified profit decay calculation
        price_change = (df['close'] - entry_price) / entry_price
        profit_decay = np.where(price_change > 0, 
                               np.maximum(0, price_change - current_pnl),
                               0)
        return pd.Series(profit_decay, index=df.index)

    def _calculate_profit_preservation_score(self, df: pd.DataFrame, entry_price: float) -> pd.Series:
        """Calculate profit preservation score."""
        price_change = (df['close'] - entry_price) / entry_price
        max_profit = price_change.expanding().max()
        
        # Score based on current profit vs max profit
        preservation_score = np.where(
            price_change > 0,
            price_change / np.maximum(max_profit, 0.001),  # Avoid division by zero
            0
        )
        
        return pd.Series(preservation_score, index=df.index)

    def _calculate_profit_time_decay(self, df: pd.DataFrame, current_pnl: float) -> pd.Series:
        """Calculate time-based profit decay."""
        # Simplified time decay model
        time_decay = np.exp(-df['time_since_entry'] / 60)  # Decay over 1 hour
        profit_decay = current_pnl * (1 - time_decay)
        return profit_decay

    def _calculate_time_decay_momentum(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate time-decay adjusted momentum."""
        momentum = df['close'].pct_change(period)
        time_decay = np.exp(-period / 30)  # Decay factor
        decayed_momentum = momentum * time_decay
        return decayed_momentum

    def _calculate_session_progress(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trading session progress."""
        # Simplified session progress (assuming 24-hour market)
        hour = df.index.hour
        session_progress = hour / 24
        return session_progress

    def _calculate_time_of_day_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate time-of-day volatility pattern."""
        hour = df.index.hour
        
        # Simplified volatility pattern (higher during active hours)
        volatility_pattern = np.where(
            (hour >= 8) & (hour <= 16), 1.2,  # High volatility hours
            np.where(
                (hour >= 0) & (hour <= 6), 0.8,  # Low volatility hours
                1.0  # Normal volatility
            )
        )
        
        return pd.Series(volatility_pattern, index=df.index)

    def _calculate_trend_regime(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend regime classification."""
        # Use multiple indicators for trend regime
        adx = self._calculate_adx(df)
        ma_alignment = self._calculate_ma_alignment(df)
        
        # Combine indicators for regime classification
        trend_regime = np.where(
            (adx > 25) & (ma_alignment > 0), 2,  # Strong uptrend
            np.where(
                (adx > 25) & (ma_alignment < 0), -2,  # Strong downtrend
                np.where(
                    (adx > 20) & (ma_alignment > 0), 1,  # Moderate uptrend
                    np.where(
                        (adx > 20) & (ma_alignment < 0), -1,  # Moderate downtrend
                        0  # Sideways
                    )
                )
            )
        )
        
        return pd.Series(trend_regime, index=df.index)

    def _calculate_market_structure(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market structure (higher highs/lower lows)."""
        # Simplified market structure analysis
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        
        # Higher highs and higher lows
        higher_highs = df['high'] > high_20.shift(1)
        higher_lows = df['low'] > low_20.shift(1)
        lower_highs = df['high'] < high_20.shift(1)
        lower_lows = df['low'] < low_20.shift(1)
        
        # Market structure classification
        structure = np.where(
            higher_highs & higher_lows, 2,  # Uptrend structure
            np.where(
                lower_highs & lower_lows, -2,  # Downtrend structure
                np.where(
                    higher_highs | higher_lows, 1,  # Mixed bullish
                    -1  # Mixed bearish
                )
            )
        )
        
        return pd.Series(structure, index=df.index)

    def _calculate_regime_transition_probability(self, df: pd.DataFrame) -> pd.Series:
        """Calculate probability of regime transition."""
        # Simplified regime transition probability
        volatility_regime = self._calculate_volatility_regime(df)
        trend_regime = self._calculate_trend_regime(df)
        
        # Transition probability based on regime changes
        vol_change = volatility_regime.diff().abs()
        trend_change = trend_regime.diff().abs()
        
        transition_prob = (vol_change + trend_change) / 4  # Normalize to 0-1
        return transition_prob

    def _calculate_timeframe_alignment(self, df: pd.DataFrame, timeframe: str) -> pd.Series:
        """Calculate multi-timeframe alignment score."""
        # Simplified timeframe alignment (placeholder)
        # In practice, this would compare signals across different timeframes
        
        # For now, use a simple momentum-based alignment
        if timeframe == "1m":
            period = 5
        elif timeframe == "5m":
            period = 20
        elif timeframe == "15m":
            period = 60
        else:
            period = 20
        
        momentum = df['close'].pct_change(period)
        alignment_score = np.where(momentum > 0, 1, -1)
        
        return pd.Series(alignment_score, index=df.index)

    def _calculate_optimal_entry_timing(self, df: pd.DataFrame) -> pd.Series:
        """Calculate optimal entry timing score."""
        # Combine multiple factors for entry timing
        rsi = self._calculate_rsi(df['close'], 14)
        macd, signal, _ = self._calculate_macd(df['close'])
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df)
        
        # Entry timing score
        rsi_score = np.where((rsi > 30) & (rsi < 70), 1, 0)  # Neutral RSI
        macd_score = np.where(macd > signal, 1, -1)  # MACD bullish/bearish
        bb_score = np.where(
            df['close'] < bb_lower, 1,  # Oversold
            np.where(df['close'] > bb_upper, -1, 0)  # Overbought
        )
        
        entry_score = (rsi_score + macd_score + bb_score) / 3
        return pd.Series(entry_score, index=df.index)

    def _calculate_entry_timing_confidence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate entry timing confidence."""
        # Confidence based on signal strength and consistency
        optimal_timing = self._calculate_optimal_entry_timing(df)
        volatility = self._calculate_atr(df, 14) / df['close']
        
        # Higher confidence for stronger signals and lower volatility
        confidence = np.abs(optimal_timing) * (1 - volatility)
        return confidence

    def _calculate_exit_urgency_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate exit urgency score."""
        # Combine reversal signals for exit urgency
        rsi_divergence = self._calculate_rsi_divergence(df, 14)
        macd_reversal = self._calculate_macd_reversal(df)
        bb_reversal = self._calculate_bb_reversal(df)
        
        # Exit urgency based on reversal signals
        urgency_score = (np.abs(rsi_divergence) + np.abs(macd_reversal) + np.abs(bb_reversal)) / 3
        return urgency_score

    def _calculate_reversal_probability(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend reversal probability."""
        # Combine multiple reversal indicators
        momentum_reversal = self._calculate_momentum_reversal(df, 10)
        volatility_reversal = self._calculate_atr_reversal(df, 14)
        volume_reversal = self._calculate_vpt_reversal(df)
        
        # Reversal probability
        reversal_signals = (np.abs(momentum_reversal) + np.abs(volatility_reversal) + np.abs(volume_reversal)) / 3
        reversal_prob = reversal_signals / 3  # Normalize to 0-1
        
        return reversal_prob

    def _calculate_optimal_exit_timing(self, df: pd.DataFrame) -> pd.Series:
        """Calculate optimal exit timing score."""
        # Combine exit factors
        exit_urgency = self._calculate_exit_urgency_score(df)
        reversal_prob = self._calculate_reversal_probability(df)
        profit_preservation = self._calculate_profit_preservation_score(df, df['close'].iloc[0])
        
        # Optimal exit timing
        exit_score = (exit_urgency + reversal_prob + profit_preservation) / 3
        return exit_score

    def _calculate_exit_timing_confidence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate exit timing confidence."""
        # Confidence based on signal strength and market conditions
        optimal_exit = self._calculate_optimal_exit_timing(df)
        volatility = self._calculate_atr(df, 14) / df['close']
        trend_strength = self._calculate_trend_strength(df)
        
        # Higher confidence for stronger signals and clearer market conditions
        confidence = np.abs(optimal_exit) * (1 - volatility) * (trend_strength / 2)
        return confidence

    def _calculate_risk_adjusted_exit_signal(self, df: pd.DataFrame) -> pd.Series:
        """Calculate risk-adjusted exit signal."""
        # Combine risk factors for exit decision
        exit_urgency = self._calculate_exit_urgency_score(df)
        reversal_prob = self._calculate_reversal_probability(df)
        volatility = self._calculate_atr(df, 14) / df['close']
        
        # Risk-adjusted exit signal
        risk_adjusted_signal = (exit_urgency + reversal_prob) * (1 + volatility)
        return risk_adjusted_signal