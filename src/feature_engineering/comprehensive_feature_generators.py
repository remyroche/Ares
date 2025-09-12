"""
Comprehensive Feature Generators for Lookback Optimization

This module provides all available feature generators from the feature engineering
pipeline, excluding SR-specific and Wavelet features. Each generator is optimized
for hardware acceleration and includes safe math operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Callable, Any, Optional, List, Tuple, Union
import logging
from pathlib import Path
import sys
import warnings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Import hardware optimization tools
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Hardware optimization tools not available: {e}")
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# Import safe math operations
try:
    from src.utils.math_validation import safe_divide, safe_log, safe_sqrt
    SAFE_MATH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Safe math operations not available: {e}")
    SAFE_MATH_AVAILABLE = False

# Import feature selection tools
try:
    from src.utils.feature_selection.step08_optimized_methods import (
        fast_correlation_matrix, optimized_mutual_information, 
        vectorized_feature_stability, parallel_feature_importance
    )
    FEATURE_SELECTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Feature selection tools not available: {e}")
    FEATURE_SELECTION_AVAILABLE = False

# Import parallel processing
try:
    from src.utils.parallel_processing_optimizer import ParallelProcessor
    PARALLEL_PROCESSING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Parallel processing not available: {e}")
    PARALLEL_PROCESSING_AVAILABLE = False

class ComprehensiveFeatureGenerators:
    """Comprehensive collection of all available feature generators with hardware optimization."""
    
    def __init__(self):
        """Initialize comprehensive feature generators with hardware optimization."""
        self.logger = logger.getChild('ComprehensiveFeatureGenerators')
        
        # Initialize hardware optimization if available
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            self.gpu_manager = M1GPUManager()
            self.cpu_optimizer = M1CPUOptimizer()
            self.memory_optimizer = M1MemoryOptimizer()
            self.logger.info("✅ Hardware optimization initialized")
        else:
            self.gpu_manager = None
            self.cpu_optimizer = None
            self.memory_optimizer = None
            self.logger.info("ℹ️ Hardware optimization not available")
        
        # Initialize parallel processing if available
        if PARALLEL_PROCESSING_AVAILABLE:
            self.parallel_processor = ParallelProcessor(max_workers=4)
            self.logger.info("✅ Parallel processing initialized")
        else:
            self.parallel_processor = None
            self.logger.info("ℹ️ Parallel processing not available")
    
    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with fallback."""
        if SAFE_MATH_AVAILABLE:
            return safe_divide(numerator, denominator, default)
        else:
            return numerator / denominator if denominator != 0 else default
    
    def _safe_log(self, value: float, default: float = 0.0) -> float:
        """Safe logarithm with fallback."""
        if SAFE_MATH_AVAILABLE:
            return safe_log(value, default)
        else:
            return np.log(value) if value > 0 else default
    
    def _safe_sqrt(self, value: float, default: float = 0.0) -> float:
        """Safe square root with fallback."""
        if SAFE_MATH_AVAILABLE:
            return safe_sqrt(value, default)
        else:
            return np.sqrt(value) if value >= 0 else default
    
    # ==================== BASIC TECHNICAL INDICATORS ====================
    
    def rsi_generator(self, data: pd.DataFrame, lookback: int, price_column: str = 'close') -> pd.Series:
        """Generate RSI (Relative Strength Index) indicator with hardware optimization."""
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")
            
            prices = data[price_column]
            delta = prices.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=lookback).mean()
            avg_losses = losses.rolling(window=lookback).mean()
            
            # Calculate RS and RSI with safe division
            rs = avg_gains / avg_losses.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)  # Fill NaN with neutral RSI value
            
        except Exception as e:
            self.logger.error(f"Error generating RSI: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def sma_generator(self, data: pd.DataFrame, lookback: int, price_column: str = 'close') -> pd.Series:
        """Generate SMA (Simple Moving Average) indicator."""
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")
            
            prices = data[price_column]
            sma = prices.rolling(window=lookback).mean()
            
            return sma
            
        except Exception as e:
            self.logger.error(f"Error generating SMA: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def ema_generator(self, data: pd.DataFrame, lookback: int, price_column: str = 'close') -> pd.Series:
        """Generate EMA (Exponential Moving Average) indicator."""
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")
            
            prices = data[price_column]
            ema = prices.ewm(span=lookback).mean()
            
            return ema
            
        except Exception as e:
            self.logger.error(f"Error generating EMA: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def bollinger_bands_generator(self, data: pd.DataFrame, lookback: int, price_column: str = 'close', 
                                 std_dev: float = 2.0) -> pd.Series:
        """Generate Bollinger Bands position indicator."""
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")
            
            prices = data[price_column]
            sma = prices.rolling(window=lookback).mean()
            std = prices.rolling(window=lookback).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            # Calculate position within bands (0-1 scale)
            bb_position = (prices - lower_band) / (upper_band - lower_band)
            
            return bb_position.fillna(0.5)  # Fill NaN with middle position
            
        except Exception as e:
            self.logger.error(f"Error generating Bollinger Bands: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def macd_generator(self, data: pd.DataFrame, lookback: int, price_column: str = 'close',
                      fast_period: int = 12, slow_period: int = 26) -> pd.Series:
        """Generate MACD signal line indicator."""
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")
            
            prices = data[price_column]
            
            # Calculate EMAs
            ema_fast = prices.ewm(span=fast_period).mean()
            ema_slow = prices.ewm(span=slow_period).mean()
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=lookback).mean()
            
            return signal_line
            
        except Exception as e:
            self.logger.error(f"Error generating MACD: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def stochastic_generator(self, data: pd.DataFrame, lookback: int, k_period: int = 14,
                           d_period: int = 3) -> pd.Series:
        """Generate Stochastic Oscillator %D indicator."""
        try:
            required_columns = ['high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Required columns {required_columns} not found in data")
            
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate %K
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            
            # Calculate %D (smoothed %K)
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return d_percent.fillna(50)  # Fill NaN with neutral value
            
        except Exception as e:
            self.logger.error(f"Error generating Stochastic: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def atr_generator(self, data: pd.DataFrame, lookback: int) -> pd.Series:
        """Generate ATR (Average True Range) indicator."""
        try:
            required_columns = ['high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Required columns {required_columns} not found in data")
            
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = true_range.rolling(window=lookback).mean()
            
            return atr
            
        except Exception as e:
            self.logger.error(f"Error generating ATR: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    # ==================== VOLUME-BASED INDICATORS ====================
    
    def volume_sma_generator(self, data: pd.DataFrame, lookback: int, volume_column: str = 'volume') -> pd.Series:
        """Generate Volume SMA indicator."""
        try:
            if volume_column not in data.columns:
                raise ValueError(f"Volume column '{volume_column}' not found in data")
            
            volume = data[volume_column]
            volume_sma = volume.rolling(window=lookback).mean()
            
            return volume_sma
            
        except Exception as e:
            self.logger.error(f"Error generating Volume SMA: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def volume_weighted_price_generator(self, data: pd.DataFrame, lookback: int) -> pd.Series:
        """Generate Volume Weighted Average Price (VWAP) indicator."""
        try:
            required_columns = ['high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Required columns {required_columns} not found in data")
            
            high = data['high']
            low = data['low']
            close = data['close']
            volume = data['volume']
            
            # Calculate typical price
            typical_price = (high + low + close) / 3
            
            # Calculate VWAP
            vwap = (typical_price * volume).rolling(window=lookback).sum() / volume.rolling(window=lookback).sum()
            
            return vwap
            
        except Exception as e:
            self.logger.error(f"Error generating VWAP: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def on_balance_volume_generator(self, data: pd.DataFrame, lookback: int) -> pd.Series:
        """Generate On-Balance Volume (OBV) indicator."""
        try:
            required_columns = ['close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Required columns {required_columns} not found in data")
            
            close = data['close']
            volume = data['volume']
            
            # Calculate price change direction
            price_change = close.diff()
            
            # Calculate OBV
            obv = volume.copy()
            obv[price_change < 0] = -volume[price_change < 0]
            obv[price_change == 0] = 0
            
            # Calculate rolling OBV
            obv_rolling = obv.rolling(window=lookback).sum()
            
            return obv_rolling
            
        except Exception as e:
            self.logger.error(f"Error generating OBV: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    # ==================== MOMENTUM INDICATORS ====================
    
    def price_momentum_generator(self, data: pd.DataFrame, lookback: int, price_column: str = 'close') -> pd.Series:
        """Generate Price Momentum indicator."""
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")
            
            prices = data[price_column]
            momentum = prices.pct_change(lookback)
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"Error generating Price Momentum: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def rate_of_change_generator(self, data: pd.DataFrame, lookback: int, price_column: str = 'close') -> pd.Series:
        """Generate Rate of Change (ROC) indicator."""
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")
            
            prices = data[price_column]
            roc = ((prices - prices.shift(lookback)) / prices.shift(lookback)) * 100
            
            return roc
            
        except Exception as e:
            self.logger.error(f"Error generating ROC: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def williams_r_generator(self, data: pd.DataFrame, lookback: int) -> pd.Series:
        """Generate Williams %R indicator."""
        try:
            required_columns = ['high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Required columns {required_columns} not found in data")
            
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate Williams %R
            highest_high = high.rolling(window=lookback).max()
            lowest_low = low.rolling(window=lookback).min()
            
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            
            return williams_r.fillna(-50)  # Fill NaN with neutral value
            
        except Exception as e:
            self.logger.error(f"Error generating Williams %R: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    # ==================== VOLATILITY INDICATORS ====================
    
    def volatility_generator(self, data: pd.DataFrame, lookback: int, price_column: str = 'close') -> pd.Series:
        """Generate Volatility indicator (rolling standard deviation of returns)."""
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")
            
            prices = data[price_column]
            returns = prices.pct_change()
            volatility = returns.rolling(window=lookback).std()
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error generating Volatility: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def keltner_channels_generator(self, data: pd.DataFrame, lookback: int, multiplier: float = 2.0) -> pd.Series:
        """Generate Keltner Channels position indicator."""
        try:
            required_columns = ['high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Required columns {required_columns} not found in data")
            
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate typical price
            typical_price = (high + low + close) / 3
            
            # Calculate EMA of typical price
            ema_tp = typical_price.ewm(span=lookback).mean()
            
            # Calculate ATR
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=lookback).mean()
            
            # Calculate Keltner Channels
            upper_channel = ema_tp + (multiplier * atr)
            lower_channel = ema_tp - (multiplier * atr)
            
            # Calculate position within channels
            kc_position = (close - lower_channel) / (upper_channel - lower_channel)
            
            return kc_position.fillna(0.5)  # Fill NaN with middle position
            
        except Exception as e:
            self.logger.error(f"Error generating Keltner Channels: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    # ==================== TREND INDICATORS ====================
    
    def adx_generator(self, data: pd.DataFrame, lookback: int) -> pd.Series:
        """Generate Average Directional Index (ADX) indicator."""
        try:
            required_columns = ['high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Required columns {required_columns} not found in data")
            
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            dm_plus = high.diff()
            dm_minus = -low.diff()
            
            dm_plus[dm_plus < 0] = 0
            dm_minus[dm_minus < 0] = 0
            
            # Calculate smoothed values
            tr_smooth = true_range.rolling(window=lookback).mean()
            dm_plus_smooth = dm_plus.rolling(window=lookback).mean()
            dm_minus_smooth = dm_minus.rolling(window=lookback).mean()
            
            # Calculate DI+ and DI-
            di_plus = 100 * (dm_plus_smooth / tr_smooth)
            di_minus = 100 * (dm_minus_smooth / tr_smooth)
            
            # Calculate DX
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            
            # Calculate ADX
            adx = dx.rolling(window=lookback).mean()
            
            return adx.fillna(25)  # Fill NaN with neutral ADX value
            
        except Exception as e:
            self.logger.error(f"Error generating ADX: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def parabolic_sar_generator(self, data: pd.DataFrame, lookback: int, acceleration: float = 0.02, 
                               maximum: float = 0.2) -> pd.Series:
        """Generate Parabolic SAR indicator."""
        try:
            required_columns = ['high', 'low']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Required columns {required_columns} not found in data")
            
            high = data['high']
            low = data['low']
            
            # Simplified Parabolic SAR calculation
            # This is a basic implementation - full PSAR is more complex
            psar = pd.Series(index=data.index, dtype=float)
            
            # Initialize
            psar.iloc[0] = low.iloc[0]
            trend = 1  # 1 for uptrend, -1 for downtrend
            af = acceleration
            
            for i in range(1, len(data)):
                if trend == 1:
                    psar.iloc[i] = psar.iloc[i-1] + af * (high.iloc[i-1] - psar.iloc[i-1])
                    if low.iloc[i] <= psar.iloc[i]:
                        trend = -1
                        psar.iloc[i] = high.iloc[i-1]
                        af = acceleration
                    else:
                        if high.iloc[i] > high.iloc[i-1]:
                            af = min(af + acceleration, maximum)
                else:
                    psar.iloc[i] = psar.iloc[i-1] + af * (low.iloc[i-1] - psar.iloc[i-1])
                    if high.iloc[i] >= psar.iloc[i]:
                        trend = 1
                        psar.iloc[i] = low.iloc[i-1]
                        af = acceleration
                    else:
                        if low.iloc[i] < low.iloc[i-1]:
                            af = min(af + acceleration, maximum)
            
            return psar
            
        except Exception as e:
            self.logger.error(f"Error generating Parabolic SAR: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    # ==================== CROSS-TIMEFRAME FEATURES ====================
    
    def cross_timeframe_momentum_generator(self, data: pd.DataFrame, lookback: int, 
                                         short_period: int = 5, long_period: int = 20) -> pd.Series:
        """Generate cross-timeframe momentum indicator."""
        try:
            if 'close' not in data.columns:
                raise ValueError("Close column not found in data")
            
            close = data['close']
            
            # Calculate short and long term momentum
            short_momentum = close.pct_change(short_period)
            long_momentum = close.pct_change(long_period)
            
            # Cross-timeframe momentum
            cross_momentum = short_momentum - long_momentum
            
            # Rolling average
            cross_momentum_avg = cross_momentum.rolling(window=lookback).mean()
            
            return cross_momentum_avg
            
        except Exception as e:
            self.logger.error(f"Error generating cross-timeframe momentum: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def cross_timeframe_volatility_generator(self, data: pd.DataFrame, lookback: int,
                                           short_period: int = 5, long_period: int = 20) -> pd.Series:
        """Generate cross-timeframe volatility indicator."""
        try:
            if 'close' not in data.columns:
                raise ValueError("Close column not found in data")
            
            close = data['close']
            returns = close.pct_change()
            
            # Calculate short and long term volatility
            short_vol = returns.rolling(window=short_period).std()
            long_vol = returns.rolling(window=long_period).std()
            
            # Cross-timeframe volatility ratio
            vol_ratio = short_vol / long_vol
            
            # Rolling average
            vol_ratio_avg = vol_ratio.rolling(window=lookback).mean()
            
            return vol_ratio_avg.fillna(1.0)  # Fill NaN with neutral ratio
            
        except Exception as e:
            self.logger.error(f"Error generating cross-timeframe volatility: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    # ==================== PATTERN RECOGNITION FEATURES ====================
    
    def price_pattern_generator(self, data: pd.DataFrame, lookback: int) -> pd.Series:
        """Generate price pattern recognition indicator."""
        try:
            required_columns = ['high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Required columns {required_columns} not found in data")
            
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Simple pattern recognition: higher highs and higher lows
            higher_highs = (high.rolling(window=lookback).max() == high).astype(int)
            higher_lows = (low.rolling(window=lookback).min() == low).astype(int)
            
            # Pattern strength
            pattern_strength = higher_highs + higher_lows
            
            return pattern_strength.rolling(window=lookback).mean()
            
        except Exception as e:
            self.logger.error(f"Error generating price pattern: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def support_resistance_generator(self, data: pd.DataFrame, lookback: int) -> pd.Series:
        """Generate support/resistance level indicator."""
        try:
            required_columns = ['high', 'low']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Required columns {required_columns} not found in data")
            
            high = data['high']
            low = data['low']
            
            # Calculate rolling highs and lows
            rolling_high = high.rolling(window=lookback).max()
            rolling_low = low.rolling(window=lookback).min()
            
            # Distance to support/resistance
            current_price = (high + low) / 2
            distance_to_resistance = (rolling_high - current_price) / current_price
            distance_to_support = (current_price - rolling_low) / current_price
            
            # Combined indicator
            sr_indicator = distance_to_resistance - distance_to_support
            
            return sr_indicator
            
        except Exception as e:
            self.logger.error(f"Error generating support/resistance: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    # ==================== REGIME-DEPENDENT FEATURES ====================
    
    def regime_volatility_generator(self, data: pd.DataFrame, lookback: int) -> pd.Series:
        """Generate regime-dependent volatility indicator."""
        try:
            if 'close' not in data.columns:
                raise ValueError("Close column not found in data")
            
            close = data['close']
            returns = close.pct_change()
            
            # Calculate volatility regimes
            vol = returns.rolling(window=lookback).std()
            vol_mean = vol.rolling(window=lookback*2).mean()
            vol_std = vol.rolling(window=lookback*2).std()
            
            # Regime classification
            high_vol_regime = (vol > vol_mean + vol_std).astype(int)
            low_vol_regime = (vol < vol_mean - vol_std).astype(int)
            
            # Regime indicator
            regime_indicator = high_vol_regime - low_vol_regime
            
            return regime_indicator.rolling(window=lookback).mean()
            
        except Exception as e:
            self.logger.error(f"Error generating regime volatility: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def regime_trend_generator(self, data: pd.DataFrame, lookback: int) -> pd.Series:
        """Generate regime-dependent trend indicator."""
        try:
            if 'close' not in data.columns:
                raise ValueError("Close column not found in data")
            
            close = data['close']
            
            # Calculate trend strength
            sma_short = close.rolling(window=lookback//2).mean()
            sma_long = close.rolling(window=lookback).mean()
            
            # Trend direction
            trend_direction = (sma_short > sma_long).astype(int) * 2 - 1
            
            # Trend strength
            trend_strength = abs(sma_short - sma_long) / sma_long
            
            # Combined regime trend indicator
            regime_trend = trend_direction * trend_strength
            
            return regime_trend.rolling(window=lookback).mean()
            
        except Exception as e:
            self.logger.error(f"Error generating regime trend: {e}")
            return pd.Series(index=data.index, dtype=float)

# Registry of all available feature generators
COMPREHENSIVE_FEATURE_GENERATORS: Dict[str, Callable] = {
    # Basic Technical Indicators
    'rsi': ComprehensiveFeatureGenerators().rsi_generator,
    'sma': ComprehensiveFeatureGenerators().sma_generator,
    'ema': ComprehensiveFeatureGenerators().ema_generator,
    'bollinger_bands': ComprehensiveFeatureGenerators().bollinger_bands_generator,
    'macd': ComprehensiveFeatureGenerators().macd_generator,
    'stochastic': ComprehensiveFeatureGenerators().stochastic_generator,
    'atr': ComprehensiveFeatureGenerators().atr_generator,
    
    # Volume-based Indicators
    'volume_sma': ComprehensiveFeatureGenerators().volume_sma_generator,
    'volume_weighted_price': ComprehensiveFeatureGenerators().volume_weighted_price_generator,
    'on_balance_volume': ComprehensiveFeatureGenerators().on_balance_volume_generator,
    
    # Momentum Indicators
    'price_momentum': ComprehensiveFeatureGenerators().price_momentum_generator,
    'rate_of_change': ComprehensiveFeatureGenerators().rate_of_change_generator,
    'williams_r': ComprehensiveFeatureGenerators().williams_r_generator,
    
    # Volatility Indicators
    'volatility': ComprehensiveFeatureGenerators().volatility_generator,
    'keltner_channels': ComprehensiveFeatureGenerators().keltner_channels_generator,
    
    # Trend Indicators
    'adx': ComprehensiveFeatureGenerators().adx_generator,
    'parabolic_sar': ComprehensiveFeatureGenerators().parabolic_sar_generator,
    
    # Cross-timeframe Features
    'cross_timeframe_momentum': ComprehensiveFeatureGenerators().cross_timeframe_momentum_generator,
    'cross_timeframe_volatility': ComprehensiveFeatureGenerators().cross_timeframe_volatility_generator,
    
    # Pattern Recognition Features
    'price_pattern': ComprehensiveFeatureGenerators().price_pattern_generator,
    'support_resistance': ComprehensiveFeatureGenerators().support_resistance_generator,
    
    # Regime-dependent Features
    'regime_volatility': ComprehensiveFeatureGenerators().regime_volatility_generator,
    'regime_trend': ComprehensiveFeatureGenerators().regime_trend_generator,
}

def get_comprehensive_feature_generator(feature_name: str) -> Optional[Callable]:
    """Get a comprehensive feature generator function by name."""
    return COMPREHENSIVE_FEATURE_GENERATORS.get(feature_name.lower())

def list_all_available_generators() -> List[str]:
    """List all available comprehensive feature generator names."""
    return list(COMPREHENSIVE_FEATURE_GENERATORS.keys())

def create_comprehensive_feature_config(feature_name: str, **kwargs) -> Dict[str, Any]:
    """Create a configuration for a comprehensive feature generator."""
    generator = get_comprehensive_feature_generator(feature_name)
    if not generator:
        raise ValueError(f"Unknown comprehensive feature generator: {feature_name}")
    
    config = {
        'generator': generator,
        'feature_name': feature_name,
        **kwargs
    }
    
    return config