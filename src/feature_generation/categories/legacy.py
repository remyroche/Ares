"""Legacy features

Legacy features are traditional technical indicators that have been used in 
financial analysis for decades. These include classic indicators like:
- Traditional RSI implementations
- Classic MACD calculations
- Original Bollinger Bands formulations
- Standard moving averages
- Conventional oscillators

These features maintain backward compatibility with existing trading systems
and provide a baseline for comparison with newer, enhanced indicators.

All legacy generators now use vectorized numpy operations for optimal performance.
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from ..core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

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
    import warnings
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Import VectorBT-optimized legacy generators
try:
    from .vectorbt_legacy import (
        create_vectorbt_legacy_generators,
        create_default_vectorbt_legacy_generators,
        VectorBTLegacyRSIGenerator,
        VectorBTLegacyMACDGenerator,
        VectorBTLegacyBollingerBandsGenerator,
        VectorBTLegacySMAGenerator,
        VectorBTLegacyEMAGenerator,
        VectorBTLegacyATRGenerator,
        VectorBTLegacyStochasticGenerator,
        VectorBTLegacyWilliamsRGenerator,
        VectorBTLegacyOBVGenerator
    )
    VECTORBT_LEGACY_AVAILABLE = True
except ImportError:
    VECTORBT_LEGACY_AVAILABLE = False

class LegacyRSIGenerator(VectorizedFeatureGenerator):
    def __init__(self, period: int = 14):
        config = FeatureConfig(
            name=f"legacy_rsi_{period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy RSI {period} - traditional implementation",
            required_columns=["close"],
            default_lookback=period * 2,
            min_lookback=period,
            max_lookback=period * 3
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        
        # Use VectorBT for optimized RSI calculation if available
        if VECTORBT_AVAILABLE and len(close) >= 1000:
            return self._calculate_rsi_vectorbt(close)
        else:
            return self._calculate_rsi_pandas(close)
    
    def _calculate_rsi_vectorbt(self, close: pd.Series) -> pd.Series:
        """Calculate RSI using VectorBT optimized operations."""
        try:
            # Use VectorBT RSI if available
            rsi_result = vbt.RSI.run(close, window=self.period)
            return rsi_result.rsi.rename(f'legacy_rsi_{self.period}')
        except Exception as e:
            # Fallback to pandas implementation
            return self._calculate_rsi_pandas(close)
    
    def _calculate_rsi_pandas(self, close: pd.Series) -> pd.Series:
        """Calculate RSI using pandas operations."""
        # Vectorized RSI calculation using numpy
        rsi = self._calculate_rsi_vectorized(close.values, self.period)
        return pd.Series(rsi, index=close.index, name=f'legacy_rsi_{self.period}')
    
    def _calculate_rsi_vectorized(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI using vectorized numpy operations."""
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        
        # Calculate price changes
        delta = np.diff(prices, prepend=prices[0])
        
        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        # Calculate rolling means using numpy
        avg_gains = self._rolling_mean_vectorized(gains, period)
        avg_losses = self._rolling_mean_vectorized(losses, period)
        
        # Calculate RSI
        rs = np.divide(avg_gains, avg_losses, out=np.ones_like(avg_gains), where=avg_losses!=0)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _rolling_mean_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean using centralized method."""
        series = pd.Series(data)
        return self._calculate_sma_vectorized(series, window).values
        
        # Use numpy's cumsum for efficient rolling mean calculation
        cumsum = np.cumsum(data)
        rolling_mean = np.full(len(data), np.nan)
        
        # Calculate rolling mean for valid windows
        for i in range(window - 1, len(data)):
            if i == window - 1:
                rolling_mean[i] = cumsum[i] / window
            else:
                rolling_mean[i] = (cumsum[i] - cumsum[i - window]) / window
        
        return rolling_meanclass LegacyMACDGenerator(VectorizedFeatureGenerator):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        config = FeatureConfig(
            name=f"legacy_macd_{fast}_{slow}_{signal}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy MACD {fast}/{slow}/{signal} - traditional implementation",
            required_columns=["close"],
            default_lookback=slow * 2,
            min_lookback=slow,
            max_lookback=slow * 3
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        
        # Use VectorBT for optimized MACD calculation if available
        if VECTORBT_AVAILABLE and len(close) >= 1000:
            return self._calculate_macd_vectorbt(close)
        else:
            return self._calculate_macd_pandas(close)
    
    def _calculate_macd_vectorbt(self, close: pd.Series) -> pd.Series:
        """Calculate MACD using VectorBT optimized operations."""
        try:
            # Use VectorBT MACD if available
            macd_result = vbt.MACD.run(close, fast_window=self.fast, slow_window=self.slow, signal_window=self.signal)
            return macd_result.macd.rename(f'legacy_macd_{self.fast}_{self.slow}_{self.signal}')
        except Exception as e:
            # Fallback to pandas implementation
            return self._calculate_macd_pandas(close)
    
    def _calculate_macd_pandas(self, close: pd.Series) -> pd.Series:
        """Calculate MACD using pandas operations."""
        # Vectorized MACD calculation using numpy
        macd = self._calculate_macd_vectorized(close.values, self.fast, self.slow)
        return pd.Series(macd, index=close.index, name=f'legacy_macd_{self.fast}_{self.slow}_{self.signal}')
    
    def _calculate_macd_vectorized(self, prices: np.ndarray, fast: int, slow: int) -> np.ndarray:
        """Calculate MACD using vectorized numpy operations."""
        if len(prices) < slow:
            return np.full(len(prices), np.nan)
        
        # Calculate EMAs using vectorized operations
        ema_fast = self._calculate_ema_vectorized(prices, fast)
        ema_slow = self._calculate_ema_vectorized(prices, slow)
        
        # MACD line
        macd = ema_fast - ema_slow
        
        return macd
    
    def _calculate_ema_vectorized(self, prices: np.ndarray, span: int) -> np.ndarray:
        """Calculate EMA using centralized method."""
        series = pd.Series(prices)
        return self._calculate_ema_vectorized(series, span).values
        
        # Calculate alpha (smoothing factor)
        alpha = 2.0 / (span + 1.0)
        
        # Initialize EMA array
        ema = np.full(len(prices), np.nan)
        ema[0] = prices[0]
        
        # Calculate EMA using vectorized operations
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        
        return emaclass LegacyBollingerBandsGenerator(VectorizedFeatureGenerator):
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        config = FeatureConfig(
            name=f"legacy_bollinger_{period}_{std_dev}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy Bollinger Bands {period}/{std_dev} - traditional implementation",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.std_dev = std_dev
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        
        # Use VectorBT for optimized Bollinger Bands calculation if available
        if VECTORBT_AVAILABLE and len(close) >= 1000:
            return self._calculate_bollinger_bands_vectorbt(close)
        else:
            return self._calculate_bollinger_bands_pandas(close)
    
    def _calculate_bollinger_bands_vectorbt(self, close: pd.Series) -> pd.Series:
        """Calculate Bollinger Bands using VectorBT optimized operations."""
        try:
            # Use VectorBT Bollinger Bands if available
            bb_result = vbt.BBANDS.run(close, window=self.period, alpha=self.std_dev)
            return bb_result.upper.rename(f'legacy_bollinger_upper_{self.period}_{self.std_dev}')
        except Exception as e:
            # Fallback to pandas implementation
            return self._calculate_bollinger_bands_pandas(close)
    
    def _calculate_bollinger_bands_pandas(self, close: pd.Series) -> pd.Series:
        """Calculate Bollinger Bands using pandas operations."""
        # Vectorized Bollinger Bands calculation using numpy
        upper_band = self._calculate_bollinger_bands_vectorized(close.values, self.period, self.std_dev)
        return pd.Series(upper_band, index=close.index, name=f'legacy_bollinger_upper_{self.period}_{self.std_dev}')
    
    def _calculate_bollinger_bands_vectorized(self, prices: np.ndarray, period: int, std_dev: float) -> np.ndarray:
        """Calculate Bollinger Bands upper band using vectorized numpy operations."""
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        # Calculate SMA using vectorized operations
        sma = self._rolling_mean_vectorized(prices, period)
        
        # Calculate rolling standard deviation using vectorized operations
        std = self._rolling_std_vectorized(prices, period)
        
        # Calculate upper band
        upper_band = sma + (std * std_dev)
        
        return upper_band
    
    def _rolling_mean_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean using centralized method."""
        series = pd.Series(data)
        return self._calculate_sma_vectorized(series, window).values
        
        # Use numpy's cumsum for efficient rolling mean calculation
        cumsum = np.cumsum(data)
        rolling_mean = np.full(len(data), np.nan)
        
        # Calculate rolling mean for valid windows
        for i in range(window - 1, len(data)):
            if i == window - 1:
                rolling_mean[i] = cumsum[i] / window
            else:
                rolling_mean[i] = (cumsum[i] - cumsum[i - window]) / window
        
        return rolling_mean
    
    def _rolling_std_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling std using centralized method."""
        series = pd.Series(data)
        return self._calculate_rolling_std_vectorized(series, window).values
        
        rolling_std = np.full(len(data), np.nan)
        
        for i in range(window - 1, len(data)):
            window_data = data[i - window + 1:i + 1]
            rolling_std[i] = np.std(window_data, ddof=0)
        
        return rolling_stdclass LegacySMAGenerator(VectorizedFeatureGenerator):
    def __init__(self, period: int = 20):
        config = FeatureConfig(
            name=f"legacy_sma_{period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy SMA {period} - traditional implementation",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        
        # Use VectorBT for optimized SMA calculation if available
        if VECTORBT_AVAILABLE and len(close) >= 1000:
            return self._calculate_sma_vectorbt(close)
        else:
            return self._calculate_sma_pandas(close)
    
    def _calculate_sma_vectorbt(self, close: pd.Series) -> pd.Series:
        """Calculate SMA using VectorBT optimized operations."""
        try:
            # Use VectorBT rolling mean if available
            sma = rolling_mean(close, window=self.period)
            return sma.rename(f'legacy_sma_{self.period}')
        except Exception as e:
            # Fallback to pandas implementation
            return self._calculate_sma_pandas(close)
    
    def _calculate_sma_pandas(self, close: pd.Series) -> pd.Series:
        """Calculate SMA using pandas operations."""
        # Vectorized SMA calculation using numpy
        sma = self._rolling_mean_vectorized(close.values, self.period)
        return pd.Series(sma, index=close.index, name=f'legacy_sma_{self.period}')
    
    def _rolling_mean_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean using centralized method."""
        series = pd.Series(data)
        return self._calculate_sma_vectorized(series, window).values
        
        # Use numpy's cumsum for efficient rolling mean calculation
        cumsum = np.cumsum(data)
        rolling_mean = np.full(len(data), np.nan)
        
        # Calculate rolling mean for valid windows
        for i in range(window - 1, len(data)):
            if i == window - 1:
                rolling_mean[i] = cumsum[i] / window
            else:
                rolling_mean[i] = (cumsum[i] - cumsum[i - window]) / window
        
        return rolling_meanclass LegacyEMAGenerator(VectorizedFeatureGenerator):
    def __init__(self, period: int = 21):
        config = FeatureConfig(
            name=f"legacy_ema_{period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy EMA {period} - traditional implementation",
            required_columns=["close"],
            default_lookback=period * 2,
            min_lookback=period,
            max_lookback=period * 3
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        
        # Use VectorBT for optimized EMA calculation if available
        if VECTORBT_AVAILABLE and len(close) >= 1000:
            return self._calculate_ema_vectorbt(close)
        else:
            return self._calculate_ema_pandas(close)
    
    def _calculate_ema_vectorbt(self, close: pd.Series) -> pd.Series:
        """Calculate EMA using VectorBT optimized operations."""
        try:
            # Use VectorBT EMA if available
            ema = close.ewm(span=self.period).mean()
            return ema.rename(f'legacy_ema_{self.period}')
        except Exception as e:
            # Fallback to pandas implementation
            return self._calculate_ema_pandas(close)
    
    def _calculate_ema_pandas(self, close: pd.Series) -> pd.Series:
        """Calculate EMA using pandas operations."""
        # Vectorized EMA calculation using numpy
        ema = self._calculate_ema_vectorized(close.values, self.period)
        return pd.Series(ema, index=close.index, name=f'legacy_ema_{self.period}')
    
    def _calculate_ema_vectorized(self, prices: np.ndarray, span: int) -> np.ndarray:
        """Calculate EMA using centralized method."""
        series = pd.Series(prices)
        return self._calculate_ema_vectorized(series, span).values
        
        # Calculate alpha (smoothing factor)
        alpha = 2.0 / (span + 1.0)
        
        # Initialize EMA array
        ema = np.full(len(prices), np.nan)
        ema[0] = prices[0]
        
        # Calculate EMA using vectorized operations
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        
        return emaclass LegacyATRGenerator(VectorizedFeatureGenerator):
    def __init__(self, period: int = 14):
        config = FeatureConfig(
            name=f"legacy_atr_{period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy ATR {period} - traditional implementation",
            required_columns=["high", "low", "close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        high = data['high']
        low = data['low']
        close = data['close']
        
        # Use VectorBT for optimized ATR calculation if available
        if VECTORBT_AVAILABLE and len(close) >= 1000:
            return self._calculate_atr_vectorbt(high, low, close)
        else:
            return self._calculate_atr_pandas(high, low, close)
    
    def _calculate_atr_vectorbt(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate ATR using VectorBT optimized operations."""
        try:
            # Use VectorBT ATR if available
            atr_result = vbt.ATR.run(high, low, close, window=self.period)
            return atr_result.atr.rename(f'legacy_atr_{self.period}')
        except Exception as e:
            # Fallback to pandas implementation
            return self._calculate_atr_pandas(high, low, close)
    
    def _calculate_atr_pandas(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate ATR using pandas operations."""
        # Vectorized ATR calculation using numpy
        atr = self._calculate_atr_vectorized(high.values, low.values, close.values, self.period)
        return pd.Series(atr, index=close.index, name=f'legacy_atr_{self.period}')
    
    def _calculate_atr_vectorized(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate ATR using vectorized numpy operations."""
        if len(high) < period or len(low) < period or len(close) < period:
            return np.full(len(close), np.nan)
        
        # Calculate True Range components
        tr1 = high - low
        
        # Shift close by 1 period for previous close
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]  # First value uses current close
        
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        # True Range is the maximum of the three components
        tr = np.maximum.reduce([tr1, tr2, tr3])
        
        # Calculate ATR as rolling mean of True Range
        atr = self._rolling_mean_vectorized(tr, period)
        
        return atr
    
    def _rolling_mean_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean using centralized method."""
        series = pd.Series(data)
        return self._calculate_sma_vectorized(series, window).values
        
        # Use numpy's cumsum for efficient rolling mean calculation
        cumsum = np.cumsum(data)
        rolling_mean = np.full(len(data), np.nan)
        
        # Calculate rolling mean for valid windows
        for i in range(window - 1, len(data)):
            if i == window - 1:
                rolling_mean[i] = cumsum[i] / window
            else:
                rolling_mean[i] = (cumsum[i] - cumsum[i - window]) / window
        
        return rolling_meanclass LegacyStochasticGenerator(VectorizedFeatureGenerator):
    def __init__(self, k_period: int = 14, d_period: int = 3):
        config = FeatureConfig(
            name=f"legacy_stochastic_{k_period}_{d_period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy Stochastic {k_period}/{d_period} - traditional implementation",
            required_columns=["high", "low", "close"],
            default_lookback=k_period,
            min_lookback=k_period,
            max_lookback=k_period
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.k_period = k_period
        self.d_period = d_period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        high = data['high']
        low = data['low']
        close = data['close']
        
        # Use VectorBT for optimized Stochastic calculation if available
        if VECTORBT_AVAILABLE and len(close) >= 1000:
            return self._calculate_stochastic_vectorbt(high, low, close)
        else:
            return self._calculate_stochastic_pandas(high, low, close)
    
    def _calculate_stochastic_vectorbt(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Stochastic using VectorBT optimized operations."""
        try:
            # Use VectorBT Stochastic if available
            stoch_result = vbt.STOCH.run(high, low, close, k_window=self.k_period, d_window=self.d_period)
            return stoch_result.stoch_k.rename(f'legacy_stochastic_k_{self.k_period}_{self.d_period}')
        except Exception as e:
            # Fallback to pandas implementation
            return self._calculate_stochastic_pandas(high, low, close)
    
    def _calculate_stochastic_pandas(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Stochastic using pandas operations."""
        # Vectorized Stochastic calculation using numpy
        k_percent = self._calculate_stochastic_vectorized(high.values, low.values, close.values, self.k_period)
        return pd.Series(k_percent, index=close.index, name=f'legacy_stochastic_k_{self.k_period}_{self.d_period}')
    
    def _calculate_stochastic_vectorized(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int) -> np.ndarray:
        """Calculate Stochastic %K using vectorized numpy operations."""
        if len(high) < k_period or len(low) < k_period or len(close) < k_period:
            return np.full(len(close), np.nan)
        
        # Calculate rolling min and max using vectorized operations
        lowest_low = self._rolling_min_vectorized(low, k_period)
        highest_high = self._rolling_max_vectorized(high, k_period)
        
        # Calculate %K
        denominator = highest_high - lowest_low
        k_percent = np.where(
            denominator != 0,
            100 * ((close - lowest_low) / denominator),
            0
        )
        
        return k_percent
    
    def _rolling_min_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling min using centralized method."""
        series = pd.Series(data)
        return self._calculate_rolling_min_vectorized(series, window).values
        
        rolling_min = np.full(len(data), np.nan)
        
        for i in range(window - 1, len(data)):
            rolling_min[i] = np.min(data[i - window + 1:i + 1])
        
        return rolling_min
    
    def _rolling_max_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling max using centralized method."""
        series = pd.Series(data)
        return self._calculate_rolling_max_vectorized(series, window).values
        
        rolling_max = np.full(len(data), np.nan)
        
        for i in range(window - 1, len(data)):
            rolling_max[i] = np.max(data[i - window + 1:i + 1])
        
        return rolling_maxclass LegacyWilliamsRGenerator(VectorizedFeatureGenerator):
    def __init__(self, period: int = 14):
        config = FeatureConfig(
            name=f"legacy_williams_r_{period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy Williams %R {period} - traditional implementation",
            required_columns=["high", "low", "close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        high = data['high']
        low = data['low']
        close = data['close']
        
        # Use VectorBT for optimized Williams %R calculation if available
        if VECTORBT_AVAILABLE and len(close) >= 1000:
            return self._calculate_williams_r_vectorbt(high, low, close)
        else:
            return self._calculate_williams_r_pandas(high, low, close)
    
    def _calculate_williams_r_vectorbt(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Williams %R using VectorBT optimized operations."""
        try:
            # Use VectorBT Williams %R if available
            willr_result = vbt.WILLR.run(high, low, close, window=self.period)
            return willr_result.willr.rename(f'legacy_williams_r_{self.period}')
        except Exception as e:
            # Fallback to pandas implementation
            return self._calculate_williams_r_pandas(high, low, close)
    
    def _calculate_williams_r_pandas(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Williams %R using pandas operations."""
        # Vectorized Williams %R calculation using numpy
        williams_r = self._calculate_williams_r_vectorized(high.values, low.values, close.values, self.period)
        return pd.Series(williams_r, index=close.index, name=f'legacy_williams_r_{self.period}')
    
    def _calculate_williams_r_vectorized(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Williams %R using vectorized numpy operations."""
        if len(high) < period or len(low) < period or len(close) < period:
            return np.full(len(close), np.nan)
        
        # Calculate rolling min and max using vectorized operations
        lowest_low = self._rolling_min_vectorized(low, period)
        highest_high = self._rolling_max_vectorized(high, period)
        
        # Calculate Williams %R
        denominator = highest_high - lowest_low
        williams_r = np.where(
            denominator != 0,
            -100 * ((highest_high - close) / denominator),
            0
        )
        
        return williams_r
    
    def _rolling_min_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling min using centralized method."""
        series = pd.Series(data)
        return self._calculate_rolling_min_vectorized(series, window).values
        
        rolling_min = np.full(len(data), np.nan)
        
        for i in range(window - 1, len(data)):
            rolling_min[i] = np.min(data[i - window + 1:i + 1])
        
        return rolling_min
    
    def _rolling_max_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling max using centralized method."""
        series = pd.Series(data)
        return self._calculate_rolling_max_vectorized(series, window).values
        
        rolling_max = np.full(len(data), np.nan)
        
        for i in range(window - 1, len(data)):
            rolling_max[i] = np.max(data[i - window + 1:i + 1])
        
        return rolling_maxclass LegacyOBVGenerator(VectorizedFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="legacy_obv",
            category=FeatureCategory.LEGACY,
            description="Legacy OBV - traditional implementation",
            required_columns=["close", "volume"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        volume = data['volume']
        
        # Use VectorBT for optimized OBV calculation if available
        if VECTORBT_AVAILABLE and len(close) >= 1000:
            return self._calculate_obv_vectorbt(close, volume)
        else:
            return self._calculate_obv_pandas(close, volume)
    
    def _calculate_obv_vectorbt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate OBV using VectorBT optimized operations."""
        try:
            # Use VectorBT OBV if available
            obv_result = self._calculate_obv_vectorized(close, volume)
            return obv_result.obv.rename('legacy_obv')
        except Exception as e:
            # Fallback to pandas implementation
            return self._calculate_obv_pandas(close, volume)
    
    def _calculate_obv_pandas(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate OBV using pandas operations."""
        # Vectorized OBV calculation using numpy
        obv = self._calculate_obv_vectorized(close.values, volume.values)
        return pd.Series(obv, index=close.index, name='legacy_obv')
    
    def _calculate_obv_vectorized(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate OBV using vectorized numpy operations."""
        if len(close) < 2 or len(volume) < 2:
            return np.full(len(close), np.nan)
        
        # Calculate price changes
        price_change = np.diff(close, prepend=close[0])
        
        # Calculate OBV based on price direction
        obv = np.where(price_change > 0, volume, 
                      np.where(price_change < 0, -volume, 0))
        
        # Cumulative sum
        obv_cumsum = np.cumsum(obv)
        
        return obv_cumsum

def create_default_legacy_generators() -> List[VectorizedFeatureGenerator]:
    """
    Create default legacy feature generators.
    
    Legacy features include traditional implementations of classic indicators
    that have been used in technical analysis for decades. These provide
    backward compatibility and serve as benchmarks for enhanced versions.
    """
    generators = []
    
    # Use VectorBT generators if available, otherwise fall back to legacy generators
    if VECTORBT_LEGACY_AVAILABLE and VECTORBT_AVAILABLE:
        # Use VectorBT-optimized generators
        generators.extend(create_default_vectorbt_legacy_generators())
    else:
        # Fall back to legacy generators
        # Classic indicators with standard parameters
        generators.extend([
            LegacyRSIGenerator(14),
            LegacyMACDGenerator(12, 26, 9),
            LegacyBollingerBandsGenerator(20, 2.0),
            LegacySMAGenerator(20),
            LegacyEMAGenerator(21),
            LegacyATRGenerator(14),
            LegacyStochasticGenerator(14, 3),
            LegacyWilliamsRGenerator(14),
            LegacyOBVGenerator(),
        ])
        
        # Additional legacy moving averages
        sma_periods = [5, 10, 50, 100, 200]
        for period in sma_periods:
            generators.append(LegacySMAGenerator(period))
        
        # Additional legacy EMAs
        ema_periods = [8, 12, 26, 50, 100]
        for period in ema_periods:
            generators.append(LegacyEMAGenerator(period))
        
        # Additional legacy RSI periods
        rsi_periods = [9, 21, 25]
        for period in rsi_periods:
            generators.append(LegacyRSIGenerator(period))
    
    return generators
