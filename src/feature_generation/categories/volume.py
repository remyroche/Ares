"""
Volume Feature Generator

This module provides feature generators for basic volume-based indicators,
including volume moving averages, ratios, rate of change, and other volume metrics.
"""

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# VectorBT imports for optimization
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

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
from ..base_calculations import (
    BaseCalculationType,
    create_base_calculator
)

logger = logging.getLogger(__name__)

class VolumeFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for basic volume-based features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="volume_features",
            category=FeatureCategory.VOLUME,
            description="Comprehensive volume features including moving averages, ratios, and rate of change",
            required_columns=["volume"],
            optional_columns=["close", "high", "low", "open"],
            default_lookback=20,
            min_lookback=1,
            max_lookback=100,
            parameters={
                "volume_windows": [5, 10, 20, 50],
                "ratio_windows": [10, 20, 50],
                "roc_windows": [1, 5, 10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'VolumeFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume']
        
        # Use VectorBT for volume moving average calculation
        if self._should_use_vectorbt(data):
            try:
                # Calculate volume SMA using VectorBT rolling operations
                if VECTORBT_AVAILABLE:
                    volume_sma = rolling_mean(volume, window=20)
                    self.performance_stats['vectorbt_operations'] += 1
                    return volume_sma
                else:
                    volume_sma = self._vectorbt_rolling_operation(volume, 'mean', 20)
                    return volume_sma
            except Exception as e:
                self.logger.warning(f"VectorBT volume calculation failed: {e}, using pandas fallback")
                self.performance_stats['pandas_fallbacks'] += 1
                return self._vectorbt_rolling_operation(volume, "mean", 20)
        else:
            return self._vectorbt_rolling_operation(volume, "mean", 20)

# Volume Simple Moving Average
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class VolumeSMAGenerator(VectorizedFeatureGenerator):
    """Generator for Volume Simple Moving Average."""
    
    def __init__(self, period: int = 20):
        config = FeatureConfig(
            name=f"volume_sma_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Simple Moving Average over {period} periods",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Volume SMA using VectorBT optimization."""
        volume = data['volume']
        
        # Use VectorBT for optimized rolling mean
        if VECTORBT_AVAILABLE and len(volume) > 100:
            try:
                return rolling_mean(volume, window=self.period)
            except Exception as e:
                logger.warning(f"VectorBT rolling mean failed: {e}, using pandas fallback")
                return volume.rolling(window=self.period).mean()
        else:
            return volume.rolling(window=self.period).mean()

# Volume Exponential Moving Average
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class VolumeEMAGenerator(VectorizedFeatureGenerator):
    """Generator for Volume Exponential Moving Average."""
    
    def __init__(self, period: int = 20, alpha: Optional[float] = None):
        if alpha is None:
            alpha = 2.0 / (period + 1)
        
        config = FeatureConfig(
            name=f"volume_ema_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Exponential Moving Average over {period} periods",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period, 'alpha': alpha}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.alpha = alpha
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Volume EMA."""
        volume = data['volume']
        return volume.ewm(alpha=self.alpha, adjust=False).mean()

# Volume Ratio
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class VolumeRatioGenerator(VectorizedFeatureGenerator):
    """Generator for Volume Ratio (current volume vs average volume)."""
    
    def __init__(self, period: int = 20):
        config = FeatureConfig(
            name=f"volume_ratio_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume ratio (current volume / average volume) over {period} periods",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Volume Ratio using VectorBT optimization."""
        volume = data['volume']
        
        # Use VectorBT for optimized rolling mean
        if VECTORBT_AVAILABLE and len(volume) > 100:
            try:
                avg_volume = rolling_mean(volume, window=self.period)
                return volume / avg_volume.replace(0, 1)  # Avoid division by zero
            except Exception as e:
                logger.warning(f"VectorBT rolling mean failed: {e}, using pandas fallback")
                avg_volume = volume.rolling(window=self.period).mean()
                return volume / avg_volume.replace(0, 1)
        else:
            avg_volume = volume.rolling(window=self.period).mean()
            return volume / avg_volume.replace(0, 1)  # Avoid division by zero

# Volume Rate of Change
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class VolumeROCGenerator(VectorizedFeatureGenerator):
    """Generator for Volume Rate of Change."""
    
    def __init__(self, period: int = 10):
        config = FeatureConfig(
            name=f"volume_roc_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Rate of Change over {period} periods",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Volume ROC."""
        volume = data['volume']
        return volume.pct_change(periods=self.period) * 100

# Volume Standard Deviation
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class VolumeStdGenerator(VectorizedFeatureGenerator):
    """Generator for Volume Standard Deviation."""
    
    def __init__(self, period: int = 20):
        config = FeatureConfig(
            name=f"volume_std_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Standard Deviation over {period} periods",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Volume Standard Deviation using VectorBT optimization."""
        volume = data['volume']
        
        # Use VectorBT for optimized rolling std
        if VECTORBT_AVAILABLE and len(volume) > 100:
            try:
                return rolling_std(volume, window=self.period)
            except Exception as e:
                logger.warning(f"VectorBT rolling std failed: {e}, using pandas fallback")
                return volume.rolling(window=self.period).std()
        else:
            return volume.rolling(window=self.period).std()

# Volume Percentile Rank
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class VolumePercentileGenerator(VectorizedFeatureGenerator):
    """Generator for Volume Percentile Rank."""
    
    def __init__(self, period: int = 50):
        config = FeatureConfig(
            name=f"volume_percentile_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Percentile Rank over {period} periods",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Volume Percentile Rank."""
        volume = data['volume']
        return volume.rolling(window=self.period).rank(pct=True) * 100

# Volume Trend Strength
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class VolumeTrendStrengthGenerator(VectorizedFeatureGenerator):
    """Generator for Volume Trend Strength."""
    
    def __init__(self, short_period: int = 10, long_period: int = 30):
        config = FeatureConfig(
            name=f"volume_trend_strength_{short_period}_{long_period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Trend Strength using {short_period} and {long_period} periods",
            required_columns=["volume"],
            default_lookback=long_period,
            min_lookback=long_period,
            max_lookback=long_period,
            parameters={'short_period': short_period, 'long_period': long_period}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.short_period = short_period
        self.long_period = long_period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Volume Trend Strength."""
        volume = data['volume']
        short_ma = volume.rolling(window=self.short_period).mean()
        long_ma = volume.rolling(window=self.long_period).mean()
        return (short_ma - long_ma) / long_ma.replace(0, 1) * 100

# Volume Oscillator
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class VolumeOscillatorGenerator(VectorizedFeatureGenerator):
    """Generator for Volume Oscillator."""
    
    def __init__(self, short_period: int = 10, long_period: int = 20):
        config = FeatureConfig(
            name=f"volume_oscillator_{short_period}_{long_period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Oscillator using {short_period} and {long_period} periods",
            required_columns=["volume"],
            default_lookback=long_period,
            min_lookback=long_period,
            max_lookback=long_period,
            parameters={'short_period': short_period, 'long_period': long_period}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.short_period = short_period
        self.long_period = long_period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Volume Oscillator."""
        volume = data['volume']
        short_ma = volume.rolling(window=self.short_period).mean()
        long_ma = volume.rolling(window=self.long_period).mean()
        return short_ma - long_ma

# Volume Momentum
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class VolumeMomentumGenerator(VectorizedFeatureGenerator):
    """Generator for Volume Momentum."""
    
    def __init__(self, period: int = 10):
        config = FeatureConfig(
            name=f"volume_momentum_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Momentum over {period} periods",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Volume Momentum."""
        volume = data['volume']
        return volume - volume.shift(self.period)

# Volume Weighted Average Price (VWAP)
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class VolumeVWAPGenerator(VectorizedFeatureGenerator):
    """Generator for Volume Weighted Average Price."""
    
    def __init__(self, period: int = 20):
        config = FeatureConfig(
            name=f"volume_vwap_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Weighted Average Price over {period} periods",
            required_columns=["close", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Volume VWAP."""
        close = data['close']
        volume = data['volume']
        return (close * volume).rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()

# Volume Price Trend (VPT)
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class VolumePriceTrendGenerator(VectorizedFeatureGenerator):
    """Generator for Volume Price Trend."""
    
    def __init__(self):
        config = FeatureConfig(
            name="volume_price_trend",
            category=FeatureCategory.VOLUME,
            description="Volume Price Trend indicator",
            required_columns=["close", "volume"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            parameters={}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Volume Price Trend."""
        close = data['close']
        volume = data['volume']
        price_change = close.pct_change()
        vpt = (price_change * volume).cumsum()
        return vpt

# Volume Accumulation/Distribution
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class VolumeAccumulationDistributionGenerator(VectorizedFeatureGenerator):
    """Generator for Volume Accumulation/Distribution."""
    
    def __init__(self):
        config = FeatureConfig(
            name="volume_accumulation_distribution",
            category=FeatureCategory.VOLUME,
            description="Volume Accumulation/Distribution indicator",
            required_columns=["close", "high", "low", "volume"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            parameters={}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Volume Accumulation/Distribution."""
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Calculate Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low).replace(0, 1)
        mfm = mfm.clip(-1, 1)  # Clamp between -1 and 1
        
        # Calculate Money Flow Volume
        mfv = mfm * volume
        
        return mfv.cumsum()


    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class VolumePriceCorrelationGenerator(VectorizedFeatureGenerator):
    """Generator for volume-price correlation features."""

    def __init__(self, period: int = 20):
        config = FeatureConfig(
            name=f"volume_price_correlation_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Correlation between volume and price over {period} periods",
            required_columns=["close", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period * 2,
            parameters={"period": period}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Calculate volume-price correlation."""
        close = data['close']
        volume = data['volume']

        # Rolling correlation between price returns and volume
        price_returns = close.pct_change()
        correlation = price_returns.rolling(window=self.config.parameters["period"]).corr(volume)

        return correlation


    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class VolumePriceDivergenceGenerator(VectorizedFeatureGenerator):
    """Generator for volume-price divergence features."""

    def __init__(self, period: int = 20):
        config = FeatureConfig(
            name=f"volume_price_divergence_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume-price divergence indicator over {period} periods",
            required_columns=["close", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period * 2,
            parameters={"period": period}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Calculate enhanced volume-price divergence with regime awareness."""
        close = data['close']
        volume = data['volume']

        # Price momentum with regime smoothing
        price_ma = close.rolling(window=self.config.parameters["period"]).mean()
        price_momentum = (close - price_ma) / (price_ma + 1e-8)  # Avoid division by zero

        # Volume momentum with regime smoothing
        volume_ma = volume.rolling(window=self.config.parameters["period"]).mean()
        volume_momentum = (volume - volume_ma) / (volume_ma + 1e-8)  # Avoid division by zero

        # Enhanced divergence with regime persistence
        divergence = price_momentum * volume_momentum
        
        # Add regime stability measure
        price_volatility = close.rolling(window=self.config.parameters["period"]).std()
        volume_volatility = volume.rolling(window=self.config.parameters["period"]).std()
        
        # Regime strength indicator (higher when both price and volume show consistent trends)
        regime_strength = np.abs(divergence) / (price_volatility * volume_volatility + 1e-8)
        
        # Combine divergence with regime strength for better clustering
        enhanced_divergence = divergence * (1 + regime_strength)
        
        return enhanced_divergence


    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class PriceVolumeOscillatorGenerator(VectorizedFeatureGenerator):
    """Generator for price-volume oscillator features."""

    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        config = FeatureConfig(
            name=f"price_volume_oscillator_{fast_period}_{slow_period}",
            category=FeatureCategory.VOLUME,
            description=f"Price-volume oscillator ({fast_period}/{slow_period})",
            required_columns=["close", "volume"],
            default_lookback=slow_period,
            min_lookback=slow_period,
            max_lookback=slow_period * 2,
            parameters={"fast_period": fast_period, "slow_period": slow_period}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Calculate price-volume oscillator."""
        close = data['close']
        volume = data['volume']

        # Price oscillator
        fast_ma = close.rolling(window=self.config.parameters["fast_period"]).mean()
        slow_ma = close.rolling(window=self.config.parameters["slow_period"]).mean()
        price_osc = (fast_ma - slow_ma) / slow_ma

        # Volume oscillator
        volume_fast_ma = volume.rolling(window=self.config.parameters["fast_period"]).mean()
        volume_slow_ma = volume.rolling(window=self.config.parameters["slow_period"]).mean()
        volume_osc = (volume_fast_ma - volume_slow_ma) / volume_slow_ma

        # Combined oscillator
        combined_osc = price_osc * volume_osc

        return combined_osc


def create_default_volume_generators() -> List[FeatureGenerator]:
    """Create default volume feature generators."""
    generators = []
    
    # Volume moving averages
    for period in [5, 10, 20, 50]:
        generators.append(VolumeSMAGenerator(period))
        generators.append(VolumeEMAGenerator(period))
    
    # Volume ratios
    for period in [10, 20, 50]:
        generators.append(VolumeRatioGenerator(period))
    
    # Volume rate of change
    for period in [1, 5, 10, 20]:
        generators.append(VolumeROCGenerator(period))
    
    # Volume standard deviation
    for period in [10, 20, 50]:
        generators.append(VolumeStdGenerator(period))
    
    # Volume percentile rank
    for period in [20, 50, 100]:
        generators.append(VolumePercentileGenerator(period))
    
    # Volume trend strength
    generators.append(VolumeTrendStrengthGenerator(10, 30))
    generators.append(VolumeTrendStrengthGenerator(20, 50))
    
    # Volume oscillator
    generators.append(VolumeOscillatorGenerator(10, 20))
    generators.append(VolumeOscillatorGenerator(5, 15))
    
    # Volume momentum
    for period in [5, 10, 20]:
        generators.append(VolumeMomentumGenerator(period))
    
    # Volume VWAP
    for period in [10, 20, 50]:
        generators.append(VolumeVWAPGenerator(period))
    
    # Volume Price Trend
    generators.append(VolumePriceTrendGenerator())
    
    # Volume Accumulation/Distribution
    generators.append(VolumeAccumulationDistributionGenerator())

    # Volume-Price Divergence features for regime identification
    for period in [10, 20]:
        generators.append(VolumePriceCorrelationGenerator(period))
        generators.append(VolumePriceDivergenceGenerator(period))

    # Price-Volume Oscillator
    generators.append(PriceVolumeOscillatorGenerator(10, 20))
    generators.append(PriceVolumeOscillatorGenerator(5, 15))

    # Analyst Features - Volume patterns
    generators.append(AnalystVolumePressureGenerator())
    generators.append(AnalystVolumeTrendGenerator())
    
    # NEW FEATURES - Enhanced Volume Analysis
    # Volume z-score generators
    for short_window in [60]:
        for long_window in [252]:
            generators.append(VolumeZScoreGenerator(short_window, long_window))
    
    # Volume MA ratios generators
    for ma_period in [20]:
        for surprise_window in [10]:
            generators.append(VolumeMARatiosGenerator(ma_period, surprise_window))
    
    # CMF generators
    for period in [20]:
        generators.append(CMFGenerator(period))
    
    # VWAP deviations generators
    for vwap_window in [20]:
        generators.append(VWAPDeviationsGenerator(vwap_window))
    
    # Order flow imbalance generators
    for window in [20]:
        generators.append(OrderFlowImbalanceGenerator(window))
    
    # Volume-volatility elasticity generators
    for window in [20]:
        generators.append(VolumeVolatilityElasticityGenerator(window))

    return generators

# Analyst Features - Volume pattern generators
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class AnalystVolumePressureGenerator(VectorizedFeatureGenerator):
    """Generator for volume pressure feature."""

    def __init__(self):
        config = FeatureConfig(
            name="analyst_volume_pressure",
            category=FeatureCategory.VOLUME,
            description="Analyst volume pressure ((buy_volume - sell_volume) / total_volume)",
            required_columns=["volume", "close"],
            default_lookback=20,
            min_lookback=10,
            max_lookback=100,
            parameters={}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate volume pressure feature."""
        volume = data['volume']
        price_change = data['close'].pct_change()

        # Use price movement direction as proxy for buy/sell pressure
        volume_up = volume.where(price_change > 0, 0)
        volume_down = volume.where(price_change < 0, 0)

        volume_pressure = (volume_up - volume_down) / volume.replace(0, 1)
        return volume_pressure

    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class AnalystVolumeTrendGenerator(VectorizedFeatureGenerator):
    """Generator for volume trend using linear regression."""

    def __init__(self, lookback: int = 20):
        config = FeatureConfig(
            name="analyst_volume_trend",
            category=FeatureCategory.VOLUME,
            description="Analyst volume trend using linear regression slope",
            required_columns=["volume"],
            default_lookback=lookback,
            min_lookback=10,
            max_lookback=100,
            parameters={"lookback": lookback}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.lookback = lookback

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate volume trend feature."""
        volume = data['volume']

        def volume_trend(x):
            if len(x) < 10:
                return 0.0
            try:
                from scipy.stats import linregress

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

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
                slope, _, _, _, _ = linregress(range(len(x)), x.values)
                return slope
            except:
                return 0.0

        volume_trend_values = volume.rolling(self.lookback).apply(volume_trend)
        return volume_trend_values

__all__ = [
    'VolumeFeatureGenerator',
    'VolumeSMAGenerator',
    'VolumeEMAGenerator',
    'VolumeRatioGenerator',
    'VolumeROCGenerator',
    'VolumeStdGenerator',
    'VolumePercentileGenerator',
    'VolumeTrendStrengthGenerator',
    'VolumeOscillatorGenerator',
    'VolumeMomentumGenerator',
    'VolumeVWAPGenerator',
    'VolumePriceTrendGenerator',
    'VolumeAccumulationDistributionGenerator',
    'VolumePriceCorrelationGenerator',
    'VolumePriceDivergenceGenerator',
    'PriceVolumeOscillatorGenerator',
    'AnalystVolumePressureGenerator',
    'AnalystVolumeTrendGenerator',
    'create_default_volume_generators'
]

# NEW FEATURES - Enhanced Volume Analysis

class VolumeZScoreGenerator(VectorizedFeatureGenerator):
    """Generator for volume z-score vs 60/252-bar history."""
    
    def __init__(self, short_window: int = 60, long_window: int = 252):
        config = FeatureConfig(
            name=f"volume_zscore_{short_window}_{long_window}",
            category=FeatureCategory.VOLUME,
            description=f"Volume z-score vs {short_window}/{long_window}-bar history",
            required_columns=["volume"],
            default_lookback=long_window,
            min_lookback=long_window,
            max_lookback=long_window,
            parameters={'short_window': short_window, 'long_window': long_window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.short_window = short_window
        self.long_window = long_window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        volume = data['volume'].values
        if len(volume) < self.long_window:
            return pd.Series(np.full(len(volume), np.nan), index=data.index)
        
        # Calculate volume z-score
        volume_zscore = np.full(len(volume), np.nan)
        for i in range(self.long_window - 1, len(volume)):
            # Short-term mean and std
            short_window_vol = volume[i - self.short_window + 1:i + 1]
            short_mean = np.mean(short_window_vol)
            short_std = np.std(short_window_vol, ddof=1)
            
            # Long-term mean and std
            long_window_vol = volume[i - self.long_window + 1:i + 1]
            long_mean = np.mean(long_window_vol)
            long_std = np.std(long_window_vol, ddof=1)
            
            if long_std > 0:
                volume_zscore[i] = (volume[i] - long_mean) / long_std
        
        return pd.Series(volume_zscore, index=data.index)

class VolumeMARatiosGenerator(VectorizedFeatureGenerator):
    """Generator for volume MA ratios and volume surprise."""
    
    def __init__(self, ma_period: int = 20, surprise_window: int = 10):
        config = FeatureConfig(
            name=f"volume_ma_ratios_{ma_period}_{surprise_window}",
            category=FeatureCategory.VOLUME,
            description=f"Volume MA ratios and surprise over {ma_period}/{surprise_window} periods",
            required_columns=["volume"],
            default_lookback=ma_period + surprise_window,
            min_lookback=ma_period + surprise_window,
            max_lookback=ma_period + surprise_window,
            parameters={'ma_period': ma_period, 'surprise_window': surprise_window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.ma_period = ma_period
        self.surprise_window = surprise_window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        volume = data['volume'].values
        if len(volume) < self.ma_period + self.surprise_window:
            return pd.Series(np.full(len(volume), np.nan), index=data.index)
        
        # Calculate volume MA ratios
        volume_ma_ratios = np.full(len(volume), np.nan)
        volume_surprise = np.full(len(volume), np.nan)
        
        for i in range(self.ma_period + self.surprise_window - 1, len(volume)):
            # Volume MA ratio
            ma_window = volume[i - self.ma_period + 1:i + 1]
            ma_volume = np.mean(ma_window)
            if ma_volume > 0:
                volume_ma_ratios[i] = volume[i] / ma_volume
            
            # Volume surprise (actual - expected)
            if i >= self.surprise_window:
                expected_window = volume[i - self.surprise_window:i]
                expected_volume = np.mean(expected_window)
                volume_surprise[i] = volume[i] - expected_volume
        
        return pd.Series(volume_ma_ratios, index=data.index)

class CMFGenerator(VectorizedFeatureGenerator):
    """Generator for Chaikin Money Flow (CMF)."""
    
    def __init__(self, period: int = 20):
        config = FeatureConfig(
            name=f"cmf_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Chaikin Money Flow over {period} periods",
            required_columns=["close", "high", "low", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values
        
        if len(close) < self.period:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate CMF
        cmf = np.full(len(close), np.nan)
        for i in range(self.period - 1, len(close)):
            # Money Flow Multiplier
            mfm = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            mfm = np.nan_to_num(mfm, nan=0.0)  # Handle division by zero
            
            # Money Flow Volume
            mfv = mfm * volume[i]
            
            # CMF = sum(MFV) / sum(Volume) over period
            period_mfv = []
            period_vol = []
            for j in range(i - self.period + 1, i + 1):
                if high[j] != low[j]:  # Avoid division by zero
                    period_mfm = ((close[j] - low[j]) - (high[j] - close[j])) / (high[j] - low[j])
                    period_mfm = np.nan_to_num(period_mfm, nan=0.0)
                    period_mfv.append(period_mfm * volume[j])
                    period_vol.append(volume[j])
            
            if len(period_vol) > 0 and sum(period_vol) > 0:
                cmf[i] = sum(period_mfv) / sum(period_vol)
        
        return pd.Series(cmf, index=data.index)

class VWAPDeviationsGenerator(VectorizedFeatureGenerator):
    """Generator for VWAP deviations and closing-VWAP gap."""
    
    def __init__(self, vwap_window: int = 20):
        config = FeatureConfig(
            name=f"vwap_deviations_{vwap_window}",
            category=FeatureCategory.VOLUME,
            description=f"VWAP deviations and closing-VWAP gap over {vwap_window} periods",
            required_columns=["close", "high", "low", "volume"],
            default_lookback=vwap_window,
            min_lookback=vwap_window,
            max_lookback=vwap_window,
            parameters={'vwap_window': vwap_window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.vwap_window = vwap_window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values
        
        if len(close) < self.vwap_window:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate VWAP deviations
        vwap_deviations = np.full(len(close), np.nan)
        closing_vwap_gap = np.full(len(close), np.nan)
        
        for i in range(self.vwap_window - 1, len(close)):
            # Calculate VWAP for the window
            window_high = high[i - self.vwap_window + 1:i + 1]
            window_low = low[i - self.vwap_window + 1:i + 1]
            window_close = close[i - self.vwap_window + 1:i + 1]
            window_volume = volume[i - self.vwap_window + 1:i + 1]
            
            # Typical price
            typical_price = (window_high + window_low + window_close) / 3
            
            # VWAP
            vwap = np.sum(typical_price * window_volume) / np.sum(window_volume)
            
            if vwap > 0:
                # VWAP deviation
                vwap_deviations[i] = (close[i] - vwap) / vwap
                
                # Closing-VWAP gap
                closing_vwap_gap[i] = close[i] - vwap
        
        return pd.Series(vwap_deviations, index=data.index)

class OrderFlowImbalanceGenerator(VectorizedFeatureGenerator):
    """Generator for order flow imbalance (signed volume)."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"order_flow_imbalance_{window}",
            category=FeatureCategory.VOLUME,
            description=f"Order flow imbalance over {window} periods",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        volume = data['volume'].values
        
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate order flow imbalance
        ofi = np.full(len(close), np.nan)
        
        for i in range(1, len(close)):
            # Price change direction
            price_change = close[i] - close[i-1]
            
            # Signed volume (positive for buying pressure, negative for selling pressure)
            if price_change > 0:
                signed_volume = volume[i]
            elif price_change < 0:
                signed_volume = -volume[i]
            else:
                signed_volume = 0
            
            # Rolling sum of signed volume
            if i >= self.window:
                window_signed_vol = []
                for j in range(i - self.window + 1, i + 1):
                    if j > 0:
                        price_chg = close[j] - close[j-1]
                        if price_chg > 0:
                            window_signed_vol.append(volume[j])
                        elif price_chg < 0:
                            window_signed_vol.append(-volume[j])
                        else:
                            window_signed_vol.append(0)
                
                ofi[i] = sum(window_signed_vol)
        
        return pd.Series(ofi, index=data.index)

class VolumeVolatilityElasticityGenerator(VectorizedFeatureGenerator):
    """Generator for volume-volatility elasticity."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"volume_volatility_elasticity_{window}",
            category=FeatureCategory.VOLUME,
            description=f"Volume-volatility elasticity over {window} periods",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        volume = data['volume'].values
        
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns and absolute returns
        returns = np.diff(close) / close[:-1]
        abs_returns = np.abs(returns)
        returns = np.concatenate([[np.nan], returns])
        abs_returns = np.concatenate([[np.nan], abs_returns])
        
        # Calculate volume-volatility elasticity
        elasticity = np.full(len(close), np.nan)
        
        for i in range(self.window, len(close)):
            window_abs_returns = abs_returns[i - self.window + 1:i + 1]
            window_volume = volume[i - self.window + 1:i + 1]
            
            # Filter out NaN values
            valid_mask = np.isfinite(window_abs_returns) & np.isfinite(window_volume)
            if np.sum(valid_mask) > 1:
                valid_abs_returns = window_abs_returns[valid_mask]
                valid_volume = window_volume[valid_mask]
                
                # Calculate correlation
                if len(valid_abs_returns) > 1 and np.std(valid_abs_returns) > 0 and np.std(valid_volume) > 0:
                    correlation = np.corrcoef(valid_abs_returns, valid_volume)[0, 1]
                    if not np.isnan(correlation):
                        elasticity[i] = correlation
        
        return pd.Series(elasticity, index=data.index)

