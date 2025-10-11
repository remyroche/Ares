"""
Volume Feature Generator

This module provides feature generators for basic volume-based indicators,
including volume moving averages, ratios, rate of change, and other volume metrics.
Enhanced with VectorBT for maximum performance.
"""

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..core.vectorbt_optimization_mixin import VectorBTOptimizationMixin

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

# VectorBT Rolling Optimizer
try:
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
    VECTORBT_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZER_AVAILABLE = False

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

class VolumeFeatureGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Feature generator for basic volume-based features with VectorBT optimization."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
    
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
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                volume_sma = self.vectorbt_optimizer.rolling_mean(volume, window=20)
                return volume_sma
            except Exception as e:
                self.logger.warning(f"VectorBT volume calculation failed: {e}, using pandas fallback")
                return volume.rolling(window=20).mean()
        else:
            return volume.rolling(window=20).mean()

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
    
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (VECTORBT_AVAILABLE and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000))
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return self.vectorbt_optimizer.rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return self.vectorbt_optimizer.rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return self.vectorbt_optimizer.rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return self.vectorbt_optimizer.rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return self.vectorbt_optimizer.rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return self.vectorbt_optimizer.rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            self.logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
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

class VolumeSMAGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Simple Moving Average with VectorBT optimization."""
    
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
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume']
        
        # Use VectorBT for optimized rolling mean
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                return self.vectorbt_optimizer.rolling_mean(volume, window=self.period)
            except Exception as e:
                self.logger.warning(f"VectorBT rolling mean failed: {e}, using pandas fallback")
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

class VolumeEMAGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Exponential Moving Average with VectorBT optimization."""
    
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
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume']
        
        # Use VectorBT for volume EMA calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                # VectorBT doesn't have direct EMA, so we use ewm with alpha
                return volume.ewm(alpha=self.alpha, adjust=False).mean()
            except Exception as e:
                self.logger.warning(f"VectorBT volume EMA calculation failed: {e}, using pandas fallback")
                return volume.ewm(alpha=self.alpha, adjust=False).mean()
        else:
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

class VolumeRatioGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Ratio (current volume vs average volume) with VectorBT optimization."""
    
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
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume']
        
        # Use VectorBT for volume ratio calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                avg_volume = self.vectorbt_optimizer.rolling_mean(volume, window=self.period)
                return volume / avg_volume.replace(0, 1)  # Avoid division by zero
            except Exception as e:
                self.logger.warning(f"VectorBT volume ratio calculation failed: {e}, using pandas fallback")
                avg_volume = volume.rolling(window=self.period).mean()
                return volume / avg_volume.replace(0, 1)  # Avoid division by zero
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

class VolumeROCGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Rate of Change with VectorBT optimization."""
    
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
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume']
        
        # Use VectorBT for volume ROC calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                # VectorBT doesn't have direct pct_change, so we calculate it manually
                roc = (volume / volume.shift(self.period) - 1) * 100
                return roc
            except Exception as e:
                self.logger.warning(f"VectorBT volume ROC calculation failed: {e}, using pandas fallback")
                return volume.pct_change(periods=self.period) * 100
        else:
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

class VolumeStdGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Standard Deviation with VectorBT optimization."""
    
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
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume']
        
        # Use VectorBT for volume standard deviation calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                return self.vectorbt_optimizer.rolling_std(volume, window=self.period)
            except Exception as e:
                self.logger.warning(f"VectorBT volume std calculation failed: {e}, using pandas fallback")
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

class VolumePercentileGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Percentile Rank with VectorBT optimization."""
    
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
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume']
        
        # Use VectorBT for volume percentile rank calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                # VectorBT doesn't have direct rank, so we use pandas rank
                return volume.rolling(window=self.period).rank(pct=True) * 100
            except Exception as e:
                self.logger.warning(f"VectorBT volume percentile calculation failed: {e}, using pandas fallback")
                return volume.rolling(window=self.period).rank(pct=True) * 100
        else:
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

class VolumeTrendStrengthGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Trend Strength with VectorBT optimization."""
    
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
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume']
        
        # Use VectorBT for volume trend strength calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                short_ma = self.vectorbt_optimizer.rolling_mean(volume, window=self.short_period)
                long_ma = self.vectorbt_optimizer.rolling_mean(volume, window=self.long_period)
                return (short_ma - long_ma) / long_ma.replace(0, 1) * 100
            except Exception as e:
                self.logger.warning(f"VectorBT volume trend strength calculation failed: {e}, using pandas fallback")
                short_ma = volume.rolling(window=self.short_period).mean()
                long_ma = volume.rolling(window=self.long_period).mean()
                return (short_ma - long_ma) / long_ma.replace(0, 1) * 100
        else:
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

class VolumeOscillatorGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Oscillator with VectorBT optimization."""
    
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
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume']
        
        # Use VectorBT for volume oscillator calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                short_ma = self.vectorbt_optimizer.rolling_mean(volume, window=self.short_period)
                long_ma = self.vectorbt_optimizer.rolling_mean(volume, window=self.long_period)
                return short_ma - long_ma
            except Exception as e:
                self.logger.warning(f"VectorBT volume oscillator calculation failed: {e}, using pandas fallback")
                short_ma = volume.rolling(window=self.short_period).mean()
                long_ma = volume.rolling(window=self.long_period).mean()
                return short_ma - long_ma
        else:
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

class VolumeMomentumGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Momentum with VectorBT optimization."""
    
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
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume']
        
        # Use VectorBT for volume momentum calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                # VectorBT doesn't have direct shift, so we use pandas shift
                return volume - volume.shift(self.period)
            except Exception as e:
                self.logger.warning(f"VectorBT volume momentum calculation failed: {e}, using pandas fallback")
                return volume - volume.shift(self.period)
        else:
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

class VolumeVWAPGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Weighted Average Price with VectorBT optimization."""
    
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
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        volume = data['volume']
        
        # Use VectorBT for volume VWAP calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                price_volume = close * volume
                price_volume_sum = self.vectorbt_optimizer.rolling_sum(price_volume, window=self.period)
                volume_sum = self.vectorbt_optimizer.rolling_sum(volume, window=self.period)
                return price_volume_sum / volume_sum
            except Exception as e:
                self.logger.warning(f"VectorBT volume VWAP calculation failed: {e}, using pandas fallback")
                return (close * volume).rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()
        else:
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

class VolumePriceTrendGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Price Trend with VectorBT optimization."""
    
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
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        volume = data['volume']
        
        # Use VectorBT for volume price trend calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                price_change = close.pct_change()
                vpt = (price_change * volume).cumsum()
                return vpt
            except Exception as e:
                self.logger.warning(f"VectorBT volume price trend calculation failed: {e}, using pandas fallback")
                price_change = close.pct_change()
                vpt = (price_change * volume).cumsum()
                return vpt
        else:
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

class VolumeAccumulationDistributionGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Accumulation/Distribution with VectorBT optimization."""
    
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
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Use VectorBT for volume accumulation/distribution calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                # Calculate Money Flow Multiplier
                mfm = ((close - low) - (high - close)) / (high - low).replace(0, 1)
                mfm = mfm.clip(-1, 1)  # Clamp between -1 and 1
                
                # Calculate Money Flow Volume
                mfv = mfm * volume
                
                return mfv.cumsum()
            except Exception as e:
                self.logger.warning(f"VectorBT volume accumulation/distribution calculation failed: {e}, using pandas fallback")
                # Calculate Money Flow Multiplier
                mfm = ((close - low) - (high - close)) / (high - low).replace(0, 1)
                mfm = mfm.clip(-1, 1)  # Clamp between -1 and 1
                
                # Calculate Money Flow Volume
                mfv = mfm * volume
                
                return mfv.cumsum()
        else:
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

class VolumePriceCorrelationGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for volume-price correlation features with VectorBT optimization."""

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
        self.period = period
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        volume = data['volume']

        # Use VectorBT for volume-price correlation calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                # Rolling correlation between price returns and volume
                price_returns = close.pct_change()
                correlation = self.vectorbt_optimizer.rolling_corr(price_returns, volume, window=self.period)
                return correlation
            except Exception as e:
                self.logger.warning(f"VectorBT volume-price correlation calculation failed: {e}, using pandas fallback")
                price_returns = close.pct_change()
                correlation = price_returns.rolling(window=self.period).corr(volume)
                return correlation
        else:
            # Rolling correlation between price returns and volume
            price_returns = close.pct_change()
            correlation = price_returns.rolling(window=self.period).corr(volume)
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

class VolumePriceDivergenceGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for volume-price divergence features with VectorBT optimization."""

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
        self.period = period
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        volume = data['volume']

        # Use VectorBT for volume-price divergence calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                # Price momentum with regime smoothing
                price_ma = self.vectorbt_optimizer.rolling_mean(close, window=self.period)
                price_momentum = (close - price_ma) / (price_ma + 1e-8)  # Avoid division by zero

                # Volume momentum with regime smoothing
                volume_ma = self.vectorbt_optimizer.rolling_mean(volume, window=self.period)
                volume_momentum = (volume - volume_ma) / (volume_ma + 1e-8)  # Avoid division by zero

                # Enhanced divergence with regime persistence
                divergence = price_momentum * volume_momentum
                
                # Add regime stability measure
                price_volatility = self.vectorbt_optimizer.rolling_std(close, window=self.period)
                volume_volatility = self.vectorbt_optimizer.rolling_std(volume, window=self.period)
                
                # Regime strength indicator (higher when both price and volume show consistent trends)
                regime_strength = np.abs(divergence) / (price_volatility * volume_volatility + 1e-8)
                
                # Combine divergence with regime strength for better clustering
                enhanced_divergence = divergence * (1 + regime_strength)
                
                return enhanced_divergence
            except Exception as e:
                self.logger.warning(f"VectorBT volume-price divergence calculation failed: {e}, using pandas fallback")
                # Price momentum with regime smoothing
                price_ma = close.rolling(window=self.period).mean()
                price_momentum = (close - price_ma) / (price_ma + 1e-8)  # Avoid division by zero

                # Volume momentum with regime smoothing
                volume_ma = volume.rolling(window=self.period).mean()
                volume_momentum = (volume - volume_ma) / (volume_ma + 1e-8)  # Avoid division by zero

                # Enhanced divergence with regime persistence
                divergence = price_momentum * volume_momentum
                
                # Add regime stability measure
                price_volatility = close.rolling(window=self.period).std()
                volume_volatility = volume.rolling(window=self.period).std()
                
                # Regime strength indicator (higher when both price and volume show consistent trends)
                regime_strength = np.abs(divergence) / (price_volatility * volume_volatility + 1e-8)
                
                # Combine divergence with regime strength for better clustering
                enhanced_divergence = divergence * (1 + regime_strength)
                
                return enhanced_divergence
        else:
            # Price momentum with regime smoothing
            price_ma = close.rolling(window=self.period).mean()
            price_momentum = (close - price_ma) / (price_ma + 1e-8)  # Avoid division by zero

            # Volume momentum with regime smoothing
            volume_ma = volume.rolling(window=self.period).mean()
            volume_momentum = (volume - volume_ma) / (volume_ma + 1e-8)  # Avoid division by zero

            # Enhanced divergence with regime persistence
            divergence = price_momentum * volume_momentum
            
            # Add regime stability measure
            price_volatility = close.rolling(window=self.period).std()
            volume_volatility = volume.rolling(window=self.period).std()
            
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

class PriceVolumeOscillatorGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for price-volume oscillator features with VectorBT optimization."""

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
        self.fast_period = fast_period
        self.slow_period = slow_period
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        volume = data['volume']

        # Use VectorBT for price-volume oscillator calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                # Price oscillator
                fast_ma = self.vectorbt_optimizer.rolling_mean(close, window=self.fast_period)
                slow_ma = self.vectorbt_optimizer.rolling_mean(close, window=self.slow_period)
                price_osc = (fast_ma - slow_ma) / slow_ma

                # Volume oscillator
                volume_fast_ma = self.vectorbt_optimizer.rolling_mean(volume, window=self.fast_period)
                volume_slow_ma = self.vectorbt_optimizer.rolling_mean(volume, window=self.slow_period)
                volume_osc = (volume_fast_ma - volume_slow_ma) / volume_slow_ma

                # Combined oscillator
                combined_osc = price_osc * volume_osc

                return combined_osc
            except Exception as e:
                self.logger.warning(f"VectorBT price-volume oscillator calculation failed: {e}, using pandas fallback")
                # Price oscillator
                fast_ma = close.rolling(window=self.fast_period).mean()
                slow_ma = close.rolling(window=self.slow_period).mean()
                price_osc = (fast_ma - slow_ma) / slow_ma

                # Volume oscillator
                volume_fast_ma = volume.rolling(window=self.fast_period).mean()
                volume_slow_ma = volume.rolling(window=self.slow_period).mean()
                volume_osc = (volume_fast_ma - volume_slow_ma) / volume_slow_ma

                # Combined oscillator
                combined_osc = price_osc * volume_osc

                return combined_osc
        else:
            # Price oscillator
            fast_ma = close.rolling(window=self.fast_period).mean()
            slow_ma = close.rolling(window=self.slow_period).mean()
            price_osc = (fast_ma - slow_ma) / slow_ma

            # Volume oscillator
            volume_fast_ma = volume.rolling(window=self.fast_period).mean()
            volume_slow_ma = volume.rolling(window=self.slow_period).mean()
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

        # Use VectorBT for volume pressure calculation
        if VECTORBT_AVAILABLE:
            try:
                # Use price movement direction as proxy for buy/sell pressure
                volume_up = volume.where(price_change > 0, 0)
                volume_down = volume.where(price_change < 0, 0)

                volume_pressure = (volume_up - volume_down) / volume.replace(0, 1)
                return volume_pressure
            except Exception as e:
                self.logger.warning(f"VectorBT volume pressure calculation failed: {e}, using pandas fallback")
                # Use price movement direction as proxy for buy/sell pressure
                volume_up = volume.where(price_change > 0, 0)
                volume_down = volume.where(price_change < 0, 0)

                volume_pressure = (volume_up - volume_down) / volume.replace(0, 1)
                return volume_pressure
        else:
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

        # Use VectorBT for volume trend calculation
        if VECTORBT_AVAILABLE:
            try:
                def volume_trend(x):
                    if len(x) < 10:
                        return 0.0
                    try:
                        from scipy.stats import linregress
                        slope, _, _, _, _ = linregress(range(len(x)), x.values)
                        return slope
                    except:
                        return 0.0

                volume_trend_values = volume.rolling(self.lookback).apply(volume_trend)
                return volume_trend_values
            except Exception as e:
                self.logger.warning(f"VectorBT volume trend calculation failed: {e}, using pandas fallback")
                def volume_trend(x):
                    if len(x) < 10:
                        return 0.0
                    try:
                        from scipy.stats import linregress
                        slope, _, _, _, _ = linregress(range(len(x)), x.values)
                        return slope
                    except:
                        return 0.0

                volume_trend_values = volume.rolling(self.lookback).apply(volume_trend)
                return volume_trend_values
        else:
            def volume_trend(x):
                if len(x) < 10:
                    return 0.0
                try:
                    from scipy.stats import linregress
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

