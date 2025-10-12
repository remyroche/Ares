"""
Refactored Volume Feature Generator

This module provides refactored volume feature generators that use centralized
utilities to eliminate code duplication and improve performance.
"""

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Any, Dict, List, Optional, Union

from ..core.unified_feature_generator import UnifiedFeatureGenerator, UnifiedFeatureConfig
from ..core.feature_generator import FeatureResult, FeatureCategory
from ..utils.centralized_rolling_manager import get_centralized_rolling_manager, RollingOperation
from ..utils.scaler_factory import get_scaler_factory, ScalerType
from ..utils.common_operations import get_common_operations

logger = logging.getLogger(__name__)

class RefactoredVolumeFeatureGenerator(UnifiedFeatureGenerator):
    """Refactored volume feature generator using centralized utilities."""
    
    def __init__(self, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config)
    
    @classmethod
    def _create_default_config(cls) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name="refactored_volume_features",
            category=FeatureCategory.VOLUME,
            description="Refactored volume features using centralized utilities",
            required_columns=["volume"],
            optional_columns=["close", "high", "low", "open"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "volume_periods": [20, 50, 100],
                "volume_ratios": [10, 20],
                "volume_oscillators": [12, 26]
            },
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='robust',
            normalization_feature_type='volume',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volume features using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        # Calculate volume using centralized rolling operations
        volume = data['volume']
        volume_ma = self.rolling_mean(volume, window=20)
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            volume_ma = self.normalize_feature(volume_ma, feature_type='volume')
        
        return volume_ma.rename('refactored_volume_ma_20')

class RefactoredVolumeSMAGenerator(UnifiedFeatureGenerator):
    """Refactored Volume SMA generator using centralized utilities."""
    
    def __init__(self, period: int = 20, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_volume_sma_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Refactored Volume SMA with period {period} using centralized utilities",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='robust',
            normalization_feature_type='volume',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume SMA using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        volume = data['volume']
        
        # Use centralized rolling operations
        volume_sma = self.rolling_mean(volume, window=self.period)
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            volume_sma = self.normalize_feature(volume_sma, feature_type='volume')
        
        return volume_sma.rename(f'refactored_volume_sma_{self.period}')

class RefactoredVolumeEMAGenerator(UnifiedFeatureGenerator):
    """Refactored Volume EMA generator using centralized utilities."""
    
    def __init__(self, period: int = 20, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_volume_ema_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Refactored Volume EMA with period {period} using centralized utilities",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='robust',
            normalization_feature_type='volume',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume EMA using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        volume = data['volume']
        
        # Calculate EMA (exponential moving average)
        alpha = 2.0 / (self.period + 1)
        volume_ema = volume.ewm(alpha=alpha).mean()
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            volume_ema = self.normalize_feature(volume_ema, feature_type='volume')
        
        return volume_ema.rename(f'refactored_volume_ema_{self.period}')

class RefactoredVolumeRatioGenerator(UnifiedFeatureGenerator):
    """Refactored Volume Ratio generator using centralized utilities."""
    
    def __init__(self, period: int = 20, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_volume_ratio_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Refactored Volume Ratio with period {period} using centralized utilities",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='robust',
            normalization_feature_type='volume',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Ratio using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        volume = data['volume']
        
        # Calculate average volume using centralized rolling operations
        avg_volume = self.rolling_mean(volume, window=self.period)
        
        # Calculate volume ratio
        volume_ratio = volume / avg_volume.replace(0, 1)  # Avoid division by zero
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            volume_ratio = self.normalize_feature(volume_ratio, feature_type='volume')
        
        return volume_ratio.rename(f'refactored_volume_ratio_{self.period}')

class RefactoredVolumeROCGenerator(UnifiedFeatureGenerator):
    """Refactored Volume Rate of Change generator using centralized utilities."""
    
    def __init__(self, period: int = 20, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_volume_roc_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Refactored Volume ROC with period {period} using centralized utilities",
            required_columns=["volume"],
            default_lookback=period + 1,
            min_lookback=period + 1,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='zscore',
            normalization_feature_type='volume',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume ROC using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        volume = data['volume']
        
        # Calculate volume rate of change
        volume_roc = (volume / volume.shift(self.period) - 1) * 100
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            volume_roc = self.normalize_feature(volume_roc, feature_type='volume')
        
        return volume_roc.rename(f'refactored_volume_roc_{self.period}')

class RefactoredVolumeStdGenerator(UnifiedFeatureGenerator):
    """Refactored Volume Standard Deviation generator using centralized utilities."""
    
    def __init__(self, period: int = 20, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_volume_std_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Refactored Volume Std with period {period} using centralized utilities",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='robust',
            normalization_feature_type='volume',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Std using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        volume = data['volume']
        
        # Use centralized rolling operations
        volume_std = self.rolling_std(volume, window=self.period)
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            volume_std = self.normalize_feature(volume_std, feature_type='volume')
        
        return volume_std.rename(f'refactored_volume_std_{self.period}')

class RefactoredVolumePercentileGenerator(UnifiedFeatureGenerator):
    """Refactored Volume Percentile generator using centralized utilities."""
    
    def __init__(self, period: int = 20, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_volume_percentile_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Refactored Volume Percentile with period {period} using centralized utilities",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='minmax',
            normalization_feature_type='volume',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Percentile using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        volume = data['volume']
        
        # Calculate percentile rank using centralized rolling operations
        volume_rank = self.rolling_rank(volume, window=self.period)
        volume_percentile = (volume_rank / self.period) * 100
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            volume_percentile = self.normalize_feature(volume_percentile, feature_type='volume')
        
        return volume_percentile.rename(f'refactored_volume_percentile_{self.period}')

class RefactoredVolumeTrendStrengthGenerator(UnifiedFeatureGenerator):
    """Refactored Volume Trend Strength generator using centralized utilities."""
    
    def __init__(self, short_period: int = 10, long_period: int = 20, 
                 config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(short_period, long_period)
        super().__init__(config)
        self.short_period = short_period
        self.long_period = long_period
    
    @classmethod
    def _create_default_config(cls, short_period: int, long_period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_volume_trend_strength_{short_period}_{long_period}",
            category=FeatureCategory.VOLUME,
            description=f"Refactored Volume Trend Strength using centralized utilities",
            required_columns=["volume"],
            default_lookback=long_period,
            min_lookback=long_period,
            max_lookback=100,
            parameters={"short_period": short_period, "long_period": long_period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='zscore',
            normalization_feature_type='volume',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Trend Strength using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        volume = data['volume']
        
        # Calculate short and long moving averages using centralized rolling operations
        short_ma = self.rolling_mean(volume, window=self.short_period)
        long_ma = self.rolling_mean(volume, window=self.long_period)
        
        # Calculate trend strength
        trend_strength = (short_ma - long_ma) / long_ma.replace(0, 1) * 100
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            trend_strength = self.normalize_feature(trend_strength, feature_type='volume')
        
        return trend_strength.rename(f'refactored_volume_trend_strength_{self.short_period}_{self.long_period}')

class RefactoredVolumeOscillatorGenerator(UnifiedFeatureGenerator):
    """Refactored Volume Oscillator generator using centralized utilities."""
    
    def __init__(self, short_period: int = 12, long_period: int = 26, 
                 config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(short_period, long_period)
        super().__init__(config)
        self.short_period = short_period
        self.long_period = long_period
    
    @classmethod
    def _create_default_config(cls, short_period: int, long_period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_volume_oscillator_{short_period}_{long_period}",
            category=FeatureCategory.VOLUME,
            description=f"Refactored Volume Oscillator using centralized utilities",
            required_columns=["volume"],
            default_lookback=long_period,
            min_lookback=long_period,
            max_lookback=100,
            parameters={"short_period": short_period, "long_period": long_period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='zscore',
            normalization_feature_type='volume',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Oscillator using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        volume = data['volume']
        
        # Calculate short and long moving averages using centralized rolling operations
        short_ma = self.rolling_mean(volume, window=self.short_period)
        long_ma = self.rolling_mean(volume, window=self.long_period)
        
        # Calculate oscillator
        oscillator = short_ma - long_ma
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            oscillator = self.normalize_feature(oscillator, feature_type='volume')
        
        return oscillator.rename(f'refactored_volume_oscillator_{self.short_period}_{self.long_period}')

class RefactoredVolumeMomentumGenerator(UnifiedFeatureGenerator):
    """Refactored Volume Momentum generator using centralized utilities."""
    
    def __init__(self, period: int = 20, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_volume_momentum_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Refactored Volume Momentum with period {period} using centralized utilities",
            required_columns=["volume"],
            default_lookback=period + 1,
            min_lookback=period + 1,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='zscore',
            normalization_feature_type='volume',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Momentum using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        volume = data['volume']
        
        # Calculate volume momentum
        volume_momentum = volume - volume.shift(self.period)
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            volume_momentum = self.normalize_feature(volume_momentum, feature_type='volume')
        
        return volume_momentum.rename(f'refactored_volume_momentum_{self.period}')

class RefactoredVolumeVWAPGenerator(UnifiedFeatureGenerator):
    """Refactored Volume VWAP generator using centralized utilities."""
    
    def __init__(self, period: int = 20, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_volume_vwap_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Refactored Volume VWAP with period {period} using centralized utilities",
            required_columns=["high", "low", "close", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='minmax',
            normalization_feature_type='volume',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume VWAP using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate VWAP using centralized rolling operations
        volume_price = typical_price * volume
        volume_sum = self.rolling_sum(volume, window=self.period)
        price_volume_sum = self.rolling_sum(volume_price, window=self.period)
        
        vwap = price_volume_sum / volume_sum.replace(0, 1)  # Avoid division by zero
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            vwap = self.normalize_feature(vwap, feature_type='volume')
        
        return vwap.rename(f'refactored_volume_vwap_{self.period}')

class RefactoredVolumePriceTrendGenerator(UnifiedFeatureGenerator):
    """Refactored Volume Price Trend generator using centralized utilities."""
    
    def __init__(self, period: int = 20, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_volume_price_trend_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Refactored Volume Price Trend with period {period} using centralized utilities",
            required_columns=["close", "volume"],
            default_lookback=period + 1,
            min_lookback=period + 1,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='zscore',
            normalization_feature_type='volume',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Price Trend using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        volume = data['volume']
        
        # Calculate price change
        price_change = close.diff()
        
        # Calculate volume price trend
        vpt = (price_change / close.shift(1)) * volume
        vpt_cumulative = vpt.cumsum()
        
        # Apply smoothing using centralized rolling operations
        vpt_smoothed = self.rolling_mean(vpt_cumulative, window=self.period)
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            vpt_smoothed = self.normalize_feature(vpt_smoothed, feature_type='volume')
        
        return vpt_smoothed.rename(f'refactored_volume_price_trend_{self.period}')

# Batch volume generator for multiple features
class RefactoredBatchVolumeGenerator(UnifiedFeatureGenerator):
    """Refactored batch volume generator using centralized utilities."""
    
    def __init__(self, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config)
    
    @classmethod
    def _create_default_config(cls) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name="refactored_batch_volume_features",
            category=FeatureCategory.VOLUME,
            description="Refactored batch volume features using centralized utilities",
            required_columns=["volume"],
            optional_columns=["close", "high", "low", "open"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "volume_periods": [20, 50, 100],
                "volume_ratios": [10, 20],
                "volume_oscillators": [12, 26]
            },
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='robust',
            normalization_feature_type='volume',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate batch volume features using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        # Create batch configuration for multiple volume features
        batch_configs = [
            {'name': 'volume_sma_20', 'operation': 'rolling_mean', 'column': 'volume', 'params': {'window': 20}},
            {'name': 'volume_sma_50', 'operation': 'rolling_mean', 'column': 'volume', 'params': {'window': 50}},
            {'name': 'volume_std_20', 'operation': 'rolling_std', 'column': 'volume', 'params': {'window': 20}},
            {'name': 'volume_ratio_20', 'operation': 'rolling_mean', 'column': 'volume', 'params': {'window': 20}},
            {'name': 'volume_roc_20', 'operation': 'rolling_mean', 'column': 'volume', 'params': {'window': 20}}
        ]
        
        # Process in batch
        batch_results = self.batch_process_features(data, batch_configs)
        
        if batch_results:
            # Combine results (use volume SMA 20 as primary feature)
            primary_feature = batch_results.get('volume_sma_20', pd.Series(dtype=float, index=data.index))
            result = primary_feature.rename('refactored_batch_volume')
            
            # Apply normalization if enabled
            if self.unified_config.auto_normalize:
                result = self.normalize_feature(result, feature_type='volume')
            
            return result
        else:
            # Fallback to simple volume SMA
            volume = data['volume']
            volume_sma = self.rolling_mean(volume, window=20)
            
            if self.unified_config.auto_normalize:
                volume_sma = self.normalize_feature(volume_sma, feature_type='volume')
            
            return volume_sma.rename('refactored_batch_volume_fallback')