"""
Refactored Trend Feature Generator

This module provides refactored trend feature generators that use centralized
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

class RefactoredTrendFeatureGenerator(UnifiedFeatureGenerator):
    """Refactored trend feature generator using centralized utilities."""
    
    def __init__(self, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config)
    
    @classmethod
    def _create_default_config(cls) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name="refactored_trend_features",
            category=FeatureCategory.TREND,
            description="Refactored trend features using centralized utilities",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "sma_periods": [20, 50, 100],
                "ema_periods": [12, 26],
                "adx_period": 14,
                "trend_windows": [10, 20, 30]
            },
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='zscore',
            normalization_feature_type='trend',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trend features using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        # Calculate trend using centralized rolling operations
        close_prices = data['close']
        trend = self.rolling_mean(close_prices, window=20)
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            trend = self.normalize_feature(trend, feature_type='trend')
        
        return trend.rename('refactored_trend_20')

class RefactoredSMAGenerator(UnifiedFeatureGenerator):
    """Refactored Simple Moving Average generator using centralized utilities."""
    
    def __init__(self, period: int = 20, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_sma_{period}",
            category=FeatureCategory.TREND,
            description=f"Refactored SMA with period {period} using centralized utilities",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='minmax',
            normalization_feature_type='trend',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate SMA using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        
        # Use centralized rolling operations
        sma = self.rolling_mean(close, window=self.period)
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            sma = self.normalize_feature(sma, feature_type='trend')
        
        return sma.rename(f'refactored_sma_{self.period}')

class RefactoredEMAGenerator(UnifiedFeatureGenerator):
    """Refactored Exponential Moving Average generator using centralized utilities."""
    
    def __init__(self, period: int = 20, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_ema_{period}",
            category=FeatureCategory.TREND,
            description=f"Refactored EMA with period {period} using centralized utilities",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='minmax',
            normalization_feature_type='trend',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate EMA using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        
        # Calculate EMA (exponential moving average)
        alpha = 2.0 / (self.period + 1)
        ema = close.ewm(alpha=alpha).mean()
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            ema = self.normalize_feature(ema, feature_type='trend')
        
        return ema.rename(f'refactored_ema_{self.period}')

class RefactoredADXGenerator(UnifiedFeatureGenerator):
    """Refactored ADX generator using centralized utilities."""
    
    def __init__(self, period: int = 14, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_adx_{period}",
            category=FeatureCategory.TREND,
            description=f"Refactored ADX with period {period} using centralized utilities",
            required_columns=["high", "low", "close"],
            default_lookback=period + 1,
            min_lookback=period + 1,
            max_lookback=100,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='minmax',
            normalization_feature_type='trend',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate ADX using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
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
        
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Use centralized rolling operations
        tr_smooth = self.rolling_mean(true_range, window=self.period)
        dm_plus_smooth = self.rolling_mean(dm_plus, window=self.period)
        dm_minus_smooth = self.rolling_mean(dm_minus, window=self.period)
        
        # Calculate DI+ and DI-
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        # Calculate DX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        
        # Calculate ADX
        adx = self.rolling_mean(dx, window=self.period)
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            adx = self.normalize_feature(adx, feature_type='trend')
        
        return adx.rename(f'refactored_adx_{self.period}')

class RefactoredDirectionalSignalGenerator(UnifiedFeatureGenerator):
    """Refactored Directional Signal generator using centralized utilities."""
    
    def __init__(self, fast_period: int = 8, slow_period: int = 20, 
                 config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(fast_period, slow_period)
        super().__init__(config)
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    @classmethod
    def _create_default_config(cls, fast_period: int, slow_period: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_directional_signal_{fast_period}_{slow_period}",
            category=FeatureCategory.TREND,
            description=f"Refactored Directional Signal using centralized utilities",
            required_columns=["close"],
            default_lookback=slow_period,
            min_lookback=slow_period,
            max_lookback=100,
            parameters={"fast_period": fast_period, "slow_period": slow_period},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='zscore',
            normalization_feature_type='trend',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Directional Signal using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        
        # Calculate EMAs
        ema_fast = close.ewm(span=self.fast_period).mean()
        ema_slow = close.ewm(span=self.slow_period).mean()
        
        # Calculate directional signal
        directional_signal = ema_fast - ema_slow
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            directional_signal = self.normalize_feature(directional_signal, feature_type='trend')
        
        return directional_signal.rename(f'refactored_directional_signal_{self.fast_period}_{self.slow_period}')

class RefactoredTrendScoreGenerator(UnifiedFeatureGenerator):
    """Refactored Trend Score generator using centralized utilities."""
    
    def __init__(self, adx_period: int = 14, signal_fast: int = 8, signal_slow: int = 20,
                 config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(adx_period, signal_fast, signal_slow)
        super().__init__(config)
        self.adx_period = adx_period
        self.signal_fast = signal_fast
        self.signal_slow = signal_slow
    
    @classmethod
    def _create_default_config(cls, adx_period: int, signal_fast: int, signal_slow: int) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_trend_score_{adx_period}_{signal_fast}_{signal_slow}",
            category=FeatureCategory.TREND,
            description=f"Refactored Trend Score using centralized utilities",
            required_columns=["high", "low", "close"],
            default_lookback=max(adx_period, signal_slow),
            min_lookback=max(adx_period, signal_slow),
            max_lookback=100,
            parameters={"adx_period": adx_period, "signal_fast": signal_fast, "signal_slow": signal_slow},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='minmax',
            normalization_feature_type='trend',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Trend Score using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate ADX (simplified version)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        tr_smooth = self.rolling_mean(true_range, window=self.adx_period)
        dm_plus_smooth = self.rolling_mean(dm_plus, window=self.adx_period)
        dm_minus_smooth = self.rolling_mean(dm_minus, window=self.adx_period)
        
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = self.rolling_mean(dx, window=self.adx_period)
        
        # Calculate directional signal
        ema_fast = close.ewm(span=self.signal_fast).mean()
        ema_slow = close.ewm(span=self.signal_slow).mean()
        directional_signal = ema_fast - ema_slow
        
        # Calculate trend score (normalized directional signal * ADX)
        trend_score = (directional_signal / close) * adx
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            trend_score = self.normalize_feature(trend_score, feature_type='trend')
        
        return trend_score.rename(f'refactored_trend_score_{self.adx_period}_{self.signal_fast}_{self.signal_slow}')

class RefactoredKeltnerChannelsGenerator(UnifiedFeatureGenerator):
    """Refactored Keltner Channels generator using centralized utilities."""
    
    def __init__(self, period: int = 20, multiplier: float = 2.0, 
                 config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period, multiplier)
        super().__init__(config)
        self.period = period
        self.multiplier = multiplier
    
    @classmethod
    def _create_default_config(cls, period: int, multiplier: float) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name=f"refactored_keltner_channels_{period}_{multiplier}",
            category=FeatureCategory.TREND,
            description=f"Refactored Keltner Channels using centralized utilities",
            required_columns=["high", "low", "close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=100,
            parameters={"period": period, "multiplier": multiplier},
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='minmax',
            normalization_feature_type='trend',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Keltner Channels using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = self.rolling_mean(true_range, window=self.period)
        
        # Calculate middle line (EMA)
        middle_line = close.ewm(span=self.period).mean()
        
        # Calculate upper and lower bands
        upper_band = middle_line + (self.multiplier * atr)
        lower_band = middle_line - (self.multiplier * atr)
        
        # Calculate channel width
        channel_width = upper_band - lower_band
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            channel_width = self.normalize_feature(channel_width, feature_type='trend')
        
        return channel_width.rename(f'refactored_keltner_width_{self.period}_{self.multiplier}')

class RefactoredOptimizedTrendFeatureGenerator(UnifiedFeatureGenerator):
    """Refactored optimized trend feature generator using centralized utilities."""
    
    def __init__(self, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config)
    
    @classmethod
    def _create_default_config(cls) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name="refactored_optimized_trend_features",
            category=FeatureCategory.TREND,
            description="Refactored optimized trend features using centralized utilities",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "sma_periods": [20, 50, 100],
                "ema_periods": [12, 26],
                "adx_period": 14,
                "trend_windows": [10, 20, 30]
            },
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='zscore',
            normalization_feature_type='trend',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate optimized trend features using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        # Use common operations for comprehensive trend calculation
        price_levels = self.calculate_price_levels(data, ['hl2', 'hlc3'])
        
        # Calculate trend using HL2 (high-low average)
        if 'hl2' in price_levels:
            trend = self.rolling_mean(price_levels['hl2'], window=20)
        else:
            close = data['close']
            trend = self.rolling_mean(close, window=20)
        
        # Apply normalization if enabled
        if self.unified_config.auto_normalize:
            trend = self.normalize_feature(trend, feature_type='trend')
        
        return trend.rename('refactored_optimized_trend')

# Batch trend generator for multiple features
class RefactoredBatchTrendGenerator(UnifiedFeatureGenerator):
    """Refactored batch trend generator using centralized utilities."""
    
    def __init__(self, config: Optional[UnifiedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config)
    
    @classmethod
    def _create_default_config(cls) -> UnifiedFeatureConfig:
        return UnifiedFeatureConfig(
            name="refactored_batch_trend_features",
            category=FeatureCategory.TREND,
            description="Refactored batch trend features using centralized utilities",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "sma_periods": [20, 50, 100],
                "ema_periods": [12, 26],
                "adx_period": 14,
                "trend_windows": [10, 20, 30]
            },
            matrix_optimized=True,
            gpu_accelerated=False,
            auto_normalize=True,
            normalization_method='zscore',
            normalization_feature_type='trend',
            enable_batch_processing=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate batch trend features using centralized utilities."""
        # Optimize DataFrame for processing
        data = self.optimize_dataframe_processing(data)
        
        # Create batch configuration for multiple trend features
        batch_configs = [
            {'name': 'sma_20', 'operation': 'rolling_mean', 'column': 'close', 'params': {'window': 20}},
            {'name': 'sma_50', 'operation': 'rolling_mean', 'column': 'close', 'params': {'window': 50}},
            {'name': 'ema_12', 'operation': 'technical_indicator', 'indicator': 'ema', 'params': {'period': 12}},
            {'name': 'ema_26', 'operation': 'technical_indicator', 'indicator': 'ema', 'params': {'period': 26}},
            {'name': 'adx_14', 'operation': 'technical_indicator', 'indicator': 'adx', 'params': {'period': 14}}
        ]
        
        # Process in batch
        batch_results = self.batch_process_features(data, batch_configs)
        
        if batch_results:
            # Combine results (use SMA 20 as primary feature)
            primary_feature = batch_results.get('sma_20', pd.Series(dtype=float, index=data.index))
            result = primary_feature.rename('refactored_batch_trend')
            
            # Apply normalization if enabled
            if self.unified_config.auto_normalize:
                result = self.normalize_feature(result, feature_type='trend')
            
            return result
        else:
            # Fallback to simple SMA
            close = data['close']
            sma = self.rolling_mean(close, window=20)
            
            if self.unified_config.auto_normalize:
                sma = self.normalize_feature(sma, feature_type='trend')
            
            return sma.rename('refactored_batch_trend_fallback')