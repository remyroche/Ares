"""
Volume Feature Generator

This module provides feature generators for basic volume-based indicators,
including volume moving averages, ratios, rate of change, and other volume metrics.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import (
    FeatureGenerator, 
    FeatureConfig, 
    FeatureCategory,
    VectorizedFeatureGenerator
)
from ..base_calculations import (
    BaseCalculationType,
    create_base_calculator
)

class VolumeFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for basic volume-based features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)
    
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
        volume = data['volume'].values
        # Placeholder - actual implementation would generate multiple volume features
        return pd.Series(volume, index=data.index, name='volume_placeholder')

# Volume Simple Moving Average
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
        super().__init__(config, enable_matrix_ops=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume SMA."""
        volume = data['volume']
        return volume.rolling(window=self.period).mean()

# Volume Exponential Moving Average
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
        super().__init__(config, enable_matrix_ops=True)
        self.period = period
        self.alpha = alpha
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume EMA."""
        volume = data['volume']
        return volume.ewm(alpha=self.alpha, adjust=False).mean()

# Volume Ratio
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
        super().__init__(config, enable_matrix_ops=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Ratio."""
        volume = data['volume']
        avg_volume = volume.rolling(window=self.period).mean()
        return volume / avg_volume.replace(0, 1)  # Avoid division by zero

# Volume Rate of Change
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
        super().__init__(config, enable_matrix_ops=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume ROC."""
        volume = data['volume']
        return volume.pct_change(periods=self.period) * 100

# Volume Standard Deviation
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
        super().__init__(config, enable_matrix_ops=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Standard Deviation."""
        volume = data['volume']
        return volume.rolling(window=self.period).std()

# Volume Percentile Rank
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
        super().__init__(config, enable_matrix_ops=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Percentile Rank."""
        volume = data['volume']
        return volume.rolling(window=self.period).rank(pct=True) * 100

# Volume Trend Strength
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
        super().__init__(config, enable_matrix_ops=True)
        self.short_period = short_period
        self.long_period = long_period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Trend Strength."""
        volume = data['volume']
        short_ma = volume.rolling(window=self.short_period).mean()
        long_ma = volume.rolling(window=self.long_period).mean()
        return (short_ma - long_ma) / long_ma.replace(0, 1) * 100

# Volume Oscillator
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
        super().__init__(config, enable_matrix_ops=True)
        self.short_period = short_period
        self.long_period = long_period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Oscillator."""
        volume = data['volume']
        short_ma = volume.rolling(window=self.short_period).mean()
        long_ma = volume.rolling(window=self.long_period).mean()
        return short_ma - long_ma

# Volume Momentum
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
        super().__init__(config, enable_matrix_ops=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Momentum."""
        volume = data['volume']
        return volume - volume.shift(self.period)

# Volume Weighted Average Price (VWAP)
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
        super().__init__(config, enable_matrix_ops=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume VWAP."""
        close = data['close']
        volume = data['volume']
        return (close * volume).rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()

# Volume Price Trend (VPT)
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
        super().__init__(config, enable_matrix_ops=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Price Trend."""
        close = data['close']
        volume = data['volume']
        price_change = close.pct_change()
        vpt = (price_change * volume).cumsum()
        return vpt

# Volume Accumulation/Distribution
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
        super().__init__(config, enable_matrix_ops=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
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
    
    return generators

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
    'create_default_volume_generators'
]
