"""
Volume Feature Generator

This module provides feature generators for basic volume-based indicators,
including volume moving averages, ratios, rate of change, and other volume metrics.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
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

        volume = data['volume'].values
        # Placeholder - actual implementation would generate multiple volume features
        return pd.Series(volume, index=data.index, name='volume_placeholder')

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

        """Generate Volume SMA."""
        volume = data['volume']
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

        """Generate Volume Ratio."""
        volume = data['volume']
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

        """Generate Volume Standard Deviation."""
        volume = data['volume']
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

