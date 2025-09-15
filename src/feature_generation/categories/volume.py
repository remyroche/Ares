"""
Volume Feature Generator

This module provides feature generators for volume-based indicators,
including volume ratios, OBV, VWAP, and other volume-related features.
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
    """
    Feature generator for volume-based features.
    
    This generator creates various volume indicators including:
    - Volume ratios and moving averages
    - On-Balance Volume (OBV)
    - Volume Weighted Average Price (VWAP)
    - Volume Rate of Change
    - Volume-Price Trend (VPT)
    - Accumulation/Distribution Line
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize the volume feature generator.
        
        Args:
            config: Feature configuration (uses default if None)
        """
        if config is None:
            config = self._create_default_config()
        
        super().__init__(config, enable_matrix_ops=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        """Create default configuration for volume features."""
        return FeatureConfig(
            name="volume_features",
            category=FeatureCategory.VOLUME,
            description="Comprehensive volume-based features including OBV, VWAP, and volume ratios",
            required_columns=["volume"],
            optional_columns=["close", "high", "low", "open"],
            default_lookback=20,
            min_lookback=1,
            max_lookback=50,
            parameters={
                "volume_ma_periods": [5, 10, 20],
                "vwap_periods": [20, 50],
                "volume_roc_periods": [10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'VolumeFeatureGenerator':
        """Create a default volume feature generator."""
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Generate volume features.
        
        Args:
            data: Input data with OHLCV columns
            **kwargs: Additional parameters
            
        Returns:
            Combined volume features (placeholder - actual implementation would return multiple features)
        """
        # This is a simplified implementation that returns a single feature
        # In practice, this would generate multiple volume features
        
        volume = data['volume'].values
        
        # Generate volume moving average as the main feature
        volume_ma = self._calculate_volume_ma(volume, period=20)
        
        return pd.Series(volume_ma, index=data.index, name='volume_ma_20')
    
    def _calculate_volume_ma(self, volume: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate volume moving average."""
        if len(volume) < period:
            return np.full(len(volume), np.nan)
        
        # Use vectorized rolling mean
        if self.enable_matrix_ops and self.matrix_ops:
            try:
                volume_series = pd.Series(volume)
                volume_ma = volume_series.rolling(window=period).mean()
                return volume_ma.values
            except Exception:
                pass
        
        # Fallback to manual calculation
        volume_ma = np.full(len(volume), np.nan)
        for i in range(period - 1, len(volume)):
            volume_ma[i] = np.mean(volume[i - period + 1:i + 1])
        
        return volume_ma
    
    def _generate_feature_with_lookback(self, data: pd.DataFrame, lookback: int, **kwargs) -> pd.Series:
        """
        Generate volume features with specific lookback period.
        
        Args:
            data: Input data
            lookback: Lookback period
            **kwargs: Additional parameters
            
        Returns:
            Volume features with specified lookback
        """
        volume = data['volume'].values
        volume_ma = self._calculate_volume_ma(volume, period=lookback)
        
        return pd.Series(volume_ma, index=data.index, name=f'volume_ma_{lookback}')

class VolumeMAGenerator(FeatureGenerator):
    """Generator for Volume Moving Average with different base calculations."""
    
    def __init__(self, 
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.VOLUME_RETURNS,
                 **base_kwargs):
        """
        Initialize Volume MA generator.
        
        Args:
            period: Volume MA period
            base_calculation: Base calculation type (volume_returns, volume_weighted, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"volume_ma_{period}_{base_calculation.value}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Moving Average over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=1,
            max_lookback=50,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Moving Average based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        volume_ma = base_values.rolling(window=self.period).mean()
        
        return volume_ma

class VolumeRatioGenerator(FeatureGenerator):
    """Generator for Volume Ratio with different base calculations."""
    
    def __init__(self, 
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.VOLUME_RETURNS,
                 **base_kwargs):
        """
        Initialize Volume Ratio generator.
        
        Args:
            period: Volume ratio period
            base_calculation: Base calculation type (volume_returns, volume_weighted, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"volume_ratio_{period}_{base_calculation.value}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Ratio over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=1,
            max_lookback=50,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Ratio based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        avg_base_values = base_values.rolling(window=self.period).mean()
        volume_ratio = base_values / avg_base_values
        
        return volume_ratio

class OBVGenerator(FeatureGenerator):
    """Generator for On-Balance Volume (OBV)."""
    
    def __init__(self):
        """Initialize OBV generator."""
        config = FeatureConfig(
            name="obv",
            category=FeatureCategory.VOLUME,
            description="On-Balance Volume",
            required_columns=["close", "volume"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate On-Balance Volume."""
        close = data['close']
        volume = data['volume']
        
        # Calculate price changes
        price_change = close.diff()
        
        # Calculate OBV
        obv = np.where(price_change > 0, volume,
                      np.where(price_change < 0, -volume, 0))
        
        # Cumulative sum
        obv = pd.Series(obv, index=data.index).cumsum()
        
        return obv

class VWAPGenerator(FeatureGenerator):
    """Generator for Volume Weighted Average Price (VWAP) with different base calculations."""
    
    def __init__(self, 
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize VWAP generator.
        
        Args:
            period: VWAP period
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"vwap_{period}_{base_calculation.value}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Weighted Average Price over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=1,
            max_lookback=50,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate VWAP based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            # Traditional VWAP calculation on price levels
            high = data['high']
            low = data['low']
            close = data['close']
            volume = data['volume']
            
            # Calculate typical price
            typical_price = (high + low + close) / 3
            
            # Calculate VWAP
            vwap = (typical_price * volume).rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()
            
            return vwap
        else:
            # For other base calculations, calculate VWAP on the base values
            base_values = self.base_calculator.calculate(data)
            volume = data['volume']
            
            # Calculate VWAP on base values
            vwap = (base_values * volume).rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()
            
            return vwap

class VolumeROCGenerator(FeatureGenerator):
    """Generator for Volume Rate of Change."""
    
    def __init__(self, period: int = 10):
        """Initialize Volume ROC generator."""
        config = FeatureConfig(
            name=f"volume_roc_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Rate of Change over {period} periods",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period
        )
        super().__init__(config)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Rate of Change."""
        volume = data['volume']
        
        # Calculate Volume ROC
        volume_roc = ((volume - volume.shift(self.period)) / volume.shift(self.period)) * 100
        
        return volume_roc

class VPTGenerator(FeatureGenerator):
    """Generator for Volume-Price Trend (VPT)."""
    
    def __init__(self):
        """Initialize VPT generator."""
        config = FeatureConfig(
            name="vpt",
            category=FeatureCategory.VOLUME,
            description="Volume-Price Trend",
            required_columns=["close", "volume"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume-Price Trend."""
        close = data['close']
        volume = data['volume']
        
        # Calculate price change percentage
        price_change_pct = close.pct_change()
        
        # Calculate VPT
        vpt = (price_change_pct * volume).cumsum()
        
        return vpt

class ADLGenerator(FeatureGenerator):
    """Generator for Accumulation/Distribution Line."""
    
    def __init__(self):
        """Initialize ADL generator."""
        config = FeatureConfig(
            name="adl",
            category=FeatureCategory.VOLUME,
            description="Accumulation/Distribution Line",
            required_columns=["high", "low", "close", "volume"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Accumulation/Distribution Line."""
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Calculate Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)  # Handle division by zero
        
        # Calculate Money Flow Volume
        mfv = mfm * volume
        
        # Calculate ADL (cumulative sum)
        adl = mfv.cumsum()
        
        return adl

class VolumeVolatilityGenerator(FeatureGenerator):
    """Generator for Volume Volatility."""
    
    def __init__(self, period: int = 20):
        """Initialize Volume Volatility generator."""
        config = FeatureConfig(
            name=f"volume_volatility_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Volatility over {period} periods",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=2,
            max_lookback=50
        )
        super().__init__(config)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Volatility."""
        volume = data['volume']
        
        # Calculate volume volatility (rolling standard deviation)
        volume_volatility = volume.rolling(window=self.period).std()
        
        return volume_volatility

class VolumeSkewnessGenerator(FeatureGenerator):
    """Generator for Volume Skewness."""
    
    def __init__(self, period: int = 20):
        """Initialize Volume Skewness generator."""
        config = FeatureConfig(
            name=f"volume_skewness_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Skewness over {period} periods",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=3,
            max_lookback=50
        )
        super().__init__(config)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Skewness."""
        volume = data['volume']
        
        # Calculate volume skewness
        volume_skewness = volume.rolling(window=self.period).skew()
        
        return volume_skewness

# Factory functions for creating volume generators
def create_volume_generators(periods: Dict[str, List[int]] = None) -> List[FeatureGenerator]:
    """
    Create a set of volume feature generators.
    
    Args:
        periods: Dictionary mapping indicator types to lists of periods
        
    Returns:
        List of volume feature generators
    """
    if periods is None:
        periods = {
            'volume_ma': [5, 10, 20],
            'volume_ratio': [10, 20],
            'vwap': [20, 50],
            'volume_roc': [10, 20],
            'volume_volatility': [20],
            'volume_skewness': [20]
        }
    
    generators = []
    
    # Volume MA generators
    for period in periods.get('volume_ma', [20]):
        generators.append(VolumeMAGenerator(period))
    
    # Volume Ratio generators
    for period in periods.get('volume_ratio', [20]):
        generators.append(VolumeRatioGenerator(period))
    
    # VWAP generators
    for period in periods.get('vwap', [20]):
        generators.append(VWAPGenerator(period))
    
    # Volume ROC generators
    for period in periods.get('volume_roc', [10, 20]):
        generators.append(VolumeROCGenerator(period))
    
    # Volume Volatility generators
    for period in periods.get('volume_volatility', [20]):
        generators.append(VolumeVolatilityGenerator(period))
    
    # Volume Skewness generators
    for period in periods.get('volume_skewness', [20]):
        generators.append(VolumeSkewnessGenerator(period))
    
    # Add non-period-based generators
    generators.extend([
        OBVGenerator(),
        VPTGenerator(),
        ADLGenerator()
    ])
    
    return generators

def create_default_volume_generators() -> List[FeatureGenerator]:
    """Create default volume feature generators."""
    return create_volume_generators()