"""
VectorBT-optimized acceleration feature generators.

This module provides high-performance acceleration indicators using VectorBT optimizations.
Includes price acceleration, volume acceleration, and combined acceleration metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from src.feature_generation.core.feature_generator import FeatureConfig, FeatureCategory, FeatureGenerator
from src.feature_generation.core.vectorbt_feature_generator import VectorBTFeatureGenerator
from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator


class VectorBTPriceAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized price acceleration generator."""

    def __init__(self, period: int = 10, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)

    def _create_default_config(self, period: int) -> FeatureConfig:
        """Create default configuration."""
        return FeatureConfig(
            name=f"vectorbt_price_acceleration_{period}",
            category=FeatureCategory.MOMENTUM,
            description=f"VectorBT-optimized price acceleration over {period} periods",
            required_columns=["close"],
            default_lookback=period * 2,
            min_lookback=period,
            max_lookback=period * 3,
            parameters={"period": period}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate price acceleration feature using VectorBT optimizations."""
        close = data['close']
        period = self.config.parameters["period"]
        
        # Calculate velocity (first derivative)
        velocity = close.pct_change(period)
        
        # Calculate acceleration (second derivative)
        acceleration = velocity.diff(period)
        
        return acceleration


class VectorBTVolumeAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized volume acceleration generator."""

    def __init__(self, period: int = 10, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)

    def _create_default_config(self, period: int) -> FeatureConfig:
        """Create default configuration."""
        return FeatureConfig(
            name=f"vectorbt_volume_acceleration_{period}",
            category=FeatureCategory.VOLUME,
            description=f"VectorBT-optimized volume acceleration over {period} periods",
            required_columns=["volume"],
            default_lookback=period * 2,
            min_lookback=period,
            max_lookback=period * 3,
            parameters={"period": period}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volume acceleration feature using VectorBT optimizations."""
        volume = data['volume']
        period = self.config.parameters["period"]
        
        # Calculate volume velocity (first derivative)
        volume_velocity = volume.pct_change(period)
        
        # Calculate volume acceleration (second derivative)
        volume_acceleration = volume_velocity.diff(period)
        
        # Normalize by volume level to make it comparable across time
        normalized_acceleration = volume_acceleration / (volume.rolling(period).mean() + 1e-8)
        
        return normalized_acceleration


class VectorBTCombinedAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized combined price and volume acceleration generator."""

    def __init__(self, period: int = 10, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)

    def _create_default_config(self, period: int) -> FeatureConfig:
        """Create default configuration."""
        return FeatureConfig(
            name=f"vectorbt_combined_acceleration_{period}",
            category=FeatureCategory.MOMENTUM,
            description=f"VectorBT-optimized combined price-volume acceleration over {period} periods",
            required_columns=["close", "volume"],
            default_lookback=period * 2,
            min_lookback=period,
            max_lookback=period * 3,
            parameters={"period": period}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate combined acceleration feature using VectorBT optimizations."""
        close = data['close']
        volume = data['volume']
        period = self.config.parameters["period"]
        
        # Price acceleration
        price_velocity = close.pct_change(period)
        price_acceleration = price_velocity.diff(period)
        
        # Volume acceleration
        volume_velocity = volume.pct_change(period)
        volume_acceleration = volume_velocity.diff(period)
        
        # Combine accelerations (weighted average)
        # Normalize both to similar scale before combining
        price_accel_norm = price_acceleration / (price_acceleration.rolling(period * 2).std() + 1e-8)
        volume_accel_norm = volume_acceleration / (volume_acceleration.rolling(period * 2).std() + 1e-8)
        
        combined_acceleration = (price_accel_norm + volume_accel_norm) / 2
        
        return combined_acceleration


class VectorBTMultiTimeframeAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized multi-timeframe acceleration generator."""

    def __init__(self, short_period: int = 5, long_period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(short_period, long_period)
        super().__init__(config)

    def _create_default_config(self, short_period: int, long_period: int) -> FeatureConfig:
        """Create default configuration."""
        return FeatureConfig(
            name=f"vectorbt_mtf_acceleration_{short_period}_{long_period}",
            category=FeatureCategory.MOMENTUM,
            description=f"VectorBT-optimized multi-timeframe acceleration ({short_period}/{long_period})",
            required_columns=["close"],
            default_lookback=long_period * 2,
            min_lookback=long_period,
            max_lookback=long_period * 3,
            parameters={"short_period": short_period, "long_period": long_period}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate multi-timeframe acceleration feature using VectorBT optimizations."""
        close = data['close']
        short_period = self.config.parameters["short_period"]
        long_period = self.config.parameters["long_period"]
        
        # Short-term acceleration
        short_velocity = close.pct_change(short_period)
        short_acceleration = short_velocity.diff(short_period)
        
        # Long-term acceleration
        long_velocity = close.pct_change(long_period)
        long_acceleration = long_velocity.diff(long_period)
        
        # Acceleration divergence (short-term vs long-term)
        acceleration_divergence = short_acceleration - long_acceleration
        
        return acceleration_divergence


def create_default_acceleration_generators() -> List[VectorBTFeatureGenerator]:
    """Create default set of acceleration feature generators.
    
    Returns:
        List of default acceleration generators with various configurations.
    """
    generators = []
    
    # Price acceleration generators
    for period in [5, 10, 20]:
        generators.append(VectorBTPriceAccelerationGenerator(period=period))
    
    # Volume acceleration generators
    for period in [5, 10, 20]:
        generators.append(VectorBTVolumeAccelerationGenerator(period=period))
    
    # Combined acceleration generators
    for period in [10, 20]:
        generators.append(VectorBTCombinedAccelerationGenerator(period=period))
    
    # Multi-timeframe acceleration generators
    generators.append(VectorBTMultiTimeframeAccelerationGenerator(short_period=5, long_period=20))
    generators.append(VectorBTMultiTimeframeAccelerationGenerator(short_period=10, long_period=30))
    
    return generators


def create_custom_acceleration_generators(configs: List[Dict[str, Any]]) -> List[VectorBTFeatureGenerator]:
    """Create custom acceleration generators from configuration list.
    
    Args:
        configs: List of configuration dictionaries containing generator parameters.
        
    Returns:
        List of custom acceleration generators.
    """
    generators = []
    
    for config in configs:
        generator_type = config.get("type", "price")
        period = config.get("period", 10)
        
        if generator_type == "price":
            generators.append(VectorBTPriceAccelerationGenerator(period=period))
        elif generator_type == "volume":
            generators.append(VectorBTVolumeAccelerationGenerator(period=period))
        elif generator_type == "combined":
            generators.append(VectorBTCombinedAccelerationGenerator(period=period))
        elif generator_type == "mtf":
            short_period = config.get("short_period", 5)
            long_period = config.get("long_period", 20)
            generators.append(VectorBTMultiTimeframeAccelerationGenerator(
                short_period=short_period, 
                long_period=long_period
            ))
    
    return generators
