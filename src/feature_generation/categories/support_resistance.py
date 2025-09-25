"""
Support/Resistance Feature Generator

This module provides feature generators for support/resistance-based indicators.
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

class SupportResistanceFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for support/resistance-based features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="support_resistance_features",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description="Comprehensive support/resistance features including pivot points, Fibonacci, and volume profile",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=1,
            max_lookback=100,
            parameters={
                "pivot_windows": [5, 10, 20],
                "fibonacci_levels": [0.236, 0.382, 0.5, 0.618, 0.786],
                "volume_profile_windows": [5, 10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'SupportResistanceFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close_prices = data['close'].values
        sr = np.zeros_like(close_prices)
        return pd.Series(sr, index=data.index, name='sr_placeholder')

# Support Level Generator
class SupportLevelGenerator(FeatureGenerator):
    """Generator for support level features."""
    
    def __init__(self, level: int = 1, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'low' not in required_columns:
            required_columns.append('low')
        
        config = FeatureConfig(
            name=f"support_level_{level}_{window}_{base_calculation.value}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Support level {level} over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level': level, 'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.level = level
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate support level."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            low = data['low']
            support_level = low.rolling(window=self.window).min()
        else:
            base_values = self.base_calculator.calculate(data)
            support_level = base_values.rolling(window=self.window).min()
        return support_level

# Resistance Level Generator
class ResistanceLevelGenerator(FeatureGenerator):
    """Generator for resistance level features."""
    
    def __init__(self, level: int = 1, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'high' not in required_columns:
            required_columns.append('high')
        
        config = FeatureConfig(
            name=f"resistance_level_{level}_{window}_{base_calculation.value}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Resistance level {level} over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level': level, 'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.level = level
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate resistance level."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            resistance_level = high.rolling(window=self.window).max()
        else:
            base_values = self.base_calculator.calculate(data)
            resistance_level = base_values.rolling(window=self.window).max()
        return resistance_level

# Pivot Point Generator
class PivotPointGenerator(FeatureGenerator):
    """Generator for pivot point features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'high' not in required_columns:
            required_columns.append('high')
        if 'low' not in required_columns:
            required_columns.append('low')
        if 'close' not in required_columns:
            required_columns.append('close')
        
        config = FeatureConfig(
            name=f"pivot_point_{window}_{base_calculation.value}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Pivot point over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate pivot point."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            close = data['close']
            pivot_point = (high + low + close) / 3
        else:
            base_values = self.base_calculator.calculate(data)
            pivot_point = base_values.rolling(window=self.window).mean()
        return pivot_point

# Fibonacci Level Generator
class FibonacciLevelGenerator(FeatureGenerator):
    """Generator for Fibonacci level features."""
    
    def __init__(self, level: float = 0.618, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'high' not in required_columns:
            required_columns.append('high')
        if 'low' not in required_columns:
            required_columns.append('low')
        
        config = FeatureConfig(
            name=f"fibonacci_{level}_{window}_{base_calculation.value}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Fibonacci level {level} over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level': level, 'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.level = level
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Fibonacci level."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            range_size = high.rolling(window=self.window).max() - low.rolling(window=self.window).min()
            fibonacci_level = low.rolling(window=self.window).min() + (range_size * self.level)
        else:
            base_values = self.base_calculator.calculate(data)
            fibonacci_level = base_values.rolling(window=self.window).quantile(self.level)
        return fibonacci_level

# Price Level Crossing Generator
class PriceLevelCrossingGenerator(FeatureGenerator):
    """Generator for price level crossing features."""

    def __init__(self, level_pct: float = 0.2, window: int = 100):
        """Initialize crossing generator.

        Args:
            level_pct: Price level percentage (e.g., 0.2 for 0.2%)
            window: Lookback window for analysis
        """
        config = FeatureConfig(
            name=f"price_level_crossings_{level_pct}_{window}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Price level crossings at {level_pct}% intervals over {window} periods",
            required_columns=["close", "high", "low"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level_pct': level_pct, 'window': window}
        )
        super().__init__(config)
        self.level_pct = level_pct
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate price level crossing counts."""
        close = data['close']
        high = data['high']
        low = data['low']

        # Calculate price levels at specified percentage intervals
        current_price = close.iloc[-1]
        price_range = current_price * self.level_pct / 100

        # Define price levels
        levels_up = [current_price + (i + 1) * price_range for i in range(10)]
        levels_down = [current_price - (i + 1) * price_range for i in range(10)]

        # Count crossings for each level
        crossings = np.zeros(len(close))

        for level in levels_up + levels_down:
            # Count upward crossings
            up_crossings = ((close.shift(1) <= level) & (close > level)).astype(int)
            # Count downward crossings
            down_crossings = ((close.shift(1) >= level) & (close < level)).astype(int)
            crossings += up_crossings + down_crossings

        return pd.Series(crossings, index=data.index, name=f'crossings_{self.level_pct}')


# Price Level Bounce Generator
class PriceLevelBounceGenerator(FeatureGenerator):
    """Generator for price level bounce features."""

    def __init__(self, level_pct: float = 0.2, window: int = 100):
        """Initialize bounce generator.

        Args:
            level_pct: Price level percentage
            window: Lookback window for analysis
        """
        config = FeatureConfig(
            name=f"price_level_bounces_{level_pct}_{window}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Price level bounces at {level_pct}% intervals over {window} periods",
            required_columns=["close", "high", "low"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level_pct': level_pct, 'window': window}
        )
        super().__init__(config)
        self.level_pct = level_pct
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate price level bounce counts."""
        close = data['close']
        high = data['high']
        low = data['low']

        current_price = close.iloc[-1]
        price_range = current_price * self.level_pct / 100

        # Define price levels
        levels_up = [current_price + (i + 1) * price_range for i in range(10)]
        levels_down = [current_price - (i + 1) * price_range for i in range(10)]

        bounces = np.zeros(len(close))

        for level in levels_up + levels_down:
            # Detect bounces: price touches level then reverses
            touches_level = ((low <= level) & (high >= level))
            reversal_up = ((close.shift(1) <= level) & (close > level))
            reversal_down = ((close.shift(1) >= level) & (close < level))

            # Count bounces (touches that result in reversal)
            bounce_up = (touches_level & reversal_up.shift(1)).astype(int)
            bounce_down = (touches_level & reversal_down.shift(1)).astype(int)
            bounces += bounce_up + bounce_down

        return pd.Series(bounces, index=data.index, name=f'bounces_{self.level_pct}')


# Volume at Price Level Generator
class VolumeAtPriceLevelGenerator(FeatureGenerator):
    """Generator for volume traded at price level features."""

    def __init__(self, level_pct: float = 0.2, window: int = 100):
        """Initialize volume at price level generator.

        Args:
            level_pct: Price level percentage
            window: Lookback window for analysis
        """
        config = FeatureConfig(
            name=f"volume_at_price_levels_{level_pct}_{window}",
            category=FeatureCategory.VOLUME,
            description=f"Volume traded at {level_pct}% price levels over {window} periods",
            required_columns=["close", "high", "low", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level_pct': level_pct, 'window': window}
        )
        super().__init__(config)
        self.level_pct = level_pct
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volume at price levels."""
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']

        current_price = close.iloc[-1]
        price_range = current_price * self.level_pct / 100

        # Define price level bins
        levels_up = [current_price + (i + 1) * price_range for i in range(10)]
        levels_down = [current_price - (i + 1) * price_range for i in range(10)]
        price_levels = levels_down[::-1] + [current_price] + levels_up

        # Calculate volume at each price level
        volume_at_levels = np.zeros(len(close))

        for i, level in enumerate(price_levels):
            # Volume traded when price is at this level
            at_level = ((low <= level) & (high >= level))
            volume_at_levels += at_level.astype(int) * volume

        return pd.Series(volume_at_levels, index=data.index, name=f'volume_at_levels_{self.level_pct}')


# Price Level Strength Generator
class PriceLevelStrengthGenerator(FeatureGenerator):
    """Generator for price level strength based on historical significance."""

    def __init__(self, level_pct: float = 0.2, window: int = 200):
        """Initialize strength generator.

        Args:
            level_pct: Price level percentage
            window: Lookback window for analysis
        """
        config = FeatureConfig(
            name=f"price_level_strength_{level_pct}_{window}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Price level strength at {level_pct}% intervals over {window} periods",
            required_columns=["close", "high", "low", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level_pct': level_pct, 'window': window}
        )
        super().__init__(config)
        self.level_pct = level_pct
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate price level strength scores."""
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']

        current_price = close.iloc[-1]
        price_range = current_price * self.level_pct / 100

        # Define price levels
        levels_up = [current_price + (i + 1) * price_range for i in range(5)]
        levels_down = [current_price - (i + 1) * price_range for i in range(5)]
        price_levels = levels_down[::-1] + [current_price] + levels_up

        # Calculate strength for each level
        strength_scores = np.zeros(len(close))

        for level in price_levels:
            # Touch frequency (how often price touches this level)
            touches = ((low <= level) & (high >= level)).astype(int)

            # Volume at level
            volume_at_level = touches * volume

            # Duration at level (consecutive periods)
            consecutive_touches = touches * (touches.groupby((touches != touches.shift()).cumsum()).cumcount() + 1)

            # Bounce strength (how strongly price reverses)
            price_change_after = close.pct_change().shift(-1)
            bounce_strength = touches * abs(price_change_after)

            # Combined strength score
            strength = (touches * 0.3 + volume_at_level * 0.3 +
                       consecutive_touches * 0.2 + bounce_strength * 0.2)

            strength_scores += strength.fillna(0)

        return pd.Series(strength_scores, index=data.index, name=f'strength_{self.level_pct}')


# Price Level Recency Generator
class PriceLevelRecencyGenerator(FeatureGenerator):
    """Generator for price level recency features."""

    def __init__(self, level_pct: float = 0.2, window: int = 200):
        """Initialize recency generator.

        Args:
            level_pct: Price level percentage
            window: Lookback window for analysis
        """
        config = FeatureConfig(
            name=f"price_level_recency_{level_pct}_{window}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Price level recency at {level_pct}% intervals over {window} periods",
            required_columns=["close", "high", "low"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level_pct': level_pct, 'window': window}
        )
        super().__init__(config)
        self.level_pct = level_pct
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate price level recency scores."""
        close = data['close']
        high = data['high']
        low = data['low']

        current_price = close.iloc[-1]
        price_range = current_price * self.level_pct / 100

        # Define price levels
        levels_up = [current_price + (i + 1) * price_range for i in range(5)]
        levels_down = [current_price - (i + 1) * price_range for i in range(5)]
        price_levels = levels_down[::-1] + [current_price] + levels_up

        # Calculate recency for each level
        recency_scores = np.zeros(len(close))

        for level in price_levels:
            # Find last touch of this level
            touches = ((low <= level) & (high >= level)).astype(int)

            # Calculate periods since last touch
            last_touch_idx = None
            for i in range(len(close)):
                if touches.iloc[i]:
                    last_touch_idx = i

            # Recency score (higher for more recent touches)
            if last_touch_idx is not None:
                periods_since_touch = len(close) - 1 - last_touch_idx
                recency_score = np.exp(-periods_since_touch / (self.window / 10))  # Decay over time
            else:
                recency_score = 0.0

            recency_scores += recency_score

        return pd.Series(recency_scores, index=data.index, name=f'recency_{self.level_pct}')


# Price Level Clustering Generator
class PriceLevelClusteringGenerator(FeatureGenerator):
    """Generator for price level clustering analysis."""

    def __init__(self, level_pct: float = 0.2, window: int = 200):
        """Initialize clustering generator.

        Args:
            level_pct: Price level percentage
            window: Lookback window for analysis
        """
        config = FeatureConfig(
            name=f"price_level_clustering_{level_pct}_{window}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Price level clustering at {level_pct}% intervals over {window} periods",
            required_columns=["close", "high", "low", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level_pct': level_pct, 'window': window}
        )
        super().__init__(config)
        self.level_pct = level_pct
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate price level clustering scores."""
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']

        current_price = close.iloc[-1]
        price_range = current_price * self.level_pct / 100

        # Define price levels
        levels_up = [current_price + (i + 1) * price_range for i in range(10)]
        levels_down = [current_price - (i + 1) * price_range for i in range(10)]
        price_levels = levels_down[::-1] + [current_price] + levels_up

        # Calculate clustering for each level
        clustering_scores = np.zeros(len(close))

        for level in price_levels:
            # Find nearby price action
            price_distance = abs(close - level)
            nearby_activity = (price_distance <= price_range / 2).astype(int)

            # Volume concentration around level
            volume_concentration = nearby_activity * volume

            # Time concentration (how much time spent near level)
            time_near_level = nearby_activity.rolling(window=10).sum()

            # Clustering score combining these factors
            clustering = (nearby_activity * 0.4 + volume_concentration * 0.4 + time_near_level * 0.2)
            clustering_scores += clustering.fillna(0)

        return pd.Series(clustering_scores, index=data.index, name=f'clustering_{self.level_pct}')


# Price Level Momentum Generator
class PriceLevelMomentumGenerator(FeatureGenerator):
    """Generator for price level momentum analysis."""

    def __init__(self, level_pct: float = 0.2, window: int = 50):
        """Initialize momentum generator.

        Args:
            level_pct: Price level percentage
            window: Lookback window for analysis
        """
        config = FeatureConfig(
            name=f"price_level_momentum_{level_pct}_{window}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Price level momentum at {level_pct}% intervals over {window} periods",
            required_columns=["close", "high", "low"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level_pct': level_pct, 'window': window}
        )
        super().__init__(config)
        self.level_pct = level_pct
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate price level momentum scores."""
        close = data['close']

        current_price = close.iloc[-1]
        price_range = current_price * self.level_pct / 100

        # Define price levels
        levels_up = [current_price + (i + 1) * price_range for i in range(5)]
        levels_down = [current_price - (i + 1) * price_range for i in range(5)]
        price_levels = levels_down[::-1] + [current_price] + levels_up

        # Calculate momentum approaching each level
        momentum_scores = np.zeros(len(close))

        for level in price_levels:
            # Price momentum (rate of change approaching level)
            price_momentum = close.pct_change(periods=5)

            # Distance to level
            distance_to_level = (close - level) / level

            # Momentum towards/away from level
            momentum_towards = price_momentum * (1 if level > close.iloc[-1] else -1)

            momentum_scores += abs(momentum_towards)

        return pd.Series(momentum_scores, index=data.index, name=f'momentum_{self.level_pct}')


def create_default_support_resistance_generators() -> List[FeatureGenerator]:
    """Create default support/resistance feature generators."""
    windows = [5, 10, 20]
    fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    price_level_pcts = [0.1, 0.2, 0.5, 1.0]

    generators = []

    # Create generators for each window
    for window in windows:
        generators.extend([
            SupportLevelGenerator(1, window),
            SupportLevelGenerator(2, window),
            SupportLevelGenerator(3, window),
            SupportLevelGenerator(4, window),
            SupportLevelGenerator(5, window),
            ResistanceLevelGenerator(1, window),
            ResistanceLevelGenerator(2, window),
            ResistanceLevelGenerator(3, window),
            ResistanceLevelGenerator(4, window),
            ResistanceLevelGenerator(5, window),
            PivotPointGenerator(window),
        ])

    # Create Fibonacci level generators
    for level in fibonacci_levels:
        for window in windows:
            generators.append(FibonacciLevelGenerator(level, window))

    # Create price level analysis generators
    for level_pct in price_level_pcts:
        for window in [50, 100, 200]:
            generators.extend([
                PriceLevelCrossingGenerator(level_pct, window),
                PriceLevelBounceGenerator(level_pct, window),
                VolumeAtPriceLevelGenerator(level_pct, window),
                PriceLevelStrengthGenerator(level_pct, window),
                PriceLevelRecencyGenerator(level_pct, window),
                PriceLevelClusteringGenerator(level_pct, window),
                PriceLevelMomentumGenerator(level_pct, window),
            ])

    return generators