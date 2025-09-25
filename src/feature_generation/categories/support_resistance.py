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

# Historical Price Level Crossing Generator
class HistoricalPriceLevelCrossingGenerator(FeatureGenerator):
    """Generator for historical price level crossing counts - backward looking for ML training."""

    def __init__(self, level_pct: float = 0.2, window: int = 100):
        """Initialize historical crossing generator.

        Args:
            level_pct: Price level percentage (e.g., 0.2 for 0.2%)
            window: Lookback window for historical analysis
        """
        config = FeatureConfig(
            name=f"historical_crossings_{level_pct}_{window}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Historical crossing counts at {level_pct}% levels over past {window} periods",
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
        """Generate historical price level crossing counts (backward looking)."""
        close = data['close']
        high = data['high']
        low = data['low']

        # Calculate historical crossings up to each point
        crossings = pd.Series(index=data.index, dtype=float)

        for idx in range(self.window, len(data)):
            # Get historical data up to current point
            historical_close = close.iloc[idx-self.window:idx]
            historical_high = high.iloc[idx-self.window:idx]
            historical_low = low.iloc[idx-self.window:idx]

            # Calculate price levels based on the current price at this historical point
            current_price = close.iloc[idx]
            price_range = current_price * self.level_pct / 100

            # Define price levels
            levels_up = [current_price + (i + 1) * price_range for i in range(10)]
            levels_down = [current_price - (i + 1) * price_range for i in range(10)]

            # Count crossings in the historical window
            total_crossings = 0

            for level in levels_up + levels_down:
                # Count upward crossings in historical data
                up_crossings = ((historical_close.shift(1) <= level) & (historical_close > level)).sum()
                # Count downward crossings in historical data
                down_crossings = ((historical_close.shift(1) >= level) & (historical_close < level)).sum()
                total_crossings += up_crossings + down_crossings

            crossings.iloc[idx] = total_crossings

        return crossings.fillna(0).astype(int)


# Historical Price Level Bounce Generator
class HistoricalPriceLevelBounceGenerator(FeatureGenerator):
    """Generator for historical price level bounce counts - backward looking for ML training."""

    def __init__(self, level_pct: float = 0.2, window: int = 100):
        """Initialize historical bounce generator.

        Args:
            level_pct: Price level percentage
            window: Lookback window for historical analysis
        """
        config = FeatureConfig(
            name=f"historical_bounces_{level_pct}_{window}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Historical bounce counts at {level_pct}% levels over past {window} periods",
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
        """Generate historical price level bounce counts (backward looking)."""
        close = data['close']
        high = data['high']
        low = data['low']

        # Calculate historical bounces up to each point
        bounces = pd.Series(index=data.index, dtype=float)

        for idx in range(self.window, len(data)):
            # Get historical data up to current point
            historical_close = close.iloc[idx-self.window:idx]
            historical_high = high.iloc[idx-self.window:idx]
            historical_low = low.iloc[idx-self.window:idx]

            # Calculate price levels based on the current price at this historical point
            current_price = close.iloc[idx]
            price_range = current_price * self.level_pct / 100

            # Define price levels
            levels_up = [current_price + (i + 1) * price_range for i in range(10)]
            levels_down = [current_price - (i + 1) * price_range for i in range(10)]

            # Count bounces in the historical window
            total_bounces = 0

            for level in levels_up + levels_down:
                # Detect touches in historical data
                touches_level = ((historical_low <= level) & (historical_high >= level))

                # Find reversal points (touches followed by price movement away)
                reversal_up = []
                reversal_down = []

                for i in range(1, len(historical_close)):
                    if touches_level.iloc[i]:
                        # Check if this touch was followed by reversal
                        price_after = historical_close.iloc[i+1] if i+1 < len(historical_close) else historical_close.iloc[i]
                        price_before = historical_close.iloc[i-1]

                        if price_before <= level and price_after > level:
                            reversal_up.append(1)
                        elif price_before >= level and price_after < level:
                            reversal_down.append(1)

                total_bounces += len(reversal_up) + len(reversal_down)

            bounces.iloc[idx] = total_bounces

        return bounces.fillna(0).astype(int)


# Historical Volume at Price Level Generator
class HistoricalVolumeAtPriceLevelGenerator(FeatureGenerator):
    """Generator for historical volume traded at price levels - backward looking for ML training."""

    def __init__(self, level_pct: float = 0.2, window: int = 100):
        """Initialize historical volume at price level generator.

        Args:
            level_pct: Price level percentage
            window: Lookback window for historical analysis
        """
        config = FeatureConfig(
            name=f"historical_volume_at_levels_{level_pct}_{window}",
            category=FeatureCategory.VOLUME,
            description=f"Historical volume traded at {level_pct}% price levels over past {window} periods",
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
        """Generate historical volume at price levels (backward looking)."""
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']

        # Calculate historical volume at levels up to each point
        volume_at_levels = pd.Series(index=data.index, dtype=float)

        for idx in range(self.window, len(data)):
            # Get historical data up to current point
            historical_close = close.iloc[idx-self.window:idx]
            historical_high = high.iloc[idx-self.window:idx]
            historical_low = low.iloc[idx-self.window:idx]
            historical_volume = volume.iloc[idx-self.window:idx]

            # Calculate price levels based on the current price at this historical point
            current_price = close.iloc[idx]
            price_range = current_price * self.level_pct / 100

            # Define price level bins
            levels_up = [current_price + (i + 1) * price_range for i in range(10)]
            levels_down = [current_price - (i + 1) * price_range for i in range(10)]
            price_levels = levels_down[::-1] + [current_price] + levels_up

            # Calculate volume at each price level in historical window
            total_volume_at_levels = 0

            for level in price_levels:
                # Volume traded when price was at this level in historical data
                at_level = ((historical_low <= level) & (historical_high >= level))
                volume_at_level = (at_level.astype(int) * historical_volume).sum()
                total_volume_at_levels += volume_at_level

            volume_at_levels.iloc[idx] = total_volume_at_levels

        return volume_at_levels.fillna(0)


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


# Historical Price Level Success Rate Generator
class HistoricalPriceLevelSuccessRateGenerator(FeatureGenerator):
    """Generator for historical success rate of price levels - perfect for ML training targets."""

    def __init__(self, level_pct: float = 0.2, window: int = 100, forward_periods: int = 20):
        """Initialize success rate generator.

        Args:
            level_pct: Price level percentage
            window: Lookback window for historical analysis
            forward_periods: How far forward to measure success
        """
        config = FeatureConfig(
            name=f"historical_success_rate_{level_pct}_{window}_{forward_periods}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Historical success rate of {level_pct}% levels over {window} periods, measured {forward_periods} periods ahead",
            required_columns=["close", "high", "low"],
            default_lookback=window + forward_periods,
            min_lookback=window + forward_periods,
            max_lookback=window + forward_periods,
            parameters={'level_pct': level_pct, 'window': window, 'forward_periods': forward_periods}
        )
        super().__init__(config)
        self.level_pct = level_pct
        self.window = window
        self.forward_periods = forward_periods

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate historical success rates for ML training (includes forward-looking labels)."""
        close = data['close']
        high = data['high']
        low = data['low']

        # This includes some forward-looking for creating training labels
        success_rates = pd.Series(index=data.index, dtype=float)

        for idx in range(self.window, len(data) - self.forward_periods):
            # Get historical data up to current point
            historical_close = close.iloc[idx-self.window:idx]
            historical_high = high.iloc[idx-self.window:idx]
            historical_low = low.iloc[idx-self.window:idx]

            # Calculate price levels based on the current price at this historical point
            current_price = close.iloc[idx]
            price_range = current_price * self.level_pct / 100

            # Define price levels
            levels_up = [current_price + (i + 1) * price_range for i in range(5)]
            levels_down = [current_price - (i + 1) * price_range for i in range(5)]

            # Count successful bounces/resistances in historical window
            successful_levels = 0
            total_levels = 0

            for level in levels_up + levels_down:
                # Check if this level was touched in historical data
                touches = ((historical_low <= level) & (historical_high >= level))

                if touches.sum() > 0:
                    total_levels += 1

                    # Check if it acted as support/resistance (price bounced)
                    # Look at price action after touches
                    bounce_count = 0

                    for i in range(len(historical_close)):
                        if touches.iloc[i]:
                            # Check if price reversed after touching
                            if i > 0 and i < len(historical_close) - 1:
                                price_before = historical_close.iloc[i-1]
                                price_at = historical_close.iloc[i]
                                price_after = historical_close.iloc[i+1]

                                # If price approached and then reversed
                                if (price_before < level < price_at and price_after > level) or \
                                   (price_before > level > price_at and price_after < level):
                                    bounce_count += 1

                    if bounce_count > 0:
                        successful_levels += 1

            # Calculate success rate
            success_rate = successful_levels / max(total_levels, 1)
            success_rates.iloc[idx] = success_rate

        return success_rates.fillna(0)


# Historical Price Level Touch Density Generator
class HistoricalPriceLevelTouchDensityGenerator(FeatureGenerator):
    """Generator for touch density analysis - how concentrated touches are around levels."""

    def __init__(self, level_pct: float = 0.2, window: int = 100):
        """Initialize touch density generator.

        Args:
            level_pct: Price level percentage
            window: Lookback window for historical analysis
        """
        config = FeatureConfig(
            name=f"historical_touch_density_{level_pct}_{window}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Historical touch density around {level_pct}% levels over past {window} periods",
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
        """Generate historical touch density scores."""
        close = data['close']
        high = data['high']
        low = data['low']

        # Calculate historical touch density up to each point
        touch_densities = pd.Series(index=data.index, dtype=float)

        for idx in range(self.window, len(data)):
            # Get historical data up to current point
            historical_close = close.iloc[idx-self.window:idx]
            historical_high = high.iloc[idx-self.window:idx]
            historical_low = low.iloc[idx-self.window:idx]

            # Calculate price levels based on the current price at this historical point
            current_price = close.iloc[idx]
            price_range = current_price * self.level_pct / 100

            # Define price levels
            levels_up = [current_price + (i + 1) * price_range for i in range(5)]
            levels_down = [current_price - (i + 1) * price_range for i in range(5)]

            # Calculate touch density for each level
            total_touch_density = 0
            level_count = 0

            for level in levels_up + levels_down:
                level_count += 1

                # Count touches around this level (± half the price range)
                level_tolerance = price_range / 2
                touches_around_level = ((historical_close >= level - level_tolerance) &
                                      (historical_close <= level + level_tolerance)).sum()

                # Normalize by window length and add to total
                density = touches_around_level / self.window
                total_touch_density += density

            # Average density across all levels
            avg_touch_density = total_touch_density / level_count
            touch_densities.iloc[idx] = avg_touch_density

        return touch_densities.fillna(0)


# Historical Price Level Time Decay Generator
class HistoricalPriceLevelTimeDecayGenerator(FeatureGenerator):
    """Generator for time decay analysis - how recency affects level importance."""

    def __init__(self, level_pct: float = 0.2, window: int = 100, decay_half_life: int = 20):
        """Initialize time decay generator.

        Args:
            level_pct: Price level percentage
            window: Lookback window for historical analysis
            decay_half_life: Half-life for exponential decay of importance
        """
        config = FeatureConfig(
            name=f"historical_time_decay_{level_pct}_{window}_{decay_half_life}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Historical time decay analysis for {level_pct}% levels over past {window} periods",
            required_columns=["close", "high", "low"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level_pct': level_pct, 'window': window, 'decay_half_life': decay_half_life}
        )
        super().__init__(config)
        self.level_pct = level_pct
        self.window = window
        self.decay_half_life = decay_half_life

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate historical time decay scores."""
        close = data['close']
        high = data['high']
        low = data['low']

        # Calculate historical time decay up to each point
        time_decay_scores = pd.Series(index=data.index, dtype=float)

        for idx in range(self.window, len(data)):
            # Get historical data up to current point
            historical_close = close.iloc[idx-self.window:idx]
            historical_high = high.iloc[idx-self.window:idx]
            historical_low = low.iloc[idx-self.window:idx]

            # Calculate price levels based on the current price at this historical point
            current_price = close.iloc[idx]
            price_range = current_price * self.level_pct / 100

            # Define price levels
            levels_up = [current_price + (i + 1) * price_range for i in range(3)]
            levels_down = [current_price - (i + 1) * price_range for i in range(3)]

            # Calculate time-decayed importance for each level
            total_decayed_importance = 0
            level_count = 0

            for level in levels_up + levels_down:
                level_count += 1

                # Find all touches in the historical window
                touches = ((historical_low <= level) & (historical_high >= level))

                # Apply exponential time decay
                decayed_touch_value = 0
                for i, touch in enumerate(touches):
                    if touch:
                        # Time since touch (0 = most recent, window-1 = oldest)
                        time_since_touch = self.window - 1 - i
                        # Exponential decay
                        decay_factor = 0.5 ** (time_since_touch / self.decay_half_life)
                        decayed_touch_value += decay_factor

                total_decayed_importance += decayed_touch_value

            # Average decayed importance across all levels
            avg_time_decay = total_decayed_importance / level_count
            time_decay_scores.iloc[idx] = avg_time_decay

        return time_decay_scores.fillna(0)


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

    # Create historical price level analysis generators (backward-looking for ML training)
    for level_pct in price_level_pcts:
        for window in [50, 100, 200]:
            generators.extend([
                HistoricalPriceLevelCrossingGenerator(level_pct, window),
                HistoricalPriceLevelBounceGenerator(level_pct, window),
                HistoricalVolumeAtPriceLevelGenerator(level_pct, window),
                HistoricalPriceLevelTouchDensityGenerator(level_pct, window),
                HistoricalPriceLevelTimeDecayGenerator(level_pct, window, decay_half_life=20),
            ])

        # Add success rate generators with different forward periods
        for forward_periods in [10, 20, 50]:
            generators.append(HistoricalPriceLevelSuccessRateGenerator(level_pct, 100, forward_periods))

    return generators