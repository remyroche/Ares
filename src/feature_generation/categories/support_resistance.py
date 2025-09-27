"""Support and resistance feature generators."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..core.feature_generator import (
    FeatureGenerator, 
    FeatureConfig, 
    FeatureCategory,
    VectorizedFeatureGenerator
)
from ..base_calculations import BaseCalculationType, create_base_calculator


logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

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
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name='support_resistance_signal')

        close = data['close'].astype(float)
        high = data['high'].astype(float) if 'high' in data.columns else close
        low = data['low'].astype(float) if 'low' in data.columns else close
        volume = data['volume'].astype(float) if 'volume' in data.columns else None

        params = self.config.parameters or {}
        pivot_windows = [window for window in params.get('pivot_windows', [self.config.default_lookback]) if window and window > 1]
        fibonacci_levels = params.get('fibonacci_levels', [0.382, 0.5, 0.618])
        volume_windows = [window for window in params.get('volume_profile_windows', []) if window and window > 1]

        aggregated = pd.Series(0.0, index=data.index, dtype=float)
        contributions = 0

        for window in pivot_windows:
            rolling_high = high.rolling(window=window, min_periods=window)
            rolling_low = low.rolling(window=window, min_periods=window)
            highest = rolling_high.max()
            lowest = rolling_low.min()
            price_range = (highest - lowest).replace(0.0, np.nan)

            pivot = (highest + lowest + close.rolling(window=window, min_periods=window).mean()) / 3.0
            pivot_score = ((close - pivot) / price_range).clip(-1.0, 1.0)
            aggregated = aggregated.add(pivot_score.fillna(0.0), fill_value=0.0)
            contributions += 1

            if fibonacci_levels:
                fib_offsets = []
                for level in fibonacci_levels:
                    fib_level = lowest + level * price_range
                    fib_offsets.append(((close - fib_level) / price_range).to_frame(name=str(level)))
                fib_df = pd.concat(fib_offsets, axis=1)

                def _nearest_offset(row: pd.Series) -> float:
                    if row.isna().all():
                        return 0.0
                    idx = row.abs().idxmin()
                    return float(row[idx]) if idx is not None else 0.0

                fib_score = fib_df.apply(_nearest_offset, axis=1).clip(-1.0, 1.0)
                aggregated = aggregated.add(fib_score.fillna(0.0), fill_value=0.0)
                contributions += 1

        if volume is not None and volume_windows:
            price_direction = np.sign(close.diff().fillna(0.0))
            up_volume = volume.where(price_direction >= 0, 0.0)
            down_volume = volume.where(price_direction < 0, 0.0)

            for window in volume_windows:
                up_sum = up_volume.rolling(window=window, min_periods=window).sum()
                down_sum = down_volume.rolling(window=window, min_periods=window).sum()
                total = (up_sum + down_sum).replace(0.0, np.nan)
                imbalance = ((up_sum - down_sum) / total).clip(-1.0, 1.0)
                aggregated = aggregated.add(imbalance.fillna(0.0), fill_value=0.0)
                contributions += 1

        if not contributions:
            return pd.Series(0.0, index=data.index, name='support_resistance_signal')

        signal = aggregated / float(contributions)
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal.clip(-1.0, 1.0).rename('support_resistance_signal')

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

    def __init__(self, level_pct: float = 0.2, window: int = 100, use_bank: bool = True):
        """Initialize historical crossing generator.

        Args:
            level_pct: Price level percentage (e.g., 0.2 for 0.2%)
            window: Lookback window for historical analysis
            use_bank: Whether to check price level bank first
        """
        config = FeatureConfig(
            name=f"historical_crossings_{level_pct}_{window}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Historical crossing counts at {level_pct}% levels over past {window} periods",
            required_columns=["close", "high", "low"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level_pct': level_pct, 'window': window, 'use_bank': use_bank}
        )
        super().__init__(config)
        self.level_pct = level_pct
        self.window = window
        self.use_bank = use_bank

        # Initialize price level bank if available
        self.price_level_bank = None
        if self.use_bank:
            try:
                from ..core.price_level_bank import get_global_price_level_bank
                self.price_level_bank = get_global_price_level_bank()
            except ImportError as exc:
                self.logger.info("Price level bank unavailable, using calculated levels: %s", exc)
            except Exception as exc:
                self.logger.warning("Failed to initialize price level bank fallback: %s", exc)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate historical price level crossing counts (backward looking)."""
        close = data['close']
        high = data['high']
        low = data['low']

        # Try to get data from bank first if available
        if self.price_level_bank and len(data) > 0:
            symbol = kwargs.get('symbol', 'BTCUSDT')  # Default symbol
            timeframe = kwargs.get('timeframe', '1h')  # Default timeframe

            try:
                # Get current price (last value in data)
                current_price = close.iloc[-1]

                # Query bank for relevant levels
                levels = self.price_level_bank.query_levels(
                    symbol=symbol,
                    timeframe=timeframe,
                    min_price=current_price * 0.5,  # Look in reasonable range
                    max_price=current_price * 1.5,
                    limit=50  # Limit to avoid too much data
                )

                if levels:
                    # Use bank data to generate features
                    return self._generate_from_bank(data, levels)

            except Exception as e:
                # Fall back to calculation if bank query fails
                self.logger.warning("Price level bank query failed (%s); using calculated levels", e)

        # Fall back to original calculation method
        return self._generate_from_calculation(data)

    def _generate_from_bank(self, data: pd.DataFrame, levels: List) -> pd.Series:
        """Generate features using bank data."""
        close = data['close']
        crossings = pd.Series(index=data.index, dtype=float)

        for idx in range(self.window, len(data)):
            total_crossings = 0

            for level in levels:
                # Check if this level is relevant for this data point
                # (Levels are stored with their historical significance)
                if level.level_pct == self.level_pct:
                    # Use the historical crossing count from the bank
                    total_crossings += level.historical_crossings

            crossings.iloc[idx] = total_crossings

        return crossings.fillna(0).astype(int)

    def _generate_from_calculation(self, data: pd.DataFrame) -> pd.Series:
        """Generate features using original calculation method."""
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

    def __init__(self, level_pct: float = 0.2, window: int = 100, use_bank: bool = True):
        """Initialize historical bounce generator.

        Args:
            level_pct: Price level percentage
            window: Lookback window for historical analysis
            use_bank: Whether to check price level bank first
        """
        config = FeatureConfig(
            name=f"historical_bounces_{level_pct}_{window}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Historical bounce counts at {level_pct}% levels over past {window} periods",
            required_columns=["close", "high", "low"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level_pct': level_pct, 'window': window, 'use_bank': use_bank}
        )
        super().__init__(config)
        self.level_pct = level_pct
        self.window = window
        self.use_bank = use_bank

        # Initialize price level bank if available
        self.price_level_bank = None
        if self.use_bank:
            try:
                from ..core.price_level_bank import get_global_price_level_bank

                self.price_level_bank = get_global_price_level_bank()
            except ImportError as exc:
                logger.info("Price level bank unavailable; calculated levels will be used: %s", exc)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("Failed to initialize price level bank fallback: %s", exc)

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

    def __init__(self, level_pct: float = 0.2, window: int = 100, use_bank: bool = True):
        """Initialize historical volume at price level generator.

        Args:
            level_pct: Price level percentage
            window: Lookback window for historical analysis
            use_bank: Whether to check price level bank first
        """
        config = FeatureConfig(
            name=f"historical_volume_at_levels_{level_pct}_{window}",
            category=FeatureCategory.VOLUME,
            description=f"Historical volume traded at {level_pct}% price levels over past {window} periods",
            required_columns=["close", "high", "low", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level_pct': level_pct, 'window': window, 'use_bank': use_bank}
        )
        super().__init__(config)
        self.level_pct = level_pct
        self.window = window
        self.use_bank = use_bank

        # Initialize price level bank if available
        self.price_level_bank = None
        if self.use_bank:
            try:
                from ..core.price_level_bank import get_global_price_level_bank

                self.price_level_bank = get_global_price_level_bank()
            except ImportError as exc:
                logger.info("Price level bank unavailable; volume levels will use local calculations: %s", exc)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("Unexpected error initialising price level bank: %s", exc)

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


# Situational Awareness Feature Generator
class SituationalAwarenessGenerator(FeatureGenerator):
    """Generator for situational awareness features based on price level bank data."""

    def __init__(self, current_price: float = None):
        """Initialize situational awareness generator.

        Args:
            current_price: Current market price (can be None to use latest from data)
        """
        config = FeatureConfig(
            name="situational_awareness",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description="Situational awareness features including closest price levels by percentage",
            required_columns=["close"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            parameters={'current_price': current_price}
        )
        super().__init__(config)
        self.current_price = current_price

        # Initialize price level bank
        self.price_level_bank = None
        try:
            from ..core.price_level_bank import get_global_price_level_bank

            self.price_level_bank = get_global_price_level_bank()
        except ImportError as exc:
            logger.info("Situational awareness running without shared price level bank: %s", exc)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Failed to initialise price level bank for situational awareness: %s", exc)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate situational awareness features."""
        if not self.price_level_bank or data.empty:
            # Return empty features if bank not available
            return {}

        # Get current price (from parameter or latest data)
        current_price = self.current_price
        if current_price is None:
            current_price = data['close'].iloc[-1]

        # Get symbol and timeframe
        symbol = kwargs.get('symbol', 'BTCUSDT')
        timeframe = kwargs.get('timeframe', '1h')

        # Get situational awareness data
        awareness = self.price_level_bank.get_situational_awareness(
            symbol, timeframe, current_price
        )

        # Convert to feature series
        features = {}

        # Distance to closest 0.2% levels (percentage only + raw historical data)
        if 0.2 in awareness['distances']['above']:
            level_data = awareness['distances']['above'][0.2]
            features['closest_0.2pct_above_pct'] = pd.Series(
                [level_data['distance_pct']] * len(data),
                index=data.index
            )
            features['closest_0.2pct_above_crossings'] = pd.Series(
                [level_data['historical_crossings']] * len(data),
                index=data.index
            )
            features['closest_0.2pct_above_bounces'] = pd.Series(
                [level_data['historical_bounces']] * len(data),
                index=data.index
            )
            features['closest_0.2pct_above_volume'] = pd.Series(
                [level_data['historical_volume']] * len(data),
                index=data.index
            )

        if 0.2 in awareness['distances']['below']:
            level_data = awareness['distances']['below'][0.2]
            features['closest_0.2pct_below_pct'] = pd.Series(
                [level_data['distance_pct']] * len(data),
                index=data.index
            )
            features['closest_0.2pct_below_crossings'] = pd.Series(
                [level_data['historical_crossings']] * len(data),
                index=data.index
            )
            features['closest_0.2pct_below_bounces'] = pd.Series(
                [level_data['historical_bounces']] * len(data),
                index=data.index
            )
            features['closest_0.2pct_below_volume'] = pd.Series(
                [level_data['historical_volume']] * len(data),
                index=data.index
            )

        # Distance to closest 0.4% levels (percentage only + raw historical data)
        if 0.4 in awareness['distances']['above']:
            level_data = awareness['distances']['above'][0.4]
            features['closest_0.4pct_above_pct'] = pd.Series(
                [level_data['distance_pct']] * len(data),
                index=data.index
            )
            features['closest_0.4pct_above_crossings'] = pd.Series(
                [level_data['historical_crossings']] * len(data),
                index=data.index
            )
            features['closest_0.4pct_above_bounces'] = pd.Series(
                [level_data['historical_bounces']] * len(data),
                index=data.index
            )
            features['closest_0.4pct_above_volume'] = pd.Series(
                [level_data['historical_volume']] * len(data),
                index=data.index
            )

        if 0.4 in awareness['distances']['below']:
            level_data = awareness['distances']['below'][0.4]
            features['closest_0.4pct_below_pct'] = pd.Series(
                [level_data['distance_pct']] * len(data),
                index=data.index
            )
            features['closest_0.4pct_below_crossings'] = pd.Series(
                [level_data['historical_crossings']] * len(data),
                index=data.index
            )
            features['closest_0.4pct_below_bounces'] = pd.Series(
                [level_data['historical_bounces']] * len(data),
                index=data.index
            )
            features['closest_0.4pct_below_volume'] = pd.Series(
                [level_data['historical_volume']] * len(data),
                index=data.index
            )

        # Distance to closest 0.8% levels (percentage only + raw historical data)
        if 0.8 in awareness['distances']['above']:
            level_data = awareness['distances']['above'][0.8]
            features['closest_0.8pct_above_pct'] = pd.Series(
                [level_data['distance_pct']] * len(data),
                index=data.index
            )
            features['closest_0.8pct_above_crossings'] = pd.Series(
                [level_data['historical_crossings']] * len(data),
                index=data.index
            )
            features['closest_0.8pct_above_bounces'] = pd.Series(
                [level_data['historical_bounces']] * len(data),
                index=data.index
            )
            features['closest_0.8pct_above_volume'] = pd.Series(
                [level_data['historical_volume']] * len(data),
                index=data.index
            )

        if 0.8 in awareness['distances']['below']:
            level_data = awareness['distances']['below'][0.8]
            features['closest_0.8pct_below_pct'] = pd.Series(
                [level_data['distance_pct']] * len(data),
                index=data.index
            )
            features['closest_0.8pct_below_crossings'] = pd.Series(
                [level_data['historical_crossings']] * len(data),
                index=data.index
            )
            features['closest_0.8pct_below_bounces'] = pd.Series(
                [level_data['historical_bounces']] * len(data),
                index=data.index
            )
            features['closest_0.8pct_below_volume'] = pd.Series(
                [level_data['historical_volume']] * len(data),
                index=data.index
            )

        # Number of significant levels nearby
        features['significant_levels_nearby'] = pd.Series(
            [len(awareness['significant_nearby'])] * len(data),
            index=data.index
        )

        # Average significance of nearby levels
        if awareness['significant_nearby']:
            avg_significance = sum(l.significance_level for l in awareness['significant_nearby']) / len(awareness['significant_nearby'])
            features['avg_significance_nearby'] = pd.Series(
                [avg_significance] * len(data),
                index=data.index
            )

        # Price ranges for context
        features['price_range_0.2pct'] = pd.Series(
            [awareness['price_ranges']['0.2%']] * len(data),
            index=data.index
        )
        features['price_range_0.4pct'] = pd.Series(
            [awareness['price_ranges']['0.4%']] * len(data),
            index=data.index
        )
        features['price_range_0.8pct'] = pd.Series(
            [awareness['price_ranges']['0.8%']] * len(data),
            index=data.index
        )
        features['price_range_1.0pct'] = pd.Series(
            [awareness['price_ranges']['1.0%']] * len(data),
            index=data.index
        )

        return features

    def generate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate situational awareness features as DataFrame."""
        features_dict = self._generate_feature(data, **kwargs)
        return pd.DataFrame(features_dict)


# Closest Price Level Generator
class ClosestPriceLevelGenerator(FeatureGenerator):
    """Generator for closest price level features by percentage."""

    def __init__(self, level_pct: float = 0.2, direction: str = 'both'):
        """Initialize closest level generator.

        Args:
            level_pct: Percentage for price level (0.2 for 0.2%)
            direction: 'above', 'below', or 'both'
        """
        config = FeatureConfig(
            name=f"closest_{level_pct}pct_levels_{direction}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Closest {level_pct}% price levels {direction}",
            required_columns=["close"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            parameters={'level_pct': level_pct, 'direction': direction}
        )
        super().__init__(config)
        self.level_pct = level_pct
        self.direction = direction

        # Initialize price level bank
        self.price_level_bank = None
        try:
            from ..core.price_level_bank import get_global_price_level_bank

            self.price_level_bank = get_global_price_level_bank()
        except ImportError as exc:
            logger.info("Closest price level generator running without shared price level bank: %s", exc)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Unexpected error initialising price level bank: %s", exc)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate closest price level features."""
        if not self.price_level_bank or data.empty:
            return {}

        current_price = data['close'].iloc[-1]
        symbol = kwargs.get('symbol', 'BTCUSDT')
        timeframe = kwargs.get('timeframe', '1h')

        # Get closest levels
        closest_levels = self.price_level_bank.get_closest_levels_by_percentage(
            symbol, timeframe, current_price, [self.level_pct]
        )

        features = {}

        if self.direction in ['above', 'both']:
            if closest_levels['above']:
                closest_above = closest_levels['above'][0]  # First (closest) level
                features[f'closest_{self.level_pct}pct_above'] = pd.Series(
                    [closest_above.price] * len(data), index=data.index
                )
                features[f'closest_{self.level_pct}pct_above_distance'] = pd.Series(
                    [closest_above.price - current_price] * len(data), index=data.index
                )
                features[f'closest_{self.level_pct}pct_above_distance_pct'] = pd.Series(
                    [(closest_above.price - current_price) / current_price * 100] * len(data),
                    index=data.index
                )
                features[f'closest_{self.level_pct}pct_above_significance'] = pd.Series(
                    [closest_above.significance_level] * len(data), index=data.index
                )

        if self.direction in ['below', 'both']:
            if closest_levels['below']:
                closest_below = closest_levels['below'][0]  # First (closest) level
                features[f'closest_{self.level_pct}pct_below'] = pd.Series(
                    [closest_below.price] * len(data), index=data.index
                )
                features[f'closest_{self.level_pct}pct_below_distance'] = pd.Series(
                    [current_price - closest_below.price] * len(data), index=data.index
                )
                features[f'closest_{self.level_pct}pct_below_distance_pct'] = pd.Series(
                    [(current_price - closest_below.price) / current_price * 100] * len(data),
                    index=data.index
                )
                features[f'closest_{self.level_pct}pct_below_significance'] = pd.Series(
                    [closest_below.significance_level] * len(data), index=data.index
                )

        return features

    def generate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate closest price level features as DataFrame."""
        features_dict = self._generate_feature(data, **kwargs)
        return pd.DataFrame(features_dict)


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
                HistoricalPriceLevelCrossingGenerator(level_pct, window, use_bank=True),
                HistoricalPriceLevelBounceGenerator(level_pct, window, use_bank=True),
                HistoricalVolumeAtPriceLevelGenerator(level_pct, window, use_bank=True),
                HistoricalPriceLevelTouchDensityGenerator(level_pct, window),
                HistoricalPriceLevelTimeDecayGenerator(level_pct, window, decay_half_life=20),
            ])

        # Add success rate generators with different forward periods
        for forward_periods in [10, 20, 50]:
            generators.append(HistoricalPriceLevelSuccessRateGenerator(level_pct, 100, forward_periods))

    # Add situational awareness generators (default features)
    generators.extend([
        SituationalAwarenessGenerator(),  # Provides comprehensive situational awareness
        ClosestPriceLevelGenerator(0.2, 'both'),  # Closest 0.2% levels above/below
        ClosestPriceLevelGenerator(0.4, 'both'),  # Closest 0.4% levels above/below
        ClosestPriceLevelGenerator(0.8, 'both'),  # Closest 0.8% levels above/below
        ClosestPriceLevelGenerator(1.0, 'above'), # Closest 1.0% level above
        ClosestPriceLevelGenerator(1.0, 'below'), # Closest 1.0% level below
    ])

    return generators
