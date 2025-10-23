"""
Mathematical Pattern Discovery & Definition Framework

This module focuses exclusively on discovering and mathematically defining price patterns
from market data. It provides precise, reproducible definitions that can be used for:
- ML target generation
- Pattern recognition research
- Economic significance testing
- Trading strategy development

Core Philosophy:
A price pattern must be:
1. Mathematically precise (exact formula)
2. Measurable (binary outcome: pattern exists or not)
3. Reproducible (same definition across datasets)
4. Frequent enough for statistical analysis
5. Economically meaningful (not random noise)

Now inherits from the production-ready BasePatternDiscoverer in core module.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from abc import ABC, abstractmethod

from src.utils.logger import system_logger

# Import the production-ready BasePatternDiscoverer
from src.core.abstract_base_classes import BasePatternDiscoverer as ProductionBasePatternDiscoverer, PatternDiscoveryResult, PatternDefinition, PatternType

class PatternType(Enum):
    """Categories of price patterns."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    TREND = "trend"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"

@dataclass
class PatternDefinition:
    """Mathematical definition of a price pattern."""
    name: str
    pattern_type: PatternType
    description: str
    mathematical_formula: str
    parameters: Dict[str, Any]
    frequency_threshold: float  # Minimum occurrence rate for validity

    def __str__(self) -> str:
        return f"{self.name}: {self.description}\nFormula: {self.mathematical_formula}"

@dataclass
class PatternDiscoveryResult:
    """Result of pattern discovery analysis."""
    definition: PatternDefinition
    labels: pd.Series  # Binary labels: 1 = pattern exists, 0 = no pattern
    frequency: float  # How often pattern occurs (0-1)
    duration_stats: Dict[str, float]  # Pattern duration statistics
    magnitude_stats: Dict[str, float]  # Pattern magnitude statistics
    predictability_score: float  # How predictable the pattern is
    noise_ratio: float  # Signal-to-noise ratio
    statistical_significance: Dict[str, float]  # Statistical tests

    @property
    def is_valid_pattern(self) -> bool:
        """Check if pattern meets validity criteria."""
        return (
            self.frequency >= self.definition.frequency_threshold and
            self.predictability_score > 0.1 and
            self.noise_ratio < 0.8
        )

class BasePatternDiscoverer(ProductionBasePatternDiscoverer):
    """Base class for pattern discovery."""

    def __init__(self, name: str, pattern_type: PatternType):
        self.name = name
        self.pattern_type = pattern_type
        self.logger = system_logger.getChild(f'PatternDiscoverer_{name}')

    def discover_pattern(self, prices: pd.Series, **kwargs) -> PatternDiscoveryResult:
        """Discover and define pattern in price data."""
        # Default implementation that can be overridden by subclasses
        try:
            # Create basic pattern labels (all zeros for base implementation)
            labels = pd.Series(0, index=prices.index)
            
            # Calculate basic statistics
            stats = self._calculate_pattern_statistics(labels, prices)
            
            # Create basic pattern definition
            definition = self.get_pattern_definition()
            
            return PatternDiscoveryResult(
                definition=definition,
                labels=labels,
                frequency=stats['frequency'],
                duration_stats=stats['duration_stats'],
                magnitude_stats=stats['magnitude_stats'],
                predictability_score=stats['predictability_score'],
                noise_ratio=stats['noise_ratio'],
                statistical_significance=stats['statistical_significance']
            )
        except Exception as e:
            self.logger.error(f"Pattern discovery failed: {e}")
            # Return empty result
            return PatternDiscoveryResult(
                definition=self.get_pattern_definition(),
                labels=pd.Series(0, index=prices.index),
                frequency=0.0,
                duration_stats={'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0},
                magnitude_stats={'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0},
                predictability_score=0.0,
                noise_ratio=1.0,
                statistical_significance={'p_value': 1.0, 't_statistic': 0.0}
            )

    def get_pattern_definition(self) -> PatternDefinition:
        """Get mathematical definition of the pattern."""
        # Default implementation that can be overridden by subclasses
        return PatternDefinition(
            name="Base Pattern",
            pattern_type=self.pattern_type,
            description="Base pattern implementation",
            mathematical_formula="Base pattern formula",
            parameters={},
            frequency_threshold=0.0
        )

    def _calculate_pattern_statistics(self,
                                    labels: pd.Series,
                                    prices: pd.Series,
                                    pattern_magnitudes: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate comprehensive pattern statistics."""

        # Basic frequency
        frequency = labels.sum() / len(labels)

        # Duration statistics
        duration_stats = self._calculate_duration_statistics(labels)

        # Magnitude statistics
        if pattern_magnitudes is not None:
            magnitude_stats = self._calculate_magnitude_statistics(pattern_magnitudes)
        else:
            magnitude_stats = {}

        # Predictability score
        predictability_score = self._calculate_predictability_score(labels)

        # Noise ratio
        noise_ratio = self._calculate_noise_ratio(labels, prices)

        # Statistical significance
        statistical_significance = self._calculate_statistical_significance(labels, prices)

        return {
            'frequency': frequency,
            'duration_stats': duration_stats,
            'magnitude_stats': magnitude_stats,
            'predictability_score': predictability_score,
            'noise_ratio': noise_ratio,
            'statistical_significance': statistical_significance
        }

    def _calculate_duration_statistics(self, labels: pd.Series) -> Dict[str, float]:
        """Calculate pattern duration statistics."""

        durations = []
        current_duration = 0
        in_pattern = False

        for label in labels:
            if label == 1:
                if not in_pattern:
                    in_pattern = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if in_pattern:
                    durations.append(current_duration)
                    in_pattern = False
                    current_duration = 0

        # Handle case where pattern continues to end
        if in_pattern:
            durations.append(current_duration)

        if not durations:
            return {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}

        return {
            'mean': float(np.mean(durations)),
            'median': float(np.median(durations)),
            'std': float(np.std(durations)),
            'min': float(np.min(durations)),
            'max': float(np.max(durations))
        }

    def _calculate_magnitude_statistics(self, magnitudes: pd.Series) -> Dict[str, float]:
        """Calculate pattern magnitude statistics."""

        non_zero_magnitudes = magnitudes[magnitudes > 0]

        if len(non_zero_magnitudes) == 0:
            return {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}

        return {
            'mean': float(non_zero_magnitudes.mean()),
            'median': float(non_zero_magnitudes.median()),
            'std': float(non_zero_magnitudes.std()),
            'min': float(non_zero_magnitudes.min()),
            'max': float(non_zero_magnitudes.max())
        }

    def _calculate_predictability_score(self, labels: pd.Series) -> float:
        """Calculate pattern predictability using entropy."""

        if len(labels) == 0:
            return 0.0

        pattern_freq = labels.sum() / len(labels)

        if pattern_freq == 0 or pattern_freq == 1:
            return 1.0  # Completely predictable

        # Shannon entropy
        entropy = -pattern_freq * np.log2(pattern_freq) - (1 - pattern_freq) * np.log2(1 - pattern_freq)

        # Convert to predictability (1 - normalized entropy)
        max_entropy = 1.0
        predictability = 1.0 - (entropy / max_entropy)

        return float(predictability)

    def _calculate_noise_ratio(self, labels: pd.Series, prices: pd.Series) -> float:
        """Calculate signal-to-noise ratio."""

        if labels.sum() == 0:
            return 1.0  # All noise

        returns = prices.pct_change().fillna(0)

        # Signal: return variance when pattern is active
        pattern_returns = returns[labels == 1]
        no_pattern_returns = returns[labels == 0]

        if len(pattern_returns) == 0 or len(no_pattern_returns) == 0:
            return 1.0

        pattern_variance = pattern_returns.var()
        no_pattern_variance = no_pattern_returns.var()

        if pattern_variance == 0:
            return 1.0

        # Noise ratio: how much of pattern variance is just normal market noise
        noise_ratio = min(no_pattern_variance / pattern_variance, 1.0)

        return float(noise_ratio)

    def _calculate_statistical_significance(self, labels: pd.Series, prices: pd.Series) -> Dict[str, float]:
        """Calculate statistical significance of pattern."""

        results = {}

        if labels.sum() == 0:
            return {'p_value': 1.0, 't_statistic': 0.0}

        returns = prices.pct_change().fillna(0)

        # Test if returns are different when pattern is active
        pattern_returns = returns[labels == 1]
        no_pattern_returns = returns[labels == 0]

        if len(pattern_returns) > 5 and len(no_pattern_returns) > 5:
            try:
                t_stat, p_value = stats.ttest_ind(pattern_returns, no_pattern_returns)
                results['t_statistic'] = float(t_stat)
                results['p_value'] = float(p_value)
            except:
                results['t_statistic'] = 0.0
                results['p_value'] = 1.0
        else:
            results['t_statistic'] = 0.0
            results['p_value'] = 1.0

        return results

class MomentumPersistenceDiscoverer(BasePatternDiscoverer):
    """Discover momentum persistence patterns."""

    def __init__(self):
        super().__init__("MomentumPersistence", PatternType.MOMENTUM)

    def get_pattern_definition(self) -> PatternDefinition:
        return PatternDefinition(
            name="Momentum Persistence",
            pattern_type=PatternType.MOMENTUM,
            description="Price momentum continues in the same direction for multiple periods with gradual decay",
            mathematical_formula="""
            Let momentum(t) = mean(returns[t-window+1:t])
            Let threshold = momentum_threshold
            Let persistence_window = P

            Pattern exists at time t IF:
            1. |momentum(t)| > threshold
            2. sign(momentum(t+k)) == sign(momentum(t)) for ≥70% of k in [1,P]
            3. |momentum(t+k)| > 0.3 * |momentum(t)| for ≥60% of k in [1,P]
            """,
            parameters={
                'momentum_window': 5,
                'persistence_window': 10,
                'momentum_threshold': 0.005,
                'direction_persistence_rate': 0.7,
                'magnitude_decay_rate': 0.6,
                'minimum_magnitude_ratio': 0.3
            },
            frequency_threshold=0.05  # Must occur at least 5% of time
        )

    def discover_pattern(self,
                        prices: pd.Series,
                        momentum_window: int = 5,
                        persistence_window: int = 10,
                        momentum_threshold: float = 0.005,
                        direction_persistence_rate: float = 0.7,
                        magnitude_decay_rate: float = 0.6,
                        minimum_magnitude_ratio: float = 0.3) -> PatternDiscoveryResult:
        """
        Discover momentum persistence patterns.

        Mathematical Definition:
        1. Calculate momentum as rolling mean of returns
        2. Identify periods with significant momentum (> threshold)
        3. Check if momentum persists in same direction for specified window
        4. Verify magnitude decays gradually (not abruptly)
        """

        self.logger.info("🚀 Discovering momentum persistence patterns")

        # Calculate momentum
        returns = prices.pct_change().fillna(0)
        momentum = returns.rolling(momentum_window).mean()

        labels = []
        magnitudes = []

        for i in range(len(momentum) - persistence_window):
            current_momentum = momentum.iloc[i]

            if abs(current_momentum) > momentum_threshold:
                # Get future momentum values
                future_momentum = momentum.iloc[i+1:i+persistence_window+1]

                # Check direction persistence
                same_direction_count = (np.sign(future_momentum) == np.sign(current_momentum)).sum()
                direction_persistence = same_direction_count / len(future_momentum)

                # Check magnitude decay (gradual vs abrupt)
                magnitude_ratios = abs(future_momentum) / abs(current_momentum)
                gradual_decay_count = (magnitude_ratios > minimum_magnitude_ratio).sum()
                magnitude_persistence = gradual_decay_count / len(magnitude_ratios)

                # Pattern exists if both conditions met
                pattern_exists = (
                    direction_persistence >= direction_persistence_rate and
                    magnitude_persistence >= magnitude_decay_rate
                )

                labels.append(1 if pattern_exists else 0)
                magnitudes.append(abs(current_momentum) if pattern_exists else 0)
            else:
                labels.append(0)
                magnitudes.append(0)

        # Create result series
        pattern_labels = pd.Series(labels, index=prices.index[:len(labels)])
        pattern_magnitudes = pd.Series(magnitudes, index=prices.index[:len(magnitudes)])

        # Calculate statistics
        stats = self._calculate_pattern_statistics(pattern_labels, prices, pattern_magnitudes)

        return PatternDiscoveryResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            noise_ratio=stats['noise_ratio'],
            statistical_significance=stats['statistical_significance']
        )

class MeanReversionSpeedDiscoverer(BasePatternDiscoverer):
    """Discover mean reversion speed patterns."""

    def __init__(self):
        super().__init__("MeanReversionSpeed", PatternType.MEAN_REVERSION)

    def get_pattern_definition(self) -> PatternDefinition:
        return PatternDefinition(
            name="Mean Reversion Speed",
            pattern_type=PatternType.MEAN_REVERSION,
            description="Price reverts toward moving average within specified timeframe",
            mathematical_formula="""
            Let MA(t) = moving_average(prices[t-window+1:t])
            Let deviation(t) = (price(t) - MA(t)) / MA(t)
            Let threshold = deviation_threshold
            Let reversion_window = R

            Pattern exists at time t IF:
            1. |deviation(t)| > threshold
            2. ∃k ∈ [1,R]: |price(t+k) - MA(t)| < 0.7 * |price(t) - MA(t)|
            """,
            parameters={
                'ma_window': 20,
                'deviation_threshold': 0.02,
                'reversion_window': 10,
                'reversion_ratio': 0.7
            },
            frequency_threshold=0.1  # Must occur at least 10% of time
        )

    def discover_pattern(self,
                        prices: pd.Series,
                        ma_window: int = 20,
                        deviation_threshold: float = 0.02,
                        reversion_window: int = 10,
                        reversion_ratio: float = 0.7) -> PatternDiscoveryResult:
        """
        Discover mean reversion speed patterns.

        Mathematical Definition:
        1. Calculate deviation from moving average
        2. Identify significant deviations (> threshold)
        3. Check if price reverts closer to mean within window
        4. Measure reversion speed and magnitude
        """

        self.logger.info("🔄 Discovering mean reversion speed patterns")

        # Calculate moving average and deviation
        ma = prices.rolling(ma_window).mean()
        deviation = (prices - ma) / ma

        labels = []
        magnitudes = []

        for i in range(ma_window, len(prices) - reversion_window):
            current_deviation = deviation.iloc[i]

            if abs(current_deviation) > deviation_threshold:
                current_price = prices.iloc[i]
                target_ma = ma.iloc[i]
                current_distance = abs(current_price - target_ma)

                # Look for reversion in future periods
                future_prices = prices.iloc[i+1:i+reversion_window+1]

                reversion_occurred = False
                reversion_speed = 0

                for j, future_price in enumerate(future_prices):
                    future_distance = abs(future_price - target_ma)

                    # Check if significantly closer to mean
                    if future_distance < current_distance * reversion_ratio:
                        reversion_occurred = True
                        reversion_speed = current_distance / (j + 1)  # Distance per period
                        break

                labels.append(1 if reversion_occurred else 0)
                magnitudes.append(reversion_speed if reversion_occurred else 0)
            else:
                labels.append(0)
                magnitudes.append(0)

        # Create result series
        pattern_labels = pd.Series(labels, index=prices.index[ma_window:ma_window+len(labels)])
        pattern_magnitudes = pd.Series(magnitudes, index=prices.index[ma_window:ma_window+len(magnitudes)])

        # Calculate statistics
        stats = self._calculate_pattern_statistics(pattern_labels, prices, pattern_magnitudes)

        return PatternDiscoveryResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            noise_ratio=stats['noise_ratio'],
            statistical_significance=stats['statistical_significance']
        )

class VolatilityExpansionDiscoverer(BasePatternDiscoverer):
    """Discover volatility expansion patterns."""

    def __init__(self):
        super().__init__("VolatilityExpansion", PatternType.VOLATILITY)

    def get_pattern_definition(self) -> PatternDefinition:
        return PatternDefinition(
            name="Volatility Expansion",
            pattern_type=PatternType.VOLATILITY,
            description="Low volatility periods followed by high volatility expansion",
            mathematical_formula="""
            Let vol(t) = std(returns[t-window+1:t])
            Let vol_percentile(t) = percentile_rank(vol(t), lookback_window)
            Let expansion_window = E

            Pattern exists at time t IF:
            1. vol_percentile(t) < low_threshold
            2. ∃k ∈ [1,E]: vol_percentile(t+k) > high_threshold
            3. Expansion rate ≥ minimum_expansion_rate
            """,
            parameters={
                'vol_window': 20,
                'lookback_window': 100,
                'expansion_window': 10,
                'low_vol_threshold': 0.2,
                'high_vol_threshold': 0.8,
                'minimum_expansion_rate': 0.3
            },
            frequency_threshold=0.08  # Must occur at least 8% of time
        )

    def discover_pattern(self,
                        prices: pd.Series,
                        vol_window: int = 20,
                        lookback_window: int = 100,
                        expansion_window: int = 10,
                        low_vol_threshold: float = 0.2,
                        high_vol_threshold: float = 0.8,
                        minimum_expansion_rate: float = 0.3) -> PatternDiscoveryResult:
        """
        Discover volatility expansion patterns.

        Mathematical Definition:
        1. Calculate rolling volatility
        2. Rank volatility in percentiles over lookback window
        3. Identify low volatility periods
        4. Check for subsequent high volatility within expansion window
        """

        self.logger.info("📈 Discovering volatility expansion patterns")

        # Calculate volatility and percentiles
        returns = prices.pct_change().fillna(0)
        volatility = returns.rolling(vol_window).std()
        vol_percentile = volatility.rolling(lookback_window).rank(pct=True)

        labels = []
        magnitudes = []

        for i in range(lookback_window, len(volatility) - expansion_window):
            current_vol_percentile = vol_percentile.iloc[i]

            if current_vol_percentile < low_vol_threshold:
                future_vol_percentiles = vol_percentile.iloc[i+1:i+expansion_window+1]

                # Check for volatility expansion
                high_vol_periods = (future_vol_percentiles > high_vol_threshold).sum()
                expansion_rate = high_vol_periods / len(future_vol_percentiles)

                # Calculate expansion magnitude
                if expansion_rate > 0:
                    max_future_vol = future_vol_percentiles.max()
                    expansion_magnitude = max_future_vol - current_vol_percentile
                else:
                    expansion_magnitude = 0

                pattern_exists = expansion_rate >= minimum_expansion_rate

                labels.append(1 if pattern_exists else 0)
                magnitudes.append(expansion_magnitude if pattern_exists else 0)
            else:
                labels.append(0)
                magnitudes.append(0)

        # Create result series
        pattern_labels = pd.Series(labels, index=volatility.index[lookback_window:lookback_window+len(labels)])
        pattern_magnitudes = pd.Series(magnitudes, index=volatility.index[lookback_window:lookback_window+len(magnitudes)])

        # Calculate statistics
        stats = self._calculate_pattern_statistics(pattern_labels, prices, pattern_magnitudes)

        return PatternDiscoveryResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            noise_ratio=stats['noise_ratio'],
            statistical_significance=stats['statistical_significance']
        )

class BreakoutConfirmationDiscoverer(BasePatternDiscoverer):
    """Discover confirmed breakout patterns."""

    def __init__(self):
        super().__init__("BreakoutConfirmation", PatternType.BREAKOUT)

    def get_pattern_definition(self) -> PatternDefinition:
        return PatternDefinition(
            name="Confirmed Breakout",
            pattern_type=PatternType.BREAKOUT,
            description="Price breaks technical level and continues in breakout direction",
            mathematical_formula="""
            Let upper_band(t) = MA(t) + 2*STD(t)
            Let lower_band(t) = MA(t) - 2*STD(t)
            Let confirmation_window = C

            Pattern exists at time t IF:
            1. price(t) > upper_band(t) OR price(t) < lower_band(t)
            2. ≥60% of price(t+k) for k∈[1,C] continue beyond breakout level
            3. Continuation magnitude > minimum_continuation
            """,
            parameters={
                'bb_window': 20,
                'bb_std': 2.0,
                'confirmation_window': 5,
                'confirmation_rate': 0.6,
                'minimum_continuation': 0.01
            },
            frequency_threshold=0.05  # Must occur at least 5% of time
        )

    def discover_pattern(self,
                        prices: pd.Series,
                        bb_window: int = 20,
                        bb_std: float = 2.0,
                        confirmation_window: int = 5,
                        confirmation_rate: float = 0.6,
                        minimum_continuation: float = 0.01) -> PatternDiscoveryResult:
        """
        Discover confirmed breakout patterns.

        Mathematical Definition:
        1. Calculate Bollinger Bands (or similar technical levels)
        2. Identify price breakouts above/below bands
        3. Confirm breakout continues in same direction
        4. Measure continuation magnitude and duration
        """

        self.logger.info("📊 Discovering confirmed breakout patterns")

        # Calculate Bollinger Bands
        ma = prices.rolling(bb_window).mean()
        std = prices.rolling(bb_window).std()
        upper_band = ma + bb_std * std
        lower_band = ma - bb_std * std

        labels = []
        magnitudes = []

        for i in range(bb_window, len(prices) - confirmation_window):
            current_price = prices.iloc[i]
            current_upper = upper_band.iloc[i]
            current_lower = lower_band.iloc[i]

            # Check for breakout
            upper_breakout = current_price > current_upper
            lower_breakout = current_price < current_lower

            if upper_breakout or lower_breakout:
                future_prices = prices.iloc[i+1:i+confirmation_window+1]

                if upper_breakout:
                    # Confirm upward breakout
                    confirmation_count = (future_prices > current_upper).sum()
                    confirmation_pct = confirmation_count / len(future_prices)

                    # Calculate continuation magnitude
                    max_future = future_prices.max()
                    continuation_magnitude = (max_future - current_price) / current_price

                    pattern_exists = (
                        confirmation_pct >= confirmation_rate and
                        continuation_magnitude > minimum_continuation
                    )

                elif lower_breakout:
                    # Confirm downward breakout
                    confirmation_count = (future_prices < current_lower).sum()
                    confirmation_pct = confirmation_count / len(future_prices)

                    # Calculate continuation magnitude
                    min_future = future_prices.min()
                    continuation_magnitude = (current_price - min_future) / current_price

                    pattern_exists = (
                        confirmation_pct >= confirmation_rate and
                        continuation_magnitude > minimum_continuation
                    )

                labels.append(1 if pattern_exists else 0)
                magnitudes.append(continuation_magnitude if pattern_exists else 0)
            else:
                labels.append(0)
                magnitudes.append(0)

        # Create result series
        pattern_labels = pd.Series(labels, index=prices.index[bb_window:bb_window+len(labels)])
        pattern_magnitudes = pd.Series(magnitudes, index=prices.index[bb_window:bb_window+len(magnitudes)])

        # Calculate statistics
        stats = self._calculate_pattern_statistics(pattern_labels, prices, pattern_magnitudes)

        return PatternDiscoveryResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            noise_ratio=stats['noise_ratio'],
            statistical_significance=stats['statistical_significance']
        )

class TrendContinuationDiscoverer(BasePatternDiscoverer):
    """Discover trend continuation patterns."""

    def __init__(self):
        super().__init__("TrendContinuation", PatternType.TREND)

    def get_pattern_definition(self) -> PatternDefinition:
        return PatternDefinition(
            name="Trend Continuation",
            pattern_type=PatternType.TREND,
            description="Established trend continues in same direction for multiple periods",
            mathematical_formula="""
            Let MA_short(t) = moving_average(prices[t-short_window+1:t])
            Let MA_long(t) = moving_average(prices[t-long_window+1:t])
            Let trend_direction(t) = sign(MA_short(t) - MA_long(t))
            Let continuation_window = C

            Pattern exists at time t IF:
            1. |MA_short(t) - MA_long(t)| > trend_strength_threshold
            2. trend_direction(t+k) == trend_direction(t) for ≥80% of k∈[1,C]
            3. Trend strength maintained or increased
            """,
            parameters={
                'short_ma_window': 10,
                'long_ma_window': 50,
                'continuation_window': 20,
                'trend_strength_threshold': 0.005,
                'direction_consistency_rate': 0.8
            },
            frequency_threshold=0.15  # Must occur at least 15% of time
        )

    def discover_pattern(self,
                        prices: pd.Series,
                        short_ma_window: int = 10,
                        long_ma_window: int = 50,
                        continuation_window: int = 20,
                        trend_strength_threshold: float = 0.005,
                        direction_consistency_rate: float = 0.8) -> PatternDiscoveryResult:
        """
        Discover trend continuation patterns.

        Mathematical Definition:
        1. Calculate short and long moving averages
        2. Determine trend direction and strength
        3. Check if trend continues in same direction
        4. Verify trend strength is maintained
        """

        self.logger.info("📈 Discovering trend continuation patterns")

        # Calculate moving averages
        ma_short = prices.rolling(short_ma_window).mean()
        ma_long = prices.rolling(long_ma_window).mean()

        # Calculate trend direction and strength
        trend_diff = (ma_short - ma_long) / ma_long
        trend_direction = np.sign(trend_diff)
        trend_strength = abs(trend_diff)

        labels = []
        magnitudes = []

        for i in range(long_ma_window, len(prices) - continuation_window):
            current_trend_strength = trend_strength.iloc[i]
            current_trend_direction = trend_direction.iloc[i]

            if current_trend_strength > trend_strength_threshold:
                # Check trend continuation
                future_trend_directions = trend_direction.iloc[i+1:i+continuation_window+1]
                future_trend_strengths = trend_strength.iloc[i+1:i+continuation_window+1]

                # Direction consistency
                same_direction_count = (future_trend_directions == current_trend_direction).sum()
                direction_consistency = same_direction_count / len(future_trend_directions)

                # Strength maintenance
                maintained_strength_count = (future_trend_strengths >= current_trend_strength * 0.7).sum()
                strength_maintenance = maintained_strength_count / len(future_trend_strengths)

                pattern_exists = (
                    direction_consistency >= direction_consistency_rate and
                    strength_maintenance >= 0.6
                )

                labels.append(1 if pattern_exists else 0)
                magnitudes.append(current_trend_strength if pattern_exists else 0)
            else:
                labels.append(0)
                magnitudes.append(0)

        # Create result series
        pattern_labels = pd.Series(labels, index=prices.index[long_ma_window:long_ma_window+len(labels)])
        pattern_magnitudes = pd.Series(magnitudes, index=prices.index[long_ma_window:long_ma_window+len(magnitudes)])

        # Calculate statistics
        stats = self._calculate_pattern_statistics(pattern_labels, prices, pattern_magnitudes)

        return PatternDiscoveryResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            noise_ratio=stats['noise_ratio'],
            statistical_significance=stats['statistical_significance']
        )

class FalseBreakoutDiscoverer(BasePatternDiscoverer):
    """Discover false breakout patterns."""

    def __init__(self):
        super().__init__("FalseBreakout", PatternType.BREAKOUT)

    def get_pattern_definition(self) -> PatternDefinition:
        return PatternDefinition(
            name="False Breakout",
            pattern_type=PatternType.BREAKOUT,
            description="Price breaks technical level but quickly reverses back inside range",
            mathematical_formula="""
            Let upper_band(t) = MA(t) + 2*STD(t)
            Let lower_band(t) = MA(t) - 2*STD(t)
            Let reversal_window = R

            Pattern exists at time t IF:
            1. price(t) > upper_band(t) OR price(t) < lower_band(t)
            2. ≥70% of price(t+k) for k∈[1,R] return inside bands
            3. Reversal occurs within R periods
            """,
            parameters={
                'bb_window': 20,
                'bb_std': 2.0,
                'reversal_window': 3,
                'reversal_rate': 0.7
            },
            frequency_threshold=0.03
        )

    def discover_pattern(self,
                        prices: pd.Series,
                        bb_window: int = 20,
                        bb_std: float = 2.0,
                        reversal_window: int = 3,
                        reversal_rate: float = 0.7) -> PatternDiscoveryResult:
        """Discover false breakout patterns."""

        self.logger.info("🔄 Discovering false breakout patterns")

        # Calculate Bollinger Bands
        ma = prices.rolling(bb_window).mean()
        std = prices.rolling(bb_window).std()
        upper_band = ma + bb_std * std
        lower_band = ma - bb_std * std

        labels = []
        magnitudes = []

        for i in range(bb_window, len(prices) - reversal_window):
            current_price = prices.iloc[i]
            current_upper = upper_band.iloc[i]
            current_lower = lower_band.iloc[i]

            # Check for initial breakout
            upper_breakout = current_price > current_upper
            lower_breakout = current_price < current_lower

            if upper_breakout or lower_breakout:
                future_prices = prices.iloc[i+1:i+reversal_window+1]

                if upper_breakout:
                    # Check for reversal back inside bands
                    reversal_count = (future_prices < current_upper).sum()
                    reversal_pct = reversal_count / len(future_prices)
                    reversal_magnitude = (current_price - future_prices.min()) / current_price

                elif lower_breakout:
                    # Check for reversal back inside bands
                    reversal_count = (future_prices > current_lower).sum()
                    reversal_pct = reversal_count / len(future_prices)
                    reversal_magnitude = (future_prices.max() - current_price) / current_price

                pattern_exists = reversal_pct >= reversal_rate
                labels.append(1 if pattern_exists else 0)
                magnitudes.append(reversal_magnitude if pattern_exists else 0)
            else:
                labels.append(0)
                magnitudes.append(0)

        pattern_labels = pd.Series(labels, index=prices.index[bb_window:bb_window+len(labels)])
        pattern_magnitudes = pd.Series(magnitudes, index=prices.index[bb_window:bb_window+len(magnitudes)])

        stats = self._calculate_pattern_statistics(pattern_labels, prices, pattern_magnitudes)

        return PatternDiscoveryResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            noise_ratio=stats['noise_ratio'],
            statistical_significance=stats['statistical_significance']
        )

class GapPatternDiscoverer(BasePatternDiscoverer):
    """Discover gap patterns (price gaps that get filled or persist)."""

    def __init__(self):
        super().__init__("GapPattern", PatternType.BREAKOUT)

    def get_pattern_definition(self) -> PatternDefinition:
        return PatternDefinition(
            name="Gap Pattern",
            pattern_type=PatternType.BREAKOUT,
            description="Price gaps that either get filled or persist as continuation signals",
            mathematical_formula="""
            Let gap(t) = (open(t) - close(t-1)) / close(t-1)
            Let gap_threshold = 0.01
            Let fill_window = F

            Pattern exists at time t IF:
            1. |gap(t)| > gap_threshold
            2. Gap either fills within F periods OR persists with continuation
            """,
            parameters={
                'gap_threshold': 0.01,
                'fill_window': 10,
                'persistence_window': 5
            },
            frequency_threshold=0.02
        )

    def discover_pattern(self,
                        market_data: pd.DataFrame,
                        gap_threshold: float = 0.01,
                        fill_window: int = 10,
                        persistence_window: int = 5) -> PatternDiscoveryResult:
        """Discover gap patterns."""

        self.logger.info("📊 Discovering gap patterns")

        if not all(col in market_data.columns for col in ['open', 'close', 'high', 'low']):
            raise ValueError("Gap pattern requires OHLC data")

        # Calculate gaps
        gaps = (market_data['open'] - market_data['close'].shift(1)) / market_data['close'].shift(1)

        labels = []
        magnitudes = []

        for i in range(1, len(gaps) - fill_window):
            current_gap = gaps.iloc[i]

            if abs(current_gap) > gap_threshold:
                gap_size = abs(current_gap)

                # Check if gap gets filled
                if current_gap > 0:  # Up gap
                    gap_level = market_data['close'].iloc[i-1]
                    future_lows = market_data['low'].iloc[i+1:i+fill_window+1]
                    gap_filled = any(future_lows <= gap_level)
                else:  # Down gap
                    gap_level = market_data['close'].iloc[i-1]
                    future_highs = market_data['high'].iloc[i+1:i+fill_window+1]
                    gap_filled = any(future_highs >= gap_level)

                # Pattern exists for both filled and unfilled significant gaps
                pattern_exists = True  # All significant gaps are patterns

                labels.append(1 if pattern_exists else 0)
                magnitudes.append(gap_size if pattern_exists else 0)
            else:
                labels.append(0)
                magnitudes.append(0)

        pattern_labels = pd.Series(labels, index=market_data.index[1:1+len(labels)])
        pattern_magnitudes = pd.Series(magnitudes, index=market_data.index[1:1+len(magnitudes)])

        stats = self._calculate_pattern_statistics(pattern_labels, market_data['close'], pattern_magnitudes)

        return PatternDiscoveryResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            noise_ratio=stats['noise_ratio'],
            statistical_significance=stats['statistical_significance']
        )

class SidewaysConsolidationDiscoverer(BasePatternDiscoverer):
    """Discover sideways consolidation patterns."""

    def __init__(self):
        super().__init__("SidewaysConsolidation", PatternType.CONSOLIDATION)

    def get_pattern_definition(self) -> PatternDefinition:
        return PatternDefinition(
            name="Sideways Consolidation",
            pattern_type=PatternType.CONSOLIDATION,
            description="Price moves sideways within narrow range for extended period",
            mathematical_formula="""
            Let price_range(t) = (high(t) - low(t)) / close(t)
            Let consolidation_window = C
            Let range_threshold = 0.02

            Pattern exists at time t IF:
            1. price_range(t+k) < range_threshold for ≥80% of k∈[0,C]
            2. |close(t+C) - close(t)| / close(t) < range_threshold
            3. No trend direction for C periods
            """,
            parameters={
                'consolidation_window': 15,
                'range_threshold': 0.02,
                'range_consistency': 0.8
            },
            frequency_threshold=0.1
        )

    def discover_pattern(self,
                        market_data: pd.DataFrame,
                        consolidation_window: int = 15,
                        range_threshold: float = 0.02,
                        range_consistency: float = 0.8) -> PatternDiscoveryResult:
        """Discover sideways consolidation patterns."""

        self.logger.info("↔️ Discovering sideways consolidation patterns")

        if not all(col in market_data.columns for col in ['high', 'low', 'close']):
            raise ValueError("Consolidation pattern requires HLC data")

        # Calculate daily ranges
        daily_ranges = (market_data['high'] - market_data['low']) / market_data['close']

        labels = []
        magnitudes = []

        for i in range(len(market_data) - consolidation_window):
            # Check range consistency over consolidation window
            future_ranges = daily_ranges.iloc[i:i+consolidation_window]
            narrow_range_count = (future_ranges < range_threshold).sum()
            range_consistency_pct = narrow_range_count / len(future_ranges)

            # Check overall price movement
            start_price = market_data['close'].iloc[i]
            end_price = market_data['close'].iloc[i+consolidation_window-1]
            total_movement = abs(end_price - start_price) / start_price

            pattern_exists = (
                range_consistency_pct >= range_consistency and
                total_movement < range_threshold
            )

            labels.append(1 if pattern_exists else 0)
            magnitudes.append(np.mean(future_ranges) if pattern_exists else 0)

        pattern_labels = pd.Series(labels, index=market_data.index[:len(labels)])
        pattern_magnitudes = pd.Series(magnitudes, index=market_data.index[:len(magnitudes)])

        stats = self._calculate_pattern_statistics(pattern_labels, market_data['close'], pattern_magnitudes)

        return PatternDiscoveryResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            noise_ratio=stats['noise_ratio'],
            statistical_significance=stats['statistical_significance']
        )

class VolumeSpikePriceImpactDiscoverer(BasePatternDiscoverer):
    """Discover volume spike patterns and their price impact."""

    def __init__(self):
        super().__init__("VolumeSpikePriceImpact", PatternType.BREAKOUT)

    def get_pattern_definition(self) -> PatternDefinition:
        return PatternDefinition(
            name="Volume Spike Price Impact",
            pattern_type=PatternType.BREAKOUT,
            description="High volume spikes followed by significant price movement",
            mathematical_formula="""
            Let volume_ratio(t) = volume(t) / mean(volume[t-19:t])
            Let volume_threshold = 2.0
            Let impact_window = I

            Pattern exists at time t IF:
            1. volume_ratio(t) > volume_threshold
            2. |price_change(t+1:t+I)| > price_impact_threshold
            3. Volume spike precedes price movement
            """,
            parameters={
                'volume_window': 20,
                'volume_threshold': 2.0,
                'impact_window': 5,
                'price_impact_threshold': 0.015
            },
            frequency_threshold=0.05
        )

    def discover_pattern(self,
                        market_data: pd.DataFrame,
                        volume_window: int = 20,
                        volume_threshold: float = 2.0,
                        impact_window: int = 5,
                        price_impact_threshold: float = 0.015) -> PatternDiscoveryResult:
        """Discover volume spike price impact patterns."""

        self.logger.info("📊 Discovering volume spike price impact patterns")

        if not all(col in market_data.columns for col in ['close', 'volume']):
            raise ValueError("Volume spike pattern requires price and volume data")

        # Calculate volume ratios
        avg_volume = market_data['volume'].rolling(volume_window).mean()
        volume_ratios = market_data['volume'] / avg_volume

        labels = []
        magnitudes = []

        for i in range(volume_window, len(market_data) - impact_window):
            current_volume_ratio = volume_ratios.iloc[i]

            if current_volume_ratio > volume_threshold:
                # Check price impact in following periods
                current_price = market_data['close'].iloc[i]
                future_prices = market_data['close'].iloc[i+1:i+impact_window+1]

                # Calculate maximum price impact
                max_impact = max(
                    abs(future_prices.max() - current_price) / current_price,
                    abs(current_price - future_prices.min()) / current_price
                )

                pattern_exists = max_impact > price_impact_threshold

                labels.append(1 if pattern_exists else 0)
                magnitudes.append(max_impact if pattern_exists else 0)
            else:
                labels.append(0)
                magnitudes.append(0)

        pattern_labels = pd.Series(labels, index=market_data.index[volume_window:volume_window+len(labels)])
        pattern_magnitudes = pd.Series(magnitudes, index=market_data.index[volume_window:volume_window+len(magnitudes)])

        stats = self._calculate_pattern_statistics(pattern_labels, market_data['close'], pattern_magnitudes)

        return PatternDiscoveryResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            noise_ratio=stats['noise_ratio'],
            statistical_significance=stats['statistical_significance']
        )

class SeasonalPatternDiscoverer(BasePatternDiscoverer):
    """Discover seasonal/time-based patterns."""

    def __init__(self):
        super().__init__("SeasonalPattern", PatternType.TREND)

    def get_pattern_definition(self) -> PatternDefinition:
        return PatternDefinition(
            name="Seasonal Pattern",
            pattern_type=PatternType.TREND,
            description="Recurring patterns based on time of day, week, month, or year",
            mathematical_formula="""
            Let time_feature(t) = extract_time_component(t)  # hour, day, month, etc.
            Let return_by_time = group_returns_by(time_feature)

            Pattern exists for time_component IF:
            1. mean_return(time_component) significantly different from overall mean
            2. Consistency across multiple periods
            3. Statistical significance of time effect
            """,
            parameters={
                'min_observations': 30,
                'significance_threshold': 0.05,
                'consistency_threshold': 0.6
            },
            frequency_threshold=0.05
        )

    def discover_pattern(self,
                        market_data: pd.DataFrame,
                        time_component: str = 'hour',
                        min_observations: int = 30,
                        significance_threshold: float = 0.05) -> PatternDiscoveryResult:
        """Discover seasonal patterns."""

        self.logger.info(f"📅 Discovering seasonal patterns for {time_component}")

        if not hasattr(market_data.index, 'hour'):
            # Create datetime index if needed
            market_data = market_data.copy()
            if not isinstance(market_data.index, pd.DatetimeIndex):
                self.logger.warning("Non-datetime index, using sequential dates")
                market_data.index = pd.date_range('2020-01-01', periods=len(market_data), freq='H')

        # Calculate returns
        returns = market_data['close'].pct_change().fillna(0)

        # Extract time component
        if time_component == 'hour':
            time_feature = market_data.index.hour
        elif time_component == 'day_of_week':
            time_feature = market_data.index.dayofweek
        elif time_component == 'month':
            time_feature = market_data.index.month
        else:
            time_feature = market_data.index.hour  # Default

        # Group returns by time component
        time_groups = pd.DataFrame({'returns': returns, 'time_comp': time_feature})
        grouped_returns = time_groups.groupby('time_comp')['returns']

        # Find significant time patterns
        labels = []
        magnitudes = []
        overall_mean = returns.mean()

        for i, (time_val, group_returns) in enumerate(grouped_returns):
            if len(group_returns) >= min_observations:
                # Test if this time period has significantly different returns
                t_stat, p_value = stats.ttest_1samp(group_returns, overall_mean)

                if p_value < significance_threshold:
                    mean_return = group_returns.mean()
                    effect_size = abs(mean_return - overall_mean)

                    # Mark all periods with this time component as pattern periods
                    time_mask = time_feature == time_val
                    pattern_strength = effect_size
                else:
                    time_mask = pd.Series(False, index=market_data.index)
                    pattern_strength = 0
            else:
                time_mask = pd.Series(False, index=market_data.index)
                pattern_strength = 0

            # Add to labels
            for j, is_pattern_time in enumerate(time_mask):
                if j == len(labels):  # Extend labels as needed
                    labels.append(1 if is_pattern_time else 0)
                    magnitudes.append(pattern_strength if is_pattern_time else 0)
                elif is_pattern_time:
                    labels[j] = 1
                    magnitudes[j] = max(magnitudes[j], pattern_strength)

        # Ensure labels match data length
        while len(labels) < len(market_data):
            labels.append(0)
            magnitudes.append(0)

        pattern_labels = pd.Series(labels[:len(market_data)], index=market_data.index)
        pattern_magnitudes = pd.Series(magnitudes[:len(market_data)], index=market_data.index)

        stats = self._calculate_pattern_statistics(pattern_labels, market_data['close'], pattern_magnitudes)

        return PatternDiscoveryResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            noise_ratio=stats['noise_ratio'],
            statistical_significance=stats['statistical_significance']
        )

class ExtremeMovementDiscoverer(BasePatternDiscoverer):
    """Discover extreme price movement patterns."""

    def __init__(self):
        super().__init__("ExtremeMovement", PatternType.VOLATILITY)

    def get_pattern_definition(self) -> PatternDefinition:
        return PatternDefinition(
            name="Extreme Movement",
            pattern_type=PatternType.VOLATILITY,
            description="Extreme price movements beyond normal volatility bounds",
            mathematical_formula="""
            Let return(t) = (close(t) - close(t-1)) / close(t-1)
            Let volatility(t) = std(return[t-19:t])
            Let extreme_threshold = 3.0

            Pattern exists at time t IF:
            1. |return(t)| > extreme_threshold * volatility(t)
            2. Movement is statistically extreme (>99th percentile)
            """,
            parameters={
                'volatility_window': 20,
                'extreme_threshold': 3.0,
                'percentile_threshold': 0.99
            },
            frequency_threshold=0.01
        )

    def discover_pattern(self,
                        prices: pd.Series,
                        volatility_window: int = 20,
                        extreme_threshold: float = 3.0,
                        percentile_threshold: float = 0.99) -> PatternDiscoveryResult:
        """Discover extreme movement patterns."""

        self.logger.info("⚡ Discovering extreme movement patterns")

        returns = prices.pct_change().fillna(0)
        volatility = returns.rolling(volatility_window).std()

        labels = []
        magnitudes = []

        # Calculate rolling percentiles
        abs_returns = abs(returns)
        rolling_percentile = abs_returns.rolling(100).quantile(percentile_threshold)

        for i in range(volatility_window, len(returns)):
            current_return = abs(returns.iloc[i])
            current_volatility = volatility.iloc[i]
            current_percentile_threshold = rolling_percentile.iloc[i]

            # Check both volatility-based and percentile-based criteria
            volatility_extreme = (
                current_volatility > 0 and
                current_return > extreme_threshold * current_volatility
            )

            percentile_extreme = current_return > current_percentile_threshold

            pattern_exists = volatility_extreme or percentile_extreme

            labels.append(1 if pattern_exists else 0)
            magnitudes.append(current_return if pattern_exists else 0)

        pattern_labels = pd.Series(labels, index=prices.index[volatility_window:volatility_window+len(labels)])
        pattern_magnitudes = pd.Series(magnitudes, index=prices.index[volatility_window:volatility_window+len(magnitudes)])

        stats = self._calculate_pattern_statistics(pattern_labels, prices, pattern_magnitudes)

        return PatternDiscoveryResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            noise_ratio=stats['noise_ratio'],
            statistical_significance=stats['statistical_significance']
        )

class PatternDiscoveryOrchestrator:
    """Main orchestrator for pattern discovery and definition."""

    def __init__(self):
        self.logger = system_logger.getChild('PatternDiscoveryOrchestrator')

        # Initialize all pattern discoverers
        self.discoverers = {
            # Original 5 patterns
            'momentum_persistence': MomentumPersistenceDiscoverer(),
            'mean_reversion_speed': MeanReversionSpeedDiscoverer(),
            'volatility_expansion': VolatilityExpansionDiscoverer(),
            'breakout_confirmation': BreakoutConfirmationDiscoverer(),
            'trend_continuation': TrendContinuationDiscoverer(),

            # New patterns
            'false_breakout': FalseBreakoutDiscoverer(),
            'gap_pattern': GapPatternDiscoverer(),
            'sideways_consolidation': SidewaysConsolidationDiscoverer(),
            'volume_spike_impact': VolumeSpikePriceImpactDiscoverer(),
            'seasonal_pattern': SeasonalPatternDiscoverer(),
            'extreme_movement': ExtremeMovementDiscoverer(),

            # Additional sophisticated patterns
            'momentum_exhaustion': MomentumExhaustionDiscoverer(),
            'volatility_clustering': VolatilityClusteringDiscoverer(),
            'support_resistance_test': SupportResistanceTestDiscoverer(),
            'volume_climax': VolumeClimaxDiscoverer(),
            'price_rejection': PriceRejectionDiscoverer(),
            'accumulation_distribution': AccumulationDistributionDiscoverer(),
            'divergence_pattern': DivergencePatternDiscoverer(),
            'exhaustion_gap': ExhaustionGapDiscoverer()
        }

    def discover_all_patterns(self, prices: pd.Series) -> Dict[str, PatternDiscoveryResult]:
        """
        Discover all patterns in price data.

        Args:
            prices: Price series (typically close prices)

        Returns:
            Dictionary of pattern discovery results
        """

        self.logger.info(f"🔍 Starting comprehensive pattern discovery on {len(prices)} price points")

        results = {}

        for pattern_name, discoverer in self.discoverers.items():
            self.logger.info(f"📊 Discovering {pattern_name}")

            try:
                result = discoverer.discover_pattern(prices)
                results[pattern_name] = result

                # Log key statistics
                self.logger.info(f"   Frequency: {result.frequency:.3f}")
                self.logger.info(f"   Predictability: {result.predictability_score:.3f}")
                self.logger.info(f"   Valid Pattern: {'✅' if result.is_valid_pattern else '❌'}")

            except Exception as e:
                self.logger.error(f"   Failed to discover {pattern_name}: {e}")
                continue

        self.logger.info(f"✅ Pattern discovery completed: {len(results)} patterns analyzed")
        return results

    def get_pattern_definitions(self) -> Dict[str, PatternDefinition]:
        """Get mathematical definitions of all patterns."""

        definitions = {}

        for pattern_name, discoverer in self.discoverers.items():
            definitions[pattern_name] = discoverer.get_pattern_definition()

        return definitions

    def generate_pattern_report(self, discovery_results: Dict[str, PatternDiscoveryResult]) -> str:
        """Generate comprehensive pattern discovery report."""

        report = []
        report.append("# Mathematical Pattern Discovery Report")
        report.append("=" * 60)
        report.append("")

        # Summary
        total_patterns = len(discovery_results)
        valid_patterns = sum(1 for result in discovery_results.values() if result.is_valid_pattern)

        report.append("## Executive Summary")
        report.append("")
        report.append(f"- **Total Patterns Analyzed**: {total_patterns}")
        report.append(f"- **Valid Patterns Found**: {valid_patterns}")
        report.append(f"- **Pattern Validity Rate**: {valid_patterns/total_patterns*100:.1f}%")
        report.append("")

        # Pattern Analysis
        report.append("## Pattern Analysis Results")
        report.append("")

        # Sort by validity and frequency
        sorted_results = sorted(
            discovery_results.items(),
            key=lambda x: (x[1].is_valid_pattern, x[1].frequency),
            reverse=True
        )

        for pattern_name, result in sorted_results:
            status = "✅ VALID" if result.is_valid_pattern else "❌ INVALID"

            report.append(f"### {pattern_name.replace('_', ' ').title()} {status}")
            report.append("")

            # Mathematical Definition
            report.append("**Mathematical Definition:**")
            report.append(f"```")
            report.append(result.definition.mathematical_formula.strip())
            report.append(f"```")
            report.append("")

            # Key Statistics
            report.append("**Key Statistics:**")
            report.append(f"- Frequency: {result.frequency:.3f} ({result.frequency*100:.1f}% of periods)")
            report.append(f"- Predictability Score: {result.predictability_score:.3f}")
            report.append(f"- Signal-to-Noise Ratio: {1-result.noise_ratio:.3f}")

            if result.statistical_significance.get('p_value'):
                p_val = result.statistical_significance['p_value']
                report.append(f"- Statistical Significance: p={p_val:.3f}")

            report.append("")

            # Duration Statistics
            if result.duration_stats['mean'] > 0:
                report.append("**Pattern Duration:**")
                report.append(f"- Average: {result.duration_stats['mean']:.1f} periods")
                report.append(f"- Range: {result.duration_stats['min']:.0f}-{result.duration_stats['max']:.0f} periods")
                report.append("")

            # Magnitude Statistics
            if result.magnitude_stats and result.magnitude_stats.get('mean', 0) > 0:
                report.append("**Pattern Magnitude:**")
                report.append(f"- Average: {result.magnitude_stats['mean']:.4f}")
                report.append(f"- Range: {result.magnitude_stats['min']:.4f}-{result.magnitude_stats['max']:.4f}")
                report.append("")

        # Recommendations
        report.append("## Recommendations")
        report.append("")

        valid_pattern_names = [name for name, result in discovery_results.items() if result.is_valid_pattern]

        if valid_pattern_names:
            report.append("✅ **Valid Patterns for ML Training:**")
            for pattern_name in valid_pattern_names:
                result = discovery_results[pattern_name]
                report.append(f"- {pattern_name}: {result.frequency:.3f} frequency, {result.predictability_score:.3f} predictability")
            report.append("")
            report.append("**Next Steps:**")
            report.append("1. Use these patterns as supervised learning targets")
            report.append("2. Analyze which market dimensions predict each pattern")
            report.append("3. Validate economic significance through backtesting")
        else:
            report.append("❌ **No Valid Patterns Found**")
            report.append("**Recommendations:**")
            report.append("1. Adjust pattern parameters to increase sensitivity")
            report.append("2. Try different timeframes or market conditions")
            report.append("3. Consider alternative pattern definitions")

        return "\n".join(report)

    def export_pattern_labels(self, discovery_results: Dict[str, PatternDiscoveryResult]) -> pd.DataFrame:
        """Export pattern labels as ML-ready DataFrame."""

        pattern_labels = {}

        for pattern_name, result in discovery_results.items():
            if result.is_valid_pattern:
                pattern_labels[pattern_name] = result.labels

        if pattern_labels:
            # Combine all pattern labels
            labels_df = pd.DataFrame(pattern_labels)

            # Add composite patterns
            if len(labels_df.columns) > 1:
                labels_df['any_momentum'] = labels_df[[col for col in labels_df.columns if 'momentum' in col]].max(axis=1)
                labels_df['any_reversion'] = labels_df[[col for col in labels_df.columns if 'reversion' in col]].max(axis=1)
                labels_df['any_volatility'] = labels_df[[col for col in labels_df.columns if 'volatility' in col]].max(axis=1)
                labels_df['any_trend'] = labels_df[[col for col in labels_df.columns if 'trend' in col]].max(axis=1)
                labels_df['any_breakout'] = labels_df[[col for col in labels_df.columns if 'breakout' in col]].max(axis=1)

            return labels_df
        else:
            return pd.DataFrame()

# Example usage
def run_pattern_discovery_example():
    """Example of how to use the pattern discovery framework."""

    print("Mathematical Pattern Discovery Framework")
    print("======================================")
    print()
    print("This framework provides:")
    print("1. Precise mathematical definitions of price patterns")
    print("2. Automated pattern discovery in price data")
    print("3. Statistical validation of pattern significance")
    print("4. ML-ready binary labels for supervised learning")
    print()
    print("Available patterns:")
    print("- Momentum Persistence: Momentum continues with gradual decay")
    print("- Mean Reversion Speed: Price reverts to moving average")
    print("- Volatility Expansion: Low vol followed by high vol")
    print("- Confirmed Breakout: Price breaks level and continues")
    print("- Trend Continuation: Established trend persists")
    print()
    print("Usage:")
    print("```python")
    print("orchestrator = PatternDiscoveryOrchestrator()")
    print("results = orchestrator.discover_all_patterns(price_series)")
    print("report = orchestrator.generate_pattern_report(results)")
    print("labels_df = orchestrator.export_pattern_labels(results)")
    print("```")

if __name__ == "__main__":
    run_pattern_discovery_example()
