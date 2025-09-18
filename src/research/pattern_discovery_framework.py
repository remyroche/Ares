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


class BasePatternDiscoverer(ABC):
    """Base class for pattern discovery."""
    
    def __init__(self, name: str, pattern_type: PatternType):
        self.name = name
        self.pattern_type = pattern_type
        self.logger = system_logger.getChild(f'PatternDiscoverer_{name}')
    
    @abstractmethod
    def discover_pattern(self, prices: pd.Series, **kwargs) -> PatternDiscoveryResult:
        """Discover and define pattern in price data."""
        pass
    
    @abstractmethod
    def get_pattern_definition(self) -> PatternDefinition:
        """Get mathematical definition of the pattern."""
        pass
    
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


class PatternDiscoveryOrchestrator:
    """Main orchestrator for pattern discovery and definition."""
    
    def __init__(self):
        self.logger = system_logger.getChild('PatternDiscoveryOrchestrator')
        
        # Initialize all pattern discoverers
        self.discoverers = {
            'momentum_persistence': MomentumPersistenceDiscoverer(),
            'mean_reversion_speed': MeanReversionSpeedDiscoverer(),
            'volatility_expansion': VolatilityExpansionDiscoverer(),
            'breakout_confirmation': BreakoutConfirmationDiscoverer(),
            'trend_continuation': TrendContinuationDiscoverer()
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