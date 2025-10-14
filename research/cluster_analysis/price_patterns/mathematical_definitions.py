"""
Core Pure Price Action Pattern Definitions

This module contains the fundamental price action patterns defined mathematically
using only price movements. No volume, fundamentals, or market structure.

Focus: WHAT price does, not WHY it moves.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from abc import ABC, abstractmethod

from src.utils.logger import system_logger


class PurePatternType(Enum):
    """Pure price action pattern categories."""
    MOMENTUM = "momentum"
    REVERSION = "reversion"
    TREND = "trend"
    RANGE = "range"
    VOLATILITY = "volatility"
    ACCELERATION = "acceleration"


@dataclass
class PurePricePattern:
    """Mathematical definition of a pure price action pattern."""
    name: str
    pattern_type: PurePatternType
    description: str
    mathematical_formula: str
    parameters: Dict[str, Any]
    frequency_threshold: float
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}\nFormula: {self.mathematical_formula}"


@dataclass
class PurePatternResult:
    """Result of pure price action pattern discovery."""
    definition: PurePricePattern
    labels: pd.Series  # Binary: 1 = pattern exists, 0 = no pattern
    intensity: pd.Series  # Gradient: strength/intensity of pattern (0-1)
    frequency: float
    duration_stats: Dict[str, float]
    magnitude_stats: Dict[str, float]
    predictability_score: float
    statistical_significance: Dict[str, float]
    
    @property
    def is_valid_pattern(self) -> bool:
        """Check if pattern meets validity criteria."""
        return (
            self.frequency >= self.definition.frequency_threshold and
            self.predictability_score > 0.1 and
            self.statistical_significance.get('p_value', 1.0) < 0.05
        )


class BasePurePricePatternDiscoverer(ABC):
    """Base class for pure price action pattern discovery."""
    
    def __init__(self, name: str, pattern_type: PurePatternType):
        self.name = name
        self.pattern_type = pattern_type
        self.logger = system_logger.getChild(f'PurePattern_{name}')
    
    @abstractmethod
    def discover_pattern(self, prices: pd.Series, **kwargs) -> PurePatternResult:
        """Discover pattern in pure price data."""
        pass
    
    @abstractmethod
    def get_pattern_definition(self) -> PurePricePattern:
        """Get mathematical definition of the pattern."""
        pass
    
    def _calculate_pattern_statistics(self, 
                                    labels: pd.Series, 
                                    intensity: pd.Series,
                                    prices: pd.Series) -> Dict[str, Any]:
        """Calculate pattern statistics."""
        
        frequency = labels.sum() / len(labels)
        duration_stats = self._calculate_durations(labels)
        magnitude_stats = self._calculate_magnitudes(labels, prices)
        predictability = self._calculate_predictability(labels)
        significance = self._calculate_significance(labels, prices)
        
        return {
            'frequency': frequency,
            'duration_stats': duration_stats,
            'magnitude_stats': magnitude_stats,
            'predictability_score': predictability,
            'statistical_significance': significance
        }
    
    def _calculate_durations(self, labels: pd.Series) -> Dict[str, float]:
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
    
    def _calculate_magnitudes(self, labels: pd.Series, prices: pd.Series) -> Dict[str, float]:
        """Calculate price movement magnitudes during patterns."""
        pattern_returns = []
        
        returns = prices.pct_change().fillna(0)
        aligned_returns = returns.loc[labels.index] if not labels.empty else pd.Series()
        
        for i, label in enumerate(labels):
            if label == 1 and i < len(aligned_returns):
                pattern_returns.append(abs(aligned_returns.iloc[i]))
        
        if not pattern_returns:
            return {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}
        
        return {
            'mean': float(np.mean(pattern_returns)),
            'median': float(np.median(pattern_returns)),
            'std': float(np.std(pattern_returns)),
            'min': float(np.min(pattern_returns)),
            'max': float(np.max(pattern_returns))
        }
    
    def _calculate_predictability(self, labels: pd.Series) -> float:
        """Calculate pattern predictability using entropy."""
        if len(labels) == 0:
            return 0.0
        
        pattern_freq = labels.sum() / len(labels)
        
        if pattern_freq == 0 or pattern_freq == 1:
            return 1.0
        
        entropy = -pattern_freq * np.log2(pattern_freq) - (1 - pattern_freq) * np.log2(1 - pattern_freq)
        return float(1.0 - entropy)
    
    def _calculate_significance(self, labels: pd.Series, prices: pd.Series) -> Dict[str, float]:
        """Calculate statistical significance."""
        if labels.sum() == 0:
            return {'p_value': 1.0, 't_statistic': 0.0}
        
        returns = prices.pct_change().fillna(0)
        aligned_returns = returns.loc[labels.index] if not labels.empty else pd.Series()
        
        if len(aligned_returns) == 0:
            return {'p_value': 1.0, 't_statistic': 0.0}
        
        pattern_returns = aligned_returns[labels == 1]
        no_pattern_returns = aligned_returns[labels == 0]
        
        if len(pattern_returns) < 5 or len(no_pattern_returns) < 5:
            return {'p_value': 1.0, 't_statistic': 0.0}
        
        try:
            t_stat, p_value = stats.ttest_ind(pattern_returns, no_pattern_returns)
            return {'p_value': float(p_value), 't_statistic': float(t_stat)}
        except:
            return {'p_value': 1.0, 't_statistic': 0.0}


class MomentumPersistencePattern(BasePurePricePatternDiscoverer):
    """Pure momentum persistence - price momentum continues with gradual decay."""
    
    def __init__(self):
        super().__init__("MomentumPersistence", PurePatternType.MOMENTUM)
    
    def get_pattern_definition(self) -> PurePricePattern:
        return PurePricePattern(
            name="Momentum Persistence",
            pattern_type=PurePatternType.MOMENTUM,
            description="Price momentum continues in same direction with gradual decay",
            mathematical_formula="""
            Let momentum(t) = (price(t) - price(t-window)) / price(t-window)
            Let persistence_window = P
            
            Pattern exists at time t IF:
            1. |momentum(t)| > momentum_threshold
            2. sign(momentum(t+k)) == sign(momentum(t)) for ≥70% of k ∈ [1,P]
            3. |momentum(t+k)| > 0.3 * |momentum(t)| for ≥60% of k ∈ [1,P]
            
            Intensity = |momentum(t)| * persistence_rate * decay_quality
            """,
            parameters={
                'momentum_window': 5,
                'persistence_window': 10,
                'momentum_threshold': 0.01,
                'direction_persistence_rate': 0.7,
                'magnitude_decay_rate': 0.6
            },
            frequency_threshold=0.05
        )
    
    def discover_pattern(self,
                        prices: pd.Series,
                        momentum_window: int = 5,
                        persistence_window: int = 10,
                        momentum_threshold: float = 0.01,
                        direction_persistence_rate: float = 0.7,
                        magnitude_decay_rate: float = 0.6) -> PurePatternResult:
        """Discover pure momentum persistence patterns."""
        
        self.logger.info("🚀 Discovering pure momentum persistence patterns")
        
        # Calculate price momentum
        momentum = (prices - prices.shift(momentum_window)) / prices.shift(momentum_window)
        momentum = momentum.fillna(0)
        
        labels = []
        intensities = []
        
        for i in range(len(momentum) - persistence_window):
            current_momentum = momentum.iloc[i]
            
            if abs(current_momentum) > momentum_threshold:
                future_momentum = momentum.iloc[i+1:i+persistence_window+1]
                
                # Direction persistence
                same_direction = (np.sign(future_momentum) == np.sign(current_momentum))
                direction_persistence = same_direction.sum() / len(future_momentum)
                
                # Magnitude decay quality
                magnitude_ratios = abs(future_momentum) / abs(current_momentum)
                gradual_decay = (magnitude_ratios > 0.3).sum() / len(magnitude_ratios)
                
                pattern_exists = (
                    direction_persistence >= direction_persistence_rate and
                    gradual_decay >= magnitude_decay_rate
                )
                
                # Calculate intensity (strength of pattern)
                if pattern_exists:
                    intensity = abs(current_momentum) * direction_persistence * gradual_decay
                    intensity = min(intensity * 10, 1.0)  # Scale to 0-1
                else:
                    intensity = 0.0
                
                labels.append(1 if pattern_exists else 0)
                intensities.append(intensity)
            else:
                labels.append(0)
                intensities.append(0.0)
        
        pattern_labels = pd.Series(labels, index=prices.index[:len(labels)])
        pattern_intensities = pd.Series(intensities, index=prices.index[:len(intensities)])
        
        stats = self._calculate_pattern_statistics(pattern_labels, pattern_intensities, prices)
        
        return PurePatternResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            intensity=pattern_intensities,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            statistical_significance=stats['statistical_significance']
        )


class PriceReversionPattern(BasePurePricePatternDiscoverer):
    """Pure price reversion - price returns to previous level."""
    
    def __init__(self):
        super().__init__("PriceReversion", PurePatternType.REVERSION)
    
    def get_pattern_definition(self) -> PurePricePattern:
        return PurePricePattern(
            name="Price Reversion",
            pattern_type=PurePatternType.REVERSION,
            description="Price moves away from level then returns to that level",
            mathematical_formula="""
            Let reference_level = price(t-lookback)
            Let deviation(t) = |price(t) - reference_level| / reference_level
            Let reversion_window = R
            
            Pattern exists at time t IF:
            1. deviation(t) > deviation_threshold
            2. ∃k ∈ [1,R]: |price(t+k) - reference_level| < 0.5 * deviation(t)
            
            Intensity = deviation(t) * reversion_speed * reversion_magnitude
            """,
            parameters={
                'lookback_window': 20,
                'deviation_threshold': 0.03,
                'reversion_window': 15,
                'reversion_ratio': 0.5
            },
            frequency_threshold=0.08
        )
    
    def discover_pattern(self,
                        prices: pd.Series,
                        lookback_window: int = 20,
                        deviation_threshold: float = 0.03,
                        reversion_window: int = 15,
                        reversion_ratio: float = 0.5) -> PurePatternResult:
        """Discover pure price reversion patterns."""
        
        self.logger.info("🔄 Discovering pure price reversion patterns")
        
        labels = []
        intensities = []
        
        for i in range(lookback_window, len(prices) - reversion_window):
            reference_level = prices.iloc[i - lookback_window]
            current_price = prices.iloc[i]
            
            # Calculate deviation from reference level
            deviation = abs(current_price - reference_level) / reference_level
            
            if deviation > deviation_threshold:
                # Look for reversion
                future_prices = prices.iloc[i+1:i+reversion_window+1]
                
                reversion_occurred = False
                reversion_speed = 0
                reversion_magnitude = 0
                
                for j, future_price in enumerate(future_prices):
                    future_deviation = abs(future_price - reference_level) / reference_level
                    if future_deviation < reversion_ratio * deviation:
                        reversion_occurred = True
                        reversion_speed = 1.0 / (j + 1)  # Faster = higher score
                        reversion_magnitude = (deviation - future_deviation) / deviation
                        break
                
                if reversion_occurred:
                    # Calculate intensity based on deviation, speed, and magnitude
                    intensity = deviation * reversion_speed * reversion_magnitude
                    intensity = min(intensity * 5, 1.0)  # Scale to 0-1
                else:
                    intensity = 0.0
                
                labels.append(1 if reversion_occurred else 0)
                intensities.append(intensity)
            else:
                labels.append(0)
                intensities.append(0.0)
        
        pattern_labels = pd.Series(labels, index=prices.index[lookback_window:lookback_window+len(labels)])
        pattern_intensities = pd.Series(intensities, index=prices.index[lookback_window:lookback_window+len(intensities)])
        
        stats = self._calculate_pattern_statistics(pattern_labels, pattern_intensities, prices)
        
        return PurePatternResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            intensity=pattern_intensities,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            statistical_significance=stats['statistical_significance']
        )


class TrendAccelerationPattern(BasePurePricePatternDiscoverer):
    """Pure trend acceleration - price movement speeds up."""
    
    def __init__(self):
        super().__init__("TrendAcceleration", PurePatternType.ACCELERATION)
    
    def get_pattern_definition(self) -> PurePricePattern:
        return PurePricePattern(
            name="Trend Acceleration",
            pattern_type=PurePatternType.ACCELERATION,
            description="Price movement accelerates (rate of change increases)",
            mathematical_formula="""
            Let velocity(t) = (price(t) - price(t-1)) / price(t-1)
            Let acceleration(t) = velocity(t) - velocity(t-1)
            Let acceleration_window = A
            
            Pattern exists at time t IF:
            1. acceleration(t) and velocity(t) same sign
            2. |acceleration(t+k)| > |acceleration(t)| for ≥60% of k ∈ [1,A]
            3. velocity maintains direction throughout
            
            Intensity = |acceleration(t)| * consistency * velocity_alignment
            """,
            parameters={
                'acceleration_window': 8,
                'min_acceleration': 0.001,
                'acceleration_consistency': 0.6,
                'velocity_consistency': 0.8
            },
            frequency_threshold=0.06
        )
    
    def discover_pattern(self,
                        prices: pd.Series,
                        acceleration_window: int = 8,
                        min_acceleration: float = 0.001,
                        acceleration_consistency: float = 0.6,
                        velocity_consistency: float = 0.8) -> PurePatternResult:
        """Discover pure trend acceleration patterns."""
        
        self.logger.info("⚡ Discovering pure trend acceleration patterns")
        
        velocity = prices.pct_change().fillna(0)
        acceleration = velocity.diff().fillna(0)
        
        labels = []
        intensities = []
        
        for i in range(2, len(acceleration) - acceleration_window):
            current_velocity = velocity.iloc[i]
            current_acceleration = acceleration.iloc[i]
            
            if (abs(current_acceleration) > min_acceleration and
                np.sign(current_acceleration) == np.sign(current_velocity) and
                current_velocity != 0):
                
                future_acceleration = acceleration.iloc[i+1:i+acceleration_window+1]
                future_velocity = velocity.iloc[i+1:i+acceleration_window+1]
                
                # Check acceleration consistency
                increasing_acceleration = (
                    abs(future_acceleration) > abs(current_acceleration)
                ).sum() / len(future_acceleration)
                
                # Check velocity direction consistency
                velocity_direction_maintained = (
                    np.sign(future_velocity) == np.sign(current_velocity)
                ).sum() / len(future_velocity)
                
                pattern_exists = (
                    increasing_acceleration >= acceleration_consistency and
                    velocity_direction_maintained >= velocity_consistency
                )
                
                if pattern_exists:
                    # Calculate intensity
                    intensity = (
                        abs(current_acceleration) * 1000 *  # Scale acceleration
                        increasing_acceleration * 
                        velocity_direction_maintained
                    )
                    intensity = min(intensity, 1.0)
                else:
                    intensity = 0.0
                
                labels.append(1 if pattern_exists else 0)
                intensities.append(intensity)
            else:
                labels.append(0)
                intensities.append(0.0)
        
        start_idx = 2
        pattern_labels = pd.Series(labels, index=prices.index[start_idx:start_idx+len(labels)])
        pattern_intensities = pd.Series(intensities, index=prices.index[start_idx:start_idx+len(intensities)])
        
        stats = self._calculate_pattern_statistics(pattern_labels, pattern_intensities, prices)
        
        return PurePatternResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            intensity=pattern_intensities,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            statistical_significance=stats['statistical_significance']
        )


class PriceRangeBreakoutPattern(BasePurePricePatternDiscoverer):
    """Pure range breakout - price breaks established range and continues."""
    
    def __init__(self):
        super().__init__("PriceRangeBreakout", PurePatternType.RANGE)
    
    def get_pattern_definition(self) -> PurePricePattern:
        return PurePricePattern(
            name="Price Range Breakout",
            pattern_type=PurePatternType.RANGE,
            description="Price breaks out of established trading range and continues",
            mathematical_formula="""
            Let range_high = max(prices[t-range_window:t])
            Let range_low = min(prices[t-range_window:t])
            Let range_size = (range_high - range_low) / range_low
            Let continuation_window = C
            
            Pattern exists at time t IF:
            1. range_size < max_range_threshold (established range)
            2. price(t) > range_high OR price(t) < range_low (breakout)
            3. price continues beyond range for ≥60% of next C periods
            
            Intensity = breakout_magnitude * continuation_strength * range_quality
            """,
            parameters={
                'range_window': 30,
                'max_range_threshold': 0.08,
                'continuation_window': 8,
                'continuation_rate': 0.6,
                'minimum_breakout': 0.01
            },
            frequency_threshold=0.04
        )
    
    def discover_pattern(self,
                        prices: pd.Series,
                        range_window: int = 30,
                        max_range_threshold: float = 0.08,
                        continuation_window: int = 8,
                        continuation_rate: float = 0.6,
                        minimum_breakout: float = 0.01) -> PurePatternResult:
        """Discover pure price range breakout patterns."""
        
        self.logger.info("📊 Discovering pure price range breakout patterns")
        
        labels = []
        intensities = []
        
        for i in range(range_window, len(prices) - continuation_window):
            # Define range
            recent_prices = prices.iloc[i-range_window:i]
            range_high = recent_prices.max()
            range_low = recent_prices.min()
            range_size = (range_high - range_low) / range_low
            
            current_price = prices.iloc[i]
            
            # Check for established range
            if range_size < max_range_threshold:
                # Check for breakout
                upper_breakout = current_price > range_high
                lower_breakout = current_price < range_low
                
                if upper_breakout or lower_breakout:
                    # Calculate breakout magnitude
                    if upper_breakout:
                        breakout_magnitude = (current_price - range_high) / range_high
                    else:
                        breakout_magnitude = (range_low - current_price) / range_low
                    
                    if breakout_magnitude > minimum_breakout:
                        # Check continuation
                        future_prices = prices.iloc[i+1:i+continuation_window+1]
                        
                        if upper_breakout:
                            continuation_count = (future_prices > range_high).sum()
                        else:
                            continuation_count = (future_prices < range_low).sum()
                        
                        continuation_strength = continuation_count / len(future_prices)
                        pattern_exists = continuation_strength >= continuation_rate
                        
                        if pattern_exists:
                            # Calculate intensity
                            range_quality = 1.0 - (range_size / max_range_threshold)  # Tighter range = higher quality
                            intensity = breakout_magnitude * continuation_strength * range_quality
                            intensity = min(intensity * 5, 1.0)  # Scale to 0-1
                        else:
                            intensity = 0.0
                        
                        labels.append(1 if pattern_exists else 0)
                        intensities.append(intensity)
                    else:
                        labels.append(0)
                        intensities.append(0.0)
                else:
                    labels.append(0)
                    intensities.append(0.0)
            else:
                labels.append(0)
                intensities.append(0.0)
        
        start_idx = range_window
        pattern_labels = pd.Series(labels, index=prices.index[start_idx:start_idx+len(labels)])
        pattern_intensities = pd.Series(intensities, index=prices.index[start_idx:start_idx+len(intensities)])
        
        stats = self._calculate_pattern_statistics(pattern_labels, pattern_intensities, prices)
        
        return PurePatternResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            intensity=pattern_intensities,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            statistical_significance=stats['statistical_significance']
        )


class PriceExtremeReversalPattern(BasePurePricePatternDiscoverer):
    """Pure extreme reversal - large price moves followed by reversal."""
    
    def __init__(self):
        super().__init__("PriceExtremeReversal", PurePatternType.REVERSION)
    
    def get_pattern_definition(self) -> PurePricePattern:
        return PurePricePattern(
            name="Price Extreme Reversal",
            pattern_type=PurePatternType.REVERSION,
            description="Extreme price movement followed by reversal in opposite direction",
            mathematical_formula="""
            Let return(t) = (price(t) - price(t-1)) / price(t-1)
            Let recent_volatility(t) = std(returns[t-vol_window:t])
            Let extreme_threshold = 2.5
            Let reversal_window = R
            
            Pattern exists at time t IF:
            1. |return(t)| > extreme_threshold * recent_volatility(t)
            2. return(t+1:t+R) moves in opposite direction
            3. Reversal magnitude > min_reversal_ratio * |return(t)|
            
            Intensity = extreme_magnitude * reversal_strength * speed
            """,
            parameters={
                'vol_window': 20,
                'extreme_threshold': 2.5,
                'reversal_window': 8,
                'min_reversal_ratio': 0.4
            },
            frequency_threshold=0.02
        )
    
    def discover_pattern(self,
                        prices: pd.Series,
                        vol_window: int = 20,
                        extreme_threshold: float = 2.5,
                        reversal_window: int = 8,
                        min_reversal_ratio: float = 0.4) -> PurePatternResult:
        """Discover pure extreme reversal patterns."""
        
        self.logger.info("⚡ Discovering pure extreme reversal patterns")
        
        returns = prices.pct_change().fillna(0)
        volatility = returns.rolling(vol_window).std()
        
        labels = []
        intensities = []
        
        for i in range(vol_window, len(returns) - reversal_window):
            current_return = returns.iloc[i]
            current_volatility = volatility.iloc[i]
            
            # Check for extreme movement
            if current_volatility > 0 and abs(current_return) > extreme_threshold * current_volatility:
                
                # Look for reversal
                future_returns = returns.iloc[i+1:i+reversal_window+1]
                cumulative_reversal = future_returns.sum()
                
                # Check opposite direction and magnitude
                opposite_direction = (
                    (current_return > 0 and cumulative_reversal < 0) or
                    (current_return < 0 and cumulative_reversal > 0)
                )
                
                if opposite_direction:
                    reversal_ratio = abs(cumulative_reversal) / abs(current_return)
                    sufficient_reversal = reversal_ratio >= min_reversal_ratio
                    
                    if sufficient_reversal:
                        # Calculate intensity
                        extreme_magnitude = abs(current_return) / current_volatility
                        reversal_strength = reversal_ratio
                        speed = 1.0 - (np.argmax(np.abs(future_returns)) / len(future_returns))  # Earlier reversal = higher score
                        
                        intensity = (extreme_magnitude / 5.0) * reversal_strength * speed
                        intensity = min(intensity, 1.0)
                    else:
                        intensity = 0.0
                    
                    pattern_exists = sufficient_reversal
                else:
                    pattern_exists = False
                    intensity = 0.0
                
                labels.append(1 if pattern_exists else 0)
                intensities.append(intensity)
            else:
                labels.append(0)
                intensities.append(0.0)
        
        start_idx = vol_window
        pattern_labels = pd.Series(labels, index=prices.index[start_idx:start_idx+len(labels)])
        pattern_intensities = pd.Series(intensities, index=prices.index[start_idx:start_idx+len(intensities)])
        
        stats = self._calculate_pattern_statistics(pattern_labels, pattern_intensities, prices)
        
        return PurePatternResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            intensity=pattern_intensities,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            statistical_significance=stats['statistical_significance']
        )


class PurePricePatternOrchestrator:
    """Main orchestrator for pure price action pattern discovery."""
    
    def __init__(self):
        self.logger = system_logger.getChild('PurePricePatternOrchestrator')
        
        # Initialize pure price action pattern discoverers
        self.pure_discoverers = {
            'momentum_persistence': MomentumPersistencePattern(),
            'price_reversion': PriceReversionPattern(),
            'trend_acceleration': TrendAccelerationPattern(),
            'range_breakout': PriceRangeBreakoutPattern(),
            'extreme_reversal': PriceExtremeReversalPattern()
        }
    
    def discover_all_pure_patterns(self, prices: pd.Series) -> Dict[str, PurePatternResult]:
        """Discover all pure price action patterns."""
        
        self.logger.info(f"🎯 Discovering pure price action patterns in {len(prices)} price points")
        
        results = {}
        
        for pattern_name, discoverer in self.pure_discoverers.items():
            self.logger.info(f"📊 Discovering {pattern_name}")
            
            try:
                result = discoverer.discover_pattern(prices)
                results[pattern_name] = result
                
                status = "✅ VALID" if result.is_valid_pattern else "❌ INVALID"
                self.logger.info(f"   {status} - Frequency: {result.frequency:.3f}")
                
            except Exception as e:
                self.logger.error(f"   ❌ Failed: {e}")
                continue
        
        valid_count = sum(1 for result in results.values() if result.is_valid_pattern)
        self.logger.info(f"🎯 Pure pattern discovery completed: {valid_count}/{len(results)} valid patterns")
        
        return results
    
    def export_binary_labels(self, results: Dict[str, PurePatternResult]) -> pd.DataFrame:
        """Export binary pattern labels for ML."""
        
        pattern_labels = {}
        
        for pattern_name, result in results.items():
            if result.is_valid_pattern:
                pattern_labels[pattern_name] = result.labels
        
        return pd.DataFrame(pattern_labels) if pattern_labels else pd.DataFrame()
    
    def export_intensity_gradients(self, results: Dict[str, PurePatternResult]) -> pd.DataFrame:
        """Export pattern intensity gradients for ML."""
        
        pattern_intensities = {}
        
        for pattern_name, result in results.items():
            if result.is_valid_pattern:
                pattern_intensities[f"{pattern_name}_intensity"] = result.intensity
        
        return pd.DataFrame(pattern_intensities) if pattern_intensities else pd.DataFrame()
    
    def export_combined_targets(self, results: Dict[str, PurePatternResult]) -> pd.DataFrame:
        """Export both binary labels and intensity gradients."""
        
        binary_labels = self.export_binary_labels(results)
        intensity_gradients = self.export_intensity_gradients(results)
        
        if not binary_labels.empty and not intensity_gradients.empty:
            return pd.concat([binary_labels, intensity_gradients], axis=1)
        elif not binary_labels.empty:
            return binary_labels
        elif not intensity_gradients.empty:
            return intensity_gradients
        else:
            return pd.DataFrame()


# Example usage
def run_core_patterns_example():
    """Example of core pure price action pattern discovery."""
    
    print("Core Pure Price Action Patterns")
    print("==============================")
    print()
    print("🎯 PURE PRICE ACTION FOCUS:")
    print("   - Only price movements (WHAT price does)")
    print("   - Mathematical precision (exact formulas)")
    print("   - Binary labels + intensity gradients")
    print("   - No volume/fundamentals/market structure")
    print()
    print("Core patterns:")
    print("1. Momentum Persistence - Price momentum continues")
    print("2. Price Reversion - Price returns to levels")
    print("3. Trend Acceleration - Price movement speeds up")
    print("4. Range Breakout - Price breaks ranges")
    print("5. Extreme Reversal - Large moves + reversal")
    print()
    print("Output formats:")
    print("- Binary labels: [0,1,0,1,0,...]")
    print("- Intensity gradients: [0.0,0.8,0.0,0.6,0.0,...]")
    print("- Combined ML targets for regression/classification")


if __name__ == "__main__":
    run_core_patterns_example()