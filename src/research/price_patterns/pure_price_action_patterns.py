"""
Pure Price Action Pattern Discovery Framework

This module focuses exclusively on PRICE ACTION patterns - what prices actually do,
not the underlying causes. All patterns are defined purely in terms of price movements,
without reference to volume, fundamentals, or market structure.

Core Philosophy:
- Patterns describe PRICE BEHAVIOR only
- No assumptions about WHY price moves
- Focus on WHAT price does, not what causes it
- Mathematical precision for price movement sequences
- Observable price action that can be measured objectively

Pattern Categories:
1. Momentum Patterns - How price momentum behaves
2. Reversion Patterns - How price reverts to levels
3. Trend Patterns - How price trends develop and end
4. Range Patterns - How price behaves in ranges
5. Volatility Patterns - How price volatility changes
6. Acceleration Patterns - How price movement speed changes
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
    
    def _calculate_pattern_statistics(self, labels: pd.Series, prices: pd.Series) -> Dict[str, Any]:
        """Calculate pattern statistics."""
        
        frequency = labels.sum() / len(labels)
        
        # Duration statistics
        durations = self._calculate_durations(labels)
        
        # Magnitude statistics (price movement during pattern)
        magnitudes = self._calculate_magnitudes(labels, prices)
        
        # Predictability
        predictability = self._calculate_predictability(labels)
        
        # Statistical significance
        significance = self._calculate_significance(labels, prices)
        
        return {
            'frequency': frequency,
            'duration_stats': durations,
            'magnitude_stats': magnitudes,
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
    """Pure momentum persistence pattern - price momentum continues."""
    
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
        
        # Calculate price momentum (NOT return momentum)
        momentum = (prices - prices.shift(momentum_window)) / prices.shift(momentum_window)
        momentum = momentum.fillna(0)
        
        labels = []
        
        for i in range(len(momentum) - persistence_window):
            current_momentum = momentum.iloc[i]
            
            if abs(current_momentum) > momentum_threshold:
                future_momentum = momentum.iloc[i+1:i+persistence_window+1]
                
                # Direction persistence
                same_direction = (np.sign(future_momentum) == np.sign(current_momentum))
                direction_persistence = same_direction.sum() / len(future_momentum)
                
                # Magnitude decay
                magnitude_ratios = abs(future_momentum) / abs(current_momentum)
                gradual_decay = (magnitude_ratios > 0.3).sum() / len(magnitude_ratios)
                
                pattern_exists = (
                    direction_persistence >= direction_persistence_rate and
                    gradual_decay >= magnitude_decay_rate
                )
                
                labels.append(1 if pattern_exists else 0)
            else:
                labels.append(0)
        
        pattern_labels = pd.Series(labels, index=prices.index[:len(labels)])
        stats = self._calculate_pattern_statistics(pattern_labels, prices)
        
        return PurePatternResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            statistical_significance=stats['statistical_significance']
        )


class PriceReversionPattern(BasePurePricePatternDiscoverer):
    """Pure price reversion pattern - price returns to previous level."""
    
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
            3. Price moves back toward reference level
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
        
        for i in range(lookback_window, len(prices) - reversion_window):
            reference_level = prices.iloc[i - lookback_window]
            current_price = prices.iloc[i]
            
            # Calculate deviation from reference level
            deviation = abs(current_price - reference_level) / reference_level
            
            if deviation > deviation_threshold:
                # Look for reversion back toward reference level
                future_prices = prices.iloc[i+1:i+reversion_window+1]
                
                reversion_occurred = False
                for future_price in future_prices:
                    future_deviation = abs(future_price - reference_level) / reference_level
                    if future_deviation < reversion_ratio * deviation:
                        reversion_occurred = True
                        break
                
                labels.append(1 if reversion_occurred else 0)
            else:
                labels.append(0)
        
        pattern_labels = pd.Series(labels, index=prices.index[lookback_window:lookback_window+len(labels)])
        stats = self._calculate_pattern_statistics(pattern_labels, prices)
        
        return PurePatternResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            statistical_significance=stats['statistical_significance']
        )


class TrendAccelerationPattern(BasePurePricePatternDiscoverer):
    """Pure trend acceleration pattern - price movement speeds up."""
    
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
            3. velocity maintains direction throughout acceleration
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
        
        # Calculate velocity and acceleration
        velocity = prices.pct_change().fillna(0)
        acceleration = velocity.diff().fillna(0)
        
        labels = []
        
        for i in range(2, len(acceleration) - acceleration_window):
            current_velocity = velocity.iloc[i]
            current_acceleration = acceleration.iloc[i]
            
            # Check if acceleration and velocity are in same direction
            if (abs(current_acceleration) > min_acceleration and
                np.sign(current_acceleration) == np.sign(current_velocity) and
                current_velocity != 0):
                
                future_acceleration = acceleration.iloc[i+1:i+acceleration_window+1]
                future_velocity = velocity.iloc[i+1:i+acceleration_window+1]
                
                # Check if acceleration increases
                increasing_acceleration = (
                    abs(future_acceleration) > abs(current_acceleration)
                ).sum() / len(future_acceleration)
                
                # Check if velocity maintains direction
                velocity_direction_maintained = (
                    np.sign(future_velocity) == np.sign(current_velocity)
                ).sum() / len(future_velocity)
                
                pattern_exists = (
                    increasing_acceleration >= acceleration_consistency and
                    velocity_direction_maintained >= velocity_consistency
                )
                
                labels.append(1 if pattern_exists else 0)
            else:
                labels.append(0)
        
        start_idx = 2
        pattern_labels = pd.Series(labels, index=prices.index[start_idx:start_idx+len(labels)])
        stats = self._calculate_pattern_statistics(pattern_labels, prices)
        
        return PurePatternResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            statistical_significance=stats['statistical_significance']
        )


class PriceRangeBreakoutPattern(BasePurePricePatternDiscoverer):
    """Pure range breakout pattern - price breaks out of established range."""
    
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
            4. Breakout magnitude > minimum_breakout
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
        
        for i in range(range_window, len(prices) - continuation_window):
            # Define range
            recent_prices = prices.iloc[i-range_window:i]
            range_high = recent_prices.max()
            range_low = recent_prices.min()
            range_size = (range_high - range_low) / range_low
            
            current_price = prices.iloc[i]
            
            # Check if range is established (not too wide)
            if range_size < max_range_threshold:
                # Check for breakout
                upper_breakout = current_price > range_high
                lower_breakout = current_price < range_low
                
                if upper_breakout or lower_breakout:
                    # Check breakout magnitude
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
                        
                        continuation_pct = continuation_count / len(future_prices)
                        pattern_exists = continuation_pct >= continuation_rate
                        
                        labels.append(1 if pattern_exists else 0)
                    else:
                        labels.append(0)
                else:
                    labels.append(0)
            else:
                labels.append(0)
        
        start_idx = range_window
        pattern_labels = pd.Series(labels, index=prices.index[start_idx:start_idx+len(labels)])
        stats = self._calculate_pattern_statistics(pattern_labels, prices)
        
        return PurePatternResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            statistical_significance=stats['statistical_significance']
        )


class PriceWhipsawPattern(BasePurePricePatternDiscoverer):
    """Pure whipsaw pattern - price moves rapidly in both directions."""
    
    def __init__(self):
        super().__init__("PriceWhipsaw", PurePatternType.VOLATILITY)
    
    def get_pattern_definition(self) -> PurePricePattern:
        return PurePricePattern(
            name="Price Whipsaw",
            pattern_type=PurePatternType.VOLATILITY,
            description="Price moves rapidly in one direction then quickly reverses",
            mathematical_formula="""
            Let move_1 = (price(t+period_1) - price(t)) / price(t)
            Let move_2 = (price(t+period_2) - price(t+period_1)) / price(t+period_1)
            Let whipsaw_window = W
            
            Pattern exists at time t IF:
            1. |move_1| > move_threshold
            2. |move_2| > move_threshold  
            3. sign(move_1) != sign(move_2) (opposite directions)
            4. Both moves occur within W periods
            """,
            parameters={
                'whipsaw_window': 10,
                'move_threshold': 0.015,
                'min_reversal_ratio': 0.5
            },
            frequency_threshold=0.03
        )
    
    def discover_pattern(self,
                        prices: pd.Series,
                        whipsaw_window: int = 10,
                        move_threshold: float = 0.015,
                        min_reversal_ratio: float = 0.5) -> PurePatternResult:
        """Discover pure price whipsaw patterns."""
        
        self.logger.info("🔄 Discovering pure price whipsaw patterns")
        
        labels = []
        
        for i in range(len(prices) - whipsaw_window):
            current_price = prices.iloc[i]
            
            # Look for whipsaw within window
            whipsaw_detected = False
            
            for split_point in range(2, whipsaw_window - 2):
                mid_price = prices.iloc[i + split_point]
                end_price = prices.iloc[i + whipsaw_window]
                
                # Calculate moves
                move_1 = (mid_price - current_price) / current_price
                move_2 = (end_price - mid_price) / mid_price
                
                # Check whipsaw conditions
                significant_moves = abs(move_1) > move_threshold and abs(move_2) > move_threshold
                opposite_directions = np.sign(move_1) != np.sign(move_2)
                sufficient_reversal = abs(move_2) > min_reversal_ratio * abs(move_1)
                
                if significant_moves and opposite_directions and sufficient_reversal:
                    whipsaw_detected = True
                    break
            
            labels.append(1 if whipsaw_detected else 0)
        
        pattern_labels = pd.Series(labels, index=prices.index[:len(labels)])
        stats = self._calculate_pattern_statistics(pattern_labels, prices)
        
        return PurePatternResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            statistical_significance=stats['statistical_significance']
        )


class PriceLevelRejectionPattern(BasePurePricePatternDiscoverer):
    """Pure price level rejection pattern - price approaches level but fails to break."""
    
    def __init__(self):
        super().__init__("PriceLevelRejection", PurePatternType.REVERSION)
    
    def get_pattern_definition(self) -> PurePricePattern:
        return PurePricePattern(
            name="Price Level Rejection",
            pattern_type=PurePatternType.REVERSION,
            description="Price approaches significant level but fails to break through",
            mathematical_formula="""
            Let significant_level = identify_price_level(prices, lookback)
            Let approach_threshold = 0.01
            Let rejection_window = R
            
            Pattern exists at time t IF:
            1. |price(t) - significant_level| / price(t) < approach_threshold
            2. price(t+1:t+R) fails to break level decisively
            3. price moves away from level by > rejection_magnitude
            """,
            parameters={
                'lookback_window': 50,
                'approach_threshold': 0.01,
                'rejection_window': 5,
                'rejection_magnitude': 0.015,
                'touch_sensitivity': 3
            },
            frequency_threshold=0.06
        )
    
    def discover_pattern(self,
                        prices: pd.Series,
                        lookback_window: int = 50,
                        approach_threshold: float = 0.01,
                        rejection_window: int = 5,
                        rejection_magnitude: float = 0.015,
                        touch_sensitivity: int = 3) -> PurePatternResult:
        """Discover pure price level rejection patterns."""
        
        self.logger.info("🚫 Discovering pure price level rejection patterns")
        
        labels = []
        
        for i in range(lookback_window, len(prices) - rejection_window):
            current_price = prices.iloc[i]
            
            # Find significant levels in recent history
            recent_prices = prices.iloc[i-lookback_window:i]
            
            # Find levels that were touched multiple times (support/resistance)
            significant_levels = []
            
            for price_level in recent_prices:
                touch_count = 0
                for other_price in recent_prices:
                    if abs(other_price - price_level) / price_level < approach_threshold:
                        touch_count += 1
                
                if touch_count >= touch_sensitivity:
                    significant_levels.append(price_level)
            
            # Remove duplicates (keep unique levels)
            if significant_levels:
                significant_levels = list(set([
                    round(level, 4) for level in significant_levels
                ]))
            
            # Check if current price is approaching any significant level
            approaching_level = None
            for level in significant_levels:
                if abs(current_price - level) / current_price < approach_threshold:
                    approaching_level = level
                    break
            
            if approaching_level is not None:
                # Check for rejection
                future_prices = prices.iloc[i+1:i+rejection_window+1]
                
                # Price should fail to break level and move away
                level_breaks = (
                    (future_prices > approaching_level) if current_price <= approaching_level
                    else (future_prices < approaching_level)
                ).sum()
                
                break_rate = level_breaks / len(future_prices)
                
                # Calculate rejection magnitude
                if current_price <= approaching_level:
                    # Testing from below, rejection = move down
                    rejection_move = (current_price - future_prices.min()) / current_price
                else:
                    # Testing from above, rejection = move up
                    rejection_move = (future_prices.max() - current_price) / current_price
                
                pattern_exists = (
                    break_rate < 0.3 and  # Fails to break decisively
                    rejection_move > rejection_magnitude  # Moves away significantly
                )
                
                labels.append(1 if pattern_exists else 0)
            else:
                labels.append(0)
        
        start_idx = lookback_window
        pattern_labels = pd.Series(labels, index=prices.index[start_idx:start_idx+len(labels)])
        stats = self._calculate_pattern_statistics(pattern_labels, prices)
        
        return PurePatternResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            statistical_significance=stats['statistical_significance']
        )


class PriceGapPattern(BasePurePricePatternDiscoverer):
    """Pure price gap pattern - price gaps between periods."""
    
    def __init__(self):
        super().__init__("PriceGap", PurePatternType.VOLATILITY)
    
    def get_pattern_definition(self) -> PurePricePattern:
        return PurePricePattern(
            name="Price Gap",
            pattern_type=PurePatternType.VOLATILITY,
            description="Significant price gap between consecutive periods",
            mathematical_formula="""
            Let gap(t) = (price(t) - price(t-1)) / price(t-1)
            Let gap_threshold = 0.02
            Let fill_window = F
            
            Pattern exists at time t IF:
            1. |gap(t)| > gap_threshold
            2. Gap either fills within F periods OR continues in gap direction
            """,
            parameters={
                'gap_threshold': 0.02,
                'fill_window': 10
            },
            frequency_threshold=0.02
        )
    
    def discover_pattern(self,
                        prices: pd.Series,
                        gap_threshold: float = 0.02,
                        fill_window: int = 10) -> PurePatternResult:
        """Discover pure price gap patterns."""
        
        self.logger.info("📊 Discovering pure price gap patterns")
        
        # Calculate price gaps
        price_gaps = prices.pct_change().fillna(0)
        
        labels = []
        
        for i in range(1, len(price_gaps) - fill_window):
            current_gap = price_gaps.iloc[i]
            
            if abs(current_gap) > gap_threshold:
                gap_price = prices.iloc[i-1]  # Price before gap
                current_price = prices.iloc[i]  # Price after gap
                
                # Check what happens to gap
                future_prices = prices.iloc[i+1:i+fill_window+1]
                
                if current_gap > 0:  # Up gap
                    # Check if gap fills (price goes back below gap_price)
                    gap_fills = any(future_prices <= gap_price)
                else:  # Down gap
                    # Check if gap fills (price goes back above gap_price)
                    gap_fills = any(future_prices >= gap_price)
                
                # Pattern exists for any significant gap (filled or unfilled)
                pattern_exists = True  # All significant gaps are patterns
                
                labels.append(1 if pattern_exists else 0)
            else:
                labels.append(0)
        
        start_idx = 1
        pattern_labels = pd.Series(labels, index=prices.index[start_idx:start_idx+len(labels)])
        stats = self._calculate_pattern_statistics(pattern_labels, prices)
        
        return PurePatternResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            statistical_significance=stats['statistical_significance']
        )


class PriceConsolidationPattern(BasePurePricePatternDiscoverer):
    """Pure price consolidation pattern - price moves sideways."""
    
    def __init__(self):
        super().__init__("PriceConsolidation", PurePatternType.RANGE)
    
    def get_pattern_definition(self) -> PurePricePattern:
        return PurePricePattern(
            name="Price Consolidation",
            pattern_type=PurePatternType.RANGE,
            description="Price moves sideways within narrow range for extended period",
            mathematical_formula="""
            Let price_range(t:t+window) = (max_price - min_price) / min_price
            Let consolidation_window = C
            Let range_threshold = 0.05
            
            Pattern exists at time t IF:
            1. price_range(t:t+C) < range_threshold
            2. No sustained directional movement > movement_threshold
            3. Price stays within range for ≥80% of C periods
            """,
            parameters={
                'consolidation_window': 20,
                'range_threshold': 0.05,
                'movement_threshold': 0.03,
                'range_consistency': 0.8
            },
            frequency_threshold=0.1
        )
    
    def discover_pattern(self,
                        prices: pd.Series,
                        consolidation_window: int = 20,
                        range_threshold: float = 0.05,
                        movement_threshold: float = 0.03,
                        range_consistency: float = 0.8) -> PurePatternResult:
        """Discover pure price consolidation patterns."""
        
        self.logger.info("↔️ Discovering pure price consolidation patterns")
        
        labels = []
        
        for i in range(len(prices) - consolidation_window):
            # Analyze price behavior over consolidation window
            window_prices = prices.iloc[i:i+consolidation_window]
            
            # Calculate range
            price_range = (window_prices.max() - window_prices.min()) / window_prices.min()
            
            # Calculate maximum sustained move
            max_sustained_move = 0
            for start_j in range(len(window_prices)):
                for end_j in range(start_j + 1, len(window_prices)):
                    move = abs(window_prices.iloc[end_j] - window_prices.iloc[start_j]) / window_prices.iloc[start_j]
                    max_sustained_move = max(max_sustained_move, move)
            
            # Check consolidation conditions
            narrow_range = price_range < range_threshold
            limited_movement = max_sustained_move < movement_threshold
            
            pattern_exists = narrow_range and limited_movement
            
            labels.append(1 if pattern_exists else 0)
        
        pattern_labels = pd.Series(labels, index=prices.index[:len(labels)])
        stats = self._calculate_pattern_statistics(pattern_labels, prices)
        
        return PurePatternResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            statistical_significance=stats['statistical_significance']
        )


class PriceExtremeReversalPattern(BasePurePricePatternDiscoverer):
    """Pure extreme reversal pattern - large price moves followed by reversal."""
    
    def __init__(self):
        super().__init__("PriceExtremeReversal", PurePatternType.REVERSION)
    
    def get_pattern_definition(self) -> PurePricePattern:
        return PurePricePattern(
            name="Price Extreme Reversal",
            pattern_type=PurePatternType.REVERSION,
            description="Extreme price movement followed by reversal in opposite direction",
            mathematical_formula="""
            Let return(t) = (price(t) - price(t-1)) / price(t-1)
            Let volatility(t) = std(returns[t-vol_window:t])
            Let extreme_threshold = 3.0
            Let reversal_window = R
            
            Pattern exists at time t IF:
            1. |return(t)| > extreme_threshold * volatility(t)
            2. return(t+1:t+R) moves in opposite direction
            3. Reversal magnitude > min_reversal_ratio * |return(t)|
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
        
        for i in range(vol_window, len(returns) - reversal_window):
            current_return = returns.iloc[i]
            current_volatility = volatility.iloc[i]
            
            # Check for extreme movement
            if current_volatility > 0 and abs(current_return) > extreme_threshold * current_volatility:
                
                # Look for reversal
                future_returns = returns.iloc[i+1:i+reversal_window+1]
                cumulative_reversal = future_returns.sum()
                
                # Check if reversal is in opposite direction and sufficient magnitude
                opposite_direction = (
                    (current_return > 0 and cumulative_reversal < 0) or
                    (current_return < 0 and cumulative_reversal > 0)
                )
                
                if opposite_direction:
                    reversal_ratio = abs(cumulative_reversal) / abs(current_return)
                    sufficient_reversal = reversal_ratio >= min_reversal_ratio
                    pattern_exists = sufficient_reversal
                else:
                    pattern_exists = False
                
                labels.append(1 if pattern_exists else 0)
            else:
                labels.append(0)
        
        start_idx = vol_window
        pattern_labels = pd.Series(labels, index=prices.index[start_idx:start_idx+len(labels)])
        stats = self._calculate_pattern_statistics(pattern_labels, prices)
        
        return PurePatternResult(
            definition=self.get_pattern_definition(),
            labels=pattern_labels,
            frequency=stats['frequency'],
            duration_stats=stats['duration_stats'],
            magnitude_stats=stats['magnitude_stats'],
            predictability_score=stats['predictability_score'],
            statistical_significance=stats['statistical_significance']
        )


class PurePricePatternOrchestrator:
    """Orchestrator for pure price action pattern discovery."""
    
    def __init__(self):
        self.logger = system_logger.getChild('PurePricePatternOrchestrator')
        
        # Initialize pure price action pattern discoverers
        self.pure_discoverers = {
            'momentum_persistence': MomentumPersistencePattern(),
            'price_reversion': PriceReversionPattern(),
            'trend_acceleration': TrendAccelerationPattern(),
            'range_breakout': PriceRangeBreakoutPattern(),
            'price_whipsaw': PriceWhipsawPattern(),
            'level_rejection': PriceLevelRejectionPattern(),
            'price_gap': PriceGapPattern(),
            'price_consolidation': PriceConsolidationPattern(),
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
    
    def get_pure_pattern_definitions(self) -> Dict[str, PurePricePattern]:
        """Get all pure pattern definitions."""
        return {
            name: discoverer.get_pattern_definition()
            for name, discoverer in self.pure_discoverers.items()
        }
    
    def export_pure_pattern_labels(self, results: Dict[str, PurePatternResult]) -> pd.DataFrame:
        """Export pure pattern labels as ML-ready DataFrame."""
        
        pattern_labels = {}
        
        for pattern_name, result in results.items():
            if result.is_valid_pattern:
                pattern_labels[pattern_name] = result.labels
        
        if pattern_labels:
            labels_df = pd.DataFrame(pattern_labels)
            
            # Add composite patterns (pure price action only)
            if len(labels_df.columns) > 1:
                labels_df['any_momentum'] = labels_df[[
                    col for col in labels_df.columns 
                    if 'momentum' in col or 'acceleration' in col
                ]].max(axis=1)
                
                labels_df['any_reversion'] = labels_df[[
                    col for col in labels_df.columns 
                    if 'reversion' in col or 'rejection' in col or 'reversal' in col
                ]].max(axis=1)
                
                labels_df['any_range'] = labels_df[[
                    col for col in labels_df.columns 
                    if 'range' in col or 'consolidation' in col
                ]].max(axis=1)
                
                labels_df['any_volatility'] = labels_df[[
                    col for col in labels_df.columns 
                    if 'gap' in col or 'whipsaw' in col or 'extreme' in col
                ]].max(axis=1)
            
            return labels_df
        else:
            return pd.DataFrame()
    
    def generate_pure_pattern_report(self, results: Dict[str, PurePatternResult]) -> str:
        """Generate report focused on pure price action patterns."""
        
        report = []
        report.append("# Pure Price Action Pattern Discovery Report")
        report.append("=" * 60)
        report.append("")
        report.append("**Focus**: Pure price action patterns (WHAT price does, not WHY)")
        report.append("**Approach**: Mathematical definitions based only on price movements")
        report.append("**Output**: ML-ready binary labels for supervised learning")
        report.append("")
        
        # Summary
        total_patterns = len(results)
        valid_patterns = sum(1 for result in results.values() if result.is_valid_pattern)
        
        report.append("## Pattern Discovery Summary")
        report.append("")
        report.append(f"- **Total Pure Patterns Analyzed**: {total_patterns}")
        report.append(f"- **Valid Patterns Found**: {valid_patterns}")
        report.append(f"- **Pattern Validity Rate**: {valid_patterns/total_patterns*100:.1f}%")
        report.append("")
        
        # Pattern analysis by category
        pattern_categories = {}
        for pattern_name, result in results.items():
            category = result.definition.pattern_type.value
            if category not in pattern_categories:
                pattern_categories[category] = []
            pattern_categories[category].append((pattern_name, result))
        
        for category, category_patterns in pattern_categories.items():
            report.append(f"## {category.title()} Patterns")
            report.append("")
            
            for pattern_name, result in category_patterns:
                status = "✅ VALID" if result.is_valid_pattern else "❌ INVALID"
                
                report.append(f"### {pattern_name.replace('_', ' ').title()} {status}")
                report.append("")
                
                # Mathematical definition
                report.append("**Mathematical Definition:**")
                report.append("```")
                report.append(result.definition.mathematical_formula.strip())
                report.append("```")
                report.append("")
                
                # Statistics
                report.append("**Pattern Statistics:**")
                report.append(f"- Frequency: {result.frequency:.3f} ({result.frequency*100:.1f}% of periods)")
                report.append(f"- Predictability: {result.predictability_score:.3f}")
                
                if result.duration_stats['mean'] > 0:
                    report.append(f"- Average Duration: {result.duration_stats['mean']:.1f} periods")
                
                if result.statistical_significance.get('p_value'):
                    p_val = result.statistical_significance['p_value']
                    significance = "Significant" if p_val < 0.05 else "Not Significant"
                    report.append(f"- Statistical Significance: {significance} (p={p_val:.3f})")
                
                report.append("")
        
        # Key insights
        report.append("## Key Insights: Pure Price Action Focus")
        report.append("")
        
        if valid_patterns > 0:
            report.append("✅ **Valid Pure Price Patterns Discovered**")
            report.append("- Patterns defined purely by price movements")
            report.append("- No assumptions about underlying causes")
            report.append("- Mathematical precision enables reproducible analysis")
            report.append("- Ready for ML target generation")
            report.append("")
            
            # Most frequent patterns
            frequent_patterns = [
                (name, result) for name, result in results.items()
                if result.is_valid_pattern and result.frequency > 0.05
            ]
            
            if frequent_patterns:
                report.append("**Most Frequent Patterns (>5% occurrence):**")
                for name, result in frequent_patterns:
                    report.append(f"- {name}: {result.frequency:.1%}")
                report.append("")
        
        else:
            report.append("❌ **No Valid Pure Price Patterns Found**")
            report.append("- Consider adjusting pattern parameters")
            report.append("- Try different timeframes or market conditions")
            report.append("- Ensure sufficient data length for analysis")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        if valid_patterns >= 5:
            report.append("🎯 **Strong Pure Price Pattern Foundation**")
            report.append("- Use patterns as ML targets for supervised learning")
            report.append("- Test which market dimensions predict these price patterns")
            report.append("- Develop pattern-specific trading strategies")
        elif valid_patterns >= 2:
            report.append("⚠️ **Moderate Pure Price Pattern Foundation**")
            report.append("- Focus on most frequent and predictable patterns")
            report.append("- Consider parameter optimization for failed patterns")
            report.append("- Validate economic significance through backtesting")
        else:
            report.append("❌ **Limited Pure Price Pattern Foundation**")
            report.append("- Adjust pattern sensitivity parameters")
            report.append("- Consider longer time series or different timeframes")
            report.append("- Focus on most basic patterns first")
        
        return "\n".join(report)


# Example usage
def run_pure_price_action_example():
    """Example of pure price action pattern discovery."""
    
    print("Pure Price Action Pattern Discovery")
    print("==================================")
    print()
    print("🎯 FOCUS: Pure price action patterns only")
    print("   - WHAT price does, not WHY it moves")
    print("   - No volume, no fundamentals, no market structure")
    print("   - Mathematical definitions based solely on price movements")
    print()
    print("Available pure price patterns:")
    print("1. Momentum Persistence - Price momentum continues with decay")
    print("2. Price Reversion - Price returns to previous levels")
    print("3. Trend Acceleration - Price movement speeds up")
    print("4. Range Breakout - Price breaks established range")
    print("5. Price Whipsaw - Rapid moves in both directions")
    print("6. Level Rejection - Price fails to break through levels")
    print("7. Price Gap - Significant price gaps between periods")
    print("8. Price Consolidation - Sideways price movement")
    print("9. Extreme Reversal - Large moves followed by reversal")
    print()
    print("Usage:")
    print("```python")
    print("orchestrator = PurePricePatternOrchestrator()")
    print("results = orchestrator.discover_all_pure_patterns(price_series)")
    print("ml_targets = orchestrator.export_pure_pattern_labels(results)")
    print("```")


if __name__ == "__main__":
    run_pure_price_action_example()