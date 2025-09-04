#!/usr/bin/env python3
"""Enhanced S/R Detection Module.

This module implements advanced S/R detection algorithms with improved accuracy
and robustness for 1-30m timeframes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

from src.core.decorators import handles_errors, traced
from src.utils.logger import system_logger


@dataclass
class SRLevel:
    """Enhanced S/R level definition with comprehensive metadata."""
    price: float
    strength: float
    type: str  # 'support' or 'resistance'
    touch_count: int
    first_touch_time: pd.Timestamp
    last_touch_time: pd.Timestamp
    age_bars: int
    avg_bounce_ratio: float
    max_bounce_ratio: float
    volume_confirmation_score: float
    consistency_score: float
    failure_count: int
    confidence_score: float
    confluence_score: float
    fibonacci_level: Optional[float] = None
    pivot_level: bool = False
    psychological_level: bool = False
    metadata: Dict[str, Any] = None


class EnhancedSRDetector:
    """Enhanced S/R detector with advanced algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced S/R detector."""
        self.config = config
        self.logger = system_logger.getChild("EnhancedSRDetector")
        
        # Detection parameters
        self.min_touches = config.get("min_touches", 3)
        self.touch_proximity_threshold = config.get("touch_proximity_threshold", 0.002)
        self.min_strength = config.get("min_strength", 0.6)
        self.volume_spike_threshold = config.get("volume_spike_threshold", 1.5)
        
        # Advanced parameters
        self.fractal_period = config.get("fractal_period", 5)
        self.pivot_period = config.get("pivot_period", 10)
        self.psychological_levels = config.get("psychological_levels", True)
        self.fibonacci_levels = config.get("fibonacci_levels", True)
        
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=[],
        context="detect enhanced SR levels"
    )
    @traced(span_name="EnhancedSR.detect_levels")
    def detect_sr_levels(self, market_data: pd.DataFrame) -> List[SRLevel]:
        """
        Detect S/R levels using multiple advanced algorithms.
        
        Args:
            market_data: OHLCV data with timestamp index
            
        Returns:
            List of detected S/R levels
        """
        try:
            self.logger.info("🔍 Starting enhanced S/R level detection...")
            
            # Method 1: Fractal-based detection
            fractal_levels = self._detect_fractal_levels(market_data)
            
            # Method 2: Pivot-based detection
            pivot_levels = self._detect_pivot_levels(market_data)
            
            # Method 3: Volume-based detection
            volume_levels = self._detect_volume_levels(market_data)
            
            # Method 4: Statistical level detection
            statistical_levels = self._detect_statistical_levels(market_data)
            
            # Method 5: Psychological level detection
            psychological_levels = self._detect_psychological_levels(market_data)
            
            # Combine and validate all levels
            all_levels = (fractal_levels + pivot_levels + volume_levels + 
                         statistical_levels + psychological_levels)
            
            # Remove duplicates and validate
            validated_levels = self._validate_and_merge_levels(all_levels, market_data)
            
            # Calculate enhanced metrics
            enhanced_levels = self._calculate_enhanced_metrics(validated_levels, market_data)
            
            self.logger.info(f"✅ Detected {len(enhanced_levels)} enhanced S/R levels")
            return enhanced_levels
            
        except Exception as e:
            self.logger.error(f"Enhanced S/R detection failed: {e}")
            return []
    
    def _detect_fractal_levels(self, data: pd.DataFrame) -> List[SRLevel]:
        """Detect S/R levels using fractal analysis."""
        try:
            levels = []
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            
            # Find fractal highs and lows
            fractal_highs = self._find_fractal_highs(high, self.fractal_period)
            fractal_lows = self._find_fractal_lows(low, self.fractal_period)
            
            # Create resistance levels from fractal highs
            for i, price in enumerate(fractal_highs):
                if i < len(data):
                    level = SRLevel(
                        price=price,
                        strength=0.7,  # Base strength for fractals
                        type='resistance',
                        touch_count=1,
                        first_touch_time=data.index[i],
                        last_touch_time=data.index[i],
                        age_bars=0,
                        avg_bounce_ratio=0.0,
                        max_bounce_ratio=0.0,
                        volume_confirmation_score=0.0,
                        consistency_score=0.0,
                        failure_count=0,
                        confidence_score=0.7,
                        confluence_score=0.0,
                        pivot_level=False,
                        psychological_level=False,
                        metadata={'method': 'fractal', 'period': self.fractal_period}
                    )
                    levels.append(level)
            
            # Create support levels from fractal lows
            for i, price in enumerate(fractal_lows):
                if i < len(data):
                    level = SRLevel(
                        price=price,
                        strength=0.7,
                        type='support',
                        touch_count=1,
                        first_touch_time=data.index[i],
                        last_touch_time=data.index[i],
                        age_bars=0,
                        avg_bounce_ratio=0.0,
                        max_bounce_ratio=0.0,
                        volume_confirmation_score=0.0,
                        consistency_score=0.0,
                        failure_count=0,
                        confidence_score=0.7,
                        confluence_score=0.0,
                        pivot_level=False,
                        psychological_level=False,
                        metadata={'method': 'fractal', 'period': self.fractal_period}
                    )
                    levels.append(level)
            
            return levels
            
        except Exception as e:
            self.logger.warning(f"Fractal detection failed: {e}")
            return []
    
    def _detect_pivot_levels(self, data: pd.DataFrame) -> List[SRLevel]:
        """Detect S/R levels using pivot point analysis."""
        try:
            levels = []
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # Calculate pivot points
            for i in range(self.pivot_period, len(data) - self.pivot_period):
                # Check for pivot high
                if self._is_pivot_high(high, i, self.pivot_period):
                    level = SRLevel(
                        price=high[i],
                        strength=0.8,  # Higher strength for pivots
                        type='resistance',
                        touch_count=1,
                        first_touch_time=data.index[i],
                        last_touch_time=data.index[i],
                        age_bars=0,
                        avg_bounce_ratio=0.0,
                        max_bounce_ratio=0.0,
                        volume_confirmation_score=0.0,
                        consistency_score=0.0,
                        failure_count=0,
                        confidence_score=0.8,
                        confluence_score=0.0,
                        pivot_level=True,
                        psychological_level=False,
                        metadata={'method': 'pivot', 'period': self.pivot_period}
                    )
                    levels.append(level)
                
                # Check for pivot low
                if self._is_pivot_low(low, i, self.pivot_period):
                    level = SRLevel(
                        price=low[i],
                        strength=0.8,
                        type='support',
                        touch_count=1,
                        first_touch_time=data.index[i],
                        last_touch_time=data.index[i],
                        age_bars=0,
                        avg_bounce_ratio=0.0,
                        max_bounce_ratio=0.0,
                        volume_confirmation_score=0.0,
                        consistency_score=0.0,
                        failure_count=0,
                        confidence_score=0.8,
                        confluence_score=0.0,
                        pivot_level=True,
                        psychological_level=False,
                        metadata={'method': 'pivot', 'period': self.pivot_period}
                    )
                    levels.append(level)
            
            return levels
            
        except Exception as e:
            self.logger.warning(f"Pivot detection failed: {e}")
            return []
    
    def _detect_volume_levels(self, data: pd.DataFrame) -> List[SRLevel]:
        """Detect S/R levels based on volume spikes and price reactions."""
        try:
            levels = []
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values
            
            # Calculate volume moving average
            volume_ma = pd.Series(volume).rolling(window=20).mean()
            volume_spikes = volume > (volume_ma * self.volume_spike_threshold)
            
            # Find levels where volume spikes occurred with price reactions
            for i in range(len(data)):
                if volume_spikes.iloc[i] if hasattr(volume_spikes, 'iloc') else volume_spikes[i]:
                    # Check for resistance level (high volume at high price)
                    if i > 0 and high[i] > high[i-1]:
                        level = SRLevel(
                            price=high[i],
                            strength=0.6,
                            type='resistance',
                            touch_count=1,
                            first_touch_time=data.index[i],
                            last_touch_time=data.index[i],
                            age_bars=0,
                            avg_bounce_ratio=0.0,
                            max_bounce_ratio=0.0,
                            volume_confirmation_score=1.0,
                            consistency_score=0.0,
                            failure_count=0,
                            confidence_score=0.6,
                            confluence_score=0.0,
                            pivot_level=False,
                            psychological_level=False,
                            metadata={'method': 'volume', 'volume_ratio': volume[i] / volume_ma.iloc[i]}
                        )
                        levels.append(level)
                    
                    # Check for support level (high volume at low price)
                    if i > 0 and low[i] < low[i-1]:
                        level = SRLevel(
                            price=low[i],
                            strength=0.6,
                            type='support',
                            touch_count=1,
                            first_touch_time=data.index[i],
                            last_touch_time=data.index[i],
                            age_bars=0,
                            avg_bounce_ratio=0.0,
                            max_bounce_ratio=0.0,
                            volume_confirmation_score=1.0,
                            consistency_score=0.0,
                            failure_count=0,
                            confidence_score=0.6,
                            confluence_score=0.0,
                            pivot_level=False,
                            psychological_level=False,
                            metadata={'method': 'volume', 'volume_ratio': volume[i] / volume_ma.iloc[i]}
                        )
                        levels.append(level)
            
            return levels
            
        except Exception as e:
            self.logger.warning(f"Volume detection failed: {e}")
            return []
    
    def _detect_statistical_levels(self, data: pd.DataFrame) -> List[SRLevel]:
        """Detect S/R levels using statistical analysis."""
        try:
            levels = []
            close = data['close'].values
            
            # Calculate statistical levels
            mean_price = np.mean(close)
            std_price = np.std(close)
            
            # Standard deviation levels
            for std_multiple in [1, 2, 3]:
                upper_level = mean_price + (std_multiple * std_price)
                lower_level = mean_price - (std_multiple * std_price)
                
                # Resistance level
                level = SRLevel(
                    price=upper_level,
                    strength=0.5 + (std_multiple * 0.1),
                    type='resistance',
                    touch_count=0,
                    first_touch_time=data.index[0],
                    last_touch_time=data.index[0],
                    age_bars=len(data),
                    avg_bounce_ratio=0.0,
                    max_bounce_ratio=0.0,
                    volume_confirmation_score=0.0,
                    consistency_score=0.0,
                    failure_count=0,
                    confidence_score=0.5 + (std_multiple * 0.1),
                    confluence_score=0.0,
                    pivot_level=False,
                    psychological_level=False,
                    metadata={'method': 'statistical', 'std_multiple': std_multiple}
                )
                levels.append(level)
                
                # Support level
                level = SRLevel(
                    price=lower_level,
                    strength=0.5 + (std_multiple * 0.1),
                    type='support',
                    touch_count=0,
                    first_touch_time=data.index[0],
                    last_touch_time=data.index[0],
                    age_bars=len(data),
                    avg_bounce_ratio=0.0,
                    max_bounce_ratio=0.0,
                    volume_confirmation_score=0.0,
                    consistency_score=0.0,
                    failure_count=0,
                    confidence_score=0.5 + (std_multiple * 0.1),
                    confluence_score=0.0,
                    pivot_level=False,
                    psychological_level=False,
                    metadata={'method': 'statistical', 'std_multiple': std_multiple}
                )
                levels.append(level)
            
            return levels
            
        except Exception as e:
            self.logger.warning(f"Statistical detection failed: {e}")
            return []
    
    def _detect_psychological_levels(self, data: pd.DataFrame) -> List[SRLevel]:
        """Detect psychological S/R levels (round numbers)."""
        try:
            levels = []
            close = data['close'].values
            current_price = close[-1]
            
            # Generate psychological levels around current price
            price_magnitude = 10 ** (int(np.log10(current_price)))
            
            # Find nearby round numbers
            for multiplier in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                level_price = round(current_price / (price_magnitude * multiplier)) * (price_magnitude * multiplier)
                
                if level_price > current_price:
                    # Resistance level
                    level = SRLevel(
                        price=level_price,
                        strength=0.4,
                        type='resistance',
                        touch_count=0,
                        first_touch_time=data.index[0],
                        last_touch_time=data.index[0],
                        age_bars=len(data),
                        avg_bounce_ratio=0.0,
                        max_bounce_ratio=0.0,
                        volume_confirmation_score=0.0,
                        consistency_score=0.0,
                        failure_count=0,
                        confidence_score=0.4,
                        confluence_score=0.0,
                        pivot_level=False,
                        psychological_level=True,
                        metadata={'method': 'psychological', 'multiplier': multiplier}
                    )
                    levels.append(level)
                else:
                    # Support level
                    level = SRLevel(
                        price=level_price,
                        strength=0.4,
                        type='support',
                        touch_count=0,
                        first_touch_time=data.index[0],
                        last_touch_time=data.index[0],
                        age_bars=len(data),
                        avg_bounce_ratio=0.0,
                        max_bounce_ratio=0.0,
                        volume_confirmation_score=0.0,
                        consistency_score=0.0,
                        failure_count=0,
                        confidence_score=0.4,
                        confluence_score=0.0,
                        pivot_level=False,
                        psychological_level=True,
                        metadata={'method': 'psychological', 'multiplier': multiplier}
                    )
                    levels.append(level)
            
            return levels
            
        except Exception as e:
            self.logger.warning(f"Psychological detection failed: {e}")
            return []
    
    def _find_fractal_highs(self, high: np.ndarray, period: int) -> List[float]:
        """Find fractal highs in price data."""
        try:
            peaks, _ = find_peaks(high, distance=period)
            return [high[i] for i in peaks]
        except Exception:
            return []
    
    def _find_fractal_lows(self, low: np.ndarray, period: int) -> List[float]:
        """Find fractal lows in price data."""
        try:
            peaks, _ = find_peaks(-low, distance=period)
            return [low[i] for i in peaks]
        except Exception:
            return []
    
    def _is_pivot_high(self, high: np.ndarray, index: int, period: int) -> bool:
        """Check if index is a pivot high."""
        try:
            if index < period or index >= len(high) - period:
                return False
            
            center_value = high[index]
            left_values = high[index - period:index]
            right_values = high[index + 1:index + period + 1]
            
            return (center_value > np.max(left_values) and 
                   center_value > np.max(right_values))
        except Exception:
            return False
    
    def _is_pivot_low(self, low: np.ndarray, index: int, period: int) -> bool:
        """Check if index is a pivot low."""
        try:
            if index < period or index >= len(low) - period:
                return False
            
            center_value = low[index]
            left_values = low[index - period:index]
            right_values = low[index + 1:index + period + 1]
            
            return (center_value < np.min(left_values) and 
                   center_value < np.min(right_values))
        except Exception:
            return False
    
    def _validate_and_merge_levels(self, levels: List[SRLevel], data: pd.DataFrame) -> List[SRLevel]:
        """Validate and merge similar levels."""
        try:
            if not levels:
                return []
            
            # Group levels by proximity
            merged_levels = []
            used_indices = set()
            
            for i, level in enumerate(levels):
                if i in used_indices:
                    continue
                
                # Find similar levels
                similar_levels = [level]
                for j, other_level in enumerate(levels[i+1:], i+1):
                    if j in used_indices:
                        continue
                    
                    price_diff = abs(level.price - other_level.price) / level.price
                    if (price_diff < self.touch_proximity_threshold and 
                        level.type == other_level.type):
                        similar_levels.append(other_level)
                        used_indices.add(j)
                
                # Merge similar levels
                if len(similar_levels) > 1:
                    merged_level = self._merge_similar_levels(similar_levels)
                    merged_levels.append(merged_level)
                else:
                    merged_levels.append(level)
                
                used_indices.add(i)
            
            return merged_levels
            
        except Exception as e:
            self.logger.warning(f"Level validation failed: {e}")
            return levels
    
    def _merge_similar_levels(self, levels: List[SRLevel]) -> SRLevel:
        """Merge similar S/R levels into one."""
        try:
            # Calculate weighted average price
            total_strength = sum(level.strength for level in levels)
            weighted_price = sum(level.price * level.strength for level in levels) / total_strength
            
            # Use the strongest level as base
            base_level = max(levels, key=lambda x: x.strength)
            
            # Create merged level
            merged_level = SRLevel(
                price=weighted_price,
                strength=min(total_strength / len(levels) * 1.2, 1.0),  # Boost strength
                type=base_level.type,
                touch_count=sum(level.touch_count for level in levels),
                first_touch_time=min(level.first_touch_time for level in levels),
                last_touch_time=max(level.last_touch_time for level in levels),
                age_bars=max(level.age_bars for level in levels),
                avg_bounce_ratio=np.mean([level.avg_bounce_ratio for level in levels]),
                max_bounce_ratio=max(level.max_bounce_ratio for level in levels),
                volume_confirmation_score=np.mean([level.volume_confirmation_score for level in levels]),
                consistency_score=np.mean([level.consistency_score for level in levels]),
                failure_count=sum(level.failure_count for level in levels),
                confidence_score=min(np.mean([level.confidence_score for level in levels]) * 1.1, 1.0),
                confluence_score=len(levels) / 10.0,  # Higher confluence for merged levels
                pivot_level=any(level.pivot_level for level in levels),
                psychological_level=any(level.psychological_level for level in levels),
                metadata={'merged_from': len(levels), 'methods': [level.metadata.get('method', 'unknown') for level in levels]}
            )
            
            return merged_level
            
        except Exception as e:
            self.logger.warning(f"Level merging failed: {e}")
            return levels[0] if levels else None
    
    def _calculate_enhanced_metrics(self, levels: List[SRLevel], data: pd.DataFrame) -> List[SRLevel]:
        """Calculate enhanced metrics for S/R levels."""
        try:
            enhanced_levels = []
            
            for level in levels:
                # Calculate touch validation
                touches = self._count_touches(level, data)
                level.touch_count = touches
                
                # Calculate bounce metrics
                bounce_metrics = self._calculate_bounce_metrics(level, data)
                level.avg_bounce_ratio = bounce_metrics['avg_bounce']
                level.max_bounce_ratio = bounce_metrics['max_bounce']
                
                # Calculate volume confirmation
                volume_score = self._calculate_volume_confirmation(level, data)
                level.volume_confirmation_score = volume_score
                
                # Calculate consistency score
                consistency = self._calculate_consistency_score(level, data)
                level.consistency_score = consistency
                
                # Calculate failure count
                failures = self._count_failures(level, data)
                level.failure_count = failures
                
                # Update strength based on metrics
                level.strength = self._calculate_enhanced_strength(level)
                
                # Calculate age
                if level.first_touch_time and level.last_touch_time:
                    level.age_bars = (level.last_touch_time - level.first_touch_time).total_seconds() / 60  # minutes
                
                enhanced_levels.append(level)
            
            return enhanced_levels
            
        except Exception as e:
            self.logger.warning(f"Enhanced metrics calculation failed: {e}")
            return levels
    
    def _count_touches(self, level: SRLevel, data: pd.DataFrame) -> int:
        """Count touches of price to S/R level."""
        try:
            touches = 0
            threshold = level.price * self.touch_proximity_threshold
            
            for i, row in data.iterrows():
                if level.type == 'support':
                    if abs(row['low'] - level.price) <= threshold:
                        touches += 1
                else:  # resistance
                    if abs(row['high'] - level.price) <= threshold:
                        touches += 1
            
            return touches
            
        except Exception as e:
            self.logger.warning(f"Touch counting failed: {e}")
            return 0
    
    def _calculate_bounce_metrics(self, level: SRLevel, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate bounce metrics for S/R level."""
        try:
            bounces = []
            threshold = level.price * self.touch_proximity_threshold
            
            for i in range(len(data) - 1):
                if level.type == 'support':
                    if abs(data.iloc[i]['low'] - level.price) <= threshold:
                        # Check for bounce
                        next_high = data.iloc[i+1]['high']
                        bounce_ratio = (next_high - level.price) / level.price
                        if bounce_ratio > 0:
                            bounces.append(bounce_ratio)
                else:  # resistance
                    if abs(data.iloc[i]['high'] - level.price) <= threshold:
                        # Check for bounce
                        next_low = data.iloc[i+1]['low']
                        bounce_ratio = (level.price - next_low) / level.price
                        if bounce_ratio > 0:
                            bounces.append(bounce_ratio)
            
            if bounces:
                return {
                    'avg_bounce': np.mean(bounces),
                    'max_bounce': np.max(bounces)
                }
            else:
                return {'avg_bounce': 0.0, 'max_bounce': 0.0}
                
        except Exception as e:
            self.logger.warning(f"Bounce calculation failed: {e}")
            return {'avg_bounce': 0.0, 'max_bounce': 0.0}
    
    def _calculate_volume_confirmation(self, level: SRLevel, data: pd.DataFrame) -> float:
        """Calculate volume confirmation score for S/R level."""
        try:
            if 'volume' not in data.columns:
                return 0.0
            
            volume_ma = data['volume'].rolling(window=20).mean()
            threshold = level.price * self.touch_proximity_threshold
            
            volume_spikes = []
            for i, row in data.iterrows():
                if level.type == 'support':
                    if abs(row['low'] - level.price) <= threshold:
                        if row['volume'] > volume_ma.iloc[i] * self.volume_spike_threshold:
                            volume_spikes.append(1.0)
                        else:
                            volume_spikes.append(0.0)
                else:  # resistance
                    if abs(row['high'] - level.price) <= threshold:
                        if row['volume'] > volume_ma.iloc[i] * self.volume_spike_threshold:
                            volume_spikes.append(1.0)
                        else:
                            volume_spikes.append(0.0)
            
            return np.mean(volume_spikes) if volume_spikes else 0.0
            
        except Exception as e:
            self.logger.warning(f"Volume confirmation calculation failed: {e}")
            return 0.0
    
    def _calculate_consistency_score(self, level: SRLevel, data: pd.DataFrame) -> float:
        """Calculate consistency score for S/R level."""
        try:
            # Simple consistency based on touch count and age
            if level.touch_count == 0:
                return 0.0
            
            # Higher consistency for more touches and longer age
            touch_score = min(level.touch_count / 5.0, 1.0)
            age_score = min(level.age_bars / 1000.0, 1.0)  # 1000 bars = 1.0
            
            return (touch_score + age_score) / 2.0
            
        except Exception as e:
            self.logger.warning(f"Consistency calculation failed: {e}")
            return 0.0
    
    def _count_failures(self, level: SRLevel, data: pd.DataFrame) -> int:
        """Count failures (breakouts) of S/R level."""
        try:
            failures = 0
            threshold = level.price * self.touch_proximity_threshold
            
            for i, row in data.iterrows():
                if level.type == 'support':
                    if row['close'] < (level.price - threshold):
                        failures += 1
                else:  # resistance
                    if row['close'] > (level.price + threshold):
                        failures += 1
            
            return failures
            
        except Exception as e:
            self.logger.warning(f"Failure counting failed: {e}")
            return 0
    
    def _calculate_enhanced_strength(self, level: SRLevel) -> float:
        """Calculate enhanced strength score for S/R level."""
        try:
            # Base strength from original calculation
            base_strength = level.strength
            
            # Boost for multiple touches
            touch_boost = min(level.touch_count * 0.1, 0.3)
            
            # Boost for volume confirmation
            volume_boost = level.volume_confirmation_score * 0.2
            
            # Boost for consistency
            consistency_boost = level.consistency_score * 0.2
            
            # Boost for confluence
            confluence_boost = level.confluence_score * 0.1
            
            # Penalty for failures
            failure_penalty = min(level.failure_count * 0.1, 0.3)
            
            # Special boosts
            special_boost = 0.0
            if level.pivot_level:
                special_boost += 0.1
            if level.psychological_level:
                special_boost += 0.05
            
            # Calculate final strength
            final_strength = (base_strength + touch_boost + volume_boost + 
                            consistency_boost + confluence_boost + special_boost - 
                            failure_penalty)
            
            return max(0.0, min(1.0, final_strength))
            
        except Exception as e:
            self.logger.warning(f"Enhanced strength calculation failed: {e}")
            return level.strength