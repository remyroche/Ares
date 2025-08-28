#!/usr/bin/env python3
"""
Centralized Support/Resistance Logic Module

This module provides a unified, centralized implementation of support/resistance
analysis that can be used by both feature engineering and analyst components.
Eliminates redundancy and provides consistent S/R calculations across the system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import system_logger
from src.utils.centralized_decorators import (
    handle_errors,
    validate_data_structure,
    monitor_feature_engineering
)


class SRType(Enum):
    """Support/Resistance level types."""
    PIVOT = "pivot"
    VOLUME_NODE = "volume_node"
    CONFLUENCE = "confluence"
    FIBONACCI = "fibonacci"
    PSYCHOLOGICAL = "psychological"


@dataclass
class SRLevel:
    """Support/Resistance level data structure."""
    price: float
    level_type: SRType
    strength: float  # 0.0 to 1.0
    touches: int
    volume: float
    age: int
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any]


class CentralizedSRAnalyzer:
    """
    Centralized Support/Resistance analyzer that provides unified S/R logic
    for use by feature engineering and analyst components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("CentralizedSRAnalyzer")
        
        # Configuration parameters
        self.pivot_window = config.get("sr_pivot_window", 20)
        self.volume_window = config.get("sr_volume_window", 50)
        self.tolerance_factor = config.get("sr_tolerance_factor", 0.1)
        self.min_touches = config.get("sr_min_touches", 2)
        self.min_strength = config.get("sr_min_strength", 0.3)
        
        # Cache for performance
        self._level_cache: Dict[str, List[SRLevel]] = {}
        self._cache_enabled = config.get("sr_cache_enabled", True)
        
    @handle_errors(
        exceptions=(Exception,),
        default_return={"supports": [], "resistances": [], "error": "S/R analysis failed"},
        context="centralized_sr_analysis"
    )
    @validate_data_structure(required_columns=["open", "high", "low", "close", "volume"])
    @monitor_feature_engineering
    def analyze_sr_levels(
        self, 
        df: pd.DataFrame, 
        current_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive S/R analysis that combines multiple methods.
        
        Args:
            df: OHLCV DataFrame
            current_price: Current market price (optional)
            
        Returns:
            Dictionary with supports, resistances, and analysis metadata
        """
        if df.empty or len(df) < self.pivot_window:
            return {"supports": [], "resistances": [], "error": "Insufficient data"}
            
        # Generate cache key
        cache_key = self._generate_cache_key(df)
        
        # Check cache
        if self._cache_enabled and cache_key in self._level_cache:
            cached_levels = self._level_cache[cache_key]
            return self._format_sr_results(cached_levels, current_price)
        
        try:
            # 1. Pivot-based S/R levels
            pivot_levels = self._calculate_pivot_levels(df)
            
            # 2. Volume-based S/R levels
            volume_levels = self._calculate_volume_levels(df)
            
            # 3. Fibonacci retracement levels
            fib_levels = self._calculate_fibonacci_levels(df)
            
            # 4. Psychological levels
            psych_levels = self._calculate_psychological_levels(df)
            
            # 5. Combine and filter levels
            all_levels = self._combine_sr_levels(
                pivot_levels, volume_levels, fib_levels, psych_levels
            )
            
            # 6. Filter by strength and relevance
            filtered_levels = self._filter_sr_levels(all_levels, current_price)
            
            # Cache results
            if self._cache_enabled:
                self._level_cache[cache_key] = filtered_levels
                
            return self._format_sr_results(filtered_levels, current_price)
            
        except Exception as e:
            self.logger.error(f"Error in S/R analysis: {e}")
            return {"supports": [], "resistances": [], "error": str(e)}
    
    def _generate_cache_key(self, df: pd.DataFrame) -> str:
        """Generate cache key for DataFrame."""
        try:
            # Use last few rows and shape for cache key
            last_rows = df.tail(10)
            key_data = f"{df.shape}_{last_rows['close'].iloc[-1]:.6f}_{len(df)}"
            return str(hash(key_data))
        except Exception:
            return str(hash(str(df.shape)))
    
    def _calculate_pivot_levels(self, df: pd.DataFrame) -> List[SRLevel]:
        """Calculate pivot-based support and resistance levels."""
        levels = []
        
        try:
            # Calculate rolling pivots
            for i in range(self.pivot_window, len(df)):
                window = df.iloc[i-self.pivot_window:i+1]
                
                high = window['high'].max()
                low = window['low'].min()
                close = window['close'].iloc[-1]
                
                pivot = (high + low + close) / 3
                
                # Calculate S/R levels
                r1 = 2 * pivot - low
                r2 = pivot + (high - low)
                s1 = 2 * pivot - high
                s2 = pivot - (high - low)
                
                # Calculate strength metrics
                for level_price, level_type in [
                    (r1, SRType.PIVOT), (r2, SRType.PIVOT),
                    (s1, SRType.PIVOT), (s2, SRType.PIVOT)
                ]:
                    if level_price > 0:
                        strength = self._calculate_level_strength(window, level_price)
                        if strength >= self.min_strength:
                            levels.append(SRLevel(
                                price=level_price,
                                level_type=level_type,
                                strength=strength,
                                touches=self._count_touches(window, level_price),
                                volume=self._calculate_volume_near_level(window, level_price),
                                age=self._calculate_level_age(window, level_price),
                                confidence=strength * 0.8,  # Pivot confidence
                                metadata={"method": "pivot", "window": self.pivot_window}
                            ))
                            
        except Exception as e:
            self.logger.warning(f"Error calculating pivot levels: {e}")
            
        return levels
    
    def _calculate_volume_levels(self, df: pd.DataFrame) -> List[SRLevel]:
        """Calculate volume-based support and resistance levels."""
        levels = []
        
        try:
            # Use ATR for dynamic binning
            atr = self._calculate_atr(df, window=14)
            bin_size = max(atr * 0.25, 1e-6)
            
            # Create price bins
            min_price = df['low'].min()
            max_price = df['high'].max()
            bins = np.arange(min_price, max_price, bin_size)
            
            if len(bins) < 2:
                return levels
                
            # Calculate volume profile
            price_bins = pd.cut(df['close'], bins=bins, right=False)
            volume_by_bin = df.groupby(price_bins, observed=False)['volume'].sum()
            
            # Find high volume nodes
            if not volume_by_bin.empty:
                # Get top volume nodes
                top_nodes = volume_by_bin.nlargest(5)
                
                for bin_level, volume in top_nodes.items():
                    level_price = bin_level.mid
                    
                    # Calculate strength metrics
                    strength = self._calculate_volume_level_strength(df, level_price, volume)
                    touches = self._count_touches(df, level_price)
                    age = self._calculate_level_age(df, level_price)
                    
                    if strength >= self.min_strength and touches >= self.min_touches:
                        levels.append(SRLevel(
                            price=level_price,
                            level_type=SRType.VOLUME_NODE,
                            strength=strength,
                            touches=touches,
                            volume=volume,
                            age=age,
                            confidence=strength * 0.9,  # Volume levels are more reliable
                            metadata={"method": "volume", "volume": volume}
                        ))
                        
        except Exception as e:
            self.logger.warning(f"Error calculating volume levels: {e}")
            
        return levels
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> List[SRLevel]:
        """Calculate Fibonacci retracement levels."""
        levels = []
        
        try:
            # Find recent swing high and low
            window = df.tail(50)
            swing_high = window['high'].max()
            swing_low = window['low'].min()
            
            # Fibonacci ratios
            fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
            
            for ratio in fib_ratios:
                # Calculate retracement levels
                range_size = swing_high - swing_low
                
                # Support levels (from swing high)
                support_level = swing_high - (range_size * ratio)
                
                # Resistance levels (from swing low)
                resistance_level = swing_low + (range_size * ratio)
                
                # Add support level
                if support_level > 0:
                    strength = self._calculate_level_strength(df, support_level)
                    levels.append(SRLevel(
                        price=support_level,
                        level_type=SRType.FIBONACCI,
                        strength=strength * 0.7,  # Fibonacci levels are theoretical
                        touches=self._count_touches(df, support_level),
                        volume=self._calculate_volume_near_level(df, support_level),
                        age=self._calculate_level_age(df, support_level),
                        confidence=strength * 0.6,
                        metadata={"method": "fibonacci", "ratio": ratio}
                    ))
                
                # Add resistance level
                if resistance_level > 0:
                    strength = self._calculate_level_strength(df, resistance_level)
                    levels.append(SRLevel(
                        price=resistance_level,
                        level_type=SRType.FIBONACCI,
                        strength=strength * 0.7,
                        touches=self._count_touches(df, resistance_level),
                        volume=self._calculate_volume_near_level(df, resistance_level),
                        age=self._calculate_level_age(df, resistance_level),
                        confidence=strength * 0.6,
                        metadata={"method": "fibonacci", "ratio": ratio}
                    ))
                    
        except Exception as e:
            self.logger.warning(f"Error calculating Fibonacci levels: {e}")
            
        return levels
    
    def _calculate_psychological_levels(self, df: pd.DataFrame) -> List[SRLevel]:
        """Calculate psychological support/resistance levels."""
        levels = []
        
        try:
            current_price = df['close'].iloc[-1]
            
            # Common psychological levels (round numbers)
            psych_levels = []
            
            # Generate round number levels around current price
            base = 10 ** (int(np.log10(current_price)) - 1)
            for i in range(-5, 6):
                level = round(current_price / base + i) * base
                if level > 0:
                    psych_levels.append(level)
            
            # Add common levels like 100, 1000, etc.
            common_levels = [100, 1000, 10000, 50000, 100000]
            for level in common_levels:
                if abs(level - current_price) / current_price < 0.5:  # Within 50%
                    psych_levels.append(level)
            
            # Remove duplicates and sort
            psych_levels = sorted(list(set(psych_levels)))
            
            for level_price in psych_levels:
                strength = self._calculate_level_strength(df, level_price)
                touches = self._count_touches(df, level_price)
                
                if strength >= self.min_strength * 0.5:  # Lower threshold for psychological levels
                    levels.append(SRLevel(
                        price=level_price,
                        level_type=SRType.PSYCHOLOGICAL,
                        strength=strength * 0.5,  # Psychological levels are less reliable
                        touches=touches,
                        volume=self._calculate_volume_near_level(df, level_price),
                        age=self._calculate_level_age(df, level_price),
                        confidence=strength * 0.4,
                        metadata={"method": "psychological"}
                    ))
                    
        except Exception as e:
            self.logger.warning(f"Error calculating psychological levels: {e}")
            
        return levels
    
    def _combine_sr_levels(
        self, 
        pivot_levels: List[SRLevel],
        volume_levels: List[SRLevel],
        fib_levels: List[SRLevel],
        psych_levels: List[SRLevel]
    ) -> List[SRLevel]:
        """Combine and deduplicate S/R levels."""
        all_levels = pivot_levels + volume_levels + fib_levels + psych_levels
        
        if not all_levels:
            return []
        
        # Sort by price
        all_levels.sort(key=lambda x: x.price)
        
        # Deduplicate nearby levels
        deduplicated = []
        tolerance = self.tolerance_factor * np.mean([level.price for level in all_levels])
        
        for level in all_levels:
            # Check if this level is close to an existing one
            is_duplicate = False
            for existing in deduplicated:
                if abs(level.price - existing.price) <= tolerance:
                    # Merge levels - keep the stronger one
                    if level.strength > existing.strength:
                        # Replace existing with current
                        deduplicated.remove(existing)
                        deduplicated.append(level)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(level)
        
        return deduplicated
    
    def _filter_sr_levels(
        self, 
        levels: List[SRLevel], 
        current_price: Optional[float] = None
    ) -> List[SRLevel]:
        """Filter S/R levels by strength and relevance."""
        if not levels:
            return []
        
        # Filter by minimum strength
        filtered = [level for level in levels if level.strength >= self.min_strength]
        
        # Filter by minimum touches
        filtered = [level for level in filtered if level.touches >= self.min_touches]
        
        # If current price provided, prioritize nearby levels
        if current_price is not None:
            # Calculate distance from current price
            for level in filtered:
                distance = abs(level.price - current_price) / current_price
                # Boost confidence for nearby levels
                if distance < 0.05:  # Within 5%
                    level.confidence *= 1.2
                elif distance > 0.2:  # Beyond 20%
                    level.confidence *= 0.8
        
        # Sort by confidence
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit to top levels
        max_levels = self.config.get("sr_max_levels", 10)
        return filtered[:max_levels]
    
    def _format_sr_results(
        self, 
        levels: List[SRLevel], 
        current_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Format S/R results for output."""
        supports = []
        resistances = []
        
        if current_price is not None:
            for level in levels:
                if level.price < current_price:
                    supports.append({
                        "price": level.price,
                        "strength": level.strength,
                        "confidence": level.confidence,
                        "type": level.level_type.value,
                        "touches": level.touches,
                        "volume": level.volume,
                        "age": level.age,
                        "metadata": level.metadata
                    })
                else:
                    resistances.append({
                        "price": level.price,
                        "strength": level.strength,
                        "confidence": level.confidence,
                        "type": level.level_type.value,
                        "touches": level.touches,
                        "volume": level.volume,
                        "age": level.age,
                        "metadata": level.metadata
                    })
        else:
            # If no current price, classify based on recent price action
            recent_price = levels[0].price if levels else 0
            for level in levels:
                level_data = {
                    "price": level.price,
                    "strength": level.strength,
                    "confidence": level.confidence,
                    "type": level.level_type.value,
                    "touches": level.touches,
                    "volume": level.volume,
                    "age": level.age,
                    "metadata": level.metadata
                }
                
                if level.price < recent_price:
                    supports.append(level_data)
                else:
                    resistances.append(level_data)
        
        return {
            "supports": sorted(supports, key=lambda x: x["price"], reverse=True),
            "resistances": sorted(resistances, key=lambda x: x["price"]),
            "analysis_metadata": {
                "total_levels": len(levels),
                "support_count": len(supports),
                "resistance_count": len(resistances),
                "cache_hit": False  # Will be set by caller if needed
            }
        }
    
    def _calculate_level_strength(self, df: pd.DataFrame, level_price: float) -> float:
        """Calculate strength of a support/resistance level."""
        try:
            tolerance = self.tolerance_factor * level_price
            
            # Count touches
            touches = self._count_touches(df, level_price)
            
            # Calculate volume near level
            volume_near = self._calculate_volume_near_level(df, level_price)
            total_volume = df['volume'].sum()
            volume_ratio = volume_near / total_volume if total_volume > 0 else 0
            
            # Calculate age
            age = self._calculate_level_age(df, level_price)
            max_age = len(df)
            age_ratio = age / max_age if max_age > 0 else 0
            
            # Weighted strength calculation
            touch_strength = min(touches / 5.0, 1.0)  # Normalize touches
            volume_strength = min(volume_ratio * 10, 1.0)  # Normalize volume
            age_strength = min(age_ratio * 2, 1.0)  # Normalize age
            
            # Combined strength (weighted average)
            strength = (
                touch_strength * 0.4 +
                volume_strength * 0.4 +
                age_strength * 0.2
            )
            
            return max(0.0, min(1.0, strength))
            
        except Exception:
            return 0.0
    
    def _count_touches(self, df: pd.DataFrame, level_price: float) -> int:
        """Count how many times price touched a level."""
        try:
            tolerance = self.tolerance_factor * level_price
            touches = 0
            
            for i in range(1, len(df)):
                prev_high = df['high'].iloc[i-1]
                prev_low = df['low'].iloc[i-1]
                curr_high = df['high'].iloc[i]
                curr_low = df['low'].iloc[i]
                
                # Check if price crossed the level
                if (prev_low < level_price < curr_high) or (prev_high > level_price > curr_low):
                    touches += 1
                    
            return touches
            
        except Exception:
            return 0
    
    def _calculate_volume_near_level(self, df: pd.DataFrame, level_price: float) -> float:
        """Calculate volume near a support/resistance level."""
        try:
            tolerance = self.tolerance_factor * level_price
            volume_near = 0.0
            
            for i in range(len(df)):
                if abs(df['close'].iloc[i] - level_price) <= tolerance:
                    volume_near += df['volume'].iloc[i]
                    
            return volume_near
            
        except Exception:
            return 0.0
    
    def _calculate_level_age(self, df: pd.DataFrame, level_price: float) -> int:
        """Calculate age of a support/resistance level."""
        try:
            tolerance = self.tolerance_factor * level_price
            
            for i in range(len(df)):
                if abs(df['close'].iloc[i] - level_price) <= tolerance:
                    return len(df) - i
                    
            return 0
            
        except Exception:
            return 0
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=window).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else df['close'].std()
            
        except Exception:
            return df['close'].std()
    
    def clear_cache(self) -> None:
        """Clear the S/R level cache."""
        self._level_cache.clear()
        self.logger.info("S/R level cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_enabled": self._cache_enabled,
            "cache_size": len(self._level_cache),
            "cache_keys": list(self._level_cache.keys())
        }