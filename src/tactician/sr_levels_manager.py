#!/usr/bin/env python3
"""
SR Levels Manager - Comprehensive Support/Resistance Level Management

This module provides:
1. SR level calculation based on backtesting data
2. Continuous updates during live trading
3. Comprehensive level information (age, strength, volume, etc.)
4. Price vs VWAP comparison logic
5. Persistent storage and retrieval
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor

logger = system_logger.getChild("SRLevelsManager")


class SRLevel:
    """Individual Support/Resistance Level with comprehensive information."""

    def __init__(
        self,
        price: float,
        level_type: str,  # "support" or "resistance"
        method: str,
        data_source: str,  # "price" or "vwap"
        timestamp: datetime,
        strength: float = 0.5,
        volume: float = 0.0,
        touch_count: int = 0,
        age_hours: float = 0.0,
        bounce_rate: float = 0.0,
        isolation_score: float = 0.0,
        confidence: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.price = price
        self.level_type = level_type
        self.method = method
        self.data_source = data_source
        self.timestamp = timestamp
        self.strength = strength
        self.volume = volume
        self.touch_count = touch_count
        self.age_hours = age_hours
        self.bounce_rate = bounce_rate
        self.isolation_score = isolation_score
        self.confidence = confidence
        self.metadata = metadata or {}

        # Calculated fields
        self.last_touch = timestamp
        self.total_touches = touch_count
        self.creation_time = timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert level to dictionary for storage."""
        return {
            "price": self.price,
            "level_type": self.level_type,
            "method": self.method,
            "data_source": self.data_source,
            "timestamp": self.timestamp.isoformat(),
            "strength": self.strength,
            "volume": self.volume,
            "touch_count": self.touch_count,
            "age_hours": self.age_hours,
            "bounce_rate": self.bounce_rate,
            "isolation_score": self.isolation_score,
            "confidence": self.confidence,
            "last_touch": self.last_touch.isoformat(),
            "total_touches": self.total_touches,
            "creation_time": self.creation_time.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SRLevel':
        """Create level from dictionary."""
        return cls(
            price=data["price"],
            level_type=data["level_type"],
            method=data["method"],
            data_source=data["data_source"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            strength=data.get("strength", 0.5),
            volume=data.get("volume", 0.0),
            touch_count=data.get("touch_count", 0),
            age_hours=data.get("age_hours", 0.0),
            bounce_rate=data.get("bounce_rate", 0.0),
            isolation_score=data.get("isolation_score", 0.0),
            confidence=data.get("confidence", 0.5),
            metadata=data.get("metadata", {})
        )

    def update_touch(self, current_time: datetime, price: float, volume: float = 0.0):
        """Update level with new touch information."""
        self.last_touch = current_time
        self.touch_count += 1
        self.total_touches += 1
        self.volume = max(self.volume, volume)

        # Update age
        self.age_hours = (current_time - self.creation_time).total_seconds() / 3600

        # Update strength based on touch count
        self.strength = min(1.0, 0.5 + (self.touch_count * 0.1))

    def calculate_quality_score(self) -> float:
        """Calculate overall quality score for this level."""
        score = 0.0

        # Base score from strength
        score += self.strength * 0.3

        # Touch count bonus
        score += min(0.3, self.touch_count * 0.05)

        # Age bonus (older levels get slight bonus)
        score += min(0.1, self.age_hours / 1000)

        # Bounce rate bonus
        score += self.bounce_rate * 0.2

        # Isolation score bonus
        score += self.isolation_score * 0.1

        return min(1.0, score)


class SRLevelsManager:
    """
    Comprehensive SR Levels Manager for trading intelligence.

    Features:
    - Calculate SR levels from backtesting data
    - Continuous updates during live trading
    - Persistent storage with comprehensive metadata
    - Price vs VWAP comparison
    - Level quality scoring and filtering
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize SR Levels Manager."""
        self.config = config
        self.logger = system_logger.getChild("SRLevelsManager")

        # Configuration
        self.sr_config = config.get("sr_levels_manager", {})
        self.storage_path = Path(self.sr_config.get("storage_path", "data/sr_levels"))
        self.max_levels = self.sr_config.get("max_levels", 50)
        self.min_strength = self.sr_config.get("min_strength", 0.3)
        self.proximity_threshold = self.sr_config.get("proximity_threshold", 0.005)

        # Storage
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.levels_file = self.storage_path / "sr_levels.json"
        self.history_file = self.storage_path / "sr_levels_history.json"

        # Current levels
        self.support_levels: List[SRLevel] = []
        self.resistance_levels: List[SRLevel] = []

        # SR predictor for calculations
        self.sr_predictor: Optional[SRBreakoutPredictor] = None

        # Performance tracking
        self.last_update = datetime.now()
        self.update_count = 0

    async def calculate_sr_levels_from_backtest(
        self,
        market_data: pd.DataFrame,
        timeframe: str = "1m"
    ) -> Dict[str, List[SRLevel]]:
        """
        Calculate SR levels from backtesting data using SR breakout predictor logic.

        Args:
            market_data: Historical market data
            timeframe: Data timeframe

        Returns:
            Dictionary with support and resistance levels
        """
        try:
            self.logger.info(f"🔍 Calculating SR levels from backtest data ({len(market_data)} points)")

            # Get current price for context
            current_price = market_data['close'].iloc[-1]

            # Use SR breakout predictor's comprehensive detection methods
            support_levels = []
            resistance_levels = []

            # Method 1: Use the main SR context method (comprehensive)
            try:
                sr_context = await self.sr_predictor.get_sr_context(market_data, current_price)

                # Process support levels from context
                for level_data in sr_context.get("support_levels", []):
                    level = self._create_sr_level_from_data(level_data, "support")
                    if level:
                        support_levels.append(level)

                # Process resistance levels from context
                for level_data in sr_context.get("resistance_levels", []):
                    level = self._create_sr_level_from_data(level_data, "resistance")
                    if level:
                        resistance_levels.append(level)

                self.logger.info(f"✅ Retrieved {len(support_levels)} support and {len(resistance_levels)} resistance levels from SR context")

            except Exception as e:
                self.logger.warning(f"⚠️ SR context method failed: {e}")

            # Method 2: Use direct detection methods if context method didn't provide enough levels
            if len(support_levels) < 3 or len(resistance_levels) < 3:
                self.logger.info("🔄 Using direct detection methods for additional levels")

                try:
                    # Direct support level detection
                    direct_support = await self.sr_predictor._detect_support_levels(market_data)
                    for level_data in direct_support:
                        level = self._create_sr_level_from_data(level_data, "support")
                        if level and not self._level_exists(level, support_levels):
                            support_levels.append(level)

                    # Direct resistance level detection
                    direct_resistance = await self.sr_predictor._detect_resistance_levels(market_data)
                    for level_data in direct_resistance:
                        level = self._create_sr_level_from_data(level_data, "resistance")
                        if level and not self._level_exists(level, resistance_levels):
                            resistance_levels.append(level)

                    self.logger.info(f"✅ Added {len(direct_support)} direct support and {len(direct_resistance)} direct resistance levels")

                except Exception as e:
                    self.logger.warning(f"⚠️ Direct detection methods failed: {e}")

            # Method 3: Use specific detection methods for comprehensive coverage
            if len(support_levels) < 5 or len(resistance_levels) < 5:
                self.logger.info("🔄 Using specific detection methods for comprehensive coverage")

                detection_methods = ["fractal", "volume", "pivot", "atr"]

                for method in detection_methods:
                    try:
                        # Temporarily set detection method
                        original_method = self.sr_predictor.sr_detection_method
                        self.sr_predictor.sr_detection_method = method

                        # Detect support levels with this method
                        method_support = await self.sr_predictor._detect_support_levels(market_data)
                        for level_data in method_support:
                            level = self._create_sr_level_from_data(level_data, "support")
                            if level and not self._level_exists(level, support_levels):
                                level.metadata["detection_method"] = method
                                support_levels.append(level)

                        # Detect resistance levels with this method
                        method_resistance = await self.sr_predictor._detect_resistance_levels(market_data)
                        for level_data in method_resistance:
                            level = self._create_sr_level_from_data(level_data, "resistance")
                            if level and not self._level_exists(level, resistance_levels):
                                level.metadata["detection_method"] = method
                                resistance_levels.append(level)

                        # Restore original method
                        self.sr_predictor.sr_detection_method = original_method

                        self.logger.info(f"✅ Added {len(method_support)} {method} support and {len(method_resistance)} {method} resistance levels")

                    except Exception as e:
                        self.logger.warning(f"⚠️ {method} detection method failed: {e}")
                        # Restore original method on error
                        self.sr_predictor.sr_detection_method = original_method

            # Filter and deduplicate levels
            support_levels = self._filter_and_deduplicate_levels(support_levels)
            resistance_levels = self._filter_and_deduplicate_levels(resistance_levels)

            # Store levels
            self.support_levels = support_levels
            self.resistance_levels = resistance_levels

            # Save to storage
            await self.save_levels()

            self.logger.info(f"✅ Final calculation: {len(support_levels)} support and {len(resistance_levels)} resistance levels")

            return {
                "support_levels": support_levels,
                "resistance_levels": resistance_levels
            }

        except Exception as e:
            self.logger.error(f"❌ Error calculating SR levels from backtest: {e}")
            return {"support_levels": [], "resistance_levels": []}

    def compare_price_vs_vwap_predictions(
        self,
        price_levels: List[SRLevel],
        vwap_levels: List[SRLevel]
    ) -> Dict[str, Any]:
        """
        Compare price vs VWAP SR level predictions.

        Args:
            price_levels: Levels detected using price data
            vwap_levels: Levels detected using VWAP data

        Returns:
            Comparison analysis
        """
        try:
            self.logger.info("🔍 Comparing price vs VWAP SR level predictions")

            # Count levels by type
            price_support = [l for l in price_levels if l.level_type == "support"]
            price_resistance = [l for l in price_levels if l.level_type == "resistance"]
            vwap_support = [l for l in vwap_levels if l.level_type == "support"]
            vwap_resistance = [l for l in vwap_levels if l.level_type == "resistance"]

            # Calculate quality metrics
            price_quality = self._calculate_levels_quality(price_levels)
            vwap_quality = self._calculate_levels_quality(vwap_levels)

            # Calculate overlap
            overlap_analysis = self._calculate_levels_overlap(price_levels, vwap_levels)

            comparison = {
                "level_counts": {
                    "price": {
                        "support": len(price_support),
                        "resistance": len(price_resistance),
                        "total": len(price_levels)
                    },
                    "vwap": {
                        "support": len(vwap_support),
                        "resistance": len(vwap_resistance),
                        "total": len(vwap_levels)
                    }
                },
                "quality_metrics": {
                    "price": price_quality,
                    "vwap": vwap_quality
                },
                "overlap_analysis": overlap_analysis,
                "recommendations": self._generate_comparison_recommendations(
                    price_quality, vwap_quality, overlap_analysis
                ),
                "timestamp": datetime.now().isoformat()
            }

            self.logger.info(f"✅ Price vs VWAP comparison completed")
            return comparison

        except Exception as e:
            self.logger.error(f"❌ Error comparing price vs VWAP predictions: {e}")
            return {}

    def _filter_and_deduplicate_levels(self, levels: List[SRLevel]) -> List[SRLevel]:
        """Filter and deduplicate levels based on quality and proximity."""
        if not levels:
            return []

        # Sort by quality score
        levels.sort(key=lambda x: x.calculate_quality_score(), reverse=True)

        # Filter by minimum strength
        levels = [l for l in levels if l.strength >= self.min_strength]

        # Deduplicate by proximity
        filtered = []
        for level in levels:
            is_duplicate = False
            for existing in filtered:
                if self._is_price_near_level(level.price, existing.price):
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(level)

        # Limit total levels
        return filtered[:self.max_levels]

    def _is_price_near_level(self, price1: float, price2: float) -> bool:
        """Check if two prices are near each other."""
        if price2 == 0:
            return False
        return abs(price1 - price2) / price2 < self.proximity_threshold

    def _find_nearest_level(self, price: float, levels: List[SRLevel]) -> Optional[SRLevel]:
        """Find the nearest level to a given price."""
        if not levels:
            return None

        nearest = min(levels, key=lambda x: abs(x.price - price))
        return nearest

    def _calculate_proximity(self, price: float, level: Optional[SRLevel]) -> float:
        """Calculate proximity to a level (0 = at level, 1 = far away)."""
        if not level or level.price == 0:
            return 1.0

        return abs(price - level.price) / level.price

    def _calculate_levels_quality(self, levels: List[SRLevel]) -> Dict[str, float]:
        """Calculate quality metrics for a set of levels."""
        if not levels:
            return {"avg_strength": 0.0, "avg_confidence": 0.0, "avg_quality": 0.0}

        avg_strength = np.mean([l.strength for l in levels])
        avg_confidence = np.mean([l.confidence for l in levels])
        avg_quality = np.mean([l.calculate_quality_score() for l in levels])

        return {
            "avg_strength": avg_strength,
            "avg_confidence": avg_confidence,
            "avg_quality": avg_quality
        }

    def _calculate_levels_overlap(self, levels1: List[SRLevel], levels2: List[SRLevel]) -> Dict[str, Any]:
        """Calculate overlap between two sets of levels."""
        if not levels1 or not levels2:
            return {"overlap_count": 0, "overlap_rate": 0.0, "overlap_details": []}

        overlap_count = 0
        overlap_details = []

        for l1 in levels1:
            for l2 in levels2:
                if (l1.level_type == l2.level_type and
                    self._is_price_near_level(l1.price, l2.price)):
                    overlap_count += 1
                    overlap_details.append({
                        "level1": l1.to_dict(),
                        "level2": l2.to_dict(),
                        "price_difference": abs(l1.price - l2.price)
                    })

        overlap_rate = overlap_count / min(len(levels1), len(levels2)) if min(len(levels1), len(levels2)) > 0 else 0.0

        return {
            "overlap_count": overlap_count,
            "overlap_rate": overlap_rate,
            "overlap_details": overlap_details
        }

    def _create_sr_level_from_data(self, level_data: Dict[str, Any], level_type: str) -> Optional[SRLevel]:
        """Create SRLevel object from level data dictionary."""
        try:
            if not level_data or not isinstance(level_data, dict):
                return None

            # Extract timestamp
            timestamp = level_data.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            elif timestamp is None:
                timestamp = datetime.now()

            # Create SRLevel object
            level = SRLevel(
                price=level_data.get("price", 0),
                level_type=level_type,
                method=level_data.get("method", "unknown"),
                data_source=level_data.get("data_source", "price"),
                timestamp=timestamp,
                strength=level_data.get("enhanced_strength", level_data.get("strength", 0.5)),
                volume=level_data.get("volume", 0.0),
                touch_count=level_data.get("touch_count", 0),
                age_hours=level_data.get("age_hours", 0.0),
                bounce_rate=level_data.get("bounce_rate", 0.0),
                isolation_score=level_data.get("isolation_score", 0.0),
                confidence=level_data.get("confidence", 0.5),
                metadata=level_data.get("metadata", {})
            )

            return level

        except Exception as e:
            self.logger.error(f"❌ Error creating SR level from data: {e}")
            return None

    def _level_exists(self, new_level: SRLevel, existing_levels: List[SRLevel]) -> bool:
        """Check if a level already exists in the list based on price proximity."""
        try:
            for existing_level in existing_levels:
                if (existing_level.level_type == new_level.level_type and
                    self._is_price_near_level(new_level.price, existing_level.price)):
                    return True
            return False

        except Exception as e:
            self.logger.error(f"❌ Error checking level existence: {e}")
            return False

    def _generate_comparison_recommendations(
        self,
        price_quality: Dict[str, float],
        vwap_quality: Dict[str, float],
        overlap_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on comparison analysis."""
        recommendations = []

        # Quality-based recommendations
        if price_quality["avg_quality"] > vwap_quality["avg_quality"]:
            recommendations.append("Price-based detection shows higher quality - consider prioritizing price data")
        elif vwap_quality["avg_quality"] > price_quality["avg_quality"]:
            recommendations.append("VWAP-based detection shows higher quality - consider prioritizing VWAP data")

        # Overlap-based recommendations
        if overlap_analysis["overlap_rate"] < 0.3:
            recommendations.append("Low overlap between approaches - consider adjusting detection parameters")
        elif overlap_analysis["overlap_rate"] > 0.8:
            recommendations.append("High overlap between approaches - both methods are detecting similar levels")

        # General recommendations
        if price_quality["avg_quality"] < 0.5:
            recommendations.append("Price-based detection quality is low - review detection parameters")
        if vwap_quality["avg_quality"] < 0.5:
            recommendations.append("VWAP-based detection quality is low - review VWAP calculation")

        return recommendations

    async def save_levels(self):
        """Save current levels to storage."""
        try:
            data = {
                "support_levels": [level.to_dict() for level in self.support_levels],
                "resistance_levels": [level.to_dict() for level in self.resistance_levels],
                "last_update": self.last_update.isoformat(),
                "update_count": self.update_count
            }

            with open(self.levels_file, 'w') as f:
                json.dump(data, f, indent=2)

            # Save to history
            await self._save_to_history(data)

        except Exception as e:
            self.logger.error(f"❌ Error saving SR levels: {e}")

    async def load_levels(self):
        """Load levels from storage."""
        try:
            if not self.levels_file.exists():
                self.logger.info("No existing SR levels found, starting fresh")
                return

            with open(self.levels_file, 'r') as f:
                data = json.load(f)

            # Load support levels
            self.support_levels = [
                SRLevel.from_dict(level_data)
                for level_data in data.get("support_levels", [])
            ]

            # Load resistance levels
            self.resistance_levels = [
                SRLevel.from_dict(level_data)
                for level_data in data.get("resistance_levels", [])
            ]

            # Load metadata
            self.last_update = datetime.fromisoformat(data.get("last_update", datetime.now().isoformat()))
            self.update_count = data.get("update_count", 0)

            self.logger.info(f"✅ Loaded {len(self.support_levels)} support and {len(self.resistance_levels)} resistance levels")

        except Exception as e:
            self.logger.error(f"❌ Error loading SR levels: {e}")

    async def _save_to_history(self, data: Dict[str, Any]):
        """Save current state to history file."""
        try:
            history_data = []

            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    history_data = json.load(f)

            # Add current state to history
            history_data.append({
                "timestamp": datetime.now().isoformat(),
                "data": data
            })

            # Keep only last 100 entries
            if len(history_data) > 100:
                history_data = history_data[-100:]

            with open(self.history_file, 'w') as f:
                json.dump(history_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"❌ Error saving to history: {e}")

