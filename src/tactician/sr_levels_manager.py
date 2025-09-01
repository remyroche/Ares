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
import warnings
from datetime import datetime
from typing import Dict, Any, List, Optional

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
        """Create SRLevel from dictionary."""
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


class SRLevelsManager:
    """
    Comprehensive Support/Resistance Levels Manager.
    
    Manages SR levels with persistent storage, continuous updates,
    and comprehensive level information including age, strength, volume, etc.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SR Levels Manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("SRLevelsManager")

        # Configuration
        self.levels_config = config.get("sr_levels_manager", {})
        self.storage_path = Path(self.levels_config.get("storage_path", "data/sr_levels"))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Level storage
        self.support_levels: List[SRLevel] = []
        self.resistance_levels: List[SRLevel] = []
        self.level_cache: Dict[str, SRLevel] = {}

        # SR predictor for level detection
        self.sr_predictor: Optional[SRBreakoutPredictor] = None

    async def initialize(self) -> bool:
        """
        Initialize the SR Levels Manager.

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing SR Levels Manager...")

            # Initialize SR predictor
            self.sr_predictor = SRBreakoutPredictor(self.config)
            await self.sr_predictor.initialize()

            # Load existing levels
            await self.load_levels()

            self.logger.info("✅ SR Levels Manager initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ SR Levels Manager initialization failed: {e}")
            return False

    async def load_levels(self) -> None:
        """Load SR levels from persistent storage."""
        try:
            support_file = self.storage_path / "support_levels.json"
            resistance_file = self.storage_path / "resistance_levels.json"

            # Load support levels
            if support_file.exists():
                with open(support_file, 'r') as f:
                    data = json.load(f)
                    self.support_levels = [SRLevel.from_dict(item) for item in data]
                    self.logger.info(f"Loaded {len(self.support_levels)} support levels")

            # Load resistance levels
            if resistance_file.exists():
                with open(resistance_file, 'r') as f:
                    data = json.load(f)
                    self.resistance_levels = [SRLevel.from_dict(item) for item in data]
                    self.logger.info(f"Loaded {len(self.resistance_levels)} resistance levels")

        except Exception as e:
            self.logger.error(f"Error loading SR levels: {e}")

    async def save_levels(self) -> None:
        """Save SR levels to persistent storage."""
        try:
            # Save support levels
            support_file = self.storage_path / "support_levels.json"
            with open(support_file, 'w') as f:
                json.dump([level.to_dict() for level in self.support_levels], f, indent=2)

            # Save resistance levels
            resistance_file = self.storage_path / "resistance_levels.json"
            with open(resistance_file, 'w') as f:
                json.dump([level.to_dict() for level in self.resistance_levels], f, indent=2)

            self.logger.info("SR levels saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving SR levels: {e}")

    async def update_levels(self, market_data: pd.DataFrame) -> None:
        """
        Update SR levels based on new market data.

        Args:
            market_data: New market data
        """
        try:
            if not self.sr_predictor:
                return

            # Get new levels from SR predictor
            new_levels = await self.sr_predictor.detect_levels(market_data)

            # Update existing levels
            for level_data in new_levels:
                await self._update_or_add_level(level_data)

            # Save updated levels
            await self.save_levels()

        except Exception as e:
            self.logger.error(f"Error updating SR levels: {e}")

    async def _update_or_add_level(self, level_data: Dict[str, Any]) -> None:
        """
        Update existing level or add new level.

        Args:
            level_data: Level data from SR predictor
        """
        try:
            price = level_data["price"]
            level_type = level_data["level_type"]
            method = level_data["method"]

            # Check if level already exists
            existing_level = self._find_existing_level(price, level_type, method)

            if existing_level:
                # Update existing level
                existing_level.touch_count += 1
                existing_level.last_touch = datetime.now()
                existing_level.strength = level_data.get("strength", existing_level.strength)
                existing_level.confidence = level_data.get("confidence", existing_level.confidence)
            else:
                # Add new level
                new_level = SRLevel(
                    price=price,
                    level_type=level_type,
                    method=method,
                    data_source=level_data.get("data_source", "price"),
                    timestamp=datetime.now(),
                    strength=level_data.get("strength", 0.5),
                    volume=level_data.get("volume", 0.0),
                    touch_count=1,
                    confidence=level_data.get("confidence", 0.5),
                    metadata=level_data.get("metadata", {})
                )

                if level_type == "support":
                    self.support_levels.append(new_level)
                else:
                    self.resistance_levels.append(new_level)

        except Exception as e:
            self.logger.error(f"Error updating/adding level: {e}")

    def _find_existing_level(self, price: float, level_type: str, method: str) -> Optional[SRLevel]:
        """
        Find existing level by price, type, and method.

        Args:
            price: Level price
            level_type: Level type (support/resistance)
            method: Detection method

        Returns:
            SRLevel: Existing level or None
        """
        try:
            levels = self.support_levels if level_type == "support" else self.resistance_levels
            
            for level in levels:
                if (abs(level.price - price) / price < 0.001 and 
                    level.method == method):
                    return level

            return None

        except Exception as e:
            self.logger.error(f"Error finding existing level: {e}")
            return None

    def get_nearby_levels(self, price: float, distance_pct: float = 0.01) -> Dict[str, List[SRLevel]]:
        """
        Get SR levels near a given price.

        Args:
            price: Current price
            distance_pct: Distance as percentage of price

        Returns:
            Dict: Nearby support and resistance levels
        """
        try:
            nearby_support = []
            nearby_resistance = []

            distance = price * distance_pct

            # Find nearby support levels
            for level in self.support_levels:
                if abs(level.price - price) <= distance:
                    nearby_support.append(level)

            # Find nearby resistance levels
            for level in self.resistance_levels:
                if abs(level.price - price) <= distance:
                    nearby_resistance.append(level)

            return {
                "support": sorted(nearby_support, key=lambda x: x.strength, reverse=True),
                "resistance": sorted(nearby_resistance, key=lambda x: x.strength, reverse=True)
            }

        except Exception as e:
            self.logger.error(f"Error getting nearby levels: {e}")
            return {"support": [], "resistance": []}

    def get_level_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about current SR levels.

        Returns:
            Dict: Level statistics
        """
        try:
            total_support = len(self.support_levels)
            total_resistance = len(self.resistance_levels)
            
            avg_support_strength = np.mean([level.strength for level in self.support_levels]) if self.support_levels else 0
            avg_resistance_strength = np.mean([level.strength for level in self.resistance_levels]) if self.resistance_levels else 0

            return {
                "total_support_levels": total_support,
                "total_resistance_levels": total_resistance,
                "avg_support_strength": avg_support_strength,
                "avg_resistance_strength": avg_resistance_strength,
                "total_levels": total_support + total_resistance
            }

        except Exception as e:
            self.logger.error(f"Error getting level statistics: {e}")
            return {}

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            # Save levels before cleanup
            await self.save_levels()

            if self.sr_predictor:
                await self.sr_predictor.cleanup()

            self.logger.info("✅ SR Levels Manager cleanup completed")

        except Exception as e:
            self.logger.error(f"❌ SR Levels Manager cleanup failed: {e}")