"""
Price Level Bank System

This module provides the PriceLevelBank class, which serves as a centralized
repository for historical price levels and their associated tags/features.
It enables efficient storage, querying, and reuse of price level data across
different feature generators and ML training processes.
"""

import logging
import pickle
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class PriceLevelData:
    """Data structure for a single price level entry."""
    price: float
    level_pct: float
    symbol: str
    timeframe: str
    timestamp: pd.Timestamp

    # Historical tags (computed from past data)
    historical_crossings: int = 0
    historical_bounces: int = 0
    historical_volume: float = 0.0
    historical_touch_density: float = 0.0
    historical_time_decay: float = 0.0
    historical_success_rate: float = 0.0

    # Additional metadata
    strength_score: float = 0.0
    recency_score: float = 0.0
    clustering_score: float = 0.0
    momentum_score: float = 0.0

    # Time-based features
    session_type: Optional[str] = None  # 'asian', 'european', 'us'
    day_of_week: Optional[int] = None
    hour_of_day: Optional[int] = None

    # Statistical measures
    significance_level: float = 0.0  # 0-1 scale
    confidence_interval: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert timestamp to string
        if isinstance(data['timestamp'], pd.Timestamp):
            data['timestamp'] = data['timestamp'].isoformat()
        # Convert confidence interval tuple to list
        data['confidence_interval'] = list(data['confidence_interval'])
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PriceLevelData':
        """Create from dictionary."""
        # Convert timestamp string back to Timestamp
        if isinstance(data['timestamp'], str):
            data['timestamp'] = pd.Timestamp(data['timestamp'])
        # Convert confidence interval list back to tuple
        if isinstance(data['confidence_interval'], list):
            data['confidence_interval'] = tuple(data['confidence_interval'])
        return cls(**data)

@dataclass
class PriceLevelBankConfig:
    """Configuration for the price level bank."""
    storage_path: str = "./data/price_level_bank"
    enable_persistence: bool = True
    auto_save_interval: int = 100  # Save every N operations
    max_levels_per_symbol: int = 10000
    default_lookback_window: int = 200
    cache_size: int = 1000
    enable_compression: bool = True

class PriceLevelBank:
    """
    Centralized bank for storing and managing historical price levels with their tags.

    This system provides:
    - Efficient storage of price levels and their historical characteristics
    - Fast lookup and querying capabilities
    - Persistence to disk for reuse across sessions
    - Integration with feature generation pipeline
    - Support for multiple symbols and timeframes
    """

    def __init__(self, config: Optional[PriceLevelBankConfig] = None):
        """
        Initialize the price level bank.

        Args:
            config: Bank configuration
        """
        self.config = config or PriceLevelBankConfig()
        self.logger = logger.getChild('PriceLevelBank')

        # Storage structures
        self.levels: Dict[str, PriceLevelData] = {}  # level_id -> PriceLevelData
        self.symbol_index: Dict[str, List[str]] = {}  # symbol -> list of level_ids
        self.price_index: Dict[str, Dict[float, List[str]]] = {}  # symbol -> price -> level_ids
        self.timeframe_index: Dict[str, Dict[str, List[str]]] = {}  # symbol -> timeframe -> level_ids

        # Metadata
        self.metadata: Dict[str, Any] = {
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'total_levels': 0,
            'symbols': set(),
            'timeframes': set()
        }

        # Operation counters for auto-save
        self.operation_count = 0

        # Cache for recent queries
        self.query_cache: Dict[str, Any] = {}

        # Ensure storage directory exists
        if self.config.enable_persistence:
            Path(self.config.storage_path).mkdir(parents=True, exist_ok=True)

        # Load existing data if available
        self._load_from_disk()

        self.logger.info("✅ Price Level Bank initialized")
        self.logger.info(f"📊 Storage: {self.config.storage_path}, "
                        f"Persistence: {self.config.enable_persistence}")

    def _generate_level_id(self, symbol: str, timeframe: str, price: float, level_pct: float) -> str:
        """Generate unique ID for a price level."""
        data = f"{symbol}_{timeframe}_{price}_{level_pct}"
        return hashlib.md5(data.encode()).hexdigest()[:16]

    def add_level(self, level_data: PriceLevelData) -> str:
        """
        Add a price level to the bank.

        Args:
            level_data: Price level data

        Returns:
            Unique level ID
        """
        level_id = self._generate_level_id(
            level_data.symbol,
            level_data.timeframe,
            level_data.price,
            level_data.level_pct
        )

        # Check if level already exists
        if level_id in self.levels:
            self.logger.debug(f"Level {level_id} already exists, updating...")
            # Update existing level
            existing = self.levels[level_id]
            # Merge data, keeping newer values for computed fields
            for field, value in level_data.__dict__.items():
                if value != 0 and value is not None:  # Don't overwrite with empty values
                    setattr(existing, field, value)
        else:
            # Add new level
            self.levels[level_id] = level_data

            # Update indices
            self._add_to_symbol_index(level_id, level_data.symbol)
            self._add_to_price_index(level_id, level_data.symbol, level_data.price)
            self._add_to_timeframe_index(level_id, level_data.symbol, level_data.timeframe)

            # Update metadata
            self.metadata['total_levels'] += 1
            self.metadata['symbols'].add(level_data.symbol)
            self.metadata['timeframes'].add(level_data.timeframe)

        # Auto-save if enabled
        self.operation_count += 1
        if (self.config.enable_persistence and
            self.operation_count % self.config.auto_save_interval == 0):
            self.save_to_disk()

        return level_id

    def add_levels(self, levels: List[PriceLevelData]) -> List[str]:
        """
        Add multiple price levels to the bank.

        Args:
            levels: List of price level data

        Returns:
            List of level IDs
        """
        level_ids = []
        for level in levels:
            level_id = self.add_level(level)
            level_ids.append(level_id)
        return level_ids

    def get_level(self, level_id: str) -> Optional[PriceLevelData]:
        """
        Get a price level by ID.

        Args:
            level_id: Unique level identifier

        Returns:
            Price level data or None if not found
        """
        return self.levels.get(level_id)

    def get_levels_by_symbol(self, symbol: str, limit: Optional[int] = None) -> List[PriceLevelData]:
        """
        Get all price levels for a symbol.

        Args:
            symbol: Trading symbol
            limit: Maximum number of levels to return

        Returns:
            List of price level data
        """
        if symbol not in self.symbol_index:
            return []

        level_ids = self.symbol_index[symbol]
        if limit:
            level_ids = level_ids[:limit]

        return [self.levels[lid] for lid in level_ids if lid in self.levels]

    def get_levels_by_price_range(self,
                                symbol: str,
                                min_price: float,
                                max_price: float,
                                limit: Optional[int] = None) -> List[PriceLevelData]:
        """
        Get price levels within a price range.

        Args:
            symbol: Trading symbol
            min_price: Minimum price
            max_price: Maximum price
            limit: Maximum number of levels to return

        Returns:
            List of price level data
        """
        if symbol not in self.price_index:
            return []

        levels = []
        price_dict = self.price_index[symbol]

        for price in price_dict:
            if min_price <= price <= max_price:
                for level_id in price_dict[price]:
                    if level_id in self.levels:
                        levels.append(self.levels[level_id])

        if limit:
            levels = levels[:limit]

        return levels

    def get_levels_by_timeframe(self,
                              symbol: str,
                              timeframe: str,
                              limit: Optional[int] = None) -> List[PriceLevelData]:
        """
        Get price levels for a specific timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            limit: Maximum number of levels to return

        Returns:
            List of price level data
        """
        if symbol not in self.timeframe_index or timeframe not in self.timeframe_index[symbol]:
            return []

        level_ids = self.timeframe_index[symbol][timeframe]
        if limit:
            level_ids = level_ids[:limit]

        return [self.levels[lid] for lid in level_ids if lid in self.levels]

    def query_levels(self,
                    symbol: Optional[str] = None,
                    timeframe: Optional[str] = None,
                    min_price: Optional[float] = None,
                    max_price: Optional[float] = None,
                    min_significance: Optional[float] = None,
                    limit: Optional[int] = None) -> List[PriceLevelData]:
        """
        Advanced query for price levels with multiple filters.

        Args:
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            min_price: Minimum price filter
            max_price: Maximum price filter
            min_significance: Minimum significance level
            limit: Maximum results to return

        Returns:
            Filtered list of price level data
        """
        levels = []

        # Start with all levels or filter by symbol/timeframe
        if symbol and timeframe:
            levels = self.get_levels_by_timeframe(symbol, timeframe)
        elif symbol:
            levels = self.get_levels_by_symbol(symbol)
        else:
            levels = list(self.levels.values())

        # Apply filters
        filtered_levels = []

        for level in levels:
            # Price filter
            if min_price is not None and level.price < min_price:
                continue
            if max_price is not None and level.price > max_price:
                continue

            # Significance filter
            if min_significance is not None and level.significance_level < min_significance:
                continue

            filtered_levels.append(level)

        if limit:
            filtered_levels = filtered_levels[:limit]

        return filtered_levels

    def get_most_significant_levels(self,
                                  symbol: str,
                                  timeframe: str,
                                  top_k: int = 10) -> List[PriceLevelData]:
        """
        Get the most significant price levels for a symbol/timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            top_k: Number of top levels to return

        Returns:
            List of most significant price level data
        """
        levels = self.get_levels_by_timeframe(symbol, timeframe)
        if not levels:
            return []

        # Sort by significance level
        sorted_levels = sorted(levels, key=lambda x: x.significance_level, reverse=True)
        return sorted_levels[:top_k]

    def get_closest_levels_by_percentage(self,
                                       symbol: str,
                                       timeframe: str,
                                       current_price: float,
                                       level_pcts: List[float] = None) -> Dict[str, List[PriceLevelData]]:
        """
        Get the closest price levels at specified percentages above and below current price.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            current_price: Current market price
            level_pcts: List of percentages to find (e.g., [0.2, 0.4, 0.8, 1.0])

        Returns:
            Dictionary with 'above' and 'below' keys containing lists of levels
        """
        if level_pcts is None:
            level_pcts = [0.2, 0.4, 0.8, 1.0]  # Default percentages including 0.8%

        result = {'above': [], 'below': []}

        # Get all levels for this symbol/timeframe
        levels = self.get_levels_by_timeframe(symbol, timeframe)
        if not levels:
            return result

        # Group levels by their percentage
        levels_by_pct = {}
        for pct in level_pcts:
            levels_by_pct[pct] = [l for l in levels if abs(l.level_pct - pct) < 0.01]  # Allow small tolerance

        # Find closest levels for each percentage
        for pct in level_pcts:
            if pct not in levels_by_pct:
                continue

            pct_levels = levels_by_pct[pct]
            if not pct_levels:
                continue

            # Find closest above current price
            above_levels = [l for l in pct_levels if l.price > current_price]
            if above_levels:
                closest_above = min(above_levels, key=lambda x: x.price - current_price)
                result['above'].append(closest_above)

            # Find closest below current price
            below_levels = [l for l in pct_levels if l.price < current_price]
            if below_levels:
                closest_below = min(below_levels, key=lambda x: current_price - x.price)
                result['below'].append(closest_below)

        return result

    def get_situational_awareness(self,
                                 symbol: str,
                                 timeframe: str,
                                 current_price: float,
                                 include_all: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive situational awareness around current price.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            current_price: Current market price
            include_all: Whether to include all nearby levels, not just closest

        Returns:
            Dictionary with situational awareness data
        """
        # Get closest levels by percentage (including 0.8%)
        default_pcts = [0.2, 0.4, 0.8, 1.0, 2.0]
        closest_levels = self.get_closest_levels_by_percentage(
            symbol, timeframe, current_price, default_pcts
        )

        # Get most significant levels nearby
        all_levels = self.get_levels_by_timeframe(symbol, timeframe)
        nearby_levels = [l for l in all_levels
                        if abs(l.price - current_price) / current_price <= 0.05]  # Within 5%
        significant_nearby = sorted(nearby_levels,
                                  key=lambda x: x.significance_level, reverse=True)[:5]

        # Calculate price ranges
        price_range_02 = current_price * 0.002  # 0.2% range
        price_range_04 = current_price * 0.004  # 0.4% range
        price_range_08 = current_price * 0.008  # 0.8% range
        price_range_10 = current_price * 0.01   # 1.0% range

        # Find levels within specific percentage ranges
        levels_in_ranges = {
            'within_0.2%': [l for l in all_levels
                          if abs(l.price - current_price) <= price_range_02],
            'within_0.4%': [l for l in all_levels
                          if abs(l.price - current_price) <= price_range_04],
            'within_0.8%': [l for l in all_levels
                          if abs(l.price - current_price) <= price_range_08],
            'within_1.0%': [l for l in all_levels
                          if abs(l.price - current_price) <= price_range_10]
        }

        # Calculate percentage-only distances to nearest levels
        distances = {'above': {}, 'below': {}}
        for pct in default_pcts:
            above_levels = [l for l in closest_levels['above'] if abs(l.level_pct - pct) < 0.01]
            below_levels = [l for l in closest_levels['below'] if abs(l.level_pct - pct) < 0.01]

            if above_levels:
                closest_above = min(above_levels, key=lambda x: x.price - current_price)
                distances['above'][pct] = {
                    'price': closest_above.price,
                    'distance_pct': (closest_above.price - current_price) / current_price * 100,
                    'historical_crossings': closest_above.historical_crossings,
                    'historical_bounces': closest_above.historical_bounces,
                    'historical_volume': closest_above.historical_volume,
                    'significance_level': closest_above.significance_level
                }

            if below_levels:
                closest_below = min(below_levels, key=lambda x: current_price - x.price)
                distances['below'][pct] = {
                    'price': closest_below.price,
                    'distance_pct': (current_price - closest_below.price) / current_price * 100,
                    'historical_crossings': closest_below.historical_crossings,
                    'historical_bounces': closest_below.historical_bounces,
                    'historical_volume': closest_below.historical_volume,
                    'significance_level': closest_below.significance_level
                }

        return {
            'current_price': current_price,
            'closest_levels': closest_levels,
            'significant_nearby': significant_nearby,
            'levels_in_ranges': levels_in_ranges,
            'distances': distances,
            'price_ranges': {
                '0.2%': price_range_02,
                '0.4%': price_range_04,
                '0.8%': price_range_08,
                '1.0%': price_range_10
            }
        }

    def get_default_situational_awareness(self,
                                        symbol: str = 'BTCUSDT',
                                        timeframe: str = '1h') -> Dict[str, Any]:
        """
        Get default situational awareness data for the most recent price.

        This method provides immediate situational awareness without requiring
        a current price parameter - it uses the latest available price data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string

        Returns:
            Dictionary with default situational awareness data
        """
        # Get all levels for this symbol/timeframe
        levels = self.get_levels_by_timeframe(symbol, timeframe)

        if not levels:
            return {
                'current_price': None,
                'closest_levels': {'above': [], 'below': []},
                'significant_nearby': [],
                'levels_in_ranges': {'within_0.2%': [], 'within_0.4%': [], 'within_1.0%': []},
                'distances': {'above': {}, 'below': {}},
                'price_ranges': {'0.2%': 0, '0.4%': 0, '1.0%': 0}
            }

        # Find the most recent price from levels (use the price of the most recently updated level)
        # In a real implementation, you'd get this from your market data feed
        recent_levels = sorted(levels, key=lambda x: x.timestamp, reverse=True)
        current_price = recent_levels[0].price  # Use most recent level's price as proxy

        # Get situational awareness for this price
        return self.get_situational_awareness(symbol, timeframe, current_price)

    def update_level_tags(self, level_id: str, **tag_updates) -> bool:
        """
        Update tags for a specific price level.

        Args:
            level_id: Unique level identifier
            **tag_updates: Tag values to update

        Returns:
            True if update successful, False otherwise
        """
        if level_id not in self.levels:
            return False

        level = self.levels[level_id]

        # Update tags
        for tag_name, tag_value in tag_updates.items():
            if hasattr(level, tag_name):
                setattr(level, tag_name, tag_value)
            else:
                self.logger.warning(f"Unknown tag: {tag_name}")

        self.metadata['last_updated'] = datetime.now().isoformat()
        return True

    def bulk_update_tags(self, updates: Dict[str, Dict[str, Any]]) -> int:
        """
        Update tags for multiple levels in bulk.

        Args:
            updates: Dictionary of level_id -> tag_updates

        Returns:
            Number of successfully updated levels
        """
        updated_count = 0

        for level_id, tag_updates in updates.items():
            if self.update_level_tags(level_id, **tag_updates):
                updated_count += 1

        return updated_count

    def save_to_disk(self, filepath: Optional[str] = None) -> bool:
        """
        Save the bank to disk.

        Args:
            filepath: Optional custom filepath

        Returns:
            True if save successful
        """
        if not self.config.enable_persistence:
            self.logger.info("Persistence disabled, skipping save")
            return True

        save_path = Path(filepath) if filepath else Path(self.config.storage_path) / "price_level_bank.pkl"

        try:
            # Create backup of existing file
            if save_path.exists():
                backup_path = save_path.with_suffix('.pkl.bak')
                save_path.rename(backup_path)
                self.logger.debug(f"Created backup: {backup_path}")

            # Prepare data for serialization
            data = {
                'levels': {lid: level.to_dict() for lid, level in self.levels.items()},
                'symbol_index': self.symbol_index,
                'price_index': self.price_index,
                'timeframe_index': self.timeframe_index,
                'metadata': self.metadata,
                'config': asdict(self.config)
            }

            # Save to disk
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)

            self.logger.info(f"✅ Saved {len(self.levels)} levels to {save_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save bank: {e}")
            return False

    def _load_from_disk(self) -> None:
        """Load the bank from disk."""
        if not self.config.enable_persistence:
            return

        load_path = Path(self.config.storage_path) / "price_level_bank.pkl"

        if not load_path.exists():
            self.logger.info("No existing bank file found, starting fresh")
            return

        try:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)

            # Restore levels
            levels_data = data.get('levels', {})
            self.levels = {lid: PriceLevelData.from_dict(level_data)
                          for lid, level_data in levels_data.items()}

            # Restore indices
            self.symbol_index = data.get('symbol_index', {})
            self.price_index = data.get('price_index', {})
            self.timeframe_index = data.get('timeframe_index', {})

            # Restore metadata
            self.metadata = data.get('metadata', {})

            self.logger.info(f"✅ Loaded {len(self.levels)} levels from {load_path}")

        except Exception as e:
            self.logger.error(f"Failed to load bank: {e}")
            # Reset to empty state
            self.levels = {}
            self.symbol_index = {}
            self.price_index = {}
            self.timeframe_index = {}
            self.metadata = {
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'total_levels': 0,
                'symbols': set(),
                'timeframes': set()
            }

    def _add_to_symbol_index(self, level_id: str, symbol: str) -> None:
        """Add level to symbol index."""
        if symbol not in self.symbol_index:
            self.symbol_index[symbol] = []
        if level_id not in self.symbol_index[symbol]:
            self.symbol_index[symbol].append(level_id)

    def _add_to_price_index(self, level_id: str, symbol: str, price: float) -> None:
        """Add level to price index."""
        if symbol not in self.price_index:
            self.price_index[symbol] = {}
        if price not in self.price_index[symbol]:
            self.price_index[symbol][price] = []
        if level_id not in self.price_index[symbol][price]:
            self.price_index[symbol][price].append(level_id)

    def _add_to_timeframe_index(self, level_id: str, symbol: str, timeframe: str) -> None:
        """Add level to timeframe index."""
        if symbol not in self.timeframe_index:
            self.timeframe_index[symbol] = {}
        if timeframe not in self.timeframe_index[symbol]:
            self.timeframe_index[symbol][timeframe] = []
        if level_id not in self.timeframe_index[symbol][timeframe]:
            self.timeframe_index[symbol][timeframe].append(level_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get bank statistics."""
        return {
            'total_levels': len(self.levels),
            'symbols': len(self.symbol_index),
            'price_points': sum(len(prices) for prices in self.price_index.values()),
            'timeframes': sum(len(timeframes) for timeframes in self.timeframe_index.values()),
            'metadata': self.metadata
        }

    def clear(self) -> None:
        """Clear all data from the bank."""
        self.levels.clear()
        self.symbol_index.clear()
        self.price_index.clear()
        self.timeframe_index.clear()
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'total_levels': 0,
            'symbols': set(),
            'timeframes': set()
        }
        self.operation_count = 0
        self.logger.info("Bank cleared")

# Global price level bank instance
_global_price_level_bank: Optional[PriceLevelBank] = None

def get_global_price_level_bank() -> PriceLevelBank:
    """
    Get the global price level bank instance.

    Returns:
        Global price level bank instance
    """
    global _global_price_level_bank
    if _global_price_level_bank is None:
        _global_price_level_bank = PriceLevelBank()
    return _global_price_level_bank

def set_global_price_level_bank(bank: PriceLevelBank) -> None:
    """
    Set the global price level bank instance.

    Args:
        bank: Price level bank instance
    """
    global _global_price_level_bank
    _global_price_level_bank = bank