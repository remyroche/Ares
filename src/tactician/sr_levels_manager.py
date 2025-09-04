from __future__ import annotations
from typing import Dict, List, Optional, Union, Any, Tuple
'\nSR Levels Manager - Comprehensive Support/Resistance Level Management\n\nThis module provides:\n1. SR level calculation based on backtesting data\n2. Continuous updates during live trading\n3. Comprehensive level information (age, strength, volume, etc.)\n4. Price vs VWAP comparison logic\n5. Persistent storage and retrieval\n'
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any
import asyncio
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.utils.logger import system_logger
logger = system_logger.getChild('SRLevelsManager')

class SRLevel:
    """Individual Support/Resistance Level with comprehensive information."""

    def __init__(self, price: float, level_type: str, method: str, data_source: str, timestamp: datetime, strength: float=0.5, volume: float=0.0, touch_count: int=0, age_hours: float=0.0, bounce_rate: float=0.0, isolation_score: float=0.0, confidence: float=0.5, metadata: dict[str, Any] | None=None) -> None:
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
        self.last_touch = timestamp
        self.total_touches = touch_count
        self.creation_time = timestamp

    def to_dict(self) -> dict[str, Any]:
        """Convert level to dictionary for storage."""
        return {'price': self.price, 'level_type': self.level_type, 'method': self.method, 'data_source': self.data_source, 'timestamp': self.timestamp.isoformat(), 'strength': self.strength, 'volume': self.volume, 'touch_count': self.touch_count, 'age_hours': self.age_hours, 'bounce_rate': self.bounce_rate, 'isolation_score': self.isolation_score, 'confidence': self.confidence, 'last_touch': self.last_touch.isoformat(), 'total_touches': self.total_touches, 'creation_time': self.creation_time.isoformat(), 'metadata': self.metadata}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'SRLevel':
        """Create level from dictionary."""
        return cls(price=data['price'], level_type=data['level_type'], method=data['method'], data_source=data['data_source'], timestamp=datetime.fromisoformat(data['timestamp']), strength=data.get('strength', 0.5), volume=data.get('volume', 0.0), touch_count=data.get('touch_count', 0), age_hours=data.get('age_hours', 0.0), bounce_rate=data.get('bounce_rate', 0.0), isolation_score=data.get('isolation_score', 0.0), confidence=data.get('confidence', 0.5), metadata=data.get('metadata', {}))

    def update_touch(self, current_time: datetime, price: float, volume: float=0.0) -> None:
        """Update level with new touch information."""
        self.last_touch = current_time
        self.touch_count += 1
        self.total_touches += 1
        self.volume = max(self.volume, volume)
        self.age_hours = (current_time - self.creation_time).total_seconds() / 3600
        self.strength = min(1.0, 0.5 + self.touch_count * 0.1)

    def calculate_quality_score(self) -> float:
        """Calculate overall quality score for this level."""
        score = 0.0
        score += self.strength * 0.3
        score += min(0.3, self.touch_count * 0.05)
        score += min(0.1, self.age_hours / 1000)
        score += self.bounce_rate * 0.2
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

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize SR Levels Manager."""
        self.config = config
        self.logger = system_logger.getChild('SRLevelsManager')
        self.sr_config = config.get('sr_levels_manager', {})
        self.storage_path = Path(self.sr_config.get('storage_path', 'data/sr_levels'))
        self.max_levels = self.sr_config.get('max_levels', 50)
        self.min_strength = self.sr_config.get('min_strength', 0.3)
        self.proximity_threshold = self.sr_config.get('proximity_threshold', 0.005)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.levels_file = self.storage_path / 'sr_levels.json'
        self.history_file = self.storage_path / 'sr_levels_history.json'
        self.support_levels: list[SRLevel] = []
        self.resistance_levels: list[SRLevel] = []
        self.sr_predictor: SRBreakoutPredictor | None = None
        self.last_update = datetime.now()
        self.update_count = 0

    async def initialize(self) -> bool:
        """Initialize the SR Levels Manager."""
        try:
            self.logger.info('🔧 Initializing SR Levels Manager...')
            self.sr_predictor = SRBreakoutPredictor(self.config)
            if not await self.sr_predictor.initialize():
                self.logger.error('❌ Failed to initialize SR predictor')
                return False
            await self.load_levels()
            self.logger.info('✅ SR Levels Manager initialized successfully')
            return True
        except Exception as e:
            self.logger.exception(f'❌ Failed to initialize SR Levels Manager: {e}')
            return False

    async def calculate_sr_levels_from_backtest(self, market_data: pd.DataFrame, timeframe: str='1m') -> dict[str, list[SRLevel]]:
        """
        Calculate SR levels from backtesting data using SR breakout predictor logic.

        Args:
            market_data: Historical market data
            timeframe: Data timeframe

        Returns:
            Dictionary with support and resistance levels
        """
        try:
            self.logger.info(f'🔍 Calculating SR levels from backtest data ({len(market_data)} points)')
            current_price = market_data['close'].iloc[-1]
            support_levels = []
            resistance_levels = []
            try:
                sr_context = await self.sr_predictor.get_sr_context(market_data, current_price)
                for level_data in sr_context.get('support_levels', []):
                    level = self._create_sr_level_from_data(level_data, 'support')
                    if level:
                        support_levels.append(level)
                for level_data in sr_context.get('resistance_levels', []):
                    level = self._create_sr_level_from_data(level_data, 'resistance')
                    if level:
                        resistance_levels.append(level)
                self.logger.info(f'✅ Retrieved {len(support_levels)} support and {len(resistance_levels)} resistance levels from SR context')
            except Exception as e:
                self.logger.warning(f'⚠️ SR context method failed: {e}')
            if len(support_levels) < 3 or len(resistance_levels) < 3:
                self.logger.info('🔄 Using direct detection methods for additional levels')
                try:
                    direct_support = await self.sr_predictor._detect_support_levels(market_data)
                    for level_data in direct_support:
                        level = self._create_sr_level_from_data(level_data, 'support')
                        if level and (not self._level_exists(level, support_levels)):
                            support_levels.append(level)
                    direct_resistance = await self.sr_predictor._detect_resistance_levels(market_data)
                    for level_data in direct_resistance:
                        level = self._create_sr_level_from_data(level_data, 'resistance')
                        if level and (not self._level_exists(level, resistance_levels)):
                            resistance_levels.append(level)
                    self.logger.info(f'✅ Added {len(direct_support)} direct support and {len(direct_resistance)} direct resistance levels')
                except Exception as e:
                    self.logger.warning(f'⚠️ Direct detection methods failed: {e}')
            if len(support_levels) < 5 or len(resistance_levels) < 5:
                self.logger.info('🔄 Using specific detection methods for comprehensive coverage')
                detection_methods = ['fractal', 'volume', 'pivot', 'atr']
                for method in detection_methods:
                    try:
                        original_method = self.sr_predictor.sr_detection_method
                        self.sr_predictor.sr_detection_method = method
                        method_support = await self.sr_predictor._detect_support_levels(market_data)
                        for level_data in method_support:
                            level = self._create_sr_level_from_data(level_data, 'support')
                            if level and (not self._level_exists(level, support_levels)):
                                level.metadata['detection_method'] = method
                                support_levels.append(level)
                        method_resistance = await self.sr_predictor._detect_resistance_levels(market_data)
                        for level_data in method_resistance:
                            level = self._create_sr_level_from_data(level_data, 'resistance')
                            if level and (not self._level_exists(level, resistance_levels)):
                                level.metadata['detection_method'] = method
                                resistance_levels.append(level)
                        self.sr_predictor.sr_detection_method = original_method
                        self.logger.info(f'✅ Added {len(method_support)} {method} support and {len(method_resistance)} {method} resistance levels')
                    except Exception as e:
                        self.logger.warning(f'⚠️ {method} detection method failed: {e}')
                        self.sr_predictor.sr_detection_method = original_method
            support_levels = self._filter_and_deduplicate_levels(support_levels)
            resistance_levels = self._filter_and_deduplicate_levels(resistance_levels)
            self.support_levels = support_levels
            self.resistance_levels = resistance_levels
            await self.save_levels()
            self.logger.info(f'✅ Final calculation: {len(support_levels)} support and {len(resistance_levels)} resistance levels')
            return {'support_levels': support_levels, 'resistance_levels': resistance_levels}
        except Exception as e:
            self.logger.exception(f'❌ Error calculating SR levels from backtest: {e}')
            return {'support_levels': [], 'resistance_levels': []}

    async def calculate_sr_levels_with_method(self, market_data: pd.DataFrame, method: str, level_type: str='both') -> dict[str, list[SRLevel]]:
        """
        Calculate SR levels using a specific detection method from SR breakout predictor.

        Args:
            market_data: Historical market data
            method: Detection method ("fractal", "volume", "pivot", "atr")
            level_type: "support", "resistance", or "both"

        Returns:
            Dictionary with support and/or resistance levels
        """
        try:
            self.logger.info(f'🔍 Calculating SR levels using {method} method')
            original_method = self.sr_predictor.sr_detection_method
            self.sr_predictor.sr_detection_method = method
            support_levels = []
            resistance_levels = []
            if level_type in ['support', 'both']:
                try:
                    support_data = await self.sr_predictor._detect_support_levels(market_data)
                    for level_data in support_data:
                        level = self._create_sr_level_from_data(level_data, 'support')
                        if level:
                            level.metadata['detection_method'] = method
                            support_levels.append(level)
                    self.logger.info(f'✅ Detected {len(support_levels)} support levels using {method} method')
                except Exception as e:
                    self.logger.exception(f'❌ Error detecting support levels with {method} method: {e}')
            if level_type in ['resistance', 'both']:
                try:
                    resistance_data = await self.sr_predictor._detect_resistance_levels(market_data)
                    for level_data in resistance_data:
                        level = self._create_sr_level_from_data(level_data, 'resistance')
                        if level:
                            level.metadata['detection_method'] = method
                            resistance_levels.append(level)
                    self.logger.info(f'✅ Detected {len(resistance_levels)} resistance levels using {method} method')
                except Exception as e:
                    self.logger.exception(f'❌ Error detecting resistance levels with {method} method: {e}')
            self.sr_predictor.sr_detection_method = original_method
            return {'support_levels': support_levels, 'resistance_levels': resistance_levels, 'method_used': method}
        except Exception as e:
            self.logger.exception(f'❌ Error calculating SR levels with {method} method: {e}')
            self.sr_predictor.sr_detection_method = original_method
            return {'support_levels': [], 'resistance_levels': [], 'method_used': method}

    async def update_levels_with_live_data(self, current_price: float, current_volume: float, current_time: datetime) -> dict[str, Any]:
        """
        Update SR levels with live trading data.

        Args:
            current_price: Current market price
            current_volume: Current volume
            current_time: Current timestamp

        Returns:
            Update summary
        """
        try:
            self.logger.info(f'🔄 Updating SR levels with live data (price: {current_price:.4f})')
            updates = {'support_touches': 0, 'resistance_touches': 0, 'new_levels_created': 0, 'levels_removed': 0}
            for level in self.support_levels + self.resistance_levels:
                if self._is_price_near_level(current_price, level.price):
                    level.update_touch(current_time, current_price, current_volume)
                    if level.level_type == 'support':
                        updates['support_touches'] += 1
                    else:
                        updates['resistance_touches'] += 1
            self._cleanup_old_levels()
            self.last_update = current_time
            self.update_count += 1
            await self.save_levels()
            self.logger.info(f'✅ Updated SR levels: {updates}')
            return updates
        except Exception as e:
            self.logger.exception(f'❌ Error updating SR levels: {e}')
            return {}

    def get_sr_levels_for_trading(self, current_price: float, include_metadata: bool=True) -> dict[str, Any]:
        """
        Get SR levels optimized for trading intelligence.

        Args:
            current_price: Current market price
            include_metadata: Whether to include detailed metadata

        Returns:
            Trading-optimized SR levels
        """
        try:
            nearest_support = self._find_nearest_level(current_price, self.support_levels)
            nearest_resistance = self._find_nearest_level(current_price, self.resistance_levels)
            support_proximity = self._calculate_proximity(current_price, nearest_support)
            resistance_proximity = self._calculate_proximity(current_price, nearest_resistance)
            response = {'current_price': current_price, 'timestamp': datetime.now().isoformat(), 'nearest_support': {'price': nearest_support.price if nearest_support else None, 'strength': nearest_support.strength if nearest_support else None, 'proximity': support_proximity, 'touch_count': nearest_support.touch_count if nearest_support else None, 'age_hours': nearest_support.age_hours if nearest_support else None, 'bounce_rate': nearest_support.bounce_rate if nearest_support else None, 'quality_score': nearest_support.calculate_quality_score() if nearest_support else None} if nearest_support else None, 'nearest_resistance': {'price': nearest_resistance.price if nearest_resistance else None, 'strength': nearest_resistance.strength if nearest_resistance else None, 'proximity': resistance_proximity, 'touch_count': nearest_resistance.touch_count if nearest_resistance else None, 'age_hours': nearest_resistance.age_hours if nearest_resistance else None, 'bounce_rate': nearest_resistance.bounce_rate if nearest_resistance else None, 'quality_score': nearest_resistance.calculate_quality_score() if nearest_resistance else None} if nearest_resistance else None, 'all_levels': {'support_count': len(self.support_levels), 'resistance_count': len(self.resistance_levels), 'total_count': len(self.support_levels) + len(self.resistance_levels)}}
            if include_metadata:
                response['detailed_levels'] = {'support_levels': [level.to_dict() for level in self.support_levels], 'resistance_levels': [level.to_dict() for level in self.resistance_levels]}
            return response
        except Exception as e:
            self.logger.exception(f'❌ Error getting SR levels for trading: {e}')
            return {}

    def compare_price_vs_vwap_predictions(self, price_levels: list[SRLevel], vwap_levels: list[SRLevel]) -> dict[str, Any]:
        """
        Compare price vs VWAP SR level predictions.

        Args:
            price_levels: Levels detected using price data
            vwap_levels: Levels detected using VWAP data

        Returns:
            Comparison analysis
        """
        try:
            self.logger.info('🔍 Comparing price vs VWAP SR level predictions')
            price_support = [l for l in price_levels if l.level_type == 'support']
            price_resistance = [l for l in price_levels if l.level_type == 'resistance']
            vwap_support = [l for l in vwap_levels if l.level_type == 'support']
            vwap_resistance = [l for l in vwap_levels if l.level_type == 'resistance']
            price_quality = self._calculate_levels_quality(price_levels)
            vwap_quality = self._calculate_levels_quality(vwap_levels)
            overlap_analysis = self._calculate_levels_overlap(price_levels, vwap_levels)
            comparison = {'level_counts': {'price': {'support': len(price_support), 'resistance': len(price_resistance), 'total': len(price_levels)}, 'vwap': {'support': len(vwap_support), 'resistance': len(vwap_resistance), 'total': len(vwap_levels)}}, 'quality_metrics': {'price': price_quality, 'vwap': vwap_quality}, 'overlap_analysis': overlap_analysis, 'recommendations': self._generate_comparison_recommendations(price_quality, vwap_quality, overlap_analysis), 'timestamp': datetime.now().isoformat()}
            self.logger.info('✅ Price vs VWAP comparison completed')
            return comparison
        except Exception as e:
            self.logger.exception(f'❌ Error comparing price vs VWAP predictions: {e}')
            return {}

    def _filter_and_deduplicate_levels(self, levels: list[SRLevel]) -> list[SRLevel]:
        """Filter and deduplicate levels based on quality and proximity."""
        if not levels:
            return []
        levels.sort(key=lambda x: x.calculate_quality_score(), reverse=True)
        levels = [l for l in levels if l.strength >= self.min_strength]
        filtered = []
        for level in levels:
            is_duplicate = False
            for existing in filtered:
                if self._is_price_near_level(level.price, existing.price):
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(level)
        return filtered[:self.max_levels]

    def _is_price_near_level(self, price1: float, price2: float) -> bool:
        """Check if two prices are near each other."""
        if price2 == 0:
            return False
        return abs(price1 - price2) / price2 < self.proximity_threshold

    def _find_nearest_level(self, price: float, levels: list[SRLevel]) -> SRLevel | None:
        """Find the nearest level to a given price."""
        if not levels:
            return None
        return min(levels, key=lambda x: abs(x.price - price))

    def _calculate_proximity(self, price: float, level: SRLevel | None) -> float:
        """Calculate proximity to a level (0 = at level, 1 = far away)."""
        if not level or level.price == 0:
            return 1.0
        return abs(price - level.price) / level.price

    def _cleanup_old_levels(self) -> None:
        """Remove old or weak levels."""
        current_time = datetime.now()
        self.support_levels = [l for l in self.support_levels if (current_time - l.creation_time).days < 30 or l.strength > 0.6]
        self.resistance_levels = [l for l in self.resistance_levels if (current_time - l.creation_time).days < 30 or l.strength > 0.6]

    def _calculate_levels_quality(self, levels: list[SRLevel]) -> dict[str, float]:
        """Calculate quality metrics for a set of levels."""
        if not levels:
            return {'avg_strength': 0.0, 'avg_confidence': 0.0, 'avg_quality': 0.0}
        avg_strength = np.mean([l.strength for l in levels])
        avg_confidence = np.mean([l.confidence for l in levels])
        avg_quality = np.mean([l.calculate_quality_score() for l in levels])
        return {'avg_strength': avg_strength, 'avg_confidence': avg_confidence, 'avg_quality': avg_quality}

    def _calculate_levels_overlap(self, levels1: list[SRLevel], levels2: list[SRLevel]) -> dict[str, Any]:
        """Calculate overlap between two sets of levels."""
        if not levels1 or not levels2:
            return {'overlap_count': 0, 'overlap_rate': 0.0, 'overlap_details': []}
        overlap_count = 0
        overlap_details = []
        for l1 in levels1:
            for l2 in levels2:
                if l1.level_type == l2.level_type and self._is_price_near_level(l1.price, l2.price):
                    overlap_count += 1
                    overlap_details.append({'level1': l1.to_dict(), 'level2': l2.to_dict(), 'price_difference': abs(l1.price - l2.price)})
        overlap_rate = overlap_count / min(len(levels1), len(levels2)) if min(len(levels1), len(levels2)) > 0 else 0.0
        return {'overlap_count': overlap_count, 'overlap_rate': overlap_rate, 'overlap_details': overlap_details}

    def _create_sr_level_from_data(self, level_data: dict[str, Any], level_type: str) -> SRLevel | None:
        """Create SRLevel object from level data dictionary."""
        try:
            if not level_data or not isinstance(level_data, dict):
                return None
            timestamp = level_data.get('timestamp')
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            elif timestamp is None:
                timestamp = datetime.now()
            return SRLevel(price=level_data.get('price', 0), level_type=level_type, method=level_data.get('method', 'unknown'), data_source=level_data.get('data_source', 'price'), timestamp=timestamp, strength=level_data.get('enhanced_strength', level_data.get('strength', 0.5)), volume=level_data.get('volume', 0.0), touch_count=level_data.get('touch_count', 0), age_hours=level_data.get('age_hours', 0.0), bounce_rate=level_data.get('bounce_rate', 0.0), isolation_score=level_data.get('isolation_score', 0.0), confidence=level_data.get('confidence', 0.5), metadata=level_data.get('metadata', {}))
        except Exception as e:
            self.logger.exception(f'❌ Error creating SR level from data: {e}')
            return None

    def _level_exists(self, new_level: SRLevel, existing_levels: list[SRLevel]) -> bool:
        """Check if a level already exists in the list based on price proximity."""
        try:
            for existing_level in existing_levels:
                if existing_level.level_type == new_level.level_type and self._is_price_near_level(new_level.price, existing_level.price):
                    return True
            return False
        except Exception as e:
            self.logger.exception(f'❌ Error checking level existence: {e}')
            return False

    def _generate_comparison_recommendations(self, price_quality: dict[str, float], vwap_quality: dict[str, float], overlap_analysis: dict[str, Any]) -> list[str]:
        """Generate recommendations based on comparison analysis."""
        recommendations = []
        if price_quality['avg_quality'] > vwap_quality['avg_quality']:
            recommendations.append('Price-based detection shows higher quality - consider prioritizing price data')
        elif vwap_quality['avg_quality'] > price_quality['avg_quality']:
            recommendations.append('VWAP-based detection shows higher quality - consider prioritizing VWAP data')
        if overlap_analysis['overlap_rate'] < 0.3:
            recommendations.append('Low overlap between approaches - consider adjusting detection parameters')
        elif overlap_analysis['overlap_rate'] > 0.8:
            recommendations.append('High overlap between approaches - both methods are detecting similar levels')
        if price_quality['avg_quality'] < 0.5:
            recommendations.append('Price-based detection quality is low - review detection parameters')
        if vwap_quality['avg_quality'] < 0.5:
            recommendations.append('VWAP-based detection quality is low - review VWAP calculation')
        return recommendations

    async def save_levels(self) -> None:
        """Save current levels to storage."""
        try:
            data = {'support_levels': [level.to_dict() for level in self.support_levels], 'resistance_levels': [level.to_dict() for level in self.resistance_levels], 'last_update': self.last_update.isoformat(), 'update_count': self.update_count}
            with open(self.levels_file, 'w') as f:
                json.dump(data, f, indent=2)
            await self._save_to_history(data)
        except Exception as e:
            self.logger.exception(f'❌ Error saving SR levels: {e}')

    async def load_levels(self) -> Any:
        """Load levels from storage."""
        try:
            if not self.levels_file.exists():
                self.logger.info('No existing SR levels found, starting fresh')
                return
            with open(self.levels_file) as f:
                data = json.load(f)
            self.support_levels = [SRLevel.from_dict(level_data) for level_data in data.get('support_levels', [])]
            self.resistance_levels = [SRLevel.from_dict(level_data) for level_data in data.get('resistance_levels', [])]
            self.last_update = datetime.fromisoformat(data.get('last_update', datetime.now().isoformat()))
            self.update_count = data.get('update_count', 0)
            self.logger.info(f'✅ Loaded {len(self.support_levels)} support and {len(self.resistance_levels)} resistance levels')
        except Exception as e:
            self.logger.exception(f'❌ Error loading SR levels: {e}')

    async def _save_to_history(self, data: dict[str, Any]) -> None:
        """Save current state to history file."""
        try:
            history_data = []
            if self.history_file.exists():
                with open(self.history_file) as f:
                    history_data = json.load(f)
            history_data.append({'timestamp': datetime.now().isoformat(), 'data': data})
            if len(history_data) > 100:
                history_data = history_data[-100:]
            with open(self.history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            self.logger.exception(f'❌ Error saving to history: {e}')

async def create_sr_levels_manager(config: dict[str, Any]) -> SRLevelsManager:
    """Factory function to create and initialize SR Levels Manager."""
    manager = SRLevelsManager(config)
    if await manager.initialize():
        return manager
    return None