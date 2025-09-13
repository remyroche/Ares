
import pandas as pd
from ...utils.logger import system_logger
'\nSR Levels Manager - Comprehensive Support/Resistance Level Management\n\nThis module provides:\n1. SR level calculation based on backtesting data\n2. Continuous updates during live trading\n3. Comprehensive level information (age, strength, volume, etc.)\n4. Price vs VWAP comparison logic\n5. Persistent storage and retrieval\n'
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any
warnings.filterwarnings('ignore')
from ...utils.logger import system_logger
from .sr_breakout_predictor_enhanced import SRBreakoutPredictor
import numpy as np
import logging
import time

logger = system_logger.getChild('SRLevelsManager')

class SRLevel:
    """Individual Support/Resistance Level with comprehensive information."""

    def __init__(self, price: float, level_type: str, method: str, data_source: str, timestamp: datetime, strength: float = 0.5, volume: float = 0.0, touch_count: int = 0, age_hours: float = 0.0, bounce_rate: float = 0.0, isolation_score: float = 0.0, confidence: float = 0.5, metadata: dict[str, Any] | None = None) -> None:
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
        import numpy as np

        def convert_value(value):
            """Convert numpy types to regular Python types for JSON serialization."""
            if isinstance(value, np.float32) or isinstance(value, np.float64):
                return float(value)
            elif isinstance(value, np.int32) or isinstance(value, np.int64):
                return int(value)
            else:
                return value

        return {
            'price': convert_value(self.price),
            'level_type': self.level_type,
            'method': self.method,
            'data_source': self.data_source,
            'timestamp': self.timestamp.isoformat(),
            'strength': convert_value(self.strength),
            'volume': convert_value(self.volume),
            'touch_count': convert_value(self.touch_count),
            'age_hours': convert_value(self.age_hours),
            'bounce_rate': convert_value(self.bounce_rate),
            'isolation_score': convert_value(self.isolation_score),
            'confidence': convert_value(self.confidence),
            'last_touch': self.last_touch.isoformat(),
            'total_touches': convert_value(self.total_touches),
            'creation_time': self.creation_time.isoformat(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'SRLevel':
        """Create level from dictionary."""
        return cls(price = data['price'], level_type = data['level_type'], method = data['method'], data_source = data['data_source'], timestamp = datetime.fromisoformat(data['timestamp']), strength = data.get('strength', 0.5), volume = data.get('volume', 0.0), touch_count = data.get('touch_count', 0), age_hours = data.get('age_hours', 0.0), bounce_rate = data.get('bounce_rate', 0.0), isolation_score = data.get('isolation_score', 0.0), confidence = data.get('confidence', 0.5), metadata = data.get('metadata', {}))

    def update_touch(self, current_time: datetime, price: float, volume: float = 0.0, 
                    market_data: pd.DataFrame = None, min_bounce_threshold: float = 0.001,
                    min_time_between_touches: int = 300, volume_spike_threshold: float = 1.2) -> bool:
        """
        Update level with new touch information using meaningful bounce detection.
        
        Args:
            current_time: Current timestamp
            price: Current price
            volume: Current volume
            market_data: Market data for context (optional)
            min_bounce_threshold: Minimum price movement away from level (0.1% = 0.001)
            min_time_between_touches: Minimum seconds between touches (5 minutes = 300)
            volume_spike_threshold: Volume spike multiplier (1.2x average)
            
        Returns:
            bool: True if touch was counted, False if filtered out
        """
        # Check time-based filtering
        if (current_time - self.last_touch).total_seconds() < min_time_between_touches:
            return False  # Too soon since last touch
        
        # Check for meaningful bounce
        price_moved_away = abs(price - self.price) / self.price > min_bounce_threshold
        if not price_moved_away:
            return False  # No meaningful bounce detected
        
        # Check volume confirmation if market data available
        volume_confirmed = True
        if market_data is not None and len(market_data) > 20:
            avg_volume = market_data['volume'].tail(20).mean()
            volume_confirmed = volume >= avg_volume * volume_spike_threshold
        
        # Only count as touch if all criteria met
        if price_moved_away and volume_confirmed:
            self.last_touch = current_time
            self.touch_count += 1
            self.total_touches += 1
            self.volume = max(self.volume, volume)
            self.age_hours = (current_time - self.creation_time).total_seconds() / 3600
            self.strength = min(1.0, 0.5 + self.touch_count * 0.1)
            
            # Update bounce rate calculation
            self._update_bounce_rate(market_data)
            return True
        
        return False
    
    def _update_bounce_rate(self, market_data: pd.DataFrame = None) -> None:
        """Update bounce rate based on recent price action."""
        if market_data is None or len(market_data) < 10:
            return
        
        # Look at recent price action around this level
        recent_data = market_data.tail(50)  # Last 50 bars
        level_price = self.price
        tolerance = level_price * 0.002  # 0.2% tolerance
        
        bounces = 0
        total_tests = 0
        
        for i in range(1, len(recent_data)):
            current = recent_data.iloc[i]
            previous = recent_data.iloc[i-1]
            
            # Check if price tested the level
            if (abs(current['low'] - level_price) <= tolerance or 
                abs(current['high'] - level_price) <= tolerance):
                total_tests += 1
                
                # Check if price bounced away
                if self.level_type == 'support':
                    if current['close'] > level_price + tolerance:
                        bounces += 1
                elif self.level_type == 'resistance':
                    if current['close'] < level_price - tolerance:
                        bounces += 1
                else:  # both
                    if (abs(current['close'] - level_price) > tolerance):
                        bounces += 1
        
        if total_tests > 0:
            self.bounce_rate = bounces / total_tests

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
        self.storage_path.mkdir(parents = True, exist_ok = True)
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
            # Initialize SRBreakoutPredictor
            try:
                self.sr_predictor = SRBreakoutPredictor(self.config)
                self.logger.info('✅ SRBreakoutPredictor initialized successfully')
            except Exception as e:
                self.logger.warning(f'⚠️ Failed to initialize SRBreakoutPredictor: {e}, using basic detection')
                self.sr_predictor = None
            
            await self.load_levels()
            self.logger.info('✅ SR Levels Manager initialized successfully')
            return True
        except Exception as e:
            self.logger.exception(f'❌ Failed to initialize SR Levels Manager: {e}')
            return False

    def calculate_sr_levels_from_backtest_sync(self, market_data: pd.DataFrame, timeframe: str='1m') -> dict[str, list[SRLevel]]:
        """Synchronous version of SR level calculation for better threading compatibility."""
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.calculate_sr_levels_from_backtest(market_data, timeframe))
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f'Synchronous SR calculation failed: {e}')
            return {'support_levels': [], 'resistance_levels': []}

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
            if self.sr_predictor is not None:
                try:
                    sr_context = await self.sr_predictor.get_sr_context(market_data, current_price)
                    for level_data in sr_context.get('support_levels', []):
                        level = self._create_sr_level_from_data(level_data, 'support', market_data)
                        if level:
                            support_levels.append(level)
                    for level_data in sr_context.get('resistance_levels', []):
                        level = self._create_sr_level_from_data(level_data, 'resistance', market_data)
                        if level:
                            resistance_levels.append(level)
                    self.logger.info(f'✅ Retrieved {len(support_levels)} support and {len(resistance_levels)} resistance levels from SR context')
                except Exception as e:
                    self.logger.warning(f'⚠️ SR context method failed: {e}')
            else:
                self.logger.warning('⚠️ SR predictor not available, skipping SR context method')
            if len(support_levels) < 3 or len(resistance_levels) < 3:
                self.logger.info('🔄 Using direct detection methods for additional levels')
                if self.sr_predictor is not None:
                    try:
                        direct_support = await self.sr_predictor._detect_support_levels(market_data)
                        for level_data in direct_support:
                            level = self._create_sr_level_from_data(level_data, 'support', market_data)
                            if level and (not self._level_exists(level, support_levels)):
                                support_levels.append(level)
                        direct_resistance = await self.sr_predictor._detect_resistance_levels(market_data)
                        for level_data in direct_resistance:
                            level = self._create_sr_level_from_data(level_data, 'resistance', market_data)
                            if level and (not self._level_exists(level, resistance_levels)):
                                resistance_levels.append(level)
                        self.logger.info(f'✅ Added {len(direct_support)} direct support and {len(direct_resistance)} direct resistance levels')
                    except Exception as e:
                        self.logger.warning(f'⚠️ Direct detection methods failed: {e}')
                else:
                    self.logger.warning('⚠️ SR predictor not available, skipping direct detection methods')
            if len(support_levels) < 5 or len(resistance_levels) < 5:
                self.logger.info('🔄 Using specific detection methods for comprehensive coverage')
                if self.sr_predictor is not None:
                    detection_methods = ['fractal', 'volume', 'pivot', 'atr']
                    for method in detection_methods:
                        try:
                            self.logger.info(f'🔄 Trying {method} detection method...')
                            original_method = self.sr_predictor.sr_detection_method
                            self.sr_predictor.sr_detection_method = method
                            
                            # Regular execution with progress updates
                            if method == 'volume':
                                self.logger.info('⏱️ Volume detection may take time for large datasets...')
                                self.logger.info('📊 Processing volume-based support level detection...')
                            
                            method_support = await self.sr_predictor._detect_support_levels(market_data)
                            for level_data in method_support:
                                level = self._create_sr_level_from_data(level_data, 'support', market_data)
                                if level and (not self._level_exists(level, support_levels)):
                                    level.metadata['detection_method'] = method
                                    support_levels.append(level)
                            # Regular execution with progress updates
                            if method == 'volume':
                                self.logger.info('📊 Processing volume-based resistance level detection...')
                            
                            method_resistance = await self.sr_predictor._detect_resistance_levels(market_data)
                            for level_data in method_resistance:
                                level = self._create_sr_level_from_data(level_data, 'resistance', market_data)
                                if level and (not self._level_exists(level, resistance_levels)):
                                    level.metadata['detection_method'] = method
                                    resistance_levels.append(level)
                            self.sr_predictor.sr_detection_method = original_method
                            self.logger.info(f'✅ Added {len(method_support)} {method} support and {len(method_resistance)} {method} resistance levels')
                        except Exception as e:
                            self.logger.warning(f'⚠️ {method} detection method failed: {e}')
                            if 'original_method' in locals():
                                self.sr_predictor.sr_detection_method = original_method
                else:
                    self.logger.warning('⚠️ SR predictor not available, skipping specific detection methods')
            support_levels = self._filter_and_deduplicate_levels(support_levels)
            resistance_levels = self._filter_and_deduplicate_levels(resistance_levels)
            
            # Merge support and resistance levels while preserving original detection info
            merged_levels = self._merge_support_resistance_levels(support_levels, resistance_levels)
            
            self.support_levels = merged_levels['support_levels']
            self.resistance_levels = merged_levels['resistance_levels']
            await self.save_levels()
            self.logger.info(f'✅ Final calculation: {len(self.support_levels)} support and {len(self.resistance_levels)} resistance levels')
            self.logger.info(f'   📊 Merged levels preserve original detection method and regime context')
            return {'support_levels': self.support_levels, 'resistance_levels': self.resistance_levels}
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
                        level = self._create_sr_level_from_data(level_data, 'support', market_data)
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
                        level = self._create_sr_level_from_data(level_data, 'resistance', market_data)
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

    async def update_levels_with_live_data(self, current_price: float, current_volume: float, current_time: datetime, 
                                         market_data: pd.DataFrame = None) -> dict[str, Any]:
        """
        Update SR levels with live trading data using meaningful bounce detection.

        Args:
            current_price: Current market price
            current_volume: Current volume
            current_time: Current timestamp
            market_data: Recent market data for context

        Returns:
            Update summary
        """
        try:
            self.logger.info(f'🔄 Updating SR levels with live data (price: {current_price:.4f})')
            updates = {'support_touches': 0, 'resistance_touches': 0, 'new_levels_created': 0, 'levels_removed': 0, 'filtered_touches': 0}
            
            for level in self.support_levels + self.resistance_levels:
                if self._is_price_near_level(current_price, level.price):
                    # Use meaningful bounce detection
                    touch_counted = level.update_touch(
                        current_time, current_price, current_volume, 
                        market_data=market_data,
                        min_bounce_threshold=0.001,  # 0.1% minimum bounce
                        min_time_between_touches=300,  # 5 minutes
                        volume_spike_threshold=1.2  # 1.2x volume spike
                    )
                    
                    if touch_counted:
                        if level.level_type == 'support':
                            updates['support_touches'] += 1
                        else:
                            updates['resistance_touches'] += 1
                    else:
                        updates['filtered_touches'] += 1
            
            self._cleanup_old_levels()
            self.last_update = current_time
            self.update_count += 1
            await self.save_levels()
            self.logger.info(f'✅ Updated SR levels: {updates}')
            return updates
        except Exception as e:
            self.logger.exception(f'❌ Error updating SR levels: {e}')
            return {}

    def get_sr_levels_for_trading(self, current_price: float, include_metadata: bool = True) -> dict[str, Any]:
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
        """Filter and deduplicate levels based on quality, proximity, and time."""
        if not levels:
            return []
        
        # Sort by quality score
        levels.sort(key = lambda x: x.calculate_quality_score(), reverse = True)
        levels = [l for l in levels if l.strength >= self.min_strength]
        
        # Apply time-based deduplication (1 hour minimum between levels)
        filtered = self._deduplicate_levels_by_time(levels, min_time_gap_hours=1.0)
        
        # Apply price-based deduplication
        final_filtered = []
        for level in filtered:
            is_duplicate = False
            for existing in final_filtered:
                if self._is_price_near_level(level.price, existing.price):
                    is_duplicate = True
                    break
            if not is_duplicate:
                final_filtered.append(level)
        
        return final_filtered[:self.max_levels]
    
    def _deduplicate_levels_by_time(self, levels: list[SRLevel], min_time_gap_hours: float = 1.0) -> list[SRLevel]:
        """Remove duplicate levels that are too close in time."""
        if not levels:
            return []
        
        # Sort by creation time
        sorted_levels = sorted(levels, key=lambda x: x.creation_time)
        filtered_levels = []
        
        for level in sorted_levels:
            is_duplicate = False
            for existing in filtered_levels:
                time_gap = (level.creation_time - existing.creation_time).total_seconds() / 3600
                price_gap = abs(level.price - existing.price) / existing.price if existing.price > 0 else 1.0
                
                # Consider duplicate if too close in time AND price
                if time_gap < min_time_gap_hours and price_gap < 0.005:  # 0.5% price threshold
                    is_duplicate = True
                    # Keep the one with higher quality score
                    if level.calculate_quality_score() > existing.calculate_quality_score():
                        filtered_levels.remove(existing)
                        filtered_levels.append(level)
                    break
            
            if not is_duplicate:
                filtered_levels.append(level)
        
        return filtered_levels

    def _is_price_near_level(self, price1: float, price2: float) -> bool:
        """Check if two prices are near each other."""
        if price2 == 0:
            return False
        return abs(price1 - price2) / price2 < self.proximity_threshold

    def _find_nearest_level(self, price: float, levels: list[SRLevel]) -> SRLevel | None:
        """Find the nearest level to a given price."""
        if not levels:
            return None
        return min(levels, key = lambda x: abs(x.price - price))

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

    def _create_sr_level_from_data(self, level_data: dict[str, Any], level_type: str, market_data: pd.DataFrame = None) -> SRLevel | None:
        """Create SRLevel object from level data dictionary with trend context."""
        try:
            if not level_data or not isinstance(level_data, dict):
                return None
            timestamp = level_data.get('timestamp')
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            elif timestamp is None:
                timestamp = datetime.now()
            
            # Create base level
            level = SRLevel(
                price = level_data.get('price', 0), 
                level_type = level_type, 
                method = level_data.get('method', 'unknown'), 
                data_source = level_data.get('data_source', 'price'), 
                timestamp = timestamp, 
                strength = level_data.get('enhanced_strength', level_data.get('strength', 0.5)), 
                volume = level_data.get('volume', 0.0), 
                touch_count = level_data.get('touch_count', 0), 
                age_hours = level_data.get('age_hours', 0.0), 
                bounce_rate = level_data.get('bounce_rate', 0.0), 
                isolation_score = level_data.get('isolation_score', 0.0), 
                confidence = level_data.get('confidence', 0.5), 
                metadata = level_data.get('metadata', {})
            )
            
            # Add trend context and regime context if market data available
            if market_data is not None and len(market_data) > 20:
                trend_context = self._classify_level_type_with_trend(level.price, market_data)
                regime_context = self._classify_market_regime(market_data)
                
                level.level_type = trend_context
                level.metadata['original_detection_type'] = level_type
                level.metadata['trend_context'] = trend_context
                level.metadata['regime_context'] = regime_context
                
                # Regime context is informational only - no level type adjustment
            
            return level
        except Exception as e:
            self.logger.exception(f'❌ Error creating SR level from data: {e}')
            return None
    
    def _classify_level_type_with_trend(self, level_price: float, market_data: pd.DataFrame, lookback_bars: int = 20) -> str:
        """Classify level as support or resistance based on how price hits the level (from above or below)."""
        try:
            if len(market_data) < lookback_bars:
                return 'both'
            
            recent_data = market_data.tail(lookback_bars)
            tolerance = level_price * 0.002  # 0.2% tolerance
            
            # Count how many times price hits the level from above vs below
            hits_from_above = 0  # Price was above level, then hit it (resistance)
            hits_from_below = 0  # Price was below level, then hit it (support)
            
            for i in range(1, len(recent_data)):
                current = recent_data.iloc[i]
                previous = recent_data.iloc[i-1]
                
                # Check if current bar touches the level
                level_touched = (current['low'] <= level_price + tolerance and 
                               current['high'] >= level_price - tolerance)
                
                if level_touched:
                    # Determine if price came from above or below
                    if previous['close'] > level_price + tolerance:
                        hits_from_above += 1  # Price came down to hit level (resistance)
                    elif previous['close'] < level_price - tolerance:
                        hits_from_below += 1  # Price came up to hit level (support)
            
            # Classify based on hit direction
            if hits_from_above > hits_from_below:
                return 'resistance'  # More hits from above = resistance
            elif hits_from_below > hits_from_above:
                return 'support'     # More hits from below = support
            else:
                return 'both'        # Equal hits = can act as both
            
        except Exception as e:
            self.logger.warning(f'Error classifying level type with trend: {e}')
            return 'both'
    
    def _classify_market_regime(self, market_data: pd.DataFrame, lookback_bars: int = 50) -> str:
        """Classify current market regime based on price action and volatility."""
        try:
            if len(market_data) < lookback_bars:
                return 'unknown'
            
            recent_data = market_data.tail(lookback_bars)
            
            # Calculate trend strength
            price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            # Calculate volatility (ATR-based)
            high_low = recent_data['high'] - recent_data['low']
            high_close = np.abs(recent_data['high'] - recent_data['close'].shift())
            low_close = np.abs(recent_data['low'] - recent_data['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=14).mean().iloc[-1]
            volatility = atr / recent_data['close'].iloc[-1]
            
            # Calculate momentum indicators
            sma_20 = recent_data['close'].rolling(window=20).mean()
            sma_50 = recent_data['close'].rolling(window=50).mean()
            sma_ratio = sma_20.iloc[-1] / sma_50.iloc[-1] if not sma_50.empty else 1.0
            
            # RSI for momentum
            delta = recent_data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - 100 / (1 + rs)
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            
            # Classify regime
            if abs(price_change) > 0.05 and volatility > 0.02:  # Strong trend with high volatility
                if price_change > 0:
                    return 'trending_up'
                else:
                    return 'trending_down'
            elif abs(price_change) > 0.02 and volatility < 0.015:  # Moderate trend with low volatility
                if price_change > 0:
                    return 'trending_up'
                else:
                    return 'trending_down'
            elif abs(sma_ratio - 1.0) < 0.01 and volatility < 0.01:  # Low volatility, sideways
                return 'ranging'
            elif volatility > 0.03:  # High volatility regardless of trend
                return 'volatile'
            else:
                return 'ranging'  # Default to ranging
                
        except Exception as e:
            self.logger.warning(f'Error classifying market regime: {e}')
            return 'unknown'
    
    def _adjust_level_type_for_regime(self, level_type: str, regime: str) -> str:
        """Adjust level type based on market regime context."""
        try:
            if regime == 'trending_up':
                # In uptrends, levels are more likely to be resistance
                if level_type == 'both':
                    return 'resistance'
                elif level_type == 'support':
                    return 'support'  # Keep support as is
                else:
                    return level_type
            elif regime == 'trending_down':
                # In downtrends, levels are more likely to be support
                if level_type == 'both':
                    return 'support'
                elif level_type == 'resistance':
                    return 'resistance'  # Keep resistance as is
                else:
                    return level_type
            elif regime == 'volatile':
                # In volatile markets, levels can act as both
                return 'both'
            else:  # ranging or unknown
                return level_type
                
        except Exception as e:
            self.logger.warning(f'Error adjusting level type for regime: {e}')
            return level_type
    
    def _merge_support_resistance_levels(self, support_levels: list[SRLevel], resistance_levels: list[SRLevel]) -> dict[str, list[SRLevel]]:
        """
        Merge support and resistance levels while preserving original detection information.
        
        This method ensures that:
        1. Original detection method is preserved in metadata
        2. Regime context is maintained
        3. Levels are properly classified based on trend and regime
        4. No information is lost during the merge process
        """
        try:
            merged_support = []
            merged_resistance = []
            
            # Process support levels
            for level in support_levels:
                # Preserve original detection info
                if 'original_detection_type' not in level.metadata:
                    level.metadata['original_detection_type'] = 'support'
                
                # Add to appropriate list based on current classification
                if level.level_type in ['support', 'both']:
                    merged_support.append(level)
                if level.level_type in ['resistance', 'both']:
                    merged_resistance.append(level)
            
            # Process resistance levels
            for level in resistance_levels:
                # Preserve original detection info
                if 'original_detection_type' not in level.metadata:
                    level.metadata['original_detection_type'] = 'resistance'
                
                # Add to appropriate list based on current classification
                if level.level_type in ['support', 'both']:
                    merged_support.append(level)
                if level.level_type in ['resistance', 'both']:
                    merged_resistance.append(level)
            
            # Remove duplicates based on price proximity
            merged_support = self._remove_duplicate_levels_by_price(merged_support)
            merged_resistance = self._remove_duplicate_levels_by_price(merged_resistance)
            
            self.logger.info(f'📊 Level merge complete:')
            self.logger.info(f'   - Support levels: {len(merged_support)} (from {len(support_levels)} original)')
            self.logger.info(f'   - Resistance levels: {len(merged_resistance)} (from {len(resistance_levels)} original)')
            
            return {
                'support_levels': merged_support,
                'resistance_levels': merged_resistance
            }
            
        except Exception as e:
            self.logger.error(f'Error merging support/resistance levels: {e}')
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels
            }
    
    def _remove_duplicate_levels_by_price(self, levels: list[SRLevel]) -> list[SRLevel]:
        """Remove duplicate levels based on price proximity while preserving the best one."""
        if not levels:
            return []
        
        # Sort by quality score (best first)
        levels.sort(key=lambda x: x.calculate_quality_score(), reverse=True)
        
        filtered_levels = []
        for level in levels:
            is_duplicate = False
            for existing in filtered_levels:
                if self._is_price_near_level(level.price, existing.price):
                    is_duplicate = True
                    # If current level is better, replace the existing one
                    if level.calculate_quality_score() > existing.calculate_quality_score():
                        filtered_levels.remove(existing)
                        filtered_levels.append(level)
                    break
            
            if not is_duplicate:
                filtered_levels.append(level)
        
        return filtered_levels

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

    async def save_levels(self, detection_config: dict[str, Any] | None = None) -> None:
        """Save current levels to storage."""
        try:
            data = {'support_levels': [level.to_dict() for level in self.support_levels], 'resistance_levels': [level.to_dict() for level in self.resistance_levels], 'last_update': self.last_update.isoformat(), 'update_count': self.update_count}
            with open(self.levels_file, 'w') as f:
                json.dump(data, f, indent = 2)

            # Save detection parameters if provided
            if detection_config:
                params_file = self.storage_path / 'detection_config.json'
                with open(params_file, 'w') as f:
                    json.dump(detection_config, f, indent = 2)
                self.logger.info(f'✅ Saved detection parameters to {params_file}')

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

    async def load_detection_config(self) -> dict[str, Any] | None:
        """Load detection configuration from storage."""
        try:
            params_file = self.storage_path / 'detection_config.json'
            if not params_file.exists():
                self.logger.info('No detection config found, using defaults')
                return None
            with open(params_file) as f:
                config = json.load(f)
            self.logger.info(f'✅ Loaded detection configuration from {params_file}')
            return config
        except Exception as e:
            self.logger.exception(f'❌ Error loading detection config: {e}')
            return None

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
                json.dump(history_data, f, indent = 2)
        except Exception as e:
            self.logger.exception(f'❌ Error saving to history: {e}')

async def create_sr_levels_manager(config: dict[str, Any]) -> SRLevelsManager:
    """Factory function to create and initialize SR Levels Manager."""
    manager = SRLevelsManager(config)
    if await manager.initialize():
        return manager
    return None