from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json
import os
from datetime import datetime
from abc import ABC, abstractmethod

# Core utilities
from ...utils.logger import system_logger
from ...core.decorators import handles_errors, traced

# Modern utilities
from ...utils.performance_utils import PerformanceProfiler
from ...utils.parallel_processing_optimizer import ParallelProcessor
from ...utils.enhanced_error_handler import EnhancedErrorHandler
from ...utils.caching import CacheManager
from ...utils.validation import ValidationFramework
from ...utils.enhanced_data_operations import DataOperations
from ...utils.monitoring_utils import MonitoringUtils

# Import the optimized SR method
from ...training.steps.data_collection.data_preparation.sr_strength_optimizer import SRLevelIdentifier, SRLevel

@dataclass
class EnhancedSRLevel:
    """Enhanced S/R level with comprehensive analysis."""
    price: float
    strength: float
    type: str
    touch_count: int
    first_touch_bar: int
    last_touch_bar: int
    age_bars: int
    avg_bounce_ratio: float
    max_bounce_ratio: float
    volume_confirmation_score: float
    consistency_score: float
    failure_count: int
    # Enhanced features
    volume_analysis: Dict[str, Any]
    psychological_level_info: Dict[str, Any]
    market_regime: str
    confidence_score: float
    metadata: Dict[str, Any]

class PsychologicalLevelDetector:
    """Detects and analyzes psychological levels."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild('PsychologicalLevelDetector')
        
    def detect_psychological_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect psychological levels with comprehensive analysis."""
        try:
            current_price = market_data['close'].iloc[-1]
            levels = []
            
            # Determine appropriate round number intervals based on price level
            if current_price > 1000:
                intervals = [100, 50, 25, 10]
                major_intervals = [1000, 500, 250, 100]
            elif current_price > 100:
                intervals = [10, 5, 2.5, 1]
                major_intervals = [100, 50, 25, 10]
            elif current_price > 10:
                intervals = [1, 0.5, 0.25, 0.1]
                major_intervals = [10, 5, 2.5, 1]
            else:
                intervals = [0.1, 0.05, 0.025, 0.01]
                major_intervals = [1, 0.5, 0.25, 0.1]
            
            # Find psychological levels around current price
            for interval in intervals:
                base = int(current_price / interval) * interval
                
                # Check levels above and below current price
                for i in range(-5, 6):
                    level_price = base + i * interval
                    if level_price > 0:
                        # Calculate psychological strength
                        strength = self._calculate_psychological_strength(level_price, interval, major_intervals)
                        
                        # Analyze volume behavior at this level
                        volume_analysis = self._analyze_volume_at_level(market_data, level_price)
                        
                        # Determine level type
                        level_type = 'support' if level_price < current_price else 'resistance'
                        
                        levels.append({
                            'price': level_price,
                            'strength': strength,
                            'type': level_type,
                            'interval': interval,
                            'is_major': interval in major_intervals,
                            'volume_analysis': volume_analysis,
                            'psychological_type': self._get_psychological_type(level_price, interval)
                        })
            
            # Sort by strength and remove duplicates
            levels = sorted(levels, key=lambda x: x['strength'], reverse=True)
            unique_levels = self._remove_duplicate_levels(levels)
            
            self.logger.info(f"🧠 Psychological levels detected: {len(unique_levels)} levels")
            for level in unique_levels[:10]:  # Log top 10
                self.logger.info(f"   - {level['type'].upper()}: {level['price']:.4f} (strength: {level['strength']:.3f}, {level['psychological_type']})")
            
            return unique_levels
            
        except Exception as e:
            self.logger.error(f"Psychological level detection failed: {e}")
            return []
    
    def _calculate_psychological_strength(self, price: float, interval: float, major_intervals: List[float]) -> float:
        """Calculate psychological strength of a level."""
        try:
            # Base strength from interval size
            if interval in major_intervals:
                base_strength = 0.8
            elif interval <= major_intervals[0] / 2:
                base_strength = 0.6
            else:
                base_strength = 0.4
            
            # Bonus for round numbers
            if price % 1 == 0:
                base_strength += 0.1
            elif price % 0.5 == 0:
                base_strength += 0.05
            elif price % 0.1 == 0:
                base_strength += 0.02
            
            # Bonus for psychological significance
            if price % 100 == 0:
                base_strength += 0.15
            elif price % 50 == 0:
                base_strength += 0.1
            elif price % 10 == 0:
                base_strength += 0.05
            
            return min(base_strength, 1.0)
            
        except Exception as e:
            self.logger.error(f"Psychological strength calculation failed: {e}")
            return 0.5
    
    def _analyze_volume_at_level(self, market_data: pd.DataFrame, level_price: float) -> Dict[str, Any]:
        """Analyze volume behavior at a psychological level."""
        try:
            threshold = 0.002  # 0.2% proximity threshold
            volume_data = []
            
            for i in range(len(market_data)):
                high = market_data['high'].iloc[i]
                low = market_data['low'].iloc[i]
                volume = market_data['volume'].iloc[i]
                
                # Check if price touched this level
                if abs(high - level_price) / level_price < threshold or abs(low - level_price) / level_price < threshold:
                    volume_data.append({
                        'index': i,
                        'volume': volume,
                        'price': market_data['close'].iloc[i],
                        'touched': True
                    })
            
            if not volume_data:
                return {
                    'touch_count': 0,
                    'avg_volume': 0,
                    'volume_spike_ratio': 1.0,
                    'volume_trend': 'none'
                }
            
            # Calculate volume statistics
            volumes = [d['volume'] for d in volume_data]
            avg_volume = np.mean(volumes)
            max_volume = np.max(volumes)
            
            # Calculate volume spike ratio
            overall_avg_volume = market_data['volume'].mean()
            volume_spike_ratio = max_volume / overall_avg_volume if overall_avg_volume > 0 else 1.0
            
            # Determine volume trend
            if len(volumes) > 1:
                volume_trend = 'increasing' if volumes[-1] > volumes[0] else 'decreasing'
            else:
                volume_trend = 'stable'
            
            return {
                'touch_count': len(volume_data),
                'avg_volume': avg_volume,
                'max_volume': max_volume,
                'volume_spike_ratio': volume_spike_ratio,
                'volume_trend': volume_trend,
                'volume_data': volume_data
            }
            
        except Exception as e:
            self.logger.error(f"Volume analysis at level failed: {e}")
            return {'touch_count': 0, 'avg_volume': 0, 'volume_spike_ratio': 1.0, 'volume_trend': 'none'}
    
    def _get_psychological_type(self, price: float, interval: float) -> str:
        """Get the psychological type of a level."""
        if price % 100 == 0:
            return "century"
        elif price % 50 == 0:
            return "half-century"
        elif price % 10 == 0:
            return "decade"
        elif price % 1 == 0:
            return "whole_number"
        elif price % 0.5 == 0:
            return "half"
        elif price % 0.1 == 0:
            return "tenth"
        else:
            return "custom_interval"
    
    def _remove_duplicate_levels(self, levels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate levels based on price proximity."""
        if not levels:
            return []
        
        unique_levels = []
        for level in levels:
            is_duplicate = False
            for existing in unique_levels:
                if abs(level['price'] - existing['price']) / existing['price'] < 0.001:  # 0.1% threshold
                    is_duplicate = True
                    # Keep the stronger level
                    if level['strength'] > existing['strength']:
                        unique_levels.remove(existing)
                        unique_levels.append(level)
                    break
            
            if not is_duplicate:
                unique_levels.append(level)
        
        return unique_levels

class VolumeAnalyzer:
    """Comprehensive volume analysis for S/R levels."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild('VolumeAnalyzer')
    
    def analyze_volume_patterns(self, market_data: pd.DataFrame, sr_levels: List[SRLevel]) -> Dict[str, Any]:
        """Analyze volume patterns for S/R levels with detailed logging."""
        try:
            self.logger.info("📊 Starting comprehensive volume analysis...")
            
            volume_analysis = {
                'overall_volume_stats': self._calculate_overall_volume_stats(market_data),
                'level_volume_analysis': [],
                'volume_regime': self._determine_volume_regime(market_data),
                'volume_trends': self._analyze_volume_trends(market_data)
            }
            
            # Analyze volume for each S/R level
            for i, level in enumerate(sr_levels):
                level_analysis = self._analyze_level_volume(market_data, level, i)
                volume_analysis['level_volume_analysis'].append(level_analysis)
                
                # Log detailed volume information
                self.logger.info(f"📈 Level {i+1} Volume Analysis:")
                self.logger.info(f"   - Price: {level.price:.4f} ({level.type})")
                self.logger.info(f"   - Touch Count: {level.touch_count}")
                self.logger.info(f"   - Volume Confirmation Score: {level.volume_confirmation_score:.3f}")
                self.logger.info(f"   - Avg Volume at Touches: {level_analysis['avg_volume_at_touches']:.0f}")
                self.logger.info(f"   - Volume Spike Ratio: {level_analysis['volume_spike_ratio']:.2f}")
                self.logger.info(f"   - Volume Trend: {level_analysis['volume_trend']}")
                self.logger.info(f"   - Volume Regime: {level_analysis['volume_regime']}")
                
                if level_analysis['volume_spikes']:
                    self.logger.info(f"   - Volume Spikes: {len(level_analysis['volume_spikes'])} detected")
                    for spike in level_analysis['volume_spikes'][:3]:  # Show top 3 spikes
                        self.logger.info(f"     * Spike at bar {spike['bar_index']}: {spike['volume']:.0f} (ratio: {spike['ratio']:.2f})")
            
            # Log overall volume statistics
            self.logger.info("📊 Overall Volume Statistics:")
            self.logger.info(f"   - Average Volume: {volume_analysis['overall_volume_stats']['avg_volume']:.0f}")
            self.logger.info(f"   - Volume Volatility: {volume_analysis['overall_volume_stats']['volume_volatility']:.3f}")
            self.logger.info(f"   - Volume Regime: {volume_analysis['volume_regime']}")
            self.logger.info(f"   - Volume Trend: {volume_analysis['volume_trends']['trend_direction']}")
            
            return volume_analysis
            
        except Exception as e:
            self.logger.error(f"Volume analysis failed: {e}")
            return {}
    
    def _calculate_overall_volume_stats(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall volume statistics."""
        try:
            volumes = market_data['volume']
            
            return {
                'avg_volume': volumes.mean(),
                'median_volume': volumes.median(),
                'max_volume': volumes.max(),
                'min_volume': volumes.min(),
                'volume_volatility': volumes.std() / volumes.mean() if volumes.mean() > 0 else 0,
                'volume_skewness': volumes.skew(),
                'volume_kurtosis': volumes.kurtosis()
            }
        except Exception as e:
            self.logger.error(f"Overall volume stats calculation failed: {e}")
            return {}
    
    def _determine_volume_regime(self, market_data: pd.DataFrame) -> str:
        """Determine the current volume regime."""
        try:
            volumes = market_data['volume']
            recent_volumes = volumes.tail(20)
            historical_volumes = volumes.head(-20)
            
            recent_avg = recent_volumes.mean()
            historical_avg = historical_volumes.mean()
            
            if recent_avg > historical_avg * 1.2:
                return "high_volume"
            elif recent_avg < historical_avg * 0.8:
                return "low_volume"
            else:
                return "normal_volume"
                
        except Exception as e:
            self.logger.error(f"Volume regime determination failed: {e}")
            return "unknown"
    
    def _analyze_volume_trends(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume trends."""
        try:
            volumes = market_data['volume']
            
            # Calculate moving averages
            short_ma = volumes.rolling(10).mean()
            long_ma = volumes.rolling(30).mean()
            
            # Determine trend
            if short_ma.iloc[-1] > long_ma.iloc[-1] * 1.1:
                trend_direction = "increasing"
            elif short_ma.iloc[-1] < long_ma.iloc[-1] * 0.9:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
            
            return {
                'trend_direction': trend_direction,
                'short_ma': short_ma.iloc[-1],
                'long_ma': long_ma.iloc[-1],
                'trend_strength': abs(short_ma.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1] if long_ma.iloc[-1] > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Volume trend analysis failed: {e}")
            return {'trend_direction': 'unknown', 'short_ma': 0, 'long_ma': 0, 'trend_strength': 0}
    
    def _analyze_level_volume(self, market_data: pd.DataFrame, level: SRLevel, level_index: int) -> Dict[str, Any]:
        """Analyze volume patterns for a specific S/R level."""
        try:
            level_price = level.price
            threshold = 0.002  # 0.2% proximity threshold
            
            # Find all touches of this level
            touches = []
            volume_spikes = []
            
            for i in range(len(market_data)):
                high = market_data['high'].iloc[i]
                low = market_data['low'].iloc[i]
                volume = market_data['volume'].iloc[i]
                
                # Check if price touched this level
                if abs(high - level_price) / level_price < threshold or abs(low - level_price) / level_price < threshold:
                    touches.append({
                        'bar_index': i,
                        'volume': volume,
                        'price': market_data['close'].iloc[i],
                        'timestamp': market_data.index[i] if hasattr(market_data.index[i], 'strftime') else i
                    })
            
            if not touches:
                return {
                    'level_index': level_index,
                    'touch_count': 0,
                    'avg_volume_at_touches': 0,
                    'volume_spike_ratio': 1.0,
                    'volume_trend': 'none',
                    'volume_regime': 'none',
                    'volume_spikes': []
                }
            
            # Calculate volume statistics
            touch_volumes = [t['volume'] for t in touches]
            avg_volume_at_touches = np.mean(touch_volumes)
            max_volume_at_touches = np.max(touch_volumes)
            
            # Calculate volume spike ratio
            overall_avg_volume = market_data['volume'].mean()
            volume_spike_ratio = max_volume_at_touches / overall_avg_volume if overall_avg_volume > 0 else 1.0
            
            # Identify volume spikes (volume > 2x average)
            for touch in touches:
                if touch['volume'] > overall_avg_volume * 2:
                    volume_spikes.append({
                        'bar_index': touch['bar_index'],
                        'volume': touch['volume'],
                        'ratio': touch['volume'] / overall_avg_volume,
                        'timestamp': touch['timestamp']
                    })
            
            # Determine volume trend at this level
            if len(touch_volumes) > 1:
                if touch_volumes[-1] > touch_volumes[0] * 1.2:
                    volume_trend = "increasing"
                elif touch_volumes[-1] < touch_volumes[0] * 0.8:
                    volume_trend = "decreasing"
                else:
                    volume_trend = "stable"
            else:
                volume_trend = "single_touch"
            
            # Determine volume regime for this level
            if avg_volume_at_touches > overall_avg_volume * 1.5:
                volume_regime = "high_volume_level"
            elif avg_volume_at_touches < overall_avg_volume * 0.7:
                volume_regime = "low_volume_level"
            else:
                volume_regime = "normal_volume_level"
            
            return {
                'level_index': level_index,
                'touch_count': len(touches),
                'avg_volume_at_touches': avg_volume_at_touches,
                'max_volume_at_touches': max_volume_at_touches,
                'volume_spike_ratio': volume_spike_ratio,
                'volume_trend': volume_trend,
                'volume_regime': volume_regime,
                'volume_spikes': sorted(volume_spikes, key=lambda x: x['ratio'], reverse=True),
                'touch_details': touches
            }
            
        except Exception as e:
            self.logger.error(f"Level volume analysis failed for level {level_index}: {e}")
            return {
                'level_index': level_index,
                'touch_count': 0,
                'avg_volume_at_touches': 0,
                'volume_spike_ratio': 1.0,
                'volume_trend': 'error',
                'volume_regime': 'error',
                'volume_spikes': []
            }

class SimplifiedSRPredictor:
    """Simplified S/R predictor using only the optimized method with enhanced analysis."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize simplified S/R predictor."""
        self.config = config
        self.logger = system_logger.getChild('SimplifiedSRPredictor')
        
        # Initialize modern utilities
        self.error_handler = EnhancedErrorHandler('SimplifiedSRPredictor')
        self.performance_profiler = PerformanceProfiler()
        self.parallel_processor = ParallelProcessor()
        self.cache_manager = CacheManager()
        self.validation_framework = ValidationFramework()
        self.data_operations = DataOperations()
        self.monitoring_utils = MonitoringUtils()
        
        # Initialize core components
        self.identifier = SRLevelIdentifier(config)
        self.psychological_detector = PsychologicalLevelDetector(config)
        self.volume_analyzer = VolumeAnalyzer(config)
        
        # Configuration
        self.min_strength = config.get('min_strength', 0.5)
        self.include_psychological = config.get('include_psychological_levels', True)
        self.volume_analysis_enabled = config.get('volume_analysis_enabled', True)
        
        self.logger.info("🚀 SimplifiedSRPredictor initialized with modern utilities")
        self.logger.info(f"   - Min strength threshold: {self.min_strength}")
        self.logger.info(f"   - Psychological levels: {'enabled' if self.include_psychological else 'disabled'}")
        self.logger.info(f"   - Volume analysis: {'enabled' if self.volume_analysis_enabled else 'disabled'}")
    
    @handles_errors(exceptions=(ValueError, AttributeError), default_return=[], context='simplified SR prediction')
    @traced(span_name='SimplifiedSR.predict')
    async def identify_enhanced_levels(self, market_data: pd.DataFrame) -> List[EnhancedSRLevel]:
        """
        Identify enhanced S/R levels using only the optimized method with comprehensive analysis.
        
        Args:
            market_data: Historical market data
            
        Returns:
            List of enhanced S/R levels with volume and psychological analysis
        """
        with self.performance_profiler.profile('enhanced_sr_prediction'):
            try:
                self.logger.info("🎯 Starting enhanced S/R level identification...")
                
                # Validate input data
                if not self.validation_framework.validate_dataframe(market_data, required_columns=['open', 'high', 'low', 'close', 'volume']):
                    self.logger.error("Invalid market data provided")
                    return []
                
                # Use cache for expensive operations
                cache_key = f'sr_levels_{hash(str(market_data.index[-10:]))}_{len(market_data)}'
                cached_result = self.cache_manager.get_cached_result(cache_key)
                if cached_result:
                    self.logger.info("📋 Using cached S/R levels")
                    return cached_result
                
                # Get optimized S/R levels
                self.logger.info("🔍 Identifying optimized S/R levels...")
                sr_levels = self.identifier.identify_strong_sr_levels(market_data, self.min_strength)
                
                self.logger.info(f"✅ Found {len(sr_levels)} optimized S/R levels")
                for i, level in enumerate(sr_levels):
                    self.logger.info(f"   - Level {i+1}: {level.type.upper()} at {level.price:.4f} (strength: {level.strength:.3f}, touches: {level.touch_count})")
                
                # Analyze volume patterns if enabled
                volume_analysis = {}
                if self.volume_analysis_enabled and sr_levels:
                    self.logger.info("📊 Analyzing volume patterns...")
                    volume_analysis = self.volume_analyzer.analyze_volume_patterns(market_data, sr_levels)
                
                # Detect psychological levels if enabled
                psychological_levels = []
                if self.include_psychological:
                    self.logger.info("🧠 Detecting psychological levels...")
                    psychological_levels = self.psychological_detector.detect_psychological_levels(market_data)
                
                # Combine and enhance levels
                enhanced_levels = await self._create_enhanced_levels(
                    sr_levels, psychological_levels, volume_analysis, market_data
                )
                
                # Cache the results
                self.cache_manager.cache_result(cache_key, enhanced_levels)
                
                # Log final results
                self.logger.info(f"🎉 Enhanced S/R analysis completed:")
                self.logger.info(f"   - Optimized levels: {len(sr_levels)}")
                self.logger.info(f"   - Psychological levels: {len(psychological_levels)}")
                self.logger.info(f"   - Total enhanced levels: {len(enhanced_levels)}")
                
                return enhanced_levels
                
            except Exception as e:
                self.logger.error(f"Enhanced S/R level identification failed: {e}")
                return []
    
    async def _create_enhanced_levels(
        self, 
        sr_levels: List[SRLevel], 
        psychological_levels: List[Dict[str, Any]], 
        volume_analysis: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> List[EnhancedSRLevel]:
        """Create enhanced S/R levels with comprehensive analysis."""
        try:
            enhanced_levels = []
            
            # Process optimized S/R levels
            for i, level in enumerate(sr_levels):
                # Get volume analysis for this level
                level_volume_analysis = {}
                if volume_analysis and 'level_volume_analysis' in volume_analysis:
                    level_volume_analysis = volume_analysis['level_volume_analysis'][i] if i < len(volume_analysis['level_volume_analysis']) else {}
                
                # Determine market regime
                market_regime = self._determine_market_regime(market_data)
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence_score(level, level_volume_analysis)
                
                enhanced_level = EnhancedSRLevel(
                    price=level.price,
                    strength=level.strength,
                    type=level.type,
                    touch_count=level.touch_count,
                    first_touch_bar=level.first_touch_bar,
                    last_touch_bar=level.last_touch_bar,
                    age_bars=level.age_bars,
                    avg_bounce_ratio=level.avg_bounce_ratio,
                    max_bounce_ratio=level.max_bounce_ratio,
                    volume_confirmation_score=level.volume_confirmation_score,
                    consistency_score=level.consistency_score,
                    failure_count=level.failure_count,
                    volume_analysis=level_volume_analysis,
                    psychological_level_info={},
                    market_regime=market_regime,
                    confidence_score=confidence_score,
                    metadata=level.metadata
                )
                
                enhanced_levels.append(enhanced_level)
            
            # Add psychological levels that don't overlap with optimized levels
            for psych_level in psychological_levels:
                # Check if this psychological level overlaps with any optimized level
                overlaps = False
                for enhanced_level in enhanced_levels:
                    if abs(psych_level['price'] - enhanced_level.price) / enhanced_level.price < 0.002:  # 0.2% threshold
                        # Enhance the existing level with psychological information
                        enhanced_level.psychological_level_info = {
                            'is_psychological': True,
                            'psychological_strength': psych_level['strength'],
                            'psychological_type': psych_level['psychological_type'],
                            'interval': psych_level['interval'],
                            'is_major': psych_level['is_major']
                        }
                        overlaps = True
                        break
                
                # If no overlap, create a new enhanced level for this psychological level
                if not overlaps and psych_level['strength'] > 0.3:  # Only include strong psychological levels
                    market_regime = self._determine_market_regime(market_data)
                    
                    enhanced_level = EnhancedSRLevel(
                        price=psych_level['price'],
                        strength=psych_level['strength'],
                        type=psych_level['type'],
                        touch_count=psych_level['volume_analysis'].get('touch_count', 0),
                        first_touch_bar=0,
                        last_touch_bar=0,
                        age_bars=0,
                        avg_bounce_ratio=0.0,
                        max_bounce_ratio=0.0,
                        volume_confirmation_score=0.5,
                        consistency_score=0.5,
                        failure_count=0,
                        volume_analysis=psych_level['volume_analysis'],
                        psychological_level_info={
                            'is_psychological': True,
                            'psychological_strength': psych_level['strength'],
                            'psychological_type': psych_level['psychological_type'],
                            'interval': psych_level['interval'],
                            'is_major': psych_level['is_major']
                        },
                        market_regime=market_regime,
                        confidence_score=psych_level['strength'],
                        metadata={'source': 'psychological_level'}
                    )
                    
                    enhanced_levels.append(enhanced_level)
            
            # Sort by strength and confidence
            enhanced_levels.sort(key=lambda x: (x.strength + x.confidence_score) / 2, reverse=True)
            
            return enhanced_levels
            
        except Exception as e:
            self.logger.error(f"Enhanced level creation failed: {e}")
            return []
    
    def _determine_market_regime(self, market_data: pd.DataFrame) -> str:
        """Determine current market regime."""
        try:
            if len(market_data) < 50:
                return "insufficient_data"
            
            # Calculate trend indicators
            sma_20 = market_data['close'].rolling(20).mean()
            sma_50 = market_data['close'].rolling(50).mean()
            
            current_price = market_data['close'].iloc[-1]
            sma_20_current = sma_20.iloc[-1]
            sma_50_current = sma_50.iloc[-1]
            
            # Determine regime
            if current_price > sma_20_current > sma_50_current:
                return "uptrend"
            elif current_price < sma_20_current < sma_50_current:
                return "downtrend"
            else:
                return "ranging"
                
        except Exception as e:
            self.logger.error(f"Market regime determination failed: {e}")
            return "unknown"
    
    def _calculate_confidence_score(self, level: SRLevel, volume_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for a level."""
        try:
            # Base confidence from level strength
            base_confidence = level.strength
            
            # Volume confirmation bonus
            volume_bonus = 0.0
            if volume_analysis and 'volume_spike_ratio' in volume_analysis:
                if volume_analysis['volume_spike_ratio'] > 2.0:
                    volume_bonus = 0.1
                elif volume_analysis['volume_spike_ratio'] > 1.5:
                    volume_bonus = 0.05
            
            # Touch count bonus
            touch_bonus = min(level.touch_count / 10.0, 0.1)
            
            # Consistency bonus
            consistency_bonus = level.consistency_score * 0.05
            
            # Age penalty (older levels are less reliable)
            age_penalty = max(0, (level.age_bars - 100) / 1000.0) * 0.1
            
            confidence = base_confidence + volume_bonus + touch_bonus + consistency_bonus - age_penalty
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Confidence score calculation failed: {e}")
            return level.strength
    
    def get_analysis_summary(self, enhanced_levels: List[EnhancedSRLevel]) -> Dict[str, Any]:
        """Get comprehensive analysis summary."""
        try:
            if not enhanced_levels:
                return {}
            
            # Calculate statistics
            total_levels = len(enhanced_levels)
            support_levels = len([l for l in enhanced_levels if l.type == 'support'])
            resistance_levels = len([l for l in enhanced_levels if l.type == 'resistance'])
            psychological_levels = len([l for l in enhanced_levels if l.psychological_level_info.get('is_psychological', False)])
            
            avg_strength = np.mean([l.strength for l in enhanced_levels])
            avg_confidence = np.mean([l.confidence_score for l in enhanced_levels])
            avg_touch_count = np.mean([l.touch_count for l in enhanced_levels])
            
            # Volume statistics
            volume_analyses = [l.volume_analysis for l in enhanced_levels if l.volume_analysis]
            avg_volume_spike_ratio = np.mean([va.get('volume_spike_ratio', 1.0) for va in volume_analyses]) if volume_analyses else 1.0
            
            summary = {
                'total_levels': total_levels,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'psychological_levels': psychological_levels,
                'avg_strength': avg_strength,
                'avg_confidence': avg_confidence,
                'avg_touch_count': avg_touch_count,
                'avg_volume_spike_ratio': avg_volume_spike_ratio,
                'strongest_level': max(enhanced_levels, key=lambda x: x.strength) if enhanced_levels else None,
                'most_confident_level': max(enhanced_levels, key=lambda x: x.confidence_score) if enhanced_levels else None
            }
            
            # Log summary
            self.logger.info("📋 Analysis Summary:")
            self.logger.info(f"   - Total levels: {total_levels}")
            self.logger.info(f"   - Support: {support_levels}, Resistance: {resistance_levels}")
            self.logger.info(f"   - Psychological levels: {psychological_levels}")
            self.logger.info(f"   - Average strength: {avg_strength:.3f}")
            self.logger.info(f"   - Average confidence: {avg_confidence:.3f}")
            self.logger.info(f"   - Average touch count: {avg_touch_count:.1f}")
            self.logger.info(f"   - Average volume spike ratio: {avg_volume_spike_ratio:.2f}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Analysis summary generation failed: {e}")
            return {}

# Backward compatibility - keep the old class name for existing imports
SREnsemblePredictor = SimplifiedSRPredictor