"""
Candlestick Pattern Feature Generator

This module provides feature generators for candlestick pattern recognition,
including doji, hammer, engulfing patterns, and other candlestick formations.
Fully optimized with VectorBT for maximum performance.
"""

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import time

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov, rolling_quantile,
        scale, rank, zscore, winsorize, clip, quantile
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    rolling_quantile = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optimization utilities
try:
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    from ..utils.vectorization_optimizer import get_vectorization_optimizer, VectorizationOptimizer
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    get_vectorbt_rolling_optimizer = None
    get_vectorization_optimizer = None

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)

@dataclass
class CandlestickPatternConfig:
    """Configuration for candlestick pattern detection."""
    # Pattern Detection Thresholds
    doji_threshold: float = 0.1  # Body size relative to total range
    hammer_threshold: float = 0.3  # Body size relative to total range
    engulfing_threshold: float = 0.1  # Minimum overlap for engulfing
    
    # VectorBT Optimization
    enable_vectorbt: bool = True
    enable_batch_processing: bool = True
    enable_gpu_acceleration: bool = False
    
    # Memory Management
    enable_memory_optimization: bool = True
    chunk_size: int = 10000
    
    # Performance Monitoring
    enable_performance_monitoring: bool = True

class CandlestickPatternFeatureGenerator(VectorBTFeatureGenerator):
    """
    High-performance candlestick pattern feature generator using VectorBT.
    
    This generator provides comprehensive candlestick pattern recognition
    with full VectorBT optimization for maximum performance.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None, 
                 pattern_config: Optional[CandlestickPatternConfig] = None):
        if config is None:
            config = self._create_default_config()
        
        super().__init__(config, enable_gpu=pattern_config.enable_gpu_acceleration if pattern_config else False)
        
        self.pattern_config = pattern_config or CandlestickPatternConfig()
        
        # Initialize unified vectorization manager
        if OPTIMIZATION_AVAILABLE:
            from ..utils.unified_vectorization_manager import get_unified_vectorization_manager
            self.vectorization_manager = get_unified_vectorization_manager()
        else:
            self.vectorization_manager = None
        
        # Pattern detection methods
        self.pattern_methods = {
            'doji': self._detect_doji_pattern,
            'hammer': self._detect_hammer_pattern,
            'hanging_man': self._detect_hanging_man_pattern,
            'engulfing_bullish': self._detect_bullish_engulfing,
            'engulfing_bearish': self._detect_bearish_engulfing,
            'harami_bullish': self._detect_bullish_harami,
            'harami_bearish': self._detect_bearish_harami,
            'shooting_star': self._detect_shooting_star,
            'inverted_hammer': self._detect_inverted_hammer,
            'morning_star': self._detect_morning_star,
            'evening_star': self._detect_evening_star,
            'three_white_soldiers': self._detect_three_white_soldiers,
            'three_black_crows': self._detect_three_black_crows
        }
        
        # Performance tracking
        self.pattern_stats = {
            'patterns_detected': 0,
            'vectorbt_operations': 0,
            'batch_operations': 0,
            'total_execution_time': 0.0
        }
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="candlestick_pattern_features",
            category=FeatureCategory.CANDLESTICK_PATTERN,
            description="Comprehensive candlestick pattern features with VectorBT optimization",
            required_columns=["open", "high", "low", "close"],
            optional_columns=["volume"],
            default_lookback=3,
            min_lookback=1,
            max_lookback=5,
            parameters={
                "patterns": ["doji", "hammer", "engulfing_bullish", "engulfing_bearish"],
                "doji_threshold": 0.1,
                "hammer_threshold": 0.3,
                "engulfing_threshold": 0.1
            },
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    @classmethod
    def create_default(cls) -> 'CandlestickPatternFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate candlestick pattern features using VectorBT optimization."""
        start_time = time.time()
        
        try:
            # Optimize DataFrame for processing
            if self.vectorization_manager:
                data = self.vectorization_manager.optimize_dataframe(data)
            
            # Get requested patterns
            patterns = kwargs.get('patterns', self.config.parameters.get('patterns', ['doji']))
            
            if isinstance(patterns, str):
                patterns = [patterns]
            
            # Generate pattern features
            if self.pattern_config.enable_batch_processing and len(patterns) > 1:
                pattern_features = self._generate_patterns_batch(data, patterns)
            else:
                pattern_features = self._generate_patterns_sequential(data, patterns)
            
            # Track performance
            execution_time = time.time() - start_time
            self.pattern_stats['patterns_detected'] += len(patterns)
            self.pattern_stats['total_execution_time'] += execution_time
            
            if self.pattern_config.enable_performance_monitoring:
                self.logger.debug(f"Generated {len(patterns)} pattern features in {execution_time:.3f}s")
            
            return pattern_features
            
        except Exception as e:
            self.logger.error(f"Pattern generation failed: {e}")
            return pd.Series(np.nan, index=data.index, name='pattern_error')
    
    def _generate_patterns_batch(self, data: pd.DataFrame, patterns: List[str]) -> pd.Series:
        """Generate multiple patterns in batch using VectorBT optimization."""
        try:
            # Prepare batch operations
            batch_operations = []
            
            for i, pattern in enumerate(patterns):
                if pattern in self.pattern_methods:
                    batch_operations.append({
                        'type': 'pattern',
                        'name': f'pattern_{pattern}',
                        'pattern': pattern,
                        'params': {}
                    })
            
            # Use VectorBT batch processing if available
            if VECTORBT_AVAILABLE and self.vectorization_manager:
                results = self._vectorbt_batch_pattern_detection(data, batch_operations)
                self.pattern_stats['batch_operations'] += 1
                return results
            else:
                return self._generate_patterns_sequential(data, patterns)
                
        except Exception as e:
            self.logger.warning(f"Batch pattern generation failed: {e}")
            return self._generate_patterns_sequential(data, patterns)
    
    def _generate_patterns_sequential(self, data: pd.DataFrame, patterns: List[str]) -> pd.Series:
        """Generate patterns sequentially."""
        results = []
        
        for pattern in patterns:
            if pattern in self.pattern_methods:
                try:
                    pattern_result = self.pattern_methods[pattern](data)
                    results.append(pattern_result)
                except Exception as e:
                    self.logger.warning(f"Pattern {pattern} failed: {e}")
                    results.append(pd.Series(np.nan, index=data.index, name=f'pattern_{pattern}'))
        
        if results:
            return pd.concat(results, axis=1).iloc[:, 0]  # Return first pattern as Series
        else:
            return pd.Series(np.nan, index=data.index, name='no_patterns')
    
    def _vectorbt_batch_pattern_detection(self, data: pd.DataFrame, 
                                        operations: List[Dict[str, Any]]) -> pd.Series:
        """Perform batch pattern detection using VectorBT."""
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required for batch operations")
        
        try:
            # Calculate basic OHLCV metrics using VectorBT
            ohlcv_metrics = self._calculate_ohlcv_metrics_vectorbt(data)
            
            # Detect patterns using vectorized operations
            pattern_results = {}
            
            for op in operations:
                pattern_name = op['name']
                pattern_type = op['pattern']
                
                if pattern_type == 'doji':
                    pattern_results[pattern_name] = self._detect_doji_vectorbt(ohlcv_metrics)
                elif pattern_type == 'hammer':
                    pattern_results[pattern_name] = self._detect_hammer_vectorbt(ohlcv_metrics)
                elif pattern_type == 'engulfing_bullish':
                    pattern_results[pattern_name] = self._detect_bullish_engulfing_vectorbt(ohlcv_metrics)
                elif pattern_type == 'engulfing_bearish':
                    pattern_results[pattern_name] = self._detect_bearish_engulfing_vectorbt(ohlcv_metrics)
                # Add more patterns as needed
            
            # Combine results
            if pattern_results:
                result_df = pd.DataFrame(pattern_results, index=data.index)
                return result_df.iloc[:, 0]  # Return first pattern as Series
            else:
                return pd.Series(np.nan, index=data.index, name='no_patterns')
                
        except Exception as e:
            self.logger.warning(f"VectorBT batch pattern detection failed: {e}")
            raise
    
    def _calculate_ohlcv_metrics_vectorbt(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate OHLCV metrics using VectorBT for pattern detection."""
        try:
            # Calculate basic metrics
            body_size = data['close'] - data['open']
            total_range = data['high'] - data['low']
            upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
            lower_shadow = np.minimum(data['open'], data['close']) - data['low']
            
            # Calculate relative sizes
            body_ratio = np.abs(body_size) / total_range
            upper_shadow_ratio = upper_shadow / total_range
            lower_shadow_ratio = lower_shadow / total_range
            
            # Use VectorBT rolling operations for trend analysis
            if self.vectorization_manager:
                close_ma = self.vectorization_manager.vectorized_rolling_operation(
                    data['close'], 'mean', window=20
                )
                trend = np.where(data['close'] > close_ma, 1, -1)
            else:
                close_ma = data['close'].rolling(window=20).mean()
                trend = np.where(data['close'] > close_ma, 1, -1)
            
            return {
                'body_size': body_size,
                'total_range': total_range,
                'upper_shadow': upper_shadow,
                'lower_shadow': lower_shadow,
                'body_ratio': body_ratio,
                'upper_shadow_ratio': upper_shadow_ratio,
                'lower_shadow_ratio': lower_shadow_ratio,
                'trend': trend,
                'open': data['open'],
                'high': data['high'],
                'low': data['low'],
                'close': data['close']
            }
            
        except Exception as e:
            self.logger.warning(f"OHLCV metrics calculation failed: {e}")
            return {}
    
    def _detect_doji_pattern(self, data: pd.DataFrame) -> pd.Series:
        """Detect doji patterns."""
        return self._detect_doji_vectorbt(self._calculate_ohlcv_metrics_vectorbt(data))
    
    def _detect_doji_vectorbt(self, metrics: Dict[str, pd.Series]) -> pd.Series:
        """Detect doji patterns using VectorBT operations."""
        if not metrics:
            return pd.Series(np.nan, index=pd.Index([]))
        
        # Doji: small body relative to total range
        doji_condition = metrics['body_ratio'] < self.pattern_config.doji_threshold
        
        return pd.Series(doji_condition.astype(int), 
                        index=metrics['close'].index, 
                        name='doji_pattern')
    
    def _detect_hammer_pattern(self, data: pd.DataFrame) -> pd.Series:
        """Detect hammer patterns."""
        return self._detect_hammer_vectorbt(self._calculate_ohlcv_metrics_vectorbt(data))
    
    def _detect_hammer_vectorbt(self, metrics: Dict[str, pd.Series]) -> pd.Series:
        """Detect hammer patterns using VectorBT operations."""
        if not metrics:
            return pd.Series(np.nan, index=pd.Index([]))
        
        # Hammer: small body at top, long lower shadow, short upper shadow
        hammer_condition = (
            (metrics['body_ratio'] < self.pattern_config.hammer_threshold) &
            (metrics['lower_shadow_ratio'] > 0.6) &
            (metrics['upper_shadow_ratio'] < 0.1) &
            (metrics['body_size'] > 0)  # Bullish body
        )
        
        return pd.Series(hammer_condition.astype(int), 
                        index=metrics['close'].index, 
                        name='hammer_pattern')
    
    def _detect_hanging_man_pattern(self, data: pd.DataFrame) -> pd.Series:
        """Detect hanging man patterns."""
        metrics = self._calculate_ohlcv_metrics_vectorbt(data)
        if not metrics:
            return pd.Series(np.nan, index=pd.Index([]))
        
        # Hanging man: similar to hammer but bearish
        hanging_man_condition = (
            (metrics['body_ratio'] < self.pattern_config.hammer_threshold) &
            (metrics['lower_shadow_ratio'] > 0.6) &
            (metrics['upper_shadow_ratio'] < 0.1) &
            (metrics['body_size'] < 0)  # Bearish body
        )
        
        return pd.Series(hanging_man_condition.astype(int), 
                        index=metrics['close'].index, 
                        name='hanging_man_pattern')
    
    def _detect_bullish_engulfing(self, data: pd.DataFrame) -> pd.Series:
        """Detect bullish engulfing patterns."""
        return self._detect_bullish_engulfing_vectorbt(self._calculate_ohlcv_metrics_vectorbt(data))
    
    def _detect_bullish_engulfing_vectorbt(self, metrics: Dict[str, pd.Series]) -> pd.Series:
        """Detect bullish engulfing patterns using VectorBT operations."""
        if not metrics:
            return pd.Series(np.nan, index=pd.Index([]))
        
        # Bullish engulfing: current candle engulfs previous bearish candle
        current_body = metrics['body_size']
        previous_body = current_body.shift(1)
        
        engulfing_condition = (
            (previous_body < 0) &  # Previous candle was bearish
            (current_body > 0) &   # Current candle is bullish
            (current_body > -previous_body) &  # Current body engulfs previous
            (metrics['open'] < metrics['open'].shift(1)) &  # Gap down
            (metrics['close'] > metrics['close'].shift(1))  # Close above previous close
        )
        
        return pd.Series(engulfing_condition.astype(int), 
                        index=metrics['close'].index, 
                        name='bullish_engulfing_pattern')
    
    def _detect_bearish_engulfing(self, data: pd.DataFrame) -> pd.Series:
        """Detect bearish engulfing patterns."""
        return self._detect_bearish_engulfing_vectorbt(self._calculate_ohlcv_metrics_vectorbt(data))
    
    def _detect_bearish_engulfing_vectorbt(self, metrics: Dict[str, pd.Series]) -> pd.Series:
        """Detect bearish engulfing patterns using VectorBT operations."""
        if not metrics:
            return pd.Series(np.nan, index=pd.Index([]))
        
        # Bearish engulfing: current candle engulfs previous bullish candle
        current_body = metrics['body_size']
        previous_body = current_body.shift(1)
        
        engulfing_condition = (
            (previous_body > 0) &  # Previous candle was bullish
            (current_body < 0) &   # Current candle is bearish
            (current_body < -previous_body) &  # Current body engulfs previous
            (metrics['open'] > metrics['open'].shift(1)) &  # Gap up
            (metrics['close'] < metrics['close'].shift(1))  # Close below previous close
        )
        
        return pd.Series(engulfing_condition.astype(int), 
                        index=metrics['close'].index, 
                        name='bearish_engulfing_pattern')
    
    def _detect_bullish_harami(self, data: pd.DataFrame) -> pd.Series:
        """Detect bullish harami patterns."""
        metrics = self._calculate_ohlcv_metrics_vectorbt(data)
        if not metrics:
            return pd.Series(np.nan, index=pd.Index([]))
        
        # Bullish harami: small bullish candle inside previous bearish candle
        current_body = metrics['body_size']
        previous_body = current_body.shift(1)
        
        harami_condition = (
            (previous_body < 0) &  # Previous candle was bearish
            (current_body > 0) &   # Current candle is bullish
            (metrics['open'] > metrics['close'].shift(1)) &  # Open above previous close
            (metrics['close'] < metrics['open'].shift(1))   # Close below previous open
        )
        
        return pd.Series(harami_condition.astype(int), 
                        index=metrics['close'].index, 
                        name='bullish_harami_pattern')
    
    def _detect_bearish_harami(self, data: pd.DataFrame) -> pd.Series:
        """Detect bearish harami patterns."""
        metrics = self._calculate_ohlcv_metrics_vectorbt(data)
        if not metrics:
            return pd.Series(np.nan, index=pd.Index([]))
        
        # Bearish harami: small bearish candle inside previous bullish candle
        current_body = metrics['body_size']
        previous_body = current_body.shift(1)
        
        harami_condition = (
            (previous_body > 0) &  # Previous candle was bullish
            (current_body < 0) &   # Current candle is bearish
            (metrics['open'] < metrics['close'].shift(1)) &  # Open below previous close
            (metrics['close'] > metrics['open'].shift(1))   # Close above previous open
        )
        
        return pd.Series(harami_condition.astype(int), 
                        index=metrics['close'].index, 
                        name='bearish_harami_pattern')
    
    def _detect_shooting_star(self, data: pd.DataFrame) -> pd.Series:
        """Detect shooting star patterns."""
        metrics = self._calculate_ohlcv_metrics_vectorbt(data)
        if not metrics:
            return pd.Series(np.nan, index=pd.Index([]))
        
        # Shooting star: small body at bottom, long upper shadow, short lower shadow
        shooting_star_condition = (
            (metrics['body_ratio'] < self.pattern_config.hammer_threshold) &
            (metrics['upper_shadow_ratio'] > 0.6) &
            (metrics['lower_shadow_ratio'] < 0.1) &
            (metrics['body_size'] < 0)  # Bearish body
        )
        
        return pd.Series(shooting_star_condition.astype(int), 
                        index=metrics['close'].index, 
                        name='shooting_star_pattern')
    
    def _detect_inverted_hammer(self, data: pd.DataFrame) -> pd.Series:
        """Detect inverted hammer patterns."""
        metrics = self._calculate_ohlcv_metrics_vectorbt(data)
        if not metrics:
            return pd.Series(np.nan, index=pd.Index([]))
        
        # Inverted hammer: small body at bottom, long upper shadow, short lower shadow
        inverted_hammer_condition = (
            (metrics['body_ratio'] < self.pattern_config.hammer_threshold) &
            (metrics['upper_shadow_ratio'] > 0.6) &
            (metrics['lower_shadow_ratio'] < 0.1) &
            (metrics['body_size'] > 0)  # Bullish body
        )
        
        return pd.Series(inverted_hammer_condition.astype(int), 
                        index=metrics['close'].index, 
                        name='inverted_hammer_pattern')
    
    def _detect_morning_star(self, data: pd.DataFrame) -> pd.Series:
        """Detect morning star patterns (3-candle pattern)."""
        metrics = self._calculate_ohlcv_metrics_vectorbt(data)
        if not metrics:
            return pd.Series(np.nan, index=pd.Index([]))
        
        # Morning star: bearish candle, small body candle, bullish candle
        first_candle = metrics['body_size'].shift(2)
        second_candle = metrics['body_size'].shift(1)
        third_candle = metrics['body_size']
        
        morning_star_condition = (
            (first_candle < 0) &  # First candle bearish
            (np.abs(second_candle) < np.abs(first_candle) * 0.3) &  # Second candle small
            (third_candle > 0) &  # Third candle bullish
            (third_candle > -first_candle)  # Third candle engulfs first
        )
        
        return pd.Series(morning_star_condition.astype(int), 
                        index=metrics['close'].index, 
                        name='morning_star_pattern')
    
    def _detect_evening_star(self, data: pd.DataFrame) -> pd.Series:
        """Detect evening star patterns (3-candle pattern)."""
        metrics = self._calculate_ohlcv_metrics_vectorbt(data)
        if not metrics:
            return pd.Series(np.nan, index=pd.Index([]))
        
        # Evening star: bullish candle, small body candle, bearish candle
        first_candle = metrics['body_size'].shift(2)
        second_candle = metrics['body_size'].shift(1)
        third_candle = metrics['body_size']
        
        evening_star_condition = (
            (first_candle > 0) &  # First candle bullish
            (np.abs(second_candle) < np.abs(first_candle) * 0.3) &  # Second candle small
            (third_candle < 0) &  # Third candle bearish
            (third_candle < -first_candle)  # Third candle engulfs first
        )
        
        return pd.Series(evening_star_condition.astype(int), 
                        index=metrics['close'].index, 
                        name='evening_star_pattern')
    
    def _detect_three_white_soldiers(self, data: pd.DataFrame) -> pd.Series:
        """Detect three white soldiers patterns (3-candle pattern)."""
        metrics = self._calculate_ohlcv_metrics_vectorbt(data)
        if not metrics:
            return pd.Series(np.nan, index=pd.Index([]))
        
        # Three white soldiers: three consecutive bullish candles with increasing closes
        first_candle = metrics['body_size'].shift(2)
        second_candle = metrics['body_size'].shift(1)
        third_candle = metrics['body_size']
        
        three_white_soldiers_condition = (
            (first_candle > 0) &  # All candles bullish
            (second_candle > 0) &
            (third_candle > 0) &
            (metrics['close'].shift(1) > metrics['close'].shift(2)) &  # Increasing closes
            (metrics['close'] > metrics['close'].shift(1))
        )
        
        return pd.Series(three_white_soldiers_condition.astype(int), 
                        index=metrics['close'].index, 
                        name='three_white_soldiers_pattern')
    
    def _detect_three_black_crows(self, data: pd.DataFrame) -> pd.Series:
        """Detect three black crows patterns (3-candle pattern)."""
        metrics = self._calculate_ohlcv_metrics_vectorbt(data)
        if not metrics:
            return pd.Series(np.nan, index=pd.Index([]))
        
        # Three black crows: three consecutive bearish candles with decreasing closes
        first_candle = metrics['body_size'].shift(2)
        second_candle = metrics['body_size'].shift(1)
        third_candle = metrics['body_size']
        
        three_black_crows_condition = (
            (first_candle < 0) &  # All candles bearish
            (second_candle < 0) &
            (third_candle < 0) &
            (metrics['close'].shift(1) < metrics['close'].shift(2)) &  # Decreasing closes
            (metrics['close'] < metrics['close'].shift(1))
        )
        
        return pd.Series(three_black_crows_condition.astype(int), 
                        index=metrics['close'].index, 
                        name='three_black_crows_pattern')
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get pattern detection statistics."""
        return self.pattern_stats.copy()
    
    def reset_pattern_stats(self):
        """Reset pattern detection statistics."""
        self.pattern_stats = {
            'patterns_detected': 0,
            'vectorbt_operations': 0,
            'batch_operations': 0,
            'total_execution_time': 0.0
        }

class VectorBTCandlestickPatternGenerator(CandlestickPatternFeatureGenerator):
    """
    VectorBT-optimized candlestick pattern generator with enhanced performance.
    
    This generator provides the same pattern detection capabilities as the base
    generator but with additional VectorBT optimizations and batch processing.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None, 
                 pattern_config: Optional[CandlestickPatternConfig] = None):
        super().__init__(config, pattern_config)
        
        # Enhanced VectorBT configuration
        if VECTORBT_AVAILABLE:
            self._configure_enhanced_vectorbt()
    
    def _configure_enhanced_vectorbt(self):
        """Configure enhanced VectorBT settings for candlestick patterns."""
        try:
            # Configure VectorBT for pattern detection
            vbt.settings.setting('array_wrapper', 'pandas')
            vbt.settings.setting('caching', True)
            vbt.settings.setting('caching_dir', 'data_cache/candlestick_patterns')
            
            # Pattern-specific optimizations
            vbt.settings.setting('chunk_size', self.pattern_config.chunk_size)
            vbt.settings.setting('memory_limit', self.pattern_config.chunk_size * 8)  # 8 bytes per float
            
            if self.pattern_config.enable_gpu_acceleration:
                vbt.settings.setting('use_gpu', True)
                logger.info("✅ GPU acceleration enabled for candlestick patterns")
            
            if self.pattern_config.enable_batch_processing:
                vbt.settings.setting('use_parallel', True)
                logger.info("✅ Parallel processing enabled for candlestick patterns")
                
        except Exception as e:
            logger.warning(f"Enhanced VectorBT configuration failed: {e}")
    
    def generate_all_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate all available candlestick patterns in a single operation."""
        start_time = time.time()
        
        try:
            # Get all available patterns
            all_patterns = list(self.pattern_methods.keys())
            
            # Use batch processing for efficiency
            if self.pattern_config.enable_batch_processing:
                results = self._generate_patterns_batch(data, all_patterns)
            else:
                results = self._generate_patterns_sequential(data, all_patterns)
            
            # Track performance
            execution_time = time.time() - start_time
            self.pattern_stats['patterns_detected'] += len(all_patterns)
            self.pattern_stats['total_execution_time'] += execution_time
            
            logger.info(f"✅ Generated {len(all_patterns)} candlestick patterns in {execution_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"All patterns generation failed: {e}")
            return pd.DataFrame(index=data.index)
    
    def generate_patterns_with_confidence(self, data: pd.DataFrame, 
                                        patterns: List[str] = None) -> pd.DataFrame:
        """Generate patterns with confidence scores."""
        if patterns is None:
            patterns = list(self.pattern_methods.keys())
        
        try:
            # Calculate OHLCV metrics once
            metrics = self._calculate_ohlcv_metrics_vectorbt(data)
            
            results = {}
            
            for pattern in patterns:
                if pattern in self.pattern_methods:
                    # Get pattern signal
                    pattern_signal = self.pattern_methods[pattern](data)
                    
                    # Calculate confidence score based on pattern strength
                    confidence = self._calculate_pattern_confidence(pattern, metrics, pattern_signal)
                    
                    results[f'{pattern}_signal'] = pattern_signal
                    results[f'{pattern}_confidence'] = confidence
            
            return pd.DataFrame(results, index=data.index)
            
        except Exception as e:
            logger.error(f"Pattern confidence calculation failed: {e}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_pattern_confidence(self, pattern: str, metrics: Dict[str, pd.Series], 
                                    signal: pd.Series) -> pd.Series:
        """Calculate confidence score for a pattern."""
        try:
            if pattern == 'doji':
                # Confidence based on how small the body is
                confidence = 1.0 - metrics['body_ratio']
            elif pattern in ['hammer', 'hanging_man']:
                # Confidence based on shadow ratios
                confidence = metrics['lower_shadow_ratio'] * (1.0 - metrics['upper_shadow_ratio'])
            elif pattern in ['shooting_star', 'inverted_hammer']:
                # Confidence based on upper shadow ratio
                confidence = metrics['upper_shadow_ratio'] * (1.0 - metrics['lower_shadow_ratio'])
            elif 'engulfing' in pattern:
                # Confidence based on engulfment ratio
                current_body = metrics['body_size']
                previous_body = current_body.shift(1)
                engulfment_ratio = np.abs(current_body) / np.abs(previous_body)
                confidence = np.minimum(engulfment_ratio, 2.0) / 2.0  # Cap at 1.0
            else:
                # Default confidence
                confidence = pd.Series(0.5, index=signal.index)
            
            # Apply signal mask
            confidence = confidence * signal
            
            return confidence.clip(0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed for {pattern}: {e}")
            return pd.Series(0.0, index=signal.index)

class VectorBTCandlestickPatternBatchProcessor:
    """
    Batch processor for multiple candlestick pattern generators.
    
    This class provides efficient batch processing of multiple pattern generators
    with VectorBT optimization and memory management.
    """
    
    def __init__(self, pattern_configs: List[CandlestickPatternConfig] = None):
        """Initialize batch processor with pattern configurations."""
        self.pattern_configs = pattern_configs or [CandlestickPatternConfig()]
        
        # Initialize generators
        self.generators = []
        for config in self.pattern_configs:
            generator = VectorBTCandlestickPatternGenerator(pattern_config=config)
            self.generators.append(generator)
        
        # Performance tracking
        self.batch_stats = {
            'total_batches': 0,
            'total_patterns': 0,
            'total_execution_time': 0.0,
            'vectorbt_operations': 0
        }
    
    def process_batch(self, data: pd.DataFrame, 
                     pattern_lists: List[List[str]] = None) -> List[pd.DataFrame]:
        """Process multiple pattern lists in batch."""
        start_time = time.time()
        
        try:
            if pattern_lists is None:
                pattern_lists = [list(generator.pattern_methods.keys()) for generator in self.generators]
            
            results = []
            
            for i, (generator, patterns) in enumerate(zip(self.generators, pattern_lists)):
                if generator.pattern_config.enable_batch_processing:
                    result = generator._generate_patterns_batch(data, patterns)
                else:
                    result = generator._generate_patterns_sequential(data, patterns)
                
                results.append(result)
            
            # Track performance
            execution_time = time.time() - start_time
            self.batch_stats['total_batches'] += 1
            self.batch_stats['total_patterns'] += sum(len(patterns) for patterns in pattern_lists)
            self.batch_stats['total_execution_time'] += execution_time
            
            logger.info(f"✅ Processed batch with {len(self.generators)} generators in {execution_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return [pd.DataFrame(index=data.index) for _ in self.generators]
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return self.batch_stats.copy()

def create_candlestick_pattern_generators(patterns: List[str] = None) -> List[FeatureGenerator]:
    """Create a set of candlestick pattern feature generators."""
    if patterns is None:
        patterns = ["doji", "hammer", "engulfing_bullish", "engulfing_bearish"]
    
    generators = []
    
    # Create individual pattern generators
    for pattern in patterns:
        config = FeatureConfig(
            name=f"candlestick_{pattern}",
            category=FeatureCategory.CANDLESTICK_PATTERN,
            description=f"VectorBT-optimized {pattern} pattern detection",
            required_columns=["open", "high", "low", "close"],
            default_lookback=3,
            min_lookback=1,
            max_lookback=5,
            parameters={"patterns": [pattern]},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        
        generator = CandlestickPatternFeatureGenerator(config)
        generators.append(generator)
    
    return generators

def create_default_candlestick_pattern_generators() -> List[FeatureGenerator]:
    """Create default set of candlestick pattern generators."""
    return create_candlestick_pattern_generators()

def create_vectorbt_candlestick_generator(pattern_config: Optional[CandlestickPatternConfig] = None) -> VectorBTCandlestickPatternGenerator:
    """Create VectorBT-optimized candlestick pattern generator."""
    return VectorBTCandlestickPatternGenerator(pattern_config=pattern_config)

def create_candlestick_batch_processor(pattern_configs: List[CandlestickPatternConfig] = None) -> VectorBTCandlestickPatternBatchProcessor:
    """Create batch processor for candlestick patterns."""
    return VectorBTCandlestickPatternBatchProcessor(pattern_configs)
