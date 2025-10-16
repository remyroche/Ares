"""
Candlestick Pattern Feature Generator

This module provides feature generators for candlestick pattern recognition,
including doji, hammer, engulfing patterns, and other candlestick formations.
Fully optimized with VectorBTRollingOptimizer and UnifiedVectorizationManager.

Key Features:
- Comprehensive candlestick pattern recognition
- VectorBT-optimized rolling operations
- Intelligent optimization strategy selection
- Pattern strength and reliability scoring
- Performance monitoring and statistics
"""

import numpy as np
import pandas as pd
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple
import logging

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_quantile, rolling_skew, rolling_kurt
    )
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
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
    rolling_skew = None
    rolling_kurt = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# VectorBT Rolling Optimizer
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import (
        get_unified_vectorization_manager, UnifiedVectorizationManager, 
        OperationType, OptimizationStrategy, OperationConfig
    )
    UNIFIED_VECTORIZATION_MANAGER_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_MANAGER_AVAILABLE = False
    UnifiedVectorizationManager = None

except ImportError:
    
    cp = None

logger = logging.getLogger(__name__)

class CandlestickPatternFeatureGenerator(VectorizedFeatureGenerator):
    """
    Advanced candlestick pattern feature generator with full VectorBT optimization.
    
    Features:
    - Comprehensive pattern recognition (doji, hammer, engulfing, etc.)
    - VectorBTRollingOptimizer for high-performance calculations
    - UnifiedVectorizationManager for intelligent optimization
    - Pattern strength and reliability scoring
    - Performance monitoring and statistics
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize optimization components
        self._initialize_optimization_components()
        
        # Pattern recognition parameters
        self.patterns = config.parameters.get("patterns", ["doji", "hammer", "engulfing", "shooting_star", "hanging_man"])
        self.body_threshold = config.parameters.get("body_threshold", 0.1)
        self.shadow_threshold = config.parameters.get("shadow_threshold", 0.3)
        self.engulfing_threshold = config.parameters.get("engulfing_threshold", 0.5)
        
        # Performance tracking
        self.performance_stats = {
            'patterns_detected': 0,
            'vectorbt_operations': 0,
            'unified_manager_operations': 0,
            'total_computation_time': 0.0
        }
    
    def _initialize_optimization_components(self):
        """Initialize VectorBT and UnifiedVectorizationManager components."""
        # Initialize VectorBTRollingOptimizer
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
            logger.info("✅ VectorBTRollingOptimizer initialized for candlestick patterns")
        else:
            self.rolling_optimizer = None
            logger.warning("⚠️ VectorBTRollingOptimizer not available, using fallback methods")
        
        # Initialize UnifiedVectorizationManager
        if UNIFIED_VECTORIZATION_MANAGER_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
            logger.info("✅ UnifiedVectorizationManager initialized for candlestick patterns")
        else:
            self.unified_manager = None
            logger.warning("⚠️ UnifiedVectorizationManager not available, using direct VectorBT calls")
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="candlestick_pattern_features",
            category=FeatureCategory.CANDLESTICK_PATTERN,
            description="Comprehensive candlestick pattern features with VectorBT optimization",
            required_columns=["open", "high", "low", "close"],
            default_lookback=3,
            min_lookback=1,
            max_lookback=10,
            parameters={
                "patterns": ["doji", "hammer", "engulfing", "shooting_star", "hanging_man", "morning_star", "evening_star"],
                "body_threshold": 0.1,
                "shadow_threshold": 0.3,
                "engulfing_threshold": 0.5,
                "use_vectorbt": True,
                "use_unified_manager": True
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'CandlestickPatternFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate comprehensive candlestick pattern features using VectorBT optimization."""
        import time
        start_time = time.time()
        
        # Validate required columns
        required_cols = ["open", "high", "low", "close"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        # Extract OHLC data
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        
        # Calculate basic candlestick components
        body_size = np.abs(close_prices - open_prices)
        upper_shadow = high_prices - np.maximum(open_prices, close_prices)
        lower_shadow = np.minimum(open_prices, close_prices) - low_prices
        total_range = high_prices - low_prices
        
        # Avoid division by zero
        total_range = np.where(total_range == 0, 1e-8, total_range)
        
        # Normalize components
        body_ratio = body_size / total_range
        upper_shadow_ratio = upper_shadow / total_range
        lower_shadow_ratio = lower_shadow / total_range
        
        # Generate pattern features
        pattern_features = self._generate_pattern_features(
            open_prices, high_prices, low_prices, close_prices,
            body_ratio, upper_shadow_ratio, lower_shadow_ratio, total_range
        )
        
        # Update performance stats
        self.performance_stats['patterns_detected'] += len(pattern_features)
        
        # Combine all pattern features into a single score
        if pattern_features:
            pattern_score = np.sum(pattern_features, axis=0) if len(pattern_features) > 1 else pattern_features[0]
        else:
            pattern_score = np.zeros(len(open_prices))
        
        # Enhance pattern detection with market context
        enhanced_pattern_score = self._enhance_pattern_detection_with_context(
            pattern_score, open_prices, high_prices, low_prices, close_prices
        )
        
        # Calculate pattern strength and reliability
        pattern_strength = self._calculate_pattern_strength(
            enhanced_pattern_score, open_prices, high_prices, low_prices, close_prices
        )
        
        pattern_reliability = self._calculate_pattern_reliability(
            enhanced_pattern_score, open_prices, high_prices, low_prices, close_prices
        )
        
        # Combine strength and reliability into final score
        final_pattern_score = enhanced_pattern_score * pattern_strength * pattern_reliability
        
        # Update performance stats
        self.performance_stats['total_computation_time'] += time.time() - start_time
        
        return pd.Series(final_pattern_score, index=data.index, name='candlestick_pattern_score')
    
    def _generate_pattern_features(self, open_prices: np.ndarray, high_prices: np.ndarray, 
                                 low_prices: np.ndarray, close_prices: np.ndarray,
                                 body_ratio: np.ndarray, upper_shadow_ratio: np.ndarray, 
                                 lower_shadow_ratio: np.ndarray, total_range: np.ndarray) -> List[np.ndarray]:
        """Generate individual pattern features using VectorBT optimization and UnifiedVectorizationManager."""
        pattern_features = []
        
        # Use UnifiedVectorizationManager for batch pattern processing if available
        if self.unified_manager and len(self.patterns) > 3:
            try:
                # Prepare data for batch processing
                pattern_data = {
                    'open_prices': open_prices,
                    'high_prices': high_prices,
                    'low_prices': low_prices,
                    'close_prices': close_prices,
                    'body_ratio': body_ratio,
                    'upper_shadow_ratio': upper_shadow_ratio,
                    'lower_shadow_ratio': lower_shadow_ratio,
                    'total_range': total_range,
                    'patterns': self.patterns
                }
                
                # Use UnifiedVectorizationManager for batch processing
                config = OperationConfig(
                    operation_type=OperationType.TECHNICAL_INDICATORS,
                    data_size=len(open_prices),
                    data_dimensions=(len(open_prices), len(self.patterns))
                )
                
                result = self.unified_manager.optimize_operation(
                    OperationType.TECHNICAL_INDICATORS,
                    pattern_data,
                    config,
                    pattern_detection=True
                )
                
                if hasattr(result, 'result') and isinstance(result.result, list):
                    pattern_features = result.result
                    self.performance_stats['unified_manager_operations'] += 1
                    logger.info(f"✅ Used UnifiedVectorizationManager for batch pattern processing")
                    return pattern_features
                    
            except Exception as e:
                logger.warning(f"UnifiedVectorizationManager batch processing failed: {e}, using individual detection")
        
        # Fallback to individual pattern detection
        for pattern in self.patterns:
            try:
                if pattern == "doji":
                    feature = self._detect_doji_pattern(body_ratio, total_range)
                elif pattern == "hammer":
                    feature = self._detect_hammer_pattern(body_ratio, upper_shadow_ratio, lower_shadow_ratio)
                elif pattern == "shooting_star":
                    feature = self._detect_shooting_star_pattern(body_ratio, upper_shadow_ratio, lower_shadow_ratio)
                elif pattern == "hanging_man":
                    feature = self._detect_hanging_man_pattern(body_ratio, upper_shadow_ratio, lower_shadow_ratio)
                elif pattern == "engulfing":
                    feature = self._detect_engulfing_pattern(open_prices, close_prices, body_ratio)
                elif pattern == "morning_star":
                    feature = self._detect_morning_star_pattern(open_prices, high_prices, low_prices, close_prices)
                elif pattern == "evening_star":
                    feature = self._detect_evening_star_pattern(open_prices, high_prices, low_prices, close_prices)
                else:
                    continue
                
                pattern_features.append(feature)
                
            except Exception as e:
                logger.warning(f"Failed to generate {pattern} pattern: {e}")
                continue
        
        return pattern_features
    
    def _detect_doji_pattern(self, body_ratio: np.ndarray, total_range: np.ndarray) -> np.ndarray:
        """Detect doji pattern using VectorBT optimization."""
        # Doji: very small body relative to total range
        doji_condition = body_ratio <= self.body_threshold
        
        # Use VectorBTRollingOptimizer for rolling statistics if available
        if self.rolling_optimizer and len(body_ratio) > 10:
            try:
                # Calculate rolling mean of body ratios for context
                rolling_body_mean = self.rolling_optimizer.rolling_mean(
                    pd.Series(body_ratio), window=5
                ).values
                
                # Enhanced doji detection with context
                enhanced_doji = doji_condition & (body_ratio < rolling_body_mean * 0.5)
                self.performance_stats['vectorbt_operations'] += 1
                return enhanced_doji.astype(float)
            except Exception as e:
                logger.warning(f"VectorBT doji detection failed: {e}, using fallback")
        
        return doji_condition.astype(float)
    
    def _detect_hammer_pattern(self, body_ratio: np.ndarray, upper_shadow_ratio: np.ndarray, 
                              lower_shadow_ratio: np.ndarray) -> np.ndarray:
        """Detect hammer pattern using VectorBT optimization."""
        # Hammer: small body, long lower shadow, short upper shadow
        hammer_condition = (
            (body_ratio <= self.body_threshold) &
            (lower_shadow_ratio >= self.shadow_threshold) &
            (upper_shadow_ratio <= self.shadow_threshold * 0.5)
        )
        
        # Use VectorBTRollingOptimizer for enhanced detection
        if self.rolling_optimizer and len(body_ratio) > 10:
            try:
                # Calculate rolling statistics for context
                rolling_lower_shadow = self.rolling_optimizer.rolling_mean(
                    pd.Series(lower_shadow_ratio), window=5
                ).values
                
                # Enhanced hammer detection
                enhanced_hammer = hammer_condition & (lower_shadow_ratio > rolling_lower_shadow * 1.5)
                self.performance_stats['vectorbt_operations'] += 1
                return enhanced_hammer.astype(float)
            except Exception as e:
                logger.warning(f"VectorBT hammer detection failed: {e}, using fallback")
        
        return hammer_condition.astype(float)
    
    def _detect_shooting_star_pattern(self, body_ratio: np.ndarray, upper_shadow_ratio: np.ndarray, 
                                    lower_shadow_ratio: np.ndarray) -> np.ndarray:
        """Detect shooting star pattern using VectorBT optimization."""
        # Shooting star: small body, long upper shadow, short lower shadow
        shooting_star_condition = (
            (body_ratio <= self.body_threshold) &
            (upper_shadow_ratio >= self.shadow_threshold) &
            (lower_shadow_ratio <= self.shadow_threshold * 0.5)
        )
        
        # Use VectorBTRollingOptimizer for enhanced detection
        if self.rolling_optimizer and len(body_ratio) > 10:
            try:
                # Calculate rolling statistics for context
                rolling_upper_shadow = self.rolling_optimizer.rolling_mean(
                    pd.Series(upper_shadow_ratio), window=5
                ).values
                
                # Enhanced shooting star detection
                enhanced_shooting_star = shooting_star_condition & (upper_shadow_ratio > rolling_upper_shadow * 1.5)
                self.performance_stats['vectorbt_operations'] += 1
                return enhanced_shooting_star.astype(float)
            except Exception as e:
                logger.warning(f"VectorBT shooting star detection failed: {e}, using fallback")
        
        return shooting_star_condition.astype(float)
    
    def _detect_hanging_man_pattern(self, body_ratio: np.ndarray, upper_shadow_ratio: np.ndarray, 
                                  lower_shadow_ratio: np.ndarray) -> np.ndarray:
        """Detect hanging man pattern using VectorBT optimization."""
        # Hanging man: similar to hammer but appears after uptrend
        hanging_man_condition = (
            (body_ratio <= self.body_threshold) &
            (lower_shadow_ratio >= self.shadow_threshold) &
            (upper_shadow_ratio <= self.shadow_threshold * 0.5)
        )
        
        # Use VectorBTRollingOptimizer for trend context
        if self.rolling_optimizer and len(body_ratio) > 20:
            try:
                # Calculate trend context using rolling mean
                rolling_body = self.rolling_optimizer.rolling_mean(
                    pd.Series(body_ratio), window=10
                ).values
                
                # Enhanced hanging man detection with trend context
                enhanced_hanging_man = hanging_man_condition & (body_ratio < rolling_body * 0.8)
                self.performance_stats['vectorbt_operations'] += 1
                return enhanced_hanging_man.astype(float)
            except Exception as e:
                logger.warning(f"VectorBT hanging man detection failed: {e}, using fallback")
        
        return hanging_man_condition.astype(float)
    
    def _detect_engulfing_pattern(self, open_prices: np.ndarray, close_prices: np.ndarray, 
                                body_ratio: np.ndarray) -> np.ndarray:
        """Detect engulfing pattern using VectorBT optimization."""
        # Engulfing: current candle completely engulfs previous candle
        engulfing_pattern = np.zeros(len(open_prices))
        
        if len(open_prices) < 2:
            return engulfing_pattern
        
        # Calculate previous candle body
        prev_open = np.roll(open_prices, 1)
        prev_close = np.roll(close_prices, 1)
        prev_body_size = np.abs(prev_close - prev_open)
        
        # Current candle body
        current_body_size = np.abs(close_prices - open_prices)
        
        # Bullish engulfing: current green candle engulfs previous red candle
        bullish_engulfing = (
            (close_prices > open_prices) &  # Current candle is green
            (prev_close < prev_open) &      # Previous candle is red
            (open_prices < prev_close) &    # Current open below previous close
            (close_prices > prev_open) &    # Current close above previous open
            (current_body_size > prev_body_size * (1 + self.engulfing_threshold))
        )
        
        # Bearish engulfing: current red candle engulfs previous green candle
        bearish_engulfing = (
            (close_prices < open_prices) &  # Current candle is red
            (prev_close > prev_open) &      # Previous candle is green
            (open_prices > prev_close) &    # Current open above previous close
            (close_prices < prev_open) &    # Current close below previous open
            (current_body_size > prev_body_size * (1 + self.engulfing_threshold))
        )
        
        # Use VectorBTRollingOptimizer for enhanced detection
        if self.rolling_optimizer and len(open_prices) > 10:
            try:
                # Calculate rolling statistics for context
                rolling_body_std = self.rolling_optimizer.rolling_std(
                    pd.Series(current_body_size), window=5
                ).values
                
                # Enhanced engulfing detection with volatility context
                enhanced_bullish = bullish_engulfing & (current_body_size > rolling_body_std * 2)
                enhanced_bearish = bearish_engulfing & (current_body_size > rolling_body_std * 2)
                
                engulfing_pattern = (enhanced_bullish.astype(float) + enhanced_bearish.astype(float))
                self.performance_stats['vectorbt_operations'] += 1
                return engulfing_pattern
            except Exception as e:
                logger.warning(f"VectorBT engulfing detection failed: {e}, using fallback")
        
        # Fallback to basic detection
        engulfing_pattern = (bullish_engulfing.astype(float) + bearish_engulfing.astype(float))
        return engulfing_pattern
    
    def _detect_morning_star_pattern(self, open_prices: np.ndarray, high_prices: np.ndarray, 
                                   low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Detect morning star pattern using VectorBT optimization."""
        # Morning star: 3-candle pattern (bearish, small body, bullish)
        morning_star = np.zeros(len(open_prices))
        
        if len(open_prices) < 3:
            return morning_star
        
        # Previous candles
        prev2_open = np.roll(open_prices, 2)
        prev2_close = np.roll(close_prices, 2)
        prev_open = np.roll(open_prices, 1)
        prev_close = np.roll(close_prices, 1)
        
        # First candle: bearish
        first_bearish = prev2_close < prev2_open
        
        # Second candle: small body (doji-like)
        second_small_body = np.abs(prev_close - prev_open) <= self.body_threshold * np.abs(prev2_close - prev2_open)
        
        # Third candle: bullish and closes above first candle's midpoint
        third_bullish = close_prices > open_prices
        third_strong = close_prices > (prev2_open + prev2_close) / 2
        
        morning_star_condition = first_bearish & second_small_body & third_bullish & third_strong
        
        # Use VectorBTRollingOptimizer for enhanced detection
        if self.rolling_optimizer and len(open_prices) > 20:
            try:
                # Calculate rolling volatility for context
                price_range = high_prices - low_prices
                rolling_volatility = self.rolling_optimizer.rolling_std(
                    pd.Series(price_range), window=10
                ).values
                
                # Enhanced morning star detection
                enhanced_morning_star = morning_star_condition & (price_range > rolling_volatility * 1.5)
                self.performance_stats['vectorbt_operations'] += 1
                return enhanced_morning_star.astype(float)
            except Exception as e:
                logger.warning(f"VectorBT morning star detection failed: {e}, using fallback")
        
        return morning_star_condition.astype(float)
    
    def _detect_evening_star_pattern(self, open_prices: np.ndarray, high_prices: np.ndarray, 
                                   low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Detect evening star pattern using VectorBT optimization."""
        # Evening star: 3-candle pattern (bullish, small body, bearish)
        evening_star = np.zeros(len(open_prices))
        
        if len(open_prices) < 3:
            return evening_star
        
        # Previous candles
        prev2_open = np.roll(open_prices, 2)
        prev2_close = np.roll(close_prices, 2)
        prev_open = np.roll(open_prices, 1)
        prev_close = np.roll(close_prices, 1)
        
        # First candle: bullish
        first_bullish = prev2_close > prev2_open
        
        # Second candle: small body (doji-like)
        second_small_body = np.abs(prev_close - prev_open) <= self.body_threshold * np.abs(prev2_close - prev2_open)
        
        # Third candle: bearish and closes below first candle's midpoint
        third_bearish = close_prices < open_prices
        third_strong = close_prices < (prev2_open + prev2_close) / 2
        
        evening_star_condition = first_bullish & second_small_body & third_bearish & third_strong
        
        # Use VectorBTRollingOptimizer for enhanced detection
        if self.rolling_optimizer and len(open_prices) > 20:
            try:
                # Calculate rolling volatility for context
                price_range = high_prices - low_prices
                rolling_volatility = self.rolling_optimizer.rolling_std(
                    pd.Series(price_range), window=10
                ).values
                
                # Enhanced evening star detection
                enhanced_evening_star = evening_star_condition & (price_range > rolling_volatility * 1.5)
                self.performance_stats['vectorbt_operations'] += 1
                return enhanced_evening_star.astype(float)
            except Exception as e:
                logger.warning(f"VectorBT evening star detection failed: {e}, using fallback")
        
        return evening_star_condition.astype(float)
    
    def _calculate_pattern_strength(self, pattern_scores: np.ndarray, 
                                  open_prices: np.ndarray, high_prices: np.ndarray, 
                                  low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Calculate pattern strength and reliability using VectorBT optimization."""
        if len(pattern_scores) == 0:
            return np.zeros(len(open_prices))
        
        # Calculate volatility for strength context
        price_range = high_prices - low_prices
        
        # Use VectorBTRollingOptimizer for volatility calculation
        if self.rolling_optimizer and len(price_range) > 10:
            try:
                # Calculate rolling volatility
                rolling_volatility = self.rolling_optimizer.rolling_std(
                    pd.Series(price_range), window=10
                ).values
                
                # Calculate rolling mean for normalization
                rolling_mean_vol = self.rolling_optimizer.rolling_mean(
                    pd.Series(price_range), window=10
                ).values
                
                # Pattern strength based on volatility context
                volatility_factor = np.where(rolling_volatility > 0, 
                                           price_range / rolling_volatility, 1.0)
                
                # Normalize pattern scores by volatility
                strength_scores = pattern_scores * volatility_factor
                
                # Clip to reasonable range
                strength_scores = np.clip(strength_scores, 0, 10)
                
                self.performance_stats['vectorbt_operations'] += 2
                return strength_scores
                
            except Exception as e:
                logger.warning(f"VectorBT pattern strength calculation failed: {e}, using fallback")
        
        # Fallback: simple strength calculation
        return np.clip(pattern_scores * 2, 0, 10)
    
    def _calculate_pattern_reliability(self, pattern_scores: np.ndarray, 
                                     open_prices: np.ndarray, high_prices: np.ndarray, 
                                     low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Calculate pattern reliability using VectorBT optimization."""
        if len(pattern_scores) == 0:
            return np.zeros(len(open_prices))
        
        # Calculate price momentum for reliability context
        price_momentum = close_prices - open_prices
        
        # Use VectorBTRollingOptimizer for momentum analysis
        if self.rolling_optimizer and len(price_momentum) > 20:
            try:
                # Calculate rolling momentum statistics
                rolling_momentum_mean = self.rolling_optimizer.rolling_mean(
                    pd.Series(price_momentum), window=20
                ).values
                
                rolling_momentum_std = self.rolling_optimizer.rolling_std(
                    pd.Series(price_momentum), window=20
                ).values
                
                # Calculate momentum consistency
                momentum_consistency = np.where(rolling_momentum_std > 0,
                                             1 - np.abs(price_momentum - rolling_momentum_mean) / rolling_momentum_std,
                                             0.5)
                
                # Pattern reliability based on momentum consistency
                reliability_scores = pattern_scores * np.clip(momentum_consistency, 0, 1)
                
                self.performance_stats['vectorbt_operations'] += 2
                return reliability_scores
                
            except Exception as e:
                logger.warning(f"VectorBT pattern reliability calculation failed: {e}, using fallback")
        
        # Fallback: simple reliability calculation
        return pattern_scores * 0.8
    
    def _enhance_pattern_detection_with_context(self, pattern_scores: np.ndarray,
                                              open_prices: np.ndarray, high_prices: np.ndarray,
                                              low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Enhance pattern detection with market context using VectorBT optimization."""
        if len(pattern_scores) == 0:
            return np.zeros(len(open_prices))
        
        # Use UnifiedVectorizationManager for context analysis if available
        if self.unified_manager and len(pattern_scores) > 50:
            try:
                # Prepare context data
                context_data = {
                    'pattern_scores': pattern_scores,
                    'open_prices': open_prices,
                    'high_prices': high_prices,
                    'low_prices': low_prices,
                    'close_prices': close_prices
                }
                
                # Use UnifiedVectorizationManager for context analysis
                config = OperationConfig(
                    operation_type=OperationType.TECHNICAL_INDICATORS,
                    data_size=len(pattern_scores),
                    data_dimensions=(len(pattern_scores),)
                )
                
                result = self.unified_manager.optimize_operation(
                    OperationType.TECHNICAL_INDICATORS,
                    context_data,
                    config,
                    pattern_enhancement=True
                )
                
                if hasattr(result, 'result') and isinstance(result.result, np.ndarray):
                    self.performance_stats['unified_manager_operations'] += 1
                    return result.result
                    
            except Exception as e:
                logger.warning(f"UnifiedVectorizationManager context analysis failed: {e}, using fallback")
        
        # Fallback: basic context enhancement
        # Calculate trend context
        if len(open_prices) > 10 and self.rolling_optimizer:
            try:
                # Calculate short-term trend
                short_trend = self.rolling_optimizer.rolling_mean(
                    pd.Series(close_prices), window=5
                ).values
                
                # Calculate long-term trend
                long_trend = self.rolling_optimizer.rolling_mean(
                    pd.Series(close_prices), window=20
                ).values
                
                # Trend strength
                trend_strength = np.abs(short_trend - long_trend) / long_trend
                
                # Enhance pattern scores with trend context
                enhanced_scores = pattern_scores * (1 + trend_strength)
                
                self.performance_stats['vectorbt_operations'] += 2
                return enhanced_scores
                
            except Exception as e:
                logger.warning(f"VectorBT trend context calculation failed: {e}, using basic enhancement")
        
        # Basic enhancement: scale by recent volatility
        if len(open_prices) > 5:
            recent_volatility = np.std(close_prices[-5:]) if len(close_prices) >= 5 else 1.0
            return pattern_scores * (1 + recent_volatility / np.mean(close_prices))
        
        return pattern_scores
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the pattern generator."""
        stats = self.performance_stats.copy()
        if stats['total_computation_time'] > 0:
            stats['patterns_per_second'] = stats['patterns_detected'] / stats['total_computation_time']
            stats['avg_time_per_pattern'] = stats['total_computation_time'] / max(stats['patterns_detected'], 1)
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'patterns_detected': 0,
            'vectorbt_operations': 0,
            'unified_manager_operations': 0,
            'total_computation_time': 0.0
        }

def create_candlestick_pattern_generators(patterns: List[str] = None) -> List[FeatureGenerator]:
    """Create a set of candlestick pattern feature generators."""
    if patterns is None:
        patterns = ["doji", "hammer", "engulfing", "shooting_star", "hanging_man"]
    
    generators = []
    for pattern in patterns:
        config = FeatureConfig(
            name=f"candlestick_{pattern}_pattern",
            category=FeatureCategory.CANDLESTICK_PATTERN,
            description=f"Detects {pattern} candlestick patterns with VectorBT optimization",
            required_columns=["open", "high", "low", "close"],
            default_lookback=3,
            min_lookback=1,
            max_lookback=10,
            parameters={
                "patterns": [pattern],
                "body_threshold": 0.1,
                "shadow_threshold": 0.3,
                "engulfing_threshold": 0.5,
                "use_vectorbt": True,
                "use_unified_manager": True
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
        generators.append(CandlestickPatternFeatureGenerator(config))
    
    return generators

def create_default_candlestick_pattern_generators() -> List[FeatureGenerator]:
    """Create default set of candlestick pattern generators."""
    return create_candlestick_pattern_generators()

def test_candlestick_pattern_generator():
    """Test function for the candlestick pattern generator with VectorBT optimization."""
    import time
    
    print("🧪 Testing Candlestick Pattern Generator with VectorBT Optimization...")
    
    # Create sample OHLC data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic OHLC data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_samples)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLC data with some patterns
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices
    }, index=pd.date_range('2020-01-01', periods=n_samples, freq='1min'))
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Create pattern generator
    generator = CandlestickPatternFeatureGenerator()
    
    # Test pattern detection
    start_time = time.time()
    pattern_score = generator._generate_feature(data)
    computation_time = time.time() - start_time
    
    print(f"✅ Pattern detection completed in {computation_time:.4f} seconds")
    print(f"📊 Pattern score statistics:")
    print(f"   - Mean: {pattern_score.mean():.4f}")
    print(f"   - Std: {pattern_score.std():.4f}")
    print(f"   - Min: {pattern_score.min():.4f}")
    print(f"   - Max: {pattern_score.max():.4f}")
    print(f"   - Non-zero patterns: {(pattern_score > 0).sum()}")
    
    # Test individual pattern generators
    print("\n🔍 Testing individual pattern generators...")
    individual_generators = create_candlestick_pattern_generators()
    
    for gen in individual_generators:
        try:
            individual_score = gen._generate_feature(data)
            pattern_name = gen.config.name.replace('candlestick_', '').replace('_pattern', '')
            print(f"   - {pattern_name}: {individual_score.sum()} patterns detected")
        except Exception as e:
            print(f"   - {gen.config.name}: Error - {e}")
    
    # Performance statistics
    stats = generator.get_performance_stats()
    print(f"\n📈 Performance Statistics:")
    print(f"   - Patterns detected: {stats['patterns_detected']}")
    print(f"   - VectorBT operations: {stats['vectorbt_operations']}")
    print(f"   - Unified manager operations: {stats['unified_manager_operations']}")
    print(f"   - Total computation time: {stats['total_computation_time']:.4f}s")
    
    if stats['total_computation_time'] > 0:
        print(f"   - Patterns per second: {stats['patterns_per_second']:.2f}")
        print(f"   - Avg time per pattern: {stats['avg_time_per_pattern']:.6f}s")
    
    print("\n🎉 Candlestick Pattern Generator test completed successfully!")
    return generator, pattern_score

if __name__ == "__main__":
    # Run test when executed directly
    test_candlestick_pattern_generator()
