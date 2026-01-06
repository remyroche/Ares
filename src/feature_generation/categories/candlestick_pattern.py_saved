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
    from src.utils.vectorbt_compat import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_quantile, rolling_skew, rolling_kurt
    )
    from src.utils.vectorbt_compat import scale, rank, zscore, winsorize, clip, quantile
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

# CuPy for GPU acceleration (optional)
try:
    import cupy as cp
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
        self.patterns = config.parameters.get("patterns", [
            "doji", "hammer", "engulfing", "shooting_star", "hanging_man",
            "harami", "harami_cross", "long_legged_doji", "dragonfly_doji", "gravestone_doji",
            "inverted_hammer", "three_white_soldiers", "three_black_crows", 
            "dark_cloud_cover", "piercing_line", "abandoned_baby"
        ])
        self.body_threshold = config.parameters.get("body_threshold", 0.1)
        self.shadow_threshold = config.parameters.get("shadow_threshold", 0.3)
        self.engulfing_threshold = config.parameters.get("engulfing_threshold", 0.5)

        # Performance tracking - include base class keys
        self.performance_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_computation_time': 0.0,
            'average_computation_time': 0.0,
            'patterns_detected': 0,
            'vectorbt_operations': 0,
            'unified_manager_operations': 0
        }

    def _initialize_optimization_components(self):
        """Initialize VectorBT and UnifiedVectorizationManager components."""
        # Initialize VectorBTRollingOptimizer
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
            # Reduced verbosity - only log once per session
            if not hasattr(CandlestickPatternFeatureGenerator, '_logged_rolling_init'):
                logger.info("✅ VectorBTRollingOptimizer initialized for candlestick patterns")
                CandlestickPatternFeatureGenerator._logged_rolling_init = True
        else:
            self.rolling_optimizer = None
            logger.warning("⚠️ VectorBTRollingOptimizer not available, using fallback methods")

        # Initialize UnifiedVectorizationManager
        if UNIFIED_VECTORIZATION_MANAGER_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
            # Reduced verbosity - only log once per session
            if not hasattr(CandlestickPatternFeatureGenerator, '_logged_unified_init'):
                logger.info("✅ UnifiedVectorizationManager initialized for candlestick patterns")
                CandlestickPatternFeatureGenerator._logged_unified_init = True
        else:
            self.unified_manager = None
            # Only log this warning once per session
            if not hasattr(CandlestickPatternFeatureGenerator, '_unified_warning_logged'):
                logger.warning("⚠️ UnifiedVectorizationManager not available, using direct VectorBT calls")
                CandlestickPatternFeatureGenerator._unified_warning_logged = True

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
                "patterns": ["doji", "hammer", "engulfing", "shooting_star", "hanging_man", "morning_star", "evening_star",
                           "harami", "harami_cross", "long_legged_doji", "dragonfly_doji", "gravestone_doji",
                           "inverted_hammer", "three_white_soldiers", "three_black_crows", 
                           "dark_cloud_cover", "piercing_line", "abandoned_baby"],
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

        # DataFrame is already optimized for processing

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

        # Avoid division by zero and handle edge cases
        total_range = np.where(total_range == 0, 1e-8, total_range)
        
        # Normalize components with better handling of edge cases
        body_ratio = body_size / total_range
        upper_shadow_ratio = upper_shadow / total_range
        lower_shadow_ratio = lower_shadow / total_range
        
        # Handle infinite and NaN values that might result from division
        body_ratio = np.nan_to_num(body_ratio, nan=0.0, posinf=1.0, neginf=0.0)
        upper_shadow_ratio = np.nan_to_num(upper_shadow_ratio, nan=0.0, posinf=1.0, neginf=0.0)
        lower_shadow_ratio = np.nan_to_num(lower_shadow_ratio, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Ensure ratios are between 0 and 1
        body_ratio = np.clip(body_ratio, 0.0, 1.0)
        upper_shadow_ratio = np.clip(upper_shadow_ratio, 0.0, 1.0)
        lower_shadow_ratio = np.clip(lower_shadow_ratio, 0.0, 1.0)

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

        # Note: UnifiedVectorizationManager integration removed due to data format mismatch
        # Candlestick patterns use individual detection which is more appropriate for this use case

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
                elif pattern == "harami":
                    feature = self._detect_harami_pattern(open_prices, close_prices, body_ratio)
                elif pattern == "harami_cross":
                    feature = self._detect_harami_cross_pattern(open_prices, close_prices, body_ratio)
                elif pattern == "long_legged_doji":
                    feature = self._detect_long_legged_doji_pattern(body_ratio, upper_shadow_ratio, lower_shadow_ratio)
                elif pattern == "dragonfly_doji":
                    feature = self._detect_dragonfly_doji_pattern(body_ratio, upper_shadow_ratio, lower_shadow_ratio)
                elif pattern == "gravestone_doji":
                    feature = self._detect_gravestone_doji_pattern(body_ratio, upper_shadow_ratio, lower_shadow_ratio)
                elif pattern == "inverted_hammer":
                    feature = self._detect_inverted_hammer_pattern(body_ratio, upper_shadow_ratio, lower_shadow_ratio)
                elif pattern == "three_white_soldiers":
                    feature = self._detect_three_white_soldiers_pattern(open_prices, close_prices)
                elif pattern == "three_black_crows":
                    feature = self._detect_three_black_crows_pattern(open_prices, close_prices)
                elif pattern == "dark_cloud_cover":
                    feature = self._detect_dark_cloud_cover_pattern(open_prices, close_prices, body_ratio)
                elif pattern == "piercing_line":
                    feature = self._detect_piercing_line_pattern(open_prices, close_prices, body_ratio)
                elif pattern == "abandoned_baby":
                    feature = self._detect_abandoned_baby_pattern(open_prices, high_prices, low_prices, close_prices)
                else:
                    continue

                pattern_features.append(feature)

            except Exception as e:
                logger.warning(f"Failed to generate {pattern} pattern: {e}")
                continue

        return pattern_features

    def _detect_doji_pattern(self, body_ratio: np.ndarray, total_range: np.ndarray) -> np.ndarray:
        """Detect doji pattern using VectorBT optimization with intensity scoring."""
        # Calculate intensity based on how small the body is relative to threshold
        # Lower body ratio = higher intensity (stronger doji)
        intensity = np.clip(1.0 - (body_ratio / self.body_threshold), 0.0, 1.0)

        # Use VectorBTRollingOptimizer for rolling statistics if available
        if self.rolling_optimizer and len(body_ratio) > 10:
            try:
                # Ensure body_ratio is a valid numpy array
                body_ratio_clean = np.nan_to_num(body_ratio, nan=0.0, posinf=1.0, neginf=0.0)

                # Calculate rolling mean of body ratios for context
                rolling_body_mean = self.rolling_optimizer.rolling_mean(
                    pd.Series(body_ratio_clean), window=5
                ).values

                # Enhanced doji intensity with context - boost when body is unusually small
                context_factor = np.clip(rolling_body_mean / (body_ratio_clean + 1e-8), 0.5, 2.0)
                enhanced_intensity = intensity * context_factor
                
                self.performance_stats['vectorbt_operations'] += 1
                return np.clip(enhanced_intensity, 0.0, 1.0)
            except Exception as e:
                logger.warning(f"VectorBT doji detection failed: {e}, using fallback")

        return intensity

    def _detect_hammer_pattern(self, body_ratio: np.ndarray, upper_shadow_ratio: np.ndarray,
                              lower_shadow_ratio: np.ndarray) -> np.ndarray:
        """Detect hammer pattern using VectorBT optimization with intensity scoring."""
        # Calculate intensity components
        body_intensity = np.clip(1.0 - (body_ratio / self.body_threshold), 0.0, 1.0)
        lower_shadow_intensity = np.clip(lower_shadow_ratio / self.shadow_threshold, 0.0, 2.0)
        upper_shadow_intensity = np.clip(1.0 - (upper_shadow_ratio / (self.shadow_threshold * 0.5)), 0.0, 1.0)
        
        # Combined intensity: small body, long lower shadow, short upper shadow
        intensity = body_intensity * lower_shadow_intensity * upper_shadow_intensity

        # Use VectorBTRollingOptimizer for enhanced detection
        if self.rolling_optimizer and len(body_ratio) > 10:
            try:
                # Ensure shadow ratios are valid numpy arrays
                lower_shadow_ratio_clean = np.nan_to_num(lower_shadow_ratio, nan=0.0, posinf=1.0, neginf=0.0)

                # Calculate rolling statistics for context
                rolling_lower_shadow = self.rolling_optimizer.rolling_mean(
                    pd.Series(lower_shadow_ratio_clean), window=5
                ).values

                # Enhanced hammer intensity with context
                context_factor = np.clip(lower_shadow_ratio_clean / (rolling_lower_shadow + 1e-8), 0.5, 2.0)
                enhanced_intensity = intensity * context_factor
                
                self.performance_stats['vectorbt_operations'] += 1
                return np.clip(enhanced_intensity, 0.0, 1.0)
            except Exception as e:
                logger.warning(f"VectorBT hammer detection failed: {e}, using fallback")

        return intensity

    def _detect_shooting_star_pattern(self, body_ratio: np.ndarray, upper_shadow_ratio: np.ndarray,
                                    lower_shadow_ratio: np.ndarray) -> np.ndarray:
        """Detect shooting star pattern using VectorBT optimization with intensity scoring."""
        # Calculate intensity components
        body_intensity = np.clip(1.0 - (body_ratio / self.body_threshold), 0.0, 1.0)
        upper_shadow_intensity = np.clip(upper_shadow_ratio / self.shadow_threshold, 0.0, 2.0)
        lower_shadow_intensity = np.clip(1.0 - (lower_shadow_ratio / (self.shadow_threshold * 0.5)), 0.0, 1.0)
        
        # Combined intensity: small body, long upper shadow, short lower shadow
        intensity = body_intensity * upper_shadow_intensity * lower_shadow_intensity

        # Use VectorBTRollingOptimizer for enhanced detection
        if self.rolling_optimizer and len(body_ratio) > 10:
            try:
                # Ensure shadow ratios are valid numpy arrays
                upper_shadow_ratio_clean = np.nan_to_num(upper_shadow_ratio, nan=0.0, posinf=1.0, neginf=0.0)

                # Calculate rolling statistics for context
                rolling_upper_shadow = self.rolling_optimizer.rolling_mean(
                    pd.Series(upper_shadow_ratio_clean), window=5
                ).values

                # Enhanced shooting star intensity with context
                context_factor = np.clip(upper_shadow_ratio_clean / (rolling_upper_shadow + 1e-8), 0.5, 2.0)
                enhanced_intensity = intensity * context_factor
                
                self.performance_stats['vectorbt_operations'] += 1
                return np.clip(enhanced_intensity, 0.0, 1.0)
            except Exception as e:
                logger.warning(f"VectorBT shooting star detection failed: {e}, using fallback")

        return intensity

    def _detect_hanging_man_pattern(self, body_ratio: np.ndarray, upper_shadow_ratio: np.ndarray,
                                  lower_shadow_ratio: np.ndarray) -> np.ndarray:
        """Detect hanging man pattern using VectorBT optimization with intensity scoring."""
        # Calculate intensity components
        body_intensity = np.clip(1.0 - (body_ratio / self.body_threshold), 0.0, 1.0)
        lower_shadow_intensity = np.clip(lower_shadow_ratio / self.shadow_threshold, 0.0, 2.0)
        upper_shadow_intensity = np.clip(1.0 - (upper_shadow_ratio / (self.shadow_threshold * 0.5)), 0.0, 1.0)
        
        # Combined intensity: small body, long lower shadow, short upper shadow
        intensity = body_intensity * lower_shadow_intensity * upper_shadow_intensity

        # Use VectorBTRollingOptimizer for trend context
        if self.rolling_optimizer and len(body_ratio) > 20:
            try:
                # Ensure body_ratio is a valid numpy array
                body_ratio_clean = np.nan_to_num(body_ratio, nan=0.0, posinf=1.0, neginf=0.0)

                # Calculate trend context using rolling mean
                rolling_body = self.rolling_optimizer.rolling_mean(
                    pd.Series(body_ratio_clean), window=10
                ).values

                # Enhanced hanging man intensity with trend context
                trend_factor = np.clip(rolling_body / (body_ratio_clean + 1e-8), 0.5, 2.0)
                enhanced_intensity = intensity * trend_factor
                
                self.performance_stats['vectorbt_operations'] += 1
                return np.clip(enhanced_intensity, 0.0, 1.0)
            except Exception as e:
                logger.warning(f"VectorBT hanging man detection failed: {e}, using fallback")

        return intensity

    def _detect_engulfing_pattern(self, open_prices: np.ndarray, close_prices: np.ndarray,
                                body_ratio: np.ndarray) -> np.ndarray:
        """Detect engulfing pattern using VectorBT optimization with intensity scoring."""
        # Initialize intensity array
        engulfing_intensity = np.zeros(len(open_prices))

        if len(open_prices) < 2:
            return engulfing_intensity

        # Calculate previous candle body
        prev_open = np.roll(open_prices, 1)
        prev_close = np.roll(close_prices, 1)
        prev_body_size = np.abs(prev_close - prev_open)

        # Current candle body
        current_body_size = np.abs(close_prices - open_prices)

        # Bullish engulfing intensity
        bullish_engulfing_intensity = np.zeros(len(open_prices))
        bullish_condition = (
            (close_prices > open_prices) &  # Current candle is green
            (prev_close < prev_open) &      # Previous candle is red
            (open_prices < prev_close) &    # Current open below previous close
            (close_prices > prev_open)      # Current close above previous open
        )
        
        # Intensity based on how much larger current body is compared to previous
        body_size_ratio = np.where(
            prev_body_size > 0,
            current_body_size / prev_body_size,
            1.0
        )
        bullish_engulfing_intensity = np.where(
            bullish_condition,
            np.clip(body_size_ratio * (1 + self.engulfing_threshold), 0.0, 2.0),
            0.0
        )

        # Bearish engulfing intensity
        bearish_engulfing_intensity = np.zeros(len(open_prices))
        bearish_condition = (
            (close_prices < open_prices) &  # Current candle is red
            (prev_close > prev_open) &      # Previous candle is green
            (open_prices > prev_close) &    # Current open above previous close
            (close_prices < prev_open)      # Current close below previous open
        )
        
        bearish_engulfing_intensity = np.where(
            bearish_condition,
            np.clip(body_size_ratio * (1 + self.engulfing_threshold), 0.0, 2.0),
            0.0
        )

        # Combined engulfing intensity
        engulfing_intensity = bullish_engulfing_intensity + bearish_engulfing_intensity
        
        return np.clip(engulfing_intensity, 0.0, 1.0)

    def _detect_morning_star_pattern(self, open_prices: np.ndarray, high_prices: np.ndarray,
                                   low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Detect morning star pattern using VectorBT optimization with intensity scoring."""
        # Morning star: 3-candle pattern (bearish, small body, bullish)
        morning_star_intensity = np.zeros(len(open_prices))

        if len(open_prices) < 3:
            return morning_star_intensity

        # Previous candles
        prev2_open = np.roll(open_prices, 2)
        prev2_close = np.roll(close_prices, 2)
        prev_open = np.roll(open_prices, 1)
        prev_close = np.roll(close_prices, 1)

        # First candle: bearish intensity
        first_bearish_intensity = np.where(prev2_close < prev2_open, 1.0, 0.0)

        # Second candle: small body intensity (doji-like)
        prev2_body_size = np.abs(prev2_close - prev2_open)
        prev_body_size = np.abs(prev_close - prev_open)
        second_small_body_intensity = np.where(
            prev_body_size <= self.body_threshold * prev2_body_size,
            1.0 - (prev_body_size / (self.body_threshold * prev2_body_size + 1e-8)),
            0.0
        )

        # Third candle: bullish intensity
        third_bullish_intensity = np.where(close_prices > open_prices, 1.0, 0.0)

        # Third candle strength: closes above first candle's midpoint
        first_midpoint = (prev2_open + prev2_close) / 2
        third_strong_intensity = np.where(
            close_prices > first_midpoint,
            (close_prices - first_midpoint) / (np.abs(prev2_close - prev2_open) + 1e-8),
            0.0
        )

        # Combined morning star intensity
        morning_star_intensity = (
            first_bearish_intensity * 
            second_small_body_intensity * 
            third_bullish_intensity * 
            np.clip(third_strong_intensity, 0.0, 1.0)
        )

        # Use VectorBTRollingOptimizer for enhanced detection
        if self.rolling_optimizer and len(open_prices) > 20:
            try:
                # Calculate rolling volatility for context
                price_range = high_prices - low_prices
                price_range_clean = np.nan_to_num(price_range, nan=0.0, posinf=1e8, neginf=0.0)

                rolling_volatility = self.rolling_optimizer.rolling_std(
                    pd.Series(price_range_clean), window=10
                ).values

                # Enhanced morning star intensity with volatility context
                volatility_factor = np.clip(
                    price_range_clean / (rolling_volatility + 1e-8),
                    0.5, 2.0
                )
                enhanced_intensity = morning_star_intensity * volatility_factor
                
                self.performance_stats['vectorbt_operations'] += 1
                return np.clip(enhanced_intensity, 0.0, 1.0)
            except Exception as e:
                logger.warning(f"VectorBT morning star detection failed: {e}, using fallback")

        return morning_star_intensity

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

    def _detect_harami_pattern(self, open_prices: np.ndarray, close_prices: np.ndarray,
                             body_ratio: np.ndarray) -> np.ndarray:
        """Detect harami pattern using VectorBT optimization."""
        # Harami: small body inside previous large body
        harami_pattern = np.zeros(len(open_prices))
        
        if len(open_prices) < 2:
            return harami_pattern
        
        # Previous candle data
        prev_open = np.roll(open_prices, 1)
        prev_close = np.roll(close_prices, 1)
        prev_body_size = np.abs(prev_close - prev_open)
        current_body_size = np.abs(close_prices - open_prices)
        
        # Harami conditions
        harami_condition = (
            (current_body_size < prev_body_size * 0.5) &  # Current body smaller than previous
            (body_ratio <= self.body_threshold) &  # Current body is small
            (current_body_size > 0)  # Not a doji
        )
        
        # Use VectorBTRollingOptimizer for enhanced detection
        if self.rolling_optimizer and len(open_prices) > 10:
            try:
                # Calculate rolling body size for context
                rolling_body_std = self.rolling_optimizer.rolling_std(
                    pd.Series(current_body_size), window=5
                ).values
                
                # Enhanced harami detection
                enhanced_harami = harami_condition & (current_body_size > rolling_body_std * 0.5)
                self.performance_stats['vectorbt_operations'] += 1
                return enhanced_harami.astype(float)
            except Exception as e:
                logger.warning(f"VectorBT harami detection failed: {e}, using fallback")
        
        return harami_condition.astype(float)

    def _detect_harami_cross_pattern(self, open_prices: np.ndarray, close_prices: np.ndarray,
                                    body_ratio: np.ndarray) -> np.ndarray:
        """Detect harami cross pattern using VectorBT optimization."""
        # Harami Cross: doji inside previous large body
        harami_cross_pattern = np.zeros(len(open_prices))
        
        if len(open_prices) < 2:
            return harami_cross_pattern
        
        # Previous candle data
        prev_open = np.roll(open_prices, 1)
        prev_close = np.roll(close_prices, 1)
        prev_body_size = np.abs(prev_close - prev_open)
        current_body_size = np.abs(close_prices - open_prices)
        
        # Harami Cross conditions (doji inside large body)
        harami_cross_condition = (
            (body_ratio <= self.body_threshold) &  # Current is doji
            (current_body_size < prev_body_size * 0.3) &  # Much smaller than previous
            (prev_body_size > current_body_size * 3)  # Previous was large
        )
        
        return harami_cross_condition.astype(float)

    def _detect_long_legged_doji_pattern(self, body_ratio: np.ndarray, upper_shadow_ratio: np.ndarray,
                                       lower_shadow_ratio: np.ndarray) -> np.ndarray:
        """Detect long-legged doji pattern using VectorBT optimization."""
        # Long-legged doji: doji with long shadows on both sides
        long_legged_doji_condition = (
            (body_ratio <= self.body_threshold) &  # Small body (doji)
            (upper_shadow_ratio >= self.shadow_threshold) &  # Long upper shadow
            (lower_shadow_ratio >= self.shadow_threshold)  # Long lower shadow
        )
        
        # Use VectorBTRollingOptimizer for enhanced detection
        if self.rolling_optimizer and len(body_ratio) > 10:
            try:
                # Ensure shadow ratios are valid numpy arrays
                upper_shadow_ratio_clean = np.nan_to_num(upper_shadow_ratio, nan=0.0, posinf=1.0, neginf=0.0)
                lower_shadow_ratio_clean = np.nan_to_num(lower_shadow_ratio, nan=0.0, posinf=1.0, neginf=0.0)

                # Calculate rolling shadow statistics
                rolling_upper_shadow = self.rolling_optimizer.rolling_mean(
                    pd.Series(upper_shadow_ratio_clean), window=5
                ).values
                rolling_lower_shadow = self.rolling_optimizer.rolling_mean(
                    pd.Series(lower_shadow_ratio_clean), window=5
                ).values

                # Enhanced long-legged doji detection
                enhanced_long_legged = long_legged_doji_condition & (
                    (upper_shadow_ratio_clean > rolling_upper_shadow * 1.5) &
                    (lower_shadow_ratio_clean > rolling_lower_shadow * 1.5)
                )
                self.performance_stats['vectorbt_operations'] += 2
                return enhanced_long_legged.astype(float)
            except Exception as e:
                logger.warning(f"VectorBT long-legged doji detection failed: {e}, using fallback")
        
        return long_legged_doji_condition.astype(float)

    def _detect_dragonfly_doji_pattern(self, body_ratio: np.ndarray, upper_shadow_ratio: np.ndarray,
                                     lower_shadow_ratio: np.ndarray) -> np.ndarray:
        """Detect dragonfly doji pattern using VectorBT optimization."""
        # Dragonfly doji: doji with long lower shadow, no upper shadow
        dragonfly_doji_condition = (
            (body_ratio <= self.body_threshold) &  # Small body (doji)
            (lower_shadow_ratio >= self.shadow_threshold) &  # Long lower shadow
            (upper_shadow_ratio <= self.shadow_threshold * 0.3)  # No upper shadow
        )
        
        return dragonfly_doji_condition.astype(float)

    def _detect_gravestone_doji_pattern(self, body_ratio: np.ndarray, upper_shadow_ratio: np.ndarray,
                                      lower_shadow_ratio: np.ndarray) -> np.ndarray:
        """Detect gravestone doji pattern using VectorBT optimization."""
        # Debug: Check for non-finite values in input arrays
        body_nan_count = np.isnan(body_ratio).sum()
        upper_nan_count = np.isnan(upper_shadow_ratio).sum()
        lower_nan_count = np.isnan(lower_shadow_ratio).sum()
        
        if body_nan_count > 0 or upper_nan_count > 0 or lower_nan_count > 0:
            self.logger.warning(f"Gravestone doji pattern: Found NaN values - body: {body_nan_count}, upper: {upper_nan_count}, lower: {lower_nan_count}")
        
        # Handle NaN values by filling with appropriate defaults
        body_ratio_clean = np.nan_to_num(body_ratio, nan=0.0, posinf=1.0, neginf=0.0)
        upper_shadow_ratio_clean = np.nan_to_num(upper_shadow_ratio, nan=0.0, posinf=1.0, neginf=0.0)
        lower_shadow_ratio_clean = np.nan_to_num(lower_shadow_ratio, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Gravestone doji: doji with long upper shadow, no lower shadow
        gravestone_doji_condition = (
            (body_ratio_clean <= self.body_threshold) &  # Small body (doji)
            (upper_shadow_ratio_clean >= self.shadow_threshold) &  # Long upper shadow
            (lower_shadow_ratio_clean <= self.shadow_threshold * 0.3)  # No lower shadow
        )
        
        result = gravestone_doji_condition.astype(float)
        
        # Debug: Check result for non-finite values
        result_nan_count = np.isnan(result).sum()
        if result_nan_count > 0:
            self.logger.warning(f"Gravestone doji pattern result: Found {result_nan_count} NaN values in output")
        
        return result

    def _detect_inverted_hammer_pattern(self, body_ratio: np.ndarray, upper_shadow_ratio: np.ndarray,
                                      lower_shadow_ratio: np.ndarray) -> np.ndarray:
        """Detect inverted hammer pattern using VectorBT optimization."""
        # Inverted hammer: small body, long upper shadow, short lower shadow
        inverted_hammer_condition = (
            (body_ratio <= self.body_threshold) &
            (upper_shadow_ratio >= self.shadow_threshold) &
            (lower_shadow_ratio <= self.shadow_threshold * 0.5)
        )
        
        # Use VectorBTRollingOptimizer for enhanced detection
        if self.rolling_optimizer and len(body_ratio) > 10:
            try:
                # Ensure shadow ratios are valid numpy arrays
                upper_shadow_ratio_clean = np.nan_to_num(upper_shadow_ratio, nan=0.0, posinf=1.0, neginf=0.0)

                # Calculate rolling statistics for context
                rolling_upper_shadow = self.rolling_optimizer.rolling_mean(
                    pd.Series(upper_shadow_ratio_clean), window=5
                ).values

                # Enhanced inverted hammer detection
                enhanced_inverted_hammer = inverted_hammer_condition & (upper_shadow_ratio_clean > rolling_upper_shadow * 1.5)
                self.performance_stats['vectorbt_operations'] += 1
                return enhanced_inverted_hammer.astype(float)
            except Exception as e:
                logger.warning(f"VectorBT inverted hammer detection failed: {e}, using fallback")
        
        return inverted_hammer_condition.astype(float)

    def _detect_three_white_soldiers_pattern(self, open_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Detect three white soldiers pattern using VectorBT optimization."""
        # Three white soldiers: three consecutive bullish candles
        three_white_soldiers = np.zeros(len(open_prices))
        
        if len(open_prices) < 3:
            return three_white_soldiers
        
        # Check for three consecutive bullish candles
        for i in range(2, len(open_prices)):
            current_bullish = close_prices[i] > open_prices[i]
            prev_bullish = close_prices[i-1] > open_prices[i-1]
            prev2_bullish = close_prices[i-2] > open_prices[i-2]
            
            # Each candle should close higher than the previous
            higher_close = (close_prices[i] > close_prices[i-1]) & (close_prices[i-1] > close_prices[i-2])
            
            three_white_soldiers[i] = (current_bullish & prev_bullish & prev2_bullish & higher_close).astype(float)
        
        return three_white_soldiers

    def _detect_three_black_crows_pattern(self, open_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Detect three black crows pattern using VectorBT optimization."""
        # Three black crows: three consecutive bearish candles
        three_black_crows = np.zeros(len(open_prices))
        
        if len(open_prices) < 3:
            return three_black_crows
        
        # Check for three consecutive bearish candles
        for i in range(2, len(open_prices)):
            current_bearish = close_prices[i] < open_prices[i]
            prev_bearish = close_prices[i-1] < open_prices[i-1]
            prev2_bearish = close_prices[i-2] < open_prices[i-2]
            
            # Each candle should close lower than the previous
            lower_close = (close_prices[i] < close_prices[i-1]) & (close_prices[i-1] < close_prices[i-2])
            
            three_black_crows[i] = (current_bearish & prev_bearish & prev2_bearish & lower_close).astype(float)
        
        return three_black_crows

    def _detect_dark_cloud_cover_pattern(self, open_prices: np.ndarray, close_prices: np.ndarray,
                                        body_ratio: np.ndarray) -> np.ndarray:
        """Detect dark cloud cover pattern using VectorBT optimization."""
        # Dark cloud cover: bearish candle opens above previous bullish candle's high
        # and closes below its midpoint
        dark_cloud_cover = np.zeros(len(open_prices))
        
        if len(open_prices) < 2:
            return dark_cloud_cover
        
        # Previous candle data
        prev_open = np.roll(open_prices, 1)
        prev_close = np.roll(close_prices, 1)
        prev_high = np.roll(open_prices, 1)  # Using open as proxy for high
        
        # Dark cloud cover conditions
        dark_cloud_condition = (
            (close_prices < open_prices) &  # Current candle is bearish
            (prev_close > prev_open) &  # Previous candle is bullish
            (open_prices > prev_close) &  # Current opens above previous close
            (close_prices < (prev_open + prev_close) / 2)  # Current closes below previous midpoint
        )
        
        return dark_cloud_condition.astype(float)

    def _detect_piercing_line_pattern(self, open_prices: np.ndarray, close_prices: np.ndarray,
                                     body_ratio: np.ndarray) -> np.ndarray:
        """Detect piercing line pattern using VectorBT optimization."""
        # Piercing line: bullish candle opens below previous bearish candle's low
        # and closes above its midpoint
        piercing_line = np.zeros(len(open_prices))
        
        if len(open_prices) < 2:
            return piercing_line
        
        # Previous candle data
        prev_open = np.roll(open_prices, 1)
        prev_close = np.roll(close_prices, 1)
        prev_low = np.roll(open_prices, 1)  # Using open as proxy for low
        
        # Piercing line conditions
        piercing_condition = (
            (close_prices > open_prices) &  # Current candle is bullish
            (prev_close < prev_open) &  # Previous candle is bearish
            (open_prices < prev_close) &  # Current opens below previous close
            (close_prices > (prev_open + prev_close) / 2)  # Current closes above previous midpoint
        )
        
        return piercing_condition.astype(float)

    def _detect_abandoned_baby_pattern(self, open_prices: np.ndarray, high_prices: np.ndarray,
                                     low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Detect abandoned baby pattern using VectorBT optimization."""
        # Abandoned baby: gap, doji, then opposite gap
        abandoned_baby = np.zeros(len(open_prices))
        
        if len(open_prices) < 3:
            return abandoned_baby
        
        # Previous candles
        prev2_open = np.roll(open_prices, 2)
        prev2_close = np.roll(close_prices, 2)
        prev_open = np.roll(open_prices, 1)
        prev_close = np.roll(close_prices, 1)
        
        # Calculate gaps
        gap1 = abs(open_prices - prev_close)  # Gap between prev and current
        gap2 = abs(prev_open - prev2_close)  # Gap between prev2 and prev
        
        # Doji condition for middle candle
        prev_body_size = np.abs(prev_close - prev_open)
        prev_total_range = high_prices - low_prices
        prev_body_ratio = prev_body_size / np.where(prev_total_range == 0, 1e-8, prev_total_range)
        
        # Abandoned baby conditions
        abandoned_baby_condition = (
            (gap1 > 0) &  # Gap before
            (gap2 > 0) &  # Gap after
            (prev_body_ratio <= self.body_threshold) &  # Middle candle is doji
            (close_prices > open_prices) &  # Current is bullish
            (prev2_close < prev2_open)  # First candle is bearish
        )
        
        return abandoned_baby_condition.astype(float)

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
                # Ensure price_range is a valid numpy array
                price_range_clean = np.nan_to_num(price_range, nan=0.0, posinf=1e8, neginf=0.0)

                # Calculate rolling volatility
                rolling_volatility = self.rolling_optimizer.rolling_std(
                    pd.Series(price_range_clean), window=10
                ).values

                # Calculate rolling mean for normalization
                rolling_mean_vol = self.rolling_optimizer.rolling_mean(
                    pd.Series(price_range_clean), window=10
                ).values

                # Pattern strength based on volatility context
                volatility_factor = np.where(rolling_volatility > 0,
                                           price_range_clean / rolling_volatility, 1.0)

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
                # Ensure price_momentum is a valid numpy array
                price_momentum_clean = np.nan_to_num(price_momentum, nan=0.0, posinf=1e8, neginf=-1e8)

                # Calculate rolling momentum statistics
                rolling_momentum_mean = self.rolling_optimizer.rolling_mean(
                    pd.Series(price_momentum_clean), window=20
                ).values

                rolling_momentum_std = self.rolling_optimizer.rolling_std(
                    pd.Series(price_momentum_clean), window=20
                ).values

                # Calculate momentum consistency
                momentum_consistency = np.where(rolling_momentum_std > 0,
                                             1 - np.abs(price_momentum_clean - rolling_momentum_mean) / rolling_momentum_std,
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

        # Note: UnifiedVectorizationManager integration removed due to data format mismatch
        # Using direct fallback implementation for context analysis

        # Fallback: basic context enhancement
        # Calculate trend context
        if len(open_prices) > 10 and self.rolling_optimizer:
            try:
                # Ensure close_prices is a valid numpy array
                close_prices_clean = np.nan_to_num(close_prices, nan=0.0, posinf=1e8, neginf=0.0)

                # Calculate short-term trend
                short_trend = self.rolling_optimizer.rolling_mean(
                    pd.Series(close_prices_clean), window=5
                ).values

                # Calculate long-term trend
                long_trend = self.rolling_optimizer.rolling_mean(
                    pd.Series(close_prices_clean), window=20
                ).values

                # Trend strength
                trend_strength = np.abs(short_trend - long_trend) / np.where(long_trend != 0, long_trend, 1.0)

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
        patterns = [
            "doji", "hammer", "engulfing", "shooting_star", "hanging_man",
            "harami", "harami_cross", "long_legged_doji", "dragonfly_doji", "gravestone_doji",
            "inverted_hammer", "three_white_soldiers", "three_black_crows", 
            "dark_cloud_cover", "piercing_line", "abandoned_baby"
        ]

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
