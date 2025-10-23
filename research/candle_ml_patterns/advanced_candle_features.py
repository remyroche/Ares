"""
Advanced Candle Feature Engineering

This module provides comprehensive feature engineering for ML-based trading indicators,
focusing on series of candles, cross-timeframe patterns, and multi-dimensional interactions.

Key Features:
- Series of candles (consecutive patterns, sequences)
- Cross-timeframe candle analysis
- Multi-dimensional interactions (volume, momentum, volatility)
- Pattern strength and size assessment
- Advanced pattern categorization
- Temporal pattern analysis
"""

import numpy as np
import pandas as pd
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from scipy.signal import find_peaks
import talib

# Core imports
from ..candlestick_pattern import CandlestickPatternFeatureGenerator
from ...core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_corr, rolling_apply
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False

logger = logging.getLogger(__name__)


class CandleSeriesType(Enum):
    """Types of candle series patterns."""
    CONSECUTIVE_BULLISH = "consecutive_bullish"
    CONSECUTIVE_BEARISH = "consecutive_bearish"
    ALTERNATING_PATTERN = "alternating_pattern"
    DOJI_SERIES = "doji_series"
    HAMMER_SERIES = "hammer_series"
    ENGULFING_SERIES = "engulfing_series"
    MOMENTUM_SERIES = "momentum_series"
    REVERSAL_SERIES = "reversal_series"


class CrossTimeframeType(Enum):
    """Types of cross-timeframe analysis."""
    MULTI_TIMEFRAME_TREND = "multi_timeframe_trend"
    TIMEFRAME_CONFLUENCE = "timeframe_confluence"
    TIMEFRAME_DIVERGENCE = "timeframe_divergence"
    TIMEFRAME_MOMENTUM = "timeframe_momentum"
    TIMEFRAME_VOLATILITY = "timeframe_volatility"


class InteractionDimension(Enum):
    """Dimensions for multi-dimensional interactions."""
    VOLUME = "volume"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    TREND = "trend"
    LIQUIDITY = "liquidity"
    MARKET_STRUCTURE = "market_structure"


@dataclass
class AdvancedFeatureConfig:
    """Configuration for advanced candle feature engineering."""
    # Series features
    enable_series_features: bool = True
    max_series_length: int = 10
    min_series_length: int = 2
    series_patterns: List[CandleSeriesType] = None
    
    # Cross-timeframe features
    enable_cross_timeframe: bool = True
    timeframes: List[str] = None  # ['1m', '5m', '15m', '1h', '4h', '1d']
    cross_timeframe_patterns: List[CrossTimeframeType] = None
    
    # Multi-dimensional interactions
    enable_multi_dimensional: bool = True
    interaction_dimensions: List[InteractionDimension] = None
    interaction_lookback: int = 20
    
    # Pattern strength features
    enable_pattern_strength: bool = True
    strength_metrics: List[str] = None
    quality_thresholds: Dict[str, float] = None
    
    # Advanced analysis
    enable_temporal_analysis: bool = True
    enable_pattern_categorization: bool = True
    enable_market_structure: bool = True
    
    def __post_init__(self):
        if self.series_patterns is None:
            self.series_patterns = list(CandleSeriesType)
        
        if self.timeframes is None:
            self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
        if self.cross_timeframe_patterns is None:
            self.cross_timeframe_patterns = list(CrossTimeframeType)
        
        if self.interaction_dimensions is None:
            self.interaction_dimensions = list(InteractionDimension)
        
        if self.strength_metrics is None:
            self.strength_metrics = ['body_size', 'shadow_ratio', 'range_ratio', 'volume_ratio', 'momentum_ratio']
        
        if self.quality_thresholds is None:
            self.quality_thresholds = {
                'high_quality': 0.8,
                'medium_quality': 0.6,
                'low_quality': 0.4
            }


class AdvancedCandleFeatureGenerator(VectorizedFeatureGenerator):
    """
    Advanced candle feature generator with comprehensive pattern analysis.
    
    This generator creates features for:
    - Series of candles (consecutive patterns, sequences)
    - Cross-timeframe candle analysis
    - Multi-dimensional interactions
    - Pattern strength and quality assessment
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None, 
                 advanced_config: Optional[AdvancedFeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.advanced_config = advanced_config or AdvancedFeatureConfig()
        self.candle_pattern_generator = CandlestickPatternFeatureGenerator()
        
        # Feature storage
        self.series_features = {}
        self.cross_timeframe_features = {}
        self.multi_dimensional_features = {}
        self.pattern_strength_features = {}
        
        logger.info("🔧 Advanced Candle Feature Generator initialized")
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="advanced_candle_features",
            category=FeatureCategory.CANDLESTICK_PATTERN,
            description="Advanced candle features with series, cross-timeframe, and multi-dimensional analysis",
            required_columns=["open", "high", "low", "close", "volume"],
            default_lookback=20,
            min_lookback=5,
            max_lookback=100,
            parameters={
                "enable_series_features": True,
                "enable_cross_timeframe": True,
                "enable_multi_dimensional": True,
                "enable_pattern_strength": True
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate comprehensive advanced candle features."""
        start_time = time.time()
        
        # Validate required columns
        required_cols = ["open", "high", "low", "close"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        # Generate different types of features
        all_features = []
        
        # 1. Series of candles features
        if self.advanced_config.enable_series_features:
            series_features = self._generate_series_features(data)
            all_features.extend(series_features)
        
        # 2. Cross-timeframe features
        if self.advanced_config.enable_cross_timeframe:
            cross_timeframe_features = self._generate_cross_timeframe_features(data)
            all_features.extend(cross_timeframe_features)
        
        # 3. Multi-dimensional interaction features
        if self.advanced_config.enable_multi_dimensional:
            multi_dimensional_features = self._generate_multi_dimensional_features(data)
            all_features.extend(multi_dimensional_features)
        
        # 4. Pattern strength features
        if self.advanced_config.enable_pattern_strength:
            pattern_strength_features = self._generate_pattern_strength_features(data)
            all_features.extend(pattern_strength_features)
        
        # 5. Advanced analysis features
        if self.advanced_config.enable_temporal_analysis:
            temporal_features = self._generate_temporal_analysis_features(data)
            all_features.extend(temporal_features)
        
        if self.advanced_config.enable_pattern_categorization:
            categorization_features = self._generate_pattern_categorization_features(data)
            all_features.extend(categorization_features)
        
        if self.advanced_config.enable_market_structure:
            market_structure_features = self._generate_market_structure_features(data)
            all_features.extend(market_structure_features)
        
        # Combine all features
        if all_features:
            combined_features = np.sum(all_features, axis=0)
        else:
            combined_features = np.zeros(len(data))
        
        # Normalize features
        if np.std(combined_features) > 0:
            combined_features = (combined_features - np.mean(combined_features)) / np.std(combined_features)
        
        logger.info(f"✅ Generated {len(all_features)} advanced candle features in {time.time() - start_time:.4f}s")
        
        return pd.Series(combined_features, index=data.index, name='advanced_candle_features')
    
    def _generate_series_features(self, data: pd.DataFrame) -> List[np.ndarray]:
        """Generate features related to series of candles."""
        series_features = []
        
        # Basic OHLC data
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        volume = data['volume'].values if 'volume' in data.columns else np.ones(len(data))
        
        # 1. Consecutive bullish/bearish series
        series_features.append(self._detect_consecutive_bullish_series(close_prices))
        series_features.append(self._detect_consecutive_bearish_series(close_prices))
        
        # 2. Alternating pattern series
        series_features.append(self._detect_alternating_pattern_series(close_prices))
        
        # 3. Doji series
        series_features.append(self._detect_doji_series(open_prices, high_prices, low_prices, close_prices))
        
        # 4. Hammer series
        series_features.append(self._detect_hammer_series(open_prices, high_prices, low_prices, close_prices))
        
        # 5. Engulfing series
        series_features.append(self._detect_engulfing_series(open_prices, high_prices, low_prices, close_prices))
        
        # 6. Momentum series
        series_features.append(self._detect_momentum_series(close_prices, volume))
        
        # 7. Reversal series
        series_features.append(self._detect_reversal_series(open_prices, high_prices, low_prices, close_prices))
        
        # 8. Series strength and consistency
        series_features.append(self._calculate_series_strength(close_prices))
        series_features.append(self._calculate_series_consistency(close_prices))
        
        return series_features
    
    def _detect_consecutive_bullish_series(self, close_prices: np.ndarray) -> np.ndarray:
        """Detect consecutive bullish candle series."""
        bullish_series = np.zeros(len(close_prices))
        
        for i in range(1, len(close_prices)):
            if close_prices[i] > close_prices[i-1]:
                # Count consecutive bullish candles
                count = 1
                j = i - 1
                while j >= 0 and close_prices[j] > close_prices[j-1] if j > 0 else False:
                    count += 1
                    j -= 1
                
                bullish_series[i] = min(count, self.advanced_config.max_series_length)
        
        return bullish_series
    
    def _detect_consecutive_bearish_series(self, close_prices: np.ndarray) -> np.ndarray:
        """Detect consecutive bearish candle series."""
        bearish_series = np.zeros(len(close_prices))
        
        for i in range(1, len(close_prices)):
            if close_prices[i] < close_prices[i-1]:
                # Count consecutive bearish candles
                count = 1
                j = i - 1
                while j >= 0 and close_prices[j] < close_prices[j-1] if j > 0 else False:
                    count += 1
                    j -= 1
                
                bearish_series[i] = min(count, self.advanced_config.max_series_length)
        
        return bearish_series
    
    def _detect_alternating_pattern_series(self, close_prices: np.ndarray) -> np.ndarray:
        """Detect alternating bullish/bearish pattern series."""
        alternating_series = np.zeros(len(close_prices))
        
        for i in range(2, len(close_prices)):
            # Check for alternating pattern
            pattern_length = 0
            j = i
            while j >= 2:
                if (close_prices[j] > close_prices[j-1] and close_prices[j-1] < close_prices[j-2]) or \
                   (close_prices[j] < close_prices[j-1] and close_prices[j-1] > close_prices[j-2]):
                    pattern_length += 1
                    j -= 2
                else:
                    break
            
            alternating_series[i] = min(pattern_length, self.advanced_config.max_series_length)
        
        return alternating_series
    
    def _detect_doji_series(self, open_prices: np.ndarray, high_prices: np.ndarray, 
                           low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Detect series of doji patterns."""
        doji_series = np.zeros(len(open_prices))
        
        for i in range(len(open_prices)):
            # Calculate body size
            body_size = abs(close_prices[i] - open_prices[i])
            total_range = high_prices[i] - low_prices[i]
            
            if total_range > 0:
                body_ratio = body_size / total_range
                is_doji = body_ratio <= 0.1  # Doji threshold
                
                if is_doji:
                    # Count consecutive dojis
                    count = 1
                    j = i - 1
                    while j >= 0:
                        prev_body_size = abs(close_prices[j] - open_prices[j])
                        prev_total_range = high_prices[j] - low_prices[j]
                        if prev_total_range > 0:
                            prev_body_ratio = prev_body_size / prev_total_range
                            if prev_body_ratio <= 0.1:
                                count += 1
                                j -= 1
                            else:
                                break
                        else:
                            break
                    
                    doji_series[i] = min(count, self.advanced_config.max_series_length)
        
        return doji_series
    
    def _detect_hammer_series(self, open_prices: np.ndarray, high_prices: np.ndarray, 
                             low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Detect series of hammer patterns."""
        hammer_series = np.zeros(len(open_prices))
        
        for i in range(len(open_prices)):
            # Calculate hammer characteristics
            body_size = abs(close_prices[i] - open_prices[i])
            upper_shadow = high_prices[i] - max(open_prices[i], close_prices[i])
            lower_shadow = min(open_prices[i], close_prices[i]) - low_prices[i]
            total_range = high_prices[i] - low_prices[i]
            
            if total_range > 0:
                body_ratio = body_size / total_range
                lower_shadow_ratio = lower_shadow / total_range
                upper_shadow_ratio = upper_shadow / total_range
                
                is_hammer = (body_ratio <= 0.3 and 
                           lower_shadow_ratio >= 0.4 and 
                           upper_shadow_ratio <= 0.2)
                
                if is_hammer:
                    # Count consecutive hammers
                    count = 1
                    j = i - 1
                    while j >= 0:
                        prev_body_size = abs(close_prices[j] - open_prices[j])
                        prev_upper_shadow = high_prices[j] - max(open_prices[j], close_prices[j])
                        prev_lower_shadow = min(open_prices[j], close_prices[j]) - low_prices[j]
                        prev_total_range = high_prices[j] - low_prices[j]
                        
                        if prev_total_range > 0:
                            prev_body_ratio = prev_body_size / prev_total_range
                            prev_lower_shadow_ratio = prev_lower_shadow / prev_total_range
                            prev_upper_shadow_ratio = prev_upper_shadow / prev_total_range
                            
                            prev_is_hammer = (prev_body_ratio <= 0.3 and 
                                            prev_lower_shadow_ratio >= 0.4 and 
                                            prev_upper_shadow_ratio <= 0.2)
                            
                            if prev_is_hammer:
                                count += 1
                                j -= 1
                            else:
                                break
                        else:
                            break
                    
                    hammer_series[i] = min(count, self.advanced_config.max_series_length)
        
        return hammer_series
    
    def _detect_engulfing_series(self, open_prices: np.ndarray, high_prices: np.ndarray, 
                                low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Detect series of engulfing patterns."""
        engulfing_series = np.zeros(len(open_prices))
        
        for i in range(1, len(open_prices)):
            # Check for engulfing pattern
            current_body = abs(close_prices[i] - open_prices[i])
            prev_body = abs(close_prices[i-1] - open_prices[i-1])
            
            is_bullish_engulfing = (close_prices[i] > open_prices[i] and  # Current bullish
                                  close_prices[i-1] < open_prices[i-1] and  # Previous bearish
                                  open_prices[i] < close_prices[i-1] and  # Current open below prev close
                                  close_prices[i] > open_prices[i-1] and  # Current close above prev open
                                  current_body > prev_body * 1.2)  # Current body larger
            
            is_bearish_engulfing = (close_prices[i] < open_prices[i] and  # Current bearish
                                  close_prices[i-1] > open_prices[i-1] and  # Previous bullish
                                  open_prices[i] > close_prices[i-1] and  # Current open above prev close
                                  close_prices[i] < open_prices[i-1] and  # Current close below prev open
                                  current_body > prev_body * 1.2)  # Current body larger
            
            is_engulfing = is_bullish_engulfing or is_bearish_engulfing
            
            if is_engulfing:
                # Count consecutive engulfing patterns
                count = 1
                j = i - 1
                while j >= 1:
                    prev_current_body = abs(close_prices[j] - open_prices[j])
                    prev_prev_body = abs(close_prices[j-1] - open_prices[j-1])
                    
                    prev_is_bullish_engulfing = (close_prices[j] > open_prices[j] and
                                               close_prices[j-1] < open_prices[j-1] and
                                               open_prices[j] < close_prices[j-1] and
                                               close_prices[j] > open_prices[j-1] and
                                               prev_current_body > prev_prev_body * 1.2)
                    
                    prev_is_bearish_engulfing = (close_prices[j] < open_prices[j] and
                                               close_prices[j-1] > open_prices[j-1] and
                                               open_prices[j] > close_prices[j-1] and
                                               close_prices[j] < open_prices[j-1] and
                                               prev_current_body > prev_prev_body * 1.2)
                    
                    if prev_is_bullish_engulfing or prev_is_bearish_engulfing:
                        count += 1
                        j -= 1
                    else:
                        break
                
                engulfing_series[i] = min(count, self.advanced_config.max_series_length)
        
        return engulfing_series
    
    def _detect_momentum_series(self, close_prices: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Detect momentum-based series patterns."""
        momentum_series = np.zeros(len(close_prices))
        
        for i in range(2, len(close_prices)):
            # Calculate momentum indicators
            price_momentum = (close_prices[i] - close_prices[i-2]) / close_prices[i-2]
            volume_momentum = (volume[i] - volume[i-2]) / (volume[i-2] + 1e-8)
            
            # Strong momentum: price and volume both increasing
            strong_momentum = price_momentum > 0.02 and volume_momentum > 0.1
            
            if strong_momentum:
                # Count consecutive strong momentum periods
                count = 1
                j = i - 1
                while j >= 2:
                    prev_price_momentum = (close_prices[j] - close_prices[j-2]) / close_prices[j-2]
                    prev_volume_momentum = (volume[j] - volume[j-2]) / (volume[j-2] + 1e-8)
                    
                    prev_strong_momentum = prev_price_momentum > 0.02 and prev_volume_momentum > 0.1
                    
                    if prev_strong_momentum:
                        count += 1
                        j -= 1
                    else:
                        break
                
                momentum_series[i] = min(count, self.advanced_config.max_series_length)
        
        return momentum_series
    
    def _detect_reversal_series(self, open_prices: np.ndarray, high_prices: np.ndarray, 
                               low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Detect reversal pattern series."""
        reversal_series = np.zeros(len(open_prices))
        
        for i in range(2, len(open_prices)):
            # Check for reversal patterns (hammer, doji, etc.)
            body_size = abs(close_prices[i] - open_prices[i])
            upper_shadow = high_prices[i] - max(open_prices[i], close_prices[i])
            lower_shadow = min(open_prices[i], close_prices[i]) - low_prices[i]
            total_range = high_prices[i] - low_prices[i]
            
            if total_range > 0:
                body_ratio = body_size / total_range
                lower_shadow_ratio = lower_shadow / total_range
                upper_shadow_ratio = upper_shadow / total_range
                
                # Hammer reversal
                is_hammer_reversal = (body_ratio <= 0.3 and 
                                    lower_shadow_ratio >= 0.4 and 
                                    upper_shadow_ratio <= 0.2)
                
                # Doji reversal
                is_doji_reversal = body_ratio <= 0.1
                
                # Shooting star reversal
                is_shooting_star = (body_ratio <= 0.3 and 
                                  upper_shadow_ratio >= 0.4 and 
                                  lower_shadow_ratio <= 0.2)
                
                is_reversal = is_hammer_reversal or is_doji_reversal or is_shooting_star
                
                if is_reversal:
                    # Count consecutive reversal patterns
                    count = 1
                    j = i - 1
                    while j >= 2:
                        prev_body_size = abs(close_prices[j] - open_prices[j])
                        prev_upper_shadow = high_prices[j] - max(open_prices[j], close_prices[j])
                        prev_lower_shadow = min(open_prices[j], close_prices[j]) - low_prices[j]
                        prev_total_range = high_prices[j] - low_prices[j]
                        
                        if prev_total_range > 0:
                            prev_body_ratio = prev_body_size / prev_total_range
                            prev_lower_shadow_ratio = prev_lower_shadow / prev_total_range
                            prev_upper_shadow_ratio = prev_upper_shadow / prev_total_range
                            
                            prev_is_hammer = (prev_body_ratio <= 0.3 and 
                                            prev_lower_shadow_ratio >= 0.4 and 
                                            prev_upper_shadow_ratio <= 0.2)
                            prev_is_doji = prev_body_ratio <= 0.1
                            prev_is_shooting_star = (prev_body_ratio <= 0.3 and 
                                                  prev_upper_shadow_ratio >= 0.4 and 
                                                  prev_lower_shadow_ratio <= 0.2)
                            
                            if prev_is_hammer or prev_is_doji or prev_is_shooting_star:
                                count += 1
                                j -= 1
                            else:
                                break
                        else:
                            break
                    
                    reversal_series[i] = min(count, self.advanced_config.max_series_length)
        
        return reversal_series
    
    def _calculate_series_strength(self, close_prices: np.ndarray) -> np.ndarray:
        """Calculate strength of candle series."""
        series_strength = np.zeros(len(close_prices))
        
        for i in range(self.advanced_config.min_series_length, len(close_prices)):
            # Calculate series strength based on consistency and magnitude
            series_data = close_prices[i-self.advanced_config.min_series_length:i+1]
            
            # Calculate price changes
            price_changes = np.diff(series_data) / series_data[:-1]
            
            # Strength based on consistency of direction and magnitude
            direction_consistency = abs(np.mean(np.sign(price_changes)))
            magnitude_consistency = 1.0 - np.std(price_changes) / (np.mean(np.abs(price_changes)) + 1e-8)
            
            series_strength[i] = direction_consistency * magnitude_consistency
        
        return series_strength
    
    def _calculate_series_consistency(self, close_prices: np.ndarray) -> np.ndarray:
        """Calculate consistency of candle series."""
        series_consistency = np.zeros(len(close_prices))
        
        for i in range(self.advanced_config.min_series_length, len(close_prices)):
            # Calculate series consistency based on pattern repetition
            series_data = close_prices[i-self.advanced_config.min_series_length:i+1]
            
            # Calculate rolling correlation with previous periods
            if i >= self.advanced_config.min_series_length * 2:
                prev_series = close_prices[i-self.advanced_config.min_series_length*2:i-self.advanced_config.min_series_length]
                if len(prev_series) == len(series_data):
                    correlation = np.corrcoef(series_data, prev_series)[0, 1]
                    series_consistency[i] = correlation if not np.isnan(correlation) else 0
                else:
                    series_consistency[i] = 0
            else:
                series_consistency[i] = 0
        
        return series_consistency
    
    def _generate_cross_timeframe_features(self, data: pd.DataFrame) -> List[np.ndarray]:
        """Generate cross-timeframe candle features."""
        cross_timeframe_features = []
        
        # For now, simulate cross-timeframe analysis with different lookback periods
        # In a real implementation, you would fetch data from different timeframes
        
        close_prices = data['close'].values
        
        # Simulate different timeframes using different lookback periods
        timeframes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        
        for tf_name, tf_period in timeframes.items():
            if len(close_prices) >= tf_period:
                # Multi-timeframe trend analysis
                trend_feature = self._analyze_multi_timeframe_trend(close_prices, tf_period)
                cross_timeframe_features.append(trend_feature)
                
                # Timeframe confluence analysis
                confluence_feature = self._analyze_timeframe_confluence(close_prices, tf_period)
                cross_timeframe_features.append(confluence_feature)
                
                # Timeframe divergence analysis
                divergence_feature = self._analyze_timeframe_divergence(close_prices, tf_period)
                cross_timeframe_features.append(divergence_feature)
        
        return cross_timeframe_features
    
    def _analyze_multi_timeframe_trend(self, close_prices: np.ndarray, timeframe: int) -> np.ndarray:
        """Analyze trend across multiple timeframes."""
        trend_strength = np.zeros(len(close_prices))
        
        for i in range(timeframe, len(close_prices)):
            # Calculate trend strength for this timeframe
            period_data = close_prices[i-timeframe:i+1]
            
            # Linear regression slope
            x = np.arange(len(period_data))
            slope, _, r_value, _, _ = stats.linregress(x, period_data)
            
            # Trend strength based on slope and correlation
            trend_strength[i] = slope * r_value * r_value
        
        return trend_strength
    
    def _analyze_timeframe_confluence(self, close_prices: np.ndarray, timeframe: int) -> np.ndarray:
        """Analyze confluence between timeframes."""
        confluence = np.zeros(len(close_prices))
        
        for i in range(timeframe * 2, len(close_prices)):
            # Compare current timeframe with longer timeframe
            short_period = close_prices[i-timeframe:i+1]
            long_period = close_prices[i-timeframe*2:i+1]
            
            # Calculate trend direction for both timeframes
            short_trend = 1 if short_period[-1] > short_period[0] else -1
            long_trend = 1 if long_period[-1] > long_period[0] else -1
            
            # Confluence when both timeframes agree
            confluence[i] = 1 if short_trend == long_trend else 0
        
        return confluence
    
    def _analyze_timeframe_divergence(self, close_prices: np.ndarray, timeframe: int) -> np.ndarray:
        """Analyze divergence between timeframes."""
        divergence = np.zeros(len(close_prices))
        
        for i in range(timeframe * 2, len(close_prices)):
            # Compare price action with momentum
            short_period = close_prices[i-timeframe:i+1]
            long_period = close_prices[i-timeframe*2:i+1]
            
            # Calculate momentum
            short_momentum = (short_period[-1] - short_period[0]) / short_period[0]
            long_momentum = (long_period[-1] - long_period[0]) / long_period[0]
            
            # Divergence when price and momentum move in opposite directions
            price_direction = 1 if short_period[-1] > long_period[-1] else -1
            momentum_direction = 1 if short_momentum > long_momentum else -1
            
            divergence[i] = 1 if price_direction != momentum_direction else 0
        
        return divergence
    
    def _generate_multi_dimensional_features(self, data: pd.DataFrame) -> List[np.ndarray]:
        """Generate multi-dimensional interaction features."""
        multi_dimensional_features = []
        
        # Basic OHLCV data
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        volume = data['volume'].values if 'volume' in data.columns else np.ones(len(data))
        
        # 1. Volume interactions
        if InteractionDimension.VOLUME in self.advanced_config.interaction_dimensions:
            multi_dimensional_features.extend(self._generate_volume_interactions(
                open_prices, high_prices, low_prices, close_prices, volume
            ))
        
        # 2. Momentum interactions
        if InteractionDimension.MOMENTUM in self.advanced_config.interaction_dimensions:
            multi_dimensional_features.extend(self._generate_momentum_interactions(
                open_prices, high_prices, low_prices, close_prices, volume
            ))
        
        # 3. Volatility interactions
        if InteractionDimension.VOLATILITY in self.advanced_config.interaction_dimensions:
            multi_dimensional_features.extend(self._generate_volatility_interactions(
                open_prices, high_prices, low_prices, close_prices, volume
            ))
        
        # 4. Trend interactions
        if InteractionDimension.TREND in self.advanced_config.interaction_dimensions:
            multi_dimensional_features.extend(self._generate_trend_interactions(
                open_prices, high_prices, low_prices, close_prices, volume
            ))
        
        return multi_dimensional_features
    
    def _generate_volume_interactions(self, open_prices: np.ndarray, high_prices: np.ndarray, 
                                    low_prices: np.ndarray, close_prices: np.ndarray, 
                                    volume: np.ndarray) -> List[np.ndarray]:
        """Generate volume interaction features."""
        volume_features = []
        
        # Volume-price relationship
        volume_price_correlation = self._calculate_rolling_correlation(close_prices, volume)
        volume_features.append(volume_price_correlation)
        
        # Volume-weighted average price (VWAP) interaction
        vwap_interaction = self._calculate_vwap_interaction(high_prices, low_prices, close_prices, volume)
        volume_features.append(vwap_interaction)
        
        # Volume breakout patterns
        volume_breakout = self._detect_volume_breakouts(close_prices, volume)
        volume_features.append(volume_breakout)
        
        # Volume exhaustion patterns
        volume_exhaustion = self._detect_volume_exhaustion(close_prices, volume)
        volume_features.append(volume_exhaustion)
        
        # Volume divergence
        volume_divergence = self._detect_volume_divergence(close_prices, volume)
        volume_features.append(volume_divergence)
        
        return volume_features
    
    def _generate_momentum_interactions(self, open_prices: np.ndarray, high_prices: np.ndarray, 
                                      low_prices: np.ndarray, close_prices: np.ndarray, 
                                      volume: np.ndarray) -> List[np.ndarray]:
        """Generate momentum interaction features."""
        momentum_features = []
        
        # Price momentum with candle patterns
        price_momentum = self._calculate_price_momentum(close_prices)
        momentum_features.append(price_momentum)
        
        # Volume momentum
        volume_momentum = self._calculate_volume_momentum(volume)
        momentum_features.append(volume_momentum)
        
        # Momentum divergence
        momentum_divergence = self._detect_momentum_divergence(close_prices, price_momentum)
        momentum_features.append(momentum_divergence)
        
        # Momentum acceleration
        momentum_acceleration = self._calculate_momentum_acceleration(price_momentum)
        momentum_features.append(momentum_acceleration)
        
        # Momentum reversal patterns
        momentum_reversal = self._detect_momentum_reversal(price_momentum)
        momentum_features.append(momentum_reversal)
        
        return momentum_features
    
    def _generate_volatility_interactions(self, open_prices: np.ndarray, high_prices: np.ndarray, 
                                        low_prices: np.ndarray, close_prices: np.ndarray, 
                                        volume: np.ndarray) -> List[np.ndarray]:
        """Generate volatility interaction features."""
        volatility_features = []
        
        # Price volatility
        price_volatility = self._calculate_price_volatility(close_prices)
        volatility_features.append(price_volatility)
        
        # Volume volatility
        volume_volatility = self._calculate_volume_volatility(volume)
        volatility_features.append(volume_volatility)
        
        # Volatility clustering
        volatility_clustering = self._detect_volatility_clustering(price_volatility)
        volatility_features.append(volatility_clustering)
        
        # Volatility breakout
        volatility_breakout = self._detect_volatility_breakout(price_volatility)
        volatility_features.append(volatility_breakout)
        
        # Volatility mean reversion
        volatility_mean_reversion = self._detect_volatility_mean_reversion(price_volatility)
        volatility_features.append(volatility_mean_reversion)
        
        return volatility_features
    
    def _generate_trend_interactions(self, open_prices: np.ndarray, high_prices: np.ndarray, 
                                   low_prices: np.ndarray, close_prices: np.ndarray, 
                                   volume: np.ndarray) -> List[np.ndarray]:
        """Generate trend interaction features."""
        trend_features = []
        
        # Trend strength
        trend_strength = self._calculate_trend_strength(close_prices)
        trend_features.append(trend_strength)
        
        # Trend persistence
        trend_persistence = self._calculate_trend_persistence(close_prices)
        trend_features.append(trend_persistence)
        
        # Trend reversal signals
        trend_reversal = self._detect_trend_reversal(close_prices)
        trend_features.append(trend_reversal)
        
        # Trend acceleration
        trend_acceleration = self._calculate_trend_acceleration(close_prices)
        trend_features.append(trend_acceleration)
        
        # Trend divergence
        trend_divergence = self._detect_trend_divergence(close_prices, volume)
        trend_features.append(trend_divergence)
        
        return trend_features
    
    def _generate_pattern_strength_features(self, data: pd.DataFrame) -> List[np.ndarray]:
        """Generate pattern strength and quality features."""
        pattern_strength_features = []
        
        # Basic OHLC data
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        volume = data['volume'].values if 'volume' in data.columns else np.ones(len(data))
        
        # 1. Body size strength
        body_size_strength = self._calculate_body_size_strength(open_prices, close_prices)
        pattern_strength_features.append(body_size_strength)
        
        # 2. Shadow ratio strength
        shadow_ratio_strength = self._calculate_shadow_ratio_strength(
            open_prices, high_prices, low_prices, close_prices
        )
        pattern_strength_features.append(shadow_ratio_strength)
        
        # 3. Range ratio strength
        range_ratio_strength = self._calculate_range_ratio_strength(
            high_prices, low_prices, close_prices
        )
        pattern_strength_features.append(range_ratio_strength)
        
        # 4. Volume ratio strength
        volume_ratio_strength = self._calculate_volume_ratio_strength(close_prices, volume)
        pattern_strength_features.append(volume_ratio_strength)
        
        # 5. Momentum ratio strength
        momentum_ratio_strength = self._calculate_momentum_ratio_strength(close_prices)
        pattern_strength_features.append(momentum_ratio_strength)
        
        # 6. Pattern quality score
        pattern_quality = self._calculate_pattern_quality(
            open_prices, high_prices, low_prices, close_prices, volume
        )
        pattern_strength_features.append(pattern_quality)
        
        # 7. Pattern reliability
        pattern_reliability = self._calculate_pattern_reliability(
            open_prices, high_prices, low_prices, close_prices
        )
        pattern_strength_features.append(pattern_reliability)
        
        return pattern_strength_features
    
    def _calculate_body_size_strength(self, open_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Calculate body size strength relative to historical patterns."""
        body_size_strength = np.zeros(len(open_prices))
        
        for i in range(1, len(open_prices)):
            current_body = abs(close_prices[i] - open_prices[i])
            
            # Calculate average body size over lookback period
            lookback = min(i, self.advanced_config.interaction_lookback)
            historical_bodies = []
            
            for j in range(max(0, i - lookback), i):
                historical_bodies.append(abs(close_prices[j] - open_prices[j]))
            
            if historical_bodies:
                avg_body_size = np.mean(historical_bodies)
                std_body_size = np.std(historical_bodies)
                
                # Strength based on how much larger than average
                if std_body_size > 0:
                    z_score = (current_body - avg_body_size) / std_body_size
                    body_size_strength[i] = np.tanh(z_score)  # Normalize to [-1, 1]
                else:
                    body_size_strength[i] = 0
        
        return body_size_strength
    
    def _calculate_shadow_ratio_strength(self, open_prices: np.ndarray, high_prices: np.ndarray, 
                                       low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Calculate shadow ratio strength."""
        shadow_ratio_strength = np.zeros(len(open_prices))
        
        for i in range(len(open_prices)):
            body_size = abs(close_prices[i] - open_prices[i])
            upper_shadow = high_prices[i] - max(open_prices[i], close_prices[i])
            lower_shadow = min(open_prices[i], close_prices[i]) - low_prices[i]
            total_range = high_prices[i] - low_prices[i]
            
            if total_range > 0:
                upper_shadow_ratio = upper_shadow / total_range
                lower_shadow_ratio = lower_shadow / total_range
                
                # Strength based on shadow dominance
                shadow_dominance = max(upper_shadow_ratio, lower_shadow_ratio)
                shadow_ratio_strength[i] = shadow_dominance
        
        return shadow_ratio_strength
    
    def _calculate_range_ratio_strength(self, high_prices: np.ndarray, low_prices: np.ndarray, 
                                      close_prices: np.ndarray) -> np.ndarray:
        """Calculate range ratio strength."""
        range_ratio_strength = np.zeros(len(high_prices))
        
        for i in range(1, len(high_prices)):
            current_range = high_prices[i] - low_prices[i]
            
            # Calculate average range over lookback period
            lookback = min(i, self.advanced_config.interaction_lookback)
            historical_ranges = []
            
            for j in range(max(0, i - lookback), i):
                historical_ranges.append(high_prices[j] - low_prices[j])
            
            if historical_ranges:
                avg_range = np.mean(historical_ranges)
                
                # Strength based on range relative to average
                if avg_range > 0:
                    range_ratio = current_range / avg_range
                    range_ratio_strength[i] = min(range_ratio, 3.0) / 3.0  # Normalize to [0, 1]
                else:
                    range_ratio_strength[i] = 0
        
        return range_ratio_strength
    
    def _calculate_volume_ratio_strength(self, close_prices: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate volume ratio strength."""
        volume_ratio_strength = np.zeros(len(close_prices))
        
        for i in range(1, len(close_prices)):
            current_volume = volume[i]
            
            # Calculate average volume over lookback period
            lookback = min(i, self.advanced_config.interaction_lookback)
            historical_volumes = volume[max(0, i - lookback):i]
            
            if len(historical_volumes) > 0:
                avg_volume = np.mean(historical_volumes)
                
                # Strength based on volume relative to average
                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    volume_ratio_strength[i] = min(volume_ratio, 5.0) / 5.0  # Normalize to [0, 1]
                else:
                    volume_ratio_strength[i] = 0
        
        return volume_ratio_strength
    
    def _calculate_momentum_ratio_strength(self, close_prices: np.ndarray) -> np.ndarray:
        """Calculate momentum ratio strength."""
        momentum_ratio_strength = np.zeros(len(close_prices))
        
        for i in range(2, len(close_prices)):
            # Calculate current momentum
            current_momentum = (close_prices[i] - close_prices[i-2]) / close_prices[i-2]
            
            # Calculate average momentum over lookback period
            lookback = min(i, self.advanced_config.interaction_lookback)
            historical_momentums = []
            
            for j in range(max(2, i - lookback), i):
                momentum = (close_prices[j] - close_prices[j-2]) / close_prices[j-2]
                historical_momentums.append(momentum)
            
            if historical_momentums:
                avg_momentum = np.mean(historical_momentums)
                std_momentum = np.std(historical_momentums)
                
                # Strength based on momentum relative to historical average
                if std_momentum > 0:
                    z_score = (current_momentum - avg_momentum) / std_momentum
                    momentum_ratio_strength[i] = np.tanh(z_score)  # Normalize to [-1, 1]
                else:
                    momentum_ratio_strength[i] = 0
        
        return momentum_ratio_strength
    
    def _calculate_pattern_quality(self, open_prices: np.ndarray, high_prices: np.ndarray, 
                                 low_prices: np.ndarray, close_prices: np.ndarray, 
                                 volume: np.ndarray) -> np.ndarray:
        """Calculate overall pattern quality score."""
        pattern_quality = np.zeros(len(open_prices))
        
        for i in range(len(open_prices)):
            quality_factors = []
            
            # Body size quality
            body_size = abs(close_prices[i] - open_prices[i])
            total_range = high_prices[i] - low_prices[i]
            if total_range > 0:
                body_ratio = body_size / total_range
                # Quality increases with moderate body size (not too small, not too large)
                body_quality = 1.0 - abs(body_ratio - 0.5) * 2
                quality_factors.append(max(0, body_quality))
            
            # Shadow balance quality
            upper_shadow = high_prices[i] - max(open_prices[i], close_prices[i])
            lower_shadow = min(open_prices[i], close_prices[i]) - low_prices[i]
            if total_range > 0:
                upper_ratio = upper_shadow / total_range
                lower_ratio = lower_shadow / total_range
                # Quality increases with balanced shadows
                shadow_balance = 1.0 - abs(upper_ratio - lower_ratio)
                quality_factors.append(max(0, shadow_balance))
            
            # Volume quality
            if i > 0:
                volume_ratio = volume[i] / (volume[i-1] + 1e-8)
                # Quality increases with moderate volume (not too low, not too high)
                volume_quality = 1.0 - abs(volume_ratio - 1.0) / 2.0
                quality_factors.append(max(0, min(1.0, volume_quality)))
            
            # Price range quality
            if i > 0:
                price_change = abs(close_prices[i] - close_prices[i-1]) / close_prices[i-1]
                # Quality increases with moderate price change
                price_quality = 1.0 - abs(price_change - 0.01) * 50  # 1% is ideal
                quality_factors.append(max(0, min(1.0, price_quality)))
            
            # Overall quality is average of all factors
            if quality_factors:
                pattern_quality[i] = np.mean(quality_factors)
        
        return pattern_quality
    
    def _calculate_pattern_reliability(self, open_prices: np.ndarray, high_prices: np.ndarray, 
                                     low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Calculate pattern reliability based on historical accuracy."""
        pattern_reliability = np.zeros(len(open_prices))
        
        for i in range(self.advanced_config.interaction_lookback, len(open_prices)):
            # Analyze recent patterns for reliability
            recent_patterns = []
            
            for j in range(i - self.advanced_config.interaction_lookback, i):
                # Simple pattern classification
                body_size = abs(close_prices[j] - open_prices[j])
                total_range = high_prices[j] - low_prices[j]
                
                if total_range > 0:
                    body_ratio = body_size / total_range
                    upper_shadow = high_prices[j] - max(open_prices[j], close_prices[j])
                    lower_shadow = min(open_prices[j], close_prices[j]) - low_prices[j]
                    upper_ratio = upper_shadow / total_range
                    lower_ratio = lower_shadow / total_range
                    
                    # Classify pattern
                    if body_ratio <= 0.1:
                        pattern_type = 'doji'
                    elif lower_ratio >= 0.4 and upper_ratio <= 0.2:
                        pattern_type = 'hammer'
                    elif upper_ratio >= 0.4 and lower_ratio <= 0.2:
                        pattern_type = 'shooting_star'
                    else:
                        pattern_type = 'normal'
                    
                    recent_patterns.append(pattern_type)
            
            # Calculate reliability based on pattern consistency
            if recent_patterns:
                pattern_counts = {}
                for pattern in recent_patterns:
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                
                # Reliability is based on most common pattern frequency
                max_frequency = max(pattern_counts.values())
                pattern_reliability[i] = max_frequency / len(recent_patterns)
        
        return pattern_reliability
    
    def _generate_temporal_analysis_features(self, data: pd.DataFrame) -> List[np.ndarray]:
        """Generate temporal analysis features."""
        temporal_features = []
        
        close_prices = data['close'].values
        
        # Time-based patterns
        temporal_features.append(self._analyze_time_of_day_patterns(data))
        temporal_features.append(self._analyze_day_of_week_patterns(data))
        temporal_features.append(self._analyze_seasonal_patterns(data))
        
        # Cyclical patterns
        temporal_features.append(self._detect_cyclical_patterns(close_prices))
        
        # Trend persistence
        temporal_features.append(self._analyze_trend_persistence(close_prices))
        
        return temporal_features
    
    def _generate_pattern_categorization_features(self, data: pd.DataFrame) -> List[np.ndarray]:
        """Generate pattern categorization features."""
        categorization_features = []
        
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        
        # Pattern categories
        categorization_features.append(self._categorize_bullish_patterns(
            open_prices, high_prices, low_prices, close_prices
        ))
        categorization_features.append(self._categorize_bearish_patterns(
            open_prices, high_prices, low_prices, close_prices
        ))
        categorization_features.append(self._categorize_reversal_patterns(
            open_prices, high_prices, low_prices, close_prices
        ))
        categorization_features.append(self._categorize_continuation_patterns(
            open_prices, high_prices, low_prices, close_prices
        ))
        
        return categorization_features
    
    def _generate_market_structure_features(self, data: pd.DataFrame) -> List[np.ndarray]:
        """Generate market structure features."""
        market_structure_features = []
        
        close_prices = data['close'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        
        # Market structure analysis
        market_structure_features.append(self._analyze_support_resistance(high_prices, low_prices))
        market_structure_features.append(self._analyze_trend_structure(close_prices))
        market_structure_features.append(self._analyze_market_regime(close_prices))
        market_structure_features.append(self._analyze_liquidity_zones(high_prices, low_prices))
        
        return market_structure_features
    
    # Helper methods for various calculations
    def _calculate_rolling_correlation(self, series1: np.ndarray, series2: np.ndarray) -> np.ndarray:
        """Calculate rolling correlation between two series."""
        correlation = np.zeros(len(series1))
        
        for i in range(self.advanced_config.interaction_lookback, len(series1)):
            window1 = series1[i-self.advanced_config.interaction_lookback:i+1]
            window2 = series2[i-self.advanced_config.interaction_lookback:i+1]
            
            if len(window1) > 1 and len(window2) > 1:
                corr = np.corrcoef(window1, window2)[0, 1]
                correlation[i] = corr if not np.isnan(corr) else 0
        
        return correlation
    
    def _calculate_vwap_interaction(self, high_prices: np.ndarray, low_prices: np.ndarray, 
                                  close_prices: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate VWAP interaction features."""
        vwap_interaction = np.zeros(len(close_prices))
        
        for i in range(1, len(close_prices)):
            # Calculate VWAP
            lookback = min(i, self.advanced_config.interaction_lookback)
            vwap = np.sum(close_prices[i-lookback:i+1] * volume[i-lookback:i+1]) / np.sum(volume[i-lookback:i+1])
            
            # Interaction with current price
            if vwap > 0:
                vwap_interaction[i] = (close_prices[i] - vwap) / vwap
        
        return vwap_interaction
    
    def _detect_volume_breakouts(self, close_prices: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Detect volume breakout patterns."""
        volume_breakouts = np.zeros(len(close_prices))
        
        for i in range(self.advanced_config.interaction_lookback, len(close_prices)):
            # Calculate volume threshold
            lookback = min(i, self.advanced_config.interaction_lookback)
            recent_volumes = volume[i-lookback:i]
            volume_threshold = np.mean(recent_volumes) + 2 * np.std(recent_volumes)
            
            # Check for volume breakout
            if volume[i] > volume_threshold:
                # Check if price also moved significantly
                price_change = abs(close_prices[i] - close_prices[i-1]) / close_prices[i-1]
                if price_change > 0.01:  # 1% price change
                    volume_breakouts[i] = 1
        
        return volume_breakouts
    
    def _detect_volume_exhaustion(self, close_prices: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Detect volume exhaustion patterns."""
        volume_exhaustion = np.zeros(len(close_prices))
        
        for i in range(self.advanced_config.interaction_lookback, len(close_prices)):
            # Check for decreasing volume with increasing price
            lookback = min(i, self.advanced_config.interaction_lookback)
            recent_volumes = volume[i-lookback:i+1]
            recent_prices = close_prices[i-lookback:i+1]
            
            # Volume trend
            volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
            price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            
            # Exhaustion: decreasing volume with increasing price
            if volume_trend < 0 and price_trend > 0:
                volume_exhaustion[i] = 1
        
        return volume_exhaustion
    
    def _detect_volume_divergence(self, close_prices: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Detect volume divergence patterns."""
        volume_divergence = np.zeros(len(close_prices))
        
        for i in range(self.advanced_config.interaction_lookback, len(close_prices)):
            # Calculate price and volume trends
            lookback = min(i, self.advanced_config.interaction_lookback)
            recent_prices = close_prices[i-lookback:i+1]
            recent_volumes = volume[i-lookback:i+1]
            
            price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
            
            # Divergence: opposite trends
            if (price_trend > 0 and volume_trend < 0) or (price_trend < 0 and volume_trend > 0):
                volume_divergence[i] = 1
        
        return volume_divergence
    
    def _calculate_price_momentum(self, close_prices: np.ndarray) -> np.ndarray:
        """Calculate price momentum."""
        momentum = np.zeros(len(close_prices))
        
        for i in range(1, len(close_prices)):
            momentum[i] = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
        
        return momentum
    
    def _calculate_volume_momentum(self, volume: np.ndarray) -> np.ndarray:
        """Calculate volume momentum."""
        momentum = np.zeros(len(volume))
        
        for i in range(1, len(volume)):
            if volume[i-1] > 0:
                momentum[i] = (volume[i] - volume[i-1]) / volume[i-1]
        
        return momentum
    
    def _detect_momentum_divergence(self, close_prices: np.ndarray, momentum: np.ndarray) -> np.ndarray:
        """Detect momentum divergence."""
        divergence = np.zeros(len(close_prices))
        
        for i in range(self.advanced_config.interaction_lookback, len(close_prices)):
            # Calculate price and momentum trends
            lookback = min(i, self.advanced_config.interaction_lookback)
            recent_prices = close_prices[i-lookback:i+1]
            recent_momentum = momentum[i-lookback:i+1]
            
            price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            momentum_trend = np.polyfit(range(len(recent_momentum)), recent_momentum, 1)[0]
            
            # Divergence: opposite trends
            if (price_trend > 0 and momentum_trend < 0) or (price_trend < 0 and momentum_trend > 0):
                divergence[i] = 1
        
        return divergence
    
    def _calculate_momentum_acceleration(self, momentum: np.ndarray) -> np.ndarray:
        """Calculate momentum acceleration."""
        acceleration = np.zeros(len(momentum))
        
        for i in range(1, len(momentum)):
            acceleration[i] = momentum[i] - momentum[i-1]
        
        return acceleration
    
    def _detect_momentum_reversal(self, momentum: np.ndarray) -> np.ndarray:
        """Detect momentum reversal patterns."""
        reversal = np.zeros(len(momentum))
        
        for i in range(2, len(momentum)):
            # Check for momentum reversal
            if (momentum[i-1] > 0 and momentum[i] < 0) or (momentum[i-1] < 0 and momentum[i] > 0):
                reversal[i] = 1
        
        return reversal
    
    def _calculate_price_volatility(self, close_prices: np.ndarray) -> np.ndarray:
        """Calculate price volatility."""
        volatility = np.zeros(len(close_prices))
        
        for i in range(self.advanced_config.interaction_lookback, len(close_prices)):
            lookback = min(i, self.advanced_config.interaction_lookback)
            recent_prices = close_prices[i-lookback:i+1]
            
            if len(recent_prices) > 1:
                returns = np.diff(recent_prices) / recent_prices[:-1]
                volatility[i] = np.std(returns)
        
        return volatility
    
    def _calculate_volume_volatility(self, volume: np.ndarray) -> np.ndarray:
        """Calculate volume volatility."""
        volatility = np.zeros(len(volume))
        
        for i in range(self.advanced_config.interaction_lookback, len(volume)):
            lookback = min(i, self.advanced_config.interaction_lookback)
            recent_volumes = volume[i-lookback:i+1]
            
            if len(recent_volumes) > 1:
                volatility[i] = np.std(recent_volumes) / np.mean(recent_volumes)
        
        return volatility
    
    def _detect_volatility_clustering(self, volatility: np.ndarray) -> np.ndarray:
        """Detect volatility clustering."""
        clustering = np.zeros(len(volatility))
        
        for i in range(self.advanced_config.interaction_lookback, len(volatility)):
            lookback = min(i, self.advanced_config.interaction_lookback)
            recent_volatility = volatility[i-lookback:i+1]
            
            # Check if current volatility is significantly higher than average
            avg_volatility = np.mean(recent_volatility[:-1])
            if avg_volatility > 0:
                volatility_ratio = recent_volatility[-1] / avg_volatility
                if volatility_ratio > 1.5:  # 50% higher than average
                    clustering[i] = 1
        
        return clustering
    
    def _detect_volatility_breakout(self, volatility: np.ndarray) -> np.ndarray:
        """Detect volatility breakout."""
        breakout = np.zeros(len(volatility))
        
        for i in range(self.advanced_config.interaction_lookback, len(volatility)):
            lookback = min(i, self.advanced_config.interaction_lookback)
            recent_volatility = volatility[i-lookback:i]
            
            if len(recent_volatility) > 0:
                volatility_threshold = np.mean(recent_volatility) + 2 * np.std(recent_volatility)
                if volatility[i] > volatility_threshold:
                    breakout[i] = 1
        
        return breakout
    
    def _detect_volatility_mean_reversion(self, volatility: np.ndarray) -> np.ndarray:
        """Detect volatility mean reversion."""
        mean_reversion = np.zeros(len(volatility))
        
        for i in range(self.advanced_config.interaction_lookback, len(volatility)):
            lookback = min(i, self.advanced_config.interaction_lookback)
            recent_volatility = volatility[i-lookback:i+1]
            
            # Check if volatility is reverting to mean
            if len(recent_volatility) > 2:
                current_vol = recent_volatility[-1]
                avg_vol = np.mean(recent_volatility[:-1])
                
                if current_vol < avg_vol * 0.8:  # 20% below average
                    mean_reversion[i] = 1
        
        return mean_reversion
    
    def _calculate_trend_strength(self, close_prices: np.ndarray) -> np.ndarray:
        """Calculate trend strength."""
        trend_strength = np.zeros(len(close_prices))
        
        for i in range(self.advanced_config.interaction_lookback, len(close_prices)):
            lookback = min(i, self.advanced_config.interaction_lookback)
            recent_prices = close_prices[i-lookback:i+1]
            
            if len(recent_prices) > 1:
                # Linear regression slope
                x = np.arange(len(recent_prices))
                slope, _, r_value, _, _ = stats.linregress(x, recent_prices)
                trend_strength[i] = abs(slope) * r_value * r_value
        
        return trend_strength
    
    def _calculate_trend_persistence(self, close_prices: np.ndarray) -> np.ndarray:
        """Calculate trend persistence."""
        persistence = np.zeros(len(close_prices))
        
        for i in range(self.advanced_config.interaction_lookback, len(close_prices)):
            lookback = min(i, self.advanced_config.interaction_lookback)
            recent_prices = close_prices[i-lookback:i+1]
            
            if len(recent_prices) > 1:
                # Count consecutive moves in same direction
                moves = np.diff(recent_prices)
                if len(moves) > 0:
                    # Count consecutive positive or negative moves
                    consecutive_count = 1
                    current_direction = 1 if moves[-1] > 0 else -1
                    
                    for j in range(len(moves) - 2, -1, -1):
                        if (moves[j] > 0 and current_direction > 0) or (moves[j] < 0 and current_direction < 0):
                            consecutive_count += 1
                        else:
                            break
                    
                    persistence[i] = consecutive_count / len(moves)
        
        return persistence
    
    def _detect_trend_reversal(self, close_prices: np.ndarray) -> np.ndarray:
        """Detect trend reversal signals."""
        reversal = np.zeros(len(close_prices))
        
        for i in range(self.advanced_config.interaction_lookback, len(close_prices)):
            lookback = min(i, self.advanced_config.interaction_lookback)
            recent_prices = close_prices[i-lookback:i+1]
            
            if len(recent_prices) > 2:
                # Check for trend reversal
                short_trend = recent_prices[-1] - recent_prices[-3]
                long_trend = recent_prices[-3] - recent_prices[-lookback]
                
                # Reversal: short trend opposite to long trend
                if (short_trend > 0 and long_trend < 0) or (short_trend < 0 and long_trend > 0):
                    reversal[i] = 1
        
        return reversal
    
    def _calculate_trend_acceleration(self, close_prices: np.ndarray) -> np.ndarray:
        """Calculate trend acceleration."""
        acceleration = np.zeros(len(close_prices))
        
        for i in range(2, len(close_prices)):
            # Calculate second derivative (acceleration)
            first_derivative = close_prices[i] - close_prices[i-1]
            second_derivative = close_prices[i] - 2 * close_prices[i-1] + close_prices[i-2]
            acceleration[i] = second_derivative
        
        return acceleration
    
    def _detect_trend_divergence(self, close_prices: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Detect trend divergence."""
        divergence = np.zeros(len(close_prices))
        
        for i in range(self.advanced_config.interaction_lookback, len(close_prices)):
            lookback = min(i, self.advanced_config.interaction_lookback)
            recent_prices = close_prices[i-lookback:i+1]
            recent_volumes = volume[i-lookback:i+1]
            
            if len(recent_prices) > 1 and len(recent_volumes) > 1:
                price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
                volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
                
                # Divergence: opposite trends
                if (price_trend > 0 and volume_trend < 0) or (price_trend < 0 and volume_trend > 0):
                    divergence[i] = 1
        
        return divergence
    
    def _analyze_time_of_day_patterns(self, data: pd.DataFrame) -> np.ndarray:
        """Analyze time-of-day patterns."""
        time_patterns = np.zeros(len(data))
        
        if hasattr(data.index, 'hour'):
            for i in range(len(data)):
                hour = data.index[i].hour
                # Simple time-based pattern (can be enhanced)
                if 9 <= hour <= 16:  # Market hours
                    time_patterns[i] = 1
                elif 20 <= hour <= 23:  # Evening
                    time_patterns[i] = 0.5
                else:  # Off hours
                    time_patterns[i] = 0
        
        return time_patterns
    
    def _analyze_day_of_week_patterns(self, data: pd.DataFrame) -> np.ndarray:
        """Analyze day-of-week patterns."""
        day_patterns = np.zeros(len(data))
        
        if hasattr(data.index, 'weekday'):
            for i in range(len(data)):
                weekday = data.index[i].weekday()
                # Simple day-based pattern (can be enhanced)
                if weekday < 5:  # Weekdays
                    day_patterns[i] = 1
                else:  # Weekend
                    day_patterns[i] = 0
        
        return day_patterns
    
    def _analyze_seasonal_patterns(self, data: pd.DataFrame) -> np.ndarray:
        """Analyze seasonal patterns."""
        seasonal_patterns = np.zeros(len(data))
        
        if hasattr(data.index, 'month'):
            for i in range(len(data)):
                month = data.index[i].month
                # Simple seasonal pattern (can be enhanced)
                if month in [1, 2, 3]:  # Q1
                    seasonal_patterns[i] = 0.8
                elif month in [4, 5, 6]:  # Q2
                    seasonal_patterns[i] = 1.0
                elif month in [7, 8, 9]:  # Q3
                    seasonal_patterns[i] = 0.9
                else:  # Q4
                    seasonal_patterns[i] = 0.7
        
        return seasonal_patterns
    
    def _detect_cyclical_patterns(self, close_prices: np.ndarray) -> np.ndarray:
        """Detect cyclical patterns."""
        cyclical = np.zeros(len(close_prices))
        
        for i in range(self.advanced_config.interaction_lookback, len(close_prices)):
            lookback = min(i, self.advanced_config.interaction_lookback)
            recent_prices = close_prices[i-lookback:i+1]
            
            if len(recent_prices) > 10:
                # Use FFT to detect cycles
                fft = np.fft.fft(recent_prices)
                freqs = np.fft.fftfreq(len(recent_prices))
                
                # Find dominant frequency
                power = np.abs(fft)
                dominant_freq_idx = np.argmax(power[1:len(power)//2]) + 1
                dominant_freq = freqs[dominant_freq_idx]
                
                # Cyclical strength
                cyclical[i] = power[dominant_freq_idx] / np.sum(power)
        
        return cyclical
    
    def _analyze_trend_persistence(self, close_prices: np.ndarray) -> np.ndarray:
        """Analyze trend persistence."""
        persistence = np.zeros(len(close_prices))
        
        for i in range(self.advanced_config.interaction_lookback, len(close_prices)):
            lookback = min(i, self.advanced_config.interaction_lookback)
            recent_prices = close_prices[i-lookback:i+1]
            
            if len(recent_prices) > 1:
                # Calculate trend consistency
                moves = np.diff(recent_prices)
                positive_moves = np.sum(moves > 0)
                negative_moves = np.sum(moves < 0)
                
                # Persistence is based on dominance of one direction
                total_moves = positive_moves + negative_moves
                if total_moves > 0:
                    persistence[i] = max(positive_moves, negative_moves) / total_moves
        
        return persistence
    
    def _categorize_bullish_patterns(self, open_prices: np.ndarray, high_prices: np.ndarray, 
                                   low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Categorize bullish patterns."""
        bullish_patterns = np.zeros(len(open_prices))
        
        for i in range(len(open_prices)):
            # Hammer pattern
            body_size = abs(close_prices[i] - open_prices[i])
            upper_shadow = high_prices[i] - max(open_prices[i], close_prices[i])
            lower_shadow = min(open_prices[i], close_prices[i]) - low_prices[i]
            total_range = high_prices[i] - low_prices[i]
            
            if total_range > 0:
                body_ratio = body_size / total_range
                lower_shadow_ratio = lower_shadow / total_range
                upper_shadow_ratio = upper_shadow / total_range
                
                # Hammer: small body, long lower shadow, short upper shadow
                is_hammer = (body_ratio <= 0.3 and 
                           lower_shadow_ratio >= 0.4 and 
                           upper_shadow_ratio <= 0.2)
                
                # Bullish engulfing (need previous candle)
                is_bullish_engulfing = False
                if i > 0:
                    prev_body = abs(close_prices[i-1] - open_prices[i-1])
                    is_bullish_engulfing = (close_prices[i] > open_prices[i] and  # Current bullish
                                          close_prices[i-1] < open_prices[i-1] and  # Previous bearish
                                          body_size > prev_body * 1.2)
                
                if is_hammer or is_bullish_engulfing:
                    bullish_patterns[i] = 1
        
        return bullish_patterns
    
    def _categorize_bearish_patterns(self, open_prices: np.ndarray, high_prices: np.ndarray, 
                                   low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Categorize bearish patterns."""
        bearish_patterns = np.zeros(len(open_prices))
        
        for i in range(len(open_prices)):
            # Shooting star pattern
            body_size = abs(close_prices[i] - open_prices[i])
            upper_shadow = high_prices[i] - max(open_prices[i], close_prices[i])
            lower_shadow = min(open_prices[i], close_prices[i]) - low_prices[i]
            total_range = high_prices[i] - low_prices[i]
            
            if total_range > 0:
                body_ratio = body_size / total_range
                lower_shadow_ratio = lower_shadow / total_range
                upper_shadow_ratio = upper_shadow / total_range
                
                # Shooting star: small body, long upper shadow, short lower shadow
                is_shooting_star = (body_ratio <= 0.3 and 
                                  upper_shadow_ratio >= 0.4 and 
                                  lower_shadow_ratio <= 0.2)
                
                # Bearish engulfing (need previous candle)
                is_bearish_engulfing = False
                if i > 0:
                    prev_body = abs(close_prices[i-1] - open_prices[i-1])
                    is_bearish_engulfing = (close_prices[i] < open_prices[i] and  # Current bearish
                                          close_prices[i-1] > open_prices[i-1] and  # Previous bullish
                                          body_size > prev_body * 1.2)
                
                if is_shooting_star or is_bearish_engulfing:
                    bearish_patterns[i] = 1
        
        return bearish_patterns
    
    def _categorize_reversal_patterns(self, open_prices: np.ndarray, high_prices: np.ndarray, 
                                    low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Categorize reversal patterns."""
        reversal_patterns = np.zeros(len(open_prices))
        
        for i in range(len(open_prices)):
            # Doji pattern
            body_size = abs(close_prices[i] - open_prices[i])
            total_range = high_prices[i] - low_prices[i]
            
            if total_range > 0:
                body_ratio = body_size / total_range
                is_doji = body_ratio <= 0.1
                
                # Hammer or shooting star (already defined above)
                upper_shadow = high_prices[i] - max(open_prices[i], close_prices[i])
                lower_shadow = min(open_prices[i], close_prices[i]) - low_prices[i]
                upper_shadow_ratio = upper_shadow / total_range
                lower_shadow_ratio = lower_shadow / total_range
                
                is_hammer = (body_ratio <= 0.3 and 
                           lower_shadow_ratio >= 0.4 and 
                           upper_shadow_ratio <= 0.2)
                
                is_shooting_star = (body_ratio <= 0.3 and 
                                  upper_shadow_ratio >= 0.4 and 
                                  lower_shadow_ratio <= 0.2)
                
                if is_doji or is_hammer or is_shooting_star:
                    reversal_patterns[i] = 1
        
        return reversal_patterns
    
    def _categorize_continuation_patterns(self, open_prices: np.ndarray, high_prices: np.ndarray, 
                                        low_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Categorize continuation patterns."""
        continuation_patterns = np.zeros(len(open_prices))
        
        for i in range(1, len(open_prices)):
            # Flag pattern (small body, long shadows)
            body_size = abs(close_prices[i] - open_prices[i])
            upper_shadow = high_prices[i] - max(open_prices[i], close_prices[i])
            lower_shadow = min(open_prices[i], close_prices[i]) - low_prices[i]
            total_range = high_prices[i] - low_prices[i]
            
            if total_range > 0:
                body_ratio = body_size / total_range
                upper_shadow_ratio = upper_shadow / total_range
                lower_shadow_ratio = lower_shadow / total_range
                
                # Flag: small body, long shadows on both sides
                is_flag = (body_ratio <= 0.2 and 
                          upper_shadow_ratio >= 0.3 and 
                          lower_shadow_ratio >= 0.3)
                
                # Spinning top (small body, long shadows)
                is_spinning_top = (body_ratio <= 0.3 and 
                                 upper_shadow_ratio >= 0.2 and 
                                 lower_shadow_ratio >= 0.2)
                
                if is_flag or is_spinning_top:
                    continuation_patterns[i] = 1
        
        return continuation_patterns
    
    def _analyze_support_resistance(self, high_prices: np.ndarray, low_prices: np.ndarray) -> np.ndarray:
        """Analyze support and resistance levels."""
        support_resistance = np.zeros(len(high_prices))
        
        for i in range(self.advanced_config.interaction_lookback, len(high_prices)):
            lookback = min(i, self.advanced_config.interaction_lookback)
            recent_highs = high_prices[i-lookback:i+1]
            recent_lows = low_prices[i-lookback:i+1]
            
            # Find peaks and troughs
            high_peaks, _ = find_peaks(recent_highs)
            low_troughs, _ = find_peaks(-recent_lows)
            
            # Support/resistance strength
            if len(high_peaks) > 0 and len(low_troughs) > 0:
                resistance_levels = recent_highs[high_peaks]
                support_levels = recent_lows[low_troughs]
                
                # Current price relative to support/resistance
                current_price = (recent_highs[-1] + recent_lows[-1]) / 2
                
                # Distance to nearest support/resistance
                if len(resistance_levels) > 0:
                    resistance_distance = min(abs(current_price - level) for level in resistance_levels)
                else:
                    resistance_distance = float('inf')
                
                if len(support_levels) > 0:
                    support_distance = min(abs(current_price - level) for level in support_levels)
                else:
                    support_distance = float('inf')
                
                # Strength based on proximity to levels
                min_distance = min(resistance_distance, support_distance)
                if min_distance != float('inf'):
                    support_resistance[i] = 1.0 / (1.0 + min_distance)
        
        return support_resistance
    
    def _analyze_trend_structure(self, close_prices: np.ndarray) -> np.ndarray:
        """Analyze trend structure."""
        trend_structure = np.zeros(len(close_prices))
        
        for i in range(self.advanced_config.interaction_lookback, len(close_prices)):
            lookback = min(i, self.advanced_config.interaction_lookback)
            recent_prices = close_prices[i-lookback:i+1]
            
            if len(recent_prices) > 2:
                # Analyze higher highs, higher lows, etc.
                highs = []
                lows = []
                
                for j in range(1, len(recent_prices) - 1):
                    if recent_prices[j] > recent_prices[j-1] and recent_prices[j] > recent_prices[j+1]:
                        highs.append(recent_prices[j])
                    elif recent_prices[j] < recent_prices[j-1] and recent_prices[j] < recent_prices[j+1]:
                        lows.append(recent_prices[j])
                
                # Trend structure analysis
                if len(highs) >= 2 and len(lows) >= 2:
                    # Higher highs and higher lows = uptrend
                    higher_highs = all(highs[i] > highs[i-1] for i in range(1, len(highs)))
                    higher_lows = all(lows[i] > lows[i-1] for i in range(1, len(lows)))
                    
                    if higher_highs and higher_lows:
                        trend_structure[i] = 1  # Strong uptrend
                    elif higher_highs or higher_lows:
                        trend_structure[i] = 0.5  # Weak uptrend
                    else:
                        trend_structure[i] = 0  # No clear trend
    
    def _analyze_market_regime(self, close_prices: np.ndarray) -> np.ndarray:
        """Analyze market regime."""
        market_regime = np.zeros(len(close_prices))
        
        for i in range(self.advanced_config.interaction_lookback, len(close_prices)):
            lookback = min(i, self.advanced_config.interaction_lookback)
            recent_prices = close_prices[i-lookback:i+1]
            
            if len(recent_prices) > 1:
                # Calculate volatility and trend
                returns = np.diff(recent_prices) / recent_prices[:-1]
                volatility = np.std(returns)
                trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                
                # Regime classification
                if volatility > 0.02 and abs(trend) > 0.01:  # High volatility, strong trend
                    market_regime[i] = 3  # Trending market
                elif volatility > 0.02:  # High volatility, weak trend
                    market_regime[i] = 2  # Volatile market
                elif abs(trend) > 0.01:  # Low volatility, strong trend
                    market_regime[i] = 1  # Stable trend
                else:  # Low volatility, weak trend
                    market_regime[i] = 0  # Sideways market
    
    def _analyze_liquidity_zones(self, high_prices: np.ndarray, low_prices: np.ndarray) -> np.ndarray:
        """Analyze liquidity zones."""
        liquidity_zones = np.zeros(len(high_prices))
        
        for i in range(self.advanced_config.interaction_lookback, len(high_prices)):
            lookback = min(i, self.advanced_config.interaction_lookback)
            recent_highs = high_prices[i-lookback:i+1]
            recent_lows = low_prices[i-lookback:i+1]
            
            # Calculate price range and volume (simplified)
            price_range = np.mean(recent_highs) - np.mean(recent_lows)
            range_volatility = np.std(recent_highs - recent_lows)
            
            # Liquidity based on range consistency
            if price_range > 0:
                liquidity_score = 1.0 - (range_volatility / price_range)
                liquidity_zones[i] = max(0, liquidity_score)
        
        return liquidity_zones


def create_advanced_candle_feature_generator(
    config: Optional[FeatureConfig] = None,
    advanced_config: Optional[AdvancedFeatureConfig] = None
) -> AdvancedCandleFeatureGenerator:
    """Create an advanced candle feature generator."""
    return AdvancedCandleFeatureGenerator(config, advanced_config)


def test_advanced_candle_features():
    """Test function for advanced candle features."""
    print("🧪 Testing Advanced Candle Features...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic OHLCV data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_samples)
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=pd.date_range('2020-01-01', periods=n_samples, freq='1min'))
    
    # Ensure OHLC constraints
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Create advanced feature generator
    advanced_config = AdvancedFeatureConfig(
        enable_series_features=True,
        enable_cross_timeframe=True,
        enable_multi_dimensional=True,
        enable_pattern_strength=True,
        enable_temporal_analysis=True,
        enable_pattern_categorization=True,
        enable_market_structure=True
    )
    
    generator = create_advanced_candle_feature_generator(advanced_config=advanced_config)
    
    # Generate features
    print("🔧 Generating advanced candle features...")
    features = generator._generate_feature(data)
    
    print(f"✅ Generated advanced features for {len(features)} samples")
    print(f"📊 Feature statistics:")
    print(f"   - Mean: {features.mean():.4f}")
    print(f"   - Std: {features.std():.4f}")
    print(f"   - Min: {features.min():.4f}")
    print(f"   - Max: {features.max():.4f}")
    print(f"   - Non-zero: {(features != 0).sum()}")
    
    print("\n🎉 Advanced Candle Features test completed successfully!")
    return generator, features


if __name__ == "__main__":
    test_advanced_candle_features()