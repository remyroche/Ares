"""
import warnings
SR Feature Extractor - Comprehensive Support/Resistance Feature Engineering

This module provides comprehensive SR (Support/Resistance) feature extraction with:
- Integration with pre-optimized parameters from sr_clustering/parameter_optimization_engine.py
- Advanced SR level detection and quality assessment
- Distance-based features to SR levels
- Bounce signal detection
- SR strength calculation
- Regime-aware SR features
- Memory-efficient processing for large datasets
- Hardware optimization support

The module is designed to be called from the main feature engineering pipeline
and uses pre-optimized parameters for maximum performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import time
from pathlib import Path
import json
from dataclasses import dataclass
from contextlib import nullcontext

# Import math validation utilities
from .math_validation import safe_divide, validate_positive

# Import optimization engine
try:
    from src.utils.sr_clustering.parameter_optimization_engine import (
        ParameterOptimizationEngine, ParameterOptimizationConfig, 
        get_parameter_optimization_engine
    )
    OPTIMIZATION_ENGINE_AVAILABLE = True
except ImportError:
    OPTIMIZATION_ENGINE_AVAILABLE = False
    ParameterOptimizationEngine = None
    ParameterOptimizationConfig = None
    get_parameter_optimization_engine = None

# Import SR detection modules
try:
    from src.tactician.sr_levels.sr_breakout_predictor_enhanced import SRBreakoutPredictor
    SR_DETECTION_AVAILABLE = True
except ImportError:
    SR_DETECTION_AVAILABLE = False
    SRBreakoutPredictor = None

# Import logging utilities
try:
    from src.utils.comprehensive_function_logger import (

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
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
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:
    
    cp = None
        log_step_functions, log_important_calls, log_all_calls
    )
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    def log_step_functions(func):
        return func
    def log_important_calls(func):
        return func
    def log_all_calls(func):
        return func

logger = logging.getLogger(__name__)

@dataclass
class SRFeatureConfig:
    """Configuration for SR feature extraction."""
    # Feature extraction settings
    enable_basic_sr_features: bool = True
    enable_advanced_sr_features: bool = True
    enable_sr_bounce_signals: bool = True
    enable_sr_strength_calculation: bool = True
    enable_regime_aware_sr: bool = True
    
    # SR level detection settings
    use_pre_optimized_parameters: bool = True
    sr_detection_window: int = 20
    min_touches_required: int = 3
    touch_tolerance: float = 0.002
    min_bounce_strength: float = 0.001
    volume_threshold_multiplier: float = 1.5
    
    # Feature calculation windows
    pivot_window: int = 20
    swing_window: int = 20
    strength_window: int = 20
    distance_calculation_window: int = 50
    
    # Memory and performance settings
    chunk_size: int = 10000
    enable_parallel_processing: bool = True
    max_parallel_workers: int = None
    
    # Quality thresholds
    min_sr_quality_score: float = 0.3
    max_sr_levels_per_type: int = 10
    
    # SR levels requirement
    require_sr_levels: bool = True  # SR levels are required for proper feature extraction

class SRFeatureExtractor:
    """
    Comprehensive SR Feature Extractor with optimization integration.
    
    This class extracts support/resistance features using pre-optimized parameters
    from the parameter optimization engine for maximum performance and accuracy.
    """
    
    def __init__(self, config: Optional[SRFeatureConfig] = None):
        """Initialize SR feature extractor."""
        self.config = config or SRFeatureConfig()
        self.logger = logger.getChild('SRFeatureExtractor')
        
        # Initialize optimization engine
        self.optimization_engine = None
        self.optimized_parameters = None
        
        if self.config.use_pre_optimized_parameters and OPTIMIZATION_ENGINE_AVAILABLE:
            self._initialize_optimization_engine()
        
        # Initialize SR detection
        self.sr_predictor = None
        if SR_DETECTION_AVAILABLE:
            self._initialize_sr_detection()
        
        self.logger.info("🚀 SR Feature Extractor initialized")
        self.logger.info(f"   Pre-optimized parameters: {self.config.use_pre_optimized_parameters}")
        self.logger.info(f"   Advanced features: {self.config.enable_advanced_sr_features}")
        self.logger.info(f"   Bounce signals: {self.config.enable_sr_bounce_signals}")
    
    def _initialize_optimization_engine(self):
        """Initialize parameter optimization engine."""
        try:
            opt_config = ParameterOptimizationConfig(
                optimization_method='adaptive_grid_search',
                enable_hardware_optimization=True,
                enable_parallel_processing=self.config.enable_parallel_processing,
                max_parallel_workers=self.config.max_parallel_workers
            )
            self.optimization_engine = get_parameter_optimization_engine(opt_config)
            self.logger.info("✅ Parameter optimization engine initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize optimization engine: {e}")
            self.optimization_engine = None
    
    def _initialize_sr_detection(self):
        """Initialize SR detection components."""
        try:
            sr_config = {
                'touch_tolerance': self.config.touch_tolerance,
                'min_touches_required': self.config.min_touches_required,
                'min_bounce_strength': self.config.min_bounce_strength,
                'volume_threshold_multiplier': self.config.volume_threshold_multiplier
            }
            self.sr_predictor = SRBreakoutPredictor(sr_config)
            self.logger.info("✅ SR detection components initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize SR detection: {e}")
            self.sr_predictor = None
    
    @log_step_functions
    def extract_sr_features(self, data: pd.DataFrame, 
                          sr_levels: Optional[Dict[str, Any]] = None,
                          regime_labels: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Extract comprehensive SR features from market data.
        
        Args:
            data: Market data with OHLCV columns
            sr_levels: Pre-computed SR levels (optional)
            regime_labels: Regime labels for regime-aware features (optional)
            
        Returns:
            DataFrame with SR features
        """
        try:
            self.logger.info(f"🔧 Extracting SR features from {len(data)} rows")
            start_time = time.time()
            
            # Initialize features DataFrame
            sr_features = pd.DataFrame(index=data.index)
            
            # Step 1: Basic SR features (pivot points, swing levels)
            if self.config.enable_basic_sr_features:
                self.logger.info("📊 Extracting basic SR features...")
                basic_features = self._extract_basic_sr_features(data)
                sr_features = pd.concat([sr_features, basic_features], axis=1)
                self.logger.info(f"   Added {basic_features.shape[1]} basic SR features")
            
            # Step 2: Advanced SR features (distance to levels, quality metrics)
            if self.config.enable_advanced_sr_features:
                self.logger.info("🎯 Extracting advanced SR features...")
                advanced_features = self._extract_advanced_sr_features(data, sr_levels)
                sr_features = pd.concat([sr_features, advanced_features], axis=1)
                self.logger.info(f"   Added {advanced_features.shape[1]} advanced SR features")
            
            # Step 3: SR bounce signals
            if self.config.enable_sr_bounce_signals:
                self.logger.info("🔄 Extracting SR bounce signals...")
                bounce_features = self._extract_sr_bounce_signals(data, sr_levels)
                sr_features = pd.concat([sr_features, bounce_features], axis=1)
                self.logger.info(f"   Added {bounce_features.shape[1]} bounce signal features")
            
            # Step 4: SR strength calculation
            if self.config.enable_sr_strength_calculation:
                self.logger.info("💪 Calculating SR strength features...")
                strength_features = self._extract_sr_strength_features(data)
                sr_features = pd.concat([sr_features, strength_features], axis=1)
                self.logger.info(f"   Added {strength_features.shape[1]} strength features")
            
            # Step 5: Regime-aware SR features
            if self.config.enable_regime_aware_sr and regime_labels is not None:
                self.logger.info("🏛️ Extracting regime-aware SR features...")
                regime_features = self._extract_regime_aware_sr_features(data, sr_levels, regime_labels)
                sr_features = pd.concat([sr_features, regime_features], axis=1)
                self.logger.info(f"   Added {regime_features.shape[1]} regime-aware features")
            
            # Clean and validate features
            sr_features = self._clean_sr_features(sr_features)
            
            processing_time = time.time() - start_time
            self.logger.info(f"✅ SR feature extraction completed in {processing_time:.2f}s")
            self.logger.info(f"   Total SR features: {sr_features.shape[1]}")
            
            return sr_features
            
        except Exception as e:
            self.logger.error(f"❌ SR feature extraction failed: {e}")
            raise
    
    def _extract_basic_sr_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract basic SR features (pivot points, swing levels)."""
        features = pd.DataFrame(index=data.index)
        
        # Pivot points
        features['pivot_point'] = (data['high'] + data['low'] + data['close']) / 3
        features['support_1'] = 2 * features['pivot_point'] - data['high']
        features['resistance_1'] = 2 * features['pivot_point'] - data['low']
        features['support_2'] = features['pivot_point'] - (data['high'] - data['low'])
        features['resistance_2'] = features['pivot_point'] + (data['high'] - data['low'])
        
        # Distance to pivot levels
        features['distance_to_support_1'] = safe_divide(
            data['close'] - features['support_1'], data['close'], default=0.0
        )
        features['distance_to_resistance_1'] = safe_divide(
            features['resistance_1'] - data['close'], data['close'], default=0.0
        )
        features['distance_to_support_2'] = safe_divide(
            data['close'] - features['support_2'], data['close'], default=0.0
        )
        features['distance_to_resistance_2'] = safe_divide(
            features['resistance_2'] - data['close'], data['close'], default=0.0
        )
        
        # Swing highs and lows
        for window in [10, 20, 50]:
            features[f'swing_high_{window}'] = data['high'].rolling(window, center=True).max()
            features[f'swing_low_{window}'] = data['low'].rolling(window, center=True).min()
            features[f'distance_to_swing_high_{window}'] = safe_divide(
                features[f'swing_high_{window}'] - data['close'], data['close'], default=0.0
            )
            features[f'distance_to_swing_low_{window}'] = safe_divide(
                data['close'] - features[f'swing_low_{window}'], data['close'], default=0.0
            )
        
        return features
    
    def _extract_advanced_sr_features(self, data: pd.DataFrame, 
                                    sr_levels: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Extract advanced SR features using detected levels."""
        features = pd.DataFrame(index=data.index)
        
        if sr_levels is None:
            # Fast fail - SR levels are required
            self.logger.error("❌ SR levels are required for advanced feature extraction")
            raise ValueError("SR levels are required for advanced SR feature extraction")
        
        if sr_levels:
            support_levels = sr_levels.get('support_levels', [])
            resistance_levels = sr_levels.get('resistance_levels', [])
            
            # Distance to nearest SR levels
            if support_levels:
                features['distance_to_support'] = self._calculate_distance_to_levels(
                    data['close'], support_levels
                )
                features['near_support'] = (features['distance_to_support'] < 0.005).astype(int)
                features['support_count'] = self._count_nearby_levels(
                    data['close'], support_levels, threshold=0.01
                )
            
            if resistance_levels:
                features['distance_to_resistance'] = self._calculate_distance_to_levels(
                    data['close'], resistance_levels
                )
                features['near_resistance'] = (features['distance_to_resistance'] < 0.005).astype(int)
                features['resistance_count'] = self._count_nearby_levels(
                    data['close'], resistance_levels, threshold=0.01
                )
            
            # SR zone features
            features['sr_zone_width'] = self._calculate_sr_zone_width(
                data['close'], support_levels, resistance_levels
            )
            features['in_sr_zone'] = (features['sr_zone_width'] > 0).astype(int)
            
            # Quality-weighted features
            if 'quality_scores' in sr_levels:
                features['avg_support_quality'] = self._calculate_avg_quality(
                    data['close'], support_levels, sr_levels.get('quality_scores', {})
                )
                features['avg_resistance_quality'] = self._calculate_avg_quality(
                    data['close'], resistance_levels, sr_levels.get('quality_scores', {})
                )
        else:
            # Add placeholder features
            features['distance_to_support'] = 1.0
            features['distance_to_resistance'] = 1.0
            features['near_support'] = 0
            features['near_resistance'] = 0
            features['support_count'] = 0
            features['resistance_count'] = 0
            features['sr_zone_width'] = 0.0
            features['in_sr_zone'] = 0
            features['avg_support_quality'] = 0.0
            features['avg_resistance_quality'] = 0.0
        
        return features
    
    def _extract_sr_bounce_signals(self, data: pd.DataFrame, 
                                 sr_levels: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Extract SR bounce signals."""
        features = pd.DataFrame(index=data.index)
        
        if sr_levels is None:
            # Fast fail - SR levels are required
            self.logger.error("❌ SR levels are required for bounce signal extraction")
            raise ValueError("SR levels are required for SR bounce signal extraction")
        
        if sr_levels:
            support_levels = sr_levels.get('support_levels', [])
            resistance_levels = sr_levels.get('resistance_levels', [])
            
            # Support bounce signals
            if support_levels:
                features['support_bounce_signal'] = self._calculate_sr_bounce_signal(
                    data, support_levels, 'support'
                )
                features['support_bounce_strength'] = self._calculate_bounce_strength(
                    data, support_levels, 'support'
                )
            
            # Resistance bounce signals
            if resistance_levels:
                features['resistance_bounce_signal'] = self._calculate_sr_bounce_signal(
                    data, resistance_levels, 'resistance'
                )
                features['resistance_bounce_strength'] = self._calculate_bounce_strength(
                    data, resistance_levels, 'resistance'
                )
            
            # Combined bounce signals
            features['sr_bounce_signal'] = (
                features.get('support_bounce_signal', 0) + 
                features.get('resistance_bounce_signal', 0)
            )
            features['sr_bounce_strength'] = (
                features.get('support_bounce_strength', 0) + 
                features.get('resistance_bounce_strength', 0)
            ) / 2
        else:
            # Add placeholder features
            features['support_bounce_signal'] = 0.0
            features['resistance_bounce_signal'] = 0.0
            features['support_bounce_strength'] = 0.0
            features['resistance_bounce_strength'] = 0.0
            features['sr_bounce_signal'] = 0.0
            features['sr_bounce_strength'] = 0.0
        
        return features
    
    def _extract_sr_strength_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract SR strength features."""
        features = pd.DataFrame(index=data.index)
        
        # Calculate SR strength using rolling windows
        for window in [10, 20, 50]:
            features[f'sr_strength_{window}'] = self._calculate_sr_strength(data, window)
            
            # SR strength momentum
            features[f'sr_strength_momentum_{window}'] = features[f'sr_strength_{window}'].diff()
            
            # SR strength volatility
            features[f'sr_strength_volatility_{window}'] = features[f'sr_strength_{window}'].rolling(5).std()
        
        # Overall SR strength
        features['sr_strength_overall'] = features[[
            col for col in features.columns if col.startswith('sr_strength_') and not col.endswith('_momentum') and not col.endswith('_volatility')
        ]].mean(axis=1)
        
        return features
    
    def _extract_regime_aware_sr_features(self, data: pd.DataFrame, 
                                        sr_levels: Optional[Dict[str, Any]],
                                        regime_labels: pd.Series) -> pd.DataFrame:
        """Extract regime-aware SR features."""
        features = pd.DataFrame(index=data.index)
        
        # Regime-specific SR proximity
        unique_regimes = regime_labels.unique()
        for regime in unique_regimes:
            if pd.notna(regime):
                regime_mask = regime_labels == regime
                regime_data = data[regime_mask]
                
                if len(regime_data) > 0:
                    # Calculate regime-specific SR metrics
                    features[f'regime_{regime}_sr_proximity'] = 0.0
                    features.loc[regime_mask, f'regime_{regime}_sr_proximity'] = self._calculate_regime_sr_proximity(
                        regime_data, sr_levels
                    )
        
        # Regime transition SR features
        features['regime_changed'] = (regime_labels != regime_labels.shift(1)).astype(int)
        features['time_in_regime'] = regime_labels.groupby(
            (regime_labels != regime_labels.shift()).cumsum()
        ).cumcount()
        
        return features
    
    def _calculate_distance_to_levels(self, prices: pd.Series, levels: List[float]) -> pd.Series:
        """Calculate normalized distance to nearest SR level."""
        if not levels or len(prices) == 0:
            return pd.Series([1.0] * len(prices), index=prices.index)
        
        # Convert levels to numpy array
        level_prices = np.array(levels, dtype=float)
        
        # Vectorized distance calculation
        distances = []
        for price in prices:
            if pd.notna(price):
                # Calculate distances to all levels
                level_distances = np.abs(level_prices - price) / price
                # Take minimum distance
                min_distance = np.min(level_distances)
                distances.append(min_distance)
            else:
                distances.append(1.0)
        
        return pd.Series(distances, index=prices.index)
    
    def _count_nearby_levels(self, prices: pd.Series, levels: List[float], 
                           threshold: float = 0.01) -> pd.Series:
        """Count number of SR levels within threshold distance."""
        if not levels:
            return pd.Series([0] * len(prices), index=prices.index)
        
        level_prices = np.array(levels, dtype=float)
        counts = []
        
        for price in prices:
            if pd.notna(price):
                # Calculate distances to all levels
                level_distances = np.abs(level_prices - price) / price
                # Count levels within threshold
                nearby_count = np.sum(level_distances <= threshold)
                counts.append(nearby_count)
            else:
                counts.append(0)
        
        return pd.Series(counts, index=prices.index)
    
    def _calculate_sr_zone_width(self, prices: pd.Series, support_levels: List[float], 
                               resistance_levels: List[float]) -> pd.Series:
        """Calculate width of SR zone around current price."""
        if not support_levels and not resistance_levels:
            return pd.Series([0.0] * len(prices), index=prices.index)
        
        widths = []
        for price in prices:
            if pd.notna(price):
                # Find nearest support and resistance
                nearest_support = None
                nearest_resistance = None
                
                if support_levels:
                    support_distances = np.abs(np.array(support_levels) - price)
                    nearest_support = support_levels[np.argmin(support_distances)]
                
                if resistance_levels:
                    resistance_distances = np.abs(np.array(resistance_levels) - price)
                    nearest_resistance = resistance_levels[np.argmin(resistance_distances)]
                
                # Calculate zone width
                if nearest_support and nearest_resistance:
                    zone_width = (nearest_resistance - nearest_support) / price
                elif nearest_support:
                    zone_width = (price - nearest_support) / price
                elif nearest_resistance:
                    zone_width = (nearest_resistance - price) / price
                else:
                    zone_width = 0.0
                
                widths.append(zone_width)
            else:
                widths.append(0.0)
        
        return pd.Series(widths, index=prices.index)
    
    def _calculate_avg_quality(self, prices: pd.Series, levels: List[float], 
                             quality_scores: Dict[str, float]) -> pd.Series:
        """Calculate average quality of nearby SR levels."""
        if not levels or not quality_scores:
            return pd.Series([0.0] * len(prices), index=prices.index)
        
        avg_qualities = []
        for price in prices:
            if pd.notna(price):
                # Find levels within 2% of current price
                nearby_levels = []
                for level in levels:
                    distance = abs(level - price) / price
                    if distance <= 0.02:  # Within 2%
                        level_key = f"level_{level:.6f}"
                        if level_key in quality_scores:
                            nearby_levels.append(quality_scores[level_key])
                
                if nearby_levels:
                    avg_quality = np.mean(nearby_levels)
                else:
                    avg_quality = 0.0
                
                avg_qualities.append(avg_quality)
            else:
                avg_qualities.append(0.0)
        
        return pd.Series(avg_qualities, index=prices.index)
    
    def _calculate_sr_bounce_signal(self, data: pd.DataFrame, levels: List[float], 
                                  level_type: str) -> pd.Series:
        """Calculate SR bounce signals based on price action near levels."""
        if not levels:
            return pd.Series([0.0] * len(data), index=data.index)
        
        signals = []
        level_prices = np.array(levels, dtype=float)
        
        for i in range(len(data)):
            current_price = data['close'].iloc[i]
            current_high = data['high'].iloc[i]
            current_low = data['low'].iloc[i]
            
            if pd.notna(current_price):
                # Find nearest level
                distances = np.abs(level_prices - current_price) / current_price
                nearest_idx = np.argmin(distances)
                nearest_level = level_prices[nearest_idx]
                distance_to_level = distances[nearest_idx]
                
                # Check if price is near level (within 1%)
                if distance_to_level <= 0.01:
                    if level_type == 'support':
                        # Check for bounce off support (price touched support and moved up)
                        if current_low <= nearest_level * 1.001 and current_price > nearest_level:
                            signals.append(1.0)
                        else:
                            signals.append(0.0)
                    elif level_type == 'resistance':
                        # Check for bounce off resistance (price touched resistance and moved down)
                        if current_high >= nearest_level * 0.999 and current_price < nearest_level:
                            signals.append(1.0)
                        else:
                            signals.append(0.0)
                    else:
                        signals.append(0.0)
                else:
                    signals.append(0.0)
            else:
                signals.append(0.0)
        
        return pd.Series(signals, index=data.index)
    
    def _calculate_bounce_strength(self, data: pd.DataFrame, levels: List[float], 
                                 level_type: str) -> pd.Series:
        """Calculate strength of SR bounces."""
        if not levels:
            return pd.Series([0.0] * len(data), index=data.index)
        
        strengths = []
        level_prices = np.array(levels, dtype=float)
        
        for i in range(len(data)):
            current_price = data['close'].iloc[i]
            prev_price = data['close'].iloc[i-1] if i > 0 else current_price
            
            if pd.notna(current_price) and pd.notna(prev_price):
                # Find nearest level
                distances = np.abs(level_prices - current_price) / current_price
                nearest_idx = np.argmin(distances)
                nearest_level = level_prices[nearest_idx]
                distance_to_level = distances[nearest_idx]
                
                # Check if price is near level
                if distance_to_level <= 0.01:
                    if level_type == 'support':
                        # Calculate bounce strength from support
                        if current_price > nearest_level:
                            bounce_strength = (current_price - nearest_level) / nearest_level
                        else:
                            bounce_strength = 0.0
                    elif level_type == 'resistance':
                        # Calculate bounce strength from resistance
                        if current_price < nearest_level:
                            bounce_strength = (nearest_level - current_price) / nearest_level
                        else:
                            bounce_strength = 0.0
                    else:
                        bounce_strength = 0.0
                    
                    strengths.append(min(bounce_strength, 0.1))  # Cap at 10%
                else:
                    strengths.append(0.0)
            else:
                strengths.append(0.0)
        
        return pd.Series(strengths, index=data.index)
    
    def _calculate_sr_strength(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate SR strength indicator."""
        high_swing = data['high'].rolling(window, center=True).max()
        low_swing = data['low'].rolling(window, center=True).min()
        current_price = data['close']
        
        high_strength = safe_divide(high_swing - current_price, high_swing, default=0.0)
        low_strength = safe_divide(current_price - low_swing, low_swing, default=0.0)
        sr_strength = (high_strength + low_strength) / 2
        
        return sr_strength
    
    def _calculate_regime_sr_proximity(self, regime_data: pd.DataFrame, 
                                     sr_levels: Optional[Dict[str, Any]]) -> pd.Series:
        """Calculate SR proximity for specific regime."""
        if sr_levels is None:
            return pd.Series([0.0] * len(regime_data), index=regime_data.index)
        
        support_levels = sr_levels.get('support_levels', [])
        resistance_levels = sr_levels.get('resistance_levels', [])
        
        # Calculate average distance to SR levels
        support_distances = self._calculate_distance_to_levels(regime_data['close'], support_levels)
        resistance_distances = self._calculate_distance_to_levels(regime_data['close'], resistance_levels)
        
        # Combine distances (lower is closer, higher proximity)
        avg_distance = (support_distances + resistance_distances) / 2
        proximity = 1.0 - np.clip(avg_distance, 0, 1)  # Convert distance to proximity
        
        return proximity
    
    def _detect_fallback_sr_levels(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Fast fail when no SR levels provided - no automatic detection."""
        self.logger.error("❌ No SR levels provided and fallback detection is disabled")
        self.logger.error("   SR levels are required for proper feature extraction")
        self.logger.error("   Please provide SR levels or enable fallback detection in configuration")
        raise ValueError("SR levels are required for feature extraction. No fallback detection available.")
    
    def _count_touches(self, data: pd.DataFrame, level: float, level_type: str) -> int:
        """Count number of times price touched a level."""
        tolerance = self.config.touch_tolerance
        touches = 0
        
        for i in range(len(data)):
            high = data['high'].iloc[i]
            low = data['low'].iloc[i]
            
            if level_type == 'support':
                if low <= level * (1 + tolerance) and high >= level * (1 - tolerance):
                    touches += 1
            elif level_type == 'resistance':
                if high >= level * (1 - tolerance) and low <= level * (1 + tolerance):
                    touches += 1
        
        return touches
    
    def _clean_sr_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate SR features."""
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill and then fill remaining NaN with 0
        features = features.ffill().fillna(0)
        
        # Remove duplicate columns
        features = features.loc[:, ~features.columns.duplicated()]
        
        # Clip extreme values
        for col in features.columns:
            if features[col].dtype in ['float64', 'float32']:
                features[col] = features[col].clip(-10, 10)
        
        return features
    
    def get_optimized_parameters(self) -> Optional[Dict[str, Any]]:
        """Get pre-optimized parameters from optimization engine."""
        if self.optimized_parameters is None and self.optimization_engine:
            try:
                # This would typically load from a saved optimization result
                # For now, return default optimized parameters
                self.optimized_parameters = {
                    'touch_tolerance': 0.002,
                    'min_touches_required': 3,
                    'min_bounce_strength': 0.001,
                    'volume_threshold_multiplier': 1.5,
                    'success_rate_multiplier': 1.0,
                    'bounce_strength_multiplier': 1.0,
                    'volume_confirmation_multiplier': 1.0,
                    'time_persistence_multiplier': 1.0,
                    'touch_frequency_multiplier': 1.0
                }
            except Exception as e:
                self.logger.warning(f"Failed to get optimized parameters: {e}")
                self.optimized_parameters = None
        
        return self.optimized_parameters
    
    def save_optimized_parameters(self, parameters: Dict[str, Any], 
                                file_path: Union[str, Path]) -> None:
        """Save optimized parameters to file."""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(parameters, f, indent=2)
            
            self.logger.info(f"💾 Saved optimized parameters to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save optimized parameters: {e}")
    
    def load_optimized_parameters(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Load optimized parameters from file."""
        try:
            file_path = Path(file_path)
            if file_path.exists():
                with open(file_path, 'r') as f:
                    parameters = json.load(f)
                
                self.optimized_parameters = parameters
                self.logger.info(f"📂 Loaded optimized parameters from {file_path}")
                return parameters
            else:
                self.logger.warning(f"Optimized parameters file not found: {file_path}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to load optimized parameters: {e}")
            return None

def get_sr_feature_extractor(config: Optional[SRFeatureConfig] = None) -> SRFeatureExtractor:
    """Get an SR feature extractor instance."""
    return SRFeatureExtractor(config)

# Convenience function for quick SR feature extraction
def extract_sr_features(data: pd.DataFrame, 
                       sr_levels: Optional[Dict[str, Any]] = None,
                       regime_labels: Optional[pd.Series] = None,
                       config: Optional[SRFeatureConfig] = None) -> pd.DataFrame:
    """
    Quick function to extract SR features from market data.
    
    Args:
        data: Market data with OHLCV columns
        sr_levels: Pre-computed SR levels (optional)
        regime_labels: Regime labels for regime-aware features (optional)
        config: SR feature configuration (optional)
        
    Returns:
        DataFrame with SR features
    """
    extractor = get_sr_feature_extractor(config)
    return extractor.extract_sr_features(data, sr_levels, regime_labels)
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
