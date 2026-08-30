"""
VectorBT Trend Coherence Feature Engineering

This module implements an enhanced Trend Coherence feature using VectorBT
for superior performance and comprehensive trend analysis.

Features:
- VectorBT-optimized trend indicators (ADX, Ichimoku, Parabolic SAR)
- Multiple timeframe trend analysis
- Advanced trend strength and direction detection
- Trend regime classification and persistence
- Parameter optimization capabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass
import time

# Import VectorBT base classes
from src.training.steps.feature_engineering.vectorbt_base import (
    VectorBTFeatureGenerator, VectorBTConfig, VectorBTTechnicalIndicators
)
from src.feature_generation.core.feature_generator import FeatureCategory, FeatureConfig, FeatureResult
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success


@dataclass
class VectorBTTrendCoherenceConfig:
    """Enhanced configuration for VectorBT Trend Coherence feature."""
    
    # Direction consistency settings
    direction_window: int = 8   # Window for direction consistency check
    min_periods: int = 1        # Minimum periods for rolling calculation
    
    # EMA settings
    ema_period: int = 12        # EMA period for slope calculation
    additional_ema_periods: List[int] = None  # Additional EMA periods
    
    # ADX settings
    adx_period: int = 14        # ADX period for trend strength
    adx_threshold: float = 25.0 # ADX threshold for trend confirmation
    
    # Ichimoku settings
    use_ichimoku: bool = True   # Enable Ichimoku analysis
    ichimoku_conversion: int = 9
    ichimoku_base: int = 26
    ichimoku_span_b: int = 52
    
    # Parabolic SAR settings
    use_psar: bool = True       # Enable Parabolic SAR
    psar_af: float = 0.02       # Parabolic SAR acceleration factor
    psar_max_af: float = 0.2    # Parabolic SAR maximum acceleration factor
    
    # Thresholds for interpretation
    min_direction_consistency: float = 0.6  # 60% of bars in same direction
    min_slope_threshold: float = 0.001      # Minimum slope for trend continuity
    strong_trend_threshold: float = 0.8     # Strong trend threshold
    
    # Output settings
    include_direction_consistency: bool = True
    include_ema_slope: bool = True
    include_adx_strength: bool = True
    include_ichimoku_signals: bool = True
    include_psar_signals: bool = True
    include_trend_coherence_grade: bool = True
    include_trend_class: bool = True
    include_trend_regime: bool = True
    include_trend_persistence: bool = True
    
    def __post_init__(self):
        if self.additional_ema_periods is None:
            self.additional_ema_periods = [5, 8, 21, 34, 55]


class VectorBTTrendCoherenceFeature:
    """
    Enhanced Trend Coherence Feature Engineering using VectorBT.
    
    Provides comprehensive trend analysis with multiple indicators,
    regime detection, and advanced trend classification capabilities.
    """
    
    def __init__(self, config: Optional[VectorBTTrendCoherenceConfig] = None):
        """Initialize VectorBT Trend Coherence feature."""
        self.config = config or VectorBTTrendCoherenceConfig()
        self.indicators = VectorBTTechnicalIndicators()
        
        tprint_info("📊 VectorBT Trend Coherence feature initialized")
        tprint_info(f"   → Direction window: {self.config.direction_window} bars")
        tprint_info(f"   → EMA period: {self.config.ema_period}")
        tprint_info(f"   → ADX period: {self.config.adx_period}, threshold: {self.config.adx_threshold}")
        tprint_info(f"   → Ichimoku: {self.config.use_ichimoku}, PSAR: {self.config.use_psar}")
    
    def calculate_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate comprehensive Trend Coherence features using VectorBT.
        
        Args:
            data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary of feature Series
        """
        tprint_info("📊 Calculating VectorBT Trend Coherence features")
        
        # Validate input data
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        features = {}
        
        try:
            # Get comprehensive trend indicators
            trend_indicators = self.indicators.get_trend_indicators(
                data, 
                windows=[self.config.ema_period] + self.config.additional_ema_periods
            )
            
            # Calculate direction consistency
            if self.config.include_direction_consistency:
                direction_features = self._calculate_direction_consistency(data)
                features.update(direction_features)
            
            # Calculate EMA slope features
            if self.config.include_ema_slope:
                ema_features = self._calculate_ema_slope_features(data, trend_indicators)
                features.update(ema_features)
            
            # Calculate ADX strength features
            if self.config.include_adx_strength:
                adx_features = self._calculate_adx_strength_features(data, trend_indicators)
                features.update(adx_features)
            
            # Calculate Ichimoku signals
            if self.config.include_ichimoku_signals and self.config.use_ichimoku:
                ichimoku_features = self._calculate_ichimoku_features(data, trend_indicators)
                features.update(ichimoku_features)
            
            # Calculate Parabolic SAR signals
            if self.config.include_psar_signals and self.config.use_psar:
                psar_features = self._calculate_psar_features(data, trend_indicators)
                features.update(psar_features)
            
            # Calculate trend coherence grade
            if self.config.include_trend_coherence_grade:
                coherence_features = self._calculate_trend_coherence_grade(features)
                features.update(coherence_features)
            
            # Calculate trend classification
            if self.config.include_trend_class:
                classification_features = self._calculate_trend_classification(features, data)
                features.update(classification_features)
            
            # Calculate trend regime
            if self.config.include_trend_regime:
                regime_features = self._calculate_trend_regime_features(features, data)
                features.update(regime_features)
            
            # Calculate trend persistence
            if self.config.include_trend_persistence:
                persistence_features = self._calculate_trend_persistence_features(features)
                features.update(persistence_features)
            
            tprint_success("✅ VectorBT Trend Coherence features calculated successfully")
            return features
            
        except Exception as e:
            tprint_error(f"❌ Error calculating VectorBT Trend Coherence features: {e}")
            raise
    
    def _calculate_direction_consistency(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate direction consistency features."""
        features = {}
        
        try:
            # Basic direction consistency
            close_direction = np.sign(data['close'].diff())
            close_direction_series = pd.Series(close_direction, index=data.index)
            
            # Calculate direction consistency as the absolute value of the mean direction
            direction_consistency = close_direction_series.rolling(
                window=self.config.direction_window,
                min_periods=1
            ).apply(lambda x: np.abs(x.mean()), raw=True)
            
            features['trend_direction_consistency'] = direction_consistency
            
            # Direction persistence (consecutive bars in same direction)
            direction_persistence = self._calculate_direction_persistence(close_direction_series)
            features['trend_direction_persistence'] = direction_persistence
            
            # Direction strength (magnitude of price changes)
            price_changes = data['close'].pct_change().abs()
            direction_strength = price_changes.rolling(
                window=self.config.direction_window,
                min_periods=1
            ).mean()
            features['trend_direction_strength'] = direction_strength
            
            tprint_info(f"   → Direction consistency: mean={direction_consistency.mean():.3f}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating direction consistency: {e}")
        
        return features
    
    def _calculate_direction_persistence(self, direction_series: pd.Series) -> pd.Series:
        """Calculate direction persistence (consecutive bars in same direction)."""
        persistence = pd.Series(0, index=direction_series.index)
        
        current_direction = None
        current_count = 0
        
        for i, direction in enumerate(direction_series):
            if np.isnan(direction):
                persistence.iloc[i] = 0
                current_count = 0
                current_direction = None
            elif direction == current_direction:
                current_count += 1
                persistence.iloc[i] = current_count
            else:
                current_direction = direction
                current_count = 1
                persistence.iloc[i] = current_count
        
        return persistence
    
    def _calculate_ema_slope_features(
        self, 
        data: pd.DataFrame, 
        trend_indicators: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Calculate EMA slope features."""
        features = {}
        
        try:
            # Primary EMA slope
            ema = trend_indicators.get(f'ema_{self.config.ema_period}')
            if ema is not None:
                ema_slope = ema.diff()
                features['trend_ema_slope'] = ema_slope
                
                # EMA slope momentum
                ema_slope_momentum = ema_slope.diff()
                features['trend_ema_slope_momentum'] = ema_slope_momentum
                
                # EMA slope strength (absolute value)
                ema_slope_strength = ema_slope.abs()
                features['trend_ema_slope_strength'] = ema_slope_strength
                
                # EMA slope trend (trend of the slope)
                ema_slope_trend = ema_slope.rolling(5).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
                features['trend_ema_slope_trend'] = ema_slope_trend
            
            # Additional EMA periods analysis
            for period in self.config.additional_ema_periods:
                ema_period = trend_indicators.get(f'ema_{period}')
                if ema_period is not None:
                    ema_slope_period = ema_period.diff()
                    features[f'trend_ema_slope_{period}'] = ema_slope_period
                    
                    # EMA convergence/divergence
                    if ema is not None:
                        ema_convergence = ema - ema_period
                        features[f'trend_ema_convergence_{period}'] = ema_convergence
            
            tprint_info(f"   → EMA slope: mean={ema_slope.mean():.3f}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating EMA slope features: {e}")
        
        return features
    
    def _calculate_adx_strength_features(
        self, 
        data: pd.DataFrame, 
        trend_indicators: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Calculate ADX strength features."""
        features = {}
        
        try:
            adx = trend_indicators.get('adx')
            adx_plus = trend_indicators.get('adx_plus')
            adx_minus = trend_indicators.get('adx_minus')
            
            if adx is not None:
                features['trend_adx_strength'] = adx
                
                # ADX trend strength classification
                adx_class = pd.Series('weak', index=adx.index)
                adx_class[adx >= self.config.adx_threshold] = 'strong'
                adx_class[adx >= self.config.adx_threshold * 1.5] = 'very_strong'
                features['trend_adx_class'] = adx_class
                
                # ADX momentum
                adx_momentum = adx.diff()
                features['trend_adx_momentum'] = adx_momentum
                
                # ADX trend (trend of ADX)
                adx_trend = adx.rolling(10).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
                features['trend_adx_trend'] = adx_trend
            
            if adx_plus is not None and adx_minus is not None:
                # ADX directional bias
                adx_bias = adx_plus - adx_minus
                features['trend_adx_bias'] = adx_bias
                
                # ADX dominance
                adx_dominance = adx_plus / (adx_plus + adx_minus)
                adx_dominance = adx_dominance.fillna(0.5)
                features['trend_adx_dominance'] = adx_dominance
                
                # ADX crossover signals
                adx_crossover = (adx_plus > adx_minus).astype(int)
                features['trend_adx_crossover'] = adx_crossover
            
            tprint_info(f"   → ADX strength: mean={adx.mean():.3f}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating ADX strength features: {e}")
        
        return features
    
    def _calculate_ichimoku_features(
        self, 
        data: pd.DataFrame, 
        trend_indicators: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Calculate Ichimoku features."""
        features = {}
        
        try:
            ichimoku_conversion = trend_indicators.get('ichimoku_conversion')
            ichimoku_base = trend_indicators.get('ichimoku_base')
            ichimoku_span_a = trend_indicators.get('ichimoku_span_a')
            ichimoku_span_b = trend_indicators.get('ichimoku_span_b')
            ichimoku_signal = trend_indicators.get('ichimoku_signal')
            
            if ichimoku_conversion is not None and ichimoku_base is not None:
                # Ichimoku conversion line vs base line
                ichimoku_conversion_signal = (ichimoku_conversion > ichimoku_base).astype(int)
                features['trend_ichimoku_conversion_signal'] = ichimoku_conversion_signal
                
                # Ichimoku conversion line slope
                ichimoku_conversion_slope = ichimoku_conversion.diff()
                features['trend_ichimoku_conversion_slope'] = ichimoku_conversion_slope
                
                # Ichimoku base line slope
                ichimoku_base_slope = ichimoku_base.diff()
                features['trend_ichimoku_base_slope'] = ichimoku_base_slope
            
            if ichimoku_span_a is not None and ichimoku_span_b is not None:
                # Ichimoku cloud analysis
                ichimoku_cloud_upper = np.maximum(ichimoku_span_a, ichimoku_span_b)
                ichimoku_cloud_lower = np.minimum(ichimoku_span_a, ichimoku_span_b)
                
                # Price vs cloud
                price_above_cloud = (data['close'] > ichimoku_cloud_upper).astype(int)
                price_below_cloud = (data['close'] < ichimoku_cloud_lower).astype(int)
                price_in_cloud = ((data['close'] >= ichimoku_cloud_lower) & 
                                (data['close'] <= ichimoku_cloud_upper)).astype(int)
                
                features['trend_ichimoku_price_above_cloud'] = price_above_cloud
                features['trend_ichimoku_price_below_cloud'] = price_below_cloud
                features['trend_ichimoku_price_in_cloud'] = price_in_cloud
                
                # Cloud thickness
                ichimoku_cloud_thickness = ichimoku_cloud_upper - ichimoku_cloud_lower
                features['trend_ichimoku_cloud_thickness'] = ichimoku_cloud_thickness
            
            if ichimoku_signal is not None:
                features['trend_ichimoku_signal'] = ichimoku_signal
            
            tprint_info("   → Ichimoku features calculated")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating Ichimoku features: {e}")
        
        return features
    
    def _calculate_psar_features(
        self, 
        data: pd.DataFrame, 
        trend_indicators: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Calculate Parabolic SAR features."""
        features = {}
        
        try:
            psar = trend_indicators.get('psar')
            psar_signal = trend_indicators.get('psar_signal')
            
            if psar is not None:
                # PSAR vs price
                psar_bullish = (data['close'] > psar).astype(int)
                features['trend_psar_bullish'] = psar_bullish
                
                # PSAR distance from price
                psar_distance = (data['close'] - psar) / data['close']
                features['trend_psar_distance'] = psar_distance
                
                # PSAR slope
                psar_slope = psar.diff()
                features['trend_psar_slope'] = psar_slope
                
                # PSAR acceleration (slope of slope)
                psar_acceleration = psar_slope.diff()
                features['trend_psar_acceleration'] = psar_acceleration
            
            if psar_signal is not None:
                features['trend_psar_signal'] = psar_signal
            
            tprint_info("   → Parabolic SAR features calculated")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating Parabolic SAR features: {e}")
        
        return features
    
    def _calculate_trend_coherence_grade(self, features: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Calculate trend coherence grade."""
        coherence_features = {}
        
        try:
            # Get key features for coherence calculation
            direction_consistency = features.get('trend_direction_consistency')
            ema_slope = features.get('trend_ema_slope')
            adx_strength = features.get('trend_adx_strength')
            
            if direction_consistency is not None and ema_slope is not None:
                # Convert to grade based on direction consistency and slope strength
                direction_grade = np.clip(direction_consistency, 0.0, 1.0)
                slope_grade = np.clip(ema_slope / self.config.min_slope_threshold, 0.0, 1.0)
                
                # Basic coherence grade
                basic_coherence_grade = (direction_grade * slope_grade).clip(0.0, 1.0)
                coherence_features['trend_coherence_grade'] = basic_coherence_grade
                
                # Enhanced coherence grade with ADX
                if adx_strength is not None:
                    adx_grade = np.clip(adx_strength / 50.0, 0.0, 1.0)  # Normalize ADX to 0-1
                    enhanced_coherence_grade = (direction_grade * slope_grade * adx_grade).clip(0.0, 1.0)
                    coherence_features['trend_coherence_grade_enhanced'] = enhanced_coherence_grade
                else:
                    coherence_features['trend_coherence_grade_enhanced'] = basic_coherence_grade
                
                # Coherence strength (how strong the coherence is)
                coherence_strength = basic_coherence_grade * direction_consistency
                coherence_features['trend_coherence_strength'] = coherence_strength
                
                tprint_info(f"   → Coherence grade: mean={basic_coherence_grade.mean():.3f}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating trend coherence grade: {e}")
        
        return coherence_features
    
    def _calculate_trend_classification(
        self, 
        features: Dict[str, pd.Series], 
        data: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """Calculate trend classification."""
        classification_features = {}
        
        try:
            # Get key features
            direction_consistency = features.get('trend_direction_consistency')
            ema_slope = features.get('trend_ema_slope')
            adx_strength = features.get('trend_adx_strength')
            coherence_grade = features.get('trend_coherence_grade')
            
            # Initialize trend class
            trend_class = pd.Series('incoherent', index=data.index)
            
            if direction_consistency is not None and ema_slope is not None:
                # Check direction consistency
                direction_consistent = direction_consistency >= self.config.min_direction_consistency
                slope_positive = ema_slope >= self.config.min_slope_threshold
                slope_negative = ema_slope <= -self.config.min_slope_threshold
                
                # Basic classification
                trend_class[direction_consistent & slope_positive] = 'coherent_uptrend'
                trend_class[direction_consistent & slope_negative] = 'coherent_downtrend'
                trend_class[direction_consistent & (np.abs(ema_slope) < self.config.min_slope_threshold)] = 'coherent_sideways'
                trend_class[~direction_consistent] = 'incoherent'
                
                # Enhanced classification with ADX
                if adx_strength is not None:
                    strong_trend = adx_strength >= self.config.adx_threshold
                    very_strong_trend = adx_strength >= self.config.adx_threshold * 1.5
                    
                    # Upgrade to strong trends
                    trend_class[trend_class == 'coherent_uptrend' & strong_trend] = 'strong_uptrend'
                    trend_class[trend_class == 'coherent_downtrend' & strong_trend] = 'strong_downtrend'
                    trend_class[trend_class == 'coherent_uptrend' & very_strong_trend] = 'very_strong_uptrend'
                    trend_class[trend_class == 'coherent_downtrend' & very_strong_trend] = 'very_strong_downtrend'
                
                # Enhanced classification with coherence grade
                if coherence_grade is not None:
                    high_coherence = coherence_grade >= self.config.strong_trend_threshold
                    trend_class[trend_class == 'coherent_uptrend' & high_coherence] = 'high_coherence_uptrend'
                    trend_class[trend_class == 'coherent_downtrend' & high_coherence] = 'high_coherence_downtrend'
            
            classification_features['trend_class'] = trend_class
            
            # Trend strength score
            trend_strength = pd.Series(0.0, index=data.index)
            if direction_consistency is not None:
                trend_strength += direction_consistency * 0.4
            if coherence_grade is not None:
                trend_strength += coherence_grade * 0.4
            if adx_strength is not None:
                trend_strength += np.clip(adx_strength / 50.0, 0.0, 1.0) * 0.2
            
            classification_features['trend_strength_score'] = trend_strength
            
            # Count classifications
            class_counts = trend_class.value_counts()
            tprint_info(f"   → Trend classification: {dict(class_counts)}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating trend classification: {e}")
        
        return classification_features
    
    def _calculate_trend_regime_features(
        self, 
        features: Dict[str, pd.Series], 
        data: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """Calculate trend regime features."""
        regime_features = {}
        
        try:
            # Get key features
            trend_class = features.get('trend_class')
            trend_strength = features.get('trend_strength_score')
            adx_strength = features.get('trend_adx_strength')
            
            if trend_class is not None:
                # Trend regime based on trend class
                trend_regime = pd.Series('unknown', index=data.index)
                
                # Uptrend regime
                uptrend_mask = trend_class.str.contains('uptrend', na=False)
                trend_regime[uptrend_mask] = 'uptrend'
                
                # Downtrend regime
                downtrend_mask = trend_class.str.contains('downtrend', na=False)
                trend_regime[downtrend_mask] = 'downtrend'
                
                # Sideways regime
                sideways_mask = trend_class.str.contains('sideways', na=False)
                trend_regime[sideways_mask] = 'sideways'
                
                # Incoherent regime
                incoherent_mask = trend_class.str.contains('incoherent', na=False)
                trend_regime[incoherent_mask] = 'incoherent'
                
                regime_features['trend_regime'] = trend_regime
                
                # Regime persistence
                regime_changes = (trend_regime != trend_regime.shift(1)).astype(int)
                regime_features['trend_regime_persistence'] = regime_changes.rolling(20).sum()
                
                # Regime strength
                if trend_strength is not None:
                    regime_strength = trend_strength.groupby(trend_regime).transform('mean')
                    regime_features['trend_regime_strength'] = regime_strength
                
                # Regime duration
                regime_duration = self._calculate_regime_duration(trend_regime)
                regime_features['trend_regime_duration'] = regime_duration
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating trend regime features: {e}")
        
        return regime_features
    
    def _calculate_regime_duration(self, regime_series: pd.Series) -> pd.Series:
        """Calculate duration of current regime."""
        duration = pd.Series(0, index=regime_series.index)
        
        current_regime = None
        current_duration = 0
        
        for i, regime in enumerate(regime_series):
            if regime == current_regime:
                current_duration += 1
                duration.iloc[i] = current_duration
            else:
                current_regime = regime
                current_duration = 1
                duration.iloc[i] = current_duration
        
        return duration
    
    def _calculate_trend_persistence_features(self, features: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Calculate trend persistence features."""
        persistence_features = {}
        
        try:
            # Direction persistence
            direction_persistence = features.get('trend_direction_persistence')
            if direction_persistence is not None:
                persistence_features['trend_direction_persistence_max'] = direction_persistence.rolling(50).max()
                persistence_features['trend_direction_persistence_avg'] = direction_persistence.rolling(20).mean()
            
            # Coherence persistence
            coherence_grade = features.get('trend_coherence_grade')
            if coherence_grade is not None:
                coherence_persistence = (coherence_grade > 0.5).astype(int)
                coherence_persistence_duration = self._calculate_regime_duration(coherence_persistence)
                persistence_features['trend_coherence_persistence'] = coherence_persistence_duration
            
            # Trend strength persistence
            trend_strength = features.get('trend_strength_score')
            if trend_strength is not None:
                strong_trend = (trend_strength > 0.7).astype(int)
                strong_trend_persistence = self._calculate_regime_duration(strong_trend)
                persistence_features['trend_strength_persistence'] = strong_trend_persistence
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating trend persistence features: {e}")
        
        return persistence_features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this class produces."""
        features = []
        
        if self.config.include_direction_consistency:
            features.extend(['trend_direction_consistency', 'trend_direction_persistence', 'trend_direction_strength'])
        if self.config.include_ema_slope:
            features.extend(['trend_ema_slope', 'trend_ema_slope_momentum', 'trend_ema_slope_strength', 'trend_ema_slope_trend'])
            features.extend([f'trend_ema_slope_{period}' for period in self.config.additional_ema_periods])
            features.extend([f'trend_ema_convergence_{period}' for period in self.config.additional_ema_periods])
        if self.config.include_adx_strength:
            features.extend(['trend_adx_strength', 'trend_adx_class', 'trend_adx_momentum', 'trend_adx_trend', 
                           'trend_adx_bias', 'trend_adx_dominance', 'trend_adx_crossover'])
        if self.config.include_ichimoku_signals:
            features.extend(['trend_ichimoku_conversion_signal', 'trend_ichimoku_conversion_slope', 'trend_ichimoku_base_slope',
                           'trend_ichimoku_price_above_cloud', 'trend_ichimoku_price_below_cloud', 'trend_ichimoku_price_in_cloud',
                           'trend_ichimoku_cloud_thickness', 'trend_ichimoku_signal'])
        if self.config.include_psar_signals:
            features.extend(['trend_psar_bullish', 'trend_psar_distance', 'trend_psar_slope', 'trend_psar_acceleration', 'trend_psar_signal'])
        if self.config.include_trend_coherence_grade:
            features.extend(['trend_coherence_grade', 'trend_coherence_grade_enhanced', 'trend_coherence_strength'])
        if self.config.include_trend_class:
            features.extend(['trend_class', 'trend_strength_score'])
        if self.config.include_trend_regime:
            features.extend(['trend_regime', 'trend_regime_persistence', 'trend_regime_strength', 'trend_regime_duration'])
        if self.config.include_trend_persistence:
            features.extend(['trend_direction_persistence_max', 'trend_direction_persistence_avg', 
                           'trend_coherence_persistence', 'trend_strength_persistence'])
        
        return features
    
    def get_feature_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about the features."""
        return {
            'trend_direction_consistency': {
                'description': f'Percentage of bars closing in same direction over {self.config.direction_window} bars',
                'range': '[0, 1]',
                'interpretation': 'Higher values indicate more consistent direction'
            },
            'trend_ema_slope': {
                'description': f'Slope of EMA({self.config.ema_period})',
                'range': '(-inf, inf)',
                'interpretation': 'Positive = uptrend, Negative = downtrend'
            },
            'trend_coherence_grade': {
                'description': 'Combined trend coherence grade (0.0-1.0)',
                'range': '[0, 1]',
                'interpretation': '1.0 = high coherence, 0.0 = low coherence'
            },
            'trend_class': {
                'description': 'Trend classification based on coherence',
                'values': ['coherent_uptrend', 'coherent_downtrend', 'coherent_sideways', 'incoherent', 
                          'strong_uptrend', 'strong_downtrend', 'high_coherence_uptrend', 'high_coherence_downtrend'],
                'interpretation': 'Categorical classification of trend state'
            },
            'trend_regime': {
                'description': 'Trend regime classification',
                'values': ['uptrend', 'downtrend', 'sideways', 'incoherent', 'unknown'],
                'interpretation': 'Current trend regime state'
            }
        }


class VectorBTTrendCoherenceGenerator(VectorBTFeatureGenerator):
    """
    VectorBT-enhanced Trend Coherence feature generator.
    
    Provides comprehensive trend analysis with VectorBT optimization,
    parameter tuning, and advanced feature generation capabilities.
    """
    
    def __init__(self, lookback: int = 8, **kwargs):
        """
        Initialize the VectorBT Trend Coherence feature generator.
        
        Args:
            lookback: Number of periods for direction consistency check
            **kwargs: Additional configuration parameters
        """
        # Create VectorBT configuration
        vectorbt_config = VectorBTConfig(
            enable_optimization=kwargs.get('enable_optimization', True),
            optimization_runs=kwargs.get('optimization_runs', 100),
            enable_caching=kwargs.get('enable_caching', True)
        )
        
        # Create feature configuration
        config = FeatureConfig(
            name="vectorbt_trend_coherence",
            category=FeatureCategory.TREND,
            description="VectorBT-enhanced trend coherence with comprehensive trend analysis",
            required_columns=['open', 'high', 'low', 'close'],
            optional_columns=['volume'],
            default_lookback=lookback,
            min_lookback=1,
            max_lookback=50,
            parameters={
                'direction_window': lookback,
                'ema_period': kwargs.get('ema_period', 12),
                'additional_ema_periods': kwargs.get('additional_ema_periods', [5, 8, 21, 34, 55]),
                'adx_period': kwargs.get('adx_period', 14),
                'adx_threshold': kwargs.get('adx_threshold', 25.0),
                'use_ichimoku': kwargs.get('use_ichimoku', True),
                'ichimoku_conversion': kwargs.get('ichimoku_conversion', 9),
                'ichimoku_base': kwargs.get('ichimoku_base', 26),
                'ichimoku_span_b': kwargs.get('ichimoku_span_b', 52),
                'use_psar': kwargs.get('use_psar', True),
                'psar_af': kwargs.get('psar_af', 0.02),
                'psar_max_af': kwargs.get('psar_max_af', 0.2),
                'min_direction_consistency': kwargs.get('min_direction_consistency', 0.6),
                'min_slope_threshold': kwargs.get('min_slope_threshold', 0.001),
                'strong_trend_threshold': kwargs.get('strong_trend_threshold', 0.8),
                'include_direction_consistency': kwargs.get('include_direction_consistency', True),
                'include_ema_slope': kwargs.get('include_ema_slope', True),
                'include_adx_strength': kwargs.get('include_adx_strength', True),
                'include_ichimoku_signals': kwargs.get('include_ichimoku_signals', True),
                'include_psar_signals': kwargs.get('include_psar_signals', True),
                'include_trend_coherence_grade': kwargs.get('include_trend_coherence_grade', True),
                'include_trend_class': kwargs.get('include_trend_class', True),
                'include_trend_regime': kwargs.get('include_trend_regime', True),
                'include_trend_persistence': kwargs.get('include_trend_persistence', True)
            },
            matrix_optimized=True,
            gpu_accelerated=False,
            enable_feature_selection=True
        )
        
        super().__init__(config, vectorbt_config)
        
        # Initialize the feature engine
        feature_config = VectorBTTrendCoherenceConfig(
            direction_window=lookback,
            ema_period=kwargs.get('ema_period', 12),
            additional_ema_periods=kwargs.get('additional_ema_periods', [5, 8, 21, 34, 55]),
            adx_period=kwargs.get('adx_period', 14),
            adx_threshold=kwargs.get('adx_threshold', 25.0),
            use_ichimoku=kwargs.get('use_ichimoku', True),
            ichimoku_conversion=kwargs.get('ichimoku_conversion', 9),
            ichimoku_base=kwargs.get('ichimoku_base', 26),
            ichimoku_span_b=kwargs.get('ichimoku_span_b', 52),
            use_psar=kwargs.get('use_psar', True),
            psar_af=kwargs.get('psar_af', 0.02),
            psar_max_af=kwargs.get('psar_max_af', 0.2),
            min_direction_consistency=kwargs.get('min_direction_consistency', 0.6),
            min_slope_threshold=kwargs.get('min_slope_threshold', 0.001),
            strong_trend_threshold=kwargs.get('strong_trend_threshold', 0.8),
            include_direction_consistency=kwargs.get('include_direction_consistency', True),
            include_ema_slope=kwargs.get('include_ema_slope', True),
            include_adx_strength=kwargs.get('include_adx_strength', True),
            include_ichimoku_signals=kwargs.get('include_ichimoku_signals', True),
            include_psar_signals=kwargs.get('include_psar_signals', True),
            include_trend_coherence_grade=kwargs.get('include_trend_coherence_grade', True),
            include_trend_class=kwargs.get('include_trend_class', True),
            include_trend_regime=kwargs.get('include_trend_regime', True),
            include_trend_persistence=kwargs.get('include_trend_persistence', True)
        )
        self.feature_engine = VectorBTTrendCoherenceFeature(feature_config)
    
    def generate_vectorbt_features(
        self, 
        data: pd.DataFrame, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate VectorBT Trend Coherence features.
        
        Args:
            data: OHLCV data with required columns
            params: Optional parameters override
            
        Returns:
            Dictionary of generated features
        """
        # Update feature engine configuration if params provided
        if params:
            for key, value in params.items():
                if hasattr(self.feature_engine.config, key):
                    setattr(self.feature_engine.config, key, value)
        
        # Generate features
        return self.feature_engine.calculate_features(data)
    
    def optimize_parameters(
        self, 
        data: pd.DataFrame, 
        target_metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Optimize parameters using VectorBT's optimization capabilities.
        
        Args:
            data: Input data for optimization
            target_metric: Target metric for optimization
            
        Returns:
            Optimized parameters
        """
        # Define parameter ranges for optimization
        param_ranges = {
            'direction_window': [5, 8, 10, 12, 15],
            'ema_period': [8, 12, 16, 20, 24],
            'adx_period': [10, 14, 18, 22],
            'adx_threshold': [20, 25, 30, 35],
            'min_direction_consistency': [0.5, 0.6, 0.7, 0.8],
            'min_slope_threshold': [0.0005, 0.001, 0.002, 0.005]
        }
        
        return super().optimize_parameters(data, param_ranges, target_metric)


# Convenience function for external usage
def calculate_vectorbt_trend_coherence_features(
    data: pd.DataFrame,
    config: Optional[VectorBTTrendCoherenceConfig] = None,
    **kwargs
) -> Dict[str, pd.Series]:
    """
    Calculate VectorBT Trend Coherence features.
    
    Args:
        data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
        config: Optional configuration
        **kwargs: Additional parameters
        
    Returns:
        Dictionary of feature Series
    """
    feature_engine = VectorBTTrendCoherenceFeature(config)
    return feature_engine.calculate_features(data)