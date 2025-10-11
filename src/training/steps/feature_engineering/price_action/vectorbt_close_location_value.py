"""
VectorBT Close Location Value (CLV) Feature Engineering

This module implements an enhanced Close Location Value feature using VectorBT
for superior performance and comprehensive price action analysis.

Features:
- VectorBT-optimized CLV calculations
- Multiple CLV measures and classifications
- Advanced volume analysis integration
- Price action control and pressure indicators
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
class VectorBTCLVConfig:
    """Enhanced configuration for VectorBT Close Location Value feature."""
    
    # Feature settings
    window: int = 8  # Rolling window for CLV smoothing
    min_periods: int = 1  # Minimum periods for rolling calculation
    additional_windows: List[int] = None  # Additional windows for analysis
    
    # Thresholds for interpretation
    positive_threshold: float = 0.2   # Sustained positive CLV = bullish
    negative_threshold: float = -0.2  # Sustained negative CLV = bearish
    volatility_threshold: float = 0.5  # Avoid when CLV fluctuates rapidly
    extreme_threshold: float = 0.5    # Extreme CLV threshold
    
    # Volume analysis settings
    enable_volume_analysis: bool = True
    volume_window: int = 20
    volume_weighted_clv: bool = True
    
    # Price action control settings
    enable_control_analysis: bool = True
    control_window: int = 10
    
    # CLV momentum settings
    enable_momentum_analysis: bool = True
    momentum_window: int = 5
    
    # Output settings
    include_raw_clv: bool = True
    include_rolling_clv: bool = True
    include_clv_volatility: bool = True
    include_clv_grade: bool = True
    include_clv_class: bool = True
    include_volume_clv: bool = True
    include_control_analysis: bool = True
    include_momentum_analysis: bool = True
    include_clv_regime: bool = True
    
    def __post_init__(self):
        if self.additional_windows is None:
            self.additional_windows = [4, 6, 10, 12, 16]


class VectorBTCloseLocationValueFeature:
    """
    Enhanced Close Location Value Feature Engineering using VectorBT.
    
    Provides comprehensive price action analysis with multiple CLV measures,
    volume integration, and advanced control analysis capabilities.
    """
    
    def __init__(self, config: Optional[VectorBTCLVConfig] = None):
        """Initialize VectorBT Close Location Value feature."""
        self.config = config or VectorBTCLVConfig()
        self.indicators = VectorBTTechnicalIndicators()
        
        tprint_info("📊 VectorBT Close Location Value feature initialized")
        tprint_info(f"   → Window: {self.config.window} bars")
        tprint_info(f"   → Additional windows: {self.config.additional_windows}")
        tprint_info(f"   → Positive threshold: {self.config.positive_threshold}")
        tprint_info(f"   → Negative threshold: {self.config.negative_threshold}")
        tprint_info(f"   → Volume analysis: {self.config.enable_volume_analysis}")
        tprint_info(f"   → Control analysis: {self.config.enable_control_analysis}")
    
    def calculate_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate comprehensive Close Location Value features using VectorBT.
        
        Args:
            data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary of feature Series
        """
        tprint_info("📊 Calculating VectorBT Close Location Value features")
        
        # Validate input data
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        features = {}
        
        try:
            # Calculate basic CLV features
            basic_features = self._calculate_basic_clv_features(data)
            features.update(basic_features)
            
            # Calculate additional window CLV features
            additional_features = self._calculate_additional_clv_features(data)
            features.update(additional_features)
            
            # Calculate volume CLV features
            if self.config.include_volume_clv and self.config.enable_volume_analysis:
                volume_features = self._calculate_volume_clv_features(data, features)
                features.update(volume_features)
            
            # Calculate control analysis features
            if self.config.include_control_analysis and self.config.enable_control_analysis:
                control_features = self._calculate_control_analysis_features(data, features)
                features.update(control_features)
            
            # Calculate momentum analysis features
            if self.config.include_momentum_analysis and self.config.enable_momentum_analysis:
                momentum_features = self._calculate_momentum_analysis_features(features)
                features.update(momentum_features)
            
            # Calculate CLV regime features
            if self.config.include_clv_regime:
                regime_features = self._calculate_clv_regime_features(features)
                features.update(regime_features)
            
            tprint_success("✅ VectorBT Close Location Value features calculated successfully")
            return features
            
        except Exception as e:
            tprint_error(f"❌ Error calculating VectorBT Close Location Value features: {e}")
            raise
    
    def _calculate_basic_clv_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate basic CLV features."""
        features = {}
        
        try:
            # Calculate raw CLV
            price_range = data['high'] - data['low']
            price_range = price_range.replace(0, np.nan)  # Avoid division by zero
            
            clv_numerator = 2 * data['close'] - data['high'] - data['low']
            raw_clv = clv_numerator / price_range
            raw_clv = raw_clv.fillna(0)  # Set to 0 for zero-range bars
            raw_clv = raw_clv.replace([np.inf, -np.inf], 0)  # Replace infinite values
            
            if self.config.include_raw_clv:
                features['clv_raw'] = raw_clv
                tprint_info(f"   → Raw CLV: mean={raw_clv.mean():.3f}, std={raw_clv.std():.3f}")
            
            # Calculate rolling mean CLV
            if self.config.include_rolling_clv:
                rolling_clv = raw_clv.rolling(
                    window=self.config.window,
                    min_periods=self.config.min_periods
                ).mean()
                features['clv_rolling'] = rolling_clv
                tprint_info(f"   → Rolling CLV: mean={rolling_clv.mean():.3f}, std={rolling_clv.std():.3f}")
            
            # Calculate CLV volatility
            if self.config.include_clv_volatility:
                clv_volatility = raw_clv.rolling(
                    window=self.config.window,
                    min_periods=self.config.min_periods
                ).std()
                features['clv_volatility'] = clv_volatility
                tprint_info(f"   → CLV volatility: mean={clv_volatility.mean():.3f}, std={clv_volatility.std():.3f}")
            
            # Calculate CLV grade (0.0-1.0)
            if self.config.include_clv_grade:
                clv_grade = self._calculate_clv_grade(raw_clv, rolling_clv, clv_volatility)
                features['clv_grade'] = clv_grade
                tprint_info(f"   → CLV grade: mean={clv_grade.mean():.3f}, std={clv_grade.std():.3f}")
            
            # Calculate CLV classification
            if self.config.include_clv_class:
                clv_class = self._calculate_clv_classification(rolling_clv, clv_volatility)
                features['clv_class'] = clv_class
                
                # Count classifications
                class_counts = clv_class.value_counts()
                tprint_info(f"   → CLV classification: {dict(class_counts)}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating basic CLV features: {e}")
        
        return features
    
    def _calculate_clv_grade(
        self, 
        raw_clv: pd.Series, 
        rolling_clv: pd.Series, 
        clv_volatility: pd.Series
    ) -> pd.Series:
        """Calculate CLV grade based on directional strength and stability."""
        # Grade based on directional strength and stability
        clv_strength = np.abs(rolling_clv)
        clv_stability = 1.0 - np.clip(clv_volatility / self.config.volatility_threshold, 0.0, 1.0)
        clv_grade = (clv_strength * clv_stability).clip(0.0, 1.0)
        return clv_grade
    
    def _calculate_clv_classification(
        self, 
        rolling_clv: pd.Series, 
        clv_volatility: pd.Series
    ) -> pd.Series:
        """Calculate CLV classification."""
        clv_class = pd.Series('neutral', index=rolling_clv.index)
        
        # Basic classification
        clv_class[rolling_clv >= self.config.positive_threshold] = 'bullish'
        clv_class[rolling_clv <= self.config.negative_threshold] = 'bearish'
        
        # Extreme classification
        clv_class[rolling_clv >= self.config.extreme_threshold] = 'extreme_bullish'
        clv_class[rolling_clv <= -self.config.extreme_threshold] = 'extreme_bearish'
        
        # Mark as unstable if volatility is too high
        if clv_volatility is not None:
            clv_class[clv_volatility > self.config.volatility_threshold] = 'unstable'
        
        return clv_class
    
    def _calculate_additional_clv_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate CLV features for additional windows."""
        features = {}
        
        try:
            # Calculate raw CLV for additional windows
            price_range = data['high'] - data['low']
            price_range = price_range.replace(0, np.nan)
            
            clv_numerator = 2 * data['close'] - data['high'] - data['low']
            raw_clv = clv_numerator / price_range
            raw_clv = raw_clv.fillna(0).replace([np.inf, -np.inf], 0)
            
            for window in self.config.additional_windows:
                # Rolling CLV for additional window
                rolling_clv = raw_clv.rolling(
                    window=window,
                    min_periods=1
                ).mean()
                features[f'clv_rolling_{window}'] = rolling_clv
                
                # CLV volatility for additional window
                clv_volatility = raw_clv.rolling(window).std()
                features[f'clv_volatility_{window}'] = clv_volatility
                
                # CLV grade for additional window
                clv_grade = self._calculate_clv_grade(raw_clv, rolling_clv, clv_volatility)
                features[f'clv_grade_{window}'] = clv_grade
                
                # CLV momentum for additional window
                clv_momentum = rolling_clv.diff()
                features[f'clv_momentum_{window}'] = clv_momentum
                
                # CLV trend for additional window
                clv_trend = rolling_clv.rolling(5).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
                features[f'clv_trend_{window}'] = clv_trend
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating additional CLV features: {e}")
        
        return features
    
    def _calculate_volume_clv_features(
        self, 
        data: pd.DataFrame, 
        features: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Calculate volume-integrated CLV features."""
        volume_features = {}
        
        try:
            if 'volume' not in data.columns:
                tprint_warning("⚠️ Volume data not available for volume CLV analysis")
                return volume_features
            
            # Get CLV data
            raw_clv = features.get('clv_raw')
            rolling_clv = features.get('clv_rolling')
            
            if raw_clv is not None:
                # Volume-weighted CLV
                if self.config.volume_weighted_clv:
                    volume_weighted_clv = (raw_clv * data['volume']).rolling(
                        self.config.volume_window
                    ).sum() / data['volume'].rolling(self.config.volume_window).sum()
                    volume_features['clv_volume_weighted'] = volume_weighted_clv.fillna(0)
                
                # Volume CLV correlation
                volume_clv_corr = data['volume'].rolling(20).corr(raw_clv)
                volume_features['clv_volume_correlation'] = volume_clv_corr.fillna(0)
                
                # Volume CLV ratio
                volume_clv_ratio = raw_clv / (data['volume'] / data['volume'].rolling(20).mean())
                volume_features['clv_volume_ratio'] = volume_clv_ratio.fillna(0)
                
                # Volume CLV strength
                volume_clv_strength = raw_clv * (data['volume'] / data['volume'].rolling(20).mean())
                volume_features['clv_volume_strength'] = volume_clv_strength.fillna(0)
            
            if rolling_clv is not None:
                # Rolling volume-weighted CLV
                rolling_volume_weighted = (rolling_clv * data['volume']).rolling(
                    self.config.volume_window
                ).sum() / data['volume'].rolling(self.config.volume_window).sum()
                volume_features['clv_rolling_volume_weighted'] = rolling_volume_weighted.fillna(0)
            
            # Volume CLV classification
            if 'clv_volume_weighted' in volume_features:
                volume_clv_class = pd.Series('normal', index=data.index)
                volume_clv_class[volume_features['clv_volume_weighted'] > 
                               volume_features['clv_volume_weighted'].rolling(50).quantile(0.8)] = 'high_volume_clv'
                volume_clv_class[volume_features['clv_volume_weighted'] < 
                               volume_features['clv_volume_weighted'].rolling(50).quantile(0.2)] = 'low_volume_clv'
                volume_features['clv_volume_class'] = volume_clv_class
            
            tprint_info("   → Volume CLV features calculated")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating volume CLV features: {e}")
        
        return volume_features
    
    def _calculate_control_analysis_features(
        self, 
        data: pd.DataFrame, 
        features: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Calculate price action control analysis features."""
        control_features = {}
        
        try:
            # Get CLV data
            raw_clv = features.get('clv_raw')
            rolling_clv = features.get('clv_rolling')
            
            if raw_clv is not None:
                # CLV control strength
                clv_control_strength = np.abs(raw_clv)
                control_features['clv_control_strength'] = clv_control_strength
                
                # CLV control persistence
                clv_control_persistence = self._calculate_control_persistence(raw_clv)
                control_features['clv_control_persistence'] = clv_control_persistence
                
                # CLV control dominance
                clv_control_dominance = self._calculate_control_dominance(raw_clv)
                control_features['clv_control_dominance'] = clv_control_dominance
                
                # CLV control momentum
                clv_control_momentum = raw_clv.diff()
                control_features['clv_control_momentum'] = clv_control_momentum
                
                # CLV control acceleration
                clv_control_acceleration = clv_control_momentum.diff()
                control_features['clv_control_acceleration'] = clv_control_acceleration
            
            if rolling_clv is not None:
                # Rolling CLV control strength
                rolling_control_strength = np.abs(rolling_clv)
                control_features['clv_rolling_control_strength'] = rolling_control_strength
                
                # Rolling CLV control persistence
                rolling_control_persistence = self._calculate_control_persistence(rolling_clv)
                control_features['clv_rolling_control_persistence'] = rolling_control_persistence
            
            # Price action control classification
            if 'clv_control_strength' in control_features:
                control_class = pd.Series('neutral', index=data.index)
                control_class[control_features['clv_control_strength'] > 0.3] = 'strong_control'
                control_class[control_features['clv_control_strength'] > 0.5] = 'extreme_control'
                control_class[control_features['clv_control_strength'] < 0.1] = 'weak_control'
                control_features['clv_control_class'] = control_class
            
            tprint_info("   → Control analysis features calculated")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating control analysis features: {e}")
        
        return control_features
    
    def _calculate_control_persistence(self, clv_series: pd.Series) -> pd.Series:
        """Calculate control persistence (consecutive bars with same control direction)."""
        persistence = pd.Series(0, index=clv_series.index)
        
        current_control = None
        current_count = 0
        
        for i, clv in enumerate(clv_series):
            if np.isnan(clv):
                persistence.iloc[i] = 0
                current_count = 0
                current_control = None
            else:
                # Determine control direction
                if clv > 0.1:
                    control = 'bullish'
                elif clv < -0.1:
                    control = 'bearish'
                else:
                    control = 'neutral'
                
                if control == current_control:
                    current_count += 1
                    persistence.iloc[i] = current_count
                else:
                    current_control = control
                    current_count = 1
                    persistence.iloc[i] = current_count
        
        return persistence
    
    def _calculate_control_dominance(self, clv_series: pd.Series) -> pd.Series:
        """Calculate control dominance over a rolling window."""
        dominance = pd.Series(0.0, index=clv_series.index)
        
        for i in range(len(clv_series)):
            window_data = clv_series.iloc[max(0, i-self.config.control_window+1):i+1]
            if len(window_data) > 0:
                bullish_count = (window_data > 0.1).sum()
                bearish_count = (window_data < -0.1).sum()
                total_count = bullish_count + bearish_count
                
                if total_count > 0:
                    dominance.iloc[i] = (bullish_count - bearish_count) / total_count
                else:
                    dominance.iloc[i] = 0.0
        
        return dominance
    
    def _calculate_momentum_analysis_features(self, features: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Calculate CLV momentum analysis features."""
        momentum_features = {}
        
        try:
            # Get CLV data
            raw_clv = features.get('clv_raw')
            rolling_clv = features.get('clv_rolling')
            
            if raw_clv is not None:
                # CLV momentum
                clv_momentum = raw_clv.diff()
                momentum_features['clv_momentum'] = clv_momentum
                
                # CLV acceleration
                clv_acceleration = clv_momentum.diff()
                momentum_features['clv_acceleration'] = clv_acceleration
                
                # CLV momentum strength
                clv_momentum_strength = clv_momentum.abs()
                momentum_features['clv_momentum_strength'] = clv_momentum_strength
                
                # CLV momentum trend
                clv_momentum_trend = clv_momentum.rolling(self.config.momentum_window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
                momentum_features['clv_momentum_trend'] = clv_momentum_trend
                
                # CLV momentum classification
                momentum_class = pd.Series('stable', index=raw_clv.index)
                momentum_class[clv_momentum > clv_momentum.rolling(20).std()] = 'increasing'
                momentum_class[clv_momentum < -clv_momentum.rolling(20).std()] = 'decreasing'
                momentum_features['clv_momentum_class'] = momentum_class
            
            if rolling_clv is not None:
                # Rolling CLV momentum
                rolling_momentum = rolling_clv.diff()
                momentum_features['clv_rolling_momentum'] = rolling_momentum
                
                # Rolling CLV acceleration
                rolling_acceleration = rolling_momentum.diff()
                momentum_features['clv_rolling_acceleration'] = rolling_acceleration
            
            tprint_info("   → Momentum analysis features calculated")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating momentum analysis features: {e}")
        
        return momentum_features
    
    def _calculate_clv_regime_features(self, features: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Calculate CLV regime features."""
        regime_features = {}
        
        try:
            # Get CLV data
            rolling_clv = features.get('clv_rolling')
            clv_grade = features.get('clv_grade')
            
            if rolling_clv is not None:
                # CLV regime based on rolling CLV
                clv_regime = pd.Series('neutral', index=rolling_clv.index)
                
                # Bullish regime
                bullish_mask = rolling_clv >= self.config.positive_threshold
                clv_regime[bullish_mask] = 'bullish'
                
                # Bearish regime
                bearish_mask = rolling_clv <= self.config.negative_threshold
                clv_regime[bearish_mask] = 'bearish'
                
                # Extreme regimes
                extreme_bullish_mask = rolling_clv >= self.config.extreme_threshold
                clv_regime[extreme_bullish_mask] = 'extreme_bullish'
                
                extreme_bearish_mask = rolling_clv <= -self.config.extreme_threshold
                clv_regime[extreme_bearish_mask] = 'extreme_bearish'
                
                regime_features['clv_regime'] = clv_regime
                
                # Regime persistence
                regime_changes = (clv_regime != clv_regime.shift(1)).astype(int)
                regime_features['clv_regime_persistence'] = regime_changes.rolling(20).sum()
                
                # Regime strength
                regime_strength = np.abs(rolling_clv).groupby(clv_regime).transform('mean')
                regime_features['clv_regime_strength'] = regime_strength
            
            # CLV regime score (composite)
            if clv_grade is not None:
                regime_score = pd.Series(0.0, index=clv_grade.index)
                regime_score += clv_grade * 0.5
                
                if rolling_clv is not None:
                    regime_score += np.clip(np.abs(rolling_clv), 0.0, 1.0) * 0.5
                
                regime_features['clv_regime_score'] = regime_score
                
                # Regime classification based on score
                regime_class = pd.Series('normal', index=regime_score.index)
                regime_class[regime_score > 0.7] = 'high_clv_regime'
                regime_class[regime_score < 0.3] = 'low_clv_regime'
                regime_class[regime_score > 0.9] = 'extreme_clv_regime'
                regime_features['clv_regime_class'] = regime_class
            
            tprint_info("   → CLV regime features calculated")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating CLV regime features: {e}")
        
        return regime_features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this class produces."""
        features = []
        
        if self.config.include_raw_clv:
            features.append('clv_raw')
        if self.config.include_rolling_clv:
            features.append('clv_rolling')
        if self.config.include_clv_volatility:
            features.append('clv_volatility')
        if self.config.include_clv_grade:
            features.append('clv_grade')
        if self.config.include_clv_class:
            features.append('clv_class')
        
        # Additional windows
        for window in self.config.additional_windows:
            features.extend([
                f'clv_rolling_{window}',
                f'clv_volatility_{window}',
                f'clv_grade_{window}',
                f'clv_momentum_{window}',
                f'clv_trend_{window}'
            ])
        
        if self.config.include_volume_clv:
            features.extend([
                'clv_volume_weighted', 'clv_volume_correlation',
                'clv_volume_ratio', 'clv_volume_strength',
                'clv_rolling_volume_weighted', 'clv_volume_class'
            ])
        
        if self.config.include_control_analysis:
            features.extend([
                'clv_control_strength', 'clv_control_persistence',
                'clv_control_dominance', 'clv_control_momentum',
                'clv_control_acceleration', 'clv_rolling_control_strength',
                'clv_rolling_control_persistence', 'clv_control_class'
            ])
        
        if self.config.include_momentum_analysis:
            features.extend([
                'clv_momentum', 'clv_acceleration',
                'clv_momentum_strength', 'clv_momentum_trend',
                'clv_momentum_class', 'clv_rolling_momentum',
                'clv_rolling_acceleration'
            ])
        
        if self.config.include_clv_regime:
            features.extend([
                'clv_regime', 'clv_regime_persistence',
                'clv_regime_strength', 'clv_regime_score',
                'clv_regime_class'
            ])
        
        return features
    
    def get_feature_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about the features."""
        return {
            'clv_raw': {
                'description': 'Raw Close-Location Value ((2*close-high-low) / (high-low))',
                'range': '[-1, 1]',
                'interpretation': 'Positive = buying pressure, Negative = selling pressure'
            },
            'clv_rolling': {
                'description': f'Rolling mean CLV over {self.config.window} bars',
                'range': '[-1, 1]',
                'interpretation': 'Smoothed CLV for trend analysis'
            },
            'clv_grade': {
                'description': 'Normalized CLV grade (0.0-1.0)',
                'range': '[0, 1]',
                'interpretation': '1.0 = strong directional CLV with low volatility'
            },
            'clv_class': {
                'description': 'CLV classification',
                'values': ['extreme_bearish', 'bearish', 'neutral', 'bullish', 'extreme_bullish', 'unstable'],
                'interpretation': 'Categorical classification based on thresholds'
            },
            'clv_regime': {
                'description': 'CLV regime classification',
                'values': ['extreme_bearish', 'bearish', 'neutral', 'bullish', 'extreme_bullish'],
                'interpretation': 'Current CLV regime state'
            }
        }


class VectorBTCloseLocationValueGenerator(VectorBTFeatureGenerator):
    """
    VectorBT-enhanced Close Location Value feature generator.
    
    Provides comprehensive price action analysis with VectorBT optimization,
    parameter tuning, and advanced feature generation capabilities.
    """
    
    def __init__(self, lookback: int = 8, **kwargs):
        """
        Initialize the VectorBT Close Location Value feature generator.
        
        Args:
            lookback: Number of periods for rolling calculation
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
            name="vectorbt_close_location_value",
            category=FeatureCategory.PRICE_ACTION,
            description="VectorBT-enhanced Close Location Value with comprehensive price action analysis",
            required_columns=['open', 'high', 'low', 'close'],
            optional_columns=['volume'],
            default_lookback=lookback,
            min_lookback=1,
            max_lookback=20,
            parameters={
                'window': lookback,
                'additional_windows': kwargs.get('additional_windows', [4, 6, 10, 12, 16]),
                'positive_threshold': kwargs.get('positive_threshold', 0.2),
                'negative_threshold': kwargs.get('negative_threshold', -0.2),
                'volatility_threshold': kwargs.get('volatility_threshold', 0.5),
                'extreme_threshold': kwargs.get('extreme_threshold', 0.5),
                'enable_volume_analysis': kwargs.get('enable_volume_analysis', True),
                'volume_window': kwargs.get('volume_window', 20),
                'volume_weighted_clv': kwargs.get('volume_weighted_clv', True),
                'enable_control_analysis': kwargs.get('enable_control_analysis', True),
                'control_window': kwargs.get('control_window', 10),
                'enable_momentum_analysis': kwargs.get('enable_momentum_analysis', True),
                'momentum_window': kwargs.get('momentum_window', 5),
                'include_raw_clv': kwargs.get('include_raw_clv', True),
                'include_rolling_clv': kwargs.get('include_rolling_clv', True),
                'include_clv_volatility': kwargs.get('include_clv_volatility', True),
                'include_clv_grade': kwargs.get('include_clv_grade', True),
                'include_clv_class': kwargs.get('include_clv_class', True),
                'include_volume_clv': kwargs.get('include_volume_clv', True),
                'include_control_analysis': kwargs.get('include_control_analysis', True),
                'include_momentum_analysis': kwargs.get('include_momentum_analysis', True),
                'include_clv_regime': kwargs.get('include_clv_regime', True)
            },
            matrix_optimized=True,
            gpu_accelerated=False,
            enable_feature_selection=True
        )
        
        super().__init__(config, vectorbt_config)
        
        # Initialize the feature engine
        feature_config = VectorBTCLVConfig(
            window=lookback,
            additional_windows=kwargs.get('additional_windows', [4, 6, 10, 12, 16]),
            positive_threshold=kwargs.get('positive_threshold', 0.2),
            negative_threshold=kwargs.get('negative_threshold', -0.2),
            volatility_threshold=kwargs.get('volatility_threshold', 0.5),
            extreme_threshold=kwargs.get('extreme_threshold', 0.5),
            enable_volume_analysis=kwargs.get('enable_volume_analysis', True),
            volume_window=kwargs.get('volume_window', 20),
            volume_weighted_clv=kwargs.get('volume_weighted_clv', True),
            enable_control_analysis=kwargs.get('enable_control_analysis', True),
            control_window=kwargs.get('control_window', 10),
            enable_momentum_analysis=kwargs.get('enable_momentum_analysis', True),
            momentum_window=kwargs.get('momentum_window', 5),
            include_raw_clv=kwargs.get('include_raw_clv', True),
            include_rolling_clv=kwargs.get('include_rolling_clv', True),
            include_clv_volatility=kwargs.get('include_clv_volatility', True),
            include_clv_grade=kwargs.get('include_clv_grade', True),
            include_clv_class=kwargs.get('include_clv_class', True),
            include_volume_clv=kwargs.get('include_volume_clv', True),
            include_control_analysis=kwargs.get('include_control_analysis', True),
            include_momentum_analysis=kwargs.get('include_momentum_analysis', True),
            include_clv_regime=kwargs.get('include_clv_regime', True)
        )
        self.feature_engine = VectorBTCloseLocationValueFeature(feature_config)
    
    def generate_vectorbt_features(
        self, 
        data: pd.DataFrame, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate VectorBT Close Location Value features.
        
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
            'window': [4, 6, 8, 10, 12],
            'positive_threshold': [0.1, 0.2, 0.3, 0.4],
            'negative_threshold': [-0.4, -0.3, -0.2, -0.1],
            'volatility_threshold': [0.3, 0.5, 0.7, 0.9],
            'extreme_threshold': [0.4, 0.5, 0.6, 0.7],
            'volume_window': [10, 20, 30, 40],
            'control_window': [5, 10, 15, 20],
            'momentum_window': [3, 5, 7, 10]
        }
        
        return super().optimize_parameters(data, param_ranges, target_metric)


# Convenience function for external usage
def calculate_vectorbt_clv_features(
    data: pd.DataFrame,
    config: Optional[VectorBTCLVConfig] = None,
    **kwargs
) -> Dict[str, pd.Series]:
    """
    Calculate VectorBT Close Location Value features.
    
    Args:
        data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
        config: Optional configuration
        **kwargs: Additional parameters
        
    Returns:
        Dictionary of feature Series
    """
    feature_engine = VectorBTCloseLocationValueFeature(config)
    return feature_engine.calculate_features(data)