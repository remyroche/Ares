"""
VectorBT Bar Efficiency Ratio Feature Engineering

This module implements an enhanced Bar Efficiency Ratio feature using VectorBT
for superior performance and comprehensive price action analysis.

Features:
- VectorBT-optimized price action calculations
- Multiple efficiency measures and classifications
- Advanced candlestick pattern analysis
- Price action momentum and strength indicators
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
class VectorBTBarEfficiencyConfig:
    """Enhanced configuration for VectorBT Bar Efficiency Ratio feature."""
    
    # Feature settings
    window: int = 3  # Rolling window for efficiency (2-4 bars = 30-60 minutes)
    min_periods: int = 1  # Minimum periods for rolling calculation
    additional_windows: List[int] = None  # Additional windows for analysis
    
    # Thresholds for interpretation
    high_efficiency_threshold: float = 0.6  # High efficiency = directional
    low_efficiency_threshold: float = 0.3   # Low efficiency = choppy
    extreme_efficiency_threshold: float = 0.8  # Extreme efficiency
    
    # Candlestick pattern settings
    enable_candlestick_patterns: bool = True
    pattern_window: int = 5  # Window for pattern analysis
    
    # Price action momentum settings
    enable_momentum_analysis: bool = True
    momentum_window: int = 10
    
    # Volume analysis settings
    enable_volume_analysis: bool = True
    volume_window: int = 20
    
    # Output settings
    include_raw_efficiency: bool = True
    include_rolling_efficiency: bool = True
    include_efficiency_grade: bool = True
    include_efficiency_class: bool = True
    include_candlestick_patterns: bool = True
    include_price_action_momentum: bool = True
    include_volume_efficiency: bool = True
    include_efficiency_regime: bool = True
    
    def __post_init__(self):
        if self.additional_windows is None:
            self.additional_windows = [2, 4, 6, 8, 10]


class VectorBTBarEfficiencyRatioFeature:
    """
    Enhanced Bar Efficiency Ratio Feature Engineering using VectorBT.
    
    Provides comprehensive price action analysis with multiple efficiency measures,
    candlestick patterns, and advanced price action classification capabilities.
    """
    
    def __init__(self, config: Optional[VectorBTBarEfficiencyConfig] = None):
        """Initialize VectorBT Bar Efficiency Ratio feature."""
        self.config = config or VectorBTBarEfficiencyConfig()
        self.indicators = VectorBTTechnicalIndicators()
        
        tprint_info("📊 VectorBT Bar Efficiency Ratio feature initialized")
        tprint_info(f"   → Window: {self.config.window} bars")
        tprint_info(f"   → Additional windows: {self.config.additional_windows}")
        tprint_info(f"   → High efficiency threshold: {self.config.high_efficiency_threshold}")
        tprint_info(f"   → Low efficiency threshold: {self.config.low_efficiency_threshold}")
        tprint_info(f"   → Candlestick patterns: {self.config.enable_candlestick_patterns}")
        tprint_info(f"   → Momentum analysis: {self.config.enable_momentum_analysis}")
    
    def calculate_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate comprehensive Bar Efficiency Ratio features using VectorBT.
        
        Args:
            data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary of feature Series
        """
        tprint_info("📊 Calculating VectorBT Bar Efficiency Ratio features")
        
        # Validate input data
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        features = {}
        
        try:
            # Calculate basic efficiency features
            basic_features = self._calculate_basic_efficiency_features(data)
            features.update(basic_features)
            
            # Calculate additional window efficiency features
            additional_features = self._calculate_additional_efficiency_features(data)
            features.update(additional_features)
            
            # Calculate candlestick pattern features
            if self.config.include_candlestick_patterns and self.config.enable_candlestick_patterns:
                pattern_features = self._calculate_candlestick_pattern_features(data)
                features.update(pattern_features)
            
            # Calculate price action momentum features
            if self.config.include_price_action_momentum and self.config.enable_momentum_analysis:
                momentum_features = self._calculate_price_action_momentum_features(data, features)
                features.update(momentum_features)
            
            # Calculate volume efficiency features
            if self.config.include_volume_efficiency and self.config.enable_volume_analysis:
                volume_features = self._calculate_volume_efficiency_features(data, features)
                features.update(volume_features)
            
            # Calculate efficiency regime features
            if self.config.include_efficiency_regime:
                regime_features = self._calculate_efficiency_regime_features(features)
                features.update(regime_features)
            
            tprint_success("✅ VectorBT Bar Efficiency Ratio features calculated successfully")
            return features
            
        except Exception as e:
            tprint_error(f"❌ Error calculating VectorBT Bar Efficiency Ratio features: {e}")
            raise
    
    def _calculate_basic_efficiency_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate basic efficiency features."""
        features = {}
        
        try:
            # Calculate raw efficiency ratio
            price_range = data['high'] - data['low']
            price_range = price_range.replace(0, np.nan)  # Avoid division by zero
            
            raw_efficiency = np.abs(data['close'] - data['open']) / price_range
            raw_efficiency = raw_efficiency.fillna(0)  # Set to 0 for zero-range bars
            raw_efficiency = raw_efficiency.replace([np.inf, -np.inf], 0)  # Replace infinite values
            
            if self.config.include_raw_efficiency:
                features['bar_efficiency_raw'] = raw_efficiency
                tprint_info(f"   → Raw efficiency: mean={raw_efficiency.mean():.3f}, std={raw_efficiency.std():.3f}")
            
            # Calculate rolling mean efficiency
            if self.config.include_rolling_efficiency:
                rolling_efficiency = raw_efficiency.rolling(
                    window=self.config.window,
                    min_periods=self.config.min_periods
                ).mean()
                features['bar_efficiency_rolling'] = rolling_efficiency
                tprint_info(f"   → Rolling efficiency: mean={rolling_efficiency.mean():.3f}, std={rolling_efficiency.std():.3f}")
            
            # Calculate efficiency grade (0.0-1.0)
            if self.config.include_efficiency_grade:
                # Normalize efficiency to 0-1 range, with 0.6+ efficiency = 1.0 grade
                efficiency_grade = np.clip(raw_efficiency / self.config.high_efficiency_threshold, 0.0, 1.0)
                features['bar_efficiency_grade'] = efficiency_grade
                tprint_info(f"   → Efficiency grade: mean={efficiency_grade.mean():.3f}, std={efficiency_grade.std():.3f}")
            
            # Calculate efficiency classification
            if self.config.include_efficiency_class:
                efficiency_class = self._calculate_efficiency_classification(raw_efficiency, rolling_efficiency)
                features['bar_efficiency_class'] = efficiency_class
                
                # Count classifications
                class_counts = efficiency_class.value_counts()
                tprint_info(f"   → Efficiency classification: {dict(class_counts)}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating basic efficiency features: {e}")
        
        return features
    
    def _calculate_efficiency_classification(
        self, 
        raw_efficiency: pd.Series, 
        rolling_efficiency: pd.Series
    ) -> pd.Series:
        """Calculate efficiency classification."""
        efficiency_class = pd.Series('choppy', index=raw_efficiency.index)
        
        # Classify based on rolling efficiency
        efficiency_class[rolling_efficiency >= self.config.extreme_efficiency_threshold] = 'extreme_directional'
        efficiency_class[rolling_efficiency >= self.config.high_efficiency_threshold] = 'directional'
        efficiency_class[rolling_efficiency < self.config.low_efficiency_threshold] = 'choppy'
        
        # Add moderate classification
        moderate_mask = (rolling_efficiency >= self.config.low_efficiency_threshold) & \
                       (rolling_efficiency < self.config.high_efficiency_threshold)
        efficiency_class[moderate_mask] = 'moderate'
        
        return efficiency_class
    
    def _calculate_additional_efficiency_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate efficiency features for additional windows."""
        features = {}
        
        try:
            # Calculate raw efficiency for additional windows
            price_range = data['high'] - data['low']
            price_range = price_range.replace(0, np.nan)
            
            raw_efficiency = np.abs(data['close'] - data['open']) / price_range
            raw_efficiency = raw_efficiency.fillna(0).replace([np.inf, -np.inf], 0)
            
            for window in self.config.additional_windows:
                # Rolling efficiency for additional window
                rolling_efficiency = raw_efficiency.rolling(
                    window=window,
                    min_periods=1
                ).mean()
                features[f'bar_efficiency_rolling_{window}'] = rolling_efficiency
                
                # Efficiency grade for additional window
                efficiency_grade = np.clip(raw_efficiency / self.config.high_efficiency_threshold, 0.0, 1.0)
                features[f'bar_efficiency_grade_{window}'] = efficiency_grade
                
                # Efficiency volatility (volatility of efficiency)
                efficiency_volatility = raw_efficiency.rolling(window).std()
                features[f'bar_efficiency_volatility_{window}'] = efficiency_volatility
                
                # Efficiency momentum (change in efficiency)
                efficiency_momentum = rolling_efficiency.diff()
                features[f'bar_efficiency_momentum_{window}'] = efficiency_momentum
                
                # Efficiency trend (trend of efficiency)
                efficiency_trend = rolling_efficiency.rolling(5).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
                features[f'bar_efficiency_trend_{window}'] = efficiency_trend
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating additional efficiency features: {e}")
        
        return features
    
    def _calculate_candlestick_pattern_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate candlestick pattern features."""
        features = {}
        
        try:
            # Get candlestick patterns from VectorBT
            patterns = self.indicators.vbt.CANDLE.run(
                data['open'], data['high'], data['low'], data['close']
            )
            
            # Basic candlestick patterns
            features['candlestick_doji'] = (patterns.doji).astype(int)
            features['candlestick_hammer'] = (patterns.hammer).astype(int)
            features['candlestick_shooting_star'] = (patterns.shooting_star).astype(int)
            features['candlestick_engulfing'] = (patterns.engulfing).astype(int)
            features['candlestick_harami'] = (patterns.harami).astype(int)
            
            # Pattern strength
            pattern_strength = patterns.patterns.astype(int).sum(axis=1)
            features['candlestick_pattern_strength'] = pattern_strength
            
            # Pattern frequency
            pattern_frequency = pattern_strength.rolling(self.config.pattern_window).mean()
            features['candlestick_pattern_frequency'] = pattern_frequency
            
            # Pattern efficiency correlation
            if 'bar_efficiency_raw' in features:
                pattern_efficiency_corr = pattern_strength.rolling(20).corr(features['bar_efficiency_raw'])
                features['candlestick_pattern_efficiency_corr'] = pattern_efficiency_corr.fillna(0)
            
            tprint_info("   → Candlestick pattern features calculated")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating candlestick pattern features: {e}")
        
        return features
    
    def _calculate_price_action_momentum_features(
        self, 
        data: pd.DataFrame, 
        features: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Calculate price action momentum features."""
        momentum_features = {}
        
        try:
            # Get efficiency data
            raw_efficiency = features.get('bar_efficiency_raw')
            rolling_efficiency = features.get('bar_efficiency_rolling')
            
            if raw_efficiency is not None:
                # Efficiency momentum
                efficiency_momentum = raw_efficiency.diff()
                momentum_features['bar_efficiency_momentum'] = efficiency_momentum
                
                # Efficiency acceleration
                efficiency_acceleration = efficiency_momentum.diff()
                momentum_features['bar_efficiency_acceleration'] = efficiency_acceleration
                
                # Efficiency momentum strength
                momentum_strength = efficiency_momentum.abs()
                momentum_features['bar_efficiency_momentum_strength'] = momentum_strength
                
                # Efficiency momentum trend
                momentum_trend = efficiency_momentum.rolling(self.config.momentum_window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
                momentum_features['bar_efficiency_momentum_trend'] = momentum_trend
            
            if rolling_efficiency is not None:
                # Rolling efficiency momentum
                rolling_momentum = rolling_efficiency.diff()
                momentum_features['bar_efficiency_rolling_momentum'] = rolling_momentum
                
                # Rolling efficiency acceleration
                rolling_acceleration = rolling_momentum.diff()
                momentum_features['bar_efficiency_rolling_acceleration'] = rolling_acceleration
            
            # Price action strength (combination of efficiency and momentum)
            if raw_efficiency is not None and 'bar_efficiency_momentum' in momentum_features:
                price_action_strength = raw_efficiency * momentum_features['bar_efficiency_momentum'].abs()
                momentum_features['price_action_strength'] = price_action_strength
                
                # Price action strength trend
                strength_trend = price_action_strength.rolling(self.config.momentum_window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
                momentum_features['price_action_strength_trend'] = strength_trend
            
            tprint_info("   → Price action momentum features calculated")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating price action momentum features: {e}")
        
        return momentum_features
    
    def _calculate_volume_efficiency_features(
        self, 
        data: pd.DataFrame, 
        features: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Calculate volume efficiency features."""
        volume_features = {}
        
        try:
            if 'volume' not in data.columns:
                tprint_warning("⚠️ Volume data not available for volume efficiency analysis")
                return volume_features
            
            # Get efficiency data
            raw_efficiency = features.get('bar_efficiency_raw')
            rolling_efficiency = features.get('bar_efficiency_rolling')
            
            if raw_efficiency is not None:
                # Volume-weighted efficiency
                volume_weighted_efficiency = (raw_efficiency * data['volume']).rolling(
                    self.config.volume_window
                ).sum() / data['volume'].rolling(self.config.volume_window).sum()
                volume_features['bar_efficiency_volume_weighted'] = volume_weighted_efficiency.fillna(0)
                
                # Volume efficiency correlation
                volume_efficiency_corr = data['volume'].rolling(20).corr(raw_efficiency)
                volume_features['bar_efficiency_volume_correlation'] = volume_efficiency_corr.fillna(0)
                
                # Volume efficiency ratio
                volume_efficiency_ratio = raw_efficiency / (data['volume'] / data['volume'].rolling(20).mean())
                volume_features['bar_efficiency_volume_ratio'] = volume_efficiency_ratio.fillna(0)
            
            if rolling_efficiency is not None:
                # Rolling volume-weighted efficiency
                rolling_volume_weighted = (rolling_efficiency * data['volume']).rolling(
                    self.config.volume_window
                ).sum() / data['volume'].rolling(self.config.volume_window).sum()
                volume_features['bar_efficiency_rolling_volume_weighted'] = rolling_volume_weighted.fillna(0)
            
            # Volume efficiency classification
            if 'bar_efficiency_volume_weighted' in volume_features:
                volume_efficiency_class = pd.Series('normal', index=data.index)
                volume_efficiency_class[volume_features['bar_efficiency_volume_weighted'] > 
                                      volume_features['bar_efficiency_volume_weighted'].rolling(50).quantile(0.8)] = 'high_volume_efficiency'
                volume_efficiency_class[volume_features['bar_efficiency_volume_weighted'] < 
                                      volume_features['bar_efficiency_volume_weighted'].rolling(50).quantile(0.2)] = 'low_volume_efficiency'
                volume_features['bar_efficiency_volume_class'] = volume_efficiency_class
            
            tprint_info("   → Volume efficiency features calculated")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating volume efficiency features: {e}")
        
        return volume_features
    
    def _calculate_efficiency_regime_features(self, features: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Calculate efficiency regime features."""
        regime_features = {}
        
        try:
            # Get efficiency data
            rolling_efficiency = features.get('bar_efficiency_rolling')
            efficiency_grade = features.get('bar_efficiency_grade')
            
            if rolling_efficiency is not None:
                # Efficiency regime based on rolling efficiency
                efficiency_regime = pd.Series('normal', index=rolling_efficiency.index)
                
                # High efficiency regime
                high_efficiency = rolling_efficiency >= self.config.high_efficiency_threshold
                efficiency_regime[high_efficiency] = 'high_efficiency'
                
                # Low efficiency regime
                low_efficiency = rolling_efficiency < self.config.low_efficiency_threshold
                efficiency_regime[low_efficiency] = 'low_efficiency'
                
                # Extreme efficiency regime
                extreme_efficiency = rolling_efficiency >= self.config.extreme_efficiency_threshold
                efficiency_regime[extreme_efficiency] = 'extreme_efficiency'
                
                regime_features['bar_efficiency_regime'] = efficiency_regime
                
                # Regime persistence
                regime_changes = (efficiency_regime != efficiency_regime.shift(1)).astype(int)
                regime_features['bar_efficiency_regime_persistence'] = regime_changes.rolling(20).sum()
                
                # Regime strength
                regime_strength = rolling_efficiency.groupby(efficiency_regime).transform('mean')
                regime_features['bar_efficiency_regime_strength'] = regime_strength
            
            # Efficiency regime score (composite)
            if efficiency_grade is not None:
                regime_score = pd.Series(0.0, index=efficiency_grade.index)
                regime_score += efficiency_grade * 0.5
                
                if rolling_efficiency is not None:
                    regime_score += np.clip(rolling_efficiency, 0.0, 1.0) * 0.5
                
                regime_features['bar_efficiency_regime_score'] = regime_score
                
                # Regime classification based on score
                regime_class = pd.Series('normal', index=regime_score.index)
                regime_class[regime_score > 0.8] = 'high_efficiency_regime'
                regime_class[regime_score < 0.2] = 'low_efficiency_regime'
                regime_class[regime_score > 0.9] = 'extreme_efficiency_regime'
                regime_features['bar_efficiency_regime_class'] = regime_class
            
            tprint_info("   → Efficiency regime features calculated")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating efficiency regime features: {e}")
        
        return regime_features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this class produces."""
        features = []
        
        if self.config.include_raw_efficiency:
            features.append('bar_efficiency_raw')
        if self.config.include_rolling_efficiency:
            features.append('bar_efficiency_rolling')
        if self.config.include_efficiency_grade:
            features.append('bar_efficiency_grade')
        if self.config.include_efficiency_class:
            features.append('bar_efficiency_class')
        
        # Additional windows
        for window in self.config.additional_windows:
            features.extend([
                f'bar_efficiency_rolling_{window}',
                f'bar_efficiency_grade_{window}',
                f'bar_efficiency_volatility_{window}',
                f'bar_efficiency_momentum_{window}',
                f'bar_efficiency_trend_{window}'
            ])
        
        if self.config.include_candlestick_patterns:
            features.extend([
                'candlestick_doji', 'candlestick_hammer', 'candlestick_shooting_star',
                'candlestick_engulfing', 'candlestick_harami', 'candlestick_pattern_strength',
                'candlestick_pattern_frequency', 'candlestick_pattern_efficiency_corr'
            ])
        
        if self.config.include_price_action_momentum:
            features.extend([
                'bar_efficiency_momentum', 'bar_efficiency_acceleration',
                'bar_efficiency_momentum_strength', 'bar_efficiency_momentum_trend',
                'bar_efficiency_rolling_momentum', 'bar_efficiency_rolling_acceleration',
                'price_action_strength', 'price_action_strength_trend'
            ])
        
        if self.config.include_volume_efficiency:
            features.extend([
                'bar_efficiency_volume_weighted', 'bar_efficiency_volume_correlation',
                'bar_efficiency_volume_ratio', 'bar_efficiency_rolling_volume_weighted',
                'bar_efficiency_volume_class'
            ])
        
        if self.config.include_efficiency_regime:
            features.extend([
                'bar_efficiency_regime', 'bar_efficiency_regime_persistence',
                'bar_efficiency_regime_strength', 'bar_efficiency_regime_score',
                'bar_efficiency_regime_class'
            ])
        
        return features
    
    def get_feature_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about the features."""
        return {
            'bar_efficiency_raw': {
                'description': 'Raw bar efficiency ratio (|close-open| / (high-low))',
                'range': '[0, 1]',
                'interpretation': 'Higher values indicate more directional price action'
            },
            'bar_efficiency_rolling': {
                'description': f'Rolling mean efficiency over {self.config.window} bars',
                'range': '[0, 1]',
                'interpretation': 'Smoothed efficiency for trend analysis'
            },
            'bar_efficiency_grade': {
                'description': 'Normalized efficiency grade (0.0-1.0)',
                'range': '[0, 1]',
                'interpretation': '1.0 = high efficiency, 0.0 = low efficiency'
            },
            'bar_efficiency_class': {
                'description': 'Efficiency classification',
                'values': ['choppy', 'moderate', 'directional', 'extreme_directional'],
                'interpretation': 'Categorical classification based on thresholds'
            },
            'bar_efficiency_regime': {
                'description': 'Efficiency regime classification',
                'values': ['low_efficiency', 'normal', 'high_efficiency', 'extreme_efficiency'],
                'interpretation': 'Current efficiency regime state'
            }
        }


class VectorBTBarEfficiencyRatioGenerator(VectorBTFeatureGenerator):
    """
    VectorBT-enhanced Bar Efficiency Ratio feature generator.
    
    Provides comprehensive price action analysis with VectorBT optimization,
    parameter tuning, and advanced feature generation capabilities.
    """
    
    def __init__(self, lookback: int = 3, **kwargs):
        """
        Initialize the VectorBT Bar Efficiency Ratio feature generator.
        
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
            name="vectorbt_bar_efficiency_ratio",
            category=FeatureCategory.PRICE_ACTION,
            description="VectorBT-enhanced bar efficiency ratio with comprehensive price action analysis",
            required_columns=['open', 'high', 'low', 'close'],
            optional_columns=['volume'],
            default_lookback=lookback,
            min_lookback=1,
            max_lookback=20,
            parameters={
                'window': lookback,
                'additional_windows': kwargs.get('additional_windows', [2, 4, 6, 8, 10]),
                'high_efficiency_threshold': kwargs.get('high_efficiency_threshold', 0.6),
                'low_efficiency_threshold': kwargs.get('low_efficiency_threshold', 0.3),
                'extreme_efficiency_threshold': kwargs.get('extreme_efficiency_threshold', 0.8),
                'enable_candlestick_patterns': kwargs.get('enable_candlestick_patterns', True),
                'pattern_window': kwargs.get('pattern_window', 5),
                'enable_momentum_analysis': kwargs.get('enable_momentum_analysis', True),
                'momentum_window': kwargs.get('momentum_window', 10),
                'enable_volume_analysis': kwargs.get('enable_volume_analysis', True),
                'volume_window': kwargs.get('volume_window', 20),
                'include_raw_efficiency': kwargs.get('include_raw_efficiency', True),
                'include_rolling_efficiency': kwargs.get('include_rolling_efficiency', True),
                'include_efficiency_grade': kwargs.get('include_efficiency_grade', True),
                'include_efficiency_class': kwargs.get('include_efficiency_class', True),
                'include_candlestick_patterns': kwargs.get('include_candlestick_patterns', True),
                'include_price_action_momentum': kwargs.get('include_price_action_momentum', True),
                'include_volume_efficiency': kwargs.get('include_volume_efficiency', True),
                'include_efficiency_regime': kwargs.get('include_efficiency_regime', True)
            },
            matrix_optimized=True,
            gpu_accelerated=False,
            enable_feature_selection=True
        )
        
        super().__init__(config, vectorbt_config)
        
        # Initialize the feature engine
        feature_config = VectorBTBarEfficiencyConfig(
            window=lookback,
            additional_windows=kwargs.get('additional_windows', [2, 4, 6, 8, 10]),
            high_efficiency_threshold=kwargs.get('high_efficiency_threshold', 0.6),
            low_efficiency_threshold=kwargs.get('low_efficiency_threshold', 0.3),
            extreme_efficiency_threshold=kwargs.get('extreme_efficiency_threshold', 0.8),
            enable_candlestick_patterns=kwargs.get('enable_candlestick_patterns', True),
            pattern_window=kwargs.get('pattern_window', 5),
            enable_momentum_analysis=kwargs.get('enable_momentum_analysis', True),
            momentum_window=kwargs.get('momentum_window', 10),
            enable_volume_analysis=kwargs.get('enable_volume_analysis', True),
            volume_window=kwargs.get('volume_window', 20),
            include_raw_efficiency=kwargs.get('include_raw_efficiency', True),
            include_rolling_efficiency=kwargs.get('include_rolling_efficiency', True),
            include_efficiency_grade=kwargs.get('include_efficiency_grade', True),
            include_efficiency_class=kwargs.get('include_efficiency_class', True),
            include_candlestick_patterns=kwargs.get('include_candlestick_patterns', True),
            include_price_action_momentum=kwargs.get('include_price_action_momentum', True),
            include_volume_efficiency=kwargs.get('include_volume_efficiency', True),
            include_efficiency_regime=kwargs.get('include_efficiency_regime', True)
        )
        self.feature_engine = VectorBTBarEfficiencyRatioFeature(feature_config)
    
    def generate_vectorbt_features(
        self, 
        data: pd.DataFrame, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate VectorBT Bar Efficiency Ratio features.
        
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
            'window': [2, 3, 4, 5, 6],
            'high_efficiency_threshold': [0.5, 0.6, 0.7, 0.8],
            'low_efficiency_threshold': [0.2, 0.3, 0.4, 0.5],
            'extreme_efficiency_threshold': [0.7, 0.8, 0.9],
            'pattern_window': [3, 5, 7, 10],
            'momentum_window': [5, 10, 15, 20],
            'volume_window': [10, 20, 30, 40]
        }
        
        return super().optimize_parameters(data, param_ranges, target_metric)


# Convenience function for external usage
def calculate_vectorbt_bar_efficiency_features(
    data: pd.DataFrame,
    config: Optional[VectorBTBarEfficiencyConfig] = None,
    **kwargs
) -> Dict[str, pd.Series]:
    """
    Calculate VectorBT Bar Efficiency Ratio features.
    
    Args:
        data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
        config: Optional configuration
        **kwargs: Additional parameters
        
    Returns:
        Dictionary of feature Series
    """
    feature_engine = VectorBTBarEfficiencyRatioFeature(config)
    return feature_engine.calculate_features(data)