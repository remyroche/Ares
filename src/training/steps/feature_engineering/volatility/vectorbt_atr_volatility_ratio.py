"""
VectorBT ATR Volatility Ratio Feature Engineering

This module implements an enhanced ATR Volatility Ratio feature using VectorBT
for superior performance and comprehensive volatility analysis.

Features:
- VectorBT-optimized ATR calculations
- Multiple volatility measures (ATR, Bollinger Bands, Keltner Channels)
- Advanced volatility ratios and classifications
- Parameter optimization capabilities
- Comprehensive volatility regime detection
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
class VectorBTATRVolatilityConfig:
    """Enhanced configuration for VectorBT ATR Volatility Ratio feature."""
    
    # ATR calculation settings
    short_window: int = 4   # Short-term ATR window (1 hour)
    long_window: int = 20   # Long-term ATR window (5 hours)
    additional_windows: List[int] = None  # Additional ATR windows for analysis
    
    # Volatility thresholds
    high_ratio_threshold: float = 1.5  # Too jumpy - skip signals
    extreme_ratio_threshold: float = 2.0  # Extreme volatility
    low_ratio_threshold: float = 0.5  # Low volatility threshold
    
    # Bollinger Bands settings
    bb_window: int = 20
    bb_std: float = 2.0
    
    # Keltner Channels settings
    kc_window: int = 20
    kc_atr_multiplier: float = 2.0
    
    # Output settings
    include_atr_short: bool = True
    include_atr_long: bool = True
    include_atr_ratio: bool = True
    include_atr_grade: bool = True
    include_atr_class: bool = True
    include_bb_volatility: bool = True
    include_kc_volatility: bool = True
    include_volatility_regime: bool = True
    include_volatility_momentum: bool = True
    
    def __post_init__(self):
        if self.additional_windows is None:
            self.additional_windows = [8, 14, 30]


class VectorBTATRVolatilityRatioFeature:
    """
    Enhanced ATR Volatility Ratio Feature Engineering using VectorBT.
    
    Provides comprehensive volatility analysis with multiple indicators,
    regime detection, and advanced classification capabilities.
    """
    
    def __init__(self, config: Optional[VectorBTATRVolatilityConfig] = None):
        """Initialize VectorBT ATR Volatility Ratio feature."""
        self.config = config or VectorBTATRVolatilityConfig()
        self.indicators = VectorBTTechnicalIndicators()
        
        tprint_info("📊 VectorBT ATR Volatility Ratio feature initialized")
        tprint_info(f"   → Short window: {self.config.short_window} bars")
        tprint_info(f"   → Long window: {self.config.long_window} bars")
        tprint_info(f"   → Additional windows: {self.config.additional_windows}")
        tprint_info(f"   → High ratio threshold: {self.config.high_ratio_threshold}")
        tprint_info(f"   → BB window: {self.config.bb_window}, KC window: {self.config.kc_window}")
    
    def calculate_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate comprehensive ATR Volatility Ratio features using VectorBT.
        
        Args:
            data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary of feature Series
        """
        tprint_info("📊 Calculating VectorBT ATR Volatility Ratio features")
        
        # Validate input data
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        features = {}
        
        try:
            # Get comprehensive volatility indicators
            volatility_indicators = self.indicators.get_volatility_indicators(
                data, 
                windows=[self.config.short_window, self.config.long_window] + self.config.additional_windows
            )
            
            # Extract ATR indicators
            atr_short = volatility_indicators.get(f'atr_{self.config.short_window}')
            atr_long = volatility_indicators.get(f'atr_{self.config.long_window}')
            
            if atr_short is None or atr_long is None:
                raise ValueError("Failed to calculate ATR indicators")
            
            # Basic ATR features
            if self.config.include_atr_short:
                features['atr_short'] = atr_short
                tprint_info(f"   → Short-term ATR: mean={atr_short.mean():.3f}, std={atr_short.std():.3f}")
            
            if self.config.include_atr_long:
                features['atr_long'] = atr_long
                tprint_info(f"   → Long-term ATR: mean={atr_long.mean():.3f}, std={atr_long.std():.3f}")
            
            # ATR ratio
            if self.config.include_atr_ratio:
                atr_ratio = atr_short / atr_long
                atr_ratio = atr_ratio.fillna(1.0).replace([np.inf, -np.inf], 1.0)
                features['atr_ratio'] = atr_ratio
                tprint_info(f"   → ATR ratio: mean={atr_ratio.mean():.3f}, std={atr_ratio.std():.3f}")
            
            # ATR grade (0.0-1.0)
            if self.config.include_atr_grade:
                atr_grade = self._calculate_atr_grade(atr_ratio)
                features['atr_grade'] = atr_grade
                tprint_info(f"   → ATR grade: mean={atr_grade.mean():.3f}, std={atr_grade.std():.3f}")
            
            # ATR classification
            if self.config.include_atr_class:
                atr_class = self._calculate_atr_classification(atr_ratio)
                features['atr_class'] = atr_class
                
                class_counts = atr_class.value_counts()
                tprint_info(f"   → ATR classification: {dict(class_counts)}")
            
            # Bollinger Bands volatility
            if self.config.include_bb_volatility:
                bb_features = self._calculate_bb_volatility_features(data, volatility_indicators)
                features.update(bb_features)
            
            # Keltner Channels volatility
            if self.config.include_kc_volatility:
                kc_features = self._calculate_kc_volatility_features(data, volatility_indicators)
                features.update(kc_features)
            
            # Volatility regime detection
            if self.config.include_volatility_regime:
                regime_features = self._calculate_volatility_regime_features(
                    data, atr_ratio, volatility_indicators
                )
                features.update(regime_features)
            
            # Volatility momentum
            if self.config.include_volatility_momentum:
                momentum_features = self._calculate_volatility_momentum_features(
                    atr_ratio, volatility_indicators
                )
                features.update(momentum_features)
            
            # Additional ATR windows analysis
            additional_features = self._calculate_additional_atr_features(
                data, volatility_indicators
            )
            features.update(additional_features)
            
            tprint_success("✅ VectorBT ATR Volatility Ratio features calculated successfully")
            return features
            
        except Exception as e:
            tprint_error(f"❌ Error calculating VectorBT ATR Volatility Ratio features: {e}")
            raise
    
    def _calculate_atr_grade(self, atr_ratio: pd.Series) -> pd.Series:
        """Calculate ATR grade based on ratio thresholds."""
        # Grade decreases as ratio approaches the threshold (too jumpy)
        # No penalty for low ratios (no "too quiet" filter)
        grade = np.clip(
            1.0 - (atr_ratio / self.config.high_ratio_threshold), 
            0.0, 1.0
        )
        return grade
    
    def _calculate_atr_classification(self, atr_ratio: pd.Series) -> pd.Series:
        """Calculate ATR classification based on ratio thresholds."""
        atr_class = pd.Series('moderate', index=atr_ratio.index)
        
        # Classify based on thresholds
        atr_class[atr_ratio > self.config.extreme_ratio_threshold] = 'extreme'
        atr_class[atr_ratio > self.config.high_ratio_threshold] = 'too_jumpy'
        atr_class[atr_ratio < self.config.low_ratio_threshold] = 'low_volatility'
        
        return atr_class
    
    def _calculate_bb_volatility_features(
        self, 
        data: pd.DataFrame, 
        volatility_indicators: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands volatility features."""
        features = {}
        
        try:
            bb_width = volatility_indicators.get(f'bb_width_{self.config.bb_window}')
            bb_position = volatility_indicators.get(f'bb_position_{self.config.bb_window}')
            
            if bb_width is not None:
                features['bb_volatility_width'] = bb_width
                features['bb_volatility_grade'] = np.clip(bb_width / bb_width.rolling(50).mean(), 0.0, 2.0)
                
                # BB volatility classification
                bb_vol_class = pd.Series('normal', index=bb_width.index)
                bb_vol_class[bb_width > bb_width.rolling(50).quantile(0.8)] = 'high_volatility'
                bb_vol_class[bb_width < bb_width.rolling(50).quantile(0.2)] = 'low_volatility'
                features['bb_volatility_class'] = bb_vol_class
            
            if bb_position is not None:
                features['bb_position'] = bb_position
                features['bb_squeeze'] = (bb_position > 0.8) | (bb_position < 0.2)
                
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating BB volatility features: {e}")
        
        return features
    
    def _calculate_kc_volatility_features(
        self, 
        data: pd.DataFrame, 
        volatility_indicators: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Calculate Keltner Channels volatility features."""
        features = {}
        
        try:
            kc_width = volatility_indicators.get(f'kc_width_{self.config.kc_window}')
            
            if kc_width is not None:
                features['kc_volatility_width'] = kc_width
                features['kc_volatility_grade'] = np.clip(kc_width / kc_width.rolling(50).mean(), 0.0, 2.0)
                
                # KC volatility classification
                kc_vol_class = pd.Series('normal', index=kc_width.index)
                kc_vol_class[kc_width > kc_width.rolling(50).quantile(0.8)] = 'high_volatility'
                kc_vol_class[kc_width < kc_width.rolling(50).quantile(0.2)] = 'low_volatility'
                features['kc_volatility_class'] = kc_vol_class
                
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating KC volatility features: {e}")
        
        return features
    
    def _calculate_volatility_regime_features(
        self, 
        data: pd.DataFrame, 
        atr_ratio: pd.Series, 
        volatility_indicators: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Calculate volatility regime detection features."""
        features = {}
        
        try:
            # Volatility regime based on multiple indicators
            regime_score = pd.Series(0.0, index=data.index)
            
            # ATR ratio component
            atr_component = np.clip(atr_ratio / self.config.high_ratio_threshold, 0.0, 1.0)
            regime_score += atr_component * 0.4
            
            # BB width component
            bb_width = volatility_indicators.get(f'bb_width_{self.config.bb_window}')
            if bb_width is not None:
                bb_component = np.clip(bb_width / bb_width.rolling(50).mean(), 0.0, 1.0)
                regime_score += bb_component * 0.3
            
            # KC width component
            kc_width = volatility_indicators.get(f'kc_width_{self.config.kc_window}')
            if kc_width is not None:
                kc_component = np.clip(kc_width / kc_width.rolling(50).mean(), 0.0, 1.0)
                regime_score += kc_component * 0.3
            
            features['volatility_regime_score'] = regime_score
            
            # Regime classification
            regime_class = pd.Series('normal', index=data.index)
            regime_class[regime_score > 0.7] = 'high_volatility'
            regime_class[regime_score < 0.3] = 'low_volatility'
            regime_class[regime_score > 0.9] = 'extreme_volatility'
            features['volatility_regime_class'] = regime_class
            
            # Regime persistence
            regime_changes = (regime_class != regime_class.shift(1)).astype(int)
            features['volatility_regime_persistence'] = regime_changes.rolling(20).sum()
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating volatility regime features: {e}")
        
        return features
    
    def _calculate_volatility_momentum_features(
        self, 
        atr_ratio: pd.Series, 
        volatility_indicators: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Calculate volatility momentum features."""
        features = {}
        
        try:
            # ATR ratio momentum
            atr_momentum = atr_ratio.diff()
            features['atr_ratio_momentum'] = atr_momentum
            features['atr_ratio_acceleration'] = atr_momentum.diff()
            
            # Volatility trend
            atr_trend = atr_ratio.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            features['atr_ratio_trend'] = atr_trend
            
            # Volatility momentum classification
            momentum_class = pd.Series('stable', index=atr_ratio.index)
            momentum_class[atr_momentum > atr_momentum.rolling(20).std()] = 'increasing'
            momentum_class[atr_momentum < -atr_momentum.rolling(20).std()] = 'decreasing'
            features['volatility_momentum_class'] = momentum_class
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating volatility momentum features: {e}")
        
        return features
    
    def _calculate_additional_atr_features(
        self, 
        data: pd.DataFrame, 
        volatility_indicators: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Calculate features for additional ATR windows."""
        features = {}
        
        try:
            for window in self.config.additional_windows:
                atr = volatility_indicators.get(f'atr_{window}')
                if atr is not None:
                    # ATR relative to price
                    atr_ratio = atr / data['close']
                    features[f'atr_price_ratio_{window}'] = atr_ratio
                    
                    # ATR percentile
                    atr_percentile = atr.rolling(100).rank(pct=True)
                    features[f'atr_percentile_{window}'] = atr_percentile
                    
                    # ATR volatility (volatility of volatility)
                    atr_volatility = atr.rolling(20).std()
                    features[f'atr_volatility_{window}'] = atr_volatility
                    
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating additional ATR features: {e}")
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this class produces."""
        features = []
        
        if self.config.include_atr_short:
            features.append('atr_short')
        if self.config.include_atr_long:
            features.append('atr_long')
        if self.config.include_atr_ratio:
            features.append('atr_ratio')
        if self.config.include_atr_grade:
            features.append('atr_grade')
        if self.config.include_atr_class:
            features.append('atr_class')
        if self.config.include_bb_volatility:
            features.extend(['bb_volatility_width', 'bb_volatility_grade', 'bb_volatility_class', 
                           'bb_position', 'bb_squeeze'])
        if self.config.include_kc_volatility:
            features.extend(['kc_volatility_width', 'kc_volatility_grade', 'kc_volatility_class'])
        if self.config.include_volatility_regime:
            features.extend(['volatility_regime_score', 'volatility_regime_class', 
                           'volatility_regime_persistence'])
        if self.config.include_volatility_momentum:
            features.extend(['atr_ratio_momentum', 'atr_ratio_acceleration', 'atr_ratio_trend', 
                           'volatility_momentum_class'])
        
        # Additional ATR windows
        for window in self.config.additional_windows:
            features.extend([
                f'atr_price_ratio_{window}',
                f'atr_percentile_{window}',
                f'atr_volatility_{window}'
            ])
        
        return features
    
    def get_feature_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about the features."""
        return {
            'atr_short': {
                'description': f'Short-term Average True Range over {self.config.short_window} bars',
                'range': '[0, inf)',
                'interpretation': 'Recent volatility measure'
            },
            'atr_long': {
                'description': f'Long-term Average True Range over {self.config.long_window} bars',
                'range': '[0, inf)',
                'interpretation': 'Baseline volatility measure'
            },
            'atr_ratio': {
                'description': 'Ratio of short-term to long-term ATR',
                'range': '[0, inf)',
                'interpretation': 'Higher values indicate increased volatility'
            },
            'atr_grade': {
                'description': 'Normalized ATR grade (0.0-1.0)',
                'range': '[0, 1]',
                'interpretation': '1.0 = moderate volatility, 0.0 = too jumpy'
            },
            'atr_class': {
                'description': 'ATR classification (moderate/too_jumpy/extreme/low_volatility)',
                'values': ['moderate', 'too_jumpy', 'extreme', 'low_volatility'],
                'interpretation': 'Categorical classification based on thresholds'
            },
            'volatility_regime_score': {
                'description': 'Composite volatility regime score (0.0-1.0)',
                'range': '[0, 1]',
                'interpretation': 'Higher values indicate higher volatility regime'
            },
            'volatility_regime_class': {
                'description': 'Volatility regime classification',
                'values': ['low_volatility', 'normal', 'high_volatility', 'extreme_volatility'],
                'interpretation': 'Current volatility regime state'
            }
        }


class VectorBTATRVolatilityRatioGenerator(VectorBTFeatureGenerator):
    """
    VectorBT-enhanced ATR Volatility Ratio feature generator.
    
    Provides comprehensive volatility analysis with VectorBT optimization,
    parameter tuning, and advanced feature generation capabilities.
    """
    
    def __init__(self, lookback: int = 4, **kwargs):
        """
        Initialize the VectorBT ATR Volatility Ratio feature generator.
        
        Args:
            lookback: Number of periods for short-term ATR calculation
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
            name="vectorbt_atr_volatility_ratio",
            category=FeatureCategory.VOLATILITY,
            description="VectorBT-enhanced ATR volatility ratio with comprehensive volatility analysis",
            required_columns=['open', 'high', 'low', 'close'],
            optional_columns=['volume'],
            default_lookback=lookback,
            min_lookback=1,
            max_lookback=100,
            parameters={
                'short_window': lookback,
                'long_window': kwargs.get('long_window', 20),
                'additional_windows': kwargs.get('additional_windows', [8, 14, 30]),
                'high_ratio_threshold': kwargs.get('high_ratio_threshold', 1.5),
                'extreme_ratio_threshold': kwargs.get('extreme_ratio_threshold', 2.0),
                'low_ratio_threshold': kwargs.get('low_ratio_threshold', 0.5),
                'bb_window': kwargs.get('bb_window', 20),
                'bb_std': kwargs.get('bb_std', 2.0),
                'kc_window': kwargs.get('kc_window', 20),
                'kc_atr_multiplier': kwargs.get('kc_atr_multiplier', 2.0),
                'include_atr_short': kwargs.get('include_atr_short', True),
                'include_atr_long': kwargs.get('include_atr_long', True),
                'include_atr_ratio': kwargs.get('include_atr_ratio', True),
                'include_atr_grade': kwargs.get('include_atr_grade', True),
                'include_atr_class': kwargs.get('include_atr_class', True),
                'include_bb_volatility': kwargs.get('include_bb_volatility', True),
                'include_kc_volatility': kwargs.get('include_kc_volatility', True),
                'include_volatility_regime': kwargs.get('include_volatility_regime', True),
                'include_volatility_momentum': kwargs.get('include_volatility_momentum', True)
            },
            matrix_optimized=True,
            gpu_accelerated=False,
            enable_feature_selection=True
        )
        
        super().__init__(config, vectorbt_config)
        
        # Initialize the feature engine
        feature_config = VectorBTATRVolatilityConfig(
            short_window=lookback,
            long_window=kwargs.get('long_window', 20),
            additional_windows=kwargs.get('additional_windows', [8, 14, 30]),
            high_ratio_threshold=kwargs.get('high_ratio_threshold', 1.5),
            extreme_ratio_threshold=kwargs.get('extreme_ratio_threshold', 2.0),
            low_ratio_threshold=kwargs.get('low_ratio_threshold', 0.5),
            bb_window=kwargs.get('bb_window', 20),
            bb_std=kwargs.get('bb_std', 2.0),
            kc_window=kwargs.get('kc_window', 20),
            kc_atr_multiplier=kwargs.get('kc_atr_multiplier', 2.0),
            include_atr_short=kwargs.get('include_atr_short', True),
            include_atr_long=kwargs.get('include_atr_long', True),
            include_atr_ratio=kwargs.get('include_atr_ratio', True),
            include_atr_grade=kwargs.get('include_atr_grade', True),
            include_atr_class=kwargs.get('include_atr_class', True),
            include_bb_volatility=kwargs.get('include_bb_volatility', True),
            include_kc_volatility=kwargs.get('include_kc_volatility', True),
            include_volatility_regime=kwargs.get('include_volatility_regime', True),
            include_volatility_momentum=kwargs.get('include_volatility_momentum', True)
        )
        self.feature_engine = VectorBTATRVolatilityRatioFeature(feature_config)
    
    def generate_vectorbt_features(
        self, 
        data: pd.DataFrame, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate VectorBT ATR Volatility Ratio features.
        
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
            'short_window': [3, 4, 5, 6, 7, 8],
            'long_window': [15, 20, 25, 30, 35],
            'high_ratio_threshold': [1.2, 1.5, 1.8, 2.0, 2.2],
            'bb_window': [15, 20, 25, 30],
            'kc_window': [15, 20, 25, 30]
        }
        
        return super().optimize_parameters(data, param_ranges, target_metric)


# Convenience function for external usage
def calculate_vectorbt_atr_volatility_features(
    data: pd.DataFrame,
    config: Optional[VectorBTATRVolatilityConfig] = None,
    **kwargs
) -> Dict[str, pd.Series]:
    """
    Calculate VectorBT ATR Volatility Ratio features.
    
    Args:
        data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
        config: Optional configuration
        **kwargs: Additional parameters
        
    Returns:
        Dictionary of feature Series
    """
    feature_engine = VectorBTATRVolatilityRatioFeature(config)
    return feature_engine.calculate_features(data)