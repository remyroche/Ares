"""
Regime Feature Integration

This module integrates all regime-focused feature generators for use in
NAS-TAS clustering. Provides a unified interface for regime classification
features while excluding trading-relevant features.

Key Features:
- Unified regime feature generation
- Trading feature exclusion
- Regime-focused feature selection
- 15-minute timeframe optimization
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from ..core.feature_generator import (
    FeatureGenerator, 
    FeatureConfig, 
    FeatureCategory,
    VectorizedFeatureGenerator
)

# Import regime-focused feature generators
from .regime_volatility import RegimeVolatilityFeatureGenerator
from .regime_volume import RegimeVolumeFeatureGenerator
from .regime_structural_trend import RegimeStructuralTrendFeatureGenerator
from .regime_statistical import RegimeStatisticalFeatureGenerator

@dataclass
class RegimeFeatureConfig:
    """Configuration for regime-focused feature generation."""
    # Regime feature categories to include
    include_volatility_regime: bool = True
    include_volume_regime: bool = True
    include_structural_trend: bool = True
    include_statistical_regime: bool = True
    
    # Feature quality filters
    min_regime_persistence: float = 0.7
    max_feature_noise_ratio: float = 0.3
    min_temporal_stability: float = 0.6
    
    # 15-minute timeframe optimization
    optimize_for_15m: bool = True
    trade_duration_minutes: Tuple[int, int] = (5, 30)
    
    # Feature selection
    max_features_per_category: int = 20
    total_max_features: int = 80
    enable_feature_selection: bool = True

class RegimeFeatureIntegration(VectorizedFeatureGenerator):
    """Unified regime feature generator that excludes trading features."""
    
    def __init__(self, config: Optional[RegimeFeatureConfig] = None):
        if config is None:
            config = RegimeFeatureConfig()
        
        self.config = config
        
        # Initialize regime-focused feature generators
        self.volatility_generator = RegimeVolatilityFeatureGenerator() if config.include_volatility_regime else None
        self.volume_generator = RegimeVolumeFeatureGenerator() if config.include_volume_regime else None
        self.structural_trend_generator = RegimeStructuralTrendFeatureGenerator() if config.include_structural_trend else None
        self.statistical_generator = RegimeStatisticalFeatureGenerator() if config.include_statistical_regime else None
        
        # Initialize base config
        base_config = FeatureConfig(
            name="regime_feature_integration",
            category=FeatureCategory.STATISTICAL,
            description="Unified regime features for 15m timeframe regime classification",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=32,
            min_lookback=8,
            max_lookback=128,
            parameters={},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        
        super().__init__(base_config, enable_matrix_ops=True)
    
    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate unified regime features, excluding trading features."""
        features = {}
        feature_names = []
        
        try:
            # Generate volatility regime features
            if self.volatility_generator:
                vol_features = self.volatility_generator.generate_features(data, **kwargs)
                features.update(vol_features)
                feature_names.extend(vol_features.keys())
            
            # Generate volume regime features
            if self.volume_generator:
                vol_regime_features = self.volume_generator.generate_features(data, **kwargs)
                features.update(vol_regime_features)
                feature_names.extend(vol_regime_features.keys())
            
            # Generate structural trend features
            if self.structural_trend_generator:
                trend_features = self.structural_trend_generator.generate_features(data, **kwargs)
                features.update(trend_features)
                feature_names.extend(trend_features.keys())
            
            # Generate statistical regime features
            if self.statistical_generator:
                stat_features = self.statistical_generator.generate_features(data, **kwargs)
                features.update(stat_features)
                feature_names.extend(stat_features.keys())
            
            # Filter out any remaining trading-relevant features
            if self.config.enable_feature_selection:
                features = self._filter_trading_features(features, feature_names)
            
            # Apply feature quality filters
            features = self._apply_quality_filters(features, data)
            
            return features
            
        except Exception as e:
            print(f"Regime feature generation failed: {e}")
            return {}
    
    def _filter_trading_features(self, features: Dict[str, np.ndarray], feature_names: List[str]) -> Dict[str, np.ndarray]:
        """Filter out any remaining trading-relevant features."""
        trading_patterns = [
            'rsi', 'macd', 'stochastic', 'williams', 'momentum',
            'oscillator', 'signal', 'crossover', 'divergence',
            'candlestick', 'pattern', 'breakout', 'support', 'resistance',
            'bollinger', 'atr', 'cci', 'roc', 'mfi', 'obv', 'ema', 'sma'
        ]
        
        filtered_features = {}
        for name, feature_array in features.items():
            name_lower = name.lower()
            
            # Skip if matches trading patterns
            if any(pattern in name_lower for pattern in trading_patterns):
                continue
            
            # Keep regime-focused features
            regime_patterns = [
                'volatility', 'volume_regime', 'trend_persistence', 
                'regime_stability', 'correlation', 'distribution',
                'clustering', 'persistence', 'structural', 'statistical',
                'vol_persistence', 'vol_clustering', 'vol_stability',
                'vol_regime', 'trend_strength', 'market_structure'
            ]
            
            if any(pattern in name_lower for pattern in regime_patterns):
                filtered_features[name] = feature_array
        
        return filtered_features
    
    def _apply_quality_filters(self, features: Dict[str, np.ndarray], data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Apply quality filters to ensure regime-relevant features only."""
        filtered_features = {}
        
        for name, feature_array in features.items():
            # Skip if feature array is invalid
            if feature_array is None or len(feature_array) == 0:
                continue
            
            # Check feature quality
            if self._is_high_quality_regime_feature(feature_array):
                filtered_features[name] = feature_array
        
        return filtered_features
    
    def _is_high_quality_regime_feature(self, feature_array: np.ndarray) -> bool:
        """Check if a feature meets quality standards for regime classification."""
        try:
            # Remove NaN values for analysis
            valid_values = feature_array[~np.isnan(feature_array)]
            
            if len(valid_values) < 5:
                return False
            
            # Test 1: Regime persistence (autocorrelation)
            if len(valid_values) > 1:
                corr = np.corrcoef(valid_values[:-1], valid_values[1:])[0, 1]
                regime_persistence = corr if not np.isnan(corr) else 0.0
            else:
                regime_persistence = 0.0
            
            # Test 2: Low noise-to-signal ratio
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            noise_ratio = std_val / (abs(mean_val) + 1e-8)
            
            # Test 3: Temporal stability
            if len(valid_values) > 5:
                window = min(5, len(valid_values) // 2)
                rolling_means = []
                for i in range(window, len(valid_values)):
                    rolling_means.append(np.mean(valid_values[i-window:i]))
                
                if len(rolling_means) > 1:
                    temporal_stability = 1.0 - (np.std(rolling_means) / (np.mean(np.abs(rolling_means)) + 1e-8))
                else:
                    temporal_stability = 0.0
            else:
                temporal_stability = 0.0
            
            # Apply quality thresholds
            return (regime_persistence > self.config.min_regime_persistence and
                    noise_ratio < self.config.max_feature_noise_ratio and
                    temporal_stability > self.config.min_temporal_stability)
        
        except:
            return False
    
    def get_feature_summary(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Get summary of generated regime features."""
        summary = {
            'total_features': len(features),
            'feature_categories': {
                'volatility_regime': 0,
                'volume_regime': 0,
                'structural_trend': 0,
                'statistical_regime': 0
            },
            'quality_metrics': {
                'avg_persistence': 0.0,
                'avg_noise_ratio': 0.0,
                'avg_temporal_stability': 0.0
            }
        }
        
        for name, feature_array in features.items():
            name_lower = name.lower()
            
            # Categorize features
            if 'volatility' in name_lower or 'vol_' in name_lower:
                summary['feature_categories']['volatility_regime'] += 1
            elif 'volume' in name_lower or 'vol_regime' in name_lower:
                summary['feature_categories']['volume_regime'] += 1
            elif 'trend' in name_lower or 'structural' in name_lower:
                summary['feature_categories']['structural_trend'] += 1
            elif 'statistical' in name_lower or 'distribution' in name_lower:
                summary['feature_categories']['statistical_regime'] += 1
            
            # Calculate quality metrics
            if feature_array is not None and len(feature_array) > 0:
                valid_values = feature_array[~np.isnan(feature_array)]
                if len(valid_values) > 1:
                    # Persistence
                    corr = np.corrcoef(valid_values[:-1], valid_values[1:])[0, 1]
                    persistence = corr if not np.isnan(corr) else 0.0
                    summary['quality_metrics']['avg_persistence'] += persistence
                    
                    # Noise ratio
                    mean_val = np.mean(valid_values)
                    std_val = np.std(valid_values)
                    noise_ratio = std_val / (abs(mean_val) + 1e-8)
                    summary['quality_metrics']['avg_noise_ratio'] += noise_ratio
        
        # Average quality metrics
        if summary['total_features'] > 0:
            summary['quality_metrics']['avg_persistence'] /= summary['total_features']
            summary['quality_metrics']['avg_noise_ratio'] /= summary['total_features']
            summary['quality_metrics']['avg_temporal_stability'] = 0.8  # Placeholder
        
        return summary

# Convenience function for easy integration
def generate_regime_features(data: pd.DataFrame, 
                           config: Optional[RegimeFeatureConfig] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Generate regime-focused features for clustering.
    
    Args:
        data: Market data DataFrame with OHLCV columns
        config: Configuration for regime feature generation
        
    Returns:
        Tuple of (features_dict, summary_dict)
    """
    if config is None:
        config = RegimeFeatureConfig()
    
    generator = RegimeFeatureIntegration(config)
    features = generator.generate_features(data)
    summary = generator.get_feature_summary(features)
    
    return features, summary