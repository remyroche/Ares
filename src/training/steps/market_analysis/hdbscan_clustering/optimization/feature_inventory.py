"""
Feature Inventory for HDBSCAN Clustering System

This module provides a comprehensive inventory of all features used in the
HDBSCAN clustering pipeline, organized by category and source.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class FeatureCategory:
    """Feature category information."""
    name: str
    description: str
    source_module: str
    feature_count: int
    key_features: List[str]
    optimization_level: str  # "high", "medium", "low"

class HDBSCANFeatureInventory:
    """
    Comprehensive feature inventory for HDBSCAN clustering system.
    
    This class provides a complete overview of all features used in the
    HDBSCAN clustering pipeline, organized by category and source.
    """
    
    def __init__(self):
        """Initialize the feature inventory."""
        self.feature_categories = self._initialize_feature_categories()
        self.total_features = sum(cat.feature_count for cat in self.feature_categories.values())
        
    def _initialize_feature_categories(self) -> Dict[str, FeatureCategory]:
        """Initialize feature categories with comprehensive information."""
        return {
            # ENTROPY FEATURES - High optimization level
            'entropy': FeatureCategory(
                name="Entropy Features",
                description="Entropy-based indicators for regime complexity analysis",
                source_module="src.feature_generation.categories.entropy",
                feature_count=45,  # Estimated from create_default_entropy_generators()
                key_features=[
                    "PriceEntropyGenerator", "VolumeEntropyGenerator", "ReturnEntropyGenerator",
                    "PriceEntropyMAGenerator", "VolumeEntropyMAGenerator", "ReturnEntropyMAGenerator",
                    "HighLowEntropyGenerator", "VolatilityEntropyGenerator", "MomentumEntropyGenerator",
                    "RSIEntropyGenerator", "MACDEntropyGenerator", "BollingerBandsEntropyGenerator",
                    "CrossAssetEntropyGenerator", "RegimeEntropyGenerator",
                    "ShannonEntropyGenerator", "PermutationEntropyGenerator", "SampleEntropyGenerator",
                    "LempelZivComplexityGenerator", "EntropyRateGenerator", "SpectralEntropyGenerator"
                ],
                optimization_level="high"
            ),
            
            # SPECTRAL & WAVELET FEATURES - High optimization level
            'spectral_wavelet': FeatureCategory(
                name="Spectral & Wavelet Features",
                description="Spectral analysis and wavelet-based features for frequency domain analysis",
                source_module="src.feature_generation.categories.spectral_wavelet",
                feature_count=25,  # Estimated from create_default_spectral_wavelet_generators()
                key_features=[
                    "WaveletEnergyGenerator", "BandLimitedVolatilityGenerator", "CycleLengthGenerator",
                    "FractalDimensionGenerator", "DFASlopesGenerator", "SpectralFeatureGenerator",
                    "WaveletFeatureGenerator", "DetrendedFluctuationAnalysisGenerator",
                    "VectorBTSpectralFeatureGenerator", "VectorBTSpectralWaveletBatchGenerator"
                ],
                optimization_level="high"
            ),
            
            # REGIME FEATURES - High optimization level
            'regime': FeatureCategory(
                name="Regime Features",
                description="Comprehensive regime-specific features for market state analysis",
                source_module="src.feature_generation.categories.regime_features",
                feature_count=60,  # Estimated from regime_features.py
                key_features=[
                    "StatisticalRegimeFeatureGenerator", "StructuralTrendRegimeFeatureGenerator",
                    "VolatilityRegimeFeatureGenerator", "VolumeRegimeFeatureGenerator",
                    "AdvancedRegimeFeatureGenerator", "RegimeEntropyGenerator",
                    "RegimeComplexityGenerator", "RegimeFractalDimensionGenerator",
                    "RegimeHurstExponentGenerator", "RegimeMemoryStrengthGenerator"
                ],
                optimization_level="high"
            ),
            
            # RETURNS FEATURES - Medium optimization level
            'returns': FeatureCategory(
                name="Returns Features",
                description="Price return-based features for trend and momentum analysis",
                source_module="src.feature_generation.categories.returns",
                feature_count=20,  # Estimated from create_default_returns_generators()
                key_features=[
                    "LogReturnsGenerator", "SimpleReturnsGenerator", "CumulativeReturnsGenerator",
                    "RollingReturnsGenerator", "ReturnsVolatilityGenerator", "ReturnsSkewnessGenerator",
                    "ReturnsKurtosisGenerator", "SharpeRatioGenerator", "AdvancedCumulativeReturnsGenerator",
                    "RollingZScoreReturnsGenerator", "ARCoefficientsGenerator", "LjungBoxTestGenerator"
                ],
                optimization_level="medium"
            ),
            
            # MOMENTUM FEATURES - Medium optimization level
            'momentum': FeatureCategory(
                name="Momentum Features",
                description="Momentum-based indicators for trend strength analysis",
                source_module="src.feature_generation.categories.momentum",
                feature_count=25,  # Estimated from create_default_momentum_generators()
                key_features=[
                    "RSIGenerator", "MACDGenerator", "StochasticGenerator", "WilliamsRGenerator",
                    "MomentumOscillatorGenerator", "RateOfChangeGenerator", "AdvancedMomentumGenerator",
                    "PriceAccelerationGenerator", "VolumeMomentumGenerator", "MomentumEndpointsGenerator",
                    "MACDDeltaGenerator", "RSIZScoreGenerator", "StochasticKDGenerator", "DonchianChannelGenerator"
                ],
                optimization_level="medium"
            ),
            
            # VOLATILITY FEATURES - Medium optimization level
            'volatility': FeatureCategory(
                name="Volatility Features",
                description="Volatility-based indicators for market uncertainty analysis",
                source_module="src.feature_generation.categories.volatility",
                feature_count=15,  # Estimated from create_default_volatility_generators()
                key_features=[
                    "BollingerBandsGenerator", "ATRGenerator", "GARCHGenerator", "VolatilityClusteringGenerator",
                    "VolatilityPersistenceGenerator", "VolatilityRegimeGenerator", "VolatilityTransitionsGenerator"
                ],
                optimization_level="medium"
            ),
            
            # VOLUME FEATURES - Medium optimization level
            'volume': FeatureCategory(
                name="Volume Features",
                description="Volume-based indicators for market participation analysis",
                source_module="src.feature_generation.categories.volume",
                feature_count=20,  # Estimated from create_default_volume_generators()
                key_features=[
                    "VolumeZScoreGenerator", "VolumeMARatiosGenerator", "CMFGenerator",
                    "VWAPDeviationsGenerator", "OrderFlowImbalanceGenerator", "VolumeVolatilityElasticityGenerator",
                    "VolumeAccelerationGenerator", "VolumeMomentumGenerator", "VolumeRegimeGenerator"
                ],
                optimization_level="medium"
            ),
            
            # TREND FEATURES - Medium optimization level
            'trend': FeatureCategory(
                name="Trend Features",
                description="Trend-based indicators for market direction analysis",
                source_module="src.feature_generation.categories.trend",
                feature_count=15,  # Estimated from create_default_trend_generators()
                key_features=[
                    "MovingAverageGenerator", "TrendStrengthGenerator", "TrendConsistencyGenerator",
                    "TrendPersistenceGenerator", "TrendRegimeGenerator", "TrendTransitionsGenerator"
                ],
                optimization_level="medium"
            ),
            
            
            # ADVANCED STATISTICAL FEATURES - High optimization level
            'advanced_statistical': FeatureCategory(
                name="Advanced Statistical Features",
                description="Advanced statistical indicators for market analysis",
                source_module="src.feature_generation.categories.advanced_statistical",
                feature_count=15,  # Estimated from create_default_advanced_statistical_generators()
                key_features=[
                    "HurstExponentGenerator", "JumpIndicatorsGenerator", "CVaRGenerator",
                    "MaxDrawdownGenerator", "RollingSkewnessKurtosisGenerator", "TrendPersistenceGenerator"
                ],
                optimization_level="high"
            ),
            
            
            # CROSS-TIMEFRAME FEATURES - Medium optimization level
            'cross_timeframe': FeatureCategory(
                name="Cross-Timeframe Features",
                description="Multi-timeframe analysis indicators",
                source_module="src.feature_generation.categories.cross_timeframe",
                feature_count=15,  # Estimated from create_default_cross_timeframe_generators()
                key_features=[
                    "CrossTimeframeMomentumGenerator", "CrossTimeframeVolatilityGenerator",
                    "CrossTimeframeVolumeGenerator", "CrossTimeframeTrendGenerator",
                    "CrossTimeframeHighLowGenerator", "CrossTimeframeRatioGenerator",
                    "CrossTimeframeCorrelationGenerator", "CrossTimeframeDivergenceGenerator"
                ],
                optimization_level="medium"
            ),
            
        }
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get comprehensive feature summary."""
        return {
            'total_categories': len(self.feature_categories),
            'total_features': self.total_features,
            'high_optimization_features': sum(
                cat.feature_count for cat in self.feature_categories.values() 
                if cat.optimization_level == "high"
            ),
            'medium_optimization_features': sum(
                cat.feature_count for cat in self.feature_categories.values() 
                if cat.optimization_level == "medium"
            ),
            'low_optimization_features': sum(
                cat.feature_count for cat in self.feature_categories.values() 
                if cat.optimization_level == "low"
            ),
            'categories': self.feature_categories
        }
    
    def get_features_by_optimization_level(self, level: str) -> List[FeatureCategory]:
        """Get features by optimization level."""
        return [
            cat for cat in self.feature_categories.values() 
            if cat.optimization_level == level
        ]
    
    def get_high_priority_features(self) -> List[str]:
        """Get high-priority features for HDBSCAN clustering."""
        high_priority = []
        
        # Entropy features (most important for regime detection)
        high_priority.extend([
            "PriceEntropyGenerator", "VolumeEntropyGenerator", "ReturnEntropyGenerator",
            "ShannonEntropyGenerator", "PermutationEntropyGenerator", "SampleEntropyGenerator"
        ])
        
        # Spectral features (important for frequency analysis)
        high_priority.extend([
            "WaveletEnergyGenerator", "FractalDimensionGenerator", "DFASlopesGenerator"
        ])
        
        # Regime features (core regime analysis)
        high_priority.extend([
            "RegimeEntropyGenerator", "RegimeComplexityGenerator", "RegimeFractalDimensionGenerator"
        ])
        
        # Advanced statistical features (market structure)
        high_priority.extend([
            "HurstExponentGenerator", "JumpIndicatorsGenerator", "TrendPersistenceGenerator"
        ])
        
        return high_priority
    
    def get_feature_imports(self) -> Dict[str, List[str]]:
        """Get feature imports organized by category."""
        imports = {}
        
        for category_name, category in self.feature_categories.items():
            imports[category_name] = [
                f"from {category.source_module} import {feature}"
                for feature in category.key_features
            ]
        
        return imports
    
    def get_optimization_recommendations(self) -> Dict[str, List[str]]:
        """Get optimization recommendations for each category."""
        recommendations = {}
        
        for category_name, category in self.feature_categories.items():
            if category.optimization_level == "high":
                recommendations[category_name] = [
                    "Use VectorBT optimization for rolling operations",
                    "Enable GPU acceleration for large datasets",
                    "Implement memory-efficient chunked processing",
                    "Use parallel processing for independent operations"
                ]
            elif category.optimization_level == "medium":
                recommendations[category_name] = [
                    "Use VectorBT optimization for rolling operations",
                    "Enable memory optimization",
                    "Consider parallel processing for large datasets"
                ]
            else:  # low
                recommendations[category_name] = [
                    "Use standard pandas operations",
                    "Enable basic memory optimization",
                    "Consider caching for repeated calculations"
                ]
        
        return recommendations

# Convenience function
def get_hdbscan_feature_inventory() -> HDBSCANFeatureInventory:
    """Get the HDBSCAN feature inventory."""
    return HDBSCANFeatureInventory()

# Example usage and feature summary
if __name__ == "__main__":
    inventory = get_hdbscan_feature_inventory()
    summary = inventory.get_feature_summary()
    
    print("=== HDBSCAN Clustering Feature Inventory ===")
    print(f"Total Categories: {summary['total_categories']}")
    print(f"Total Features: {summary['total_features']}")
    print(f"High Optimization Features: {summary['high_optimization_features']}")
    print(f"Medium Optimization Features: {summary['medium_optimization_features']}")
    print(f"Low Optimization Features: {summary['low_optimization_features']}")
    
    print("\n=== High Priority Features ===")
    high_priority = inventory.get_high_priority_features()
    for feature in high_priority:
        print(f"- {feature}")
    
    print("\n=== Optimization Recommendations ===")
    recommendations = inventory.get_optimization_recommendations()
    for category, recs in recommendations.items():
        print(f"\n{category.upper()}:")
        for rec in recs:
            print(f"  - {rec}")
