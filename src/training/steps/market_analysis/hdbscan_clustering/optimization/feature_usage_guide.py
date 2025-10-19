"""
Feature Usage Guide for HDBSCAN Clustering System

This module provides a comprehensive guide on which features are used
in the HDBSCAN clustering pipeline and how they are integrated.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class FeatureUsage:
    """Feature usage information."""
    feature_name: str
    category: str
    source_module: str
    usage_frequency: str  # "high", "medium", "low"
    optimization_level: str  # "high", "medium", "low"
    description: str
    example_usage: str

class HDBSCANFeatureUsageGuide:
    """
    Comprehensive feature usage guide for HDBSCAN clustering system.
    
    This class provides detailed information about which features are used
    in the HDBSCAN clustering pipeline and how they are integrated.
    """
    
    def __init__(self):
        """Initialize the feature usage guide."""
        self.feature_usage = self._initialize_feature_usage()
        self.integration_patterns = self._initialize_integration_patterns()
        
    def _initialize_feature_usage(self) -> Dict[str, FeatureUsage]:
        """Initialize feature usage information."""
        return {
            # HIGH USAGE FEATURES - Core regime detection
            'price_entropy': FeatureUsage(
                feature_name="PriceEntropyGenerator",
                category="entropy",
                source_module="src.feature_generation.categories.entropy",
                usage_frequency="high",
                optimization_level="high",
                description="Price entropy for regime complexity analysis",
                example_usage="PriceEntropyGenerator(window=20).generate(data)"
            ),
            
            'volume_entropy': FeatureUsage(
                feature_name="VolumeEntropyGenerator",
                category="entropy",
                source_module="src.feature_generation.categories.entropy",
                usage_frequency="high",
                optimization_level="high",
                description="Volume entropy for market participation analysis",
                example_usage="VolumeEntropyGenerator(window=20).generate(data)"
            ),
            
            'return_entropy': FeatureUsage(
                feature_name="ReturnEntropyGenerator",
                category="entropy",
                source_module="src.feature_generation.categories.entropy",
                usage_frequency="high",
                optimization_level="high",
                description="Return entropy for volatility regime analysis",
                example_usage="ReturnEntropyGenerator(window=20).generate(data)"
            ),
            
            'shannon_entropy': FeatureUsage(
                feature_name="ShannonEntropyGenerator",
                category="entropy",
                source_module="src.feature_generation.categories.entropy",
                usage_frequency="high",
                optimization_level="high",
                description="Shannon entropy for information content analysis",
                example_usage="ShannonEntropyGenerator(window=20, q_bins=10).generate(data)"
            ),
            
            'permutation_entropy': FeatureUsage(
                feature_name="PermutationEntropyGenerator",
                category="entropy",
                source_module="src.feature_generation.categories.entropy",
                usage_frequency="high",
                optimization_level="high",
                description="Permutation entropy for pattern complexity analysis",
                example_usage="PermutationEntropyGenerator(window=20, embedding_dim=3, delay=1).generate(data)"
            ),
            
            'wavelet_energy': FeatureUsage(
                feature_name="WaveletEnergyGenerator",
                category="spectral_wavelet",
                source_module="src.feature_generation.categories.spectral_wavelet",
                usage_frequency="high",
                optimization_level="high",
                description="Wavelet energy for frequency domain analysis",
                example_usage="WaveletEnergyGenerator(window=20).generate(data)"
            ),
            
            'fractal_dimension': FeatureUsage(
                feature_name="FractalDimensionGenerator",
                category="spectral_wavelet",
                source_module="src.feature_generation.categories.spectral_wavelet",
                usage_frequency="high",
                optimization_level="high",
                description="Fractal dimension for market structure analysis",
                example_usage="FractalDimensionGenerator(window=20).generate(data)"
            ),
            
            'regime_entropy': FeatureUsage(
                feature_name="RegimeEntropyGenerator",
                category="regime",
                source_module="src.feature_generation.categories.regime_features",
                usage_frequency="high",
                optimization_level="high",
                description="Regime entropy for regime stability analysis",
                example_usage="RegimeEntropyGenerator(window=20).generate(data)"
            ),
            
            'regime_complexity': FeatureUsage(
                feature_name="RegimeComplexityGenerator",
                category="regime",
                source_module="src.feature_generation.categories.regime_features",
                usage_frequency="high",
                optimization_level="high",
                description="Regime complexity for regime transition analysis",
                example_usage="RegimeComplexityGenerator(window=20).generate(data)"
            ),
            
            'hurst_exponent': FeatureUsage(
                feature_name="HurstExponentGenerator",
                category="advanced_statistical",
                source_module="src.feature_generation.categories.advanced_statistical",
                usage_frequency="high",
                optimization_level="high",
                description="Hurst exponent for long-range dependence analysis",
                example_usage="HurstExponentGenerator(window=20).generate(data)"
            ),
            
            # MEDIUM USAGE FEATURES - Supporting regime analysis
            'rsi_entropy': FeatureUsage(
                feature_name="RSIEntropyGenerator",
                category="entropy",
                source_module="src.feature_generation.categories.entropy",
                usage_frequency="medium",
                optimization_level="medium",
                description="RSI entropy for momentum regime analysis",
                example_usage="RSIEntropyGenerator(window=20).generate(data)"
            ),
            
            'macd_entropy': FeatureUsage(
                feature_name="MACDEntropyGenerator",
                category="entropy",
                source_module="src.feature_generation.categories.entropy",
                usage_frequency="medium",
                optimization_level="medium",
                description="MACD entropy for trend regime analysis",
                example_usage="MACDEntropyGenerator(window=20).generate(data)"
            ),
            
            'bollinger_entropy': FeatureUsage(
                feature_name="BollingerBandsEntropyGenerator",
                category="entropy",
                source_module="src.feature_generation.categories.entropy",
                usage_frequency="medium",
                optimization_level="medium",
                description="Bollinger Bands entropy for volatility regime analysis",
                example_usage="BollingerBandsEntropyGenerator(window=20).generate(data)"
            ),
            
            'sample_entropy': FeatureUsage(
                feature_name="SampleEntropyGenerator",
                category="entropy",
                source_module="src.feature_generation.categories.entropy",
                usage_frequency="medium",
                optimization_level="medium",
                description="Sample entropy for irregularity analysis",
                example_usage="SampleEntropyGenerator(window=20, m=2, r=0.2).generate(data)"
            ),
            
            'lempel_ziv_complexity': FeatureUsage(
                feature_name="LempelZivComplexityGenerator",
                category="entropy",
                source_module="src.feature_generation.categories.entropy",
                usage_frequency="medium",
                optimization_level="medium",
                description="Lempel-Ziv complexity for pattern analysis",
                example_usage="LempelZivComplexityGenerator(window=20).generate(data)"
            ),
            
            'band_limited_volatility': FeatureUsage(
                feature_name="BandLimitedVolatilityGenerator",
                category="spectral_wavelet",
                source_module="src.feature_generation.categories.spectral_wavelet",
                usage_frequency="medium",
                optimization_level="medium",
                description="Band-limited volatility for frequency-specific analysis",
                example_usage="BandLimitedVolatilityGenerator(window=20).generate(data)"
            ),
            
            'cycle_length': FeatureUsage(
                feature_name="CycleLengthGenerator",
                category="spectral_wavelet",
                source_module="src.feature_generation.categories.spectral_wavelet",
                usage_frequency="medium",
                optimization_level="medium",
                description="Cycle length for periodic pattern analysis",
                example_usage="CycleLengthGenerator(window=20).generate(data)"
            ),
            
            'dfa_slopes': FeatureUsage(
                feature_name="DFASlopesGenerator",
                category="spectral_wavelet",
                source_module="src.feature_generation.categories.spectral_wavelet",
                usage_frequency="medium",
                optimization_level="medium",
                description="DFA slopes for scaling behavior analysis",
                example_usage="DFASlopesGenerator(window=20).generate(data)"
            ),
            
            'regime_fractal_dimension': FeatureUsage(
                feature_name="RegimeFractalDimensionGenerator",
                category="regime",
                source_module="src.feature_generation.categories.regime_features",
                usage_frequency="medium",
                optimization_level="medium",
                description="Regime fractal dimension for regime structure analysis",
                example_usage="RegimeFractalDimensionGenerator(window=20).generate(data)"
            ),
            
            'regime_hurst_exponent': FeatureUsage(
                feature_name="RegimeHurstExponentGenerator",
                category="regime",
                source_module="src.feature_generation.categories.regime_features",
                usage_frequency="medium",
                optimization_level="medium",
                description="Regime Hurst exponent for regime persistence analysis",
                example_usage="RegimeHurstExponentGenerator(window=20).generate(data)"
            ),
            
            'regime_memory_strength': FeatureUsage(
                feature_name="RegimeMemoryStrengthGenerator",
                category="regime",
                source_module="src.feature_generation.categories.regime_features",
                usage_frequency="medium",
                optimization_level="medium",
                description="Regime memory strength for regime persistence analysis",
                example_usage="RegimeMemoryStrengthGenerator(window=20).generate(data)"
            ),
            
            'jump_indicators': FeatureUsage(
                feature_name="JumpIndicatorsGenerator",
                category="advanced_statistical",
                source_module="src.feature_generation.categories.advanced_statistical",
                usage_frequency="medium",
                optimization_level="medium",
                description="Jump indicators for discontinuity analysis",
                example_usage="JumpIndicatorsGenerator(window=20).generate(data)"
            ),
            
            'trend_persistence': FeatureUsage(
                feature_name="TrendPersistenceGenerator",
                category="advanced_statistical",
                source_module="src.feature_generation.categories.advanced_statistical",
                usage_frequency="medium",
                optimization_level="medium",
                description="Trend persistence for trend strength analysis",
                example_usage="TrendPersistenceGenerator(window=20).generate(data)"
            ),
            
            # LOW USAGE FEATURES - Supplementary analysis
            'cross_asset_entropy': FeatureUsage(
                feature_name="CrossAssetEntropyGenerator",
                category="entropy",
                source_module="src.feature_generation.categories.entropy",
                usage_frequency="low",
                optimization_level="low",
                description="Cross-asset entropy for correlation analysis",
                example_usage="CrossAssetEntropyGenerator(window=20).generate(data)"
            ),
            
            'entropy_rate': FeatureUsage(
                feature_name="EntropyRateGenerator",
                category="entropy",
                source_module="src.feature_generation.categories.entropy",
                usage_frequency="low",
                optimization_level="low",
                description="Entropy rate for information flow analysis",
                example_usage="EntropyRateGenerator(window=20).generate(data)"
            ),
            
            'spectral_entropy': FeatureUsage(
                feature_name="SpectralEntropyGenerator",
                category="entropy",
                source_module="src.feature_generation.categories.entropy",
                usage_frequency="low",
                optimization_level="low",
                description="Spectral entropy for frequency domain analysis",
                example_usage="SpectralEntropyGenerator(window=20).generate(data)"
            )
        }
    
    def _initialize_integration_patterns(self) -> Dict[str, List[str]]:
        """Initialize integration patterns for feature usage."""
        return {
            'entropy_features': [
                "PriceEntropyGenerator", "VolumeEntropyGenerator", "ReturnEntropyGenerator",
                "ShannonEntropyGenerator", "PermutationEntropyGenerator", "SampleEntropyGenerator"
            ],
            'spectral_features': [
                "WaveletEnergyGenerator", "FractalDimensionGenerator", "BandLimitedVolatilityGenerator",
                "CycleLengthGenerator", "DFASlopesGenerator"
            ],
            'regime_features': [
                "RegimeEntropyGenerator", "RegimeComplexityGenerator", "RegimeFractalDimensionGenerator",
                "RegimeHurstExponentGenerator", "RegimeMemoryStrengthGenerator"
            ],
            'advanced_statistical_features': [
                "HurstExponentGenerator", "JumpIndicatorsGenerator", "TrendPersistenceGenerator"
            ],
            'supporting_features': [
                "RSIEntropyGenerator", "MACDEntropyGenerator", "BollingerBandsEntropyGenerator",
                "LempelZivComplexityGenerator", "CrossAssetEntropyGenerator"
            ]
        }
    
    def get_feature_usage_by_frequency(self, frequency: str) -> List[FeatureUsage]:
        """Get features by usage frequency."""
        return [
            usage for usage in self.feature_usage.values()
            if usage.usage_frequency == frequency
        ]
    
    def get_feature_usage_by_optimization_level(self, level: str) -> List[FeatureUsage]:
        """Get features by optimization level."""
        return [
            usage for usage in self.feature_usage.values()
            if usage.optimization_level == level
        ]
    
    def get_integration_example(self) -> str:
        """Get example of how to integrate features in HDBSCAN clustering."""
        return '''
# Example: HDBSCAN Clustering with Feature Integration

from src.training.steps.market_analysis.hdbscan_clustering.optimization import (
    create_enhanced_hdbscan_integration
)
from src.feature_generation.categories.entropy import create_default_entropy_generators
from src.feature_generation.categories.spectral_wavelet import create_default_spectral_wavelet_generators
from src.feature_generation.categories.regime_features import create_default_regime_generators

# Initialize enhanced HDBSCAN integration
hdbscan_integration = create_enhanced_hdbscan_integration(
    memory_optimization=True,
    hyperparameter_optimization=True,
    vectorized_processing=True,
    enable_vectorbt=True,
    enable_parallel=True,
    max_memory_gb=8.0,
    optimization_strategy="hybrid",
    n_trials=50,
    primary_metric="silhouette"
)

# Process data with optimizations
features_df = hdbscan_integration.process_data_with_optimizations(data)

# Optimize hyperparameters
optimization_results = hdbscan_integration.optimize_hyperparameters(features_df)

# Perform optimized clustering
cluster_labels, clustering_info = hdbscan_integration.perform_optimized_clustering(
    features_df, 
    hdbscan_params=optimization_results.get('best_params', {})
)

# Get performance statistics
performance_stats = hdbscan_integration.get_comprehensive_performance_stats()
print(f"Clustering completed: {clustering_info['n_clusters']} clusters found")
print(f"Performance: {performance_stats['total_processing_time']:.2f}s")
'''
    
    def get_feature_imports_for_hdbscan(self) -> str:
        """Get feature imports for HDBSCAN clustering."""
        return '''
# Feature imports for HDBSCAN clustering system

# Core entropy features
from src.feature_generation.categories.entropy import (
    PriceEntropyGenerator, VolumeEntropyGenerator, ReturnEntropyGenerator,
    ShannonEntropyGenerator, PermutationEntropyGenerator, SampleEntropyGenerator,
    LempelZivComplexityGenerator, create_default_entropy_generators
)

# Spectral and wavelet features
from src.feature_generation.categories.spectral_wavelet import (
    WaveletEnergyGenerator, FractalDimensionGenerator, BandLimitedVolatilityGenerator,
    CycleLengthGenerator, DFASlopesGenerator, create_default_spectral_wavelet_generators
)

# Regime features
from src.feature_generation.categories.regime_features import (
    RegimeEntropyGenerator, RegimeComplexityGenerator, RegimeFractalDimensionGenerator,
    RegimeHurstExponentGenerator, RegimeMemoryStrengthGenerator, create_default_regime_generators
)

# Advanced statistical features
from src.feature_generation.categories.advanced_statistical import (
    HurstExponentGenerator, JumpIndicatorsGenerator, TrendPersistenceGenerator,
    create_default_advanced_statistical_generators
)

# Supporting features
from src.feature_generation.categories.entropy import (
    RSIEntropyGenerator, MACDEntropyGenerator, BollingerBandsEntropyGenerator,
    CrossAssetEntropyGenerator, EntropyRateGenerator, SpectralEntropyGenerator
)
'''
    
    def get_optimization_recommendations(self) -> Dict[str, List[str]]:
        """Get optimization recommendations for feature usage."""
        return {
            'high_usage_features': [
                "Use VectorBT optimization for all rolling operations",
                "Enable GPU acceleration for large datasets",
                "Implement memory-efficient chunked processing",
                "Use parallel processing for independent operations",
                "Cache results for repeated calculations"
            ],
            'medium_usage_features': [
                "Use VectorBT optimization for rolling operations",
                "Enable memory optimization",
                "Consider parallel processing for large datasets",
                "Use caching for repeated calculations"
            ],
            'low_usage_features': [
                "Use standard pandas operations",
                "Enable basic memory optimization",
                "Consider caching for repeated calculations",
                "Use lazy evaluation when possible"
            ]
        }
    
    def get_feature_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive feature usage summary."""
        high_usage = len(self.get_feature_usage_by_frequency('high'))
        medium_usage = len(self.get_feature_usage_by_frequency('medium'))
        low_usage = len(self.get_feature_usage_by_frequency('low'))
        
        high_optimization = len(self.get_feature_usage_by_optimization_level('high'))
        medium_optimization = len(self.get_feature_usage_by_optimization_level('medium'))
        low_optimization = len(self.get_feature_usage_by_optimization_level('low'))
        
        return {
            'total_features': len(self.feature_usage),
            'usage_frequency': {
                'high': high_usage,
                'medium': medium_usage,
                'low': low_usage
            },
            'optimization_level': {
                'high': high_optimization,
                'medium': medium_optimization,
                'low': low_optimization
            },
            'integration_patterns': self.integration_patterns,
            'recommendations': self.get_optimization_recommendations()
        }

# Convenience function
def get_hdbscan_feature_usage_guide() -> HDBSCANFeatureUsageGuide:
    """Get the HDBSCAN feature usage guide."""
    return HDBSCANFeatureUsageGuide()

# Example usage and feature summary
if __name__ == "__main__":
    guide = get_hdbscan_feature_usage_guide()
    summary = guide.get_feature_usage_summary()
    
    print("=== HDBSCAN Clustering Feature Usage Guide ===")
    print(f"Total Features: {summary['total_features']}")
    print(f"High Usage Features: {summary['usage_frequency']['high']}")
    print(f"Medium Usage Features: {summary['usage_frequency']['medium']}")
    print(f"Low Usage Features: {summary['usage_frequency']['low']}")
    
    print("\n=== High Usage Features ===")
    high_usage = guide.get_feature_usage_by_frequency('high')
    for feature in high_usage:
        print(f"- {feature.feature_name}: {feature.description}")
    
    print("\n=== Integration Example ===")
    print(guide.get_integration_example())
