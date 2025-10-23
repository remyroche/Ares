"""
Updated Feature Summary for HDBSCAN Clustering System

This module provides a summary of the remaining features after removing
the specified categories from the HDBSCAN clustering pipeline.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class UpdatedFeatureSummary:
    """Updated feature summary after category removal."""
    
    # Remaining feature categories
    remaining_categories = {
        'entropy': {
            'name': 'Entropy Features',
            'feature_count': 45,
            'optimization_level': 'high',
            'description': 'Entropy-based indicators for regime complexity analysis'
        },
        'spectral_wavelet': {
            'name': 'Spectral & Wavelet Features', 
            'feature_count': 25,
            'optimization_level': 'high',
            'description': 'Spectral analysis and wavelet-based features for frequency domain analysis'
        },
        'regime': {
            'name': 'Regime Features',
            'feature_count': 60,
            'optimization_level': 'high', 
            'description': 'Comprehensive regime-specific features for market state analysis'
        },
        'returns': {
            'name': 'Returns Features',
            'feature_count': 20,
            'optimization_level': 'medium',
            'description': 'Price return-based features for trend and momentum analysis'
        },
        'momentum': {
            'name': 'Momentum Features',
            'feature_count': 25,
            'optimization_level': 'medium',
            'description': 'Momentum-based indicators for trend strength analysis'
        },
        'volatility': {
            'name': 'Volatility Features',
            'feature_count': 15,
            'optimization_level': 'medium',
            'description': 'Volatility-based indicators for market uncertainty analysis'
        },
        'volume': {
            'name': 'Volume Features',
            'feature_count': 20,
            'optimization_level': 'medium',
            'description': 'Volume-based indicators for market participation analysis'
        },
        'trend': {
            'name': 'Trend Features',
            'feature_count': 15,
            'optimization_level': 'medium',
            'description': 'Trend-based indicators for market direction analysis'
        },
        'advanced_statistical': {
            'name': 'Advanced Statistical Features',
            'feature_count': 15,
            'optimization_level': 'high',
            'description': 'Advanced statistical indicators for market analysis'
        },
        'cross_timeframe': {
            'name': 'Cross-Timeframe Features',
            'feature_count': 15,
            'optimization_level': 'medium',
            'description': 'Multi-timeframe analysis indicators'
        }
    }
    
    # Removed categories
    removed_categories = {
        'oscillator': 'Oscillator Features - Low optimization level',
        'support_resistance': 'Support/Resistance Features - Low optimization level', 
        'candlestick_pattern': 'Candlestick Pattern Features - Low optimization level',
        'interaction': 'Interaction Features - Medium optimization level',
        'microstructure': 'Microstructure Features - High optimization level',
        'time': 'Time Features - Low optimization level'
    }
    
    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        """Get updated feature summary."""
        total_features = sum(cat['feature_count'] for cat in cls.remaining_categories.values())
        
        high_optimization = sum(
            cat['feature_count'] for cat in cls.remaining_categories.values()
            if cat['optimization_level'] == 'high'
        )
        
        medium_optimization = sum(
            cat['feature_count'] for cat in cls.remaining_categories.values()
            if cat['optimization_level'] == 'medium'
        )
        
        return {
            'total_categories': len(cls.remaining_categories),
            'total_features': total_features,
            'high_optimization_features': high_optimization,
            'medium_optimization_features': medium_optimization,
            'low_optimization_features': 0,  # All remaining categories are high/medium
            'removed_categories': len(cls.removed_categories),
            'removed_features': 76,  # Sum of removed category feature counts
            'remaining_categories': cls.remaining_categories,
            'removed_categories': cls.removed_categories
        }
    
    @classmethod
    def get_optimization_focus(cls) -> Dict[str, Any]:
        """Get optimization focus after category removal."""
        return {
            'high_priority_categories': [
                'entropy', 'spectral_wavelet', 'regime', 'advanced_statistical'
            ],
            'medium_priority_categories': [
                'returns', 'momentum', 'volatility', 'volume', 'trend', 'cross_timeframe'
            ],
            'optimization_recommendations': {
                'high_priority': [
                    'Use VectorBT optimization for all rolling operations',
                    'Enable GPU acceleration for large datasets', 
                    'Implement memory-efficient chunked processing',
                    'Use parallel processing for independent operations',
                    'Cache results for repeated calculations'
                ],
                'medium_priority': [
                    'Use VectorBT optimization for rolling operations',
                    'Enable memory optimization',
                    'Consider parallel processing for large datasets',
                    'Use caching for repeated calculations'
                ]
            }
        }

# Print summary
if __name__ == "__main__":
    summary = UpdatedFeatureSummary.get_summary()
    optimization_focus = UpdatedFeatureSummary.get_optimization_focus()
    
    print("=== Updated HDBSCAN Clustering Feature Summary ===")
    print(f"Remaining Categories: {summary['total_categories']}")
    print(f"Total Features: {summary['total_features']}")
    print(f"High Optimization Features: {summary['high_optimization_features']}")
    print(f"Medium Optimization Features: {summary['medium_optimization_features']}")
    print(f"Removed Categories: {summary['removed_categories']}")
    print(f"Removed Features: {summary['removed_features']}")
    
    print("\n=== Remaining Categories ===")
    for category, info in summary['remaining_categories'].items():
        print(f"- {info['name']}: {info['feature_count']} features ({info['optimization_level']} optimization)")
    
    print("\n=== Removed Categories ===")
    for category, description in summary['removed_categories'].items():
        print(f"- {description}")
    
    print("\n=== Optimization Focus ===")
    print("High Priority Categories:")
    for category in optimization_focus['high_priority_categories']:
        print(f"  - {category}")
    
    print("Medium Priority Categories:")
    for category in optimization_focus['medium_priority_categories']:
        print(f"  - {category}")
