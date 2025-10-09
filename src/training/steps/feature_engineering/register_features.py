"""
Feature Registration Script

This script registers all the new advanced features with the feature bank system
for integration with the framework's feature generation and lookback optimization.
"""

import logging
from typing import Dict, List, Type
from src.feature_generation.core.feature_bank import FeatureBank
from src.feature_generation.core.feature_registry import FeatureRegistry

# Import the new feature generators
from src.training.steps.feature_engineering.price_action.bar_efficiency_ratio import BarEfficiencyRatioGenerator
from src.training.steps.feature_engineering.price_action.close_location_value import CloseLocationValueGenerator
from src.training.steps.feature_engineering.volatility.atr_volatility_ratio import ATRVolatilityRatioGenerator
from src.training.steps.feature_engineering.trend.trend_coherence import TrendCoherenceGenerator

logger = logging.getLogger(__name__)

def register_advanced_features(feature_bank: FeatureBank) -> None:
    """
    Register all advanced features with the feature bank.
    
    Args:
        feature_bank: FeatureBank instance to register features with
    """
    logger.info("🔧 Registering advanced features with feature bank")
    
    # Define feature generators with their configurations
    feature_generators = [
        # Price Action Features
        {
            'generator_class': BarEfficiencyRatioGenerator,
            'name': 'bar_efficiency_ratio',
            'lookback': 3,
            'parameters': {
                'high_efficiency_threshold': 0.6,
                'low_efficiency_threshold': 0.3,
                'include_raw_efficiency': True,
                'include_rolling_efficiency': True,
                'include_efficiency_grade': True
            }
        },
        {
            'generator_class': CloseLocationValueGenerator,
            'name': 'close_location_value',
            'lookback': 8,
            'parameters': {
                'positive_threshold': 0.2,
                'negative_threshold': -0.2,
                'volatility_threshold': 0.5,
                'include_raw_clv': True,
                'include_rolling_clv': True,
                'include_clv_volatility': True,
                'include_clv_grade': True,
                'include_clv_class': True
            }
        },
        {
            'generator_class': ATRVolatilityRatioGenerator,
            'name': 'atr_volatility_ratio',
            'lookback': 4,
            'parameters': {
                'long_window': 20,
                'high_ratio_threshold': 1.5,
                'include_atr_short': True,
                'include_atr_long': True,
                'include_atr_ratio': True,
                'include_atr_grade': True,
                'include_atr_class': True
            }
        },
        {
            'generator_class': TrendCoherenceGenerator,
            'name': 'trend_coherence',
            'lookback': 8,
            'parameters': {
                'ema_period': 12,
                'min_direction_consistency': 0.6,
                'min_slope_threshold': 0.001,
                'include_direction_consistency': True,
                'include_ema_slope': True,
                'include_trend_coherence_grade': True,
                'include_trend_class': True
            }
        }
    ]
    
    # Register each feature generator
    for feature_config in feature_generators:
        try:
            generator_class = feature_config['generator_class']
            lookback = feature_config['lookback']
            parameters = feature_config.get('parameters', {})
            
            # Create generator instance
            generator = generator_class(lookback=lookback, **parameters)
            
            # Register with feature bank
            feature_bank.register_generator(generator)
            
            logger.info(f"✅ Registered {feature_config['name']} with lookback {lookback}")
            
        except Exception as e:
            logger.error(f"❌ Failed to register {feature_config['name']}: {e}")
            raise
    
    logger.info("🎉 Advanced features registration completed")


def get_advanced_feature_generators() -> Dict[str, Type]:
    """
    Get dictionary of advanced feature generator classes.
    
    Returns:
        Dictionary mapping feature names to generator classes
    """
    return {
        'bar_efficiency_ratio': BarEfficiencyRatioGenerator,
        'close_location_value': CloseLocationValueGenerator,
        'atr_volatility_ratio': ATRVolatilityRatioGenerator,
        'trend_coherence': TrendCoherenceGenerator
    }


def create_feature_bank_with_advanced_features() -> FeatureBank:
    """
    Create a feature bank with all advanced features registered.
    
    Returns:
        FeatureBank instance with advanced features
    """
    from src.feature_generation.core.feature_bank import FeatureBankConfig
    
    # Create feature bank with optimized configuration
    config = FeatureBankConfig(
        enable_matrix_operations=True,
        enable_gpu_acceleration=True,
        enable_lookback_optimization=True,
        enable_parallel_processing=True,
        max_workers=4,
        chunk_size=1000,
        memory_efficient=True,
        cache_results=True,
        default_lookback=10
    )
    
    feature_bank = FeatureBank(config)
    
    # Register advanced features
    register_advanced_features(feature_bank)
    
    return feature_bank


# Convenience function for quick access
def get_advanced_features_by_category() -> Dict[str, List[str]]:
    """
    Get advanced features organized by category.
    
    Returns:
        Dictionary mapping categories to lists of feature names
    """
    return {
        'price_action': [
            'bar_efficiency_ratio',
            'close_location_value'
        ],
        'volatility': [
            'atr_volatility_ratio'
        ],
        'trend': [
            'trend_coherence'
        ]
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create feature bank with advanced features
    feature_bank = create_feature_bank_with_advanced_features()
    
    # Print registered features
    print("📊 Registered advanced features:")
    for category, features in get_advanced_features_by_category().items():
        print(f"  {category}: {features}")
    
    print("✅ Feature bank setup completed")