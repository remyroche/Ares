"""
SR Feature Integration Example

This example demonstrates how to use the SR feature integration system
to add SR-specific features to existing feature sets for ML training.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from src.utils.ml_common.sr_feature_integration import SRFeatureIntegration

async def example_sr_feature_integration():
    """Example of integrating SR features into existing feature pipeline."""
    
    # Mock existing features from feature engineering pipeline
    existing_features = {
        'price_sma_20': 50000.0,
        'price_sma_50': 49500.0,
        'rsi_14': 65.5,
        'macd': 0.002,
        'volume_ratio': 1.2,
        'volatility_20': 0.015,
        'momentum_10': 0.03,
        'bollinger_position': 0.7
    }
    
    # Mock SR levels from market analysis pipeline
    sr_levels = [
        {
            'id': 'support_1',
            'price': 48500.0,
            'level_type': 'support',
            'strength': 0.85,
            'touch_count': 5,
            'volume': 1000000
        },
        {
            'id': 'resistance_1',
            'price': 51500.0,
            'level_type': 'resistance',
            'strength': 0.75,
            'touch_count': 3,
            'volume': 800000
        },
        {
            'id': 'support_2',
            'price': 47800.0,
            'level_type': 'support',
            'strength': 0.92,
            'touch_count': 8,
            'volume': 1200000
        },
        {
            'id': 'resistance_2',
            'price': 52200.0,
            'level_type': 'resistance',
            'strength': 0.68,
            'touch_count': 2,
            'volume': 600000
        }
    ]
    
    # Mock market data
    market_data = pd.DataFrame({
        'close': [50000.0, 50100.0, 49900.0, 50050.0, 50120.0],
        'volume': [1000000, 1100000, 950000, 1050000, 1080000]
    })
    
    # Mock pipeline state
    pipeline_state = {
        'features': existing_features,
        'sr_levels': sr_levels,
        'market_data': market_data
    }
    
    # Initialize SR feature integration
    sr_config = {
        'sr_features': {
            'enabled': True,
            'proximity_threshold': 0.05,
            'strength_weights': {
                'touch_count': 0.4,
                'volume_confirmation': 0.3,
                'time_decay': 0.2,
                'confluence': 0.1
            }
        }
    }
    
    sr_integration = SRFeatureIntegration(sr_config)
    
    print("🔧 SR Feature Integration Example")
    print("=" * 50)
    
    # Show original features
    print(f"📊 Original features: {len(existing_features)}")
    for key, value in existing_features.items():
        print(f"   {key}: {value}")
    
    print("\n📈 SR Levels:")
    for level in sr_levels:
        print(f"   {level['level_type']} at {level['price']} (strength: {level['strength']})")
    
    # Integrate SR features
    enhanced_features = sr_integration.integrate_sr_features_into_pipeline(
        existing_features=existing_features,
        pipeline_state=pipeline_state
    )
    
    # Show enhanced features
    print(f"\n✅ Enhanced features: {len(enhanced_features)}")
    
    # Show only SR features
    sr_features = {k: v for k, v in enhanced_features.items() if k.startswith('sr_')}
    print(f"\n🎯 SR Features Added: {len(sr_features)}")
    for key, value in sr_features.items():
        print(f"   {key}: {value:.4f}")
    
    # Show feature names for reference
    print(f"\n📋 Available SR Feature Names:")
    feature_names = sr_integration.get_sr_feature_names()
    for name in feature_names:
        print(f"   {name}")
    
    return enhanced_features

async def example_individual_feature_extraction():
    """Example of extracting individual SR features."""
    
    # Mock SR levels
    sr_levels = [
        {'price': 48500.0, 'level_type': 'support', 'strength': 0.85},
        {'price': 51500.0, 'level_type': 'resistance', 'strength': 0.75},
        {'price': 47800.0, 'level_type': 'support', 'strength': 0.92}
    ]
    
    current_price = 50000.0
    
    # Initialize SR feature integration
    sr_integration = SRFeatureIntegration()
    
    print("\n🔍 Individual Feature Extraction Example")
    print("=" * 50)
    
    # Extract proximity features
    proximity_features = sr_integration.extract_sr_proximity_features(
        current_price=current_price,
        sr_levels=sr_levels,
        previous_balance=None  # No previous data for this example
    )
    
    print("📍 Proximity Features:")
    for key, value in proximity_features.items():
        print(f"   {key}: {value:.4f}")
    
    # Extract strength features
    strength_features = sr_integration.extract_sr_strength_features(sr_levels, current_price)
    
    # Extract trading features
    trading_features = sr_integration.extract_sr_trading_features(sr_levels, current_price, market_data)
    
    print("\n💪 Strength Features:")
    for key, value in strength_features.items():
        print(f"   {key}: {value:.4f}")
    
    print("\n📈 Trading Features:")
    for key, value in trading_features.items():
        print(f"   {key}: {value:.4f}")
    
    print(f"\n📊 Total SR Features: {len(proximity_features) + len(strength_features) + len(trading_features)}")
    print("   - Proximity Features: 11")
    print("   - Strength Features: 3") 
    print("   - Trading Features: 4")

if __name__ == "__main__":
    # Run examples
    asyncio.run(example_sr_feature_integration())
    asyncio.run(example_individual_feature_extraction())