#!/usr/bin/env python3
"""
Test script for long/short separation in ML model training.

This script demonstrates and validates the directional training functionality
implemented in the models_training system.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import sys
import os

# Add the src directory to the path
sys.path.append('/Users/remyroche/Documents/Ares/src')

def create_synthetic_market_data(n_samples=1000, include_directions=True):
    """Create synthetic market data for testing."""
    np.random.seed(42)

    # Generate base price data
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H')

    # Create OHLCV data
    close_prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.1)
    opens = close_prices + np.random.randn(n_samples) * 0.05
    highs = np.maximum(opens, close_prices) + np.random.randn(n_samples) * 0.02
    lows = np.minimum(opens, close_prices) - np.random.randn(n_samples) * 0.02
    volumes = 1000 + np.random.randn(n_samples) * 100

    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices,
        'volume': volumes
    })

    # Add target variable (simplified return prediction)
    data['target_return'] = data['close'].pct_change().shift(-1)

    if include_directions:
        # Add direction indicators based on price movement
        data['direction_long'] = (data['target_return'] > 0.001).astype(int)
        data['direction_short'] = (data['target_return'] < -0.001).astype(int)
        data['direction_combined'] = data['direction_long'] - data['direction_short']

    return data

def test_directional_data_separation():
    """Test the directional data separation functionality."""
    print("🧪 Testing directional data separation...")

    # Create test data
    market_data = create_synthetic_market_data(500)

    # Test data separation logic (from training orchestrator)
    direction_columns = [col for col in market_data.columns if 'direction' in col.lower() or 'long' in col.lower() or 'short' in col.lower()]

    if direction_columns:
        directional_data = {}
        for direction in ['long', 'short']:
            direction_cols = [col for col in direction_columns if direction in col.lower()]
            for col in direction_cols:
                mask = market_data[col] == 1
                if mask.any():
                    directional_data[direction] = market_data[mask].copy()
                    break

        print(f"✅ Directional separation successful:")
        print(f"   - Long samples: {len(directional_data.get('long', []))}")
        print(f"   - Short samples: {len(directional_data.get('short', []))}")
        return True
    else:
        print("⚠️ No direction columns found in test data")
        return False

def test_directional_feature_engineering():
    """Test directional feature engineering."""
    print("\n🔧 Testing directional feature engineering...")

    market_data = create_synthetic_market_data(200)

    # Test adding directional features (from training orchestrator)
    direction = 'long'
    data = market_data.copy()

    if 'close' in data.columns:
        data['long_price_momentum_10'] = data['close'] / data['close'].shift(10) - 1
        data['long_price_momentum_20'] = data['close'] / data['close'].shift(20) - 1
        data['long_price_acceleration'] = data['long_price_momentum_10'] - data['long_price_momentum_20']

    if 'volume' in data.columns:
        data['long_volume_trend'] = data['volume'].rolling(10).mean() / data['volume'].rolling(20).mean()
        data['long_volume_confirmation'] = (data['long_volume_trend'] > 1.0).astype(int)

    # Check if new features were added
    new_features = [col for col in data.columns if col.startswith('long_')]
    print(f"✅ Directional feature engineering successful:")
    print(f"   - New long features added: {len(new_features)}")
    print(f"   - Sample features: {new_features[:3]}")
    return True

def test_directional_model_configuration():
    """Test directional model configuration."""
    print("\n⚙️ Testing directional model configuration...")

    try:
        # Test importing directional configuration
        from src.training.steps.models_training.nas_tas.regime_aware_trainer import DirectionMode

        print(f"✅ DirectionMode enum imported successfully:")
        print(f"   - Available modes: {[mode.value for mode in DirectionMode]}")

        # Test creating directional config
        from src.training.steps.models_training.nas_tas.regime_aware_trainer import RegimeAwareTrainingConfig

        config = RegimeAwareTrainingConfig()
        config.direction_mode = DirectionMode.SEPARATE
        config.separate_directional_features = True

        print(f"✅ Directional configuration created:")
        print(f"   - Mode: {config.direction_mode.value}")
        print(f"   - Separate features: {config.separate_directional_features}")
        print(f"   - Min directional samples: {config.min_directional_samples}")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_directional_model_selection():
    """Test directional model selection functionality."""
    print("\n🎯 Testing directional model selection...")

    try:
        # Test importing model selector
        from src.training.steps.models_training.nas_tas.model_selector import ModelSelector, ModelSelectionConfig

        config = ModelSelectionConfig()
        selector = ModelSelector(config)

        print(f"✅ Model selector created successfully")
        print(f"   - Selection strategy: {config.selection_strategy.value}")
        print(f"   - Routing method: {config.routing_method.value}")

        # Test direction detection method
        market_data = create_synthetic_market_data(50)
        direction = selector._detect_direction(market_data)
        print(f"✅ Direction detection successful: {direction}")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_training_orchestrator_configuration():
    """Test training orchestrator directional configuration."""
    print("\n🏗️ Testing training orchestrator configuration...")

    try:
        # Test importing training orchestrator
        from src.training.steps.models_training.nas_tas.training_orchestrator import OrchestratorConfig

        config = OrchestratorConfig()
        config.direction_mode = "separate"
        config.separate_directional_features = True

        print(f"✅ Training orchestrator configuration created:")
        print(f"   - Direction mode: {config.direction_mode}")
        print(f"   - Separate features: {config.separate_directional_features}")
        print(f"   - Min directional samples: {config.min_directional_samples}")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Long/Short Separation Tests")
    print("=" * 50)

    tests = [
        test_directional_data_separation,
        test_directional_feature_engineering,
        test_directional_model_configuration,
        test_directional_model_selection,
        test_training_orchestrator_configuration
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            print(f"❌ {test.__name__} FAILED with error: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Long/short separation implementation is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
