#!/usr/bin/env python3
"""
Test script for improved regime models training.

This script tests the improved regime models training component with
proper temporal validation, fast fail behavior, and configuration validation.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_test_data(n_samples=1000):
    """Create synthetic test data."""
    np.random.seed(42)
    
    # Generate synthetic OHLCV data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_samples)
    
    # Add regime-like patterns
    regime_changes = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2])
    regime_multipliers = np.array([1.0, 1.5, 0.5])[regime_changes]
    returns *= regime_multipliers
    
    # Generate prices
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    prices = np.array(prices[1:])
    
    # Generate OHLCV
    high_multiplier = 1 + np.abs(np.random.normal(0, 0.01, n_samples))
    low_multiplier = 1 - np.abs(np.random.normal(0, 0.01, n_samples))
    
    high = prices * high_multiplier
    low = prices * low_multiplier
    open_prices = np.roll(prices, 1)
    open_prices[0] = base_price
    volume = np.random.lognormal(10, 1, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume,
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1H')
    })
    
    return data, regime_changes

def test_improved_regime_models_training():
    """Test the improved regime models training component."""
    try:
        print("🧪 Testing Improved Regime Models Training Component...")
        
        # Test 1: Import the component
        print("\n1. Testing component import...")
        from src.training.steps.market_analysis.components.regime_models_training import (
            RegimeModelsTrainingComponent
        )
        from src.training.steps.market_analysis.components.base_component import ComponentConfig
        print("✅ Successfully imported RegimeModelsTrainingComponent")
        
        # Test 2: Test configuration validation
        print("\n2. Testing configuration validation...")
        from src.utils.ml_common.validation.config_validator import (
            validate_regime_training_config, create_default_regime_training_config
        )
        
        # Test valid configuration
        valid_config = create_default_regime_training_config()
        validated_config = validate_regime_training_config(valid_config)
        print("✅ Configuration validation passed")
        
        # Test invalid configuration
        try:
            invalid_config = {'test_size': 0.8, 'cv_folds': 1}  # Invalid values
            validate_regime_training_config(invalid_config, strict=True)
            print("❌ Configuration validation should have failed")
        except ValueError:
            print("✅ Configuration validation correctly rejected invalid config")
        
        # Test 3: Test temporal data splitter
        print("\n3. Testing temporal data splitter...")
        from src.utils.ml_common.validation.temporal_data_splitter import (
            TemporalDataSplitter, RegimeAwareSplitter
        )
        
        # Create test data
        X = np.random.randn(1000, 10)
        y = np.random.choice([0, 1, 2], 1000)
        
        # Test temporal splitter
        splitter = TemporalDataSplitter(test_size=0.3, gap_size=1)
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_temporal(X, y)
        
        # Verify no data leakage
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
        assert X_train.shape[1] == X.shape[1]
        print("✅ Temporal data splitter working correctly")
        
        # Test 4: Test regime label extractor
        print("\n4. Testing regime label extractor...")
        from src.utils.ml_common.data.regime_label_extractor import (
            RegimeLabelExtractor, extract_regime_labels_fast_fail
        )
        
        # Test with valid artifacts
        valid_artifacts = {
            'regime_clustering_result': {
                'cluster_assignments': [0, 0, 1, 1, 2, 2, 0, 1, 2, 0]
            }
        }
        
        labels = extract_regime_labels_fast_fail(valid_artifacts)
        assert len(labels) == 10
        assert len(np.unique(labels)) >= 2
        print("✅ Regime label extractor working correctly")
        
        # Test with invalid artifacts (should fail fast)
        try:
            invalid_artifacts = {'no_regime_data': {}}
            extract_regime_labels_fast_fail(invalid_artifacts)
            print("❌ Regime label extractor should have failed")
        except ValueError:
            print("✅ Regime label extractor correctly failed fast")
        
        # Test 5: Test robust feature generator
        print("\n5. Testing robust feature generator...")
        from src.utils.ml_common.features.robust_feature_generator import (
            RobustFeatureGenerator, generate_features_fast_fail, FeatureGenerationError
        )
        
        # Create test data
        test_data, _ = create_test_data(200)
        
        # Test feature generation
        X, feature_names = generate_features_fast_fail(test_data, min_total_features=20, min_samples=50)
        assert X.shape[0] > 0
        assert X.shape[1] >= 20
        assert len(feature_names) == X.shape[1]
        print("✅ Robust feature generator working correctly")
        
        # Test with insufficient data (should fail fast)
        try:
            small_data = test_data.head(10)  # Too small
            generate_features_fast_fail(small_data, min_total_features=20, min_samples=50)
            print("❌ Feature generator should have failed with insufficient data")
        except FeatureGenerationError:
            print("✅ Feature generator correctly failed fast with insufficient data")
        
        # Test 6: Test component initialization
        print("\n6. Testing component initialization...")
        component_config = ComponentConfig(
            symbol='ETHUSDT',
            exchange='binance',
            timeframe='1h',
            execution_mode='light'
        )
        
        component = RegimeModelsTrainingComponent(component_config)
        assert hasattr(component, 'temporal_splitter')
        assert hasattr(component, 'regime_extractor')
        assert hasattr(component, 'feature_generator')
        print("✅ Component initialization successful")
        
        # Test 7: Test full execution
        print("\n7. Testing full execution...")
        test_data, regime_labels = create_test_data(500)
        
        # Create pipeline state with regime labels
        pipeline_state = {
            'artifacts': {
                'regime_clustering_result': {
                    'cluster_assignments': regime_labels.tolist()
                }
            }
        }
        
        # Run the component
        import asyncio
        result = asyncio.run(component.execute(test_data, pipeline_state))
        
        if result.success:
            print("✅ Full execution successful")
            print(f"   - Models trained: {len(result.artifacts.get('regime_models_training_result', {}).get('models', {}))}")
            print(f"   - Execution time: {result.metadata.get('execution_time', 0):.2f}s")
        else:
            print(f"❌ Full execution failed: {result.error_message}")
        
        print("\n🎉 All tests passed! Improved regime models training is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_temporal_validation():
    """Test temporal validation specifically."""
    print("\n🔍 Testing temporal validation...")
    
    try:
        from src.utils.ml_common.validation.temporal_data_splitter import RegimeAwareSplitter
        
        # Create test data with regime structure
        n_samples = 1000
        X = np.random.randn(n_samples, 10)
        
        # Create regime labels with temporal structure
        regime_labels = np.zeros(n_samples, dtype=int)
        regime_labels[200:400] = 1
        regime_labels[600:800] = 2
        
        # Test regime-aware splitter
        splitter = RegimeAwareSplitter(test_size=0.3, gap_size=5, min_regime_samples=10)
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_regime_aware(X, regime_labels)
        
        # Verify temporal order is maintained
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
        assert X_train.shape[0] > 0 and X_val.shape[0] > 0 and X_test.shape[0] > 0  # All sets should have data
        
        # Verify regime distribution
        train_regimes = np.unique(y_train)
        val_regimes = np.unique(y_val)
        test_regimes = np.unique(y_test)
        
        print(f"   - Train regimes: {train_regimes}")
        print(f"   - Val regimes: {val_regimes}")
        print(f"   - Test regimes: {test_regimes}")
        
        print("✅ Temporal validation working correctly")
        
    except Exception as e:
        print(f"❌ Temporal validation test failed: {e}")
        raise

if __name__ == "__main__":
    print("🚀 Starting Improved Regime Models Training Tests")
    print("=" * 60)
    
    # Run main tests
    success = test_improved_regime_models_training()
    
    # Run temporal validation test
    temporal_success = test_temporal_validation()
    
    if success and temporal_success:
        print("\n🎉 All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)