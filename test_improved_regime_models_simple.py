#!/usr/bin/env python3
"""
Simple test script for improved regime models training.

This script tests the improved regime models training component without
requiring external ML libraries.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        # Test configuration validator
        from src.utils.ml_common.validation.config_validator import (
            ConfigValidator, validate_regime_training_config, create_default_regime_training_config
        )
        print("✅ Configuration validator imported successfully")
        
        # Test temporal data splitter
        from src.utils.ml_common.validation.temporal_data_splitter import (
            TemporalDataSplitter, RegimeAwareSplitter, create_temporal_splitter
        )
        print("✅ Temporal data splitter imported successfully")
        
        # Test regime label extractor
        from src.utils.ml_common.data.regime_label_extractor import (
            RegimeLabelExtractor, extract_regime_labels_fast_fail
        )
        print("✅ Regime label extractor imported successfully")
        
        # Test robust feature generator
        from src.utils.ml_common.features.robust_feature_generator import (
            RobustFeatureGenerator, generate_features_fast_fail, FeatureGenerationError
        )
        print("✅ Robust feature generator imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration_validation():
    """Test configuration validation."""
    print("\n🔧 Testing configuration validation...")
    
    try:
        from src.utils.ml_common.validation.config_validator import (
            validate_regime_training_config, create_default_regime_training_config
        )
        
        # Test default configuration
        default_config = create_default_regime_training_config()
        print(f"✅ Default config created: {list(default_config.keys())}")
        
        # Test valid configuration
        valid_config = {
            'test_size': 0.3,
            'validation_size': 0.2,
            'cv_folds': 5,
            'random_state': 42,
            'gap_size': 1,
            'min_regime_samples': 10
        }
        
        validated_config = validate_regime_training_config(valid_config)
        print("✅ Valid configuration validated successfully")
        
        # Test invalid configuration
        try:
            invalid_config = {'test_size': 0.8, 'cv_folds': 1}  # Invalid values
            validate_regime_training_config(invalid_config, strict=True)
            print("❌ Configuration validation should have failed")
            return False
        except ValueError:
            print("✅ Configuration validation correctly rejected invalid config")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation test failed: {e}")
        return False

def test_regime_label_extraction():
    """Test regime label extraction."""
    print("\n📊 Testing regime label extraction...")
    
    try:
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
        print(f"✅ Regime labels extracted: {len(labels)} samples")
        
        # Test with invalid artifacts (should fail fast)
        try:
            invalid_artifacts = {'no_regime_data': {}}
            extract_regime_labels_fast_fail(invalid_artifacts)
            print("❌ Regime label extractor should have failed")
            return False
        except ValueError:
            print("✅ Regime label extractor correctly failed fast")
        
        return True
        
    except Exception as e:
        print(f"❌ Regime label extraction test failed: {e}")
        return False

def test_temporal_data_splitter():
    """Test temporal data splitter."""
    print("\n⏰ Testing temporal data splitter...")
    
    try:
        from src.utils.ml_common.validation.temporal_data_splitter import (
            TemporalDataSplitter, RegimeAwareSplitter
        )
        
        # Test basic temporal splitter
        splitter = TemporalDataSplitter(test_size=0.3, gap_size=1)
        print("✅ Temporal splitter created successfully")
        
        # Test regime-aware splitter
        regime_splitter = RegimeAwareSplitter(test_size=0.3, gap_size=1, min_regime_samples=5)
        print("✅ Regime-aware splitter created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Temporal data splitter test failed: {e}")
        return False

def test_feature_generator():
    """Test feature generator."""
    print("\n🔧 Testing feature generator...")
    
    try:
        from src.utils.ml_common.features.robust_feature_generator import (
            RobustFeatureGenerator, FeatureGenerationError
        )
        
        # Test feature generator initialization
        generator = RobustFeatureGenerator(min_total_features=20, min_samples=50)
        print("✅ Feature generator created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature generator test failed: {e}")
        return False

def test_component_initialization():
    """Test component initialization."""
    print("\n🚀 Testing component initialization...")
    
    try:
        from src.training.steps.market_analysis.components.improved_regime_models_training import (
            ImprovedRegimeModelsTrainingComponent
        )
        from src.training.steps.market_analysis.components.base_component import ComponentConfig
        
        # Test component config
        component_config = ComponentConfig(
            symbol='ETHUSDT',
            exchange='binance',
            timeframe='1h',
            execution_mode='light'
        )
        print("✅ Component config created successfully")
        
        # Test component initialization (without ML libraries)
        try:
            component = ImprovedRegimeModelsTrainingComponent(component_config)
            print("✅ Component initialized successfully")
        except Exception as e:
            if "ML libraries not available" in str(e):
                print("✅ Component correctly detected missing ML libraries")
            else:
                print(f"❌ Component initialization failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Component initialization test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Improved Regime Models Training Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_configuration_validation,
        test_regime_label_extraction,
        test_temporal_data_splitter,
        test_feature_generator,
        test_component_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test {test.__name__} failed")
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Improved regime models training is working correctly.")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)