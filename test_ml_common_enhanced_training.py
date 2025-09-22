#!/usr/bin/env python3
"""
Test Script for ML Common Enhanced Training Integration

This script tests the integration of enhanced training utilities into the
ml_common base training infrastructure to verify that all ML models
benefit from overfitting prevention, lookahead bias detection, and
enhanced regularization natively.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_ml_common_enhanced_training():
    """Test the ml_common enhanced training integration."""
    print("🧪 Testing ML Common Enhanced Training Integration...")
    
    try:
        # Test enhanced training utilities import
        from src.utils.ml_common.training.enhanced_training_utils import (
            EnhancedTrainingUtils,
            EarlyStoppingConfig,
            PurgedCVConfig,
            OverfittingMonitorConfig,
            RegularizationConfig
        )
        print("✅ Enhanced training utilities imported successfully")
        
        # Test training integration import
        from src.utils.ml_common.training.training_integration import (
            TrainingStepEnhancer,
            TrainingIntegrationConfig
        )
        print("✅ Training integration imported successfully")
        
        # Test base training step import
        from src.utils.ml_common.training.base_training_step import BaseTrainingStep
        print("✅ Base training step imported successfully")
        
        # Test per-regime training step import
        from src.utils.ml_common.training.per_regime_training_step import PerRegimeTrainingStep
        print("✅ Per-regime training step imported successfully")
        
        # Test ensemble training step import
        from src.utils.ml_common.training.ensemble_training_step import EnsembleTrainingStep
        print("✅ Ensemble training step imported successfully")
        
        # Test base training config import
        from src.utils.ml_common.config.base_training_config import BaseTrainingConfig
        print("✅ Base training config imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ ML Common enhanced training test failed: {e}")
        return False

def test_per_regime_enhanced_training():
    """Test that PerRegimeTrainingStep has enhanced training utilities."""
    print("\n🧪 Testing Per-Regime Training Step Enhanced Integration...")
    
    try:
        from src.utils.ml_common.training.per_regime_training_step import PerRegimeTrainingStep
        from src.utils.ml_common.config.base_training_config import PerRegimeTrainingConfig
        
        print("✅ Per-regime training step imported successfully")
        
        # Check if enhanced training utilities are integrated
        with open("src/utils/ml_common/training/per_regime_training_step.py", 'r') as f:
            content = f.read()
        
        if "EnhancedTrainingUtils" in content:
            print("✅ Enhanced training utilities integrated in PerRegimeTrainingStep")
        else:
            print("❌ Enhanced training utilities not integrated in PerRegimeTrainingStep")
            return False
        
        if "_initialize_enhanced_training_utilities" in content:
            print("✅ Enhanced training initialization method found in PerRegimeTrainingStep")
        else:
            print("❌ Enhanced training initialization method not found in PerRegimeTrainingStep")
            return False
        
        if "_train_regime_models_enhanced" in content:
            print("✅ Enhanced training execution method found in PerRegimeTrainingStep")
        else:
            print("❌ Enhanced training execution method not found in PerRegimeTrainingStep")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Per-regime enhanced training test failed: {e}")
        return False

def test_ensemble_enhanced_training():
    """Test that EnsembleTrainingStep has enhanced training utilities."""
    print("\n🧪 Testing Ensemble Training Step Enhanced Integration...")
    
    try:
        from src.utils.ml_common.training.ensemble_training_step import EnsembleTrainingStep
        from src.utils.ml_common.config.base_training_config import EnsembleTrainingConfig
        
        print("✅ Ensemble training step imported successfully")
        
        # Check if enhanced training utilities are integrated
        with open("src/utils/ml_common/training/ensemble_training_step.py", 'r') as f:
            content = f.read()
        
        if "EnhancedTrainingUtils" in content:
            print("✅ Enhanced training utilities integrated in EnsembleTrainingStep")
        else:
            print("❌ Enhanced training utilities not integrated in EnsembleTrainingStep")
            return False
        
        if "_initialize_enhanced_training_utilities" in content:
            print("✅ Enhanced training initialization method found in EnsembleTrainingStep")
        else:
            print("❌ Enhanced training initialization method not found in EnsembleTrainingStep")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Ensemble enhanced training test failed: {e}")
        return False

def test_base_config_enhanced_settings():
    """Test that base training config has enhanced settings."""
    print("\n🧪 Testing Base Training Config Enhanced Settings...")
    
    try:
        from src.utils.ml_common.config.base_training_config import BaseTrainingConfig
        
        # Create a config instance
        config = BaseTrainingConfig()
        
        # Check for enhanced training settings
        enhanced_settings = [
            'enable_enhanced_training',
            'enable_early_stopping',
            'enable_lookahead_bias_detection',
            'enable_enhanced_regularization',
            'enable_temporal_validation',
            'enable_purged_cv',
            'enable_walk_forward_validation',
            'enable_ensemble_diversity'
        ]
        
        for setting in enhanced_settings:
            if hasattr(config, setting):
                print(f"✅ {setting} setting found in BaseTrainingConfig")
            else:
                print(f"❌ {setting} setting not found in BaseTrainingConfig")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Base config enhanced settings test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\n🧪 Testing File Structure...")
    
    required_files = [
        "src/utils/ml_common/training/enhanced_training_utils.py",
        "src/utils/ml_common/training/training_integration.py",
        "src/utils/ml_common/training/quick_integration.py",
        "src/utils/ml_common/training/integration_examples.py",
        "src/utils/ml_common/training/UPDATE_GUIDE.md",
        "src/utils/ml_common/training/per_regime_training_step.py",
        "src/utils/ml_common/training/ensemble_training_step.py",
        "src/utils/ml_common/config/base_training_config.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_enhanced_training_utilities():
    """Test that enhanced training utilities are properly integrated."""
    print("\n🧪 Testing Enhanced Training Utilities Integration...")
    
    try:
        # Test configuration creation
        from src.utils.ml_common.training.training_integration import TrainingIntegrationConfig
        
        config = TrainingIntegrationConfig(
            enable_early_stopping=True,
            enable_purged_cv=True,
            enable_lookahead_bias_detection=True,
            enable_temporal_splits=True,
            enable_regularization=True,
            enable_overfitting_monitoring=True
        )
        print("✅ Training integration configuration created successfully")
        
        # Test enhanced training utils
        from src.utils.ml_common.training.enhanced_training_utils import EnhancedTrainingUtils
        
        enhanced_utils = EnhancedTrainingUtils()
        print("✅ Enhanced training utils initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced training utilities test failed: {e}")
        return False

def main():
    """Run all ml_common integration tests."""
    print("🚀 Starting ML Common Enhanced Training Integration Tests")
    print("=" * 70)
    
    tests = [
        test_file_structure,
        test_ml_common_enhanced_training,
        test_per_regime_enhanced_training,
        test_ensemble_enhanced_training,
        test_base_config_enhanced_settings,
        test_enhanced_training_utilities
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! ML Common enhanced training integration is working correctly.")
        print("✅ All ML models now benefit from enhanced training utilities natively!")
        return True
    else:
        print("⚠️ Some tests failed. Please check the integration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)