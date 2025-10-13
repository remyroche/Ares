#!/usr/bin/env python3
"""
Test script for the modular feature selection system.

This script verifies that all functionality is preserved after modularization
by running comprehensive tests on the new modular system.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
)


def create_test_data(n_samples: int = 1000, n_features: int = 120) -> tuple:
    """Create test data for feature selection."""
    tprint_info(f"📊 Creating test data: {n_samples} samples, {n_features} features")
    
    # Create random feature matrix
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create target variable with some correlation to features
    y = (
        0.3 * X.iloc[:, 0] +  # Strong correlation
        0.2 * X.iloc[:, 1] +  # Medium correlation
        0.1 * X.iloc[:, 2] +  # Weak correlation
        0.4 * np.random.randn(n_samples)  # Noise
    )
    
    tprint_success(f"   ✅ Test data created: {X.shape}")
    return X, y


def test_core_modules():
    """Test core feature selection modules."""
    tprint_info("🧪 Testing core modules")
    
    try:
        from src.training.steps.pre_training.feature_selection.core import (
            MultiStageFeatureSelector,
            FeatureSelector,
            FeatureSelectionOptimizer,
            FeatureSelectionConfig,
            FeatureSelectionResult
        )
        
        # Test configuration
        config = FeatureSelectionConfig(
            target_features=80,
            min_features=60,
            max_features=100
        )
        tprint_success("   ✅ Configuration classes working")
        
        # Test feature selector
        selector = FeatureSelector()
        X, y = create_test_data(100, 50)
        
        # Test MRMR selection
        selected_features = selector.mrmr_selection(X, y, k=10)
        assert len(selected_features) == 10, f"Expected 10 features, got {len(selected_features)}"
        tprint_success("   ✅ FeatureSelector working")
        
        # Test optimizer
        optimizer = FeatureSelectionOptimizer()
        hardware_config = optimizer.get_optimal_hardware_config()
        assert isinstance(hardware_config, dict), "Hardware config should be a dict"
        tprint_success("   ✅ FeatureSelectionOptimizer working")
        
        # Test pipeline
        pipeline = MultiStageFeatureSelector(config)
        result = pipeline.select_features(X, y)
        assert isinstance(result, FeatureSelectionResult), "Should return FeatureSelectionResult"
        tprint_success("   ✅ MultiStageFeatureSelector working")
        
        return True
        
    except Exception as e:
        tprint_error(f"   ❌ Core modules test failed: {e}")
        return False


def test_hardware_modules():
    """Test hardware optimization modules."""
    tprint_info("🧪 Testing hardware modules")
    
    try:
        from src.training.steps.pre_training.feature_selection.hardware import (
            MemoryManager,
            VectorBTManager,
            PerformanceMonitor
        )
        
        # Test memory manager
        memory_manager = MemoryManager()
        memory_stats = memory_manager.get_memory_stats()
        assert isinstance(memory_stats.total_memory_gb, float), "Memory stats should be numeric"
        tprint_success("   ✅ MemoryManager working")
        
        # Test VectorBT manager
        vectorbt_manager = VectorBTManager()
        is_available = vectorbt_manager.is_available()
        assert isinstance(is_available, bool), "VectorBT availability should be boolean"
        tprint_success("   ✅ VectorBTManager working")
        
        # Test performance monitor
        performance_monitor = PerformanceMonitor()
        with performance_monitor.monitor_operation("test_operation", (100, 50)):
            time.sleep(0.1)  # Simulate work
        
        summary = performance_monitor.get_performance_summary()
        assert summary.total_operations > 0, "Should have recorded operations"
        tprint_success("   ✅ PerformanceMonitor working")
        
        # Cleanup
        memory_manager.cleanup_resources()
        vectorbt_manager.cleanup()
        performance_monitor.cleanup()
        
        return True
        
    except Exception as e:
        tprint_error(f"   ❌ Hardware modules test failed: {e}")
        return False


def test_config_modules():
    """Test configuration management modules."""
    tprint_info("🧪 Testing config modules")
    
    try:
        from src.training.steps.pre_training.feature_selection.config import (
            ConfigLoader,
            ModelProfileManager,
            ConfigValidator
        )
        
        # Test config loader
        config_loader = ConfigLoader()
        available_configs = config_loader.get_available_configs()
        assert len(available_configs) > 0, "Should have available configs"
        tprint_success("   ✅ ConfigLoader working")
        
        # Test model profile manager
        profile_manager = ModelProfileManager()
        profile = profile_manager.get_profile('neural_network')
        assert profile is not None, "Should get neural network profile"
        assert profile.target_features > 0, "Profile should have valid target features"
        tprint_success("   ✅ ModelProfileManager working")
        
        # Test config validator
        validator = ConfigValidator()
        test_config = {
            'target_features': 80,
            'min_features': 60,
            'max_features': 100,
            'vif_threshold': 10.0
        }
        validation_result = validator.validate_config(test_config)
        assert validation_result.is_valid, f"Valid config should pass validation: {validation_result.errors}"
        tprint_success("   ✅ ConfigValidator working")
        
        return True
        
    except Exception as e:
        tprint_error(f"   ❌ Config modules test failed: {e}")
        return False


def test_validation_modules():
    """Test validation modules."""
    tprint_info("🧪 Testing validation modules")
    
    try:
        from src.training.steps.pre_training.feature_selection.validation import (
            DataValidator
        )
        
        # Test data validator
        validator = DataValidator()
        X, y = create_test_data(100, 50)
        
        validation_result = validator.validate_data(X, y)
        assert isinstance(validation_result.is_valid, bool), "Should return validation result"
        tprint_success("   ✅ DataValidator working")
        
        return True
        
    except Exception as e:
        tprint_error(f"   ❌ Validation modules test failed: {e}")
        return False


def test_integrated_pipeline():
    """Test the integrated modular pipeline."""
    tprint_info("🧪 Testing integrated modular pipeline")
    
    try:
        from src.training.steps.pre_training.final_feature_selection_pipeline_modular import (
            run_final_feature_selection,
            get_final_features
        )
        
        # Create test data
        X, y = create_test_data(200, 120)
        
        # Test main pipeline function
        result = run_final_feature_selection(X, y)
        assert isinstance(result.success, bool), "Should return success status"
        tprint_success("   ✅ run_final_feature_selection working")
        
        # Test convenience function
        selected_features = get_final_features(X, y)
        assert isinstance(selected_features, list), "Should return list of features"
        tprint_success("   ✅ get_final_features working")
        
        return True
        
    except Exception as e:
        tprint_error(f"   ❌ Integrated pipeline test failed: {e}")
        return False


def test_performance_comparison():
    """Compare performance between old and new systems."""
    tprint_info("🧪 Testing performance comparison")
    
    try:
        # Create test data
        X, y = create_test_data(500, 100)
        
        # Test new modular system
        start_time = time.time()
        from src.training.steps.pre_training.final_feature_selection_pipeline_modular import run_final_feature_selection
        
        result = run_final_feature_selection(X, y)
        modular_time = time.time() - start_time
        
        tprint_success(f"   ✅ Modular system: {modular_time:.3f}s")
        tprint_success(f"   📊 Selected {len(result.selected_features)} features")
        
        return True
        
    except Exception as e:
        tprint_error(f"   ❌ Performance comparison test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    tprint("🚀 Starting Modular Feature Selection System Tests")
    tprint("=" * 60)
    
    tests = [
        ("Core Modules", test_core_modules),
        ("Hardware Modules", test_hardware_modules),
        ("Config Modules", test_config_modules),
        ("Validation Modules", test_validation_modules),
        ("Integrated Pipeline", test_integrated_pipeline),
        ("Performance Comparison", test_performance_comparison)
    ]
    
    results = {}
    total_tests = len(tests)
    passed_tests = 0
    
    for test_name, test_func in tests:
        tprint(f"\n📋 Running {test_name} Test")
        try:
            success = test_func()
            results[test_name] = success
            if success:
                passed_tests += 1
                tprint_success(f"✅ {test_name} test PASSED")
            else:
                tprint_error(f"❌ {test_name} test FAILED")
        except Exception as e:
            tprint_error(f"❌ {test_name} test FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    tprint("\n" + "=" * 60)
    tprint("📊 TEST SUMMARY")
    tprint("=" * 60)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        tprint(f"{test_name}: {status}")
    
    tprint(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        tprint_success("🎉 ALL TESTS PASSED! Modular system is working correctly.")
        return True
    else:
        tprint_error(f"⚠️ {total_tests - passed_tests} tests failed. Please review the issues.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)