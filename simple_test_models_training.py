#!/usr/bin/env python3
"""
Simple Test Script for Models Training Fixes

This script tests the basic functionality without external dependencies.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, '/workspace/src')

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        # Test if we can import the modules (this will test tprint imports)
        from training.steps.models_training.nas_tas.model_manager import ModelManager, ModelManagerConfig
        print("✅ Model Manager imports successfully")
    except Exception as e:
        print(f"❌ Model Manager import failed: {e}")
        return False
    
    try:
        from training.steps.models_training.nas_tas.performance_tracker import PerformanceTracker, PerformanceConfig
        print("✅ Performance Tracker imports successfully")
    except Exception as e:
        print(f"❌ Performance Tracker import failed: {e}")
        return False
    
    try:
        from training.steps.models_training.nas_tas.model_selector import ModelSelector, ModelSelectionConfig
        print("✅ Model Selector imports successfully")
    except Exception as e:
        print(f"❌ Model Selector import failed: {e}")
        return False
    
    try:
        from training.steps.models_training.nas_tas.regime_aware_trainer import RegimeAwareTrainer, RegimeAwareTrainingConfig
        print("✅ Regime Aware Trainer imports successfully")
    except Exception as e:
        print(f"❌ Regime Aware Trainer import failed: {e}")
        return False
    
    try:
        from training.steps.models_training.nas_tas.training_orchestrator import TrainingOrchestrator, OrchestratorConfig
        print("✅ Training Orchestrator imports successfully")
    except Exception as e:
        print(f"❌ Training Orchestrator import failed: {e}")
        return False
    
    return True

def test_tprint_availability():
    """Test that tprint functions are available in the modules."""
    print("\n🧪 Testing tprint availability...")
    
    try:
        from training.steps.models_training.nas_tas.model_manager import tprint, tprint_error, tprint_warning
        print("✅ Model Manager has tprint functions")
    except ImportError as e:
        print(f"❌ Model Manager missing tprint functions: {e}")
        return False
    
    try:
        from training.steps.models_training.nas_tas.performance_tracker import tprint, tprint_error, tprint_warning
        print("✅ Performance Tracker has tprint functions")
    except ImportError as e:
        print(f"❌ Performance Tracker missing tprint functions: {e}")
        return False
    
    try:
        from training.steps.models_training.nas_tas.model_selector import tprint, tprint_error, tprint_warning
        print("✅ Model Selector has tprint functions")
    except ImportError as e:
        print(f"❌ Model Selector missing tprint functions: {e}")
        return False
    
    return True

def test_configuration_creation():
    """Test that configuration objects can be created."""
    print("\n🧪 Testing configuration creation...")
    
    try:
        from training.steps.models_training.nas_tas.model_manager import ModelManagerConfig
        config = ModelManagerConfig()
        print("✅ ModelManagerConfig created successfully")
    except Exception as e:
        print(f"❌ ModelManagerConfig creation failed: {e}")
        return False
    
    try:
        from training.steps.models_training.nas_tas.performance_tracker import PerformanceConfig
        config = PerformanceConfig()
        print("✅ PerformanceConfig created successfully")
    except Exception as e:
        print(f"❌ PerformanceConfig creation failed: {e}")
        return False
    
    try:
        from training.steps.models_training.nas_tas.model_selector import ModelSelectionConfig
        config = ModelSelectionConfig()
        print("✅ ModelSelectionConfig created successfully")
    except Exception as e:
        print(f"❌ ModelSelectionConfig creation failed: {e}")
        return False
    
    return True

def test_placeholder_warnings():
    """Test that placeholder functions exist and can be called."""
    print("\n🧪 Testing placeholder function existence...")
    
    try:
        from training.steps.models_training.nas_tas.model_manager import ModelManager, ModelManagerConfig
        
        config = ModelManagerConfig()
        manager = ModelManager(config)
        
        # Check that placeholder methods exist
        if hasattr(manager, '_immediate_deployment'):
            print("✅ _immediate_deployment method exists")
        else:
            print("❌ _immediate_deployment method missing")
            return False
        
        if hasattr(manager, '_gradual_deployment'):
            print("✅ _gradual_deployment method exists")
        else:
            print("❌ _gradual_deployment method missing")
            return False
        
        if hasattr(manager, '_ab_testing_deployment'):
            print("✅ _ab_testing_deployment method exists")
        else:
            print("❌ _ab_testing_deployment method missing")
            return False
        
        if hasattr(manager, '_canary_deployment'):
            print("✅ _canary_deployment method exists")
        else:
            print("❌ _canary_deployment method missing")
            return False
    
    except Exception as e:
        print(f"❌ Placeholder function test failed: {e}")
        return False
    
    return True

def run_simple_tests():
    """Run all simple tests."""
    print("🚀 Starting simple test suite for models_training fixes...")
    print("=" * 60)
    
    tests = [
        ("imports", test_imports),
        ("tprint_availability", test_tprint_availability),
        ("configuration_creation", test_configuration_creation),
        ("placeholder_warnings", test_placeholder_warnings)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 TEST RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Basic functionality is working.")
        return True
    else:
        print(f"⚠️ {failed} tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = run_simple_tests()
    sys.exit(0 if success else 1)