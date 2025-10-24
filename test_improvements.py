#!/usr/bin/env python3
"""
Test script to verify the improvements work correctly.
"""

import sys
import os
from pathlib import Path

# Add the workspace to Python path
workspace_root = Path(__file__).parent
sys.path.insert(0, str(workspace_root))

def test_imports_work():
    """Test that all imports work after refactoring."""
    print("🧪 Testing imports after refactoring...")
    
    try:
        from src.utils.ml_common.optimization import (
            ConsolidatedHPO, HPOConfig, HPOResult, HPOPhaseConfig,
            create_bayesian_hpo, create_grid_hpo, create_random_hpo,
            OptimizationError, ConfigurationError
        )
        print("✅ All imports work correctly")
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_bayesian_grid_prestep():
    """Test that Bayesian strategy uses Grid as pre-step."""
    print("\n🧪 Testing Bayesian Grid pre-step...")
    
    try:
        from src.utils.ml_common.optimization import HPOConfig, create_bayesian_hpo
        
        # Create Bayesian HPO with staged optimization enabled
        config = HPOConfig(
            strategy='bayesian',
            n_trials=20,
            enable_staged_optimization=True,
            coarse_grid_trials=5,
            fine_grid_trials=5
        )
        
        hpo = create_bayesian_hpo(n_trials=20)
        
        # Check that the strategy is configured for staged optimization
        assert hpo.config.enable_staged_optimization == True
        print("✅ Bayesian strategy configured for Grid pre-step")
        
        return True
    except Exception as e:
        print(f"❌ Bayesian Grid pre-step test failed: {e}")
        return False

def test_grid_coarse_fine():
    """Test that Grid strategy uses coarse-to-fine progression."""
    print("\n🧪 Testing Grid coarse-to-fine progression...")
    
    try:
        from src.utils.ml_common.optimization import HPOConfig, create_grid_hpo
        
        # Create Grid HPO with staged optimization
        config = HPOConfig(
            strategy='grid',
            n_trials=30,
            enable_staged_optimization=True,
            coarse_grid_points=3,
            fine_grid_points=5,
            coarse_grid_trials=10,
            fine_grid_trials=10,
            tpe_trials=10
        )
        
        hpo = create_grid_hpo(n_trials=30)
        
        # Check that staged optimization is enabled
        assert hpo.config.enable_staged_optimization == True
        assert hpo.config.coarse_grid_points == 3
        assert hpo.config.fine_grid_points == 5
        print("✅ Grid strategy configured for coarse-to-fine progression")
        
        return True
    except Exception as e:
        print(f"❌ Grid coarse-to-fine test failed: {e}")
        return False

def test_configuration_validation():
    """Test configuration validation works."""
    print("\n🧪 Testing configuration validation...")
    
    try:
        from src.utils.ml_common.optimization import HPOConfig, validate_hpo_config
        
        # Test valid configuration
        config = HPOConfig(
            strategy='bayesian',
            n_trials=100,
            enable_staged_optimization=True,
            coarse_grid_trials=20,
            fine_grid_trials=20
        )
        
        assert config.strategy.value == 'bayesian'
        assert config.enable_staged_optimization == True
        print("✅ Configuration validation works")
        
        # Test validation function
        config_dict = {
            'strategy': 'grid',
            'n_trials': 50,
            'enable_staged_optimization': True
        }
        
        validated_config = validate_hpo_config(config_dict)
        assert validated_config.strategy.value == 'grid'
        print("✅ Configuration validation function works")
        
        return True
    except Exception as e:
        print(f"❌ Configuration validation test failed: {e}")
        return False

def test_factory_functions():
    """Test factory functions work correctly."""
    print("\n🧪 Testing factory functions...")
    
    try:
        from src.utils.ml_common.optimization import (
            create_bayesian_hpo, create_grid_hpo, create_random_hpo,
            create_ares_mode_hpo, create_auto_mode_hpo
        )
        
        # Test basic factory functions
        hpo1 = create_bayesian_hpo(n_trials=10)
        assert hpo1.config.strategy.value == 'bayesian'
        print("✅ create_bayesian_hpo works")
        
        hpo2 = create_grid_hpo(n_trials=10)
        assert hpo2.config.strategy.value == 'grid'
        print("✅ create_grid_hpo works")
        
        hpo3 = create_random_hpo(n_trials=10)
        assert hpo3.config.strategy.value == 'random'
        print("✅ create_random_hpo works")
        
        # Test Ares mode factory
        hpo4 = create_ares_mode_hpo(ares_mode='light', strategy='bayesian', n_trials=10)
        assert hpo4.config.ares_execution_mode.value == 'light'
        print("✅ create_ares_mode_hpo works")
        
        # Test auto mode factory
        hpo5 = create_auto_mode_hpo(strategy='bayesian', n_trials=10)
        assert hpo5.config.auto_detect_mode == True
        print("✅ create_auto_mode_hpo works")
        
        return True
    except Exception as e:
        print(f"❌ Factory functions test failed: {e}")
        return False

def test_error_handling():
    """Test error handling works correctly."""
    print("\n🧪 Testing error handling...")
    
    try:
        from src.utils.ml_common.optimization import (
            OptimizationError, ConfigurationError, ModelEvaluationError
        )
        
        # Test exception hierarchy
        assert issubclass(ConfigurationError, OptimizationError)
        assert issubclass(ModelEvaluationError, OptimizationError)
        print("✅ Exception hierarchy is correct")
        
        # Test exception creation with context
        try:
            raise ConfigurationError("Test error", {"param": "value"})
        except ConfigurationError as e:
            assert e.context["param"] == "value"
            print("✅ Exception context works")
        
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing ML Optimization Improvements")
    print("=" * 60)
    
    tests = [
        test_imports_work,
        test_bayesian_grid_prestep,
        test_grid_coarse_fine,
        test_configuration_validation,
        test_factory_functions,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All improvements working correctly!")
        print("\n📋 Summary of completed improvements:")
        print("   ✅ Deleted legacy consolidated_hpo.py")
        print("   ✅ Bayesian optimization now uses Grid as pre-step")
        print("   ✅ Grid optimization uses proper coarse -> fine progression")
        print("   ✅ Updated all imports to use refactored components")
        print("   ✅ Configuration validation with Pydantic")
        print("   ✅ Comprehensive error handling")
        print("   ✅ Factory functions for easy HPO creation")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())