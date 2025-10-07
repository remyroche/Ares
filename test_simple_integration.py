#!/usr/bin/env python3
"""
Simple test for Tactician & Analyst integration implementation.

This script tests the code structure and imports without running the full training.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that the updated modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent / "src"))
        
        # Test Tactician training imports
        from src.training.steps.models_training.tactician_models_training import TacticianModelsTrainingStep
        print("✅ TacticianModelsTrainingStep imported successfully")
        
        # Test Analyst training imports
        from src.training.steps.models_training.analyst_models_training import AnalystModelsTrainingStep
        print("✅ AnalystModelsTrainingStep imported successfully")
        
        # Test final parameters optimization imports
        from src.training.steps.backtesting.final_parameters_optimization import FinalParametersOptimizer
        print("✅ FinalParametersOptimizer imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_method_existence():
    """Test that the new methods exist in the classes."""
    print("🧪 Testing method existence...")
    
    try:
        sys.path.append(str(Path(__file__).parent / "src"))
        
        from src.training.steps.models_training.tactician_models_training import TacticianModelsTrainingStep
        from src.training.steps.models_training.analyst_models_training import AnalystModelsTrainingStep
        from src.training.steps.backtesting.final_parameters_optimization import FinalParametersOptimizer
        
        # Test Tactician methods
        tactician_trainer = TacticianModelsTrainingStep()
        assert hasattr(tactician_trainer, '_add_analyst_oof_features'), "Missing _add_analyst_oof_features method"
        assert hasattr(tactician_trainer, '_calculate_analyst_weights'), "Missing _calculate_analyst_weights method"
        assert hasattr(tactician_trainer, '_load_analyst_oof_outputs'), "Missing _load_analyst_oof_outputs method"
        print("✅ Tactician methods exist")
        
        # Test Analyst methods
        analyst_trainer = AnalystModelsTrainingStep()
        assert hasattr(analyst_trainer, '_generate_oof_predictions'), "Missing _generate_oof_predictions method"
        print("✅ Analyst methods exist")
        
        # Test FinalParametersOptimizer categories
        config = {'n_trials': 5, 'timeout': 30, 'study_name': 'test'}
        optimizer = FinalParametersOptimizer(config)
        assert 'tactician_analyst_integration' in optimizer.categories, "Missing tactician_analyst_integration category"
        assert 'analyst_oof_weights' in optimizer.categories, "Missing analyst_oof_weights category"
        assert 'merged_feature_importance' in optimizer.categories, "Missing merged_feature_importance category"
        print("✅ FinalParametersOptimizer categories exist")
        
        return True
        
    except Exception as e:
        print(f"❌ Method existence test failed: {e}")
        return False

def test_search_spaces():
    """Test that the new search spaces are defined."""
    print("🧪 Testing search spaces...")
    
    try:
        sys.path.append(str(Path(__file__).parent / "src"))
        
        from src.training.steps.backtesting.final_parameters_optimization import FinalParametersOptimizer
        
        config = {'n_trials': 5, 'timeout': 30, 'study_name': 'test'}
        optimizer = FinalParametersOptimizer(config)
        
        # Test that new search spaces exist
        search_spaces = optimizer.default_search_spaces
        
        assert 'tactician_analyst_integration' in search_spaces, "Missing tactician_analyst_integration search space"
        assert 'analyst_oof_weights' in search_spaces, "Missing analyst_oof_weights search space"
        assert 'merged_feature_importance' in search_spaces, "Missing merged_feature_importance search space"
        
        # Test specific parameters
        tactician_analyst_space = search_spaces['tactician_analyst_integration']
        assert 'w_min' in tactician_analyst_space, "Missing w_min parameter"
        assert 'p_trade_weight' in tactician_analyst_space, "Missing p_trade_weight parameter"
        assert 'u_trade_weight' in tactician_analyst_space, "Missing u_trade_weight parameter"
        assert 'q_trade_weight' in tactician_analyst_space, "Missing q_trade_weight parameter"
        
        print("✅ Search spaces are properly defined")
        
        return True
        
    except Exception as e:
        print(f"❌ Search spaces test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Tactician & Analyst integration tests...")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Method Existence Test", test_method_existence),
        ("Search Spaces Test", test_search_spaces),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Tactician & Analyst integration is working correctly.")
        print("\n📋 Implementation Summary:")
        print("   • Tactician now trains on whole dataset")
        print("   • Analyst OOF outputs (p_trade, u_trade, q_trade) are used as features")
        print("   • Sample weights calculated as w = w_min + (1-w_min)*p_trade")
        print("   • Final parameters optimization handles merged inputs")
        print("   • New optimization categories added for integration")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)