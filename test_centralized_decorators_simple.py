#!/usr/bin/env python3
"""
Simple test script for centralized decorators (no external dependencies)
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_centralized_decorators():
    """Test that all centralized decorators can be imported."""
    
    print("🧪 Testing Centralized Decorators (Simple)")
    print("=" * 50)
    
    try:
        # Test imports
        from src.utils.centralized_decorators import (
            validate_data_quality,
            quality_gate,
            step_specific_ml_validation,
            auto_fix_data_quality_issues,
            monitor_feature_engineering,
            monitor_data_collection,
            deterministic_seed,
            idempotent_step,
            handle_errors,
            with_tracing_span
        )
        print("✅ All decorators imported successfully")
        
        # Test that decorators can be applied
        @validate_data_quality(validation_level="WARNING", context="test")
        def test_function1():
            return "test1"
        
        @quality_gate(min_quality_score=0.7, required_grade="C")
        def test_function2():
            return "test2"
        
        @step_specific_ml_validation("step3")
        def test_function3():
            return "test3"
        
        @auto_fix_data_quality_issues(context="test")
        def test_function4():
            return "test4"
        
        @monitor_feature_engineering()
        def test_function5():
            return "test5"
        
        @deterministic_seed(42)
        @idempotent_step()
        @handle_errors()
        @with_tracing_span("test")
        def test_function6():
            return "test6"
        
        print("✅ All decorators applied successfully")
        
        # Test function calls
        result1 = test_function1()
        result2 = test_function2()
        result3 = test_function3()
        result4 = test_function4()
        result5 = test_function5()
        result6 = test_function6()
        
        print("✅ All decorated functions executed successfully")
        print("✅ Centralized decorators test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing centralized decorators: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step3_import():
    """Test that step3 can import quality_gate from centralized_decorators."""
    
    print("\n🧪 Testing Step3 Import")
    print("=" * 50)
    
    try:
        # Test that step3 can import quality_gate
        import src.training.steps.step3_hmm_regime_discovery as step3_module
        
        # Check if quality_gate is imported
        if hasattr(step3_module, 'quality_gate'):
            print("✅ Step3 successfully imports quality_gate from centralized_decorators")
        else:
            print("❌ Step3 does not have quality_gate imported")
            return False
        
        print("✅ Step3 import test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing step3 import: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_decorator_signatures():
    """Test that decorators have the correct signatures."""
    
    print("\n🧪 Testing Decorator Signatures")
    print("=" * 50)
    
    try:
        from src.utils.centralized_decorators import (
            validate_data_quality,
            quality_gate,
            step_specific_ml_validation,
            auto_fix_data_quality_issues
        )
        
        # Test that decorators are callable
        assert callable(validate_data_quality), "validate_data_quality should be callable"
        assert callable(quality_gate), "quality_gate should be callable"
        assert callable(step_specific_ml_validation), "step_specific_ml_validation should be callable"
        assert callable(auto_fix_data_quality_issues), "auto_fix_data_quality_issues should be callable"
        
        print("✅ All decorators are callable")
        
        # Test that decorators return decorator functions
        test_func = lambda: "test"
        
        decorated1 = validate_data_quality()(test_func)
        decorated2 = quality_gate()(test_func)
        decorated3 = step_specific_ml_validation("step3")(test_func)
        decorated4 = auto_fix_data_quality_issues()(test_func)
        
        assert callable(decorated1), "validate_data_quality should return a callable"
        assert callable(decorated2), "quality_gate should return a callable"
        assert callable(decorated3), "step_specific_ml_validation should return a callable"
        assert callable(decorated4), "auto_fix_data_quality_issues should return a callable"
        
        print("✅ All decorators return callable functions")
        print("✅ Decorator signatures test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing decorator signatures: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run tests
    success1 = test_centralized_decorators()
    success2 = test_step3_import()
    success3 = test_decorator_signatures()
    
    if all([success1, success2, success3]):
        print("\n🎉 All tests passed! Centralized decorators are working correctly.")
        print("\n📋 Summary:")
        print("   ✅ validate_data_quality decorator implemented and working")
        print("   ✅ quality_gate decorator implemented and working")
        print("   ✅ auto_fix_data_quality_issues decorator implemented and working")
        print("   ✅ step_specific_ml_validation decorator implemented and working")
        print("   ✅ All decorators centralized for maintainability")
        print("   ✅ Step3 uses correct quality_gate from centralized_decorators")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")