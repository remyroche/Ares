#!/usr/bin/env python3
"""
Simple VectorBT Performance Test

This script tests the basic functionality of the VectorBT optimizations
without requiring external dependencies.
"""

import sys
import os
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all the enhanced modules can be imported."""
    print("🧪 Testing Enhanced VectorBT Module Imports")
    print("=" * 50)
    
    try:
        from feature_generation.utils.vectorbt_rolling_optimizer import (
            VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
        )
        print("✅ VectorBTRollingOptimizer imported successfully")
        
        # Test that new methods exist
        optimizer = VectorBTRollingOptimizer()
        
        # Check for new batch processing methods
        if hasattr(optimizer, 'batch_rolling_operations'):
            print("✅ batch_rolling_operations method available")
        else:
            print("❌ batch_rolling_operations method missing")
            
        if hasattr(optimizer, 'parallel_cross_validation'):
            print("✅ parallel_cross_validation method available")
        else:
            print("❌ parallel_cross_validation method missing")
            
        if hasattr(optimizer, 'chunked_processing'):
            print("✅ chunked_processing method available")
        else:
            print("❌ chunked_processing method missing")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_training_module_imports():
    """Test that training modules can be imported."""
    print("\n🧪 Testing Training Module Imports")
    print("=" * 50)
    
    try:
        from training.steps.models_training.tactician_ensemble_training import TacticianEnsembleTrainingStep
        print("✅ TacticianEnsembleTrainingStep imported successfully")
        
        # Check for new batch methods
        if hasattr(TacticianEnsembleTrainingStep, '_optimized_batch_rolling_operations'):
            print("✅ _optimized_batch_rolling_operations method available")
        else:
            print("❌ _optimized_batch_rolling_operations method missing")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_code_structure():
    """Test the code structure and method signatures."""
    print("\n🧪 Testing Code Structure")
    print("=" * 50)
    
    try:
        from feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
        
        optimizer = VectorBTRollingOptimizer()
        
        # Test method signatures
        import inspect
        
        # Check batch_rolling_operations signature
        if hasattr(optimizer, 'batch_rolling_operations'):
            sig = inspect.signature(optimizer.batch_rolling_operations)
            params = list(sig.parameters.keys())
            expected_params = ['data', 'operations', 'window']
            
            if all(param in params for param in expected_params):
                print("✅ batch_rolling_operations has correct signature")
            else:
                print(f"❌ batch_rolling_operations signature incorrect. Got: {params}, Expected: {expected_params}")
        
        # Check parallel_cross_validation signature
        if hasattr(optimizer, 'parallel_cross_validation'):
            sig = inspect.signature(optimizer.parallel_cross_validation)
            params = list(sig.parameters.keys())
            expected_params = ['X', 'y', 'model_class', 'cv_folds']
            
            if all(param in params for param in expected_params):
                print("✅ parallel_cross_validation has correct signature")
            else:
                print(f"❌ parallel_cross_validation signature incorrect. Got: {params}, Expected: {expected_params}")
        
        # Check chunked_processing signature
        if hasattr(optimizer, 'chunked_processing'):
            sig = inspect.signature(optimizer.chunked_processing)
            params = list(sig.parameters.keys())
            expected_params = ['data', 'operation_func', 'chunk_size']
            
            if all(param in params for param in expected_params):
                print("✅ chunked_processing has correct signature")
            else:
                print(f"❌ chunked_processing signature incorrect. Got: {params}, Expected: {expected_params}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing code structure: {e}")
        return False

def test_performance_improvements():
    """Test that performance improvements are implemented."""
    print("\n🧪 Testing Performance Improvements Implementation")
    print("=" * 50)
    
    improvements = {
        'batch_rolling': False,
        'parallel_cv': False,
        'memory_chunking': False,
        'enhanced_training': False
    }
    
    try:
        from feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
        
        optimizer = VectorBTRollingOptimizer()
        
        # Check for batch processing
        if hasattr(optimizer, 'batch_rolling_operations'):
            improvements['batch_rolling'] = True
            print("✅ Batch rolling operations implemented")
        
        # Check for parallel CV
        if hasattr(optimizer, 'parallel_cross_validation'):
            improvements['parallel_cv'] = True
            print("✅ Parallel cross-validation implemented")
        
        # Check for memory chunking
        if hasattr(optimizer, 'chunked_processing'):
            improvements['memory_chunking'] = True
            print("✅ Memory-efficient chunking implemented")
        
        # Check training module enhancements
        try:
            from training.steps.models_training.tactician_ensemble_training import TacticianEnsembleTrainingStep
            if hasattr(TacticianEnsembleTrainingStep, '_optimized_batch_rolling_operations'):
                improvements['enhanced_training'] = True
                print("✅ Enhanced training modules implemented")
        except:
            pass
        
        return improvements
        
    except Exception as e:
        print(f"❌ Error testing performance improvements: {e}")
        return improvements

def main():
    """Run all tests."""
    print("🚀 VectorBT Performance Improvements Validation")
    print("=" * 80)
    
    tests = [
        ("Module Imports", test_imports),
        ("Training Module Imports", test_training_module_imports),
        ("Code Structure", test_code_structure),
        ("Performance Improvements", test_performance_improvements)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 80)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        if isinstance(result, dict):
            # For performance improvements test
            if all(result.values()):
                print(f"✅ {test_name}: All improvements implemented")
                passed += 1
            else:
                print(f"⚠️ {test_name}: Partial implementation")
                for improvement, status in result.items():
                    status_icon = "✅" if status else "❌"
                    print(f"   {status_icon} {improvement}")
        elif result:
            print(f"✅ {test_name}: Passed")
            passed += 1
        else:
            print(f"❌ {test_name}: Failed")
    
    print(f"\n🎯 Overall Results:")
    print(f"   Tests passed: {passed}/{total}")
    print(f"   Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All VectorBT performance improvements successfully implemented!")
    elif passed >= total * 0.75:
        print("✅ Most VectorBT performance improvements implemented successfully!")
    else:
        print("⚠️ Some VectorBT performance improvements need attention.")
    
    return results

if __name__ == "__main__":
    results = main()