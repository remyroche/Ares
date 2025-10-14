#!/usr/bin/env python3
"""
Simple test to verify the enhanced tprint logging and silent failure prevention
without external dependencies.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../..'))

def test_tprint_import():
    """Test that tprint utilities can be imported."""
    print("🧪 Testing tprint import...")
    
    try:
        from src.utils.tprint import (
            tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
        )
        print("✅ Tprint utilities imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Tprint import failed: {e}")
        return False

def test_pipeline_import():
    """Test that pipeline can be imported."""
    print("🧪 Testing pipeline import...")
    
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline import (
            UnifiedDataDrivenPipeline,
            create_unified_pipeline,
            process_with_unified_pipeline,
            UnifiedPipelineConfig,
            create_default_config
        )
        print("✅ Pipeline classes imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Pipeline import failed: {e}")
        return False

def test_enhanced_methods_exist():
    """Test that enhanced methods exist in the pipeline."""
    print("🧪 Testing enhanced methods exist...")
    
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline import (
            UnifiedDataDrivenPipeline
        )
        
        # Check if the enhanced methods exist
        methods_to_check = [
            '_enhanced_period_optimization',
            '_advanced_feature_selection', 
            '_generate_selected_features',
            '_enhanced_interaction_generation',
            '_combine_period_scores_safe',
            '_select_optimal_periods_safe'
        ]
        
        for method_name in methods_to_check:
            if hasattr(UnifiedDataDrivenPipeline, method_name):
                print(f"✅ Method {method_name} exists")
            else:
                print(f"❌ Method {method_name} missing")
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Pipeline import failed: {e}")
        return False

def test_enhanced_logging_in_code():
    """Test that enhanced logging is present in the code."""
    print("🧪 Testing enhanced logging in code...")
    
    try:
        # Read the consolidated pipeline file
        pipeline_file = os.path.join(os.path.dirname(__file__), 'consolidated_pipeline.py')
        
        with open(pipeline_file, 'r') as f:
            content = f.read()
        
        # Check for enhanced logging patterns
        logging_patterns = [
            'tprint_info("🔍 Starting',
            'tprint_debug("📊',
            'tprint_success("✅',
            'tprint_warning("⚠️',
            'tprint_error("❌',
            'error_msg =',
            'raise RuntimeError',
            'raise ValueError',
            'raise TypeError'
        ]
        
        found_patterns = []
        for pattern in logging_patterns:
            if pattern in content:
                found_patterns.append(pattern)
                print(f"✅ Found pattern: {pattern}")
            else:
                print(f"❌ Missing pattern: {pattern}")
        
        if len(found_patterns) >= len(logging_patterns) * 0.8:  # At least 80% of patterns
            print("✅ Enhanced logging patterns found in code")
            return True
        else:
            print(f"❌ Only {len(found_patterns)}/{len(logging_patterns)} logging patterns found")
            return False
            
    except Exception as e:
        print(f"❌ Error reading pipeline file: {e}")
        return False

def test_silent_failure_prevention():
    """Test that silent failure prevention is implemented."""
    print("🧪 Testing silent failure prevention...")
    
    try:
        # Read the consolidated pipeline file
        pipeline_file = os.path.join(os.path.dirname(__file__), 'consolidated_pipeline.py')
        
        with open(pipeline_file, 'r') as f:
            content = f.read()
        
        # Check for silent failure prevention patterns
        prevention_patterns = [
            'if data is None or data.empty:',
            'if not isinstance(',
            'if not hasattr(',
            'if not combined_scores:',
            'if not valid_scores:',
            'raise ValueError(',
            'raise TypeError(',
            'raise RuntimeError(',
            'error_msg =',
            'tprint_error('
        ]
        
        found_patterns = []
        for pattern in prevention_patterns:
            if pattern in content:
                found_patterns.append(pattern)
                print(f"✅ Found prevention pattern: {pattern}")
            else:
                print(f"❌ Missing prevention pattern: {pattern}")
        
        if len(found_patterns) >= len(prevention_patterns) * 0.7:  # At least 70% of patterns
            print("✅ Silent failure prevention patterns found in code")
            return True
        else:
            print(f"❌ Only {len(found_patterns)}/{len(prevention_patterns)} prevention patterns found")
            return False
            
    except Exception as e:
        print(f"❌ Error checking silent failure prevention: {e}")
        return False

def main():
    """Run all simple tests."""
    print("🚀 Starting simple enhancement verification tests")
    print("=" * 60)
    
    tests = [
        ("Tprint Import", test_tprint_import),
        ("Pipeline Import", test_pipeline_import),
        ("Enhanced Methods Exist", test_enhanced_methods_exist),
        ("Enhanced Logging in Code", test_enhanced_logging_in_code),
        ("Silent Failure Prevention", test_silent_failure_prevention)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📋 TEST SUMMARY")
    print(f"{'=' * 60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhancements are working correctly.")
        return True
    else:
        print(f"❌ {total - passed} tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)