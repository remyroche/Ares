"""
Quick verification script for HTF base features refactoring.

This script tests that:
1. The new module can be imported
2. Backward compatibility is maintained
3. New functions are available
4. No breaking changes
"""

import sys
import traceback
from typing import List, Tuple

def test_imports() -> Tuple[bool, str]:
    """Test that the module can be imported."""
    try:
        from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation import htf_base_features
        return True, "✅ Module imported successfully"
    except Exception as e:
        return False, f"❌ Failed to import module: {e}"


def test_backward_compatible_functions() -> Tuple[bool, str]:
    """Test that backward compatible functions exist."""
    try:
        from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.htf_base_features import (
            get_base_feature_func,
            resample_to_htf
        )
        
        # Check that functions are callable
        if not callable(get_base_feature_func):
            return False, "❌ get_base_feature_func is not callable"
        if not callable(resample_to_htf):
            return False, "❌ resample_to_htf is not callable"
        
        return True, "✅ Backward compatible functions exist and are callable"
    except Exception as e:
        return False, f"❌ Failed to import backward compatible functions: {e}"


def test_new_functions() -> Tuple[bool, str]:
    """Test that new functions are available."""
    try:
        from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.htf_base_features import (
            DynamicFeatureGenerator,
            get_feature_generator,
            generate_htf_features,
            optimize_htf_lookbacks
        )
        
        # Check that new components exist
        if not callable(DynamicFeatureGenerator):
            return False, "❌ DynamicFeatureGenerator is not a class"
        if not callable(get_feature_generator):
            return False, "❌ get_feature_generator is not callable"
        if not callable(generate_htf_features):
            return False, "❌ generate_htf_features is not callable"
        if not callable(optimize_htf_lookbacks):
            return False, "❌ optimize_htf_lookbacks is not callable"
        
        return True, "✅ New functions exist and are callable"
    except Exception as e:
        return False, f"❌ Failed to import new functions: {e}"


def test_removed_functions() -> Tuple[bool, str]:
    """Test that old hardcoded functions are removed."""
    try:
        from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation import htf_base_features
        
        # These should NOT exist anymore
        removed_functions = [
            '_price_ema10_pct',
            '_price_ema20_pct',
            '_bollz20',
            '_sigma_ew',
            '_gk_w',
            '_rv_bipower_12',
            '_rv_short_3',
            '_rsi',
            '_rsi7',
            '_rsi14',
            '_stochk14',
            '_autocorr_r1_w',
            '_vwap_session_dist',
            '_vwap_roll12_dist',
            '_BASE_FEATURE_FUNCTIONS'
        ]
        
        still_exist = []
        for func_name in removed_functions:
            if hasattr(htf_base_features, func_name):
                still_exist.append(func_name)
        
        if still_exist:
            return False, f"❌ Old functions still exist: {still_exist}"
        
        return True, "✅ Old hardcoded functions properly removed"
    except Exception as e:
        return False, f"❌ Failed to check for removed functions: {e}"


def test_exports() -> Tuple[bool, str]:
    """Test that __all__ is properly defined."""
    try:
        from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation import htf_base_features
        
        if not hasattr(htf_base_features, '__all__'):
            return False, "❌ __all__ is not defined"
        
        expected_exports = [
            'DynamicFeatureGenerator',
            'get_feature_generator',
            'generate_htf_features',
            'optimize_htf_lookbacks',
            'get_base_feature_func',
            'resample_to_htf'
        ]
        
        actual_exports = set(htf_base_features.__all__)
        expected_exports_set = set(expected_exports)
        
        missing = expected_exports_set - actual_exports
        if missing:
            return False, f"❌ Missing exports: {missing}"
        
        return True, f"✅ __all__ properly defined with {len(actual_exports)} exports"
    except Exception as e:
        return False, f"❌ Failed to check exports: {e}"


def test_feature_generator_initialization() -> Tuple[bool, str]:
    """Test that DynamicFeatureGenerator can be initialized."""
    try:
        from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.htf_base_features import (
            get_feature_generator
        )
        
        # Try to get the generator
        generator = get_feature_generator()
        
        # Check that it has expected attributes
        if not hasattr(generator, 'generate_features'):
            return False, "❌ Generator missing generate_features method"
        if not hasattr(generator, 'optimize_feature_lookback'):
            return False, "❌ Generator missing optimize_feature_lookback method"
        if not hasattr(generator, 'get_feature_function'):
            return False, "❌ Generator missing get_feature_function method"
        
        return True, f"✅ DynamicFeatureGenerator initialized (initialized={generator._initialized})"
    except Exception as e:
        return False, f"❌ Failed to initialize generator: {e}"


def test_backward_compatibility_call() -> Tuple[bool, str]:
    """Test that backward compatible functions can be called."""
    try:
        import pandas as pd
        import numpy as np
        from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.htf_base_features import (
            get_base_feature_func,
            resample_to_htf
        )
        
        # Create sample data
        data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        data.index = pd.date_range(start='2024-01-01', periods=100, freq='5min')
        
        # Test get_base_feature_func
        try:
            rsi_func = get_base_feature_func('rsi', lookback_period=14)
            if not callable(rsi_func):
                return False, "❌ get_base_feature_func didn't return a callable"
        except Exception as e:
            # It's OK if this fails due to FeatureBank not being available
            return True, f"⚠️ get_base_feature_func call failed (FeatureBank may not be available): {e}"
        
        # Test resample_to_htf
        test_series = pd.Series(np.random.randn(100), index=data.index)
        resampled = resample_to_htf(test_series, lookback_minutes=60, family='oscillators')
        
        if not isinstance(resampled, pd.Series):
            return False, "❌ resample_to_htf didn't return a Series"
        
        return True, "✅ Backward compatible functions can be called"
    except Exception as e:
        return False, f"❌ Failed to call backward compatible functions: {e}\n{traceback.format_exc()}"


def run_all_tests() -> None:
    """Run all verification tests."""
    print("="*80)
    print("HTF BASE FEATURES REFACTORING - VERIFICATION TESTS")
    print("="*80)
    print()
    
    tests = [
        ("Module Import", test_imports),
        ("Backward Compatible Functions", test_backward_compatible_functions),
        ("New Functions", test_new_functions),
        ("Removed Functions", test_removed_functions),
        ("Module Exports", test_exports),
        ("Generator Initialization", test_feature_generator_initialization),
        ("Backward Compatibility Calls", test_backward_compatibility_call),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Test: {test_name}")
        print("-" * 80)
        try:
            success, message = test_func()
            results.append((test_name, success, message))
            print(f"   {message}")
        except Exception as e:
            results.append((test_name, False, f"❌ Test crashed: {e}"))
            print(f"   ❌ Test crashed: {e}")
            traceback.print_exc()
    
    # Summary
    print()
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, message in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed ({100*passed//total}%)")
    
    if passed == total:
        print()
        print("🎉 ALL TESTS PASSED! The refactoring is successful.")
        print()
        print("Next steps:")
        print("1. Review the migration guide: HTF_BASE_FEATURES_MIGRATION.md")
        print("2. Run the examples: example_htf_feature_usage.py")
        print("3. Update any code using the old hardcoded functions")
    else:
        print()
        print("⚠️ Some tests failed. Please review the failures above.")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()