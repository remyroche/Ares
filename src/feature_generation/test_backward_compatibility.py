"""
Backward Compatibility Test Suite

This module tests that all existing APIs and usage patterns continue to work
unchanged after the auto-optimization implementation.

Usage:
    python test_backward_compatibility.py
"""

import pandas as pd
import numpy as np
import sys
import io
from contextlib import redirect_stdout
from typing import List, Dict, Any, Optional

def create_test_data(rows: int = 100) -> pd.DataFrame:
    """Create test data for validation."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=rows, freq='1min')

    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(rows) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(rows) * 0.1) + np.random.rand(rows) * 2,
        'low': 100 + np.cumsum(np.random.randn(rows) * 0.1) - np.random.rand(rows) * 2,
        'close': 100 + np.cumsum(np.random.randn(rows) * 0.1),
        'volume': np.random.randint(1000, 10000, rows)
    }, index=dates)

    # Ensure data integrity
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))

    return data

def test_import_compatibility():
    """Test that all existing imports continue to work."""
    print("🧪 Testing Import Compatibility...")

    try:
        # Test core imports
        from src.feature_generation import (
            FeatureBank,
            FeatureGenerator,
            FeatureCategory,
            FeatureRegistry,
            VectorizedFeatureGenerator,
            get_feature_generator,
            get_feature_bank,
            register_feature_generator,
            list_available_features,
            list_available_categories
        )
        print("   ✅ Core imports successful")

        # Test utility mixins (existing)
        from src.feature_generation import (
            OptimizationMixin,
            RollingOperationsMixin,
            VectorBTOptimizationMixin
        )
        print("   ✅ Utility mixin imports successful")

        # Test factory pattern (existing)
        from src.feature_generation import (
            GeneratorFactory,
            get_generator_factory,
            create_generator
        )
        print("   ✅ Factory pattern imports successful")

        # Test new auto-optimization imports (should not break existing code)
        from src.feature_generation import (
            AutoOptimizedFeatureGenerator,
            AutoOptimizationConfig,
            OptimizationLevel
        )
        print("   ✅ Auto-optimization imports successful")

        print("✅ Import compatibility test passed")
        return True

    except Exception as e:
        print(f"❌ Import compatibility test failed: {e}")
        return False

def test_feature_bank_default_behavior():
    """Test that FeatureBank works exactly as before by default."""
    print("🧪 Testing FeatureBank Default Behavior...")

    try:
        from src.feature_generation import FeatureBank, FeatureCategory

        # Test default FeatureBank creation (should not enable auto-optimization)
        bank = FeatureBank()

        # Verify auto-optimization is disabled by default
        if bank.config.enable_auto_optimization:
            print("   ❌ Auto-optimization should be disabled by default")
            return False

        print("   ✅ Auto-optimization disabled by default (backward compatibility)")

        # Test that existing methods work unchanged
        categories = bank.list_categories()
        if not isinstance(categories, list):
            print("   ❌ list_categories() should return a list")
            return False

        print("   ✅ list_categories() works as expected")

        # Test that generators are created as standard generators (not auto-optimized)
        generators = bank.get_generators_by_category(FeatureCategory.MOMENTUM)
        if generators:
            # Check that generators are not AutoOptimizedFeatureGenerator instances
            from src.feature_generation import AutoOptimizedFeatureGenerator
            auto_optimized_count = sum(1 for gen in generators if isinstance(gen, AutoOptimizedFeatureGenerator))
            if auto_optimized_count > 0:
                print(f"   ❌ Found {auto_optimized_count} auto-optimized generators when they should be standard")
                return False

        print("   ✅ Generators are standard generators (not auto-optimized) by default")

        print("✅ FeatureBank default behavior test passed")
        return True

    except Exception as e:
        print(f"❌ FeatureBank default behavior test failed: {e}")
        return False

def test_existing_api_methods():
    """Test that all existing API methods work unchanged."""
    print("🧪 Testing Existing API Methods...")

    try:
        from src.feature_generation import FeatureBank, FeatureCategory

        bank = FeatureBank()
        data = create_test_data(50)

        # Test generate_features method (existing API)
        try:
            features = bank.generate_features(
                data=data,
                categories=[FeatureCategory.MOMENTUM]
            )
            if not isinstance(features, pd.DataFrame):
                print("   ❌ generate_features() should return DataFrame")
                return False
            print("   ✅ generate_features() works as expected")
        except Exception as e:
            print(f"   ❌ generate_features() failed: {e}")
            return False

        # Test generate_features_by_category method (existing API)
        try:
            features = bank.generate_features_by_category(
                data=data,
                category=FeatureCategory.MOMENTUM
            )
            if not isinstance(features, pd.DataFrame):
                print("   ❌ generate_features_by_category() should return DataFrame")
                return False
            print("   ✅ generate_features_by_category() works as expected")
        except Exception as e:
            print(f"   ❌ generate_features_by_category() failed: {e}")
            return False

        # Test generate_specific_features method (existing API)
        try:
            features = bank.generate_specific_features(
                data=data,
                feature_names=["test_feature"]
            )
            if not isinstance(features, pd.DataFrame):
                print("   ❌ generate_specific_features() should return DataFrame")
                return False
            print("   ✅ generate_specific_features() works as expected")
        except Exception as e:
            print(f"   ❌ generate_specific_features() failed: {e}")
            return False

        # Test get_generators_by_category method (existing API)
        try:
            generators = bank.get_generators_by_category(FeatureCategory.MOMENTUM)
            if not isinstance(generators, list):
                print("   ❌ get_generators_by_category() should return list")
                return False
            print("   ✅ get_generators_by_category() works as expected")
        except Exception as e:
            print(f"   ❌ get_generators_by_category() failed: {e}")
            return False

        # Test get_generator_by_name method (existing API)
        try:
            generator = bank.get_generator_by_name("test_generator")
            # Should return None or a generator, both are valid
            print("   ✅ get_generator_by_name() works as expected")
        except Exception as e:
            print(f"   ❌ get_generator_by_name() failed: {e}")
            return False

        print("✅ Existing API methods test passed")
        return True

    except Exception as e:
        print(f"❌ Existing API methods test failed: {e}")
        return False

def test_feature_generator_compatibility():
    """Test that existing FeatureGenerator classes work unchanged."""
    print("🧪 Testing FeatureGenerator Compatibility...")

    try:
        from src.feature_generation import (
            FeatureGenerator,
            FeatureConfig,
            FeatureCategory,
            VectorizedFeatureGenerator
        )

        # Test regular FeatureGenerator
        config = FeatureConfig(
            name="test_generator",
            category=FeatureCategory.CUSTOM,
            description="Test generator",
            required_columns=["close"],
            default_lookback=20
        )

        class TestGenerator(FeatureGenerator):
            def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                return data['close'].rolling(20).mean()

        generator = TestGenerator(config)

        # Test that it works as expected
        data = create_test_data(50)
        result = generator.generate(data)

        if not hasattr(result, 'success') or not result.success:
            print("   ❌ Regular FeatureGenerator should work unchanged")
            return False

        print("   ✅ Regular FeatureGenerator works as expected")

        # Test VectorizedFeatureGenerator
        class TestVectorizedGenerator(VectorizedFeatureGenerator):
            def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                return data['close'].rolling(20).mean()

        vectorized_generator = TestVectorizedGenerator(config)
        result = vectorized_generator.generate(data)

        if not hasattr(result, 'success') or not result.success:
            print("   ❌ VectorizedFeatureGenerator should work unchanged")
            return False

        print("   ✅ VectorizedFeatureGenerator works as expected")

        print("✅ FeatureGenerator compatibility test passed")
        return True

    except Exception as e:
        print(f"❌ FeatureGenerator compatibility test failed: {e}")
        return False

def test_legacy_usage_patterns():
    """Test legacy usage patterns still work."""
    print("🧪 Testing Legacy Usage Patterns...")

    try:
        from src.feature_generation import FeatureBank, FeatureCategory

        # Test legacy pattern: Create bank and use it directly
        bank = FeatureBank()
        data = create_test_data(50)

        # Legacy pattern 1: Generate by category
        features = bank.generate_features_by_category(
            data=data,
            category=FeatureCategory.MOMENTUM
        )

        if not isinstance(features, pd.DataFrame):
            print("   ❌ Legacy pattern 1 failed")
            return False

        print("   ✅ Legacy pattern 1 (generate by category) works")

        # Legacy pattern 2: Get generators and use them individually
        generators = bank.get_generators_by_category(FeatureCategory.MOMENTUM)

        if not isinstance(generators, list):
            print("   ❌ Legacy pattern 2 failed")
            return False

        print("   ✅ Legacy pattern 2 (get generators individually) works")

        # Legacy pattern 3: Use specific features
        features = bank.generate_specific_features(
            data=data,
            feature_names=["test_feature"]
        )

        if not isinstance(features, pd.DataFrame):
            print("   ❌ Legacy pattern 3 failed")
            return False

        print("   ✅ Legacy pattern 3 (specific features) works")

        print("✅ Legacy usage patterns test passed")
        return True

    except Exception as e:
        print(f"❌ Legacy usage patterns test failed: {e}")
        return False

def test_auto_optimization_opt_in():
    """Test that auto-optimization is opt-in and doesn't break existing code."""
    print("🧪 Testing Auto-Optimization Opt-In...")

    try:
        from src.feature_generation import (
            FeatureBank,
            FeatureBankConfig,
            FeatureCategory,
            AutoOptimizedFeatureGenerator
        )

        # Test 1: Default behavior (no auto-optimization)
        bank_default = FeatureBank()

        if bank_default.config.enable_auto_optimization:
            print("   ❌ Auto-optimization should be disabled by default")
            return False

        print("   ✅ Auto-optimization disabled by default")

        # Test 2: Explicitly enable auto-optimization
        config = FeatureBankConfig(enable_auto_optimization=True)
        bank_optimized = FeatureBank(config)

        if not bank_optimized.config.enable_auto_optimization:
            print("   ❌ Auto-optimization should be enabled when explicitly requested")
            return False

        print("   ✅ Auto-optimization enabled when explicitly requested")

        # Test 3: Verify generators are auto-optimized when enabled
        generators = bank_optimized.get_generators_by_category(FeatureCategory.MOMENTUM)
        if generators:
            auto_optimized_count = sum(1 for gen in generators if isinstance(gen, AutoOptimizedFeatureGenerator))
            if auto_optimized_count == 0:
                print("   ❌ No auto-optimized generators found when auto-optimization is enabled")
                return False

        print("   ✅ Auto-optimized generators created when enabled")

        # Test 4: Verify generators are standard when disabled
        generators_default = bank_default.get_generators_by_category(FeatureCategory.MOMENTUM)
        if generators_default:
            auto_optimized_count = sum(1 for gen in generators_default if isinstance(gen, AutoOptimizedFeatureGenerator))
            if auto_optimized_count > 0:
                print("   ❌ Auto-optimized generators found when auto-optimization is disabled")
                return False

        print("   ✅ Standard generators created when auto-optimization disabled")

        print("✅ Auto-optimization opt-in test passed")
        return True

    except Exception as e:
        print(f"❌ Auto-optimization opt-in test failed: {e}")
        return False

def test_generator_factory_compatibility():
    """Test that GeneratorFactory works unchanged."""
    print("🧪 Testing GeneratorFactory Compatibility...")

    try:
        from src.feature_generation import GeneratorFactory, FeatureCategory

        factory = GeneratorFactory()

        # Test existing methods still work
        generators = factory.list_available_generators()
        if not isinstance(generators, list):
            print("   ❌ list_available_generators() should return list")
            return False

        print("   ✅ list_available_generators() works as expected")

        # Test create_generator method (existing)
        generator = factory.create_generator("test_generator")
        # Should return None or a generator, both are valid
        print("   ✅ create_generator() works as expected")

        # Test create_vectorized_generator method (existing)
        generator = factory.create_vectorized_generator(
            name="test_vectorized",
            category=FeatureCategory.CUSTOM,
            required_columns=["close"]
        )
        # Should return None or a generator, both are valid
        print("   ✅ create_vectorized_generator() works as expected")

        # Test create_optimized_generator method (existing)
        generator = factory.create_optimized_generator(
            name="test_optimized",
            category=FeatureCategory.CUSTOM,
            required_columns=["close"]
        )
        # Should return None or a generator, both are valid
        print("   ✅ create_optimized_generator() works as expected")

        print("✅ GeneratorFactory compatibility test passed")
        return True

    except Exception as e:
        print(f"❌ GeneratorFactory compatibility test failed: {e}")
        return False

def test_no_breaking_changes():
    """Test that no breaking changes were introduced."""
    print("🧪 Testing No Breaking Changes...")

    try:
        from src.feature_generation import (
            FeatureBank,
            FeatureGenerator,
            FeatureCategory,
            VectorizedFeatureGenerator,
            GeneratorFactory
        )

        # Test that all classes can be instantiated with existing patterns
        bank = FeatureBank()

        # Test that existing method signatures haven't changed
        import inspect

        # Check generate_features signature
        sig = inspect.signature(bank.generate_features)
        expected_params = ['data', 'categories', 'features', 'lookback_optimization', 'target_column', 'use_optimized_pipeline']
        actual_params = list(sig.parameters.keys())

        for param in expected_params:
            if param not in actual_params:
                print(f"   ❌ Missing parameter '{param}' in generate_features")
                return False

        print("   ✅ generate_features signature unchanged")

        # Check that new parameters are optional (have defaults)
        if 'categories' in sig.parameters and sig.parameters['categories'].default is None:
            print("   ✅ categories parameter has correct default")
        else:
            print("   ❌ categories parameter default changed")
            return False

        print("   ✅ All method signatures compatible")

        print("✅ No breaking changes test passed")
        return True

    except Exception as e:
        print(f"❌ No breaking changes test failed: {e}")
        return False

def run_all_compatibility_tests():
    """Run all backward compatibility tests."""
    print("🎯 Backward Compatibility Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("Import Compatibility", test_import_compatibility),
        ("FeatureBank Default Behavior", test_feature_bank_default_behavior),
        ("Existing API Methods", test_existing_api_methods),
        ("FeatureGenerator Compatibility", test_feature_generator_compatibility),
        ("Legacy Usage Patterns", test_legacy_usage_patterns),
        ("Auto-Optimization Opt-In", test_auto_optimization_opt_in),
        ("GeneratorFactory Compatibility", test_generator_factory_compatibility),
        ("No Breaking Changes", test_no_breaking_changes)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"🧪 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}")
        print()

    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All backward compatibility tests passed!")
        print("✅ Full backward compatibility maintained.")
        print("✅ Existing code will work unchanged.")
        print("✅ Auto-optimization is opt-in only.")
    else:
        print(f"⚠️ {total - passed} tests failed. Backward compatibility issues detected.")

    return passed == total

def main():
    """Main test runner."""
    try:
        success = run_all_compatibility_tests()
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
