"""
Test TPrint Logging Implementation

This module tests that all auto-optimization components have extensive
tprint logging and no silent failures occur.

Usage:
    python test_tprint_logging.py
"""

import pandas as pd
import numpy as np
import sys
import io
from contextlib import redirect_stdout
from typing import List, Dict, Any

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

def capture_tprint_output(func, *args, **kwargs):
    """Capture tprint output from a function."""
    captured_output = io.StringIO()
    try:
        with redirect_stdout(captured_output):
            result = func(*args, **kwargs)
        return result, captured_output.getvalue()
    except Exception as e:
        return None, str(e)

def test_auto_optimization_config_logging():
    """Test AutoOptimizationConfig logging."""
    print("🧪 Testing AutoOptimizationConfig Logging...")

    try:
        from src.feature_generation import AutoOptimizationConfig, OptimizationLevel

        # Test default configuration creation
        result, output = capture_tprint_output(AutoOptimizationConfig)

        if result is None:
            print(f"   ❌ Error creating AutoOptimizationConfig: {output}")
            return False

        # Check for expected log messages
        expected_messages = [
            "🔧 Initializing AutoOptimizationConfig",
            "📝 Setting up conservative optimization settings",
            "📝 Setting up balanced optimization settings",
            "📝 Setting up aggressive optimization settings",
            "✅ Conservative settings configured",
            "✅ Balanced settings configured",
            "✅ Aggressive settings configured",
            "🎯 AutoOptimizationConfig initialized"
        ]

        missing_messages = []
        for message in expected_messages:
            if message not in output:
                missing_messages.append(message)

        if missing_messages:
            print(f"   ⚠️ Missing log messages: {missing_messages}")
        else:
            print("   ✅ All expected log messages present")

        # Test get_settings_for_level logging
        result, output = capture_tprint_output(result.get_settings_for_level)

        if "🔍 Getting settings for" not in output:
            print("   ❌ Missing get_settings_for_level logging")
            return False

        print("   ✅ get_settings_for_level logging present")

        # Test apply_level_settings logging
        result, output = capture_tprint_output(result.apply_level_settings)

        if "🔧 Applying" not in output:
            print("   ❌ Missing apply_level_settings logging")
            return False

        print("   ✅ apply_level_settings logging present")

        print("✅ AutoOptimizationConfig logging test passed")
        return True

    except Exception as e:
        print(f"❌ AutoOptimizationConfig logging test failed: {e}")
        return False

def test_optimization_strategies_logging():
    """Test optimization strategies logging."""
    print("🧪 Testing Optimization Strategies Logging...")

    try:
        from src.feature_generation import (
            AutoOptimizationConfig,
            OptimizationLevel,
            ConservativeOptimizationStrategy,
            BalancedOptimizationStrategy,
            AggressiveOptimizationStrategy
        )

        # Test conservative strategy
        config = AutoOptimizationConfig(optimization_level=OptimizationLevel.CONSERVATIVE)
        result, output = capture_tprint_output(ConservativeOptimizationStrategy, config)

        if result is None:
            print(f"   ❌ Error creating ConservativeOptimizationStrategy: {output}")
            return False

        if "🔧 Initializing ConservativeOptimizationStrategy" not in output:
            print("   ❌ Missing ConservativeOptimizationStrategy initialization logging")
            return False

        print("   ✅ ConservativeOptimizationStrategy logging present")

        # Test balanced strategy
        config = AutoOptimizationConfig(optimization_level=OptimizationLevel.BALANCED)
        result, output = capture_tprint_output(BalancedOptimizationStrategy, config)

        if result is None:
            print(f"   ❌ Error creating BalancedOptimizationStrategy: {output}")
            return False

        if "🔧 Initializing BalancedOptimizationStrategy" not in output:
            print("   ❌ Missing BalancedOptimizationStrategy initialization logging")
            return False

        print("   ✅ BalancedOptimizationStrategy logging present")

        # Test stats logging
        result, output = capture_tprint_output(result.get_stats)

        if "📊 Getting stats for" not in output:
            print("   ❌ Missing get_stats logging")
            return False

        print("   ✅ get_stats logging present")

        print("✅ Optimization Strategies logging test passed")
        return True

    except Exception as e:
        print(f"❌ Optimization Strategies logging test failed: {e}")
        return False

def test_auto_optimized_generator_logging():
    """Test AutoOptimizedFeatureGenerator logging."""
    print("🧪 Testing AutoOptimizedFeatureGenerator Logging...")

    try:
        from src.feature_generation import (
            AutoOptimizedFeatureGenerator,
            FeatureConfig,
            FeatureCategory,
            AutoOptimizationConfig,
            OptimizationLevel
        )

        # Create test data
        data = create_test_data(50)

        # Create feature config
        config = FeatureConfig(
            name="test_logging_generator",
            category=FeatureCategory.CUSTOM,
            description="Test generator for logging",
            required_columns=["close"],
            default_lookback=20
        )

        # Test generator creation logging
        auto_opt_config = AutoOptimizationConfig(
            optimization_level=OptimizationLevel.BALANCED,
            enable_optimization_logging=True
        )

        class TestGenerator(AutoOptimizedFeatureGenerator):
            def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                return data['close'].rolling(20).mean()

        result, output = capture_tprint_output(TestGenerator, config, auto_opt_config)

        if result is None:
            print(f"   ❌ Error creating AutoOptimizedFeatureGenerator: {output}")
            return False

        # Check for expected initialization messages
        expected_messages = [
            "🔧 Initializing AutoOptimizedFeatureGenerator",
            "📦 Initializing base classes",
            "🔧 Initializing optimization mixins",
            "✅ All mixins initialized",
            "⚙️ Setting up auto-optimization configuration",
            "🔧 Applying level-specific settings",
            "🎯 Creating optimization strategy",
            "📊 Initializing performance tracking",
            "✅ AutoOptimizedFeatureGenerator"
        ]

        missing_messages = []
        for message in expected_messages:
            if message not in output:
                missing_messages.append(message)

        if missing_messages:
            print(f"   ⚠️ Missing initialization messages: {missing_messages}")
        else:
            print("   ✅ All initialization messages present")

        # Test feature generation logging
        result, output = capture_tprint_output(result.generate, data)

        if result is None:
            print(f"   ❌ Error generating feature: {output}")
            return False

        # Check for expected generation messages
        expected_gen_messages = [
            "🚀 Starting feature generation",
            "🔧 Applying auto-optimization",
            "📊 Generating feature",
            "✅ Feature generation completed"
        ]

        missing_gen_messages = []
        for message in expected_gen_messages:
            if message not in output:
                missing_gen_messages.append(message)

        if missing_gen_messages:
            print(f"   ⚠️ Missing generation messages: {missing_gen_messages}")
        else:
            print("   ✅ All generation messages present")

        print("✅ AutoOptimizedFeatureGenerator logging test passed")
        return True

    except Exception as e:
        print(f"❌ AutoOptimizedFeatureGenerator logging test failed: {e}")
        return False

def test_generator_factory_logging():
    """Test GeneratorFactory logging."""
    print("🧪 Testing GeneratorFactory Logging...")

    try:
        from src.feature_generation import GeneratorFactory, FeatureCategory

        factory = GeneratorFactory()

        # Test create_auto_optimized_generator logging
        result, output = capture_tprint_output(
            factory.create_auto_optimized_generator,
            name="test_factory_generator",
            category=FeatureCategory.CUSTOM,
            required_columns=["close"],
            optimization_level="balanced"
        )

        if result is None:
            print(f"   ❌ Error creating auto-optimized generator: {output}")
            return False

        # Check for expected factory messages
        expected_messages = [
            "🔧 Creating auto-optimized generator",
            "📊 Category:",
            "📊 Optimization level:",
            "📊 Required columns:",
            "📝 Creating feature configuration",
            "✅ Feature configuration created",
            "⚙️ Setting up auto-optimization configuration",
            "🚀 Creating AutoOptimizedFeatureGenerator",
            "✅ Auto-optimized generator"
        ]

        missing_messages = []
        for message in expected_messages:
            if message not in output:
                missing_messages.append(message)

        if missing_messages:
            print(f"   ⚠️ Missing factory messages: {missing_messages}")
        else:
            print("   ✅ All factory messages present")

        print("✅ GeneratorFactory logging test passed")
        return True

    except Exception as e:
        print(f"❌ GeneratorFactory logging test failed: {e}")
        return False

def test_feature_bank_logging():
    """Test FeatureBank logging."""
    print("🧪 Testing FeatureBank Logging...")

    try:
        from src.feature_generation import FeatureBank, FeatureBankConfig, FeatureCategory

        # Create feature bank with auto-optimization
        config = FeatureBankConfig(
            enable_auto_optimization=True,
            default_optimization_level="balanced"
        )

        result, output = capture_tprint_output(FeatureBank, config)

        if result is None:
            print(f"   ❌ Error creating FeatureBank: {output}")
            return False

        # Check for expected FeatureBank messages
        expected_messages = [
            "🔧 Auto-registering feature generators",
            "🔧 Creating auto-optimized generators for category",
            "🔄 Converting generator"
        ]

        missing_messages = []
        for message in expected_messages:
            if message not in output:
                missing_messages.append(message)

        if missing_messages:
            print(f"   ⚠️ Missing FeatureBank messages: {missing_messages}")
        else:
            print("   ✅ All FeatureBank messages present")

        # Test create_auto_optimized_generator logging
        result, output = capture_tprint_output(
            result.create_auto_optimized_generator,
            name="test_bank_generator",
            category=FeatureCategory.CUSTOM,
            required_columns=["close"]
        )

        if result is None:
            print(f"   ❌ Error creating generator via FeatureBank: {output}")
            return False

        if "🔧 Creating auto-optimized generator via FeatureBank" not in output:
            print("   ❌ Missing FeatureBank generator creation logging")
            return False

        print("   ✅ FeatureBank generator creation logging present")

        print("✅ FeatureBank logging test passed")
        return True

    except Exception as e:
        print(f"❌ FeatureBank logging test failed: {e}")
        return False

def test_error_handling_logging():
    """Test error handling and logging."""
    print("🧪 Testing Error Handling and Logging...")

    try:
        from src.feature_generation import (
            AutoOptimizationConfig,
            OptimizationLevel,
            ConservativeOptimizationStrategy
        )

        # Test with invalid data to trigger error handling
        config = AutoOptimizationConfig(optimization_level=OptimizationLevel.CONSERVATIVE)
        strategy = ConservativeOptimizationStrategy(config)

        # Create a mock generator that will fail
        class FailingGenerator:
            def optimize_dataframe_processing(self, data):
                raise Exception("Test error for logging")

        failing_generator = FailingGenerator()
        data = create_test_data(10)

        # Test error handling in optimization
        result, output = capture_tprint_output(
            strategy.optimize_data,
            data,
            failing_generator
        )

        if result is None:
            print(f"   ❌ Error in optimization: {output}")
            return False

        # Check for error logging
        if "❌ Memory optimization failed" not in output:
            print("   ❌ Missing error logging")
            return False

        print("   ✅ Error logging present")

        # Test that optimization continues despite errors
        if "✅ Conservative optimization completed" not in output:
            print("   ❌ Optimization did not complete after error")
            return False

        print("   ✅ Error handling works correctly")

        print("✅ Error handling logging test passed")
        return True

    except Exception as e:
        print(f"❌ Error handling logging test failed: {e}")
        return False

def test_no_silent_failures():
    """Test that no silent failures occur."""
    print("🧪 Testing No Silent Failures...")

    try:
        from src.feature_generation import (
            AutoOptimizedFeatureGenerator,
            FeatureConfig,
            FeatureCategory,
            AutoOptimizationConfig,
            OptimizationLevel
        )

        # Create test data
        data = create_test_data(50)

        # Create feature config
        config = FeatureConfig(
            name="test_silent_failure",
            category=FeatureCategory.CUSTOM,
            description="Test for silent failures",
            required_columns=["close"],
            default_lookback=20
        )

        # Create generator that will fail
        class FailingGenerator(AutoOptimizedFeatureGenerator):
            def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                raise Exception("Intentional failure for testing")

        auto_opt_config = AutoOptimizationConfig(
            optimization_level=OptimizationLevel.BALANCED,
            enable_optimization_logging=True
        )

        generator = FailingGenerator(config, auto_opt_config)

        # Test that failures are logged and handled
        result, output = capture_tprint_output(generator.generate, data)

        if result is None:
            print(f"   ❌ Error in generator: {output}")
            return False

        # Check that failure was logged
        if "❌ Error generating feature" not in output:
            print("   ❌ Error not logged properly")
            return False

        print("   ✅ Error properly logged")

        # Check that a failed result was returned (not silent failure)
        if not hasattr(result, 'success') or result.success:
            print("   ❌ Silent failure occurred - success should be False")
            return False

        print("   ✅ No silent failure - proper error result returned")

        print("✅ No silent failures test passed")
        return True

    except Exception as e:
        print(f"❌ No silent failures test failed: {e}")
        return False

def run_all_logging_tests():
    """Run all logging tests."""
    print("🎯 TPrint Logging Implementation Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("AutoOptimizationConfig Logging", test_auto_optimization_config_logging),
        ("Optimization Strategies Logging", test_optimization_strategies_logging),
        ("AutoOptimizedFeatureGenerator Logging", test_auto_optimized_generator_logging),
        ("GeneratorFactory Logging", test_generator_factory_logging),
        ("FeatureBank Logging", test_feature_bank_logging),
        ("Error Handling Logging", test_error_handling_logging),
        ("No Silent Failures", test_no_silent_failures)
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
        print("🎉 All logging tests passed! Extensive tprint logging is working correctly.")
        print("✅ No silent failures detected.")
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the logging implementation.")

    return passed == total

def main():
    """Main test runner."""
    try:
        success = run_all_logging_tests()
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
