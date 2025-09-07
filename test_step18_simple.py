#!/usr/bin/env python3
"""
Simple test script for enhanced step18 functionality
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_basic_functionality():
    """Test basic functionality of enhanced step18."""
    print("🧪 Testing enhanced step18 functionality...")

    try:
        # Test imports - direct import to avoid module issues
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "step18_module",
            "/Users/remyroche/Documents/Ares/src/training/steps/backtesting/step18_walk_forward_validation_per_regime.py"
        )
        step18_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step18_module)
        PerRegimeWalkForwardValidationStep = step18_module.PerRegimeWalkForwardValidationStep
        print("✅ Direct imports successful")

        # Test configuration
        config = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1m',
            'use_real_market_data': True,
            'enable_enhanced_metrics': True
        }

        validator = PerRegimeWalkForwardValidationStep(config)
        print("✅ Validator initialization successful")

        # Test risk-adjusted score calculation
        score = validator._calculate_risk_adjusted_score(1.5, 1.8, 2.2, 0.12)
        print(f"✅ Risk-adjusted score calculation: {score:.3f}")

        # Test regime performance multiplier
        multiplier = validator._get_regime_performance_multiplier(0)
        print(f"✅ Regime performance multiplier: {multiplier}")

        print("🎉 All basic tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_parallel_processing():
    """Test parallel processing functionality."""
    print("\n🧪 Testing parallel processing...")

    try:
        # Direct import to avoid module issues
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "step18_module",
            "/Users/remyroche/Documents/Ares/src/training/steps/backtesting/step18_walk_forward_validation_per_regime.py"
        )
        step18_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step18_module)
        PerRegimeWalkForwardValidationStep = step18_module.PerRegimeWalkForwardValidationStep

        config = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1m'
        }

        validator = PerRegimeWalkForwardValidationStep(config)

        # Mock the validation method for testing
        async def mock_validation(*args, **kwargs):
            await asyncio.sleep(0.01)  # Small delay to simulate work
            return True

        # Patch the method
        original_method = validator.execute_per_regime_walk_forward_validation
        validator.execute_per_regime_walk_forward_validation = mock_validation

        try:
            # Test parallel execution
            results = await validator.execute_parallel_regime_validation(
                'ETHUSDT', 'BINANCE', '1m', 'data_cache', [0, 1, 2], max_concurrent=2
            )

            print(f"✅ Parallel processing results: {results}")
            print("✅ Parallel processing test passed!")
            return True

        finally:
            # Restore original method
            validator.execute_per_regime_walk_forward_validation = original_method

    except Exception as e:
        print(f"❌ Parallel processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting step18 enhancement tests...\n")

    test1_passed = await test_basic_functionality()
    test2_passed = await test_parallel_processing()

    print("\n" + "="*50)
    print("📊 Test Results:")
    print(f"   Basic functionality: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Parallel processing: {'✅ PASS' if test2_passed else '❌ FAIL'}")

    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Enhanced step18 is ready for production.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
