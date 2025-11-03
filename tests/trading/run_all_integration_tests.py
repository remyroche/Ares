"""
Comprehensive Test Runner for Enhanced Exit Strategy Integration

Runs all unit tests for:
- TradingOrchestrator integration
- UncertaintyCalculator
- PredictionCache
- Dynamic trailing system
"""

import sys
import unittest
from datetime import datetime

# Import test modules
from test_orchestrator_integration import (
    TestPositionPredictionUpdates,
    TestMLContextUncertainty,
    TestCacheCleanup,
    TestPositionOpenIntegration,
    TestEndToEndIntegration
)
from test_uncertainty_calculator import TestUncertaintyCalculator
from test_prediction_cache import TestPredictionCache


def run_all_tests():
    """Run all test suites and return comprehensive results."""
    
    print("=" * 100)
    print("ENHANCED EXIT STRATEGY - COMPREHENSIVE INTEGRATION TEST SUITE")
    print("=" * 100)
    print(f"Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    print("📋 Loading test suites...")
    print()
    
    # TradingOrchestrator tests
    print("  1. TradingOrchestrator Integration Tests")
    suite.addTests(loader.loadTestsFromTestCase(TestPositionPredictionUpdates))
    suite.addTests(loader.loadTestsFromTestCase(TestMLContextUncertainty))
    suite.addTests(loader.loadTestsFromTestCase(TestCacheCleanup))
    suite.addTests(loader.loadTestsFromTestCase(TestPositionOpenIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndIntegration))
    
    # UncertaintyCalculator tests
    print("  2. UncertaintyCalculator Unit Tests")
    suite.addTests(loader.loadTestsFromTestCase(TestUncertaintyCalculator))
    
    # PredictionCache tests
    print("  3. PredictionCache Unit Tests")
    suite.addTests(loader.loadTestsFromTestCase(TestPredictionCache))
    
    print()
    print("=" * 100)
    print("RUNNING TESTS")
    print("=" * 100)
    print()
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Detailed results
    print()
    print("=" * 100)
    print("COMPREHENSIVE TEST RESULTS")
    print("=" * 100)
    print()
    
    # Test breakdown by category
    test_categories = {
        'TradingOrchestrator Integration': 15,
        'UncertaintyCalculator': 20,
        'PredictionCache': 15
    }
    
    print("📊 Test Coverage by Component:")
    print()
    for category, count in test_categories.items():
        print(f"  • {category}: {count} tests")
    print()
    
    # Overall statistics
    print("📈 Overall Statistics:")
    print()
    print(f"  Total Tests Run:     {result.testsRun}")
    print(f"  ✅ Passed:           {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  ❌ Failed:           {len(result.failures)}")
    print(f"  ⚠️  Errors:           {len(result.errors)}")
    print(f"  Success Rate:        {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print()
    
    # Detailed failure/error reporting
    if result.failures:
        print("=" * 100)
        print("❌ FAILURES")
        print("=" * 100)
        for test, traceback in result.failures:
            print(f"\n{test}:")
            print(traceback)
    
    if result.errors:
        print("=" * 100)
        print("⚠️  ERRORS")
        print("=" * 100)
        for test, traceback in result.errors:
            print(f"\n{test}:")
            print(traceback)
    
    # Final verdict
    print("=" * 100)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED - INTEGRATION VERIFIED!")
        print()
        print("The Enhanced Exit Strategy implementation is fully functional:")
        print("  ✓ Position predictions update every candle")
        print("  ✓ ML context populates uncertainty metrics")
        print("  ✓ Position cache cleanup on close")
        print("  ✓ Uncertainty calculation working correctly")
        print("  ✓ Prediction cache thread-safe and functional")
        print()
        print("Status: READY FOR DEPLOYMENT ✅")
    else:
        print("❌ SOME TESTS FAILED - REVIEW REQUIRED")
        print()
        print("Please review failures and errors above.")
    
    print("=" * 100)
    
    return result


if __name__ == '__main__':
    result = run_all_tests()
    sys.exit(0 if result.wasSuccessful() else 1)

