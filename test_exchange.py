#!/usr/bin/env python3
"""
Simple Exchange Interface Tester

A simplified CLI tool for testing ExchangeInterface functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from exchange_interface_test_suite import ExchangeInterfaceTestSuite, TestConfig

async def quick_test():
    """Run a quick test of the exchange interface."""
    print("🚀 Quick Exchange Interface Test")
    print("=" * 50)
    
    # Create a simple test configuration
    config = TestConfig(
        exchange_type="simulated",
        test_symbol="BTCUSDT",
        test_interval="1m",
        test_quantity=0.001,
        verbose=True,
        test_operations=["connection", "klines", "balance", "ticker"]
    )
    
    # Run the test suite
    test_suite = ExchangeInterfaceTestSuite(config)
    
    try:
        results = await test_suite.run_all_tests()
        
        print("\n" + "=" * 50)
        print("📊 Quick Test Results:")
        print(f"✅ Passed: {results.passed_tests}")
        print(f"❌ Failed: {results.failed_tests}")
        print(f"⏱️ Duration: {results.total_duration:.2f}s")
        
        if results.failed_tests > 0:
            print("\n❌ Some tests failed:")
            for result in results.results:
                if not result.success:
                    print(f"  - {result.operation}: {result.error}")
        
        return results.failed_tests == 0
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    finally:
        await test_suite.cleanup()

if __name__ == "__main__":
    success = asyncio.run(quick_test())
    sys.exit(0 if success else 1)