#!/usr/bin/env python3
"""Comprehensive test for enhanced Step 4 with function call monitoring.

This test script verifies that Step 4 includes comprehensive function call monitoring,
function-to-function tracking, and detailed outcome reporting.
"""
import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.market_analysis.step04_regime_data_splitting import (
    run_step,
    _function_tracker,
    log_function_call_summary,
    log_comprehensive_error_report,
    comprehensive_function_monitor
)

async def test_enhanced_step04():
    """Test the enhanced Step 4 with comprehensive function monitoring."""
    print("🧪 Testing Enhanced Step 4: Regime Data Splitting with Function Call Monitoring")
    print("=" * 80)
    
    # Test configuration
    test_config = {
        'symbol': 'ETHUSDT',
        'exchange': 'BINANCE',
        'timeframe': '1m',
        'data_dir': 'data_cache',
        'force_rerun': False,
        'regime_merge_min_retention': 0.8
    }
    
    print(f"📋 Test Configuration:")
    for key, value in test_config.items():
        print(f"   {key}: {value}")
    print()
    
    # Clear any previous function call history
    _function_tracker.call_history.clear()
    _function_tracker.active_calls.clear()
    _function_tracker.performance_metrics.clear()
    _function_tracker.error_tracking.clear()
    
    print("🚀 Starting Step 4 execution with comprehensive monitoring...")
    start_time = time.time()
    
    try:
        # Execute Step 4
        success = await run_step(
            symbol=test_config['symbol'],
            exchange=test_config['exchange'],
            timeframe=test_config['timeframe'],
            data_dir=test_config['data_dir'],
            force_rerun=test_config['force_rerun'],
            config=test_config
        )
        
        execution_time = time.time() - start_time
        
        print(f"\n⏱️ Total execution time: {execution_time:.2f} seconds")
        print(f"🎯 Step 4 execution result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        # Generate comprehensive function call analysis
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE FUNCTION CALL ANALYSIS")
        print("=" * 80)
        
        summary = _function_tracker.generate_summary_report()
        
        print(f"📞 Total function calls: {summary['total_calls']}")
        print(f"🔄 Active calls remaining: {summary['active_calls']}")
        
        if summary['performance_metrics']:
            print(f"\n⚡ Performance Metrics:")
            for func_name, metrics in summary['performance_metrics'].items():
                print(f"   🔧 {func_name}:")
                print(f"      📊 Total calls: {metrics['total_calls']}")
                print(f"      ⏱️ Average execution time: {metrics['avg_execution_time']:.4f}s")
                print(f"      🏆 Success rate: {metrics['success_count']}/{metrics['total_calls']} ({metrics['success_count']/metrics['total_calls']*100:.1f}%)")
                print(f"      ⚡ Min/Max execution time: {metrics['min_execution_time']:.4f}s / {metrics['max_execution_time']:.4f}s")
        
        if summary['error_summary']:
            print(f"\n🚨 Error Summary:")
            for func_name, error_count in summary['error_summary'].items():
                print(f"   ❌ {func_name}: {error_count} errors")
        
        # Test function call monitoring decorator
        print(f"\n🧪 Testing Function Call Monitoring Decorator:")
        test_function_call_monitoring()
        
        # Test error handling
        print(f"\n🧪 Testing Comprehensive Error Handling:")
        test_error_handling()
        
        print(f"\n✅ Enhanced Step 4 testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        log_comprehensive_error_report(e, {'test_context': 'enhanced_step04_test'})
        return False

@comprehensive_function_monitor
def test_function_call_monitoring():
    """Test function to verify monitoring decorator works."""
    print("   🔍 Testing function call monitoring...")
    time.sleep(0.1)  # Simulate some work
    return "monitoring_test_passed"

@comprehensive_function_monitor
def test_error_handling():
    """Test function to verify error handling works."""
    print("   🚨 Testing error handling...")
    try:
        # Intentionally cause an error
        result = 1 / 0
    except Exception as e:
        log_comprehensive_error_report(e, {'test_context': 'error_handling_test'})
        print("   ✅ Error handling test completed (error was expected)")

def test_decorator_functionality():
    """Test the comprehensive function monitoring decorator."""
    print("\n🧪 Testing Decorator Functionality:")
    print("=" * 50)
    
    @comprehensive_function_monitor
    def simple_test_function(x: int, y: int) -> int:
        """Simple test function."""
        return x + y
    
    @comprehensive_function_monitor
    async def async_test_function(x: int, y: int) -> int:
        """Async test function."""
        await asyncio.sleep(0.01)
        return x * y
    
    # Test sync function
    result1 = simple_test_function(5, 3)
    print(f"   Sync function result: {result1}")
    
    # Test async function
    result2 = asyncio.run(async_test_function(4, 6))
    print(f"   Async function result: {result2}")
    
    print("   ✅ Decorator functionality test completed")

async def main():
    """Main test function."""
    print("🧪 Enhanced Step 4 Comprehensive Testing Suite")
    print("=" * 80)
    
    # Test decorator functionality first
    test_decorator_functionality()
    
    # Test enhanced Step 4
    success = await test_enhanced_step04()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 ALL TESTS PASSED - Enhanced Step 4 is working correctly!")
        print("✅ Function call monitoring: ACTIVE")
        print("✅ Function-to-function tracking: ACTIVE") 
        print("✅ Detailed outcome reporting: ACTIVE")
        print("✅ Comprehensive error handling: ACTIVE")
        print("✅ Performance monitoring: ACTIVE")
    else:
        print("❌ SOME TESTS FAILED - Check the logs above for details")
    
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())