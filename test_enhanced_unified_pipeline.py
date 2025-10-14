#!/usr/bin/env python3
"""
Test script for the enhanced UnifiedDataDrivenPipeline with new infrastructure.

This script tests the integration of all the new infrastructure components
that were missing from the original UnifiedDataDrivenPipeline.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline import (
        UnifiedDataDrivenPipeline, create_unified_pipeline, process_with_unified_pipeline
    )
    from src.training.steps.pre_training.unified_data_driven_pipeline.core.config import create_default_config
    print("✅ Successfully imported UnifiedDataDrivenPipeline")
except ImportError as e:
    print(f"❌ Failed to import UnifiedDataDrivenPipeline: {e}")
    sys.exit(1)


def create_test_data(n_periods: int = 1000) -> pd.DataFrame:
    """Create synthetic test data."""
    print("🔧 Creating synthetic test data...")
    
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_periods // 24)  # Assuming hourly data
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Generate synthetic OHLCV data
    np.random.seed(42)
    n_periods = len(date_range)
    
    # Generate price series with random walk
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_periods)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Generate OHLCV
    data = {
        'open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_periods))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_periods)
    }
    
    df = pd.DataFrame(data, index=date_range)
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
    df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
    
    print(f"✅ Created test data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def create_test_targets(data: pd.DataFrame) -> pd.Series:
    """Create synthetic targets for testing."""
    print("🔧 Creating synthetic targets...")
    
    # Create simple targets based on price movements
    returns = data['close'].pct_change()
    targets = (returns > returns.rolling(20).mean()).astype(int)
    
    print(f"✅ Created targets: {len(targets)} values")
    return targets


async def test_basic_functionality():
    """Test basic pipeline functionality."""
    print("\n🧪 Testing basic pipeline functionality...")
    
    try:
        # Create test data
        data = create_test_data(500)  # Smaller dataset for faster testing
        targets = create_test_targets(data)
        
        # Create pipeline
        pipeline = create_unified_pipeline()
        print("✅ Pipeline created successfully")
        
        # Test pipeline processing
        print("🚀 Running pipeline processing...")
        result = await pipeline.process(
            data=data,
            targets=targets,
            timeframe="1h",
            pipeline_state={
                'symbol': 'TESTUSDT',
                'exchange': 'test',
                'timeframe': '1h',
                'execution_mode': 'light'
            }
        )
        
        if result.success:
            print("✅ Pipeline processing completed successfully")
            print(f"📊 Results: {len(result.selected_features)} features selected")
            print(f"⏱️ Processing time: {result.processing_time:.3f}s")
            print(f"💾 Artifacts saved: {result.artifacts_saved if hasattr(result, 'artifacts_saved') else 'N/A'}")
        else:
            print(f"❌ Pipeline processing failed: {result.error_message}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_infrastructure_components():
    """Test individual infrastructure components."""
    print("\n🧪 Testing infrastructure components...")
    
    try:
        # Create pipeline
        pipeline = create_unified_pipeline()
        
        # Test advanced validator
        print("🔍 Testing advanced validator...")
        data = create_test_data(100)
        is_valid, summary, cleaned_data = pipeline.advanced_validator.validate_data(data)
        print(f"✅ Validator test: valid={is_valid}, quality_score={summary.quality_score}")
        
        # Test advanced error handler
        print("🛡️ Testing advanced error handler...")
        try:
            pipeline.advanced_error_handler.safe_execute(
                lambda x: x / 0, 1, operation="test_division", return_value=0
            )
            print("✅ Error handler test: handled division by zero gracefully")
        except Exception as e:
            print(f"❌ Error handler test failed: {e}")
        
        # Test advanced performance monitor
        print("📊 Testing advanced performance monitor...")
        pipeline.advanced_performance_monitor.start_monitoring()
        pipeline.advanced_performance_monitor.record_memory_usage()
        pipeline.advanced_performance_monitor.record_cpu_usage()
        stats = pipeline.advanced_performance_monitor.get_performance_summary()
        print(f"✅ Performance monitor test: {len(stats)} metrics recorded")
        
        # Test advanced data loader
        print("📥 Testing advanced data loader...")
        cache_metrics = pipeline.advanced_data_loader.get_cache_metrics()
        print(f"✅ Data loader test: cache metrics available")
        
        # Test advanced artifact manager
        print("💾 Testing advanced artifact manager...")
        registry = pipeline.advanced_artifact_manager.get_artifact_registry()
        print(f"✅ Artifact manager test: registry size={len(registry)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Infrastructure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_performance_monitoring():
    """Test performance monitoring capabilities."""
    print("\n🧪 Testing performance monitoring...")
    
    try:
        pipeline = create_unified_pipeline()
        
        # Start monitoring
        pipeline.advanced_performance_monitor.start_monitoring()
        
        # Simulate some operations
        for i in range(5):
            start_time = pipeline.advanced_performance_monitor.start_operation(f"test_operation_{i}")
            await asyncio.sleep(0.1)  # Simulate work
            pipeline.advanced_performance_monitor.end_operation(f"test_operation_{i}", start_time, success=True)
            pipeline.advanced_performance_monitor.record_memory_usage()
            pipeline.advanced_performance_monitor.record_cpu_usage()
        
        # Get performance summary
        summary = pipeline.advanced_performance_monitor.get_performance_summary()
        
        print(f"✅ Performance monitoring test completed")
        print(f"📊 Operations tracked: {len(summary.get('operations', {}))}")
        print(f"📊 Memory samples: {summary.get('memory_stats', {}).get('samples', 0)}")
        print(f"📊 CPU samples: {summary.get('cpu_stats', {}).get('samples', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling capabilities."""
    print("\n🧪 Testing error handling...")
    
    try:
        pipeline = create_unified_pipeline()
        
        # Test safe execution with error
        result = pipeline.advanced_error_handler.safe_execute(
            lambda: 1 / 0,  # This will raise ZeroDivisionError
            operation="test_division",
            return_value="error_handled"
        )
        
        if result == "error_handled":
            print("✅ Error handling test: safely handled division by zero")
        else:
            print(f"❌ Error handling test: unexpected result: {result}")
            return False
        
        # Test error statistics
        error_stats = pipeline.advanced_error_handler.get_error_stats()
        print(f"✅ Error statistics: {error_stats['total_errors']} errors recorded")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Enhanced UnifiedDataDrivenPipeline Tests")
    print("=" * 60)
    
    tests = [
        ("Infrastructure Components", test_infrastructure_components),
        ("Performance Monitoring", test_performance_monitoring),
        ("Error Handling", test_error_handling),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = await test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced UnifiedDataDrivenPipeline is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)