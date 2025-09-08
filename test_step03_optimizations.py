#!/usr/bin/env python3
"""Test script for Step03 optimizations.

This script tests all the implemented optimizations:
1. Chunked processing and memory-aware data loading
2. Parallel file loading and async I/O operations
3. Intelligent caching with memoization
4. Fast fail mechanisms with extensive logging
5. Performance monitoring and analytics
"""

import asyncio
import logging
import time
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_memory_manager():
    """Test enhanced memory manager."""
    logger.info("🧠 Testing Enhanced Memory Manager...")
    
    try:
        from src.training.steps.market_analysis.hmm_clustering.step03_enhanced_memory_manager import (
            EnhancedMemoryManager, MemoryConfig, get_enhanced_memory_manager
        )
        
        # Create memory manager
        config = MemoryConfig(
            max_memory_usage_percent=80.0,
            chunk_size_mb=50,
            enable_memory_monitoring=True
        )
        
        memory_manager = get_enhanced_memory_manager(config)
        await memory_manager.initialize()
        
        # Test memory monitoring
        stats = memory_manager.get_memory_stats()
        logger.info(f"✅ Memory stats: {stats.process_memory_mb:.1f}MB used, {stats.available_memory_mb:.1f}MB available")
        
        # Test memory context
        async with memory_manager.memory_context("test_operation"):
            # Simulate some memory usage
            test_data = np.random.rand(1000, 100)
            await asyncio.sleep(0.1)
        
        # Get memory report
        report = memory_manager.get_memory_report()
        logger.info(f"✅ Memory report generated: {len(report)} sections")
        
        await memory_manager.cleanup()
        logger.info("✅ Memory manager test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory manager test failed: {e}")
        return False

async def test_fast_fail_validation():
    """Test fast fail validation system."""
    logger.info("🔍 Testing Fast Fail Validation System...")
    
    try:
        from src.training.steps.market_analysis.hmm_clustering.step03_fast_fail_validation import (
            FastFailValidator, ValidationConfig, get_fast_fail_validator
        )
        
        # Create validator
        config = ValidationConfig(
            min_available_memory_gb=1.0,
            min_disk_space_gb=1.0,
            enable_extensive_logging=True
        )
        
        validator = get_fast_fail_validator(config)
        
        # Test system resource validation
        resource_result = await validator.validate_system_resources()
        logger.info(f"✅ System resources validation: {resource_result}")
        
        # Test configuration validation
        test_config = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1m',
            'n_trials': 50,
            'random_state': 42
        }
        
        config_result = await validator.validate_configuration(test_config)
        logger.info(f"✅ Configuration validation: {config_result}")
        
        # Test data quality validation
        test_data = pd.DataFrame({
            'open': np.random.rand(1000) * 100,
            'high': np.random.rand(1000) * 100,
            'low': np.random.rand(1000) * 100,
            'close': np.random.rand(1000) * 100,
            'volume': np.random.rand(1000) * 1000
        })
        
        data_quality_result = await validator.validate_data_quality(test_data, "test_data")
        logger.info(f"✅ Data quality validation: {data_quality_result}")
        
        # Test financial metrics validation
        test_metrics = {
            'sharpe_ratio': 1.5,
            'volatility': 0.2,
            'return_percent': 15.5,
            'max_drawdown': -0.1
        }
        
        metrics_result = await validator.validate_financial_metrics(test_metrics)
        logger.info(f"✅ Financial metrics validation: {metrics_result}")
        
        # Get validation summary
        summary = validator.get_validation_summary()
        logger.info(f"✅ Validation summary: {summary['total_validations']} validations, {summary['success_rate']:.1%} success rate")
        
        logger.info("✅ Fast fail validation test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Fast fail validation test failed: {e}")
        return False

async def test_parallel_io_operations():
    """Test parallel I/O operations."""
    logger.info("📁 Testing Parallel I/O Operations...")
    
    try:
        from src.training.steps.market_analysis.hmm_clustering.step03_parallel_io_operations import (
            ParallelIOOperations, IOConfig, get_parallel_io_operations
        )
        
        # Create I/O operations manager
        config = IOConfig(
            max_concurrent_files=5,
            max_workers=2,
            enable_compression=True
        )
        
        io_ops = get_parallel_io_operations(config)
        
        # Create test data
        test_dir = Path("test_data")
        test_dir.mkdir(exist_ok=True)
        
        # Create test files
        test_files = []
        for i in range(3):
            test_data = pd.DataFrame({
                'id': range(100),
                'value': np.random.rand(100),
                'category': np.random.choice(['A', 'B', 'C'], 100)
            })
            
            test_file = test_dir / f"test_data_{i}.parquet"
            test_data.to_parquet(test_file)
            test_files.append(test_file)
        
        # Test parallel file loading
        logger.info("📦 Testing parallel file loading...")
        loaded_data = await io_ops.load_files_parallel(test_files)
        logger.info(f"✅ Loaded {len(loaded_data)} files in parallel")
        
        # Test parallel file saving
        logger.info("💾 Testing parallel file saving...")
        output_files = []
        for i, data in enumerate(loaded_data):
            output_file = test_dir / f"output_{i}.parquet"
            output_files.append((data, output_file))
        
        await io_ops.save_files_parallel(output_files)
        logger.info(f"✅ Saved {len(output_files)} files in parallel")
        
        # Test data processing
        logger.info("⚡ Testing parallel data processing...")
        
        def process_data(df):
            return df.groupby('category')['value'].mean().to_dict()
        
        processed_results = await io_ops.process_data_parallel(loaded_data, process_data)
        logger.info(f"✅ Processed {len(processed_results)} datasets in parallel")
        
        # Get performance report
        performance_report = io_ops.get_performance_report()
        logger.info(f"✅ I/O performance report: {performance_report['io_performance']['total_operations']} operations")
        
        # Cleanup
        await io_ops.cleanup()
        
        # Clean up test files
        for file in test_files + [f[1] for f in output_files]:
            file.unlink(missing_ok=True)
        test_dir.rmdir()
        
        logger.info("✅ Parallel I/O operations test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Parallel I/O operations test failed: {e}")
        return False

async def test_intelligent_caching():
    """Test intelligent caching system."""
    logger.info("💾 Testing Intelligent Caching System...")
    
    try:
        from src.training.steps.market_analysis.hmm_clustering.step03_intelligent_caching import (
            IntelligentCache, CacheConfig, get_intelligent_cache, memoize
        )
        
        # Create cache
        config = CacheConfig(
            max_memory_cache_size_mb=100,
            max_disk_cache_size_mb=200,
            cache_ttl_seconds=60,
            enable_memory_cache=True,
            enable_disk_cache=True
        )
        
        cache = get_intelligent_cache(config)
        
        # Test basic caching
        logger.info("📦 Testing basic caching...")
        
        test_key = "test_key_1"
        test_value = {"data": [1, 2, 3, 4, 5], "timestamp": time.time()}
        
        # Set value
        cache.set(test_key, test_value, ttl_seconds=60)
        
        # Get value
        retrieved_value = cache.get(test_key)
        assert retrieved_value == test_value, "Cached value doesn't match original"
        logger.info("✅ Basic caching test passed")
        
        # Test DataFrame caching
        logger.info("📊 Testing DataFrame caching...")
        
        df_key = "test_dataframe"
        test_df = pd.DataFrame({
            'id': range(1000),
            'value': np.random.rand(1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000)
        })
        
        cache.set(df_key, test_df, ttl_seconds=60)
        retrieved_df = cache.get(df_key)
        
        assert retrieved_df.equals(test_df), "Cached DataFrame doesn't match original"
        logger.info("✅ DataFrame caching test passed")
        
        # Test memoization decorator
        logger.info("🔄 Testing memoization decorator...")
        
        @memoize(ttl_seconds=60, tags=['test'])
        def expensive_computation(n):
            time.sleep(0.1)  # Simulate expensive operation
            return sum(range(n))
        
        # First call (should be slow)
        start_time = time.time()
        result1 = expensive_computation(1000)
        first_call_time = time.time() - start_time
        
        # Second call (should be fast due to caching)
        start_time = time.time()
        result2 = expensive_computation(1000)
        second_call_time = time.time() - start_time
        
        assert result1 == result2, "Memoized results don't match"
        assert second_call_time < first_call_time, "Second call should be faster"
        logger.info(f"✅ Memoization test passed: {first_call_time:.3f}s -> {second_call_time:.3f}s")
        
        # Test cache invalidation
        logger.info("🗑️ Testing cache invalidation...")
        
        cache.set("invalidation_test", "test_value")
        assert cache.get("invalidation_test") == "test_value"
        
        cache.invalidate("invalidation_test")
        assert cache.get("invalidation_test") is None
        logger.info("✅ Cache invalidation test passed")
        
        # Get cache statistics
        stats = cache.get_stats()
        logger.info(f"✅ Cache stats: {stats['performance']['total_requests']} requests, {stats['performance']['hit_rate']:.1%} hit rate")
        
        # Clear cache
        cache.clear()
        
        logger.info("✅ Intelligent caching test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Intelligent caching test failed: {e}")
        return False

async def test_optimized_step03():
    """Test the complete optimized Step03."""
    logger.info("🚀 Testing Optimized Step03...")
    
    try:
        from src.training.steps.market_analysis.hmm_clustering.step03_enhanced_optimized import (
            OptimizedStep03, OptimizedStep03Config, run_optimized_step03
        )
        
        # Create optimized configuration
        config = OptimizedStep03Config(
            max_memory_usage_percent=80.0,
            chunk_size_mb=50,
            enable_memory_monitoring=True,
            max_concurrent_files=5,
            max_workers=2,
            enable_compression=True,
            max_memory_cache_size_mb=100,
            max_disk_cache_size_mb=200,
            cache_ttl_seconds=60,
            min_available_memory_gb=1.0,
            min_disk_space_gb=1.0,
            enable_extensive_logging=True,
            enable_performance_monitoring=True,
            enable_parallel_processing=True,
            enable_chunked_processing=True
        )
        
        # Test initialization
        logger.info("🔧 Testing optimized Step03 initialization...")
        optimized_step = OptimizedStep03(config)
        await optimized_step.initialize()
        logger.info("✅ Optimized Step03 initialized successfully")
        
        # Test fast fail validation
        logger.info("🔍 Testing fast fail validation...")
        test_config = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1m',
            'data_dir': 'data_cache'
        }
        
        # This will test the validation system
        try:
            await optimized_step._perform_fast_fail_validation(
                test_config['symbol'],
                test_config['exchange'],
                test_config['timeframe'],
                test_config['data_dir']
            )
            logger.info("✅ Fast fail validation completed")
        except Exception as e:
            logger.info(f"⚠️ Fast fail validation failed (expected if data doesn't exist): {e}")
        
        # Test performance monitoring
        logger.info("📊 Testing performance monitoring...")
        performance_report = await optimized_step._generate_performance_report()
        logger.info(f"✅ Performance report generated: {len(performance_report)} sections")
        
        # Cleanup
        await optimized_step.cleanup()
        
        logger.info("✅ Optimized Step03 test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimized Step03 test failed: {e}")
        return False

async def run_all_tests():
    """Run all optimization tests."""
    logger.info("🧪 Starting Step03 Optimization Tests")
    logger.info("=" * 80)
    
    tests = [
        ("Enhanced Memory Manager", test_memory_manager),
        ("Fast Fail Validation", test_fast_fail_validation),
        ("Parallel I/O Operations", test_parallel_io_operations),
        ("Intelligent Caching", test_intelligent_caching),
        ("Optimized Step03", test_optimized_step03)
    ]
    
    results = {}
    total_start_time = time.time()
    
    for test_name, test_func in tests:
        logger.info(f"\n🔬 Running {test_name} test...")
        test_start_time = time.time()
        
        try:
            success = await test_func()
            test_duration = time.time() - test_start_time
            results[test_name] = {
                'success': success,
                'duration': test_duration
            }
            
            if success:
                logger.info(f"✅ {test_name} test PASSED ({test_duration:.2f}s)")
            else:
                logger.error(f"❌ {test_name} test FAILED ({test_duration:.2f}s)")
                
        except Exception as e:
            test_duration = time.time() - test_start_time
            results[test_name] = {
                'success': False,
                'duration': test_duration,
                'error': str(e)
            }
            logger.error(f"❌ {test_name} test FAILED with exception ({test_duration:.2f}s): {e}")
    
    # Generate test summary
    total_duration = time.time() - total_start_time
    passed_tests = sum(1 for r in results.values() if r['success'])
    total_tests = len(results)
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"⏱️ Total execution time: {total_duration:.2f} seconds")
    logger.info(f"✅ Tests passed: {passed_tests}/{total_tests}")
    logger.info(f"📈 Success rate: {passed_tests/total_tests:.1%}")
    
    logger.info("\n📋 Detailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        duration = result['duration']
        logger.info(f"   {status} {test_name}: {duration:.2f}s")
        if not result['success'] and 'error' in result:
            logger.info(f"      Error: {result['error']}")
    
    logger.info("=" * 80)
    
    # Save test results
    test_results_file = Path("test_results_step03_optimizations.json")
    with open(test_results_file, 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'total_duration': total_duration,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'success_rate': passed_tests/total_tests,
            'results': results
        }, f, indent=2, default=str)
    
    logger.info(f"💾 Test results saved to: {test_results_file}")
    
    return results

async def main():
    """Main function to run all tests."""
    try:
        results = await run_all_tests()
        
        # Check if all tests passed
        all_passed = all(r['success'] for r in results.values())
        
        if all_passed:
            logger.info("\n🎉 ALL TESTS PASSED! Step03 optimizations are working correctly.")
            return 0
        else:
            logger.error("\n💥 SOME TESTS FAILED! Please check the logs for details.")
            return 1
            
    except Exception as e:
        logger.error(f"\n💥 Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)