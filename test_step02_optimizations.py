#!/usr/bin/env python3
"""Test script for Step 2 optimizations.

This script demonstrates the performance improvements and optimizations
implemented in the optimized Step 2 data reading module.
"""
import asyncio
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the optimized modules
from src.training.steps.data_collection.step02_data_reading_optimized import (
    OptimizedDataReadingStep,
    run_step_optimized,
    vectorized_price_validation,
    vectorized_timestamp_validation,
    vectorized_volume_validation,
    fast_fail_file_check,
    fast_fail_schema_check,
    fast_fail_data_size_check
)

from src.training.steps.data_collection.step02_data_reading_validator_optimized import (
    OptimizedStep2Validator,
    run_validator_optimized
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_data(num_rows: int = 10000) -> pd.DataFrame:
    """Create test data for performance testing."""
    logger.info(f"Creating test data with {num_rows} rows...")
    
    # Generate timestamps
    start_time = pd.Timestamp('2024-01-01 00:00:00')
    timestamps = pd.date_range(start=start_time, periods=num_rows, freq='1min')
    
    # Generate OHLCV data
    np.random.seed(42)  # For reproducible results
    base_price = 100.0
    
    # Generate price data with some realistic patterns
    price_changes = np.random.normal(0, 0.01, num_rows)
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))  # Ensure positive prices
    
    # Generate OHLC data
    data = []
    for i, (timestamp, close_price) in enumerate(zip(timestamps, prices)):
        # Generate realistic OHLC from close price
        volatility = np.random.uniform(0.001, 0.005)
        high = close_price * (1 + volatility)
        low = close_price * (1 - volatility)
        open_price = prices[i-1] if i > 0 else close_price
        
        # Generate volume
        volume = np.random.exponential(1000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    
    # Add some data quality issues for testing
    if num_rows > 1000:
        # Add some duplicate timestamps
        df.loc[100:105, 'timestamp'] = df.loc[100, 'timestamp']
        
        # Add some negative prices (should fail validation)
        df.loc[200, 'close'] = -1.0
        
        # Add some infinite values
        df.loc[300, 'volume'] = float('inf')
        
        # Add some NaN values
        df.loc[400, 'open'] = np.nan
    
    logger.info(f"✅ Created test data: {df.shape}")
    return df

def test_vectorized_operations():
    """Test vectorized operations performance."""
    logger.info("🧪 Testing vectorized operations...")
    
    # Create test data
    test_data = create_test_data(50000)
    
    # Test vectorized price validation
    start_time = time.time()
    price_results = vectorized_price_validation(test_data)
    price_time = time.time() - start_time
    
    # Test vectorized timestamp validation
    start_time = time.time()
    timestamp_results = vectorized_timestamp_validation(test_data)
    timestamp_time = time.time() - start_time
    
    # Test vectorized volume validation
    start_time = time.time()
    volume_results = vectorized_volume_validation(test_data)
    volume_time = time.time() - start_time
    
    logger.info("📊 Vectorized Operations Results:")
    logger.info(f"   - Price validation: {price_time:.4f}s")
    logger.info(f"   - Timestamp validation: {timestamp_time:.4f}s")
    logger.info(f"   - Volume validation: {volume_time:.4f}s")
    logger.info(f"   - Total time: {price_time + timestamp_time + volume_time:.4f}s")
    
    # Log validation results
    logger.info("🔍 Validation Results:")
    logger.info(f"   - Negative prices: {price_results['negative_prices']}")
    logger.info(f"   - Infinite prices: {price_results['infinite_prices']}")
    logger.info(f"   - NaN prices: {price_results['nan_prices']}")
    logger.info(f"   - OHLC inconsistencies: {price_results['ohlc_inconsistencies']}")
    logger.info(f"   - Duplicate timestamps: {timestamp_results['duplicate_timestamps']}")
    logger.info(f"   - Large gaps: {timestamp_results['large_gaps']}")
    logger.info(f"   - Negative volumes: {volume_results['negative_volumes']}")
    logger.info(f"   - Zero volumes: {volume_results['zero_volumes']}")
    
    return {
        'price_validation': price_results,
        'timestamp_validation': timestamp_results,
        'volume_validation': volume_results,
        'timings': {
            'price_time': price_time,
            'timestamp_time': timestamp_time,
            'volume_time': volume_time,
            'total_time': price_time + timestamp_time + volume_time
        }
    }

def test_fast_fail_validation():
    """Test fast-fail validation functions."""
    logger.info("🧪 Testing fast-fail validation...")
    
    # Test with valid data
    valid_data = create_test_data(1000)
    
    # Test schema check
    start_time = time.time()
    is_valid, error = fast_fail_schema_check(valid_data)
    schema_time = time.time() - start_time
    
    logger.info(f"✅ Schema check (valid data): {is_valid}, {error}, {schema_time:.6f}s")
    
    # Test data size check
    start_time = time.time()
    is_valid, error = fast_fail_data_size_check(valid_data, 500)
    size_time = time.time() - start_time
    
    logger.info(f"✅ Size check (valid data): {is_valid}, {error}, {size_time:.6f}s")
    
    # Test with invalid data
    invalid_data = valid_data.copy()
    invalid_data = invalid_data.drop(columns=['open', 'high'])  # Remove required columns
    
    start_time = time.time()
    is_valid, error = fast_fail_schema_check(invalid_data)
    invalid_schema_time = time.time() - start_time
    
    logger.info(f"❌ Schema check (invalid data): {is_valid}, {error}, {invalid_schema_time:.6f}s")
    
    # Test with insufficient data
    small_data = create_test_data(100)
    
    start_time = time.time()
    is_valid, error = fast_fail_data_size_check(small_data, 500)
    invalid_size_time = time.time() - start_time
    
    logger.info(f"❌ Size check (insufficient data): {is_valid}, {error}, {invalid_size_time:.6f}s")
    
    return {
        'valid_schema_time': schema_time,
        'valid_size_time': size_time,
        'invalid_schema_time': invalid_schema_time,
        'invalid_size_time': invalid_size_time
    }

async def test_parallel_reading():
    """Test parallel file reading performance."""
    logger.info("🧪 Testing parallel file reading...")
    
    # Create test files
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create multiple test files
    num_files = 10
    file_paths = []
    
    for i in range(num_files):
        test_data = create_test_data(1000)
        file_path = test_dir / f"test_data_{i}.parquet"
        test_data.to_parquet(file_path, index=False)
        file_paths.append(file_path)
    
    logger.info(f"Created {num_files} test files")
    
    # Test parallel reading
    from src.training.steps.data_collection.step02_data_reading_optimized import read_parquet_files_parallel
    
    start_time = time.time()
    dataframes = await read_parquet_files_parallel(file_paths, max_workers=4)
    parallel_time = time.time() - start_time
    
    logger.info(f"✅ Parallel reading: {len(dataframes)} files in {parallel_time:.4f}s")
    
    # Test sequential reading for comparison
    start_time = time.time()
    sequential_dataframes = []
    for file_path in file_paths:
        df = pd.read_parquet(file_path)
        sequential_dataframes.append(df)
    sequential_time = time.time() - start_time
    
    logger.info(f"📊 Sequential reading: {len(sequential_dataframes)} files in {sequential_time:.4f}s")
    
    # Calculate speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    logger.info(f"🚀 Speedup: {speedup:.2f}x")
    
    # Cleanup
    for file_path in file_paths:
        file_path.unlink()
    test_dir.rmdir()
    
    return {
        'parallel_time': parallel_time,
        'sequential_time': sequential_time,
        'speedup': speedup,
        'files_processed': len(dataframes)
    }

async def test_memory_efficient_concat():
    """Test memory-efficient concatenation."""
    logger.info("🧪 Testing memory-efficient concatenation...")
    
    # Create test dataframes
    num_dataframes = 20
    dataframes = []
    
    for i in range(num_dataframes):
        df = create_test_data(1000)
        dataframes.append(df)
    
    logger.info(f"Created {num_dataframes} dataframes")
    
    # Test memory-efficient concatenation
    from src.training.steps.data_collection.step02_data_reading_optimized import memory_efficient_concat
    
    start_time = time.time()
    result_df = memory_efficient_concat(dataframes, chunk_size=5)
    memory_efficient_time = time.time() - start_time
    
    logger.info(f"✅ Memory-efficient concatenation: {result_df.shape} in {memory_efficient_time:.4f}s")
    
    # Test standard concatenation for comparison
    start_time = time.time()
    standard_result = pd.concat(dataframes, ignore_index=True)
    standard_time = time.time() - start_time
    
    logger.info(f"📊 Standard concatenation: {standard_result.shape} in {standard_time:.4f}s")
    
    # Verify results are the same
    assert result_df.shape == standard_result.shape, "Results should have the same shape"
    
    return {
        'memory_efficient_time': memory_efficient_time,
        'standard_time': standard_time,
        'result_shape': result_df.shape
    }

async def test_optimized_step():
    """Test the optimized step implementation."""
    logger.info("🧪 Testing optimized step implementation...")
    
    # Create test data directory structure
    test_data_dir = Path("test_data_cache")
    test_data_dir.mkdir(exist_ok=True)
    
    unified_dir = test_data_dir / "unified" / "BINANCE" / "ETHUSDT" / "1m"
    unified_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test data file
    test_data = create_test_data(5000)
    test_file = unified_dir / "test_data.parquet"
    test_data.to_parquet(test_file, index=False)
    
    logger.info(f"Created test data file: {test_file}")
    
    # Test optimized step
    config = {
        'max_workers': 4,
        'chunk_size': 1000,
        'min_rows': 1000,
        'max_duplicate_ratio': 0.01,
        'max_gap_seconds': 0.5
    }
    
    step = OptimizedDataReadingStep(config)
    await step.initialize()
    
    start_time = time.time()
    result = await step.execute("ETHUSDT", "BINANCE", "1m", str(test_data_dir))
    execution_time = time.time() - start_time
    
    logger.info(f"✅ Optimized step execution: {execution_time:.4f}s")
    logger.info(f"   - Success: {result['success']}")
    
    if result['success']:
        logger.info(f"   - Data path: {result['data_path']}")
        logger.info(f"   - Quality score: {result['validation_results']['quality_score']}")
        logger.info(f"   - Issues: {len(result['validation_results']['issues'])}")
        logger.info(f"   - Warnings: {len(result['validation_results']['warnings'])}")
    
    # Cleanup
    test_file.unlink()
    unified_dir.rmdir()
    (unified_dir.parent).rmdir()
    (unified_dir.parent.parent).rmdir()
    (unified_dir.parent.parent.parent).rmdir()
    test_data_dir.rmdir()
    
    return {
        'execution_time': execution_time,
        'success': result['success'],
        'result': result
    }

async def test_optimized_validator():
    """Test the optimized validator implementation."""
    logger.info("🧪 Testing optimized validator implementation...")
    
    # Create test data directory structure
    test_data_dir = Path("test_data_cache")
    test_data_dir.mkdir(exist_ok=True)
    
    unified_dir = test_data_dir / "unified" / "BINANCE" / "ETHUSDT" / "1m"
    unified_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test data file
    test_data = create_test_data(5000)
    test_file = unified_dir / "test_data.parquet"
    test_data.to_parquet(test_file, index=False)
    
    logger.info(f"Created test data file: {test_file}")
    
    # Test optimized validator
    training_input = {
        'symbol': 'ETHUSDT',
        'exchange': 'BINANCE',
        'timeframe': '1m',
        'data_dir': str(test_data_dir)
    }
    
    start_time = time.time()
    result = await run_validator_optimized(training_input, {})
    validation_time = time.time() - start_time
    
    logger.info(f"✅ Optimized validator execution: {validation_time:.4f}s")
    
    validation_result = result['validation_result']
    logger.info(f"   - Validation passed: {validation_result['validation_passed']}")
    logger.info(f"   - Data shape: {validation_result.get('data_shape', 'Unknown')}")
    logger.info(f"   - Issues: {validation_result.get('total_issues', 0)}")
    logger.info(f"   - Warnings: {validation_result.get('total_warnings', 0)}")
    
    # Cleanup
    test_file.unlink()
    unified_dir.rmdir()
    (unified_dir.parent).rmdir()
    (unified_dir.parent.parent).rmdir()
    (unified_dir.parent.parent.parent).rmdir()
    test_data_dir.rmdir()
    
    return {
        'validation_time': validation_time,
        'validation_passed': validation_result['validation_passed'],
        'result': result
    }

async def main():
    """Run all optimization tests."""
    logger.info("🚀 Starting Step 2 Optimization Tests")
    logger.info("=" * 60)
    
    # Test vectorized operations
    logger.info("\n1. Testing Vectorized Operations")
    logger.info("-" * 40)
    vectorized_results = test_vectorized_operations()
    
    # Test fast-fail validation
    logger.info("\n2. Testing Fast-Fail Validation")
    logger.info("-" * 40)
    fast_fail_results = test_fast_fail_validation()
    
    # Test parallel reading
    logger.info("\n3. Testing Parallel File Reading")
    logger.info("-" * 40)
    parallel_results = await test_parallel_reading()
    
    # Test memory-efficient concatenation
    logger.info("\n4. Testing Memory-Efficient Concatenation")
    logger.info("-" * 40)
    concat_results = await test_memory_efficient_concat()
    
    # Test optimized step
    logger.info("\n5. Testing Optimized Step Implementation")
    logger.info("-" * 40)
    step_results = await test_optimized_step()
    
    # Test optimized validator
    logger.info("\n6. Testing Optimized Validator Implementation")
    logger.info("-" * 40)
    validator_results = await test_optimized_validator()
    
    # Summary
    logger.info("\n📊 OPTIMIZATION TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Vectorized Operations Total Time: {vectorized_results['timings']['total_time']:.4f}s")
    logger.info(f"Fast-Fail Validation (Valid): {fast_fail_results['valid_schema_time'] + fast_fail_results['valid_size_time']:.6f}s")
    logger.info(f"Fast-Fail Validation (Invalid): {fast_fail_results['invalid_schema_time'] + fast_fail_results['invalid_size_time']:.6f}s")
    logger.info(f"Parallel Reading Speedup: {parallel_results['speedup']:.2f}x")
    logger.info(f"Memory-Efficient Concatenation: {concat_results['memory_efficient_time']:.4f}s")
    logger.info(f"Optimized Step Execution: {step_results['execution_time']:.4f}s")
    logger.info(f"Optimized Validator Execution: {validator_results['validation_time']:.4f}s")
    
    logger.info("\n✅ All optimization tests completed successfully!")
    
    return {
        'vectorized_results': vectorized_results,
        'fast_fail_results': fast_fail_results,
        'parallel_results': parallel_results,
        'concat_results': concat_results,
        'step_results': step_results,
        'validator_results': validator_results
    }

if __name__ == "__main__":
    asyncio.run(main())