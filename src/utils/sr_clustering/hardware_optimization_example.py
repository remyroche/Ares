#!/usr/bin/env python3
"""
Hardware Optimization Example for SR Parameter Optimization.

This example demonstrates the comprehensive hardware optimizations integrated
into the SR parameter optimization system, including:

1. M1 Memory Optimization
2. M1 CPU Optimization  
3. M1 GPU Acceleration
4. Parallel Processing
5. Vectorized Operations
6. Caching
7. Numba Acceleration
8. Computation Optimization

Usage:
    python hardware_optimization_example.py
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.sr_clustering.parameter_optimization_engine import (
    ParameterOptimizationEngine, 
    ParameterOptimizationConfig
)
from utils.sr_clustering.sr_backtesting_engine import (
    SRBacktestingEngine, 
    BacktestConfig
)
from utils.sr_clustering.sr_level import SRLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_market_data(n_bars: int = 10000) -> pd.DataFrame:
    """Create sample market data for testing."""
    logger.info(f"Creating {n_bars} bars of sample market data")
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_bars)  # 2% daily volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLC data
    data = []
    for i, price in enumerate(prices):
        # Generate realistic OHLC from close price
        volatility = abs(np.random.normal(0, 0.01))
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = prices[i-1] if i > 0 else price
        
        data.append({
            'timestamp': i,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': np.random.randint(1000, 10000)
        })
    
    return pd.DataFrame(data)

def create_sample_sr_levels() -> List[SRLevel]:
    """Create sample SR levels for testing."""
    logger.info("Creating sample SR levels")
    
    levels = []
    base_prices = [95.0, 100.0, 105.0, 110.0, 115.0]
    
    for i, price in enumerate(base_prices):
        level = SRLevel(
            price=price,
            level_type='support' if i % 2 == 0 else 'resistance',
            strength=0.8,
            touches=3 + i,
            first_touch_time=1000 + i * 100,
            last_touch_time=2000 + i * 100,
            detection_method='swing_points'
        )
        levels.append(level)
    
    return levels

def test_hardware_optimization_performance():
    """Test the performance impact of hardware optimizations."""
    logger.info("🚀 Testing Hardware Optimization Performance")
    
    # Create sample data
    market_data = create_sample_market_data(5000)
    sr_levels = create_sample_sr_levels()
    
    # Test configurations
    configs = [
        {
            'name': 'No Optimizations',
            'config': ParameterOptimizationConfig(
                enable_hardware_optimization=False,
                enable_parallel_processing=False,
                enable_gpu_acceleration=False,
                enable_vectorized_operations=False,
                enable_caching=False,
                enable_numba_acceleration=False,
                n_trials=50
            )
        },
        {
            'name': 'Memory + CPU Optimization',
            'config': ParameterOptimizationConfig(
                enable_hardware_optimization=True,
                enable_parallel_processing=True,
                enable_gpu_acceleration=False,
                enable_vectorized_operations=True,
                enable_caching=True,
                enable_numba_acceleration=False,
                n_trials=50
            )
        },
        {
            'name': 'Full Hardware Optimization',
            'config': ParameterOptimizationConfig(
                enable_hardware_optimization=True,
                enable_parallel_processing=True,
                enable_gpu_acceleration=True,
                enable_vectorized_operations=True,
                enable_caching=True,
                enable_numba_acceleration=True,
                n_trials=50
            )
        }
    ]
    
    results = []
    
    for config_test in configs:
        logger.info(f"Testing: {config_test['name']}")
        
        # Create backtesting engine
        backtest_config = BacktestConfig(
            enable_m1_optimizations=config_test['config'].enable_hardware_optimization,
            enable_parallel_processing=config_test['config'].enable_parallel_processing,
            enable_vectorized_operations=config_test['config'].enable_vectorized_operations,
            enable_caching=config_test['config'].enable_caching,
            enable_numba_acceleration=config_test['config'].enable_numba_acceleration
        )
        
        backtest_engine = SRBacktestingEngine(backtest_config)
        
        # Generate backtest results
        backtest_results = []
        for level in sr_levels:
            result = backtest_engine.backtest_sr_level(level, market_data)
            backtest_results.append(result)
        
        # Create optimization engine
        opt_engine = ParameterOptimizationEngine(config_test['config'])
        
        # Measure optimization time
        start_time = time.time()
        optimization_result = opt_engine.optimize_parameters(backtest_results, market_data)
        end_time = time.time()
        
        optimization_time = end_time - start_time
        
        results.append({
            'name': config_test['name'],
            'time': optimization_time,
            'best_score': optimization_result.best_score,
            'success': optimization_result.optimization_success,
            'n_trials': optimization_result.n_trials
        })
        
        logger.info(f"✅ {config_test['name']}: {optimization_time:.2f}s, Score: {optimization_result.best_score:.4f}")
    
    return results

def test_memory_optimization():
    """Test memory optimization features."""
    logger.info("🧠 Testing Memory Optimization")
    
    # Create large dataset
    large_market_data = create_sample_market_data(20000)
    sr_levels = create_sample_sr_levels()
    
    config = ParameterOptimizationConfig(
        enable_hardware_optimization=True,
        memory_limit_gb=4.0,  # Low memory limit to test optimization
        chunk_size=500,
        n_trials=100
    )
    
    backtest_config = BacktestConfig(
        enable_m1_optimizations=True,
        memory_limit_gb=4.0,
        chunk_size=500
    )
    
    backtest_engine = SRBacktestingEngine(backtest_config)
    opt_engine = ParameterOptimizationEngine(config)
    
    # Generate backtest results
    backtest_results = []
    for level in sr_levels:
        result = backtest_engine.backtest_sr_level(level, large_market_data)
        backtest_results.append(result)
    
    # Test optimization with memory constraints
    start_time = time.time()
    optimization_result = opt_engine.optimize_parameters(backtest_results, large_market_data)
    end_time = time.time()
    
    logger.info(f"✅ Memory optimization test completed in {end_time - start_time:.2f}s")
    logger.info(f"   Best score: {optimization_result.best_score:.4f}")
    logger.info(f"   Success: {optimization_result.optimization_success}")
    
    return optimization_result

def test_parallel_processing():
    """Test parallel processing capabilities."""
    logger.info("⚡ Testing Parallel Processing")
    
    market_data = create_sample_market_data(8000)
    sr_levels = create_sample_sr_levels()
    
    # Test different worker counts
    worker_configs = [
        {'workers': 1, 'name': 'Single Thread'},
        {'workers': 2, 'name': '2 Threads'},
        {'workers': 4, 'name': '4 Threads'},
        {'workers': None, 'name': 'Auto-detect'}
    ]
    
    results = []
    
    for worker_config in worker_configs:
        config = ParameterOptimizationConfig(
            enable_hardware_optimization=True,
            enable_parallel_processing=True,
            max_parallel_workers=worker_config['workers'],
            n_trials=100
        )
        
        backtest_config = BacktestConfig(enable_parallel_processing=True)
        backtest_engine = SRBacktestingEngine(backtest_config)
        opt_engine = ParameterOptimizationEngine(config)
        
        # Generate backtest results
        backtest_results = []
        for level in sr_levels:
            result = backtest_engine.backtest_sr_level(level, market_data)
            backtest_results.append(result)
        
        # Test optimization
        start_time = time.time()
        optimization_result = opt_engine.optimize_parameters(backtest_results, market_data)
        end_time = time.time()
        
        results.append({
            'name': worker_config['name'],
            'time': end_time - start_time,
            'score': optimization_result.best_score
        })
        
        logger.info(f"✅ {worker_config['name']}: {end_time - start_time:.2f}s")
    
    return results

def test_gpu_acceleration():
    """Test GPU acceleration capabilities."""
    logger.info("🎮 Testing GPU Acceleration")
    
    market_data = create_sample_market_data(10000)
    sr_levels = create_sample_sr_levels()
    
    # Test with and without GPU
    configs = [
        {
            'name': 'CPU Only',
            'config': ParameterOptimizationConfig(
                enable_hardware_optimization=True,
                enable_gpu_acceleration=False,
                n_trials=100
            )
        },
        {
            'name': 'GPU Accelerated',
            'config': ParameterOptimizationConfig(
                enable_hardware_optimization=True,
                enable_gpu_acceleration=True,
                n_trials=100
            )
        }
    ]
    
    results = []
    
    for config_test in configs:
        backtest_config = BacktestConfig(
            enable_gpu_acceleration=config_test['config'].enable_gpu_acceleration
        )
        
        backtest_engine = SRBacktestingEngine(backtest_config)
        opt_engine = ParameterOptimizationEngine(config_test['config'])
        
        # Generate backtest results
        backtest_results = []
        for level in sr_levels:
            result = backtest_engine.backtest_sr_level(level, market_data)
            backtest_results.append(result)
        
        # Test optimization
        start_time = time.time()
        optimization_result = opt_engine.optimize_parameters(backtest_results, market_data)
        end_time = time.time()
        
        results.append({
            'name': config_test['name'],
            'time': end_time - start_time,
            'score': optimization_result.best_score,
            'gpu_available': opt_engine.m1_gpu_manager.mps_available if opt_engine.m1_gpu_manager else False
        })
        
        logger.info(f"✅ {config_test['name']}: {end_time - start_time:.2f}s")
        if opt_engine.m1_gpu_manager:
            logger.info(f"   GPU Available: {opt_engine.m1_gpu_manager.mps_available}")
    
    return results

def demonstrate_optimization_features():
    """Demonstrate all optimization features."""
    logger.info("🎯 Demonstrating All Optimization Features")
    
    # Create comprehensive configuration
    config = ParameterOptimizationConfig(
        # Hardware optimization
        enable_hardware_optimization=True,
        enable_parallel_processing=True,
        enable_gpu_acceleration=True,
        memory_limit_gb=8.0,
        chunk_size=1000,
        
        # Computation optimization
        enable_vectorized_operations=True,
        enable_caching=True,
        cache_size_mb=100,
        enable_numba_acceleration=True,
        
        # Optimization settings
        optimization_method='adaptive_grid_search',
        n_trials=200,
        objective_metric='quality_score_correlation'
    )
    
    backtest_config = BacktestConfig(
        # Hardware optimization
        enable_m1_optimizations=True,
        enable_gpu_acceleration=True,
        enable_memory_optimization=True,
        memory_limit_gb=8.0,
        chunk_size=1000,
        
        # Computation optimization
        enable_parallel_processing=True,
        enable_vectorized_operations=True,
        enable_caching=True,
        cache_size_mb=100,
        enable_numba_acceleration=True
    )
    
    # Create engines
    backtest_engine = SRBacktestingEngine(backtest_config)
    opt_engine = ParameterOptimizationEngine(config)
    
    # Create test data
    market_data = create_sample_market_data(15000)
    sr_levels = create_sample_sr_levels()
    
    logger.info("📊 Generating backtest results...")
    backtest_results = []
    for level in sr_levels:
        result = backtest_engine.backtest_sr_level(level, market_data)
        backtest_results.append(result)
    
    logger.info("🔧 Running parameter optimization...")
    start_time = time.time()
    optimization_result = opt_engine.optimize_parameters(backtest_results, market_data)
    end_time = time.time()
    
    # Display results
    logger.info("📈 Optimization Results:")
    logger.info(f"   Time: {end_time - start_time:.2f}s")
    logger.info(f"   Best Score: {optimization_result.best_score:.4f}")
    logger.info(f"   Success: {optimization_result.optimization_success}")
    logger.info(f"   Method: {optimization_result.optimization_method}")
    logger.info(f"   Trials: {optimization_result.n_trials}")
    
    logger.info("🎯 Best Parameters:")
    for param, value in optimization_result.best_parameters.items():
        logger.info(f"   {param}: {value}")
    
    # Display hardware optimization status
    logger.info("🖥️ Hardware Optimization Status:")
    if opt_engine.m1_memory_optimizer:
        logger.info("   ✅ M1 Memory Optimizer: Active")
    if opt_engine.m1_cpu_optimizer:
        logger.info("   ✅ M1 CPU Optimizer: Active")
    if opt_engine.m1_gpu_manager and opt_engine.m1_gpu_manager.mps_available:
        logger.info("   ✅ M1 GPU Manager: Active with MPS")
    elif opt_engine.m1_gpu_manager:
        logger.info("   ⚠️ M1 GPU Manager: Active without MPS")
    
    return optimization_result

def main():
    """Main function to run all hardware optimization tests."""
    logger.info("🚀 Starting Hardware Optimization Tests")
    
    try:
        # Test 1: Performance comparison
        logger.info("\n" + "="*60)
        logger.info("TEST 1: Performance Comparison")
        logger.info("="*60)
        perf_results = test_hardware_optimization_performance()
        
        # Test 2: Memory optimization
        logger.info("\n" + "="*60)
        logger.info("TEST 2: Memory Optimization")
        logger.info("="*60)
        memory_result = test_memory_optimization()
        
        # Test 3: Parallel processing
        logger.info("\n" + "="*60)
        logger.info("TEST 3: Parallel Processing")
        logger.info("="*60)
        parallel_results = test_parallel_processing()
        
        # Test 4: GPU acceleration
        logger.info("\n" + "="*60)
        logger.info("TEST 4: GPU Acceleration")
        logger.info("="*60)
        gpu_results = test_gpu_acceleration()
        
        # Test 5: Full demonstration
        logger.info("\n" + "="*60)
        logger.info("TEST 5: Full Optimization Demonstration")
        logger.info("="*60)
        full_result = demonstrate_optimization_features()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        
        logger.info("Performance Comparison Results:")
        for result in perf_results:
            logger.info(f"   {result['name']}: {result['time']:.2f}s (Score: {result['best_score']:.4f})")
        
        logger.info("\nParallel Processing Results:")
        for result in parallel_results:
            logger.info(f"   {result['name']}: {result['time']:.2f}s")
        
        logger.info("\nGPU Acceleration Results:")
        for result in gpu_results:
            logger.info(f"   {result['name']}: {result['time']:.2f}s (GPU: {result['gpu_available']})")
        
        logger.info("\n✅ All hardware optimization tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()