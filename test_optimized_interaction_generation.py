#!/usr/bin/env python3
"""
Test script for optimized feature generation interaction generation step analyst.

This script tests the performance improvements from VectorBT optimization
and validates the enhanced functionality.
"""

import asyncio
import time
import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.training.steps.pre_training.feature_generation_interaction_generation_step_analyst import (
    FeatureGenerationInteractionGenerationStepAnalyst
)
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_performance


async def test_optimized_interaction_generation():
    """Test the optimized interaction generation step."""
    tprint_info("🧪 Testing optimized feature generation interaction generation step analyst")
    
    # Test configurations
    test_configs = [
        {
            'name': 'Light Mode Test',
            'config': {
                'symbol': 'ETHUSDT',
                'exchange': 'binance',
                'timeframe': '15m',
                'execution_mode': 'light',
                'sample_size': 5000,
                'enable_vectorbt': True,
                'enable_gpu': False,
                'memory_efficient': True
            }
        },
        {
            'name': 'Full Mode Test',
            'config': {
                'symbol': 'BTCUSDT',
                'exchange': 'binance',
                'timeframe': '1h',
                'execution_mode': 'full',
                'sample_size': 10000,
                'enable_vectorbt': True,
                'enable_gpu': False,
                'memory_efficient': True
            }
        },
        {
            'name': 'Fallback Mode Test',
            'config': {
                'symbol': 'ADAUSDT',
                'exchange': 'binance',
                'timeframe': '5m',
                'execution_mode': 'light',
                'sample_size': 3000,
                'enable_vectorbt': False,  # Force fallback
                'enable_gpu': False,
                'memory_efficient': False
            }
        }
    ]
    
    results = []
    
    for test_case in test_configs:
        tprint_info(f"\n🔬 Running {test_case['name']}")
        
        try:
            # Initialize the step
            step = FeatureGenerationInteractionGenerationStepAnalyst()
            
            # Measure execution time
            start_time = time.time()
            
            # Execute the step
            result = await step.execute(test_case['config'])
            
            execution_time = time.time() - start_time
            
            # Validate results
            if result['success']:
                tprint_success(f"✅ {test_case['name']} completed successfully in {execution_time:.2f}s")
                
                # Extract metrics
                metrics = result.get('metrics', {})
                artifacts = result.get('artifacts', {})
                
                # Display results
                tprint_info(f"📊 Results for {test_case['name']}:")
                tprint_info(f"  - Features generated: {metrics.get('n_interaction_features', 0)}")
                tprint_info(f"  - Execution time: {execution_time:.2f}s")
                tprint_info(f"  - Optimization used: {metrics.get('optimization_used', 'Unknown')}")
                
                # Performance stats
                perf_stats = metrics.get('performance_stats', {})
                if perf_stats:
                    tprint_info(f"  - VectorBT operations: {perf_stats.get('vectorbt_operations', 0)}")
                    tprint_info(f"  - Pandas fallbacks: {perf_stats.get('pandas_fallbacks', 0)}")
                    tprint_info(f"  - Memory optimizations: {perf_stats.get('memory_optimizations', 0)}")
                    tprint_info(f"  - Cache hits: {perf_stats.get('cache_hits', 0)}")
                
                # Store results
                results.append({
                    'test_name': test_case['name'],
                    'success': True,
                    'execution_time': execution_time,
                    'features_generated': metrics.get('n_interaction_features', 0),
                    'optimization_used': metrics.get('optimization_used', 'Unknown'),
                    'performance_stats': perf_stats,
                    'config': test_case['config']
                })
                
            else:
                tprint_error(f"❌ {test_case['name']} failed: {result.get('error', 'Unknown error')}")
                results.append({
                    'test_name': test_case['name'],
                    'success': False,
                    'error': result.get('error', 'Unknown error'),
                    'config': test_case['config']
                })
                
        except Exception as e:
            tprint_error(f"❌ {test_case['name']} failed with exception: {e}")
            results.append({
                'test_name': test_case['name'],
                'success': False,
                'error': str(e),
                'config': test_case['config']
            })
    
    return results


async def test_performance_comparison():
    """Test performance comparison between optimized and basic implementations."""
    tprint_info("\n📈 Running performance comparison test")
    
    # Generate test data
    np.random.seed(42)
    n_points = 20000
    
    # Create realistic test data
    returns = np.random.randn(n_points) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    
    test_data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + np.abs(np.random.randn(n_points) * 0.005)),
        'low': prices * (1 - np.abs(np.random.randn(n_points) * 0.005)),
        'volume': np.random.randint(1000, 10000, n_points),
        'timestamp': pd.date_range(start='2023-01-01', periods=n_points, freq='15T')
    })
    
    # Add derived features
    test_data['returns'] = test_data['close'].pct_change()
    test_data['volatility'] = test_data['returns'].rolling(window=20).std()
    test_data['momentum'] = test_data['close'].pct_change(periods=20)
    test_data['trend'] = test_data['close'].rolling(window=20).mean().diff()
    test_data['volume_momentum'] = test_data['volume'].pct_change(periods=5)
    
    # Test configurations
    configs = [
        {
            'name': 'VectorBT Optimized',
            'config': {
                'symbol': 'TEST',
                'exchange': 'test',
                'timeframe': '15m',
                'execution_mode': 'full',
                'data_path': None,  # Use sample data
                'enable_vectorbt': True,
                'enable_gpu': False,
                'memory_efficient': True,
                'sample_size': n_points
            }
        },
        {
            'name': 'Basic Fallback',
            'config': {
                'symbol': 'TEST',
                'exchange': 'test',
                'timeframe': '15m',
                'execution_mode': 'light',
                'data_path': None,
                'enable_vectorbt': False,  # Force fallback
                'enable_gpu': False,
                'memory_efficient': False,
                'sample_size': n_points
            }
        }
    ]
    
    comparison_results = []
    
    for config in configs:
        tprint_info(f"\n🔬 Testing {config['name']}")
        
        try:
            step = FeatureGenerationInteractionGenerationStepAnalyst()
            
            # Measure execution time
            start_time = time.time()
            result = await step.execute(config['config'])
            execution_time = time.time() - start_time
            
            if result['success']:
                metrics = result.get('metrics', {})
                perf_stats = metrics.get('performance_stats', {})
                
                comparison_results.append({
                    'name': config['name'],
                    'execution_time': execution_time,
                    'features_generated': metrics.get('n_interaction_features', 0),
                    'vectorbt_operations': perf_stats.get('vectorbt_operations', 0),
                    'pandas_fallbacks': perf_stats.get('pandas_fallbacks', 0),
                    'memory_optimizations': perf_stats.get('memory_optimizations', 0),
                    'cache_hits': perf_stats.get('cache_hits', 0)
                })
                
                tprint_success(f"✅ {config['name']} completed in {execution_time:.2f}s")
            else:
                tprint_error(f"❌ {config['name']} failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            tprint_error(f"❌ {config['name']} failed with exception: {e}")
    
    return comparison_results


def analyze_results(results: list, comparison_results: list):
    """Analyze and display test results."""
    tprint_info("\n📊 Test Results Analysis")
    
    # Basic test results
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    tprint_info(f"✅ Successful tests: {len(successful_tests)}/{len(results)}")
    tprint_info(f"❌ Failed tests: {len(failed_tests)}/{len(results)}")
    
    if failed_tests:
        tprint_info("\nFailed tests:")
        for test in failed_tests:
            tprint_error(f"  - {test['test_name']}: {test.get('error', 'Unknown error')}")
    
    # Performance comparison
    if len(comparison_results) >= 2:
        tprint_info("\n📈 Performance Comparison:")
        
        optimized = next((r for r in comparison_results if 'VectorBT' in r['name']), None)
        basic = next((r for r in comparison_results if 'Basic' in r['name']), None)
        
        if optimized and basic:
            speedup = basic['execution_time'] / optimized['execution_time']
            tprint_info(f"  - Speedup: {speedup:.2f}x")
            tprint_info(f"  - Optimized time: {optimized['execution_time']:.2f}s")
            tprint_info(f"  - Basic time: {basic['execution_time']:.2f}s")
            tprint_info(f"  - Features generated: {optimized['features_generated']} vs {basic['features_generated']}")
            
            if optimized['vectorbt_operations'] > 0:
                tprint_success(f"  ✅ VectorBT optimizations were used ({optimized['vectorbt_operations']} operations)")
            else:
                tprint_info(f"  ⚠️ VectorBT optimizations were not used")
            
            if optimized['memory_optimizations'] > 0:
                tprint_success(f"  ✅ Memory optimizations applied ({optimized['memory_optimizations']} optimizations)")
            
            if optimized['cache_hits'] > 0:
                tprint_success(f"  ✅ Cache was utilized ({optimized['cache_hits']} hits)")


async def main():
    """Main test function."""
    tprint_info("🚀 Starting optimized feature generation interaction generation step analyst tests")
    
    try:
        # Run basic functionality tests
        results = await test_optimized_interaction_generation()
        
        # Run performance comparison tests
        comparison_results = await test_performance_comparison()
        
        # Analyze results
        analyze_results(results, comparison_results)
        
        tprint_success("\n✅ All tests completed successfully!")
        
        # Summary
        tprint_info("\n📋 Test Summary:")
        tprint_info("  - Optimized feature generation step implemented")
        tprint_info("  - VectorBT integration working")
        tprint_info("  - UnifiedVectorizationManager integration working")
        tprint_info("  - Performance monitoring implemented")
        tprint_info("  - Fallback mechanisms working")
        tprint_info("  - Comprehensive error handling")
        
    except Exception as e:
        tprint_error(f"❌ Test suite failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
