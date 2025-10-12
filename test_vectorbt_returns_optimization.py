#!/usr/bin/env python3
"""
Test script for VectorBT optimizations in returns module.

This script validates that the returns module is fully utilizing VectorBT
optimizations including VectorBTRollingOptimizer and UnifiedVectorizationManager.
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vectorbt_imports():
    """Test VectorBT imports and availability."""
    logger.info("Testing VectorBT imports...")
    
    try:
        import vectorbt as vbt
        from vectorbt.generic import rolling_mean, rolling_std, rolling_var
        logger.info("✅ VectorBT imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ VectorBT import failed: {e}")
        return False

def test_returns_module_imports():
    """Test returns module imports."""
    logger.info("Testing returns module imports...")
    
    try:
        from feature_generation.categories.returns import (
            ReturnsFeatureGenerator,
            LogReturnsGenerator,
            SimpleReturnsGenerator,
            CumulativeReturnsGenerator,
            RollingReturnsGenerator,
            ReturnsVolatilityGenerator,
            ReturnsSkewnessGenerator,
            ReturnsKurtosisGenerator,
            SharpeRatioGenerator
        )
        logger.info("✅ Returns module imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Returns module import failed: {e}")
        return False

def test_optimization_utilities():
    """Test optimization utilities."""
    logger.info("Testing optimization utilities...")
    
    try:
        from feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
        from feature_generation.utils.unified_vectorization_manager import UnifiedVectorizationManager, get_unified_vectorization_manager
        
        # Test VectorBTRollingOptimizer
        rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        logger.info("✅ VectorBTRollingOptimizer initialized")
        
        # Test UnifiedVectorizationManager
        unified_manager = get_unified_vectorization_manager(enable_gpu=False, enable_parallel=True)
        logger.info("✅ UnifiedVectorizationManager initialized")
        
        return True
    except ImportError as e:
        logger.error(f"❌ Optimization utilities import failed: {e}")
        return False

def create_test_data(size: int = 10000) -> pd.DataFrame:
    """Create test data for performance testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=size, freq='1min')
    
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(size) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(size) * 0.01) + np.abs(np.random.randn(size) * 0.5),
        'low': 100 + np.cumsum(np.random.randn(size) * 0.01) - np.abs(np.random.randn(size) * 0.5),
        'open': 100 + np.cumsum(np.random.randn(size) * 0.01),
        'volume': np.random.lognormal(10, 1, size)
    }, index=dates)
    
    return data

def test_returns_generators_performance():
    """Test performance of returns generators with VectorBT optimization."""
    logger.info("Testing returns generators performance...")
    
    try:
        from feature_generation.categories.returns import (
            ReturnsFeatureGenerator,
            LogReturnsGenerator,
            SimpleReturnsGenerator,
            CumulativeReturnsGenerator,
            ReturnsVolatilityGenerator,
            SharpeRatioGenerator
        )
        
        # Create test data
        data = create_test_data(5000)
        
        # Test generators
        generators = [
            ReturnsFeatureGenerator(),
            LogReturnsGenerator(period=1),
            SimpleReturnsGenerator(period=1),
            CumulativeReturnsGenerator(window=20),
            ReturnsVolatilityGenerator(window=20),
            SharpeRatioGenerator(window=20)
        ]
        
        results = {}
        
        for generator in generators:
            generator_name = generator.__class__.__name__
            logger.info(f"Testing {generator_name}...")
            
            start_time = time.time()
            try:
                feature_result = generator.generate_feature(data)
                end_time = time.time()
                
                results[generator_name] = {
                    'success': True,
                    'time': end_time - start_time,
                    'shape': feature_result.shape if hasattr(feature_result, 'shape') else len(feature_result),
                    'has_nan': feature_result.isna().sum() if hasattr(feature_result, 'isna') else 0
                }
                
                logger.info(f"✅ {generator_name} completed in {end_time - start_time:.4f}s")
                
            except Exception as e:
                results[generator_name] = {
                    'success': False,
                    'error': str(e)
                }
                logger.error(f"❌ {generator_name} failed: {e}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Performance testing failed: {e}")
        return {}

def test_vectorbt_rolling_optimizer():
    """Test VectorBTRollingOptimizer functionality."""
    logger.info("Testing VectorBTRollingOptimizer...")
    
    try:
        from feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
        
        optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        data = create_test_data(1000)
        
        # Test various operations
        operations = ['mean', 'std', 'var', 'min', 'max', 'sum']
        results = {}
        
        for operation in operations:
            start_time = time.time()
            result = optimizer.rolling_operation(data['close'], operation, window=20)
            end_time = time.time()
            
            results[operation] = {
                'success': True,
                'time': end_time - start_time,
                'shape': result.shape
            }
            
            logger.info(f"✅ Rolling {operation} completed in {end_time - start_time:.4f}s")
        
        # Get performance stats
        stats = optimizer.get_performance_stats()
        logger.info(f"VectorBTRollingOptimizer stats: {stats}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ VectorBTRollingOptimizer testing failed: {e}")
        return {}

def test_unified_vectorization_manager():
    """Test UnifiedVectorizationManager functionality."""
    logger.info("Testing UnifiedVectorizationManager...")
    
    try:
        from feature_generation.utils.unified_vectorization_manager import get_unified_vectorization_manager
        
        manager = get_unified_vectorization_manager(enable_gpu=False, enable_parallel=True)
        data = create_test_data(1000)
        
        # Test single operations
        operations = ['mean', 'std', 'var', 'min', 'max', 'sum']
        results = {}
        
        for operation in operations:
            start_time = time.time()
            result = manager.rolling_operation(data['close'], operation, window=20)
            end_time = time.time()
            
            results[operation] = {
                'success': True,
                'time': end_time - start_time,
                'shape': result.shape
            }
            
            logger.info(f"✅ Unified {operation} completed in {end_time - start_time:.4f}s")
        
        # Test batch operations
        batch_operations = [
            {'type': 'rolling', 'name': 'sma_20', 'params': {'column': 'close', 'operation': 'mean', 'window': 20}},
            {'type': 'rolling', 'name': 'std_20', 'params': {'column': 'close', 'operation': 'std', 'window': 20}},
            {'type': 'rolling', 'name': 'volume_sma_10', 'params': {'column': 'volume', 'operation': 'mean', 'window': 10}}
        ]
        
        start_time = time.time()
        batch_result = manager.batch_operations(data, batch_operations)
        end_time = time.time()
        
        results['batch'] = {
            'success': True,
            'time': end_time - start_time,
            'shape': batch_result.shape,
            'columns': list(batch_result.columns)
        }
        
        logger.info(f"✅ Batch operations completed in {end_time - start_time:.4f}s")
        
        # Get performance stats
        stats = manager.get_performance_stats()
        logger.info(f"UnifiedVectorizationManager stats: {stats}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ UnifiedVectorizationManager testing failed: {e}")
        return {}

def test_vectorbt_vs_pandas_performance():
    """Compare VectorBT vs pandas performance."""
    logger.info("Testing VectorBT vs pandas performance...")
    
    try:
        from feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
        
        optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        data = create_test_data(10000)
        
        # Test VectorBT performance
        start_time = time.time()
        vectorbt_result = optimizer.rolling_operation(data['close'], 'mean', window=20)
        vectorbt_time = time.time() - start_time
        
        # Test pandas performance
        start_time = time.time()
        pandas_result = data['close'].rolling(window=20).mean()
        pandas_time = time.time() - start_time
        
        # Compare results
        results_match = np.allclose(vectorbt_result.dropna(), pandas_result.dropna(), rtol=1e-10)
        
        logger.info(f"VectorBT time: {vectorbt_time:.4f}s")
        logger.info(f"Pandas time: {pandas_time:.4f}s")
        logger.info(f"Speedup: {pandas_time / vectorbt_time:.2f}x")
        logger.info(f"Results match: {results_match}")
        
        return {
            'vectorbt_time': vectorbt_time,
            'pandas_time': pandas_time,
            'speedup': pandas_time / vectorbt_time,
            'results_match': results_match
        }
        
    except Exception as e:
        logger.error(f"❌ Performance comparison failed: {e}")
        return {}

def generate_optimization_report():
    """Generate a comprehensive optimization report."""
    logger.info("Generating optimization report...")
    
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'vectorbt_available': test_vectorbt_imports(),
        'returns_module_available': test_returns_module_imports(),
        'optimization_utilities_available': test_optimization_utilities(),
        'generators_performance': test_returns_generators_performance(),
        'rolling_optimizer_performance': test_vectorbt_rolling_optimizer(),
        'unified_manager_performance': test_unified_vectorization_manager(),
        'performance_comparison': test_vectorbt_vs_pandas_performance()
    }
    
    return report

def main():
    """Main test function."""
    logger.info("Starting VectorBT optimization testing...")
    
    # Generate comprehensive report
    report = generate_optimization_report()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("VECTORBT OPTIMIZATION TEST SUMMARY")
    logger.info("="*60)
    
    logger.info(f"VectorBT Available: {'✅' if report['vectorbt_available'] else '❌'}")
    logger.info(f"Returns Module Available: {'✅' if report['returns_module_available'] else '❌'}")
    logger.info(f"Optimization Utilities Available: {'✅' if report['optimization_utilities_available'] else '❌'}")
    
    if report['generators_performance']:
        successful_generators = sum(1 for gen in report['generators_performance'].values() if gen.get('success', False))
        total_generators = len(report['generators_performance'])
        logger.info(f"Generators Working: {successful_generators}/{total_generators}")
    
    if report['performance_comparison']:
        speedup = report['performance_comparison'].get('speedup', 0)
        logger.info(f"VectorBT Speedup: {speedup:.2f}x")
    
    logger.info("="*60)
    
    # Save report
    import json
    with open('vectorbt_optimization_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info("Report saved to vectorbt_optimization_report.json")
    
    return report

if __name__ == "__main__":
    main()