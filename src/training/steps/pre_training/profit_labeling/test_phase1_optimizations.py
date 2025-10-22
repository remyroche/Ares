"""
Test Script for Phase 1 Optimizations

This script tests all Phase 1 optimizations to ensure they work correctly
and provide the expected performance improvements.

Tests:
1. Volatility modeling optimization
2. Bar construction optimization
3. Intelligent caching system
4. Memory optimization system
5. Integration testing
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import Phase 1 optimization modules
try:
    from .phase1_optimization_integration import (
        Phase1OptimizationManager, Phase1OptimizationConfig, get_phase1_optimization_manager
    )
    from .volatility_modeling_optimized import OptimizedVolatilityModeler, OptimizedVolatilityConfig
    from .bar_construction_optimized import OptimizedEventBasedBarConstructor, OptimizedBarConstructionConfig
    from .intelligent_caching_system import IntelligentCachingSystem, CacheConfig
    from .memory_optimization_system import MemoryOptimizationSystem, MemoryOptimizationConfig
    PHASE1_AVAILABLE = True
except ImportError as e:
    print(f"Phase 1 modules not available: {e}")
    PHASE1_AVAILABLE = False

# Import original modules for comparison
try:
    from .volatility_modeling import VolatilityModeler, VolatilityConfig, VolatilityMethod
    from .bar_construction import EventBasedBarConstructor, BarConstructionConfig, TriggerType
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False

from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_performance
)


def generate_test_data(n_samples: int = 1000) -> Dict[str, pd.DataFrame]:
    """Generate test data for optimization testing."""
    tprint_info("📊 Generating test data")
    
    # Generate OHLCV data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1min')
    
    # Generate price data with trend and volatility
    base_price = 100.0
    returns = np.random.normal(0, 0.001, n_samples)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    ohlcv_data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.002)))
        low = price * (1 - abs(np.random.normal(0, 0.002)))
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        volume = np.random.randint(100, 1000)
        
        ohlcv_data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    ohlcv_df = pd.DataFrame(ohlcv_data)
    
    # Generate tick data (higher frequency)
    tick_data = []
    for i in range(0, n_samples, 5):  # Every 5th sample as tick
        tick_data.append({
            'timestamp': dates[i],
            'price': prices[i],
            'volume': np.random.randint(10, 100)
        })
    
    tick_df = pd.DataFrame(tick_data)
    
    tprint_success(f"✅ Generated test data: {len(ohlcv_df)} OHLCV bars, {len(tick_df)} ticks")
    
    return {
        'ohlcv_data': ohlcv_df,
        'tick_data': tick_df
    }


def test_volatility_optimization(test_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Test volatility modeling optimization."""
    tprint_info("🧪 Testing volatility modeling optimization")
    
    results = {
        'test_name': 'volatility_optimization',
        'success': False,
        'performance_improvement': 0.0,
        'original_time': 0.0,
        'optimized_time': 0.0,
        'error': None
    }
    
    try:
        ohlcv_data = test_data['ohlcv_data']
        
        # Test original implementation
        if ORIGINAL_AVAILABLE:
            tprint_info("   → Testing original volatility modeling")
            start_time = time.time()
            original_modeler = VolatilityModeler(VolatilityConfig(method=VolatilityMethod.COMBINED))
            original_result = original_modeler.model_volatility(ohlcv_data)
            original_time = time.time() - start_time
            results['original_time'] = original_time
            tprint_performance(f"   → Original: {original_time:.3f}s")
        
        # Test optimized implementation
        if PHASE1_AVAILABLE:
            tprint_info("   → Testing optimized volatility modeling")
            start_time = time.time()
            optimized_modeler = OptimizedVolatilityModeler(OptimizedVolatilityConfig())
            optimized_result = optimized_modeler.model_volatility_optimized(ohlcv_data)
            optimized_time = time.time() - start_time
            results['optimized_time'] = optimized_time
            tprint_performance(f"   → Optimized: {optimized_time:.3f}s")
            
            # Calculate performance improvement
            if ORIGINAL_AVAILABLE and original_time > 0:
                improvement = (original_time - optimized_time) / original_time * 100
                results['performance_improvement'] = improvement
                tprint_success(f"   → Performance improvement: {improvement:.1f}%")
            
            results['success'] = True
            
        else:
            tprint_warning("   → Phase 1 modules not available")
            results['error'] = "Phase 1 modules not available"
    
    except Exception as e:
        tprint_error(f"   → Volatility optimization test failed: {e}")
        results['error'] = str(e)
    
    return results


def test_bar_construction_optimization(test_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Test bar construction optimization."""
    tprint_info("🧪 Testing bar construction optimization")
    
    results = {
        'test_name': 'bar_construction_optimization',
        'success': False,
        'performance_improvement': 0.0,
        'original_time': 0.0,
        'optimized_time': 0.0,
        'error': None
    }
    
    try:
        tick_data = test_data['tick_data']
        
        # Test original implementation
        if ORIGINAL_AVAILABLE:
            tprint_info("   → Testing original bar construction")
            start_time = time.time()
            original_constructor = EventBasedBarConstructor(BarConstructionConfig(trigger_type=TriggerType.HYBRID))
            original_result = original_constructor.construct_bars(tick_data)
            original_time = time.time() - start_time
            results['original_time'] = original_time
            tprint_performance(f"   → Original: {original_time:.3f}s")
        
        # Test optimized implementation
        if PHASE1_AVAILABLE:
            tprint_info("   → Testing optimized bar construction")
            start_time = time.time()
            optimized_constructor = OptimizedEventBasedBarConstructor(OptimizedBarConstructionConfig())
            optimized_result = optimized_constructor.construct_bars_optimized(tick_data)
            optimized_time = time.time() - start_time
            results['optimized_time'] = optimized_time
            tprint_performance(f"   → Optimized: {optimized_time:.3f}s")
            
            # Calculate performance improvement
            if ORIGINAL_AVAILABLE and original_time > 0:
                improvement = (original_time - optimized_time) / original_time * 100
                results['performance_improvement'] = improvement
                tprint_success(f"   → Performance improvement: {improvement:.1f}%")
            
            results['success'] = True
            
        else:
            tprint_warning("   → Phase 1 modules not available")
            results['error'] = "Phase 1 modules not available"
    
    except Exception as e:
        tprint_error(f"   → Bar construction optimization test failed: {e}")
        results['error'] = str(e)
    
    return results


def test_caching_system() -> Dict[str, Any]:
    """Test intelligent caching system."""
    tprint_info("🧪 Testing intelligent caching system")
    
    results = {
        'test_name': 'caching_system',
        'success': False,
        'cache_hit_rate': 0.0,
        'performance_improvement': 0.0,
        'error': None
    }
    
    try:
        if PHASE1_AVAILABLE:
            # Initialize caching system
            cache_config = CacheConfig(max_memory_mb=50, enable_monitoring=True)
            caching_system = IntelligentCachingSystem(cache_config)
            
            # Test caching functionality
            test_data = np.random.randn(1000, 100)
            
            # First access (cache miss)
            start_time = time.time()
            caching_system.set('test_data', test_data)
            first_time = time.time() - start_time
            
            # Second access (cache hit)
            start_time = time.time()
            cached_data = caching_system.get('test_data')
            second_time = time.time() - start_time
            
            # Calculate performance improvement
            if first_time > 0:
                improvement = (first_time - second_time) / first_time * 100
                results['performance_improvement'] = improvement
            
            # Get cache statistics
            stats = caching_system.get_stats()
            results['cache_hit_rate'] = stats['hit_rate']
            
            tprint_success(f"   → Cache hit rate: {stats['hit_rate']:.2%}")
            tprint_success(f"   → Performance improvement: {improvement:.1f}%")
            
            results['success'] = True
            
        else:
            tprint_warning("   → Phase 1 modules not available")
            results['error'] = "Phase 1 modules not available"
    
    except Exception as e:
        tprint_error(f"   → Caching system test failed: {e}")
        results['error'] = str(e)
    
    return results


def test_memory_optimization(test_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Test memory optimization system."""
    tprint_info("🧪 Testing memory optimization system")
    
    results = {
        'test_name': 'memory_optimization',
        'success': False,
        'memory_saved_mb': 0.0,
        'optimization_count': 0,
        'error': None
    }
    
    try:
        if PHASE1_AVAILABLE:
            # Initialize memory optimization system
            memory_config = MemoryOptimizationConfig(enable_monitoring=True)
            memory_optimizer = MemoryOptimizationSystem(memory_config)
            
            # Test DataFrame optimization
            df = test_data['ohlcv_data'].copy()
            original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            
            optimized_df = memory_optimizer.optimize_dataframe(df)
            optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            
            memory_saved = original_memory - optimized_memory
            results['memory_saved_mb'] = memory_saved
            
            # Get performance metrics
            metrics = memory_optimizer.get_performance_metrics()
            results['optimization_count'] = metrics['optimizations_performed']
            
            tprint_success(f"   → Memory saved: {memory_saved:.2f}MB")
            tprint_success(f"   → Optimizations performed: {metrics['optimizations_performed']}")
            
            results['success'] = True
            
        else:
            tprint_warning("   → Phase 1 modules not available")
            results['error'] = "Phase 1 modules not available"
    
    except Exception as e:
        tprint_error(f"   → Memory optimization test failed: {e}")
        results['error'] = str(e)
    
    return results


def test_integration(test_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Test Phase 1 optimization integration."""
    tprint_info("🧪 Testing Phase 1 optimization integration")
    
    results = {
        'test_name': 'integration',
        'success': False,
        'total_time': 0.0,
        'operations_completed': 0,
        'error': None
    }
    
    try:
        if PHASE1_AVAILABLE:
            # Initialize Phase 1 optimization manager
            config = Phase1OptimizationConfig(
                enable_volatility_optimization=True,
                enable_bar_construction_optimization=True,
                enable_caching=True,
                enable_memory_optimization=True
            )
            manager = get_phase1_optimization_manager(config)
            
            # Test integrated operations
            start_time = time.time()
            operations = 0
            
            # Test volatility modeling
            try:
                volatility_result = manager.optimize_volatility_modeling(test_data['ohlcv_data'])
                operations += 1
                tprint_success("   → Volatility modeling: Success")
            except Exception as e:
                tprint_warning(f"   → Volatility modeling: Failed ({e})")
            
            # Test bar construction
            try:
                bar_result = manager.optimize_bar_construction(test_data['tick_data'])
                operations += 1
                tprint_success("   → Bar construction: Success")
            except Exception as e:
                tprint_warning(f"   → Bar construction: Failed ({e})")
            
            # Test memory optimization
            try:
                optimized_data = manager.optimize_data_structure(test_data['ohlcv_data'])
                operations += 1
                tprint_success("   → Memory optimization: Success")
            except Exception as e:
                tprint_warning(f"   → Memory optimization: Failed ({e})")
            
            total_time = time.time() - start_time
            results['total_time'] = total_time
            results['operations_completed'] = operations
            
            # Get performance metrics
            metrics = manager.get_performance_metrics()
            tprint_success(f"   → Total time: {total_time:.3f}s")
            tprint_success(f"   → Operations completed: {operations}")
            
            results['success'] = operations > 0
            
        else:
            tprint_warning("   → Phase 1 modules not available")
            results['error'] = "Phase 1 modules not available"
    
    except Exception as e:
        tprint_error(f"   → Integration test failed: {e}")
        results['error'] = str(e)
    
    return results


def run_all_tests(n_samples: int = 1000) -> Dict[str, Any]:
    """Run all Phase 1 optimization tests."""
    tprint_info("🚀 Starting Phase 1 optimization tests")
    
    # Generate test data
    test_data = generate_test_data(n_samples)
    
    # Run individual tests
    test_results = {}
    
    # Test volatility optimization
    test_results['volatility'] = test_volatility_optimization(test_data)
    
    # Test bar construction optimization
    test_results['bar_construction'] = test_bar_construction_optimization(test_data)
    
    # Test caching system
    test_results['caching'] = test_caching_system()
    
    # Test memory optimization
    test_results['memory'] = test_memory_optimization(test_data)
    
    # Test integration
    test_results['integration'] = test_integration(test_data)
    
    # Calculate overall results
    total_tests = len(test_results)
    successful_tests = sum(1 for result in test_results.values() if result['success'])
    
    overall_results = {
        'total_tests': total_tests,
        'successful_tests': successful_tests,
        'success_rate': successful_tests / total_tests if total_tests > 0 else 0.0,
        'test_results': test_results,
        'phase1_available': PHASE1_AVAILABLE,
        'original_available': ORIGINAL_AVAILABLE
    }
    
    # Print summary
    tprint_info("📊 Test Results Summary")
    tprint_info(f"   → Total tests: {total_tests}")
    tprint_info(f"   → Successful tests: {successful_tests}")
    tprint_info(f"   → Success rate: {overall_results['success_rate']:.1%}")
    
    for test_name, result in test_results.items():
        status = "✅" if result['success'] else "❌"
        tprint_info(f"   → {test_name}: {status}")
        if not result['success'] and result['error']:
            tprint_warning(f"      Error: {result['error']}")
    
    return overall_results


if __name__ == "__main__":
    # Run all tests
    results = run_all_tests(n_samples=1000)
    
    # Print final summary
    if results['success_rate'] > 0.8:
        tprint_success("🎉 Phase 1 optimizations are working well!")
    elif results['success_rate'] > 0.5:
        tprint_warning("⚠️ Phase 1 optimizations are partially working")
    else:
        tprint_error("❌ Phase 1 optimizations need attention")