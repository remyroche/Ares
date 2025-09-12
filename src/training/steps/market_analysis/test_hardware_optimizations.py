"""
Test Hardware Optimizations for Triple Barrier Labeling

This module tests the hardware optimization features of the triple barrier labeling system,
including M1/M2/M3 optimizations, memory management, and performance improvements.
"""

import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

# Import the hardware-optimized triple barrier labeling
try:
    from .hardware_optimized_triple_barrier_labeling import (
        HardwareOptimizedTripleBarrierLabeling,
        HardwareOptimizedConfig,
        apply_hardware_optimized_triple_barrier_labeling,
        apply_hardware_optimized_regime_aware_triple_barrier_labeling,
        get_hardware_optimization_info
    )
    HARDWARE_OPTIMIZED_AVAILABLE = True
except ImportError as e:
    HARDWARE_OPTIMIZED_AVAILABLE = False
    logging.warning(f"Hardware-optimized triple barrier labeling not available: {e}")

# Import the standard triple barrier labeling for comparison
from .triple_barrier_labeling import (
    MarketAnalysisTripleBarrierLabeling,
    TripleBarrierConfig,
    apply_triple_barrier_labeling
)

# Import hardware optimization utilities
try:
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, is_m1_available, is_mps_available
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_optimizations import get_m1_memory_optimizer as get_advanced_m1_optimizer
    HARDWARE_UTILS_AVAILABLE = True
except ImportError:
    HARDWARE_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)

def create_test_data(size: int = 10000, num_regimes: int = 3) -> pd.DataFrame:
    """Create test data for hardware optimization testing."""
    dates = pd.date_range('2024-01-01', periods=size, freq='1min')
    
    # Create realistic market data
    np.random.seed(42)  # For reproducible results
    
    # Generate price data with some trend and volatility
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, size)  # Small positive drift with 2% volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Generate OHLC data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, size)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, size))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, size))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, size),
        'hmm_regime': np.random.choice(range(num_regimes), size, p=[0.4, 0.4, 0.2])
    }, index=dates)
    
    # Ensure OHLC consistency
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

def test_hardware_optimization_availability():
    """Test hardware optimization availability and configuration."""
    logger.info("🔧 Testing hardware optimization availability...")
    
    results = {
        'hardware_optimized_available': HARDWARE_OPTIMIZED_AVAILABLE,
        'hardware_utils_available': HARDWARE_UTILS_AVAILABLE,
        'hardware_info': {}
    }
    
    if HARDWARE_OPTIMIZED_AVAILABLE:
        # Get hardware optimization info
        hardware_info = get_hardware_optimization_info()
        results['hardware_info'] = hardware_info
        
        logger.info(f"✅ Hardware-optimized triple barrier labeling available")
        logger.info(f"   → Numba available: {hardware_info.get('numba_available', False)}")
        logger.info(f"   → Torch available: {hardware_info.get('torch_available', False)}")
        logger.info(f"   → MPS available: {hardware_info.get('mps_available', False)}")
        logger.info(f"   → M1 available: {hardware_info.get('m1_available', False)}")
    
    if HARDWARE_UTILS_AVAILABLE:
        try:
            # Test M1 CPU optimizer
            cpu_optimizer = get_m1_cpu_optimizer()
            cpu_info = cpu_optimizer.get_cpu_info()
            results['cpu_info'] = cpu_info
            
            # Test M1 GPU manager
            gpu_manager = get_m1_gpu_manager()
            gpu_info = gpu_manager.get_gpu_info()
            results['gpu_info'] = gpu_info
            
            # Test M1 memory optimizer
            memory_optimizer = get_m1_memory_optimizer()
            memory_stats = memory_optimizer.get_memory_stats()
            results['memory_stats'] = memory_stats
            
            logger.info(f"✅ Hardware utilities available")
            logger.info(f"   → CPU info: {cpu_info}")
            logger.info(f"   → GPU info: {gpu_info}")
            logger.info(f"   → Memory stats: {memory_stats}")
            
        except Exception as e:
            logger.warning(f"⚠️ Hardware utilities test failed: {e}")
            results['hardware_utils_error'] = str(e)
    
    return results

def benchmark_hardware_optimizations(data: pd.DataFrame) -> Dict[str, Any]:
    """Benchmark hardware optimizations against standard implementation."""
    logger.info("⚡ Benchmarking hardware optimizations...")
    
    results = {
        'standard_performance': {},
        'hardware_optimized_performance': {},
        'performance_comparison': {}
    }
    
    # Test standard implementation
    logger.info("📊 Testing standard implementation...")
    start_time = time.time()
    
    standard_config = TripleBarrierConfig(
        profit_take_multiplier=0.004,
        stop_loss_multiplier=0.003,
        time_barrier_minutes=30,
        max_lookahead=100,
        transaction_cost=0.0008,
        binary_classification=True,
        regime_aware=True
    )
    
    standard_labeler = MarketAnalysisTripleBarrierLabeling(standard_config)
    standard_labeled = standard_labeler.apply_triple_barrier_labeling(data)
    
    standard_time = time.time() - start_time
    results['standard_performance'] = {
        'execution_time': standard_time,
        'labeled_samples': len(standard_labeled),
        'label_distribution': standard_labeled['label'].value_counts().to_dict() if len(standard_labeled) > 0 else {}
    }
    
    logger.info(f"✅ Standard implementation completed in {standard_time:.3f}s")
    
    # Test hardware-optimized implementation
    if HARDWARE_OPTIMIZED_AVAILABLE:
        logger.info("🚀 Testing hardware-optimized implementation...")
        start_time = time.time()
        
        hardware_config = HardwareOptimizedConfig(
            profit_take_multiplier=0.004,
            stop_loss_multiplier=0.003,
            time_barrier_minutes=30,
            max_lookahead=100,
            transaction_cost=0.0008,
            binary_classification=True,
            enable_regime_aware=True,
            enable_hardware_optimization=True,
            enable_numba_acceleration=True,
            enable_gpu_acceleration=True,
            memory_limit_gb=8.0,
            enable_memory_monitoring=True,
            enable_parallel_processing=True
        )
        
        hardware_labeler = HardwareOptimizedTripleBarrierLabeling(hardware_config)
        hardware_labeled = hardware_labeler.apply_triple_barrier_labeling(data)
        hardware_labeler.cleanup()  # Cleanup resources
        
        hardware_time = time.time() - start_time
        results['hardware_optimized_performance'] = {
            'execution_time': hardware_time,
            'labeled_samples': len(hardware_labeled),
            'label_distribution': hardware_labeled['label'].value_counts().to_dict() if len(hardware_labeled) > 0 else {}
        }
        
        logger.info(f"✅ Hardware-optimized implementation completed in {hardware_time:.3f}s")
        
        # Calculate performance comparison
        speedup = standard_time / hardware_time if hardware_time > 0 else 0
        results['performance_comparison'] = {
            'speedup': speedup,
            'time_saved': standard_time - hardware_time,
            'efficiency_improvement': (speedup - 1) * 100 if speedup > 0 else 0
        }
        
        logger.info(f"📈 Performance comparison:")
        logger.info(f"   → Speedup: {speedup:.2f}x")
        logger.info(f"   → Time saved: {standard_time - hardware_time:.3f}s")
        logger.info(f"   → Efficiency improvement: {(speedup - 1) * 100:.1f}%")
    
    return results

def test_memory_optimization(data: pd.DataFrame) -> Dict[str, Any]:
    """Test memory optimization features."""
    logger.info("🧠 Testing memory optimization...")
    
    results = {
        'memory_optimization_available': False,
        'memory_stats': {},
        'optimization_results': {}
    }
    
    if HARDWARE_UTILS_AVAILABLE:
        try:
            # Get memory optimizer
            memory_optimizer = get_m1_memory_optimizer()
            
            # Get initial memory stats
            initial_stats = memory_optimizer.get_memory_stats()
            results['memory_stats']['initial'] = initial_stats
            
            # Test DataFrame memory optimization
            logger.info("📊 Testing DataFrame memory optimization...")
            optimized_data = memory_optimizer.optimize_dataframe_memory(data.copy())
            
            # Get memory stats after optimization
            after_stats = memory_optimizer.get_memory_stats()
            results['memory_stats']['after_optimization'] = after_stats
            
            # Test advanced memory optimization
            logger.info("🔧 Testing advanced memory optimization...")
            advanced_optimizer = get_advanced_m1_optimizer(
                memory_limit_gb=8.0,
                enable_gc_tuning=True,
                enable_memory_leak_detection=True,
                enable_swap_management=True
            )
            
            optimization_results = advanced_optimizer.optimize_memory()
            results['optimization_results'] = optimization_results
            
            results['memory_optimization_available'] = True
            
            logger.info(f"✅ Memory optimization completed")
            logger.info(f"   → Initial memory: {initial_stats}")
            logger.info(f"   → After optimization: {after_stats}")
            logger.info(f"   → Optimization results: {optimization_results}")
            
        except Exception as e:
            logger.warning(f"⚠️ Memory optimization test failed: {e}")
            results['error'] = str(e)
    
    return results

def test_regime_aware_hardware_optimization(data: pd.DataFrame) -> Dict[str, Any]:
    """Test regime-aware hardware optimization."""
    logger.info("🎯 Testing regime-aware hardware optimization...")
    
    results = {
        'regime_aware_available': False,
        'regime_performance': {},
        'regime_comparison': {}
    }
    
    if HARDWARE_OPTIMIZED_AVAILABLE:
        try:
            # Test regime-aware hardware optimization
            start_time = time.time()
            
            hardware_labeled = apply_hardware_optimized_regime_aware_triple_barrier_labeling(
                data,
                regime_column='hmm_regime',
                profit_take_multiplier=0.004,
                stop_loss_multiplier=0.003,
                time_barrier_minutes=30,
                max_lookahead=100,
                transaction_cost=0.0008,
                binary_classification=True,
                enable_hardware_optimization=True,
                enable_numba_acceleration=True,
                enable_gpu_acceleration=True,
                memory_limit_gb=8.0
            )
            
            regime_time = time.time() - start_time
            
            results['regime_aware_available'] = True
            results['regime_performance'] = {
                'execution_time': regime_time,
                'labeled_samples': len(hardware_labeled),
                'label_distribution': hardware_labeled['label'].value_counts().to_dict() if len(hardware_labeled) > 0 else {},
                'regime_distribution': hardware_labeled['hmm_regime'].value_counts().to_dict() if 'hmm_regime' in hardware_labeled.columns else {}
            }
            
            logger.info(f"✅ Regime-aware hardware optimization completed in {regime_time:.3f}s")
            
        except Exception as e:
            logger.warning(f"⚠️ Regime-aware hardware optimization test failed: {e}")
            results['error'] = str(e)
    
    return results

def run_comprehensive_hardware_test():
    """Run comprehensive hardware optimization tests."""
    logger.info("🚀 Starting comprehensive hardware optimization tests...")
    
    # Create test data
    test_data = create_test_data(size=5000, num_regimes=3)
    logger.info(f"📊 Created test data with {len(test_data)} samples")
    
    # Run all tests
    test_results = {
        'test_data_info': {
            'size': len(test_data),
            'columns': list(test_data.columns),
            'regime_distribution': test_data['hmm_regime'].value_counts().to_dict()
        },
        'hardware_availability': test_hardware_optimization_availability(),
        'performance_benchmark': benchmark_hardware_optimizations(test_data),
        'memory_optimization': test_memory_optimization(test_data),
        'regime_aware_optimization': test_regime_aware_hardware_optimization(test_data)
    }
    
    # Generate summary
    logger.info("📋 Generating test summary...")
    
    summary = {
        'total_tests': len(test_results) - 1,  # Exclude test_data_info
        'hardware_optimized_available': HARDWARE_OPTIMIZED_AVAILABLE,
        'hardware_utils_available': HARDWARE_UTILS_AVAILABLE,
        'performance_improvement': 0,
        'memory_optimization_available': False,
        'regime_aware_available': False
    }
    
    # Extract summary information
    if 'performance_benchmark' in test_results:
        perf_comp = test_results['performance_benchmark'].get('performance_comparison', {})
        summary['performance_improvement'] = perf_comp.get('efficiency_improvement', 0)
    
    if 'memory_optimization' in test_results:
        summary['memory_optimization_available'] = test_results['memory_optimization'].get('memory_optimization_available', False)
    
    if 'regime_aware_optimization' in test_results:
        summary['regime_aware_available'] = test_results['regime_aware_optimization'].get('regime_aware_available', False)
    
    test_results['summary'] = summary
    
    # Log summary
    logger.info("📊 Test Summary:")
    logger.info(f"   → Hardware-optimized available: {summary['hardware_optimized_available']}")
    logger.info(f"   → Hardware utils available: {summary['hardware_utils_available']}")
    logger.info(f"   → Performance improvement: {summary['performance_improvement']:.1f}%")
    logger.info(f"   → Memory optimization available: {summary['memory_optimization_available']}")
    logger.info(f"   → Regime-aware available: {summary['regime_aware_available']}")
    
    return test_results

def main():
    """Main function to run hardware optimization tests."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🧪 Starting Hardware Optimization Tests for Triple Barrier Labeling")
    logger.info("=" * 70)
    
    try:
        # Run comprehensive tests
        results = run_comprehensive_hardware_test()
        
        # Save results
        results_file = Path(__file__).parent / 'hardware_optimization_test_results.json'
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Test results saved to: {results_file}")
        
        # Print final summary
        summary = results['summary']
        logger.info("\n🎉 Hardware Optimization Tests Completed!")
        logger.info("=" * 50)
        logger.info(f"✅ Hardware-optimized triple barrier labeling: {'Available' if summary['hardware_optimized_available'] else 'Not Available'}")
        logger.info(f"✅ Hardware utilities: {'Available' if summary['hardware_utils_available'] else 'Not Available'}")
        logger.info(f"✅ Memory optimization: {'Available' if summary['memory_optimization_available'] else 'Not Available'}")
        logger.info(f"✅ Regime-aware optimization: {'Available' if summary['regime_aware_available'] else 'Not Available'}")
        
        if summary['performance_improvement'] > 0:
            logger.info(f"🚀 Performance improvement: {summary['performance_improvement']:.1f}%")
        else:
            logger.info("⚠️ No performance improvement measured")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Hardware optimization tests failed: {e}")
        raise

if __name__ == '__main__':
    main()