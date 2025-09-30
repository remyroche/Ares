"""
Enhanced TAS Regime Detection Example

This example demonstrates the enhanced TAS regime detection capabilities with:
- Performance optimizations (memory, GPU, parallel processing)
- Advanced validation (cross-validation, out-of-sample testing, regime persistence)
- Intelligent caching
- Comprehensive performance monitoring
"""

import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.utils.ml_common.optimization.tas.enhanced_regime_detection import get_enhanced_tas_regime_detection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 10000, n_features: int = 20) -> pd.DataFrame:
    """Create sample financial data for regime detection."""
    logger.info(f"📊 Creating sample data: {n_samples} samples, {n_features} features")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create time index
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='1H')
    
    # Create sample financial data with different regimes
    data = {}
    
    # Regime 1: Low volatility (first 40% of data)
    regime1_size = int(n_samples * 0.4)
    data['price'] = np.cumsum(np.random.normal(0.001, 0.01, regime1_size))
    data['volume'] = np.random.lognormal(10, 0.5, regime1_size)
    data['volatility'] = np.random.gamma(2, 0.01, regime1_size)
    
    # Regime 2: High volatility (middle 30% of data)
    regime2_size = int(n_samples * 0.3)
    data['price'] = np.concatenate([
        data['price'],
        np.cumsum(np.random.normal(0.002, 0.05, regime2_size)) + data['price'][-1]
    ])
    data['volume'] = np.concatenate([
        data['volume'],
        np.random.lognormal(11, 0.8, regime2_size)
    ])
    data['volatility'] = np.concatenate([
        data['volatility'],
        np.random.gamma(3, 0.03, regime2_size)
    ])
    
    # Regime 3: Medium volatility (last 30% of data)
    regime3_size = n_samples - regime1_size - regime2_size
    data['price'] = np.concatenate([
        data['price'],
        np.cumsum(np.random.normal(0.0015, 0.02, regime3_size)) + data['price'][-1]
    ])
    data['volume'] = np.concatenate([
        data['volume'],
        np.random.lognormal(10.5, 0.6, regime3_size)
    ])
    data['volatility'] = np.concatenate([
        data['volatility'],
        np.random.gamma(2.5, 0.02, regime3_size)
    ])
    
    # Add additional features
    for i in range(n_features - 3):
        data[f'feature_{i}'] = np.random.randn(n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates)
    
    # Add regime labels for validation
    df['true_regime'] = 0
    df.iloc[regime1_size:regime1_size + regime2_size, -1] = 1
    df.iloc[regime1_size + regime2_size:, -1] = 2
    
    logger.info(f"✅ Sample data created: {df.shape}")
    logger.info(f"📈 Regime distribution: {df['true_regime'].value_counts().to_dict()}")
    
    return df

def demonstrate_memory_optimization():
    """Demonstrate memory optimization capabilities."""
    logger.info("\n🧠 Memory Optimization Demo")
    logger.info("="*50)
    
    # Create large dataset
    large_data = create_sample_data(n_samples=50000, n_features=50)
    logger.info(f"📊 Large dataset created: {large_data.shape}")
    logger.info(f"💾 Memory usage: {large_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    # Initialize enhanced regime detection with memory optimization
    regime_detector = get_enhanced_tas_regime_detection(
        enable_gpu=True,
        enable_memory_optimization=True,
        enable_parallel=True,
        cache_dir='./tas_cache',
        max_memory_gb=4.0  # Limit memory usage
    )
    
    # Perform memory optimization
    logger.info("🔧 Optimizing memory usage...")
    memory_stats = regime_detector.optimize_memory_usage()
    logger.info(f"✅ Memory optimization: {memory_stats}")
    
    # Test regime detection with memory optimization
    start_time = time.time()
    results = regime_detector.detect_regimes_enhanced(
        data=large_data,
        timeframes=['1h', '4h'],
        methods=['unsupervised', 'clustering'],
        use_cache=True,
        parallel=True
    )
    execution_time = time.time() - start_time
    
    logger.info(f"⏱️ Execution time: {execution_time:.3f}s")
    logger.info(f"📊 Results: {len(results)} regime analyses")
    
    return results

def demonstrate_parallel_processing():
    """Demonstrate parallel processing capabilities."""
    logger.info("\n🔄 Parallel Processing Demo")
    logger.info("="*50)
    
    # Create medium dataset
    medium_data = create_sample_data(n_samples=20000, n_features=30)
    
    # Initialize with parallel processing
    regime_detector = get_enhanced_tas_regime_detection(
        enable_gpu=True,
        enable_memory_optimization=True,
        enable_parallel=True,
        cache_dir='./tas_cache'
    )
    
    # Test sequential vs parallel processing
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    methods = ['unsupervised', 'clustering', 'qualification']
    
    # Sequential processing
    logger.info("🐌 Testing sequential processing...")
    start_time = time.time()
    sequential_results = regime_detector.detect_regimes_enhanced(
        data=medium_data,
        timeframes=timeframes,
        methods=methods,
        use_cache=False,
        parallel=False
    )
    sequential_time = time.time() - start_time
    
    # Parallel processing
    logger.info("🚀 Testing parallel processing...")
    start_time = time.time()
    parallel_results = regime_detector.detect_regimes_enhanced(
        data=medium_data,
        timeframes=timeframes,
        methods=methods,
        use_cache=False,
        parallel=True
    )
    parallel_time = time.time() - start_time
    
    # Compare results
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    logger.info(f"⏱️ Sequential time: {sequential_time:.3f}s")
    logger.info(f"⏱️ Parallel time: {parallel_time:.3f}s")
    logger.info(f"🚀 Speedup: {speedup:.2f}x")
    
    return parallel_results

def demonstrate_caching():
    """Demonstrate intelligent caching capabilities."""
    logger.info("\n💾 Caching Demo")
    logger.info("="*50)
    
    # Create dataset
    data = create_sample_data(n_samples=10000, n_features=20)
    
    # Initialize with caching
    regime_detector = get_enhanced_tas_regime_detection(
        enable_gpu=True,
        enable_memory_optimization=True,
        enable_parallel=True,
        cache_dir='./tas_cache'
    )
    
    # First run (cache miss)
    logger.info("🔍 First run (cache miss)...")
    start_time = time.time()
    results1 = regime_detector.detect_regimes_enhanced(
        data=data,
        timeframes=['1h', '4h'],
        methods=['unsupervised', 'clustering'],
        use_cache=True,
        parallel=True
    )
    first_run_time = time.time() - start_time
    
    # Second run (cache hit)
    logger.info("🔍 Second run (cache hit)...")
    start_time = time.time()
    results2 = regime_detector.detect_regimes_enhanced(
        data=data,
        timeframes=['1h', '4h'],
        methods=['unsupervised', 'clustering'],
        use_cache=True,
        parallel=True
    )
    second_run_time = time.time() - start_time
    
    # Compare results
    cache_speedup = first_run_time / second_run_time if second_run_time > 0 else 0
    logger.info(f"⏱️ First run time: {first_run_time:.3f}s")
    logger.info(f"⏱️ Second run time: {second_run_time:.3f}s")
    logger.info(f"🚀 Cache speedup: {cache_speedup:.2f}x")
    
    # Check cache statistics
    stats = regime_detector.get_performance_stats()
    logger.info(f"📊 Cache hits: {stats['cache_hits']}")
    logger.info(f"📊 Cache misses: {stats['cache_misses']}")
    
    return results2

def demonstrate_advanced_validation():
    """Demonstrate advanced validation capabilities."""
    logger.info("\n🔬 Advanced Validation Demo")
    logger.info("="*50)
    
    # Create dataset with known regimes
    data = create_sample_data(n_samples=15000, n_features=25)
    
    # Initialize enhanced regime detection
    regime_detector = get_enhanced_tas_regime_detection(
        enable_gpu=True,
        enable_memory_optimization=True,
        enable_parallel=True,
        cache_dir='./tas_cache'
    )
    
    # Perform regime detection with advanced validation
    logger.info("🔍 Performing regime detection with advanced validation...")
    results = regime_detector.detect_regimes_enhanced(
        data=data,
        timeframes=['1h', '4h', '1d'],
        methods=['unsupervised', 'clustering'],
        use_cache=True,
        parallel=True
    )
    
    # Analyze validation results
    logger.info("📊 Analyzing validation results...")
    
    # Cross-validation results
    if 'cross_validation' in results:
        cv_results = results['cross_validation']
        logger.info(f"✅ Cross-validation completed for {len(cv_results)} analyses")
        
        for key, cv_result in cv_results.items():
            if 'stability' in cv_result:
                logger.info(f"  {key}: Stability = {cv_result['stability']:.3f}")
    
    # Out-of-sample results
    if 'out_of_sample' in results:
        oos_results = results['out_of_sample']
        logger.info(f"✅ Out-of-sample validation completed for {len(oos_results)} analyses")
        
        for key, oos_result in oos_results.items():
            if 'oos_score' in oos_result:
                logger.info(f"  {key}: OOS Score = {oos_result['oos_score']:.3f}")
    
    # Regime persistence results
    if 'persistence' in results:
        persistence_results = results['persistence']
        logger.info(f"✅ Regime persistence analysis completed for {len(persistence_results)} analyses")
        
        for key, persistence_result in persistence_results.items():
            if 'average_stability' in persistence_result:
                logger.info(f"  {key}: Average Stability = {persistence_result['average_stability']:.3f}")
    
    # Performance metrics
    if 'performance_metrics' in results:
        perf_metrics = results['performance_metrics']
        logger.info(f"📈 Performance Metrics:")
        logger.info(f"  Success Rate: {perf_metrics.get('success_rate', 0):.3f}")
        logger.info(f"  Performance Score: {perf_metrics.get('performance_score', 0):.3f}")
    
    return results

def demonstrate_comprehensive_analysis():
    """Demonstrate comprehensive regime analysis."""
    logger.info("\n🎯 Comprehensive Regime Analysis Demo")
    logger.info("="*50)
    
    # Create comprehensive dataset
    data = create_sample_data(n_samples=25000, n_features=40)
    
    # Initialize enhanced regime detection
    regime_detector = get_enhanced_tas_regime_detection(
        enable_gpu=True,
        enable_memory_optimization=True,
        enable_parallel=True,
        cache_dir='./tas_cache',
        max_memory_gb=8.0
    )
    
    # Comprehensive regime detection
    logger.info("🔍 Performing comprehensive regime analysis...")
    start_time = time.time()
    
    results = regime_detector.detect_regimes_enhanced(
        data=data,
        timeframes=['1m', '5m', '15m', '1h', '4h', '1d'],
        methods=['unsupervised', 'clustering', 'qualification'],
        use_cache=True,
        parallel=True
    )
    
    execution_time = time.time() - start_time
    
    # Analyze results
    logger.info(f"⏱️ Total execution time: {execution_time:.3f}s")
    logger.info(f"📊 Total analyses: {len(results)}")
    
    # Performance statistics
    stats = regime_detector.get_performance_stats()
    logger.info(f"📈 Performance Statistics:")
    logger.info(f"  Total detections: {stats['total_detections']}")
    logger.info(f"  Cache hits: {stats['cache_hits']}")
    logger.info(f"  Cache misses: {stats['cache_misses']}")
    logger.info(f"  Parallel detections: {stats['parallel_detections']}")
    logger.info(f"  Memory optimized detections: {stats['memory_optimized_detections']}")
    logger.info(f"  Average detection time: {stats['average_detection_time']:.3f}s")
    
    # Hardware information
    if 'gpu_info' in stats:
        gpu_info = stats['gpu_info']
        logger.info(f"🖥️ GPU Info: {gpu_info}")
    
    if 'memory_info' in stats:
        memory_info = stats['memory_info']
        logger.info(f"🧠 Memory Info: {memory_info}")
    
    if 'cpu_info' in stats:
        cpu_info = stats['cpu_info']
        logger.info(f"🖥️ CPU Info: {cpu_info}")
    
    return results

def main():
    """Main demonstration function."""
    logger.info("🚀 Enhanced TAS Regime Detection Demonstration")
    logger.info("="*60)
    
    try:
        # Create cache directory
        cache_dir = Path('./tas_cache')
        cache_dir.mkdir(exist_ok=True)
        
        # Run demonstrations
        logger.info("\n1. Memory Optimization Demo")
        memory_results = demonstrate_memory_optimization()
        
        logger.info("\n2. Parallel Processing Demo")
        parallel_results = demonstrate_parallel_processing()
        
        logger.info("\n3. Caching Demo")
        caching_results = demonstrate_caching()
        
        logger.info("\n4. Advanced Validation Demo")
        validation_results = demonstrate_advanced_validation()
        
        logger.info("\n5. Comprehensive Analysis Demo")
        comprehensive_results = demonstrate_comprehensive_analysis()
        
        # Summary
        logger.info("\n🎉 Enhanced TAS Regime Detection Demonstration Complete!")
        logger.info("="*60)
        logger.info("✅ All demonstrations completed successfully")
        logger.info("📊 Key features demonstrated:")
        logger.info("  - Memory optimization for large datasets")
        logger.info("  - Parallel processing across timeframes")
        logger.info("  - Intelligent caching for performance")
        logger.info("  - Cross-validation for regime stability")
        logger.info("  - Out-of-sample testing for validation")
        logger.info("  - Regime persistence analysis")
        logger.info("  - Comprehensive performance monitoring")
        
        # Cleanup
        logger.info("\n🧹 Cleaning up cache...")
        regime_detector = get_enhanced_tas_regime_detection()
        regime_detector.clear_cache()
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise

if __name__ == "__main__":
    main()