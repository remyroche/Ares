"""
Regime Statistical VectorBT Optimization Comparison

This script compares the performance of the original regime_statistical.py
implementation with the optimized version using VectorBTRollingOptimizer
and UnifiedVectorizationManager.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any, List
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import both implementations
try:
    from .regime_statistical import RegimeStatisticalFeatureGenerator
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False
    RegimeStatisticalFeatureGenerator = None

try:
    from .regime_statistical_optimized import OptimizedRegimeStatisticalFeatureGenerator
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False
    OptimizedRegimeStatisticalFeatureGenerator = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeStatisticalComparison:
    """Compare original and optimized regime statistical implementations."""
    
    def __init__(self):
        self.results = {}
        self.performance_data = {}
    
    def create_test_data(self, size: int = 1000, seed: int = 42) -> pd.DataFrame:
        """Create test data for comparison."""
        np.random.seed(seed)
        
        # Generate realistic price data
        returns = np.random.normal(0, 0.02, size)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate volume data
        volume = np.random.lognormal(10, 1, size)
        
        data = pd.DataFrame({
            'close': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, size))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, size))),
            'open': np.roll(prices, 1),
            'volume': volume
        })
        
        # Ensure high >= low
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])
        
        return data
    
    def benchmark_original(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Benchmark the original implementation."""
        if not ORIGINAL_AVAILABLE:
            return {'error': 'Original implementation not available'}
        
        logger.info("Benchmarking original implementation...")
        
        try:
            generator = RegimeStatisticalFeatureGenerator()
            
            start_time = time.time()
            features = generator.generate_features(data)
            end_time = time.time()
            
            # Get performance stats if available
            stats = {}
            if hasattr(generator, 'get_performance_stats'):
                stats = generator.get_performance_stats()
            
            return {
                'success': True,
                'execution_time': end_time - start_time,
                'feature_count': len(features),
                'feature_names': list(features.keys()),
                'performance_stats': stats,
                'memory_usage': data.memory_usage(deep=True).sum() / 1024**2  # MB
            }
            
        except Exception as e:
            logger.error(f"Original implementation failed: {e}")
            return {'error': str(e)}
    
    def benchmark_optimized(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Benchmark the optimized implementation."""
        if not OPTIMIZED_AVAILABLE:
            return {'error': 'Optimized implementation not available'}
        
        logger.info("Benchmarking optimized implementation...")
        
        try:
            generator = OptimizedRegimeStatisticalFeatureGenerator()
            
            start_time = time.time()
            features = generator.generate_features_optimized(data)
            end_time = time.time()
            
            # Get performance stats
            stats = generator.get_performance_stats()
            
            return {
                'success': True,
                'execution_time': end_time - start_time,
                'feature_count': len(features),
                'feature_names': list(features.keys()),
                'performance_stats': stats,
                'memory_usage': data.memory_usage(deep=True).sum() / 1024**2  # MB
            }
            
        except Exception as e:
            logger.error(f"Optimized implementation failed: {e}")
            return {'error': str(e)}
    
    def compare_features(self, original_features: Dict[str, np.ndarray], 
                        optimized_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compare feature outputs between implementations."""
        comparison = {}
        
        # Find common features
        common_features = set(original_features.keys()) & set(optimized_features.keys())
        
        for feature_name in common_features:
            orig_feat = original_features[feature_name]
            opt_feat = optimized_features[feature_name]
            
            # Ensure same length
            min_len = min(len(orig_feat), len(opt_feat))
            orig_feat = orig_feat[:min_len]
            opt_feat = opt_feat[:min_len]
            
            # Calculate correlation
            try:
                correlation = np.corrcoef(orig_feat, opt_feat)[0, 1]
                if not np.isnan(correlation):
                    comparison[feature_name] = {
                        'correlation': correlation,
                        'original_mean': np.mean(orig_feat),
                        'optimized_mean': np.mean(opt_feat),
                        'original_std': np.std(orig_feat),
                        'optimized_std': np.std(opt_feat),
                        'mean_difference': abs(np.mean(orig_feat) - np.mean(opt_feat)),
                        'std_difference': abs(np.std(orig_feat) - np.std(opt_feat))
                    }
            except Exception as e:
                comparison[feature_name] = {'error': str(e)}
        
        return comparison
    
    def run_comparison(self, data_sizes: List[int] = [100, 500, 1000, 2000, 5000]) -> Dict[str, Any]:
        """Run comprehensive comparison across different data sizes."""
        logger.info("Starting comprehensive comparison...")
        
        results = {
            'data_sizes': data_sizes,
            'original_results': {},
            'optimized_results': {},
            'comparisons': {},
            'summary': {}
        }
        
        for size in data_sizes:
            logger.info(f"Testing with data size: {size}")
            
            # Create test data
            data = self.create_test_data(size)
            
            # Benchmark both implementations
            original_result = self.benchmark_original(data)
            optimized_result = self.benchmark_optimized(data)
            
            results['original_results'][size] = original_result
            results['optimized_results'][size] = optimized_result
            
            # Compare features if both succeeded
            if (original_result.get('success') and optimized_result.get('success') and 
                'features' in original_result and 'features' in optimized_result):
                
                feature_comparison = self.compare_features(
                    original_result['features'], 
                    optimized_result['features']
                )
                results['comparisons'][size] = feature_comparison
        
        # Calculate summary statistics
        results['summary'] = self.calculate_summary(results)
        
        return results
    
    def calculate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics from comparison results."""
        summary = {
            'speedup_ratios': {},
            'memory_improvements': {},
            'feature_correlations': {},
            'performance_improvements': {}
        }
        
        for size in results['data_sizes']:
            orig = results['original_results'].get(size, {})
            opt = results['optimized_results'].get(size, {})
            
            if orig.get('success') and opt.get('success'):
                # Calculate speedup
                if orig.get('execution_time') and opt.get('execution_time'):
                    speedup = orig['execution_time'] / opt['execution_time']
                    summary['speedup_ratios'][size] = speedup
                
                # Calculate memory improvement
                if orig.get('memory_usage') and opt.get('memory_usage'):
                    memory_improvement = (orig['memory_usage'] - opt['memory_usage']) / orig['memory_usage'] * 100
                    summary['memory_improvements'][size] = memory_improvement
                
                # Calculate feature correlations
                if size in results['comparisons']:
                    correlations = []
                    for feature_name, comparison in results['comparisons'][size].items():
                        if 'correlation' in comparison:
                            correlations.append(comparison['correlation'])
                    
                    if correlations:
                        summary['feature_correlations'][size] = {
                            'mean_correlation': np.mean(correlations),
                            'min_correlation': np.min(correlations),
                            'max_correlation': np.max(correlations),
                            'std_correlation': np.std(correlations)
                        }
                
                # Calculate performance improvements
                if 'performance_stats' in opt:
                    stats = opt['performance_stats']
                    summary['performance_improvements'][size] = {
                        'vectorbt_usage': stats.get('vectorbt_usage_percentage', 0),
                        'batch_usage': stats.get('batch_usage_percentage', 0),
                        'chunked_usage': stats.get('chunked_usage_percentage', 0),
                        'gpu_usage': stats.get('gpu_usage_percentage', 0),
                        'cache_hit_rate': stats.get('cache_hit_rate', 0)
                    }
        
        return summary
    
    def print_results(self, results: Dict[str, Any]):
        """Print comparison results in a readable format."""
        print("\n" + "="*80)
        print("REGRIME STATISTICAL VECTORBT OPTIMIZATION COMPARISON")
        print("="*80)
        
        # Print summary
        summary = results['summary']
        
        print("\n📊 PERFORMANCE SUMMARY")
        print("-" * 40)
        
        if summary['speedup_ratios']:
            print("Speedup Ratios (Higher is Better):")
            for size, speedup in summary['speedup_ratios'].items():
                print(f"  Data size {size:4d}: {speedup:.2f}x speedup")
        
        if summary['memory_improvements']:
            print("\nMemory Improvements (Higher is Better):")
            for size, improvement in summary['memory_improvements'].items():
                print(f"  Data size {size:4d}: {improvement:.1f}% reduction")
        
        if summary['feature_correlations']:
            print("\nFeature Correlations (Closer to 1.0 is Better):")
            for size, corr_stats in summary['feature_correlations'].items():
                print(f"  Data size {size:4d}: Mean={corr_stats['mean_correlation']:.4f}, "
                      f"Min={corr_stats['min_correlation']:.4f}, Max={corr_stats['max_correlation']:.4f}")
        
        if summary['performance_improvements']:
            print("\nVectorBT Usage Statistics:")
            for size, perf_stats in summary['performance_improvements'].items():
                print(f"  Data size {size:4d}: VectorBT={perf_stats['vectorbt_usage']:.1f}%, "
                      f"Batch={perf_stats['batch_usage']:.1f}%, "
                      f"Chunked={perf_stats['chunked_usage']:.1f}%, "
                      f"Cache={perf_stats['cache_hit_rate']:.1f}%")
        
        # Print detailed results for each data size
        print("\n📈 DETAILED RESULTS")
        print("-" * 40)
        
        for size in results['data_sizes']:
            print(f"\nData Size: {size}")
            print("-" * 20)
            
            orig = results['original_results'].get(size, {})
            opt = results['optimized_results'].get(size, {})
            
            if orig.get('success'):
                print(f"Original:  {orig['execution_time']:.4f}s, {orig['feature_count']} features")
            else:
                print(f"Original:  FAILED - {orig.get('error', 'Unknown error')}")
            
            if opt.get('success'):
                print(f"Optimized: {opt['execution_time']:.4f}s, {opt['feature_count']} features")
            else:
                print(f"Optimized: FAILED - {opt.get('error', 'Unknown error')}")
            
            if orig.get('success') and opt.get('success'):
                speedup = orig['execution_time'] / opt['execution_time']
                print(f"Speedup:   {speedup:.2f}x")
        
        print("\n" + "="*80)


def main():
    """Run the comparison."""
    comparison = RegimeStatisticalComparison()
    
    # Test with different data sizes
    data_sizes = [100, 500, 1000, 2000, 5000]
    
    print("Starting Regime Statistical VectorBT Optimization Comparison...")
    print(f"Testing with data sizes: {data_sizes}")
    
    # Run comparison
    results = comparison.run_comparison(data_sizes)
    
    # Print results
    comparison.print_results(results)
    
    # Save results to file
    import json
    with open('regime_statistical_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: regime_statistical_comparison_results.json")
    
    return results


if __name__ == "__main__":
    main()