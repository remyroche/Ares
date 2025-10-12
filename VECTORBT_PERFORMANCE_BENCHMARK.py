#!/usr/bin/env python3
"""
VectorBT Performance Benchmarking Utility

This script provides comprehensive benchmarking tools to measure the performance
improvements achieved through VectorBT optimizations across different feature
generation categories.

Usage:
    python VECTORBT_PERFORMANCE_BENCHMARK.py --category all --sizes 1000,10000,100000
    python VECTORBT_PERFORMANCE_BENCHMARK.py --category momentum --iterations 10
"""

import argparse
import time
import psutil
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src')

try:
    from tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
except ImportError:
    def tprint(*args, **kwargs):
        print(*args, **kwargs)
    def tprint_info(*args, **kwargs):
        print(f"ℹ️  {' '.join(map(str, args))}")
    def tprint_success(*args, **kwargs):
        print(f"✅ {' '.join(map(str, args))}")
    def tprint_warning(*args, **kwargs):
        print(f"⚠️  {' '.join(map(str, args))}")
    def tprint_error(*args, **kwargs):
        print(f"❌ {' '.join(map(str, args))}")
    def tprint_debug(*args, **kwargs):
        print(f"🔍 {' '.join(map(str, args))}")

class PerformanceBenchmark:
    """Comprehensive performance benchmarking for VectorBT optimizations."""
    
    def __init__(self):
        self.results = {}
        self.memory_usage = {}
        self.gpu_available = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            import cupy as cp
            return True
        except ImportError:
            return False
    
    def generate_test_data(self, size: int, columns: List[str] = None) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'volume']
        
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic price data
        returns = np.random.normal(0, 0.02, size)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = pd.DataFrame(index=pd.date_range('2020-01-01', periods=size, freq='1min'))
        
        if 'close' in columns:
            data['close'] = prices
        
        if 'open' in columns:
            data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
        
        if 'high' in columns:
            data['high'] = data['close'] * (1 + np.abs(np.random.normal(0, 0.01, size)))
        
        if 'low' in columns:
            data['low'] = data['close'] * (1 - np.abs(np.random.normal(0, 0.01, size)))
        
        if 'volume' in columns:
            data['volume'] = np.random.randint(1000, 10000, size)
        
        return data
    
    def benchmark_trend_features(self, data: pd.DataFrame, iterations: int = 5) -> Dict[str, Any]:
        """Benchmark trend feature generation."""
        tprint_info(f"🔄 Benchmarking trend features with {len(data)} rows, {iterations} iterations")
        
        try:
            from feature_generation.categories.trend import TrendFeatureGenerator
            generator = TrendFeatureGenerator()
        except ImportError as e:
            tprint_error(f"Failed to import TrendFeatureGenerator: {e}")
            return {}
        
        # Test individual operations
        individual_times = []
        for _ in range(iterations):
            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            # Generate individual features
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean()
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            
            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            individual_times.append(end_time - start_time)
        
        # Test batch operations
        batch_times = []
        for _ in range(iterations):
            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            # Generate batch features
            batch_features = generator.generate_moving_averages_batch(
                data, 
                windows=[20, 50], 
                columns=['close'],
                operation='mean'
            )
            
            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            batch_times.append(end_time - start_time)
        
        avg_individual = np.mean(individual_times)
        avg_batch = np.mean(batch_times)
        speedup = avg_individual / avg_batch if avg_batch > 0 else 0
        
        return {
            'category': 'trend',
            'individual_time': avg_individual,
            'batch_time': avg_batch,
            'speedup': speedup,
            'memory_usage_mb': end_memory - start_memory,
            'iterations': iterations
        }
    
    def benchmark_volatility_features(self, data: pd.DataFrame, iterations: int = 5) -> Dict[str, Any]:
        """Benchmark volatility feature generation."""
        tprint_info(f"🔄 Benchmarking volatility features with {len(data)} rows, {iterations} iterations")
        
        try:
            from feature_generation.categories.volatility import VolatilityFeatureGenerator
            generator = VolatilityFeatureGenerator()
        except ImportError as e:
            tprint_error(f"Failed to import VolatilityFeatureGenerator: {e}")
            return {}
        
        # Test individual operations
        individual_times = []
        for _ in range(iterations):
            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            # Generate individual Bollinger Bands
            sma_20 = data['close'].rolling(20).mean()
            std_20 = data['close'].rolling(20).std()
            bb_upper = sma_20 + (std_20 * 2)
            bb_lower = sma_20 - (std_20 * 2)
            
            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            individual_times.append(end_time - start_time)
        
        # Test batch operations
        batch_times = []
        for _ in range(iterations):
            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            # Generate batch features
            batch_features = generator.generate_bollinger_bands_batch(
                data, 
                windows=[20, 50], 
                std_devs=[2.0, 2.5]
            )
            
            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            batch_times.append(end_time - start_time)
        
        avg_individual = np.mean(individual_times)
        avg_batch = np.mean(batch_times)
        speedup = avg_individual / avg_batch if avg_batch > 0 else 0
        
        return {
            'category': 'volatility',
            'individual_time': avg_individual,
            'batch_time': avg_batch,
            'speedup': speedup,
            'memory_usage_mb': end_memory - start_memory,
            'iterations': iterations
        }
    
    def benchmark_volume_features(self, data: pd.DataFrame, iterations: int = 5) -> Dict[str, Any]:
        """Benchmark volume feature generation."""
        tprint_info(f"🔄 Benchmarking volume features with {len(data)} rows, {iterations} iterations")
        
        try:
            from feature_generation.categories.volume import VolumeFeatureGenerator
            generator = VolumeFeatureGenerator()
        except ImportError as e:
            tprint_error(f"Failed to import VolumeFeatureGenerator: {e}")
            return {}
        
        # Test individual operations
        individual_times = []
        for _ in range(iterations):
            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            # Generate individual volume features
            volume_sma_20 = data['volume'].rolling(20).mean()
            volume_ema_12 = data['volume'].ewm(span=12).mean()
            volume_ratio = data['volume'] / volume_sma_20
            
            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            individual_times.append(end_time - start_time)
        
        # Test batch operations
        batch_times = []
        for _ in range(iterations):
            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            # Generate batch features
            batch_features = generator.generate_volume_indicators_batch(
                data,
                sma_windows=[20, 50],
                ema_windows=[12, 26],
                ratio_windows=[5, 10]
            )
            
            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            batch_times.append(end_time - start_time)
        
        avg_individual = np.mean(individual_times)
        avg_batch = np.mean(batch_times)
        speedup = avg_individual / avg_batch if avg_batch > 0 else 0
        
        return {
            'category': 'volume',
            'individual_time': avg_individual,
            'batch_time': avg_batch,
            'speedup': speedup,
            'memory_usage_mb': end_memory - start_memory,
            'iterations': iterations
        }
    
    def benchmark_momentum_features(self, data: pd.DataFrame, iterations: int = 5) -> Dict[str, Any]:
        """Benchmark momentum feature generation."""
        tprint_info(f"🔄 Benchmarking momentum features with {len(data)} rows, {iterations} iterations")
        
        try:
            from feature_generation.categories.momentum import OptimizedMomentumFeatureGenerator
            generator = OptimizedMomentumFeatureGenerator()
        except ImportError as e:
            tprint_error(f"Failed to import OptimizedMomentumFeatureGenerator: {e}")
            return {}
        
        # Test individual operations
        individual_times = []
        for _ in range(iterations):
            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            # Generate individual RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            individual_times.append(end_time - start_time)
        
        # Test batch operations
        batch_times = []
        for _ in range(iterations):
            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            # Generate batch features
            batch_features = generator.generate_rsi_features_batch(
                data,
                periods=[14, 21, 30],
                columns=['close']
            )
            
            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            batch_times.append(end_time - start_time)
        
        avg_individual = np.mean(individual_times)
        avg_batch = np.mean(batch_times)
        speedup = avg_individual / avg_batch if avg_batch > 0 else 0
        
        return {
            'category': 'momentum',
            'individual_time': avg_individual,
            'batch_time': avg_batch,
            'speedup': speedup,
            'memory_usage_mb': end_memory - start_memory,
            'iterations': iterations
        }
    
    def benchmark_interactive_features(self, data: pd.DataFrame, iterations: int = 5) -> Dict[str, Any]:
        """Benchmark interactive feature generation."""
        tprint_info(f"🔄 Benchmarking interactive features with {len(data)} rows, {iterations} iterations")
        
        try:
            from training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.interactive_feature_generation_component import InteractiveFeatureGenerationComponent
            component = InteractiveFeatureGenerationComponent()
        except ImportError as e:
            tprint_error(f"Failed to import InteractiveFeatureGenerationComponent: {e}")
            return {}
        
        # Test individual operations
        individual_times = []
        for _ in range(iterations):
            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            # Generate individual rolling features
            rolling_mean_20 = data['close'].rolling(20).mean()
            rolling_std_20 = data['close'].rolling(20).std()
            rolling_corr = data['close'].rolling(20).corr(data['volume'])
            
            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            individual_times.append(end_time - start_time)
        
        # Test batch operations
        batch_times = []
        for _ in range(iterations):
            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            # Generate batch features
            batch_features = component.generate_rolling_features_batch(
                data,
                windows=[20, 50],
                operations=['mean', 'std', 'var'],
                columns=['close', 'volume']
            )
            
            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            
            batch_times.append(end_time - start_time)
        
        avg_individual = np.mean(individual_times)
        avg_batch = np.mean(batch_times)
        speedup = avg_individual / avg_batch if avg_batch > 0 else 0
        
        return {
            'category': 'interactive',
            'individual_time': avg_individual,
            'batch_time': avg_batch,
            'speedup': speedup,
            'memory_usage_mb': end_memory - start_memory,
            'iterations': iterations
        }
    
    def run_comprehensive_benchmark(self, sizes: List[int], categories: List[str], iterations: int = 5) -> Dict[str, Any]:
        """Run comprehensive benchmark across multiple data sizes and categories."""
        tprint_info(f"🚀 Starting comprehensive benchmark: sizes={sizes}, categories={categories}, iterations={iterations}")
        
        all_results = {}
        
        for size in sizes:
            tprint_info(f"📊 Testing with {size:,} rows")
            data = self.generate_test_data(size)
            
            size_results = {}
            
            for category in categories:
                try:
                    if category == 'trend':
                        result = self.benchmark_trend_features(data, iterations)
                    elif category == 'volatility':
                        result = self.benchmark_volatility_features(data, iterations)
                    elif category == 'volume':
                        result = self.benchmark_volume_features(data, iterations)
                    elif category == 'momentum':
                        result = self.benchmark_momentum_features(data, iterations)
                    elif category == 'interactive':
                        result = self.benchmark_interactive_features(data, iterations)
                    else:
                        tprint_warning(f"Unknown category: {category}")
                        continue
                    
                    if result:
                        size_results[category] = result
                        tprint_success(f"✅ {category}: {result['speedup']:.2f}x speedup")
                    
                except Exception as e:
                    tprint_error(f"❌ {category} benchmark failed: {e}")
                    continue
            
            all_results[size] = size_results
        
        return all_results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive performance report."""
        report = []
        report.append("# VectorBT Performance Benchmark Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        all_speedups = []
        for size_results in results.values():
            for category_result in size_results.values():
                all_speedups.append(category_result['speedup'])
        
        if all_speedups:
            avg_speedup = np.mean(all_speedups)
            max_speedup = np.max(all_speedups)
            min_speedup = np.min(all_speedups)
            
            report.append("## Summary Statistics")
            report.append(f"- Average Speedup: {avg_speedup:.2f}x")
            report.append(f"- Maximum Speedup: {max_speedup:.2f}x")
            report.append(f"- Minimum Speedup: {min_speedup:.2f}x")
            report.append("")
        
        # Detailed results by size
        for size, size_results in results.items():
            report.append(f"## Data Size: {size:,} rows")
            report.append("")
            
            for category, result in size_results.items():
                report.append(f"### {category.title()} Features")
                report.append(f"- Individual Time: {result['individual_time']:.4f}s")
                report.append(f"- Batch Time: {result['batch_time']:.4f}s")
                report.append(f"- Speedup: {result['speedup']:.2f}x")
                report.append(f"- Memory Usage: {result['memory_usage_mb']:.1f} MB")
                report.append("")
        
        # Performance recommendations
        report.append("## Performance Recommendations")
        report.append("")
        
        if all_speedups:
            if avg_speedup > 2.0:
                report.append("✅ Excellent performance improvements achieved!")
            elif avg_speedup > 1.5:
                report.append("✅ Good performance improvements achieved.")
            elif avg_speedup > 1.2:
                report.append("⚠️ Moderate performance improvements. Consider further optimization.")
            else:
                report.append("❌ Limited performance improvements. Review implementation.")
        
        report.append("")
        report.append("### Optimization Tips:")
        report.append("- Use batch processing for multiple features")
        report.append("- Enable GPU acceleration for large datasets")
        report.append("- Use memory-efficient data types")
        report.append("- Process features in chunks for very large datasets")
        report.append("")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], filename: str = "vectorbt_benchmark_results.json"):
        """Save benchmark results to JSON file."""
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        converted_results = recursive_convert(results)
        
        with open(filename, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        tprint_success(f"💾 Results saved to {filename}")


def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description='VectorBT Performance Benchmark')
    parser.add_argument('--category', nargs='+', default=['all'], 
                       choices=['all', 'trend', 'volatility', 'volume', 'momentum', 'interactive'],
                       help='Categories to benchmark')
    parser.add_argument('--sizes', type=str, default='1000,10000,100000',
                       help='Data sizes to test (comma-separated)')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of iterations per test')
    parser.add_argument('--output', type=str, default='vectorbt_benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--report', type=str, default='vectorbt_benchmark_report.md',
                       help='Output file for report')
    
    args = parser.parse_args()
    
    # Parse sizes
    sizes = [int(s.strip()) for s in args.sizes.split(',')]
    
    # Determine categories
    if 'all' in args.category:
        categories = ['trend', 'volatility', 'volume', 'momentum', 'interactive']
    else:
        categories = args.category
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark()
    
    # Run benchmark
    tprint_info("🚀 Starting VectorBT Performance Benchmark")
    tprint_info(f"📊 Categories: {categories}")
    tprint_info(f"📏 Sizes: {sizes}")
    tprint_info(f"🔄 Iterations: {args.iterations}")
    tprint_info(f"🖥️  GPU Available: {benchmark.gpu_available}")
    tprint_info("")
    
    results = benchmark.run_comprehensive_benchmark(sizes, categories, args.iterations)
    
    # Generate and save report
    report = benchmark.generate_report(results)
    with open(args.report, 'w') as f:
        f.write(report)
    
    # Save results
    benchmark.save_results(results, args.output)
    
    # Print summary
    tprint_success("🎉 Benchmark completed!")
    tprint_info(f"📄 Report saved to: {args.report}")
    tprint_info(f"📊 Results saved to: {args.output}")
    
    # Print quick summary
    all_speedups = []
    for size_results in results.values():
        for category_result in size_results.values():
            all_speedups.append(category_result['speedup'])
    
    if all_speedups:
        avg_speedup = np.mean(all_speedups)
        tprint_success(f"📈 Average speedup: {avg_speedup:.2f}x")


if __name__ == "__main__":
    main()