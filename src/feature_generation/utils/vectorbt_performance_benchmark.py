"""
VectorBT Performance Benchmarking

This module provides comprehensive benchmarking tools to compare
VectorBT-optimized implementations against the original implementations.

Key Features:
- Performance comparison (speed, memory usage)
- Accuracy validation (results match within tolerance)
- Scalability testing (different data sizes)
- Feature generation benchmarking
- Normalization/scaling benchmarking
"""

import time
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import psutil
import gc
from contextlib import contextmanager

# VectorBT imports
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False

from ..core.feature_generator import FeatureGenerator
from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator
from ...features_common.transforms.base_scaler import create_optimized_scaler
from ...features_common.transforms.vectorbt_scaler import VectorBTScaler

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    name: str
    method: str
    execution_time: float
    memory_usage: float
    accuracy_score: float
    data_size: int
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ComparisonResult:
    """Results from comparing two implementations."""
    original: BenchmarkResult
    vectorbt: BenchmarkResult
    speedup: float
    memory_improvement: float
    accuracy_difference: float
    recommendation: str


class PerformanceBenchmark:
    """Comprehensive performance benchmarking system."""
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize performance benchmark.
        
        Args:
            tolerance: Tolerance for accuracy comparison
        """
        self.tolerance = tolerance
        self.results = []
        
    @contextmanager
    def _measure_performance(self):
        """Context manager to measure execution time and memory usage."""
        # Measure initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        try:
            yield
        finally:
            # Measure final memory and time
            end_time = time.time()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            execution_time = end_time - start_time
            memory_usage = final_memory - initial_memory
            
            return execution_time, memory_usage
    
    def _generate_test_data(self, size: int = 1000, columns: List[str] = None) -> pd.DataFrame:
        """Generate test data for benchmarking."""
        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'volume']
        
        np.random.seed(42)  # For reproducible results
        
        data = {}
        for col in columns:
            if col == 'volume':
                data[col] = np.random.randint(1000, 100000, size)
            else:
                # Generate realistic price data
                base_price = 100
                returns = np.random.normal(0, 0.02, size)
                prices = base_price * np.exp(np.cumsum(returns))
                data[col] = prices
        
        # Ensure high >= low and high/low are reasonable relative to close
        data['high'] = np.maximum(data['high'], data['close'] * 1.01)
        data['low'] = np.minimum(data['low'], data['close'] * 0.99)
        
        index = pd.date_range('2020-01-01', periods=size, freq='1min')
        return pd.DataFrame(data, index=index)
    
    def _calculate_accuracy_score(self, original: pd.Series, vectorbt: pd.Series) -> float:
        """Calculate accuracy score between two series."""
        try:
            # Align indices
            common_index = original.index.intersection(vectorbt.index)
            if len(common_index) == 0:
                return 0.0
            
            orig_aligned = original.reindex(common_index)
            vbt_aligned = vectorbt.reindex(common_index)
            
            # Remove NaN values
            mask = ~(orig_aligned.isna() | vbt_aligned.isna())
            if not mask.any():
                return 0.0
            
            orig_clean = orig_aligned[mask]
            vbt_clean = vbt_aligned[mask]
            
            if len(orig_clean) == 0:
                return 0.0
            
            # Calculate correlation
            correlation = orig_clean.corr(vbt_clean)
            if pd.isna(correlation):
                return 0.0
            
            # Calculate mean absolute percentage error
            mape = np.mean(np.abs((orig_clean - vbt_clean) / orig_clean)) * 100
            
            # Calculate accuracy score (higher is better)
            accuracy = max(0, correlation - mape / 100)
            
            return accuracy
            
        except Exception as e:
            logger.warning(f"Error calculating accuracy score: {e}")
            return 0.0
    
    def benchmark_feature_generator(self, 
                                  original_generator: FeatureGenerator,
                                  vectorbt_generator: VectorBTFeatureGenerator,
                                  data_sizes: List[int] = [100, 500, 1000, 5000],
                                  iterations: int = 3) -> List[ComparisonResult]:
        """
        Benchmark feature generators.
        
        Args:
            original_generator: Original feature generator
            vectorbt_generator: VectorBT feature generator
            data_sizes: List of data sizes to test
            iterations: Number of iterations per test
            
        Returns:
            List of comparison results
        """
        results = []
        
        for size in data_sizes:
            logger.info(f"Benchmarking feature generators with data size {size}")
            
            # Generate test data
            test_data = self._generate_test_data(size)
            
            # Benchmark original generator
            original_results = []
            for i in range(iterations):
                try:
                    with self._measure_performance() as perf:
                        original_result = original_generator.generate(test_data)
                    
                    execution_time, memory_usage = perf
                    original_results.append(BenchmarkResult(
                        name=f"{original_generator.config.name}_original",
                        method="original",
                        execution_time=execution_time,
                        memory_usage=memory_usage,
                        accuracy_score=1.0,  # Reference
                        data_size=size,
                        success=original_result.success,
                        error_message=original_result.error_message,
                        metadata={'iteration': i}
                    ))
                except Exception as e:
                    original_results.append(BenchmarkResult(
                        name=f"{original_generator.config.name}_original",
                        method="original",
                        execution_time=0.0,
                        memory_usage=0.0,
                        accuracy_score=0.0,
                        data_size=size,
                        success=False,
                        error_message=str(e)
                    ))
            
            # Benchmark VectorBT generator
            vectorbt_results = []
            for i in range(iterations):
                try:
                    with self._measure_performance() as perf:
                        vectorbt_result = vectorbt_generator.generate(test_data)
                    
                    execution_time, memory_usage = perf
                    vectorbt_results.append(BenchmarkResult(
                        name=f"{vectorbt_generator.config.name}_vectorbt",
                        method="vectorbt",
                        execution_time=execution_time,
                        memory_usage=memory_usage,
                        accuracy_score=1.0,  # Will be calculated later
                        data_size=size,
                        success=vectorbt_result.success,
                        error_message=vectorbt_result.error_message,
                        metadata={'iteration': i}
                    ))
                except Exception as e:
                    vectorbt_results.append(BenchmarkResult(
                        name=f"{vectorbt_generator.config.name}_vectorbt",
                        method="vectorbt",
                        execution_time=0.0,
                        memory_usage=0.0,
                        accuracy_score=0.0,
                        data_size=size,
                        success=False,
                        error_message=str(e)
                    ))
            
            # Calculate average results
            original_avg = self._average_benchmark_results(original_results)
            vectorbt_avg = self._average_benchmark_results(vectorbt_results)
            
            # Calculate accuracy if both succeeded
            if original_avg.success and vectorbt_avg.success:
                try:
                    original_data = original_generator.generate(test_data).data
                    vectorbt_data = vectorbt_generator.generate(test_data).data
                    accuracy_score = self._calculate_accuracy_score(original_data, vectorbt_data)
                    vectorbt_avg.accuracy_score = accuracy_score
                except Exception as e:
                    logger.warning(f"Error calculating accuracy: {e}")
                    vectorbt_avg.accuracy_score = 0.0
            
            # Create comparison result
            comparison = ComparisonResult(
                original=original_avg,
                vectorbt=vectorbt_avg,
                speedup=original_avg.execution_time / vectorbt_avg.execution_time if vectorbt_avg.execution_time > 0 else 0,
                memory_improvement=(original_avg.memory_usage - vectorbt_avg.memory_usage) / original_avg.memory_usage if original_avg.memory_usage > 0 else 0,
                accuracy_difference=vectorbt_avg.accuracy_score - original_avg.accuracy_score,
                recommendation=self._generate_recommendation(original_avg, vectorbt_avg)
            )
            
            results.append(comparison)
            
            # Force garbage collection
            gc.collect()
        
        return results
    
    def benchmark_scaler(self, 
                        method: str = 'zscore',
                        data_sizes: List[int] = [100, 500, 1000, 5000],
                        iterations: int = 3) -> List[ComparisonResult]:
        """
        Benchmark scalers.
        
        Args:
            method: Scaling method to test
            data_sizes: List of data sizes to test
            iterations: Number of iterations per test
            
        Returns:
            List of comparison results
        """
        results = []
        
        for size in data_sizes:
            logger.info(f"Benchmarking scalers with data size {size}")
            
            # Generate test data
            test_data = self._generate_test_data(size)
            test_series = test_data['close']
            
            # Benchmark original scaler
            original_results = []
            for i in range(iterations):
                try:
                    from ...features_common.transforms.base_scaler import SimpleScaler
                    scaler = SimpleScaler()
                    
                    with self._measure_performance() as perf:
                        original_result = scaler.fit_transform(test_series)
                    
                    execution_time, memory_usage = perf
                    original_results.append(BenchmarkResult(
                        name=f"simple_scaler_{method}",
                        method="original",
                        execution_time=execution_time,
                        memory_usage=memory_usage,
                        accuracy_score=1.0,
                        data_size=size,
                        success=True,
                        metadata={'iteration': i}
                    ))
                except Exception as e:
                    original_results.append(BenchmarkResult(
                        name=f"simple_scaler_{method}",
                        method="original",
                        execution_time=0.0,
                        memory_usage=0.0,
                        accuracy_score=0.0,
                        data_size=size,
                        success=False,
                        error_message=str(e)
                    ))
            
            # Benchmark VectorBT scaler
            vectorbt_results = []
            for i in range(iterations):
                try:
                    scaler = VectorBTScaler(method)
                    
                    with self._measure_performance() as perf:
                        vectorbt_result = scaler.fit_transform(test_series)
                    
                    execution_time, memory_usage = perf
                    vectorbt_results.append(BenchmarkResult(
                        name=f"vectorbt_scaler_{method}",
                        method="vectorbt",
                        execution_time=execution_time,
                        memory_usage=memory_usage,
                        accuracy_score=1.0,
                        data_size=size,
                        success=True,
                        metadata={'iteration': i}
                    ))
                except Exception as e:
                    vectorbt_results.append(BenchmarkResult(
                        name=f"vectorbt_scaler_{method}",
                        method="vectorbt",
                        execution_time=0.0,
                        memory_usage=0.0,
                        accuracy_score=0.0,
                        data_size=size,
                        success=False,
                        error_message=str(e)
                    ))
            
            # Calculate average results
            original_avg = self._average_benchmark_results(original_results)
            vectorbt_avg = self._average_benchmark_results(vectorbt_results)
            
            # Calculate accuracy if both succeeded
            if original_avg.success and vectorbt_avg.success:
                try:
                    from ...features_common.transforms.base_scaler import SimpleScaler
                    original_scaler = SimpleScaler()
                    original_data = original_scaler.fit_transform(test_series)
                    
                    vectorbt_scaler = VectorBTScaler(method)
                    vectorbt_data = vectorbt_scaler.fit_transform(test_series)
                    
                    accuracy_score = self._calculate_accuracy_score(original_data, vectorbt_data)
                    vectorbt_avg.accuracy_score = accuracy_score
                except Exception as e:
                    logger.warning(f"Error calculating accuracy: {e}")
                    vectorbt_avg.accuracy_score = 0.0
            
            # Create comparison result
            comparison = ComparisonResult(
                original=original_avg,
                vectorbt=vectorbt_avg,
                speedup=original_avg.execution_time / vectorbt_avg.execution_time if vectorbt_avg.execution_time > 0 else 0,
                memory_improvement=(original_avg.memory_usage - vectorbt_avg.memory_usage) / original_avg.memory_usage if original_avg.memory_usage > 0 else 0,
                accuracy_difference=vectorbt_avg.accuracy_score - original_avg.accuracy_score,
                recommendation=self._generate_recommendation(original_avg, vectorbt_avg)
            )
            
            results.append(comparison)
            
            # Force garbage collection
            gc.collect()
        
        return results
    
    def _average_benchmark_results(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        """Calculate average of benchmark results."""
        if not results:
            return BenchmarkResult(
                name="empty",
                method="unknown",
                execution_time=0.0,
                memory_usage=0.0,
                accuracy_score=0.0,
                data_size=0,
                success=False
            )
        
        successful_results = [r for r in results if r.success]
        if not successful_results:
            return results[0]  # Return first failed result
        
        return BenchmarkResult(
            name=successful_results[0].name,
            method=successful_results[0].method,
            execution_time=np.mean([r.execution_time for r in successful_results]),
            memory_usage=np.mean([r.memory_usage for r in successful_results]),
            accuracy_score=np.mean([r.accuracy_score for r in successful_results]),
            data_size=successful_results[0].data_size,
            success=True,
            metadata={'iterations': len(successful_results)}
        )
    
    def _generate_recommendation(self, original: BenchmarkResult, vectorbt: BenchmarkResult) -> str:
        """Generate recommendation based on benchmark results."""
        if not original.success and not vectorbt.success:
            return "Both implementations failed - investigate errors"
        elif not original.success:
            return "Use VectorBT implementation - original failed"
        elif not vectorbt.success:
            return "Use original implementation - VectorBT failed"
        
        speedup = original.execution_time / vectorbt.execution_time if vectorbt.execution_time > 0 else 0
        memory_improvement = (original.memory_usage - vectorbt.memory_usage) / original.memory_usage if original.memory_usage > 0 else 0
        accuracy_diff = vectorbt.accuracy_score - original.accuracy_score
        
        if speedup > 2.0 and memory_improvement > 0.1 and accuracy_diff > -0.05:
            return f"Strongly recommend VectorBT - {speedup:.1f}x faster, {memory_improvement:.1%} less memory, similar accuracy"
        elif speedup > 1.5 and accuracy_diff > -0.1:
            return f"Recommend VectorBT - {speedup:.1f}x faster, acceptable accuracy difference"
        elif speedup > 1.2 and accuracy_diff > -0.05:
            return f"Consider VectorBT - {speedup:.1f}x faster, good accuracy"
        elif accuracy_diff < -0.1:
            return f"Consider original - VectorBT has significant accuracy loss ({accuracy_diff:.3f})"
        else:
            return "Both implementations perform similarly - choose based on other factors"
    
    def generate_report(self, results: List[ComparisonResult]) -> str:
        """Generate a comprehensive benchmark report."""
        if not results:
            return "No benchmark results available."
        
        report = ["# VectorBT Performance Benchmark Report\n"]
        
        # Summary statistics
        speedups = [r.speedup for r in results if r.speedup > 0]
        memory_improvements = [r.memory_improvement for r in results if r.memory_improvement > 0]
        accuracy_diffs = [r.accuracy_difference for r in results]
        
        if speedups:
            report.append(f"## Summary Statistics")
            report.append(f"- Average speedup: {np.mean(speedups):.2f}x")
            report.append(f"- Median speedup: {np.median(speedups):.2f}x")
            report.append(f"- Max speedup: {np.max(speedups):.2f}x")
            report.append(f"- Min speedup: {np.min(speedups):.2f}x\n")
        
        if memory_improvements:
            report.append(f"- Average memory improvement: {np.mean(memory_improvements):.1%}")
            report.append(f"- Median memory improvement: {np.median(memory_improvements):.1%}\n")
        
        if accuracy_diffs:
            report.append(f"- Average accuracy difference: {np.mean(accuracy_diffs):.3f}")
            report.append(f"- Median accuracy difference: {np.median(accuracy_diffs):.3f}\n")
        
        # Detailed results
        report.append("## Detailed Results\n")
        
        for i, result in enumerate(results, 1):
            report.append(f"### Test {i} - Data Size: {result.original.data_size}")
            report.append(f"- **Speedup**: {result.speedup:.2f}x")
            report.append(f"- **Memory Improvement**: {result.memory_improvement:.1%}")
            report.append(f"- **Accuracy Difference**: {result.accuracy_difference:.3f}")
            report.append(f"- **Recommendation**: {result.recommendation}")
            report.append("")
        
        return "\n".join(report)


def run_comprehensive_benchmark() -> str:
    """Run comprehensive benchmark of all VectorBT implementations."""
    if not VECTORBT_AVAILABLE:
        return "VectorBT not available - cannot run benchmarks"
    
    benchmark = PerformanceBenchmark()
    
    # Import generators
    try:
        from ..categories.volatility import VolatilityFeatureGenerator, VectorBTVolatilityFeatureGenerator
        from ..categories.momentum import MomentumFeatureGenerator, VectorBTMomentumFeatureGenerator
        from ..categories.trend import TrendFeatureGenerator, VectorBTTrendFeatureGenerator
    except ImportError as e:
        return f"Error importing generators: {e}"
    
    all_results = []
    
    # Benchmark volatility generators
    try:
        logger.info("Benchmarking volatility generators...")
        vol_results = benchmark.benchmark_feature_generator(
            VolatilityFeatureGenerator(20),
            VectorBTVolatilityFeatureGenerator(20),
            data_sizes=[100, 500, 1000]
        )
        all_results.extend(vol_results)
    except Exception as e:
        logger.warning(f"Volatility benchmark failed: {e}")
    
    # Benchmark momentum generators
    try:
        logger.info("Benchmarking momentum generators...")
        mom_results = benchmark.benchmark_feature_generator(
            MomentumFeatureGenerator(),
            VectorBTMomentumFeatureGenerator(14),
            data_sizes=[100, 500, 1000]
        )
        all_results.extend(mom_results)
    except Exception as e:
        logger.warning(f"Momentum benchmark failed: {e}")
    
    # Benchmark trend generators
    try:
        logger.info("Benchmarking trend generators...")
        trend_results = benchmark.benchmark_feature_generator(
            TrendFeatureGenerator(),
            VectorBTTrendFeatureGenerator(20),
            data_sizes=[100, 500, 1000]
        )
        all_results.extend(trend_results)
    except Exception as e:
        logger.warning(f"Trend benchmark failed: {e}")
    
    # Benchmark scalers
    try:
        logger.info("Benchmarking scalers...")
        scaler_results = benchmark.benchmark_scaler(
            method='zscore',
            data_sizes=[100, 500, 1000]
        )
        all_results.extend(scaler_results)
    except Exception as e:
        logger.warning(f"Scaler benchmark failed: {e}")
    
    # Generate report
    return benchmark.generate_report(all_results)


if __name__ == "__main__":
    # Run benchmark when executed directly
    print("Running VectorBT performance benchmark...")
    report = run_comprehensive_benchmark()
    print(report)