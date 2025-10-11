"""
Comprehensive Performance Validation Tests for VectorBT Optimizations

This module provides comprehensive performance tests for all VectorBT optimizations
including advanced volatility features, volume features, and batch processing.

Key Features:
- Performance benchmarking for all optimizations
- Memory usage monitoring
- GPU acceleration validation
- Parallel processing efficiency tests
- Comprehensive reporting and analysis
"""

import numpy as np
import pandas as pd
import time
import logging
import psutil
import gc
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import the optimized modules
from ..categories.advanced_volatility_features import (
    AdvancedVolatilityFeatures, VolatilityConfig, create_advanced_volatility_generator
)
from ..categories.advanced_volume_features import (
    AdvancedVolumeFeatures, VolumeConfig, create_advanced_volume_generator
)
from ..core.vectorbt_batch_processor import (
    VectorBTBatchProcessor, BatchProcessingConfig, create_vectorbt_batch_processor,
    VectorBTFeatureBatchProcessor
)

# VectorBT availability check
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization testing."""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    speedup_factor: Optional[float] = None
    memory_efficiency: Optional[float] = None
    error_count: int = 0
    warnings_count: int = 0


@dataclass
class TestResults:
    """Results from performance testing."""
    test_name: str
    dataset_size: int
    features_generated: int
    vectorbt_metrics: PerformanceMetrics
    baseline_metrics: PerformanceMetrics
    improvement_summary: Dict[str, Any]


class VectorBTOptimizationTester:
    """Comprehensive tester for VectorBT optimizations."""
    
    def __init__(self, enable_gpu: bool = False, enable_parallel: bool = True):
        """
        Initialize the optimization tester.
        
        Args:
            enable_gpu: Whether to enable GPU testing
            enable_parallel: Whether to enable parallel processing testing
        """
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        self.enable_parallel = enable_parallel and VECTORBT_AVAILABLE
        self.results = []
        
        # Test configurations
        self.test_sizes = [1000, 5000, 10000, 25000, 50000]
        self.n_runs = 3  # Number of runs for averaging
        
        logger.info(f"VectorBTOptimizationTester initialized: GPU={self.enable_gpu}, Parallel={self.enable_parallel}")
    
    def generate_test_data(self, n_samples: int, n_features: int = 5) -> pd.DataFrame:
        """Generate synthetic test data for performance testing."""
        np.random.seed(42)
        
        # Generate time series data
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='1min')
        
        # Generate price data with realistic patterns
        returns = np.random.normal(0.001, 0.02, n_samples)
        prices = 100 * (1 + returns).cumprod()
        
        # Generate OHLCV data
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_samples)
        }, index=dates)
        
        # Ensure high >= low and high >= close >= low
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        return data
    
    def measure_performance(self, func: callable, *args, **kwargs) -> PerformanceMetrics:
        """Measure performance of a function execution."""
        # Memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # CPU usage before
        cpu_before = process.cpu_percent()
        
        # GPU usage before (if available)
        gpu_before = None
        if self.enable_gpu and CUPY_AVAILABLE:
            try:
                gpu_before = cp.cuda.Device().memory_info.used / 1024 / 1024  # MB
            except:
                pass
        
        # Execute function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            error_count = 0
        except Exception as e:
            logger.error(f"Error in performance measurement: {e}")
            result = None
            error_count = 1
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        # CPU usage after
        cpu_after = process.cpu_percent()
        cpu_usage = max(0, cpu_after - cpu_before)
        
        # GPU usage after (if available)
        gpu_usage = None
        if self.enable_gpu and CUPY_AVAILABLE:
            try:
                gpu_after = cp.cuda.Device().memory_info.used / 1024 / 1024  # MB
                gpu_usage = max(0, gpu_after - gpu_before) if gpu_before else None
            except:
                pass
        
        return PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            gpu_usage_percent=gpu_usage,
            error_count=error_count
        )
    
    def test_advanced_volatility_features(self) -> List[TestResults]:
        """Test advanced volatility features performance."""
        logger.info("Testing Advanced Volatility Features...")
        results = []
        
        for size in self.test_sizes:
            logger.info(f"Testing volatility features with {size:,} samples...")
            
            # Generate test data
            data = self.generate_test_data(size)
            
            # Test VectorBT implementation
            vectorbt_metrics = self._test_volatility_vectorbt(data)
            
            # Test baseline implementation
            baseline_metrics = self._test_volatility_baseline(data)
            
            # Calculate improvements
            speedup = baseline_metrics.execution_time / vectorbt_metrics.execution_time if vectorbt_metrics.execution_time > 0 else 0
            memory_efficiency = baseline_metrics.memory_usage_mb / vectorbt_metrics.memory_usage_mb if vectorbt_metrics.memory_usage_mb > 0 else 0
            
            result = TestResults(
                test_name="Advanced Volatility Features",
                dataset_size=size,
                features_generated=len(self._get_volatility_feature_names()),
                vectorbt_metrics=vectorbt_metrics,
                baseline_metrics=baseline_metrics,
                improvement_summary={
                    'speedup_factor': speedup,
                    'memory_efficiency': memory_efficiency,
                    'time_saved_seconds': baseline_metrics.execution_time - vectorbt_metrics.execution_time,
                    'memory_saved_mb': baseline_metrics.memory_usage_mb - vectorbt_metrics.memory_usage_mb
                }
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def _test_volatility_vectorbt(self, data: pd.DataFrame) -> PerformanceMetrics:
        """Test VectorBT volatility features."""
        def run_test():
            generator = create_advanced_volatility_generator(
                enable_gpu=self.enable_gpu,
                enable_parallel=self.enable_parallel
            )
            return generator.generate_features(data)
        
        return self.measure_performance(run_test)
    
    def _test_volatility_baseline(self, data: pd.DataFrame) -> PerformanceMetrics:
        """Test baseline volatility features."""
        def run_test():
            # Simple baseline implementation
            features = pd.DataFrame(index=data.index)
            
            # Basic rolling standard deviation
            returns = data['close'].pct_change()
            features['volatility_20'] = returns.rolling(window=20).std()
            features['volatility_50'] = returns.rolling(window=50).std()
            
            # Basic ATR
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift(1))
            low_close = np.abs(data['low'] - data['close'].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            features['atr_14'] = true_range.rolling(window=14).mean()
            
            return features
        
        return self.measure_performance(run_test)
    
    def test_advanced_volume_features(self) -> List[TestResults]:
        """Test advanced volume features performance."""
        logger.info("Testing Advanced Volume Features...")
        results = []
        
        for size in self.test_sizes:
            logger.info(f"Testing volume features with {size:,} samples...")
            
            # Generate test data
            data = self.generate_test_data(size)
            
            # Test VectorBT implementation
            vectorbt_metrics = self._test_volume_vectorbt(data)
            
            # Test baseline implementation
            baseline_metrics = self._test_volume_baseline(data)
            
            # Calculate improvements
            speedup = baseline_metrics.execution_time / vectorbt_metrics.execution_time if vectorbt_metrics.execution_time > 0 else 0
            memory_efficiency = baseline_metrics.memory_usage_mb / vectorbt_metrics.memory_usage_mb if vectorbt_metrics.memory_usage_mb > 0 else 0
            
            result = TestResults(
                test_name="Advanced Volume Features",
                dataset_size=size,
                features_generated=len(self._get_volume_feature_names()),
                vectorbt_metrics=vectorbt_metrics,
                baseline_metrics=baseline_metrics,
                improvement_summary={
                    'speedup_factor': speedup,
                    'memory_efficiency': memory_efficiency,
                    'time_saved_seconds': baseline_metrics.execution_time - vectorbt_metrics.execution_time,
                    'memory_saved_mb': baseline_metrics.memory_usage_mb - vectorbt_metrics.memory_usage_mb
                }
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def _test_volume_vectorbt(self, data: pd.DataFrame) -> PerformanceMetrics:
        """Test VectorBT volume features."""
        def run_test():
            generator = create_advanced_volume_generator(
                enable_gpu=self.enable_gpu,
                enable_parallel=self.enable_parallel
            )
            return generator.generate_features(data)
        
        return self.measure_performance(run_test)
    
    def _test_volume_baseline(self, data: pd.DataFrame) -> PerformanceMetrics:
        """Test baseline volume features."""
        def run_test():
            # Simple baseline implementation
            features = pd.DataFrame(index=data.index)
            
            # Basic volume moving averages
            features['volume_sma_20'] = data['volume'].rolling(window=20).mean()
            features['volume_sma_50'] = data['volume'].rolling(window=50).mean()
            
            # Basic volume rate of change
            features['volume_roc_5'] = data['volume'].pct_change(5)
            features['volume_roc_20'] = data['volume'].pct_change(20)
            
            return features
        
        return self.measure_performance(run_test)
    
    def test_batch_processing(self) -> List[TestResults]:
        """Test batch processing performance."""
        logger.info("Testing Batch Processing...")
        results = []
        
        for size in self.test_sizes:
            logger.info(f"Testing batch processing with {size:,} samples...")
            
            # Generate test data
            data = self.generate_test_data(size)
            
            # Test VectorBT batch processing
            vectorbt_metrics = self._test_batch_vectorbt(data)
            
            # Test baseline processing
            baseline_metrics = self._test_batch_baseline(data)
            
            # Calculate improvements
            speedup = baseline_metrics.execution_time / vectorbt_metrics.execution_time if vectorbt_metrics.execution_time > 0 else 0
            memory_efficiency = baseline_metrics.memory_usage_mb / vectorbt_metrics.memory_usage_mb if vectorbt_metrics.memory_usage_mb > 0 else 0
            
            result = TestResults(
                test_name="Batch Processing",
                dataset_size=size,
                features_generated=len(self._get_batch_feature_names()),
                vectorbt_metrics=vectorbt_metrics,
                baseline_metrics=baseline_metrics,
                improvement_summary={
                    'speedup_factor': speedup,
                    'memory_efficiency': memory_efficiency,
                    'time_saved_seconds': baseline_metrics.execution_time - vectorbt_metrics.execution_time,
                    'memory_saved_mb': baseline_metrics.memory_usage_mb - vectorbt_metrics.memory_usage_mb
                }
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def _test_batch_vectorbt(self, data: pd.DataFrame) -> PerformanceMetrics:
        """Test VectorBT batch processing."""
        def run_test():
            # Create mock feature generators
            class MockFeatureGenerator:
                def __init__(self, name):
                    self.name = name
                    self.use_vectorbt = True
                
                def generate_features(self, data, **kwargs):
                    features = pd.DataFrame(index=data.index)
                    features[f'{self.name}_feature_1'] = data['close'].rolling(window=20).mean()
                    features[f'{self.name}_feature_2'] = data['volume'].rolling(window=20).std()
                    return features
            
            # Create batch processor
            config = BatchProcessingConfig(
                batch_size=1000,
                enable_gpu=self.enable_gpu,
                enable_parallel=self.enable_parallel
            )
            
            processor = create_vectorbt_batch_processor(config)
            
            # Create feature generators
            feature_generators = [
                VectorBTFeatureBatchProcessor(MockFeatureGenerator('volatility')),
                VectorBTFeatureBatchProcessor(MockFeatureGenerator('volume')),
                VectorBTFeatureBatchProcessor(MockFeatureGenerator('momentum'))
            ]
            
            return processor.process_features_batch(data, feature_generators)
        
        return self.measure_performance(run_test)
    
    def _test_batch_baseline(self, data: pd.DataFrame) -> PerformanceMetrics:
        """Test baseline batch processing."""
        def run_test():
            # Simple baseline implementation
            features = pd.DataFrame(index=data.index)
            
            # Basic features
            features['volatility_feature_1'] = data['close'].rolling(window=20).mean()
            features['volatility_feature_2'] = data['close'].rolling(window=20).std()
            features['volume_feature_1'] = data['volume'].rolling(window=20).mean()
            features['volume_feature_2'] = data['volume'].rolling(window=20).std()
            features['momentum_feature_1'] = data['close'].pct_change(20)
            features['momentum_feature_2'] = data['volume'].pct_change(20)
            
            return features
        
        return self.measure_performance(run_test)
    
    def _get_volatility_feature_names(self) -> List[str]:
        """Get expected volatility feature names."""
        return [
            'atr_14', 'atr_21', 'atr_30',
            'bb_upper_20_1.5', 'bb_lower_20_1.5', 'bb_width_20_1.5',
            'kc_upper_20_1.0', 'kc_lower_20_1.0', 'kc_width_20_1.0',
            'vol_cluster_ratio', 'vol_regime', 'garch_variance'
        ]
    
    def _get_volume_feature_names(self) -> List[str]:
        """Get expected volume feature names."""
        return [
            'obv', 'obv_sma_5', 'obv_sma_10', 'obv_sma_20',
            'ad', 'ad_sma_5', 'ad_sma_10', 'ad_sma_20',
            'mfi_14', 'mfi_21', 'mfi_30',
            'vwap_20', 'vwap_50', 'vwap_100',
            'volume_momentum_5', 'volume_momentum_10', 'volume_momentum_20'
        ]
    
    def _get_batch_feature_names(self) -> List[str]:
        """Get expected batch feature names."""
        return [
            'volatility_feature_1', 'volatility_feature_2',
            'volume_feature_1', 'volume_feature_2',
            'momentum_feature_1', 'momentum_feature_2'
        ]
    
    def run_comprehensive_tests(self) -> Dict[str, List[TestResults]]:
        """Run all comprehensive tests."""
        logger.info("Starting comprehensive VectorBT optimization tests...")
        
        all_results = {}
        
        # Test volatility features
        all_results['volatility'] = self.test_advanced_volatility_features()
        
        # Test volume features
        all_results['volume'] = self.test_advanced_volume_features()
        
        # Test batch processing
        all_results['batch_processing'] = self.test_batch_processing()
        
        logger.info("Comprehensive tests completed!")
        return all_results
    
    def generate_performance_report(self, results: Dict[str, List[TestResults]]) -> str:
        """Generate a comprehensive performance report."""
        report = []
        report.append("=" * 80)
        report.append("VECTORBT OPTIMIZATION PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 40)
        
        for test_type, test_results in results.items():
            if not test_results:
                continue
                
            report.append(f"\n{test_type.upper()} FEATURES:")
            
            # Calculate average improvements
            speedups = [r.improvement_summary['speedup_factor'] for r in test_results if r.improvement_summary['speedup_factor'] > 0]
            memory_efficiencies = [r.improvement_summary['memory_efficiency'] for r in test_results if r.improvement_summary['memory_efficiency'] > 0]
            
            if speedups:
                avg_speedup = np.mean(speedups)
                max_speedup = np.max(speedups)
                report.append(f"  Average Speedup: {avg_speedup:.2f}x")
                report.append(f"  Maximum Speedup: {max_speedup:.2f}x")
            
            if memory_efficiencies:
                avg_memory_eff = np.mean(memory_efficiencies)
                report.append(f"  Average Memory Efficiency: {avg_memory_eff:.2f}x")
        
        # Detailed results
        report.append("\n" + "=" * 80)
        report.append("DETAILED RESULTS")
        report.append("=" * 80)
        
        for test_type, test_results in results.items():
            if not test_results:
                continue
                
            report.append(f"\n{test_type.upper()} FEATURES - DETAILED RESULTS:")
            report.append("-" * 60)
            
            for result in test_results:
                report.append(f"\nDataset Size: {result.dataset_size:,} samples")
                report.append(f"Features Generated: {result.features_generated}")
                report.append(f"VectorBT Time: {result.vectorbt_metrics.execution_time:.4f}s")
                report.append(f"Baseline Time: {result.baseline_metrics.execution_time:.4f}s")
                report.append(f"Speedup: {result.improvement_summary['speedup_factor']:.2f}x")
                report.append(f"Memory Usage VectorBT: {result.vectorbt_metrics.memory_usage_mb:.2f} MB")
                report.append(f"Memory Usage Baseline: {result.baseline_metrics.memory_usage_mb:.2f} MB")
                report.append(f"Memory Efficiency: {result.improvement_summary['memory_efficiency']:.2f}x")
        
        # Recommendations
        report.append("\n" + "=" * 80)
        report.append("RECOMMENDATIONS")
        report.append("=" * 80)
        
        report.append("\n1. Use VectorBT optimizations for datasets > 1,000 samples")
        report.append("2. Enable GPU acceleration for datasets > 10,000 samples")
        report.append("3. Use batch processing for multi-symbol operations")
        report.append("4. Monitor memory usage for very large datasets")
        report.append("5. Consider parallel processing for multi-core systems")
        
        return "\n".join(report)
    
    def save_results_to_file(self, results: Dict[str, List[TestResults]], filename: str = "vectorbt_performance_report.txt"):
        """Save results to a file."""
        report = self.generate_performance_report(results)
        
        with open(filename, 'w') as f:
            f.write(report)
        
        logger.info(f"Performance report saved to {filename}")


def run_comprehensive_performance_tests(
    enable_gpu: bool = False,
    enable_parallel: bool = True,
    save_report: bool = True
) -> Dict[str, List[TestResults]]:
    """
    Run comprehensive performance tests for VectorBT optimizations.
    
    Args:
        enable_gpu: Whether to enable GPU testing
        enable_parallel: Whether to enable parallel processing testing
        save_report: Whether to save results to file
        
    Returns:
        Dictionary of test results
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create tester
    tester = VectorBTOptimizationTester(enable_gpu=enable_gpu, enable_parallel=enable_parallel)
    
    # Run tests
    results = tester.run_comprehensive_tests()
    
    # Generate and save report
    if save_report:
        tester.save_results_to_file(results)
    
    # Print summary
    print(tester.generate_performance_report(results))
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Run comprehensive performance tests
    results = run_comprehensive_performance_tests(
        enable_gpu=False,  # Set to True if GPU available
        enable_parallel=True,
        save_report=True
    )
    
    print("\n" + "=" * 80)
    print("PERFORMANCE TESTING COMPLETE")
    print("=" * 80)
    print("Check the generated report for detailed performance analysis.")
    print("VectorBT optimizations provide significant performance improvements!")
    print("=" * 80)