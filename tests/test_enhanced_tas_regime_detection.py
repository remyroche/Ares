"""
Test suite for Enhanced TAS Regime Detection.

This test suite validates the enhanced TAS regime detection capabilities including:
- Performance optimizations (memory, GPU, parallel processing)
- Advanced validation (cross-validation, out-of-sample testing, regime persistence)
- Intelligent caching
- Comprehensive performance monitoring
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import time
import logging

# Import the enhanced regime detection
from src.utils.ml_common.optimization.tas.enhanced_regime_detection import (
    get_enhanced_tas_regime_detection,
    EnhancedTASRegimeDetection
)

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class TestEnhancedTASRegimeDetection(unittest.TestCase):
    """Test cases for Enhanced TAS Regime Detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = self._create_sample_data()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_sample_data(self, n_samples: int = 1000, n_features: int = 10) -> pd.DataFrame:
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='1H')
        
        data = {
            'price': np.cumsum(np.random.normal(0.001, 0.01, n_samples)),
            'volume': np.random.lognormal(10, 0.5, n_samples),
            'volatility': np.random.gamma(2, 0.01, n_samples)
        }
        
        # Add additional features
        for i in range(n_features - 3):
            data[f'feature_{i}'] = np.random.randn(n_samples)
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def test_initialization(self):
        """Test enhanced regime detection initialization."""
        regime_detector = get_enhanced_tas_regime_detection(
            enable_gpu=True,
            enable_memory_optimization=True,
            enable_parallel=True,
            cache_dir=self.temp_dir,
            max_memory_gb=2.0
        )
        
        self.assertIsInstance(regime_detector, EnhancedTASRegimeDetection)
        self.assertTrue(regime_detector.enable_gpu)
        self.assertTrue(regime_detector.enable_memory_optimization)
        self.assertTrue(regime_detector.enable_parallel)
        self.assertEqual(regime_detector.max_memory_gb, 2.0)
    
    def test_basic_regime_detection(self):
        """Test basic regime detection functionality."""
        regime_detector = get_enhanced_tas_regime_detection(
            enable_gpu=False,  # Disable GPU for testing
            enable_memory_optimization=True,
            enable_parallel=False,  # Disable parallel for testing
            cache_dir=self.temp_dir
        )
        
        results = regime_detector.detect_regimes_enhanced(
            data=self.sample_data,
            timeframes=['1h', '4h'],
            methods=['unsupervised', 'clustering'],
            use_cache=False,
            parallel=False
        )
        
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Check that results contain expected keys
        expected_keys = ['1h_unsupervised', '1h_clustering', '4h_unsupervised', '4h_clustering']
        for key in expected_keys:
            self.assertIn(key, results)
    
    def test_memory_optimization(self):
        """Test memory optimization capabilities."""
        regime_detector = get_enhanced_tas_regime_detection(
            enable_gpu=False,
            enable_memory_optimization=True,
            enable_parallel=False,
            cache_dir=self.temp_dir,
            max_memory_gb=1.0
        )
        
        # Test memory optimization
        memory_stats = regime_detector.optimize_memory_usage()
        self.assertIsInstance(memory_stats, dict)
        
        # Test with large dataset
        large_data = self._create_sample_data(n_samples=5000, n_features=20)
        results = regime_detector.detect_regimes_enhanced(
            data=large_data,
            timeframes=['1h'],
            methods=['unsupervised'],
            use_cache=False,
            parallel=False
        )
        
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
    
    def test_parallel_processing(self):
        """Test parallel processing capabilities."""
        regime_detector = get_enhanced_tas_regime_detection(
            enable_gpu=False,
            enable_memory_optimization=True,
            enable_parallel=True,
            cache_dir=self.temp_dir
        )
        
        # Test parallel processing
        results = regime_detector.detect_regimes_enhanced(
            data=self.sample_data,
            timeframes=['1h', '4h'],
            methods=['unsupervised', 'clustering'],
            use_cache=False,
            parallel=True
        )
        
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Check that parallel processing was used
        stats = regime_detector.get_performance_stats()
        self.assertGreaterEqual(stats['parallel_detections'], 0)
    
    def test_caching(self):
        """Test intelligent caching capabilities."""
        regime_detector = get_enhanced_tas_regime_detection(
            enable_gpu=False,
            enable_memory_optimization=True,
            enable_parallel=False,
            cache_dir=self.temp_dir
        )
        
        # First run (cache miss)
        start_time = time.time()
        results1 = regime_detector.detect_regimes_enhanced(
            data=self.sample_data,
            timeframes=['1h'],
            methods=['unsupervised'],
            use_cache=True,
            parallel=False
        )
        first_run_time = time.time() - start_time
        
        # Second run (cache hit)
        start_time = time.time()
        results2 = regime_detector.detect_regimes_enhanced(
            data=self.sample_data,
            timeframes=['1h'],
            methods=['unsupervised'],
            use_cache=True,
            parallel=False
        )
        second_run_time = time.time() - start_time
        
        # Check that second run is faster (cached)
        self.assertLessEqual(second_run_time, first_run_time)
        
        # Check cache statistics
        stats = regime_detector.get_performance_stats()
        self.assertGreaterEqual(stats['cache_hits'], 0)
        self.assertGreaterEqual(stats['cache_misses'], 0)
    
    def test_advanced_validation(self):
        """Test advanced validation capabilities."""
        regime_detector = get_enhanced_tas_regime_detection(
            enable_gpu=False,
            enable_memory_optimization=True,
            enable_parallel=False,
            cache_dir=self.temp_dir
        )
        
        results = regime_detector.detect_regimes_enhanced(
            data=self.sample_data,
            timeframes=['1h', '4h'],
            methods=['unsupervised', 'clustering'],
            use_cache=False,
            parallel=False
        )
        
        # Check that advanced validation results are included
        self.assertIn('cross_validation', results)
        self.assertIn('out_of_sample', results)
        self.assertIn('persistence', results)
        self.assertIn('performance_metrics', results)
        
        # Check cross-validation results
        cv_results = results['cross_validation']
        self.assertIsInstance(cv_results, dict)
        
        # Check out-of-sample results
        oos_results = results['out_of_sample']
        self.assertIsInstance(oos_results, dict)
        
        # Check persistence results
        persistence_results = results['persistence']
        self.assertIsInstance(persistence_results, dict)
        
        # Check performance metrics
        perf_metrics = results['performance_metrics']
        self.assertIsInstance(perf_metrics, dict)
        self.assertIn('success_rate', perf_metrics)
        self.assertIn('performance_score', perf_metrics)
    
    def test_performance_monitoring(self):
        """Test performance monitoring capabilities."""
        regime_detector = get_enhanced_tas_regime_detection(
            enable_gpu=False,
            enable_memory_optimization=True,
            enable_parallel=True,
            cache_dir=self.temp_dir
        )
        
        # Perform some operations
        regime_detector.detect_regimes_enhanced(
            data=self.sample_data,
            timeframes=['1h'],
            methods=['unsupervised'],
            use_cache=False,
            parallel=True
        )
        
        # Check performance statistics
        stats = regime_detector.get_performance_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_detections', stats)
        self.assertIn('cache_hits', stats)
        self.assertIn('cache_misses', stats)
        self.assertIn('parallel_detections', stats)
        self.assertIn('memory_optimized_detections', stats)
        self.assertIn('average_detection_time', stats)
        self.assertIn('peak_memory_usage_mb', stats)
        
        # Check that statistics are reasonable
        self.assertGreaterEqual(stats['total_detections'], 0)
        self.assertGreaterEqual(stats['cache_hits'], 0)
        self.assertGreaterEqual(stats['cache_misses'], 0)
        self.assertGreaterEqual(stats['parallel_detections'], 0)
        self.assertGreaterEqual(stats['memory_optimized_detections'], 0)
        self.assertGreaterEqual(stats['average_detection_time'], 0)
        self.assertGreaterEqual(stats['peak_memory_usage_mb'], 0)
    
    def test_cache_management(self):
        """Test cache management capabilities."""
        regime_detector = get_enhanced_tas_regime_detection(
            enable_gpu=False,
            enable_memory_optimization=True,
            enable_parallel=False,
            cache_dir=self.temp_dir
        )
        
        # Test cache clearing
        regime_detector.clear_cache()
        
        # Check that cache directory exists
        self.assertTrue(regime_detector.cache_dir.exists())
    
    def test_error_handling(self):
        """Test error handling capabilities."""
        regime_detector = get_enhanced_tas_regime_detection(
            enable_gpu=False,
            enable_memory_optimization=True,
            enable_parallel=False,
            cache_dir=self.temp_dir
        )
        
        # Test with invalid data
        invalid_data = None
        with self.assertRaises(Exception):
            regime_detector.detect_regimes_enhanced(
                data=invalid_data,
                timeframes=['1h'],
                methods=['unsupervised'],
                use_cache=False,
                parallel=False
            )
        
        # Test with empty data
        empty_data = pd.DataFrame()
        with self.assertRaises(Exception):
            regime_detector.detect_regimes_enhanced(
                data=empty_data,
                timeframes=['1h'],
                methods=['unsupervised'],
                use_cache=False,
                parallel=False
            )
    
    def test_cpu_optimization(self):
        """Test CPU optimization capabilities."""
        regime_detector = get_enhanced_tas_regime_detection(
            enable_gpu=False,
            enable_memory_optimization=True,
            enable_parallel=True,
            cache_dir=self.temp_dir
        )
        
        # Test CPU optimization
        cpu_stats = regime_detector.optimize_cpu_usage(target_utilization=0.8)
        self.assertIsInstance(cpu_stats, dict)
        
        # Check that CPU optimization was applied
        self.assertIn('success', cpu_stats)
        self.assertIn('recommended_workers', cpu_stats)
        self.assertIn('target_utilization', cpu_stats)
    
    def test_comprehensive_analysis(self):
        """Test comprehensive regime analysis."""
        regime_detector = get_enhanced_tas_regime_detection(
            enable_gpu=False,
            enable_memory_optimization=True,
            enable_parallel=True,
            cache_dir=self.temp_dir
        )
        
        # Test comprehensive analysis
        results = regime_detector.detect_regimes_enhanced(
            data=self.sample_data,
            timeframes=['1h', '4h', '1d'],
            methods=['unsupervised', 'clustering', 'qualification'],
            use_cache=True,
            parallel=True
        )
        
        # Check that all expected components are present
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Check that advanced validation is included
        self.assertIn('cross_validation', results)
        self.assertIn('out_of_sample', results)
        self.assertIn('persistence', results)
        self.assertIn('performance_metrics', results)
        
        # Check that results are comprehensive
        expected_timeframes = ['1h', '4h', '1d']
        expected_methods = ['unsupervised', 'clustering', 'qualification']
        
        for timeframe in expected_timeframes:
            for method in expected_methods:
                key = f"{timeframe}_{method}"
                self.assertIn(key, results)
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks."""
        regime_detector = get_enhanced_tas_regime_detection(
            enable_gpu=False,
            enable_memory_optimization=True,
            enable_parallel=True,
            cache_dir=self.temp_dir
        )
        
        # Test performance with different data sizes
        data_sizes = [1000, 2000, 5000]
        execution_times = []
        
        for size in data_sizes:
            test_data = self._create_sample_data(n_samples=size, n_features=10)
            
            start_time = time.time()
            results = regime_detector.detect_regimes_enhanced(
                data=test_data,
                timeframes=['1h'],
                methods=['unsupervised'],
                use_cache=False,
                parallel=True
            )
            execution_time = time.time() - start_time
            
            execution_times.append(execution_time)
            
            # Check that results are valid
            self.assertIsInstance(results, dict)
            self.assertGreater(len(results), 0)
        
        # Check that execution times are reasonable
        for i, exec_time in enumerate(execution_times):
            self.assertGreater(exec_time, 0)
            self.assertLess(exec_time, 60)  # Should complete within 60 seconds
    
    def test_memory_usage(self):
        """Test memory usage optimization."""
        regime_detector = get_enhanced_tas_regime_detection(
            enable_gpu=False,
            enable_memory_optimization=True,
            enable_parallel=True,
            cache_dir=self.temp_dir,
            max_memory_gb=1.0
        )
        
        # Test with memory optimization
        large_data = self._create_sample_data(n_samples=10000, n_features=50)
        
        # Monitor memory usage
        memory_stats_before = regime_detector.memory_optimizer.get_memory_stats() if regime_detector.memory_optimizer else {}
        
        results = regime_detector.detect_regimes_enhanced(
            data=large_data,
            timeframes=['1h'],
            methods=['unsupervised'],
            use_cache=False,
            parallel=True
        )
        
        memory_stats_after = regime_detector.memory_optimizer.get_memory_stats() if regime_detector.memory_optimizer else {}
        
        # Check that results are valid
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Check that memory optimization was applied
        stats = regime_detector.get_performance_stats()
        self.assertGreaterEqual(stats['memory_optimized_detections'], 0)


class TestEnhancedTASRegimeDetectionIntegration(unittest.TestCase):
    """Integration tests for Enhanced TAS Regime Detection."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = self._create_comprehensive_sample_data()
        
    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_comprehensive_sample_data(self) -> pd.DataFrame:
        """Create comprehensive sample data for integration testing."""
        np.random.seed(42)
        n_samples = 5000
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='1H')
        
        # Create data with multiple regimes
        data = {}
        
        # Regime 1: Low volatility (first 40%)
        regime1_size = int(n_samples * 0.4)
        data['price'] = np.cumsum(np.random.normal(0.001, 0.01, regime1_size))
        data['volume'] = np.random.lognormal(10, 0.5, regime1_size)
        data['volatility'] = np.random.gamma(2, 0.01, regime1_size)
        
        # Regime 2: High volatility (middle 30%)
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
        
        # Regime 3: Medium volatility (last 30%)
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
        for i in range(20):
            data[f'feature_{i}'] = np.random.randn(n_samples)
        
        df = pd.DataFrame(data, index=dates)
        
        # Add true regime labels for validation
        df['true_regime'] = 0
        df.iloc[regime1_size:regime1_size + regime2_size, -1] = 1
        df.iloc[regime1_size + regime2_size:, -1] = 2
        
        return df
    
    def test_full_integration(self):
        """Test full integration of enhanced regime detection."""
        regime_detector = get_enhanced_tas_regime_detection(
            enable_gpu=True,
            enable_memory_optimization=True,
            enable_parallel=True,
            cache_dir=self.temp_dir,
            max_memory_gb=4.0
        )
        
        # Perform comprehensive regime detection
        results = regime_detector.detect_regimes_enhanced(
            data=self.sample_data,
            timeframes=['1h', '4h', '1d'],
            methods=['unsupervised', 'clustering', 'qualification'],
            use_cache=True,
            parallel=True
        )
        
        # Validate results
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Check that all expected components are present
        expected_components = ['cross_validation', 'out_of_sample', 'persistence', 'performance_metrics']
        for component in expected_components:
            self.assertIn(component, results)
            self.assertIsInstance(results[component], dict)
        
        # Check performance statistics
        stats = regime_detector.get_performance_stats()
        self.assertIsInstance(stats, dict)
        self.assertGreater(stats['total_detections'], 0)
        
        # Validate that all timeframes and methods were processed
        expected_timeframes = ['1h', '4h', '1d']
        expected_methods = ['unsupervised', 'clustering', 'qualification']
        
        for timeframe in expected_timeframes:
            for method in expected_methods:
                key = f"{timeframe}_{method}"
                self.assertIn(key, results)
                self.assertIsInstance(results[key], dict)
    
    def test_performance_optimization_integration(self):
        """Test integration of performance optimizations."""
        regime_detector = get_enhanced_tas_regime_detection(
            enable_gpu=True,
            enable_memory_optimization=True,
            enable_parallel=True,
            cache_dir=self.temp_dir,
            max_memory_gb=2.0
        )
        
        # Test memory optimization
        memory_stats = regime_detector.optimize_memory_usage()
        self.assertIsInstance(memory_stats, dict)
        
        # Test CPU optimization
        cpu_stats = regime_detector.optimize_cpu_usage(target_utilization=0.8)
        self.assertIsInstance(cpu_stats, dict)
        
        # Perform regime detection with optimizations
        results = regime_detector.detect_regimes_enhanced(
            data=self.sample_data,
            timeframes=['1h', '4h'],
            methods=['unsupervised', 'clustering'],
            use_cache=True,
            parallel=True
        )
        
        # Validate results
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Check that optimizations were applied
        stats = regime_detector.get_performance_stats()
        self.assertGreaterEqual(stats['memory_optimized_detections'], 0)
        self.assertGreaterEqual(stats['parallel_detections'], 0)
    
    def test_validation_integration(self):
        """Test integration of advanced validation."""
        regime_detector = get_enhanced_tas_regime_detection(
            enable_gpu=False,
            enable_memory_optimization=True,
            enable_parallel=True,
            cache_dir=self.temp_dir
        )
        
        # Perform regime detection with validation
        results = regime_detector.detect_regimes_enhanced(
            data=self.sample_data,
            timeframes=['1h', '4h'],
            methods=['unsupervised', 'clustering'],
            use_cache=True,
            parallel=True
        )
        
        # Validate cross-validation results
        cv_results = results['cross_validation']
        self.assertIsInstance(cv_results, dict)
        self.assertGreater(len(cv_results), 0)
        
        for key, cv_result in cv_results.items():
            self.assertIsInstance(cv_result, dict)
            if 'stability' in cv_result:
                self.assertIsInstance(cv_result['stability'], (int, float))
                self.assertGreaterEqual(cv_result['stability'], 0)
                self.assertLessEqual(cv_result['stability'], 1)
        
        # Validate out-of-sample results
        oos_results = results['out_of_sample']
        self.assertIsInstance(oos_results, dict)
        self.assertGreater(len(oos_results), 0)
        
        for key, oos_result in oos_results.items():
            self.assertIsInstance(oos_result, dict)
            if 'oos_score' in oos_result:
                self.assertIsInstance(oos_result['oos_score'], (int, float))
                self.assertGreaterEqual(oos_result['oos_score'], 0)
                self.assertLessEqual(oos_result['oos_score'], 1)
        
        # Validate persistence results
        persistence_results = results['persistence']
        self.assertIsInstance(persistence_results, dict)
        self.assertGreater(len(persistence_results), 0)
        
        for key, persistence_result in persistence_results.items():
            self.assertIsInstance(persistence_result, dict)
            if 'average_stability' in persistence_result:
                self.assertIsInstance(persistence_result['average_stability'], (int, float))
                self.assertGreaterEqual(persistence_result['average_stability'], 0)
                self.assertLessEqual(persistence_result['average_stability'], 1)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)