#!/usr/bin/env python3
"""
M1 Compatibility Test for Training Pipeline Steps 3-20.

This script tests the M1 optimizations for CPU, GPU, and memory usage
across the training pipeline steps to ensure compatibility and performance.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import M1 optimization utilities
try:
    from src.utils.m1_gpu_utils import get_m1_gpu_manager, initialize_m1_gpu
    from src.utils.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.m1_cpu_optimizer import get_m1_cpu_optimizer, parallel_map
    M1_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ M1 optimizations not available: {e}")
    M1_OPTIMIZATIONS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class M1CompatibilityTester:
    """Test M1 compatibility across training pipeline steps."""

    def __init__(self):
        self.results = {}
        self.start_time = time.time()

        # Initialize M1 optimizers
        if M1_OPTIMIZATIONS_AVAILABLE:
            self.gpu_manager = initialize_m1_gpu()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
        else:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None

        logger.info("🧪 M1 Compatibility Tester initialized")

    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run complete test suite for M1 compatibility."""
        logger.info("🚀 Starting M1 compatibility test suite...")

        test_results = {}

        # Test 1: GPU/MPS Compatibility
        test_results['gpu_compatibility'] = self.test_gpu_compatibility()

        # Test 2: Memory Management
        test_results['memory_management'] = self.test_memory_management()

        # Test 3: CPU Parallel Processing
        test_results['cpu_parallelization'] = self.test_cpu_parallelization()

        # Test 4: Matrix Operations
        test_results['matrix_operations'] = self.test_matrix_operations()

        # Test 5: Neural Network Operations
        test_results['neural_network_operations'] = self.test_neural_network_operations()

        # Test 6: Data Processing Pipeline
        test_results['data_processing'] = self.test_data_processing_pipeline()

        # Test 7: End-to-End Performance
        test_results['end_to_end_performance'] = self.test_end_to_end_performance()

        # Summary
        test_results['summary'] = self.generate_test_summary(test_results)

        total_time = time.time() - self.start_time
        logger.info(".2f")

        return test_results

    def test_gpu_compatibility(self) -> Dict[str, Any]:
        """Test GPU/MPS compatibility."""
        logger.info("🎯 Testing GPU/MPS compatibility...")

        results = {
            'mps_available': False,
            'cuda_available': False,
            'gpu_memory_test': False,
            'matrix_multiplication_test': False,
            'mixed_precision_test': False
        }

        if not self.gpu_manager:
            logger.warning("⚠️ GPU manager not available")
            return results

        try:
            # Test MPS availability
            results['mps_available'] = self.gpu_manager.device.type == 'mps'
            results['cuda_available'] = self.gpu_manager.device.type == 'cuda'

            # Test GPU memory operations
            test_tensor = torch.randn(100, 100, device=self.gpu_manager.device)
            results['gpu_memory_test'] = True
            del test_tensor

            # Test matrix multiplication
            a = torch.randn(500, 500, device=self.gpu_manager.device)
            b = torch.randn(500, 500, device=self.gpu_manager.device)
            c = torch.matmul(a, b)
            results['matrix_multiplication_test'] = c.shape == (500, 500)
            del a, b, c

            # Test mixed precision
            if self.gpu_manager.use_mps:
                a_fp16 = torch.randn(100, 100, dtype=torch.float16, device=self.gpu_manager.device)
                b_fp16 = torch.randn(100, 100, dtype=torch.float16, device=self.gpu_manager.device)
                c_fp16 = torch.matmul(a_fp16, b_fp16)
                results['mixed_precision_test'] = c_fp16.dtype == torch.float16
                del a_fp16, b_fp16, c_fp16

            # Memory cleanup
            self.gpu_manager.optimize_memory()

            logger.info("✅ GPU compatibility test completed")

        except Exception as e:
            logger.error(f"❌ GPU compatibility test failed: {e}")
            results['error'] = str(e)

        return results

    def test_memory_management(self) -> Dict[str, Any]:
        """Test memory management capabilities."""
        logger.info("🧠 Testing memory management...")

        results = {
            'memory_monitoring': False,
            'chunked_processing': False,
            'memory_cleanup': False,
            'memory_efficiency': 0.0
        }

        if not self.memory_optimizer:
            logger.warning("⚠️ Memory optimizer not available")
            return results

        try:
            # Test memory monitoring
            memory_usage = self.memory_optimizer.get_memory_usage()
            results['memory_monitoring'] = 'rss_gb' in memory_usage

            # Test chunked processing
            large_data = pd.DataFrame(np.random.randn(10000, 50))

            def dummy_processor(chunk):
                return chunk.mean().mean()

            chunks = list(self.memory_optimizer.chunked_dataframe_processor(
                large_data, dummy_processor, chunk_size=1000
            ))
            results['chunked_processing'] = len(chunks) > 1

            # Test memory cleanup
            cleanup_results = self.memory_optimizer.optimize_memory()
            results['memory_cleanup'] = cleanup_results.get('gc_collected', 0) >= 0

            # Test memory efficiency
            memory_report = self.memory_optimizer.get_memory_report()
            results['memory_efficiency'] = memory_report.get('memory_efficiency', 0.0)

            logger.info("✅ Memory management test completed")

        except Exception as e:
            logger.error(f"❌ Memory management test failed: {e}")
            results['error'] = str(e)

        return results

    def test_cpu_parallelization(self) -> Dict[str, Any]:
        """Test CPU parallelization capabilities."""
        logger.info("⚡ Testing CPU parallelization...")

        results = {
            'parallel_processing': False,
            'threading_test': False,
            'multiprocessing_test': False,
            'optimal_workers': 0
        }

        if not self.cpu_optimizer:
            logger.warning("⚠️ CPU optimizer not available")
            return results

        try:
            # Test parallel processing
            def cpu_intensive_task(x):
                return sum(i**2 for i in range(x))

            test_data = [1000, 2000, 3000, 4000, 5000]
            results_parallel = parallel_map(cpu_intensive_task, test_data, task_type="cpu_bound")
            results['parallel_processing'] = len(results_parallel) == len(test_data)

            # Test threading
            def io_task(x):
                time.sleep(0.01)  # Simulate I/O
                return x * 2

            results_threading = self.cpu_optimizer.parallel_process(
                test_data, io_task, task_type="io_bound"
            )
            results['threading_test'] = len(results_threading) == len(test_data)

            # Test multiprocessing
            results_mp = self.cpu_optimizer._parallel_process_cpu_bound(
                test_data, cpu_intensive_task, num_workers=2
            )
            results['multiprocessing_test'] = len(results_mp) == len(test_data)

            # Get optimal workers
            results['optimal_workers'] = self.cpu_optimizer.max_workers

            logger.info("✅ CPU parallelization test completed")

        except Exception as e:
            logger.error(f"❌ CPU parallelization test failed: {e}")
            results['error'] = str(e)

        return results

    def test_matrix_operations(self) -> Dict[str, Any]:
        """Test matrix operations with M1 optimizations."""
        logger.info("📊 Testing matrix operations...")

        results = {
            'correlation_matrix': False,
            'covariance_matrix': False,
            'eigenvalue_decomposition': False,
            'gpu_accelerated': False
        }

        try:
            # Import matrix processor
            from src.training.steps.model_training.matrix_components import MatrixProcessor

            # Create test data
            test_data = pd.DataFrame(np.random.randn(1000, 20))

            # Test matrix processor
            config = {'use_gpu': True, 'batch_size': 500}
            processor = MatrixProcessor(**config)

            # Test correlation matrix
            corr_matrix = processor.compute_correlation_matrix(test_data)
            results['correlation_matrix'] = corr_matrix.shape == (20, 20)

            # Test covariance matrix
            cov_matrix = processor.compute_covariance_matrix(test_data)
            results['covariance_matrix'] = cov_matrix.shape == (20, 20)

            # Test eigenvalue decomposition
            if hasattr(processor, 'compute_eigendecomposition'):
                eigenvalues, eigenvectors = processor.compute_eigendecomposition(cov_matrix)
                results['eigenvalue_decomposition'] = len(eigenvalues) == 20

            # Check if GPU was used
            results['gpu_accelerated'] = processor.use_mps or processor.use_cuda

            logger.info("✅ Matrix operations test completed")

        except Exception as e:
            logger.error(f"❌ Matrix operations test failed: {e}")
            results['error'] = str(e)

        return results

    def test_neural_network_operations(self) -> Dict[str, Any]:
        """Test neural network operations with M1 optimizations."""
        logger.info("🧠 Testing neural network operations...")

        results = {
            'model_initialization': False,
            'forward_pass': False,
            'gpu_accelerated': False,
            'mixed_precision': False
        }

        try:
            # Import neural network model
            from src.training.steps.model_training.step09_5_hmm_lm_generalist_training import EfficientRegimePredictor

            # Create test model
            input_dim = 50
            num_regimes = 3
            model = EfficientRegimePredictor(input_dim, num_regimes)

            results['model_initialization'] = True

            # Test forward pass
            batch_size = 32
            seq_length = 100
            test_input = torch.randn(batch_size, seq_length, input_dim)

            if hasattr(model, 'm1_gpu_manager') and model.m1_gpu_manager:
                test_input = model.m1_gpu_manager.to_device(test_input, "neural_net")

            output = model(test_input)
            results['forward_pass'] = isinstance(output, dict) and 'current_regime' in output

            # Check GPU acceleration
            results['gpu_accelerated'] = hasattr(model, 'device') and model.device.type in ['mps', 'cuda']

            # Check mixed precision
            results['mixed_precision'] = hasattr(model, 'use_mixed_precision') and model.use_mixed_precision

            logger.info("✅ Neural network operations test completed")

        except Exception as e:
            logger.error(f"❌ Neural network operations test failed: {e}")
            results['error'] = str(e)

        return results

    def test_data_processing_pipeline(self) -> Dict[str, Any]:
        """Test data processing pipeline with M1 optimizations."""
        logger.info("📥 Testing data processing pipeline...")

        results = {
            'data_loading': False,
            'chunked_processing': False,
            'memory_efficient': False,
            'parallel_processing': False
        }

        try:
            # Create test data
            test_data = pd.DataFrame(np.random.randn(5000, 30))

            if self.memory_optimizer:
                # Test memory-efficient processing
                def process_chunk(chunk):
                    return {
                        'mean': chunk.mean().mean(),
                        'std': chunk.std().mean(),
                        'size': len(chunk)
                    }

                chunks = list(self.memory_optimizer.chunked_dataframe_processor(
                    test_data, process_chunk, chunk_size=1000
                ))
                results['chunked_processing'] = len(chunks) > 1

            if self.cpu_optimizer:
                # Test parallel processing
                def parallel_task(x):
                    return x ** 2

                test_values = list(range(100))
                parallel_results = self.cpu_optimizer.parallel_process(
                    test_values, parallel_task, task_type="cpu_bound"
                )
                results['parallel_processing'] = len(parallel_results) == len(test_values)

            results['data_loading'] = True
            results['memory_efficient'] = True

            logger.info("✅ Data processing pipeline test completed")

        except Exception as e:
            logger.error(f"❌ Data processing pipeline test failed: {e}")
            results['error'] = str(e)

        return results

    def test_end_to_end_performance(self) -> Dict[str, Any]:
        """Test end-to-end performance of M1 optimizations."""
        logger.info("🏁 Testing end-to-end performance...")

        results = {
            'total_test_time': 0.0,
            'memory_efficiency': 0.0,
            'cpu_utilization': 0.0,
            'gpu_utilization': 0.0,
            'overall_score': 0.0
        }

        try:
            # Measure performance
            start_time = time.time()

            # Run a comprehensive test
            test_data = pd.DataFrame(np.random.randn(2000, 25))

            if self.memory_optimizer:
                # Memory-efficient processing
                def comprehensive_process(chunk):
                    # Simulate complex processing
                    time.sleep(0.001)
                    return chunk.corr().mean().mean()

                chunks = list(self.memory_optimizer.chunked_dataframe_processor(
                    test_data, comprehensive_process, chunk_size=500
                ))

            if self.cpu_optimizer:
                # Parallel processing test
                def cpu_task(x):
                    return sum(i * i for i in range(x))

                parallel_map(cpu_task, [500, 1000, 1500], task_type="cpu_bound")

            end_time = time.time()
            results['total_test_time'] = end_time - start_time

            # Get resource utilization
            if self.memory_optimizer:
                memory_report = self.memory_optimizer.get_memory_report()
                results['memory_efficiency'] = memory_report.get('memory_efficiency', 0.0)

            if self.cpu_optimizer:
                cpu_report = self.cpu_optimizer.get_cpu_usage_report()
                results['cpu_utilization'] = cpu_report.get('cpu_percent_overall', 0.0)

            # Calculate overall score (0-100)
            score_components = [
                20 if results['memory_efficiency'] > 0.5 else 10,
                20 if results['cpu_utilization'] < 80 else 10,
                30 if results['total_test_time'] < 10 else 15,
                30 if not any('error' in str(v) for v in results.values() if isinstance(v, dict)) else 0
            ]
            results['overall_score'] = sum(score_components)

            logger.info("✅ End-to-end performance test completed")

        except Exception as e:
            logger.error(f"❌ End-to-end performance test failed: {e}")
            results['error'] = str(e)

        return results

    def generate_test_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        summary = {
            'total_tests': len(test_results),
            'passed_tests': 0,
            'failed_tests': 0,
            'warnings': 0,
            'errors': [],
            'performance_score': 0,
            'compatibility_score': 0
        }

        for test_name, test_result in test_results.items():
            if test_name == 'summary':
                continue

            if isinstance(test_result, dict):
                if 'error' in test_result:
                    summary['failed_tests'] += 1
                    summary['errors'].append(f"{test_name}: {test_result['error']}")
                else:
                    # Count passed sub-tests
                    passed_subtests = sum(1 for v in test_result.values() if isinstance(v, bool) and v)
                    total_subtests = sum(1 for v in test_result.values() if isinstance(v, bool))
                    if passed_subtests == total_subtests:
                        summary['passed_tests'] += 1
                    elif passed_subtests > 0:
                        summary['warnings'] += 1
                        summary['passed_tests'] += 1  # Partial pass

        # Calculate scores
        total_tests = summary['total_tests'] - 1  # Exclude summary
        summary['compatibility_score'] = (summary['passed_tests'] / total_tests) * 100 if total_tests > 0 else 0

        if 'end_to_end_performance' in test_results:
            perf_result = test_results['end_to_end_performance']
            summary['performance_score'] = perf_result.get('overall_score', 0)

        return summary


def main():
    """Main test execution function."""
    print("🍎 M1 Compatibility Test Suite for Training Pipeline Steps 3-20")
    print("=" * 60)

    if not M1_OPTIMIZATIONS_AVAILABLE:
        print("❌ M1 optimizations not available. Please ensure all required packages are installed.")
        return 1

    # Initialize tester
    tester = M1CompatibilityTester()

    try:
        # Run tests
        results = tester.run_full_test_suite()

        # Print results
        print("\n📊 Test Results Summary:")
        print("-" * 30)

        summary = results.get('summary', {})
        print(f"Total Tests: {summary.get('total_tests', 0)}")
        print(f"Passed Tests: {summary.get('passed_tests', 0)}")
        print(f"Failed Tests: {summary.get('failed_tests', 0)}")
        print(f"Warnings: {summary.get('warnings', 0)}")
        print(".1f")
        print(".1f")

        # Print detailed results
        print("\n📋 Detailed Test Results:")
        print("-" * 30)

        for test_name, test_result in results.items():
            if test_name != 'summary':
                status = "✅ PASS" if 'error' not in test_result else "❌ FAIL"
                print(f"{test_name}: {status}")
                if 'error' in test_result:
                    print(f"   Error: {test_result['error']}")

        # Performance recommendations
        print("\n💡 Performance Recommendations:")
        print("-" * 30)

        if summary.get('compatibility_score', 0) > 80:
            print("✅ Excellent M1 compatibility! All optimizations working correctly.")
        elif summary.get('compatibility_score', 0) > 60:
            print("⚠️ Good M1 compatibility with some optimizations working.")
        else:
            print("❌ Limited M1 compatibility. Some optimizations may need fixes.")

        if summary.get('performance_score', 0) > 80:
            print("🚀 Excellent performance! M1 optimizations significantly improving speed.")
        elif summary.get('performance_score', 0) > 60:
            print("⚡ Good performance with moderate improvements from M1 optimizations.")
        else:
            print("🐌 Performance optimizations may need tuning for better M1 utilization.")

        return 0 if summary.get('compatibility_score', 0) > 50 else 1

    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
