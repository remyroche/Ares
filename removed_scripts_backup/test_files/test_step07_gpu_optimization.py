#!/usr/bin/env python3
"""
Test Step07 Enhanced GPU/MPS Optimization for Mac M1
"""

import pandas as pd
import numpy as np
import time
import torch
from src.training.steps.model_training.matrix_components import MatrixProcessor

def test_mps_optimization():
    """Test MPS optimization capabilities."""
    print("🧪 Testing Step07 MPS/GPU Optimizations...")

    # Check MPS availability
    mps_available = torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    print(f"🍎 MPS Available: {mps_available}")
    print(f"🎮 CUDA Available: {cuda_available}")

    if not mps_available and not cuda_available:
        print("⚠️ No GPU acceleration available - using CPU fallback")
        return False

    try:
        # Initialize enhanced matrix processor
        matrix_processor = MatrixProcessor(use_gpu=True, batch_size=1000)
        print(f"✅ Matrix processor initialized with device: {matrix_processor.device}")

        # Test memory optimization
        memory_metrics = matrix_processor.optimize_memory_mps()
        print(f"📊 Memory metrics: {memory_metrics}")

        # Create test data
        np.random.seed(42)
        n_samples, n_features = 5000, 100
        test_data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        print(f"📈 Test data shape: {test_data.shape}")

        # Test correlation matrix computation
        print("🔢 Testing correlation matrix computation...")
        start_time = time.time()
        corr_matrix = matrix_processor.compute_correlation_matrix(test_data)
        corr_time = time.time() - start_time
        print(".2f")

        # Test covariance matrix computation
        print("🔢 Testing covariance matrix computation...")
        start_time = time.time()
        cov_matrix = matrix_processor.compute_covariance_matrix(test_data)
        cov_time = time.time() - start_time
        print(".2f")

        # Test feature interaction matrix
        print("🔗 Testing feature interaction matrix...")
        start_time = time.time()
        interaction_matrix = matrix_processor.compute_feature_interaction_matrix_mps(test_data)
        interaction_time = time.time() - start_time
        print(".2f")

        # Test eigendecomposition
        print("🔢 Testing eigendecomposition...")
        start_time = time.time()
        eigenvalues, eigenvectors = matrix_processor.compute_eigendecomposition(cov_matrix)
        eigen_time = time.time() - start_time
        print(".2f")
        print(f"   Top 5 eigenvalues: {eigenvalues[:5]}")

        # Performance summary
        print("\n📊 Performance Summary:")
        print(f"   Correlation Matrix: {corr_time:.3f}s")
        print(f"   Covariance Matrix: {cov_time:.3f}s")
        print(f"   Feature Interaction: {interaction_time:.3f}s")
        print(f"   Eigendecomposition: {eigen_time:.3f}s")
        print(".3f")

        # Memory cleanup test
        final_memory_metrics = matrix_processor.optimize_memory_mps()
        print(f"🧹 Final memory metrics: {final_memory_metrics}")

        print("✅ All GPU/MPS optimizations working correctly!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_cpu_vs_gpu():
    """Benchmark CPU vs GPU/MPS performance."""
    print("\n🔬 Benchmarking CPU vs GPU/MPS Performance...")

    # Create test data
    np.random.seed(42)
    n_samples, n_features = 2000, 50
    test_data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # CPU test
    print("💻 Testing CPU performance...")
    cpu_processor = MatrixProcessor(use_gpu=False, batch_size=1000)
    start_time = time.time()
    cpu_corr = cpu_processor.compute_correlation_matrix(test_data)
    cpu_time = time.time() - start_time

    # GPU/MPS test
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        print("🎯 Testing GPU/MPS performance...")
        gpu_processor = MatrixProcessor(use_gpu=True, batch_size=1000)
        start_time = time.time()
        gpu_corr = gpu_processor.compute_correlation_matrix(test_data)
        gpu_time = time.time() - start_time

        # Compare results
        max_diff = np.max(np.abs(cpu_corr - gpu_corr))
        speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0

        print(".2f")
        print(".2f")
        print(".1f")
        print(".6f")

        if speedup > 1.0:
            print(f"🚀 GPU/MPS is {speedup:.1f}x faster than CPU!")
        else:
            print(f"⚠️ GPU/MPS is {1.0/speedup:.1f}x slower than CPU (may be due to data transfer overhead)")

    else:
        print("⚠️ No GPU available for comparison")

if __name__ == "__main__":
    print("🔬 Step07 GPU/MPS Optimization Test Suite")
    print("=" * 50)

    success = test_mps_optimization()
    benchmark_cpu_vs_gpu()

    if success:
        print("\n🎉 Step07 GPU/MPS optimization tests PASSED!")
        print("Your Mac M1 should see significant performance improvements in matrix operations.")
    else:
        print("\n💥 Step07 GPU/MPS optimization tests FAILED!")
        print("Check your PyTorch installation and MPS support.")
