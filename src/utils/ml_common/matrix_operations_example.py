#!/usr/bin/env python3
"""
Example usage of M1-Enhanced Matrix Operations.

This script demonstrates how to use the improved matrix operations with M1 optimization
integration, showcasing GPU acceleration, memory optimization, and parallel processing.
"""

import numpy as np
import pandas as pd
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_basic_operations():
    """Demonstrate basic M1-optimized matrix operations."""
    logger.info("🔧 Demonstrating Basic M1-Enhanced Matrix Operations")
    
    try:
        from matrix_operations_improved import (
            get_enhanced_matrix_operations, 
            m1_matrix_multiply, 
            m1_correlation_matrix,
            m1_eigendecomposition,
            m1_svd_decomposition
        )
        
        # Get the enhanced operations instance
        ops = get_enhanced_matrix_operations()
        
        # Create sample matrices
        np.random.seed(42)
        A = np.random.randn(1000, 1000).astype(np.float32)
        B = np.random.randn(1000, 1000).astype(np.float32)
        
        logger.info(f"Created matrices A: {A.shape}, B: {B.shape}")
        
        # 1. Matrix Multiplication
        logger.info("🚀 Testing M1-optimized matrix multiplication...")
        start_time = time.time()
        result = m1_matrix_multiply(A, B)
        multiplication_time = time.time() - start_time
        logger.info(f"✅ Matrix multiplication completed in {multiplication_time:.3f}s")
        logger.info(f"Result shape: {result.shape}")
        
        # 2. Correlation Matrix
        logger.info("📊 Testing M1-optimized correlation matrix...")
        # Create a sample DataFrame
        data = pd.DataFrame(np.random.randn(1000, 50))
        start_time = time.time()
        corr_matrix = m1_correlation_matrix(data)
        correlation_time = time.time() - start_time
        logger.info(f"✅ Correlation matrix completed in {correlation_time:.3f}s")
        logger.info(f"Correlation matrix shape: {corr_matrix.shape}")
        
        # 3. Eigendecomposition
        logger.info("🔍 Testing M1-optimized eigendecomposition...")
        # Create a symmetric matrix for eigendecomposition
        symmetric_matrix = A @ A.T
        start_time = time.time()
        eigenvalues, eigenvectors = m1_eigendecomposition(symmetric_matrix)
        eigen_time = time.time() - start_time
        logger.info(f"✅ Eigendecomposition completed in {eigen_time:.3f}s")
        logger.info(f"Eigenvalues shape: {eigenvalues.shape}, Eigenvectors shape: {eigenvectors.shape}")
        
        # 4. SVD Decomposition
        logger.info("📐 Testing M1-optimized SVD decomposition...")
        start_time = time.time()
        U, S, V = m1_svd_decomposition(A, k=100)  # Truncated SVD
        svd_time = time.time() - start_time
        logger.info(f"✅ SVD decomposition completed in {svd_time:.3f}s")
        logger.info(f"U shape: {U.shape}, S shape: {S.shape}, V shape: {V.shape}")
        
        return {
            'multiplication_time': multiplication_time,
            'correlation_time': correlation_time,
            'eigen_time': eigen_time,
            'svd_time': svd_time
        }
        
    except ImportError as e:
        logger.error(f"Failed to import M1-enhanced matrix operations: {e}")
        return None

def demonstrate_batch_operations():
    """Demonstrate batch matrix operations with parallel processing."""
    logger.info("🔄 Demonstrating Batch Matrix Operations")
    
    try:
        from matrix_operations_improved import (
            get_enhanced_matrix_operations,
            m1_parallel_operations
        )
        
        ops = get_enhanced_matrix_operations()
        
        # Create multiple matrices for batch processing
        np.random.seed(42)
        matrices = []
        for i in range(10):
            matrix = np.random.randn(500, 500).astype(np.float32)
            matrices.append(matrix)
        
        logger.info(f"Created {len(matrices)} matrices for batch processing")
        
        # Test parallel eigendecomposition
        logger.info("⚡ Testing parallel eigendecomposition...")
        start_time = time.time()
        eigen_results = m1_parallel_operations(matrices, operation="eigen")
        parallel_eigen_time = time.time() - start_time
        logger.info(f"✅ Parallel eigendecomposition completed in {parallel_eigen_time:.3f}s")
        logger.info(f"Processed {len(eigen_results)} matrices")
        
        # Test parallel SVD
        logger.info("📐 Testing parallel SVD decomposition...")
        start_time = time.time()
        svd_results = m1_parallel_operations(matrices, operation="svd")
        parallel_svd_time = time.time() - start_time
        logger.info(f"✅ Parallel SVD decomposition completed in {parallel_svd_time:.3f}s")
        logger.info(f"Processed {len(svd_results)} matrices")
        
        return {
            'parallel_eigen_time': parallel_eigen_time,
            'parallel_svd_time': parallel_svd_time,
            'matrices_processed': len(matrices)
        }
        
    except ImportError as e:
        logger.error(f"Failed to import M1-enhanced matrix operations: {e}")
        return None

def demonstrate_memory_optimization():
    """Demonstrate memory optimization features."""
    logger.info("🧠 Demonstrating Memory Optimization")
    
    try:
        from matrix_operations_improved import (
            get_enhanced_matrix_operations,
            m1_optimize_memory,
            get_m1_performance_stats
        )
        
        ops = get_enhanced_matrix_operations()
        
        # Get initial memory stats
        initial_stats = get_m1_performance_stats()
        logger.info("📊 Initial performance stats:")
        logger.info(f"  GPU enabled: {initial_stats.get('gpu_enabled', False)}")
        logger.info(f"  Memory optimization enabled: {initial_stats.get('memory_optimization_enabled', False)}")
        logger.info(f"  Parallel processing enabled: {initial_stats.get('parallel_processing_enabled', False)}")
        
        # Perform some operations to generate memory usage
        np.random.seed(42)
        large_matrices = []
        for i in range(5):
            matrix = np.random.randn(2000, 2000).astype(np.float32)
            large_matrices.append(matrix)
        
        logger.info(f"Created {len(large_matrices)} large matrices")
        
        # Perform operations
        results = []
        for i, matrix in enumerate(large_matrices):
            logger.info(f"Processing matrix {i+1}/{len(large_matrices)}")
            result = ops.matrix_multiply(matrix, matrix.T)
            results.append(result)
        
        # Optimize memory
        logger.info("🧹 Performing memory optimization...")
        start_time = time.time()
        memory_optimization_result = m1_optimize_memory()
        optimization_time = time.time() - start_time
        logger.info(f"✅ Memory optimization completed in {optimization_time:.3f}s")
        logger.info(f"Memory optimization result: {memory_optimization_result}")
        
        # Get final performance stats
        final_stats = get_m1_performance_stats()
        logger.info("📊 Final performance stats:")
        logger.info(f"  Total operations: {final_stats['m1_enhanced_operations']['total_operations']}")
        logger.info(f"  GPU operations: {final_stats['m1_enhanced_operations']['gpu_operations']}")
        logger.info(f"  CPU operations: {final_stats['m1_enhanced_operations']['cpu_operations']}")
        logger.info(f"  Memory optimizations: {final_stats['m1_enhanced_operations']['memory_optimizations']}")
        logger.info(f"  Average execution time: {final_stats['m1_enhanced_operations']['average_execution_time']:.4f}s")
        
        return {
            'optimization_time': optimization_time,
            'memory_optimization_result': memory_optimization_result,
            'final_stats': final_stats
        }
        
    except ImportError as e:
        logger.error(f"Failed to import M1-enhanced matrix operations: {e}")
        return None

def demonstrate_vectorized_processing():
    """Demonstrate vectorized processing capabilities."""
    logger.info("🔄 Demonstrating Vectorized Processing")
    
    try:
        from matrix_operations_improved import get_enhanced_matrix_operations
        
        ops = get_enhanced_matrix_operations()
        
        # Create sample financial data
        np.random.seed(42)
        n_samples = 10000
        n_features = 100
        
        # Generate synthetic financial time series data
        data = pd.DataFrame()
        for i in range(n_features):
            # Generate random walk with drift
            returns = np.random.normal(0.001, 0.02, n_samples)
            prices = np.cumprod(1 + returns)
            data[f'asset_{i}'] = prices
        
        logger.info(f"Created financial dataset: {data.shape}")
        
        # Test vectorized correlation analysis
        logger.info("📊 Testing vectorized correlation analysis...")
        start_time = time.time()
        corr_matrix = ops.correlation_matrix(data, method='pearson')
        correlation_time = time.time() - start_time
        logger.info(f"✅ Vectorized correlation analysis completed in {correlation_time:.3f}s")
        logger.info(f"Correlation matrix shape: {corr_matrix.shape}")
        
        # Test with different correlation methods
        for method in ['spearman', 'kendall']:
            logger.info(f"📈 Testing {method} correlation...")
            start_time = time.time()
            corr_matrix = ops.correlation_matrix(data, method=method)
            method_time = time.time() - start_time
            logger.info(f"✅ {method.capitalize()} correlation completed in {method_time:.3f}s")
        
        return {
            'correlation_time': correlation_time,
            'dataset_shape': data.shape
        }
        
    except ImportError as e:
        logger.error(f"Failed to import M1-enhanced matrix operations: {e}")
        return None

def compare_with_baseline():
    """Compare M1-optimized operations with baseline NumPy operations."""
    logger.info("⚖️ Comparing M1-optimized vs Baseline Operations")
    
    try:
        from matrix_operations_improved import m1_matrix_multiply
        
        # Create test matrices
        np.random.seed(42)
        A = np.random.randn(1000, 1000).astype(np.float32)
        B = np.random.randn(1000, 1000).astype(np.float32)
        
        # Baseline NumPy operation
        logger.info("🐌 Testing baseline NumPy matrix multiplication...")
        start_time = time.time()
        numpy_result = np.dot(A, B)
        numpy_time = time.time() - start_time
        logger.info(f"✅ NumPy matrix multiplication completed in {numpy_time:.3f}s")
        
        # M1-optimized operation
        logger.info("🚀 Testing M1-optimized matrix multiplication...")
        start_time = time.time()
        m1_result = m1_matrix_multiply(A, B)
        m1_time = time.time() - start_time
        logger.info(f"✅ M1-optimized matrix multiplication completed in {m1_time:.3f}s")
        
        # Verify results are equivalent
        if np.allclose(numpy_result, m1_result, rtol=1e-5):
            logger.info("✅ Results are numerically equivalent")
        else:
            logger.warning("⚠️ Results differ - this might indicate precision differences")
        
        # Calculate speedup
        speedup = numpy_time / m1_time if m1_time > 0 else 0
        logger.info(f"📈 Speedup: {speedup:.2f}x")
        
        return {
            'numpy_time': numpy_time,
            'm1_time': m1_time,
            'speedup': speedup,
            'results_equivalent': np.allclose(numpy_result, m1_result, rtol=1e-5)
        }
        
    except ImportError as e:
        logger.error(f"Failed to import M1-enhanced matrix operations: {e}")
        return None

def main():
    """Run all demonstrations."""
    logger.info("🎯 Starting M1-Enhanced Matrix Operations Demonstration")
    logger.info("=" * 60)
    
    results = {}
    
    # Run demonstrations
    results['basic_operations'] = demonstrate_basic_operations()
    results['batch_operations'] = demonstrate_batch_operations()
    results['memory_optimization'] = demonstrate_memory_optimization()
    results['vectorized_processing'] = demonstrate_vectorized_processing()
    results['baseline_comparison'] = compare_with_baseline()
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 Demonstration Summary")
    logger.info("=" * 60)
    
    for demo_name, demo_results in results.items():
        if demo_results:
            logger.info(f"✅ {demo_name.replace('_', ' ').title()}: Completed successfully")
        else:
            logger.warning(f"❌ {demo_name.replace('_', ' ').title()}: Failed or skipped")
    
    logger.info("🎉 M1-Enhanced Matrix Operations demonstration completed!")

if __name__ == "__main__":
    main()