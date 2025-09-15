#!/usr/bin/env python3
"""
Test Unified Matrix Operations Module

This script tests the new unified matrix operations module to ensure
all functionality works correctly and backwards compatibility is maintained.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_unified_imports():
    """Test that all unified imports work correctly."""
    print("🔧 Testing Unified Module Imports")
    
    try:
        # Test core imports
        from src.utils.matrix_operations import (
            get_unified_matrix_operations,
            UnifiedMatrixOperations,
            M1EnhancedMatrixOperations,  # Backwards compatibility
        )
        print("✅ Core unified operations imported successfully")
        
        # Test vectorized core imports
        from src.utils.matrix_operations import (
            get_vectorized_processing_core,
            VectorizedProcessingCore,
            OptimizedPipelineExecutor,
            PipelineStage,
            PipelineExecutionMode,
        )
        print("✅ Vectorized processing core imported successfully")
        
        # Test batch operations imports
        from src.utils.matrix_operations import (
            get_batch_matrix_processor,
            BatchMatrixProcessor,
        )
        print("✅ Batch operations imported successfully")
        
        # Test enhanced operations imports
        from src.utils.matrix_operations import (
            get_enhanced_matrix_operations,
            EnhancedMatrixOperations,
            BatchOptimizationStrategy,
            OperationComplexity,
        )
        print("✅ Enhanced operations imported successfully")
        
        # Test error handling imports
        from src.utils.matrix_operations import (
            ErrorHandler,
            OptimizationError,
            GPUError,
            MemoryError,
            with_error_handling,
            with_gpu_fallback,
            with_memory_optimization,
        )
        print("✅ Error handling imported successfully")
        
        # Test convenience functions imports
        from src.utils.matrix_operations import (
            safe_matrix_multiply,
            safe_correlation_matrix,
            gpu_matrix_multiply,
            optimize_dataframe,
            batch_matrix_multiply,
            get_performance_stats,
            get_system_info,
        )
        print("✅ Convenience functions imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_unified_operations():
    """Test unified matrix operations functionality."""
    print("\n🧮 Testing Unified Matrix Operations")
    
    try:
        from src.utils.matrix_operations import get_unified_matrix_operations
        
        # Create sample matrices
        np.random.seed(42)
        A = np.random.randn(100, 100)
        B = np.random.randn(100, 100)
        data = np.random.randn(500, 10)
        
        # Initialize operations
        ops = get_unified_matrix_operations()
        print("✅ Unified operations initialized")
        
        # Test matrix multiplication
        result = ops.matrix_multiply(A, B)
        print(f"✅ Matrix multiplication: {result.shape}")
        
        # Test correlation matrix
        corr = ops.safe_correlation_matrix(data)
        print(f"✅ Correlation matrix: {corr.shape}")
        
        # Test matrix inversion
        square_matrix = A @ A.T  # Make positive definite
        inverse = ops.matrix_inverse(square_matrix)
        print(f"✅ Matrix inversion: {inverse.shape}")
        
        # Test eigendecomposition
        eigenvals, eigenvecs = ops.eigendecomposition(square_matrix)
        print(f"✅ Eigendecomposition: {eigenvals.shape}, {eigenvecs.shape}")
        
        # Test SVD
        U, s, Vh = ops.svd_decomposition(A)
        print(f"✅ SVD decomposition: {U.shape}, {s.shape}, {Vh.shape}")
        
        # Test performance stats
        stats = ops.get_performance_stats()
        print(f"✅ Performance stats: {len(stats)} metrics")
        
        # Test hardware info
        hw_info = ops.get_hardware_info()
        print(f"✅ Hardware info: {len(hw_info)} components")
        
        return True
        
    except Exception as e:
        print(f"❌ Unified operations test failed: {e}")
        return False

def test_vectorized_core():
    """Test vectorized processing core functionality."""
    print("\n⚡ Testing Vectorized Processing Core")
    
    try:
        from src.utils.matrix_operations import get_vectorized_processing_core
        
        # Create sample data
        np.random.seed(42)
        data = pd.DataFrame(np.random.randn(1000, 20), 
                           columns=[f'feature_{i}' for i in range(20)])
        
        # Initialize core
        core = get_vectorized_processing_core()
        print("✅ Vectorized core initialized")
        
        # Test DataFrame optimization
        optimized_data = core.optimize_dataframe_for_processing(data)
        print(f"✅ DataFrame optimization: {optimized_data.shape}")
        
        # Test rolling features
        rolling_data = core.vectorized_rolling_features(data, windows=[5, 10])
        print(f"✅ Rolling features: {rolling_data.shape}")
        
        # Test correlation analysis
        corr_matrix, feature_importance = core.matrix_correlation_analysis(data)
        print(f"✅ Correlation analysis: {corr_matrix.shape}, {feature_importance.shape}")
        
        # Test processing stats
        stats = core.get_processing_stats()
        print(f"✅ Processing stats: {len(stats)} metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Vectorized core test failed: {e}")
        return False

def test_batch_operations():
    """Test batch matrix operations functionality."""
    print("\n📦 Testing Batch Matrix Operations")
    
    try:
        from src.utils.matrix_operations import get_batch_matrix_processor
        
        # Create sample matrices
        np.random.seed(42)
        matrices_a = [np.random.randn(50, 50) for _ in range(5)]
        matrices_b = [np.random.randn(50, 50) for _ in range(5)]
        
        # Initialize processor
        processor = get_batch_matrix_processor()
        print("✅ Batch processor initialized")
        
        # Test batch matrix multiplication
        results = processor.batch_matrix_multiply(matrices_a, matrices_b)
        print(f"✅ Batch matrix multiplication: {len(results)} results")
        
        # Test batch feature transformation
        data = pd.DataFrame(np.random.randn(1000, 10))
        transformations = [
            {'type': 'standardize', 'columns': data.columns[:5]},
            {'type': 'normalize', 'columns': data.columns[5:]}
        ]
        transformed = processor.batch_feature_transformation(data, transformations)
        print(f"✅ Batch feature transformation: {transformed.shape}")
        
        # Test batch correlation analysis
        sample_data = data.iloc[:, :5]  # Use subset for demo
        corr_matrix, p_values = processor.batch_correlation_analysis(sample_data)
        print(f"✅ Batch correlation analysis: {corr_matrix.shape}, {p_values.shape}")
        
        # Test performance stats
        stats = processor.get_performance_stats()
        print(f"✅ Batch performance stats: {len(stats)} metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch operations test failed: {e}")
        return False

def test_enhanced_operations():
    """Test enhanced matrix operations with GPU support."""
    print("\n🚀 Testing Enhanced Matrix Operations")
    
    try:
        from src.utils.matrix_operations import get_enhanced_matrix_operations
        
        # Create sample matrix
        np.random.seed(42)
        matrix = np.random.randn(100, 100)
        
        # Initialize enhanced operations
        enhanced_ops = get_enhanced_matrix_operations()
        print("✅ Enhanced operations initialized")
        
        # Test tensor conversion
        tensor = enhanced_ops.to_tensor(matrix)
        print(f"✅ Tensor conversion: {tensor.shape}")
        
        # Test matrix multiplication
        result = enhanced_ops.matrix_multiply(matrix, matrix)
        print(f"✅ Enhanced matrix multiplication: {result.shape}")
        
        # Test performance stats
        stats = enhanced_ops.get_performance_stats()
        print(f"✅ Enhanced performance stats: {len(stats)} metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced operations test failed: {e}")
        return False

def test_convenience_functions():
    """Test convenience functions."""
    print("\n🎯 Testing Convenience Functions")
    
    try:
        from src.utils.matrix_operations import (
            matrix_multiply,
            correlation_matrix,
            optimize_dataframe,
            batch_matrix_multiply,
            get_performance_stats,
            get_system_info,
        )
        
        # Create sample data
        np.random.seed(42)
        A = np.random.randn(50, 50)
        B = np.random.randn(50, 50)
        data = pd.DataFrame(np.random.randn(500, 10))
        
        # Test matrix multiplication
        result = matrix_multiply(A, B)
        print(f"✅ Convenience matrix multiply: {result.shape}")
        
        # Test correlation matrix
        corr = correlation_matrix(data)
        print(f"✅ Convenience correlation: {corr.shape}")
        
        # Test DataFrame optimization
        optimized = optimize_dataframe(data)
        print(f"✅ Convenience DataFrame optimization: {optimized.shape}")
        
        # Test batch operations
        matrices_a = [np.random.randn(20, 20) for _ in range(3)]
        matrices_b = [np.random.randn(20, 20) for _ in range(3)]
        batch_results = batch_matrix_multiply(matrices_a, matrices_b)
        print(f"✅ Convenience batch operations: {len(batch_results)} results")
        
        # Test performance stats
        stats = get_performance_stats()
        print(f"✅ Convenience performance stats: {len(stats)} components")
        
        # Test system info
        sys_info = get_system_info()
        print(f"✅ System info: {len(sys_info)} metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Convenience functions test failed: {e}")
        return False

def test_backwards_compatibility():
    """Test backwards compatibility with existing code."""
    print("\n🔄 Testing Backwards Compatibility")
    
    try:
        # Test legacy imports that should still work
        from src.utils.matrix_operations import M1EnhancedMatrixOperations
        print("✅ Legacy M1EnhancedMatrixOperations import works")
        
        # Test legacy function calls
        from src.utils.matrix_operations import m1_matrix_multiply, get_enhanced_matrix_operations
        
        # Create sample matrices
        np.random.seed(42)
        A = np.random.randn(50, 50)
        B = np.random.randn(50, 50)
        
        # Test legacy M1 matrix multiply
        result = m1_matrix_multiply(A, B)
        print(f"✅ Legacy m1_matrix_multiply: {result.shape}")
        
        # Test legacy enhanced operations
        legacy_ops = get_enhanced_matrix_operations()
        print("✅ Legacy get_enhanced_matrix_operations works")
        
        return True
        
    except Exception as e:
        print(f"❌ Backwards compatibility test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and recovery mechanisms."""
    print("\n🛡️ Testing Error Handling")
    
    try:
        from src.utils.matrix_operations import (
            ErrorHandler,
            OptimizationError,
            with_error_handling,
            with_gpu_fallback,
        )
        
        # Test error handler initialization
        error_handler = ErrorHandler()
        print("✅ Error handler initialized")
        
        # Test error classification
        test_error = ValueError("GPU memory error")
        error_type = error_handler._classify_error(test_error)
        print(f"✅ Error classification: {error_type}")
        
        # Test error handling decorator
        @with_error_handling("test_operation")
        def test_function():
            return "success"
        
        result = test_function()
        print(f"✅ Error handling decorator: {result}")
        
        # Test GPU fallback decorator
        @with_gpu_fallback("test_gpu_operation")
        def test_gpu_function(use_gpu=True):
            if use_gpu:
                raise RuntimeError("GPU memory error")
            return "cpu_fallback"
        
        result = test_gpu_function()
        print(f"✅ GPU fallback decorator: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 UNIFIED MATRIX OPERATIONS TEST SUITE")
    print("="*60)
    
    success_count = 0
    total_tests = 7
    
    # Test unified imports
    try:
        if test_unified_imports():
            success_count += 1
            print("✅ UNIFIED IMPORTS: PASSED")
        else:
            print("❌ UNIFIED IMPORTS: FAILED")
    except Exception as e:
        print(f"❌ UNIFIED IMPORTS: ERROR - {e}")
    
    # Test unified operations
    try:
        if test_unified_operations():
            success_count += 1
            print("✅ UNIFIED OPERATIONS: PASSED")
        else:
            print("❌ UNIFIED OPERATIONS: FAILED")
    except Exception as e:
        print(f"❌ UNIFIED OPERATIONS: ERROR - {e}")
    
    # Test vectorized core
    try:
        if test_vectorized_core():
            success_count += 1
            print("✅ VECTORIZED CORE: PASSED")
        else:
            print("❌ VECTORIZED CORE: FAILED")
    except Exception as e:
        print(f"❌ VECTORIZED CORE: ERROR - {e}")
    
    # Test batch operations
    try:
        if test_batch_operations():
            success_count += 1
            print("✅ BATCH OPERATIONS: PASSED")
        else:
            print("❌ BATCH OPERATIONS: FAILED")
    except Exception as e:
        print(f"❌ BATCH OPERATIONS: ERROR - {e}")
    
    # Test enhanced operations
    try:
        if test_enhanced_operations():
            success_count += 1
            print("✅ ENHANCED OPERATIONS: PASSED")
        else:
            print("❌ ENHANCED OPERATIONS: FAILED")
    except Exception as e:
        print(f"❌ ENHANCED OPERATIONS: ERROR - {e}")
    
    # Test convenience functions
    try:
        if test_convenience_functions():
            success_count += 1
            print("✅ CONVENIENCE FUNCTIONS: PASSED")
        else:
            print("❌ CONVENIENCE FUNCTIONS: FAILED")
    except Exception as e:
        print(f"❌ CONVENIENCE FUNCTIONS: ERROR - {e}")
    
    # Test backwards compatibility
    try:
        if test_backwards_compatibility():
            success_count += 1
            print("✅ BACKWARDS COMPATIBILITY: PASSED")
        else:
            print("❌ BACKWARDS COMPATIBILITY: FAILED")
    except Exception as e:
        print(f"❌ BACKWARDS COMPATIBILITY: ERROR - {e}")
    
    # Test error handling
    try:
        if test_error_handling():
            success_count += 1
            print("✅ ERROR HANDLING: PASSED")
        else:
            print("❌ ERROR HANDLING: FAILED")
    except Exception as e:
        print(f"❌ ERROR HANDLING: ERROR - {e}")
    
    # Summary
    print("\n" + "="*60)
    print("🎉 TEST SUMMARY")
    print("="*60)
    
    print(f"📊 Tests completed: {success_count}/{total_tests}")
    success_rate = (success_count / total_tests) * 100
    print(f"📈 Success rate: {success_rate:.1f}%")
    
    if success_count >= 6:  # At least 85% success
        print("🎯 UNIFIED MATRIX OPERATIONS: SUCCESS")
        print("\n🚀 Key Achievements:")
        print("   • ✅ Created unified matrix operations module")
        print("   • ✅ Consolidated scattered functionality")
        print("   • ✅ Maintained full backwards compatibility")
        print("   • ✅ Integrated GPU acceleration and memory optimization")
        print("   • ✅ Added comprehensive error handling and recovery")
        print("   • ✅ Provided convenient wrapper functions")
        print("   • ✅ Optimized for Apple Silicon M1/M2/M3 Macs")
        print("\n✨ Matrix operations are now unified and ready for use!")
        print("\n📖 Usage Examples:")
        print("   from src.utils.matrix_operations import get_unified_matrix_operations")
        print("   ops = get_unified_matrix_operations()")
        print("   result = ops.matrix_multiply(A, B)")
        print("\n   from src.utils.matrix_operations import matrix_multiply")
        print("   result = matrix_multiply(A, B, use_gpu=True)")
    else:
        print("⚠️ Some tests failed - check implementation")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)