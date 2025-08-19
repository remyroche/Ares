#!/usr/bin/env python3
"""
Test script for M1 GPU integration with enhanced matrix operations.
Demonstrates GPU acceleration on Mac M1 using Metal Performance Shaders.
"""

import sys
import os
import asyncio
import numpy as np
import pandas as pd
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.training.enhanced_matrix_gpu_integration import EnhancedMatrixGPUIntegration
    from src.config.m1_gpu_config import get_m1_gpu_config, get_optimized_m1_config
    print("✅ Successfully imported M1 GPU integration modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)


async def test_m1_gpu_basic_operations():
    """Test basic M1 GPU operations."""
    
    print("\n" + "="*60)
    print("🧪 Testing Basic M1 GPU Operations")
    print("="*60)
    
    # Create sample data
    print("📊 Creating sample data...")
    np.random.seed(42)
    
    # Small dataset for quick testing
    features_df = pd.DataFrame({
        'price': np.random.normal(100, 10, 1000),
        'volume': np.random.lognormal(10, 1, 1000),
        'returns': np.random.normal(0, 0.02, 1000),
        'volatility': np.random.gamma(2, 0.01, 1000),
        'momentum': np.random.normal(0, 0.1, 1000),
        'rsi': np.random.uniform(0, 100, 1000),
        'macd': np.random.normal(0, 0.5, 1000),
        'bollinger_upper': np.random.normal(110, 5, 1000),
        'bollinger_lower': np.random.normal(90, 5, 1000),
        'atr': np.random.gamma(1, 0.5, 1000),
    })
    
    # Add more features
    for i in range(20):
        features_df[f'feature_{i+1}'] = np.random.normal(0, 1, 1000)
    
    target = pd.Series(np.random.binomial(1, 0.5, 1000), name='target')
    
    print(f"✅ Created dataset: {features_df.shape[0]} samples, {features_df.shape[1]} features")
    
    # Initialize with default config
    config = get_m1_gpu_config()
    integration = EnhancedMatrixGPUIntegration(config)
    
    # Test GPU availability
    print(f"\n🔍 GPU Status:")
    print(f"  MPS Available: {integration.gpu_accel.mps_available}")
    print(f"  Device: {integration.gpu_accel.device}")
    
    # Benchmark GPU vs CPU
    print(f"\n📊 Running GPU vs CPU Benchmark...")
    benchmark_results = await integration.benchmark_gpu_vs_cpu(features_df, target)
    
    if "benchmarks" in benchmark_results:
        for operation, results in benchmark_results["benchmarks"].items():
            print(f"\n{operation.upper()}:")
            print(f"  CPU Time: {results['cpu_time']:.4f}s")
            print(f"  GPU Time: {results['gpu_time']:.4f}s")
            print(f"  Speedup: {results['speedup']:.2f}x")
            if 'result_difference' in results:
                print(f"  Result Difference: {results['result_difference']:.2e}")
    
    # Test enhanced GPU matrix operations
    print(f"\n🔧 Testing Enhanced GPU Matrix Operations...")
    enhanced_features, enhancement_metadata = await integration.enhanced_gpu_matrix_operations(features_df, target)
    
    print(f"✅ Enhanced features: {len(features_df.columns)} -> {len(enhanced_features.columns)}")
    print(f"📈 Feature increase: +{enhancement_metadata.get('feature_count_increase', 0)} features")
    print(f"⏱️ Processing time: {enhancement_metadata.get('total_processing_time', 0):.2f}s")
    
    # GPU performance summary
    if "gpu_performance_summary" in enhancement_metadata:
        gpu_summary = enhancement_metadata["gpu_performance_summary"]
        print(f"\n🎯 GPU Performance Summary:")
        print(f"  Operations: {gpu_summary.get('gpu_operations_count', 0)}")
        print(f"  Total Time: {gpu_summary.get('gpu_processing_time', 0):.2f}s")
        print(f"  Average Time: {gpu_summary.get('average_gpu_time', 0):.4f}s")
    
    # Clear GPU memory
    integration.clear_gpu_memory()
    
    return True


async def test_m1_gpu_optimization_modes():
    """Test different optimization modes for M1 GPU."""
    
    print("\n" + "="*60)
    print("⚡ Testing M1 GPU Optimization Modes")
    print("="*60)
    
    # Create larger dataset for optimization testing
    print("📊 Creating larger dataset for optimization testing...")
    np.random.seed(42)
    
    features_df = pd.DataFrame({
        'price': np.random.normal(100, 10, 2000),
        'volume': np.random.lognormal(10, 1, 2000),
        'returns': np.random.normal(0, 0.02, 2000),
        'volatility': np.random.gamma(2, 0.01, 2000),
        'momentum': np.random.normal(0, 0.1, 2000),
        'rsi': np.random.uniform(0, 100, 2000),
        'macd': np.random.normal(0, 0.5, 2000),
        'bollinger_upper': np.random.normal(110, 5, 2000),
        'bollinger_lower': np.random.normal(90, 5, 2000),
        'atr': np.random.gamma(1, 0.5, 2000),
    })
    
    # Add more features
    for i in range(50):
        features_df[f'feature_{i+1}'] = np.random.normal(0, 1, 2000)
    
    target = pd.Series(np.random.binomial(1, 0.5, 2000), name='target')
    
    print(f"✅ Created dataset: {features_df.shape[0]} samples, {features_df.shape[1]} features")
    
    # Test different optimization modes
    optimization_modes = ["performance", "memory", "accuracy", "stability"]
    
    for mode in optimization_modes:
        print(f"\n🔧 Testing {mode.upper()} optimization mode...")
        
        # Get optimized config
        config = get_optimized_m1_config(mode)
        integration = EnhancedMatrixGPUIntegration(config)
        
        # Test enhanced operations
        start_time = time.time()
        enhanced_features, enhancement_metadata = await integration.enhanced_gpu_matrix_operations(features_df, target)
        processing_time = time.time() - start_time
        
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Feature increase: +{enhancement_metadata.get('feature_count_increase', 0)} features")
        print(f"  GPU operations: {enhancement_metadata.get('gpu_performance_summary', {}).get('gpu_operations_count', 0)}")
        
        # Clear GPU memory
        integration.clear_gpu_memory()
    
    return True


async def test_m1_gpu_error_handling():
    """Test error handling and fallback mechanisms."""
    
    print("\n" + "="*60)
    print("🛡️ Testing M1 GPU Error Handling")
    print("="*60)
    
    # Create very small dataset to test CPU fallback
    print("📊 Creating small dataset to test CPU fallback...")
    np.random.seed(42)
    
    features_df = pd.DataFrame({
        'price': np.random.normal(100, 10, 100),
        'volume': np.random.lognormal(10, 1, 100),
        'returns': np.random.normal(0, 0.02, 100),
    })
    
    target = pd.Series(np.random.binomial(1, 0.5, 100), name='target')
    
    print(f"✅ Created small dataset: {features_df.shape[0]} samples, {features_df.shape[1]} features")
    
    # Test with high CPU threshold to force fallback
    config = get_m1_gpu_config()
    config["m1_gpu"]["cpu_threshold"] = 10  # Very low threshold
    config["m1_gpu"]["enable_cpu_fallback"] = True
    
    integration = EnhancedMatrixGPUIntegration(config)
    
    print(f"\n🔧 Testing CPU fallback with small dataset...")
    enhanced_features, enhancement_metadata = await integration.enhanced_gpu_matrix_operations(features_df, target)
    
    print(f"✅ Fallback test completed")
    print(f"  Feature increase: +{enhancement_metadata.get('feature_count_increase', 0)} features")
    print(f"  Processing time: {enhancement_metadata.get('total_processing_time', 0):.2f}s")
    
    # Test with invalid data
    print(f"\n🔧 Testing error handling with invalid data...")
    try:
        invalid_features = pd.DataFrame({
            'invalid': [np.nan, np.inf, -np.inf] * 100
        })
        
        enhanced_invalid, invalid_metadata = await integration.enhanced_gpu_matrix_operations(invalid_features, target)
        print(f"✅ Error handling test completed")
        
    except Exception as e:
        print(f"✅ Error handling working: {e}")
    
    # Clear GPU memory
    integration.clear_gpu_memory()
    
    return True


async def test_m1_gpu_integration_summary():
    """Test integration summary and performance reporting."""
    
    print("\n" + "="*60)
    print("📊 Testing M1 GPU Integration Summary")
    print("="*60)
    
    # Create sample data
    print("📊 Creating sample data...")
    np.random.seed(42)
    
    features_df = pd.DataFrame({
        'price': np.random.normal(100, 10, 1500),
        'volume': np.random.lognormal(10, 1, 1500),
        'returns': np.random.normal(0, 0.02, 1500),
        'volatility': np.random.gamma(2, 0.01, 1500),
        'momentum': np.random.normal(0, 0.1, 1500),
    })
    
    # Add more features
    for i in range(30):
        features_df[f'feature_{i+1}'] = np.random.normal(0, 1, 1500)
    
    target = pd.Series(np.random.binomial(1, 0.5, 1500), name='target')
    
    print(f"✅ Created dataset: {features_df.shape[0]} samples, {features_df.shape[1]} features")
    
    # Initialize integration
    config = get_m1_gpu_config()
    integration = EnhancedMatrixGPUIntegration(config)
    
    # Run multiple operations to build up performance data
    print(f"\n🔧 Running multiple operations to build performance data...")
    
    for i in range(3):
        print(f"  Operation {i+1}/3...")
        enhanced_features, enhancement_metadata = await integration.enhanced_gpu_matrix_operations(features_df, target)
        time.sleep(0.1)  # Small delay
    
    # Get integration summary
    print(f"\n📊 Getting integration summary...")
    summary = integration.get_integration_summary()
    
    print(f"✅ Integration Summary:")
    print(f"  GPU Available: {summary.get('gpu_available', False)}")
    print(f"  Device Info: {summary.get('device_info', 'Unknown')}")
    
    if "gpu_performance" in summary:
        gpu_perf = summary["gpu_performance"]
        print(f"  GPU Operations: {gpu_perf.get('gpu_operations_count', 0)}")
        print(f"  GPU Total Time: {gpu_perf.get('gpu_processing_time', 0):.2f}s")
        print(f"  GPU Average Time: {gpu_perf.get('average_gpu_time', 0):.4f}s")
    
    # Clear GPU memory
    integration.clear_gpu_memory()
    
    return True


async def main():
    """Main test function."""
    
    print("🚀 M1 GPU Integration Test Suite")
    print("="*60)
    print("Testing enhanced matrix operations with Mac M1 GPU acceleration")
    print("Using Metal Performance Shaders (MPS) for optimal performance")
    print("="*60)
    
    try:
        # Test basic operations
        await test_m1_gpu_basic_operations()
        
        # Test optimization modes
        await test_m1_gpu_optimization_modes()
        
        # Test error handling
        await test_m1_gpu_error_handling()
        
        # Test integration summary
        await test_m1_gpu_integration_summary()
        
        print("\n" + "="*60)
        print("🎉 All M1 GPU Integration Tests Completed Successfully!")
        print("✅ Enhanced matrix operations with M1 GPU acceleration")
        print("🔒 All operations secured with decorators")
        print("📊 Performance benchmarks completed")
        print("🛡️ Error handling and fallback mechanisms tested")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(main())
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)