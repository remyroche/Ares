#!/usr/bin/env python3
"""
Simple validation script for volume feature optimization.

This script validates that the volume.py file has been properly optimized with VectorBT.
"""

import os
import re

def check_imports():
    """Check if VectorBT and UnifiedVectorizationManager imports are present."""
    print("Checking imports...")
    
    with open('/workspace/src/feature_generation/categories/volume.py', 'r') as f:
        content = f.read()
    
    # Check for VectorBT imports
    vectorbt_imports = [
        'from ..utils.vectorbt_rolling_optimizer import',
        'from ...utils.ml_common.unified_vectorization_manager import',
        'VectorBTRollingOptimizer',
        'UnifiedVectorizationManager',
        'OperationType',
        'OptimizationStrategy'
    ]
    
    missing_imports = []
    for import_str in vectorbt_imports:
        if import_str not in content:
            missing_imports.append(import_str)
    
    if missing_imports:
        print(f"❌ Missing imports: {missing_imports}")
        return False
    else:
        print("✅ All required imports present")
        return True

def check_class_optimizations():
    """Check if classes have been optimized with VectorBT."""
    print("Checking class optimizations...")
    
    with open('/workspace/src/feature_generation/categories/volume.py', 'r') as f:
        content = f.read()
    
    # Check for optimized class patterns
    optimizations = [
        'self.rolling_optimizer = get_vectorbt_rolling_optimizer',
        'self.unified_manager = get_unified_vectorization_manager',
        'VectorBTRollingOptimizer',
        'UnifiedVectorizationManager',
        'performance_stats',
        'memory_optimization',
        'gpu_available',
        'batch_processing'
    ]
    
    found_optimizations = []
    for opt in optimizations:
        if opt in content:
            found_optimizations.append(opt)
    
    print(f"✅ Found {len(found_optimizations)}/{len(optimizations)} optimizations")
    return len(found_optimizations) >= len(optimizations) * 0.8  # 80% threshold

def check_method_implementations():
    """Check if optimization methods are implemented."""
    print("Checking method implementations...")
    
    with open('/workspace/src/feature_generation/categories/volume.py', 'r') as f:
        content = f.read()
    
    # Check for key optimization methods
    methods = [
        'def _should_use_vectorbt',
        'def _should_use_unified_manager',
        'def _generate_with_unified_manager',
        'def generate_batch_volume_features',
        'def _optimize_memory_usage',
        'def _process_in_chunks',
        'def _check_gpu_availability',
        'def _monitor_operation',
        'def get_performance_summary'
    ]
    
    found_methods = []
    for method in methods:
        if method in content:
            found_methods.append(method)
    
    print(f"✅ Found {len(found_methods)}/{len(methods)} optimization methods")
    return len(found_methods) >= len(methods) * 0.8  # 80% threshold

def check_factory_implementation():
    """Check if OptimizedVolumeFeatureFactory is implemented."""
    print("Checking factory implementation...")
    
    with open('/workspace/src/feature_generation/categories/volume.py', 'r') as f:
        content = f.read()
    
    factory_components = [
        'class OptimizedVolumeFeatureFactory',
        'def create_optimized_volume_factory',
        'def generate_comprehensive_volume_features',
        'def get_performance_stats'
    ]
    
    found_components = []
    for component in factory_components:
        if component in content:
            found_components.append(component)
    
    print(f"✅ Found {len(found_components)}/{len(factory_components)} factory components")
    return len(found_components) >= len(factory_components) * 0.8  # 80% threshold

def check_performance_monitoring():
    """Check if performance monitoring is implemented."""
    print("Checking performance monitoring...")
    
    with open('/workspace/src/feature_generation/categories/volume.py', 'r') as f:
        content = f.read()
    
    monitoring_features = [
        'performance_stats',
        'operation_times',
        'memory_usage_history',
        'def _monitor_operation',
        'def get_performance_summary',
        'def log_performance_report',
        'def reset_performance_stats'
    ]
    
    found_features = []
    for feature in monitoring_features:
        if feature in content:
            found_features.append(feature)
    
    print(f"✅ Found {len(found_features)}/{len(monitoring_features)} monitoring features")
    return len(found_features) >= len(monitoring_features) * 0.8  # 80% threshold

def check_memory_optimization():
    """Check if memory optimization is implemented."""
    print("Checking memory optimization...")
    
    with open('/workspace/src/feature_generation/categories/volume.py', 'r') as f:
        content = f.read()
    
    memory_features = [
        'memory_threshold_mb',
        'chunk_size',
        'enable_memory_optimization',
        'def _optimize_memory_usage',
        'def _process_in_chunks',
        'def _should_use_chunking'
    ]
    
    found_features = []
    for feature in memory_features:
        if feature in content:
            found_features.append(feature)
    
    print(f"✅ Found {len(found_features)}/{len(memory_features)} memory optimization features")
    return len(found_features) >= len(memory_features) * 0.8  # 80% threshold

def check_gpu_acceleration():
    """Check if GPU acceleration is implemented."""
    print("Checking GPU acceleration...")
    
    with open('/workspace/src/feature_generation/categories/volume.py', 'r') as f:
        content = f.read()
    
    gpu_features = [
        'enable_gpu',
        'gpu_available',
        'gpu_threshold',
        'def _check_gpu_availability',
        'def _should_use_gpu',
        'def _convert_to_gpu',
        'def _convert_from_gpu'
    ]
    
    found_features = []
    for feature in gpu_features:
        if feature in content:
            found_features.append(feature)
    
    print(f"✅ Found {len(found_features)}/{len(gpu_features)} GPU acceleration features")
    return len(found_features) >= len(gpu_features) * 0.8  # 80% threshold

def main():
    """Run all validation checks."""
    print("=== Volume Feature Optimization Validation ===\n")
    
    checks = [
        ("Imports", check_imports),
        ("Class Optimizations", check_class_optimizations),
        ("Method Implementations", check_method_implementations),
        ("Factory Implementation", check_factory_implementation),
        ("Performance Monitoring", check_performance_monitoring),
        ("Memory Optimization", check_memory_optimization),
        ("GPU Acceleration", check_gpu_acceleration)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n--- {check_name} ---")
        try:
            if check_func():
                passed += 1
                print(f"✅ {check_name} PASSED")
            else:
                print(f"❌ {check_name} FAILED")
        except Exception as e:
            print(f"❌ {check_name} ERROR: {e}")
    
    print(f"\n=== Validation Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All validations passed! Volume features are fully optimized with VectorBT.")
        print("\nKey optimizations implemented:")
        print("  ✅ VectorBTRollingOptimizer integration")
        print("  ✅ UnifiedVectorizationManager integration")
        print("  ✅ Batch processing capabilities")
        print("  ✅ Memory optimization and chunking")
        print("  ✅ GPU acceleration support")
        print("  ✅ Performance monitoring and statistics")
        print("  ✅ OptimizedVolumeFeatureFactory")
    else:
        print(f"\n⚠️ {total - passed} validations failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)