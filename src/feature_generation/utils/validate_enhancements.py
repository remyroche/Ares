"""
Validation Script for Enhanced VectorBT Classes

This script validates that the enhanced VectorBT classes are properly implemented
and maintain backward compatibility without requiring external dependencies.
"""

import sys
import os
from pathlib import Path

def validate_file_structure():
    """Validate that all required files exist."""
    print("🔍 Validating file structure...")
    
    required_files = [
        "enhanced_vectorbt_rolling_optimizer.py",
        "enhanced_unified_vectorization_manager.py", 
        "test_backward_compatibility.py",
        "enhancement_migration_guide.md"
    ]
    
    base_path = Path(__file__).parent
    
    for file_name in required_files:
        file_path = base_path / file_name
        if file_path.exists():
            print(f"✅ {file_name} exists")
        else:
            print(f"❌ {file_name} missing")
            return False
    
    return True

def validate_imports():
    """Validate that imports work correctly."""
    print("\n🔍 Validating imports...")
    
    try:
        # Test enhanced vectorbt rolling optimizer imports
        sys.path.insert(0, str(Path(__file__).parent))
        
        from enhanced_vectorbt_rolling_optimizer import (
            EnhancedVectorBTRollingOptimizer,
            VectorBTRollingOptimizer,
            MemoryConfig,
            CacheConfig,
            get_vectorbt_rolling_optimizer
        )
        print("✅ Enhanced VectorBT Rolling Optimizer imports work")
        
        from enhanced_unified_vectorization_manager import (
            EnhancedUnifiedVectorizationManager,
            UnifiedVectorizationManager,
            EnhancedVectorizationConfig,
            get_unified_vectorization_manager
        )
        print("✅ Enhanced Unified Vectorization Manager imports work")
        
        # Test that aliases work
        assert VectorBTRollingOptimizer is EnhancedVectorBTRollingOptimizer
        print("✅ VectorBTRollingOptimizer alias works")
        
        assert UnifiedVectorizationManager is EnhancedUnifiedVectorizationManager
        print("✅ UnifiedVectorizationManager alias works")
        
        return True
        
    except Exception as e:
        print(f"❌ Import validation failed: {e}")
        return False

def validate_class_structure():
    """Validate that classes have required methods and attributes."""
    print("\n🔍 Validating class structure...")
    
    try:
        from enhanced_vectorbt_rolling_optimizer import EnhancedVectorBTRollingOptimizer
        from enhanced_unified_vectorization_manager import EnhancedUnifiedVectorizationManager
        
        # Test EnhancedVectorBTRollingOptimizer
        optimizer_class = EnhancedVectorBTRollingOptimizer
        
        # Check required methods
        required_methods = [
            'rolling_mean', 'rolling_std', 'rolling_var', 'rolling_min', 'rolling_max',
            'rolling_sum', 'rolling_quantile', 'rolling_skew', 'rolling_kurt',
            'rolling_corr', 'rolling_cov', 'rolling_apply', 'rolling_median',
            'rolling_percentile', 'rolling_rank', 'rolling_ewm', 'rolling_ewm_std',
            'rolling_ewm_var', 'rolling_correlation_matrix', 'rolling_covariance_matrix',
            'get_performance_stats', 'reset_stats', 'cleanup', '__enter__', '__exit__'
        ]
        
        for method_name in required_methods:
            if hasattr(optimizer_class, method_name):
                print(f"✅ EnhancedVectorBTRollingOptimizer.{method_name} exists")
            else:
                print(f"❌ EnhancedVectorBTRollingOptimizer.{method_name} missing")
                return False
        
        # Check required attributes
        required_attrs = [
            'enable_gpu', 'enable_parallel', 'memory_efficient', 'chunk_size',
            'use_vectorbt', 'fast_fail', 'enable_logging', 'performance_stats'
        ]
        
        for attr_name in required_attrs:
            if hasattr(optimizer_class, attr_name):
                print(f"✅ EnhancedVectorBTRollingOptimizer.{attr_name} exists")
            else:
                print(f"❌ EnhancedVectorBTRollingOptimizer.{attr_name} missing")
                return False
        
        # Test EnhancedUnifiedVectorizationManager
        manager_class = EnhancedUnifiedVectorizationManager
        
        # Check required methods
        required_methods = [
            'rolling_operation', 'scale_data', 'batch_process_features',
            'optimize_dataframe', 'get_performance_stats', 'reset_stats',
            'performance_monitoring', 'cleanup', '__enter__', '__exit__'
        ]
        
        for method_name in required_methods:
            if hasattr(manager_class, method_name):
                print(f"✅ EnhancedUnifiedVectorizationManager.{method_name} exists")
            else:
                print(f"❌ EnhancedUnifiedVectorizationManager.{method_name} missing")
                return False
        
        # Check required attributes
        required_attrs = [
            'config', 'fast_fail', 'enable_logging', 'performance_stats',
            'rolling_optimizer', 'performance_monitor'
        ]
        
        for attr_name in required_attrs:
            if hasattr(manager_class, attr_name):
                print(f"✅ EnhancedUnifiedVectorizationManager.{attr_name} exists")
            else:
                print(f"❌ EnhancedUnifiedVectorizationManager.{attr_name} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Class structure validation failed: {e}")
        return False

def validate_enhanced_features():
    """Validate that enhanced features are properly implemented."""
    print("\n🔍 Validating enhanced features...")
    
    try:
        from enhanced_vectorbt_rolling_optimizer import (
            MemoryConfig, CacheConfig, MemoryManager, AdvancedCacheManager, M1GPUOptimizer
        )
        from enhanced_unified_vectorization_manager import (
            EnhancedVectorizationConfig, PerformanceMonitor
        )
        
        # Test MemoryConfig
        memory_config = MemoryConfig()
        assert hasattr(memory_config, 'max_memory_gb')
        assert hasattr(memory_config, 'memory_pressure_threshold')
        assert hasattr(memory_config, 'adaptive_chunking')
        assert hasattr(memory_config, 'memory_pooling')
        print("✅ MemoryConfig structure valid")
        
        # Test CacheConfig
        cache_config = CacheConfig()
        assert hasattr(cache_config, 'l1_cache_size')
        assert hasattr(cache_config, 'l2_cache_size')
        assert hasattr(cache_config, 'l2_cache_dir')
        assert hasattr(cache_config, 'cache_ttl')
        print("✅ CacheConfig structure valid")
        
        # Test EnhancedVectorizationConfig
        vectorization_config = EnhancedVectorizationConfig()
        assert hasattr(vectorization_config, 'enable_m1_gpu')
        assert hasattr(vectorization_config, 'adaptive_chunking')
        assert hasattr(vectorization_config, 'enable_caching')
        assert hasattr(vectorization_config, 'l1_cache_size')
        assert hasattr(vectorization_config, 'l2_cache_size')
        print("✅ EnhancedVectorizationConfig structure valid")
        
        # Test MemoryManager
        memory_manager = MemoryManager(memory_config)
        assert hasattr(memory_manager, 'get_memory_pressure')
        assert hasattr(memory_manager, 'calculate_optimal_chunk_size')
        assert hasattr(memory_manager, 'allocate_memory')
        assert hasattr(memory_manager, 'deallocate_memory')
        print("✅ MemoryManager structure valid")
        
        # Test AdvancedCacheManager
        cache_manager = AdvancedCacheManager(cache_config)
        assert hasattr(cache_manager, 'get')
        assert hasattr(cache_manager, 'put')
        assert hasattr(cache_manager, 'get_stats')
        print("✅ AdvancedCacheManager structure valid")
        
        # Test M1GPUOptimizer
        m1_optimizer = M1GPUOptimizer()
        assert hasattr(m1_optimizer, 'can_optimize')
        assert hasattr(m1_optimizer, 'rolling_mean_m1')
        assert hasattr(m1_optimizer, 'rolling_std_m1')
        print("✅ M1GPUOptimizer structure valid")
        
        # Test PerformanceMonitor
        performance_monitor = PerformanceMonitor()
        assert hasattr(performance_monitor, 'record_operation')
        assert hasattr(performance_monitor, 'record_cache_performance')
        assert hasattr(performance_monitor, 'get_performance_summary')
        print("✅ PerformanceMonitor structure valid")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced features validation failed: {e}")
        return False

def validate_backward_compatibility():
    """Validate backward compatibility features."""
    print("\n🔍 Validating backward compatibility...")
    
    try:
        from enhanced_vectorbt_rolling_optimizer import (
            EnhancedVectorBTRollingOptimizer,
            VectorBTRollingOptimizer,
            get_vectorbt_rolling_optimizer
        )
        from enhanced_unified_vectorization_manager import (
            EnhancedUnifiedVectorizationManager,
            UnifiedVectorizationManager,
            get_unified_vectorization_manager
        )
        
        # Test that aliases point to enhanced classes
        assert VectorBTRollingOptimizer is EnhancedVectorBTRollingOptimizer
        print("✅ VectorBTRollingOptimizer alias points to enhanced class")
        
        assert UnifiedVectorizationManager is EnhancedUnifiedVectorizationManager
        print("✅ UnifiedVectorizationManager alias points to enhanced class")
        
        # Test that global functions exist
        assert callable(get_vectorbt_rolling_optimizer)
        print("✅ get_vectorbt_rolling_optimizer function exists")
        
        assert callable(get_unified_vectorization_manager)
        print("✅ get_unified_vectorization_manager function exists")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility validation failed: {e}")
        return False

def validate_documentation():
    """Validate that documentation exists and is properly formatted."""
    print("\n🔍 Validating documentation...")
    
    try:
        migration_guide_path = Path(__file__).parent / "enhancement_migration_guide.md"
        
        if migration_guide_path.exists():
            content = migration_guide_path.read_text()
            
            # Check for key sections
            required_sections = [
                "# Enhanced VectorBT Classes Migration Guide",
                "## Overview",
                "## Key Enhancements",
                "## Migration Steps",
                "## Usage Examples",
                "## Performance Benefits",
                "## Configuration Options"
            ]
            
            for section in required_sections:
                if section in content:
                    print(f"✅ Migration guide contains: {section}")
                else:
                    print(f"❌ Migration guide missing: {section}")
                    return False
            
            print("✅ Migration guide is properly formatted")
        else:
            print("❌ Migration guide not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Documentation validation failed: {e}")
        return False

def main():
    """Run all validation checks."""
    print("🚀 Starting Enhanced VectorBT Classes Validation")
    print("=" * 60)
    
    validations = [
        ("File Structure", validate_file_structure),
        ("Imports", validate_imports),
        ("Class Structure", validate_class_structure),
        ("Enhanced Features", validate_enhanced_features),
        ("Backward Compatibility", validate_backward_compatibility),
        ("Documentation", validate_documentation)
    ]
    
    results = []
    
    for name, validation_func in validations:
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        try:
            result = validation_func()
            results.append((name, result))
            if result:
                print(f"\n✅ {name} validation PASSED")
            else:
                print(f"\n❌ {name} validation FAILED")
        except Exception as e:
            print(f"\n❌ {name} validation ERROR: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} validations passed")
    
    if passed == total:
        print("\n🎉 All validations PASSED! Enhanced VectorBT classes are ready for use.")
        return True
    else:
        print(f"\n⚠️ {total - passed} validations FAILED. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)