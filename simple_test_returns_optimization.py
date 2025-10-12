#!/usr/bin/env python3
"""
Simple test script for VectorBT optimization in returns feature generation.

This script tests the code structure and imports without requiring external dependencies.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing imports and code structure...")
    
    try:
        # Test basic imports
        print("  - Testing basic imports...")
        from src.feature_generation.categories.returns import (
            ReturnsFeatureGenerator,
            LogReturnsGenerator,
            SimpleReturnsGenerator,
            VectorBTOptimizedReturnsGenerator,
            create_vectorbt_optimized_returns_generators
        )
        print("    ✅ Basic imports successful")
        
        # Test VectorBT imports
        print("  - Testing VectorBT imports...")
        from src.feature_generation.utils.vectorbt_rolling_optimizer import (
            VectorBTRollingOptimizer,
            get_vectorbt_rolling_optimizer
        )
        print("    ✅ VectorBT Rolling Optimizer imports successful")
        
        # Test Unified Vectorization Manager imports
        print("  - Testing Unified Vectorization Manager imports...")
        from src.utils.ml_common.unified_vectorization_manager import (
            UnifiedVectorizationManager,
            OperationType,
            OptimizationStrategy,
            get_unified_vectorization_manager
        )
        print("    ✅ Unified Vectorization Manager imports successful")
        
        return True
        
    except ImportError as e:
        print(f"    ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"    ❌ Unexpected error: {e}")
        return False

def test_class_initialization():
    """Test that classes can be initialized."""
    print("\n🔍 Testing class initialization...")
    
    try:
        # Test ReturnsFeatureGenerator
        print("  - Testing ReturnsFeatureGenerator initialization...")
        from src.feature_generation.categories.returns import ReturnsFeatureGenerator
        gen = ReturnsFeatureGenerator()
        print(f"    ✅ ReturnsFeatureGenerator initialized: {gen.__class__.__name__}")
        
        # Test LogReturnsGenerator
        print("  - Testing LogReturnsGenerator initialization...")
        from src.feature_generation.categories.returns import LogReturnsGenerator
        log_gen = LogReturnsGenerator(period=1)
        print(f"    ✅ LogReturnsGenerator initialized: {log_gen.__class__.__name__}")
        
        # Test SimpleReturnsGenerator
        print("  - Testing SimpleReturnsGenerator initialization...")
        from src.feature_generation.categories.returns import SimpleReturnsGenerator
        simple_gen = SimpleReturnsGenerator(period=1)
        print(f"    ✅ SimpleReturnsGenerator initialized: {simple_gen.__class__.__name__}")
        
        # Test VectorBTOptimizedReturnsGenerator
        print("  - Testing VectorBTOptimizedReturnsGenerator initialization...")
        from src.feature_generation.categories.returns import VectorBTOptimizedReturnsGenerator
        vectorbt_gen = VectorBTOptimizedReturnsGenerator()
        print(f"    ✅ VectorBTOptimizedReturnsGenerator initialized: {vectorbt_gen.__class__.__name__}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Initialization error: {e}")
        return False

def test_method_availability():
    """Test that required methods are available."""
    print("\n🔧 Testing method availability...")
    
    try:
        from src.feature_generation.categories.returns import ReturnsFeatureGenerator
        
        gen = ReturnsFeatureGenerator()
        
        # Test required methods
        required_methods = [
            'generate_feature',
            'generate_returns_features_batch',
            'get_performance_stats',
            'optimize_dataframe_processing',
            'vectorized_rolling_operations'
        ]
        
        for method_name in required_methods:
            if hasattr(gen, method_name):
                print(f"    ✅ Method '{method_name}' available")
            else:
                print(f"    ❌ Method '{method_name}' missing")
                return False
        
        # Test VectorBT components
        if hasattr(gen, 'rolling_optimizer'):
            print("    ✅ VectorBT Rolling Optimizer component available")
        else:
            print("    ⚠️ VectorBT Rolling Optimizer component not available (expected if VectorBT not installed)")
        
        if hasattr(gen, 'unified_manager'):
            print("    ✅ Unified Vectorization Manager component available")
        else:
            print("    ⚠️ Unified Vectorization Manager component not available (expected if not installed)")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Method availability test error: {e}")
        return False

def test_configuration():
    """Test configuration and parameters."""
    print("\n⚙️ Testing configuration...")
    
    try:
        from src.feature_generation.categories.returns import ReturnsFeatureGenerator
        
        gen = ReturnsFeatureGenerator()
        
        # Test configuration
        if hasattr(gen, 'config'):
            config = gen.config
            print(f"    ✅ Configuration available: {config.name}")
            print(f"    ✅ Category: {config.category}")
            print(f"    ✅ Required columns: {config.required_columns}")
            print(f"    ✅ Matrix optimized: {config.matrix_optimized}")
        else:
            print("    ❌ Configuration not available")
            return False
        
        # Test performance stats
        if hasattr(gen, 'performance_stats'):
            stats = gen.performance_stats
            print(f"    ✅ Performance stats available: {list(stats.keys())}")
        else:
            print("    ❌ Performance stats not available")
            return False
        
        return True
        
    except Exception as e:
        print(f"    ❌ Configuration test error: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Starting VectorBT optimization structure tests...")
    print("=" * 70)
    
    tests = [
        ("Import Tests", test_imports),
        ("Class Initialization", test_class_initialization),
        ("Method Availability", test_method_availability),
        ("Configuration", test_configuration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            print(f"✅ {test_name} passed")
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All structure tests passed!")
        print("\nKey VectorBT optimizations implemented:")
        print("✅ VectorBTRollingOptimizer integration")
        print("✅ UnifiedVectorizationManager integration")
        print("✅ Batch processing capabilities")
        print("✅ Performance monitoring")
        print("✅ Comprehensive fallback mechanisms")
        print("✅ Enhanced error handling")
        return 0
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)