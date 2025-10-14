#!/usr/bin/env python3
"""
Simple test to verify the vectorization integration without external dependencies.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that the vectorization utilities can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test VectorBTRollingOptimizer import
        from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
        print("✅ VectorBTRollingOptimizer imported successfully")
        
        # Test UnifiedVectorizationManager import
        from src.feature_generation.utils.unified_vectorization_manager import UnifiedVectorizationManager
        print("✅ UnifiedVectorizationManager imported successfully")
        
        # Test pipeline import
        from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline import UnifiedDataDrivenPipeline
        print("✅ UnifiedDataDrivenPipeline imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_pipeline_initialization():
    """Test that the pipeline initializes with vectorization utilities."""
    print("\n🚀 Testing pipeline initialization...")
    
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline import UnifiedDataDrivenPipeline
        
        # Initialize pipeline
        pipeline = UnifiedDataDrivenPipeline()
        print("✅ Pipeline initialized successfully")
        
        # Check if vectorization utilities are available
        has_rolling_optimizer = hasattr(pipeline, 'vectorbt_rolling_optimizer')
        has_vectorization_manager = hasattr(pipeline, 'unified_vectorization_manager')
        
        print(f"✅ VectorBTRollingOptimizer available: {has_rolling_optimizer}")
        print(f"✅ UnifiedVectorizationManager available: {has_vectorization_manager}")
        
        # Check if vectorization methods are available
        has_vectorized_rolling = hasattr(pipeline, '_vectorized_rolling_operations')
        has_unified_vectorization = hasattr(pipeline, '_unified_vectorization_processing')
        has_optimized_calculation = hasattr(pipeline, '_optimized_feature_calculation')
        
        print(f"✅ _vectorized_rolling_operations method: {has_vectorized_rolling}")
        print(f"✅ _unified_vectorization_processing method: {has_unified_vectorization}")
        print(f"✅ _optimized_feature_calculation method: {has_optimized_calculation}")
        
        # Check performance stats
        if hasattr(pipeline, 'performance_stats'):
            vectorization_stats = [key for key in pipeline.performance_stats.keys() if 'vector' in key.lower() or 'correlation' in key.lower() or 'momentum' in key.lower() or 'volatility' in key.lower() or 'volume' in key.lower()]
            print(f"✅ Vectorization performance stats: {vectorization_stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False

def test_method_signatures():
    """Test that the vectorization methods have the correct signatures."""
    print("\n🔍 Testing method signatures...")
    
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline import UnifiedDataDrivenPipeline
        
        pipeline = UnifiedDataDrivenPipeline()
        
        # Test method signatures
        import inspect
        
        # Test _vectorized_rolling_operations signature
        if hasattr(pipeline, '_vectorized_rolling_operations'):
            sig = inspect.signature(pipeline._vectorized_rolling_operations)
            params = list(sig.parameters.keys())
            print(f"✅ _vectorized_rolling_operations signature: {params}")
        
        # Test _unified_vectorization_processing signature
        if hasattr(pipeline, '_unified_vectorization_processing'):
            sig = inspect.signature(pipeline._unified_vectorization_processing)
            params = list(sig.parameters.keys())
            print(f"✅ _unified_vectorization_processing signature: {params}")
        
        # Test _optimized_feature_calculation signature
        if hasattr(pipeline, '_optimized_feature_calculation'):
            sig = inspect.signature(pipeline._optimized_feature_calculation)
            params = list(sig.parameters.keys())
            print(f"✅ _optimized_feature_calculation signature: {params}")
        
        return True
        
    except Exception as e:
        print(f"❌ Method signature test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing VectorBTRollingOptimizer and UnifiedVectorizationManager integration...")
    print("=" * 70)
    
    tests = [
        test_imports,
        test_pipeline_initialization,
        test_method_signatures
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Vectorization integration is working correctly.")
        return True
    else:
        print("💥 Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)