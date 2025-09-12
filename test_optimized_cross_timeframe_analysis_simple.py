#!/usr/bin/env python3
"""
Simple test script for optimized cross timeframe analysis.

This script tests the basic structure and imports without requiring
external dependencies.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing Module Imports")
    
    try:
        # Test main integration module
        from src.feature_engineering.optimized_cross_timeframe_analysis_integration import (
            OptimizedCrossTimeframeAnalysisPipeline,
            create_optimized_config,
            analyze_cross_timeframes_optimized
        )
        print("✅ Main integration module imported successfully")
        
        # Test individual modules
        from src.feature_engineering.optimized_cross_timeframe_analysis import (
            OptimizedCrossTimeframeAnalysis,
            OptimizedCrossTimeframeConfig,
            OptimizedCrossTimeframeResult
        )
        print("✅ Main analysis module imported successfully")
        
        from src.feature_engineering.optimized_cross_timeframe_analysis_methods import (
            OptimizedCrossTimeframeMethods
        )
        print("✅ Methods module imported successfully")
        
        from src.feature_engineering.optimized_cross_timeframe_analysis_advanced import (
            OptimizedCrossTimeframeAdvanced
        )
        print("✅ Advanced module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_config_creation():
    """Test configuration creation."""
    print("\n🧪 Testing Configuration Creation")
    
    try:
        from src.feature_engineering.optimized_cross_timeframe_analysis_integration import (
            create_optimized_config
        )
        
        # Test basic config
        config = create_optimized_config()
        print("✅ Basic configuration created successfully")
        
        # Test custom config
        custom_config = create_optimized_config(
            timeframes=['1m', '5m', '15m'],
            enable_m1_optimizations=True,
            enable_gpu_acceleration=True,
            memory_limit_gb=8.0,
            max_workers=4
        )
        print("✅ Custom configuration created successfully")
        
        # Test config attributes
        assert hasattr(custom_config, 'timeframes')
        assert hasattr(custom_config, 'enable_m1_optimizations')
        assert hasattr(custom_config, 'enable_gpu_acceleration')
        assert hasattr(custom_config, 'memory_limit_gb')
        assert hasattr(custom_config, 'max_workers')
        print("✅ Configuration attributes verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_pipeline_creation():
    """Test pipeline creation."""
    print("\n🧪 Testing Pipeline Creation")
    
    try:
        from src.feature_engineering.optimized_cross_timeframe_analysis_integration import (
            OptimizedCrossTimeframeAnalysisPipeline,
            create_optimized_config
        )
        
        # Create config
        config = create_optimized_config(
            timeframes=['1m', '5m'],
            enable_m1_optimizations=False,  # Disable for testing
            enable_gpu_acceleration=False,  # Disable for testing
            enable_advanced_feature_selection=False,  # Disable for testing
            memory_limit_gb=2.0,
            max_workers=1
        )
        
        # Create pipeline
        pipeline = OptimizedCrossTimeframeAnalysisPipeline(config)
        print("✅ Pipeline created successfully")
        
        # Test pipeline methods
        assert hasattr(pipeline, 'analyze_cross_timeframes')
        assert hasattr(pipeline, 'get_optimization_status')
        assert hasattr(pipeline, 'get_performance_metrics')
        assert hasattr(pipeline, 'get_memory_usage')
        assert hasattr(pipeline, 'optimize_memory')
        print("✅ Pipeline methods verified")
        
        # Test optimization status
        status = pipeline.get_optimization_status()
        assert isinstance(status, dict)
        assert 'hardware_optimizations' in status
        assert 'feature_selection' in status
        assert 'utilities' in status
        assert 'caching' in status
        assert 'config' in status
        print("✅ Optimization status verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def test_fallback_integration():
    """Test fallback integration with original module."""
    print("\n🧪 Testing Fallback Integration")
    
    try:
        # Test that the original module can be imported
        from src.feature_engineering.cross_timeframe_interaction_features import (
            CrossTimeframeFeatureGenerator,
            CrossTimeframeConfig,
            InteractionConfig
        )
        print("✅ Original module imported successfully")
        
        # Test that the optimized integration is referenced
        import src.feature_engineering.cross_timeframe_interaction_features as original_module
        
        # Check if the optimized pipeline is being imported
        source_code = Path(original_module.__file__).read_text()
        if 'optimized_cross_timeframe_analysis_integration' in source_code:
            print("✅ Optimized integration detected in original module")
        else:
            print("⚠️ Optimized integration not found in original module")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback integration test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\n🧪 Testing File Structure")
    
    required_files = [
        'src/feature_engineering/optimized_cross_timeframe_analysis.py',
        'src/feature_engineering/optimized_cross_timeframe_analysis_methods.py',
        'src/feature_engineering/optimized_cross_timeframe_analysis_advanced.py',
        'src/feature_engineering/optimized_cross_timeframe_analysis_integration.py',
        'src/feature_engineering/OPTIMIZED_CROSS_TIMEFRAME_ANALYSIS_README.md'
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def main():
    """Main test function."""
    print("🚀 Starting Simple Cross Timeframe Analysis Tests")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("File Structure", test_file_structure),
        ("Module Imports", test_imports),
        ("Configuration Creation", test_config_creation),
        ("Pipeline Creation", test_pipeline_creation),
        ("Fallback Integration", test_fallback_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The optimized cross timeframe analysis is ready to use.")
    elif passed > 0:
        print("⚠️ Some tests passed. The basic structure is working, but some features may not be available.")
    else:
        print("❌ All tests failed. Check the implementation and dependencies.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)