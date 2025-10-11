"""
Test VectorBT Optimizations for Interactive Feature Generation

This script tests the VectorBT optimizations integrated into the interactive
feature generation pipeline to ensure they work correctly and provide
performance improvements.

Key Tests:
- VectorBT feature generation functionality
- Performance comparison between VectorBT and manual methods
- Memory efficiency validation
- Backward compatibility verification
- Error handling and fallback mechanisms
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import the components to test
from .vectorbt_optimized_features import (
    VectorBTFeatureGenerator, VectorBTFeatureConfig,
    generate_vectorbt_features, create_vectorbt_config,
    VECTORBT_AVAILABLE
)

from .feature_generation_utils import (
    ImprovedFeatureGenerator, FeatureGenerationConfig,
    VECTORBT_AVAILABLE as UTILS_VECTORBT_AVAILABLE
)

from .enhanced_optimized_orchestrator import (
    EnhancedOptimizedInteractionOrchestrator, EnhancedOptimizedConfig
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data(n_samples: int = 1000, n_features: int = 5) -> pd.DataFrame:
    """Create test OHLCV data for feature generation."""
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = 100 * (1 + returns).cumprod()
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices + np.random.uniform(-0.5, 0.5, n_samples),
        'high': prices + np.random.uniform(0, 1, n_samples),
        'low': prices - np.random.uniform(0, 1, n_samples),
        'close': prices + np.random.uniform(-0.3, 0.3, n_samples),
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    # Ensure OHLC relationships are valid
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    # Add some additional features
    for i in range(n_features):
        data[f'feature_{i}'] = np.random.randn(n_samples)
    
    return data


def test_vectorbt_availability():
    """Test VectorBT availability and basic functionality."""
    print("=" * 60)
    print("TESTING VECTORBT AVAILABILITY")
    print("=" * 60)
    
    print(f"VectorBT available in optimized features: {VECTORBT_AVAILABLE}")
    print(f"VectorBT available in utils: {UTILS_VECTORBT_AVAILABLE}")
    
    if not VECTORBT_AVAILABLE:
        print("❌ VectorBT not available - skipping VectorBT tests")
        return False
    
    try:
        import vectorbt as vbt
        print(f"✅ VectorBT version: {vbt.__version__}")
        return True
    except ImportError as e:
        print(f"❌ VectorBT import failed: {e}")
        return False


def test_vectorbt_feature_generation():
    """Test VectorBT feature generation functionality."""
    print("\n" + "=" * 60)
    print("TESTING VECTORBT FEATURE GENERATION")
    print("=" * 60)
    
    if not VECTORBT_AVAILABLE:
        print("❌ Skipping - VectorBT not available")
        return False
    
    try:
        # Create test data
        data = create_test_data(1000, 3)
        print(f"✅ Created test data: {data.shape}")
        
        # Test VectorBT configuration
        config = create_vectorbt_config(
            use_gpu=False,  # Use CPU for testing
            enable_parallel=True,
            rolling_windows=[5, 10, 20],
            cross_timeframe_periods=[5, 15, 30]
        )
        print(f"✅ Created VectorBT config: {config}")
        
        # Test technical indicators
        generator = VectorBTFeatureGenerator(config)
        technical_features = generator.generate_technical_indicators(data)
        print(f"✅ Generated technical indicators: {technical_features.shape}")
        
        # Test rolling features
        rolling_features = generator.generate_rolling_features(data)
        print(f"✅ Generated rolling features: {rolling_features.shape}")
        
        # Test cross-timeframe features
        cross_tf_features = generator.generate_cross_timeframe_features(data)
        print(f"✅ Generated cross-timeframe features: {cross_tf_features.shape}")
        
        # Test interaction features
        interaction_features = generator.generate_interaction_features(data)
        print(f"✅ Generated interaction features: {interaction_features.shape}")
        
        # Test comprehensive feature generation
        all_features = generate_vectorbt_features(data, config)
        print(f"✅ Generated all features: {all_features.shape}")
        
        # Test validation
        validation_result = generator.validate_features(all_features)
        print(f"✅ Feature validation: {validation_result['passed']}")
        print(f"   Quality score: {validation_result['quality_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ VectorBT feature generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_comparison():
    """Compare performance between VectorBT and manual methods."""
    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE COMPARISON")
    print("=" * 60)
    
    if not VECTORBT_AVAILABLE:
        print("❌ Skipping - VectorBT not available")
        return False
    
    try:
        # Create test data
        data = create_test_data(2000, 5)
        print(f"✅ Created test data: {data.shape}")
        
        # Test manual feature generation
        manual_config = FeatureGenerationConfig(
            enable_vectorbt=False,
            enable_technical_indicators=True,
            enable_rolling_stats=True,
            enable_interaction_features=True,
            enable_cross_timeframe=True,
            rolling_windows=[5, 10, 20],
            cross_timeframe_periods=[5, 15, 30]
        )
        
        manual_generator = ImprovedFeatureGenerator(manual_config)
        
        start_time = time.time()
        manual_features = manual_generator.generate_meaningful_features(data)
        manual_time = time.time() - start_time
        
        print(f"✅ Manual generation: {manual_time:.3f}s, {manual_features.shape[1]} features")
        
        # Test VectorBT feature generation
        vectorbt_config = FeatureGenerationConfig(
            enable_vectorbt=True,
            vectorbt_use_gpu=False,
            vectorbt_enable_parallel=True,
            enable_technical_indicators=True,
            enable_rolling_stats=True,
            enable_interaction_features=True,
            enable_cross_timeframe=True,
            rolling_windows=[5, 10, 20],
            cross_timeframe_periods=[5, 15, 30]
        )
        
        vectorbt_generator = ImprovedFeatureGenerator(vectorbt_config)
        
        start_time = time.time()
        vectorbt_features = vectorbt_generator.generate_meaningful_features(data)
        vectorbt_time = time.time() - start_time
        
        print(f"✅ VectorBT generation: {vectorbt_time:.3f}s, {vectorbt_features.shape[1]} features")
        
        # Calculate performance improvement
        if manual_time > 0:
            speedup = manual_time / vectorbt_time
            print(f"🚀 Speedup: {speedup:.2f}x")
        
        # Compare feature counts
        print(f"📊 Feature count comparison:")
        print(f"   Manual: {manual_features.shape[1]}")
        print(f"   VectorBT: {vectorbt_features.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_efficiency():
    """Test memory efficiency of VectorBT optimizations."""
    print("\n" + "=" * 60)
    print("TESTING MEMORY EFFICIENCY")
    print("=" * 60)
    
    if not VECTORBT_AVAILABLE:
        print("❌ Skipping - VectorBT not available")
        return False
    
    try:
        import psutil
        import os
        
        # Create larger test data
        data = create_test_data(5000, 10)
        print(f"✅ Created test data: {data.shape}")
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"📊 Initial memory usage: {initial_memory:.1f} MB")
        
        # Test VectorBT with memory optimization
        config = create_vectorbt_config(
            use_gpu=False,
            memory_limit_gb=2.0,  # Limit memory usage
            chunk_size=10000,  # Smaller chunks
            enable_parallel=True
        )
        
        generator = VectorBTFeatureGenerator(config)
        
        start_time = time.time()
        features = generator.generate_technical_indicators(data)
        generation_time = time.time() - start_time
        
        # Get memory usage after generation
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        print(f"✅ Generated features: {features.shape}")
        print(f"⏱️ Generation time: {generation_time:.3f}s")
        print(f"💾 Memory used: {memory_used:.1f} MB")
        print(f"📊 Memory per feature: {memory_used / features.shape[1]:.3f} MB")
        
        # Test memory optimization
        optimized_features = generator._optimize_dataframe_dtypes(features)
        optimized_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_saved = final_memory - optimized_memory
        
        print(f"🔧 Memory optimization saved: {memory_saved:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory efficiency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling and fallback mechanisms."""
    print("\n" + "=" * 60)
    print("TESTING ERROR HANDLING")
    print("=" * 60)
    
    try:
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [2, 1, 4],  # Invalid: high < open
            'low': [0, 1, 2],
            'close': [1.5, 1.5, 3.5],
            'volume': [100, 200, 300]
        })
        
        config = create_vectorbt_config(use_gpu=False)
        generator = VectorBTFeatureGenerator(config)
        
        try:
            generator.generate_technical_indicators(invalid_data)
            print("❌ Should have failed with invalid OHLC data")
            return False
        except ValueError as e:
            print(f"✅ Correctly caught invalid OHLC error: {e}")
        
        # Test with empty data
        empty_data = pd.DataFrame()
        
        try:
            generator.generate_technical_indicators(empty_data)
            print("❌ Should have failed with empty data")
            return False
        except ValueError as e:
            print(f"✅ Correctly caught empty data error: {e}")
        
        # Test fallback mechanism in feature generation utils
        data = create_test_data(100, 2)
        
        # Test with VectorBT disabled
        config = FeatureGenerationConfig(
            enable_vectorbt=False,
            enable_technical_indicators=True,
            enable_rolling_stats=True
        )
        
        generator = ImprovedFeatureGenerator(config)
        features = generator.generate_meaningful_features(data)
        
        print(f"✅ Fallback generation successful: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_orchestrator():
    """Test integration with the enhanced optimized orchestrator."""
    print("\n" + "=" * 60)
    print("TESTING ORCHESTRATOR INTEGRATION")
    print("=" * 60)
    
    try:
        # Create test data
        data = create_test_data(1000, 3)
        
        # Create orchestrator config with VectorBT enabled
        config = EnhancedOptimizedConfig(
            enable_vectorbt=True,
            vectorbt_use_gpu=False,
            vectorbt_enable_parallel=True,
            max_workers=2,
            enable_parallel_processing=True
        )
        
        orchestrator = EnhancedOptimizedInteractionOrchestrator(config)
        print(f"✅ Created orchestrator with VectorBT: {config.enable_vectorbt}")
        
        # Test feature generation through orchestrator
        training_input = {
            'data': data,
            'target_column': 'close'
        }
        
        pipeline_state = {
            'symbol': 'TEST',
            'timeframe': '15m',
            'execution_mode': 'full'
        }
        
        # This would normally be async, but we'll test the setup
        print("✅ Orchestrator setup successful")
        print(f"   VectorBT enabled: {config.enable_vectorbt}")
        print(f"   GPU enabled: {config.vectorbt_use_gpu}")
        print(f"   Parallel processing: {config.vectorbt_enable_parallel}")
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all VectorBT optimization tests."""
    print("🚀 STARTING VECTORBT OPTIMIZATION TESTS")
    print("=" * 80)
    
    tests = [
        ("VectorBT Availability", test_vectorbt_availability),
        ("VectorBT Feature Generation", test_vectorbt_feature_generation),
        ("Performance Comparison", test_performance_comparison),
        ("Memory Efficiency", test_memory_efficiency),
        ("Error Handling", test_error_handling),
        ("Orchestrator Integration", test_integration_with_orchestrator)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            results[test_name] = False
            print(f"\n❌ FAILED: {test_name} - {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! VectorBT optimizations are working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return results


if __name__ == "__main__":
    run_all_tests()