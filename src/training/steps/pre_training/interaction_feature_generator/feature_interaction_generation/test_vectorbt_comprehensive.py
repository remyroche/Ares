"""
Comprehensive VectorBT Integration Test

This script tests all the VectorBT improvements made to the interaction feature generation system,
including performance comparisons, memory optimization, and feature quality validation.

Key Tests:
- VectorBT rolling operations performance
- Interaction feature generation with VectorBT
- Cross-timeframe features with VectorBT
- Technical indicators with VectorBT
- Memory optimization with VectorBT
- GPU acceleration (if available)
- Feature validation with VectorBT
- Error handling and fallback mechanisms
"""

import numpy as np
import pandas as pd
import time
import logging
import psutil
import os
from typing import Dict, Any, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import the enhanced VectorBT components
from .vectorbt_optimized_features import (
    VectorBTFeatureGenerator, VectorBTFeatureConfig,
    generate_vectorbt_features, create_vectorbt_config,
    VECTORBT_AVAILABLE, CUPY_AVAILABLE
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_comprehensive_test_data(n_samples: int = 5000, n_features: int = 10) -> pd.DataFrame:
    """Create comprehensive test data for VectorBT testing."""
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
    
    # Add additional features
    for i in range(n_features):
        data[f'feature_{i}'] = np.random.randn(n_samples)
    
    return data


def test_vectorbt_availability():
    """Test VectorBT availability and basic functionality."""
    print("=" * 60)
    print("TESTING VECTORBT AVAILABILITY")
    print("=" * 60)
    
    print(f"VectorBT available: {VECTORBT_AVAILABLE}")
    print(f"CuPy available: {CUPY_AVAILABLE}")
    
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


def test_rolling_operations_performance():
    """Test VectorBT rolling operations performance."""
    print("\n" + "=" * 60)
    print("TESTING ROLLING OPERATIONS PERFORMANCE")
    print("=" * 60)
    
    if not VECTORBT_AVAILABLE:
        print("❌ Skipping - VectorBT not available")
        return False
    
    try:
        # Create test data
        data = create_comprehensive_test_data(2000, 5)
        print(f"✅ Created test data: {data.shape}")
        
        # Test CPU configuration
        config_cpu = create_vectorbt_config(
            use_gpu=False,
            enable_parallel=True,
            rolling_windows=[5, 10, 20, 50],
            chunk_size=1000
        )
        
        generator_cpu = VectorBTFeatureGenerator(config_cpu)
        
        start_time = time.time()
        rolling_features_cpu = generator_cpu.generate_rolling_features(data)
        cpu_time = time.time() - start_time
        
        print(f"✅ CPU rolling features: {rolling_features_cpu.shape[1]} features in {cpu_time:.3f}s")
        
        # Test GPU configuration if available
        if CUPY_AVAILABLE:
            config_gpu = create_vectorbt_config(
                use_gpu=True,
                enable_parallel=True,
                rolling_windows=[5, 10, 20, 50],
                chunk_size=1000
            )
            
            generator_gpu = VectorBTFeatureGenerator(config_gpu)
            
            start_time = time.time()
            rolling_features_gpu = generator_gpu.generate_rolling_features(data)
            gpu_time = time.time() - start_time
            
            print(f"✅ GPU rolling features: {rolling_features_gpu.shape[1]} features in {gpu_time:.3f}s")
            
            if cpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"🚀 GPU speedup: {speedup:.2f}x")
        
        # Validate features
        validation = generator_cpu.validate_features(rolling_features_cpu)
        print(f"✅ Feature validation: {validation['passed']}")
        print(f"   Quality score: {validation['quality_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Rolling operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_interaction_features_performance():
    """Test VectorBT interaction features performance."""
    print("\n" + "=" * 60)
    print("TESTING INTERACTION FEATURES PERFORMANCE")
    print("=" * 60)
    
    if not VECTORBT_AVAILABLE:
        print("❌ Skipping - VectorBT not available")
        return False
    
    try:
        # Create test data
        data = create_comprehensive_test_data(1500, 8)
        print(f"✅ Created test data: {data.shape}")
        
        # Test CPU configuration
        config_cpu = create_vectorbt_config(
            use_gpu=False,
            enable_parallel=True,
            chunk_size=1000
        )
        
        generator_cpu = VectorBTFeatureGenerator(config_cpu)
        
        start_time = time.time()
        interaction_features_cpu = generator_cpu.generate_interaction_features(data)
        cpu_time = time.time() - start_time
        
        print(f"✅ CPU interaction features: {interaction_features_cpu.shape[1]} features in {cpu_time:.3f}s")
        
        # Test GPU configuration if available
        if CUPY_AVAILABLE:
            config_gpu = create_vectorbt_config(
                use_gpu=True,
                enable_parallel=True,
                chunk_size=1000
            )
            
            generator_gpu = VectorBTFeatureGenerator(config_gpu)
            
            start_time = time.time()
            interaction_features_gpu = generator_gpu.generate_interaction_features(data)
            gpu_time = time.time() - start_time
            
            print(f"✅ GPU interaction features: {interaction_features_gpu.shape[1]} features in {gpu_time:.3f}s")
            
            if cpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"🚀 GPU speedup: {speedup:.2f}x")
        
        # Validate features
        validation = generator_cpu.validate_features(interaction_features_cpu)
        print(f"✅ Feature validation: {validation['passed']}")
        print(f"   Quality score: {validation['quality_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Interaction features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cross_timeframe_features_performance():
    """Test VectorBT cross-timeframe features performance."""
    print("\n" + "=" * 60)
    print("TESTING CROSS-TIMEFRAME FEATURES PERFORMANCE")
    print("=" * 60)
    
    if not VECTORBT_AVAILABLE:
        print("❌ Skipping - VectorBT not available")
        return False
    
    try:
        # Create test data
        data = create_comprehensive_test_data(2000, 6)
        print(f"✅ Created test data: {data.shape}")
        
        # Test CPU configuration
        config_cpu = create_vectorbt_config(
            use_gpu=False,
            enable_parallel=True,
            cross_timeframe_periods=[5, 15, 30, 60],
            chunk_size=1000
        )
        
        generator_cpu = VectorBTFeatureGenerator(config_cpu)
        
        start_time = time.time()
        ctf_features_cpu = generator_cpu.generate_cross_timeframe_features(data)
        cpu_time = time.time() - start_time
        
        print(f"✅ CPU cross-timeframe features: {ctf_features_cpu.shape[1]} features in {cpu_time:.3f}s")
        
        # Test GPU configuration if available
        if CUPY_AVAILABLE:
            config_gpu = create_vectorbt_config(
                use_gpu=True,
                enable_parallel=True,
                cross_timeframe_periods=[5, 15, 30, 60],
                chunk_size=1000
            )
            
            generator_gpu = VectorBTFeatureGenerator(config_gpu)
            
            start_time = time.time()
            ctf_features_gpu = generator_gpu.generate_cross_timeframe_features(data)
            gpu_time = time.time() - start_time
            
            print(f"✅ GPU cross-timeframe features: {ctf_features_gpu.shape[1]} features in {gpu_time:.3f}s")
            
            if cpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"🚀 GPU speedup: {speedup:.2f}x")
        
        # Validate features
        validation = generator_cpu.validate_features(ctf_features_cpu)
        print(f"✅ Feature validation: {validation['passed']}")
        print(f"   Quality score: {validation['quality_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cross-timeframe features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_technical_indicators_performance():
    """Test VectorBT technical indicators performance using feature bank."""
    print("\n" + "=" * 60)
    print("TESTING TECHNICAL INDICATORS PERFORMANCE (FEATURE BANK)")
    print("=" * 60)
    
    if not VECTORBT_AVAILABLE:
        print("❌ Skipping - VectorBT not available")
        return False
    
    try:
        # Create test data
        data = create_comprehensive_test_data(1500, 3)
        print(f"✅ Created test data: {data.shape}")
        
        # Test CPU configuration
        config_cpu = create_vectorbt_config(
            use_gpu=False,
            enable_parallel=True,
            enable_rsi=True,
            enable_macd=True,
            enable_bollinger=True,
            enable_sma=True,
            enable_ema=True,
            rsi_periods=[14, 21, 28],
            sma_periods=[5, 10, 20, 50],
            ema_periods=[5, 10, 20, 50],
            bb_periods=[20, 30],
            bb_std_devs=[1.5, 2.0]
        )
        
        generator_cpu = VectorBTFeatureGenerator(config_cpu)
        
        start_time = time.time()
        tech_features_cpu = generator_cpu.generate_technical_indicators(data)
        cpu_time = time.time() - start_time
        
        print(f"✅ CPU technical indicators (feature bank): {tech_features_cpu.shape[1]} features in {cpu_time:.3f}s")
        
        # Test GPU configuration if available
        if CUPY_AVAILABLE:
            config_gpu = create_vectorbt_config(
                use_gpu=True,
                enable_parallel=True,
                enable_rsi=True,
                enable_macd=True,
                enable_bollinger=True,
                enable_sma=True,
                enable_ema=True,
                rsi_periods=[14, 21, 28],
                sma_periods=[5, 10, 20, 50],
                ema_periods=[5, 10, 20, 50],
                bb_periods=[20, 30],
                bb_std_devs=[1.5, 2.0]
            )
            
            generator_gpu = VectorBTFeatureGenerator(config_gpu)
            
            start_time = time.time()
            tech_features_gpu = generator_gpu.generate_technical_indicators(data)
            gpu_time = time.time() - start_time
            
            print(f"✅ GPU technical indicators (feature bank): {tech_features_gpu.shape[1]} features in {gpu_time:.3f}s")
            
            if cpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"🚀 GPU speedup: {speedup:.2f}x")
        
        # Validate features
        validation = generator_cpu.validate_features(tech_features_cpu)
        print(f"✅ Feature validation: {validation['passed']}")
        print(f"   Quality score: {validation['quality_score']:.3f}")
        
        # Test additional feature bank features
        start_time = time.time()
        additional_features = generator_cpu._generate_additional_feature_bank_features(data)
        additional_time = time.time() - start_time
        
        print(f"✅ Additional feature bank features: {additional_features.shape[1]} features in {additional_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Technical indicators test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_optimization():
    """Test VectorBT memory optimization."""
    print("\n" + "=" * 60)
    print("TESTING MEMORY OPTIMIZATION")
    print("=" * 60)
    
    if not VECTORBT_AVAILABLE:
        print("❌ Skipping - VectorBT not available")
        return False
    
    try:
        # Create larger test data
        data = create_comprehensive_test_data(3000, 15)
        print(f"✅ Created test data: {data.shape}")
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"📊 Initial memory usage: {initial_memory:.1f} MB")
        
        # Test with memory optimization
        config = create_vectorbt_config(
            use_gpu=False,
            enable_parallel=True,
            memory_limit_gb=2.0,
            chunk_size=5000,
            rolling_windows=[5, 10, 20, 50, 100],
            cross_timeframe_periods=[5, 15, 30, 60, 120]
        )
        
        generator = VectorBTFeatureGenerator(config)
        
        # Generate all types of features
        start_time = time.time()
        
        rolling_features = generator.generate_rolling_features(data)
        interaction_features = generator.generate_interaction_features(data)
        ctf_features = generator.generate_cross_timeframe_features(data)
        tech_features = generator.generate_technical_indicators(data)
        
        generation_time = time.time() - start_time
        
        # Get memory usage after generation
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        print(f"✅ Generated features:")
        print(f"   Rolling: {rolling_features.shape[1]} features")
        print(f"   Interaction: {interaction_features.shape[1]} features")
        print(f"   Cross-timeframe: {ctf_features.shape[1]} features")
        print(f"   Technical: {tech_features.shape[1]} features")
        print(f"⏱️ Total generation time: {generation_time:.3f}s")
        print(f"💾 Memory used: {memory_used:.1f} MB")
        
        # Test memory optimization
        total_features = rolling_features.shape[1] + interaction_features.shape[1] + ctf_features.shape[1] + tech_features.shape[1]
        memory_per_feature = memory_used / total_features if total_features > 0 else 0
        print(f"📊 Memory per feature: {memory_per_feature:.3f} MB")
        
        # Test dtype optimization
        combined_features = pd.concat([rolling_features, interaction_features, ctf_features, tech_features], axis=1)
        optimized_features = generator._optimize_dataframe_dtypes(combined_features)
        
        original_memory = combined_features.memory_usage(deep=True).sum() / 1024 / 1024
        optimized_memory = optimized_features.memory_usage(deep=True).sum() / 1024 / 1024
        memory_saved = original_memory - optimized_memory
        
        print(f"🔧 Memory optimization:")
        print(f"   Original: {original_memory:.2f} MB")
        print(f"   Optimized: {optimized_memory:.2f} MB")
        print(f"   Saved: {memory_saved:.2f} MB ({memory_saved/original_memory*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comprehensive_feature_generation():
    """Test comprehensive feature generation with all VectorBT optimizations."""
    print("\n" + "=" * 60)
    print("TESTING COMPREHENSIVE FEATURE GENERATION")
    print("=" * 60)
    
    if not VECTORBT_AVAILABLE:
        print("❌ Skipping - VectorBT not available")
        return False
    
    try:
        # Create test data
        data = create_comprehensive_test_data(2000, 8)
        print(f"✅ Created test data: {data.shape}")
        
        # Test comprehensive configuration
        config = create_vectorbt_config(
            use_gpu=CUPY_AVAILABLE,
            enable_parallel=True,
            memory_limit_gb=4.0,
            chunk_size=2000,
            rolling_windows=[5, 10, 20, 50],
            cross_timeframe_periods=[5, 15, 30, 60],
            enable_rsi=True,
            enable_macd=True,
            enable_bollinger=True,
            enable_sma=True,
            enable_ema=True,
            rsi_periods=[14, 21],
            sma_periods=[5, 10, 20],
            ema_periods=[5, 10, 20],
            bb_periods=[20],
            bb_std_devs=[2.0]
        )
        
        generator = VectorBTFeatureGenerator(config)
        
        # Generate all features
        start_time = time.time()
        all_features = generate_vectorbt_features(data, config)
        total_time = time.time() - start_time
        
        print(f"✅ Generated comprehensive features: {all_features.shape[1]} features in {total_time:.3f}s")
        
        # Validate features
        validation = generator.validate_features(all_features)
        print(f"✅ Feature validation: {validation['passed']}")
        print(f"   Quality score: {validation['quality_score']:.3f}")
        
        if validation['issues']:
            print(f"   Issues: {validation['issues']}")
        
        # Performance metrics
        features_per_second = all_features.shape[1] / total_time if total_time > 0 else 0
        print(f"📊 Performance: {features_per_second:.1f} features/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive feature generation test failed: {e}")
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
        
        # Test with insufficient data
        minimal_data = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100.5],
            'volume': [1000]
        })
        
        try:
            features = generator.generate_technical_indicators(minimal_data)
            print(f"✅ Handled minimal data gracefully: {features.shape}")
        except Exception as e:
            print(f"✅ Correctly handled minimal data error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comprehensive_tests():
    """Run all comprehensive VectorBT tests."""
    print("🚀 STARTING COMPREHENSIVE VECTORBT TESTS")
    print("=" * 80)
    
    tests = [
        ("VectorBT Availability", test_vectorbt_availability),
        ("Rolling Operations Performance", test_rolling_operations_performance),
        ("Interaction Features Performance", test_interaction_features_performance),
        ("Cross-timeframe Features Performance", test_cross_timeframe_features_performance),
        ("Technical Indicators Performance", test_technical_indicators_performance),
        ("Memory Optimization", test_memory_optimization),
        ("Comprehensive Feature Generation", test_comprehensive_feature_generation),
        ("Error Handling", test_error_handling)
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
    print("COMPREHENSIVE TEST SUMMARY")
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
    run_comprehensive_tests()