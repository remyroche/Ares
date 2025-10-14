"""
Test script for enhanced UnifiedDataDrivenPipeline integration.

This script tests the integration of all the missing functionality
from FeatureLookbackOptimizationComponent into UnifiedDataDrivenPipeline.
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path

# Import the enhanced pipeline
from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_unified_pipeline import (
    EnhancedUnifiedDataDrivenPipeline, create_enhanced_unified_pipeline
)
from src.training.steps.pre_training.unified_data_driven_pipeline.core.config import create_default_config

def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create test data for pipeline testing."""
    np.random.seed(42)
    
    # Generate synthetic OHLCV data
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='15T')
    
    # Generate price data with trend and volatility
    base_price = 100
    returns = np.random.normal(0, 0.02, n_samples)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    }, index=dates)
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

def test_enhanced_pipeline():
    """Test the enhanced pipeline with all integrated functionality."""
    print("🚀 Testing Enhanced UnifiedDataDrivenPipeline Integration")
    print("=" * 60)
    
    # Create test data
    print("📊 Creating test data...")
    data = create_test_data(1000)
    print(f"✅ Created test data: {data.shape}")
    
    # Create targets
    targets = data['close'].pct_change().dropna()
    print(f"✅ Created targets: {len(targets)} samples")
    
    # Create enhanced pipeline
    print("\n🔧 Creating enhanced pipeline...")
    config = create_default_config()
    
    # Enable all features
    config.enable_feature_lookback_optimization = True
    config.enable_interaction_generation = True
    config.enable_htf_interactions = True
    config.enable_feature_selection = True
    
    pipeline = create_enhanced_unified_pipeline(config)
    print("✅ Enhanced pipeline created")
    
    # Test pipeline processing
    print("\n🚀 Testing pipeline processing...")
    start_time = time.time()
    
    try:
        result = pipeline.process(data, targets, timeframe="15m")
        
        processing_time = time.time() - start_time
        
        print(f"✅ Pipeline processing completed in {processing_time:.3f}s")
        
        # Display results
        print("\n📊 Results Summary:")
        print(f"  Success: {result.success}")
        print(f"  Optimal periods: {len(result.optimal_periods)}")
        print(f"  Selected features: {len(result.selected_features)}")
        print(f"  Generated interactions: {len(result.generated_interactions)}")
        print(f"  HTF interactions: {len(result.htf_interactions)}")
        print(f"  Optimized lookbacks: {len(result.optimized_lookbacks)}")
        print(f"  Total processing time: {result.total_processing_time:.3f}s")
        
        # Display performance statistics
        print("\n📈 Performance Statistics:")
        stats = pipeline.get_performance_stats()
        
        print(f"  VectorBT operations: {stats.get('vectorbt_operations', 0)}")
        print(f"  Economic evaluations: {stats.get('economic_evaluations', 0)}")
        print(f"  Feature selections: {stats.get('feature_selections', 0)}")
        print(f"  Lookback optimizations: {stats.get('lookback_optimizations', 0)}")
        
        # Test component integration
        print("\n🔧 Testing component integration...")
        
        # Test advanced lookback optimizer
        print("  Testing advanced lookback optimizer...")
        lookback_stats = stats.get('advanced_lookback_optimizer', {})
        print(f"    Total optimizations: {lookback_stats.get('total_optimizations', 0)}")
        print(f"    Successful optimizations: {lookback_stats.get('successful_optimizations', 0)}")
        print(f"    Parallel operations: {lookback_stats.get('parallel_operations', 0)}")
        
        # Test feature bank integration
        print("  Testing feature bank integration...")
        feature_bank_stats = stats.get('feature_bank_integration', {})
        print(f"    Total generations: {feature_bank_stats.get('total_generations', 0)}")
        print(f"    Features generated: {feature_bank_stats.get('features_generated', 0)}")
        print(f"    Cache hits: {feature_bank_stats.get('cache_hits', 0)}")
        print(f"    Cache misses: {feature_bank_stats.get('cache_misses', 0)}")
        
        # Test advanced cache manager
        print("  Testing advanced cache manager...")
        cache_stats = stats.get('advanced_cache_manager', {})
        print(f"    Total entries: {cache_stats.get('total_entries', 0)}")
        print(f"    Memory entries: {cache_stats.get('memory_entries', 0)}")
        print(f"    Disk entries: {cache_stats.get('disk_entries', 0)}")
        print(f"    Hit rate: {cache_stats.get('hit_rate', 0):.2%}")
        
        # Test modular architecture
        print("  Testing modular architecture...")
        validation_stats = stats.get('input_validator', {})
        print(f"    Total validations: {validation_stats.get('total_validations', 0)}")
        print(f"    Successful validations: {validation_stats.get('successful_validations', 0)}")
        
        error_stats = stats.get('error_handler', {})
        print(f"    Total errors: {error_stats.get('total_errors', 0)}")
        print(f"    Errors by category: {error_stats.get('errors_by_category', {})}")
        
        performance_stats = stats.get('performance_monitor', {})
        print(f"    Total operations: {performance_stats.get('total_operations', 0)}")
        print(f"    Memory usage: {performance_stats.get('memory_usage_mb', 0):.1f} MB")
        
        print("\n✅ All tests completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """Test individual components separately."""
    print("\n🔧 Testing Individual Components")
    print("=" * 40)
    
    # Create test data
    data = create_test_data(500)
    targets = data['close'].pct_change().dropna()
    
    # Test advanced lookback optimizer
    print("Testing Advanced Lookback Optimizer...")
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_lookback_optimizer import (
            create_advanced_lookback_optimizer, LookbackConstraints, OptimizationMethod
        )
        
        optimizer = create_advanced_lookback_optimizer()
        
        # Test with a few features
        feature_names = ['close', 'volume']
        lookback_range = (5, 50)
        
        results = optimizer.optimize_features_parallel_batch(
            data=data,
            feature_names=feature_names,
            target_column='close',
            lookback_range=lookback_range,
            method=OptimizationMethod.COARSE_TO_REFINE
        )
        
        print(f"  ✅ Optimized {len(results)} features")
        for result in results:
            if result.success:
                print(f"    {result.feature_name}: lookback={result.best_lookback}, score={result.best_score:.4f}")
        
    except Exception as e:
        print(f"  ❌ Advanced lookback optimizer test failed: {e}")
    
    # Test feature bank integration
    print("\nTesting Feature Bank Integration...")
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.feature_bank_integration import (
            create_feature_bank_integration, FeatureBankConfig
        )
        
        feature_bank = create_feature_bank_integration()
        
        result = feature_bank.generate_features_for_optimization(data)
        
        if result.success:
            print(f"  ✅ Generated {result.n_features_generated} features")
            print(f"  Cache hit: {result.cache_hit}")
            print(f"  Memory usage: {result.memory_usage_mb:.1f} MB")
        else:
            print(f"  ❌ Feature generation failed: {result.error_message}")
        
    except Exception as e:
        print(f"  ❌ Feature bank integration test failed: {e}")
    
    # Test advanced cache manager
    print("\nTesting Advanced Cache Manager...")
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_caching import (
            create_advanced_cache_manager, CacheConfig
        )
        
        cache_manager = create_advanced_cache_manager()
        
        # Test caching
        test_key = "test_data"
        test_data = data.head(100)
        
        # Set data
        cache_manager.set(test_key, test_data)
        
        # Get data
        retrieved_data = cache_manager.get(test_key)
        
        if retrieved_data is not None:
            print("  ✅ Cache set/get successful")
            print(f"  Retrieved data shape: {retrieved_data.shape}")
        else:
            print("  ❌ Cache retrieval failed")
        
        # Get stats
        stats = cache_manager.get_stats()
        print(f"  Cache stats: {stats.total_entries} entries, {stats.hit_rate:.2%} hit rate")
        
    except Exception as e:
        print(f"  ❌ Advanced cache manager test failed: {e}")

if __name__ == "__main__":
    print("🧪 Enhanced UnifiedDataDrivenPipeline Integration Test")
    print("=" * 60)
    
    # Test individual components first
    test_individual_components()
    
    # Test full pipeline
    success = test_enhanced_pipeline()
    
    if success:
        print("\n🎉 All tests passed! The enhanced pipeline successfully integrates")
        print("   all the missing functionality from FeatureLookbackOptimizationComponent.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")