"""
Test script for the consolidated Unified Data-Driven Pipeline

This script validates that the consolidated pipeline works correctly
and provides all the expected functionality.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from src.training.steps.pre_training.unified_data_driven_pipeline import (
        UnifiedDataDrivenPipeline,
        ConsolidatedPipelineResult,
        create_unified_pipeline,
        process_with_unified_pipeline,
        UnifiedPipelineConfig
    )
    print("✅ Successfully imported consolidated pipeline")
except ImportError as e:
    print(f"❌ Failed to import consolidated pipeline: {e}")
    sys.exit(1)


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample data for testing."""
    np.random.seed(42)
    
    # Create OHLCV data
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='15T')
    
    # Generate realistic price data
    returns = np.random.normal(0, 0.01, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
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


def create_sample_targets(data: pd.DataFrame) -> pd.Series:
    """Create sample targets for testing."""
    # Create future returns as targets
    future_returns = data['close'].pct_change().shift(-1)
    return future_returns.dropna()


def test_basic_functionality():
    """Test basic pipeline functionality."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Create sample data
        data = create_sample_data(500)
        targets = create_sample_targets(data)
        
        # Create pipeline
        pipeline = create_unified_pipeline()
        
        # Process data
        result = pipeline.process(
            data=data,
            targets=targets,
            feature_columns=None,
            timeframe="15m"
        )
        
        # Validate result
        assert isinstance(result, ConsolidatedPipelineResult), "Result should be ConsolidatedPipelineResult"
        assert result.success, f"Pipeline should succeed, but got error: {result.error_message}"
        assert len(result.selected_features) > 0, "Should select some features"
        assert result.processing_time > 0, "Processing time should be positive"
        
        print("✅ Basic functionality test passed")
        print(f"   - Selected {len(result.selected_features)} features")
        print(f"   - Processing time: {result.processing_time:.3f}s")
        print(f"   - Success: {result.success}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_enhanced_features():
    """Test enhanced features are available."""
    print("\n🧪 Testing enhanced features...")
    
    try:
        # Create sample data
        data = create_sample_data(500)
        targets = create_sample_targets(data)
        
        # Create pipeline
        pipeline = create_unified_pipeline()
        
        # Process data
        result = pipeline.process(
            data=data,
            targets=targets,
            feature_columns=None,
            timeframe="15m"
        )
        
        # Test enhanced features
        assert result.optimal_periods is not None, "Should have optimal periods"
        assert result.period_scores is not None, "Should have period scores"
        assert result.economic_evaluation_results is not None, "Should have economic evaluation"
        assert result.generated_interactions is not None, "Should have generated interactions"
        assert result.htf_interactions is not None, "Should have HTF interactions"
        assert result.optimized_lookbacks is not None, "Should have optimized lookbacks"
        assert result.cross_timeframe_features is not None, "Should have cross-timeframe features"
        assert result.interaction_features is not None, "Should have interaction features"
        assert result.no_features is not None, "Should have no features"
        assert result.comparison_features is not None, "Should have comparison features"
        
        print("✅ Enhanced features test passed")
        print(f"   - Optimal periods: {len(result.optimal_periods)}")
        print(f"   - Generated interactions: {len(result.generated_interactions)}")
        print(f"   - HTF interactions: {len(result.htf_interactions)}")
        print(f"   - Optimized lookbacks: {len(result.optimized_lookbacks)}")
        print(f"   - Cross-timeframe features: {len(result.cross_timeframe_features)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced features test failed: {e}")
        return False


def test_performance_metrics():
    """Test performance metrics are available."""
    print("\n🧪 Testing performance metrics...")
    
    try:
        # Create sample data
        data = create_sample_data(500)
        targets = create_sample_targets(data)
        
        # Create pipeline
        pipeline = create_unified_pipeline()
        
        # Process data
        result = pipeline.process(
            data=data,
            targets=targets,
            feature_columns=None,
            timeframe="15m"
        )
        
        # Test performance metrics
        assert result.memory_usage_mb >= 0, "Memory usage should be non-negative"
        assert result.vectorbt_operations >= 0, "VectorBT operations should be non-negative"
        assert result.cache_hit_rate >= 0, "Cache hit rate should be non-negative"
        assert result.processing_time > 0, "Processing time should be positive"
        
        # Test performance stats
        stats = pipeline.get_performance_stats()
        assert 'total_pipeline_runs' in stats, "Should have total pipeline runs"
        assert 'successful_pipeline_runs' in stats, "Should have successful pipeline runs"
        assert 'vectorbt_operations' in stats, "Should have VectorBT operations"
        
        print("✅ Performance metrics test passed")
        print(f"   - Memory usage: {result.memory_usage_mb:.2f} MB")
        print(f"   - VectorBT operations: {result.vectorbt_operations}")
        print(f"   - Cache hit rate: {result.cache_hit_rate:.2%}")
        print(f"   - Processing time: {result.processing_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance metrics test failed: {e}")
        return False


def test_convenience_function():
    """Test convenience function works."""
    print("\n🧪 Testing convenience function...")
    
    try:
        # Create sample data
        data = create_sample_data(500)
        targets = create_sample_targets(data)
        
        # Use convenience function
        result = process_with_unified_pipeline(
            data=data,
            targets=targets,
            feature_columns=None,
            timeframe="15m"
        )
        
        # Validate result
        assert isinstance(result, ConsolidatedPipelineResult), "Result should be ConsolidatedPipelineResult"
        assert result.success, f"Pipeline should succeed, but got error: {result.error_message}"
        
        print("✅ Convenience function test passed")
        print(f"   - Selected {len(result.selected_features)} features")
        print(f"   - Processing time: {result.processing_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Convenience function test failed: {e}")
        return False


def test_configuration():
    """Test configuration system works."""
    print("\n🧪 Testing configuration system...")
    
    try:
        # Test default configuration
        config = UnifiedPipelineConfig()
        pipeline = UnifiedDataDrivenPipeline(config)
        
        # Test custom configuration
        custom_config = UnifiedPipelineConfig()
        custom_config.feature_selection.multi_objective.max_features = 20
        custom_pipeline = UnifiedDataDrivenPipeline(custom_config)
        
        print("✅ Configuration system test passed")
        print(f"   - Default config created successfully")
        print(f"   - Custom config created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration system test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Consolidated Unified Data-Driven Pipeline")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_enhanced_features,
        test_performance_metrics,
        test_convenience_function,
        test_configuration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Consolidated pipeline is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)