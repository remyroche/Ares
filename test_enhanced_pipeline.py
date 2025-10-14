"""
Test script for Enhanced Unified Data-Driven Pipeline

This script demonstrates the comprehensive functionality of the enhanced pipeline
with all integrated tools and utilities.
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
import warnings

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import the enhanced pipeline
from enhanced_unified_data_driven_pipeline import (
    EnhancedUnifiedDataDrivenPipeline, EnhancedPipelineConfig,
    create_enhanced_pipeline, process_data_with_enhanced_pipeline,
    LogLevel
)

def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample financial data for testing."""
    print("📊 Creating sample financial data...")
    
    # Generate realistic financial data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    
    # Generate price data with trend and volatility
    base_price = 100
    trend = np.linspace(0, 20, n_samples)  # Upward trend
    noise = np.random.randn(n_samples) * 2  # Random noise
    prices = base_price + trend + noise
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.randn(n_samples) * 0.5,
        'high': prices + np.abs(np.random.randn(n_samples)) * 2,
        'low': prices - np.abs(np.random.randn(n_samples)) * 2,
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Ensure high >= low and high/low contain open/close
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    print(f"✅ Created sample data with {len(data)} rows and {len(data.columns)} columns")
    return data

def test_basic_functionality():
    """Test basic pipeline functionality."""
    print("\n🧪 Testing Basic Functionality")
    print("=" * 50)
    
    # Create sample data
    data = create_sample_data(500)
    
    # Create pipeline with basic config
    config = EnhancedPipelineConfig(
        log_level=LogLevel.INFO,
        fail_fast=True,
        enable_vectorbt_optimization=True,
        enable_unified_vectorization=True,
        enable_comprehensive_validation=True
    )
    
    pipeline = create_enhanced_pipeline(config)
    
    try:
        # Test feature engineering
        print("\n🔧 Testing Feature Engineering...")
        result = pipeline.process_data(data, operation_type="feature_engineering")
        print(f"✅ Feature engineering completed")
        print(f"   - Features shape: {result['features'].shape if 'features' in result else 'N/A'}")
        print(f"   - Optimization used: {result.get('optimization_used', 'N/A')}")
        
        # Test backtesting
        print("\n📊 Testing Backtesting...")
        result = pipeline.process_data(data, operation_type="backtesting")
        print(f"✅ Backtesting completed")
        print(f"   - Strategy used: {result.get('strategy_used', 'N/A')}")
        print(f"   - Performance gain: {result.get('performance_gain', 'N/A')}")
        
        # Test cross-validation
        print("\n🔄 Testing Cross-Validation...")
        result = pipeline.process_data(data, operation_type="cross_validation")
        print(f"✅ Cross-validation completed")
        print(f"   - CV results: {result.get('cv_results', 'N/A')}")
        
        # Test VectorBT optimization
        print("\n⚡ Testing VectorBT Optimization...")
        result = pipeline.process_data(data, operation_type="vectorbt_optimization")
        print(f"✅ VectorBT optimization completed")
        print(f"   - Optimization strategy: {result.get('optimization_strategy', 'N/A')}")
        
        # Get pipeline status
        print("\n📊 Getting Pipeline Status...")
        status = pipeline.get_pipeline_status()
        print(f"✅ Pipeline status retrieved")
        print(f"   - Components available: {sum(status['components_available'].values())}/{len(status['components_available'])}")
        print(f"   - Validation success rate: {status['validation_summary']['success_rate']:.2%}")
        print(f"   - Error count: {status['error_count']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pipeline.cleanup()

def test_error_handling():
    """Test error handling and validation."""
    print("\n🚨 Testing Error Handling")
    print("=" * 50)
    
    # Create pipeline with strict validation
    config = EnhancedPipelineConfig(
        fail_fast=True,
        strict_validation=True,
        validate_inputs=True,
        validate_outputs=True
    )
    
    pipeline = create_enhanced_pipeline(config)
    
    try:
        # Test with invalid data (empty DataFrame)
        print("\n🔍 Testing with empty DataFrame...")
        empty_data = pd.DataFrame()
        try:
            pipeline.process_data(empty_data, operation_type="feature_engineering")
            print("❌ Should have failed with empty data")
        except Exception as e:
            print(f"✅ Correctly caught error: {type(e).__name__}")
        
        # Test with data containing NaN values
        print("\n🔍 Testing with NaN values...")
        data_with_nan = create_sample_data(100)
        data_with_nan.loc[50:60, 'close'] = np.nan
        try:
            pipeline.process_data(data_with_nan, operation_type="feature_engineering")
            print("❌ Should have failed with NaN values")
        except Exception as e:
            print(f"✅ Correctly caught error: {type(e).__name__}")
        
        # Test with wrong data type
        print("\n🔍 Testing with wrong data type...")
        try:
            pipeline.process_data("not a dataframe", operation_type="feature_engineering")
            print("❌ Should have failed with wrong data type")
        except Exception as e:
            print(f"✅ Correctly caught error: {type(e).__name__}")
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    finally:
        pipeline.cleanup()

def test_performance_monitoring():
    """Test performance monitoring capabilities."""
    print("\n⏱️ Testing Performance Monitoring")
    print("=" * 50)
    
    # Create pipeline with performance monitoring
    config = EnhancedPipelineConfig(
        enable_performance_monitoring=True,
        log_level=LogLevel.PERFORMANCE
    )
    
    pipeline = create_enhanced_pipeline(config)
    
    try:
        # Create larger dataset for performance testing
        data = create_sample_data(2000)
        
        # Test multiple operations to see performance metrics
        operations = [
            "feature_engineering",
            "backtesting", 
            "cross_validation",
            "vectorbt_optimization"
        ]
        
        for operation in operations:
            print(f"\n🔄 Testing {operation} performance...")
            start_time = datetime.now()
            
            result = pipeline.process_data(data, operation_type=operation)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"✅ {operation} completed in {duration:.3f}s")
        
        # Get performance metrics
        status = pipeline.get_pipeline_status()
        print(f"\n📊 Performance Metrics:")
        for operation, metrics in status['performance_metrics'].items():
            print(f"   - {operation}: {metrics.get('duration', 'N/A'):.3f}s")
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
    
    finally:
        pipeline.cleanup()

def test_configuration_options():
    """Test different configuration options."""
    print("\n⚙️ Testing Configuration Options")
    print("=" * 50)
    
    # Test with different configurations
    configs = [
        ("Minimal", EnhancedPipelineConfig(
            enable_vectorbt_optimization=False,
            enable_unified_vectorization=False,
            enable_comprehensive_validation=False,
            fail_fast=False
        )),
        ("Maximum", EnhancedPipelineConfig(
            enable_vectorbt_optimization=True,
            enable_unified_vectorization=True,
            enable_comprehensive_validation=True,
            enable_performance_monitoring=True,
            enable_caching=True,
            enable_data_quality_checks=True,
            fail_fast=True
        )),
        ("Balanced", EnhancedPipelineConfig(
            enable_vectorbt_optimization=True,
            enable_unified_vectorization=True,
            enable_comprehensive_validation=True,
            fail_fast=True,
            memory_limit_mb=2048
        ))
    ]
    
    data = create_sample_data(500)
    
    for config_name, config in configs:
        print(f"\n🔧 Testing {config_name} Configuration...")
        
        pipeline = create_enhanced_pipeline(config)
        
        try:
            result = pipeline.process_data(data, operation_type="feature_engineering")
            status = pipeline.get_pipeline_status()
            
            print(f"✅ {config_name} configuration test passed")
            print(f"   - Components available: {sum(status['components_available'].values())}")
            print(f"   - Validation errors: {status['validation_summary']['error_count']}")
            
        except Exception as e:
            print(f"❌ {config_name} configuration test failed: {e}")
        
        finally:
            pipeline.cleanup()

def test_convenience_functions():
    """Test convenience functions."""
    print("\n🛠️ Testing Convenience Functions")
    print("=" * 50)
    
    data = create_sample_data(300)
    
    try:
        # Test process_data_with_enhanced_pipeline
        print("\n🔄 Testing process_data_with_enhanced_pipeline...")
        result = process_data_with_enhanced_pipeline(
            data, 
            operation_type="feature_engineering"
        )
        print(f"✅ Convenience function test passed")
        print(f"   - Result type: {type(result)}")
        print(f"   - Keys: {list(result.keys())}")
        
    except Exception as e:
        print(f"❌ Convenience function test failed: {e}")

def main():
    """Run all tests."""
    print("🚀 Enhanced Unified Data-Driven Pipeline Test Suite")
    print("=" * 60)
    
    # Run all tests
    test_basic_functionality()
    test_error_handling()
    test_performance_monitoring()
    test_configuration_options()
    test_convenience_functions()
    
    print("\n🎉 All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()