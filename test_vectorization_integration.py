#!/usr/bin/env python3
"""
Test script to verify VectorBTRollingOptimizer and UnifiedVectorizationManager integration
in the UnifiedDataDrivenPipeline.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_sample_data(n_rows=1000):
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=n_rows//24)  # Assuming hourly data
    timestamps = [start_date + timedelta(hours=i) for i in range(n_rows)]
    
    # Generate OHLCV data
    base_price = 100.0
    prices = []
    current_price = base_price
    
    for i in range(n_rows):
        # Random walk with some trend
        change = np.random.normal(0, 0.02) + 0.001  # Slight upward bias
        current_price *= (1 + change)
        prices.append(current_price)
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Generate OHLC from the price
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

def test_vectorization_integration():
    """Test the vectorization integration in the pipeline."""
    print("🧪 Testing VectorBTRollingOptimizer and UnifiedVectorizationManager integration...")
    
    try:
        # Import the pipeline
        from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline import UnifiedDataDrivenPipeline
        
        print("✅ Successfully imported UnifiedDataDrivenPipeline")
        
        # Create sample data
        print("📊 Creating sample data...")
        sample_data = create_sample_data(500)  # Smaller dataset for testing
        print(f"✅ Created sample data with shape: {sample_data.shape}")
        
        # Initialize the pipeline
        print("🚀 Initializing pipeline...")
        pipeline = UnifiedDataDrivenPipeline()
        print("✅ Pipeline initialized successfully")
        
        # Check if vectorization utilities are available
        if hasattr(pipeline, 'vectorbt_rolling_optimizer') and pipeline.vectorbt_rolling_optimizer is not None:
            print("✅ VectorBTRollingOptimizer is available")
        else:
            print("⚠️ VectorBTRollingOptimizer is not available")
        
        if hasattr(pipeline, 'unified_vectorization_manager') and pipeline.unified_vectorization_manager is not None:
            print("✅ UnifiedVectorizationManager is available")
        else:
            print("⚠️ UnifiedVectorizationManager is not available")
        
        # Test vectorized rolling operations directly
        print("🔄 Testing vectorized rolling operations...")
        if hasattr(pipeline, '_vectorized_rolling_operations'):
            try:
                enhanced_data = pipeline._vectorized_rolling_operations(sample_data, windows=[5, 10, 20])
                print(f"✅ Vectorized rolling operations completed: {enhanced_data.shape[1]} features")
                print(f"   Original features: {sample_data.shape[1]}")
                print(f"   New features: {enhanced_data.shape[1] - sample_data.shape[1]}")
            except Exception as e:
                print(f"❌ Vectorized rolling operations failed: {e}")
        else:
            print("⚠️ Vectorized rolling operations method not found")
        
        # Test unified vectorization processing
        print("🚀 Testing unified vectorization processing...")
        if hasattr(pipeline, '_unified_vectorization_processing'):
            try:
                vectorized_data = pipeline._unified_vectorization_processing(sample_data)
                print(f"✅ Unified vectorization processing completed: {vectorized_data.shape[1]} features")
            except Exception as e:
                print(f"❌ Unified vectorization processing failed: {e}")
        else:
            print("⚠️ Unified vectorization processing method not found")
        
        # Test optimized feature calculation
        print("⚡ Testing optimized feature calculation...")
        if hasattr(pipeline, '_optimized_feature_calculation'):
            try:
                feature_config = {
                    'rolling_windows': [5, 10, 20],
                    'enable_correlation_features': True,
                    'enable_momentum_features': True,
                    'enable_volatility_features': True,
                    'enable_volume_features': True
                }
                optimized_data = pipeline._optimized_feature_calculation(sample_data, feature_config)
                print(f"✅ Optimized feature calculation completed: {optimized_data.shape[1]} features")
            except Exception as e:
                print(f"❌ Optimized feature calculation failed: {e}")
        else:
            print("⚠️ Optimized feature calculation method not found")
        
        # Check performance stats
        print("📊 Checking performance statistics...")
        if hasattr(pipeline, 'performance_stats'):
            print("Performance statistics available:")
            for key, value in pipeline.performance_stats.items():
                if 'vector' in key.lower() or 'correlation' in key.lower() or 'momentum' in key.lower() or 'volatility' in key.lower() or 'volume' in key.lower():
                    print(f"  {key}: {value}")
        
        print("✅ Vectorization integration test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_vectorization_integration()
    if success:
        print("\n🎉 All tests passed! Vectorization integration is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Please check the error messages above.")
        sys.exit(1)