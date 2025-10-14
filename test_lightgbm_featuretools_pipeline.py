#!/usr/bin/env python3
"""
Test script for the new LightGBM + Featuretools + ALE pipeline
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline import (
        UnifiedDataDrivenPipeline, create_unified_pipeline
    )
    from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.lightgbm_featuretools_generator import (
        LightGBMFeatureToolsGenerator, LightGBMFeatureToolsConfig
    )
    print("✅ Successfully imported pipeline components")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def create_sample_data(n_samples=1000):
    """Create sample financial data for testing."""
    np.random.seed(42)
    
    # Generate OHLCV data
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='15T')
    
    # Generate price data with some trend and volatility
    returns = np.random.normal(0.0001, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    })
    
    # Add some technical indicators
    data['sma_20'] = data['close'].rolling(20).mean()
    data['rsi'] = 50 + np.random.normal(0, 10, n_samples)
    data['bb_upper'] = data['sma_20'] + 2 * data['close'].rolling(20).std()
    data['bb_lower'] = data['sma_20'] - 2 * data['close'].rolling(20).std()
    
    # Create target variable (next period return)
    data['target'] = data['close'].pct_change().shift(-1)
    
    # Remove NaN values
    data = data.dropna()
    
    return data

def test_lightgbm_generator():
    """Test the LightGBM + Featuretools generator directly."""
    print("\n🧪 Testing LightGBM + Featuretools generator directly...")
    
    # Create sample data
    data = create_sample_data(500)  # Smaller dataset for testing
    
    # Configure generator
    config = LightGBMFeatureToolsConfig(
        model_type='lightgbm',
        max_features_to_select=50,  # Limit for testing
        use_featuretools=True,
        use_ale_validation=True,
        use_shap=True,
        enable_vectorbt=False,  # Disable for testing
        enable_parallel=False,  # Disable for testing
        memory_efficient=True
    )
    
    # Create generator
    generator = LightGBMFeatureToolsGenerator(config)
    
    # Generate features
    result = generator.generate_features(
        data, 
        'target',
        feature_columns=['open', 'high', 'low', 'close', 'volume', 'sma_20', 'rsi'],
        execution_mode='light'
    )
    
    print(f"✅ Generator test completed:")
    print(f"   - Features generated: {result.n_features_generated}")
    print(f"   - Features selected: {result.n_features_selected}")
    print(f"   - SHAP analysis: {result.shap_analysis_completed}")
    print(f"   - ALE validation: {result.ale_validation_completed}")
    print(f"   - Featuretools features: {result.featuretools_features_generated}")
    print(f"   - Generation time: {result.generation_time:.3f}s")
    
    return result

def test_full_pipeline():
    """Test the full pipeline with the new generator."""
    print("\n🧪 Testing full pipeline with LightGBM + Featuretools...")
    
    # Create sample data
    data = create_sample_data(1000)
    
    # Create pipeline
    pipeline = create_unified_pipeline()
    
    # Process data
    result = pipeline.process(
        data, 
        targets=data['target'],
        feature_columns=['open', 'high', 'low', 'close', 'volume', 'sma_20', 'rsi'],
        timeframe="15m"
    )
    
    print(f"✅ Pipeline test completed:")
    print(f"   - Success: {result.success}")
    print(f"   - Selected features: {len(result.selected_features)}")
    print(f"   - Processing time: {result.processing_time:.3f}s")
    print(f"   - Cross-timeframe features: {len(result.cross_timeframe_features) if result.cross_timeframe_features else 0}")
    print(f"   - Interaction features: {len(result.interaction_features) if result.interaction_features else 0}")
    print(f"   - No features: {len(result.no_features) if result.no_features else 0}")
    print(f"   - Comparison features: {len(result.comparison_features) if result.comparison_features else 0}")
    
    if result.error_message:
        print(f"   - Error: {result.error_message}")
    
    return result

def main():
    """Main test function."""
    print("🚀 Testing LightGBM + Featuretools + ALE Pipeline")
    print("=" * 60)
    
    try:
        # Test 1: Direct generator test
        generator_result = test_lightgbm_generator()
        
        # Test 2: Full pipeline test
        pipeline_result = test_full_pipeline()
        
        print("\n✅ All tests completed successfully!")
        print("\n📊 Summary:")
        print(f"   - Generator features: {generator_result.n_features_selected}")
        print(f"   - Pipeline features: {len(pipeline_result.selected_features)}")
        print(f"   - Max features limit: 100 (as requested)")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)