"""
Usage Example for Unified Data-Driven Feature Pipeline

This example demonstrates how to use the unified pipeline for feature
engineering and selection with proper time series validation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the consolidated unified pipeline
from .. import (
    UnifiedDataDrivenPipeline,
    ConsolidatedPipelineResult,
    create_unified_pipeline,
    process_with_unified_pipeline,
    create_default_config
)

def create_sample_data(n_samples: int = 1000, n_features: int = 50) -> Tuple[pd.DataFrame, pd.Series]:
    """Create sample financial data for demonstration."""
    print("Creating sample financial data...")
    
    # Create date index
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Create features with different characteristics
    np.random.seed(42)
    features = {}
    
    # Price features (trending)
    features['price'] = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)
    features['high'] = features['price'] + np.abs(np.random.randn(n_samples) * 0.5)
    features['low'] = features['price'] - np.abs(np.random.randn(n_samples) * 0.5)
    features['open'] = features['price'] + np.random.randn(n_samples) * 0.1
    
    # Volatility features
    features['volatility'] = np.abs(np.random.randn(n_samples) * 0.02)
    features['realized_vol'] = features['volatility'].rolling(20).std().fillna(0.01)
    
    # Momentum features
    features['momentum_5'] = features['price'].pct_change(5)
    features['momentum_20'] = features['price'].pct_change(20)
    features['rsi'] = 50 + np.random.randn(n_samples) * 10  # Simulated RSI
    
    # Volume features
    features['volume'] = np.random.lognormal(10, 1, n_samples)
    features['volume_ma'] = features['volume'].rolling(20).mean().fillna(features['volume'].mean())
    
    # Technical indicators
    features['sma_20'] = features['price'].rolling(20).mean().fillna(features['price'])
    features['sma_50'] = features['price'].rolling(50).mean().fillna(features['price'])
    features['bb_upper'] = features['sma_20'] + 2 * features['realized_vol']
    features['bb_lower'] = features['sma_20'] - 2 * features['realized_vol']
    
    # Add some noise features
    for i in range(n_features - len(features)):
        features[f'noise_{i}'] = np.random.randn(n_samples)
    
    # Create DataFrame
    data = pd.DataFrame(features, index=dates)
    
    # Create targets (returns)
    targets = data['price'].pct_change().dropna()
    data = data.iloc[1:]  # Align with targets
    
    print(f"Created data with shape: {data.shape}")
    print(f"Targets shape: {targets.shape}")
    
    return data, targets


def example_basic_usage():
    """Basic usage example."""
    print("\n" + "="*60)
    print("BASIC USAGE EXAMPLE")
    print("="*60)
    
    # Create sample data
    data, targets = create_sample_data(n_samples=500, n_features=30)
    
    # Use default configuration
    print("\n1. Using default configuration...")
    result = process_features(data, targets)
    
    print(f"\nResults:")
    print(f"- Selected features: {len(result.selected_features)}")
    print(f"- Processing time: {result.processing_time:.2f}s")
    print(f"- Out-of-sample Sharpe: {result.out_of_sample_sharpe:.3f}")
    print(f"- Max drawdown: {result.max_drawdown:.3f}")
    print(f"- Stability score: {result.stability_score:.3f}")
    print(f"- Diversity score: {result.diversity_score:.3f}")
    
    print(f"\nSelected features: {result.selected_features[:10]}...")  # Show first 10


def example_custom_configuration():
    """Custom configuration example."""
    print("\n" + "="*60)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("="*60)
    
    # Create sample data
    data, targets = create_sample_data(n_samples=800, n_features=40)
    
    # Create custom configuration
    config = create_default_config()
    
    # Modify configuration
    config.feature_selection.multi_objective.max_features = 20
    config.feature_selection.multi_objective.min_features = 5
    config.feature_selection.cv_config.n_splits = 3
    
    # Adjust objective weights
    config.feature_selection.multi_objective.objectives = {
        'out_of_sample_sharpe': 0.4,
        'drawdown': 0.3,
        'stability': 0.2,
        'diversity': 0.1
    }
    
    print("\n2. Using custom configuration...")
    print(f"Max features: {config.feature_selection.multi_objective.max_features}")
    print(f"Min features: {config.feature_selection.multi_objective.min_features}")
    print(f"CV splits: {config.feature_selection.cv_config.n_splits}")
    print(f"Objective weights: {config.feature_selection.multi_objective.objectives}")
    
    # Create pipeline with custom config
    pipeline = create_unified_pipeline(config)
    result = pipeline.process(data, targets, feature_columns=None, timeframe="15m")
    
    print(f"\nResults:")
    print(f"- Selected features: {len(result.selected_features)}")
    print(f"- Processing time: {result.processing_time:.2f}s")
    print(f"- Objective values: {result.objective_values}")


def example_high_performance_config():
    """High performance configuration example."""
    print("\n" + "="*60)
    print("HIGH PERFORMANCE CONFIGURATION EXAMPLE")
    print("="*60)
    
    # Create larger sample data
    data, targets = create_sample_data(n_samples=2000, n_features=100)
    
    # Use high performance configuration
    config = create_high_performance_config()
    
    print("\n3. Using high performance configuration...")
    print(f"VectorBT GPU enabled: {config.vectorization.enable_gpu}")
    print(f"Parallel processing: {config.vectorization.enable_parallel}")
    
    # Create pipeline
    pipeline = create_unified_pipeline(config)
    result = pipeline.process(data, targets, feature_columns=None, timeframe="15m")
    
    print(f"\nResults:")
    print(f"- Selected features: {len(result.selected_features)}")
    print(f"- Processing time: {result.processing_time:.2f}s")
    print(f"- Performance stats: {pipeline.get_performance_stats()}")


def example_memory_efficient_config():
    """Memory efficient configuration example."""
    print("\n" + "="*60)
    print("MEMORY EFFICIENT CONFIGURATION EXAMPLE")
    print("="*60)
    
    # Create sample data
    data, targets = create_sample_data(n_samples=1000, n_features=60)
    
    # Use memory efficient configuration
    config = create_memory_efficient_config()
    
    print("\n4. Using memory efficient configuration...")
    print(f"Memory limit: {config.vectorization.max_memory_gb}GB")
    print(f"Chunk size: {config.vectorization.chunk_size}")
    print(f"Memory efficient: {config.vectorization.memory_efficient}")
    
    # Create pipeline
    pipeline = create_unified_pipeline(config)
    result = pipeline.process(data, targets, feature_columns=None, timeframe="15m")
    
    print(f"\nResults:")
    print(f"- Selected features: {len(result.selected_features)}")
    print(f"- Processing time: {result.processing_time:.2f}s")


def example_fast_config():
    """Fast configuration example."""
    print("\n" + "="*60)
    print("FAST CONFIGURATION EXAMPLE")
    print("="*60)
    
    # Create sample data
    data, targets = create_sample_data(n_samples=600, n_features=40)
    
    # Use fast configuration
    config = create_fast_config()
    
    print("\n5. Using fast configuration...")
    print(f"Max features: {config.feature_selection.multi_objective.max_features}")
    print(f"CV splits: {config.feature_selection.cv_config.n_splits}")
    print(f"Test size: {config.feature_selection.cv_config.test_size}")
    
    # Create pipeline
    pipeline = create_unified_pipeline(config)
    result = pipeline.process(data, targets, feature_columns=None, timeframe="15m")
    
    print(f"\nResults:")
    print(f"- Selected features: {len(result.selected_features)}")
    print(f"- Processing time: {result.processing_time:.2f}s")


def example_save_results():
    """Example of saving results."""
    print("\n" + "="*60)
    print("SAVING RESULTS EXAMPLE")
    print("="*60)
    
    # Create sample data
    data, targets = create_sample_data(n_samples=500, n_features=30)
    
    # Process features
    result = process_features(data, targets)
    
    # Save results
    output_path = "pipeline_results"
    result.save_result(result, output_path)
    
    print(f"\n6. Results saved to: {output_path}")
    print("Files created:")
    print("- selected_features.csv")
    print("- objective_values.csv")
    print("- metadata.json")


def example_validation():
    """Example of validation and error handling."""
    print("\n" + "="*60)
    print("VALIDATION EXAMPLE")
    print("="*60)
    
    print("\n7. Testing validation...")
    
    # Test with invalid data
    try:
        result = process_features(None, None)
        print("ERROR: Should have failed with None data")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    # Test with mismatched lengths
    try:
        data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        targets = pd.Series([1, 2])  # Different length
        result = process_features(data, targets)
        print("ERROR: Should have failed with mismatched lengths")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    # Test with valid data
    try:
        data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        targets = pd.Series([0.1, 0.2, 0.3])
        result = process_features(data, targets)
        print(f"✓ Successfully processed valid data: {len(result.selected_features)} features selected")
    except Exception as e:
        print(f"ERROR: Unexpected error with valid data: {e}")


def main():
    """Run all examples."""
    print("UNIFIED DATA-DRIVEN FEATURE PIPELINE EXAMPLES")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_usage()
        example_custom_configuration()
        example_high_performance_config()
        example_memory_efficient_config()
        example_fast_config()
        example_save_results()
        example_validation()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()