"""
Refactored Usage Example for Unified Data-Driven Pipeline

This example demonstrates how to use the refactored unified pipeline with
simplified configuration presets and modular stages.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the refactored pipeline
from ..refactored_pipeline import (
    RefactoredUnifiedPipeline,
    RefactoredPipelineResult,
    create_refactored_pipeline,
    create_full_pipeline,
    create_blank_pipeline,
    create_light_pipeline
)

# Import simplified configuration
from ..core.simplified_config import (
    create_full_config, create_blank_config, create_light_config,
    create_config_by_intensity, list_available_intensities
)


def create_sample_data(n_samples: int = 1000, n_features: int = 50) -> Tuple[pd.DataFrame, pd.Series]:
    """Create sample financial data for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        
    Returns:
        Tuple of (DataFrame, Series) containing sample data and targets
    """
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


def example_full_intensity():
    """Full intensity pipeline example (100%)."""
    print("\n" + "="*60)
    print("FULL INTENSITY PIPELINE EXAMPLE")
    print("="*60)
    
    # Create sample data
    data, targets = create_sample_data(n_samples=500, n_features=30)
    
    # Create full intensity pipeline
    print("\n1. Using full intensity pipeline...")
    pipeline = create_full_pipeline()
    
    # Process data
    result = pipeline.process(data, targets, timeframe="15m")
    
    print(f"\nResults:")
    print(f"- Selected features: {len(result.selected_features)}")
    print(f"- Processing time: {result.total_processing_time:.2f}s")
    print(f"- Memory usage: {result.memory_usage:.2f} MB")
    print(f"- Quality score: {result.quality_score:.3f}")
    print(f"- Warnings: {len(result.warnings)}")
    print(f"- Errors: {len(result.errors)}")
    
    # Show pipeline summary
    summary = pipeline.get_pipeline_summary()
    print(f"\nPipeline Summary:")
    print(f"- Pipeline type: {summary['pipeline_type']}")
    print(f"- Intensity: {summary['intensity']}")
    print(f"- Stages: {', '.join(summary['stages'])}")
    
    # Cleanup
    pipeline.cleanup()


def example_blank_intensity():
    """Blank intensity pipeline example (25%)."""
    print("\n" + "="*60)
    print("BLANK INTENSITY PIPELINE EXAMPLE (25%)")
    print("="*60)
    
    # Create sample data
    data, targets = create_sample_data(n_samples=800, n_features=40)
    
    # Create blank intensity pipeline
    print("\n2. Using blank intensity pipeline...")
    pipeline = create_blank_pipeline()
    
    # Process data
    result = pipeline.process(data, targets, timeframe="15m")
    
    print(f"\nResults:")
    print(f"- Selected features: {len(result.selected_features)}")
    print(f"- Processing time: {result.total_processing_time:.2f}s")
    print(f"- Memory usage: {result.memory_usage:.2f} MB")
    print(f"- Quality score: {result.quality_score:.3f}")
    
    # Show stage results
    if result.validation_result:
        print(f"- Validation quality: {result.validation_result.quality_score:.3f}")
    if result.generation_result:
        print(f"- Generation quality: {result.generation_result.quality_score:.3f}")
    if result.selection_result:
        print(f"- Selection quality: {result.selection_result.quality_score:.3f}")
    if result.optimization_result:
        print(f"- Optimization quality: {result.optimization_result.quality_score:.3f}")
    
    # Cleanup
    pipeline.cleanup()


def example_light_intensity():
    """Light intensity pipeline example (10%)."""
    print("\n" + "="*60)
    print("LIGHT INTENSITY PIPELINE EXAMPLE (10%)")
    print("="*60)
    
    # Create sample data
    data, targets = create_sample_data(n_samples=600, n_features=40)
    
    # Create light intensity pipeline
    print("\n3. Using light intensity pipeline...")
    pipeline = create_light_pipeline()
    
    # Process data
    result = pipeline.process(data, targets, timeframe="15m")
    
    print(f"\nResults:")
    print(f"- Selected features: {len(result.selected_features)}")
    print(f"- Processing time: {result.total_processing_time:.2f}s")
    print(f"- Memory usage: {result.memory_usage:.2f} MB")
    print(f"- Quality score: {result.quality_score:.3f}")
    
    # Cleanup
    pipeline.cleanup()


def example_custom_configuration():
    """Custom configuration example."""
    print("\n" + "="*60)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("="*60)
    
    # Create sample data
    data, targets = create_sample_data(n_samples=500, n_features=30)
    
    # Create custom configuration
    custom_overrides = {
        'feature_selection.multi_objective.max_features': 15,
        'feature_selection.multi_objective.min_features': 5,
        'period_optimization.max_period': 50
    }
    
    print("\n4. Using custom configuration...")
    print(f"Custom overrides: {custom_overrides}")
    
    # Create pipeline with custom configuration
    pipeline = create_refactored_pipeline(
        intensity="blank",
        custom_overrides=custom_overrides
    )
    
    # Process data
    result = pipeline.process(data, targets, timeframe="15m")
    
    print(f"\nResults:")
    print(f"- Selected features: {len(result.selected_features)}")
    print(f"- Processing time: {result.total_processing_time:.2f}s")
    print(f"- Quality score: {result.quality_score:.3f}")
    
    # Cleanup
    pipeline.cleanup()


def example_save_results():
    """Example of saving results."""
    print("\n" + "="*60)
    print("SAVING RESULTS EXAMPLE")
    print("="*60)
    
    # Create sample data
    data, targets = create_sample_data(n_samples=500, n_features=30)
    
    # Create pipeline
    pipeline = create_blank_pipeline()
    
    # Process data
    result = pipeline.process(data, targets, timeframe="15m")
    
    # Save results
    output_path = "refactored_pipeline_results"
    success = result.save_result(output_path)
    
    if success:
        print(f"\n5. Results saved to: {output_path}")
        print("Files created:")
        print("- processed_data.csv")
        print("- selected_features.csv")
        print("- metadata.json")
    else:
        print("Failed to save results")
    
    # Cleanup
    pipeline.cleanup()


def example_intensity_comparison():
    """Compare different intensity levels."""
    print("\n" + "="*60)
    print("INTENSITY COMPARISON EXAMPLE")
    print("="*60)
    
    # Create sample data
    data, targets = create_sample_data(n_samples=400, n_features=25)
    
    intensities = ["light", "blank", "full"]
    results = {}
    
    for intensity in intensities:
        print(f"\nTesting {intensity} intensity...")
        
        # Create pipeline
        pipeline = create_refactored_pipeline(intensity=intensity)
        
        # Process data
        result = pipeline.process(data, targets, timeframe="15m")
        
        # Store results
        results[intensity] = {
            'features': len(result.selected_features),
            'time': result.total_processing_time,
            'memory': result.memory_usage,
            'quality': result.quality_score
        }
        
        # Cleanup
        pipeline.cleanup()
    
    # Print comparison
    print(f"\nIntensity Comparison:")
    print(f"{'Intensity':<10} {'Features':<10} {'Time (s)':<10} {'Memory (MB)':<12} {'Quality':<8}")
    print("-" * 60)
    
    for intensity, metrics in results.items():
        print(f"{intensity:<10} {metrics['features']:<10} {metrics['time']:<10.2f} {metrics['memory']:<12.2f} {metrics['quality']:<8.3f}")


def example_available_intensities():
    """Show available intensity levels."""
    print("\n" + "="*60)
    print("AVAILABLE INTENSITY LEVELS")
    print("="*60)
    
    intensities = list_available_intensities()
    
    print("\n6. Available intensity levels:")
    for intensity, description in intensities.items():
        print(f"  - {intensity}: {description}")


def example_error_handling():
    """Example of error handling."""
    print("\n" + "="*60)
    print("ERROR HANDLING EXAMPLE")
    print("="*60)
    
    print("\n7. Testing error handling...")
    
    # Test with invalid data
    try:
        pipeline = create_light_pipeline()
        result = pipeline.process(None, None)
        print("ERROR: Should have failed with None data")
    except Exception as e:
        print(f"✓ Correctly caught error: {e}")
    
    # Test with mismatched lengths
    try:
        data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        targets = pd.Series([1, 2])  # Different length
        pipeline = create_light_pipeline()
        result = pipeline.process(data, targets)
        print("ERROR: Should have failed with mismatched lengths")
    except Exception as e:
        print(f"✓ Correctly caught error: {e}")
    
    # Test with valid data
    try:
        data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        targets = pd.Series([0.1, 0.2, 0.3])
        pipeline = create_light_pipeline()
        result = pipeline.process(data, targets)
        print(f"✓ Successfully processed valid data: {len(result.selected_features)} features selected")
        pipeline.cleanup()
    except Exception as e:
        print(f"ERROR: Unexpected error with valid data: {e}")


def main():
    """Run all examples."""
    print("REFACTORED UNIFIED DATA-DRIVEN PIPELINE EXAMPLES")
    print("=" * 60)
    
    try:
        # Run examples
        example_full_intensity()
        example_blank_intensity()
        example_light_intensity()
        example_custom_configuration()
        example_save_results()
        example_intensity_comparison()
        example_available_intensities()
        example_error_handling()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()