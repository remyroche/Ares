#!/usr/bin/env python3
"""
Example usage of the Multi-Stage Feature Selection Pipeline.

This script demonstrates how to use the MultiStageFeatureSelectionPipeline
class for feature selection in different scenarios.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.training.steps.pre_training.feature_selection import (
    MultiStageFeatureSelectionPipeline,
    run_multi_stage_feature_selection,
    FeatureSelectionConfig
)


def create_sample_data(n_samples: int = 1000, n_features: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    """Create sample data for demonstration."""
    np.random.seed(42)
    
    # Create feature matrix
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i:03d}" for i in range(n_features)]
    )
    
    # Create target variable with some features being more important
    y = (
        2 * X.iloc[:, 0] +  # Most important feature
        1.5 * X.iloc[:, 1] +  # Second most important
        1.0 * X.iloc[:, 2] +  # Third most important
        0.5 * X.iloc[:, 3] +  # Fourth most important
        np.random.randn(n_samples) * 0.1  # Noise
    )
    
    return X, y


def example_1_basic_usage():
    """Example 1: Basic usage with default configuration."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage with Default Configuration")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data(n_samples=1000, n_features=120)
    print(f"Created sample data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Use the convenience function
    result = run_multi_stage_feature_selection(
        X=X,
        y=y,
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="15m"
    )
    
    print(f"\nFeature selection completed:")
    print(f"  - Success: {result.success}")
    print(f"  - Selected features: {len(result.selected_features)}")
    print(f"  - Execution time: {result.execution_time:.2f}s")
    print(f"  - Stage results: {list(result.stage_results.keys())}")
    
    print(f"\nFirst 10 selected features:")
    for i, feature in enumerate(result.selected_features[:10], 1):
        print(f"  {i:2d}. {feature}")


def example_2_custom_configuration():
    """Example 2: Using custom configuration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data(n_samples=800, n_features=100)
    print(f"Created sample data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Create custom configuration
    config = FeatureSelectionConfig(
        target_features=50,  # Reduce target features
        enable_vectorbt_optimization=True,
        vectorbt_memory_efficient=True,
        vectorbt_chunk_size=500,
        stage1_mrmr_weight=0.8,  # Increase mRMR weight
        stage1_spearman_weight=0.2,  # Decrease Spearman weight
        rfe_step_size=0.15,  # Increase RFE step size
        bootstrap_n_samples=50  # Reduce bootstrap samples
    )
    
    # Use the pipeline class directly
    pipeline = MultiStageFeatureSelectionPipeline(config)
    
    try:
        result = pipeline.select_features(
            X=X,
            y=y,
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="5m"
        )
        
        print(f"\nFeature selection completed:")
        print(f"  - Success: {result.success}")
        print(f"  - Selected features: {len(result.selected_features)}")
        print(f"  - Execution time: {result.execution_time:.2f}s")
        print(f"  - Target features: {config.target_features}")
        
        print(f"\nStage 1 results:")
        stage1 = result.stage_results.get('stage_1', {})
        print(f"  - Method: {stage1.get('method', 'unknown')}")
        print(f"  - Target count: {stage1.get('target_count', 0)}")
        
        print(f"\nStage 2 results:")
        stage2 = result.stage_results.get('stage_2', {})
        print(f"  - Method: {stage2.get('method', 'unknown')}")
        print(f"  - Bootstrap/CV used: {stage2.get('use_bootstrap_cv', False)}")
        
    finally:
        pipeline.cleanup()


def example_3_error_handling():
    """Example 3: Error handling with fast fail."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Error Handling with Fast Fail")
    print("=" * 60)
    
    # Test with invalid data
    print("Testing with empty feature matrix...")
    try:
        X_empty = pd.DataFrame()
        y = pd.Series([1, 2, 3])
        
        result = run_multi_stage_feature_selection(X_empty, y)
        print(f"Unexpected success: {result.success}")
    except ValueError as e:
        print(f"✅ Fast fail caught error: {e}")
    
    # Test with mismatched dimensions
    print("\nTesting with mismatched dimensions...")
    try:
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.random.randn(50))  # Different length
        
        result = run_multi_stage_feature_selection(X, y)
        print(f"Unexpected success: {result.success}")
    except ValueError as e:
        print(f"✅ Fast fail caught error: {e}")


def example_4_performance_comparison():
    """Example 4: Performance comparison with different configurations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Performance Comparison")
    print("=" * 60)
    
    # Create larger dataset
    X, y = create_sample_data(n_samples=2000, n_features=150)
    print(f"Created larger dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test with VectorBT optimization enabled
    print("\nTesting with VectorBT optimization enabled...")
    config_with_vectorbt = FeatureSelectionConfig(
        enable_vectorbt_optimization=True,
        target_features=80
    )
    
    start_time = pd.Timestamp.now()
    result_with_vectorbt = run_multi_stage_feature_selection(
        X, y, config=config_with_vectorbt
    )
    time_with_vectorbt = (pd.Timestamp.now() - start_time).total_seconds()
    
    print(f"  - Success: {result_with_vectorbt.success}")
    print(f"  - Selected features: {len(result_with_vectorbt.selected_features)}")
    print(f"  - Execution time: {time_with_vectorbt:.2f}s")
    
    # Test with VectorBT optimization disabled
    print("\nTesting with VectorBT optimization disabled...")
    config_without_vectorbt = FeatureSelectionConfig(
        enable_vectorbt_optimization=False,
        target_features=80
    )
    
    start_time = pd.Timestamp.now()
    result_without_vectorbt = run_multi_stage_feature_selection(
        X, y, config=config_without_vectorbt
    )
    time_without_vectorbt = (pd.Timestamp.now() - start_time).total_seconds()
    
    print(f"  - Success: {result_without_vectorbt.success}")
    print(f"  - Selected features: {len(result_without_vectorbt.selected_features)}")
    print(f"  - Execution time: {time_without_vectorbt:.2f}s")
    
    print(f"\nPerformance comparison:")
    print(f"  - VectorBT enabled: {time_with_vectorbt:.2f}s")
    print(f"  - VectorBT disabled: {time_without_vectorbt:.2f}s")
    print(f"  - Speedup: {time_without_vectorbt/time_with_vectorbt:.2f}x")


def main():
    """Run all examples."""
    print("Multi-Stage Feature Selection Pipeline Examples")
    print("=" * 60)
    
    try:
        example_1_basic_usage()
        example_2_custom_configuration()
        example_3_error_handling()
        example_4_performance_comparison()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()