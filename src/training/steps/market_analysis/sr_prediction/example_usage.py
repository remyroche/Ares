"""
Example Usage of SR Performance Predictor

Demonstrates training and prediction workflows.
"""

import asyncio
import pandas as pd
from pathlib import Path

from src.training.steps.market_analysis.sr_prediction import (
    SRPerformancePredictor,
    SRTrainingDataBuilder
)


async def example_basic_training():
    """Example: Basic training workflow."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Training")
    print("=" * 80)
    
    # Initialize data builder
    builder = SRTrainingDataBuilder()
    
    # Collect training data
    print("\n1. Collecting training data...")
    data = await builder.collect_data(
        symbol='BTCUSDT',
        exchange='binance',
        start_date='2023-01-01',
        end_date='2023-06-01',
        timeframe='1h',
        forward_days=10,
        sample_freq_days=7
    )
    
    # Check data quality
    print("\n2. Checking data quality...")
    stats = builder.check_data_quality(data)
    
    # Filter untested levels
    print("\n3. Filtering untested levels...")
    data = builder.filter_untested_levels(data)
    
    # Train model
    print("\n4. Training model...")
    predictor = SRPerformancePredictor()
    metrics = predictor.train(data, n_folds=3, num_boost_round=500)
    
    print("\n5. Training complete!")
    print("Metrics:")
    for target, target_metrics in metrics.items():
        print(f"\n{target}:")
        for metric, value in target_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Save model
    save_dir = Path('outputs/sr_prediction/example')
    predictor.save(save_dir)
    print(f"\n6. Model saved to {save_dir}")
    
    return predictor, data


async def example_multi_symbol_training():
    """Example: Multi-symbol training for better generalization."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Multi-Symbol Training")
    print("=" * 80)
    
    builder = SRTrainingDataBuilder()
    
    # Collect data for multiple symbols
    print("\n1. Collecting multi-symbol data...")
    data = await builder.collect_multi_symbol(
        symbols=['BTCUSDT', 'ETHUSDT'],
        exchange='binance',
        start_date='2023-01-01',
        end_date='2023-06-01',
        timeframe='4h',
        forward_days=15
    )
    
    # Apply confidence weighting
    print("\n2. Applying confidence weighting...")
    data = builder.apply_confidence_weighting(data, method='tiered')
    
    # Train with validation split
    print("\n3. Splitting train/validation...")
    train_data, val_data = builder.prepare_train_val_split(data, val_ratio=0.2)
    
    print("\n4. Training model...")
    predictor = SRPerformancePredictor()
    metrics = predictor.train(train_data, n_folds=3)
    
    # Validate
    print("\n5. Validating on hold-out set...")
    from sklearn.metrics import mean_absolute_error
    
    predictions = predictor.predict(val_data)
    
    for target in predictor.targets:
        if target in val_data.columns:
            mae = mean_absolute_error(val_data[target], predictions[target])
            print(f"{target} MAE: {mae:.4f}")
    
    return predictor, data


def example_prediction():
    """Example: Making predictions with trained model."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Making Predictions")
    print("=" * 80)
    
    # Load trained model
    print("\n1. Loading trained model...")
    predictor = SRPerformancePredictor()
    predictor.load('outputs/sr_prediction/example')
    
    # Prepare example features
    print("\n2. Preparing features...")
    features = pd.DataFrame([
        {
            'feature_strength': 0.85,
            'feature_prominence': 0.72,
            'feature_width': 1.5,
            'feature_volume_confirmation': 0.68,
            'feature_consistency': 0.75,
            'feature_touch_count': 5,
            'feature_age_bars': 50,
            'feature_failure_count': 0,
            'feature_avg_bounce_ratio': 0.018,
            'feature_max_bounce_ratio': 0.035,
            'feature_median_bounce_ratio': 0.015,
            'feature_bounce_consistency': 0.72,
            'feature_volume_weighted_bounce': 0.68,
            'feature_strong_bounce_count': 3,
            'feature_strong_bounce_ratio': 0.6,
            'feature_avg_touch_volume_ratio': 1.8,
            # ... (add all other required features with default values)
        }
    ])
    
    # Fill missing features with 0
    for feature in predictor.feature_names:
        if feature not in features.columns:
            features[feature] = 0.0
    
    # Make predictions
    print("\n3. Making predictions...")
    predictions = predictor.predict(features)
    
    print("\nPredictions for SR level:")
    for target, values in predictions.items():
        print(f"  {target}: {values[0]:.3f}")
    
    # Get SHAP explanation
    print("\n4. Generating SHAP explanation for bounce_strength...")
    explanation = predictor.explain_prediction(
        features, 
        target='bounce_strength',
        sample_idx=0
    )
    
    print(f"\nBase value: {explanation['base_value']:.3f}")
    print(f"Prediction: {explanation['prediction']:.3f}")
    print("\nTop 10 feature contributions:")
    
    shap_items = sorted(
        explanation['shap_values'].items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]
    
    for feature, shap_value in shap_items:
        feature_value = explanation['feature_values'][feature]
        direction = "↑" if shap_value > 0 else "↓"
        print(f"  {direction} {feature}: {shap_value:+.4f} (value={feature_value:.3f})")


def example_feature_importance():
    """Example: Analyzing feature importance."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Feature Importance Analysis")
    print("=" * 80)
    
    # Load model
    predictor = SRPerformancePredictor()
    predictor.load('outputs/sr_prediction/example')
    
    # Get feature importance for each target
    for target in predictor.targets:
        print(f"\n{target.upper()} - Top 15 Features:")
        print("-" * 60)
        
        importance = predictor.get_feature_importance(
            target=target,
            method='gain',
            top_n=15
        )
        
        for idx, row in importance.iterrows():
            bar_length = int(row['importance'] / importance['importance'].max() * 40)
            bar = "█" * bar_length
            print(f"  {row['feature']:<40} {bar} {row['importance']:>10.0f}")


async def main():
    """Run all examples."""
    
    # Example 1: Basic training
    predictor, data = await example_basic_training()
    
    # Example 2: Multi-symbol training (commented out to save time)
    # await example_multi_symbol_training()
    
    # Example 3: Predictions (requires trained model from example 1)
    example_prediction()
    
    # Example 4: Feature importance (requires trained model)
    example_feature_importance()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == '__main__':
    # Run examples
    asyncio.run(main())

