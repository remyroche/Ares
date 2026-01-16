#!/usr/bin/env python3
"""
Example usage of the enhanced Layer 2.5 Chaser with weak Huber constraints
and strong regularization.
"""

import numpy as np
import pandas as pd
from src.training.steps.labeling.layer2_5_chaser import Layer25Chaser

def create_sample_data(n_samples=1000, n_features=10):
    """Create sample data for demonstration."""
    np.random.seed(42)
    
    # Generate features with some structure
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    
    # Generate binary target with some relationship to features
    # Make it somewhat predictable but not trivial
    signal = (X.iloc[:, 0] + 0.5 * X.iloc[:, 1] - 0.3 * X.iloc[:, 2]).values
    prob = 1 / (1 + np.exp(-signal))
    y = pd.Series(np.random.binomial(1, prob), name="target")
    
    # Generate regime probabilities (3 regimes)
    regime_probs = pd.DataFrame(
        np.random.dirichlet([1, 1, 1], size=n_samples),
        columns=[f"regime_{i}" for i in range(3)]
    )
    
    # Sample weights
    sample_weights = pd.Series(np.ones(n_samples), name="weights")
    
    return X, y, regime_probs, sample_weights

def main():
    """Demonstrate the enhanced chaser model."""
    print("🚀 Enhanced Layer 2.5 Chaser Demo")
    print("=" * 50)
    
    # Create sample data
    X, y, regime_probs, sample_weights = create_sample_data()
    print(f"📊 Data shape: {X.shape}")
    print(f"📊 Target distribution: {y.value_counts().to_dict()}")
    
    # Create chaser with weak Huber constraints and strong regularization
    chaser = Layer25Chaser(
        mode="classification",
        regime_split=True,
        feature_engineering=True,
        correlation_threshold=0.7,
        verbose=True,
        models_to_train=["xgb", "lgb", "cat"],  # Skip ET for demo
        use_huber_constraints=True,
        constraint_tier="weak"
    )
    
    print("\n🔧 Training chaser model...")
    
    # Fit the model
    chaser.fit(
        X=X,
        y=y,
        regime_probs=regime_probs,
        sample_weight=sample_weights
    )
    
    print("\n🎯 Making predictions...")
    
    # Make predictions
    predictions = chaser.predict(X, regime_probs=regime_probs)
    
    print(f"📈 Predictions shape: {predictions.shape}")
    print(f"📈 Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"📈 Mean prediction: {predictions.mean():.3f}")
    
    # Simple evaluation
    if len(np.unique(y)) == 2:  # Binary classification
        # Convert to binary predictions at 0.5 threshold
        binary_preds = (predictions > 0.5).astype(int)
        accuracy = np.mean(binary_preds == y.values)
        print(f"📊 Accuracy (threshold 0.5): {accuracy:.3f}")
    
    print("\n✅ Demo completed successfully!")
    
    # Show model information
    print("\n📋 Model Information:")
    print(f"   - Mode: {chaser.mode}")
    print(f"   - Features used: {len(chaser.feature_names)}")
    print(f"   - Regime models: {len(chaser.regime_models)}")
    print(f"   - Global model: {'Yes' if chaser.global_models else 'No'}")
    print(f"   - Huber constraints: {'Yes' if chaser.use_huber_constraints else 'No'}")
    print(f"   - Constraint tier: {chaser.constraint_tier}")

if __name__ == "__main__":
    main()
