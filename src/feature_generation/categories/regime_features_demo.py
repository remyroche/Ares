"""
Demo script for new regime-specific features.

This script demonstrates how to use the new regime feature categories:
- REGIME_TRANSITIONS
- REGIME_PERSISTENCE
- MARKET_STRUCTURE
- REGIME_PROBABILITY
- REGIME_UNCERTAINTY

These features are designed to work with HMM regime models from
training/steps/market_analysis/rolling_hmm_clustering.
"""

import numpy as np
import pandas as pd

from .regime_transitions import create_regime_transition_generators
from .regime_persistence import create_regime_persistence_generators
from .market_structure import create_market_structure_generators
from .regime_probability import create_regime_probability_generators
from .regime_uncertainty import create_regime_uncertainty_generators


def demo_regime_features():
    """
    Demonstrate usage of new regime features.

    This shows how to:
    1. Load regime data (labels and probabilities from HMM)
    2. Create feature generators
    3. Generate features
    4. Inspect results
    """
    print("=" * 80)
    print("REGIME FEATURES DEMO")
    print("=" * 80)

    # Create synthetic market data for demonstration
    n_samples = 1000
    np.random.seed(42)

    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1h'),
        'open': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'high': 101 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'low': 99 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'volume': np.random.lognormal(10, 1, n_samples)
    })

    data['high'] = data[['open', 'high', 'close']].max(axis=1) + 0.5
    data['low'] = data[['open', 'low', 'close']].min(axis=1) - 0.5

    # Simulate HMM outputs
    n_regimes = 3

    # Simulate regime labels (regime switches every ~50-150 periods)
    regime_labels = np.zeros(n_samples)
    current_regime = 0
    i = 0
    while i < n_samples:
        duration = np.random.randint(50, 150)
        regime_labels[i:min(i + duration, n_samples)] = current_regime
        current_regime = (current_regime + 1) % n_regimes
        i += duration

    regime_labels = pd.Series(regime_labels, index=data.index)

    # Simulate regime probabilities (softmax-like)
    regime_probabilities = np.random.dirichlet(np.ones(n_regimes) * 5, size=n_samples)
    # Make probabilities align with labels (high prob for assigned regime)
    for i in range(n_samples):
        regime_probabilities[i, int(regime_labels.iloc[i])] += 3.0
    # Re-normalize
    regime_probabilities = regime_probabilities / regime_probabilities.sum(axis=1, keepdims=True)

    # Simulate transition matrix
    transition_matrix = np.array([
        [0.95, 0.03, 0.02],  # Regime 0
        [0.02, 0.95, 0.03],  # Regime 1
        [0.03, 0.02, 0.95]   # Regime 2
    ])

    print(f"\nData shape: {data.shape}")
    print(f"Regime labels shape: {regime_labels.shape}")
    print(f"Regime probabilities shape: {regime_probabilities.shape}")
    print(f"Number of regimes: {n_regimes}")
    print(f"Transition matrix shape: {transition_matrix.shape}")

    # ========================================================================
    # 1. REGIME TRANSITIONS
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. REGIME TRANSITIONS FEATURES")
    print("=" * 80)

    transition_generators = create_regime_transition_generators()
    for gen in transition_generators:
        print(f"\nGenerator: {gen.config.name}")
        features = gen.generate_features(
            data,
            regime_labels=regime_labels,
            regime_probabilities=regime_probabilities,
            transition_matrix=transition_matrix
        )
        print(f"Number of features: {len(features)}")
        print(f"Feature names:")
        for name in sorted(features.keys())[:10]:  # Show first 10
            print(f"  - {name}: shape={features[name].shape}, "
                  f"non-null={(~np.isnan(features[name])).sum()}/{len(features[name])}")
        if len(features) > 10:
            print(f"  ... and {len(features) - 10} more")

    # ========================================================================
    # 2. REGIME PERSISTENCE
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. REGIME PERSISTENCE FEATURES")
    print("=" * 80)

    persistence_generators = create_regime_persistence_generators()
    for gen in persistence_generators:
        print(f"\nGenerator: {gen.config.name}")
        features = gen.generate_features(
            data,
            regime_labels=regime_labels
        )
        print(f"Number of features: {len(features)}")
        print(f"Feature names:")
        for name in sorted(features.keys()):
            non_null = (~np.isnan(features[name])).sum()
            if non_null > 0:
                print(f"  - {name}: shape={features[name].shape}, "
                      f"non-null={non_null}/{len(features[name])}, "
                      f"mean={np.nanmean(features[name]):.2f}")

    # ========================================================================
    # 3. MARKET STRUCTURE
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. MARKET STRUCTURE FEATURES")
    print("=" * 80)

    structure_generators = create_market_structure_generators()
    for gen in structure_generators:
        print(f"\nGenerator: {gen.config.name}")
        features = gen.generate_features(
            data,
            regime_labels=regime_labels
        )
        print(f"Number of features: {len(features)}")
        print(f"Feature names:")
        for name in sorted(features.keys()):
            non_null = (~np.isnan(features[name])).sum()
            if non_null > 0:
                print(f"  - {name}: shape={features[name].shape}, "
                      f"non-null={non_null}/{len(features[name])}")

    # ========================================================================
    # 4. REGIME PROBABILITY
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. REGIME PROBABILITY FEATURES")
    print("=" * 80)

    probability_generators = create_regime_probability_generators()
    for gen in probability_generators:
        print(f"\nGenerator: {gen.config.name}")
        features = gen.generate_features(
            data,
            regime_probabilities=regime_probabilities
        )
        print(f"Number of features: {len(features)}")
        print(f"Feature names:")
        for name in sorted(features.keys())[:10]:
            non_null = (~np.isnan(features[name])).sum()
            if non_null > 0:
                print(f"  - {name}: shape={features[name].shape}, "
                      f"non-null={non_null}/{len(features[name])}, "
                      f"mean={np.nanmean(features[name]):.4f}")
        if len(features) > 10:
            print(f"  ... and {len(features) - 10} more")

    # ========================================================================
    # 5. REGIME UNCERTAINTY
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. REGIME UNCERTAINTY FEATURES")
    print("=" * 80)

    uncertainty_generators = create_regime_uncertainty_generators()
    for gen in uncertainty_generators:
        print(f"\nGenerator: {gen.config.name}")
        features = gen.generate_features(
            data,
            regime_probabilities=regime_probabilities
        )
        print(f"Number of features: {len(features)}")
        print(f"Feature names:")
        for name in sorted(features.keys()):
            non_null = (~np.isnan(features[name])).sum()
            if non_null > 0:
                print(f"  - {name}: shape={features[name].shape}, "
                      f"non-null={non_null}/{len(features[name])}, "
                      f"mean={np.nanmean(features[name]):.4f}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)

    return {
        'transition_generators': transition_generators,
        'persistence_generators': persistence_generators,
        'structure_generators': structure_generators,
        'probability_generators': probability_generators,
        'uncertainty_generators': uncertainty_generators
    }


def integration_example():
    """
    Example of how to integrate these features with regime_models_training.

    In practice, you would:
    1. Load regime labels and probabilities from HMM artifacts
    2. Create feature generators
    3. Pass them to feature engineering pipeline
    """
    print("\n" + "=" * 80)
    print("INTEGRATION EXAMPLE")
    print("=" * 80)

    code_example = '''
# In your training pipeline:

from src.feature_generation.categories import (
    create_regime_transition_generators,
    create_regime_persistence_generators,
    create_market_structure_generators,
    create_regime_probability_generators,
    create_regime_uncertainty_generators
)

# After HMM regime discovery step:
# - Load regime_labels.parquet
# - Load regime_probabilities.h5
# - Extract transition_matrix from HMM model

# Create all regime feature generators
regime_generators = []
regime_generators.extend(create_regime_transition_generators())
regime_generators.extend(create_regime_persistence_generators())
regime_generators.extend(create_market_structure_generators())
regime_generators.extend(create_regime_probability_generators())
regime_generators.extend(create_regime_uncertainty_generators())

# Generate features
all_regime_features = {}
for generator in regime_generators:
    features = generator.generate_features(
        data=market_data,
        regime_labels=regime_labels,
        regime_probabilities=regime_probabilities,
        transition_matrix=transition_matrix
    )
    all_regime_features.update(features)

# Add to feature matrix
feature_df = pd.DataFrame(all_regime_features, index=market_data.index)
    '''

    print(code_example)
    print("=" * 80)


if __name__ == "__main__":
    # Run demo
    generators = demo_regime_features()

    # Show integration example
    integration_example()

    print("\n✓ All regime feature generators working correctly!")
