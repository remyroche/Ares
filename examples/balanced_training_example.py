#!/usr/bin/env python3
"""
Balanced Training Example

This example demonstrates how to use the comprehensive label balancing
and sample weighting system with the Tactician training pipeline.

The example shows:
1. Basic balancing and weighting setup
2. Integration with Tactician training
3. Validation fairness checking
4. Performance monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import asyncio
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.training.steps.pre_training.label_balancing import (
        ComprehensiveBalancingSystem,
        BalancingConfig,
        WeightingConfig,
        RegimeConfig,
        ValidationFairnessConfig,
        DEFAULT_BALANCING_CONFIG,
        DEFAULT_WEIGHTING_CONFIG,
        DEFAULT_REGIME_CONFIG,
        DEFAULT_FAIRNESS_CONFIG,
        BalancingTechnique,
        WeightingScheme
    )

    from src.training.steps.model_training.tactician_balanced_training import (
        BalancedTacticianTrainingStep,
        BalancedTrainingConfig
    )

    # Import Tactician components
    from src.training.steps.model_training.tactician_training_step import TacticianTrainingConfig

    COMPONENTS_AVAILABLE = True

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("⚠️ Some components may not be available")
    COMPONENTS_AVAILABLE = False

try:
    from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False


def generate_synthetic_imbalanced_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic imbalanced financial dataset.

    Creates a dataset with extreme class imbalance typical of financial data:
    - 85% "no-trade" (class 0)
    - 10% "long" (class 1)
    - 5% "short" (class -1)
    """
    if TPRINT_AVAILABLE:
        tprint_info("📊 Generating synthetic imbalanced dataset...")

    np.random.seed(42)

    # Generate class labels with extreme imbalance
    class_probs = [0.85, 0.10, 0.05]  # 85% no-trade, 10% long, 5% short
    classes = np.random.choice([0, 1, -1], size=n_samples, p=class_probs)

    # Generate features
    n_features = 20
    features = {}

    for i in range(n_features):
        if i < 5:  # Price-related features
            features[f'price_{i}'] = np.random.randn(n_samples) * 0.1 + classes * 0.05
        elif i < 10:  # Volume-related features
            features[f'volume_{i}'] = np.random.exponential(1, n_samples) * (1 + abs(classes) * 0.5)
        elif i < 15:  # Technical indicator features
            features[f'technical_{i}'] = np.random.randn(n_samples) * 0.2 + classes * 0.1
        else:  # Volatility features
            features[f'volatility_{i}'] = np.random.exponential(0.1, n_samples) * (1 + abs(classes) * 0.3)

    # Create DataFrame
    df = pd.DataFrame(features)
    df['target'] = classes

    # Add some metadata
    df['timestamp'] = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    df['regime'] = np.random.choice(['trending', 'ranging', 'volatile'], n_samples, p=[0.3, 0.5, 0.2])

    # Add volatility for weighting
    df['realized_volatility'] = np.random.exponential(0.05, n_samples) * (1 + abs(classes) * 0.2)

    if TPRINT_AVAILABLE:
        tprint_success(f"✅ Generated dataset: {len(df)} samples")
        tprint_info(f"📊 Class distribution: {df['target'].value_counts().to_dict()}")

    return df


def demonstrate_basic_balancing():
    """Demonstrate basic balancing and weighting."""
    if TPRINT_AVAILABLE:
        tprint_info("🔄 Demonstrating basic balancing and weighting...")

    # Generate imbalanced data
    data = generate_synthetic_imbalanced_data(5000)

    # Split features and target
    feature_cols = [col for col in data.columns if col not in ['target', 'timestamp', 'regime']]
    X = data[feature_cols]
    y = data['target']

    # Create balancing system
    balancing_system = ComprehensiveBalancingSystem(
        DEFAULT_BALANCING_CONFIG,
        DEFAULT_WEIGHTING_CONFIG,
        DEFAULT_REGIME_CONFIG,
        DEFAULT_FAIRNESS_CONFIG
    )

    # Prepare additional features for weighting
    additional_features = {
        'volatility': data['realized_volatility'],
        'regime': data['regime'],
        'timestamp': data['timestamp']
    }

    # Apply balancing and weighting
    X_balanced, y_balanced, sample_weights = balancing_system.balance_and_weight(
        X, y, additional_features=additional_features
    )

    # Display results
    if TPRINT_AVAILABLE:
        tprint_success("✅ Balancing completed!")
        print(f"\n{'='*60}")
        print("BALANCING RESULTS")
        print(f"{'='*60}")
        print(f"Original samples: {len(X)}")
        print(f"Balanced samples: {len(X_balanced)}")
        print(f"Reduction ratio: {len(X_balanced)/len(X):.2%}")
        print()
        print("CLASS DISTRIBUTION:")
        print(f"Original: {y.value_counts().to_dict()}")
        print(f"Balanced: {y_balanced.value_counts().to_dict()}")
        print()
        print("WEIGHT STATISTICS:")
        print(f"Mean weight: {sample_weights.mean():.3f}")
        print(f"Weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
        print(f"Weight std: {sample_weights.std():.3f}")
        print(f"{'='*60}")

    return X_balanced, y_balanced, sample_weights


def demonstrate_validation_fairness():
    """Demonstrate validation fairness checking."""
    if TPRINT_AVAILABLE:
        tprint_info("⚖️ Demonstrating validation fairness checking...")

    # Generate training and validation data
    train_data = generate_synthetic_imbalanced_data(8000)
    val_data = generate_synthetic_imbalanced_data(2000)

    # Create balancing system for fairness checking
    balancing_system = ComprehensiveBalancingSystem(
        DEFAULT_BALANCING_CONFIG,
        DEFAULT_WEIGHTING_CONFIG,
        DEFAULT_REGIME_CONFIG,
        DEFAULT_FAIRNESS_CONFIG
    )

    # Prepare data for fairness check
    train_dict = {
        'y': train_data['target'],
        'regime': train_data['regime']
    }

    val_dict = {
        'y': val_data['target'],
        'regime': val_data['regime']
    }

    # Check fairness
    fairness_report = balancing_system.check_validation_fairness(train_dict, val_dict)

    # Display fairness report
    if TPRINT_AVAILABLE:
        print(f"\n{'='*60}")
        print("VALIDATION FAIRNESS REPORT")
        print(f"{'='*60}")

        if fairness_report.get('class_ratio_fair', True):
            tprint_success("✅ Class ratios are fair")
        else:
            tprint_warning("⚠️ Class ratios are not fair")
            print(f"   Max deviation: {fairness_report.get('class_ratio_deviation', 0):.3f}")

        if fairness_report.get('regime_mix_fair', True):
            tprint_success("✅ Regime mix is fair")
        else:
            tprint_warning("⚠️ Regime mix is not fair")
            print(f"   Max deviation: {fairness_report.get('regime_mix_deviation', 0):.3f}")

        print(f"\nTrain class ratios: {fairness_report.get('train_ratios', {})}")
        print(f"Val class ratios: {fairness_report.get('val_ratios', {})}")
        print(f"{'='*60}")

    return fairness_report


async def demonstrate_balanced_training_integration():
    """Demonstrate integration with Tactician training."""
    if not COMPONENTS_AVAILABLE:
        if TPRINT_AVAILABLE:
            tprint_warning("⚠️ Tactician training components not available, skipping integration demo")
        return None

    if TPRINT_AVAILABLE:
        tprint_info("🚀 Demonstrating balanced training integration...")

    # Generate training data
    data = generate_synthetic_imbalanced_data(10000)

    # Split into analyst signals and market data (simplified)
    analyst_signals = data[['target']].copy()
    analyst_signals['confidence'] = np.random.uniform(0.5, 0.95, len(data))

    market_data = data.drop(['target'], axis=1)
    feature_names = [col for col in market_data.columns if col not in ['timestamp', 'regime']]

    # Create balanced training configuration
    balanced_config = BalancedTrainingConfig(
        enable_balancing=True,
        enable_weighting=True,
        enable_regime_balancing=True,
        enable_validation_fairness=True
    )

    # Create balanced trainer
    trainer = BalancedTacticianTrainingStep(
        training_config=TacticianTrainingConfig(),
        balanced_config=balanced_config
    )

    # Note: This would require full Tactician setup in practice
    # For demo purposes, we'll just show the setup
    if TPRINT_AVAILABLE:
        tprint_success("✅ Balanced trainer created successfully")
        tprint_info("   → Balancing: Enabled")
        tprint_info("   → Weighting: Enabled")
        tprint_info("   → Regime balancing: Enabled")
        tprint_info("   → Validation fairness: Enabled")

    return trainer


def run_comprehensive_demo():
    """Run comprehensive demonstration of all features."""
    if TPRINT_AVAILABLE:
        tprint_info("🚀 Starting comprehensive balancing system demo...")

    print("=" * 80)
    print("COMPREHENSIVE LABEL BALANCING & SAMPLE WEIGHTING DEMO")
    print("=" * 80)

    # Demo 1: Basic balancing and weighting
    print("\n1. BASIC BALANCING AND WEIGHTING")
    print("-" * 40)
    X_balanced, y_balanced, weights = demonstrate_basic_balancing()

    # Demo 2: Validation fairness
    print("\n2. VALIDATION FAIRNESS CHECKING")
    print("-" * 40)
    fairness_report = demonstrate_validation_fairness()

    # Demo 3: Training integration
    print("\n3. BALANCED TRAINING INTEGRATION")
    print("-" * 40)
    trainer = asyncio.run(demonstrate_balanced_training_integration())

    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    if TPRINT_AVAILABLE:
        tprint_success("✅ All demonstrations completed")
        tprint_info("📚 See README_BALANCING_SYSTEM.md for detailed usage instructions")


if __name__ == "__main__":
    run_comprehensive_demo()