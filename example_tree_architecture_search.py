#!/usr/bin/env python3
"""
Example: Tree Architecture Search (TAS) Usage

This script demonstrates how to use the Tree Architecture Search
implementation to find optimal tree-based model architectures.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score

# Import our Tree Architecture Search
from src.utils.ml_common.optimization.tree_architecture_search import (
    TreeArchitectureSearch,
    TreeArchitectureConfig,
    TreeArchitectureCandidate,
    search_tree_architecture
)


def create_sample_data(problem_type: str = 'classification', n_samples: int = 1000, n_features: int = 20):
    """Create sample data for demonstration."""
    if problem_type == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            noise=0.1,
            random_state=42
        )

    return X, y


def demonstrate_tas():
    """Demonstrate Tree Architecture Search."""
    print("🌲 Tree Architecture Search (TAS) Demo")
    print("=" * 50)

    # Create sample classification data
    print("\n1. Creating sample classification data...")
    X, y = create_sample_data('classification')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"   Training set: {X_train.shape}")
    print(f"   Validation set: {X_val.shape}")
    print(f"   Classes: {np.unique(y)}")

    # Configure TAS
    print("\n2. Configuring Tree Architecture Search...")
    config = TreeArchitectureConfig(
        n_trials=10,  # Reduced for demo
        population_size=8,  # Reduced for demo
        min_depth=3,
        max_depth=8,
        min_trees=10,
        max_trees=50,
        objectives=['accuracy', 'efficiency', 'interpretability'],
        objective_weights=[0.5, 0.3, 0.2]
    )

    print(f"   Search space: {config.min_trees}-{config.max_trees} trees")
    print(f"   Depth range: {config.min_depth}-{config.max_depth}")
    print(f"   Objectives: {config.objectives}")
    print(f"   Weights: {config.objective_weights}")

    # Run TAS
    print("\n3. Running Tree Architecture Search...")
    tas = TreeArchitectureSearch(config)

    best_architecture = tas.search(X_train, y_train, X_val, y_val)

    # Display results
    print("\n4. Results Summary")
    print("-" * 30)
    print(f"   Best Architecture Found:")
    print(f"   - Number of trees: {best_architecture.n_trees}")
    print(f"   - Max depth: {best_architecture.max_depth}")
    print(f"   - Min samples split: {best_architecture.min_samples_split}")
    print(f"   - Min samples leaf: {best_architecture.min_samples_leaf}")
    print(f"   - Max features: {best_architecture.max_features}")
    print(f"   - Splitting strategy: {best_architecture.splitting_strategy}")

    print("
   Performance Metrics:")
    print(f"   - Accuracy: {best_architecture.accuracy:.4".4f"
    print(f"   - Efficiency score: {best_architecture.efficiency_score:.4".4f"
    print(f"   - Interpretability score: {best_architecture.interpretability_score:.4".4f"
    print(f"   - Overall score: {best_architecture.overall_score:.4".4f"

    print("
   Training Info:")
    print(f"   - Training time: {best_architecture.training_time:.2".2f"conds")
    print(f"   - Model size: {best_architecture.model_size_mb:.2".2f"B")

    # Test the best architecture
    print("\n5. Testing Best Architecture...")
    model = tas._create_model_from_candidate(best_architecture, y_train)
    model.fit(X_train, y_train)

    # Make predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, train_pred)
    val_accuracy = accuracy_score(y_val, val_pred)

    print(f"   Training accuracy: {train_accuracy:.4".4f"
    print(f"   Validation accuracy: {val_accuracy:.4".4f"

    # Compare with default Random Forest
    print("\n6. Comparison with Default Models")
    print("-" * 35)

    from sklearn.ensemble import RandomForestClassifier
    default_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    default_rf.fit(X_train, y_train)

    default_train_acc = default_rf.score(X_train, y_train)
    default_val_acc = default_rf.score(X_val, y_val)

    print(f"   Default Random Forest:")
    print(f"   - Training accuracy: {default_train_acc:.4".4f"
    print(f"   - Validation accuracy: {default_val_acc:.4".4f"

    print("
   TAS Improvement:")
    improvement = val_accuracy - default_val_acc
    print(f"   - Accuracy improvement: {improvement:+.4".4f"{improvement:+.2%}")

    return best_architecture


def demonstrate_convenience_function():
    """Demonstrate the convenience function."""
    print("\n" + "=" * 50)
    print("🚀 Using Convenience Function")
    print("=" * 50)

    # Create sample data
    X, y = create_sample_data('regression')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use convenience function
    best_architecture = search_tree_architecture(X_train, y_train, X_val, y_val)

    print("
   Best Architecture (Regression):"    print(f"   - Trees: {best_architecture.n_trees}")
    print(f"   - Max depth: {best_architecture.max_depth}")
    print(f"   - R² Score: {best_architecture.accuracy:.4".4f"

    return best_architecture


if __name__ == "__main__":
    try:
        # Run demonstrations
        best_clf = demonstrate_tas()
        best_reg = demonstrate_convenience_function()

        print("\n" + "=" * 50)
        print("✅ Tree Architecture Search Demo Complete!")
        print("=" * 50)
        print("\nKey Takeaways:")
        print("• TAS can find optimal tree structures automatically")
        print("• Multi-objective optimization balances accuracy, efficiency, and interpretability")
        print("• Evolutionary algorithms efficiently explore the architecture space")
        print("• TAS works with various tree-based models (Random Forest, XGBoost, LightGBM)")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()