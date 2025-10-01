#!/usr/bin/env python3
"""
Example: Using the Improved HPO with Diagnostics

This script demonstrates how to use the new HPO diagnostics and fixes
to avoid the common problem of identical scores across trials.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import the improved HPO utilities
from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization
from src.utils.ml_common.optimization.hpo_diagnostics_and_fixes import (
    HPODiagnostics,
    apply_hpo_fixes
)


def example_1_automatic_diagnostics():
    """Example 1: Using automatic diagnostics in bayesian_optimization."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Automatic Diagnostics")
    print("="*80)
    
    # Simulate regime data with imbalance (80% class 0, 15% class 1, 5% class 2)
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.8, 0.15, 0.05])
    
    print(f"Created dataset: {n_samples} samples, {n_features} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Initialize HPO
    hpo = HyperparameterOptimization(config={
        'enable_parallel': False,
        'enable_monitoring': True
    })
    
    # Use RandomForest search space
    search_space = hpo.default_search_spaces['random_forest']
    
    # Run optimization with automatic diagnostics
    print("\nRunning Bayesian optimization with diagnostics enabled...")
    results = hpo.bayesian_optimization(
        model_factory=RandomForestClassifier,
        X=X,
        y=y,
        search_space=search_space,
        n_trials=10,
        scoring='accuracy',  # Will auto-switch to balanced_accuracy if needed
        enable_diagnostics=True,  # 🔑 KEY: Enables automatic diagnostics
        pruner='median',
        timeout=120  # 2 minutes
    )
    
    print("\n" + "="*80)
    print("Results:")
    print(f"  Best Score: {results.get('best_score', 'N/A')}")
    print(f"  Best Params: {results.get('best_params', {})}")
    print(f"  N Trials: {results.get('n_trials', 'N/A')}")
    print("="*80)


def example_2_manual_diagnostics():
    """Example 2: Manually running diagnostics before HPO."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Manual Diagnostics")
    print("="*80)
    
    # Create dataset
    np.random.seed(42)
    X = np.random.randn(500, 30)
    y = np.random.choice([0, 1], size=500, p=[0.7, 0.3])
    
    # Step 1: Run diagnostics manually
    print("\nStep 1: Running diagnostics...")
    diagnostics = HPODiagnostics.check_data_variance(X, y, "Regime Data")
    HPODiagnostics.print_diagnostics(diagnostics)
    
    if not diagnostics["is_valid"]:
        print("❌ Data validation failed! Fix issues before HPO.")
        return
    
    # Step 2: Get recommended scoring metric
    recommended_scoring = HPODiagnostics.recommend_scoring_metric(diagnostics)
    print(f"\n✅ Recommended scoring metric: {recommended_scoring}")
    
    # Step 3: Run HPO with recommendations
    hpo = HyperparameterOptimization()
    search_space = hpo.default_search_spaces['random_forest']
    
    results = hpo.bayesian_optimization(
        model_factory=RandomForestClassifier,
        X=X,
        y=y,
        search_space=search_space,
        n_trials=5,
        scoring=recommended_scoring,
        enable_diagnostics=False  # Already did manual diagnostics
    )
    
    print(f"\nBest score: {results.get('best_score', 'N/A')}")


def example_3_apply_all_fixes():
    """Example 3: Using apply_hpo_fixes convenience function."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Apply All Fixes Automatically")
    print("="*80)
    
    # Create dataset
    np.random.seed(42)
    X = np.random.randn(800, 40)
    y = np.random.choice([0, 1, 2], size=800, p=[0.6, 0.3, 0.1])
    
    # Apply all fixes automatically
    print("\nApplying all HPO fixes...")
    search_space, hpo_params = apply_hpo_fixes(X, y, model_type="random_forest")
    
    print(f"\n✅ Using improved configuration:")
    print(f"  Scoring: {hpo_params['scoring']}")
    print(f"  CV Strategy: {hpo_params['cv_description']}")
    print(f"  N Trials: {hpo_params['n_trials']}")
    print(f"  Acquisition: {hpo_params['acquisition_function']}")
    
    # Run HPO with improved configuration
    hpo = HyperparameterOptimization()
    
    results = hpo.bayesian_optimization(
        model_factory=RandomForestClassifier,
        X=X,
        y=y,
        search_space=search_space,
        n_trials=hpo_params['n_trials'],
        scoring=hpo_params['scoring'],
        cv=hpo_params['cv_strategy'],
        acquisition_function=hpo_params['acquisition_function'],
        timeout=hpo_params['timeout'],
        enable_diagnostics=False  # Already applied fixes
    )
    
    print(f"\nBest score: {results.get('best_score', 'N/A')}")
    
    # Show score variance across trials
    if 'optimization_curve' in results:
        scores = results['optimization_curve']
        print(f"Score variance: {np.var(scores):.6f}")
        print(f"Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        print(f"Unique scores: {len(set(scores))}/{len(scores)}")


def example_4_monitoring_in_action():
    """Example 4: Demonstrating real-time monitoring."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Real-time Monitoring")
    print("="*80)
    
    # Create problematic dataset (very imbalanced)
    np.random.seed(42)
    X = np.random.randn(500, 20)
    y = np.random.choice([0, 1], size=500, p=[0.95, 0.05])  # 95% class 0!
    
    print(f"Created highly imbalanced dataset:")
    print(f"  Class 0: {np.sum(y==0)} samples ({np.sum(y==0)/len(y)*100:.1f}%)")
    print(f"  Class 1: {np.sum(y==1)} samples ({np.sum(y==1)/len(y)*100:.1f}%)")
    
    # Run with diagnostics to see warnings
    hpo = HyperparameterOptimization()
    search_space = hpo.default_search_spaces['random_forest']
    
    print("\nRunning HPO with monitoring...")
    results = hpo.bayesian_optimization(
        model_factory=RandomForestClassifier,
        X=X,
        y=y,
        search_space=search_space,
        n_trials=8,
        scoring='accuracy',  # Will trigger warnings
        enable_diagnostics=True  # Will catch the imbalance issue
    )
    
    print(f"\nFinal best score: {results.get('best_score', 'N/A')}")


def main():
    """Run all examples."""
    print("\n" + "="*100)
    print(" "*30 + "HPO OPTIMIZATION EXAMPLES")
    print("="*100)
    
    try:
        example_1_automatic_diagnostics()
    except Exception as e:
        print(f"\n❌ Example 1 failed: {e}")
    
    try:
        example_2_manual_diagnostics()
    except Exception as e:
        print(f"\n❌ Example 2 failed: {e}")
    
    try:
        example_3_apply_all_fixes()
    except Exception as e:
        print(f"\n❌ Example 3 failed: {e}")
    
    try:
        example_4_monitoring_in_action()
    except Exception as e:
        print(f"\n❌ Example 4 failed: {e}")
    
    print("\n" + "="*100)
    print("Examples complete!")
    print("="*100)


if __name__ == "__main__":
    main()

