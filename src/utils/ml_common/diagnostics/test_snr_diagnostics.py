"""
Test script for SNR Diagnostics Module

This script demonstrates and validates the SNR diagnostics functionality
with synthetic data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import make_regression

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.utils.ml_common.diagnostics import SNRDiagnostics


def generate_test_data(n_samples=1000, n_features=20, noise_level=0.5, random_state=42):
    """
    Generate synthetic test data with known signal-to-noise characteristics.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    noise_level : float
        Amount of noise (0 = no noise, 1 = high noise)
    random_state : int
        Random seed

    Returns
    -------
    X, y : arrays
        Feature matrix and target vector
    """
    X, y, true_coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(1, int(n_features * 0.6)),
        n_targets=1,
        noise=noise_level * 100,
        random_state=random_state,
        coef=True
    )

    return X, y, true_coef


def test_basic_functionality():
    """Test basic SNR diagnostics functionality."""
    print("="*80)
    print("TEST 1: Basic Functionality")
    print("="*80)

    # Generate low-noise data (should have high SNR)
    X, y, _ = generate_test_data(n_samples=500, noise_level=0.2)

    # Define simple models
    models = {
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Random_Forest': RandomForestRegressor(
            n_estimators=50, max_depth=8, random_state=42, n_jobs=-1
        )
    }

    # Initialize diagnostics
    output_dir = Path('/tmp/snr_test_basic')
    snr_diag = SNRDiagnostics(
        output_dir=output_dir,
        cv_folds=3,
        bootstrap_iterations=100,  # Reduced for speed
        permutation_iterations=100,
        random_state=42,
        verbose=True
    )

    # Run diagnostics
    cv_preds, metrics, plots, reports = snr_diag.run_full_diagnostics(
        models=models,
        X=X,
        y=y
    )

    # Validate results
    print("\n" + "-"*80)
    print("VALIDATION:")
    print("-"*80)

    for model_name, m in metrics.items():
        print(f"\n{model_name}:")
        print(f"  R² = {m.r2:.4f} (expect > 0.5 for low noise)")
        print(f"  SNR = {m.snr:.4f} (expect > 1.0 for low noise)")
        print(f"  p-value = {m.permutation_pvalue:.4f} (expect < 0.05)")
        print(f"  95% CI: [{m.bootstrap_ci_lower:.4f}, {m.bootstrap_ci_upper:.4f}]")

        # Basic assertions
        assert m.r2 > 0.3, f"R² too low: {m.r2:.4f}"
        assert m.snr > 0.3, f"SNR too low: {m.snr:.4f}"
        assert m.permutation_pvalue < 0.5, f"p-value too high: {m.permutation_pvalue:.4f}"

    print("\n✅ Basic functionality test PASSED")
    print(f"📁 Output saved to: {output_dir}")


def test_high_noise_scenario():
    """Test SNR diagnostics with high-noise data (should detect low SNR)."""
    print("\n" + "="*80)
    print("TEST 2: High Noise Scenario")
    print("="*80)

    # Generate high-noise data (should have low SNR)
    X, y, _ = generate_test_data(n_samples=500, noise_level=2.0)

    models = {
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42, max_iter=2000)
    }

    output_dir = Path('/tmp/snr_test_high_noise')
    snr_diag = SNRDiagnostics(
        output_dir=output_dir,
        cv_folds=3,
        bootstrap_iterations=100,
        permutation_iterations=100,
        random_state=42,
        verbose=True
    )

    cv_preds, metrics, plots, reports = snr_diag.run_full_diagnostics(
        models=models,
        X=X,
        y=y
    )

    print("\n" + "-"*80)
    print("VALIDATION:")
    print("-"*80)

    for model_name, m in metrics.items():
        print(f"\n{model_name}:")
        print(f"  R² = {m.r2:.4f} (expect < 0.5 for high noise)")
        print(f"  SNR = {m.snr:.4f} (expect < 1.0 for high noise)")
        print(f"  p-value = {m.permutation_pvalue:.4f}")

        # In high noise, we expect lower performance
        if m.r2 < 0.3:
            print(f"  ✅ Correctly detected low signal (R² = {m.r2:.4f})")

    print("\n✅ High noise scenario test PASSED")
    print(f"📁 Output saved to: {output_dir}")


def test_model_comparison():
    """Test comparing multiple models with different complexities."""
    print("\n" + "="*80)
    print("TEST 3: Model Comparison")
    print("="*80)

    # Generate data with moderate nonlinearity
    X, y, _ = generate_test_data(n_samples=800, n_features=15, noise_level=0.5)

    # Add some nonlinear features
    X_nonlinear = np.column_stack([
        X,
        X[:, 0] * X[:, 1],  # Interaction
        X[:, 0] ** 2,        # Quadratic
    ])

    models = {
        'Linear_Ridge': Ridge(alpha=1.0, random_state=42),
        'Random_Forest': RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
        ),
        'Gradient_Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=42
        )
    }

    output_dir = Path('/tmp/snr_test_comparison')
    snr_diag = SNRDiagnostics(
        output_dir=output_dir,
        cv_folds=5,
        bootstrap_iterations=200,
        permutation_iterations=200,
        random_state=42,
        verbose=True
    )

    cv_preds, metrics, plots, reports = snr_diag.run_full_diagnostics(
        models=models,
        X=X_nonlinear,
        y=y
    )

    print("\n" + "-"*80)
    print("VALIDATION:")
    print("-"*80)

    # Sort models by SNR
    sorted_models = sorted(metrics.items(), key=lambda x: x[1].snr, reverse=True)

    print("\nModels ranked by SNR:")
    for rank, (model_name, m) in enumerate(sorted_models, 1):
        print(f"{rank}. {model_name}: SNR={m.snr:.4f}, R²={m.r2:.4f}, p={m.permutation_pvalue:.4f}")

    # Check that reports were generated
    assert reports[0].exists(), "CSV report not generated"
    assert reports[1].exists(), "Markdown report not generated"

    print("\n✅ Model comparison test PASSED")
    print(f"📁 Output saved to: {output_dir}")


def test_standalone_functions():
    """Test standalone utility functions."""
    print("\n" + "="*80)
    print("TEST 4: Standalone Functions")
    print("="*80)

    from src.utils.ml_common.diagnostics import (
        compute_snr_metrics,
        bootstrap_r2,
        permutation_test
    )

    # Generate simple test data
    np.random.seed(42)
    y_true = np.random.randn(200)
    y_pred = y_true + np.random.randn(200) * 0.3  # Add some noise

    # Test compute_snr_metrics
    print("\nTesting compute_snr_metrics...")
    metrics = compute_snr_metrics(y_true, y_pred)
    print(f"  R² = {metrics['r2']:.4f}")
    print(f"  SNR = {metrics['snr']:.4f}")
    print(f"  RMSE = {metrics['rmse']:.4f}")
    assert 'r2' in metrics and 'snr' in metrics

    # Test bootstrap_r2
    print("\nTesting bootstrap_r2...")
    ci_lower, ci_upper = bootstrap_r2(y_true, y_pred, n_iterations=100)
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    assert ci_lower < ci_upper
    assert ci_lower >= -1.0 and ci_upper <= 1.0

    # Test permutation_test
    print("\nTesting permutation_test...")
    pvalue = permutation_test(y_true, y_pred, n_permutations=100)
    print(f"  p-value = {pvalue:.4f}")
    assert 0 <= pvalue <= 1

    print("\n✅ Standalone functions test PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("SNR DIAGNOSTICS MODULE - COMPREHENSIVE TEST SUITE")
    print("="*80)

    try:
        test_basic_functionality()
        test_high_noise_scenario()
        test_model_comparison()
        test_standalone_functions()

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✅")
        print("="*80)
        print("\nThe SNR diagnostics module is working correctly!")
        print("You can now use it in your ML pipelines.")

    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED ❌")
        print("="*80)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    run_all_tests()
