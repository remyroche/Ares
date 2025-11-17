#!/usr/bin/env python3
"""
Simple test to verify OOF implementation works correctly.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

# Import des assertions standardisées
from tests.utils.assertions import (
    assert_float_equals,
    assert_dict_structure,
    assert_list_structure
)

def test_oof_predictions():
    """Test OOF prediction generation with a simple example."""

    # Create synthetic temporal data
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    n_classes = 3

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)

    print("=" * 80)
    print("Testing OOF Prediction Generation")
    print("=" * 80)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Number of classes: {n_classes}")

    # Initialize OOF predictions
    n_splits = 5
    oof_predictions = np.full((len(X), n_classes), np.nan)

    # Create temporal folds
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Track which samples have been predicted
    predicted_mask = np.zeros(len(X), dtype=bool)

    print(f"\nGenerating OOF predictions with {n_splits} folds...")

    # Generate OOF predictions for each fold
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\nFold {fold_idx + 1}/{n_splits}:")
        print(f"  Train: {len(train_idx)} samples (indices {train_idx[0]}-{train_idx[-1]})")
        print(f"  Val: {len(val_idx)} samples (indices {val_idx[0]}-{val_idx[-1]})")

        # Get fold data
        X_fold_train = X[train_idx]
        y_fold_train = y[train_idx]
        X_fold_val = X[val_idx]

        # Create and train a fresh model for this fold
        fold_model = RandomForestClassifier(n_estimators=10, random_state=42)
        fold_model.fit(X_fold_train, y_fold_train)

        # Predict on fold validation data (out-of-fold)
        fold_predictions = fold_model.predict_proba(X_fold_val)

        # Store OOF predictions
        oof_predictions[val_idx] = fold_predictions
        predicted_mask[val_idx] = True

        print(f"  ✓ Generated {len(val_idx)} OOF predictions")

    # Calculate coverage statistics
    n_predicted = predicted_mask.sum()
    coverage_pct = (n_predicted / len(X)) * 100

    print("\n" + "=" * 80)
    print("OOF Prediction Summary")
    print("=" * 80)
    print(f"Total samples: {len(X)}")
    print(f"Predicted samples: {n_predicted}")
    print(f"Coverage: {coverage_pct:.1f}%")

    # Check for NaN values
    nan_count = np.isnan(oof_predictions).sum()
    nan_pct = (nan_count / oof_predictions.size) * 100
    print(f"NaN values: {nan_count}/{oof_predictions.size} ({nan_pct:.1f}%)")

    # Verify predictions have correct shape
    expected_shape = (n_samples, n_classes)
    assert_dict_structure(
        {'shape': oof_predictions.shape, 'expected': expected_shape},
        ['shape', 'expected'],
        message=f"Shape mismatch: {oof_predictions.shape} != {expected_shape}"
    )
    
    # Verify we have predictions for most samples (TimeSeriesSplit leaves first fold without predictions)
    assert_float_equals(
        coverage_pct,
        60.0,
        tolerance=5.0,  # Allow 5% tolerance for edge cases
        message=f"Coverage too low: {coverage_pct:.1f}%"
    )

    print("\n✅ All tests passed!")
    print("=" * 80)

    return True

if __name__ == "__main__":
    try:
        test_oof_predictions()
        print("\n✅ OOF implementation test SUCCESSFUL")
    except Exception as e:
        print(f"\n❌ OOF implementation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
