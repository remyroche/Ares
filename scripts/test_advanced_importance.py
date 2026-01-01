#!/usr/bin/env python3
"""
Test script for advanced feature importance analysis (MDA/SHAP).

Demonstrates the benefits of MDA and SHAP over basic LGBM importance.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Add the project root to Python path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.steps.labeling.advanced_feature_importance import (
    compute_mda_importance,
    compute_shap_importance,
    compute_feature_importance_analysis
)
from src.utils.tprint import tprint_info, tprint_success, tprint_warning


def create_synthetic_dataset(n_samples=1000, n_features=50, n_informative=10, random_state=42):
    """Create a synthetic dataset with known important features."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=random_state
    )

    # Convert to DataFrame with meaningful feature names
    feature_names = []
    for i in range(n_features):
        if i < n_informative:
            feature_names.append(f"important_{i+1}")
        elif i < n_informative + 5:
            feature_names.append(f"redundant_{i-n_informative+1}")
        else:
            feature_names.append(f"noise_{i-n_informative-5+1}")

    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')

    return X_df, y_series


def compare_importance_methods():
    """Compare MDA, SHAP, and basic feature importance."""
    tprint_info("🔍 Testing Advanced Feature Importance Analysis")
    tprint_info("=" * 60)

    # Create synthetic dataset
    tprint_info("Creating synthetic dataset...")
    X, y = create_synthetic_dataset(n_samples=1000, n_features=30, n_informative=8)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tprint_info(f"Dataset: {len(X_train)} train, {len(X_test)} test samples")
    tprint_info(f"Features: {len(X.columns)} total")
    tprint_info(f"Known important features: {[col for col in X.columns if 'important' in col]}")
    tprint_info("")

    # 1. Basic RandomForest importance
    tprint_info("1️⃣ Computing basic RandomForest importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    basic_importance = dict(zip(X.columns, rf.feature_importances_))
    basic_top_10 = sorted(basic_importance.items(), key=lambda x: x[1], reverse=True)[:10]

    tprint_info("Top 10 features by basic importance:")
    for i, (feat, imp) in enumerate(basic_top_10, 1):
        feat_type = "✅ IMPORTANT" if "important" in feat else "❌ NOISE/REDUNDANT"
        tprint_info(f"  {i}. {feat}: {imp:.4f} {feat_type}")
    tprint_info("")

    # 2. MDA (Mean Decrease Accuracy)
    tprint_info("2️⃣ Computing MDA (Mean Decrease Accuracy)...")
    mda_results = compute_mda_importance(X_train, y_train, n_estimators=50, n_repeats=3)

    if "error" not in mda_results:
        mda_scores = mda_results["feature_scores"]
        mda_top_10 = sorted(mda_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        tprint_info("Top 10 features by MDA:")
        for i, (feat, score) in enumerate(mda_top_10, 1):
            feat_type = "✅ IMPORTANT" if "important" in feat else "❌ NOISE/REDUNDANT"
            tprint_info(f"  {i}. {feat}: {score:.4f} {feat_type}")

        tprint_info(f"MDA concentration metrics:")
        tprint_info(".3f")
        tprint_info(".3f")
    else:
        tprint_warning(f"MDA failed: {mda_results['error']}")
    tprint_info("")

    # 3. SHAP
    tprint_info("3️⃣ Computing SHAP importance...")
    shap_results = compute_shap_importance(X_train, y_train, max_evals=500, n_samples=200)

    if "error" not in shap_results:
        shap_scores = shap_results["feature_scores"]
        shap_top_10 = sorted(shap_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        tprint_info("Top 10 features by SHAP:")
        for i, (feat, score) in enumerate(shap_top_10, 1):
            feat_type = "✅ IMPORTANT" if "important" in feat else "❌ NOISE/REDUNDANT"
            tprint_info(f"  {i}. {feat}: {score:.4f} {feat_type}")

        tprint_info(f"SHAP concentration metrics:")
        tprint_info(".3f")
        tprint_info(".3f")
    else:
        tprint_warning(f"SHAP failed: {shap_results['error']}")
    tprint_info("")

    # 4. Comprehensive analysis
    tprint_info("4️⃣ Running comprehensive feature importance analysis...")
    config = {
        "methods": ["mda", "shap"],
        "mda_estimators": 30,
        "shap_max_evals": 300,
        "shap_n_samples": 150
    }

    full_results = compute_feature_importance_analysis(X_train, y_train, config, verbose=False)

    if full_results.get("methods_used"):
        tprint_success("Comprehensive analysis completed!")

        # Show method comparison
        if "comparison" in full_results:
            comparison = full_results["comparison"]
            if "method_agreement" in comparison and "top_5" in comparison["method_agreement"]:
                agreements = comparison["method_agreement"]["top_5"]
                if agreements:
                    agreement = agreements[0]  # MDA vs SHAP
                    tprint_info("Method agreement analysis:")
                    tprint_info(".1f")
                    tprint_info(".3f")

    else:
        tprint_warning("Comprehensive analysis failed")

    # Summary
    tprint_info("")
    tprint_info("🎯 SUMMARY: Benefits of MDA/SHAP vs Basic Importance")
    tprint_info("=" * 60)
    tprint_info("✅ MDA (Mean Decrease Accuracy):")
    tprint_info("   - Shows true predictive impact (accuracy drop when shuffled)")
    tprint_info("   - Less biased toward high-cardinality features")
    tprint_info("   - Better at detecting feature interactions")
    tprint_info("")
    tprint_info("✅ SHAP (SHapley Additive exPlanations):")
    tprint_info("   - Shows direction and magnitude of feature effects")
    tprint_info("   - Handles feature interactions explicitly")
    tprint_info("   - Provides individual prediction explanations")
    tprint_info("   - More interpretable than permutation importance")
    tprint_info("")
    tprint_info("✅ Combined Benefits:")
    tprint_info("   - More robust feature selection")
    tprint_info("   - Better detection of redundant features")
    tprint_info("   - Reduced risk of overfitting to noise")
    tprint_info("   - Improved model interpretability and trust")


if __name__ == "__main__":
    compare_importance_methods()







