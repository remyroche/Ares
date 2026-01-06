#!/usr/bin/env python3
"""
Test script for MDA/SHAP Feature Selection Module.

Tests the comprehensive feature selection pipeline with synthetic data.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Add the project root to Python path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.steps.labeling.mda_shap_feature_selection import (
    MDA_SHAP_FeatureSelector,
    run_mda_shap_feature_selection
)
from src.utils.tprint import tprint_info, tprint_success, tprint_warning


def create_test_dataset(n_samples=1000, n_features=100, n_informative=20, random_state=42):
    """Create a synthetic dataset with realistic financial characteristics and sample weights."""
    # Create correlated features (like in financial data)
    np.random.seed(random_state)

    # Base informative features
    X_base = np.random.randn(n_samples, n_informative)

    # Add correlated noise features
    X_noisy = np.zeros((n_samples, n_features - n_informative))
    for i in range(n_features - n_informative):
        # Some features correlated with informative ones, others pure noise
        if i % 3 == 0:
            base_idx = i % n_informative
            X_noisy[:, i] = X_base[:, base_idx] + 0.1 * np.random.randn(n_samples)
        else:
            X_noisy[:, i] = np.random.randn(n_samples)

    X = np.column_stack([X_base, X_noisy])
    y = (X_base[:, 0] + X_base[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)

    # Create sample weights (emphasize minority class like in imbalanced financial labels)
    pos_weight = 2.0  # Weight positive class more
    neg_weight = 1.0  # Base weight for negative class
    weights = np.where(y == 1, pos_weight, neg_weight)

    # Add some variation to weights
    weights = weights * (0.8 + 0.4 * np.random.RandomState(random_state + 1).rand(n_samples))

    # Create DataFrame with meaningful names
    feature_names = []
    for i in range(n_features):
        if i < n_informative:
            feature_names.append(f"informative_{i+1}")
        elif i < n_informative + (n_features - n_informative) // 2:
            feature_names.append(f"correlated_{i-n_informative+1}")
        else:
            feature_names.append(f"noise_{i-n_informative-(n_features-n_informative)//2+1}")

    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    weights_series = pd.Series(weights, name='sample_weight')

    return X_df, y_series, weights_series


def test_mda_shap_selection():
    """Test the MDA/SHAP feature selection pipeline."""
    tprint_info("🧬 Testing MDA/SHAP Feature Selection")
    tprint_info("=" * 60)

    # Create test dataset
    tprint_info("Creating synthetic financial-like dataset...")
    X, y, weights = create_test_dataset(n_samples=1000, n_features=80, n_informative=15)

    tprint_info(f"Dataset: {len(X)} samples, {len(X.columns)} features")
    tprint_info(f"Target distribution: {y.value_counts().to_dict()}")
    tprint_info(f"Known informative features: {[col for col in X.columns if 'informative' in col]}")
    tprint_info("")

    # Configure selection
    config = {
        "model_type": "rf",
        "n_folds": 3,  # Faster for testing
        "pre_filters": {
            "enable_lgbm_mdi_filter": True,
            "enable_correlation_filter": True,
            "enable_variance_filter": True,
            "enable_anova_filter": True
        },
        "corr_threshold": 0.85,
        "top_clusters": 3,
        "shap_sample_size": 500,
        "verbose": True
    }

    try:
        # Run selection
        tprint_info("Running MDA/SHAP feature selection with target sample weights...")
        selected_features, results = run_mda_shap_feature_selection(X, y, weights, config)

        # Results analysis
        tprint_success("✅ Selection completed successfully!")

        original_count = results['n_features_original']
        after_prefilters = results['n_features_after_prefilters']
        selected_count = results['n_features_selected']

        tprint_info("📊 Selection Summary:")
        tprint_info(f"   Original features: {original_count}")
        tprint_info(f"   After pre-filters: {after_prefilters}")
        tprint_info(f"   Final selected: {selected_count}")
        tprint_info("")

        # Show cluster analysis
        if 'clusters' in results:
            clusters = results['clusters']
            tprint_info(f"🗂️ Feature Clusters Created: {len(clusters)}")
            for cluster_name, features in list(clusters.items())[:3]:
                tprint_info(f"   {cluster_name}: {len(features)} features")
        tprint_info("")

        # Show MDA results
        if 'mda_results' in results and results['mda_results']:
            mda_scores = results['mda_results']
            top_mda = sorted(mda_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            tprint_info("🏆 Top MDA Clusters:")
            for cluster, score in top_mda:
                tprint_info(".4f")
        tprint_info("")

        # Show SHAP results
        if 'shap_results' in results and results['shap_results']:
            shap_scores = results['shap_results']
            top_shap = sorted(shap_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            tprint_info("🎯 Top SHAP Features:")
            for feature, score in top_shap:
                feature_type = "✅ INFORMATIVE" if "informative" in feature else "❌ NOISE"
                tprint_info(".4f")
        tprint_info("")

        # Show final selection
        tprint_info("🎯 Final Selected Features:")
        informative_selected = [f for f in selected_features if "informative" in f]
        correlated_selected = [f for f in selected_features if "correlated" in f]
        noise_selected = [f for f in selected_features if "noise" in f]

        tprint_info(f"   ✅ Informative: {len(informative_selected)}/{len([col for col in X.columns if 'informative' in col])}")
        tprint_info(f"   ⚠️ Correlated: {len(correlated_selected)}/{len([col for col in X.columns if 'correlated' in col])}")
        tprint_info(f"   ❌ Noise: {len(noise_selected)}/{len([col for col in X.columns if 'noise' in col])}")
        tprint_info("")

        # Performance check
        if informative_selected:
            tprint_success("✅ SUCCESS: Selected informative features!")
        else:
            tprint_warning("⚠️ WARNING: No informative features selected")

        return True

    except Exception as e:
        tprint_warning(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pre_filters():
    """Test individual pre-filter components."""
    tprint_info("🔍 Testing Pre-filter Components")
    tprint_info("=" * 40)

    X, y = create_test_dataset(n_samples=500, n_features=50, n_informative=10)

    # Test LGBM MDI filter
    try:
        import lightgbm as lgb

        lgbm_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, verbosity=-1)
        lgbm_model.fit(X, y)

        mdi_scores = dict(zip(X.columns, lgbm_model.feature_importances_))
        top_20 = sorted(mdi_scores.items(), key=lambda x: x[1], reverse=True)[:20]

        informative_in_top20 = sum(1 for feat, _ in top_20 if "informative" in feat)
        tprint_info(f"LGBM MDI: {informative_in_top20}/10 informative features in top 20")

    except Exception as e:
        tprint_warning(f"LGBM MDI test failed: {e}")

    # Test correlation filter
    try:
        corr_matrix = X.corr()
        upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))
        high_corr_pairs = [(col, upper.columns[i], upper.loc[col, upper.columns[i]])
                          for i, col in enumerate(upper.columns)
                          if any(upper[col] > 0.95)]
        tprint_info(f"Correlation filter: Found {len(high_corr_pairs)} highly correlated pairs")
    except Exception as e:
        tprint_warning(f"Correlation filter test failed: {e}")

    # Test ANOVA filter
    try:
        from sklearn.feature_selection import f_classif, SelectKBest

        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        scores = selector.scores_

        # Top 75th percentile
        percentile_25 = np.percentile(scores, 25)
        selected_count = sum(scores >= percentile_25)
        tprint_info(f"ANOVA F-test: {selected_count}/{len(X.columns)} features in top 75th percentile")

        # Check informative features
        informative_scores = [scores[i] for i, col in enumerate(X.columns) if "informative" in col]
        if informative_scores:
            avg_informative_score = np.mean(informative_scores)
            avg_noise_score = np.mean([scores[i] for i, col in enumerate(X.columns) if "noise" in col])
            tprint_info(".2f")
    except Exception as e:
        tprint_warning(f"ANOVA filter test failed: {e}")


if __name__ == "__main__":
    # Test pre-filters
    test_pre_filters()
    print("\n" + "="*60 + "\n")

    # Test full pipeline
    success = test_mda_shap_selection()

    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)









