"""
Advanced Feature Importance Analysis Module

Implements MDA (Mean Decrease Accuracy) and SHAP for comprehensive feature importance analysis.
Provides better feature selection and interpretability than basic LGBM importance.
"""

import warnings
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.tprint import tprint_info, tprint_warning, tprint_success

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    tprint_warning("SHAP not available - install with: pip install shap")

try:
    from sklearn.inspection import permutation_importance
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def compute_mda_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 50,
    n_repeats: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Compute Mean Decrease Accuracy (MDA) feature importance.

    MDA measures how much prediction accuracy decreases when a feature is randomly shuffled.
    Higher MDA values indicate more important features.

    Args:
        X: Feature matrix
        y: Target labels
        n_estimators: Number of trees in RandomForest
        n_repeats: Number of permutation repeats
        random_state: Random state for reproducibility

    Returns:
        Dict with MDA scores and rankings
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn not available for MDA"}

    try:
        # Train baseline model
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X, y)

        # Compute baseline accuracy
        baseline_score = rf.score(X, y)

        # Compute permutation importance
        perm_importance = permutation_importance(
            rf, X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        # Create results dictionary
        feature_names = X.columns.tolist()
        mda_scores = perm_importance.importances_mean
        mda_std = perm_importance.importances_std

        # Create feature ranking
        sorted_indices = np.argsort(mda_scores)[::-1]
        top_features = [feature_names[i] for i in sorted_indices[:20]]

        # Calculate concentration metrics
        if mda_scores.sum() > 0:
            normalized_scores = mda_scores / mda_scores.sum()
            importance_concentration = normalized_scores[0]  # Top feature concentration
            top_5_concentration = normalized_scores[:5].sum()
        else:
            importance_concentration = 0.0
            top_5_concentration = 0.0

        results = {
            "method": "mda",
            "baseline_accuracy": float(baseline_score),
            "feature_scores": dict(zip(feature_names, mda_scores.tolist())),
            "feature_std": dict(zip(feature_names, mda_std.tolist())),
            "top_features": top_features,
            "importance_concentration": float(importance_concentration),
            "top_5_concentration": float(top_5_concentration),
            "n_estimators": n_estimators,
            "n_repeats": n_repeats
        }

        return results

    except Exception as e:
        return {"error": f"MDA computation failed: {str(e)}"}


def compute_shap_importance(
    X: pd.DataFrame,
    y: pd.Series,
    max_evals: int = 1000,
    n_samples: int = 1000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Compute SHAP feature importance using TreeExplainer.

    SHAP provides more detailed explanations than MDA, showing both magnitude
    and direction of feature impacts.

    Args:
        X: Feature matrix
        y: Target labels
        max_evals: Maximum SHAP evaluations
        n_samples: Number of samples to explain
        random_state: Random state for reproducibility

    Returns:
        Dict with SHAP values and importance rankings
    """
    if not SHAP_AVAILABLE or not SKLEARN_AVAILABLE:
        return {"error": "SHAP or scikit-learn not available"}

    try:
        # Sample data if too large
        if len(X) > n_samples:
            sample_indices = np.random.RandomState(random_state).choice(
                len(X), size=n_samples, replace=False
            )
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
        else:
            X_sample = X
            y_sample = y

        # Train model
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X_sample, y_sample)

        # Compute SHAP values
        explainer = shap.TreeExplainer(rf, seed=random_state)

        # Use shap.sample for efficiency if dataset is large
        if len(X_sample) > 100:
            X_shap = shap.sample(X_sample, nsamples=min(100, len(X_sample)))
        else:
            X_shap = X_sample

        shap_values = explainer(X_shap, max_evals=max_evals)

        # Compute mean absolute SHAP values for each feature
        feature_names = X.columns.tolist()
        shap_importance = np.abs(shap_values.values).mean(axis=0)

        # Handle multi-class case
        if len(shap_values.values.shape) > 2:
            shap_importance = np.abs(shap_values.values).mean(axis=(0, 1))

        # Create feature ranking
        sorted_indices = np.argsort(shap_importance)[::-1]
        top_features = [feature_names[i] for i in sorted_indices[:20]]

        # Calculate concentration metrics
        if shap_importance.sum() > 0:
            normalized_scores = shap_importance / shap_importance.sum()
            importance_concentration = normalized_scores[0]
            top_5_concentration = normalized_scores[:5].sum()
        else:
            importance_concentration = 0.0
            top_5_concentration = 0.0

        # Store individual SHAP values for top features (first 100 samples)
        individual_shap = {}
        max_samples = min(100, len(X_shap))
        for i, feature in enumerate(feature_names[:10]):  # Top 10 features
            if i < len(shap_importance):
                individual_shap[feature] = shap_values.values[:max_samples, i].tolist()

        results = {
            "method": "shap",
            "feature_scores": dict(zip(feature_names, shap_importance.tolist())),
            "top_features": top_features,
            "importance_concentration": float(importance_concentration),
            "top_5_concentration": float(top_5_concentration),
            "shap_values_sample": individual_shap,
            "n_samples_explained": len(X_shap),
            "max_evals": max_evals
        }

        return results

    except Exception as e:
        return {"error": f"SHAP computation failed: {str(e)}"}


def compute_feature_importance_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive feature importance analysis using multiple methods.

    Args:
        X: Feature matrix
        y: Target labels
        config: Configuration dictionary
        verbose: Whether to print progress

    Returns:
        Dict with results from all enabled methods
    """
    methods = config.get("methods", ["mda", "shap"])
    results = {
        "methods_used": [],
        "config": config
    }

    # MDA Analysis
    if "mda" in methods:
        if verbose:
            tprint_info("   Computing MDA (Mean Decrease Accuracy)...")

        mda_config = {
            "n_estimators": config.get("mda_estimators", 50),
            "n_repeats": config.get("mda_n_repeats", 5),
            "random_state": 42
        }

        mda_results = compute_mda_importance(X, y, **mda_config)

        if "error" not in mda_results:
            results["methods_used"].append("mda")
            results["mda"] = mda_results
            if verbose:
                top_5 = mda_results["top_features"][:5]
                tprint_success(f"   MDA completed - top features: {', '.join(top_5)}")
        else:
            if verbose:
                tprint_warning(f"   MDA failed: {mda_results['error']}")

    # SHAP Analysis
    if "shap" in methods:
        if verbose:
            tprint_info("   Computing SHAP values...")

        shap_config = {
            "max_evals": config.get("shap_max_evals", 1000),
            "n_samples": config.get("shap_n_samples", min(1000, len(X))),
            "random_state": 42
        }

        shap_results = compute_shap_importance(X, y, **shap_config)

        if "error" not in shap_results:
            results["methods_used"].append("shap")
            results["shap"] = shap_results
            if verbose:
                top_5 = shap_results["top_features"][:5]
                tprint_success(f"   SHAP completed - top features: {', '.join(top_5)}")
        else:
            if verbose:
                tprint_warning(f"   SHAP failed: {shap_results['error']}")

    # Cross-method comparison
    if len(results["methods_used"]) >= 2:
        results["comparison"] = compute_importance_comparison(results)

    return results


def compute_importance_comparison(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare importance rankings across different methods.

    Args:
        results: Results dictionary from compute_feature_importance_analysis

    Returns:
        Dict with comparison metrics
    """
    comparison = {
        "method_agreement": {},
        "feature_stability": {},
        "rank_correlations": {}
    }

    methods = results.get("methods_used", [])
    if len(methods) < 2:
        return comparison

    # Compare top-k overlap
    for k in [5, 10, 20]:
        top_sets = {}
        for method in methods:
            if method in results:
                top_features = set(results[method].get("top_features", [])[:k])
                top_sets[method] = top_features

        # Calculate pairwise overlaps
        method_pairs = []
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                if i < j:
                    overlap = len(top_sets[m1] & top_sets[m2])
                    union = len(top_sets[m1] | top_sets[m2])
                    jaccard = overlap / union if union > 0 else 0
                    method_pairs.append({
                        "methods": f"{m1}_vs_{m2}",
                        "overlap": overlap,
                        "jaccard_similarity": jaccard,
                        "k": k
                    })

        comparison["method_agreement"][f"top_{k}"] = method_pairs

    return comparison


def select_features_by_advanced_importance(
    X: pd.DataFrame,
    y: pd.Series,
    importance_results: Dict[str, Any],
    max_features: int = 50,
    min_importance_threshold: float = 0.001,
    method_preference: str = "ensemble"
) -> List[str]:
    """
    Select features based on advanced importance analysis.

    Args:
        X: Feature matrix
        y: Target labels
        importance_results: Results from compute_feature_importance_analysis
        max_features: Maximum number of features to select
        min_importance_threshold: Minimum importance threshold
        method_preference: How to combine methods ("ensemble", "mda", "shap")

    Returns:
        List of selected feature names
    """
    if not importance_results.get("methods_used"):
        # Fallback to simple selection
        return X.columns.tolist()[:max_features]

    selected_features = set()

    if method_preference == "ensemble" and len(importance_results["methods_used"]) >= 2:
        # Ensemble selection: features that rank high in multiple methods
        feature_scores = {}

        for feature in X.columns:
            scores = []
            for method in importance_results["methods_used"]:
                if method in importance_results:
                    method_scores = importance_results[method].get("feature_scores", {})
                    score = method_scores.get(feature, 0)
                    if score > min_importance_threshold:
                        scores.append(score)

            if scores:
                # Use geometric mean of scores across methods
                ensemble_score = np.exp(np.mean(np.log(np.array(scores) + 1e-8)))
                feature_scores[feature] = ensemble_score

        # Sort by ensemble score and select top features
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f for f, _ in sorted_features[:max_features]]

    else:
        # Single method selection
        preferred_method = method_preference if method_preference in importance_results.get("methods_used", []) else importance_results["methods_used"][0]

        method_results = importance_results[preferred_method]
        feature_scores = method_results.get("feature_scores", {})

        # Filter and sort
        filtered_scores = {k: v for k, v in feature_scores.items() if v > min_importance_threshold}
        sorted_features = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f for f, _ in sorted_features[:max_features]]

    return list(selected_features)


# Export functions
__all__ = [
    "compute_mda_importance",
    "compute_shap_importance",
    "compute_feature_importance_analysis",
    "select_features_by_advanced_importance"
]




