#!/usr/bin/env python3
import psutil
from tqdm import tqdm
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""Specialist Feature Diagnostics CLI.

Analyzes specialist model outputs (Risk, Liquidity, Breakout/Bounce)
loaded via `get_specialist_models_outputs` against meta-labeling targets
produced by `FeatureGenerationMetaLabelingStep`.

Metrics per specialist feature:
- Event-aware correlation-based MI proxy (cheap mutual information proxy)
- MI stability across time-series CV folds (mean and coefficient of variation)
- Pearson correlation with target label and corresponding R^2

Outputs Markdown and CSV reports under the `outcomes/` directory.
Optionally restricts analysis to the last N calendar days via --lookback-days.

Usage example (from project root):

  # Uses direction-aware default target (binary_label_long for longs)
  python scripts/specialist_feature_diagnostics.py \
      --symbol ETHUSDT --exchange binance --timeframe 15m \
      --direction long --regime-timeframe 1h --lookback-days 365

  # Explicit target column (regression target for regressors)
  python scripts/specialist_feature_diagnostics.py \
      --symbol ETHUSDT --exchange binance --timeframe 15m \
      --direction long --target-col target_long --regime-timeframe 1h

  # Compare multiple targets (classifiers vs regressors)
  python scripts/specialist_feature_diagnostics.py \
      --symbol ETHUSDT --exchange binance --timeframe 15m \
      --direction long --compare-targets --regime-timeframe 1h
"""

import argparse
import gc
from src.utils.thresholding.dynamic_thresholds import DynamicThresholdCalculator, calculate_dynamic_thresholds_batch
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Ensure project root is on sys.path
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# if str(PROJECT_ROOT) not in sys.path:
#    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import system_logger  # type: ignore
from src.utils.ml_common.get_specialist_models_outputs import (  # type: ignore
    get_specialist_models_outputs,
    get_enhanced_specialist_models_outputs,
)
from src.utils.ml_common.feature_selection import get_feature_selection_utils  # type: ignore
from src.training.steps.labeling.feature_generation_meta_labeling_step import (  # type: ignore
    FeatureGenerationMetaLabelingStep,
)
from src.training.steps.labeling.snr_diagnostics import (  # type: ignore
    _load_labeled_data,
)
from src.training.steps.pre_training.components.final_feature_selection import (  # type: ignore
    FinalFeatureSelectionConfig,
    FinalFeatureSelectionComponent,
)
from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
from src.training.steps.market_analysis import step_registry  # type: ignore
from scripts.generate_orthogonal_comparison_report import generate_orthogonal_comparison_report
from src.analysis.tv_var_system import TVVARSystem
from src.analysis.tv_var_regime_definition import EightFeatureRegimeDetector
from src.analysis.tv_var_decision_tree_rules import TVVARDecisionTreeRules
from src.analysis.tv_var_monthly_trainer import TVVARMonthlyTrainer
from src.analysis.tv_var_backtesting import TVVARBacktester, backtest_tv_var_enhanced
from src.training.steps.labeling.unified_price_layer2 import (  # type: ignore
    load_layer0_params,
    apply_hampel_filter,
    apply_savgol_filter,
)


logger = system_logger.getChild("specialist_feature_diagnostics")

OUTCOMES_DIR = Path("outcomes")
_LAYER0_PARAMS_CACHE: Optional[Dict[str, Any]] = None


def _load_latest_layer0_params() -> Dict[str, Any]:
    """Load (and cache) the most recent Layer0 smoothing parameters."""
    global _LAYER0_PARAMS_CACHE
    if _LAYER0_PARAMS_CACHE is not None:
        return _LAYER0_PARAMS_CACHE

    try:
        params = load_layer0_params()
        _LAYER0_PARAMS_CACHE = params
        logger.info(
            "🧽 Loaded latest Layer0 params (Q=%.2e, R=%.2e, vwap_weight=%.3f)",
            params.get("kalman_Q", 0.0),
            params.get("kalman_R", 0.0),
            params.get("vwap_weight", 0.0),
        )
        return params
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("⚠️ Failed to load Layer0 params, skipping global denoising: %s", exc)
        _LAYER0_PARAMS_CACHE = {}
        return _LAYER0_PARAMS_CACHE


def _kalman_smooth_series(series: pd.Series, Q: float = 1e-5, R: float = 0.01) -> pd.Series:
    """Apply simple Kalman filter smoothing to a continuous series (probabilities)."""
    vals = series.values
    n = len(vals)
    x_hat = np.zeros(n)
    P = np.ones(n)
    
    x_hat[0] = vals[0]
    P[0] = 1.0
    
    for i in range(1, n):
        # Prediction
        x_hat_minus = x_hat[i-1]
        P_minus = P[i-1] + Q
        
        # Update
        K = P_minus / (P_minus + R)
        x_hat[i] = x_hat_minus + K * (vals[i] - x_hat_minus)
        P[i] = (1 - K) * P_minus
        
    return pd.Series(x_hat, index=series.index)

def _calibrate_probabilities(y_tr: np.ndarray, p_tr: np.ndarray, p_te: np.ndarray) -> np.ndarray:
    """Calibrate probabilities using Platt Scaling (Logistic Regression on scores)."""
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(y_tr)) < 2:
        return p_te
    
    # Platt scaling: train a small LogReg on the probabilities
    calibrator = LogisticRegression(C=1e10, solver='lbfgs')
    # Reshape for sklearn
    X_tr = p_tr.reshape(-1, 1)
    X_te = p_te.reshape(-1, 1)
    
    calibrator.fit(X_tr, y_tr)
    return calibrator.predict_proba(X_te)[:, 1]

def _apply_global_layer0_denoising(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Layer0-derived Hampel/Savitzky-Golay denoising to all numeric specialist features once.
    """
    params = _load_latest_layer0_params()
    if not params:
        return df

    hampel_enabled = params.get("hampel_filter_enabled", False)
    savgol_enabled = params.get("savgol_filter_enabled", False)
    if not (hampel_enabled or savgol_enabled):
        return df

    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns
    if not len(numeric_cols):
        return df

    denoised_df = df.copy()

    hampel_window = int(params.get("hampel_window", 5))
    hampel_threshold = float(params.get("hampel_threshold", 3.0))
    savgol_window = int(params.get("savgol_window", 21))
    savgol_order = int(params.get("savgol_order", 3))

    logger.info(
        "🔧 Applying global Layer0 denoising (hampel=%s, savgol=%s) to %d specialist columns",
        hampel_enabled,
        savgol_enabled,
        len(numeric_cols),
    )

    for col in numeric_cols:
        series = denoised_df[col].astype(float)
        series = series.replace([np.inf, -np.inf], np.nan)
        series = series.ffill().bfill()

        if hampel_enabled and len(series.dropna()) >= max(3, hampel_window + 1):
            try:
                series = apply_hampel_filter(series, window=hampel_window, threshold=hampel_threshold)
            except Exception as exc:  # pragma: no cover - logging only
                logger.debug("Hampel filter failed for %s: %s", col, exc)

        if savgol_enabled and len(series.dropna()) >= max(savgol_window + 1, savgol_order + 2):
            try:
                series = apply_savgol_filter(series, window_length=savgol_window, poly_order=savgol_order)
            except Exception as exc:  # pragma: no cover - logging only
                logger.debug("Savitzky-Golay filter failed for %s: %s", col, exc)

        # Restore original dtype if it was boolean
        if pd.api.types.is_bool_dtype(df[col].dtype):
            denoised_df[col] = (series >= 0.5).astype(bool)
        else:
            denoised_df[col] = series

    return denoised_df

# Performance Optimization Utilities


def _get_data_hash(X: pd.DataFrame, y: pd.Series) -> str:
    """Generate hash for caching based on data shape and content."""
    try:
        # Use shape and first/last few values for efficient hashing
        x_hash = hashlib.md5(f"{X.shape}{X.iloc[:5].values.tobytes()}{X.iloc[-5:].values.tobytes()}".encode()).hexdigest()[:16]
        y_hash = hashlib.md5(f"{len(y)}{y.iloc[:5].values.tobytes()}{y.iloc[-5:].values.tobytes()}".encode()).hexdigest()[:16]
        return f"{x_hash}_{y_hash}"
    except Exception:
        return f"{X.shape}_{len(y)}"
def _correlation_clustering_feature_selection(
    X: pd.DataFrame, 
    correlation_threshold: float = 0.7,
    method: str = "hierarchical"
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Advanced correlation clustering for feature redundancy removal.
    
    Uses hierarchical clustering to group correlated features and selects
    the best representative from each cluster based on variance and MI scores.
    
    Args:
        X: Feature DataFrame
        correlation_threshold: Correlation threshold for clustering
        method: Clustering method ('hierarchical', 'spectral', 'ward')
    
    Returns:
        Tuple of (filtered DataFrame, removed_features, clustering_info)
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr
    
    logger.info(f"🔗 Starting correlation clustering: {X.shape[1]} features, threshold={correlation_threshold}")
    
    if X.shape[1] <= 1:
        return X.copy(), [], {"method": method, "clusters": 0}
    
    # Calculate correlation matrix (use Spearman for robustness)
    try:
        corr_matrix = X.corr(method='spearman')
        corr_matrix = corr_matrix.fillna(0)
    except Exception as e:
        logger.warning(f"Spearman correlation failed: {e}, using Pearson")
        corr_matrix = X.corr()
        corr_matrix = corr_matrix.fillna(0)
    
    # Convert to distance matrix
    distance_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix.values, 0)
    
    # Hierarchical clustering
    try:
        if method == "ward":
            # Ward requires Euclidean distance, so we use correlation-based distance
            condensed_distances = squareform(distance_matrix.values)
            linkage_matrix = linkage(condensed_distances, method='ward')
        else:
            # Use correlation directly for other methods
            condensed_distances = squareform(distance_matrix.values)
            linkage_matrix = linkage(condensed_distances, method='average')
        
        # Form clusters based on threshold
        cluster_labels = fcluster(
            linkage_matrix, 
            t=1 - correlation_threshold, 
            criterion='distance'
        )
        
    except Exception as e:
        logger.warning(f"Clustering failed: {e}, using simple correlation filtering")
        # Fallback to simple correlation filtering
        return _simple_correlation_filtering(X, correlation_threshold)
    
    # Analyze clusters
    feature_clusters = {}
    for i, (feature, cluster_id) in enumerate(zip(X.columns, cluster_labels)):
        if cluster_id not in feature_clusters:
            feature_clusters[cluster_id] = []
        feature_clusters[cluster_id].append(feature)
    
    # Select best feature from each cluster
    selected_features = []
    removed_features = []
    cluster_info = {
        "method": method,
        "threshold": correlation_threshold,
        "total_clusters": len(feature_clusters),
        "clusters": {}
    }
    
    for cluster_id, features_in_cluster in feature_clusters.items():
        if len(features_in_cluster) == 1:
            # Single feature cluster - keep it
            selected_features.extend(features_in_cluster)
            cluster_info["clusters"][f"cluster_{cluster_id}"] = {
                "features": features_in_cluster,
                "selected": features_in_cluster,
                "size": 1,
                "action": "kept_single"
            }
        else:
            # Multiple features in cluster - select best one
            best_feature = _select_best_feature_from_cluster(X, features_in_cluster)
            selected_features.append(best_feature)
            
            # Remove other features
            removed = [f for f in features_in_cluster if f != best_feature]
            removed_features.extend(removed)
            
            cluster_info["clusters"][f"cluster_{cluster_id}"] = {
                "features": features_in_cluster,
                "selected": [best_feature],
                "removed": removed,
                "size": len(features_in_cluster),
                "action": "selected_best"
            }
    
    # Create filtered DataFrame
    X_filtered = X[selected_features].copy()
    
    logger.info(
        f"✅ Correlation clustering completed: "
        f"{len(selected_features)} features kept, {len(removed_features)} removed "
        f"({len(feature_clusters)} clusters)"
    )
    
    return X_filtered, removed_features, cluster_info


def _select_best_feature_from_cluster(X: pd.DataFrame, features: List[str]) -> str:
    """
    Select the best feature from a cluster based on multiple criteria.
    
    Args:
        X: Feature DataFrame
        features: List of features in the cluster
    
    Returns:
        Best feature name
    """
    if len(features) == 1:
        return features[0]
    
    # Calculate scores for each feature
    feature_scores = {}
    
    for feature in features:
        scores = []
        
        # 1. Variance score (higher is better)
        variance = X[feature].var()
        variance_score = variance / (variance + 1e-8)  # Normalized
        scores.append(variance_score)
        
        # 2. Stability score (lower coefficient of variation is better)
        if variance > 0:
            cv = np.std(X[feature]) / np.mean(np.abs(X[feature]))
            stability_score = 1 / (1 + cv)  # Inverse CV
        else:
            stability_score = 0.0
        scores.append(stability_score)
        
        # 3. Information content (approximate using entropy-like measure)
        try:
            # Discretize for entropy calculation
            discretized = pd.cut(X[feature], bins=10, labels=False)
            value_counts = pd.Series(discretized).value_counts(normalize=True)
            entropy = -np.sum(value_counts * np.log2(value_counts + 1e-8))
            entropy_score = entropy / np.log2(10)  # Normalized to [0,1]
            scores.append(entropy_score)
        except:
            scores.append(0.0)
        
        # 4. Name preference (shorter, simpler names might be better)
        name_score = 1.0 / (1 + len(feature.split('_')))  # Prefer shorter names
        scores.append(name_score)
        
        # Combined score (weighted average)
        combined_score = np.mean(scores)
        feature_scores[feature] = combined_score
    
    # Select feature with highest score
    best_feature = max(feature_scores, key=feature_scores.get)
    
    return best_feature


def _simple_correlation_filtering(
    X: pd.DataFrame, 
    correlation_threshold: float = 0.7
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Simple correlation filtering as fallback.
    
    Args:
        X: Feature DataFrame
        correlation_threshold: Correlation threshold
    
    Returns:
        Tuple of (filtered DataFrame, removed_features, info)
    """
    corr_matrix = X.corr().abs()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > correlation_threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    # Remove redundant features (keep first occurrence)
    features_to_remove = set()
    for feat_i, feat_j, corr in high_corr_pairs:
        if feat_i not in features_to_remove and feat_j not in features_to_remove:
            features_to_remove.add(feat_j)  # Remove the second feature
    
    selected_features = [f for f in X.columns if f not in features_to_remove]
    X_filtered = X[selected_features].copy()
    
    info = {
        "method": "simple_filtering",
        "threshold": correlation_threshold,
        "high_corr_pairs": len(high_corr_pairs),
        "features_removed": len(features_to_remove)
    }
    
    return X_filtered, list(features_to_remove), info


def _enhanced_smart_feature_pruning(
    X: pd.DataFrame, 
    y: pd.Series, 
    variance_threshold: float = 1e-8, 
    correlation_threshold: float = 0.7,
    use_clustering: bool = True
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Enhanced smart feature pruning with correlation clustering.
    
    Combines variance filtering with advanced correlation clustering
    for more intelligent feature selection.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        variance_threshold: Variance threshold for filtering
        correlation_threshold: Correlation threshold for clustering
        use_clustering: Whether to use clustering or simple filtering
    
    Returns:
        Tuple of (filtered DataFrame, removed_features, pruning_info)
    """
    logger.info(f"🔍 Starting enhanced smart feature pruning: {X.shape[1]} features")
    
    original_features = list(X.columns)
    removed_features = []
    pruning_info = {
        "original_count": len(original_features),
        "steps": []
    }
    
    # Step 1: Variance filtering
    feature_variances = X.var()
    low_variance_features = feature_variances[feature_variances < variance_threshold].index.tolist()
    
    if low_variance_features:
        X = X.drop(columns=low_variance_features)
        removed_features.extend(low_variance_features)
        pruning_info["steps"].append({
            "step": "variance_filtering",
            "removed": low_variance_features,
            "threshold": variance_threshold,
            "remaining": len(X.columns)
        })
        logger.info(f"🗑️  Removed {len(low_variance_features)} low-variance features")
    
    # Step 2: Correlation clustering
    if len(X.columns) > 1:
        if use_clustering:
            X_corr, corr_removed, corr_info = _correlation_clustering_feature_selection(
                X, correlation_threshold, method="hierarchical"
            )
        else:
            X_corr, corr_removed, corr_info = _simple_correlation_filtering(
                X, correlation_threshold
            )
        
        removed_features.extend(corr_removed)
        pruning_info["steps"].append({
            "step": "correlation_clustering" if use_clustering else "correlation_filtering",
            "removed": corr_removed,
            "threshold": correlation_threshold,
            "info": corr_info,
            "remaining": len(X_corr.columns)
        })
        
        X = X_corr
        logger.info(f"🗑️  Removed {len(corr_removed)} correlated features")
    
    # Final summary
    pruning_info["final_count"] = len(X.columns)
    pruning_info["total_removed"] = len(removed_features)
    pruning_info["reduction_ratio"] = len(removed_features) / len(original_features)
    
    logger.info(
        f"✅ Enhanced pruning completed: {len(X.columns)} features remaining "
        f"(removed {len(removed_features)} of {len(original_features)}, "
        f"{pruning_info['reduction_ratio']:.1%} reduction)"
    )
    
    return X, removed_features, pruning_info


@lru_cache(maxsize=128)
def _cached_mutual_info_regression(X_hash: str, y_hash: str, n_features: int, random_state: int = 42) -> tuple:
    """Cached mutual information computation."""
    return None  # Placeholder - will be populated with actual data


def _compute_vectorized_mi_fast(X: pd.DataFrame, y: np.ndarray, n_neighbors: int = 3) -> np.ndarray:
    """Vectorized mutual information computation using sklearn's optimized implementation."""
    try:
        # Use sklearn's optimized mutual_info_regression with fast parameters
        mi_scores = mutual_info_regression(
            X, y, 
            discrete_features='auto',
            n_neighbors=n_neighbors,
            random_state=42
        )
        return mi_scores
    except Exception as e:
        logger.warning(f"Vectorized MI computation failed: {e}, falling back to per-feature computation")
        # Fallback to per-feature computation
        mi_scores = []
        for col in X.columns:
            try:
                score = mutual_info_regression(
                    X[[col]], y, 
                    discrete_features='auto',
                    n_neighbors=n_neighbors,
                    random_state=42
                )[0]
                mi_scores.append(score)
            except Exception:
                mi_scores.append(0.0)
        return np.array(mi_scores)


def _smart_feature_pruning(X: pd.DataFrame, y: pd.Series, variance_threshold: float = 1e-8, 
                          correlation_threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    """Smart feature pruning to remove redundant and low-variance features."""
    logger.info(f"🔍 Starting smart feature pruning: {X.shape[1]} features")
    
    original_features = list(X.columns)
    pruned_features = []
    
    # 1. Remove constant/near-constant features
    feature_variances = X.var()
    low_variance_features = feature_variances[feature_variances < variance_threshold].index.tolist()
    
    if low_variance_features:
        logger.info(f"🗑️  Removing {len(low_variance_features)} low-variance features (< {variance_threshold})")
        X = X.drop(columns=low_variance_features)
        pruned_features.extend(low_variance_features)
    
    # 2. Remove highly correlated features (keep first occurrence)
    if len(X.columns) > 1:
        correlation_matrix = X.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_pairs = []
        for i in range(len(upper_triangle.columns)):
            for j in range(i):
                if upper_triangle.iloc[i, j] > correlation_threshold:
                    col_i = upper_triangle.columns[i]
                    col_j = upper_triangle.columns[j]
                    high_corr_pairs.append((col_i, col_j, upper_triangle.iloc[i, j]))
        
        # Remove features with high correlations (keep the first one encountered)
        features_to_remove = set()
        for col_i, col_j, corr in high_corr_pairs:
            if col_i not in features_to_remove and col_j not in features_to_remove:
                features_to_remove.add(col_j)  # Remove the second feature
        
        if features_to_remove:
            logger.info(f"🗑️  Removing {len(features_to_remove)} highly correlated features (> {correlation_threshold})")
            X = X.drop(columns=list(features_to_remove))
            pruned_features.extend(features_to_remove)
    
    logger.info(f"✅ Smart pruning completed: {X.shape[1]} features remaining (removed {len(pruned_features)})")
    return X, pruned_features


def _median_pruner_callback(current_scores: List[float], median_history: List[float], 
                           prune_threshold: float = 0.5) -> List[int]:
    """Optuna-inspired median pruner for early stopping during feature evaluation."""
    if len(median_history) < 2:
        return []  # Don't prune early
    
    current_median = np.median(current_scores)
    historical_median = np.median(median_history)
    
    # Prune features that are significantly below historical median
    prune_indices = []
    for i, score in enumerate(current_scores):
        if score < historical_median * prune_threshold:
            prune_indices.append(i)
    
    return prune_indices


def _parallel_feature_computation(features: List[str], X: pd.DataFrame, y: np.ndarray, 
                                 compute_func, n_workers: int = 2) -> Dict[str, Any]:
    """Parallel computation of feature metrics."""
    results = {}
    
    def compute_single_feature(feature_name):
        try:
            feature_data = X[feature_name]
            return feature_name, compute_func(feature_data, y)
        except Exception as e:
            logger.debug(f"Failed to compute metric for {feature_name}: {e}")
            return feature_name, None
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs
        future_to_feature = {
            executor.submit(compute_single_feature, feature): feature 
            for feature in features
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_feature):
            feature_name, result = future.result()
            if result is not None:
                results[feature_name] = result
    
    return results


def _optimized_shap_interactions(model, X_sample: pd.DataFrame, max_features: int = 20, 
                                sample_size: int = 1000) -> Dict[str, Any]:
    """Optimized SHAP interaction computation with sampling and feature limiting."""
    try:
        import shap
    except ImportError:
        return {"error": "shap not available"}
    
    # Limit features to top by importance if needed
    if X_sample.shape[1] > max_features:
        # Simple variance-based feature selection for SHAP
        feature_variances = X_sample.var()
        top_features = feature_variances.nlargest(max_features).index
        X_sample = X_sample[top_features]
    
    # Sample data for efficiency
    if len(X_sample) > sample_size:
        X_sample = X_sample.sample(n=sample_size, random_state=42)
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_interactions = explainer.shap_interaction_values(X_sample)
        
        # Handle interaction values shape: (n_samples, n_features, n_features)
        # or (n_outputs, n_samples, n_features, n_features)
        if isinstance(shap_interactions, list):
            # (n_outputs, n_samples, n_features, n_features) -> (n_features, n_features)
            shap_interactions = np.mean(np.abs(np.array(shap_interactions)), axis=(0, 1))
        else:
            if len(shap_interactions.shape) == 4:
                # (n_outputs, n_samples, n_features, n_features) -> (n_features, n_features)
                shap_interactions = np.mean(np.abs(shap_interactions), axis=(0, 1))
            else:
                # (n_samples, n_features, n_features) -> (n_features, n_features)
                shap_interactions = np.mean(np.abs(shap_interactions), axis=0)
        
        # Extract top interactions
        n_features = shap_interactions.shape[0]
        interactions = []
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interaction_strength = float(shap_interactions[i, j])
                if interaction_strength > 1e-6:  # Filter very weak interactions
                    interactions.append({
                        "feature_i": X_sample.columns[i],
                        "feature_j": X_sample.columns[j],
                        "interaction_strength": interaction_strength
                    })
        
        # Sort by strength and return top 20
        interactions.sort(key=lambda x: x["interaction_strength"], reverse=True)
        
        return {
            "n_features": len(X_sample.columns),
            "sample_size": len(X_sample),
            "top_pairs": interactions[:20],
            "method": "optimized_tree_shap"
        }
        
    except Exception as e:
        return {"error": f"SHAP interaction computation failed: {e}"}


def _garbage_collect_optimized():
    """Optimized garbage collection with memory pressure awareness."""
    try:
        # Check memory usage
        memory_usage = psutil.virtual_memory().percent
        
        # Aggressive GC if memory usage is high
        if memory_usage > 80:
            gc.collect(2)  # Perform aggressive collection
        elif memory_usage > 60:
            gc.collect(1)  # Perform standard collection
        else:
            gc.collect(0)  # Minimal collection
        
        logger.debug(f"🧹 Garbage collection completed (memory usage: {memory_usage:.1f}%)")
    except Exception as e:
        logger.debug(f"Garbage collection failed: {e}")



async def _run_specialist_training(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    regime_timeframe: str,
    lookback_days: Optional[float] = None,
    selected_specialists: Optional[list[str]] = None,
    train_per_regime: bool = False,
    regime_labels: Optional[list[str]] = None,
) -> None:
    """Run training for all (or selected) specialist models sequentially."""
    logger.info("🚀 Starting specialist model training sequence")
    tprint_info(
        f"🚀 [Specialists] Training kickoff for {symbol}-{exchange}-{timeframe}-{direction} "
        f"(regime_tf={regime_timeframe}, lookback={lookback_days or 'full'})"
    )

    # Order matters: some steps might depend on artifacts from others,
    # though ideally they should be independent or use shared feature generation.
    default_specialist_steps_enhanced = [
        "enhanced_ml_momentum_persistence_step",
        "enhanced_ml_smc_regime_step",
        "enhanced_ml_volatility_burst_step",
        "enhanced_ml_volume_force_step",
        "enhanced_ml_breakout_bounce_regime_step",
        "enhanced_xgb_macro_regime_step",
        "enhanced_ml_liquidity_regime_step",
        "enhanced_ml_path_regime_step",
        "enhanced_ml_risk_regime_step",
        "enhanced_xgb_meso_regime_step",
        "enhanced_ml_microstructure_step",
        "enhanced_ml_spectral_step",
    ]
    
    specialist_steps = selected_specialists if selected_specialists else default_specialist_steps_enhanced

    # Defensive registry hydration: some enhanced steps can fail to register
    # due to import-order/circular-import issues. If a step is missing, we
    # attempt a targeted import and explicit registration before skipping.
    import importlib

    step_import_map: Dict[str, Tuple[str, str]] = {
        "enhanced_xgb_meso_regime_step": (
            "src.training.steps.market_analysis.xgb_meso_regime_step_enhanced",
            "EnhancedXGBMesoRegimeStep",
        ),
        "enhanced_ml_microstructure_step": (
            "src.training.steps.market_analysis.ml_microstructure_step_enhanced",
            "EnhancedMLMicrostructureStep",
        ),
        "enhanced_ml_spectral_step": (
            "src.training.steps.market_analysis.ml_spectral_step_enhanced",
            "EnhancedMLSpectralStep",
        ),
    }

    config_base = {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "direction": direction,
        "regime_timeframe": regime_timeframe,
        "execution_mode": "full", # default to full for training
    }

    if lookback_days:
        config_base["lookback_days"] = lookback_days

    if train_per_regime and not regime_labels:
        regime_labels = ["Quiet", "Trending", "Chaos"]

    for idx, step_key in enumerate(specialist_steps, start=1):
        # Verify registry key and handle potential aliases (e.g. missing _step suffix)
        try:
            if not step_registry.is_registered(step_key):
                # Attempt targeted import + explicit register for known enhanced steps
                mapping = step_import_map.get(step_key)
                if mapping is not None:
                    module_name, class_name = mapping
                    try:
                        mod = importlib.import_module(module_name)
                        StepCls = getattr(mod, class_name, None)
                        if StepCls is not None and not step_registry.is_registered(step_key):
                            step_registry.register(step_key, StepCls)
                    except Exception as exc:
                        logger.warning(
                            "⚠️ Failed to auto-register %s from %s: %s",
                            step_key,
                            module_name,
                            exc,
                        )

                # Alias fallback
                if step_registry.is_registered(step_key + "_step"):
                    step_key = step_key + "_step"
                else:
                    logger.warning(f"⚠️ Step '{step_key}' not found in registry. Skipping.")
                    continue
        except Exception:
             logger.warning(f"⚠️ Error checking registry for '{step_key}'. Skipping.")
             continue

        progress_tag = f"[{idx}/{len(specialist_steps)}]"
        tprint_info(f"▶️ {progress_tag} Running {step_key} …")
        logger.info(f"▶️ Running {step_key}...")
        try:
            StepClass = step_registry.get_step(step_key)
            step_instance = StepClass(step_name=step_key)

            # Run the step (per-step overrides allowed)
            def _build_step_config() -> Dict[str, Any]:
                step_config = dict(config_base)
                if step_key == "ml_liquidity_regime_step":
                    step_config.setdefault("liquidity_quality_skip_generic_cluster_assessor", True)
                    step_config.setdefault("liquidity_quality_fast_mode", True)
                return step_config

            if train_per_regime and regime_labels:
                for rlab in regime_labels:
                    step_config = _build_step_config()
                    step_config["train_regime_label"] = rlab
                    result = await step_instance.run(step_config)

                    if result.get("success"):
                        msg = f"{progress_tag} {step_key} ({rlab}) completed successfully."
                        logger.info(f"✅ {msg}")
                        tprint_success(f"✅ {msg}")
                    else:
                        err = result.get("error")
                        msg = f"{progress_tag} {step_key} ({rlab}) failed: {err}"
                        logger.error(f"❌ {msg}")
                        tprint_error(f"❌ {msg}")
            else:
                step_config = _build_step_config()
                result = await step_instance.run(step_config)

                if result.get("success"):
                    msg = f"{progress_tag} {step_key} completed successfully."
                    logger.info(f"✅ {msg}")
                    tprint_success(f"✅ {msg}")
                else:
                    err = result.get("error")
                    msg = f"{progress_tag} {step_key} failed: {err}"
                    logger.error(f"❌ {msg}")
                    tprint_error(f"❌ {msg}")
        except Exception as exc:
            logger.error(f"❌ Exception running {step_key}: {exc}")
            tprint_error(f"❌ {progress_tag} Exception in {step_key}: {exc}")

    logger.info("🏁 Specialist model training sequence finished.")
    tprint_success("🏁 All requested specialist trainings finished.")


def _ensure_outcomes_dir() -> Path:
    """Ensure outcomes directory exists and return it."""
    OUTCOMES_DIR.mkdir(exist_ok=True)
    return OUTCOMES_DIR


def _export_report(
    prefix: str,
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    model: str,
    payload: Dict[str, Any],
    markdown_lines: list[str],
) -> Tuple[Path, Path]:
    """Export diagnostics payload as Markdown and CSV into outcomes/.

    Filenames are of the form:
        outcomes/{prefix}_{symbol}_{timeframe}_{YYYYMMDD_%H%M%S}.md/csv
    """
    out_dir = _ensure_outcomes_dir()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_name = f"{prefix}_{symbol}_{timeframe}_{ts}"

    md_path = out_dir / f"{base_name}.md"
    csv_path = out_dir / f"{base_name}.csv"

    with md_path.open("w") as f_md:
        f_md.write("\n".join(markdown_lines))

    # Optional CSV export when feature_metrics are provided in the payload
    feature_metrics = payload.get("feature_metrics")
    model_reliability = payload.get("model_reliability")
    model_pairwise = payload.get("model_pairwise")

    if isinstance(feature_metrics, dict) and feature_metrics:
        try:
            frames: list[pd.DataFrame] = []

            # Base per-feature metrics
            # Filter out special metadata entries
            clean_metrics = {k: v for k, v in feature_metrics.items() if k != "_tv_var_system"}
            df_features = pd.DataFrame.from_dict(clean_metrics, orient="index")

            df_features.index.name = "feature"
            df_features.insert(0, "row_type", "feature")
            frames.append(df_features)

            # Optional per-model reliability metrics
            if isinstance(model_reliability, dict):
                per_model = model_reliability.get("per_model")
                if isinstance(per_model, dict) and per_model:
                    df_models = pd.DataFrame.from_dict(per_model, orient="index")
                    df_models.index.name = "feature"
                    df_models.insert(0, "row_type", "model_reliability")
                    frames.append(df_models)

            # Optional pairwise relationships between specialist models
            if isinstance(model_pairwise, dict):
                pairs = model_pairwise.get("pairs")
                if isinstance(pairs, list) and pairs:
                    df_pairs = pd.DataFrame(pairs)
                    if not df_pairs.empty:
                        df_pairs = df_pairs.copy()
                        pair_index = (
                            df_pairs["model_i"].astype(str)
                            + "|"
                            + df_pairs["model_j"].astype(str)
                        )
                        df_pairs.index = pair_index
                        df_pairs.index.name = "feature"
                        df_pairs.insert(0, "row_type", "model_pairwise")
                        frames.append(df_pairs)

            if frames:
                df_all = pd.concat(frames, axis=0, sort=False)
                df_all.to_csv(csv_path)
                logger.info(
                    "Saved %s diagnostics to %s and %s",
                    prefix,
                    md_path,
                    csv_path,
                )
            else:
                # Fallback: feature metrics only (should not normally trigger)
                df = pd.DataFrame.from_dict(feature_metrics, orient="index")
                df.index.name = "feature"
                df.to_csv(csv_path)
                logger.info(
                    "Saved %s diagnostics to %s and %s",
                    prefix,
                    md_path,
                    csv_path,
                )
        except Exception as csv_exc:  # pragma: no cover - best-effort CSV export
            logger.warning("Failed to export CSV diagnostics table: %s", csv_exc)
    else:
        logger.info("Saved %s diagnostics to %s", prefix, md_path)

    return md_path, csv_path


def _prepare_labels(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    model: str,
    target_col: str,
    lookback_days: Optional[float] = None,
) -> Tuple[pd.Series, pd.DatetimeIndex, pd.Series]:
    """Load labeled_data and return (y, datetime index, realized_return).

    Uses the same loader as snr_diagnostics to ensure compatibility with
    FeatureGenerationMetaLabelingStep artifacts.
    
    Returns:
        Tuple of (target_series, training_index, realized_return_series)
    """
    labeled_df = _load_labeled_data(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
    )

    if target_col not in labeled_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in labeled_data; "
            f"available columns: {sorted(labeled_df.columns)}"
        )

    # Normalize timestamp index. Prefer an existing DatetimeIndex on
    # labeled_df if available; only fall back to timestamp/close_time
    # columns when the index is not already datetime-like.
    if isinstance(labeled_df.index, pd.DatetimeIndex):
        idx = labeled_df.index
        if idx.tz is not None:
            try:
                idx = idx.tz_convert("UTC").tz_localize(None)
            except Exception:
                idx = idx.tz_localize(None)
        labeled_df = labeled_df.copy()
        labeled_df.index = idx
    elif "timestamp" in labeled_df.columns:
        ts_raw = labeled_df["timestamp"]

        # Handle both datetime-like and numeric Unix timestamp columns.
        # Meta-labeling typically stores exchange timestamps in milliseconds,
        # so we interpret numeric values as ms since epoch to avoid bogus
        # 1970-01-01 ranges.
        if pd.api.types.is_datetime64_any_dtype(ts_raw):
            ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
        else:
            numeric = pd.to_numeric(ts_raw, errors="coerce")
            if numeric.notna().any():
                ts = pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
            else:
                ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")

        try:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            ts = ts.dt.tz_localize(None)

        valid_mask = ~ts.isna()
        labeled_df = labeled_df.loc[valid_mask].copy()
        ts = ts[valid_mask]
        labeled_df.index = ts
    elif "close_time" in labeled_df.columns:
        # Fallback for labeled_data artifacts that store event timestamps
        # in a 'close_time' column rather than a generic 'timestamp'.
        close_col = labeled_df["close_time"]
        try:
            if pd.api.types.is_datetime64_any_dtype(close_col):
                ts = pd.to_datetime(close_col, utc=True, errors="coerce")
            else:
                # Infer epoch unit (s, ms, ns) from numeric magnitude to avoid
                # misinterpreting seconds as milliseconds (which would produce
                # bogus 1970-era timestamps).
                close_numeric = pd.to_numeric(close_col, errors="coerce")
                if close_numeric.notna().any():
                    q = float(close_numeric.dropna().quantile(0.5))
                    if q > 1e14:
                        unit = "ns"
                    elif q > 1e11:
                        unit = "ms"
                    elif q > 1e9:
                        unit = "s"
                    else:
                        unit = "ms"
                    ts = pd.to_datetime(close_numeric, unit=unit, utc=True, errors="coerce")
                else:
                    ts = pd.to_datetime(close_col, utc=True, errors="coerce")
        except Exception:
            ts = pd.to_datetime(close_col, utc=True, errors="coerce")
        try:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            ts = ts.dt.tz_localize(None)
        valid_mask = ~ts.isna()
        labeled_df = labeled_df.loc[valid_mask].copy()
        ts = ts[valid_mask]
        labeled_df.index = ts
    else:
        raise ValueError(
            "labeled_data has neither 'timestamp'/'close_time' column nor DatetimeIndex"
        )

    y = labeled_df[target_col].astype(float)
    valid_y = ~y.isna()
    y = y[valid_y]

    # Optional time-based lookback restriction
    if lookback_days is not None and lookback_days > 0:
        try:
            cutoff = y.index.max() - pd.Timedelta(days=float(lookback_days))
            y = y.loc[y.index >= cutoff]
        except Exception as lb_exc:
            logger.warning("Failed to apply lookback_days filter: %s", lb_exc)

    # Debug: inspect target distribution after cleaning/lookback
    try:
        vc = y.value_counts(dropna=False).to_dict()
        logger.info(
            "🎯 Target '%s' distribution after cleaning/lookback_days=%s: n=%d, counts=%s",
            target_col,
            str(lookback_days),
            len(y),
            vc,
        )
    except Exception as dist_exc:
        logger.warning("Failed to compute target distribution summary: %s", dist_exc)

    if len(y) < 100:
        logger.warning(
            "Only %d valid target samples after cleaning/lookback; diagnostics may be noisy",
            len(y),
        )

    # Diagnostics: report training index range used for specialist alignment
    if isinstance(y.index, pd.DatetimeIndex) and len(y.index) > 0:
        logger.info(
            "🎯 Labeled training index range (%s): %s → %s (n=%d)",
            target_col,
            y.index.min(),
            y.index.max(),
            len(y.index),
        )

    # Extract realized_return for PnL simulation (if available)
    if "realized_return" in labeled_df.columns:
        realized_return = labeled_df["realized_return"].astype(float)
        realized_return = realized_return.loc[y.index]
    else:
        # Fallback: use the target column as a proxy if it's a continuous value
        realized_return = pd.Series(index=y.index, dtype=float)
        logger.warning("realized_return column not found in labeled_data; PnL simulation will be limited")

    return y, y.index, realized_return


def _load_specialist_features(
    symbol: str,
    exchange: str,
    base_timeframe: str,
    regime_timeframe: str,
    direction: str,
    model: str,
    training_index: pd.DatetimeIndex,
    enable_risk_hmm_specialist: bool,
    load_regime_label: Optional[str] = None,
) -> pd.DataFrame:
    """Load specialist model outputs aligned to training_index.

    Uses FeatureGenerationMetaLabelingStep's BaseStep machinery to obtain an
    ArtifactRouter instance and then delegates to get_specialist_models_outputs.
    """
    step = FeatureGenerationMetaLabelingStep()
    step.set_context(
        symbol=symbol,
        exchange=exchange,
        timeframe=base_timeframe,
        direction=direction,
        model=model,
    )

    specialist_config: Dict[str, Any] = {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": base_timeframe,
        "regime_timeframe": regime_timeframe,
        "direction": direction,
        "model": model,
        "execution_mode": "full",  # Ensure full dataset usage
        # Mirror training behavior: include the optional HMM risk specialist
        # only when explicitly enabled so diagnostics can match the exact
        # specialist feature block used for training.
        "enable_risk_hmm_specialist": enable_risk_hmm_specialist,
        # For diagnostics we prefer to start from the raw specialist blocks
        # and collapse to scalars locally via _select_specialist_scalars,
        # so that we can choose alternative MR scalars when dense series are
        # effectively constant over the evaluation window.
        # IMPORTANT: keep raw per-specialist feature blocks at load-time.
        # Diagnostics can optionally collapse to scalars later via projection_mode,
        # but orthogonalization/coverage checks need access to the full blocks.
        "use_canonical_specialist_scalars": False,
    }

    if load_regime_label:
        specialist_config["load_regime_label"] = load_regime_label

    specialist_frames: list[pd.DataFrame] = []

    # Load enhanced specialists (best-effort)
    enhanced_specialist_df = get_enhanced_specialist_models_outputs(
        symbol=specialist_config.get("symbol", "ETHUSDT"),
        exchange=specialist_config.get("exchange", "binance"),
        timeframe=specialist_config.get("timeframe", "15m"),
        direction=specialist_config.get("direction", "long"),
        model=specialist_config.get("model", "analyst"),
        training_index=training_index,
        config=specialist_config,
        strict=False,
    )
    if enhanced_specialist_df is not None and not enhanced_specialist_df.empty:
        specialist_frames.append(enhanced_specialist_df)
        step.logger.info("Using enhanced specialist outputs")

    # Also load classic specialists so diagnostics sees full coverage even when
    # enhanced artifacts are only partially available.
    classic_specialist_df = get_specialist_models_outputs(
        artifact_router=step.artifact_router,
        training_index=training_index,
        config=specialist_config,
        logger=step.logger,
        strict=False,
    )
    if classic_specialist_df is not None and not classic_specialist_df.empty:
        specialist_frames.append(classic_specialist_df)

    specialist_df: Optional[pd.DataFrame]
    if specialist_frames:
        specialist_df = pd.concat(specialist_frames, axis=1)
        specialist_df = specialist_df.loc[:, ~specialist_df.columns.duplicated()].copy()
    else:
        specialist_df = None

    if specialist_df is None or specialist_df.empty:
        raise ValueError(
            "No specialist model outputs found for the given context; "
            "ensure the specialist steps have been run first."
        )

    # Ensure strict alignment to the training index
    specialist_df = specialist_df.reindex(training_index, method="ffill")

    # Keep only numeric columns; let downstream metrics logic drop degenerate
    # features (all-NaN or zero variance) rather than failing early here.
    numeric = specialist_df.select_dtypes(include=[np.number, "bool"]).copy()
    if numeric.shape[1] == 0:
        raise ValueError("No numeric specialist features found in specialist_df")

    # Apply Layer0-based global denoising once across all specialist columns.
    try:
        numeric = _apply_global_layer0_denoising(numeric)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("⚠️ Global Layer0 denoising failed; continuing with raw signals: %s", exc)

    # Apply Kalman smoothing to probability-like columns
    for col in numeric.columns:
        if col.endswith("_probability") or col.endswith("_score"):
            logger.info(f"🔮 Applying Kalman smoothing to {col}")
            numeric[col] = _kalman_smooth_series(numeric[col])

    return numeric


def compute_binned_mi(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Compute fast MI approximation using sklearn's optimized implementation."""
    try:
        if len(x) < 2 or len(np.unique(y)) < 2:
            return 0.0
        
        # Clean data
        mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
        x_c, y_c = x[mask], y[mask]
        if len(x_c) < 2:
            return 0.0
        
        # Use sklearn's fast mutual_info_regression with n_neighbors=3 for speed
        mi_score = mutual_info_regression(
            x_c.reshape(-1, 1), y_c,
            discrete_features='auto',
            n_neighbors=3,  # Fast approximation
            random_state=42
        )[0]
        
        return float(mi_score)
    except Exception:
        return 0.0

def _compute_feature_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Compute MI proxy, HSIC, MI stability, correlation, and R^2 per feature with optimizations."""
    logger.info("🚀 Starting optimized feature metrics computation")
    
    # 1. Smart Feature Pruning
    X_original = X.copy()
    X, pruned_features, pruning_info = _enhanced_smart_feature_pruning(X, y, correlation_threshold=0.7, use_clustering=True)
    if X.shape[1] == 0 and X_original.shape[1] > 0:
        logger.warning("⚠️ Smart pruning removed all features; falling back to unpruned feature set")
        X = X_original.copy()
    
    # 2. Align and clean
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index].copy()
    y = y.loc[common_index].astype(float)

    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Replace infinities and fill remaining NaNs in X for robust correlation-based metrics
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Drop breakout/bounce and specialist features from diagnostics
    drop_cols = [
        col
        for col in X.columns
        if col.startswith("breakout_")
        or col in {
            "is_resistance",
            "is_support",
            "support_scalar",
            "resistance_scalar",
            "breakout_success_prob",
            "breakout_high_conf_signal",
        }
    ]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    # Pre-compute numeric target array and detect binary vs continuous target
    y_arr = y.to_numpy(dtype=float)
    uniq = np.unique(y_arr[~np.isnan(y_arr)])
    
    logger.info(f"📊 Target distribution: {pd.Series(y_arr).value_counts(dropna=False).to_dict()}")
    logger.info(f"📊 Target unique values: {uniq.tolist()}")

    # Robust binary check
    is_binary_target = False
    if len(uniq) <= 2 and uniq.size > 0:
        if set(uniq).issubset({0.0, 1.0}):
            is_binary_target = True

    # Debug: log alignment statistics
    try:
        y_var = np.var(y_arr)
        logger.info(
            "📊 Feature metrics alignment: n_samples=%d, n_features=%d, target_var=%.6f, is_binary=%s, unique_vals=%s",
            len(y),
            X.shape[1],
            y_var,
            is_binary_target,
            uniq[:10].tolist() if len(uniq) > 10 else uniq.tolist(),
        )
        
        feat_vars = X.var()
        constant_feats = feat_vars[feat_vars < 1e-12].index.tolist()
        if constant_feats:
            logger.warning("⚠️ Found %d constant features: %s", len(constant_feats), constant_feats[:10])
        logger.info("📊 Mean feature variance: %.6f", feat_vars.mean())

        if is_binary_target:
            class_counts = {
                int(c): int((y_arr == c).sum()) for c in uniq
            }
            logger.info(
                "📊 Binary target class counts after alignment: %s",
                class_counts,
            )
    except Exception as align_exc:
        logger.warning("Failed to log feature-metrics alignment statistics: %s", align_exc)

    if len(y) < max(cv_folds * 5, 50):
        logger.warning(
            "Limited samples (%d) for CV=%d; MI stability estimates may be noisy",
            len(y),
            cv_folds,
        )

    # 3. Vectorized MI Computation
    logger.info("🔥 Computing vectorized MI scores")
    mi_full_vectorized = _compute_vectorized_mi_fast(X, y_arr, n_neighbors=3)
    mi_full = pd.Series(mi_full_vectorized, index=X.columns)

    # 4. HSIC scores with caching
    logger.info("🔗 Computing HSIC scores")
    hsic_scores: Dict[str, float] = {}
    try:
        data_hash = _get_data_hash(X, y)
        fs_utils = get_feature_selection_utils()
        X_mat = X.to_numpy(dtype=float)
        hsic_scores = fs_utils.compute_hsic_features(
            X=X_mat,
            y=y_arr,
            feature_names=list(X.columns),
            kernel="rbf",
        )
    except Exception as hsic_exc:
        logger.warning("Failed to compute HSIC scores; defaulting to 0.0: %s", hsic_exc)
        hsic_scores = {col: 0.0 for col in X.columns}

    # 5. Parallel MI stability computation
    logger.info("⚡ Computing MI stability with parallel processing")
    mi_mean: Dict[str, float] = {}
    mi_cv: Dict[str, float] = {}
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    def compute_mi_stability(feature_name):
        fold_scores = []
        for train_idx, _ in tscv.split(X):
            y_tr = y_arr[train_idx]
            x_tr = X.iloc[train_idx][feature_name].to_numpy(dtype=float)

            mask_tr = ~np.isnan(y_tr)
            if not np.any(mask_tr):
                continue
            y_tr_clean = y_tr[mask_tr]
            x_tr_clean = x_tr[mask_tr]

            if len(np.unique(y_tr_clean)) < 2:
                continue

            score = compute_binned_mi(x_tr_clean, y_tr_clean)
            fold_scores.append(score)

        if fold_scores:
            m = float(np.mean(fold_scores))
            s = float(np.std(fold_scores))
            return feature_name, m, float(s / max(abs(m), 1e-12))
        else:
            return feature_name, 0.0, float("nan")

    # Use parallel processing for MI stability
    stability_results = _parallel_feature_computation(
        list(X.columns), X, y_arr, 
        lambda x, y: compute_binned_mi(x, y),  # This will be replaced in the parallel function
        n_workers=2
    )
    
    # Actually compute stability properly
    for col in X.columns:
        fold_scores = []
        for train_idx, _ in tscv.split(X):
            y_tr = y_arr[train_idx]
            x_tr = X.iloc[train_idx][col].to_numpy(dtype=float)

            mask_tr = ~np.isnan(y_tr)
            if not np.any(mask_tr):
                continue
            y_tr_clean = y_tr[mask_tr]
            x_tr_clean = x_tr[mask_tr]

            if len(np.unique(y_tr_clean)) < 2:
                continue

            score = compute_binned_mi(x_tr_clean, y_tr_clean)
            fold_scores.append(score)

        if fold_scores:
            m = float(np.mean(fold_scores))
            s = float(np.std(fold_scores))
            mi_mean[col] = m
            mi_cv[col] = float(s / max(abs(m), 1e-12))
        else:
            mi_mean[col] = 0.0
            mi_cv[col] = float("nan")

    # 6. Simple Pearson correlation and R^2 per feature
    logger.info("📈 Computing correlation and R^2 metrics")
    metrics: Dict[str, Dict[str, float]] = {}

    for col in X.columns:
        x = X[col].to_numpy(dtype=float)

        # Skip only fully-missing features
        if np.all(np.isnan(x)):
            continue

        try:
            corr = float(np.corrcoef(x, y_arr)[0, 1])
        except Exception:
            corr = float("nan")

        if not np.isfinite(corr):
            r2 = float("nan")
        else:
            r2 = float(corr ** 2)

        metrics[col] = {
            "mi_proxy_full": float(mi_full.get(col, 0.0)),
            "mi_mean_cv": float(mi_mean.get(col, 0.0)),
            "mi_cv": float(mi_cv.get(col, float("inf"))),
            "hsic": float(hsic_scores.get(col, 0.0)),
            "pearson_corr": corr,
            "r2": r2,
        }

    # 7. Garbage collection
    _garbage_collect_optimized()
    
    logger.info(f"✅ Optimized feature metrics computation completed: {len(metrics)} features")
    return metrics

def _compute_feature_metrics_with_tv_var(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    enable_tv_var: bool = True,
    tv_var_window: int = 100,
    tv_var_regime_aware: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Compute enhanced metrics with TV-VAR integration.
    
    Returns:
        tuple: (feature_metrics, tv_var_info)
    """
    
    # First compute standard metrics
    standard_metrics = _compute_feature_metrics(X, y, cv_folds)
    tv_var_info = {"enabled": False}
    
    if not enable_tv_var or len(X) < tv_var_window + 50:
        logger.info("TV-VAR disabled or insufficient data, using standard metrics")
        return standard_metrics, tv_var_info
    
    logger.info("🚀 Computing TV-VAR enhanced metrics")
    
    try:
        # Generate 8 core features for TV-VAR if not already present
        tv_var_features = _generate_tv_var_features_from_specialists(X)
        
        if len(tv_var_features.columns) < 8:
            logger.warning("Insufficient features for TV-VAR, using standard metrics")
            return standard_metrics, tv_var_info
        
        from src.analysis.tv_var_system import TVVARSystem
        
        # Initialize TV-VAR system
        tv_var_system = TVVARSystem(window_size=tv_var_window)
        
        # Fit TV-VAR model
        tv_var_results = tv_var_system.fit_tv_var_monthly_stable(tv_var_features)
        
        # Apply TV-VAR orthogonalization
        X_orthogonalized = tv_var_system.apply_orthogonalization(X, use_decision_tree_rules=True)
        
        # Compute metrics on orthogonalized features
        orthogonalized_metrics = _compute_feature_metrics(X_orthogonalized, y, cv_folds)
        
        # Merge standard and TV-VAR enhanced metrics
        enhanced_metrics = {}
        
        for col in X.columns:
            if col in standard_metrics:
                enhanced_metrics[col] = standard_metrics[col].copy()
                
                # Add TV-VAR enhanced metrics if available
                if col in orthogonalized_metrics:
                    orth_metrics = orthogonalized_metrics[col]
                    enhanced_metrics[col].update({
                        f"tv_var_{k}": v for k, v in orth_metrics.items()
                    })
        
        # Collect TV-VAR system metrics
        tv_var_info = {
            "enabled": True,
            "stability_score": tv_var_results.stability_score,
            "n_features": len(tv_var_features.columns),
            "n_samples": len(tv_var_features),
            "regime_count": len(tv_var_results.regime_assignments.unique()) if hasattr(tv_var_results.regime_assignments, 'unique') else 0,
            "tv_var_regime": tv_var_results.regime_assignments,
        }
        
        logger.info(f"✅ TV-VAR enhanced metrics computed - Stability: {tv_var_results.stability_score:.3f}")
        
        return enhanced_metrics, tv_var_info
        
    except Exception as e:
        logger.error(f"TV-VAR computation failed: {e}, falling back to standard metrics")
        return standard_metrics, tv_var_info


def _generate_tv_var_features_from_specialists(specialist_df: pd.DataFrame) -> pd.DataFrame:
    """Generate 8 core TV-VAR features from specialist outputs."""
    
    features = pd.DataFrame(index=specialist_df.index)
    
    try:
        # Calculate returns from price data if available
        if any("close" in col.lower() for col in specialist_df.columns):
            close_col = next(col for col in specialist_df.columns if "close" in col.lower())
            returns = specialist_df[close_col].pct_change()
        else:
            # Use specialist outputs as proxy for market activity
            returns = specialist_df.mean(axis=1).pct_change()
        
        # 1. Volatility Regime features
        # Short-term realized volatility Z-score
        rv_short = returns.rolling(window=12).std() * np.sqrt(252)
        rv_long = returns.rolling(window=48).std() * np.sqrt(252)
        
        features["rv_z_short"] = (rv_short - rv_short.rolling(100).mean()) / (rv_short.rolling(100).std() + 1e-8)
        features["rv_z_long"] = (rv_long - rv_long.rolling(100).mean()) / (rv_long.rolling(100).std() + 1e-8)
        features["vol_ratio"] = rv_short / (rv_long + 1e-8)
        
        # 2. Liquidity/Participation features
        if "volume" in specialist_df.columns:
            volume = specialist_df["volume"]
        else:
            volume = specialist_df.abs().sum(axis=1)
        
        features["volume_z"] = (volume - volume.rolling(50).mean()) / (volume.rolling(50).std() + 1e-8)
        
        # Spread proxy (using specialist dispersion)
        specialist_std = specialist_df.std(axis=1)
        features["spread_proxy_z"] = (specialist_std - specialist_std.rolling(50).mean()) / (specialist_std.rolling(50).std() + 1e-8)
        
        # 3. Trend/Directional features
        # Use mean specialist output as trend proxy
        trend_proxy = specialist_df.mean(axis=1)
        trend_slope = trend_proxy.rolling(window=24).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 24 else 0)
        features["trend_slope_z"] = (trend_slope - trend_slope.rolling(100).mean()) / (trend_slope.rolling(100).std() + 1e-8)
        features["trend_strength"] = abs(trend_slope)
        
        # 4. Stress/Tail Risk feature
        # Drawdown based on trend proxy
        rolling_max = trend_proxy.rolling(window=50).max()
        drawdown = (trend_proxy - rolling_max) / (rolling_max + 1e-8)
        features["drawdown_z"] = (drawdown - drawdown.rolling(100).mean()) / (drawdown.rolling(100).std() + 1e-8)
        
        # Clean up infinities and NaNs
        features = features.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        
        logger.info(f"✅ Generated {len(features.columns)} TV-VAR features from specialist outputs")
        
    except Exception as e:
        logger.error(f"Failed to generate TV-VAR features: {e}")
        # Return minimal features as fallback
        features = pd.DataFrame(index=specialist_df.index, columns=[
            "rv_z_short", "rv_z_long", "vol_ratio", "volume_z", 
            "spread_proxy_z", "trend_slope_z", "trend_strength", "drawdown_z"
        ]).fillna(0)
    
    return features




def _compute_range_specific_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    min_target: float = 0.015,
    max_target: float = 0.03,
    cv_folds: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Compute range-specific metrics for 1.5-3% trading range."""
    from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
    from sklearn.model_selection import TimeSeriesSplit
    
    # Align and clean
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index].copy()
    y = y.loc[common_index].astype(float)
    
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]
    
    # Replace infinities and fill remaining NaNs
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Create range-specific labels (1.5-3% returns)
    if 'realized_return' in X.columns:
        returns = X['realized_return']
        range_labels = ((returns >= min_target) & (returns <= max_target)).astype(int)
    else:
        # Fallback: use binary target as proxy
        range_labels = (y > 0.5).astype(int)
    
    metrics = {}
    
    # Time series cross-validation for range-specific performance
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    for col in X.columns:
        x = X[col].to_numpy(dtype=float)
        
        if np.all(np.isnan(x)):
            continue
            
        fold_metrics = []
        
        for train_idx, val_idx in tscv.split(X):
            if len(train_idx) < 50 or len(val_idx) < 20:
                continue
                
            x_train, x_val = x[train_idx], x[val_idx]
            y_train, y_val = range_labels.iloc[train_idx], range_labels.iloc[val_idx]
            
            # Skip if no positive examples
            if y_train.sum() == 0 or y_val.sum() == 0:
                continue
            
            try:
                # Simple threshold-based prediction for range
                threshold = np.percentile(x_train, 70)  # Top 30% as signals
                pred = (x_val >= threshold).astype(int)
                
                if len(np.unique(pred)) < 2:
                    continue
                
                # Check if target is binary or continuous
                uniq = np.unique(y_val[~np.isnan(y_val)])
                if len(uniq) <= 2 and set(uniq).issubset({0.0, 1.0}):
                    # Binary classification
                    if len(uniq) > 1:
                        auc = roc_auc_score(y_val, x_val)
                        # Precision-Recall AUC (better for imbalanced data)
                        pr_auc = average_precision_score(y_val, x_val)
                    else:
                        auc = 0.5
                        pr_auc = 0.0
                else:
                    # Regression (continuous target)
                    corr = np.corrcoef(x_val, y_val)[0, 1] if np.var(x_val) > 0 and np.var(y_val) > 0 else 0
                    auc = 0.5 + 0.5 * abs(corr)
                    pr_auc = abs(corr)
                
                # Range hit rate (precision for range targets)
                precision = precision_recall_curve(y_val, x_val)[0][:2].mean()
                
                fold_metrics.append({
                    'auc': auc,
                    'pr_auc': pr_auc,
                    'precision': precision
                })
                
            except Exception as e:
                logger.debug(f"Failed to calculate fold metrics: {e}")
                continue
        
        if fold_metrics:
            avg_metrics = {
                'range_auc': np.mean([m['auc'] for m in fold_metrics]),
                'range_auc_std': np.std([m['auc'] for m in fold_metrics]),
                'range_pr_auc': np.mean([m['pr_auc'] for m in fold_metrics]),
                'range_precision': np.mean([m['precision'] for m in fold_metrics]),
                'range_coverage': len(fold_metrics) / cv_folds
            }
            metrics[col] = avg_metrics
    
    return metrics



def _compute_model_quality_diagnostics(
    feature_metrics: Dict[str, Dict[str, float]],
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, Any]:
    """Compute model quality diagnostics following de Prado framework."""
    diagnostics = {}
    
    # Feature importance stability
    mi_values = [feat_met.get('mi_mean_cv', 0) for k, feat_met in feature_metrics.items() if k != "_tv_var_system"]
    r2_values = [feat_met.get('r2', 0) for k, feat_met in feature_metrics.items() if k != "_tv_var_system"]
    
    if mi_values:
        diagnostics['feature_importance_stability'] = {
            'mi_mean': np.mean(mi_values),
            'mi_std': np.std(mi_values),
            'mi_cv': np.std(mi_values) / (np.mean(mi_values) + 1e-8),
            'r2_mean': np.mean(r2_values),
            'r2_std': np.std(r2_values),
            'n_features': len(mi_values)
        }
    
    # Prediction correlation matrix (if we have predictions)
    # This is a placeholder - would need actual model predictions
    diagnostics['diversity_metrics'] = {
        'avg_feature_correlation': 0.0,  # Would compute from predictions
        'max_pairwise_correlation': 0.0,
        'effective_n_features': len(feature_metrics),
        'redundancy_ratio': 0.0
    }
    
    # Calibration diagnostics
    diagnostics['calibration_quality'] = {
        'reliability_score': 0.0,  # Would compute from calibration curves
        'brier_score': 0.0,
        'expected_calibration_error': 0.0
    }
    
    return diagnostics


def _infer_model_group(feature_name: str) -> str:
    name = feature_name.lower()
    if "risk" in name:
        return "risk"
    if "liquidity" in name:
        return "liquidity"
    if "path" in name:
        return "path"
    if "smc" in name:
        return "smc"
    if "macro" in name:
        return "xgb_macro"
    if "meso" in name:
        return "xgb_meso"
    if "volume" in name or "vol_force" in name:
        return "volume"
    if "momentum" in name:
        return "momentum"
    if "micro" in name:
        return "microstructure"
    if "spectral" in name:
        return "spectral"
    if "volatility" in name:
        return "volatility"
    if "causal" in name or "surprise" in name:
        return "causal"
    return "other"


def _select_specialist_scalars(X: pd.DataFrame) -> pd.DataFrame:
    cols = list(X.columns)

    # ------------------------------------------------------------------
    # Risk: prefer explicit risk_score, otherwise derive from risk_regime
    # ------------------------------------------------------------------
    if "risk_score" not in cols and "risk_regime" in cols:
        rr = X["risk_regime"].astype(float)
        max_rr = float(np.nanmax(rr)) if rr.notna().any() else 0.0
        if np.isfinite(max_rr) and max_rr > 0.0:
            X["risk_score"] = rr / max_rr
        else:
            X["risk_score"] = 0.0
        cols = list(X.columns)

    risk_cols = [c for c in cols if c.startswith("risk_") or c.startswith("enhanced_ml_risk_")]
    if "risk_score" in X.columns or any(c.endswith("_specialist_probability") for c in cols if "risk" in c):
        # Prefer probability from enhanced model if available
        prob_col = next((c for c in cols if "risk" in c and c.endswith("_specialist_probability")), None)
        if prob_col:
            X["risk_score"] = X[prob_col]
        
        risk_keep = {"risk_score"}
        risk_drop = [c for c in risk_cols if c not in risk_keep]
        X = X.drop(columns=risk_drop, errors="ignore")
        cols = list(X.columns)

    # ------------------------------------------------------------------
    # Liquidity: collapse to attractiveness scalar (Regime 3 - Regime 0)
    # ------------------------------------------------------------------
    liq_cols = [c for c in cols if c.startswith("liquidity_") or c.startswith("enhanced_ml_liquidity_")]
    prob_col = next((c for c in cols if "liquidity" in c and c.endswith("_specialist_probability")), None)
    if prob_col:
        X["liquidity_score"] = X[prob_col]
    elif "liquidity_regime_3_prob" in X.columns and "liquidity_regime_0_prob" in X.columns:
        X["liquidity_score"] = X["liquidity_regime_3_prob"] - X["liquidity_regime_0_prob"]
    
    if "liquidity_score" in X.columns:
        liq_keep = {"liquidity_score"}
        liq_drop = [c for c in liq_cols if c not in liq_keep]
        X = X.drop(columns=liq_drop, errors="ignore")
        cols = list(X.columns)

    # ------------------------------------------------------------------
    # Momentum: prefer enhanced probability
    # ------------------------------------------------------------------
    mom_cols = [c for c in cols if c.startswith("momentum_") or c.startswith("enhanced_ml_momentum_")]
    prob_col = next((c for c in cols if "momentum" in c and c.endswith("_specialist_probability")), None)
    if prob_col:
        X["momentum_score"] = X[prob_col]
        mom_keep = {"momentum_score"}
        mom_drop = [c for c in mom_cols if c not in mom_keep]
        X = X.drop(columns=mom_drop, errors="ignore")
        cols = list(X.columns)

    # ------------------------------------------------------------------
    # SMC: prefer enhanced probability
    # ------------------------------------------------------------------
    smc_cols = [c for c in cols if "smc" in c.lower()]
    prob_col = next((c for c in cols if "smc" in c.lower() and c.endswith("_specialist_probability")), None)
    if prob_col:
        X["smc_score"] = X[prob_col]
        smc_keep = {"smc_score"}
        smc_drop = [c for c in smc_cols if c not in smc_keep]
        X = X.drop(columns=smc_drop, errors="ignore")
        cols = list(X.columns)

    # ------------------------------------------------------------------
    # Volatility Burst: prefer enhanced probability
    # ------------------------------------------------------------------
    vol_burst_cols = [c for c in cols if "volatility_burst" in c.lower()]
    prob_col = next((c for c in cols if "volatility_burst" in c.lower() and c.endswith("_specialist_probability")), None)
    if prob_col:
        X["volatility_burst_score"] = X[prob_col]
        vol_burst_keep = {"volatility_burst_score"}
        vol_burst_drop = [c for c in vol_burst_cols if c not in vol_burst_keep]
        X = X.drop(columns=vol_burst_drop, errors="ignore")
        cols = list(X.columns)

    # ------------------------------------------------------------------
    # Volume Force: prefer enhanced probability
    # ------------------------------------------------------------------
    vol_force_cols = [c for c in cols if "volume_force" in c.lower() or "vol_force" in c.lower()]
    prob_col = next((c for c in cols if ("volume_force" in c.lower() or "vol_force" in c.lower()) and c.endswith("_specialist_probability")), None)
    if prob_col:
        X["volume_force_score"] = X[prob_col]
        vol_force_keep = {"volume_force_score"}
        vol_force_drop = [c for c in vol_force_cols if c not in vol_force_keep]
        X = X.drop(columns=vol_force_drop, errors="ignore")
        cols = list(X.columns)

    # ------------------------------------------------------------------
    # Macro/Meso Regime: prefer enhanced probability
    # ------------------------------------------------------------------
    macro_meso_cols = [c for c in cols if "macro" in c.lower() or "meso" in c.lower()]
    for prefix in ["macro", "meso"]:
        prob_col = next((c for c in cols if prefix in c.lower() and c.endswith("_specialist_probability")), None)
        if prob_col:
            X[f"{prefix}_regime_score"] = X[prob_col]
            prefix_cols = [c for c in cols if prefix in c.lower() and c != f"{prefix}_regime_score"]
            X = X.drop(columns=prefix_cols, errors="ignore")
            cols = list(X.columns)

    # ------------------------------------------------------------------
    # Path/Risk: prefer enhanced probability
    # ------------------------------------------------------------------
    path_cols = [c for c in cols if "path" in c.lower()]
    prob_col = next((c for c in cols if "path" in c.lower() and c.endswith("_specialist_probability")), None)
    if prob_col:
        X["path_score"] = X[prob_col]
        path_keep = {"path_score"}
        path_drop = [c for c in path_cols if c not in path_keep]
        X = X.drop(columns=path_drop, errors="ignore")
        cols = list(X.columns)

    # ------------------------------------------------------------------
    # Microstructure/Spectral: prefer enhanced probability
    # ------------------------------------------------------------------
    for prefix in ["microstructure", "spectral"]:
        prob_col = next((c for c in cols if prefix in c.lower() and c.endswith("_specialist_probability")), None)
        if prob_col:
            X[f"{prefix}_score"] = X[prob_col]
            prefix_cols = [c for c in cols if prefix in c.lower() and c != f"{prefix}_score"]
            X = X.drop(columns=prefix_cols, errors="ignore")
            cols = list(X.columns)

    # ------------------------------------------------------------------
    # Candlestick: prefer enhanced probability
    # ------------------------------------------------------------------
    candle_cols = [c for c in cols if "candlestick" in c.lower()]
    prob_col = next((c for c in cols if "candlestick" in c.lower() and c.endswith("_specialist_probability")), None)
    if prob_col:
        X["candlestick_score"] = X[prob_col]
        candle_keep = {"candlestick_score"}
        candle_drop = [c for c in candle_cols if c not in candle_keep]
        X = X.drop(columns=candle_drop, errors="ignore")
        cols = list(X.columns)

    # ------------------------------------------------------------------
    # Reversion: prefer enhanced probability
    # ------------------------------------------------------------------
    reversion_cols = [c for c in cols if "reversion" in c.lower()]
    prob_col = next((c for c in cols if "reversion" in c.lower() and c.endswith("_specialist_probability")), None)
    if prob_col:
        X["reversion_score"] = X[prob_col]
        reversion_keep = {"reversion_score"}
        reversion_drop = [c for c in reversion_cols if c not in reversion_keep]
        X = X.drop(columns=reversion_drop, errors="ignore")
        cols = list(X.columns)
        X = X.drop(columns=mom_drop, errors="ignore")
        cols = list(X.columns)

    # ------------------------------------------------------------------
    # Path: prefer dedicated risk-style scalar if present
    # ------------------------------------------------------------------
    path_cols = [c for c in cols if c.startswith("path_") or c.startswith("enhanced_ml_path_")]
    prob_col = next((c for c in cols if "path" in c and c.endswith("_specialist_probability")), None)
    path_scalar_col: Optional[str] = None
    if prob_col:
        X["path_score"] = X[prob_col]
        path_scalar_col = "path_score"
    elif "path_risk_score" in X.columns:
        path_scalar_col = "path_risk_score"
    elif "path_regime" in X.columns:
        pr = X["path_regime"].astype(float)
        max_pr = float(np.nanmax(pr)) if pr.notna().any() else 0.0
        if np.isfinite(max_pr) and max_pr > 0.0:
            X["path_risk_score"] = pr / max_pr
        else:
            X["path_risk_score"] = 0.0
        path_scalar_col = "path_risk_score"

    if path_scalar_col is not None:
        path_keep = {path_scalar_col}
        path_drop = [c for c in path_cols if c not in path_keep]
        X = X.drop(columns=path_drop, errors="ignore")
        cols = list(X.columns)

    # ------------------------------------------------------------------
    # SMC: keep a single scalar
    # ------------------------------------------------------------------
    smc_cols = [c for c in cols if c.startswith("smc_") or c.startswith("enhanced_ml_smc_")]
    prob_col = next((c for c in cols if "smc" in c and c.endswith("_specialist_probability")), None)
    if prob_col:
        X["smc_predicted"] = X[prob_col]
    
    smc_keep = set()
    if "smc_predicted" in X.columns:
        smc_keep.add("smc_predicted")

    smc_drop = [c for c in smc_cols if c not in smc_keep]
    if smc_drop:
        X = X.drop(columns=smc_drop, errors="ignore")
        cols = list(X.columns)

    # ------------------------------------------------------------------
    # Others: generic handling for enhanced probability
    # ------------------------------------------------------------------
    remaining_specialists = ["spectral", "microstructure", "volatility_burst", "volume_force", "macro", "meso", "candlestick"]
    for spec in remaining_specialists:
        spec_cols = [c for c in cols if spec in c.lower()]
        prob_col = next((c for c in cols if spec in c.lower() and c.endswith("_specialist_probability")), None)
        if prob_col:
            keep = {prob_col}
            drop = [c for c in spec_cols if c not in keep]
            if drop:
                X = X.drop(columns=drop, errors="ignore")
            cols = list(X.columns)

    return X


def _compute_model_reliability(
    feature_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = {}
    for feat, met in feature_metrics.items():
        group = _infer_model_group(feat)
        if group == "other":
            continue
        g = groups.setdefault(
            group,
            {
                "features": [],
                "mi_values": [],
                "r2_values": [],
            },
        )
        mi_val = float(met.get("mi_mean_cv", 0.0))
        r2_val = float(met.get("r2", 0.0))
        g["features"].append(feat)
        if np.isfinite(mi_val):
            g["mi_values"].append(mi_val)
        if np.isfinite(r2_val):
            g["r2_values"].append(r2_val)

    per_model: Dict[str, Dict[str, float]] = {}
    for group, data in groups.items():
        mi_arr = np.array(data["mi_values"], dtype=float)
        r2_arr = np.array(data["r2_values"], dtype=float)
        # Optional HSIC aggregation if present in the metrics
        hsic_vals: list[float] = []
        for feat in data["features"]:
            met = feature_metrics.get(feat, {})
            hsic_val = float(met.get("hsic", 0.0))
            if np.isfinite(hsic_val):
                hsic_vals.append(hsic_val)
        hsic_arr = np.array(hsic_vals, dtype=float) if hsic_vals else np.array([], dtype=float)
        n_features = len(data["features"])
        model_summary: Dict[str, float] = {
            "n_features": int(n_features),
            "mi_mean_avg": float(mi_arr.mean()) if mi_arr.size else 0.0,
            "mi_mean_median": float(np.median(mi_arr)) if mi_arr.size else 0.0,
            "r2_mean": float(r2_arr.mean()) if r2_arr.size else 0.0,
            "r2_median": float(np.median(r2_arr)) if r2_arr.size else 0.0,
            "n_high_mi": int(np.sum(mi_arr > 0.1)) if mi_arr.size else 0,
            "n_high_r2": int(np.sum(r2_arr > 0.05)) if r2_arr.size else 0,
            "hsic_mean": float(hsic_arr.mean()) if hsic_arr.size else 0.0,
            "hsic_median": float(np.median(hsic_arr)) if hsic_arr.size else 0.0,
        }

        best_mi: Optional[float] = None
        best_mi_feat: Optional[str] = None
        best_r2: Optional[float] = None
        best_r2_feat: Optional[str] = None
        best_hsic: Optional[float] = None
        best_hsic_feat: Optional[str] = None

        for feat in data["features"]:
            met = feature_metrics.get(feat, {})
            mi_val = float(met.get("mi_mean_cv", 0.0))
            r2_val = float(met.get("r2", 0.0))
            hsic_val = float(met.get("hsic", 0.0))
            if np.isfinite(mi_val) and (best_mi is None or mi_val > best_mi):
                best_mi = mi_val
                best_mi_feat = feat
            if np.isfinite(r2_val) and (best_r2 is None or r2_val > best_r2):
                best_r2 = r2_val
                best_r2_feat = feat
            if np.isfinite(hsic_val) and (best_hsic is None or hsic_val > best_hsic):
                best_hsic = hsic_val
                best_hsic_feat = feat

        if best_mi is not None:
            model_summary["best_mi_feature_value"] = float(best_mi)
        if best_mi_feat is not None:
            model_summary["best_mi_feature"] = best_mi_feat
        if best_r2 is not None:
            model_summary["best_r2_feature_value"] = float(best_r2)
        if best_r2_feat is not None:
            model_summary["best_r2_feature"] = best_r2_feat
        if best_hsic is not None:
            model_summary["best_hsic_feature_value"] = float(best_hsic)
        if best_hsic_feat is not None:
            model_summary["best_hsic_feature"] = best_hsic_feat

        per_model[group] = model_summary

    ranked_by_mi = sorted(
        per_model.items(), key=lambda kv: kv[1].get("mi_mean_avg", 0.0), reverse=True
    )
    ranked_by_r2 = sorted(
        per_model.items(), key=lambda kv: kv[1].get("r2_mean", 0.0), reverse=True
    )

    return {
        "per_model": per_model,
        "ranked_by_mi": [g for g, _ in ranked_by_mi],
        "ranked_by_r2": [g for g, _ in ranked_by_r2],
    }


def _compute_model_coverage(
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, Any]:
    """Compute data coverage (date range and sample count) per specialist group.

    Coverage is defined over the intersection of the specialist feature index
    and the target index. For each specialist group (alpha, macro_trend,
    liquidity, breakout_bounce, risk, smc), we report:

    - n_samples: number of target samples where at least one feature in that
      group is non-null
    - start / end: first and last timestamps (on the target index) where the
      group has any non-null coverage
    """
    coverage: Dict[str, Dict[str, Any]] = {}

    if not isinstance(X.index, pd.DatetimeIndex):
        return coverage

    common_index = X.index.intersection(y.index)
    if len(common_index) == 0:
        return coverage

    Xc = X.loc[common_index]
    yc = y.loc[common_index]

    # Initialize groups based on features actually present
    for feat in Xc.columns:
        group = _infer_model_group(feat)
        if group == "other":
            continue
        coverage.setdefault(
            group,
            {
                "n_samples": 0,
                "start": None,
                "end": None,
                "fraction_of_target": 0.0,
            },
        )

    if not coverage:
        return coverage

    total = int(len(yc)) if len(yc) > 0 else 0

    for group, info in coverage.items():
        group_cols = [c for c in Xc.columns if _infer_model_group(c) == group]
        if not group_cols:
            continue
        mask = (~Xc[group_cols].isna()).any(axis=1)
        if not mask.any():
            continue
        idx = yc.index[mask]
        if len(idx) == 0:
            continue

        n_samples = int(mask.sum())
        info["n_samples"] = n_samples
        info["start"] = idx.min()
        info["end"] = idx.max()
        info["fraction_of_target"] = float(n_samples / total) if total > 0 else 0.0

    return coverage


def _compute_model_pairwise_relationships(
    X: pd.DataFrame,
    feature_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    group_to_best: Dict[str, str] = {}
    group_to_best_score: Dict[str, float] = {}

    for feat, met in feature_metrics.items():
        group = _infer_model_group(feat)
        if group == "other":
            continue
        score = float(met.get("mi_mean_cv", 0.0))
        prev = group_to_best_score.get(group)
        if prev is None or score > prev:
            group_to_best_score[group] = score
            group_to_best[group] = feat

    if len(group_to_best) < 2:
        return {"error": "Not enough specialist model groups for pairwise analysis"}

    reps: Dict[str, pd.Series] = {}
    for group, feat in group_to_best.items():
        if feat not in X.columns:
            continue
        s = X[feat].astype(float).replace([np.inf, -np.inf], np.nan)
        reps[group] = s

    if len(reps) < 2:
        return {"error": "Representative features missing in X for pairwise analysis"}

    common_index: Optional[pd.DatetimeIndex] = None
    for s in reps.values():
        if common_index is None:
            common_index = s.index
        else:
            common_index = common_index.intersection(s.index)

    if common_index is None or len(common_index) == 0:
        return {"error": "No overlapping samples for pairwise analysis"}

    matrix = pd.DataFrame(
        {g: s.loc[common_index] for g, s in reps.items()},
        index=common_index,
    )
    matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    config = FinalFeatureSelectionConfig()
    component = FinalFeatureSelectionComponent(config=config)

    groups = sorted(matrix.columns)
    pairwise: list[Dict[str, Any]] = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            gi = groups[i]
            gj = groups[j]

            xi = matrix[gi].to_numpy(dtype=float)
            xj = matrix[gj].to_numpy(dtype=float)

            try:
                corr = float(np.corrcoef(xi, xj)[0, 1])
            except Exception:
                corr = float("nan")

            if np.isfinite(corr):
                r2_val = float(corr ** 2)
            else:
                r2_val = float("nan")

            df_i = pd.DataFrame({gi: matrix[gi]})
            df_j = pd.DataFrame({gj: matrix[gj]})

            try:
                # Use absolute correlation as a cheap symmetric MI proxy
                mi_forward = abs(corr) if np.isfinite(corr) else 0.0
                mi_backward = mi_forward
            except Exception:
                mi_forward = 0.0
                mi_backward = 0.0

            mi_sym = 0.5 * (mi_forward + mi_backward)

            pairwise.append(
                {
                    "model_i": gi,
                    "model_j": gj,
                    "rep_feature_i": group_to_best.get(gi),
                    "rep_feature_j": group_to_best.get(gj),
                    "mi_proxy": float(mi_sym),
                    "mi_forward": float(mi_forward),
                    "mi_backward": float(mi_backward),
                    "r2": r2_val,
                }
            )

    pairwise.sort(key=lambda d: d["mi_proxy"], reverse=True)

    return {
        "representatives": group_to_best,
        "pairs": pairwise,
    }


def _compute_probe_models(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Fit simple probe models (LogReg, LightGBM) and report metrics.

    For now this assumes a binary meta-label target (0/1). If the target is
    not approximately binary, the function returns a descriptive payload and
    skips model fitting.
    """
    # Align and clean
    common_index = X.index.intersection(y.index)
    Xc = X.loc[common_index].copy()
    yc = y.loc[common_index].astype(float)

    mask = ~yc.isna()
    Xc = Xc.loc[mask]
    yc = yc.loc[mask]

    # Ensure there are no NaNs in X for sklearn probe models
    Xc = Xc.fillna(0.0)

    result: Dict[str, Any] = {
        "n_samples": int(len(yc)),
        "n_features": int(Xc.shape[1]),
        "task_type": "unknown",
    }

    if len(yc) < max(100, n_splits * 10):
        result["warning"] = (
            "Very few samples for probe models; metrics may be unstable"
        )

    # Determine if target looks binary
    uniq = np.unique(yc.values[~np.isnan(yc.values)])
    if uniq.size == 0:
        result["error"] = "No valid target samples for probe models"
        return result
    # Check if target is binary (subset of {0, 1})
    is_binary = False
    if len(uniq) <= 2:
        if set(uniq).issubset({0.0, 1.0}):
            is_binary = True

    tscv = TimeSeriesSplit(n_splits=n_splits)

    if is_binary:
        # --- Binary Classification Path ---
        y_bin = (yc > 0.5).astype(int)
        pos_frac = float(y_bin.mean())
        result["task_type"] = "binary_classification"
        result["class_balance"] = {"pos_frac": pos_frac, "neg_frac": 1.0 - pos_frac}

        def _collect_scores_clf(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
            try:
                uniq_y = np.unique(y_true)
                if len(uniq_y) < 2:
                    auc = 0.5
                else:
                    auc = roc_auc_score(y_true, probs)
            except Exception:
                auc = 0.5
            
            acc = accuracy_score(y_true, (probs >= 0.5).astype(int))
            brier = brier_score_loss(y_true, probs)
            mse = mean_squared_error(y_true, probs)
            rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")
            r2 = r2_score(y_true, probs)
            return {
                "auc": float(auc) if np.isfinite(auc) else float("nan"),
                "accuracy": float(acc),
                "brier": float(brier),
                "rmse": float(rmse),
                "pseudo_r2": float(r2),
            }

        # Logistic Regression probe
        logreg_scores: Dict[str, list[float]] = {
            "auc": [], "accuracy": [], "brier": [], "rmse": [], "pseudo_r2": []
        }

        for train_idx, test_idx in tscv.split(Xc):
            X_tr, X_te = Xc.iloc[train_idx], Xc.iloc[test_idx]
            y_tr, y_te = y_bin.iloc[train_idx], y_bin.iloc[test_idx]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
            try:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=200, n_jobs=-1, class_weight="balanced")),
                ])
                pipe.fit(X_tr, y_tr)
                p_te = pipe.predict_proba(X_te)[:, 1]
                
                # Calibrate probabilities
                p_tr = pipe.predict_proba(X_tr)[:, 1]
                p_te = _calibrate_probabilities(y_tr.values, p_tr, p_te)
                
                fold = _collect_scores_clf(p_te, y_te.values)
                for k, v in fold.items():
                    if np.isfinite(v):
                        logreg_scores[k].append(v)
            except Exception:
                continue

        if any(logreg_scores.values()):
            result["logreg"] = {}
            for k, v in logreg_scores.items():
                result["logreg"][f"{k}_mean"] = float(np.mean(v)) if v else float("nan")
                result["logreg"][f"{k}_std"] = float(np.std(v)) if v else float("nan")

        # LightGBM probe
        try:
            import lightgbm as lgb  # type: ignore
            lgbm_scores: Dict[str, list[float]] = {
                "auc": [], "accuracy": [], "brier": [], "rmse": [], "pseudo_r2": []
            }
            for train_idx, test_idx in tscv.split(Xc):
                X_tr, X_te = Xc.iloc[train_idx], Xc.iloc[test_idx]
                y_tr, y_te = y_bin.iloc[train_idx], y_bin.iloc[test_idx]
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                    continue
                try:
                    model = lgb.LGBMClassifier(
                        objective="binary", n_estimators=100, learning_rate=0.05,
                        num_leaves=8, max_depth=3, min_child_samples=100,
                        reg_alpha=5.0, reg_lambda=5.0,
                        subsample=0.6, colsample_bytree=0.6,
                        random_state=42, n_jobs=-1, verbose=-1
                    )
                    model.fit(X_tr, y_tr)
                    p_te = model.predict_proba(X_te)[:, 1]
                    
                    # Calibrate probabilities
                    p_tr = model.predict_proba(X_tr)[:, 1]
                    p_te = _calibrate_probabilities(y_tr.values, p_tr, p_te)
                    
                    fold = _collect_scores_clf(p_te, y_te.values)
                    for k, v in fold.items():
                        if np.isfinite(v):
                            lgbm_scores[k].append(v)
                except Exception:
                    continue

            if any(lgbm_scores.values()):
                result["lgbm"] = {}
                for k, v in lgbm_scores.items():
                    result["lgbm"][f"{k}_mean"] = float(np.mean(v)) if v else float("nan")
                    result["lgbm"][f"{k}_std"] = float(np.std(v)) if v else float("nan")
        except ImportError:
            result["lgbm"] = {"error": "lightgbm not available"}

    else:
        # --- Regression Path ---
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error

        result["task_type"] = "regression"

        def _collect_scores_reg(preds: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
            mse = mean_squared_error(y_true, preds)
            rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")
            mae = mean_absolute_error(y_true, preds)
            r2 = r2_score(y_true, preds)
            return {
                "rmse": rmse,
                "mae": float(mae),
                "r2": float(r2),
            }

        # Ridge Regression probe (Linear)
        ridge_scores: Dict[str, list[float]] = {
            "rmse": [], "mae": [], "r2": []
        }

        for train_idx, test_idx in tscv.split(Xc):
            X_tr, X_te = Xc.iloc[train_idx], Xc.iloc[test_idx]
            y_tr, y_te = yc.iloc[train_idx], yc.iloc[test_idx]

            try:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("reg", Ridge(alpha=1.0)),
                ])
                pipe.fit(X_tr, y_tr)
                p_te = pipe.predict(X_te)
                fold = _collect_scores_reg(p_te, y_te.values)
                for k, v in fold.items():
                    if np.isfinite(v):
                        ridge_scores[k].append(v)
            except Exception:
                continue

        if any(ridge_scores.values()):
            result["logreg"] = {}  # Store under logreg key for unified reporting, or separate
            # Actually better to rename for clarity, but keeping structure for report function
            # Renaming key to 'linear_reg' or reuse 'logreg' but report reg metrics

            # We will use 'linear_reg' and update report function to handle it
            result["linear_reg"] = {}
            for k, v in ridge_scores.items():
                result["linear_reg"][f"{k}_mean"] = float(np.mean(v)) if v else float("nan")
                result["linear_reg"][f"{k}_std"] = float(np.std(v)) if v else float("nan")

        # LightGBM Regressor probe
        try:
            import lightgbm as lgb
            lgbm_reg_scores: Dict[str, list[float]] = {
                "rmse": [], "mae": [], "r2": []
            }
            for train_idx, test_idx in tscv.split(Xc):
                X_tr, X_te = Xc.iloc[train_idx], Xc.iloc[test_idx]
                y_tr, y_te = yc.iloc[train_idx], yc.iloc[test_idx]

                try:
                    model = lgb.LGBMRegressor(
                        objective="regression", n_estimators=100, learning_rate=0.05,
                        num_leaves=8, max_depth=3, min_child_samples=100,
                        reg_alpha=5.0, reg_lambda=5.0,
                        subsample=0.6, colsample_bytree=0.6,
                        random_state=42, n_jobs=-1, verbose=-1
                    )
                    model.fit(X_tr, y_tr)
                    p_te = model.predict(X_te)
                    fold = _collect_scores_reg(p_te, y_te.values)
                    for k, v in fold.items():
                        if np.isfinite(v):
                            lgbm_reg_scores[k].append(v)
                except Exception:
                    continue

            if any(lgbm_reg_scores.values()):
                result["lgbm"] = {}
                for k, v in lgbm_reg_scores.items():
                    result["lgbm"][f"{k}_mean"] = float(np.mean(v)) if v else float("nan")
                    result["lgbm"][f"{k}_std"] = float(np.std(v)) if v else float("nan")
        except ImportError:
            result["lgbm"] = {"error": "lightgbm not available"}

    return result


def _compute_per_regime_probe_models(
    X: pd.DataFrame,
    y: pd.Series,
    regimes: pd.Series,
    *,
    n_splits: int = 5,
    min_samples: int = 400,
) -> Dict[str, Any]:
    try:
        if not isinstance(regimes, pd.Series):
            return {"error": "regimes is not a pandas Series"}

        common_index = X.index.intersection(y.index).intersection(regimes.index)
        if len(common_index) == 0:
            return {"error": "No overlapping index between X/y/regimes"}

        Xc = X.loc[common_index].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        yc = y.loc[common_index].astype(float)
        rc = regimes.loc[common_index]

        mask = ~yc.isna() & ~rc.isna()
        Xc = Xc.loc[mask]
        yc = yc.loc[mask]
        rc = rc.loc[mask]

        if len(Xc) < min_samples:
            return {
                "error": f"Insufficient samples for per-regime probes: {len(Xc)} < {min_samples}",
                "n_samples": int(len(Xc)),
            }

        uniq_target = np.unique(yc.values[~np.isnan(yc.values)])
        is_binary = len(uniq_target) <= 2 and set(uniq_target).issubset({0.0, 1.0})

        per_regime: Dict[str, Any] = {}
        for regime_val in sorted(pd.unique(rc)):
            idx = rc.index[rc == regime_val]
            if len(idx) < min_samples:
                per_regime[str(regime_val)] = {
                    "n_samples": int(len(idx)),
                    "error": f"Insufficient samples (<{min_samples})",
                }
                continue

            Xr = Xc.loc[idx]
            yr = yc.loc[idx]

            if len(Xr) < max(100, n_splits * 25):
                per_regime[str(regime_val)] = {
                    "n_samples": int(len(Xr)),
                    "error": "Too few samples for requested CV folds",
                }
                continue

            tscv = TimeSeriesSplit(n_splits=n_splits)

            if is_binary:
                y_bin = (yr > 0.5).astype(int)
                pos_frac = float(y_bin.mean())

                logreg_aucs: list[float] = []
                lgbm_aucs: list[float] = []

                for train_idx, test_idx in tscv.split(Xr):
                    X_tr, X_te = Xr.iloc[train_idx], Xr.iloc[test_idx]
                    y_tr, y_te = y_bin.iloc[train_idx], y_bin.iloc[test_idx]

                    if y_tr.nunique() < 2 or y_te.nunique() < 2:
                        continue

                    try:
                        pipe = Pipeline(
                            [
                                ("scaler", StandardScaler()),
                                (
                                    "clf",
                                    LogisticRegression(
                                        max_iter=300,
                                        solver="lbfgs",
                                        class_weight="balanced",
                                    ),
                                ),
                            ]
                        )
                        pipe.fit(X_tr, y_tr)
                        p_te = pipe.predict_proba(X_te)[:, 1]
                        auc = roc_auc_score(y_te.values, p_te)
                        if np.isfinite(auc):
                            logreg_aucs.append(float(auc))
                    except Exception:
                        pass

                    try:
                        import lightgbm as lgb  # type: ignore

                        clf = lgb.LGBMClassifier(
                            objective="binary",
                            n_estimators=100,
                            learning_rate=0.05,
                            num_leaves=8,
                            max_depth=3,
                            min_child_samples=100,
                            reg_alpha=5.0,
                            reg_lambda=5.0,
                            subsample=0.6,
                            colsample_bytree=0.6,
                            random_state=42,
                            n_jobs=1,
                            verbose=-1,
                        )
                        clf.fit(X_tr, y_tr)
                        p_te = clf.predict_proba(X_te)[:, 1]
                        auc = roc_auc_score(y_te.values, p_te)
                        if np.isfinite(auc):
                            lgbm_aucs.append(float(auc))
                    except Exception:
                        pass

                per_regime[str(regime_val)] = {
                    "n_samples": int(len(Xr)),
                    "pos_frac": float(pos_frac),
                    "logreg_auc_mean": float(np.mean(logreg_aucs)) if logreg_aucs else float("nan"),
                    "logreg_auc_std": float(np.std(logreg_aucs)) if logreg_aucs else float("nan"),
                    "lgbm_auc_mean": float(np.mean(lgbm_aucs)) if lgbm_aucs else float("nan"),
                    "lgbm_auc_std": float(np.std(lgbm_aucs)) if lgbm_aucs else float("nan"),
                    "n_folds_used": int(max(len(logreg_aucs), len(lgbm_aucs))),
                }
            else:
                per_regime[str(regime_val)] = {
                    "n_samples": int(len(Xr)),
                    "task_type": "regression",
                    "note": "per-regime regression probes not implemented",
                }

        return {
            "task_type": "binary_classification" if is_binary else "regression",
            "n_regimes": int(len(per_regime)),
            "min_samples": int(min_samples),
            "per_regime": per_regime,
        }

    except Exception as exc:
        return {"error": str(exc)}

# Add this after the _compute_probe_models function definition

def _compute_probe_models_parallel(X_train, y_train, X_test, y_test, model_type: str = "logreg"):
    """Parallel computation of probe model metrics for a single fold."""
    try:
        if model_type == "logreg":
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                return None
            
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200, n_jobs=-1, class_weight="balanced")),
            ])
            pipe.fit(X_train, y_train)
            p_te = pipe.predict_proba(X_test)[:, 1]
            
            from sklearn.metrics import accuracy_score, brier_score_loss, mean_squared_error, r2_score, roc_auc_score
            
            uniq_y = np.unique(y_test)
            if len(uniq_y) < 2:
                auc = 0.5
            else:
                auc = roc_auc_score(y_test, p_te)
            
            acc = accuracy_score(y_test, (p_te >= 0.5).astype(int))
            brier = brier_score_loss(y_test, p_te)
            mse = mean_squared_error(y_test, p_te)
            rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")
            r2 = r2_score(y_test, p_te)
            
            return {
                "auc": float(auc) if np.isfinite(auc) else float("nan"),
                "accuracy": float(acc),
                "brier": float(brier),
                "rmse": float(rmse),
                "pseudo_r2": float(r2),
            }
        
        elif model_type == "lgbm":
            try:
                import lightgbm as lgb
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    return None
                
                model = lgb.LGBMClassifier(
                    objective="binary", n_estimators=100, learning_rate=0.05,
                    num_leaves=8, max_depth=3, min_child_samples=100,
                    reg_alpha=5.0, reg_lambda=5.0,
                    subsample=0.6, colsample_bytree=0.6,
                    random_state=42, n_jobs=-1, verbose=-1
                )
                model.fit(X_train, y_train)
                p_te = model.predict_proba(X_test)[:, 1]
                
                from sklearn.metrics import accuracy_score, brier_score_loss, mean_squared_error, r2_score, roc_auc_score
                
                uniq_y = np.unique(y_test)
                if len(uniq_y) < 2:
                    auc = 0.5
                else:
                    auc = roc_auc_score(y_test, p_te)
                
                acc = accuracy_score(y_test, (p_te >= 0.5).astype(int))
                brier = brier_score_loss(y_test, p_te)
                mse = mean_squared_error(y_test, p_te)
                rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")
                r2 = r2_score(y_test, p_te)
                
                return {
                    "auc": float(auc) if np.isfinite(auc) else float("nan"),
                    "accuracy": float(acc),
                    "brier": float(brier),
                    "rmse": float(rmse),
                    "pseudo_r2": float(r2),
                }
            except ImportError:
                return {"error": "lightgbm not available"}
        
        return None
    except Exception:
        return None

def _detect_feature_leakage(
    X: pd.DataFrame,
    y: pd.Series,
    component: FinalFeatureSelectionComponent,
) -> Dict[str, Any]:
    """Use FinalFeatureSelectionComponent's leakage detector on specialist features."""
    try:
        leakage = component.detect_potential_leakage(
            X=X,
            y=y,
            selected_features=list(X.columns),
        )
        return leakage
    except Exception as exc:
        logger.warning("Leakage detection failed: %s", exc)
        return {"error": str(exc)}


def _compute_global_stability(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Compute a simple global stability metric via CV variability (AUC for clf, R2 for reg)."""
    common_index = X.index.intersection(y.index)
    Xc = X.loc[common_index].copy()
    yc = y.loc[common_index].astype(float)
    mask = ~yc.isna()
    Xc = Xc.loc[mask]
    yc = yc.loc[mask]

    # Ensure there are no NaNs in X for sklearn probe model
    Xc = Xc.fillna(0.0)

    # Determine task type
    uniq = np.unique(yc.values[~np.isnan(yc.values)])
    is_binary = len(uniq) <= 2 and set(uniq).issubset({0.0, 1.0})
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores: list[float] = []

    for train_idx, test_idx in tscv.split(Xc):
        X_tr, X_te = Xc.iloc[train_idx], Xc.iloc[test_idx]
        y_tr, y_te = yc.iloc[train_idx], yc.iloc[test_idx]
        
        if is_binary:
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
            try:
                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=200, n_jobs=-1, class_weight="balanced")),
                    ]
                )
                pipe.fit(X_tr, y_tr)
                p_te = pipe.predict_proba(X_te)[:, 1]
                auc = roc_auc_score(y_te.values, p_te)
                if np.isfinite(auc):
                    scores.append(float(auc))
            except Exception:
                continue
        else:
            try:
                from sklearn.linear_model import Ridge
                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("reg", Ridge(alpha=1.0)),
                    ]
                )
                pipe.fit(X_tr, y_tr)
                p_te = pipe.predict(X_te)
                r2 = r2_score(y_te.values, p_te)
                if np.isfinite(r2):
                    scores.append(float(r2))
            except Exception:
                continue

    if not scores:
        return {"error": "Insufficient folds for stability analysis"}

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    stability = float(1.0 - std_score / abs(mean_score)) if abs(mean_score) > 1e-6 else float("nan")
    
    metric_name = "auc" if is_binary else "r2"
    return {
        "n_splits": int(n_splits),
        "task_type": "binary_classification" if is_binary else "regression",
        f"fold_{metric_name}s": scores,
        f"mean_{metric_name}": mean_score,
        f"std_{metric_name}": std_score,
        "stability_score": stability,
    }



def _compute_tree_shap_interactions(
    X: pd.DataFrame,
    y: pd.Series,
    feature_metrics: Dict[str, Dict[str, float]],
    max_features: int = 20,  # Reduced from 30 for speed
    sample_size: int = 1000,  # Reduced from 2000 for speed
) -> Dict[str, Any]:
    """Optimized SHAP interaction computation with sampling and feature limiting."""
    logger.info("🔍 Starting optimized SHAP interaction computation")
    
    try:
        import lightgbm as lgb  # type: ignore
        import shap  # type: ignore
    except ImportError as exc:
        return {"error": f"lightgbm or shap not available: {exc}"}

    # Align and clean
    common_index = X.index.intersection(y.index)
    Xc = X.loc[common_index].copy()
    yc = y.loc[common_index].astype(float)
    mask = ~yc.isna()
    Xc = Xc.loc[mask]
    yc = yc.loc[mask]

    # Select top features by MI_mean (or fall back to all)
    if feature_metrics:
        # Filter out system metrics and ensure we have valid MI scores
        valid_items = [
            item for item in feature_metrics.items() 
            if item[0] != "_tv_var_system" and isinstance(item[1], dict) and "mi_mean_cv" in item[1]
        ]
        ranked = sorted(
            valid_items,
            key=lambda kv: kv[1].get("mi_mean_cv", 0.0),
            reverse=True,
        )
        top_names = [name for name, _ in ranked[:max_features]]
    else:
        # Simple variance-based selection if no metrics available
        feature_variances = Xc.var()
        top_names = feature_variances.nlargest(max_features).index.tolist()

    X_sel = Xc[top_names].copy()
    logger.info(f"📊 Selected {len(top_names)} features for SHAP analysis")

    # Optimized sampling
    if len(X_sel) > sample_size:
        # Use stratified sampling for better representation
        if len(yc) > sample_size * 2:
            # Take samples from different parts of the dataset
            first_part = X_sel.iloc[:sample_size//3]
            middle_part = X_sel.iloc[len(X_sel)//2 - sample_size//6:len(X_sel)//2 + sample_size//6]
            last_part = X_sel.iloc[-sample_size//3:]
            
            X_sample = pd.concat([first_part, middle_part, last_part]).head(sample_size)
            y_sample = yc.loc[X_sample.index]
        else:
            X_sample = X_sel.sample(n=sample_size, random_state=42)
            y_sample = yc.loc[X_sample.index]
    else:
        X_sample = X_sel
        y_sample = yc

    logger.info(f"📈 Using {len(X_sample)} samples for SHAP computation")

    # Determine task type based on target
    uniq = np.unique(y_sample[~np.isnan(y_sample)])
    is_binary = len(uniq) <= 2 and set(uniq).issubset({0.0, 1.0})

    # Use optimized model parameters for speed
    if is_binary:
        if len(uniq) < 2:
            return {"error": "Insufficient classes for SHAP classification"}
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=8,
            max_depth=3,
            min_child_samples=100,
            reg_alpha=5.0,
            reg_lambda=5.0,
            subsample=0.6,
            colsample_bytree=0.6,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    else:
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=100,  # Reduced from 200
            learning_rate=0.1,  # Increased for faster convergence
            num_leaves=15,
            max_depth=3,  # Reduced from 4
            min_child_samples=50,
            reg_alpha=0.1,
            reg_lambda=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,  # Reduced for memory efficiency
            verbose=-1
        )
    
    model.fit(X_sample, y_sample)

    # Use optimized SHAP computation
    try:
        return _optimized_shap_interactions(model, X_sample, max_features, sample_size)
    except Exception as exc:
        logger.warning(f"Optimized SHAP failed, trying fallback: {exc}")
        
        # Fallback to original method with optimizations
        try:
            explainer = shap.TreeExplainer(model)
            shap_int = explainer.shap_interaction_values(X_sample)
            
            # shap_int shape: (n_samples, n_features, n_features) or list of such
            if isinstance(shap_int, list):
                # For binary classification, shap returns list per class; use average across classes and then samples
                shap_int_arr = np.mean(np.abs(np.array(shap_int)), axis=(0, 1))
            else:
                # For regression, it's (n_samples, n_features, n_features)
                if len(shap_int.shape) == 4: # (n_outputs, n_samples, n_features, n_features)
                    shap_int_arr = np.mean(np.abs(shap_int), axis=(0, 1))
                else:
                    shap_int_arr = np.mean(np.abs(shap_int), axis=0)

            # Ensure we are (n_features, n_features)
            if len(shap_int_arr.shape) > 2:
                shap_int_arr = np.mean(shap_int_arr, axis=tuple(range(len(shap_int_arr.shape) - 2)))

            n_feat = shap_int_arr.shape[0]
            pairs: list[Dict[str, Any]] = []
            for i in range(n_feat):
                for j in range(i + 1, n_feat):
                    score = float(shap_int_arr[i, j])
                    # Only include meaningful interactions
                    if score > 1e-6:
                        pairs.append(
                            {
                                "feature_i": top_names[i],
                                "feature_j": top_names[j],
                                "interaction_strength": score,
                            }
                        )

            if not pairs:
                return {"error": "No meaningful interaction pairs computed"}

            pairs.sort(key=lambda d: d["interaction_strength"], reverse=True)
            top_pairs = pairs[:20]

            # Garbage collection
            _garbage_collect_optimized()

            return {
                "n_features": len(top_names),
                "sample_size": int(len(X_sample)),
                "top_pairs": top_pairs,
                "method": "optimized_tree_shap_fallback",
            }
        except Exception as fallback_exc:
            return {"error": f"Both optimized and fallback SHAP failed: {fallback_exc}"}

def _compute_trading_simulation_pnl_dynamic(
    confidence_scores: pd.Series,
    realized_returns: pd.Series,
    thresholds: Optional[list[float]] = None,
    tp_pct: float = 0.02,  # 2% take profit
    sl_pct: float = 0.007,  # 0.7% stop loss
    fee_per_trade: float = 0.0015,  # 0.15% per trade (0.3% round trip)
    dynamic_thresholds: bool = True,
    returns_for_volatility: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Enhanced PnL computation with dynamic thresholding."""
    
    # Use dynamic thresholds if enabled or no thresholds provided
    if dynamic_thresholds or thresholds is None:
        try:
            if returns_for_volatility is not None:
                dynamic_threshs = calculate_dynamic_thresholds_batch(
                    predictions=confidence_scores,
                    returns=returns_for_volatility,
                    method="adaptive",
                    base_threshold=0.55,  # Lower base threshold
                    min_trades_target=3
                )
            else:
                # Fallback to percentile-based thresholds
                clean_preds = confidence_scores.dropna()
                if len(clean_preds) > 30:
                    dynamic_threshs = [
                        clean_preds.quantile(0.55),
                        clean_preds.quantile(0.65),
                        clean_preds.quantile(0.75),
                        clean_preds.quantile(0.85)
                    ]
                else:
                    dynamic_threshs = [0.5, 0.6, 0.7, 0.8]
            
            thresholds = dynamic_threshs
            logger.info(f"🔄 Using dynamic thresholds: {[f'{t:.3f}' for t in thresholds]}")
        except Exception as e:
            logger.warning(f"Dynamic threshold calculation failed: {e}, using defaults")
            thresholds = [0.5, 0.6, 0.7, 0.8]
    
    # Ensure thresholds are sorted and valid
    thresholds = sorted([t for t in thresholds if 0.5 <= t <= 0.95])
    if not thresholds:
        thresholds = [0.5, 0.6, 0.7, 0.8]
    
    # Rest of the original function with enhanced logging
    conf = confidence_scores.copy()
    ret = realized_returns.copy()
    
    # Align and clean
    common_idx = conf.index.intersection(ret.index)
    conf = conf.loc[common_idx]
    ret = ret.loc[common_idx]
    
    mask = ~(conf.isna() | ret.isna() | np.isinf(conf) | np.isinf(ret))
    conf = conf.loc[mask]
    ret = ret.loc[mask]
    
    if len(conf) == 0:
        return {"error": "No valid data after cleaning"}
    
    results = {
        "per_threshold": {},
        "summary": {},
        "thresholds": thresholds,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "fee_round_trip_pct": fee_per_trade * 2,
    }
    total_trades_by_threshold = {}
    
    for thresh in thresholds:
        # Generate signals
        signals = (conf >= thresh).astype(int)
        
        # Calculate trade metrics
        trades = signals.sum()
        thresh_key = f"{thresh:.3f}"
        
        if trades == 0:
            results["per_threshold"][thresh_key] = {
                "n_trades": 0,
                "trades_per_day": 0.0,
                "win_rate": 0.0,
                "avg_pnl_per_trade_pct": 0.0,
                "avg_pnl_per_day_pct": 0.0,
                "avg_pnl_per_month_pct": 0.0,
                "sharpe_estimate": 0.0,
            }
            total_trades_by_threshold[thresh] = 0
            continue
        
        # Calculate PnL for trades
        trade_returns = ret[signals == 1]
        
        # Apply transaction costs
        net_returns = trade_returns - fee_per_trade
        
        # Calculate metrics
        n_wins = int((net_returns > 0).sum())
        win_rate = n_wins / trades if trades > 0 else 0.0
        
        total_pnl_pct = float(net_returns.sum()) * 100  # As percentage
        avg_pnl_per_trade_pct = float(net_returns.mean()) * 100
        
        date_range_days = (conf.index.max() - conf.index.min()).days
        trades_per_day = trades / max(date_range_days, 1)
        pnl_per_day_pct = total_pnl_pct / max(date_range_days, 1)
        pnl_per_month_pct = total_pnl_pct / (date_range_days / 30.44)  # Average days per month
        
        # Estimate Sharpe-like metric (annualized)
        if len(net_returns) > 1 and net_returns.std() > 0:
            sharpe = (net_returns.mean() * 252) / (net_returns.std() * np.sqrt(252))
        else:
            sharpe = 0.0
        
        results["per_threshold"][thresh_key] = {
            "n_trades": int(trades),
            "trades_per_day": float(trades_per_day),
            "win_rate": float(win_rate),
            "avg_pnl_per_trade_pct": float(avg_pnl_per_trade_pct),
            "avg_pnl_per_day_pct": float(pnl_per_day_pct),
            "avg_pnl_per_month_pct": float(pnl_per_month_pct),
            "sharpe_estimate": float(sharpe),
        }
        
        total_trades_by_threshold[thresh] = trades
    
    # Summary statistics
    total_trades = sum(total_trades_by_threshold.values())
    results["summary"] = {
        "total_trades_all_thresholds": int(total_trades),
        "date_range_days": float(max((conf.index.max() - conf.index.min()).days, 1)),
        "total_samples": int(len(conf)),
        "thresholds_used": thresholds,
        "dynamic_thresholds": dynamic_thresholds,
    }
    
    # Also add top level keys for legacy compatibility
    results["date_range_days"] = float(max((conf.index.max() - conf.index.min()).days, 1))
    results["total_samples"] = int(len(conf))
    
    # Log summary
    logger.info(
        f"📊 PnL Summary: {total_trades} total trades across {len(thresholds)} thresholds "
        f"(dynamic: {dynamic_thresholds})"
    )
    
    return results
def _compute_trading_simulation_pnl(
    confidence_scores: pd.Series,
    realized_returns: pd.Series,
    thresholds: list[float] = [0.6, 0.7, 0.8, 0.9],
    tp_pct: float = 0.02,  # 2% take profit (User requested standard)
    sl_pct: float = 0.007,  # 0.7% stop loss (User requested standard)
    fee_per_trade: float = 0.0015,  # 0.15% per trade (0.3% round trip)
    dynamic_thresholds: bool = True,
) -> Dict[str, Any]:
    """Compute trading simulation PnL for specialist/probe models at various confidence thresholds.
    
    Uses a simplified triple-barrier approach:
    - TP: Take profit at +2%
    - SL: Stop loss at -0.7%
    - Fees: 0.15% per trade (0.3% round-trip)
    
    For each threshold, computes:
    - Number of trades
    - Average trades per day
    - Win rate
    - Average PnL % per-day and per-month
    
    Args:
        confidence_scores: Model confidence/probability scores (0-1)
        realized_returns: Actual realized returns for each event
        thresholds: Confidence thresholds to evaluate
        tp_pct: Take profit percentage (default: 2%)
        sl_pct: Stop loss percentage (default: 0.7%)
        fee_per_trade: Fee per trade as fraction (default: 0.15%)
        
    Returns:
        Dictionary with results per threshold
    """
    results = {
        "thresholds": thresholds,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "fee_round_trip_pct": fee_per_trade * 2,
        "per_threshold": {},
    }
    
    # Align confidence and returns
    common_idx = confidence_scores.index.intersection(realized_returns.index)
    if len(common_idx) < 10:
        return {"error": f"Insufficient overlapping data: {len(common_idx)} samples"}
    
    conf = confidence_scores.loc[common_idx].astype(float)
    rets = realized_returns.loc[common_idx].astype(float)
    
    # Drop NaN values
    valid_mask = conf.notna() & rets.notna()
    conf = conf[valid_mask]
    rets = rets[valid_mask]
    
    if len(conf) < 10:
        return {"error": f"Insufficient valid data after NaN removal: {len(conf)} samples"}
    
    # Calculate date range for per-day/per-month metrics
    if isinstance(conf.index, pd.DatetimeIndex):
        date_range_days = (conf.index.max() - conf.index.min()).days
        if date_range_days <= 0:
            date_range_days = 1
    else:
        date_range_days = len(conf) / 96  # Assume 15m bars, ~96 per day
    
    date_range_months = date_range_days / 30.44  # Average days per month
    
    for threshold in thresholds:
        # Filter trades above threshold
        trade_mask = conf >= threshold
        n_trades = int(trade_mask.sum())
        
        if n_trades == 0:
            results["per_threshold"][f"{threshold:.0%}"] = {
                "threshold": threshold,
                "n_trades": 0,
                "trades_per_day": 0.0,
                "win_rate": 0.0,
                "total_pnl_pct": 0.0,
                "avg_pnl_per_trade_pct": 0.0,
                "avg_pnl_per_day_pct": 0.0,
                "avg_pnl_per_month_pct": 0.0,
                "sharpe_estimate": 0.0,
            }
            continue
        
        # Get returns for trades above threshold
        trade_returns = rets[trade_mask]
        
        # Apply triple-barrier logic (simplified):
        # If return >= tp_pct: profit = tp_pct - fees
        # If return <= -sl_pct: loss = -sl_pct - fees  
        # Otherwise: use actual return - fees
        def apply_barrier(ret: float) -> float:
            if ret >= tp_pct:
                return tp_pct - fee_per_trade * 2  # Hit TP
            elif ret <= -sl_pct:
                return -sl_pct - fee_per_trade * 2  # Hit SL
            else:
                return ret - fee_per_trade * 2  # Actual return minus fees
        
        barrier_returns = trade_returns.apply(apply_barrier)
        
        # Calculate metrics
        n_wins = int((barrier_returns > 0).sum())
        win_rate = n_wins / n_trades if n_trades > 0 else 0.0
        
        total_pnl_pct = float(barrier_returns.sum()) * 100  # As percentage
        avg_pnl_per_trade_pct = float(barrier_returns.mean()) * 100
        
        trades_per_day = n_trades / date_range_days if date_range_days > 0 else 0.0
        avg_pnl_per_day_pct = total_pnl_pct / date_range_days if date_range_days > 0 else 0.0
        avg_pnl_per_month_pct = total_pnl_pct / date_range_months if date_range_months > 0 else 0.0
        
        # Estimate Sharpe-like metric (annualized)
        if len(barrier_returns) > 1 and barrier_returns.std() > 0:
            daily_sharpe = (barrier_returns.mean() / barrier_returns.std()) * np.sqrt(trades_per_day)
            annualized_sharpe = daily_sharpe * np.sqrt(252)
        else:
            annualized_sharpe = 0.0
        
        results["per_threshold"][f"{threshold:.0%}"] = {
            "threshold": threshold,
            "n_trades": n_trades,
            "trades_per_day": round(trades_per_day, 2),
            "win_rate": round(win_rate, 4),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "avg_pnl_per_trade_pct": round(avg_pnl_per_trade_pct, 4),
            "avg_pnl_per_day_pct": round(avg_pnl_per_day_pct, 4),
            "avg_pnl_per_month_pct": round(avg_pnl_per_month_pct, 2),
            "sharpe_estimate": round(annualized_sharpe, 2),
        }
    
    results["date_range_days"] = round(date_range_days, 1)
    results["total_samples"] = len(conf)
    
    return results


def _compute_probe_model_pnl(
    X: pd.DataFrame,
    y: pd.Series,
    realized_returns: pd.Series,
    n_splits: int = 5,
    thresholds: Optional[list[float]] = None,
    dynamic_thresholds: bool = True,
) -> Dict[str, Any]:
    """Compute trading PnL metrics for probe models using cross-validation predictions.
    
    Supports both classification (LogisticRegression, LGBMClassifier) and
    regression (Ridge, LGBMRegressor) depending on the target type.
    """
    results = {}
    
    # Check for valid data
    common_idx = X.index.intersection(y.index).intersection(realized_returns.index)
    if len(common_idx) < 100:
        return {"error": f"Insufficient overlapping data: {len(common_idx)} samples"}
    
    X_aligned = X.loc[common_idx].copy()
    y_aligned = y.loc[common_idx].copy()
    rets_aligned = realized_returns.loc[common_idx].copy()
    
    # Drop NaN values
    valid_mask = y_aligned.notna() & rets_aligned.notna()
    X_aligned = X_aligned.loc[valid_mask]
    y_aligned = y_aligned[valid_mask]
    rets_aligned = rets_aligned[valid_mask]
    
    # Fill NaN in features
    X_aligned = X_aligned.fillna(0.0)
    
    if len(X_aligned) < 100:
        return {"error": f"Insufficient valid data: {len(X_aligned)} samples"}
    
    # Determine task type
    y_vals = y_aligned.values
    uniq = np.unique(y_vals)
    is_binary = False
    if len(uniq) <= 2:
        if set(uniq).issubset({0.0, 1.0}):
            is_binary = True

    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    if is_binary:
        # --- Classification PnL ---
        y_train_target = (y_aligned > 0.5).astype(int)
        
        oos_probs_logreg = pd.Series(index=X_aligned.index, dtype=float)
        oos_probs_lgbm = pd.Series(index=X_aligned.index, dtype=float)
        
        for train_idx, val_idx in tscv.split(X_aligned):
            X_train, X_val = X_aligned.iloc[train_idx], X_aligned.iloc[val_idx]
            y_train = y_train_target.iloc[train_idx]

            # Logistic Regression
            try:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)),
                ])
                pipe.fit(X_train, y_train)
                # Calibrate probabilities if enabled
                p_tr = pipe.predict_proba(X_train)[:, 1]
                p_te = pipe.predict_proba(X_val)[:, 1]
                p_te = _calibrate_probabilities(y_train.values, p_tr, p_te)
                oos_probs_logreg.iloc[val_idx] = p_te
            except Exception:
                pass

            # LightGBM
            try:
                import lightgbm as lgb
                lgbm_clf = lgb.LGBMClassifier(
                    objective="binary", n_estimators=100, learning_rate=0.05,
                    num_leaves=8, max_depth=3, min_child_samples=100,
                    reg_alpha=5.0, reg_lambda=5.0,
                    subsample=0.6, colsample_bytree=0.6,
                    random_state=42, n_jobs=-1, verbose=-1
                )
                lgbm_clf.fit(X_train, y_train)
                # Calibrate probabilities if enabled
                p_tr = lgbm_clf.predict_proba(X_train)[:, 1]
                p_te = lgbm_clf.predict_proba(X_val)[:, 1]
                p_te = _calibrate_probabilities(y_train.values, p_tr, p_te)
                oos_probs_lgbm.iloc[val_idx] = p_te
            except Exception:
                pass

        # Compute trading PnL for classification using dynamic thresholds
        if oos_probs_logreg.notna().sum() >= 10:
            results["logreg"] = _compute_trading_simulation_pnl_dynamic(
                confidence_scores=oos_probs_logreg,
                realized_returns=rets_aligned,
                thresholds=thresholds,
                dynamic_thresholds=dynamic_thresholds,
                returns_for_volatility=rets_aligned
            )
        else:
            results["logreg"] = {"error": "Insufficient LogReg predictions"}

        if oos_probs_lgbm.notna().sum() >= 10:
            results["lgbm"] = _compute_trading_simulation_pnl_dynamic(
                confidence_scores=oos_probs_lgbm,
                realized_returns=rets_aligned,
                thresholds=thresholds,
                dynamic_thresholds=dynamic_thresholds,
                returns_for_volatility=rets_aligned
            )
        else:
            results["lgbm"] = {"error": "Insufficient LGBM predictions"}

    else:
        # --- Regression PnL ---
        from sklearn.linear_model import Ridge

        oos_preds_linear = pd.Series(index=X_aligned.index, dtype=float)
        oos_preds_lgbm = pd.Series(index=X_aligned.index, dtype=float)

        for train_idx, val_idx in tscv.split(X_aligned):
            X_train, X_val = X_aligned.iloc[train_idx], X_aligned.iloc[val_idx]
            y_train = y_aligned.iloc[train_idx]

            # Ridge Regression
            try:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("reg", Ridge(alpha=1.0)),
                ])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_val)
                oos_preds_linear.iloc[val_idx] = preds
            except Exception:
                pass

            # LightGBM Regressor
            try:
                import lightgbm as lgb
                lgbm_reg = lgb.LGBMRegressor(
                    objective="regression", n_estimators=100, learning_rate=0.05,
                    num_leaves=8, max_depth=3, min_child_samples=100,
                    reg_alpha=5.0, reg_lambda=5.0,
                    subsample=0.6, colsample_bytree=0.6,
                    random_state=42, n_jobs=-1, verbose=-1
                )
                lgbm_reg.fit(X_train, y_train)
                preds = lgbm_reg.predict(X_val)
                oos_preds_lgbm.iloc[val_idx] = preds
            except Exception:
                pass

        # Compute trading PnL for regression (using predicted return as score)
        # Note: dynamic thresholding for regression expects probabilities, but we can pass returns
        if oos_preds_linear.notna().sum() >= 10:
            results["linear_reg"] = _compute_trading_simulation_pnl_dynamic(
                confidence_scores=oos_preds_linear,
                realized_returns=rets_aligned,
                thresholds=thresholds,
                dynamic_thresholds=dynamic_thresholds,
                returns_for_volatility=rets_aligned
            )
        else:
            results["linear_reg"] = {"error": "Insufficient LinearReg predictions"}

        if oos_preds_lgbm.notna().sum() >= 10:
            results["lgbm"] = _compute_trading_simulation_pnl_dynamic(
                confidence_scores=oos_preds_lgbm,
                realized_returns=rets_aligned,
                thresholds=thresholds,
                dynamic_thresholds=dynamic_thresholds,
                returns_for_volatility=rets_aligned
            )
        else:
            results["lgbm"] = {"error": "Insufficient LGBM predictions"}

    return results



def _compute_group_lgbm_auc(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 3,
) -> Dict[str, Any]:
    """Train simple LGBM classifiers on combinations of specialist groups with parallel processing."""
    logger.info("🚀 Starting optimized group LGBM computation")
    
    try:
        import lightgbm as lgb  # type: ignore
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import TimeSeriesSplit
    except Exception:
        return {"error": "lightgbm or sklearn not available for group LGBM probes"}

    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx].copy()
    y = y.loc[common_idx].astype(float)

    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    if len(X) < max(cv_folds * 20, 120):
        return {"error": f"Insufficient samples for group LGBM probe: {len(X)}"}

    y_bin = (y > 0).astype(int)
    if y_bin.nunique() < 2:
        return {"error": "Binary target for group LGBM probe is single-class"}

    # Group features by specialist model
    group_features: Dict[str, List[str]] = {}
    for col in X.columns:
        group = _infer_model_group(col)
        if group == "other":
            continue
        group_features.setdefault(group, []).append(col)

    if not group_features:
        return {"error": "No specialist groups found for group LGBM probe"}

    logger.info(f"📊 Found {len(group_features)} specialist groups: {list(group_features.keys())}")

    tscv = TimeSeriesSplit(n_splits=cv_folds)

    def _cv_auc_for_features(feat_list: List[str]) -> Optional[Tuple[float, int]]:
        if not feat_list:
            return None
        X_sub = X[feat_list]
        oof = pd.Series(index=X_sub.index, dtype=float)
        for train_idx, val_idx in tscv.split(X_sub):
            X_tr = X_sub.iloc[train_idx]
            X_val = X_sub.iloc[val_idx]
            y_tr = y_bin.iloc[train_idx]
            if y_tr.nunique() < 2:
                continue
            try:
                clf = lgb.LGBMClassifier(
                    objective="binary",
                    n_estimators=50,  # Reduced for speed
                    learning_rate=0.1,  # Increased for faster convergence
                    num_leaves=8,
                    max_depth=3,
                    min_child_samples=100,
                    reg_alpha=5.0,
                    reg_lambda=5.0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                    n_jobs=1,  # Single job for memory efficiency
                )
                clf.fit(X_tr, y_tr)
                probs = clf.predict_proba(X_val)[:, 1]
                oof.iloc[val_idx] = probs
            except Exception:
                continue

        valid = oof.notna()
        try:
            if valid.sum() < max(20, 2 * cv_folds):
                return None
            
            # Determine task type
            uniq_val = np.unique(y_bin[valid])
            if len(uniq_val) <= 2 and set(uniq_val).issubset({0.0, 1.0}):
                if len(uniq_val) < 2:
                    return 0.5, int(valid.sum())
                try:
                    auc = roc_auc_score(y_bin[valid], oof[valid])
                except Exception:
                    return 0.5, int(valid.sum())
            else:
                # Regression
                corr = np.corrcoef(oof[valid], y_bin[valid])[0, 1] if np.var(oof[valid]) > 0 and np.var(y_bin[valid]) > 0 else 0
                auc = 0.5 + 0.5 * abs(corr)
        except Exception:
            return None
        return float(auc), int(valid.sum())

    results: Dict[str, Dict[str, Any]] = {}
    groups = sorted(group_features.keys())

    # Prepare all combinations for parallel processing
    combinations = []
    
    # Single-group probes
    for g in groups:
        feats = group_features[g]
        combinations.append((g, feats))
    
    # Pairwise and triple-group probes (limit to avoid explosion)
    import itertools
    max_combinations = 20  # Limit total combinations for performance
    
    for r in (2, 3):
        for combo in itertools.combinations(groups, r):
            if len(combinations) >= max_combinations:
                break
            combo_key = "|".join(combo)
            feat_list: List[str] = []
            for g in combo:
                feat_list.extend(group_features.get(g, []))
            if feat_list:
                combinations.append((combo_key, feat_list))
        if len(combinations) >= max_combinations:
            break

    logger.info(f"🔄 Processing {len(combinations)} group combinations with parallel processing")

    # Parallel processing of combinations
    def process_combination(combo_data):
        key, feat_list = combo_data
        res = _cv_auc_for_features(feat_list)
        if res is None:
            return None
        auc, n_valid = res
        return key, {
            "combination": key,
            "n_features": int(len(feat_list)),
            "auc": auc,
            "n_samples": n_valid,
        }

    # Use parallel processing with 2 workers
    processed_results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_combo = {
            executor.submit(process_combination, combo): combo 
            for combo in combinations
        }
        
        for future in as_completed(future_to_combo):
            try:
                result = future.result()
                if result is not None:
                    key, data = result
                    processed_results.append((key, data))
            except Exception as e:
                logger.debug(f"Failed to process combination: {e}")

    # Convert to results dictionary
    for key, data in processed_results:
        results[key] = data

    # Sort by AUC and limit results
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1].get('auc', 0), reverse=True))
    
    # Limit to top results to avoid overwhelming output
    if len(sorted_results) > 50:
        sorted_results = dict(list(sorted_results.items())[:50])

    # Garbage collection
    _garbage_collect_optimized()
    
    logger.info(f"✅ Group LGBM computation completed: {len(sorted_results)} combinations")
    return {"group_lgbm_auc": sorted_results}

def run_diagnostics(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    model: str,
    regime_timeframe: str,
    target_col: str,
    cv_folds: int,
    lookback_days: Optional[float] = None,
    enable_risk_hmm_specialist: bool = True,
    projection_mode: str = "canonical_scalars",
    enable_orthogonalization: bool = False,
    run_optimized_orthogonalization: bool = False,
    anchor_specialist: str = "xgb_macro",
    enable_cache: bool = False,
    feature_optimization: bool = False,
    orthogonal_hpo: bool = False,
    conservative_pruning: bool = False,
    run_lgbm_comparison: bool = False,
    orthogonal_comparison_report: bool = False,
    enable_moe: bool = False,
    enable_adversarial_orthogonalization: bool = False,
    adversarial_penalty: float = 0.1,
    enable_tv_var: bool = True,
    tv_var_window: int = 100,
    tv_var_regime_aware: bool = True,
    tv_var_backtest: bool = False,
    load_regime_label: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Run full specialist feature diagnostics and export reports."""
    # Determine default target column based on direction if not explicitly provided
    if not target_col:
        target_col = "binary_label"
    
    # Initialize optional result holders
    moe_results = None
    adversarial_results = None
    orthogonalization_results = None
    
    # 1) Load labels and realized_return for PnL simulation
    y, training_index, realized_return = _prepare_labels(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
        target_col=target_col,
        lookback_days=lookback_days,
    )

    # 2) Load specialist features aligned to the same index
    specialist_df = _load_specialist_features(
        symbol=symbol,
        exchange=exchange,
        base_timeframe=timeframe,
        regime_timeframe=regime_timeframe,
        direction=direction,
        model=model,
        training_index=training_index,
        enable_risk_hmm_specialist=enable_risk_hmm_specialist,
        load_regime_label=load_regime_label,
    )

    logger.info(f"📊 Loaded specialist_df with {len(specialist_df)} rows and columns: {list(specialist_df.columns)}")
    logger.info(f"📊 Training index size: {len(training_index)}")
    logger.info(f"📊 Target y size: {len(y)}")

    
    # Deduplicate columns in specialist_df to prevent LightGBM errors
    specialist_df = specialist_df.loc[:, ~specialist_df.columns.duplicated()]

    # Optionally collapse raw specialist outputs to canonical scalars. 
    # Must do this before metrics computation.
    if projection_mode == "canonical_scalars":
        X = _select_specialist_scalars(specialist_df)
    else:
        X = specialist_df

    # 3) Compute per-feature metrics and TV-VAR info (Needed for MoE)
    feature_metrics, tv_var_info = _compute_feature_metrics_with_tv_var(
        X=X, y=y, cv_folds=cv_folds, 
        enable_tv_var=enable_tv_var, 
        tv_var_window=tv_var_window, 
        tv_var_regime_aware=tv_var_regime_aware
    )
    # Garbage collection after feature metrics
    _garbage_collect_optimized()

    if not feature_metrics:
        raise ValueError("No feature metrics computed; check inputs and artifacts")

    # ---------------------------------------------------------
    # Specialist Category Orthogonalization (ENHANCED)
    # ---------------------------------------------------------
    orthogonalizer = None
    if enable_orthogonalization or run_optimized_orthogonalization or enable_moe or enable_adversarial_orthogonalization:
        from src.utils.ml_common.specialist_orthogonalizer import OptimizedSpecialistOrthogonalizer
        
        # Initialize orthogonalizer with optimization features
        orthogonalizer = OptimizedSpecialistOrthogonalizer(
            anchor_specialists=[anchor_specialist] if anchor_specialist else None,
            enable_cache=enable_cache,
            enable_feature_optimization=feature_optimization,
            enable_orthogonal_hpo=orthogonal_hpo,
            enable_conservative_pruning=conservative_pruning
        )
        
        # Validate specialist coverage
        coverage = orthogonalizer.validate_specialist_coverage(specialist_df)
        available_specialists = [s for s, has in coverage.items() if has]
        
        tprint_info(f"🎯 Found {len(available_specialists)} available specialists for orthogonalization/MoE")
        for specialist, has_features in coverage.items():
            status = "✅" if has_features else "❌"
            tprint_info(f"  {status} {specialist}")
        
        if len(available_specialists) < 2:
            tprint_warning("⚠️ Need at least 2 specialists for orthogonalization/MoE")
        else:
            # Get sample weights
            sample_weights = None
            if 'target_sample_weight' in specialist_df.columns:
                sample_weights = specialist_df['target_sample_weight']
            
            # Run orthogonalization if requested
            if enable_orthogonalization or run_optimized_orthogonalization:
                if run_optimized_orthogonalization:
                    tprint_info("🚀 Running optimized orthogonalization pipeline...")
                    
                    optimization_results = orthogonalizer.run_optimized_orthogonalization(
                        specialist_df=specialist_df,
                        target_series=y,
                        sample_weights=sample_weights,
                        run_hpo=orthogonal_hpo,
                        run_pruning=conservative_pruning,
                        optimize_features=feature_optimization
                    )
                    
                    # Store results for reporting
                    orthogonal_targets = optimization_results['orthogonal_targets']
                    auc_weights = optimization_results.get('hpo_results', {}).get('ensemble_config', {}).get('weights', {})
                    
                    # Log optimization summary
                    perf_summary = optimization_results.get('performance_summary', {})
                    tprint_success(f"✅ Optimized orthogonalization completed:")
                    tprint_info(f"  Specialists: {len(optimization_results.get('pruned_specialists', []))}")
                    tprint_info(f"  Mean AUC: {perf_summary.get('mean_auc', 0.5):.4f}")
                    tprint_info(f"  Total time: {optimization_results.get('optimization_time', 0):.1f}s")
                    
                else:
                    # Run standard orthogonalization
                    orthogonal_targets, auc_weights = orthogonalizer.generate_auc_weighted_orthogonal_targets(
                        specialist_df=specialist_df,
                            target_series=y,
                            sample_weights=sample_weights
                        )
                    
                    # Run LGBM comparison if requested
                    if run_lgbm_comparison:
                        tprint_info("🔬 Running LGBM comparison analysis...")
                        
                        if run_optimized_orthogonalization and 'optimization_results' in locals():
                            # Use optimized results
                            comparison_results = orthogonalizer.run_2_core_lgbm_comparison(
                                specialist_df, y, sample_weights
                            )
                        else:
                            # Run standard comparison
                            comparison_results = orthogonalizer.run_2_core_lgbm_comparison(
                                specialist_df, y, sample_weights
                            )
                        
                        # Generate comprehensive comparison report
                        if orthogonal_comparison_report:
                            from scripts.generate_orthogonal_comparison_report import generate_orthogonal_comparison_report
                            # Create a simple args-like object for the report generator
                            class ArgsProxy:
                                def __init__(self):
                                    self.symbol = symbol
                                    self.exchange = exchange
                                    self.timeframe = timeframe
                                    self.direction = direction
                                    self.model = model
                                    self.regime_timeframe = regime_timeframe
                                    self.target_col = target_col
                                    self.anchor_specialist = anchor_specialist
                            generate_orthogonal_comparison_report(comparison_results, ArgsProxy())
                        
                        # Store comparison results
                        orthogonalization_results = {
                            'comparison_results': comparison_results,
                            'orthogonal_targets': orthogonal_targets,
                            'auc_weights': auc_weights,
                            'specialist_coverage': coverage
                        }
                        
                        tprint_success("✅ Orthogonalization and LGBM comparison completed")

            # ---------------------------------------------------------
            # Adversarial Orthogonalization (NEW)
            # ---------------------------------------------------------
            if enable_adversarial_orthogonalization:
                tprint_info("🛡️ Running adversarial orthogonalization...")
                anchor_features = orthogonalizer.extract_specialist_features(specialist_df, anchor_specialist)
                if not anchor_features.empty:
                    adv_targets = orthogonalizer.generate_adversarial_orthogonal_targets(
                        specialist_df=specialist_df,
                        target_series=y,
                        anchor_features=anchor_features,
                        penalty_lambda=adversarial_penalty
                    )
                    tprint_success(f"✅ Generated {len(adv_targets.columns)} adversarial orthogonal targets")
                    
                    # Store adversarial results for payload
                    adversarial_results = {
                        "n_targets": len(adv_targets.columns),
                        "targets": list(adv_targets.columns),
                        "penalty_lambda": adversarial_penalty,
                        "anchor": anchor_specialist
                    }
                    
                    # Optionally run a probe on one of the adversarial targets to show value
                    # For diagnostics, we'll just store the targets themselves for now
                else:
                    tprint_warning(f"⚠️ Anchor specialist {anchor_specialist} has no features; skipping adversarial orthogonalization")
                    adversarial_results = {"error": "Anchor specialist missing features"}
            else:
                adversarial_results = None

            # ---------------------------------------------------------
            # Regime-Gated MoE (NEW)
            # ---------------------------------------------------------
            if enable_moe and 'tv_var_regime' in tv_var_info:
                from src.utils.ml_common.specialist_orthogonalizer import RegimeGatedMoE
                tprint_info("🧠 Fitting Regime-Gated Mixture of Experts...")
                moe = RegimeGatedMoE(n_regimes=8)
                moe.fit(specialist_df, y, tv_var_info['tv_var_regime'])
                
                # Report MoE weights for latest regime
                latest_regime = tv_var_info['tv_var_regime'].iloc[-1]
                latest_weights = moe.get_weights(latest_regime)
                tprint_success(f"✅ MoE fitted. Latest regime ({latest_regime}) weights: {latest_weights}")

                # Evaluate MoE as a probe signal
                try:
                    # Store MoE for payload
                    moe_results = {
                        "latest_regime": int(latest_regime),
                        "latest_weights": latest_weights,
                        "regime_distribution": tv_var_info['tv_var_regime'].value_counts().to_dict()
                    }
                except Exception as moe_exc:
                    logger.warning(f"MoE evaluation failed: {moe_exc}")
                    moe_results = {"error": str(moe_exc)}
            else:
                moe_results = None

    # TV-VAR Enhanced Analysis and Backtesting
    if enable_tv_var and tv_var_backtest and len(X) > 500:
        tprint_info("�� Running TV-VAR backtesting validation...")
        
        try:
            # Generate TV-VAR features for backtesting
            tv_var_features = _generate_tv_var_features_from_specialists(X)
            
            if len(tv_var_features.columns) >= 8:
                # Run TV-VAR backtesting
                backtest_results = backtest_tv_var_enhanced(
                    tv_var_features, X, y, symbol, 2024, 2024
                )
                
                # Store TV-VAR results
                tv_var_results = {
                    "backtest_results": backtest_results,
                    "stability_score": backtest_results.validation_summary.get("success_rate", 0),
                    "improvement": backtest_results.validation_summary.get("avg_auc_improvement", 0),
                    "production_ready": backtest_results.validation_summary.get("production_ready", False)
                }
                
                tprint_success(
                    f"✅ TV-VAR backtesting completed - Success Rate: {tv_var_results['stability_score']:.1%}, "
                    f"Improvement: {tv_var_results['improvement']:.2f}%"
                )
            else:
                tprint_warning("⚠️ Insufficient TV-VAR features for backtesting")
                tv_var_results = {"error": "Insufficient features"}
        
        except Exception as e:
            tprint_error(f"❌ TV-VAR backtesting failed: {e}")
            tv_var_results = {"error": str(e)}
    else:
        tv_var_results = {"disabled": True}

# Optionally collapse raw specialist outputs to canonical scalars. When
    # projection_mode == "raw", use the specialist block exactly as training
    # sees it (same get_specialist_models_outputs output).
    if projection_mode == "canonical_scalars":
        X = _select_specialist_scalars(specialist_df)
    else:
        X = specialist_df

    # 3) Compute per-feature metrics
    # feature_metrics, tv_var_info = _compute_feature_metrics_with_tv_var(X=X, y=y, cv_folds=cv_folds, enable_tv_var=enable_tv_var, tv_var_window=tv_var_window, tv_var_regime_aware=tv_var_regime_aware)
    # Garbage collection after feature metrics
    # _garbage_collect_optimized()

    # if not feature_metrics:
    #    raise ValueError("No feature metrics computed; check inputs and artifacts")

    model_reliability = _compute_model_reliability(feature_metrics=feature_metrics)
    model_coverage = _compute_model_coverage(X=X, y=y)
    model_relationships = _compute_model_pairwise_relationships(X=X, feature_metrics=feature_metrics)
    group_lgbm_auc = _compute_group_lgbm_auc(X=X, y=y, cv_folds=cv_folds)

    # 3b) Compute target date range
    y_start = y.index.min()
    y_end = y.index.max()
    # Garbage collection after model reliability
    _garbage_collect_optimized()
    y_duration = (y_end - y_start).days if len(y) > 0 else 0

    # 4) Probe models (LogReg / LGBM), leakage, stability, interactions
    probe_models = _compute_probe_models(X=X, y=y, n_splits=cv_folds)

    per_regime_probe_models: Dict[str, Any] = {}
    try:
        tv_regime = tv_var_info.get("tv_var_regime") if isinstance(tv_var_info, dict) else None
        if isinstance(tv_regime, pd.Series):
            per_regime_probe_models = _compute_per_regime_probe_models(
                X=X,
                y=y,
                regimes=tv_regime,
                n_splits=cv_folds,
            )
    except Exception as exc:
        per_regime_probe_models = {"error": str(exc)}
    
    # 4b) Compute probe model trading PnL simulation
    probe_pnl = _compute_probe_model_pnl(
        X=X,
        y=y,
        realized_returns=realized_return,
        n_splits=cv_folds,
        thresholds=None, dynamic_thresholds=True,
    )

    # Garbage collection after probe models
    _garbage_collect_optimized()
    # Reuse a FinalFeatureSelectionComponent instance for leakage detection
    fs_config = FinalFeatureSelectionConfig()
    fs_component = FinalFeatureSelectionComponent(config=fs_config)
    
    # Use detect_redundant_features as a proxy for leakage diagnostics if detect_potential_leakage is missing
    if hasattr(fs_component, 'detect_potential_leakage'):
        leakage_diagnostics = _detect_feature_leakage(X=X, y=y, component=fs_component)
    elif hasattr(fs_component, 'detect_redundant_features'):
        # Fallback to redundancy check
        logger.info("Using detect_redundant_features as proxy for leakage diagnostics")
        redundancy = fs_component.detect_redundant_features(X, list(X.columns))
        leakage_diagnostics = {
            "suspicious_features": [(f, 0.95) for f in redundancy.get("high_correlation_pairs", [])],
            "perfect_features": [],
            "redundancy_info": redundancy
        }
    else:
        leakage_diagnostics = {"error": "No leakage detection method found in FinalFeatureSelectionComponent"}

    global_stability = _compute_global_stability(X=X, y=y, n_splits=cv_folds)
    interactions = _compute_tree_shap_interactions(
        X=X,
        y=y,
        feature_metrics=feature_metrics,
    )

    # Aggregate summary stats
    mi_values = np.array([m["mi_mean_cv"] for k, m in feature_metrics.items() if k != "_tv_var_system"], dtype=float)
    r2_values = np.array([m["r2"] for k, m in feature_metrics.items() if k != "_tv_var_system"], dtype=float)

    mi_values = mi_values[np.isfinite(mi_values)]
    r2_values = r2_values[np.isfinite(r2_values)]

    summary: Dict[str, Any] = {
        "n_features": len(feature_metrics),
        "mi_mean_avg": float(mi_values.mean()) if mi_values.size else 0.0,
        "mi_mean_median": float(np.median(mi_values)) if mi_values.size else 0.0,
        "r2_mean": float(r2_values.mean()) if r2_values.size else 0.0,
        "r2_median": float(np.median(r2_values)) if r2_values.size else 0.0,
        "n_high_mi": int(np.sum(mi_values > 0.1)) if mi_values.size else 0,
        "n_high_r2": int(np.sum(r2_values > 0.05)) if r2_values.size else 0,
    }

    # Build Markdown summary (top features by MI and R^2)
    sorted_by_mi = sorted(
        [item for item in feature_metrics.items() if item[0] != "_tv_var_system"], 
        key=lambda kv: kv[1].get("mi_mean_cv", 0.0), 
        reverse=True
    )
    sorted_by_r2 = sorted(
        [item for item in feature_metrics.items() if item[0] != "_tv_var_system"], 
        key=lambda kv: kv[1].get("r2", 0.0), 
        reverse=True
    )

    top_k = 20
    md_lines: list[str] = [
        "# Specialist Feature Diagnostics",
        "",
        f"**Symbol**: {symbol}",
        f"**Exchange**: {exchange}",
        f"**Timeframe**: {timeframe}",
        f"**Direction**: {direction}",
        f"**Model**: {model}",
        f"**Regime timeframe**: {regime_timeframe}",
        f"**Target column**: {target_col}",
        "",
        "## Data Range Analysis",
        f"- Target start date: {y_start}",
        f"- Target end date: {y_end}",
        f"- Target duration: {y_duration} days",
        f"- Target samples: {len(y)}",
        "",
        "## Overview",
        f"- Number of specialist features: {summary['n_features']}",
        f"- Mean MI (CV-averaged): {summary['mi_mean_avg']:.4f}",
        f"- Median MI (CV-averaged): {summary['mi_mean_median']:.4f}",
        f"- Mean R^2 (univariate): {summary['r2_mean']:.4f}",
        f"- Median R^2 (univariate): {summary['r2_median']:.4f}",
        f"- High-MI features (MI>0.10): {summary['n_high_mi']}",
        f"- High-R^2 features (R^2>0.05): {summary['n_high_r2']}",
        "",
    ]
    
    if tv_var_info.get("enabled"):
        md_lines.extend([
            "### TV-VAR System Metrics",
            f"- Stability Score: {tv_var_info['stability_score']:.3f}",
            f"- TV-VAR Samples: {tv_var_info['n_samples']}",
            f"- Market Regimes Detected: {tv_var_info['regime_count']}",
            "",
        ])

    md_lines.append("### Probe model summary (LogReg / LGBM)")


    # Add brief probe model summary if available
    task_type = probe_models.get("task_type", "unknown")

    if task_type == "binary_classification":
        logreg_summary = probe_models.get("logreg", {}) if isinstance(probe_models, dict) else {}
        lgbm_summary = probe_models.get("lgbm", {}) if isinstance(probe_models, dict) else {}

        def _fmt_probe_clf(model_name: str, summary_dict: Dict[str, Any]) -> str:
            if not summary_dict or "auc_mean" not in summary_dict:
                return f"- {model_name}: not available"
            auc_mean = summary_dict.get("auc_mean", float("nan"))
            auc_std = summary_dict.get("auc_std", float("nan"))
            acc_mean = summary_dict.get("accuracy_mean", float("nan"))
            return (
                f"- {model_name}: AUC={auc_mean:.3f}±{auc_std:.3f}, "
                f"Accuracy={acc_mean:.3f}"
            )

        md_lines.extend(
            [
                _fmt_probe_clf("Logistic Regression", logreg_summary),
                _fmt_probe_clf("LightGBM", lgbm_summary),
            ]
        )
    elif task_type == "regression":
        linear_summary = probe_models.get("linear_reg", {}) if isinstance(probe_models, dict) else {}
        lgbm_summary = probe_models.get("lgbm", {}) if isinstance(probe_models, dict) else {}

        def _fmt_probe_reg(model_name: str, summary_dict: Dict[str, Any]) -> str:
            if not summary_dict or "rmse_mean" not in summary_dict:
                return f"- {model_name}: not available"
            rmse_mean = summary_dict.get("rmse_mean", float("nan"))
            rmse_std = summary_dict.get("rmse_std", float("nan"))
            r2_mean = summary_dict.get("r2_mean", float("nan"))
            return (
                f"- {model_name}: RMSE={rmse_mean:.4f}±{rmse_std:.4f}, "
                f"R2={r2_mean:.4f}"
            )

        md_lines.extend(
            [
                _fmt_probe_reg("Linear Regression (Ridge)", linear_summary),
                _fmt_probe_reg("LightGBM Regressor", lgbm_summary),
            ]
        )

    # Per-regime probe diagnostics (only when regime labels exist)
    if isinstance(per_regime_probe_models, dict) and per_regime_probe_models and "per_regime" in per_regime_probe_models:
        md_lines.extend(
            [
                "",
                "### Per-regime probe models (TimeSeriesSplit within each regime)",
                "",
                "| Regime | n_samples | pos_frac | LogReg AUC | LGBM AUC |",
                "|--------|----------:|---------:|----------:|---------:|",
            ]
        )
        per_reg = per_regime_probe_models.get("per_regime", {})
        for regime_name, stats in sorted(per_reg.items(), key=lambda kv: kv[0]):
            if not isinstance(stats, dict):
                continue
            n_samples = stats.get("n_samples")
            pos_frac = stats.get("pos_frac")
            lr_auc = stats.get("logreg_auc_mean")
            lgbm_auc = stats.get("lgbm_auc_mean")

            if "error" in stats:
                md_lines.append(
                    f"| {regime_name} | {int(n_samples) if n_samples is not None else 0} |  |  |  |"
                )
                continue

            md_lines.append(
                "| "
                + f"{regime_name} | "
                + f"{int(n_samples) if n_samples is not None else 0} | "
                + f"{float(pos_frac) if pos_frac is not None else float('nan'):.3f} | "
                + f"{float(lr_auc) if lr_auc is not None else float('nan'):.3f} | "
                + f"{float(lgbm_auc) if lgbm_auc is not None else float('nan'):.3f} |"
            )
    elif isinstance(per_regime_probe_models, dict) and per_regime_probe_models.get("error"):
        md_lines.extend(
            [
                "",
                "### Per-regime probe models",
                f"- Unavailable: {per_regime_probe_models.get('error')}",
            ]
        )

    # Add Trading PnL Simulation section
    md_lines.extend(
        [
            "",
            "### Trading PnL Simulation (TP=2%, SL=0.7%, Fees=0.3% round-trip)",
            "",
        ]
    )

    if "error" in probe_pnl:
        md_lines.append(f"- Trading simulation unavailable: {probe_pnl['error']}")
    else:
        # Determine models to display based on task type
        models_to_display = []
        if task_type == "regression":
            models_to_display = [("Linear Regression", "linear_reg"), ("LightGBM Regressor", "lgbm")]
        else:
            models_to_display = [("Logistic Regression", "logreg"), ("LightGBM", "lgbm")]

        # Display results for each model
        for model_name, model_key in models_to_display:
            model_pnl = probe_pnl.get(model_key, {})
            if "error" in model_pnl:
                md_lines.append(f"**{model_name}**: {model_pnl['error']}")
                md_lines.append("")
                continue
            
            per_threshold = model_pnl.get("per_threshold", {})
            if not per_threshold:
                md_lines.append(f"**{model_name}**: No threshold data available")
                md_lines.append("")
                continue
            
            md_lines.append(f"**{model_name}** (data range: {model_pnl.get('date_range_days', 0):.0f} days, {model_pnl.get('total_samples', 0)} samples)")
            md_lines.append("")
            md_lines.append("| Threshold | Trades | Trades/Day | Win Rate | PnL/Trade | PnL/Day | PnL/Month | Sharpe |")
            md_lines.append("|----------:|-------:|-----------:|---------:|----------:|--------:|----------:|-------:|")
            
            # Use dynamic threshold keys since regression thresholds differ from classification
            sorted_thresholds = sorted(per_threshold.keys())

            for threshold_key in sorted_thresholds:
                t = per_threshold[threshold_key]
                md_lines.append(
                    f"| {threshold_key} | "
                    f"{t['n_trades']:,} | "
                    f"{t['trades_per_day']:.2f} | "
                    f"{t['win_rate']:.1%} | "
                    f"{t['avg_pnl_per_trade_pct']:.4f}% | "
                    f"{t['avg_pnl_per_day_pct']:.4f}% | "
                    f"{t['avg_pnl_per_month_pct']:.2f}% | "
                    f"{t['sharpe_estimate']:.2f} |"
                )
            md_lines.append("")

    md_lines.extend(
        [
            "",
            "### Per-specialist model reliability vs target (MI / R^2)",
        ]
    )

    per_model = model_reliability.get("per_model", {}) if isinstance(model_reliability, dict) else {}
    if per_model:
        for group_name, stats in per_model.items():
            md_lines.append(
                "- "
                + f"{group_name}: "
                + f"n_features={int(stats.get('n_features', 0))}, "
                + f"MI_mean={float(stats.get('mi_mean_avg', 0.0)):.4f}, "
                + f"R^2_mean={float(stats.get('r2_mean', 0.0)):.4f}, "
                + f"high_MI={int(stats.get('n_high_mi', 0))}, "
                + f"high_R^2={int(stats.get('n_high_r2', 0))}"
            )
    else:
        md_lines.append("- Per-model reliability metrics unavailable")

    md_lines.extend(
        [
            "",
            "### Per-specialist data coverage",
        ]
    )

    coverage_info = model_coverage if isinstance(model_coverage, dict) else {}
    if coverage_info:
        total_target = int(len(y))
        if total_target > 0:
            md_lines.append(f"*(Target samples: {total_target})*")

        for group_name in sorted(coverage_info.keys()):
            info = coverage_info.get(group_name, {})
            n_samples = int(info.get("n_samples", 0) or 0)
            frac = float(info.get("fraction_of_target", 0.0) or 0.0)
            start = info.get("start")
            end = info.get("end")

            line = f"- **{group_name}**: n={n_samples} ({frac:.1%} coverage)"
            if start is not None and end is not None:
                line += f", range: {start} → {end}"

                # Check for significant mismatch
                if start > y_start + pd.Timedelta(days=7):
                    line += " ⚠️ Starts late"
                if end < y_end - pd.Timedelta(days=7):
                    line += " ⚠️ Ends early"
            else:
                line += ", range unavailable"

            if frac < 0.5:
                line += " ⚠️ Low coverage (<50%)"

            md_lines.append(line)
    else:
        md_lines.append("- Per-specialist coverage unavailable")

    md_lines.extend(
        [
            "",
            "### Pairwise relationships between specialist models (MI / R^2)",
        ]
    )

    pairwise_info = model_relationships if isinstance(model_relationships, dict) else {}
    pair_list = pairwise_info.get("pairs", []) if isinstance(pairwise_info, dict) else []

    if not pair_list:
        error_msg = pairwise_info.get("error") if isinstance(pairwise_info, dict) else None
        if error_msg:
            md_lines.append(f"- Pairwise model analysis unavailable: {error_msg}")
        else:
            md_lines.append("- Pairwise model analysis unavailable")
    else:
        md_lines.append("")
        md_lines.append("| Model i | Model j | Rep feature i | Rep feature j | MI_proxy | R^2 |")
        md_lines.append("|---------|---------|---------------|---------------|---------:|----:|")
        for entry in pair_list:
            md_lines.append(
                "| "
                + f"{entry.get('model_i', '')} | "
                + f"{entry.get('model_j', '')} | "
                + f"{entry.get('rep_feature_i', '')} | "
                + f"{entry.get('rep_feature_j', '')} | "
                + f"{float(entry.get('mi_proxy', 0.0)):.4f} | "
                + f"{float(entry.get('r2', 0.0)):.4f} |"
            )

    md_lines.extend(
        [
            "",
            "### LGBM interaction probes (specialist groups)",
        ]
    )

    if isinstance(group_lgbm_auc, dict) and "group_lgbm_auc" in group_lgbm_auc:
        group_auc_entries = group_lgbm_auc.get("group_lgbm_auc", {})
        if group_auc_entries:
            md_lines.append("")
            md_lines.append("| Groups | n_features | AUC | n_oof_samples |")
            md_lines.append("|--------|-----------:|----:|--------------:|")
            for key, entry in sorted(
                group_auc_entries.items(),
                key=lambda kv: kv[1].get("auc", 0.0),
                reverse=True,
            )[:20]:
                md_lines.append(
                    f"| {entry.get('combination', key)} | "
                    f"{int(entry.get('n_features', 0))} | "
                    f"{entry.get('auc', 0.0):.3f} | "
                    f"{int(entry.get('n_samples', 0))} |"
                )
        else:
            err_msg = group_lgbm_auc.get("error")
            if err_msg:
                md_lines.append(f"- Group LGBM probes unavailable: {err_msg}")
            else:
                md_lines.append("- Group LGBM probes unavailable")
    else:
        md_lines.append("- Group LGBM probes unavailable")

    md_lines.extend(
        [
            "",
            "### Global stability (TimeSeriesSplit AUC)",
        ]
    )

    if "error" in global_stability:
        md_lines.append(f"- Stability analysis unavailable: {global_stability['error']}")
    else:
        md_lines.append(
            f"- Mean AUC={global_stability.get('mean_auc', float('nan')):.3f}, "
            f"std={global_stability.get('std_auc', float('nan')):.3f}, "
            f"stability score={global_stability.get('stability_score', float('nan')):.3f}"
        )

    md_lines.extend(
        [
            "",
            "## Top Features by MI Proxy (CV-averaged)",
            "",
            "| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |",
            "|---------|--------:|--------:|------:|-----:|----:|",
        ]
    )

    for name, met in sorted_by_mi[:top_k]:
        md_lines.append(
            f"| {name} | {met['mi_proxy_full']:.4f} | {met['mi_mean_cv']:.4f} | "
            f"{met['mi_cv']:.3f} | {met['pearson_corr']:.3f} | {met['r2']:.4f} |"
        )

    md_lines.extend(
        [
            "",
            "## Top Features by R^2 (Univariate)",
            "",
            "| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |",
            "|---------|--------:|--------:|------:|-----:|----:|",
        ]
    )

    for name, met in sorted_by_r2[:top_k]:
        md_lines.append(
            f"| {name} | {met['mi_proxy_full']:.4f} | {met['mi_mean_cv']:.4f} | "
            f"{met['mi_cv']:.3f} | {met['pearson_corr']:.3f} | {met['r2']:.4f} |"
        )

    # Append constant/near-constant feature check
    md_lines.extend(
        [
            "",
            "## Constant / Near-Constant Feature Check",
        ]
    )
    constant_feats = []
    for col in X.columns:
        if np.std(X[col]) < 1e-9:
            val = float(X[col].mean()) if len(X) > 0 else 0.0
            constant_feats.append(f"{col} (val={val:.4f})")

    if constant_feats:
        md_lines.append(f"⚠️ Found {len(constant_feats)} constant features:")
        for f in constant_feats[:10]:
            md_lines.append(f"- {f}")
        if len(constant_feats) > 10:
            md_lines.append(f"- ... and {len(constant_feats) - 10} more")
    else:
        md_lines.append("- No constant features found (std < 1e-9).")

    # Append leakage and interaction summaries
    md_lines.extend(
        [
            "",
            "## Leakage diagnostics",
        ]
    )

    if "error" in leakage_diagnostics:
        md_lines.append(f"- Leakage detection unavailable: {leakage_diagnostics['error']}")
    else:
        susp = leakage_diagnostics.get("suspicious_features", [])
        perf = leakage_diagnostics.get("perfect_features", [])
        md_lines.append(f"- Suspicious features (|corr|>=0.95): {len(susp)}")
        md_lines.append(f"- Perfect-correlation features (|corr|>=0.99): {len(perf)}")
        if susp:
            md_lines.append("- Examples (suspicious): " + ", ".join(f"{f[0]}({f[1]:.3f})" for f in susp[:5]))
        if perf:
            md_lines.append("- Examples (perfect): " + ", ".join(f"{f[0]}({f[1]:.3f})" for f in perf[:5]))

    md_lines.extend(
        [
            "",
            "## Notable pairwise interactions (TreeSHAP)",
        ]
    )

    if "error" in interactions:
        md_lines.append(f"- Interaction analysis unavailable: {interactions['error']}")
    else:
        top_pairs = interactions.get("top_pairs", [])
        md_lines.append(
            f"- Computed on {interactions.get('n_features', 0)} features, "
            f"sample_size={interactions.get('sample_size', 0)}"
        )
        md_lines.append("")
        md_lines.append("| Feature i | Feature j | Interaction strength |")
        md_lines.append("|----------|----------|---------------------:|")
        for p in top_pairs[:20]:
            md_lines.append(
                f"| {p['feature_i']} | {p['feature_j']} | {p['interaction_strength']:.4e} |"
            )

    # ---------------------------------------------------------
    # Regime-Gated MoE Report (NEW)
    # ---------------------------------------------------------
    if moe_results:
        md_lines.extend([
            "",
            "## Regime-Gated Mixture of Experts (MoE)",
            f"- **Latest Regime**: {moe_results.get('latest_regime')}",
            "",
            "### Latest Regime Weights",
            "| Specialist | Weight |",
            "|------------|--------:|",
        ])
        weights = moe_results.get('latest_weights', {})
        for spec, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            md_lines.append(f"| {spec} | {weight:.4f} |")
        
        md_lines.extend([
            "",
            "### Regime Distribution (Samples)",
            "| Regime | Count |",
            "|--------|-------:|",
        ])
        dist = moe_results.get('regime_distribution', {})
        for r, count in sorted(dist.items()):
            md_lines.append(f"| {r} | {count:,} |")

    # ---------------------------------------------------------
    # Adversarial Orthogonalization Report (NEW)
    # ---------------------------------------------------------
    if adversarial_results:
        md_lines.extend([
            "",
            "## Adversarial Orthogonalization",
            f"- **Anchor**: {adversarial_results.get('anchor')}",
            f"- **Penalty (Lambda)**: {adversarial_results.get('penalty_lambda')}",
            f"- **Generated Targets**: {adversarial_results.get('n_targets')}",
            "",
            "### Targets List",
        ])
        for target in adversarial_results.get('targets', []):
            md_lines.append(f"- {target}")

    payload: Dict[str, Any] = {
        "summary": summary,
        "feature_metrics": feature_metrics,
        "tv_var_info": tv_var_info,
        "cv_folds": int(cv_folds),
        "probe_models": probe_models,
        "per_regime_probe_models": per_regime_probe_models,
        "probe_pnl_simulation": probe_pnl,
        "leakage_diagnostics": leakage_diagnostics,
        "stability_metrics": global_stability,
        "interactions": interactions,
        "model_reliability": model_reliability,
        "model_pairwise": model_relationships,
        "group_lgbm_auc": group_lgbm_auc,
        "model_coverage": model_coverage,
        "tv_var_results": tv_var_results,
        "moe_results": moe_results,
        "adversarial_results": adversarial_results,
    }

    return _export_report(
        prefix="specialist_feature_diagnostics",
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
        payload=payload,
        markdown_lines=md_lines,
    )


def _check_range_specific_config() -> bool:
    """Check if 1.5-3% range optimization is enabled."""
    try:
        import yaml
        config_path = "config/labeling/layer2_coverage_relax_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("target_range_optimization", {}).get("enabled", False)
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diagnostics for specialist model outputs vs meta-label targets",
    )
    ap.add_argument(
        "--independent-mode",
        action="store_true",
        help="Run specialist training before diagnostics"
    )
    ap.add_argument(
        "--auto-train",
        action="store_true",
        help="Automatically run training for specialists"
    )
    ap.add_argument(
        "--train-per-regime",
        action="store_true",
        help="Train specialists separately for each GMM router regime (Quiet/Trending/Chaos)"
    )
    ap.add_argument(
        "--regime-labels",
        type=str,
        help="Comma-separated regime labels to train (default: Quiet,Trending,Chaos)"
    )
    ap.add_argument(
        "--load-regime-label",
        type=str,
        help="Load regime-suffixed specialist artifacts for a single regime label"
    )
    ap.add_argument(
        "--enable-moe",
        action="store_true",
        help="Enable Regime-Gated Mixture of Experts selection"
    )
    ap.add_argument(
        "--enable-adversarial-orthogonalization",
        action="store_true",
        help="Enable Adversarial Orthogonalization with macro penalty"
    )
    ap.add_argument(
        "--adversarial-penalty",
        type=float,
        default=0.1,
        help="Penalty lambda for adversarial orthogonalization (default: 0.1)"
    )
    ap.add_argument("--symbol", type=str, default="ETHUSDT")
    ap.add_argument("--exchange", type=str, default="binance")
    ap.add_argument("--timeframe", type=str, default="15m")
    ap.add_argument("--direction", type=str, default="long", choices=["long", "short", "both"])
    ap.add_argument("--model", type=str, default="analyst")
    ap.add_argument("--regime-timeframe", type=str, default="1h")
    # Default target column is now direction-aware: binary_label_long for longs, binary_label_short for shorts
    # Falls back to unified binary_label if directional labels not available
    
        # Orthogonalization arguments
    ap.add_argument(
        "--enable-orthogonalization",
        action="store_true",
        help="Enable specialist category orthogonalization using XGB Macro Trend as anchor"
    )
    ap.add_argument(
        "--anchor-specialist",
        type=str,
        default="xgb_macro",
        help="Anchor specialist for orthogonalization (default: xgb_macro)"
    )
    ap.add_argument(
        "--run-lgbm-comparison",
        action="store_true",
        help="Run LGBM comparison between baseline and orthogonal features"
    )
    ap.add_argument(
        "--weighting-metric",
        type=str,
        default="auc",
        choices=["auc", "mi", "hybrid"],
        help="Weighting metric for orthogonal targets (default: auc)"
    )
    ap.add_argument(
        "--orthogonal-comparison-report",
        action="store_true",
        help="Generate comprehensive orthogonal comparison report"
    )
    
    # Target denoising arguments
    ap.add_argument(
        "--target-denoising",
        action="store_true",
        help="Enable target denoising before orthogonalization"
    )
    ap.add_argument(
        "--denoising-method",
        type=str,
        default="kalman",
        choices=["kalman", "hampel", "savgol", "volume", "ensemble"],
        help="Target denoising method (default: kalman)"
    )
    ap.add_argument(
        "--denoising-confidence-threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for denoising (default: 0.7)"
    )
    ap.add_argument(
        "--volume-column",
        type=str,
        default="volume",
        help="Volume column name for volume-weighted denoising (default: volume)"
    )
    
    # New optimization arguments    # New optimization arguments
    ap.add_argument(
        "--orthogonal-hpo",
        action="store_true",
        help="Enable orthogonalization-aware HPO with narrow search spaces"
    )
    ap.add_argument(
        "--conservative-pruning",
        action="store_true",
        help="Enable conservative ensemble pruning (8-12 specialists)"
    )
    ap.add_argument(
        "--feature-optimization",
        action="store_true",
        help="Enable feature deduplication and optimization"
    )
    ap.add_argument(
        "--enable-cache",
        action="store_true",
        help="Enable specialist model caching for faster execution"
    )
    ap.add_argument(
        "--narrow-hpo",
        action="store_true",
        help="Use narrow HPO search spaces for faster convergence (30 trials max)"
    )
    ap.add_argument(
        "--min-ensemble-size",
        type=int,
        default=8,
        help="Minimum ensemble size after pruning (default: 8)"
    )
    ap.add_argument(
        "--max-ensemble-size",
        type=int,
        default=12,
        help="Maximum ensemble size after pruning (default: 12)"
    )
    ap.add_argument(
        "--run-optimized-orthogonalization",
        action="store_true",
        help="Run complete optimized orthogonalization pipeline"
    )
    ap.add_argument(
        "--enable-calibration",
        action="store_true",
        default=True,
        help="Enable probability calibration for probe models (default: True)"
    )
    
    # Additional missing arguments
    ap.add_argument(
        "--compare-targets",
        action="store_true",
        help="Compare multiple targets (classifiers vs regressors)"
    )
    ap.add_argument(
        "--target-col",
        type=str,
        help="Explicit target column (overrides direction-aware default)"
    )
    ap.add_argument(
        "--lookback-days",
        type=float,
        help="Restrict analysis to last N calendar days"
    )
    ap.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    ap.add_argument(
        "--enable-hmm-risk-specialist",
        action="store_true",
        help="Enable HMM risk specialist"
    )
    ap.add_argument(
        "--projection-mode",
        type=str,
        default="canonical_scalars",
        help="Projection mode (default: canonical_scalars)"
    )

    # TV-VAR Enhanced Analysis arguments (default enabled)
    ap.add_argument(
        "--enable-tv-var",
        action="store_true",
        default=True,
        help="Enable TV-VAR enhanced analysis (default: True)"
    )
    ap.add_argument(
        "--tv-var-window",
        type=int,
        default=100,
        help="TV-VAR rolling window size (default: 100)"
    )
    ap.add_argument(
        "--tv-var-regime-aware",
        action="store_true",
        default=True,
        help="Use regime-aware TV-VAR analysis (default: True)"
    )
    ap.add_argument(
        "--tv-var-backtest",
        action="store_true",
        help="Run TV-VAR backtesting validation"
    )

    args = ap.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # Optional: Compare targets if requested
    if hasattr(args, 'compare_targets') and args.compare_targets:
        targets_to_run = [args.target_col]
        # Additional comparison logic would go here
        pass

    # Determine default target column based on direction if not explicitly provided
    if not hasattr(args, 'target_col') or args.target_col is None:
        # Priority: binary_label (often has better coverage/balance in AFML)
        args.target_col = "binary_label"
        print(f"Using unified default target: {args.target_col}")

    # Normalize legacy/unified binary_label to direction-aware labels when possible
    if args.target_col == "binary_label":
        if args.direction == "long":
            args.target_col = "binary_label_long"
            print("Normalizing target_col 'binary_label' to 'binary_label_long' for long direction.")
        elif args.direction == "short":
            args.target_col = "binary_label_short"
            print("Normalizing target_col 'binary_label' to 'binary_label_short' for short direction.")

    targets_to_run = [args.target_col]
    if args.compare_targets:
        # Add regression targets for comparison
        if args.direction == "long":
            if "binary_label_long" not in targets_to_run:
                targets_to_run.append("binary_label_long")
            # Also compare with unified binary_label for reference
            if "binary_label" not in targets_to_run:
                targets_to_run.append("binary_label")
        elif args.direction == "short":
            if "binary_label_short" not in targets_to_run:
                targets_to_run.append("binary_label_short")
            if "binary_label" not in targets_to_run:
                targets_to_run.append("binary_label")
        elif args.direction == "both":
            if "binary_label_long" not in targets_to_run:
                targets_to_run.append("binary_label_long")
            if "binary_label_short" not in targets_to_run:
                targets_to_run.append("binary_label_short")
            if "binary_label_long" not in targets_to_run:
                targets_to_run.append("binary_label_long")
            if "binary_label_short" not in targets_to_run:
                targets_to_run.append("binary_label_short")

    for tgt in targets_to_run:
        if args.independent_mode or args.auto_train:
            import asyncio
            regime_labels = None
            if getattr(args, "regime_labels", None):
                regime_labels = [s.strip() for s in str(args.regime_labels).split(",") if s.strip()]
            asyncio.run(_run_specialist_training(
                symbol=args.symbol,
                exchange=args.exchange,
                timeframe=args.timeframe,
                direction=args.direction,
                regime_timeframe=args.regime_timeframe,
                lookback_days=args.lookback_days,
                train_per_regime=bool(getattr(args, "train_per_regime", False)),
                regime_labels=regime_labels,
            ))

        print(f"\n--- Running diagnostics for target: {tgt} ---")
        try:
            md_path, csv_path = run_diagnostics(
                symbol=args.symbol,
                exchange=args.exchange,
                timeframe=args.timeframe,
                direction=args.direction,
                model=args.model,
                regime_timeframe=args.regime_timeframe,
                target_col=tgt,
                cv_folds=args.cv_folds,
                lookback_days=args.lookback_days,
                enable_risk_hmm_specialist=args.enable_hmm_risk_specialist,
                projection_mode=args.projection_mode,
                enable_orthogonalization=args.enable_orthogonalization,
                run_optimized_orthogonalization=args.run_optimized_orthogonalization,
                anchor_specialist=args.anchor_specialist,
                enable_cache=args.enable_cache,
                feature_optimization=args.feature_optimization,
                orthogonal_hpo=args.orthogonal_hpo,
                conservative_pruning=args.conservative_pruning,
                run_lgbm_comparison=args.run_lgbm_comparison,
                orthogonal_comparison_report=args.orthogonal_comparison_report,
                enable_moe=args.enable_moe,
                enable_adversarial_orthogonalization=args.enable_adversarial_orthogonalization,
                adversarial_penalty=args.adversarial_penalty,
                enable_tv_var=args.enable_tv_var,
                tv_var_window=args.tv_var_window,
                tv_var_regime_aware=args.tv_var_regime_aware,
                tv_var_backtest=args.tv_var_backtest,
                load_regime_label=getattr(args, "load_regime_label", None),
            )
            print(
                f"Specialist feature diagnostics for {tgt} saved to: {md_path} "
                f"and {csv_path}"
            )
        except Exception as e:
            print(f"Failed diagnostics for {tgt}: {e}")


if __name__ == "__main__":
    main()
