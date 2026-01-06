"""
Hierarchical Feature Selection Module.

This module implements a robust feature selection mechanism using Hierarchical Clustering
(De Prado's method) combined with quality-based filtering to address concept dominance
and ensure feature diversity.

Key components:
1. Quality Score: Stability (low std), Signal (mean), Low Fat Tails (kurtosis).
2. Concept Dominance Defense: Drop low variance features and bottom 20% quality.
3. Hierarchical Clustering: Select best-in-class features from orthogonal clusters.

Created: 2025-12-11
"""

import logging
import warnings
from typing import List, Tuple, Dict, Optional, Any, Union
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import kurtosis

from src.utils.tprint import tprint_info, tprint_warning, tprint_success, tprint_error

logger = logging.getLogger(__name__)

def _base_quality_score(series: np.ndarray) -> float:
    """
    The actual math engine. 
    High Stability (low std), High Signal (mean), Low Fat Tails (kurtosis).
    """
    # 1. Handle constant or empty data
    if len(series) < 10 or np.std(series) == 0:
        return 0.0
    
    # 2. Calculate components
    # Using annualized Sharpe-like ratio logic or similar stability metric
    mu = np.mean(series)
    sigma = np.std(series)
    
    # Avoid zero division - Drop features with effectively zero variance
    if sigma < 1e-9: 
        return 0.0
        
    kurt = kurtosis(series, fisher=True) # Fisher=True means normal is 0.0
    
    # 3. Formulate Score
    # We penalize high kurtosis (fat tails/risk) and reward high Sharpe
    # Adding 3.0 to kurtosis to ensure strictly positive denominator if desired, 
    # or just using max(1.0, ...) to avoid division blowouts.
    
    signal_to_noise = np.abs(mu / sigma)
    # Penalty: If kurtosis is high (unstable), reduce score. 
    # (1 + abs(kurt)) ensures we don't divide by zero or negative.
    penalty = 1 + np.abs(kurt) 
    
    return signal_to_noise / penalty

def calculate_time_robust_quality(series: Union[pd.Series, np.ndarray], chunk_size: int = 2000) -> float:
    """
    Calculates Quality Score in chunks and returns the Worst-Case score.
    Non-recursive implementation.
    """
    # Convert to numpy and handle NaNs
    if isinstance(series, pd.Series):
        series = series.values
    series = np.nan_to_num(series)

    n = len(series)

    # Fallback for short series: just calculate global score
    if n < chunk_size:
        return _base_quality_score(series)

    scores = []

    # Rolling/Chunked Evaluation
    for i in range(0, n, chunk_size):
        chunk = series[i : i + chunk_size]

        # Skip small incomplete chunks at the end to ensure statistical significance
        if len(chunk) < chunk_size // 2:
            continue

        # Call the helper (Base Calculation)
        q_score = _base_quality_score(chunk)
        scores.append(q_score)

    if not scores:
        return _base_quality_score(series)

    # Return 10th Percentile (Conservative - robustness check)
    return np.percentile(scores, 10)


def calculate_residual_autocorr_penalty(
    feature_series: Union[pd.Series, np.ndarray],
    target_series: Union[pd.Series, np.ndarray],
    max_lag: int = 5
) -> float:
    """
    Calculate residual autocorrelation penalty for a feature.

    Higher penalty (closer to 1.0) for features whose residuals show strong autocorrelation,
    indicating they miss sequential patterns in the target.

    Args:
        feature_series: Feature values
        target_series: Target values (must be aligned)
        max_lag: Maximum lag to check for autocorrelation

    Returns:
        Penalty factor between 0 and 1 (1 = severe penalty, 0 = no penalty)
    """
    try:
        # Ensure aligned series
        if isinstance(feature_series, pd.Series) and isinstance(target_series, pd.Series):
            common_idx = feature_series.index.intersection(target_series.index)
            if len(common_idx) < 50:
                return 0.0  # Not enough data
            feature_vals = feature_series.loc[common_idx].values
            target_vals = target_series.loc[common_idx].values
        else:
            feature_vals = np.asarray(feature_series)
            target_vals = np.asarray(target_series)
            min_len = min(len(feature_vals), len(target_vals))
            feature_vals = feature_vals[:min_len]
            target_vals = target_vals[:min_len]

        # Remove NaNs
        valid_mask = ~(np.isnan(feature_vals) | np.isnan(target_vals))
        feature_vals = feature_vals[valid_mask]
        target_vals = target_vals[valid_mask]

        if len(feature_vals) < 50:
            return 0.0

        # Simple linear regression to get residuals
        # feature_vals = residuals + predicted
        try:
            # Use numpy polyfit for speed
            coeffs = np.polyfit(feature_vals, target_vals, 1)
            predicted = np.polyval(coeffs, feature_vals)
            residuals = target_vals - predicted
        except:
            # Fallback: just use feature as proxy for residuals
            residuals = target_vals - feature_vals

        # Calculate autocorrelation of residuals
        max_autocorr = 0.0
        for lag in range(1, min(max_lag + 1, len(residuals) // 4)):
            autocorr = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
            if not np.isnan(autocorr):
                max_autocorr = max(max_autocorr, abs(autocorr))

        # Penalty increases with autocorr strength
        # 0 autocorr = 0 penalty, 0.5+ autocorr = significant penalty
        penalty = min(1.0, max_autocorr * 2.0)
        return penalty

    except Exception:
        # On any error, return no penalty
        return 0.0


def detect_feature_leakage(
    feature_series: Union[pd.Series, np.ndarray],
    target_series: Union[pd.Series, np.ndarray],
    threshold: float = 0.95
) -> bool:
    """
    Detect potential feature leakage by checking for suspiciously high correlation
    with the target (forward-looking bias).

    Args:
        feature_series: Feature values
        target_series: Target values
        threshold: Correlation threshold above which leakage is suspected

    Returns:
        True if leakage is suspected
    """
    try:
        # Ensure aligned series
        if isinstance(feature_series, pd.Series) and isinstance(target_series, pd.Series):
            common_idx = feature_series.index.intersection(target_series.index)
            if len(common_idx) < 30:
                return False
            feature_vals = feature_series.loc[common_idx].values
            target_vals = target_series.loc[common_idx].values
        else:
            feature_vals = np.asarray(feature_series)
            target_vals = np.asarray(target_series)
            min_len = min(len(feature_vals), len(target_vals))
            feature_vals = feature_vals[:min_len]
            target_vals = target_vals[:min_len]

        # Remove NaNs
        valid_mask = ~(np.isnan(feature_vals) | np.isnan(target_vals))
        feature_vals = feature_vals[valid_mask]
        target_vals = target_vals[valid_mask]

        if len(feature_vals) < 30:
            return False

        # Calculate correlation
        corr = np.corrcoef(feature_vals, target_vals)[0, 1]
        if np.isnan(corr):
            return False

        return abs(corr) > threshold

    except Exception:
        return False


def calculate_redundancy_score(
    feature_series: Union[pd.Series, np.ndarray],
    other_features: List[Union[pd.Series, np.ndarray]],
    threshold: float = 0.85
) -> float:
    """
    Calculate redundancy score for a feature based on correlation with other features.

    Higher score indicates more redundancy (feature is highly correlated with others).

    Args:
        feature_series: Feature to check
        other_features: List of other feature series
        threshold: Correlation threshold for redundancy

    Returns:
        Redundancy score between 0 and 1
    """
    if not other_features:
        return 0.0

    try:
        # Convert to numpy arrays and align
        if isinstance(feature_series, pd.Series):
            feature_vals = feature_series.values
            feature_idx = feature_series.index
        else:
            feature_vals = np.asarray(feature_series)
            feature_idx = None

        max_corr = 0.0
        for other_feat in other_features:
            try:
                if isinstance(other_feat, pd.Series):
                    if feature_idx is not None:
                        common_idx = feature_idx.intersection(other_feat.index)
                        if len(common_idx) < 30:
                            continue
                        feat_a = feature_series.loc[common_idx].values
                        feat_b = other_feat.loc[common_idx].values
                    else:
                        continue
                else:
                    feat_a = feature_vals
                    feat_b = np.asarray(other_feat)
                    min_len = min(len(feat_a), len(feat_b))
                    feat_a = feat_a[:min_len]
                    feat_b = feat_b[:min_len]

                # Remove NaNs
                valid_mask = ~(np.isnan(feat_a) | np.isnan(feat_b))
                feat_a = feat_a[valid_mask]
                feat_b = feat_b[valid_mask]

                if len(feat_a) < 30:
                    continue

                corr = np.corrcoef(feat_a, feat_b)[0, 1]
                if not np.isnan(corr):
                    max_corr = max(max_corr, abs(corr))

            except Exception:
                continue

        # Convert correlation to redundancy score
        # corr >= threshold gets score proportional to correlation strength
        if max_corr >= threshold:
            return (max_corr - threshold) / (1.0 - threshold)
        else:
            return 0.0

    except Exception:
        return 0.0

def select_features_hierarchical(
    df_features: pd.DataFrame,
    target_n: int = 70,
    drop_bottom_percentile: float = 0.20,
    log_dir: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
    target_series: Optional[Union[pd.Series, np.ndarray]] = None,
    autocorr_penalty_weight: float = 0.3,
    enable_leakage_detection: bool = True,
    enable_redundancy_pruning: bool = True,
    leakage_threshold: float = 0.95,
    redundancy_threshold: float = 0.85,
    enable_advanced_importance: bool = False,
    importance_config: Optional[Dict[str, Any]] = None
) -> Tuple[List[str], Dict[str, Any]]:
    """
    De Prado's Hierarchical Feature Selection with Concept Dominance Fixes and Autocorr Penalization.

    Guarantees diversity by picking best-in-class from N distinct clusters.
    Optionally penalizes features whose residuals show strong autocorrelation.

    Steps:
    1. Drop features with std_dev < 1e-9 (Zero Variance)
    2. Calculate Quality Score for all features (with optional autocorr penalty)
    3. Drop bottom 20% of features based on quality (Concept Dominance Fix)
    4. Hierarchical Clustering on remaining features
    5. Select best feature per cluster based on quality score

    Args:
        df_features: DataFrame of features
        target_n: Target number of features to select
        drop_bottom_percentile: Percentile of low-quality features to drop before clustering
        log_dir: Optional directory to save selection log
        metadata: Optional metadata for logging
        target_series: Optional target series for autocorr penalization
        autocorr_penalty_weight: Weight for autocorr penalty (0-1, higher = stronger penalty)

    Returns:
        Tuple of (selected_feature_names, selection_log)
    """
    tprint_info(f"🌳 Starting Hierarchical Selection: {df_features.shape[1]} -> {target_n}")
    
    initial_features_count = df_features.shape[1]
    selection_log = {
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {},
        "initial_count": initial_features_count,
        "steps": {}
    }

    # 0. Safety: Drop constant columns first to prevent NaN correlations
    # (Concept Dominance Fix Step 1: Drop std_dev < 1e-9)
    # Using numpy std for speed on whole dataframe
    stds = df_features.std()
    valid_std_mask = stds >= 1e-9
    df_features = df_features.loc[:, valid_std_mask]
    
    dropped_std_count = initial_features_count - df_features.shape[1]
    if dropped_std_count > 0:
        tprint_info(f"   ↪ Dropped {dropped_std_count} features with near-zero variance")
    
    selection_log["steps"]["variance_filter"] = {
        "dropped_count": dropped_std_count,
        "remaining_count": df_features.shape[1]
    }

    if df_features.empty:
        tprint_error("❌ All features dropped due to low variance!")
        return [], selection_log

    # 1.5. Limit to last 6 months for selection (User Request)
    # We use a 6-month lookback to ensure features are relevant to current regime
    selection_df = df_features
    if isinstance(df_features.index, pd.DatetimeIndex):
        six_months_ago = df_features.index.max() - pd.Timedelta(days=180)
        selection_df = df_features[df_features.index >= six_months_ago]
        tprint_info(f"   📅 Limiting selection calculation to last 6 months (since {six_months_ago.date()})")
        tprint_info(f"      Rows used: {len(selection_df)}/{len(df_features)}")
        
        if len(selection_df) < 100:
            tprint_warning("   ⚠️ < 100 rows in last 6 months! Falling back to full history.")
            selection_df = df_features
    else:
        tprint_warning("   ⚠️ DataFrame index is not DatetimeIndex; cannot apply 6-month lookback.")

    # 1.6 Re-check variance on the SLICED dataframe
    # Features that had variance historically might be dead/constant now.
    slice_stds = selection_df.std()
    valid_slice_mask = slice_stds >= 1e-9
    
    dropped_slice_count = (~valid_slice_mask).sum()
    if dropped_slice_count > 0:
         tprint_info(f"   ↪ Dropped {dropped_slice_count} features that are constant in the last 6 months")
         selection_df = selection_df.loc[:, valid_slice_mask]
         df_features = df_features.loc[:, valid_slice_mask]

    if selection_df.empty:
         tprint_error("❌ All features dropped due to low variance in recent period!")
         return [], selection_log

    # 2. Calculate Quality and Concept Dominance Fix Step 2
    # Pre-calculate quality for speed using the SLICED dataframe
    qualities = {}
    autocorr_penalties = {}
    leakage_flags = {}
    redundancy_scores = {}

    for col in selection_df.columns:
        base_quality = calculate_time_robust_quality(selection_df[col].values)

        # Apply autocorr penalty if target series provided
        autocorr_penalty = 0.0
        if target_series is not None:
            autocorr_penalty = calculate_residual_autocorr_penalty(
                selection_df[col], target_series
            )
            autocorr_penalties[col] = autocorr_penalty

        # Check for leakage if enabled
        is_leaky = False
        if enable_leakage_detection and target_series is not None:
            is_leaky = detect_feature_leakage(selection_df[col], target_series, leakage_threshold)
            leakage_flags[col] = is_leaky

        # Calculate redundancy score if enabled
        redundancy_score = 0.0
        if enable_redundancy_pruning and len(selection_df.columns) > 1:
            other_cols = [c for c in selection_df.columns if c != col]
            other_series = [selection_df[c] for c in other_cols]
            redundancy_score = calculate_redundancy_score(
                selection_df[col], other_series, redundancy_threshold
            )
            redundancy_scores[col] = redundancy_score

        # Apply penalties: autocorr, leakage (severe), redundancy
        penalty_multiplier = 1.0
        penalty_multiplier *= (1.0 - autocorr_penalty_weight * autocorr_penalty)
        if is_leaky:
            penalty_multiplier *= 0.1  # Severe penalty for leakage
        penalty_multiplier *= (1.0 - 0.5 * redundancy_score)  # Moderate penalty for redundancy

        # Final quality
        final_quality = base_quality * penalty_multiplier
        qualities[col] = max(0.0, final_quality)  # Ensure non-negative

    # Sort features by quality
    sorted_features = sorted(qualities.keys(), key=lambda x: qualities[x], reverse=True)
    
    # Drop bottom 20% (or specified percentile)
    n_features = len(sorted_features)
    n_keep_quality = int(n_features * (1.0 - drop_bottom_percentile))
    # Ensure we keep at least target_n if possible
    n_keep_quality = max(n_keep_quality, target_n)
    n_keep_quality = max(n_keep_quality, 1) # Safety
    
    features_to_keep = sorted_features[:n_keep_quality]
    features_to_drop = sorted_features[n_keep_quality:]
    
    # Update both dataframes (we need selection_df for clustering later)
    df_features = df_features[features_to_keep] 
    selection_df = selection_df[features_to_keep]
    
    tprint_info(f"   ↪ Dropped {len(features_to_drop)} features (bottom {drop_bottom_percentile*100:.0f}% quality)")
    tprint_info(f"   ↪ Remaining for clustering: {selection_df.shape[1]}")
    
    # ------------------------------------------------------------------
    # Advanced importance filtering (optional)
    # ------------------------------------------------------------------
    importance_results = None
    if enable_advanced_importance and target_series is not None and len(df_features.columns) > target_n:
        try:
            from .advanced_feature_importance import (
                compute_feature_importance_analysis,
                select_features_by_advanced_importance
            )

            # Prepare target data aligned with features
            if isinstance(target_series, pd.Series):
                common_idx = df_features.index.intersection(target_series.index)
                if len(common_idx) < 50:
                    tprint_warning("Insufficient overlapping data for advanced importance analysis")
                else:
                    X_importance = df_features.loc[common_idx]
                    y_importance = target_series.loc[common_idx]

                    # Run advanced importance analysis
                    importance_config = importance_config or {
                        "methods": ["mda", "shap"],
                        "mda_estimators": 30,  # Faster for feature selection
                        "shap_max_evals": 500,
                        "shap_n_samples": min(500, len(X_importance))
                    }

                    importance_results = compute_feature_importance_analysis(
                        X_importance, y_importance, importance_config, verbose=False
                    )

                    if importance_results.get("methods_used"):
                        # Select features using advanced importance
                        importance_selected = select_features_by_advanced_importance(
                            X_importance, y_importance, importance_results,
                            max_features=max(len(df_features.columns) // 2, target_n * 2),  # Select more for diversity
                            method_preference="ensemble"
                        )

                        # Keep only importance-selected features
                        original_count = len(df_features.columns)
                        df_features = df_features[importance_selected]
                        selection_df = selection_df[importance_selected]

                        tprint_info(f"   ↪ Advanced importance filtering: {original_count} → {len(df_features.columns)} features")

                        selection_log["steps"]["importance_filter"] = {
                            "applied": True,
                            "methods_used": importance_results.get("methods_used", []),
                            "original_count": original_count,
                            "filtered_count": len(df_features.columns),
                            "top_features_sample": importance_selected[:5]
                        }
                    else:
                        tprint_warning("   Advanced importance analysis failed, proceeding without it")
                        selection_log["steps"]["importance_filter"] = {"applied": False, "error": "analysis_failed"}

            else:
                tprint_warning("   Advanced importance requires pandas Series target, skipping")
                selection_log["steps"]["importance_filter"] = {"applied": False, "error": "invalid_target_format"}

        except ImportError:
            tprint_warning("   Advanced importance module not available, skipping")
            selection_log["steps"]["importance_filter"] = {"applied": False, "error": "module_not_available"}
        except Exception as e:
            tprint_warning(f"   Advanced importance filtering failed: {e}")
            selection_log["steps"]["importance_filter"] = {"applied": False, "error": str(e)}

    selection_log["steps"]["quality_filter"] = {
        "dropped_count": len(features_to_drop),
        "remaining_count": df_features.shape[1],
        "dropped_examples": features_to_drop[:5],
        "autocorr_penalty_applied": target_series is not None,
        "autocorr_penalty_weight": autocorr_penalty_weight,
        "autocorr_penalties_sample": dict(list(autocorr_penalties.items())[:5]) if autocorr_penalties else {},
        "leakage_detection_enabled": enable_leakage_detection,
        "leakage_threshold": leakage_threshold,
        "leaky_features_detected": sum(leakage_flags.values()) if leakage_flags else 0,
        "redundancy_pruning_enabled": enable_redundancy_pruning,
        "redundancy_threshold": redundancy_threshold,
        "redundancy_scores_sample": dict(list(redundancy_scores.items())[:5]) if redundancy_scores else {},
        "advanced_importance_applied": importance_results is not None,
        "importance_methods_used": importance_results.get("methods_used", []) if importance_results else []
    }

    # If we have fewer features than target, just return them all
    if df_features.shape[1] <= target_n:
        tprint_warning(f"⚠️ Remaining features ({df_features.shape[1]}) <= Target ({target_n}). Returning all.")
        return df_features.columns.tolist(), selection_log
    
    # 4. Hierarchical Clustering logic
    # 1. Calculate Correlation & Distance
    # Spearman is often safer for financial data (non-linear relationships)
    # USE SELECTION_DF (Recent 6 months) for correlation structure
    corr_matrix = selection_df.corr(method='spearman').fillna(0)
    try:
        np.fill_diagonal(corr_matrix.values, 1.0)
    except Exception:
        pass
    
    # Distance metric d = sqrt(2(1-rho))
    # Clip to avoid negative due to float precision errors
    dist_matrix = np.sqrt(np.clip(2 * (1 - np.abs(corr_matrix)), 0, None))
    try:
        np.fill_diagonal(dist_matrix.values, 0.0)
    except Exception:
        pass
    try:
        dist_array = squareform(dist_matrix) 
        
        # 2. Hierarchical Clustering (Linkage)
        linkage_matrix = hierarchy.linkage(dist_array, method='ward')
        
        # 3. Form Clusters
        cluster_labels = hierarchy.fcluster(linkage_matrix, t=target_n, criterion='maxclust')
        
        # 4. Select Best Feature per Cluster
        selected_feats = []
        
        for cluster_id in range(1, target_n + 1):
            # Get all features in this cluster
            # Note: cluster_labels aligns with corr_matrix.index
            members_mask = (cluster_labels == cluster_id)
            members = corr_matrix.index[members_mask].tolist()
            
            if not members:
                continue
                
            # Pick the one with highest Quality Score (Calculated on recent data)
            best_in_cluster = max(members, key=lambda x: qualities[x])
            selected_feats.append(best_in_cluster)
            
        tprint_success(f"✅ Selected {len(selected_feats)} orthogonal features from {df_features.shape[1]} candidates.")
        
        selection_log["steps"]["clustering"] = {
            "input_count": df_features.shape[1],
            "target_clusters": target_n,
            "selected_count": len(selected_feats)
        }
        selection_log["selected_features"] = selected_feats
        
        if log_dir:
            try:
                log_dir = Path(log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = log_dir / f"hierarchical_selection_{timestamp}.json"
                with open(save_path, 'w') as f:
                    json.dump(selection_log, f, indent=2, default=str)
            except Exception as e:
                tprint_error(f"Failed to save selection log: {e}")

        return selected_feats, selection_log

    except Exception as e:
        tprint_error(f"Hierarchical clustering failed: {e}. Fallback to Top-N quality.")
        # Fallback: Just take top N highest quality features
        fallback_features = features_to_keep[:target_n]
        return fallback_features, selection_log

