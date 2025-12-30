"""
LGBM-Based Feature Selection Module for Meta-Labeling.

This module implements a sophisticated feature selection pipeline using LightGBM:
1. Correlation pruning to remove highly correlated features
2. Iterative importance-based filtering with external subsampling (65%)
3. Permutation importance with RFE for final feature sets (50/60/70/80 features)
4. Feature set persistence with exchange/asset/datetime metadata

Based on guidance from "Advances in Financial Machine Learning" by Marcos López de Prado.

Created: 2025-12-08
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
except ImportError:
    permutation_importance = None
    train_test_split = None

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error

logger = logging.getLogger(__name__)

# Default LGBM parameters for feature selection
DEFAULT_LGBM_PARAMS = {
    "boosting_type": "gbdt",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "num_leaves": 16,  # ≤ 2^max_depth
    "min_data_in_leaf": 30,  # 20-50 range
    "subsample": 1.0,  # Handled externally
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbosity": -1,
    "objective": "binary",
    "metric": "auc",
    "n_jobs": -1,
}

# Feature selection configuration
ITERATIVE_SELECTION_BUFFER_RATIO = 1.5

FEATURE_SELECTION_CONFIG = {
    "subsample_fraction": 0.65,  # 65% subsampling
    "n_iterations": 10,  # Number of LGBM runs
    "bottom_percentile": 0.30,  # Bottom 30% of importance
    "discard_threshold": 0.70,  # Discard if in bottom 70% of the time
    "target_features_initial": 80,  # Initial target after iteration
    "rfe_features_per_step": 10,  # Remove 10 features per RFE step
    "permutation_sample_fraction": 0.50,  # 50% sample for permutation importance
    "permutation_n_repeats": 3,  # 3 shuffles per feature
    "target_feature_sets": [80, 70, 60, 50],  # Final feature set sizes
}


def correlation_pruning(
    X: pd.DataFrame,
    correlation_threshold: float = 0.95,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Remove highly correlated features using correlation matrix.
    
    Args:
        X: Feature matrix
        correlation_threshold: Remove features with correlation > this value
        
    Returns:
        Tuple of (pruned_X, kept_features, dropped_features)
    """
    tprint_info(f"🔗 Correlation pruning: Starting with {len(X.columns)} features")
    
    # Compute correlation matrix
    X_clean = X.fillna(0)
    corr_matrix = X_clean.corr().abs()
    
    # Get upper triangle
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to drop (keep first of each correlated pair)
    to_drop = [
        col for col in upper_tri.columns 
        if any(upper_tri[col] > correlation_threshold)
    ]
    
    kept_features = [col for col in X.columns if col not in to_drop]
    
    tprint_info(f"   ↪ Removed {len(to_drop)} highly correlated features (>{correlation_threshold})")
    tprint_info(f"   ↪ Remaining features: {len(kept_features)}")
    
    return X[kept_features], kept_features, to_drop


def _get_lgbm_params(random_state: int) -> Dict[str, Any]:
    """Get LGBM parameters with specified random state."""
    params = DEFAULT_LGBM_PARAMS.copy()
    params["random_state"] = random_state
    return params


def iterative_lgbm_importance_selection(
    X: pd.DataFrame,
    y: pd.Series,
    target_n_features: int = 80,
    subsample_fraction: float = 0.65,
    n_iterations: int = 10,
    bottom_percentile: float = 0.30,
    discard_threshold: float = 0.70,
    log_dir: Optional[Path] = None,
    returns: Optional[pd.Series] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Iterative LGBM-based feature selection with external subsampling.
    
    This method:
    1. Subsamples the training set (65% randomly, done EXTERNALLY - not LGBM internal bagging)
    2. Runs LGBM 10 times to compute feature importance
    3. Removes features appearing in bottom 30% of importance rankings ≥70% of the time
    4. Iterates until target_n_features are reached
    
    Args:
        X: Feature matrix
        y: Binary labels (fallback if returns not provided)
        target_n_features: Target number of features (default 80)
        subsample_fraction: Fraction of samples to use per run (0.65)
        n_iterations: Number of LGBM runs (10)
        bottom_percentile: Bottom percentile threshold (0.30)
        discard_threshold: Discard if in bottom this fraction of runs (0.70)
        log_dir: Optional directory to save iteration logs
        returns: Optional realized returns - if provided, uses sign(returns) as target
        sample_weight: Optional sample weights (e.g., return-magnitude weighted)
        
    Returns:
        Tuple of (selected_features, selection_log)
    """
    if lgb is None:
        raise ImportError("lightgbm is required for LGBM-based feature selection")
    
    tprint_info(f"🌳 Starting iterative LGBM importance selection")
    tprint_info(f"   ↪ Target features: {target_n_features}")
    tprint_info(f"   ↪ Subsample fraction: {subsample_fraction}")
    tprint_info(f"   ↪ Iterations per round: {n_iterations}")
    
    # Determine target: use sign(returns) if returns provided, otherwise use y
    target_type = "labels"
    if returns is not None:
        try:
            returns_arr = np.asarray(returns, dtype=float)
            # sign(returns): positive return = 1 (profit), else = 0
            y_target = pd.Series((returns_arr > 0).astype(int), index=X.index[:len(returns_arr)])
            target_type = "sign_returns"
            tprint_info(f"   ↪ Using sign(returns) as target (profit=1, loss=0)")
        except Exception:
            y_target = y
            tprint_warning(f"   ⚠️ Failed to use returns, falling back to labels")
    else:
        y_target = y
    
    # Prepare data
    clean_mask = ~y_target.isna()
    X_clean = X[clean_mask].fillna(0)
    y_clean = y_target[clean_mask]
    
    # Prepare sample weights
    w_arr = None
    if sample_weight is not None:
        try:
            w_full = np.asarray(sample_weight, dtype=float)
            w_arr = w_full[clean_mask] if len(w_full) == len(clean_mask) else w_full[:len(y_clean)]
            w_arr = np.where(np.isfinite(w_arr) & (w_arr > 0), w_arr, 1.0)
        except Exception:
            w_arr = None
    
    n_samples = len(y_clean)
    if n_samples < 100:
        tprint_warning(f"⚠️ Too few samples ({n_samples}) for LGBM feature selection")
        return list(X.columns), {"error": "too_few_samples"}
    
    current_features = list(X_clean.columns)
    selection_log: Dict[str, Any] = {
        "initial_features": len(current_features),
        "target_features": target_n_features,
        "target_type": target_type,
        "iterations": [],
    }

    
    iteration = 0
    while len(current_features) > target_n_features:
        iteration += 1
        X_iter = X_clean[current_features]
        
        tprint_info(f"   📊 Iteration {iteration}: {len(current_features)} features")
        
        # Collect importance rankings across runs
        importance_rankings: Dict[str, List[int]] = {f: [] for f in current_features}
        
        for run_idx in range(n_iterations):
            # External subsampling (NOT LGBM internal bagging)
            rng = np.random.default_rng(42 + iteration * 100 + run_idx)
            sample_size = int(len(X_iter) * subsample_fraction)
            sample_indices = rng.choice(len(X_iter), size=sample_size, replace=False)
            
            X_sample = X_iter.iloc[sample_indices]
            y_sample = y_clean.iloc[sample_indices]
            w_sample = w_arr[sample_indices] if w_arr is not None else None
            
            # Train LGBM
            params = _get_lgbm_params(random_state=42 + iteration * 100 + run_idx)
            model = lgb.LGBMClassifier(**params)
            
            try:
                fit_kwargs = {"eval_set": [(X_sample, y_sample)], "callbacks": [lgb.early_stopping(50, verbose=False)]}
                if w_sample is not None:
                    fit_kwargs["sample_weight"] = w_sample
                model.fit(X_sample, y_sample, **fit_kwargs)

                
                # Get feature importance
                importances = model.feature_importances_
                
                # Rank features (0 = most important)
                sorted_indices = np.argsort(importances)[::-1]
                for rank, idx in enumerate(sorted_indices):
                    feature_name = current_features[idx]
                    importance_rankings[feature_name].append(rank)
                    
            except Exception as e:
                tprint_warning(f"   ⚠️ LGBM run {run_idx + 1} failed: {e}")
                continue
        
        # Calculate how often each feature appears in bottom percentile
        n_features = len(current_features)
        bottom_threshold_rank = int(n_features * (1 - bottom_percentile))  # Rank threshold
        
        features_to_discard = []
        features_to_keep = []
        
        for feature, rankings in importance_rankings.items():
            if not rankings:
                features_to_discard.append(feature)
                continue
                
            # Count how often feature is in bottom percentile
            n_in_bottom = sum(1 for r in rankings if r >= bottom_threshold_rank)
            fraction_in_bottom = n_in_bottom / len(rankings)
            
            if fraction_in_bottom >= discard_threshold:
                features_to_discard.append(feature)
            else:
                features_to_keep.append(feature)
        
        # Log iteration results
        iter_log = {
            "iteration": iteration,
            "features_before": len(current_features),
            "features_discarded": len(features_to_discard),
            "features_kept": len(features_to_keep),
            "discarded_features": features_to_discard,
            "kept_features": features_to_keep[:20],  # Sample of kept features
        }
        selection_log["iterations"].append(iter_log)
        
        tprint_info(f"   ↪ Discarding {len(features_to_discard)} features (in bottom {bottom_percentile*100:.0f}% ≥{discard_threshold*100:.0f}% of runs)")
        
        # If we discarded too many, keep highest ranked ones
        if len(features_to_keep) < target_n_features:
            # Rank all features by mean ranking and keep top N
            mean_rankings = {
                f: np.mean(r) if r else float('inf') 
                for f, r in importance_rankings.items()
            }
            sorted_features = sorted(mean_rankings.keys(), key=lambda x: mean_rankings[x])
            features_to_keep = sorted_features[:target_n_features]
            tprint_info(f"   ↪ Keeping top {target_n_features} features by mean ranking")
        
        current_features = features_to_keep
        
        # Safety check to prevent infinite loop
        if len(features_to_discard) == 0:
            tprint_warning(f"   ⚠️ No features discarded, stopping iteration")
            break
        
        if iteration > 50:  # Safety limit
            tprint_warning(f"   ⚠️ Max iterations reached")
            break
    
    # If we have fewer than target, keep all remaining
    if len(current_features) < target_n_features:
        tprint_info(f"   ⚠️ Reached {len(current_features)} features (below target {target_n_features})")
    
    selection_log["final_features"] = len(current_features)
    selection_log["selected_feature_names"] = current_features
    
    # Save log if directory provided
    if log_dir is not None:
        try:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"lgbm_importance_selection_{timestamp}.json"
            with open(log_path, "w") as f:
                json.dump(selection_log, f, indent=2, default=str)
            tprint_info(f"   💾 Saved selection log to {log_path}")
        except Exception as e:
            tprint_warning(f"   ⚠️ Failed to save selection log: {e}")
    
    tprint_success(f"✅ LGBM importance selection complete: {len(current_features)} features")
    return current_features, selection_log


def permutation_importance_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    feature_sets: List[int] = [70, 60, 50],
    sample_fraction: float = 0.50,
    n_repeats: int = 3,
    features_per_step: int = 10,
    log_dir: Optional[Path] = None,
    returns: Optional[pd.Series] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[Dict[int, List[str]], Dict[str, Any]]:
    """
    Permutation importance with Recursive Feature Elimination (RFE).
    
    Uses permutation importance to progressively eliminate features:
    - Sample 50% of data (drop every other sample)
    - 3 shuffles per feature for importance calculation
    - Remove 10 features per RFE step
    
    Args:
        X: Feature matrix (should start from 80 features)
        y: Binary labels (fallback if returns not provided)
        feature_sets: Target feature set sizes [70, 60, 50]
        sample_fraction: Fraction of samples to use (0.50)
        n_repeats: Number of permutation repeats (3)
        features_per_step: Features to remove per RFE step (10)
        log_dir: Optional directory to save iteration logs
        returns: Optional realized returns - if provided, uses sign(returns) as target
        sample_weight: Optional sample weights (e.g., return-magnitude weighted)
        
    Returns:
        Tuple of (feature_sets_dict, rfe_log)
    """
    if lgb is None or permutation_importance is None:
        raise ImportError("lightgbm and sklearn are required for permutation importance RFE")
    
    tprint_info(f"🔄 Starting permutation importance RFE")
    tprint_info(f"   ↪ Starting features: {len(X.columns)}")
    tprint_info(f"   ↪ Target feature sets: {feature_sets}")
    tprint_info(f"   ↪ Sample fraction: {sample_fraction}")
    tprint_info(f"   ↪ Permutation repeats: {n_repeats}")
    
    # Determine target: use sign(returns) if returns provided, otherwise use y
    target_type = "labels"
    if returns is not None:
        try:
            returns_arr = np.asarray(returns, dtype=float)
            y_target = pd.Series((returns_arr > 0).astype(int), index=X.index[:len(returns_arr)])
            target_type = "sign_returns"
            tprint_info(f"   ↪ Using sign(returns) as target (profit=1, loss=0)")
        except Exception:
            y_target = y
            tprint_warning(f"   ⚠️ Failed to use returns, falling back to labels")
    else:
        y_target = y
    
    # Prepare data
    clean_mask = ~y_target.isna()
    X_clean = X[clean_mask].fillna(0)
    y_clean = y_target[clean_mask]
    
    # Prepare sample weights
    w_arr = None
    if sample_weight is not None:
        try:
            w_full = np.asarray(sample_weight, dtype=float)
            w_arr = w_full[clean_mask] if len(w_full) == len(clean_mask) else w_full[:len(y_clean)]
            w_arr = np.where(np.isfinite(w_arr) & (w_arr > 0), w_arr, 1.0)
        except Exception:
            w_arr = None
    
    n_samples = len(y_clean)
    if n_samples < 100:
        tprint_warning(f"⚠️ Too few samples ({n_samples}) for permutation importance")
        return {s: list(X.columns)[:s] for s in feature_sets}, {"error": "too_few_samples"}
    
    # Subsample: drop every other sample (50%)
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(X_clean), size=int(len(X_clean) * sample_fraction), replace=False)
    X_sample = X_clean.iloc[sample_indices]
    y_sample = y_clean.iloc[sample_indices]
    w_sample = w_arr[sample_indices] if w_arr is not None else None
    
    current_features = list(X_clean.columns)
    feature_sets_dict: Dict[int, List[str]] = {}
    rfe_log: Dict[str, Any] = {
        "initial_features": len(current_features),
        "target_sets": feature_sets,
        "target_type": target_type,
        "iterations": [],
    }

    
    # Sort target sets in descending order
    sorted_targets = sorted(feature_sets, reverse=True)
    
    iteration = 0
    for target_size in sorted_targets:
        while len(current_features) > target_size:
            iteration += 1
            X_iter = X_sample[current_features]
            
            tprint_info(f"   📊 RFE Iteration {iteration}: {len(current_features)} → {max(target_size, len(current_features) - features_per_step)} features")
            
            # Train model
            params = _get_lgbm_params(random_state=42 + iteration)
            model = lgb.LGBMClassifier(**params)
            
            try:
                if w_sample is not None:
                    model.fit(X_iter, y_sample, sample_weight=w_sample)
                else:
                    model.fit(X_iter, y_sample)

                
                # Calculate permutation importance
                perm_result = permutation_importance(
                    model, X_iter, y_sample,
                    n_repeats=n_repeats,
                    random_state=42 + iteration,
                    n_jobs=-1,
                )
                
                # Get mean importance per feature
                importances_mean = perm_result.importances_mean
                
                # Rank features by importance
                sorted_indices = np.argsort(importances_mean)
                
                # Determine how many to remove
                n_to_remove = min(features_per_step, len(current_features) - target_size)
                if n_to_remove <= 0:
                    break
                
                # Remove lowest importance features
                features_to_remove = [current_features[i] for i in sorted_indices[:n_to_remove]]
                features_to_keep = [f for f in current_features if f not in features_to_remove]
                
                # Log iteration
                iter_log = {
                    "iteration": iteration,
                    "features_before": len(current_features),
                    "features_removed": n_to_remove,
                    "removed_features": features_to_remove,
                    "features_after": len(features_to_keep),
                }
                rfe_log["iterations"].append(iter_log)
                
                tprint_info(f"   ↪ Removed {n_to_remove} features: {features_to_remove[:5]}...")
                
                current_features = features_to_keep
                
            except Exception as e:
                tprint_warning(f"   ⚠️ RFE iteration {iteration} failed: {e}")
                break
            
            if iteration > 100:  # Safety limit
                tprint_warning(f"   ⚠️ Max RFE iterations reached")
                break
        
        # Store feature set at this target size
        feature_sets_dict[target_size] = current_features.copy()
        tprint_info(f"   ✅ Feature set {target_size}: {len(current_features)} features")
    
    rfe_log["feature_sets"] = {k: len(v) for k, v in feature_sets_dict.items()}
    
    # Save log if directory provided
    if log_dir is not None:
        try:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"permutation_rfe_{timestamp}.json"
            with open(log_path, "w") as f:
                json.dump(rfe_log, f, indent=2, default=str)
            tprint_info(f"   💾 Saved RFE log to {log_path}")
        except Exception as e:
            tprint_warning(f"   ⚠️ Failed to save RFE log: {e}")
    
    tprint_success(f"✅ Permutation RFE complete: {list(feature_sets_dict.keys())} feature sets")
    return feature_sets_dict, rfe_log


def lgbm_feature_selection_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    correlation_threshold: float = 0.95,
    target_feature_sets: List[int] = [80, 70, 60, 50],
    log_dir: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
    returns: Optional[pd.Series] = None,
    sample_weight: Optional[np.ndarray] = None,
    samples_per_feature_ratio: int = 100,
) -> Tuple[Dict[int, List[str]], Dict[str, Any]]:
    """
    Complete LGBM-based feature selection pipeline.
    
    Pipeline stages:
    1. Correlation pruning (remove features with correlation > threshold)
    2. Iterative LGBM importance filtering (subsample 65%, 10 runs, remove bottom 30% features appearing ≥70% of runs)
    3. Permutation importance RFE (50% sample, 3 shuffles, remove 10 features per step)
    
    Adaptive Logic:
    - Enforces max 1 feature per N samples (default 100).
    - Adjusts target feature sets dynamically.
    - Scales Iterative LGBM target to avoid bottleneck for RFE.

    Args:
        X: Feature matrix
        y: Binary labels (fallback if returns not provided)
        correlation_threshold: Threshold for correlation pruning (0.95)
        target_feature_sets: List of target feature set sizes [80, 70, 60, 50]
        log_dir: Optional directory to save logs
        metadata: Optional metadata (exchange, asset, etc.)
        returns: Optional realized returns - if provided, uses sign(returns) as target
        sample_weight: Optional sample weights (e.g., return-magnitude weighted)
        samples_per_feature_ratio: Minimum samples per feature ratio (default 100)
        
    Returns:
        Tuple of (feature_sets_dict, pipeline_log)
    """
    target_type = "sign_returns" if returns is not None else "labels"
    tprint_info(f"🚀 Starting LGBM feature selection pipeline")
    tprint_info(f"   ↪ Initial features: {len(X.columns)}")
    tprint_info(f"   ↪ Target sets (Requested): {target_feature_sets}")
    tprint_info(f"   ↪ Target type: {target_type}")

    # --- Adaptive Feature Limit ---
    n_samples = len(y)
    max_features_allowed = max(1, n_samples // samples_per_feature_ratio)
    tprint_info(f"   📏 Adaptive Limit: Max {max_features_allowed} features (1 per {samples_per_feature_ratio} samples)")

    # Filter/Adjust target sets
    adaptive_target_sets = sorted([s for s in target_feature_sets if s <= max_features_allowed], reverse=True)
    if not adaptive_target_sets:
        # Fallback: Create steps down from max_allowed
        step = max(1, max_features_allowed // 4)
        adaptive_target_sets = sorted(list(set([
            max_features_allowed,
            max(1, max_features_allowed - step),
            max(1, max_features_allowed - 2 * step)
        ])), reverse=True)
        # Ensure we don't have duplicates or empty list
        if not adaptive_target_sets:
            adaptive_target_sets = [max_features_allowed]

    tprint_info(f"   ↪ Adaptive Target Sets: {adaptive_target_sets}")
    
    pipeline_log: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {},
        "initial_features": len(X.columns),
        "target_sets": adaptive_target_sets,
        "target_type": target_type,
        "max_features_allowed": max_features_allowed,
        "stages": {},
    }
    
    # Stage 1: Correlation pruning
    X_pruned, kept_features, dropped_features = correlation_pruning(
        X, correlation_threshold
    )
    pipeline_log["stages"]["correlation_pruning"] = {
        "features_before": len(X.columns),
        "features_after": len(kept_features),
        "dropped_features": dropped_features,
    }
    
    # Stage 2: Iterative LGBM importance selection
    # We aim for the largest target set + some buffer for RFE to work with
    # But strictly capped to avoid "10x" issue.
    # If largest target is 10, we don't want to output 100.
    largest_target = adaptive_target_sets[0]
    # Use max(1, ...) for safety with tiny datasets
    iterative_target = min(
        len(kept_features),
        int(max(1, largest_target) * ITERATIVE_SELECTION_BUFFER_RATIO)
    )
    # Ensure at least largest target (unless input features < target)
    iterative_target = max(iterative_target, largest_target)
    # Clamp to input count again just in case
    iterative_target = min(iterative_target, len(kept_features))

    tprint_info(f"   🎯 Iterative Selection Target: {iterative_target} (buffer for RFE)")

    selected_features_iter, importance_log = iterative_lgbm_importance_selection(
        X_pruned[kept_features],
        y,
        target_n_features=iterative_target,
        subsample_fraction=FEATURE_SELECTION_CONFIG["subsample_fraction"],
        n_iterations=FEATURE_SELECTION_CONFIG["n_iterations"],
        bottom_percentile=FEATURE_SELECTION_CONFIG["bottom_percentile"],
        discard_threshold=FEATURE_SELECTION_CONFIG["discard_threshold"],
        log_dir=log_dir,
        returns=returns,
        sample_weight=sample_weight,
    )
    pipeline_log["stages"]["iterative_importance"] = importance_log
    
    # Stage 3: Permutation importance RFE
    # We use adaptive_target_sets here
    # The first set in adaptive_target_sets might be == iterative_target or close to it.
    # RFE should start from selected_features_iter.
    
    # If the iterative process landed on exactly the largest target, we might skip RFE for that target?
    # No, keep it consistent.
    
    # Define RFE targets: All adaptive sets
    feature_sets_rfe, rfe_log = permutation_importance_rfe(
        X_pruned[selected_features_iter],
        y,
        feature_sets=adaptive_target_sets,
        sample_fraction=FEATURE_SELECTION_CONFIG["permutation_sample_fraction"],
        n_repeats=FEATURE_SELECTION_CONFIG["permutation_n_repeats"],
        features_per_step=max(1, FEATURE_SELECTION_CONFIG["rfe_features_per_step"]), # Ensure > 0
        log_dir=log_dir,
        returns=returns,
        sample_weight=sample_weight,
    )
    pipeline_log["stages"]["permutation_rfe"] = rfe_log

    feature_sets_dict = feature_sets_rfe

    # Ensure all original requested targets are present if possible (clipping to max allowed)
    # If user asked for 80, but we maxed at 15, we return 15 for '80' key?
    # Or we just return what we found. The calling code should handle it.
    # For compatibility, let's map requested keys to best available.

    for req_target in target_feature_sets:
        if req_target not in feature_sets_dict:
            # Find closest available set
            available_sizes = sorted(feature_sets_dict.keys())
            if available_sizes:
                # If requested 80, but we have [15, 12, 10], pick 15.
                # If requested 5, pick 10? No, pick closest.
                closest = min(available_sizes, key=lambda x: abs(x - req_target))

                # If closest is significantly smaller than requested due to limit, it's fine.
                # Just alias it.
                feature_sets_dict[req_target] = feature_sets_dict[closest]
                # Don't warn excessively, just log
                # tprint_warning(f"   ⚠️ mapped requested {req_target} to available {closest}")
    
    pipeline_log["feature_sets"] = {
        k: {"count": len(v), "features": v} 
        for k, v in feature_sets_dict.items()
    }
    
    tprint_success(f"✅ Feature selection pipeline complete")
    for size, features in sorted(feature_sets_dict.items(), reverse=True):
        tprint(f"   ↪ {size}-feature set: {len(features)} features", "INFO")
    
    return feature_sets_dict, pipeline_log


class FeatureSetPersistence:
    """
    Manages persistence of selected feature sets with metadata.
    
    Feature sets are saved with:
    - Exchange, asset, datetime identification
    - Human-readable feature list
    - Selection process logs
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize feature set persistence manager."""
        self.base_dir = Path(base_dir) if base_dir else Path("versioned_artifacts")
    
    def _get_feature_set_dir(self, exchange: str, asset: str) -> Path:
        """Get directory for feature sets."""
        return self.base_dir / f"feature_sets_{asset}_{exchange}"
    
    def _get_feature_set_path(
        self, 
        exchange: str, 
        asset: str, 
        timestamp: Optional[str] = None
    ) -> Path:
        """Get path for feature set file."""
        fs_dir = self._get_feature_set_dir(exchange, asset)
        fs_dir.mkdir(parents=True, exist_ok=True)
        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        return fs_dir / f"feature_sets_{ts}.json"
    
    def save_feature_sets(
        self,
        feature_sets: Dict[int, List[str]],
        exchange: str,
        asset: str,
        pipeline_log: Optional[Dict[str, Any]] = None,
        winning_set_size: Optional[int] = None,
        winning_metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save feature sets with metadata.
        
        Args:
            feature_sets: Dictionary mapping feature count to feature list
            exchange: Exchange name
            asset: Asset symbol
            pipeline_log: Optional pipeline log from selection process
            winning_set_size: Size of the winning feature set
            winning_metrics: Metrics for the winning set
            
        Returns:
            Path to saved feature set file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data
        save_data = {
            "metadata": {
                "exchange": exchange,
                "asset": asset,
                "timestamp": timestamp,
                "datetime_iso": datetime.now().isoformat(),
            },
            "feature_sets": {},
            "winning_set": {
                "size": winning_set_size,
                "metrics": winning_metrics,
            } if winning_set_size else None,
            "pipeline_log": pipeline_log,
        }
        
        # Add feature sets with human-readable format
        for size, features in feature_sets.items():
            save_data["feature_sets"][str(size)] = {
                "count": len(features),
                "features": features,
                "feature_list_readable": "\n".join(f"  - {f}" for f in features),
            }
        
        # Save
        save_path = self._get_feature_set_path(exchange, asset, timestamp)
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        
        # Also save latest symlink-style pointer
        latest_path = self._get_feature_set_dir(exchange, asset) / "latest_feature_sets.json"
        with open(latest_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        
        tprint_success(f"💾 Saved feature sets to {save_path}")
        return save_path
    
    def load_feature_sets(
        self,
        exchange: str,
        asset: str,
        timestamp: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load feature sets.
        
        Args:
            exchange: Exchange name
            asset: Asset symbol
            timestamp: Optional specific timestamp (default: latest)
            
        Returns:
            Feature sets data or None if not found
        """
        if timestamp:
            load_path = self._get_feature_set_path(exchange, asset, timestamp)
        else:
            load_path = self._get_feature_set_dir(exchange, asset) / "latest_feature_sets.json"
        
        if not load_path.exists():
            tprint_warning(f"⚠️ Feature sets not found at {load_path}")
            return None
        
        try:
            with open(load_path, "r") as f:
                data = json.load(f)
            tprint_info(f"📂 Loaded feature sets from {load_path}")
            return data
        except Exception as e:
            tprint_error(f"❌ Failed to load feature sets: {e}")
            return None
    
    def get_feature_set(
        self,
        exchange: str,
        asset: str,
        size: int,
        timestamp: Optional[str] = None,
    ) -> Optional[List[str]]:
        """
        Get a specific feature set by size.
        
        Args:
            exchange: Exchange name
            asset: Asset symbol
            size: Feature set size (50, 60, 70, 80)
            timestamp: Optional specific timestamp
            
        Returns:
            List of feature names or None if not found
        """
        data = self.load_feature_sets(exchange, asset, timestamp)
        if data is None:
            return None
        
        feature_sets = data.get("feature_sets", {})
        size_key = str(size)
        
        if size_key not in feature_sets:
            available = list(feature_sets.keys())
            tprint_warning(f"⚠️ Feature set size {size} not found. Available: {available}")
            return None
        
        return feature_sets[size_key].get("features", [])
    
    def should_reselect(
        self,
        exchange: str,
        asset: str,
        force: bool = False,
    ) -> bool:
        """
        Check if feature selection should be re-run.
        
        Args:
            exchange: Exchange name
            asset: Asset symbol
            force: Force re-selection
            
        Returns:
            True if re-selection should be performed
        """
        if force:
            return True
        
        data = self.load_feature_sets(exchange, asset)
        if data is None:
            return True  # No existing feature sets
        
        # Check metadata
        metadata = data.get("metadata", {})
        saved_exchange = metadata.get("exchange")
        saved_asset = metadata.get("asset")
        
        if saved_exchange != exchange or saved_asset != asset:
            return True  # Different exchange/asset
        
        return False  # Use existing feature sets


def select_features_lgbm_for_meta_labeling(
    X: pd.DataFrame,
    y: pd.Series,
    exchange: str,
    asset: str,
    correlation_threshold: float = 0.95,
    force_reselection: bool = False,
    log_dir: Optional[Path] = None,
    persist: bool = True,
) -> Tuple[Dict[int, List[str]], Dict[str, Any]]:
    """
    Main entry point for LGBM-based feature selection for meta-labeling.
    
    This function is called by feature_generation_meta_labeling_step to:
    1. Check if feature sets exist for this exchange/asset
    2. If not (or force=True), run the full selection pipeline
    3. Return feature sets and selection log
    
    Args:
        X: Feature matrix
        y: Binary labels
        exchange: Exchange name
        asset: Asset symbol
        correlation_threshold: Threshold for correlation pruning
        force_reselection: Force re-run even if cached sets exist
        log_dir: Optional directory for logs
        persist: Whether to persist results
        
    Returns:
        Tuple of (feature_sets_dict, selection_log)
    """
    persistence = FeatureSetPersistence()
    
    # Check if we should use existing feature sets
    if not force_reselection and not persistence.should_reselect(exchange, asset):
        tprint_info(f"📂 Using existing feature sets for {asset} on {exchange}")
        data = persistence.load_feature_sets(exchange, asset)
        if data:
            feature_sets = {}
            for size_str, fs_data in data.get("feature_sets", {}).items():
                size = int(size_str)
                feature_sets[size] = fs_data.get("features", [])
            return feature_sets, data.get("pipeline_log", {})
    
    tprint_info(f"🔄 Running LGBM feature selection for {asset} on {exchange}")
    
    # Run full pipeline
    metadata = {
        "exchange": exchange,
        "asset": asset,
    }
    
    feature_sets, pipeline_log = lgbm_feature_selection_pipeline(
        X, y,
        correlation_threshold=correlation_threshold,
        target_feature_sets=FEATURE_SELECTION_CONFIG["target_feature_sets"],
        log_dir=log_dir,
        metadata=metadata,
    )
    
    # Persist results
    if persist:
        persistence.save_feature_sets(
            feature_sets,
            exchange,
            asset,
            pipeline_log,
        )
    
    return feature_sets, pipeline_log


# Backward compatibility wrapper
def select_features_by_importance_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = 60,
    correlation_threshold: float = 0.95,
) -> List[str]:
    """
    Backward-compatible wrapper for LGBM feature selection.
    
    This replaces the original RandomForest-based select_features_by_importance
    with LGBM-based selection.
    
    Args:
        X: Feature matrix
        y: Binary labels
        max_features: Maximum number of features (default 60)
        correlation_threshold: Correlation threshold
        
    Returns:
        List of selected feature names
    """
    # Use simplified single-pass selection for backward compatibility
    tprint_info(f"🔍 LGBM feature selection: Starting with {len(X.columns)} features")
    
    # Correlation pruning
    X_pruned, kept_features, _ = correlation_pruning(X, correlation_threshold)
    
    if len(kept_features) <= max_features:
        return kept_features
    
    # Train single LGBM for importance
    clean_mask = ~y.isna()
    X_clean = X_pruned[clean_mask].fillna(0)
    y_clean = y[clean_mask]
    
    if len(y_clean) < 50:
        tprint_warning(f"⚠️ Too few samples for LGBM selection")
        return kept_features[:max_features]
    
    try:
        params = _get_lgbm_params(random_state=42)
        model = lgb.LGBMClassifier(**params)
        model.fit(X_clean, y_clean)
        
        importances = model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1][:max_features]
        selected = [kept_features[i] for i in sorted_indices]
        
        tprint_info(f"   ↪ Selected top {max_features} features by LGBM importance")
        return selected
        
    except Exception as e:
        tprint_warning(f"⚠️ LGBM selection failed: {e}, using correlation-pruned features")
        return kept_features[:max_features]
