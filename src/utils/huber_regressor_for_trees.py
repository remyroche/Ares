import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import RobustScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from typing import Dict, List, Tuple, Optional, Union

def _fit_huber_split(
    X_split: pd.DataFrame,
    y_split: pd.Series,
    split_weights: Optional[np.ndarray],
    epsilons: List[float],
    alphas: List[float]
) -> np.ndarray:
    """Helper to fit a single Huber split."""
    scaler_split = RobustScaler()
    X_split_scaled = scaler_split.fit_transform(X_split)

    eps_median = np.median(epsilons)
    alpha_median = np.median(alphas)

    h_split = HuberRegressor(epsilon=eps_median, alpha=alpha_median, max_iter=5000)
    if split_weights is not None:
        h_split.fit(X_split_scaled, y_split, sample_weight=split_weights)
    else:
        h_split.fit(X_split_scaled, y_split)

    return h_split.coef_

def prepare_huber_teacher_outputs(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    vol_proxy: Optional[pd.Series] = None,
    sample_weight: Optional[np.ndarray] = None,
    epsilons: List[float] = [1.1, 1.35, 1.75],
    alphas: List[float] = [1e-4, 1e-3, 1e-2],
    pruning_percentile: int = 15,
    corr_threshold: float = 0.7,
    n_jobs: int = -1,
    sign_agree_threshold: float = 0.8,  # Same sign in ≥ 80% of splits
    nonzero_rate_threshold: float = 0.7,  # Non-zero in ≥ 70% of splits
    n_time_splits: int = 5  # Number of walk-forward time splits for stability
) -> Dict:
    """
    Advanced Huber Orchestrator for 15m Crypto Specialists.
    Includes: Vol-Weighting, Robust Scaling, and Named Interaction Constraints.
    Optimized for performance without parallelization.
    """
    # 1. Sample Weighting (De Prado alignment)
    if sample_weight is not None:
        actual_weights = np.asarray(sample_weight)
        actual_weights /= actual_weights.mean() # Normalize to preserve scale
    elif vol_proxy is not None:
        actual_weights = (1.0 / vol_proxy).fillna(vol_proxy.median()).values
        actual_weights /= actual_weights.mean() # Normalize to preserve scale
    else:
        actual_weights = None

    # 2. Robust Scaling
    scaler_full = RobustScaler()
    X_tr_scaled = scaler_full.fit_transform(X_train)
    feature_names = np.asarray(X_train.columns)
    n_features = len(feature_names)

    # 3. Walk-Forward Time Splits for Stability Analysis
    print(f"\n🔄 Creating {n_time_splits} walk-forward time splits for stability analysis...")
    
    n_samples = len(X_train)
    split_size = n_samples // n_time_splits
    
    coeffs_array = np.zeros((n_time_splits, n_features))
    
    for split_idx in range(n_time_splits):
        # Walk-forward: each split uses data up to that point
        if split_idx == n_time_splits - 1:
            train_end = n_samples
        else:
            train_end = (split_idx + 1) * split_size
        
        # Ensure minimum samples for training
        train_start = max(0, train_end - split_size * 2)
        
        X_split = X_train.iloc[train_start:train_end]
        y_split = y_train.iloc[train_start:train_end]
        
        split_weights = actual_weights[train_start:train_end] if actual_weights is not None else None
        
        print(f"   📊 Split {split_idx + 1}/{n_time_splits}: {len(X_split)} samples [{train_start}:{train_end}]")
        
        coeffs_array[split_idx] = _fit_huber_split(
            X_split, y_split, split_weights, epsilons, alphas
        )

    # 4. Consensus Logic (Median Coeffs across time splits)
    avg_coeffs = np.median(coeffs_array, axis=0)
    abs_avg_coeffs = np.abs(avg_coeffs)
    
    # Generate warm start predictions using median model on full data
    h_full = HuberRegressor(epsilon=np.median(epsilons), alpha=np.median(alphas), max_iter=5000)
    if actual_weights is not None:
        h_full.fit(X_tr_scaled, y_train, sample_weight=actual_weights)
    else:
        h_full.fit(X_tr_scaled, y_train)
    warm_start_tr = h_full.predict(X_tr_scaled)

    # 5. O(n) Feature Pruning
    kth_val = np.percentile(abs_avg_coeffs, pruning_percentile)
    keep_mask = abs_avg_coeffs > kth_val
    selected_feats = feature_names[keep_mask]
    
    # 6. Stability Analysis: Vectorized
    print(f"\n🔍 Huber Stability Analysis across {n_time_splits} time splits:")
    
    nonzero_threshold = 1e-6
    # Boolean mask of meaningful coefficients across all splits [n_splits, n_features]
    is_nonzero = np.abs(coeffs_array) > nonzero_threshold
    
    # Nonzero rate per feature
    nonzero_rate = np.mean(is_nonzero, axis=0)
    
    # Sign consensus
    median_signs = np.sign(avg_coeffs)  # [n_features]
    split_signs = np.sign(coeffs_array) # [n_splits, n_features]
    
    # Check where signs match median (broadcast median_signs)
    signs_match = (split_signs == median_signs[None, :])
    
    # Count matches where coefficient is nonzero
    # We only care about sign agreement for non-zero coefficients
    nonzero_counts = np.sum(is_nonzero, axis=0)

    # Initialize with zeros
    sign_agreement = np.zeros(n_features)

    # Avoid division by zero
    valid_mask = nonzero_counts > 0

    # Calculate agreement: sum(match & nonzero) / sum(nonzero)
    # If median is 0, agreement is 0 (handled by initialization)
    if np.any(valid_mask):
        # Where median is not zero
        nonzero_median = median_signs != 0
        mask_final = valid_mask & nonzero_median

        agreement_counts = np.sum(signs_match & is_nonzero, axis=0)
        sign_agreement[mask_final] = agreement_counts[mask_final] / nonzero_counts[mask_final]

    print(f"   📈 Average sign agreement: {np.mean(sign_agreement):.3f}")
    print(f"   📈 Average nonzero rate: {np.mean(nonzero_rate):.3f}")
    
    # 7. Enhanced Monotonicity: Vectorized
    local_scale = np.mean(abs_avg_coeffs[keep_mask])
    strength_threshold = max(0.15 * local_scale, np.percentile(abs_avg_coeffs[keep_mask], 10))
    
    # Calculate pass masks for ALL features first
    strength_pass_all = abs_avg_coeffs > strength_threshold
    stability_pass_all = (sign_agreement >= sign_agree_threshold) & (nonzero_rate >= nonzero_rate_threshold)
    
    # Filter to selected features
    strength_pass_sel = strength_pass_all[keep_mask]
    stability_pass_sel = stability_pass_all[keep_mask]
    
    # Determine constraints
    # If both pass: sign of median coefficient. Else: 0.
    mono_cst_sel = np.where(
        strength_pass_sel & stability_pass_sel,
        np.sign(avg_coeffs[keep_mask]),
        0.0
    ).astype(int)
    
    # Logging
    # We can reconstruct lists for logging if needed, or skip for speed/cleanliness
    # Here we replicate the logging summary roughly
    negative_mask = mono_cst_sel == -1
    positive_mask = mono_cst_sel == 1
    unconstrained_mask = mono_cst_sel == 0
    
    n_negative = np.sum(negative_mask)
    n_positive = np.sum(positive_mask)
    n_unconstrained = np.sum(unconstrained_mask)
    
    print(f"\n🔗 Huber Enhanced Monotonic Constraints Analysis:")
    print(f"   📊 Total features: {len(selected_feats)}")
    print(f"   🔻 Negative constraints: {n_negative}")
    print(f"   🔺 Positive constraints: {n_positive}")
    print(f"   ⚪ Unconstrained: {n_unconstrained}")
    
    # Failure analysis for unconstrained
    # A feature is unconstrained if mono_cst is 0.
    # This happens if not (strength_pass and stability_pass).
    # Specifically, we want to know why.
    unconstrained_indices = np.where(unconstrained_mask)[0]
    
    n_strength_failed = np.sum(~strength_pass_sel[unconstrained_indices] & stability_pass_sel[unconstrained_indices])
    n_stability_failed = np.sum(strength_pass_sel[unconstrained_indices] & ~stability_pass_sel[unconstrained_indices])
    n_both_failed = np.sum(~strength_pass_sel[unconstrained_indices] & ~stability_pass_sel[unconstrained_indices])

    print(f"\n📈 Constraint Failure Analysis:")
    print(f"   ❌ Strength failed: {n_strength_failed}")
    print(f"   🔄 Stability failed: {n_stability_failed}")
    print(f"   💥 Both failed: {n_both_failed}")
    
    # 8. Interaction Constraints (Named Output for Tree Learners)
    # Efficiency: Use Rank-based correlation (O(n log n)) with rankdata
    imp_mask_sub = abs_avg_coeffs[keep_mask] > np.median(abs_avg_coeffs[keep_mask])
    # imp_mask_sub is relative to keep_mask.
    # imp_feat_names = selected_feats[imp_mask_sub]
    
    interaction_constraints = []
    
    # Slice the scaled matrix to kept features, then important features
    # X_tr_scaled is [n_samples, n_features]
    # We need X_tr_scaled[:, keep_mask][:, imp_mask_sub]
    
    if np.sum(imp_mask_sub) > 1:
        X_imp = X_tr_scaled[:, keep_mask][:, imp_mask_sub]
        imp_feat_names = selected_feats[imp_mask_sub]

        # Vectorized Rank correlation using scipy rankdata
        # rankdata ranks flattened array by default, need axis=0
        X_imp_ranks = rankdata(X_imp, axis=0)
        corr_matrix = np.corrcoef(X_imp_ranks.T)
        
        D = np.clip(1 - np.abs(corr_matrix), 0, 1)
        Z = linkage(squareform(D, checks=False), method='complete')
        
        labels = fcluster(Z, corr_threshold, criterion='distance')
        
        for l in np.unique(labels):
            group = imp_feat_names[labels == l].tolist()
            interaction_constraints.append(group)
    else:
        interaction_constraints = None

    # 9. Consensus Inference
    def get_consensus_pred(df):
        if df is None: return None
        df_s = scaler_full.transform(df)
        pred = h_full.predict(df_s)
        return pred

    return {
        'selected_features': selected_feats.tolist(),
        'monotonic_constraints': dict(zip(selected_feats, mono_cst_sel)),
        'interaction_constraints': interaction_constraints,
        'warm_start': {
            'train': warm_start_tr,
            'val': get_consensus_pred(X_val),
            'test': get_consensus_pred(X_test)
        },
        'huber_models': [h_full],
        'quantile_meta_targets': y_train - warm_start_tr,
        'scaler': scaler_full
    }

# Backward compatibility alias
def prepare_huber_production_orchestrator(*args, **kwargs):
    """Deprecated alias for prepare_huber_teacher_outputs"""
    return prepare_huber_teacher_outputs(*args, **kwargs)
