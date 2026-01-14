import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import RobustScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Optional, Union

def _fit_single_split_optimized(
    X_full: np.ndarray,
    y_full: np.ndarray,
    sample_weight_full: Optional[np.ndarray],
    start_idx: int,
    end_idx: int,
    epsilons: List[float],
    alphas: List[float]
) -> np.ndarray:
    """
    Helper function to fit a single Huber Regressor split.
    Accepts full numpy arrays and indices to minimize pickling overhead.
    """
    # Slice the data (creates a view usually for numpy)
    X_split = X_full[start_idx:end_idx]
    y_split = y_full[start_idx:end_idx]

    # Scale data for this split
    # RobustScaler usually fits on the data.
    scaler_split = RobustScaler()
    X_split_scaled = scaler_split.fit_transform(X_split) # Already float32 if input is float32

    # Fit best Huber model for this split (use median parameters)
    eps_median = np.median(epsilons)
    alpha_median = np.median(alphas)

    h_split = HuberRegressor(epsilon=eps_median, alpha=alpha_median, max_iter=5000)

    if sample_weight_full is not None:
        split_weights = sample_weight_full[start_idx:end_idx]
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
    Includes: Vol-Weighting, Parallel Grid Fitting, and Named Interaction Constraints.
    """
    # 1. Sample Weighting (De Prado alignment)
    # Priority 1: Direct sample_weight (e.g. from Sequential Bootstrap)
    # Priority 2: Inverse Volatility (if vol_proxy provided)
    if sample_weight is not None:
        actual_weights = np.asarray(sample_weight)
        actual_weights /= actual_weights.mean() # Normalize to preserve scale
    elif vol_proxy is not None:
        actual_weights = (1.0 / vol_proxy).fillna(vol_proxy.median()).values
        actual_weights /= actual_weights.mean() # Normalize to preserve scale
    else:
        actual_weights = None

    # 2. Robust Scaling (NumPy-first for speed)
    # Convert to float32 numpy array immediately to reduce memory footprint and copy overhead
    X_train_np = X_train.values.astype(np.float32)
    y_train_np = y_train.values
    feature_names = np.asarray(X_train.columns)

    # 3. Walk-Forward Time Splits for Stability Analysis
    print(f"\n🔄 Creating {n_time_splits} walk-forward time splits for stability analysis (Parallel n_jobs={n_jobs})...")
    
    # Create time-based splits (walk-forward)
    n_samples = len(X_train)
    split_size = n_samples // n_time_splits
    
    tasks = []
    
    for split_idx in range(n_time_splits):
        # Walk-forward: each split uses data up to that point
        if split_idx == n_time_splits - 1:
            # Last split uses all data
            train_end = n_samples
        else:
            # Other splits use progressive portions
            train_end = (split_idx + 1) * split_size
        
        # Ensure minimum samples for training
        train_start = max(0, train_end - split_size * 2)  # Use rolling window
        
        print(f"   📊 Split {split_idx + 1}/{n_time_splits}: {train_end - train_start} samples [{train_start}:{train_end}]")
        
        # Pass full arrays and indices
        tasks.append((X_train_np, y_train_np, actual_weights, train_start, train_end, epsilons, alphas))

    # Parallel Execution
    split_coeffs = Parallel(n_jobs=n_jobs)(
        delayed(_fit_single_split_optimized)(*args) for args in tasks
    )
    
    # Convert to array for analysis
    coeffs_array = np.array(split_coeffs)  # Shape: (n_splits, n_features)

    # 4. Consensus Logic (Median Coeffs across time splits)
    avg_coeffs = np.median(coeffs_array, axis=0)
    abs_avg_coeffs = np.abs(avg_coeffs)
    
    # Generate warm start predictions using median model on full data
    scaler_full = RobustScaler()
    X_full_scaled = scaler_full.fit_transform(X_train_np) # float32

    h_full = HuberRegressor(epsilon=np.median(epsilons), alpha=np.median(alphas), max_iter=5000)
    if actual_weights is not None:
        h_full.fit(X_full_scaled, y_train, sample_weight=actual_weights)
    else:
        h_full.fit(X_full_scaled, y_train)
    warm_start_tr = h_full.predict(X_full_scaled)

    # 5. O(n) Feature Pruning
    kth_val = np.percentile(abs_avg_coeffs, pruning_percentile)
    keep_mask = abs_avg_coeffs > kth_val
    selected_feats = feature_names[keep_mask]
    
    # 6. Stability Analysis: Sign Consensus and Nonzero Rate across Time Splits
    print(f"\n🔍 Huber Stability Analysis across {n_time_splits} time splits:")
    
    # Calculate stability metrics for each feature
    # n_splits = len(coeffs_array)
    
    # Sign consensus: proportion of splits with same sign as median
    median_signs = np.sign(avg_coeffs)
    sign_agreement = np.zeros(len(feature_names))
    
    # Nonzero rate: proportion of splits with meaningful coefficients
    nonzero_threshold = 1e-6  # Threshold for "meaningfully non-zero"
    nonzero_rate = np.zeros(len(feature_names))
    
    for j, feat_name in enumerate(feature_names):
        feat_coeffs = coeffs_array[:, j]
        
        # Sign consensus (exclude zeros from sign calculation)
        nonzero_mask = np.abs(feat_coeffs) > nonzero_threshold
        if np.sum(nonzero_mask) > 0:
            nonzero_signs = np.sign(feat_coeffs[nonzero_mask])
            if median_signs[j] != 0:
                sign_agreement[j] = np.mean(nonzero_signs == median_signs[j])
            else:
                sign_agreement[j] = 0.0  # No consensus if median is zero
        else:
            sign_agreement[j] = 0.0
        
        # Nonzero rate
        nonzero_rate[j] = np.mean(nonzero_mask)
        
        # Debug logging for a few features
        if j < 5:  # Log first 5 features
            print(f"   📊 {feat_name}: sign_agree={sign_agreement[j]:.2f}, nonzero_rate={nonzero_rate[j]:.2f}, median_coeff={avg_coeffs[j]:.4f}")
    
    print(f"   📈 Average sign agreement: {np.mean(sign_agreement):.3f}")
    print(f"   📈 Average nonzero rate: {np.mean(nonzero_rate):.3f}")
    
    # 7. Enhanced Monotonicity: Strength + Stability Criteria
    local_scale = np.mean(abs_avg_coeffs[keep_mask])
    strength_threshold = max(0.15 * local_scale, np.percentile(abs_avg_coeffs[keep_mask], 10))
    
    # Stability thresholds (configurable parameters)
    # sign_agree_threshold = 0.8  # Same sign in ≥ 80% of splits
    # nonzero_rate_threshold = 0.7  # Non-zero in ≥ 70% of splits
    
    # Apply constraints only when both strength and stability pass
    mono_cst = np.zeros(len(selected_feats))  # Default: unconstrained
    
    for i, (feat_idx, feat_name) in enumerate(zip(np.where(keep_mask)[0], selected_feats)):
        # Strength criterion
        strength_pass = abs_avg_coeffs[feat_idx] > strength_threshold
        
        # Stability criteria
        stability_pass = (sign_agreement[feat_idx] >= sign_agree_threshold and 
                         nonzero_rate[feat_idx] >= nonzero_rate_threshold)
        
        # Apply constraint only if both pass
        if strength_pass and stability_pass:
            mono_cst[i] = np.sign(avg_coeffs[feat_idx])
        else:
            mono_cst[i] = 0  # Unconstrained
    
    # Enhanced logging: Show which features have which constraints
    negative_features = []
    positive_features = []
    unconstrained_features = []
    
    # Also track why features were unconstrained
    strength_failed = []
    stability_failed = []
    both_failed = []
    
    for i, (feat_name, constraint) in enumerate(zip(selected_feats, mono_cst)):
        feat_idx = np.where(keep_mask)[0][i]
        
        if constraint == -1:
            negative_features.append(feat_name)
        elif constraint == 1:
            positive_features.append(feat_name)
        else:  # constraint == 0
            unconstrained_features.append(feat_name)
            
            # Track failure reason
            strength_pass = abs_avg_coeffs[feat_idx] > strength_threshold
            stability_pass = (sign_agreement[feat_idx] >= sign_agree_threshold and 
                             nonzero_rate[feat_idx] >= nonzero_rate_threshold)
            
            if not strength_pass and not stability_pass:
                both_failed.append(feat_name)
            elif not strength_pass:
                strength_failed.append(feat_name)
            else:
                stability_failed.append(feat_name)
    
    # Print detailed constraint information
    print(f"\n🔗 Huber Enhanced Monotonic Constraints Analysis:")
    print(f"   📊 Total features: {len(selected_feats)}")
    print(f"   🔻 Negative constraints: {len(negative_features)}")
    print(f"   🔺 Positive constraints: {len(positive_features)}")
    print(f"   ⚪ Unconstrained: {len(unconstrained_features)}")
    
    # Summary: Candidates by magnitude → Constraints after stability gate
    magnitude_candidates = len(selected_feats)  # Features that passed magnitude threshold
    stability_constraints = len(negative_features) + len(positive_features)  # Features that passed stability
    dropped_instability = magnitude_candidates - stability_constraints  # Dropped due to instability
    
    print(f"\n📋 Constraint Summary:")
    print(f"   🎯 Candidates by magnitude: {magnitude_candidates}")
    print(f"   ✅ Constraints after stability gate: {stability_constraints}")
    print(f"   ❌ Dropped due to instability: {dropped_instability}")
    print(f"   📊 Stability retention rate: {stability_constraints/magnitude_candidates*100:.1f}%")
    
    # Print failure breakdown
    print(f"\n📈 Constraint Failure Analysis:")
    print(f"   ❌ Strength failed: {len(strength_failed)}")
    print(f"   🔄 Stability failed: {len(stability_failed)}")
    print(f"   💥 Both failed: {len(both_failed)}")
    
    # Print stability statistics
    print(f"\n📊 Stability Statistics (Thresholds: sign≥{sign_agree_threshold}, nonzero≥{nonzero_rate_threshold}):")
    print(f"   🔄 Based on {n_time_splits} walk-forward time splits")
    constrained_features = negative_features + positive_features
    if constrained_features:
        constrained_sign_agree = [sign_agreement[np.where(feature_names == feat)[0][0]] 
                                for feat in constrained_features]
        constrained_nonzero_rate = [nonzero_rate[np.where(feature_names == feat)[0][0]] 
                                  for feat in constrained_features]
        print(f"   ✅ Constrained features:")
        print(f"      - Avg sign agreement: {np.mean(constrained_sign_agree):.3f}")
        print(f"      - Avg nonzero rate: {np.mean(constrained_nonzero_rate):.3f}")
    
    if unconstrained_features:
        unconstrained_sign_agree = [sign_agreement[np.where(feature_names == feat)[0][0]] 
                                  for feat in unconstrained_features]
        unconstrained_nonzero_rate = [nonzero_rate[np.where(feature_names == feat)[0][0]] 
                                    for feat in unconstrained_features]
        print(f"   ⚪ Unconstrained features:")
        print(f"      - Avg sign agreement: {np.mean(unconstrained_sign_agree):.3f}")
        print(f"      - Avg nonzero rate: {np.mean(unconstrained_nonzero_rate):.3f}")
    
    if negative_features:
        print(f"   🔻 Negative features: {negative_features[:5]}{'...' if len(negative_features) > 5 else ''}")
    if positive_features:
        print(f"   🔺 Positive features: {positive_features[:5]}{'...' if len(positive_features) > 5 else ''}")
    if unconstrained_features:
        print(f"   ⚪ Unconstrained features: {unconstrained_features[:5]}{'...' if len(unconstrained_features) > 5 else ''}")
    
    if strength_failed:
        print(f"   ❌ Strength failed: {strength_failed[:3]}{'...' if len(strength_failed) > 3 else ''}")
    if stability_failed:
        print(f"   🔄 Stability failed: {stability_failed[:3]}{'...' if len(stability_failed) > 3 else ''}")
    if both_failed:
        print(f"   💥 Both failed: {both_failed[:3]}{'...' if len(both_failed) > 3 else ''}")

    # 7. Interaction Constraints (Named Output for Tree Learners)
    # Efficiency: Use Rank-based correlation (O(n log n))
    imp_mask = abs_avg_coeffs[keep_mask] > np.median(abs_avg_coeffs[keep_mask])
    imp_feat_names = selected_feats[imp_mask]
    
    interaction_constraints = []
    if imp_feat_names.size > 1:
        # Optimized Rank correlation: Use scipy rankdata on numpy array
        # X_tr_scaled is already numpy float32.
        # Subset features first
        X_subset = X_train_np[:, keep_mask][:, imp_mask]

        # rankdata computes rank along axis. We want rank of each feature (column) across samples.
        # axis=0.
        X_imp_ranks = rankdata(X_subset, axis=0)

        # Compute correlation on ranks (Spearman)
        corr_matrix = np.corrcoef(X_imp_ranks.T)
        
        D = np.clip(1 - np.abs(corr_matrix), 0, 1)
        # squareform requires 1D condensed distance matrix or 2D square. D is 2D square.
        # checks=False skips symmetry check for speed.
        Z = linkage(squareform(D, checks=False), method='complete')
        
        labels = fcluster(Z, corr_threshold, criterion='distance')
        
        # Save as feature names to prevent indexing breaks in HPO
        for l in np.unique(labels):
            group = imp_feat_names[labels == l].tolist()
            interaction_constraints.append(group)
    else:
        interaction_constraints = None

    # 8. Consensus Inference for Validation/Test
    def get_consensus_pred(df):
        if df is None: return None
        # Use full scaler on dataframe
        df_s = scaler_full.transform(df).astype(np.float32)
        pred = h_full.predict(df_s)  # Use the median model
        return pred

    return {
        'selected_features': selected_feats.tolist(),
        'monotonic_constraints': dict(zip(selected_feats, mono_cst.astype(int))),
        'interaction_constraints': interaction_constraints,
        'warm_start': {
            'train': warm_start_tr,
            'val': get_consensus_pred(X_val),
            'test': get_consensus_pred(X_test)
        },
        'huber_models': [h_full], # For future inspection and prediction (median model)
        'quantile_meta_targets': y_train - warm_start_tr,
        'scaler': scaler_full
    }

# Backward compatibility alias
def prepare_huber_production_orchestrator(*args, **kwargs):
    """Deprecated alias for prepare_huber_teacher_outputs"""
    return prepare_huber_teacher_outputs(*args, **kwargs)
