import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import RobustScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

@dataclass(frozen=True)
class ConstraintSelectionConfig:
    # Eligibility (Full-mode default)
    p_min: float = 0.80          # sign-stability gate
    nz_min: float = 0.00         # optional nonzero-rate gate (set >0 to enforce)

    # Band (your updated choice)
    band_min: float = 0.20       # minimum constrained fraction (of eligible pool)
    band_max: float = 0.60       # maximum constrained fraction (of eligible pool)

    # Adaptive-quantile mapping (within eligible pool)
    # q is the quantile threshold on Q: select Q >= quantile(Q, q)
    q_low: float = 0.40          # when quality is high, constrain more (lower threshold)
    q_high: float = 0.80         # when quality is low, constrain fewer (higher threshold)

    # How to compute dataset quality health
    top_frac_for_H: float = 0.10 # "top 10%" for top-heaviness H
    H_ref: float = 2.0           # reference top-heaviness where we start tightening
    H_strength: float = 0.10     # how strongly H pushes q upward (0.0 disables)

    # Quality score details
    lambda_econ: float = 0.0     # optional economic prior boost (0 disables)
    eps: float = 1e-12           # numerical stability

    # Behavior controls
    tie_breaker: str = "stable"  # "stable" or "magnitude": how to break ties near cutoff


def _safe_quantile(x: np.ndarray, q: float) -> float:
    """Robust quantile helper for possibly empty arrays."""
    if x.size == 0:
        return np.nan
    q = float(np.clip(q, 0.0, 1.0))
    return float(np.quantile(x, q))


def compute_constraint_quality(
    a: np.ndarray,              # |avg_coef| or other strength proxy, shape (N,)
    p: np.ndarray,              # sign stability in [0,1], shape (N,)
    nz: np.ndarray,             # nonzero rate in [0,1], shape (N,)
    e: Optional[np.ndarray],    # economic prior flag in {0,1}, shape (N,)
    cfg: ConstraintSelectionConfig,
) -> np.ndarray:
    """
    Q_j = a_j * (2 p_j - 1) * nz_j * (1 + lambda * e_j)
    - a: must be comparable across features (ensure consistent scaling upstream).
    """
    a = np.asarray(a, dtype=float)
    p = np.asarray(p, dtype=float)
    nz = np.asarray(nz, dtype=float)

    if e is None:
        econ = 1.0
    else:
        e = np.asarray(e, dtype=float)
        econ = 1.0 + cfg.lambda_econ * e

    stability_penalty = np.clip(2.0 * p - 1.0, 0.0, 1.0)  # maps 0.5->0, 1.0->1
    Q = a * stability_penalty * np.clip(nz, 0.0, 1.0) * econ
    return Q


def compute_quality_health(Q_eligible: np.ndarray, p_eligible: np.ndarray, cfg: ConstraintSelectionConfig) -> Dict[str, float]:
    """
    - S: stability prevalence among all features (or eligible subset if provided)
    - H: top-heaviness ratio over eligible Q
    """
    Qe = np.asarray(Q_eligible, dtype=float)
    pe = np.asarray(p_eligible, dtype=float)

    # S computed on the same population passed in
    S = float(np.mean(pe >= cfg.p_min)) if pe.size else 0.0

    # H = mean(top X%) / mean(all)
    if Qe.size == 0:
        H = 0.0
    else:
        mean_all = float(np.mean(Qe))
        if mean_all <= cfg.eps:
            H = 0.0
        else:
            k = max(1, int(np.ceil(cfg.top_frac_for_H * Qe.size)))
            top_vals = np.sort(Qe)[-k:]
            H = float(np.mean(top_vals) / (mean_all + cfg.eps))

    return {"S": S, "H": H}


def adaptive_quantile_from_health(S: float, H: float, cfg: ConstraintSelectionConfig) -> float:
    """
    Adaptive quantile q (within eligible pool), primarily driven by S.
    - Higher S => lower q (constrain more)
    - Lower S  => higher q (constrain fewer)
    Optionally: if H is high (quality concentrated), push q upward.
    """
    # Map S in [0,1] to q in [q_low, q_high] with inverse relation.
    # q = q_high - S * (q_high - q_low)
    q = cfg.q_high - float(np.clip(S, 0.0, 1.0)) * (cfg.q_high - cfg.q_low)

    # Optional H adjustment: if quality is very concentrated (H > H_ref), constrain fewer.
    if cfg.H_strength > 0.0 and cfg.H_ref > 0.0:
        excess = max(0.0, H - cfg.H_ref)
        q += cfg.H_strength * excess

    return float(np.clip(q, 0.0, 1.0))


def select_monotonic_constraints(
    a: np.ndarray,
    p: np.ndarray,
    nz: np.ndarray,
    sign: np.ndarray,                # dominant sign per feature: -1, 0, +1 (0 => unconstrained)
    feature_names: Optional[np.ndarray] = None,
    e: Optional[np.ndarray] = None,
    cfg: ConstraintSelectionConfig = ConstraintSelectionConfig(),
) -> Dict[str, Any]:
    """
    Returns a monotonic constraint vector (same order as inputs), plus diagnostics.

    Band policy (your update):
    - Apply adaptive quantile cutoff within eligible features.
    - Clamp constrained fraction to [band_min, band_max] of eligible pool.
    - Never force constraints on unstable / ineligible features.

    Eligibility:
    - p >= p_min
    - nz >= nz_min (optional)
    - sign != 0

    Selection:
    - Rank by Q within eligible.
    - Choose top-K where K is clamped to [ceil(band_min*Ne), floor(band_max*Ne)].
    """
    a = np.asarray(a, dtype=float)
    p = np.asarray(p, dtype=float)
    nz = np.asarray(nz, dtype=float)
    sgn = np.asarray(sign, dtype=int)

    N = a.size
    if feature_names is None:
        feature_names = np.array([f"f{i}" for i in range(N)], dtype=object)
    else:
        feature_names = np.asarray(feature_names, dtype=object)

    Q = compute_constraint_quality(a=a, p=p, nz=nz, e=e, cfg=cfg)

    eligible = (p >= cfg.p_min) & (nz >= cfg.nz_min) & (sgn != 0)
    idx_eligible = np.where(eligible)[0]
    Ne = idx_eligible.size

    # Default: no constraints
    mono = np.zeros(N, dtype=int)

    # If nothing eligible, return early with diagnostics
    if Ne == 0:
        return {
            "monotonic_constraints": mono,
            "selected_features": [],
            "selected_fraction_of_eligible": 0.0,
            "eligible_count": 0,
            "total_count": N,
            "q_adaptive": np.nan,
            "threshold_Q": np.nan,
            "health": {"S": float(np.mean(p >= cfg.p_min)), "H": 0.0},
            "Q": Q,
            "eligible_mask": eligible,
            "band": {"min": cfg.band_min, "max": cfg.band_max, "K_min": 0, "K_max": 0, "K": 0},
        }

    Qe = Q[idx_eligible]
    pe = p[idx_eligible]

    health = compute_quality_health(Q_eligible=Qe, p_eligible=pe, cfg=cfg)
    q_adaptive = adaptive_quantile_from_health(S=health["S"], H=health["H"], cfg=cfg)
    threshold_Q = _safe_quantile(Qe, q_adaptive)

    # Initial pick based on quantile threshold
    pick = idx_eligible[Qe >= threshold_Q]

    # Convert to top-K selection to enforce band robustly
    K_min = int(np.ceil(cfg.band_min * Ne))
    K_max = int(np.floor(cfg.band_max * Ne))
    K_min = max(0, min(K_min, Ne))
    K_max = max(0, min(K_max, Ne))
    if K_max < K_min:
        # If band is inconsistent due to small Ne, collapse to feasible
        K_max = K_min

    # Rank eligible by Q descending; optional tie breaker
    order = np.argsort(-Qe)  # descending
    ranked = idx_eligible[order]

    # Determine K0 from the quantile-based pick
    K0 = pick.size

    # Clamp K to band
    K = int(np.clip(K0, K_min, K_max))

    # If quantile pick produced too few/many, use top-K directly
    selected_idx = ranked[:K]

    # Apply constraints with provided signs
    mono[selected_idx] = sgn[selected_idx]

    selected_features = [(str(feature_names[i]), int(mono[i]), float(Q[i])) for i in selected_idx]
    selected_fraction = (K / Ne) if Ne else 0.0

    return {
        "monotonic_constraints": mono,                  # int vector: -1, 0, +1
        "selected_features": selected_features,          # (name, direction, Q) tuples
        "selected_fraction_of_eligible": selected_fraction,
        "eligible_count": Ne,
        "total_count": N,
        "q_adaptive": q_adaptive,
        "threshold_Q": threshold_Q,
        "health": health,
        "Q": Q,
        "eligible_mask": eligible,
        "band": {"min": cfg.band_min, "max": cfg.band_max, "K_min": K_min, "K_max": K_max, "K": K},
    }

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
    
    # 7. Enhanced Monotonicity: Adaptive Thresholds
    print(f"\n🔗 Huber Enhanced Monotonic Constraints Analysis (Adaptive):")
    
    # Extract metrics for the selected (pruned) features
    indices_kept = np.where(keep_mask)[0]
    
    a_kept = abs_avg_coeffs[indices_kept]
    p_kept = sign_agreement[indices_kept]
    nz_kept = nonzero_rate[indices_kept]
    sign_kept = np.sign(avg_coeffs[indices_kept])
    
    # Configure selection
    # Map function arguments to config
    sel_cfg = ConstraintSelectionConfig(
        p_min=sign_agree_threshold,
        nz_min=nonzero_rate_threshold
        # default band_min=0.20, band_max=0.60, etc.
    )
    
    selection_result = select_monotonic_constraints(
        a=a_kept,
        p=p_kept,
        nz=nz_kept,
        sign=sign_kept,
        feature_names=selected_feats,
        cfg=sel_cfg
    )
    
    mono_cst = selection_result["monotonic_constraints"]
    
    # Logging
    print(f"   📊 Total features: {selection_result['total_count']}")
    print(f"   ✅ Eligible features: {selection_result['eligible_count']}")
    print(f"   🎯 Target Band: {selection_result['band']['min']*100:.0f}% - {selection_result['band']['max']*100:.0f}%")
    print(f"   📏 Adaptive Quantile (q): {selection_result['q_adaptive']:.3f} (Threshold Q: {selection_result['threshold_Q']:.4f})")
    print(f"   🏥 Quality Health: S={selection_result['health']['S']:.3f}, H={selection_result['health']['H']:.3f}")

    selected_feats_info = selection_result["selected_features"]
    negative_features = [name for name, d, q in selected_feats_info if d == -1]
    positive_features = [name for name, d, q in selected_feats_info if d == 1]
    
    print(f"   🔻 Negative constraints: {len(negative_features)}")
    print(f"   🔺 Positive constraints: {len(positive_features)}")
    print(f"   ⚪ Unconstrained: {selection_result['total_count'] - len(negative_features) - len(positive_features)}")
    
    if negative_features:
        print(f"   🔻 Negative features: {negative_features[:5]}{'...' if len(negative_features) > 5 else ''}")
    if positive_features:
        print(f"   🔺 Positive features: {positive_features[:5]}{'...' if len(positive_features) > 5 else ''}")

    # 8. Interaction Constraints (Named Output for Tree Learners)
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

    # 9. Consensus Inference for Validation/Test
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
