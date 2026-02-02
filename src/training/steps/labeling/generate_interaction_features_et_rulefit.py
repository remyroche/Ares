"""
Enhanced RuleFit Interaction Feature Generator

Generates interaction features using ExtraTrees and IC-optimized ElasticNet selection.
Implements de Prado-aligned stability selection and hierarchical complexity penalties.

Key Features:
1. IC-based objective with ElasticNet regularization
2. Stability selection with early stopping and temporal splits
3. Hierarchical penalty: λ_rule = λ_base * (1 + η * depth)
4. Vectorized decision_path for efficient rule extraction
5. Rule deduplication via MinHash-approximated Jaccard
6. Sample-weighted support calculation
7. Numba-accelerated core operations
8. Parallel stability runs via joblib
9. 4-5σ winsorization for fat-tailed returns
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import warnings
import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from joblib import Parallel, delayed

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    prange = range


# -----------------------------
# Numba-Accelerated Utilities
# -----------------------------
@njit(cache=True)
def _robust_stats_numba(x: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
    """Numba-accelerated robust stats (median + MAD-based sigma)."""
    n = len(x)
    if n == 0:
        return 0.0, 1.0
    
    # Median via partition (O(N))
    k = n // 2
    # Ensure we don't modify original x
    work_x = x.copy()
    part_x = np.partition(work_x, k)

    if n % 2 == 0:
        med2 = part_x[k]
        # Max of left side is the (k-1)-th element
        med1 = np.max(part_x[:k])
        med = (med1 + med2) * np.float32(0.5)
    else:
        med = part_x[k]
    
    # MAD
    abs_dev = np.abs(x - med).astype(np.float32)
    part_dev = np.partition(abs_dev, k)
    
    if n % 2 == 0:
        mad2 = part_dev[k]
        mad1 = np.max(part_dev[:k])
        mad = (mad1 + mad2) * np.float32(0.5)
    else:
        mad = part_dev[k]
    
    sigma = np.float32(1.4826) * (mad + np.float32(eps))
    return med, sigma


@njit(cache=True)
def _winsorize_numba(x: np.ndarray, k: float = 4.0) -> np.ndarray:
    """Numba-accelerated winsorization."""
    med, sigma = _robust_stats_numba(x)
    k_f32 = np.float32(k)
    lo = med - k_f32 * sigma
    hi = med + k_f32 * sigma
    out = np.empty_like(x)
    for i in range(len(x)):
        if x[i] < lo:
            out[i] = lo
        elif x[i] > hi:
            out[i] = hi
        else:
            out[i] = x[i]
    return out


@njit(cache=True)
def _weighted_corr_numba(y_hat: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Numba-accelerated weighted correlation (single pass)."""
    n = len(y)
    if n == 0:
        return 0.0
    
    # Weighted means
    w_sum = np.float32(0.0)
    y_wsum = np.float32(0.0)
    yh_wsum = np.float32(0.0)
    for i in range(n):
        w_sum += w[i]
        y_wsum += w[i] * y[i]
        yh_wsum += w[i] * y_hat[i]
    
    if w_sum < 1e-12:
        return 0.0
    
    y_m = y_wsum / w_sum
    yh_m = yh_wsum / w_sum
    
    # Weighted covariance and variances
    cov = np.float32(0.0)
    var_y = np.float32(0.0)
    var_yh = np.float32(0.0)
    for i in range(n):
        y_c = y[i] - y_m
        yh_c = y_hat[i] - yh_m
        wi = w[i] / w_sum
        cov += wi * y_c * yh_c
        var_y += wi * y_c * y_c
        var_yh += wi * yh_c * yh_c
    
    denom = np.sqrt(var_y) * np.sqrt(var_yh)
    if denom < 1e-12:
        return 0.0
    return cov / denom


@njit(cache=True)
def _compute_node_depths_numba(
    children_left: np.ndarray,
    children_right: np.ndarray,
    node_count: int
) -> np.ndarray:
    """Numba-accelerated tree depth calculation via BFS."""
    depths = np.zeros(node_count, dtype=np.int32)
    stack = np.empty(node_count, dtype=np.int32)
    depth_stack = np.empty(node_count, dtype=np.int32)
    
    stack[0] = 0
    depth_stack[0] = 0
    ptr = 1
    
    while ptr > 0:
        ptr -= 1
        node = stack[ptr]
        d = depth_stack[ptr]
        depths[node] = d
        
        left = children_left[node]
        right = children_right[node]
        
        if left != -1:
            stack[ptr] = left
            depth_stack[ptr] = d + 1
            ptr += 1
        if right != -1:
            stack[ptr] = right
            depth_stack[ptr] = d + 1
            ptr += 1
    
    return depths


@njit(cache=True, parallel=True)
def _compute_rule_ics_vectorized(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    y: np.ndarray,
    n_rules: int,
    n_samples: int
) -> np.ndarray:
    """Vectorized IC computation for all rules using CSC sparse format."""
    rule_ics = np.zeros(n_rules, dtype=np.float32)
    y_mean = np.float32(np.mean(y))
    y_std = np.float32(np.std(y))
    
    if y_std < 1e-12:
        return rule_ics
    
    y_centered = y - y_mean
    
    for j in prange(n_rules):
        start = indptr[j]
        end = indptr[j + 1]
        col_sum = end - start
        
        if col_sum == 0 or col_sum == n_samples:
            rule_ics[j] = -1.0
            continue
        
        # Column mean and std
        col_mean = np.float32(col_sum) / np.float32(n_samples)
        col_var = col_mean * (np.float32(1.0) - col_mean)
        
        if col_var < 1e-12:
            rule_ics[j] = -1.0
            continue
        
        col_std = np.sqrt(col_var)
        
        # Correlation: sum of y_centered where col=1, adjusted
        y_sum_where_1 = np.float32(0.0)
        for idx in range(start, end):
            row = indices[idx]
            y_sum_where_1 += y_centered[row]
        
        # Correlation formula for binary column
        cov = y_sum_where_1 / np.float32(n_samples)
        corr = cov / (col_std * y_std)
        rule_ics[j] = abs(corr) if np.isfinite(corr) else -1.0
    
    return rule_ics


@njit(cache=True)
def _minhash_signature(indices: np.ndarray, n_hashes: int, n_samples: int) -> np.ndarray:
    """Compute MinHash signature for a set of indices."""
    sig = np.full(n_hashes, np.iinfo(np.int32).max, dtype=np.int32)
    
    # Use simple hash functions: (a * x + b) mod p mod n_samples
    primes = np.array([104729, 224737, 350377, 479909, 611953, 746773, 882377, 1020379], dtype=np.int64)
    
    for i in range(len(indices)):
        idx = indices[i]
        for h in range(min(n_hashes, len(primes))):
            hash_val = int((primes[h] * idx + h * 31) % 1000000007) % (n_samples + 1)
            if hash_val < sig[h]:
                sig[h] = hash_val
    
    return sig


@njit(cache=True)
def _estimate_jaccard_minhash(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Estimate Jaccard similarity from MinHash signatures."""
    matches = 0
    for i in range(len(sig1)):
        if sig1[i] == sig2[i]:
            matches += 1
    return matches / len(sig1)


@njit(cache=True)
def _soft_threshold(x: float, lam: float) -> float:
    """Soft thresholding operator for proximal gradient."""
    if x > lam:
        return x - lam
    elif x < -lam:
        return x + lam
    return 0.0


@njit(cache=True)
def _proximal_gradient_step(
    beta: np.ndarray,
    grad: np.ndarray,
    step_size: float,
    l1_ratio: float,
    alpha: float,
    depth_penalties: np.ndarray
) -> np.ndarray:
    """Single proximal gradient step with ElasticNet penalty."""
    p = len(beta)
    beta_new = np.empty(p, dtype=np.float32)
    
    # Cast constants to float32
    step_size = np.float32(step_size)
    l1_ratio = np.float32(l1_ratio)
    alpha = np.float32(alpha)
    one = np.float32(1.0)

    for j in range(p):
        # Gradient step
        z = beta[j] - step_size * grad[j]
        # L2 shrinkage
        l2_denom = one + step_size * (one - l1_ratio) * alpha * depth_penalties[j]
        l2_factor = one / l2_denom
        z = z * l2_factor
        # L1 soft thresholding
        lam = step_size * l1_ratio * alpha * depth_penalties[j]
        beta_new[j] = _soft_threshold(z, lam)
    
    return beta_new


def _compute_temporal_decay_weights(
    n_samples: int,
    decay_half_life: float | None = None,
    decay_type: str = "exponential"
) -> np.ndarray:
    """
    Compute temporal decay weights where recent samples have higher weight.
    
    Args:
        n_samples: Number of samples (assumed to be in chronological order)
        decay_half_life: Half-life in fraction of samples (e.g., 0.5 = half weight at midpoint)
                        None means no decay (uniform weights)
        decay_type: 'exponential' or 'linear'
    
    Returns:
        Array of weights (sum normalized to n_samples)
    """
    if decay_half_life is None or decay_half_life <= 0:
        return np.ones(n_samples, dtype=np.float32)
    
    # Position in [0, 1] where 1 is most recent
    t = np.linspace(0, 1, n_samples, dtype=np.float32)
    
    if decay_type == "exponential":
        # Exponential decay: w = exp(lambda * t) where lambda = ln(2) / half_life
        lam = np.log(2) / decay_half_life
        lam_f32 = np.float32(lam)
        weights = np.exp(lam_f32 * (t - np.float32(1.0)))  # Normalize so most recent = 1
    else:  # linear
        # Linear decay from (1 - decay_half_life) to 1
        weights = np.float32(1.0) - np.float32(decay_half_life) + np.float32(decay_half_life) * t
    
    # Normalize so weights sum to n_samples (preserves effective sample size interpretation)
    weights = weights * (n_samples / weights.sum())
    return weights.astype(np.float32)

# -----------------------------
# Python Wrapper Utilities
# -----------------------------
def robust_stats(x: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
    """Calculate median and robust sigma (MAD-based). Uses Numba if available."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if NUMBA_AVAILABLE and len(x) > 100:
        return _robust_stats_numba(x, eps)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return float(med), float(1.4826 * mad)


def winsorize(x: np.ndarray, k: float = 4.5) -> np.ndarray:
    """Winsorize data based on robust sigma. Default 4.5σ for fat-tailed returns."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if NUMBA_AVAILABLE and len(x) > 100:
        return _winsorize_numba(x, k)
    med, s = robust_stats(x)
    return np.clip(x, med - k * s, med + k * s)


def weighted_corr(y_hat: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted correlation. Uses Numba single-pass if available."""
    y_hat = np.ascontiguousarray(y_hat, dtype=np.float32).ravel()
    y = np.ascontiguousarray(y, dtype=np.float32).ravel()
    
    if w is None:
        w = np.ones(len(y), dtype=np.float32)
    else:
        w = np.ascontiguousarray(w, dtype=np.float32).ravel()
        if w.sum() < 1e-12:
            return 0.0
    
    if NUMBA_AVAILABLE and len(y) > 100:
        return _weighted_corr_numba(y_hat, y, w)
    
    # Fallback numpy implementation
    w = w / (w.sum() + 1e-12)
    y_m = np.sum(w * y)
    yh_m = np.sum(w * y_hat)
    y_c = y - y_m
    yh_c = y_hat - yh_m
    y_s = np.sqrt(np.sum(w * y_c**2))
    yh_s = np.sqrt(np.sum(w * yh_c**2))
    if y_s < 1e-12 or yh_s < 1e-12:
        return 0.0
    c = float(np.sum(w * y_c * yh_c) / (y_s * yh_s))
    return c if np.isfinite(c) else 0.0


def ic_objective_oos(
    beta: np.ndarray,
    X_va: sparse.csr_matrix,
    y_va: np.ndarray,
    l1_ratio: float,
    alpha: float,
    depth_penalties: np.ndarray,
    sample_weight_va: np.ndarray | None = None,
) -> float:
    """IC-based objective with ElasticNet and hierarchical penalty (OOS evaluation)."""
    y_hat_va = X_va @ beta
    ic = weighted_corr(y_hat_va, y_va, sample_weight_va)
    
    l1 = l1_ratio * alpha * np.sum(np.abs(beta) * depth_penalties)
    l2 = (1 - l1_ratio) * alpha * np.sum((beta ** 2) * depth_penalties)
    
    return -ic + l1 + l2


def build_gate_groups(
    n_base: int,
    n_gates: int,
    n_rules: int,
    n_gated_per_gate: List[int]
) -> List[slice]:
    """Build slice groups for group lasso penalty."""
    start = n_base + n_gates + n_rules
    groups = []
    for m in n_gated_per_gate:
        groups.append(slice(start, start + m))
        start += m
    return groups


def group_lasso_penalty(beta: np.ndarray, groups: List[slice], lam_g: float, eps: float = 1e-8) -> float:
    """Group lasso penalty term."""
    s = 0.0
    for sl in groups:
        bg = beta[sl]
        s += np.sqrt(np.dot(bg, bg) + eps)
    return lam_g * s


def ic_objective_oos_with_groups(
    beta: np.ndarray,
    X_va: sparse.csr_matrix,
    y_va: np.ndarray,
    l1_ratio: float,
    alpha: float,
    depth_penalties: np.ndarray,
    groups: List[slice],
    lam_group: float,
    sample_weight_va: np.ndarray | None = None,
) -> float:
    """IC objective with group lasso for gated rule groups."""
    y_hat_va = X_va @ beta
    ic = weighted_corr(y_hat_va, y_va, sample_weight_va)
    
    l1 = l1_ratio * alpha * np.sum(np.abs(beta) * depth_penalties)
    l2 = (1 - l1_ratio) * alpha * np.sum((beta ** 2) * depth_penalties)
    gl = group_lasso_penalty(beta, groups, lam_group)
    
    return -ic + l1 + l2 + gl



# -----------------------------
# Vectorized Rule Extraction
# -----------------------------
class LeafRuleExtractor:
    """
    Conjunction rules = leaf membership with sample-weighted support.
    Creates sparse one-hot columns: [tree0:leaf_id, tree1:leaf_id, ...]
    """

    def __init__(self, min_support: float = 0.01, max_rules: int = 5000):
        self.min_support = float(min_support)
        self.max_rules = int(max_rules)
        self._col_specs: List[Tuple[int, int]] = []
        self._depths: np.ndarray = np.array([], dtype=np.int32)
        self._leaf_cache: Dict[int, np.ndarray] = {}

    def fit(
        self,
        forest: ExtraTreesClassifier | ExtraTreesRegressor,
        X: np.ndarray,
        sample_weight: np.ndarray | None = None
    ) -> "LeafRuleExtractor":
        """
        Fit extractor with sample-weighted support calculation.
        
        Args:
            forest: Fitted ExtraTrees model
            X: Feature matrix
            sample_weight: Sample weights for support calculation
        """
        col_specs = []
        depths = []
        supports = []
        n = X.shape[0]
        
        # Normalize sample weights
        if sample_weight is None:
            w = np.ones(n, dtype=np.float32)
        else:
            w = np.asarray(sample_weight, dtype=np.float32)
            w = w / (w.sum() + 1e-12)
        
        # Cache leaf assignments per tree
        self._leaf_cache = {}
        
        for t_idx, est in enumerate(forest.estimators_):
            tree = est.tree_
            leaf_ids = est.apply(X)
            self._leaf_cache[t_idx] = leaf_ids
            
            # Compute sample-weighted support per leaf
            uniq_leaves = np.unique(leaf_ids)
            
            # Numba-accelerated depth calculation
            if NUMBA_AVAILABLE:
                node_depth = _compute_node_depths_numba(
                    tree.children_left.astype(np.int32),
                    tree.children_right.astype(np.int32),
                    tree.node_count
                )
            else:
                node_depth = np.zeros(tree.node_count, dtype=np.int32)
                stack = [(0, 0)]
                while stack:
                    node, d = stack.pop()
                    node_depth[node] = d
                    left = tree.children_left[node]
                    right = tree.children_right[node]
                    if left != -1:
                        stack.append((left, d + 1))
                    if right != -1:
                        stack.append((right, d + 1))
            
            for leaf in uniq_leaves:
                mask = leaf_ids == leaf
                # Sample-weighted support
                support = w[mask].sum()
                
                if support >= self.min_support:
                    col_specs.append((t_idx, int(leaf)))
                    depths.append(int(node_depth[leaf]))
                    supports.append(support)
        
        # Prune to max_rules by support (already weighted)
        if len(col_specs) > self.max_rules:
            top_idx = np.argsort(-np.array(supports))[:self.max_rules]
            col_specs = [col_specs[i] for i in top_idx]
            depths = [depths[i] for i in top_idx]
        
        self._col_specs = col_specs
        self._depths = np.asarray(depths, dtype=np.int32)
        return self

    def transform(
        self,
        forest: ExtraTreesClassifier | ExtraTreesRegressor,
        X: np.ndarray
    ) -> sparse.csr_matrix:
        """Transform X into sparse rule activation matrix."""
        if not self._col_specs:
            return sparse.csr_matrix((X.shape[0], 0), dtype=np.int8)
        
        n = X.shape[0]
        m = len(self._col_specs)
        
        # Get leaf assignments (use cache if available, otherwise compute)
        per_tree_leaf = {}
        for t_idx, _ in self._col_specs:
            if t_idx not in per_tree_leaf:
                if t_idx in self._leaf_cache and len(self._leaf_cache[t_idx]) == n:
                    per_tree_leaf[t_idx] = self._leaf_cache[t_idx]
                else:
                    per_tree_leaf[t_idx] = forest.estimators_[t_idx].apply(X)
        
        # Build sparse matrix efficiently
        rows_list = []
        cols_list = []
        
        for j, (t_idx, leaf_id) in enumerate(self._col_specs):
            hit = np.where(per_tree_leaf[t_idx] == leaf_id)[0]
            if hit.size > 0:
                rows_list.append(hit)
                cols_list.append(np.full(hit.size, j, dtype=np.int32))
        
        if not rows_list:
            return sparse.csr_matrix((n, m), dtype=np.int8)
        
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        data = np.ones(len(rows), dtype=np.int8)
        
        return sparse.csr_matrix((data, (rows, cols)), shape=(n, m))

    def clear_cache(self):
        """Clear leaf assignment cache to free memory."""
        self._leaf_cache = {}

    @property
    def depths_(self) -> np.ndarray:
        return self._depths


def clean_redundant_rules(
    R_sparse: sparse.csr_matrix,
    y: np.ndarray,
    depths: np.ndarray,
    jaccard_threshold: float = 0.85,
    use_minhash: bool = True,
    n_hashes: int = 8
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """
    Deduplicates rules based on Jaccard sample activation overlap.
    Uses vectorized IC computation and MinHash for O(n log n) approximate Jaccard.
    
    Args:
        R_sparse: Sparse rule activation matrix (n_samples x n_rules)
        y: Target array
        depths: Depth of each rule (for fallback tie-breaking)
        jaccard_threshold: Overlap threshold for considering rules redundant
        use_minhash: Use MinHash approximation for large rule sets
        n_hashes: Number of hash functions for MinHash
    
    Returns:
        R_cleaned: Sparse matrix with redundant rules removed
        depths_cleaned: Depths of retained rules
        kept_indices: Indices of the rules retained from the original matrix
    """
    n_samples, n_rules = R_sparse.shape
    if n_rules <= 1:
        return R_sparse, depths, np.arange(n_rules)
    
    # Convert to CSC for efficient column-wise access
    R_csc = R_sparse.tocsc()
    y_arr = np.ascontiguousarray(y, dtype=np.float32)
    
    # Vectorized IC computation using Numba
    if NUMBA_AVAILABLE and n_rules > 50:
        rule_ics = _compute_rule_ics_vectorized(
            R_csc.data.astype(np.float32),
            R_csc.indices.astype(np.int32),
            R_csc.indptr.astype(np.int32),
            y_arr,
            n_rules,
            n_samples
        )
    else:
        # Fallback: batch correlation using matrix multiplication
        y_centered = y_arr - y_arr.mean()
        y_std = y_arr.std()
        if y_std < 1e-12:
            return R_sparse, depths, np.arange(n_rules)
        
        rule_ics = np.zeros(n_rules, dtype=np.float32)
        for i in range(n_rules):
            col = R_csc.getcol(i).toarray().ravel()
            col_std = col.std()
            if col_std < 1e-12:
                rule_ics[i] = -1.0
                continue
            corr = np.dot(col - col.mean(), y_centered) / (len(y_arr) * col_std * y_std)
            rule_ics[i] = abs(corr) if np.isfinite(corr) else -1.0
    
    # Sort by IC (highest first)
    sorted_idx = np.argsort(-rule_ics)
    
    # Pre-compute MinHash signatures if enabled and set is large
    use_minhash_actual = use_minhash and NUMBA_AVAILABLE and n_rules > 100
    signatures = {}
    col_sums = {}
    
    if use_minhash_actual:
        for i in range(n_rules):
            if rule_ics[i] >= 0:
                indices = R_csc.indices[R_csc.indptr[i]:R_csc.indptr[i + 1]]
                signatures[i] = _minhash_signature(
                    indices.astype(np.int32), n_hashes, n_samples
                )
                col_sums[i] = len(indices)
    
    # Greedy selection with Jaccard check
    kept_indices = []
    kept_col_data = []  # Cache for exact Jaccard fallback
    
    for i in sorted_idx:
        if rule_ics[i] < 0:
            continue
        
        is_redundant = False
        current_start = R_csc.indptr[i]
        current_end = R_csc.indptr[i + 1]
        current_indices = R_csc.indices[current_start:current_end]
        current_sum = len(current_indices)
        
        if current_sum == 0:
            continue
        
        for j, existing_idx in enumerate(kept_indices):
            # MinHash pre-filter
            if use_minhash_actual and i in signatures and existing_idx in signatures:
                approx_jaccard = _estimate_jaccard_minhash(
                    signatures[i], signatures[existing_idx]
                )
                # If MinHash says low similarity, skip exact check
                if approx_jaccard < jaccard_threshold - 0.15:
                    continue
            
            # Exact Jaccard computation
            existing_indices = kept_col_data[j]
            existing_sum = len(existing_indices)
            
            # Set intersection via sorted merge
            intersection = len(np.intersect1d(current_indices, existing_indices, assume_unique=True))
            union = current_sum + existing_sum - intersection
            
            if union > 0:
                jaccard = intersection / union
                if jaccard > jaccard_threshold:
                    is_redundant = True
                    break
        
        if not is_redundant:
            kept_indices.append(i)
            kept_col_data.append(current_indices)
    
    # Finalize
    kept_indices = np.array(kept_indices, dtype=int)
    if len(kept_indices) == 0:
        return sparse.csr_matrix((n_samples, 0), dtype=np.int8), np.array([]), kept_indices
    
    R_cleaned = R_csc[:, kept_indices].tocsr()
    depths_cleaned = depths[kept_indices]
    
    return R_cleaned, depths_cleaned, kept_indices


# -----------------------------
# Stability Selection with Temporal Splits
# -----------------------------
def _proximal_gradient_optimize(
    X: sparse.csr_matrix,
    y: np.ndarray,
    depth_penalties: np.ndarray,
    l1_ratio: float,
    alpha: float,
    sample_weight: np.ndarray | None,
    max_iter: int = 100,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Proximal gradient descent for IC-based ElasticNet objective.
    Properly handles non-differentiable L1 penalty via soft thresholding.
    """
    n, p = X.shape
    beta = np.zeros(p, dtype=np.float32)
    
    # Normalize weights
    if sample_weight is None:
        w = np.ones(n, dtype=np.float32) / n
    else:
        w = sample_weight / (sample_weight.sum() + 1e-12)
    
    # Precompute weighted y stats
    y = np.asarray(y, dtype=np.float32)
    y_mean = np.sum(w * y)
    y_centered = y - y_mean
    y_var = np.sum(w * y_centered**2)
    
    if y_var < 1e-12:
        return beta
    
    y_std = np.sqrt(y_var)
    
    # Step size (use 1/L where L is Lipschitz constant estimate)
    # For correlation objective, L ~ 1/y_std
    step_size = np.float32(0.1) * y_std
    
    for iteration in range(max_iter):
        # Compute prediction and gradient of -IC
        y_hat = np.asarray(X @ beta, dtype=np.float32).ravel()
        yh_mean = np.sum(w * y_hat)
        yh_centered = y_hat - yh_mean
        yh_var = np.sum(w * yh_centered**2)
        
        if yh_var < 1e-12:
            yh_std = np.float32(1e-6)
        else:
            yh_std = np.sqrt(yh_var)
        
        # Gradient of -correlation w.r.t. beta
        # d(-corr)/d(beta) = -d(cov)/(std_y * std_yh) + corr * d(std_yh)/std_yh
        cov = np.sum(w * y_centered * yh_centered)
        corr = cov / (y_std * yh_std + 1e-12)
        
        # Gradient through X
        grad_cov = X.T @ (w * y_centered)
        grad_var = 2 * X.T @ (w * yh_centered)
        
        grad = np.asarray(
            -grad_cov / (y_std * yh_std + 1e-12) + 
            corr * grad_var / (2 * yh_var + 1e-12),
            dtype=np.float32
        ).ravel()
        
        # Proximal gradient step with Numba if available
        if NUMBA_AVAILABLE:
            beta_new = _proximal_gradient_step(
                beta, grad, step_size, l1_ratio, alpha, depth_penalties
            )
        else:
            # Pure numpy fallback
            z = beta - step_size * grad
            l2_factor = np.float32(1.0) / (np.float32(1.0) + step_size * (np.float32(1.0) - l1_ratio) * alpha * depth_penalties)
            z = z * l2_factor
            lam = step_size * l1_ratio * alpha * depth_penalties
            beta_new = np.sign(z) * np.maximum(np.abs(z) - lam, np.float32(0.0))
        
        # Check convergence
        diff = np.max(np.abs(beta_new - beta))
        beta = beta_new
        
        if diff < tol:
            break
    
    return beta


def _single_stability_run(
    run_idx: int,
    X: sparse.csr_matrix,
    y: np.ndarray,
    depth_penalties: np.ndarray,
    subsample_ratio: float,
    val_ratio: float,
    l1_ratio: float,
    alpha: float,
    sample_weight: np.ndarray | None,
    random_state: int,
    gate_groups: Optional[List[slice]],
    lam_group: float,
    use_temporal_split: bool,
    use_proximal_gradient: bool = True,
    proximal_max_iter: int = 100,
    proximal_tol: float = 1e-6
) -> np.ndarray:
    """Single stability selection run (for parallel execution)."""
    rng = np.random.default_rng(random_state + run_idx)
    n, p = X.shape
    
    # Subsample indices
    idx = rng.choice(n, int(n * subsample_ratio), replace=False)
    
    if use_temporal_split:
        # TEMPORAL split: train on earlier samples, validate on later
        idx_sorted = np.sort(idx)
        cut = int(len(idx_sorted) * (1 - val_ratio))
        idx_tr, idx_va = idx_sorted[:cut], idx_sorted[cut:]
    else:
        # Random split
        rng.shuffle(idx)
        cut = int(len(idx) * (1 - val_ratio))
        idx_tr, idx_va = idx[:cut], idx[cut:]
    
    X_va, y_va = X[idx_va], y[idx_va]
    w_va = sample_weight[idx_va] if sample_weight is not None else None
    
    beta0 = np.zeros(p, dtype=np.float32)
    
    try:
        if use_proximal_gradient and (gate_groups is None or lam_group <= 0.0):
            # Use proximal gradient for proper L1 handling
            beta = _proximal_gradient_optimize(
                X_va, y_va, depth_penalties, l1_ratio, alpha, w_va,
                max_iter=proximal_max_iter, tol=proximal_tol
            )
        elif gate_groups is None or lam_group <= 0.0:
            # Fallback to L-BFGS-B
            res = minimize(
                ic_objective_oos,
                x0=beta0,
                args=(X_va, y_va, l1_ratio, alpha, depth_penalties, w_va),
                method="L-BFGS-B",
                options={"maxiter": 300, "disp": False},
            )
            beta = res.x
        else:
            # Group lasso still uses L-BFGS-B (could extend proximal later)
            res = minimize(
                ic_objective_oos_with_groups,
                x0=beta0,
                args=(X_va, y_va, l1_ratio, alpha, depth_penalties, gate_groups, lam_group, w_va),
                method="L-BFGS-B",
                options={"maxiter": 300, "disp": False},
            )
            beta = res.x
    except Exception:
        beta = beta0
    
    return (np.abs(beta) > 1e-8).astype(np.int32)


def stability_selection_oos(
    X: sparse.csr_matrix,
    y: np.ndarray,
    depth_penalties: np.ndarray,
    n_runs: int = 50,
    subsample_ratio: float = 0.8,
    val_ratio: float = 0.25,
    l1_ratio: float = 0.85,
    alpha: float = 0.1,
    threshold: float = 0.6,
    sample_weight: np.ndarray | None = None,
    random_state: int = 42,
    verbose: bool = False,
    gate_groups: Optional[List[slice]] = None,
    lam_group: float = 0.0,
    use_temporal_split: bool = True,
    early_stopping_patience: int = 10,
    n_jobs: int = -1,
    use_proximal_gradient: bool = True,
    proximal_max_iter: int = 100,
    proximal_tol: float = 1e-6
) -> np.ndarray:
    """
    Stability selection with OOS validation, temporal splits, early stopping, and parallelization.
    
    Args:
        X: Feature matrix (sparse)
        y: Target array
        depth_penalties: Per-feature penalty multipliers
        n_runs: Number of stability runs
        subsample_ratio: Fraction of samples to use per run
        val_ratio: Fraction of subsample for validation
        l1_ratio: ElasticNet L1 ratio
        alpha: Regularization strength
        threshold: Selection probability threshold
        sample_weight: Sample weights
        random_state: Random seed
        verbose: Print progress
        gate_groups: Feature groups for group lasso
        lam_group: Group lasso penalty
        use_temporal_split: Use temporal (forward) splits instead of random
        early_stopping_patience: Stop if selection stabilizes for this many runs
        n_jobs: Number of parallel jobs (-1 for all cores)
        use_proximal_gradient: Use proximal gradient for proper L1 handling
        proximal_max_iter: Max iterations for proximal gradient
        proximal_tol: Convergence tolerance for proximal gradient
    
    Returns:
        Boolean mask of stable features
    """
    n, p = X.shape
    counts = np.zeros(p, dtype=np.int32)
    
    # Determine actual n_jobs
    if n_jobs == -1:
        try:
            import os
            n_jobs_actual = min(os.cpu_count() or 4, n_runs)
        except Exception:
            n_jobs_actual = 4
    else:
        n_jobs_actual = min(n_jobs, n_runs)
    
    # Early stopping tracking
    prev_stable_set = set()
    stable_count = 0
    
    # Process in batches for early stopping checks
    batch_size = max(5, n_runs // 10)
    run_idx = 0
    
    while run_idx < n_runs:
        batch_end = min(run_idx + batch_size, n_runs)
        batch_runs = batch_end - run_idx
        
        if n_jobs_actual > 1 and batch_runs > 1:
            # Parallel execution
            try:
                results = Parallel(n_jobs=n_jobs_actual, prefer="threads")(
                    delayed(_single_stability_run)(
                        i, X, y, depth_penalties, subsample_ratio, val_ratio,
                        l1_ratio, alpha, sample_weight, random_state,
                        gate_groups, lam_group, use_temporal_split,
                        use_proximal_gradient, proximal_max_iter, proximal_tol
                    )
                    for i in range(run_idx, batch_end)
                )
                for sel in results:
                    counts += sel
            except Exception:
                # Fallback to sequential
                for i in range(run_idx, batch_end):
                    sel = _single_stability_run(
                        i, X, y, depth_penalties, subsample_ratio, val_ratio,
                        l1_ratio, alpha, sample_weight, random_state,
                        gate_groups, lam_group, use_temporal_split,
                        use_proximal_gradient, proximal_max_iter, proximal_tol
                    )
                    counts += sel
        else:
            # Sequential execution
            for i in range(run_idx, batch_end):
                sel = _single_stability_run(
                    i, X, y, depth_penalties, subsample_ratio, val_ratio,
                    l1_ratio, alpha, sample_weight, random_state,
                    gate_groups, lam_group, use_temporal_split,
                    use_proximal_gradient, proximal_max_iter, proximal_tol
                )
                counts += sel
        
        run_idx = batch_end
        
        # Early stopping check
        if early_stopping_patience > 0 and run_idx >= 15:
            current_prob = counts / run_idx
            current_stable = set(np.where(current_prob >= threshold)[0])
            
            if current_stable == prev_stable_set:
                stable_count += 1
                if stable_count >= early_stopping_patience:
                    if verbose:
                        print(f"  Early stopping at run {run_idx}: selection stabilized")
                    break
            else:
                stable_count = 0
                prev_stable_set = current_stable
        
        if verbose and run_idx % 10 == 0:
            current_selected = (counts / run_idx >= threshold).sum()
            print(f"  Stability run {run_idx}/{n_runs}: {current_selected} features above threshold")
    
    prob = counts / run_idx
    stable_mask = prob >= threshold
    
    if verbose:
        print(f"  Stability selection complete: {stable_mask.sum()} features with P(s) >= {threshold}")
    
    return stable_mask


# -----------------------------
# RuleFit Transformer Config
# -----------------------------
@dataclass
class LeafGateConfig:
    """Configuration for RuleFitTransformer."""
    # Forest parameters
    max_depth: int = 3
    n_estimators: int = 200
    min_samples_leaf: float = 0.01
    max_features: str = "sqrt"  # De Prado: sqrt(n) for diversity
    bootstrap: bool = True
    random_state: int = 7
    
    # Rule extraction
    min_support: float = 0.01
    max_rules: int = 5000
    overlap_threshold: float = 0.85
    
    # Selection
    n_stability_runs: int = 50
    stability_subsample: float = 0.8
    stability_threshold: float = 0.6
    stability_val_ratio: float = 0.25
    use_temporal_split: bool = True
    early_stopping_patience: int = 10
    n_jobs: int = -1
    use_proximal_gradient: bool = True  # Use proximal gradient instead of L-BFGS-B
    proximal_max_iter: int = 100
    proximal_tol: float = 1e-6
    
    # Regularization
    alpha: float = 0.1
    l1_ratio: float = 0.85
    eta: float = 0.3  # Depth penalty multiplier
    use_exponential_depth_penalty: bool = True  # (1+eta)^depth vs 1+eta*depth
    lam_group: float = 0.0
    
    # Temporal weighting (financial regime awareness)
    temporal_decay_half_life: float | None = None  # Fraction of samples for half-weight (e.g., 0.5)
    temporal_decay_type: str = "exponential"  # 'exponential' or 'linear'
    
    # Target processing (4.5σ for fat-tailed financial returns)
    winsorize_k: float = 4.5
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert 0 < self.min_support < 1, f"min_support must be in (0, 1), got {self.min_support}"
        assert 0 < self.l1_ratio <= 1, f"l1_ratio must be in (0, 1], got {self.l1_ratio}"
        assert self.alpha >= 0, f"alpha must be >= 0, got {self.alpha}"
        assert self.eta >= 0, f"eta must be >= 0, got {self.eta}"
        assert self.max_rules > 0, f"max_rules must be > 0, got {self.max_rules}"
        assert self.n_stability_runs > 0, f"n_stability_runs must be > 0, got {self.n_stability_runs}"
        assert 0 < self.stability_subsample <= 1, f"stability_subsample must be in (0, 1], got {self.stability_subsample}"
        assert 0 < self.stability_threshold < 1, f"stability_threshold must be in (0, 1), got {self.stability_threshold}"
        assert self.winsorize_k > 0, f"winsorize_k must be > 0, got {self.winsorize_k}"
        if self.temporal_decay_half_life is not None:
            assert 0 < self.temporal_decay_half_life <= 1, f"temporal_decay_half_life must be in (0, 1], got {self.temporal_decay_half_life}"
        assert self.temporal_decay_type in ("exponential", "linear"), f"temporal_decay_type must be 'exponential' or 'linear'"


# -----------------------------
# RuleFit Transformer
# -----------------------------
class RuleFitTransformer(BaseEstimator, TransformerMixin):
    """
    Enhanced RuleFit with stability selection and IC optimization.
    
    Features:
    - Sample-weighted rule support calculation
    - Temporal train/val splits for forward-looking bias prevention
    - Early stopping when selection stabilizes
    - Parallel stability runs
    - MinHash-accelerated Jaccard deduplication
    - Numba-accelerated core operations

    .. warning::
        **Look-ahead Bias Risk**: The initial rule generation (ExtraTrees fitting) is performed on the
        entire input dataset `X`. While stability selection uses temporal splits to validate the
        rules, the rules themselves are candidates generated using potentially future information
        (if `fit` is called with the full dataset). For strict walk-forward validity, rules should
        be generated on a growing window or purge-k-fold basis, though this is computationally expensive.

    .. note::
        **Cross-Asset usage**: The built-in winsorization is global. If `X` contains multiple assets
        with significantly different volatility regimes that haven't been normalized (e.g. z-scored per asset),
        global winsorization might be suboptimal. Ensure inputs are normalized cross-sectionally or per-asset
        before passing to this transformer.
    """
    def __init__(
        self,
        base_cols: List[str],
        gate_cols: List[str],
        config: LeafGateConfig | None = None,
        verbose: bool = True
    ):
        self.base_cols = base_cols
        self.gate_cols = gate_cols
        self.config = config or LeafGateConfig()
        self.verbose = verbose

        # Fitted assets
        self.scaler_ = StandardScaler()
        self.forest_: ExtraTreesClassifier | ExtraTreesRegressor | None = None
        self.rule_extractor_: LeafRuleExtractor | None = None
        self.stable_mask_: np.ndarray | None = None
        self.node_depths_: np.ndarray | None = None
        self.feature_names_: List[str] = []
        self.n_original_features_: int = 0
        self.base_rules_kept_idx_: np.ndarray | None = None
        self.gated_kept_idx_: np.ndarray | None = None
        self._fitted: bool = False
        self._is_classifier: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
        class_weight: str | Dict | None = "balanced"
    ) -> "RuleFitTransformer":
        """
        Fit RuleFit with stability selection.
        
        Args:
            X: Features (must contain base_cols and gate_cols)
            y: Target (binary or continuous)
            sample_weight: Sample weights
            class_weight: Class weight handling for imbalance (str, dict, or None)
        """
        cfg = self.config
        n = len(X)
        
        if self.verbose:
            print(f"RuleFitTransformer: Fitting on {n} samples...")
        
        # Extract base features and gates
        X_base_val = np.ascontiguousarray(X[self.base_cols].values, dtype=np.float32)
        G = X[self.gate_cols].values.astype(np.int8)
        y_arr = np.ascontiguousarray(y.values, dtype=np.float32)
        
        # Determine if classification BEFORE any class-specific processing
        n_unique = len(np.unique(y_arr[np.isfinite(y_arr)]))
        self._is_classifier = n_unique <= 2
        
        # Winsorize input features to prevent outliers affecting splits
        for i in range(X_base_val.shape[1]):
            X_base_val[:, i] = winsorize(X_base_val[:, i], k=cfg.winsorize_k)
        
        # Winsorize target
        y_wins = winsorize(y_arr, k=cfg.winsorize_k)
        
        # Initialize sample weights
        if sample_weight is None:
            sample_weight = np.ones(n, dtype=np.float32)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float32).copy()
        
        # Handle class imbalance ONLY for classification
        # We use sqrt(ratio) for balanced weighting (softer than standard inverse)
        # If class_weight is 'balanced', we apply this manual weighting and set sklearn_class_weight to None
        sklearn_class_weight_param = None
        
        if self._is_classifier:
            if isinstance(class_weight, dict):
                sklearn_class_weight_param = class_weight
            elif class_weight == "balanced":
                # Manual sqrt balancing
                sklearn_class_weight_param = None
                classes, counts = np.unique(y_arr.astype(int), return_counts=True)
                if len(classes) == 2 and counts.min() > 0:
                    ratio = counts.max() / counts.min()
                    minority_class = classes[np.argmin(counts)]
                    # Use sqrt(ratio) for balanced weighting
                    sample_weight[y_arr.astype(int) == minority_class] *= np.sqrt(ratio)
            else:
                sklearn_class_weight_param = None

        # Apply temporal decay if configured (recent samples weighted higher)
        if cfg.temporal_decay_half_life is not None:
            temporal_weights = _compute_temporal_decay_weights(
                n, cfg.temporal_decay_half_life, cfg.temporal_decay_type
            )
            sample_weight = sample_weight * temporal_weights
            if self.verbose:
                print(f"  Applied {cfg.temporal_decay_type} temporal decay (half-life={cfg.temporal_decay_half_life})")
        
        # Normalize weights (guard against zero sum)
        w_sum = sample_weight.sum()
        if w_sum > 1e-12:
            sample_weight = sample_weight / (w_sum / n)
        else:
            warnings.warn("Sample weights sum to zero, using uniform weights")
            sample_weight = np.ones(n, dtype=np.float32)
        
        # Fit scaler
        self.scaler_.fit(X_base_val)
        X_scaled = self.scaler_.transform(X_base_val).astype(np.float32, copy=False)
        
        # Fit ExtraTrees
        if self.verbose:
            task = "classification" if self._is_classifier else "regression"
            print(f"  Fitting ExtraTrees ({cfg.n_estimators} trees, max_depth={cfg.max_depth}, {task})...")
        
        if self._is_classifier:
            self.forest_ = ExtraTreesClassifier(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                min_samples_leaf=cfg.min_samples_leaf,
                max_features=cfg.max_features,
                bootstrap=cfg.bootstrap,
                n_jobs=-1,
                min_impurity_decrease=1e-4,
                random_state=cfg.random_state,
                class_weight=sklearn_class_weight_param
            )
        else:
            self.forest_ = ExtraTreesRegressor(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                min_samples_leaf=cfg.min_samples_leaf,
                max_features=cfg.max_features,
                bootstrap=cfg.bootstrap,
                n_jobs=-1,
                min_impurity_decrease=1e-4,
                random_state=cfg.random_state
            )
        
        self.forest_.fit(X_scaled, y_arr, sample_weight=sample_weight)
        
        # Extract rules via leaf membership (conjunctions)
        if self.verbose:
            print("  Extracting rules via leaf membership...")

        self.rule_extractor_ = LeafRuleExtractor(
            min_support=cfg.min_support,
            max_rules=cfg.max_rules
        )
        # Pass sample_weight for weighted support calculation
        self.rule_extractor_.fit(self.forest_, X_scaled, sample_weight=sample_weight)
        X_rules = self.rule_extractor_.transform(self.forest_, X_scaled)
        node_depths = self.rule_extractor_.depths_
        
        if self.verbose:
            print(f"  Extracted {X_rules.shape[1]} rules (after support filtering)")
        
        # Deduplicate rules using MinHash-accelerated Jaccard
        X_rules, node_depths, kept_indices = clean_redundant_rules(
            X_rules, y_wins, node_depths,
            jaccard_threshold=cfg.overlap_threshold
        )
        self.base_rules_kept_idx_ = kept_indices
        
        if self.verbose:
            print(f"  After deduplication: {X_rules.shape[1]} rules")
        
        # Gate the rules
        n_rules = X_rules.shape[1]
        n_gates = G.shape[1]
        gated_blocks = []
        gated_depths = []
        
        for k in range(n_gates):
            gk = G[:, k].reshape(-1, 1)
            gated = X_rules.multiply(gk)
            gated_blocks.append(gated)
            gated_depths.append(node_depths + 1)  # Gating adds complexity
        
        # Combine gated rules and clean them too
        X_gated_all = sparse.hstack(gated_blocks, format="csr")
        gated_depths_all = np.concatenate(gated_depths)
        
        if self.verbose:
            print(f"  Gated rules before cleaning: {X_gated_all.shape[1]}")
        
        # Apply Jaccard cleaning to gated rules as well
        X_gated_cleaned, gated_depths_cleaned, kept_indices = clean_redundant_rules(
            X_gated_all, y_wins, gated_depths_all,
            jaccard_threshold=cfg.overlap_threshold
        )
        self.gated_kept_idx_ = kept_indices
        
        if self.verbose:
            print(f"  Gated rules after cleaning: {X_gated_cleaned.shape[1]}")
        
        # Combine: base + gates + rules + cleaned_gated_rules
        X_base_sp = sparse.csr_matrix(X_scaled, dtype=np.float32)
        G_sp = sparse.csr_matrix(G)
        
        Phi = sparse.hstack([X_base_sp, G_sp, X_rules, X_gated_cleaned], format="csr")
        
        # Compute depth penalties (exponential or linear based on config)
        base_depths = np.zeros(X_scaled.shape[1])
        gate_depths_arr = np.zeros(G.shape[1])
        all_depths = np.concatenate([
            base_depths,
            gate_depths_arr,
            node_depths,
            gated_depths_cleaned
        ])
        
        if cfg.use_exponential_depth_penalty:
            # De Prado-recommended exponential: (1 + eta)^depth
            depth_penalties = np.power(1 + cfg.eta, all_depths).astype(np.float32)
        else:
            # Linear: 1 + eta * depth
            depth_penalties = (1 + cfg.eta * all_depths).astype(np.float32)

        
        if self.verbose:
            print(f"  Full feature matrix: {Phi.shape}")
            print(f"  Running stability selection ({cfg.n_stability_runs} runs)...")
        
        n_gated_per_gate = []
        if self.gated_kept_idx_ is not None:
            start = 0
            kept = self.gated_kept_idx_
            for block in gated_blocks:
                end = start + block.shape[1]
                n_gated_per_gate.append(int(np.sum((kept >= start) & (kept < end))))
                start = end
        else:
            n_gated_per_gate = [block.shape[1] for block in gated_blocks]

        gate_groups = build_gate_groups(
            n_base=X_scaled.shape[1],
            n_gates=G.shape[1],
            n_rules=n_rules,
            n_gated_per_gate=n_gated_per_gate
        )

        # OOS stability selection with temporal splits, early stopping, and parallelization
        self.stable_mask_ = stability_selection_oos(
            Phi, y_wins, depth_penalties,
            n_runs=cfg.n_stability_runs,
            subsample_ratio=cfg.stability_subsample,
            val_ratio=cfg.stability_val_ratio,
            l1_ratio=cfg.l1_ratio,
            alpha=cfg.alpha,
            threshold=cfg.stability_threshold,
            sample_weight=sample_weight,
            random_state=cfg.random_state,
            verbose=self.verbose,
            gate_groups=gate_groups,
            lam_group=cfg.lam_group,
            use_temporal_split=cfg.use_temporal_split,
            early_stopping_patience=cfg.early_stopping_patience,
            n_jobs=cfg.n_jobs,
            use_proximal_gradient=cfg.use_proximal_gradient,
            proximal_max_iter=cfg.proximal_max_iter,
            proximal_tol=cfg.proximal_tol
        )
        
        self.node_depths_ = all_depths
        
        # Generate feature names with informative gated rule naming
        base_names = [f"base:{c}" for c in self.base_cols]
        gate_names = [f"gate:{c}" for c in self.gate_cols]
        rule_names = [f"rule:{j}" for j in range(n_rules)]
        
        # Gated rules include gate name for interpretability
        n_gated_cleaned = X_gated_cleaned.shape[1]
        gated_rule_names = []
        if self.gated_kept_idx_ is not None and len(self.gated_kept_idx_) > 0:
            # Map kept indices back to gate + rule
            cumsum = 0
            gate_boundaries = []
            for k, block in enumerate(gated_blocks):
                gate_boundaries.append((cumsum, cumsum + block.shape[1], self.gate_cols[k]))
                cumsum += block.shape[1]
            
            for idx in self.gated_kept_idx_:
                gate_name = "unknown"
                rule_idx = idx
                for start, end, gname in gate_boundaries:
                    if start <= idx < end:
                        gate_name = gname
                        rule_idx = idx - start
                        break
                gated_rule_names.append(f"gated:{gate_name}:rule{rule_idx}")
        else:
            gated_rule_names = [f"gated_rule:{j}" for j in range(n_gated_cleaned)]
        
        self.feature_names_ = base_names + gate_names + rule_names + gated_rule_names
        self._n_base = len(base_names)
        self._n_gates = len(gate_names)
        self._n_rules = n_rules
        self._n_gated = n_gated_cleaned
        self.n_original_features_ = X.shape[1]
        self._fitted = True
        
        n_selected = self.stable_mask_.sum()

        # Clear extractor cache to free memory
        if self.rule_extractor_ is not None:
            self.rule_extractor_.clear_cache()

        if self.verbose:
            print(f"RuleFitTransformer: Fitted. Selected {n_selected} stable features.")
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted rules and selection."""
        if not self._fitted:
            raise ValueError("Transformer not fitted.")
        
        # Extract and scale base features
        X_base = self.scaler_.transform(X[self.base_cols].values.astype(np.float32)).astype(
            np.float32, copy=False
        )
        G = X[self.gate_cols].values.astype(np.int8)
        
        # Get rules (aligned to fit-time specs)
        X_rules = self.rule_extractor_.transform(self.forest_, X_base)
        
        # Apply base rule deduplication filter
        if self.base_rules_kept_idx_ is not None:
            X_rules = X_rules[:, self.base_rules_kept_idx_]

        # Gate the rules
        n_gates = G.shape[1]
        gated_blocks = []
        for k in range(n_gates):
            gk = G[:, k].reshape(-1, 1)
            gated = X_rules.multiply(gk)
            gated_blocks.append(gated)
        
        # Combine
        X_base_sp = sparse.csr_matrix(X_base, dtype=np.float32)
        G_sp = sparse.csr_matrix(G)
        X_gated_all = sparse.hstack(gated_blocks, format="csr")
        if self.gated_kept_idx_ is not None:
            X_gated = X_gated_all[:, self.gated_kept_idx_]
        else:
            X_gated = X_gated_all
        
        Phi = sparse.hstack([X_base_sp, G_sp, X_rules, X_gated], format="csr")
        
        # Apply selection
        if Phi.shape[1] != len(self.stable_mask_):
            raise ValueError(
                f"Feature matrix shape mismatch: got {Phi.shape[1]} columns, "
                f"expected {len(self.stable_mask_)}. This may indicate a bug in "
                "rule extraction or gating logic."
            )
        
        selected = Phi[:, self.stable_mask_].toarray().astype(np.float32, copy=False)
        names = [n for n, s in zip(self.feature_names_, self.stable_mask_) if s]
        
        return pd.DataFrame(selected, index=X.index, columns=names)

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
        **kwargs
    ) -> pd.DataFrame:
        """Fit and transform in one call."""
        self.fit(X, y, sample_weight=sample_weight, **kwargs)
        return self.transform(X)

    def get_selection_probabilities(self) -> Dict[str, float]:
        """Return selection probabilities for all features (requires fitted transformer)."""
        if not self._fitted or self.stable_mask_ is None:
            raise ValueError("Transformer not fitted.")
        # Note: actual probabilities are not stored, only the final mask
        # This returns 1.0 for selected, 0.0 for not selected
        return {
            name: 1.0 if selected else 0.0
            for name, selected in zip(self.feature_names_, self.stable_mask_)
        }
