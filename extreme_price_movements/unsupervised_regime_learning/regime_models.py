"""Unsupervised regime discovery from selected regime-learning features."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import NearestNeighbors

try:  # pragma: no cover - optional dependency path
    import lightgbm as lgb

    _LGBM_AVAILABLE = True
except Exception:  # pragma: no cover
    lgb = None
    _LGBM_AVAILABLE = False

try:  # pragma: no cover - optional dependency path
    from hmmlearn.hmm import GaussianHMM

    _HMM_AVAILABLE = True
except Exception:  # pragma: no cover
    GaussianHMM = None
    _HMM_AVAILABLE = False

try:  # pragma: no cover - optional dependency path
    import hdbscan

    _HDBSCAN_AVAILABLE = True
except Exception:  # pragma: no cover
    hdbscan = None
    _HDBSCAN_AVAILABLE = False

try:  # pragma: no cover - optional dependency path
    import umap

    _UMAP_AVAILABLE = True
except Exception:  # pragma: no cover
    umap = None
    _UMAP_AVAILABLE = False

try:  # pragma: no cover - optional dependency path
    import torch
    from torch import nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None
    _TORCH_AVAILABLE = False

try:  # pragma: no cover - optional acceleration path
    from numba import njit as _numba_njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    _NUMBA_AVAILABLE = False

    def _numba_njit(*args: Any, **kwargs: Any) -> Any:
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _wrap(fn: Any) -> Any:
            return fn

        return _wrap

from extreme_price_movements.unsupervised_regime_learning.diagnostics import (
    numeric_feature_frame,
    stratified_period_sample_positions,
    time_sort_order,
)


ADVANCED_REGIME_LEARNING_SCHEMA_VERSION = "unsupervised_regime_learning_v2"


@dataclass(frozen=True)
class AdvancedRegimeLearningConfig:
    random_state: int = 42
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"
    max_rows: int = 50000
    sample_time_bins: int = 32
    scaling_mode: str = "causal_expanding"
    scaling_min_periods: int = 64

    selector_backend: str = "lgbm"
    null_ratio: float = 1.0
    null_block_size: int = 24
    stability_bootstraps: int = 12
    stability_top_m: int = 80
    bootstrap_block_hours: int = 24 * 7
    max_depth: int = 3
    large_sample_depth_threshold: int = 100000
    max_depth_large_sample: int = 4
    min_leaf_fraction: float = 0.025
    n_estimators: int = 80
    learning_rate: float = 0.05
    max_classifier_rows: int = 60000
    lgbm_feature_fraction: float = 0.85
    lgbm_bagging_fraction: float = 0.85
    lgbm_bagging_freq: int = 1
    lgbm_min_gain_to_split: float = 0.0
    lgbm_lambda_l1: float = 0.0
    lgbm_lambda_l2: float = 0.0

    conservative_threshold: float = 0.80
    strong_threshold: float = 0.70
    exploratory_threshold: float = 0.50

    leaf_trees: int = 80
    leaf_embedding_dim: int = 8
    raw_embedding_dim: int = 8
    n_regimes: int = 5
    min_regime_duration: int = 4
    bayesian_gmm_covariance_type: str = "diag"
    bayesian_gmm_weight_concentration_prior: float = 0.0
    bayesian_gmm_reg_covar: float = 1e-6
    bayesian_gmm_max_iter: int = 200
    hdbscan_min_cluster_size: int = 0
    hdbscan_min_cluster_size_fraction: float = 0.0
    hdbscan_min_samples: int = 0
    hdbscan_cluster_selection_epsilon: float = 0.0
    hdbscan_cluster_selection_method: str = "eom"
    hmm_covariance_type: str = "diag"
    hmm_n_iter: int = 100
    hmm_tol: float = 1e-2
    hmm_min_covar: float = 1e-3
    hmm_transmat_self_bias: float = 0.0
    hmm_startprob_prior: float = 1.0
    spectral_n_neighbors: int = 10
    spectral_affinity: str = "nearest_neighbors"
    spectral_assign_labels: str = "kmeans"
    spectral_gamma: float = 1.0
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    kmeans_tol: float = 1e-4
    kmeans_algorithm: str = "lloyd"

    mfa_regimes: int = 5
    mfa_factors: int = 3
    mfa_max_iter: int = 25
    mfa_l1_lambda: float = 0.001
    mfa_tol: float = 1e-4
    mfa_relevance_min: float = 0.0
    mfa_min_keep_features: int = 8

    ae_latent_dim: int = 8
    ae_hidden_dim: int = 32
    ae_epochs: int = 40
    ae_batch_size: int = 256
    ae_backend: str = "numpy"
    ae_torch_enabled: bool = False
    ae_max_train_rows: int = 20000
    ae_learning_rate: float = 1e-3
    ae_weight_decay: float = 1e-4
    ae_dropout: float = 0.05
    ae_noise: float = 0.03
    ae_family_mask_rate: float = 0.15
    ae_lambda_sparse: float = 1e-3
    ae_lambda_contrastive: float = 0.20
    ae_lambda_smooth: float = 0.01
    ae_temperature: float = 0.20

    keep_candidate_margin: float = 0.0
    regime_assessment_oos_folds: int = 3
    regime_assessment_bootstraps: int = 3
    regime_assessment_windows: int = 4
    regime_assessment_null_repeats: int = 3
    regime_assessment_feature_top_n: int = 20
    regime_assessment_transition_tau: float = 0.50
    regime_assessment_geometry_min_rows: int = 8
    regime_assessment_max_auc_features: int = 96
    regime_assessment_max_auc_rows: int = 20000
    regime_assessment_max_robustness_rows: int = 20000
    regime_assessment_max_geometry_rows_per_regime: int = 512
    leaf_embedding_max_trees: int = 64
    persistence_split_dataframes: bool = True
    eps: float = 1e-8


@dataclass(frozen=True)
class AdvancedRegimeLearningArtifact:
    schema_version: str
    selected_features: list[str]
    conservative_features: list[str]
    strong_features: list[str]
    exploratory_features: list[str]
    stability_frequencies: pd.DataFrame
    real_vs_null_importances: pd.DataFrame
    leaf_embeddings: pd.DataFrame
    raw_baseline_embeddings: pd.DataFrame
    ae_latents: pd.DataFrame
    contrastive_ae_latents: pd.DataFrame
    contrastive_leaf_latents: pd.DataFrame
    mfa_responsibilities: pd.DataFrame
    mfa_feature_relevance: pd.DataFrame
    ae_feature_gates: pd.DataFrame
    regime_labels: pd.DataFrame
    regime_probabilities: pd.DataFrame
    regime_transition_features: pd.DataFrame
    regime_feature_importance: pd.DataFrame
    regime_tradability_diagnostics: pd.DataFrame
    regime_diagnostics: pd.DataFrame
    pipeline_steps: pd.DataFrame
    model_regime_features: pd.DataFrame
    model_regime_feature_metrics: pd.DataFrame
    materialized_features: pd.DataFrame
    materialized_feature_groups: dict[str, list[str]]
    specialist_candidate_features: list[str]
    method_keep_decisions: pd.DataFrame
    row_keys: pd.DataFrame = field(default_factory=pd.DataFrame)
    method_embeddings: dict[str, pd.DataFrame] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _stable_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = str(value)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _feature_family(name: str) -> str:
    low = str(name).lower()
    if low.startswith("cov_"):
        return "covariance"
    if low.startswith("corr_"):
        return "correlation"
    if low.startswith("q_"):
        return "quantile"
    if low.startswith("autocorr"):
        return "autocorr"
    if low.startswith("eig_"):
        return "eigen"
    if low.startswith("svd") or "knn" in low:
        return "svd_knn"
    if "fund" in low:
        return "funding"
    if "oi" in low or "open_interest" in low:
        return "open_interest"
    if "volume" in low or "amihud" in low or "liquidity" in low:
        return "liquidity"
    if "rvol" in low:
        return "liquidity"
    if "vol" in low or "atr" in low or "range" in low or low.startswith("rv_") or "variance" in low:
        return "volatility"
    if "trend" in low or "ema" in low or "momentum" in low:
        return "trend"
    if "entropy" in low or "efficiency" in low or "coherence" in low:
        return "path_structure"
    return "primitive"


def _numeric_scaled_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    max_rows: int,
    timestamp_col: str,
    symbol_col: str,
    sample_time_bins: int,
    scaling_mode: str,
    scaling_min_periods: int,
    eps: float,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    features = [str(c) for c in dict.fromkeys(feature_columns) if str(c) in frame.columns]
    x = numeric_feature_frame(frame, features)
    matrix = x[features].to_numpy(dtype=np.float32, copy=True) if features else np.zeros((len(frame), 0), dtype=np.float32)
    finite_frac = np.isfinite(matrix).mean(axis=0) if matrix.size else np.zeros(0)
    keep = finite_frac >= 0.50
    features = [feature for feature, ok in zip(features, keep) if bool(ok)]
    matrix = matrix[:, keep] if matrix.size else matrix
    if matrix.shape[1] == 0:
        return matrix.astype(np.float32), [], {"sampled_rows": 0, "kept_features": 0}
    mode = str(scaling_mode or "causal_expanding").strip().lower()
    if mode in {"causal", "causal_expanding", "expanding"}:
        scaled = _causal_expanding_scale_matrix(
            frame,
            matrix,
            timestamp_col=timestamp_col,
            symbol_col=symbol_col,
            min_periods=int(scaling_min_periods),
            eps=float(eps),
        )
        return scaled, features, {
            "sampled_rows": int(len(frame)),
            "kept_features": int(len(features)),
            "scaling_mode": "causal_expanding",
            "scaling_min_periods": int(scaling_min_periods),
        }
    fit_pos = stratified_period_sample_positions(
        frame,
        np.arange(len(frame), dtype=int),
        max_rows=int(max_rows or 0) or None,
        timestamp_col=timestamp_col,
        symbol_col=symbol_col,
        n_periods=sample_time_bins,
    )
    fit = matrix[fit_pos]
    center = np.nanmedian(fit, axis=0)
    center = np.where(np.isfinite(center), center, 0.0).astype(np.float32)
    q25 = np.nanpercentile(fit, 25.0, axis=0)
    q75 = np.nanpercentile(fit, 75.0, axis=0)
    scale = (q75 - q25).astype(np.float32)
    std = np.nanstd(fit, axis=0).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > eps), scale, std)
    scale = np.where(np.isfinite(scale) & (scale > eps), scale, 1.0).astype(np.float32)
    missing = ~np.isfinite(matrix)
    if missing.any():
        matrix[missing] = np.take(center, np.where(missing)[1])
    matrix = (matrix - center.reshape(1, -1)) / np.maximum(scale.reshape(1, -1), eps)
    return np.clip(matrix, -8.0, 8.0).astype(np.float32), features, {
        "sampled_rows": int(len(fit_pos)),
        "kept_features": int(len(features)),
        "scaling_mode": "sample_fit",
    }


def _causal_expanding_scale_matrix(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    *,
    timestamp_col: str,
    symbol_col: str,
    min_periods: int,
    eps: float,
) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    out = np.empty_like(arr, dtype=np.float32)
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        return out
    order = time_sort_order(frame, symbol_col=symbol_col, timestamp_col=timestamp_col)
    ordered = arr[order]
    if symbol_col in frame.columns:
        ordered_symbols = frame.iloc[order][symbol_col].astype(str).to_numpy()
    else:
        ordered_symbols = np.repeat("__all__", len(order))
    symbol_codes = pd.factorize(ordered_symbols, sort=False)[0].astype(np.int64, copy=False)
    min_count = max(2, int(min_periods or 2))
    if _NUMBA_AVAILABLE:
        ordered_scaled = _causal_expanding_scale_ordered_numba(
            ordered.astype(np.float32, copy=False),
            symbol_codes,
            int(min_count),
            float(eps),
        )
        out[order] = ordered_scaled
        return out
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and symbol_codes[end] == symbol_codes[start]:
            end += 1
        block = ordered[start:end].astype(np.float64, copy=False)
        finite = np.isfinite(block)
        val = np.where(finite, block, 0.0)
        count = np.cumsum(finite.astype(np.float64), axis=0)
        csum = np.cumsum(val, axis=0)
        csum2 = np.cumsum(val * val, axis=0)
        cur_count = finite.astype(np.float64)
        prior_count = count - cur_count
        prior_sum = csum - val
        prior_sum2 = csum2 - val * val
        incl_count = np.maximum(count, 1.0)
        incl_mean = csum / incl_count
        incl_var = np.maximum(csum2 / incl_count - incl_mean * incl_mean, 0.0)
        prior_denom = np.maximum(prior_count, 1.0)
        prior_mean = prior_sum / prior_denom
        prior_var = np.maximum(prior_sum2 / prior_denom - prior_mean * prior_mean, 0.0)
        use_prior = prior_count >= float(min_count)
        center = np.where(use_prior, prior_mean, incl_mean)
        scale = np.where(use_prior, np.sqrt(prior_var), np.sqrt(incl_var))
        center = np.where(np.isfinite(center), center, 0.0)
        scale = np.where(np.isfinite(scale) & (scale > eps), scale, 1.0)
        filled = np.where(finite, block, center)
        out[order[start:end]] = np.clip((filled - center) / np.maximum(scale, eps), -8.0, 8.0).astype(np.float32)
        start = end
    return out


@_numba_njit(cache=True)
def _causal_expanding_scale_ordered_numba(
    ordered: np.ndarray,
    symbol_codes: np.ndarray,
    min_count: int,
    eps: float,
) -> np.ndarray:
    n, p = ordered.shape
    out = np.empty((n, p), dtype=np.float32)
    start = 0
    while start < n:
        end = start + 1
        code = symbol_codes[start]
        while end < n and symbol_codes[end] == code:
            end += 1
        counts = np.zeros(p, dtype=np.float64)
        sums = np.zeros(p, dtype=np.float64)
        sums2 = np.zeros(p, dtype=np.float64)
        for i in range(start, end):
            for j in range(p):
                value = float(ordered[i, j])
                finite = np.isfinite(value)
                prior_count = counts[j]
                prior_sum = sums[j]
                prior_sum2 = sums2[j]
                if finite:
                    incl_count = prior_count + 1.0
                    incl_sum = prior_sum + value
                    incl_sum2 = prior_sum2 + value * value
                else:
                    incl_count = prior_count
                    incl_sum = prior_sum
                    incl_sum2 = prior_sum2
                if prior_count >= float(min_count):
                    center = prior_sum / max(prior_count, 1.0)
                    variance = prior_sum2 / max(prior_count, 1.0) - center * center
                elif incl_count > 0.0:
                    center = incl_sum / incl_count
                    variance = incl_sum2 / incl_count - center * center
                else:
                    center = 0.0
                    variance = 0.0
                if not np.isfinite(center):
                    center = 0.0
                if not np.isfinite(variance) or variance < 0.0:
                    variance = 0.0
                scale = np.sqrt(variance)
                if not np.isfinite(scale) or scale <= eps:
                    scale = 1.0
                filled = value if finite else center
                scaled = (filled - center) / max(scale, eps)
                if scaled > 8.0:
                    scaled = 8.0
                elif scaled < -8.0:
                    scaled = -8.0
                out[i, j] = np.float32(scaled)
                if finite:
                    counts[j] = incl_count
                    sums[j] = incl_sum
                    sums2[j] = incl_sum2
        start = end
    return out


def _coarse_bucket(values: np.ndarray, n_bins: int = 4) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(vals)
    if int(finite.sum()) < n_bins:
        return np.zeros(vals.size, dtype=np.int64)
    qs = np.nanpercentile(vals[finite], np.linspace(0.0, 100.0, n_bins + 1)[1:-1])
    return np.digitize(np.nan_to_num(vals, nan=np.nanmedian(vals[finite])), qs).astype(np.int64)


def _bucket_ids(matrix: np.ndarray, features: Sequence[str]) -> np.ndarray:
    vol_idx = next((i for i, f in enumerate(features) if "vol" in f.lower() or "atr" in f.lower()), 0)
    trend_idx = next((i for i, f in enumerate(features) if "trend" in f.lower() or "ema" in f.lower()), min(1, matrix.shape[1] - 1))
    vol_bucket = _coarse_bucket(matrix[:, vol_idx])
    trend_bucket = _coarse_bucket(matrix[:, trend_idx])
    return (vol_bucket * 10 + trend_bucket).astype(np.int64)


def _block_shuffle_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    block = max(1, int(block_size or 1))
    starts = np.arange(0, n, block, dtype=int)
    order = starts.copy()
    rng.shuffle(order)
    pieces = [np.arange(start, min(start + block, n), dtype=int) for start in order]
    return np.concatenate(pieces)[:n] if pieces else np.arange(n, dtype=int)


def _take_positions_with_replacement(
    positions: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.int64)
    if int(size) <= 0:
        return np.zeros(0, dtype=np.int64)
    if pos.size == 0:
        return np.zeros(int(size), dtype=np.int64)
    if pos.size >= int(size):
        return pos[: int(size)]
    extra = rng.choice(pos, size=int(size) - pos.size, replace=True)
    return np.concatenate([pos, extra]).astype(np.int64)


def generate_real_vs_null_samples(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    *,
    config: AdvancedRegimeLearningConfig,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create real-vs-null samples using four joint-structure-destroying nulls."""

    rng = np.random.default_rng(int(config.random_state))
    x = np.asarray(matrix, dtype=np.float32)
    n, p = x.shape
    if n == 0 or p == 0:
        return x, np.ones(n, dtype=np.int8), pd.DataFrame()
    null_parts: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    per_mode = max(1, int(np.ceil(n * float(config.null_ratio) / 4.0)))
    row_sel = rng.choice(np.arange(n), size=per_mode, replace=n < per_mode)
    buckets = _bucket_ids(x, feature_names)
    col_index = np.arange(p)

    # 1. Within-bucket per-feature shuffles.
    part = x[row_sel].copy()
    for bucket in np.unique(buckets):
        src = np.flatnonzero(buckets == bucket)
        dst_local = np.flatnonzero(buckets[row_sel] == bucket)
        if src.size < 2 or dst_local.size == 0:
            continue
        sampled = rng.choice(src, size=(dst_local.size, p), replace=True)
        part[dst_local] = x[sampled, col_index]
    null_parts.append(part)
    rows.append({"mode": "bucket_shuffle", "rows": int(len(part))})

    # 2. Family-level row permutations.
    part = x[row_sel].copy()
    families = np.asarray([_feature_family(f) for f in feature_names])
    unique_families = pd.unique(families)
    for family in unique_families:
        cols = np.flatnonzero(families == family)
        perm = _take_positions_with_replacement(rng.permutation(n), per_mode, rng)
        part[:, cols] = x[np.ix_(perm, cols)]
    null_parts.append(part)
    rows.append({"mode": "operator_family_permutation", "rows": int(len(part))})

    # 3. Circular shifts by related feature groups.
    part = x[row_sel].copy()
    for family in unique_families:
        cols = np.flatnonzero(families == family)
        lag = int(rng.integers(1, max(2, n)))
        source = (row_sel - lag) % n
        part[:, cols] = x[np.ix_(source, cols)]
    null_parts.append(part)
    rows.append({"mode": "circular_group_shift", "rows": int(len(part))})

    # 4. Per-feature block shuffles preserving local temporal texture.
    source_idx = np.empty((per_mode, p), dtype=np.int64)
    for j in range(p):
        idx = _block_shuffle_indices(n, int(config.null_block_size), rng)
        source_idx[:, j] = _take_positions_with_replacement(idx, per_mode, rng)
    part = x[source_idx, col_index]
    null_parts.append(part)
    rows.append({"mode": "block_shuffle_individual_texture", "rows": int(len(part))})

    null_x = np.vstack(null_parts).astype(np.float32)
    real_x = x
    combined = np.vstack([real_x, null_x]).astype(np.float32)
    y = np.concatenate(
        [np.ones(len(real_x), dtype=np.int8), np.zeros(len(null_x), dtype=np.int8)]
    )
    return combined, y, pd.DataFrame(rows)


def _temporal_bootstrap_indices(
    frame: pd.DataFrame,
    n_rows: int,
    *,
    timestamp_col: str,
    block_hours: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if timestamp_col not in frame.columns:
        return rng.choice(np.arange(n_rows), size=n_rows, replace=True)
    ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    if not ts.notna().any():
        return rng.choice(np.arange(n_rows), size=n_rows, replace=True)
    ts_ns = ts.to_numpy(dtype="datetime64[ns]").astype("int64")
    valid = ts.notna().to_numpy(dtype=bool)
    start = int(np.nanmin(ts_ns[valid]))
    block_ns = max(1, int(block_hours) * 3600 * 1_000_000_000)
    ids = (ts_ns - start) // block_ns
    blocks = [np.flatnonzero(valid & (ids == bid)) for bid in np.unique(ids[valid])]
    blocks = [b for b in blocks if b.size]
    if not blocks:
        return rng.choice(np.arange(n_rows), size=n_rows, replace=True)
    selected: list[np.ndarray] = []
    while sum(len(x) for x in selected) < n_rows:
        selected.append(blocks[int(rng.integers(0, len(blocks)))])
    out = np.concatenate(selected)[:n_rows]
    rng.shuffle(out)
    return out.astype(np.int64)


def _train_real_null_classifier(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    config: AdvancedRegimeLearningConfig,
    random_state: int,
) -> tuple[Any, np.ndarray, str]:
    n = len(y)
    min_leaf = max(2, int(np.ceil(float(config.min_leaf_fraction) * max(n, 1))))
    depth = int(config.max_depth)
    if n >= int(config.large_sample_depth_threshold):
        depth = min(int(config.max_depth_large_sample), 5)
    backend = str(config.selector_backend).lower()
    if backend == "lgbm" and _LGBM_AVAILABLE:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": float(config.learning_rate),
            "num_leaves": int(min(2 ** depth, 31)),
            "max_depth": depth,
            "min_data_in_leaf": min_leaf,
            "feature_fraction": float(np.clip(config.lgbm_feature_fraction, 0.05, 1.0)),
            "bagging_fraction": float(np.clip(config.lgbm_bagging_fraction, 0.05, 1.0)),
            "bagging_freq": max(0, int(config.lgbm_bagging_freq)),
            "min_gain_to_split": max(float(config.lgbm_min_gain_to_split), 0.0),
            "lambda_l1": max(float(config.lgbm_lambda_l1), 0.0),
            "lambda_l2": max(float(config.lgbm_lambda_l2), 0.0),
            "verbosity": -1,
            "seed": int(random_state),
            "num_threads": 1,
        }
        dtrain = lgb.Dataset(x, label=y, feature_name=list(feature_names), free_raw_data=True)
        model = lgb.train(params, dtrain, num_boost_round=int(config.n_estimators))
        imp = model.feature_importance(importance_type="gain").astype(np.float64)
        return model, imp, "lightgbm"
    model = RandomForestClassifier(
        n_estimators=int(config.n_estimators),
        max_depth=depth,
        min_samples_leaf=min_leaf,
        max_features="sqrt",
        random_state=int(random_state),
        n_jobs=1,
    )
    model.fit(x, y)
    return model, model.feature_importances_.astype(np.float64), "random_forest"


def real_vs_null_stability_selection(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    feature_names: Sequence[str],
    *,
    config: AdvancedRegimeLearningConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, Any, pd.DataFrame]:
    """Mandatory temporal bootstrap stability selection for real-vs-null structure."""

    rng = np.random.default_rng(int(config.random_state))
    full_x, full_y, null_report = generate_real_vs_null_samples(
        matrix,
        feature_names,
        config=config,
    )
    n_real = matrix.shape[0]
    n_null = full_x.shape[0] - n_real
    counts = np.zeros(len(feature_names), dtype=np.float64)
    importances = np.zeros(len(feature_names), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    bootstraps = max(1, int(config.stability_bootstraps))
    top_m = max(1, min(int(config.stability_top_m), len(feature_names)))
    null_idx_all = np.arange(n_real, n_real + n_null, dtype=np.int64)
    for b in range(bootstraps):
        real_idx = _temporal_bootstrap_indices(
            frame,
            n_real,
            timestamp_col=config.timestamp_col,
            block_hours=config.bootstrap_block_hours,
            rng=rng,
        )
        null_idx = rng.choice(null_idx_all, size=len(real_idx), replace=len(null_idx_all) < len(real_idx))
        idx = np.concatenate([real_idx, null_idx])
        if len(idx) > int(config.max_classifier_rows):
            idx = rng.choice(idx, size=int(config.max_classifier_rows), replace=False)
        rng.shuffle(idx)
        model, imp, backend = _train_real_null_classifier(
            full_x[idx],
            full_y[idx],
            feature_names,
            config=config,
            random_state=int(config.random_state + b),
        )
        order = np.argsort(imp)[::-1][:top_m]
        counts[order] += 1.0
        importances += imp / max(float(np.nanmax(imp)), config.eps)
        rows.append(
            {
                "bootstrap": b,
                "backend": backend,
                "rows": int(len(idx)),
                "top_m": int(top_m),
            }
        )
    freq = counts / float(bootstraps)
    imp_mean = importances / float(bootstraps)
    stability = pd.DataFrame(
        {
            "feature": list(feature_names),
            "selection_frequency": freq,
            "mean_importance": imp_mean,
            "tier": np.where(
                freq >= float(config.conservative_threshold),
                "conservative",
                np.where(
                    freq >= float(config.strong_threshold),
                    "strong",
                    np.where(freq >= float(config.exploratory_threshold), "exploratory", "drop"),
                ),
            ),
        }
    ).sort_values(
        ["selection_frequency", "mean_importance"],
        ascending=False,
        kind="mergesort",
    )
    final_model, final_imp, backend = _train_real_null_classifier(
        full_x,
        full_y,
        feature_names,
        config=config,
        random_state=int(config.random_state + 999),
    )
    importances_df = pd.DataFrame(
        {"feature": list(feature_names), "importance": final_imp, "backend": backend}
    ).sort_values("importance", ascending=False, kind="mergesort")
    diagnostics = pd.DataFrame(rows)
    return stability, importances_df, final_model, pd.concat(
        [null_report.assign(section="null_generation"), diagnostics.assign(section="bootstrap")],
        ignore_index=True,
        sort=False,
    )


def _leaf_indices(model: Any, matrix: np.ndarray, backend: str) -> np.ndarray:
    if backend == "lightgbm" and hasattr(model, "predict"):
        leaves = model.predict(matrix, pred_leaf=True)
        return np.asarray(leaves, dtype=np.int64)
    if hasattr(model, "apply"):
        return np.asarray(model.apply(matrix), dtype=np.int64)
    return np.zeros((len(matrix), 0), dtype=np.int64)


def leaf_embedding_from_classifier(
    model: Any,
    backend: str,
    real_matrix: np.ndarray,
    train_matrix: np.ndarray,
    train_y: np.ndarray,
    *,
    max_trees: int = 0,
) -> pd.DataFrame:
    train_leaves = _leaf_indices(model, train_matrix, backend)
    real_leaves = _leaf_indices(model, real_matrix, backend)
    if real_leaves.ndim == 1:
        real_leaves = real_leaves.reshape(-1, 1)
        train_leaves = train_leaves.reshape(-1, 1)
    tree_count = int(real_leaves.shape[1])
    max_tree_count = int(max_trees or 0)
    if max_tree_count > 0 and tree_count > max_tree_count:
        keep = np.linspace(0, tree_count - 1, max_tree_count).round().astype(np.int64)
        keep = np.unique(keep)
        real_leaves = real_leaves[:, keep]
        train_leaves = train_leaves[:, keep]
    cols: dict[str, np.ndarray] = {}
    y_float = np.asarray(train_y, dtype=np.float64)
    for tree_idx in range(real_leaves.shape[1]):
        train_ids = train_leaves[:, tree_idx]
        real_ids = real_leaves[:, tree_idx]
        leaf_values, inverse = np.unique(train_ids, return_inverse=True)
        if leaf_values.size == 0:
            cols[f"leaf_real_rate_{tree_idx:03d}"] = np.full(len(real_ids), 0.5, dtype=np.float32)
            cols[f"leaf_log_count_{tree_idx:03d}"] = np.zeros(len(real_ids), dtype=np.float32)
            continue
        counts = np.bincount(inverse, minlength=len(leaf_values)).astype(np.float64, copy=False)
        real_rate = np.divide(
            np.bincount(inverse, weights=y_float, minlength=len(leaf_values)),
            counts,
            out=np.full(len(leaf_values), 0.5, dtype=np.float64),
            where=counts > 0,
        )
        log_count = np.log1p(counts)
        pos = np.searchsorted(leaf_values, real_ids)
        matched = (pos < len(leaf_values)) & (leaf_values[np.minimum(pos, len(leaf_values) - 1)] == real_ids)
        rate_col = np.full(len(real_ids), 0.5, dtype=np.float32)
        count_col = np.zeros(len(real_ids), dtype=np.float32)
        if bool(np.any(matched)):
            rate_col[matched] = real_rate[pos[matched]].astype(np.float32)
            count_col[matched] = log_count[pos[matched]].astype(np.float32)
        cols[f"leaf_real_rate_{tree_idx:03d}"] = rate_col
        cols[f"leaf_log_count_{tree_idx:03d}"] = count_col
    return pd.DataFrame(cols, dtype=np.float32)


def _bayesian_gmm_model(
    n_components: int,
    *,
    config: AdvancedRegimeLearningConfig,
    random_state: int,
    max_iter: int | None = None,
) -> BayesianGaussianMixture:
    prior = float(config.bayesian_gmm_weight_concentration_prior)
    covariance_type = "diag"
    # Keep the public knob visible for manifests/HPO while enforcing the
    # requested diag-only Bayesian GMM variant.
    if str(config.bayesian_gmm_covariance_type).strip().lower() != "diag":
        covariance_type = "diag"
    kwargs: dict[str, Any] = {}
    if prior > 0.0 and np.isfinite(prior):
        kwargs["weight_concentration_prior"] = prior
    return BayesianGaussianMixture(
        n_components=int(n_components),
        covariance_type=covariance_type,
        weight_concentration_prior_type="dirichlet_process",
        reg_covar=max(float(config.bayesian_gmm_reg_covar), 1e-12),
        random_state=int(random_state),
        max_iter=int(max_iter if max_iter is not None else config.bayesian_gmm_max_iter),
        **kwargs,
    )


def _hdbscan_min_cluster_size(
    n: int,
    k: int,
    config: AdvancedRegimeLearningConfig,
) -> int:
    explicit = int(config.hdbscan_min_cluster_size or 0)
    if explicit > 0:
        return max(2, min(explicit, max(int(n), 2)))
    fraction = float(config.hdbscan_min_cluster_size_fraction or 0.0)
    if fraction > 0.0 and np.isfinite(fraction):
        return max(2, min(int(np.ceil(float(n) * fraction)), max(int(n), 2)))
    return max(5, int(n) // max(10, int(k) * 3))


def _hdbscan_model(
    n: int,
    k: int,
    *,
    config: AdvancedRegimeLearningConfig,
    prediction_data: bool = False,
) -> Any:
    min_cluster_size = _hdbscan_min_cluster_size(n, k, config)
    min_samples = int(config.hdbscan_min_samples or 0)
    kwargs: dict[str, Any] = {
        "min_cluster_size": min_cluster_size,
        "cluster_selection_epsilon": max(float(config.hdbscan_cluster_selection_epsilon), 0.0),
        "cluster_selection_method": (
            str(config.hdbscan_cluster_selection_method)
            if str(config.hdbscan_cluster_selection_method) in {"eom", "leaf"}
            else "eom"
        ),
        "prediction_data": bool(prediction_data),
    }
    if min_samples > 0:
        kwargs["min_samples"] = max(1, min(min_samples, min_cluster_size))
    return hdbscan.HDBSCAN(**kwargs)


def _hmm_model(
    n_components: int,
    *,
    config: AdvancedRegimeLearningConfig,
    random_state: int,
    n_iter: int | None = None,
) -> Any:
    cov_type = str(config.hmm_covariance_type).strip().lower()
    if cov_type not in {"diag", "spherical", "full", "tied"}:
        cov_type = "diag"
    k = int(n_components)
    trans_prior: float | np.ndarray = 1.0
    self_bias = float(config.hmm_transmat_self_bias or 0.0)
    if self_bias > 0.0 and np.isfinite(self_bias):
        trans_prior = np.ones((k, k), dtype=np.float64)
        np.fill_diagonal(trans_prior, 1.0 + self_bias)
    return GaussianHMM(
        n_components=k,
        covariance_type=cov_type,
        n_iter=int(n_iter if n_iter is not None else config.hmm_n_iter),
        tol=max(float(config.hmm_tol), 1e-8),
        min_covar=max(float(config.hmm_min_covar), 1e-12),
        startprob_prior=max(float(config.hmm_startprob_prior), 1e-8),
        transmat_prior=trans_prior,
        random_state=int(random_state),
    )


def _spectral_kwargs(
    n: int,
    *,
    config: AdvancedRegimeLearningConfig,
) -> dict[str, Any]:
    affinity = str(config.spectral_affinity).strip().lower()
    if affinity not in {"nearest_neighbors", "rbf"}:
        affinity = "nearest_neighbors"
    assign_labels = str(config.spectral_assign_labels).strip().lower()
    if assign_labels not in {"kmeans", "discretize", "cluster_qr"}:
        assign_labels = "kmeans"
    kwargs: dict[str, Any] = {
        "affinity": affinity,
        "assign_labels": assign_labels,
    }
    if affinity == "nearest_neighbors":
        kwargs["n_neighbors"] = max(1, min(int(config.spectral_n_neighbors), max(int(n) - 1, 1)))
    else:
        kwargs["gamma"] = max(float(config.spectral_gamma), 1e-8)
    return kwargs


def _kmeans_model(
    n_clusters: int,
    *,
    config: AdvancedRegimeLearningConfig,
    random_state: int,
    n_init: int | None = None,
) -> KMeans:
    algorithm = str(config.kmeans_algorithm).strip().lower()
    if algorithm not in {"lloyd", "elkan"}:
        algorithm = "lloyd"
    return KMeans(
        n_clusters=int(n_clusters),
        random_state=int(random_state),
        n_init=max(1, int(n_init if n_init is not None else config.kmeans_n_init)),
        max_iter=max(1, int(config.kmeans_max_iter)),
        tol=max(float(config.kmeans_tol), 1e-12),
        algorithm=algorithm,
    )


def _reduce_embedding(
    matrix: np.ndarray,
    *,
    method: str,
    n_components: int,
    random_state: int,
    config: AdvancedRegimeLearningConfig = AdvancedRegimeLearningConfig(),
) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float32)
    if x.shape[0] == 0 or x.shape[1] == 0:
        return np.zeros((x.shape[0], 0), dtype=np.float32)
    dim = max(1, min(int(n_components), x.shape[1], max(1, x.shape[0] - 1)))
    if x.shape[0] < 2:
        return np.zeros((x.shape[0], dim), dtype=np.float32)
    if method == "umap" and _UMAP_AVAILABLE and x.shape[0] > 10:
        reducer = umap.UMAP(n_components=dim, random_state=random_state, n_neighbors=min(30, max(2, x.shape[0] // 10)))
        return reducer.fit_transform(x).astype(np.float32)
    if method == "spectral" and x.shape[0] > dim + 2:
        try:
            reducer = SpectralClustering(
                n_clusters=max(2, dim),
                random_state=random_state,
                **_spectral_kwargs(x.shape[0], config=config),
            )
            labels = reducer.fit_predict(x)
            one_hot = np.eye(int(labels.max()) + 1, dtype=np.float32)[labels]
            return PCA(n_components=min(dim, one_hot.shape[1]), random_state=random_state).fit_transform(one_hot).astype(np.float32)
        except Exception:
            pass
    return PCA(n_components=dim, random_state=random_state).fit_transform(x).astype(np.float32)


def _cluster_embedding(
    z: np.ndarray,
    *,
    method: str,
    n_regimes: int,
    random_state: int,
    config: AdvancedRegimeLearningConfig = AdvancedRegimeLearningConfig(),
) -> tuple[np.ndarray, np.ndarray | None, str]:
    x = np.asarray(z, dtype=np.float32)
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64), None, "empty"
    if n < 2:
        return np.zeros(n, dtype=np.int64), np.ones((n, 1), dtype=np.float32), "single_cluster"
    k = max(2, min(int(n_regimes), n))
    if method == "direct":
        probs = np.asarray(x, dtype=np.float32)
        if probs.ndim != 2 or probs.shape[1] == 0:
            return np.zeros(n, dtype=np.int64), None, "direct_empty"
        probs = np.clip(probs, 0.0, np.inf)
        denom = probs.sum(axis=1, keepdims=True)
        probs = np.divide(
            probs,
            np.maximum(denom, 1e-12),
            out=np.full_like(probs, 1.0 / max(probs.shape[1], 1)),
            where=denom > 1e-12,
        ).astype(np.float32)
        return np.argmax(probs, axis=1).astype(np.int64), probs, "direct_probabilities"
    if method == "hdbscan" and _HDBSCAN_AVAILABLE and n >= 10:
        try:
            labels = _hdbscan_model(
                n,
                k,
                config=config,
                prediction_data=False,
            ).fit_predict(x)
            return labels.astype(np.int64), None, "hdbscan"
        except Exception:
            pass
    if method == "hmm" and _HMM_AVAILABLE and n >= k * 4:
        try:
            model = _hmm_model(k, config=config, random_state=random_state)
            model.fit(x)
            labels = model.predict(x)
            probs = model.predict_proba(x).astype(np.float32)
            return labels.astype(np.int64), probs, "gaussian_hmm"
        except Exception:
            pass
    if method == "bayesian_gmm":
        model = _bayesian_gmm_model(k, config=config, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            labels = model.fit_predict(x)
        return labels.astype(np.int64), model.predict_proba(x).astype(np.float32), "bayesian_gmm"
    if method == "spectral" and n >= k + 2:
        try:
            labels = SpectralClustering(
                n_clusters=k,
                random_state=random_state,
                **_spectral_kwargs(n, config=config),
            ).fit_predict(x)
            return labels.astype(np.int64), None, "spectral"
        except Exception:
            pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        labels = _kmeans_model(k, config=config, random_state=random_state).fit_predict(x)
    return labels.astype(np.int64), None, "kmeans"


def minimum_duration_smooth(labels: Sequence[int], min_duration: int = 4) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int64).copy()
    if arr.size == 0 or int(min_duration) <= 1:
        return arr
    n = len(arr)
    start = 0
    while start < n:
        end = start + 1
        while end < n and arr[end] == arr[start]:
            end += 1
        if end - start < int(min_duration):
            prev_label = arr[start - 1] if start > 0 else None
            next_label = arr[end] if end < n else None
            fill = prev_label if prev_label is not None else next_label
            if fill is not None:
                arr[start:end] = fill
        start = end
    return arr


def minimum_duration_smooth_by_frame(
    labels: Sequence[int],
    frame: pd.DataFrame,
    *,
    min_duration: int = 4,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int64).copy()
    if arr.size == 0 or int(min_duration) <= 1 or len(frame) != arr.size:
        return arr
    out = arr.copy()
    for group in _time_symbol_groups(
        frame,
        timestamp_col=timestamp_col,
        symbol_col=symbol_col,
    ):
        if group.size:
            out[group] = minimum_duration_smooth(arr[group], int(min_duration))
    return out


def _label_diagnostics(labels: np.ndarray, z: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return {"regime_count": 0.0, "turnover": 0.0, "persistence": 0.0, "min_support": 0.0, "silhouette": 0.0}
    valid = labels >= 0
    unique, counts = np.unique(labels[valid], return_counts=True)
    turnover = float(np.mean(labels[1:] != labels[:-1])) if labels.size > 1 else 0.0
    support = float(np.min(counts) / max(int(valid.sum()), 1)) if counts.size else 0.0
    sil = 0.0
    if unique.size >= 2 and z.shape[0] == labels.size:
        try:
            sil = float(silhouette_score(z[valid], labels[valid]))
        except Exception:
            sil = 0.0
    return {
        "regime_count": float(unique.size),
        "turnover": turnover,
        "persistence": 1.0 - turnover,
        "min_support": support,
        "silhouette": sil,
    }


def _stability_score(z: np.ndarray, labels: np.ndarray, config: AdvancedRegimeLearningConfig) -> float:
    n = len(labels)
    if n < 10:
        return 0.0
    rng = np.random.default_rng(int(config.random_state) + 1729)
    scores: list[float] = []
    for b in range(min(5, max(1, int(config.stability_bootstraps) // 2))):
        idx = rng.choice(np.arange(n), size=max(5, int(0.7 * n)), replace=False)
        try:
            boot = _kmeans_model(
                max(2, min(int(config.n_regimes), len(idx))),
                config=config,
                random_state=int(config.random_state + b),
                n_init=min(max(1, int(config.kmeans_n_init)), 5),
            ).fit_predict(z[idx])
            scores.append(float(adjusted_mutual_info_score(labels[idx], boot)))
        except Exception:
            continue
    return float(np.nanmean(scores)) if scores else 0.0


def _clamp01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _time_order_positions(frame: pd.DataFrame, *, timestamp_col: str) -> np.ndarray:
    n = len(frame)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if timestamp_col not in frame.columns:
        return np.arange(n, dtype=np.int64)
    ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    valid = ts.notna().to_numpy(dtype=bool)
    ts_ns = ts.to_numpy(dtype="datetime64[ns]").astype("int64")
    pos = np.arange(n, dtype=np.int64)
    valid_pos = pos[valid]
    invalid_pos = pos[~valid]
    if valid_pos.size:
        ordered = valid_pos[np.argsort(ts_ns[valid_pos], kind="mergesort")]
        return np.concatenate([ordered, invalid_pos.astype(np.int64)])
    return pos


def _assessment_blocks(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    n_blocks: int,
    min_rows: int,
) -> list[np.ndarray]:
    order = _time_order_positions(frame, timestamp_col=timestamp_col)
    if order.size == 0:
        return []
    split_count = max(1, min(int(n_blocks or 1), order.size))
    return [
        block.astype(np.int64, copy=False)
        for block in np.array_split(order, split_count)
        if block.size >= int(max(1, min_rows))
    ]


def _assessment_sample_positions(
    frame: pd.DataFrame,
    positions: np.ndarray,
    *,
    config: AdvancedRegimeLearningConfig,
    max_rows: int,
    random_state: int,
) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.int64)
    cap = int(max_rows or 0)
    if cap <= 0 or pos.size <= cap:
        return pos
    try:
        sampled = stratified_period_sample_positions(
            frame,
            pos,
            max_rows=cap,
            timestamp_col=config.timestamp_col,
            symbol_col=config.symbol_col,
            n_periods=int(config.sample_time_bins),
        )
        sampled = np.asarray(sampled, dtype=np.int64)
        if sampled.size:
            return sampled
    except Exception:
        pass
    rng = np.random.default_rng(int(random_state))
    return np.sort(rng.choice(pos, size=cap, replace=False)).astype(np.int64)


def _time_symbol_groups(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    symbol_col: str,
) -> list[np.ndarray]:
    order = time_sort_order(frame, symbol_col=symbol_col, timestamp_col=timestamp_col)
    if order.size == 0:
        return []
    if symbol_col not in frame.columns:
        return [order.astype(np.int64, copy=False)]
    symbols = frame[symbol_col].astype(str).to_numpy()
    ordered_symbols = symbols[order]
    breaks = np.flatnonzero(ordered_symbols[1:] != ordered_symbols[:-1]) + 1
    starts = np.concatenate(
        [
            np.zeros(1, dtype=np.int64),
            breaks.astype(np.int64, copy=False),
            np.asarray([order.size], dtype=np.int64),
        ]
    )
    return [
        order[int(starts[i]) : int(starts[i + 1])].astype(np.int64, copy=False)
        for i in range(len(starts) - 1)
        if int(starts[i + 1]) > int(starts[i])
    ]


def _feature_family_matrix(
    matrix: np.ndarray,
    features: Sequence[str],
    *,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((arr.shape[0] if arr.ndim == 2 else 0, 0), dtype=np.float32)
    include = set(include or set())
    exclude = set(exclude or set())
    families = [_feature_family(str(feature)) for feature in features]
    idx = [
        i
        for i, family in enumerate(families)
        if (not include or family in include)
        and family not in exclude
    ]
    if not idx:
        return np.zeros((arr.shape[0], 0), dtype=np.float32)
    return arr[:, idx].astype(np.float32, copy=False)


def _trend_vol_matrix(matrix: np.ndarray, features: Sequence[str]) -> np.ndarray:
    return _feature_family_matrix(
        matrix,
        features,
        include={"trend", "volatility"},
    )


def _non_trend_vol_matrix(matrix: np.ndarray, features: Sequence[str]) -> np.ndarray:
    return _feature_family_matrix(
        matrix,
        features,
        exclude={"trend", "volatility"},
    )


def _cv_auc_regime_classifier(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    blocks: Sequence[np.ndarray],
    random_state: int,
    max_features: int,
    max_rows: int,
) -> float:
    x = np.asarray(values, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[0] != y.size or x.shape[1] == 0:
        return 0.5
    max_cols = int(max_features or 0)
    if max_cols > 0 and x.shape[1] > max_cols:
        variances = np.nanvar(x, axis=0)
        variances = np.nan_to_num(variances, nan=0.0, posinf=0.0, neginf=0.0)
        keep = np.argpartition(variances, -max_cols)[-max_cols:]
        keep = keep[np.argsort(variances[keep])[::-1]]
        x = x[:, keep].astype(np.float32, copy=False)
    valid = (y >= 0) & np.isfinite(x).all(axis=1)
    classes, counts = np.unique(y[valid], return_counts=True)
    if int(np.sum(valid)) < 12 or classes.size < 2 or int(np.min(counts)) < 2:
        return 0.5
    if len(blocks) < 2:
        return 0.5
    all_pos = np.flatnonzero(valid)
    aucs: list[float] = []
    for fold_i, block in enumerate(blocks):
        test_idx = block[valid[block]]
        if test_idx.size < 3:
            continue
        train_idx = np.setdiff1d(all_pos, test_idx, assume_unique=False)
        train_classes = np.unique(y[train_idx])
        test_classes = np.unique(y[test_idx])
        if train_idx.size < 6 or train_classes.size < 2 or test_classes.size < 2:
            continue
        cap = int(max_rows or 0)
        if cap > 0 and train_idx.size + test_idx.size > cap:
            train_cap = max(6, int(cap * 0.75))
            test_cap = max(3, int(cap - train_cap))
            rng = np.random.default_rng(int(random_state) + 17000 + fold_i)
            if train_idx.size > train_cap:
                train_idx = np.sort(rng.choice(train_idx, size=train_cap, replace=False)).astype(np.int64)
            if test_idx.size > test_cap:
                test_idx = np.sort(rng.choice(test_idx, size=test_cap, replace=False)).astype(np.int64)
        try:
            model = RandomForestClassifier(
                n_estimators=40,
                max_depth=3,
                min_samples_leaf=max(2, int(0.05 * len(train_idx))),
                max_features=None,
                random_state=int(random_state) + fold_i,
                n_jobs=1,
            )
            model.fit(x[train_idx], y[train_idx])
            proba = model.predict_proba(x[test_idx])
            y_test = y[test_idx]
            fold_scores: list[float] = []
            fold_weights: list[float] = []
            for class_idx, cls in enumerate(model.classes_):
                binary = y_test == int(cls)
                positives = int(np.sum(binary))
                negatives = int(binary.size - positives)
                if positives <= 0 or negatives <= 0:
                    continue
                fold_scores.append(float(roc_auc_score(binary.astype(np.int8), proba[:, class_idx])))
                fold_weights.append(float(positives))
            if fold_scores:
                aucs.append(float(np.average(fold_scores, weights=fold_weights)))
        except Exception:
            continue
    if not aucs:
        return 0.5
    return float(np.nanmean(aucs))


def _cv_auc_trend_vol(
    trend_vol: np.ndarray,
    labels: np.ndarray,
    *,
    blocks: Sequence[np.ndarray],
    random_state: int,
    max_features: int,
    max_rows: int,
) -> float:
    return _cv_auc_regime_classifier(
        trend_vol,
        labels,
        blocks=blocks,
        random_state=random_state,
        max_features=max_features,
        max_rows=max_rows,
    )


def _nontriviality_components(
    *,
    auc_tv: float,
    auc_non_trend_vol: float,
    auc_all: float,
) -> dict[str, float]:
    tv_signal = _clamp01((float(auc_tv) - 0.5) / 0.35)
    non_tv_signal = _clamp01((float(auc_non_trend_vol) - 0.5) / 0.35)
    incremental_auc = max(float(auc_all) - max(float(auc_tv), 0.5), 0.0)
    incremental_signal = _clamp01(incremental_auc / 0.15)
    replica_penalty = _clamp01(tv_signal * (1.0 - max(non_tv_signal, incremental_signal)))
    nontriviality = _clamp01(
        0.60 * incremental_signal
        + 0.30 * non_tv_signal
        + 0.10 * (1.0 - replica_penalty)
    )
    return {
        "Incremental_AUC_over_trend_vol": float(incremental_auc),
        "IncrementalNonTriviality": incremental_signal,
        "NonTrendVolSignal": non_tv_signal,
        "TrendVolReplicaPenalty": replica_penalty,
        "NonTriviality": nontriviality,
    }


def _assign_by_centroid(
    train: np.ndarray,
    train_labels: np.ndarray,
    test: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(train_labels, dtype=np.int64)
    states = np.asarray(sorted(int(v) for v in np.unique(labels) if int(v) >= 0), dtype=np.int64)
    if states.size == 0 or test.shape[0] == 0:
        return np.full(test.shape[0], -1, dtype=np.int64)
    centroids = []
    centroid_states = []
    for state in states:
        mask = labels == int(state)
        if int(np.sum(mask)) == 0:
            continue
        centroids.append(np.nanmean(train[mask], axis=0))
        centroid_states.append(int(state))
    if not centroids:
        return np.full(test.shape[0], -1, dtype=np.int64)
    centers = np.nan_to_num(np.vstack(centroids).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x = np.nan_to_num(np.asarray(test, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    dist = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    chosen = np.argmin(dist, axis=1)
    state_arr = np.asarray(centroid_states, dtype=np.int64)
    return state_arr[chosen].astype(np.int64, copy=False)


def _fit_predict_cluster_for_positions(
    z: np.ndarray,
    *,
    method: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_regimes: int,
    random_state: int,
    config: AdvancedRegimeLearningConfig,
) -> np.ndarray:
    x = np.asarray(z, dtype=np.float32)
    if test_idx.size == 0:
        return np.zeros(0, dtype=np.int64)
    if str(method) == "direct":
        labels, _probs, _used = _cluster_embedding(
            x[test_idx],
            method="direct",
            n_regimes=n_regimes,
            random_state=random_state,
            config=config,
        )
        return labels
    train = x[train_idx]
    test = x[test_idx]
    k = max(2, min(int(n_regimes), len(train_idx)))
    if len(train_idx) < k or len(test_idx) == 0:
        labels, _probs, _used = _cluster_embedding(
            test,
            method="kmeans",
            n_regimes=k,
            random_state=random_state,
            config=config,
        )
        return labels
    try:
        if str(method) == "bayesian_gmm":
            model = _bayesian_gmm_model(
                k,
                config=config,
                random_state=int(random_state),
                max_iter=min(int(config.bayesian_gmm_max_iter), 100),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(train)
            return model.predict(test).astype(np.int64)
        if str(method) == "hdbscan" and _HDBSCAN_AVAILABLE and len(train_idx) >= 10:
            model = _hdbscan_model(
                len(train_idx),
                k,
                config=config,
                prediction_data=True,
            )
            train_labels = model.fit_predict(train).astype(np.int64)
            if hasattr(hdbscan, "approximate_predict"):
                pred, _strength = hdbscan.approximate_predict(model, test)
                return np.asarray(pred, dtype=np.int64)
            return _assign_by_centroid(train, train_labels, test)
        if str(method) == "spectral" and len(train_idx) >= k + 2:
            train_labels = SpectralClustering(
                n_clusters=k,
                random_state=int(random_state),
                **_spectral_kwargs(len(train_idx), config=config),
            ).fit_predict(train)
            return _assign_by_centroid(train, np.asarray(train_labels, dtype=np.int64), test)
        if str(method) == "hmm" and _HMM_AVAILABLE and len(train_idx) >= k * 4:
            model = _hmm_model(
                k,
                config=config,
                random_state=int(random_state),
                n_iter=min(int(config.hmm_n_iter), 80),
            )
            model.fit(train)
            return model.predict(test).astype(np.int64)
        model = _kmeans_model(
            k,
            config=config,
            random_state=int(random_state),
            n_init=min(max(1, int(config.kmeans_n_init)), 5),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(train)
        return model.predict(test).astype(np.int64)
    except Exception:
        labels, _probs, _used = _cluster_embedding(
            test,
            method="kmeans",
            n_regimes=k,
            random_state=random_state,
            config=config,
        )
        return labels


def _oos_stability_score(
    z: np.ndarray,
    labels: np.ndarray,
    *,
    cluster_method: str,
    frame: pd.DataFrame,
    blocks: Sequence[np.ndarray],
    config: AdvancedRegimeLearningConfig,
) -> float:
    base = np.asarray(labels, dtype=np.int64)
    if base.size < 12 or np.unique(base[base >= 0]).size < 2:
        return 0.0
    if len(blocks) < 2:
        return 0.0
    all_pos = np.arange(len(base), dtype=np.int64)
    rng = np.random.default_rng(int(config.random_state) + 4200)
    bootstraps = max(1, int(config.regime_assessment_bootstraps))
    scores: list[float] = []
    for fold_i, test_idx in enumerate(blocks):
        train_idx = np.setdiff1d(all_pos, test_idx, assume_unique=False)
        test_use = np.asarray(test_idx, dtype=np.int64)
        cap = int(config.regime_assessment_max_robustness_rows or 0)
        if cap > 0 and train_idx.size + test_use.size > cap:
            rng_fold = np.random.default_rng(int(config.random_state) + 4300 + fold_i)
            train_cap = max(3, int(0.75 * cap))
            test_cap = max(3, cap - train_cap)
            if train_idx.size > train_cap:
                train_idx = np.sort(rng_fold.choice(train_idx, size=train_cap, replace=False)).astype(np.int64)
            if test_use.size > test_cap:
                test_use = np.sort(rng_fold.choice(test_use, size=test_cap, replace=False)).astype(np.int64)
        valid = base[test_use] >= 0
        if int(np.sum(valid)) < 3 or train_idx.size < 3:
            continue
        for boot_i in range(bootstraps):
            if boot_i == 0:
                train_use = train_idx
            else:
                size = max(3, int(np.ceil(0.8 * train_idx.size)))
                replace = train_idx.size < size
                train_use = rng.choice(train_idx, size=size, replace=replace)
                train_use = np.asarray(sorted(np.unique(train_use)), dtype=np.int64)
                if train_use.size < 3:
                    train_use = train_idx
            pred = _fit_predict_cluster_for_positions(
                z,
                method=cluster_method,
                train_idx=train_use,
                test_idx=test_use,
                n_regimes=int(config.n_regimes),
                random_state=int(config.random_state) + 4100 + fold_i * 100 + boot_i,
                config=config,
            )
            pred = minimum_duration_smooth_by_frame(
                pred,
                frame.iloc[test_use],
                min_duration=int(config.min_regime_duration),
                timestamp_col=config.timestamp_col,
                symbol_col=config.symbol_col,
            )
            try:
                scores.append(_clamp01(float(adjusted_rand_score(base[test_use][valid], pred[valid]))))
            except Exception:
                continue
    return float(np.nanmean(scores)) if scores else 0.0


def _dwell_quality(
    labels: np.ndarray,
    frame: pd.DataFrame,
    *,
    min_duration: int,
    timestamp_col: str,
    symbol_col: str,
) -> float:
    arr = np.asarray(labels, dtype=np.int64)
    valid = arr >= 0
    if int(np.sum(valid)) == 0:
        return 0.0
    unique, counts = np.unique(arr[valid], return_counts=True)
    total = float(np.sum(counts))
    groups = _time_symbol_groups(
        frame,
        timestamp_col=timestamp_col,
        symbol_col=symbol_col,
    )
    if not groups:
        groups = [np.arange(arr.size, dtype=np.int64)]
    scores: list[float] = []
    weights: list[float] = []
    for regime, count in zip(unique, counts):
        run_lengths: list[int] = []
        for group in groups:
            values = arr[group]
            start = 0
            while start < values.size:
                while start < values.size and int(values[start]) != int(regime):
                    start += 1
                if start >= values.size:
                    break
                end = start + 1
                while end < values.size and int(values[end]) == int(regime):
                    end += 1
                run_lengths.append(int(end - start))
                start = end
        if not run_lengths:
            continue
        mean_dwell = float(np.mean(run_lengths))
        dwell_score = mean_dwell / (mean_dwell + max(float(min_duration), 1.0))
        scores.append(_clamp01(dwell_score))
        weights.append(float(count) / max(total, 1.0))
    return float(np.average(scores, weights=weights)) if scores else 0.0


def _transition_matrix(
    labels: np.ndarray,
    states: np.ndarray,
    groups: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int64)
    states = np.asarray(states, dtype=np.int64)
    k = int(states.size)
    if k == 0:
        return np.zeros((0, 0), dtype=np.float32)
    pos = {int(state): i for i, state in enumerate(states)}
    counts = np.zeros((k, k), dtype=np.float64)
    if groups is None:
        groups = [np.arange(arr.size, dtype=np.int64)]
    for group in groups:
        values = arr[np.asarray(group, dtype=np.int64)]
        for left, right in zip(values[:-1], values[1:]):
            if int(left) in pos and int(right) in pos:
                counts[pos[int(left)], pos[int(right)]] += 1.0
    denom = counts.sum(axis=1, keepdims=True)
    return np.divide(
        counts,
        np.maximum(denom, 1.0),
        out=np.zeros_like(counts),
        where=denom > 0,
    ).astype(np.float32)


def _transition_stability_score(
    labels: np.ndarray,
    frame: pd.DataFrame,
    *,
    config: AdvancedRegimeLearningConfig,
) -> float:
    arr = np.asarray(labels, dtype=np.int64)
    states = np.asarray(sorted(int(v) for v in np.unique(arr) if int(v) >= 0), dtype=np.int64)
    if states.size < 2 or arr.size < 4:
        return 0.0
    full_groups = _time_symbol_groups(
        frame,
        timestamp_col=config.timestamp_col,
        symbol_col=config.symbol_col,
    )
    full = _transition_matrix(arr, states, groups=full_groups)
    blocks = _assessment_blocks(
        frame,
        timestamp_col=config.timestamp_col,
        n_blocks=int(config.regime_assessment_windows),
        min_rows=max(4, int(config.n_regimes)),
    )
    distances: list[float] = []
    for block in blocks:
        if np.unique(arr[block][arr[block] >= 0]).size < 2:
            continue
        block_frame = frame.iloc[block]
        block_groups = _time_symbol_groups(
            block_frame,
            timestamp_col=config.timestamp_col,
            symbol_col=config.symbol_col,
        )
        block_matrix = _transition_matrix(arr[block], states, groups=block_groups)
        distances.append(float(np.linalg.norm(block_matrix - full, ord="fro")))
    if not distances:
        return 0.0
    tau = max(float(config.regime_assessment_transition_tau), 1e-6)
    return _clamp01(float(np.exp(-float(np.nanmean(distances)) / tau)))


def _top_shift_feature_set(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    labels: np.ndarray,
    positions: np.ndarray,
    *,
    top_n: int,
    eps: float,
) -> set[str]:
    arr = np.asarray(matrix, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    pos = np.asarray(positions, dtype=np.int64)
    if pos.size < 3 or arr.shape[1] == 0:
        return set()
    y = labels[pos]
    x = arr[pos]
    valid = y >= 0
    if int(np.sum(valid)) < 3 or np.unique(y[valid]).size < 2:
        return set()
    y = y[valid]
    x = x[valid]
    global_mean = np.nanmean(x, axis=0)
    global_var = np.nanvar(x, axis=0)
    score = np.zeros(arr.shape[1], dtype=np.float64)
    total = max(len(y), 1)
    for regime in np.unique(y):
        mask = y == regime
        if int(np.sum(mask)) < 2:
            continue
        mean = np.nanmean(x[mask], axis=0)
        score += (float(np.sum(mask)) / total) * (mean - global_mean) ** 2
    score = score / np.maximum(global_var, eps)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    top = min(max(1, int(top_n)), score.size)
    if top < score.size:
        idx = np.argpartition(score, -top)[-top:]
        idx = idx[np.argsort(score[idx])[::-1]]
    else:
        idx = np.argsort(score)[::-1]
    return {str(feature_names[int(i)]) for i in idx if score[int(i)] > 0.0}


def _feature_stability_score(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    labels: np.ndarray,
    frame: pd.DataFrame,
    *,
    config: AdvancedRegimeLearningConfig,
) -> float:
    blocks = _assessment_blocks(
        frame,
        timestamp_col=config.timestamp_col,
        n_blocks=int(config.regime_assessment_windows),
        min_rows=max(4, int(config.n_regimes)),
    )
    sets = [
        _top_shift_feature_set(
            matrix,
            feature_names,
            labels,
            block,
            top_n=int(config.regime_assessment_feature_top_n),
            eps=float(config.eps),
        )
        for block in blocks
    ]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return 0.0
    scores: list[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if not union:
                continue
            scores.append(float(len(sets[i] & sets[j]) / len(union)))
    return float(np.nanmean(scores)) if scores else 0.0


def _cluster_labels_for_assessment(
    z: np.ndarray,
    *,
    cluster_method: str,
    config: AdvancedRegimeLearningConfig,
    random_state: int,
    frame: pd.DataFrame | None = None,
) -> np.ndarray:
    labels, _probs, _used = _cluster_embedding(
        z,
        method=cluster_method,
        n_regimes=int(config.n_regimes),
        random_state=int(random_state),
        config=config,
    )
    if frame is not None and len(frame) == len(labels):
        return minimum_duration_smooth_by_frame(
            labels,
            frame,
            min_duration=int(config.min_regime_duration),
            timestamp_col=config.timestamp_col,
            symbol_col=config.symbol_col,
        )
    return minimum_duration_smooth(labels, int(config.min_regime_duration))


def _null_robustness_score(
    z: np.ndarray,
    labels: np.ndarray,
    *,
    cluster_method: str,
    frame: pd.DataFrame,
    sample_pos: np.ndarray,
    config: AdvancedRegimeLearningConfig,
) -> float:
    pos = np.asarray(sample_pos, dtype=np.int64)
    x = np.asarray(z, dtype=np.float32)[pos]
    base = np.asarray(labels, dtype=np.int64)[pos]
    sub_frame = frame.iloc[pos]
    valid = base >= 0
    if x.shape[0] < 6 or np.unique(base[valid]).size < 2:
        return 0.0
    rng = np.random.default_rng(int(config.random_state) + 5300)
    repeats = max(1, int(config.regime_assessment_null_repeats))
    scale = np.nanstd(x, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0).astype(np.float32)
    real_scores: list[float] = []
    null_scores: list[float] = []
    for i in range(repeats):
        noisy = x + rng.normal(0.0, 0.03, size=x.shape).astype(np.float32) * scale.reshape(1, -1)
        real_labels = _cluster_labels_for_assessment(
            noisy,
            cluster_method=cluster_method,
            config=config,
            random_state=int(config.random_state) + 5400 + i,
            frame=sub_frame,
        )
        try:
            real_scores.append(_clamp01(float(adjusted_rand_score(base[valid], real_labels[valid]))))
        except Exception:
            pass
        null = x.copy()
        for col in range(null.shape[1]):
            null[:, col] = null[rng.permutation(null.shape[0]), col]
        null_labels = _cluster_labels_for_assessment(
            null,
            cluster_method=cluster_method,
            config=config,
            random_state=int(config.random_state) + 5500 + i,
            frame=sub_frame,
        )
        try:
            null_scores.append(_clamp01(float(adjusted_rand_score(base[valid], null_labels[valid]))))
        except Exception:
            pass
    real_mean = float(np.nanmean(real_scores)) if real_scores else 0.0
    null_mean = float(np.nanmean(null_scores)) if null_scores else 1.0
    return _clamp01(real_mean * (1.0 - null_mean))


def _window_robustness_score(
    z: np.ndarray,
    labels: np.ndarray,
    frame: pd.DataFrame,
    *,
    cluster_method: str,
    blocks: Sequence[np.ndarray],
    config: AdvancedRegimeLearningConfig,
) -> float:
    base = np.asarray(labels, dtype=np.int64)
    scores: list[float] = []
    for i, block in enumerate(blocks):
        block = np.asarray(block, dtype=np.int64)
        cap = int(config.regime_assessment_max_robustness_rows or 0)
        if cap > 0 and block.size > cap:
            rng = np.random.default_rng(int(config.random_state) + 5800 + i)
            block = np.sort(rng.choice(block, size=cap, replace=False)).astype(np.int64)
        valid = base[block] >= 0
        if int(np.sum(valid)) < 3 or np.unique(base[block][valid]).size < 2:
            continue
        labels_w = _cluster_labels_for_assessment(
            np.asarray(z, dtype=np.float32)[block],
            cluster_method=cluster_method,
            config=config,
            random_state=int(config.random_state) + 5700 + i,
            frame=frame.iloc[block],
        )
        try:
            scores.append(_clamp01(float(adjusted_rand_score(base[block][valid], labels_w[valid]))))
        except Exception:
            continue
    return float(np.nanmean(scores)) if scores else 0.0


def _corr_matrix(values: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return None
    med = np.nanmedian(arr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    missing = ~np.isfinite(arr)
    if missing.any():
        arr = arr.copy()
        arr[missing] = np.take(med, np.where(missing)[1])
    std = np.nanstd(arr, axis=0)
    if float(np.nanmax(std)) <= 1e-12:
        return None
    centered = arr - np.nanmean(arr, axis=0, keepdims=True)
    scale = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    standardized = centered / scale.reshape(1, -1)
    standardized[:, ~(np.isfinite(std) & (std > 1e-12))] = 0.0
    corr = (standardized.T @ standardized) / max(float(arr.shape[0] - 1), 1.0)
    active = np.isfinite(std) & (std > 1e-12)
    if active.any():
        diag_idx = np.diag_indices_from(corr)
        corr[diag_idx] = active.astype(np.float64)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def _geometry_separation_score(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    labels: np.ndarray,
    *,
    config: AdvancedRegimeLearningConfig,
) -> float:
    arr = np.asarray(matrix, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    valid_regimes = [int(v) for v in np.unique(y) if int(v) >= 0]
    if arr.shape[1] < 2 or len(valid_regimes) < 2:
        return 0.0
    groups: dict[str, list[int]] = {}
    for idx, feature in enumerate(feature_names):
        groups.setdefault(_feature_family(str(feature)), []).append(int(idx))
    min_rows = max(3, int(config.regime_assessment_geometry_min_rows))
    geometry_scores: list[float] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        if len(indices) > 32:
            variances = np.nanvar(arr[:, indices], axis=0)
            order = np.argsort(np.nan_to_num(variances, nan=0.0))[::-1][:32]
            indices = [indices[int(i)] for i in order]
        corr_by_regime: list[np.ndarray] = []
        within_distances: list[float] = []
        for regime in valid_regimes:
            pos = np.flatnonzero(y == regime)
            if pos.size < min_rows:
                continue
            max_rows = int(config.regime_assessment_max_geometry_rows_per_regime or 0)
            if max_rows > 0 and pos.size > max_rows:
                rng = np.random.default_rng(
                    int(config.random_state) + 6800 + int(regime) * 101 + len(indices)
                )
                pos = np.sort(rng.choice(pos, size=max_rows, replace=False)).astype(np.int64)
            values = arr[np.ix_(pos, indices)]
            corr = _corr_matrix(values)
            if corr is None:
                continue
            corr_by_regime.append(corr)
            left = values[::2]
            right = values[1::2]
            corr_left = _corr_matrix(left)
            corr_right = _corr_matrix(right)
            if corr_left is not None and corr_right is not None:
                within_distances.append(float(np.linalg.norm(corr_left - corr_right, ord="fro")))
        if len(corr_by_regime) < 2:
            continue
        between: list[float] = []
        for i in range(len(corr_by_regime)):
            for j in range(i + 1, len(corr_by_regime)):
                between.append(float(np.linalg.norm(corr_by_regime[i] - corr_by_regime[j], ord="fro")))
        between_mean = float(np.nanmean(between)) if between else 0.0
        within_mean = float(np.nanmean(within_distances)) if within_distances else 1.0
        sep = between_mean / max(within_mean, float(config.eps))
        geometry_scores.append(_clamp01(sep / (1.0 + sep)))
    return float(np.nanmean(geometry_scores)) if geometry_scores else 0.0


def _assess_regime_method(
    *,
    method: str,
    cluster_method: str,
    embedding: np.ndarray,
    labels: np.ndarray,
    matrix: np.ndarray,
    feature_names: Sequence[str],
    trend_vol: np.ndarray,
    non_trend_vol: np.ndarray,
    frame: pd.DataFrame,
    oos_blocks: Sequence[np.ndarray],
    window_blocks: Sequence[np.ndarray],
    robustness_positions: np.ndarray,
    config: AdvancedRegimeLearningConfig,
) -> dict[str, float]:
    auc_tv = _cv_auc_trend_vol(
        trend_vol,
        labels,
        blocks=oos_blocks,
        random_state=int(config.random_state) + 6100,
        max_features=int(config.regime_assessment_max_auc_features),
        max_rows=int(config.regime_assessment_max_auc_rows),
    )
    auc_non_trend_vol = _cv_auc_regime_classifier(
        non_trend_vol,
        labels,
        blocks=oos_blocks,
        random_state=int(config.random_state) + 6200,
        max_features=int(config.regime_assessment_max_auc_features),
        max_rows=int(config.regime_assessment_max_auc_rows),
    )
    auc_all = _cv_auc_regime_classifier(
        matrix,
        labels,
        blocks=oos_blocks,
        random_state=int(config.random_state) + 6300,
        max_features=int(config.regime_assessment_max_auc_features),
        max_rows=int(config.regime_assessment_max_auc_rows),
    )
    nontriviality_parts = _nontriviality_components(
        auc_tv=auc_tv,
        auc_non_trend_vol=auc_non_trend_vol,
        auc_all=auc_all,
    )
    nontriviality = float(nontriviality_parts["NonTriviality"])
    oos_stability = _oos_stability_score(
        embedding,
        labels,
        cluster_method=cluster_method,
        frame=frame,
        blocks=oos_blocks,
        config=config,
    )
    dwell = _dwell_quality(
        labels,
        frame,
        min_duration=int(config.min_regime_duration),
        timestamp_col=config.timestamp_col,
        symbol_col=config.symbol_col,
    )
    transition = _transition_stability_score(labels, frame, config=config)
    feature_stability = _feature_stability_score(
        matrix,
        feature_names,
        labels,
        frame,
        config=config,
    )
    null_robustness = _null_robustness_score(
        embedding,
        labels,
        cluster_method=cluster_method,
        frame=frame,
        sample_pos=robustness_positions,
        config=config,
    )
    window_robustness = _window_robustness_score(
        embedding,
        labels,
        frame,
        cluster_method=cluster_method,
        blocks=window_blocks,
        config=config,
    )
    geometry = _geometry_separation_score(
        matrix,
        feature_names,
        labels,
        config=config,
    )
    total = (
        0.20 * nontriviality
        + 0.15 * oos_stability
        + 0.10 * dwell
        + 0.10 * transition
        + 0.15 * feature_stability
        + 0.10 * null_robustness
        + 0.10 * window_robustness
        + 0.10 * geometry
    )
    return {
        "AUC_tv": float(auc_tv),
        "AUC_non_trend_vol": float(auc_non_trend_vol),
        "AUC_all_structure": float(auc_all),
        "Incremental_AUC_over_trend_vol": float(nontriviality_parts["Incremental_AUC_over_trend_vol"]),
        "IncrementalNonTriviality": float(nontriviality_parts["IncrementalNonTriviality"]),
        "NonTrendVolSignal": float(nontriviality_parts["NonTrendVolSignal"]),
        "TrendVolReplicaPenalty": float(nontriviality_parts["TrendVolReplicaPenalty"]),
        "NonTriviality": _clamp01(nontriviality),
        "OOS_Stability": _clamp01(oos_stability),
        "Dwell_Quality": _clamp01(dwell),
        "Transition_Stability": _clamp01(transition),
        "Feature_Stability": _clamp01(feature_stability),
        "Null_Robustness": _clamp01(null_robustness),
        "Window_Robustness": _clamp01(window_robustness),
        "Geometry_Separation": _clamp01(geometry),
        "TotalScore": _clamp01(total),
    }


class MixtureFactorAnalyzer:
    def __init__(
        self,
        n_components: int,
        n_factors: int,
        *,
        max_iter: int = 25,
        l1_lambda: float = 0.0,
        tol: float = 1e-4,
        random_state: int = 42,
        eps: float = 1e-8,
    ) -> None:
        self.n_components = int(n_components)
        self.n_factors = int(n_factors)
        self.max_iter = int(max_iter)
        self.l1_lambda = float(l1_lambda)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.eps = float(eps)

    def fit(self, x: np.ndarray) -> "MixtureFactorAnalyzer":
        arr = np.asarray(x, dtype=np.float64)
        n, p = arr.shape
        k = max(1, min(self.n_components, n))
        q = max(1, min(self.n_factors, p))
        rng = np.random.default_rng(self.random_state)
        labels = KMeans(n_clusters=k, random_state=self.random_state, n_init=5).fit_predict(arr)
        self.pi_ = np.full(k, 1.0 / k)
        self.mu_ = np.vstack([arr[labels == i].mean(axis=0) if np.any(labels == i) else arr[rng.integers(0, n)] for i in range(k)])
        self.lambda_ = rng.normal(0.0, 0.05, size=(k, p, q))
        self.psi_ = np.tile(np.var(arr, axis=0) + self.eps, (k, 1))
        prev_ll = -np.inf
        self.log_likelihood_ = []
        for _it in range(self.max_iter):
            gamma, ef, eff, ll = self._e_step(arr)
            nk = gamma.sum(axis=0) + self.eps
            self.pi_ = nk / n
            for comp in range(k):
                w = gamma[:, comp]
                mu = (w[:, None] * arr).sum(axis=0) / nk[comp]
                self.mu_[comp] = mu
                xc = arr - mu.reshape(1, -1)
                sxf = np.einsum("n,np,nq->pq", w, xc, ef[:, comp, :])
                sff = np.einsum("n,nqr->qr", w, eff[:, comp, :, :])
                lam = sxf @ np.linalg.pinv(sff + self.eps * np.eye(q))
                if self.l1_lambda > 0.0:
                    lam = np.sign(lam) * np.maximum(np.abs(lam) - self.l1_lambda, 0.0)
                self.lambda_[comp] = lam
                recon_cross = 2.0 * np.sum((xc @ lam) * ef[:, comp, :], axis=1)
                quad = np.einsum("ij,njk,ik->n", lam, eff[:, comp, :, :], lam)
                resid = xc * xc - recon_cross[:, None] / max(p, 1)
                psi = (w[:, None] * np.maximum(resid, 0.0)).sum(axis=0) / nk[comp]
                # Blend with exact diagonal expectation for numerical stability.
                diag_quad = np.einsum("jq,nqr,jr->nj", lam, eff[:, comp, :, :], lam)
                psi_exact = (w[:, None] * (xc * xc - 2.0 * xc * (ef[:, comp, :] @ lam.T) + diag_quad)).sum(axis=0) / nk[comp]
                self.psi_[comp] = np.maximum(0.5 * psi + 0.5 * psi_exact, self.eps)
            self.log_likelihood_.append(float(ll))
            if np.isfinite(prev_ll) and abs(ll - prev_ll) < self.tol * max(1.0, abs(prev_ll)):
                break
            prev_ll = ll
        self.responsibilities_, _ef, _eff, _ll = self._e_step(arr)
        return self

    def _e_step(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        n, p = arr.shape
        k, _p, q = self.lambda_.shape
        log_prob = np.empty((n, k), dtype=np.float64)
        ef = np.empty((n, k, q), dtype=np.float64)
        eff = np.empty((n, k, q, q), dtype=np.float64)
        eye_q = np.eye(q)
        for comp in range(k):
            lam = self.lambda_[comp]
            psi = np.maximum(self.psi_[comp], self.eps)
            inv_psi_lam = lam / psi.reshape(-1, 1)
            m = eye_q + lam.T @ inv_psi_lam
            v = np.linalg.pinv(m)
            xc = arr - self.mu_[comp].reshape(1, -1)
            mean_f = (xc / psi.reshape(1, -1)) @ lam @ v
            ef[:, comp, :] = mean_f
            for i in range(n):
                eff[i, comp] = v + np.outer(mean_f[i], mean_f[i])
            sigma = lam @ lam.T + np.diag(psi)
            sign, logdet = np.linalg.slogdet(sigma)
            if sign <= 0:
                logdet = float(np.log(np.maximum(np.linalg.det(sigma + self.eps * np.eye(p)), self.eps)))
            inv_sigma = np.linalg.pinv(sigma)
            quad = np.sum((xc @ inv_sigma) * xc, axis=1)
            log_prob[:, comp] = np.log(max(self.pi_[comp], self.eps)) - 0.5 * (p * np.log(2.0 * np.pi) + logdet + quad)
        norm = logsumexp(log_prob, axis=1)
        gamma = np.exp(log_prob - norm.reshape(-1, 1))
        return gamma, ef, eff, float(np.sum(norm))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        gamma, _ef, _eff, _ll = self._e_step(np.asarray(x, dtype=np.float64))
        return gamma.astype(np.float32)

    def feature_relevance(self, feature_names: Sequence[str]) -> pd.DataFrame:
        rel = np.linalg.norm(self.lambda_, axis=2) / np.sqrt(np.maximum(self.psi_, self.eps))
        rows = []
        for j, feature in enumerate(feature_names):
            vals = rel[:, j]
            rows.append(
                {
                    "feature": str(feature),
                    "mfa_relevance": float(np.max(vals)),
                    **{f"regime_{k}_relevance": float(vals[k]) for k in range(rel.shape[0])},
                }
            )
        return pd.DataFrame(rows).sort_values("mfa_relevance", ascending=False, kind="mergesort")


if _TORCH_AVAILABLE:

    class _SparseAutoEncoder(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, gated: bool = True) -> None:
            super().__init__()
            self.gated = bool(gated)
            self.gate_logits = nn.Parameter(torch.zeros(in_dim)) if gated else None
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, in_dim),
            )

        def gate(self) -> Any:
            if self.gate_logits is None:
                return None
            return torch.sigmoid(self.gate_logits)

        def encode(self, x: Any) -> Any:
            gate = self.gate()
            if gate is not None:
                x = x * gate.reshape(1, -1)
            return self.encoder(x)

        def forward(self, x: Any) -> tuple[Any, Any]:
            z = self.encode(x)
            return self.decoder(z), z


def _sigmoid(x: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(x, dtype=np.float32), -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-arr))).astype(np.float32)


def _family_column_groups(feature_names: Sequence[str] | None, p: int) -> list[np.ndarray]:
    if feature_names:
        families = np.asarray([_feature_family(str(f)) for f in feature_names[:p]])
        if families.size < p:
            families = np.concatenate(
                [families, np.repeat("latent", p - families.size)]
            )
        return [np.flatnonzero(families == family) for family in pd.unique(families)]
    return [np.arange(p, dtype=np.int64)]


def _row_normalize(z: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    norm = np.maximum(norm, eps).astype(np.float32)
    return (z / norm).astype(np.float32), norm


def _row_normalize_backward(
    grad_normalized: np.ndarray,
    normalized: np.ndarray,
    norm: np.ndarray,
) -> np.ndarray:
    dot = np.sum(grad_normalized * normalized, axis=1, keepdims=True)
    return ((grad_normalized - normalized * dot) / np.maximum(norm, 1e-8)).astype(np.float32)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.nanmax(logits, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -60.0, 60.0))
    denom = np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
    return (exp / denom).astype(np.float32)


def _ae_init_params(
    in_dim: int,
    hidden_dim: int,
    latent_dim: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    hidden = max(2, int(hidden_dim))
    latent = max(1, min(int(latent_dim), in_dim))

    def weight(rows: int, cols: int) -> np.ndarray:
        scale = np.sqrt(2.0 / max(rows + cols, 1))
        return rng.normal(0.0, scale, size=(rows, cols)).astype(np.float32)

    return {
        "gate_logits": np.zeros(in_dim, dtype=np.float32),
        "w1": weight(in_dim, hidden),
        "b1": np.zeros(hidden, dtype=np.float32),
        "w2": weight(hidden, latent),
        "b2": np.zeros(latent, dtype=np.float32),
        "w3": weight(latent, hidden),
        "b3": np.zeros(hidden, dtype=np.float32),
        "w4": weight(hidden, in_dim),
        "b4": np.zeros(in_dim, dtype=np.float32),
    }


def _ae_forward(
    params: Mapping[str, np.ndarray],
    x: np.ndarray,
    *,
    decode: bool,
) -> tuple[np.ndarray | None, np.ndarray, dict[str, np.ndarray]]:
    gate = _sigmoid(params["gate_logits"])
    xg = x * gate.reshape(1, -1)
    h1_pre = xg @ params["w1"] + params["b1"].reshape(1, -1)
    h1 = np.maximum(h1_pre, 0.0).astype(np.float32)
    z = (h1 @ params["w2"] + params["b2"].reshape(1, -1)).astype(np.float32)
    cache = {
        "x": x,
        "gate": gate,
        "xg": xg,
        "h1_pre": h1_pre,
        "h1": h1,
        "z": z,
    }
    if not decode:
        return None, z, cache
    h2_pre = z @ params["w3"] + params["b3"].reshape(1, -1)
    h2 = np.maximum(h2_pre, 0.0).astype(np.float32)
    recon = (h2 @ params["w4"] + params["b4"].reshape(1, -1)).astype(np.float32)
    cache.update({"h2_pre": h2_pre, "h2": h2})
    return recon, z, cache


def _ae_encode_numpy_batches(
    params: Mapping[str, np.ndarray],
    x: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    latent_dim = int(params["b2"].shape[0])
    out = np.empty((arr.shape[0], latent_dim), dtype=np.float32)
    batch = max(256, int(batch_size or 256))
    for start in range(0, arr.shape[0], batch):
        end = min(start + batch, arr.shape[0])
        _recon, z, _cache = _ae_forward(params, arr[start:end], decode=False)
        out[start:end] = z.astype(np.float32, copy=False)
    return out


def _ae_empty_grads(params: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.zeros_like(value, dtype=np.float32) for key, value in params.items()}


def _ae_backward(
    params: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    grads: dict[str, np.ndarray],
    *,
    grad_recon: np.ndarray | None = None,
    grad_z_extra: np.ndarray | None = None,
) -> None:
    grad_z = np.zeros_like(cache["z"], dtype=np.float32)
    if grad_recon is not None:
        h2 = cache["h2"]
        z = cache["z"]
        grads["w4"] += h2.T @ grad_recon
        grads["b4"] += grad_recon.sum(axis=0)
        grad_h2 = grad_recon @ params["w4"].T
        grad_h2_pre = grad_h2 * (cache["h2_pre"] > 0.0)
        grads["w3"] += z.T @ grad_h2_pre
        grads["b3"] += grad_h2_pre.sum(axis=0)
        grad_z += grad_h2_pre @ params["w3"].T
    if grad_z_extra is not None:
        grad_z += grad_z_extra.astype(np.float32, copy=False)
    if not np.any(grad_z):
        return
    h1 = cache["h1"]
    grads["w2"] += h1.T @ grad_z
    grads["b2"] += grad_z.sum(axis=0)
    grad_h1 = grad_z @ params["w2"].T
    grad_h1_pre = grad_h1 * (cache["h1_pre"] > 0.0)
    grads["w1"] += cache["xg"].T @ grad_h1_pre
    grads["b1"] += grad_h1_pre.sum(axis=0)
    grad_xg = grad_h1_pre @ params["w1"].T
    gate = cache["gate"]
    grads["gate_logits"] += np.sum(grad_xg * cache["x"], axis=0) * gate * (1.0 - gate)


def _apply_grads(
    params: dict[str, np.ndarray],
    grads: Mapping[str, np.ndarray],
    *,
    learning_rate: float,
    weight_decay: float,
) -> None:
    total_sq = 0.0
    for grad in grads.values():
        total_sq += float(np.sum(np.asarray(grad, dtype=np.float64) ** 2))
    clip = 1.0
    norm = np.sqrt(max(total_sq, 0.0))
    if np.isfinite(norm) and norm > 10.0:
        clip = 10.0 / max(norm, 1e-12)
    for key, value in params.items():
        grad = np.asarray(grads[key], dtype=np.float32) * np.float32(clip)
        if key.startswith("w") and float(weight_decay) > 0.0:
            grad = grad + np.float32(weight_decay) * value
        value -= np.float32(learning_rate) * grad


def _augment_positive_view(
    xb: np.ndarray,
    train_x: np.ndarray,
    train_positions: np.ndarray,
    *,
    feature_names: Sequence[str] | None,
    buckets: np.ndarray | None,
    rng: np.random.Generator,
    config: AdvancedRegimeLearningConfig,
) -> np.ndarray:
    out = xb.astype(np.float32, copy=True)
    if float(config.ae_noise) > 0.0:
        out += np.float32(config.ae_noise) * rng.normal(0.0, 1.0, size=out.shape).astype(np.float32)
    if float(config.ae_dropout) > 0.0:
        keep = rng.random(out.shape) >= float(config.ae_dropout)
        out *= keep.astype(np.float32)
    groups = _family_column_groups(feature_names, out.shape[1])
    mask_rate = float(np.clip(config.ae_family_mask_rate, 0.0, 1.0))
    if mask_rate > 0.0 and len(groups) > 1:
        for cols in groups:
            if cols.size and rng.random() < mask_rate:
                out[:, cols] = 0.0
    if buckets is not None and len(train_positions) == len(out) and train_x.shape[0] == len(buckets):
        for bucket in np.unique(buckets[train_positions]):
            local = np.flatnonzero(buckets[train_positions] == bucket)
            global_peer_pool = np.flatnonzero(buckets == bucket)
            if local.size > 0 and global_peer_pool.size > 1:
                peers = rng.choice(global_peer_pool, size=local.size, replace=True)
                out[local] = 0.95 * out[local] + 0.05 * train_x[peers]
    return np.clip(out, -10.0, 10.0).astype(np.float32)


def _make_negative_view(
    xb: np.ndarray,
    train_x: np.ndarray,
    train_positions: np.ndarray,
    *,
    feature_names: Sequence[str] | None,
    buckets: np.ndarray | None,
    rng: np.random.Generator,
    config: AdvancedRegimeLearningConfig,
) -> np.ndarray:
    out = xb.astype(np.float32, copy=True)
    p = out.shape[1]
    groups = _family_column_groups(feature_names, p)
    mode = int(rng.integers(0, 4))
    if mode == 0:
        for j in range(p):
            out[:, j] = out[rng.permutation(len(out)), j]
    elif mode == 1:
        for cols in groups:
            if cols.size:
                out[:, cols] = out[rng.permutation(len(out))][:, cols]
    elif mode == 2:
        for cols in groups:
            if cols.size and len(out) > 1:
                lag = int(rng.integers(1, len(out)))
                out[:, cols] = np.roll(out[:, cols], lag, axis=0)
    else:
        if buckets is not None and train_x.shape[0] == len(buckets) and len(train_positions) == len(out):
            for i, pos in enumerate(train_positions):
                other = np.flatnonzero(buckets != buckets[int(pos)])
                if other.size == 0:
                    other = np.arange(train_x.shape[0], dtype=np.int64)
                out[i] = train_x[int(rng.choice(other))]
        else:
            idx = _block_shuffle_indices(train_x.shape[0], int(config.null_block_size), rng)
            out = train_x[_take_positions_with_replacement(idx, len(out), rng)].astype(np.float32, copy=True)
    if float(config.ae_noise) > 0.0:
        out += np.float32(config.ae_noise) * rng.normal(0.0, 1.0, size=out.shape).astype(np.float32)
    return np.clip(out, -10.0, 10.0).astype(np.float32)


def _train_numpy_autoencoder(
    matrix: np.ndarray,
    *,
    config: AdvancedRegimeLearningConfig,
    contrastive: bool,
    random_state: int,
    feature_names: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x = np.asarray(matrix, dtype=np.float32)
    rng = np.random.default_rng(int(random_state))
    n, p = x.shape
    max_train = int(config.ae_max_train_rows or 0)
    if max_train > 0 and n > max_train:
        train_positions = np.sort(rng.choice(np.arange(n), size=max_train, replace=False)).astype(np.int64)
    else:
        train_positions = np.arange(n, dtype=np.int64)
    train_x = x[train_positions].astype(np.float32, copy=True)
    train_features = list(feature_names or [])
    try:
        buckets = _bucket_ids(train_x, train_features) if train_x.shape[1] else None
    except Exception:
        buckets = None
    params = _ae_init_params(
        p,
        int(config.ae_hidden_dim),
        int(config.ae_latent_dim),
        rng,
    )
    batch = max(8, min(int(config.ae_batch_size), len(train_x)))
    lr = float(config.ae_learning_rate)
    if lr <= 0.0 or not np.isfinite(lr):
        lr = 1e-3
    losses: list[float] = []
    for epoch in range(max(1, int(config.ae_epochs))):
        order = rng.permutation(len(train_x))
        epoch_loss = 0.0
        epoch_batches = 0
        for start in range(0, len(order), batch):
            local_idx = order[start : start + batch]
            xb = train_x[local_idx]
            batch_positions = local_idx.astype(np.int64)
            view1 = _augment_positive_view(
                xb,
                train_x,
                batch_positions,
                feature_names=train_features,
                buckets=buckets,
                rng=rng,
                config=config,
            )
            recon, z, cache = _ae_forward(params, view1, decode=True)
            assert recon is not None
            denom = max(float(xb.shape[0] * xb.shape[1]), 1.0)
            diff = recon - xb
            grad_recon = (2.0 / denom) * diff
            grad_z = (
                float(config.ae_lambda_sparse)
                * np.sign(z).astype(np.float32)
                / max(float(np.prod(z.shape)), 1.0)
            )
            grads = _ae_empty_grads(params)
            _ae_backward(params, cache, grads, grad_recon=grad_recon.astype(np.float32), grad_z_extra=grad_z)
            loss = float(np.mean(diff * diff)) + float(config.ae_lambda_sparse) * float(np.mean(np.abs(z)))
            gate = _sigmoid(params["gate_logits"])
            grads["gate_logits"] += (
                float(config.ae_lambda_sparse)
                * np.sign(gate).astype(np.float32)
                * gate
                * (1.0 - gate)
                / max(float(len(gate)), 1.0)
            )
            if contrastive and len(xb) > 2:
                view2 = _augment_positive_view(
                    xb,
                    train_x,
                    batch_positions,
                    feature_names=train_features,
                    buckets=buckets,
                    rng=rng,
                    config=config,
                )
                neg = _make_negative_view(
                    xb,
                    train_x,
                    batch_positions,
                    feature_names=train_features,
                    buckets=buckets,
                    rng=rng,
                    config=config,
                )
                _r1, z1, c1 = _ae_forward(params, view1, decode=False)
                _r2, z2, c2 = _ae_forward(params, view2, decode=False)
                _rn, zn, cn = _ae_forward(params, neg, decode=False)
                z1n, n1 = _row_normalize(z1, float(config.eps))
                z2n, n2 = _row_normalize(z2, float(config.eps))
                znn, nnorm = _row_normalize(zn, float(config.eps))
                keys = np.concatenate([z2n, znn], axis=0).astype(np.float32)
                tau = max(float(config.ae_temperature), 1e-3)
                logits = (z1n @ keys.T) / np.float32(tau)
                probs = _softmax_rows(logits)
                labels = np.arange(len(z1n), dtype=np.int64)
                contrast_loss = -float(np.mean(np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-12))))
                grad_logits = probs
                grad_logits[np.arange(len(labels)), labels] -= 1.0
                grad_logits *= np.float32(float(config.ae_lambda_contrastive) / max(len(labels), 1) / tau)
                grad_z1n = grad_logits @ keys
                grad_keys = grad_logits.T @ z1n
                grad_z2n = grad_keys[: len(z2n)]
                grad_znn = grad_keys[len(z2n) :]
                _ae_backward(
                    params,
                    c1,
                    grads,
                    grad_z_extra=_row_normalize_backward(grad_z1n, z1n, n1),
                )
                _ae_backward(
                    params,
                    c2,
                    grads,
                    grad_z_extra=_row_normalize_backward(grad_z2n, z2n, n2),
                )
                _ae_backward(
                    params,
                    cn,
                    grads,
                    grad_z_extra=_row_normalize_backward(grad_znn, znn, nnorm),
                )
                loss += float(config.ae_lambda_contrastive) * contrast_loss
            if float(config.ae_lambda_smooth) > 0.0 and len(local_idx) > 2:
                smooth_idx = np.sort(local_idx)
                _rs, zseq, cseq = _ae_forward(params, train_x[smooth_idx], decode=False)
                delta = zseq[1:] - zseq[:-1]
                grad_seq = np.zeros_like(zseq, dtype=np.float32)
                scale = (
                    2.0
                    * float(config.ae_lambda_smooth)
                    / max(float(delta.size), 1.0)
                )
                grad_seq[1:] += scale * delta
                grad_seq[:-1] -= scale * delta
                _ae_backward(params, cseq, grads, grad_z_extra=grad_seq)
                loss += float(config.ae_lambda_smooth) * float(np.mean(delta * delta))
            _apply_grads(
                params,
                grads,
                learning_rate=lr,
                weight_decay=float(config.ae_weight_decay),
            )
            epoch_loss += loss
            epoch_batches += 1
        losses.append(float(epoch_loss / max(epoch_batches, 1)))
        if epoch >= 3 and len(losses) >= 4:
            prev = losses[-2]
            if np.isfinite(prev) and abs(prev - losses[-1]) < 1e-5 * max(1.0, abs(prev)):
                break
    z_all = _ae_encode_numpy_batches(
        params,
        x,
        batch_size=max(int(config.ae_batch_size), 1024),
    )
    gate = _sigmoid(params["gate_logits"]).astype(np.float32)
    return z_all.astype(np.float32), gate, {
        "enabled": True,
        "backend": "numpy",
        "contrastive": bool(contrastive),
        "epochs": int(len(losses)),
        "train_rows": int(len(train_x)),
        "loss_final": float(losses[-1]) if losses else float("nan"),
        "loss_initial": float(losses[0]) if losses else float("nan"),
    }


def _train_autoencoder(
    matrix: np.ndarray,
    *,
    config: AdvancedRegimeLearningConfig,
    contrastive: bool,
    random_state: int,
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    x = np.asarray(matrix, dtype=np.float32)
    if x.shape[0] == 0 or x.shape[1] == 0:
        return np.zeros((x.shape[0], 0), dtype=np.float32), None, {"enabled": False, "reason": "empty"}
    backend = str(config.ae_backend or "numpy").strip().lower()
    if backend != "torch" or not bool(config.ae_torch_enabled):
        return _train_numpy_autoencoder(
            x,
            config=config,
            contrastive=contrastive,
            random_state=random_state,
            feature_names=feature_names,
        )
    if not _TORCH_AVAILABLE:
        return _train_numpy_autoencoder(
            x,
            config=config,
            contrastive=contrastive,
            random_state=random_state,
            feature_names=feature_names,
        )
    torch.manual_seed(int(random_state))
    device = torch.device("cpu")
    rng = np.random.default_rng(int(random_state))
    max_train = int(config.ae_max_train_rows or 0)
    if max_train > 0 and x.shape[0] > max_train:
        train_positions = np.sort(rng.choice(np.arange(x.shape[0]), size=max_train, replace=False)).astype(np.int64)
    else:
        train_positions = np.arange(x.shape[0], dtype=np.int64)
    train_x = x[train_positions].astype(np.float32, copy=False)
    model = _SparseAutoEncoder(
        x.shape[1],
        int(config.ae_hidden_dim),
        min(int(config.ae_latent_dim), x.shape[1]),
        gated=True,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(config.ae_learning_rate))
    data = torch.tensor(train_x, dtype=torch.float32, device=device)
    batch = max(16, min(int(config.ae_batch_size), train_x.shape[0]))
    for _epoch in range(max(1, int(config.ae_epochs))):
        order = rng.permutation(train_x.shape[0])
        for start in range(0, len(order), batch):
            idx = torch.tensor(order[start : start + batch], dtype=torch.long, device=device)
            xb = data[idx]
            noise = float(config.ae_noise)
            drop = float(config.ae_dropout)
            view1 = xb + noise * torch.randn_like(xb)
            if drop > 0.0:
                view1 = view1 * (torch.rand_like(view1) > drop).float()
            recon, z = model(view1)
            loss = F.mse_loss(recon, xb) + float(config.ae_lambda_sparse) * torch.mean(torch.abs(z))
            gate = model.gate()
            if gate is not None:
                loss = loss + float(config.ae_lambda_sparse) * torch.mean(torch.abs(gate))
            if contrastive and xb.shape[0] > 2:
                view2 = xb + noise * torch.randn_like(xb)
                if drop > 0.0:
                    view2 = view2 * (torch.rand_like(view2) > drop).float()
                z1 = F.normalize(model.encode(view1), dim=1)
                z2 = F.normalize(model.encode(view2), dim=1)
                logits = z1 @ z2.T / max(float(config.ae_temperature), 1e-3)
                labels = torch.arange(logits.shape[0], device=device)
                loss = loss + float(config.ae_lambda_contrastive) * F.cross_entropy(logits, labels)
                if z.shape[0] > 1:
                    loss = loss + float(config.ae_lambda_smooth) * torch.mean((z[1:] - z[:-1]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        chunks: list[np.ndarray] = []
        encode_batch = max(batch, 1024)
        for start in range(0, x.shape[0], encode_batch):
            xb = torch.tensor(x[start : start + encode_batch], dtype=torch.float32, device=device)
            chunks.append(model.encode(xb).cpu().numpy().astype(np.float32))
        z = np.vstack(chunks).astype(np.float32, copy=False) if chunks else np.zeros((0, 0), dtype=np.float32)
        gate = model.gate().cpu().numpy().astype(np.float32) if model.gate() is not None else None
    return z, gate, {
        "enabled": True,
        "contrastive": bool(contrastive),
        "epochs": int(config.ae_epochs),
        "train_rows": int(len(train_x)),
        "encode_batch_rows": int(max(batch, 1024)),
    }


def _build_probability_frame(index: pd.Index, method: str, probs: np.ndarray | None, labels: np.ndarray) -> pd.DataFrame:
    if probs is None or probs.ndim != 2:
        unique = sorted(int(v) for v in np.unique(labels) if int(v) >= 0)
        probs = np.zeros((len(labels), len(unique)), dtype=np.float32)
        for i, label in enumerate(unique):
            probs[:, i] = labels == label
    return pd.DataFrame(
        {f"{method}_regime_prob_{i:02d}": probs[:, i].astype(np.float32) for i in range(probs.shape[1])},
        index=index,
    )


def _embedding_frame(index: pd.Index, prefix: str, matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {f"{prefix}_{i:02d}": matrix[:, i].astype(np.float32) for i in range(matrix.shape[1])},
        index=index,
    )


def _method_probability_columns(columns: Sequence[str], method: str) -> list[str]:
    prefix = f"{method}_regime_prob_"
    return [str(col) for col in columns if str(col).startswith(prefix)]


def _probability_matrix(probabilities: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    if not cols:
        return np.ones((len(probabilities), 1), dtype=np.float32)
    raw = probabilities.loc[:, list(cols)].to_numpy(dtype=np.float32, copy=False)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.clip(raw, 0.0, np.inf)
    denom = raw.sum(axis=1, keepdims=True)
    return np.divide(
        raw,
        np.maximum(denom, 1e-12),
        out=np.full_like(raw, 1.0 / max(raw.shape[1], 1), dtype=np.float32),
        where=denom > 1e-12,
    ).astype(np.float32)


def _label_array(values: pd.Series | np.ndarray | Sequence[Any]) -> np.ndarray:
    try:
        return np.asarray(values, dtype=np.int64)
    except Exception:
        return pd.to_numeric(values, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)


def _lagged_positions(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    symbol_col: str,
    horizon_hours: float,
) -> np.ndarray:
    n = len(frame)
    out = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return out
    order = time_sort_order(frame, symbol_col=symbol_col, timestamp_col=timestamp_col)
    if timestamp_col in frame.columns:
        ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
        valid_ts = ts.notna().to_numpy(dtype=bool)
        ts_ns = ts.to_numpy(dtype="datetime64[ns]").astype("int64")
    else:
        valid_ts = np.zeros(n, dtype=bool)
        ts_ns = np.zeros(n, dtype=np.int64)
    if symbol_col in frame.columns:
        symbols = frame[symbol_col].astype(str).to_numpy()
    else:
        symbols = np.repeat("__all__", n)
    step = max(1, int(round(float(horizon_hours))))
    for symbol in pd.unique(symbols[order]):
        pos = order[symbols[order] == symbol]
        if pos.size == 0:
            continue
        valid_pos = pos[valid_ts[pos]]
        if valid_pos.size:
            sort = np.argsort(ts_ns[valid_pos], kind="mergesort")
            sorted_pos = valid_pos[sort]
            sorted_ts = ts_ns[sorted_pos]
            delta = int(round(float(horizon_hours) * 3600.0 * 1_000_000_000.0))
            targets = sorted_ts - delta
            lag_local = np.searchsorted(sorted_ts, targets, side="right") - 1
            ok = lag_local >= 0
            out[sorted_pos[ok]] = sorted_pos[lag_local[ok]]
            invalid_pos = pos[~valid_ts[pos]]
            if invalid_pos.size:
                for i in range(step, invalid_pos.size):
                    out[invalid_pos[i]] = invalid_pos[i - step]
        else:
            for i in range(step, pos.size):
                out[pos[i]] = pos[i - step]
    return out


def _transition_duration_arrays(
    frame: pd.DataFrame,
    labels: np.ndarray,
    prob_max: np.ndarray,
    *,
    timestamp_col: str,
    symbol_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(labels)
    hazard = np.zeros(n, dtype=np.float32)
    time_since = np.zeros(n, dtype=np.float32)
    expected_duration = np.ones(n, dtype=np.float32)
    if n == 0:
        return hazard, time_since, expected_duration
    order = time_sort_order(frame, symbol_col=symbol_col, timestamp_col=timestamp_col)
    if symbol_col in frame.columns:
        symbols = frame[symbol_col].astype(str).to_numpy()
    else:
        symbols = np.repeat("__all__", n)
    ordered_symbol_codes = pd.factorize(symbols[order], sort=False)[0].astype(np.int64, copy=False)
    if timestamp_col in frame.columns:
        ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
        ts_ns = ts.to_numpy(dtype="datetime64[ns]").astype("int64")
        valid_ts = ts.notna().to_numpy(dtype=bool)
    else:
        ts_ns = np.zeros(n, dtype=np.int64)
        valid_ts = np.zeros(n, dtype=bool)
    if _NUMBA_AVAILABLE:
        return _transition_duration_arrays_numba(
            np.asarray(labels, dtype=np.int64),
            np.asarray(prob_max, dtype=np.float32),
            order.astype(np.int64, copy=False),
            ordered_symbol_codes,
            ts_ns.astype(np.int64, copy=False),
            valid_ts.astype(np.bool_, copy=False),
        )
    alpha = 2.0 / (24.0 + 1.0)
    for code in pd.unique(ordered_symbol_codes):
        pos = order[ordered_symbol_codes == code]
        if pos.size == 0:
            continue
        prev_label: int | None = None
        run_start_pos = int(pos[0])
        run_start_i = 0
        transition_rate = 0.0
        completed_sum: dict[int, float] = {}
        completed_count: dict[int, int] = {}
        for local_i, row_pos_raw in enumerate(pos):
            row_pos = int(row_pos_raw)
            label = int(labels[row_pos])
            changed = bool(prev_label is not None and label != prev_label)
            if changed and prev_label is not None:
                if valid_ts[row_pos] and valid_ts[run_start_pos]:
                    duration = max(
                        float((ts_ns[row_pos] - ts_ns[run_start_pos]) / 3_600_000_000_000.0),
                        0.0,
                    )
                else:
                    duration = float(max(local_i - run_start_i, 1))
                completed_sum[prev_label] = completed_sum.get(prev_label, 0.0) + duration
                completed_count[prev_label] = completed_count.get(prev_label, 0) + 1
                run_start_pos = row_pos
                run_start_i = local_i
            if valid_ts[row_pos] and valid_ts[run_start_pos]:
                elapsed = max(
                    float((ts_ns[row_pos] - ts_ns[run_start_pos]) / 3_600_000_000_000.0),
                    0.0,
                )
            else:
                elapsed = float(max(local_i - run_start_i, 0))
            count = completed_count.get(label, 0)
            if count > 0:
                expected = completed_sum.get(label, 0.0) / float(count)
            else:
                expected = max(elapsed, 1.0)
            hazard[row_pos] = np.float32(
                np.clip(0.5 * transition_rate + 0.5 * (1.0 - float(prob_max[row_pos])), 0.0, 1.0)
            )
            time_since[row_pos] = np.float32(elapsed)
            expected_duration[row_pos] = np.float32(max(expected, 1e-6))
            transition_rate = (1.0 - alpha) * transition_rate + alpha * float(changed)
            prev_label = label
    return hazard, time_since, expected_duration


@_numba_njit(cache=True)
def _transition_duration_arrays_numba(
    labels: np.ndarray,
    prob_max: np.ndarray,
    order: np.ndarray,
    ordered_symbol_codes: np.ndarray,
    ts_ns: np.ndarray,
    valid_ts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = labels.shape[0]
    hazard = np.zeros(n, dtype=np.float32)
    time_since = np.zeros(n, dtype=np.float32)
    expected_duration = np.ones(n, dtype=np.float32)
    if n == 0:
        return hazard, time_since, expected_duration
    max_label = -1
    for i in range(n):
        if labels[i] > max_label:
            max_label = labels[i]
    label_count = max(max_label + 2, 1)
    alpha = 2.0 / 25.0
    start = 0
    while start < order.shape[0]:
        end = start + 1
        code = ordered_symbol_codes[start]
        while end < order.shape[0] and ordered_symbol_codes[end] == code:
            end += 1
        completed_sum = np.zeros(label_count, dtype=np.float64)
        completed_count = np.zeros(label_count, dtype=np.int64)
        prev_label = 0
        prev_label_index = -1
        has_prev = False
        run_start_pos = int(order[start])
        run_start_i = 0
        transition_rate = 0.0
        for absolute_i in range(start, end):
            local_i = absolute_i - start
            row_pos = int(order[absolute_i])
            label = int(labels[row_pos])
            label_index = label + 1 if label >= 0 else 0
            changed = has_prev and label != prev_label
            if changed:
                if valid_ts[row_pos] and valid_ts[run_start_pos]:
                    duration = (float(ts_ns[row_pos]) - float(ts_ns[run_start_pos])) / 3_600_000_000_000.0
                    if duration < 0.0:
                        duration = 0.0
                else:
                    duration = float(local_i - run_start_i)
                    if duration < 1.0:
                        duration = 1.0
                if prev_label_index >= 0 and prev_label_index < label_count:
                    completed_sum[prev_label_index] += duration
                    completed_count[prev_label_index] += 1
                run_start_pos = row_pos
                run_start_i = local_i
            if valid_ts[row_pos] and valid_ts[run_start_pos]:
                elapsed = (float(ts_ns[row_pos]) - float(ts_ns[run_start_pos])) / 3_600_000_000_000.0
                if elapsed < 0.0:
                    elapsed = 0.0
            else:
                elapsed = float(local_i - run_start_i)
                if elapsed < 0.0:
                    elapsed = 0.0
            if label_index < label_count and completed_count[label_index] > 0:
                expected = completed_sum[label_index] / float(completed_count[label_index])
            else:
                expected = elapsed
                if expected < 1.0:
                    expected = 1.0
            haz = 0.5 * transition_rate + 0.5 * (1.0 - float(prob_max[row_pos]))
            if haz < 0.0:
                haz = 0.0
            elif haz > 1.0:
                haz = 1.0
            hazard[row_pos] = np.float32(haz)
            time_since[row_pos] = np.float32(elapsed)
            if expected < 1e-6:
                expected = 1e-6
            expected_duration[row_pos] = np.float32(expected)
            transition_rate = (1.0 - alpha) * transition_rate + alpha * (1.0 if changed else 0.0)
            prev_label = label
            prev_label_index = label_index
            has_prev = True
        start = end
    return hazard, time_since, expected_duration


def _build_regime_transition_features(
    frame: pd.DataFrame,
    regime_probabilities: pd.DataFrame,
    regime_labels: pd.DataFrame,
    methods: Sequence[str],
    *,
    label_arrays: Mapping[str, np.ndarray] | None = None,
    config: AdvancedRegimeLearningConfig,
) -> pd.DataFrame:
    if len(frame) == 0 or not methods:
        return pd.DataFrame(index=frame.index)
    cols: dict[str, np.ndarray] = {}
    lag_positions = {
        1: _lagged_positions(
            frame,
            timestamp_col=config.timestamp_col,
            symbol_col=config.symbol_col,
            horizon_hours=1.0,
        ),
        4: _lagged_positions(
            frame,
            timestamp_col=config.timestamp_col,
            symbol_col=config.symbol_col,
            horizon_hours=4.0,
        ),
        24: _lagged_positions(
            frame,
            timestamp_col=config.timestamp_col,
            symbol_col=config.symbol_col,
            horizon_hours=24.0,
        ),
    }
    for method in methods:
        prob_cols = _method_probability_columns(regime_probabilities.columns, str(method))
        probs = _probability_matrix(regime_probabilities, prob_cols)
        k = max(probs.shape[1], 1)
        entropy = -np.sum(
            probs * np.log(np.maximum(probs, float(config.eps))),
            axis=1,
        )
        if k > 1:
            entropy = entropy / np.log(float(k))
        else:
            entropy = np.zeros(len(probs), dtype=np.float32)
        prob_max = np.max(probs, axis=1).astype(np.float32)
        prefix = f"url_{method}"
        cols[f"{prefix}_regime_prob_entropy"] = np.clip(entropy, 0.0, 1.0).astype(np.float32)
        cols[f"{prefix}_regime_prob_max"] = prob_max.astype(np.float32)
        for horizon, lag_pos in lag_positions.items():
            change = np.zeros(len(frame), dtype=np.float32)
            valid = lag_pos >= 0
            if valid.any():
                change[valid] = (
                    0.5
                    * np.sum(np.abs(probs[valid] - probs[lag_pos[valid]]), axis=1)
                ).astype(np.float32)
            cols[f"{prefix}_regime_prob_change_{horizon}h"] = np.clip(change, 0.0, 1.0)
        label_col = f"{method}_smoothed_regime"
        if label_arrays is not None and label_col in label_arrays:
            labels = np.asarray(label_arrays[label_col], dtype=np.int64)
        elif label_col in regime_labels.columns:
            labels = _label_array(regime_labels[label_col])
        else:
            labels = np.argmax(probs, axis=1).astype(np.int64)
        hazard, since, duration = _transition_duration_arrays(
            frame,
            labels,
            prob_max,
            timestamp_col=config.timestamp_col,
            symbol_col=config.symbol_col,
        )
        cols[f"{prefix}_regime_transition_hazard"] = hazard.astype(np.float32)
        cols[f"{prefix}_time_since_regime_change"] = since.astype(np.float32)
        cols[f"{prefix}_expected_regime_duration"] = duration.astype(np.float32)
    return pd.DataFrame(cols, index=frame.index, dtype=np.float32)


def _compute_regime_feature_importance(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    regime_labels: pd.DataFrame,
    methods: Sequence[str],
    *,
    label_arrays: Mapping[str, np.ndarray] | None = None,
    eps: float,
) -> pd.DataFrame:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.shape[0] == 0 or arr.shape[1] == 0 or not methods:
        return pd.DataFrame()
    finite_all = np.isfinite(arr)
    values_all = np.where(finite_all, arr, 0.0).astype(np.float32, copy=False)
    values2_all = (values_all * values_all).astype(np.float32, copy=False)
    rows: list[dict[str, Any]] = []
    feature_labels = [str(name) for name in feature_names]
    feature_count = int(arr.shape[1])
    max_rank = min(feature_count, 50)
    for method in methods:
        label_col = f"{method}_smoothed_regime"
        if label_arrays is not None and label_col in label_arrays:
            labels = np.asarray(label_arrays[label_col], dtype=np.int64)
        elif label_col in regime_labels.columns:
            labels = _label_array(regime_labels[label_col])
        else:
            continue
        valid = labels >= 0
        valid_rows = int(np.count_nonzero(valid))
        if valid_rows < 4:
            continue
        valid_labels = labels[valid]
        if valid_labels.size == 0:
            continue
        max_label = int(valid_labels.max())
        if max_label < 0:
            continue
        label_count = max_label + 1
        row_counts = np.bincount(valid_labels, minlength=label_count).astype(np.int64, copy=False)
        finite = finite_all[valid]
        values = values_all[valid]
        values2 = values2_all[valid]
        counts = np.zeros((label_count, feature_count), dtype=np.float32)
        sums = np.zeros_like(counts)
        sums2 = np.zeros_like(counts)
        np.add.at(counts, valid_labels, finite.astype(np.float32, copy=False))
        np.add.at(sums, valid_labels, values)
        np.add.at(sums2, valid_labels, values2)
        total_counts = counts.sum(axis=0)
        total_sums = sums.sum(axis=0)
        total_sums2 = sums2.sum(axis=0)

        for regime in np.flatnonzero(row_counts):
            support_rows = int(row_counts[regime])
            outside_rows = valid_rows - support_rows
            if support_rows < 2 or outside_rows < 2:
                continue
            in_count = counts[regime]
            out_count = total_counts - in_count
            mean_in = np.divide(sums[regime], in_count, out=np.zeros(feature_count, dtype=np.float32), where=in_count > 0)
            mean_out = np.divide(
                total_sums - sums[regime],
                out_count,
                out=np.zeros(feature_count, dtype=np.float32),
                where=out_count > 0,
            )
            var_in = np.divide(
                sums2[regime],
                in_count,
                out=np.zeros(feature_count, dtype=np.float32),
                where=in_count > 0,
            ) - mean_in * mean_in
            var_out = np.divide(
                total_sums2 - sums2[regime],
                out_count,
                out=np.zeros(feature_count, dtype=np.float32),
                where=out_count > 0,
            ) - mean_out * mean_out
            var_in = np.maximum(var_in, 0.0)
            var_out = np.maximum(var_out, 0.0)
            denom = np.sqrt(np.maximum(0.5 * (var_in + var_out), eps))
            signed = (mean_in - mean_out) / denom
            importance = np.abs(signed)
            usable = (in_count > 0) & (out_count > 0) & np.isfinite(importance)
            if not bool(np.any(usable)):
                continue
            importance = np.where(usable, importance, -np.inf)
            top_n = min(max_rank, int(np.count_nonzero(usable)))
            if top_n < feature_count:
                order = np.argpartition(importance, -top_n)[-top_n:]
                order = order[np.argsort(importance[order])[::-1]]
            else:
                order = np.argsort(importance)[::-1]
            support_fraction = float(support_rows) / float(len(labels))
            for rank, j in enumerate(order[:top_n], start=1):
                rows.append(
                    {
                        "method": str(method),
                        "regime": int(regime),
                        "feature": feature_labels[j],
                        "rank": int(rank),
                        "importance": float(importance[j]),
                        "signed_shift": float(signed[j]),
                        "regime_mean": float(mean_in[j]),
                        "outside_mean": float(mean_out[j]),
                        "support_fraction": support_fraction,
                        "support_rows": support_rows,
                    }
                )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(
        ["method", "regime", "importance"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    for col in ["importance", "signed_shift", "regime_mean", "outside_mean", "support_fraction"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float32)
    for col in ["regime", "rank", "support_rows"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.int32)
    return out


def _tradability_feature_groups(feature_names: Sequence[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {
        "volume_activity": [],
        "liquidity_cost": [],
        "funding_pressure": [],
        "oi_crowding": [],
        "volatility_stress": [],
    }
    for idx, feature in enumerate(feature_names):
        low = str(feature).lower()
        family = _feature_family(str(feature))
        is_cost = any(token in low for token in ["amihud", "illiq", "spread", "slippage", "cost"])
        if is_cost:
            groups["liquidity_cost"].append(int(idx))
            continue
        if family == "funding" or "fund" in low:
            groups["funding_pressure"].append(int(idx))
            continue
        if family == "open_interest" or "open_interest" in low or "oi_" in low or low.startswith("oi"):
            groups["oi_crowding"].append(int(idx))
            continue
        if (
            family == "liquidity"
            or "volume" in low
            or "rvol" in low
            or "liquidity" in low
        ):
            groups["volume_activity"].append(int(idx))
            continue
        if family == "volatility":
            groups["volatility_stress"].append(int(idx))
    return groups


def _row_category_means(
    arr: np.ndarray,
    cols: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    n = arr.shape[0]
    if not cols:
        empty = np.full(n, np.nan, dtype=np.float32)
        return empty, empty.copy()
    values = arr[:, np.asarray(cols, dtype=np.int64)]
    finite = np.isfinite(values)
    counts = finite.sum(axis=1, dtype=np.float32)
    sums = np.where(finite, values, 0.0).sum(axis=1, dtype=np.float32)
    abs_sums = np.where(finite, np.abs(values), 0.0).sum(axis=1, dtype=np.float32)
    mean = np.divide(
        sums,
        counts,
        out=np.full(n, np.nan, dtype=np.float32),
        where=counts > 0,
    )
    abs_mean = np.divide(
        abs_sums,
        counts,
        out=np.full(n, np.nan, dtype=np.float32),
        where=counts > 0,
    )
    return mean.astype(np.float32, copy=False), abs_mean.astype(np.float32, copy=False)


def _label_mean_stats(
    values: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    *,
    label_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=np.float32)
    finite = valid & np.isfinite(vals)
    if int(np.count_nonzero(finite)) == 0 or int(label_count) <= 0:
        empty = np.full(max(int(label_count), 0), np.nan, dtype=np.float32)
        return empty, empty.copy()
    label_vals = labels[finite]
    sums = np.bincount(
        label_vals,
        weights=vals[finite].astype(np.float64, copy=False),
        minlength=int(label_count),
    )
    counts = np.bincount(label_vals, minlength=int(label_count)).astype(np.float64, copy=False)
    total_sum = float(np.sum(sums))
    total_count = float(np.sum(counts))
    mean = np.divide(
        sums,
        counts,
        out=np.full(int(label_count), np.nan, dtype=np.float64),
        where=counts > 0,
    )
    outside_count = total_count - counts
    outside_mean = np.divide(
        total_sum - sums,
        outside_count,
        out=np.full(int(label_count), np.nan, dtype=np.float64),
        where=outside_count > 0,
    )
    return mean.astype(np.float32), outside_mean.astype(np.float32)


def _bounded_tanh(value: float, scale: float = 2.0) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.tanh(float(value) / max(float(scale), 1e-6)))


def _compute_regime_tradability_diagnostics(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    feature_names: Sequence[str],
    regime_labels: pd.DataFrame,
    methods: Sequence[str],
    *,
    label_arrays: Mapping[str, np.ndarray] | None = None,
    timestamp_col: str,
    symbol_col: str,
) -> pd.DataFrame:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.shape[0] == 0 or arr.shape[1] == 0 or not methods:
        return pd.DataFrame()
    groups = _tradability_feature_groups(feature_names)
    total_group_features = int(sum(len(v) for v in groups.values()))
    category_means = {
        group_name: _row_category_means(arr, cols)
        for group_name, cols in groups.items()
    }
    rows: list[dict[str, Any]] = []
    if symbol_col in frame.columns:
        symbols = frame[symbol_col].astype(str).to_numpy()
    else:
        symbols = np.repeat("__all__", len(frame))
    symbol_codes = pd.factorize(symbols, sort=False)[0].astype(np.int32, copy=False)
    if timestamp_col in frame.columns:
        ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
        ts_ns = ts.to_numpy(dtype="datetime64[ns]").astype("int64")
        valid_ts = ts.notna().to_numpy(dtype=bool)
    else:
        ts_ns = np.zeros(len(frame), dtype=np.int64)
        valid_ts = np.zeros(len(frame), dtype=bool)
    for method in methods:
        label_col = f"{method}_smoothed_regime"
        if label_arrays is not None and label_col in label_arrays:
            labels = np.asarray(label_arrays[label_col], dtype=np.int64)
        elif label_col in regime_labels.columns:
            labels = _label_array(regime_labels[label_col])
        else:
            continue
        valid = labels >= 0
        valid_total = int(np.count_nonzero(valid))
        if valid_total < 2:
            continue
        valid_labels = labels[valid]
        if valid_labels.size == 0:
            continue
        label_count = int(valid_labels.max()) + 1
        if label_count <= 0:
            continue
        row_counts = np.bincount(valid_labels, minlength=label_count).astype(np.int32, copy=False)
        group_stats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for group_name, (row_mean, row_abs_mean) in category_means.items():
            mean_by_label, outside_mean_by_label = _label_mean_stats(
                row_mean,
                labels,
                valid,
                label_count=label_count,
            )
            abs_mean_by_label, _outside_abs = _label_mean_stats(
                row_abs_mean,
                labels,
                valid,
                label_count=label_count,
            )
            group_stats[group_name] = (
                mean_by_label,
                outside_mean_by_label,
                abs_mean_by_label,
            )
        for regime in np.flatnonzero(row_counts):
            pos = np.flatnonzero(labels == int(regime)).astype(np.int64, copy=False)
            if pos.size == 0:
                continue
            record: dict[str, Any] = {
                "method": str(method),
                "regime": int(regime),
                "support_rows": int(pos.size),
                "support_fraction": float(pos.size) / float(max(valid_total, 1)),
                "symbol_count": int(np.unique(symbol_codes[pos]).size),
                "tradability_feature_count": total_group_features,
            }
            valid_pos_ts = pos[valid_ts[pos]]
            if valid_pos_ts.size >= 2:
                span_ns = int(np.max(ts_ns[valid_pos_ts]) - np.min(ts_ns[valid_pos_ts]))
                record["timestamp_span_hours"] = float(max(span_ns, 0) / 3_600_000_000_000.0)
            else:
                record["timestamp_span_hours"] = float("nan")

            for group_name, cols in groups.items():
                mean_by_label, outside_mean_by_label, abs_mean_by_label = group_stats[group_name]
                mean = float(mean_by_label[int(regime)]) if int(regime) < mean_by_label.size else float("nan")
                outside_mean = (
                    float(outside_mean_by_label[int(regime)])
                    if int(regime) < outside_mean_by_label.size
                    else float("nan")
                )
                abs_mean = (
                    float(abs_mean_by_label[int(regime)])
                    if int(regime) < abs_mean_by_label.size
                    else float("nan")
                )
                record[f"{group_name}_feature_count"] = int(len(cols))
                record[f"{group_name}_mean_z"] = mean
                record[f"{group_name}_outside_mean_z"] = outside_mean
                record[f"{group_name}_shift_z"] = (
                    float(mean - outside_mean)
                    if np.isfinite(mean) and np.isfinite(outside_mean)
                    else float("nan")
                )
                record[f"{group_name}_abs_mean_z"] = abs_mean

            volume_shift = float(record.get("volume_activity_shift_z", float("nan")))
            cost_shift = float(record.get("liquidity_cost_shift_z", float("nan")))
            cost_abs = float(record.get("liquidity_cost_abs_mean_z", float("nan")))
            funding_abs = float(record.get("funding_pressure_abs_mean_z", float("nan")))
            oi_abs = float(record.get("oi_crowding_abs_mean_z", float("nan")))
            vol_shift = float(record.get("volatility_stress_shift_z", float("nan")))
            volume_component = _bounded_tanh(volume_shift, scale=2.0)
            illiquidity_risk = _clamp01(
                0.5 * _bounded_tanh(max(cost_shift, 0.0), scale=2.0)
                + 0.5 * _bounded_tanh(max(cost_abs, 0.0), scale=3.0)
            )
            crowding_risk = _clamp01(
                _bounded_tanh(
                    (max(funding_abs, 0.0) if np.isfinite(funding_abs) else 0.0)
                    + (max(oi_abs, 0.0) if np.isfinite(oi_abs) else 0.0),
                    scale=4.0,
                )
            )
            volatility_stress = _clamp01(_bounded_tanh(max(vol_shift, 0.0), scale=2.0))
            tradability_score = _clamp01(
                0.50
                + 0.20 * volume_component
                - 0.25 * illiquidity_risk
                - 0.15 * volatility_stress
                - 0.10 * crowding_risk
            )
            record["illiquidity_risk_score"] = illiquidity_risk
            record["crowding_risk_score"] = crowding_risk
            record["volatility_stress_score"] = volatility_stress
            record["tradability_score"] = tradability_score
            rows.append(record)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(
        ["method", "regime"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    float_cols = [
        col
        for col in out.columns
        if col not in {"method", "regime", "support_rows", "symbol_count"}
    ]
    for col in float_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float32)
    for col in ["regime", "support_rows", "symbol_count"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.int32)
    return out


def _metric_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (list, tuple, set)):
        return json.dumps(_json_ready(list(value)), sort_keys=True)
    if isinstance(value, Mapping):
        return json.dumps(_json_ready(dict(value)), sort_keys=True)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _pipeline_step_row(
    step: str,
    *,
    status: str = "completed",
    **metrics: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "step": str(step),
        "status": str(status),
    }
    for key, value in metrics.items():
        row[str(key)] = _metric_scalar(value)
    return row


def _pipeline_steps_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["step", "status"])
    out = pd.DataFrame([dict(row) for row in rows])
    order = ["step", "status"]
    remaining = [col for col in out.columns if col not in set(order)]
    return out.loc[:, order + remaining]


def _methods_with_prefix(methods: Sequence[str], prefix: str) -> list[str]:
    return [str(method) for method in methods if str(method).startswith(prefix)]


def _infer_method_from_feature(feature: str, methods: Sequence[str]) -> str | None:
    raw = str(feature)
    stripped = raw[4:] if raw.startswith("url_") else raw
    for method in sorted([str(m) for m in methods], key=len, reverse=True):
        if stripped == method or stripped.startswith(f"{method}_") or raw.startswith(f"{method}_"):
            return method
    return None


def _method_score_metric(
    methods: Sequence[str],
    diag_by_method: Mapping[str, Mapping[str, Any]],
    metric: str,
) -> float:
    values: list[float] = []
    for method in methods:
        row = diag_by_method.get(str(method), {})
        value = row.get(metric, np.nan) if isinstance(row, Mapping) else np.nan
        try:
            f = float(value)
        except Exception:
            continue
        if np.isfinite(f):
            values.append(f)
    return float(max(values)) if values else float("nan")


def _method_tradability_metric(
    methods: Sequence[str],
    tradability_by_method: Mapping[str, float],
) -> float:
    values: list[float] = []
    for method in methods:
        try:
            value = float(tradability_by_method.get(str(method), np.nan))
        except Exception:
            continue
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def _feature_quality_row(values: pd.Series) -> dict[str, Any]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32, copy=False)
    finite = np.isfinite(arr)
    finite_count = int(np.count_nonzero(finite))
    if finite_count:
        finite_values = arr[finite]
        std = float(np.std(finite_values, dtype=np.float64))
        mean_abs = float(np.mean(np.abs(finite_values), dtype=np.float64))
        unique_values = int(np.unique(finite_values).size)
    else:
        std = float("nan")
        mean_abs = float("nan")
        unique_values = 0
    return {
        "finite_fraction": float(finite_count) / float(max(len(arr), 1)),
        "non_null_fraction": float(values.notna().mean()) if len(values) else 0.0,
        "std": std,
        "mean_abs": mean_abs,
        "unique_values": unique_values,
    }


def _model_regime_feature_metrics(
    features: pd.DataFrame,
    feature_groups: Mapping[str, Sequence[str]],
    *,
    model_methods: Sequence[str],
    kept_methods: set[str],
    all_methods: Sequence[str],
    regime_diagnostics: pd.DataFrame,
    regime_tradability_diagnostics: pd.DataFrame,
    candidate_tier: str,
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    source_by_feature: dict[str, str] = {}
    for group, cols in feature_groups.items():
        for col in cols:
            source_by_feature[str(col)] = str(group)
    model_method_list = [str(method) for method in model_methods]
    diag_by_method: dict[str, dict[str, Any]] = {}
    if isinstance(regime_diagnostics, pd.DataFrame) and not regime_diagnostics.empty and "method" in regime_diagnostics.columns:
        diag_by_method = {
            str(row["method"]): dict(row)
            for row in regime_diagnostics.to_dict(orient="records")
        }
    tradability_by_method: dict[str, float] = {}
    if (
        isinstance(regime_tradability_diagnostics, pd.DataFrame)
        and not regime_tradability_diagnostics.empty
        and {"method", "tradability_score"}.issubset(regime_tradability_diagnostics.columns)
    ):
        tradability_by_method = {
            str(method): float(value)
            for method, value in regime_tradability_diagnostics.groupby("method")["tradability_score"].mean().items()
        }
    rows: list[dict[str, Any]] = []
    group_method_scope = {
        "raw_pca_embedding": [
            method for method in model_method_list if method.startswith("raw_pca") or method == "raw_selected_kmeans"
        ],
        "raw_spectral_embedding": [method for method in model_method_list if method.startswith("raw_spectral")],
        "leaf_embedding": _methods_with_prefix(model_method_list, "leaf_"),
        "leaf_umap_embedding": _methods_with_prefix(model_method_list, "leaf_umap_"),
        "leaf_spectral_embedding": _methods_with_prefix(model_method_list, "leaf_spectral_"),
        "sparse_ae_latent": _methods_with_prefix(model_method_list, "sparse_ae_"),
        "contrastive_ae_latent": _methods_with_prefix(model_method_list, "contrastive_ae_"),
        "contrastive_leaf_latent": _methods_with_prefix(model_method_list, "contrastive_leaf_"),
        "mfa_responsibility": [method for method in model_method_list if method == "mfa"],
    }
    for col in features.columns:
        col_name = str(col)
        source_group = source_by_feature.get(col_name, "unknown")
        exact_method = _infer_method_from_feature(col_name, all_methods)
        scope = [exact_method] if exact_method else group_method_scope.get(source_group, [])
        scope = [method for method in _stable_unique(scope) if method]
        if not scope:
            scope = model_method_list
        quality = _feature_quality_row(features[col])
        rows.append(
            {
                "feature": col_name,
                "source_group": source_group,
                "method": ",".join(scope[:6]),
                "method_count": int(len(scope)),
                "candidate_tier": str(candidate_tier),
                "selected_by_method_keep": bool(any(method in kept_methods for method in scope)),
                "method_total_score": _method_score_metric(scope, diag_by_method, "TotalScore"),
                "method_oos_stability": _method_score_metric(scope, diag_by_method, "OOS_Stability"),
                "method_null_robustness": _method_score_metric(scope, diag_by_method, "Null_Robustness"),
                "method_window_robustness": _method_score_metric(scope, diag_by_method, "Window_Robustness"),
                "method_geometry_separation": _method_score_metric(scope, diag_by_method, "Geometry_Separation"),
                "method_nontriviality": _method_score_metric(scope, diag_by_method, "NonTriviality"),
                "mean_regime_tradability": _method_tradability_metric(scope, tradability_by_method),
                **quality,
            }
        )
    out = pd.DataFrame(rows)
    float_cols = [
        "method_total_score",
        "method_oos_stability",
        "method_null_robustness",
        "method_window_robustness",
        "method_geometry_separation",
        "method_nontriviality",
        "mean_regime_tradability",
        "finite_fraction",
        "non_null_fraction",
        "std",
        "mean_abs",
    ]
    for col in float_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float32)
    for col in ["method_count", "unique_values"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int32)
    return out


def fit_advanced_regime_learning(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    config: AdvancedRegimeLearningConfig = AdvancedRegimeLearningConfig(),
) -> AdvancedRegimeLearningArtifact:
    """Fit the unsupervised regime-learning stack on selected/generated features."""

    matrix, features, matrix_diag = _numeric_scaled_matrix(
        frame,
        feature_columns,
        max_rows=int(config.max_rows),
        timestamp_col=config.timestamp_col,
        symbol_col=config.symbol_col,
        sample_time_bins=int(config.sample_time_bins),
        scaling_mode=str(config.scaling_mode),
        scaling_min_periods=int(config.scaling_min_periods),
        eps=float(config.eps),
    )
    index = frame.index
    row_key_cols = [
        col
        for col in [str(config.timestamp_col), str(config.symbol_col)]
        if col in frame.columns
    ]
    row_keys = frame.loc[:, row_key_cols].copy(deep=False) if row_key_cols else pd.DataFrame(index=index)
    step_rows: list[dict[str, Any]] = [
        _pipeline_step_row(
            "01_matrix_scaling",
            input_rows=int(len(frame)),
            input_feature_count=int(len(feature_columns)),
            usable_feature_count=int(len(features)),
            sampled_rows=int(matrix.shape[0]),
            scaling_mode=str(config.scaling_mode),
            sample_time_bins=int(config.sample_time_bins),
            warmup_min_periods=int(config.scaling_min_periods),
        )
    ]
    if matrix.shape[1] == 0:
        empty = pd.DataFrame(index=index)
        return AdvancedRegimeLearningArtifact(
            schema_version=ADVANCED_REGIME_LEARNING_SCHEMA_VERSION,
            selected_features=[],
            conservative_features=[],
            strong_features=[],
            exploratory_features=[],
            stability_frequencies=pd.DataFrame(),
            real_vs_null_importances=pd.DataFrame(),
            leaf_embeddings=empty,
            raw_baseline_embeddings=empty,
            ae_latents=empty,
            contrastive_ae_latents=empty,
            contrastive_leaf_latents=empty,
            mfa_responsibilities=empty,
            mfa_feature_relevance=pd.DataFrame(),
            ae_feature_gates=pd.DataFrame(),
            regime_labels=empty,
            regime_probabilities=empty,
            regime_transition_features=empty,
            regime_feature_importance=pd.DataFrame(),
            regime_tradability_diagnostics=pd.DataFrame(),
            regime_diagnostics=pd.DataFrame(),
            pipeline_steps=_pipeline_steps_frame(
                step_rows
                + [
                    _pipeline_step_row(
                        "terminal",
                        status="skipped",
                        reason="no_numeric_features",
                    )
                ]
            ),
            model_regime_features=empty,
            model_regime_feature_metrics=pd.DataFrame(),
            materialized_features=empty,
            materialized_feature_groups={},
            specialist_candidate_features=[],
            method_keep_decisions=pd.DataFrame(),
            row_keys=row_keys,
            method_embeddings={},
            diagnostics={"enabled": False, "reason": "no_numeric_features"},
        )

    stability, importances, selector_model, selector_diag = real_vs_null_stability_selection(
        frame,
        matrix,
        features,
        config=config,
    )
    selected = stability.loc[
        stability["selection_frequency"] >= float(config.exploratory_threshold),
        "feature",
    ].astype(str).tolist()
    if not selected:
        selected = stability.head(min(len(stability), int(config.stability_top_m)))["feature"].astype(str).tolist()
    selected = _stable_unique(selected)
    selected_idx = [features.index(f) for f in selected if f in features]
    selected_matrix = matrix[:, selected_idx] if selected_idx else matrix
    selected_features = [features[i] for i in selected_idx] if selected_idx else features
    if not selector_diag.empty and {"section", "rows"}.issubset(selector_diag.columns):
        selector_sections = selector_diag["section"].astype(str)
        selector_bootstrap_rows = selector_diag.loc[selector_sections.eq("bootstrap"), "rows"]
        selector_null_rows = selector_diag.loc[selector_sections.eq("null_generation"), "rows"]
    else:
        selector_bootstrap_rows = pd.Series(dtype=float)
        selector_null_rows = pd.Series(dtype=float)
    step_rows.append(
        _pipeline_step_row(
            "02_real_vs_null_stability_selection",
            backend=(str(importances["backend"].iloc[0]) if not importances.empty and "backend" in importances.columns else None),
            bootstrap_count=int(config.stability_bootstraps),
            mean_bootstrap_rows=(
                float(pd.to_numeric(selector_bootstrap_rows, errors="coerce").mean())
                if not selector_bootstrap_rows.empty
                else None
            ),
            synthetic_rows=int(pd.to_numeric(selector_null_rows, errors="coerce").sum()) if not selector_null_rows.empty else 0,
            selected_feature_count=int(len(selected_features)),
            conservative_feature_count=int((stability["tier"] == "conservative").sum()) if "tier" in stability.columns else 0,
            strong_feature_count=int(stability["tier"].isin(["conservative", "strong"]).sum()) if "tier" in stability.columns else 0,
            exploratory_feature_count=int((stability["tier"] != "drop").sum()) if "tier" in stability.columns else 0,
            top_selection_frequency=(
                float(pd.to_numeric(stability["selection_frequency"], errors="coerce").max())
                if "selection_frequency" in stability.columns
                else None
            ),
        )
    )

    backend = str(importances["backend"].iloc[0]) if not importances.empty else "random_forest"
    full_x, full_y, null_report = generate_real_vs_null_samples(matrix, features, config=config)
    leaf_raw = leaf_embedding_from_classifier(
        selector_model,
        backend,
        matrix,
        full_x,
        full_y,
        max_trees=int(config.leaf_embedding_max_trees),
    )
    leaf_reduced = _reduce_embedding(
        leaf_raw.to_numpy(dtype=np.float32),
        method="pca",
        n_components=int(config.leaf_embedding_dim),
        random_state=int(config.random_state),
        config=config,
    )
    leaf_embeddings = _embedding_frame(index, "url_leaf", leaf_reduced)
    leaf_umap_reduced = _reduce_embedding(
        leaf_raw.to_numpy(dtype=np.float32),
        method="umap",
        n_components=int(config.leaf_embedding_dim),
        random_state=int(config.random_state),
        config=config,
    )
    leaf_spectral_reduced = _reduce_embedding(
        leaf_raw.to_numpy(dtype=np.float32),
        method="spectral",
        n_components=int(config.leaf_embedding_dim),
        random_state=int(config.random_state),
        config=config,
    )
    leaf_umap_embeddings = _embedding_frame(index, "url_leaf_umap", leaf_umap_reduced)
    leaf_spectral_embeddings = _embedding_frame(index, "url_leaf_spectral", leaf_spectral_reduced)
    raw_pca = _reduce_embedding(
        selected_matrix,
        method="pca",
        n_components=int(config.raw_embedding_dim),
        random_state=int(config.random_state),
        config=config,
    )
    raw_spectral = _reduce_embedding(
        selected_matrix,
        method="spectral",
        n_components=int(config.raw_embedding_dim),
        random_state=int(config.random_state),
        config=config,
    )
    raw_baseline_embeddings = _embedding_frame(index, "url_raw_pca", raw_pca)
    raw_spectral_embeddings = _embedding_frame(index, "url_raw_spectral", raw_spectral)
    step_rows.append(
        _pipeline_step_row(
            "03_leaf_and_raw_embeddings",
            real_rows=int(matrix.shape[0]),
            synthetic_rows=int(pd.to_numeric(null_report.get("rows", pd.Series(dtype=float)), errors="coerce").sum())
            if not null_report.empty
            else 0,
            leaf_raw_feature_count=int(leaf_raw.shape[1]),
            leaf_pca_dim=int(leaf_embeddings.shape[1]),
            leaf_umap_dim=int(leaf_umap_embeddings.shape[1]),
            leaf_spectral_dim=int(leaf_spectral_embeddings.shape[1]),
            raw_pca_dim=int(raw_baseline_embeddings.shape[1]),
            raw_spectral_dim=int(raw_spectral_embeddings.shape[1]),
            max_leaf_trees=int(config.leaf_embedding_max_trees),
        )
    )

    ae_z, ae_gate, ae_diag = _train_autoencoder(
        selected_matrix,
        config=config,
        contrastive=False,
        random_state=int(config.random_state) + 100,
        feature_names=selected_features,
    )
    contrast_z, contrast_gate, contrast_diag = _train_autoencoder(
        selected_matrix,
        config=config,
        contrastive=True,
        random_state=int(config.random_state) + 200,
        feature_names=selected_features,
    )
    leaf_contrast_z, _leaf_gate, leaf_contrast_diag = _train_autoencoder(
        leaf_reduced,
        config=config,
        contrastive=True,
        random_state=int(config.random_state) + 300,
        feature_names=list(leaf_embeddings.columns),
    )
    ae_latents = _embedding_frame(index, "url_sparse_ae", ae_z)
    contrastive_ae_latents = _embedding_frame(index, "url_contrastive_ae", contrast_z)
    contrastive_leaf_latents = _embedding_frame(index, "url_contrastive_leaf", leaf_contrast_z)
    step_rows.append(
        _pipeline_step_row(
            "04_autoencoder_latents",
            backend=str(ae_diag.get("backend", config.ae_backend)) if isinstance(ae_diag, Mapping) else str(config.ae_backend),
            sparse_ae_dim=int(ae_latents.shape[1]),
            contrastive_ae_dim=int(contrastive_ae_latents.shape[1]),
            contrastive_leaf_dim=int(contrastive_leaf_latents.shape[1]),
            sparse_train_rows=int(ae_diag.get("train_rows", 0)) if isinstance(ae_diag, Mapping) else 0,
            contrastive_train_rows=int(contrast_diag.get("train_rows", 0)) if isinstance(contrast_diag, Mapping) else 0,
            leaf_contrastive_train_rows=int(leaf_contrast_diag.get("train_rows", 0)) if isinstance(leaf_contrast_diag, Mapping) else 0,
            epochs=int(config.ae_epochs),
            batch_size=int(config.ae_batch_size),
        )
    )

    mfa = MixtureFactorAnalyzer(
        n_components=int(config.mfa_regimes),
        n_factors=int(config.mfa_factors),
        max_iter=int(config.mfa_max_iter),
        l1_lambda=float(config.mfa_l1_lambda),
        tol=float(config.mfa_tol),
        random_state=int(config.random_state),
        eps=float(config.eps),
    ).fit(selected_matrix)
    gamma = mfa.predict_proba(selected_matrix)
    mfa_responsibilities = pd.DataFrame(
        {f"url_mfa_gamma_{i:02d}": gamma[:, i].astype(np.float32) for i in range(gamma.shape[1])},
        index=index,
    )
    mfa_feature_relevance = mfa.feature_relevance(selected_features)
    ae_feature_gates = pd.DataFrame({"feature": selected_features})
    if ae_gate is not None and len(ae_gate) == len(selected_features):
        ae_feature_gates["sparse_ae_gate"] = ae_gate.astype(np.float32)
    if contrast_gate is not None and len(contrast_gate) == len(selected_features):
        ae_feature_gates["contrastive_ae_gate"] = contrast_gate.astype(np.float32)
    if len(ae_feature_gates.columns) > 1:
        score_cols = [col for col in ae_feature_gates.columns if col.endswith("_gate")]
        ae_feature_gates["max_gate"] = ae_feature_gates[score_cols].max(axis=1)
        ae_feature_gates = ae_feature_gates.sort_values("max_gate", ascending=False, kind="mergesort")
    mfa_threshold = float(config.mfa_relevance_min)
    mfa_ranked = mfa_feature_relevance["feature"].astype(str).tolist() if not mfa_feature_relevance.empty else selected_features
    if mfa_threshold > 0.0 and not mfa_feature_relevance.empty:
        supported = mfa_feature_relevance.loc[
            mfa_feature_relevance["mfa_relevance"] >= mfa_threshold,
            "feature",
        ].astype(str).tolist()
        min_keep = max(0, min(int(config.mfa_min_keep_features), len(mfa_ranked)))
        supported = _stable_unique(supported + mfa_ranked[:min_keep])
    else:
        supported = list(selected_features)
    mfa_supported_features = [feature for feature in supported if feature in selected_features]
    mfa_deprioritized_features = [feature for feature in selected_features if feature not in set(mfa_supported_features)]
    mfa_ll = [float(v) for v in getattr(mfa, "log_likelihood_", [])]
    step_rows.append(
        _pipeline_step_row(
            "05_mixture_factor_analyzers",
            regime_count=int(gamma.shape[1]) if gamma.ndim == 2 else 0,
            factor_count=int(config.mfa_factors),
            responsibility_columns=int(mfa_responsibilities.shape[1]),
            relevance_rows=int(len(mfa_feature_relevance)),
            supported_feature_count=int(len(mfa_supported_features)),
            deprioritized_feature_count=int(len(mfa_deprioritized_features)),
            final_log_likelihood=float(mfa_ll[-1]) if mfa_ll else None,
            iterations=int(len(mfa_ll)),
        )
    )

    method_specs: list[tuple[str, np.ndarray, str]] = [
        ("raw_selected_kmeans", selected_matrix, "kmeans"),
        ("raw_pca_kmeans", raw_pca, "kmeans"),
        ("raw_pca_bayesian_gmm", raw_pca, "bayesian_gmm"),
        ("raw_spectral_kmeans", raw_spectral, "kmeans"),
        ("raw_spectral_bayesian_gmm", raw_spectral, "bayesian_gmm"),
        ("raw_spectral_spectral", raw_spectral, "spectral"),
        ("leaf_pca_bayesian_gmm", leaf_reduced, "bayesian_gmm"),
        ("leaf_pca_hdbscan", leaf_reduced, "hdbscan"),
        ("leaf_pca_hmm", leaf_reduced, "hmm"),
        ("leaf_pca_spectral", leaf_reduced, "spectral"),
        ("leaf_umap_bayesian_gmm", leaf_umap_reduced, "bayesian_gmm"),
        ("leaf_umap_hdbscan", leaf_umap_reduced, "hdbscan"),
        ("leaf_spectral_spectral", leaf_spectral_reduced, "spectral"),
        ("sparse_ae_bayesian_gmm", ae_z, "bayesian_gmm"),
        ("sparse_ae_hmm", ae_z, "hmm"),
        ("sparse_ae_hdbscan", ae_z, "hdbscan"),
        ("sparse_ae_spectral", ae_z, "spectral"),
        ("contrastive_ae_bayesian_gmm", contrast_z, "bayesian_gmm"),
        ("contrastive_ae_hmm", contrast_z, "hmm"),
        ("contrastive_ae_hdbscan", contrast_z, "hdbscan"),
        ("contrastive_ae_spectral", contrast_z, "spectral"),
        ("contrastive_leaf_bayesian_gmm", leaf_contrast_z, "bayesian_gmm"),
        ("contrastive_leaf_hmm", leaf_contrast_z, "hmm"),
        ("contrastive_leaf_hdbscan", leaf_contrast_z, "hdbscan"),
        ("contrastive_leaf_spectral", leaf_contrast_z, "spectral"),
        ("mfa", gamma, "direct"),
    ]
    baseline_methods = {
        "raw_selected_kmeans",
        "raw_pca_kmeans",
        "raw_pca_bayesian_gmm",
        "raw_spectral_kmeans",
        "raw_spectral_bayesian_gmm",
        "raw_spectral_spectral",
    }
    label_cols: dict[str, np.ndarray] = {}
    prob_frames: list[pd.DataFrame] = []
    diag_rows: list[dict[str, Any]] = []
    trend_vol = _trend_vol_matrix(selected_matrix, selected_features)
    non_trend_vol = _non_trend_vol_matrix(selected_matrix, selected_features)
    oos_blocks = _assessment_blocks(
        frame,
        timestamp_col=config.timestamp_col,
        n_blocks=int(config.regime_assessment_oos_folds),
        min_rows=max(3, int(config.n_regimes)),
    )
    window_blocks = _assessment_blocks(
        frame,
        timestamp_col=config.timestamp_col,
        n_blocks=int(config.regime_assessment_windows),
        min_rows=max(4, int(config.n_regimes)),
    )
    robustness_positions = _assessment_sample_positions(
        frame,
        np.arange(len(frame), dtype=np.int64),
        config=config,
        max_rows=int(config.regime_assessment_max_robustness_rows),
        random_state=int(config.random_state) + 7150,
    )
    for name, z, cluster_method in method_specs:
        labels, probs, used = _cluster_embedding(
            z,
            method=cluster_method,
            n_regimes=int(config.n_regimes),
            random_state=int(config.random_state),
            config=config,
        )
        smoothed = minimum_duration_smooth_by_frame(
            labels,
            frame,
            min_duration=int(config.min_regime_duration),
            timestamp_col=config.timestamp_col,
            symbol_col=config.symbol_col,
        )
        label_cols[f"{name}_raw_regime"] = labels.astype(np.int16)
        label_cols[f"{name}_smoothed_regime"] = smoothed.astype(np.int16)
        prob_frames.append(_build_probability_frame(index, name, probs, labels))
        metrics = _label_diagnostics(smoothed, z)
        metrics["method"] = name
        metrics["cluster_method"] = used
        legacy_stability = _stability_score(z, smoothed, config)
        metrics["stability"] = legacy_stability
        if cluster_method == "direct":
            assessment_cluster_method = "direct"
        elif used == "bayesian_gmm":
            assessment_cluster_method = "bayesian_gmm"
        elif used == "gaussian_hmm":
            assessment_cluster_method = "hmm"
        elif used == "hdbscan":
            assessment_cluster_method = "hdbscan"
        elif used == "spectral":
            assessment_cluster_method = "spectral"
        else:
            assessment_cluster_method = "kmeans"
        metrics["assessment_cluster_method"] = assessment_cluster_method
        metrics["OOS_Stability_Scope"] = "temporal_cluster_refit_on_full_sample_embedding"
        assessment = _assess_regime_method(
            method=name,
            cluster_method=assessment_cluster_method,
            embedding=z,
            labels=smoothed,
            matrix=selected_matrix,
            feature_names=selected_features,
            trend_vol=trend_vol,
            non_trend_vol=non_trend_vol,
            frame=frame,
            oos_blocks=oos_blocks,
            window_blocks=window_blocks,
            robustness_positions=robustness_positions,
            config=config,
        )
        metrics.update(assessment)
        metrics["score"] = float(assessment["TotalScore"])
        metrics["is_baseline"] = bool(name in baseline_methods)
        diag_rows.append(metrics)
    label_arrays = {
        str(col): np.asarray(values, dtype=np.int64)
        for col, values in label_cols.items()
    }
    regime_labels = pd.DataFrame(label_cols, index=index)
    for col in regime_labels.columns:
        regime_labels[col] = regime_labels[col].astype("category")
    regime_probabilities = pd.concat(prob_frames, axis=1) if prob_frames else pd.DataFrame(index=index)
    regime_diagnostics = pd.DataFrame(diag_rows).sort_values("TotalScore", ascending=False, kind="mergesort")
    baseline_values = pd.to_numeric(
        regime_diagnostics.loc[
            regime_diagnostics["method"].isin(baseline_methods),
            "TotalScore",
        ],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    baseline_values = baseline_values[np.isfinite(baseline_values)]
    baseline_score = float(np.max(baseline_values)) if baseline_values.size else float("-inf")
    baseline_stability_values = pd.to_numeric(
        regime_diagnostics.loc[
            regime_diagnostics["method"].isin(baseline_methods),
            "stability",
        ],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    baseline_stability_values = baseline_stability_values[np.isfinite(baseline_stability_values)]
    baseline_stability = float(np.max(baseline_stability_values)) if baseline_stability_values.size else float("-inf")
    keep_rows = []
    for row in regime_diagnostics.itertuples(index=False):
        is_baseline = bool(getattr(row, "is_baseline", False))
        total_score = float(getattr(row, "TotalScore", getattr(row, "score", 0.0)))
        beats_baseline = total_score > baseline_score + float(config.keep_candidate_margin)
        beats_stability = float(row.stability) > baseline_stability + float(config.keep_candidate_margin)
        keep_rows.append(
            {
                "method": row.method,
                "score": total_score,
                "TotalScore": total_score,
                "baseline_score": baseline_score,
                "stability": float(row.stability),
                "baseline_stability": baseline_stability,
                "is_baseline": is_baseline,
                "beats_baseline": bool(beats_baseline),
                "beats_stability": bool(beats_stability),
                "keep": bool((not is_baseline) and beats_baseline and beats_stability),
            }
        )
    method_keep = pd.DataFrame(keep_rows)
    kept_methods = set(method_keep.loc[method_keep["keep"], "method"].astype(str))
    top_method = (
        str(regime_diagnostics["method"].iloc[0])
        if not regime_diagnostics.empty and "method" in regime_diagnostics.columns
        else None
    )
    top_total_score = (
        float(pd.to_numeric(regime_diagnostics["TotalScore"], errors="coerce").iloc[0])
        if not regime_diagnostics.empty and "TotalScore" in regime_diagnostics.columns
        else None
    )
    step_rows.append(
        _pipeline_step_row(
            "06_regime_discovery_assessment",
            candidate_method_count=int(len(method_specs)),
            assessed_method_count=int(len(regime_diagnostics)),
            baseline_method_count=int(len(baseline_methods)),
            kept_method_count=int(len(kept_methods)),
            top_method=top_method,
            top_total_score=top_total_score,
            baseline_score=baseline_score if np.isfinite(baseline_score) else None,
            baseline_stability=baseline_stability if np.isfinite(baseline_stability) else None,
            label_column_count=int(regime_labels.shape[1]),
            probability_column_count=int(regime_probabilities.shape[1]),
        )
    )
    all_methods = [name for name, _z, _cluster_method in method_specs]
    regime_transition_features = _build_regime_transition_features(
        frame,
        regime_probabilities,
        regime_labels,
        all_methods,
        label_arrays=label_arrays,
        config=config,
    )
    regime_feature_importance = _compute_regime_feature_importance(
        selected_matrix,
        selected_features,
        regime_labels,
        all_methods,
        label_arrays=label_arrays,
        eps=float(config.eps),
    )
    regime_tradability_diagnostics = _compute_regime_tradability_diagnostics(
        frame,
        selected_matrix,
        selected_features,
        regime_labels,
        all_methods,
        label_arrays=label_arrays,
        timestamp_col=config.timestamp_col,
        symbol_col=config.symbol_col,
    )
    model_methods = sorted(kept_methods)
    model_candidate_tier = "production_candidate"
    model_package_meaningful = bool(model_methods)
    if not model_methods and top_method:
        model_methods = [str(top_method)]
        model_candidate_tier = "fallback_top_method_assessment_only"
        model_package_meaningful = False
    selected_label_cols = [
        col
        for col in regime_labels.columns
        if any(col.startswith(method) for method in model_methods)
    ]
    selected_prob_cols = [
        col
        for col in regime_probabilities.columns
        if any(col.startswith(f"{method}_regime_prob_") for method in model_methods)
    ]
    selected_transition_cols = [
        col
        for col in regime_transition_features.columns
        if any(col.startswith(f"url_{method}_") for method in model_methods)
    ]
    materialized_groups: dict[str, list[str]] = {}
    materialized_parts: list[pd.DataFrame] = []

    def _add_materialized_part(group: str, part: pd.DataFrame) -> None:
        if isinstance(part, pd.DataFrame) and not part.empty and part.shape[1] > 0:
            cols = [str(col) for col in part.columns]
            materialized_groups[str(group)] = cols
            materialized_parts.append(part.astype(np.float32, copy=False))

    use_raw_pca = any(method.startswith("raw_pca") or method == "raw_selected_kmeans" for method in model_methods)
    use_raw_spectral = any(method.startswith("raw_spectral") for method in model_methods)
    use_leaf = any(method.startswith("leaf_") for method in model_methods)
    use_sparse_ae = any(method.startswith("sparse_ae_") for method in model_methods)
    use_contrastive_ae = any(method.startswith("contrastive_ae_") for method in model_methods)
    use_contrastive_leaf = any(method.startswith("contrastive_leaf_") for method in model_methods)
    use_mfa = "mfa" in model_methods
    if use_raw_pca:
        _add_materialized_part("raw_pca_embedding", raw_baseline_embeddings)
    if use_raw_spectral:
        _add_materialized_part("raw_spectral_embedding", raw_spectral_embeddings)
    if use_leaf:
        _add_materialized_part("leaf_embedding", leaf_embeddings)
        _add_materialized_part("leaf_umap_embedding", leaf_umap_embeddings)
        _add_materialized_part("leaf_spectral_embedding", leaf_spectral_embeddings)
    if use_sparse_ae:
        _add_materialized_part("sparse_ae_latent", ae_latents)
    if use_contrastive_ae:
        _add_materialized_part("contrastive_ae_latent", contrastive_ae_latents)
    if use_contrastive_leaf:
        _add_materialized_part("contrastive_leaf_latent", contrastive_leaf_latents)
    if use_mfa:
        _add_materialized_part("mfa_responsibility", mfa_responsibilities)
    label_part = pd.DataFrame(
        {
            col: label_arrays[col].astype(np.float32, copy=False)
            for col in selected_label_cols
            if col in label_arrays
        },
        index=index,
    )
    _add_materialized_part("regime_label", label_part)
    _add_materialized_part("regime_probability", regime_probabilities.reindex(columns=selected_prob_cols))
    _add_materialized_part("regime_transition", regime_transition_features.reindex(columns=selected_transition_cols))
    materialized = pd.concat(
        materialized_parts if materialized_parts else [pd.DataFrame(index=index)],
        axis=1,
    )
    if not materialized.empty:
        materialized = materialized.loc[:, ~materialized.columns.duplicated()].astype(np.float32, copy=False)
    model_regime_features = materialized
    model_regime_feature_metrics = _model_regime_feature_metrics(
        model_regime_features,
        materialized_groups,
        model_methods=model_methods,
        kept_methods=kept_methods,
        all_methods=all_methods,
        regime_diagnostics=regime_diagnostics,
        regime_tradability_diagnostics=regime_tradability_diagnostics,
        candidate_tier=model_candidate_tier,
    )
    step_rows.append(
        _pipeline_step_row(
            "07_regime_feature_generation",
            transition_feature_count=int(regime_transition_features.shape[1]),
            selected_transition_feature_count=int(len(selected_transition_cols)),
            feature_importance_rows=int(len(regime_feature_importance)),
            tradability_diagnostic_rows=int(len(regime_tradability_diagnostics)),
            model_method_count=int(len(model_methods)),
            model_candidate_tier=model_candidate_tier,
            model_package_meaningful=bool(model_package_meaningful),
        )
    )
    step_rows.append(
        _pipeline_step_row(
            "08_model_regime_feature_package",
            model_feature_count=int(model_regime_features.shape[1]),
            model_feature_row_count=int(model_regime_features.shape[0]),
            model_feature_metric_rows=int(len(model_regime_feature_metrics)),
            source_group_count=int(len(materialized_groups)),
            finite_fraction=(
                float(np.isfinite(model_regime_features.to_numpy(dtype=np.float32, copy=False)).mean())
                if not model_regime_features.empty
                else None
            ),
            candidate_tier=model_candidate_tier,
            kept_method_count=int(len(kept_methods)),
            fallback_method=(str(model_methods[0]) if (not kept_methods and model_methods) else None),
        )
    )
    specialist_candidates: list[str] = []
    pipeline_steps = _pipeline_steps_frame(step_rows)
    method_embeddings = {
        "raw_pca": raw_baseline_embeddings,
        "raw_spectral": raw_spectral_embeddings,
        "leaf_pca": leaf_embeddings,
        "leaf_umap": leaf_umap_embeddings,
        "leaf_spectral": leaf_spectral_embeddings,
        "sparse_ae": ae_latents,
        "contrastive_ae": contrastive_ae_latents,
        "contrastive_leaf": contrastive_leaf_latents,
    }
    diagnostics = {
        "schema_version": ADVANCED_REGIME_LEARNING_SCHEMA_VERSION,
        "matrix": matrix_diag,
        "selected_feature_count": int(len(selected)),
        "mfa_supported_feature_count": int(len(mfa_supported_features)),
        "mfa_deprioritized_feature_count": int(len(mfa_deprioritized_features)),
        "mfa_deprioritized_features": mfa_deprioritized_features[:100],
        "regime_transition_feature_count": int(regime_transition_features.shape[1]),
        "selected_regime_transition_feature_count": int(len(selected_transition_cols)),
        "regime_feature_importance_rows": int(len(regime_feature_importance)),
        "regime_tradability_diagnostic_rows": int(len(regime_tradability_diagnostics)),
        "selector": selector_diag.to_dict(orient="records"),
        "autoencoder": ae_diag,
        "contrastive_autoencoder": contrast_diag,
        "contrastive_leaf_autoencoder": leaf_contrast_diag,
        "mfa_log_likelihood": mfa_ll,
        "baseline_methods": sorted(baseline_methods),
        "baseline_score": baseline_score,
        "baseline_stability": baseline_stability,
        "kept_methods": sorted(kept_methods),
        "model_regime_methods": model_methods,
        "model_regime_candidate_tier": model_candidate_tier,
        "model_regime_package_meaningful": bool(model_package_meaningful),
        "model_regime_feature_count": int(model_regime_features.shape[1]),
        "model_regime_feature_metric_rows": int(len(model_regime_feature_metrics)),
        "model_regime_feature_groups": {key: len(value) for key, value in materialized_groups.items()},
        "pipeline_step_count": int(len(pipeline_steps)),
        "persistence_split_dataframes": bool(config.persistence_split_dataframes),
        "assessment": {
            "score": "TotalScore",
            "formula": (
                "0.20*NonTriviality(incremental_after_trend_vol_control) + 0.15*OOS_Stability + "
                "0.10*Dwell_Quality + 0.10*Transition_Stability + "
                "0.15*Feature_Stability + 0.10*Null_Robustness + "
                "0.10*Window_Robustness + 0.10*Geometry_Separation"
            ),
            "nontriviality": (
                "Penalizes pure trend/vol replicas and rewards AUC improvement from "
                "non-trend/vol structure after controlling for trend/vol predictability."
            ),
            "top_method": (
                str(regime_diagnostics["method"].iloc[0])
                if not regime_diagnostics.empty
                else None
            ),
            "top_total_score": (
                float(regime_diagnostics["TotalScore"].iloc[0])
                if not regime_diagnostics.empty and "TotalScore" in regime_diagnostics.columns
                else None
            ),
        },
        "specialist_integration": "disabled_assessment_only",
    }
    return AdvancedRegimeLearningArtifact(
        schema_version=ADVANCED_REGIME_LEARNING_SCHEMA_VERSION,
        selected_features=selected,
        conservative_features=stability.loc[stability["tier"].eq("conservative"), "feature"].astype(str).tolist(),
        strong_features=stability.loc[stability["tier"].isin(["conservative", "strong"]), "feature"].astype(str).tolist(),
        exploratory_features=stability.loc[stability["tier"].ne("drop"), "feature"].astype(str).tolist(),
        stability_frequencies=stability.reset_index(drop=True),
        real_vs_null_importances=importances.reset_index(drop=True),
        leaf_embeddings=leaf_embeddings,
        raw_baseline_embeddings=raw_baseline_embeddings,
        ae_latents=ae_latents,
        contrastive_ae_latents=contrastive_ae_latents,
        contrastive_leaf_latents=contrastive_leaf_latents,
        mfa_responsibilities=mfa_responsibilities,
        mfa_feature_relevance=mfa_feature_relevance.reset_index(drop=True),
        ae_feature_gates=ae_feature_gates.reset_index(drop=True),
        regime_labels=regime_labels,
        regime_probabilities=regime_probabilities,
        regime_transition_features=regime_transition_features,
        regime_feature_importance=regime_feature_importance,
        regime_tradability_diagnostics=regime_tradability_diagnostics,
        regime_diagnostics=regime_diagnostics.reset_index(drop=True),
        pipeline_steps=pipeline_steps.reset_index(drop=True),
        model_regime_features=model_regime_features,
        model_regime_feature_metrics=model_regime_feature_metrics.reset_index(drop=True),
        materialized_features=materialized,
        materialized_feature_groups=materialized_groups,
        specialist_candidate_features=specialist_candidates,
        method_keep_decisions=method_keep,
        row_keys=row_keys,
        method_embeddings=method_embeddings,
        diagnostics=diagnostics,
    )


def augment_frame_with_regime_artifact(
    frame: pd.DataFrame,
    artifact: AdvancedRegimeLearningArtifact | Any,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Return a frame augmented with materialized regime-learning outputs."""

    materialized = getattr(artifact, "materialized_features", pd.DataFrame(index=frame.index))
    out = frame.copy(deep=False)
    added: list[str] = []
    if isinstance(materialized, pd.DataFrame) and not materialized.empty:
        aligned = materialized.reindex(frame.index)
        row_keys = getattr(artifact, "row_keys", pd.DataFrame())
        if isinstance(row_keys, pd.DataFrame) and not row_keys.empty:
            key_cols = [col for col in row_keys.columns if col in frame.columns]
            if key_cols:
                try:
                    left_keys = _normalise_alignment_keys(frame.loc[:, key_cols], key_cols)
                    right_keys = _normalise_alignment_keys(row_keys.loc[:, key_cols], key_cols)
                    right = pd.concat(
                        [
                            right_keys.reset_index(drop=True),
                            materialized.reset_index(drop=True),
                        ],
                        axis=1,
                    )
                    right = right.drop_duplicates(key_cols, keep="last")
                    merged = left_keys.reset_index(drop=True).merge(
                        right,
                        on=key_cols,
                        how="left",
                        sort=False,
                    )
                    keyed = pd.DataFrame(
                        merged.loc[:, materialized.columns].to_numpy(dtype=np.float32, copy=False),
                        index=frame.index,
                        columns=materialized.columns,
                    )
                    keyed_coverage = float(np.isfinite(keyed.to_numpy(dtype=np.float32, copy=False)).mean())
                    index_coverage = float(np.isfinite(aligned.to_numpy(dtype=np.float32, copy=False)).mean())
                    if keyed_coverage >= index_coverage:
                        aligned = keyed
                    else:
                        aligned = aligned.where(aligned.notna(), keyed)
                except Exception:
                    pass
        for col in aligned.columns:
            if col not in out.columns:
                out[col] = pd.to_numeric(aligned[col], errors="coerce").astype(np.float32)
                added.append(str(col))
    candidates = _stable_unique(
        [str(c) for c in getattr(artifact, "specialist_candidate_features", []) if str(c) in out.columns]
        + added
    )
    diag = {
        "enabled": True,
        "used": bool(candidates),
        "added_columns": added,
        "candidate_features": candidates,
        "candidate_feature_count": int(len(candidates)),
        "schema_version": getattr(artifact, "schema_version", None),
    }
    return out, candidates, diag


def regime_artifact_assessment_summary(
    artifact: AdvancedRegimeLearningArtifact | Any,
) -> dict[str, Any]:
    """Summarise an unsupervised regime artifact without injecting its features."""

    diagnostics = getattr(artifact, "diagnostics", {}) or {}
    out: dict[str, Any] = {
        "enabled": True,
        "used": False,
        "reason": "assessment_only_not_injected",
        "schema_version": getattr(artifact, "schema_version", None),
        "selected_feature_count": int(len(getattr(artifact, "selected_features", []) or [])),
        "specialist_candidate_feature_count": int(
            len(getattr(artifact, "specialist_candidate_features", []) or [])
        ),
        "model_regime_feature_count": int(
            getattr(artifact, "model_regime_features", pd.DataFrame()).shape[1]
        ),
        "model_regime_feature_metric_rows": int(
            len(getattr(artifact, "model_regime_feature_metrics", pd.DataFrame()))
        ),
        "pipeline_step_count": int(len(getattr(artifact, "pipeline_steps", pd.DataFrame()))),
        "regime_tradability_diagnostic_rows": int(
            len(getattr(artifact, "regime_tradability_diagnostics", pd.DataFrame()))
        ),
        "kept_methods": list(diagnostics.get("kept_methods", [])),
        "model_regime_methods": list(diagnostics.get("model_regime_methods", [])),
        "model_regime_candidate_tier": diagnostics.get("model_regime_candidate_tier"),
        "model_regime_package_meaningful": bool(diagnostics.get("model_regime_package_meaningful", False)),
    }
    assessment = diagnostics.get("assessment", {})
    if isinstance(assessment, Mapping):
        out["assessment"] = dict(assessment)
    regime_diagnostics = getattr(artifact, "regime_diagnostics", pd.DataFrame())
    if isinstance(regime_diagnostics, pd.DataFrame) and not regime_diagnostics.empty:
        out["assessment_method_count"] = int(len(regime_diagnostics))
        if "TotalScore" in regime_diagnostics.columns:
            ordered = regime_diagnostics.sort_values(
                "TotalScore",
                ascending=False,
                kind="mergesort",
            )
            out["top_method"] = str(ordered["method"].iloc[0]) if "method" in ordered.columns else None
            out["top_total_score"] = float(pd.to_numeric(ordered["TotalScore"], errors="coerce").iloc[0])
    return out


def _normalise_alignment_keys(keys: pd.DataFrame, key_cols: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=keys.index)
    for col in key_cols:
        values = keys[col]
        if "time" in str(col).lower() or np.issubdtype(values.dtype, np.datetime64):
            out[col] = pd.to_datetime(values, utc=True, errors="coerce")
        else:
            out[col] = values.astype(str)
    return out


def _json_ready(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


_ARTIFACT_DATAFRAME_FIELDS = [
    "stability_frequencies",
    "real_vs_null_importances",
    "leaf_embeddings",
    "raw_baseline_embeddings",
    "ae_latents",
    "contrastive_ae_latents",
    "contrastive_leaf_latents",
    "mfa_responsibilities",
    "mfa_feature_relevance",
    "ae_feature_gates",
    "regime_labels",
    "regime_probabilities",
    "regime_transition_features",
    "regime_feature_importance",
    "regime_tradability_diagnostics",
    "regime_diagnostics",
    "pipeline_steps",
    "model_regime_features",
    "model_regime_feature_metrics",
    "materialized_features",
    "method_keep_decisions",
    "row_keys",
]


def _write_pickle(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _split_artifact_dataframes(
    artifact: AdvancedRegimeLearningArtifact,
    output_dir: Path,
) -> tuple[AdvancedRegimeLearningArtifact, dict[str, Any]]:
    frames_dir = output_dir / "advanced_regime_learning_frames"
    fields: dict[str, str] = {}
    empty_updates: dict[str, Any] = {}
    for field_name in _ARTIFACT_DATAFRAME_FIELDS:
        value = getattr(artifact, field_name, None)
        if isinstance(value, pd.DataFrame) and not value.empty:
            rel = Path("advanced_regime_learning_frames") / f"{field_name}.pkl"
            _write_pickle(value, output_dir / rel)
            fields[field_name] = str(rel)
        empty_updates[field_name] = pd.DataFrame()
    method_embeddings: dict[str, str] = {}
    for key, value in (artifact.method_embeddings or {}).items():
        if isinstance(value, pd.DataFrame) and not value.empty:
            safe_key = str(key).replace("/", "_")
            rel = Path("advanced_regime_learning_frames") / "method_embeddings" / f"{safe_key}.pkl"
            _write_pickle(value, output_dir / rel)
            method_embeddings[str(key)] = str(rel)
    split_meta = {
        "enabled": True,
        "format": "pickle",
        "frames_dir": str(frames_dir.relative_to(output_dir)),
        "fields": fields,
        "method_embeddings": method_embeddings,
    }
    diagnostics = dict(artifact.diagnostics or {})
    diagnostics["split_persistence"] = split_meta
    core = replace(
        artifact,
        **empty_updates,
        method_embeddings={},
        diagnostics=diagnostics,
    )
    return core, split_meta


def _restore_split_artifact_dataframes(
    artifact: AdvancedRegimeLearningArtifact,
    base_dir: Path,
) -> AdvancedRegimeLearningArtifact:
    diagnostics = dict(artifact.diagnostics or {})
    split_meta = diagnostics.get("split_persistence", {})
    if not isinstance(split_meta, Mapping) or not bool(split_meta.get("enabled", False)):
        return artifact
    updates: dict[str, Any] = {}
    fields = split_meta.get("fields", {})
    if isinstance(fields, Mapping):
        for field_name, rel in fields.items():
            if str(field_name) in _ARTIFACT_DATAFRAME_FIELDS:
                path = base_dir / str(rel)
                if path.exists():
                    updates[str(field_name)] = _read_pickle(path)
    method_embeddings: dict[str, pd.DataFrame] = {}
    raw_embeddings = split_meta.get("method_embeddings", {})
    if isinstance(raw_embeddings, Mapping):
        for key, rel in raw_embeddings.items():
            path = base_dir / str(rel)
            if path.exists():
                value = _read_pickle(path)
                if isinstance(value, pd.DataFrame):
                    method_embeddings[str(key)] = value
    if method_embeddings:
        updates["method_embeddings"] = method_embeddings
    return replace(artifact, **updates) if updates else artifact


def save_advanced_regime_learning_artifact(
    artifact: AdvancedRegimeLearningArtifact,
    output_dir: str | Path,
) -> dict[str, str]:
    """Persist the full regime-learning artifact and a compact JSON manifest."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_enabled = bool((artifact.diagnostics or {}).get("persistence_split_dataframes", True))
    split_meta: dict[str, Any] = {"enabled": False}
    artifact_to_save = artifact
    if split_enabled:
        artifact_to_save, split_meta = _split_artifact_dataframes(artifact, out_dir)
    artifact_path = out_dir / "advanced_regime_learning_artifact.pkl"
    _write_pickle(artifact_to_save, artifact_path)
    manifest = {
        "schema_version": artifact.schema_version,
        "selected_feature_count": int(len(artifact.selected_features)),
        "conservative_feature_count": int(len(artifact.conservative_features)),
        "strong_feature_count": int(len(artifact.strong_features)),
        "exploratory_feature_count": int(len(artifact.exploratory_features)),
        "specialist_candidate_feature_count": int(len(artifact.specialist_candidate_features)),
        "materialized_feature_count": int(artifact.materialized_features.shape[1]),
        "model_regime_feature_count": int(
            getattr(artifact, "model_regime_features", pd.DataFrame()).shape[1]
        ),
        "model_regime_feature_metric_rows": int(
            len(getattr(artifact, "model_regime_feature_metrics", pd.DataFrame()))
        ),
        "pipeline_step_count": int(len(getattr(artifact, "pipeline_steps", pd.DataFrame()))),
        "regime_transition_feature_count": int(artifact.regime_transition_features.shape[1]),
        "regime_feature_importance_rows": int(len(artifact.regime_feature_importance)),
        "regime_tradability_diagnostic_rows": int(
            len(getattr(artifact, "regime_tradability_diagnostics", pd.DataFrame()))
        ),
        "split_persistence": split_meta,
        "kept_methods": list(artifact.diagnostics.get("kept_methods", [])),
        "diagnostics": artifact.diagnostics,
    }
    manifest_path = out_dir / "advanced_regime_learning_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths = {"artifact": str(artifact_path), "manifest": str(manifest_path)}
    if bool(split_meta.get("enabled", False)):
        paths["dataframes"] = str(out_dir / str(split_meta.get("frames_dir", "advanced_regime_learning_frames")))
    return paths


def load_advanced_regime_learning_artifact(
    path_or_dir: str | Path,
) -> AdvancedRegimeLearningArtifact:
    """Load a persisted advanced regime-learning artifact."""

    path = Path(path_or_dir)
    if path.is_dir():
        base_dir = path
        path = path / "advanced_regime_learning_artifact.pkl"
    else:
        base_dir = path.parent
    with path.open("rb") as fh:
        artifact = pickle.load(fh)
    if not isinstance(artifact, AdvancedRegimeLearningArtifact):
        raise TypeError(f"Unexpected artifact type: {type(artifact).__name__}")
    return _restore_split_artifact_dataframes(artifact, base_dir)
