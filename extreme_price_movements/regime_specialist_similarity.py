"""Current-regime analogue scoring and specialist sample weights.

This module is intentionally standalone. It computes a current-regime
fingerprint, finds historical analogue windows, assigns
``similarity_to_current`` to rows, and builds sample-weight multipliers for a
shadow current-regime specialist. It does not mutate base/meta training by
itself.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REGIME_SPECIALIST_SCHEMA_VERSION = "regime_specialist_similarity_v1"


DEFAULT_MARKET_REGIME_TOKENS: tuple[str, ...] = (
    "volatility_percentile",
    "vol_percentile",
    "volatility_pct",
    "vol_pct",
    "volume_percentile",
    "volume_pct",
    "volume_rank",
    "dollar_volume",
    "correlation_percentile",
    "correlation_pct",
    "corr_percentile",
    "dispersion",
    "funding_average",
    "funding_avg",
    "funding_dispersion",
    "funding",
    "oi_growth",
    "open_interest_growth",
    "oi_volume",
    "oi_over_volume",
    "oi_to_volume",
    "breadth",
    "trend_strength",
    "price_entropy",
    "entropy",
)

DEFAULT_DRIFT_TOKENS: tuple[str, ...] = (
    "drift",
    "psi",
    "ks",
    "wasserstein",
    "mahalanobis",
    "cov_shift",
    "mean_shift",
    "std_shift",
    "quantile_shift",
    "prediction_distribution",
    "inference_drift",
    "feature_drift",
    "population_drift",
)

DEFAULT_COVARIANCE_TOKENS: tuple[str, ...] = (
    "cov",
    "corr",
    "eigen",
    "pc1",
    "dispersion",
    "volatility",
    "volume",
    "funding",
    "oi",
    "breadth",
    "trend",
    "drift",
    "psi",
    "ks",
)

DRIFT_FAMILY_ORDER: tuple[str, ...] = (
    "psi",
    "ks",
    "wasserstein",
    "mahalanobis",
    "prediction_distribution",
    "covariance",
    "contribution",
    "base_model",
    "meta_model",
    "row_drift",
    "raw_state",
    "other",
)

DRIFT_GLOBAL_METRIC_COUNT = 11
DRIFT_FAMILY_METRIC_COUNT = 5

DEFAULT_ASSET_RETURN_CANDIDATES: tuple[str, ...] = (
    "beta_neutral_residual_return",
    "cluster_neutral_residual_return",
    "residual_return_1h",
    "return_1h",
    "ret_1h",
    "log_return",
    "returns",
)

DEFAULT_EXCLUDE_TOKENS: tuple[str, ...] = (
    "target",
    "label",
    "future",
    "forward",
    "wallet",
    "net_pnl",
    "pnl",
    "exit_price",
)


@dataclass(frozen=True)
class RegimeSimilarityConfig:
    current_window_days: float = 28.0
    candidate_window_days: float = 28.0
    day_window_days: float = 1.0
    embargo_days: float = 0.0
    label_horizon_hours: float = 0.0
    label_end_col: Optional[str] = None
    recency_decay_per_week: float = 0.67

    drift_weight: float = 0.40
    covariance_weight: float = 0.35
    regime_weight: float = 0.15
    knn_weight: float = 0.10
    domain_classifier_weight: float = 0.10
    assessment_min_aligned_fraction: float = 0.80
    assessment_allow_timestamp_only_alignment: bool = False
    ae_weight: float = 0.10
    alpha: float = 1.5
    tau: Optional[float] = None

    analogue_threshold: float = 0.55
    normal_threshold: float = 0.15

    knn_k: int = 25
    max_knn_current_rows: int = 2000
    max_knn_candidate_rows: int = 5000
    max_knn_historical_rows: int = 50000
    knn_fallback_chunk_pairs: int = 2_000_000
    max_fingerprint_rows_per_window: int = 20000
    max_day_fingerprint_rows: int = 10000
    max_covariance_features: int = 48
    max_asset_covariance_assets: int = 100
    max_asset_covariance_time_rows: int = 50000
    min_asset_observation_fraction: float = 0.60
    asset_covariance_shrinkage: float = 0.10
    cov_feature_cov_eig_weight: float = 0.15
    cov_feature_corr_eig_weight: float = 0.20
    cov_feature_concentration_weight: float = 0.20
    cov_asset_cov_eig_weight: float = 0.10
    cov_asset_corr_eig_weight: float = 0.20
    cov_asset_concentration_weight: float = 0.15
    drift_psi_scale: float = 0.25
    drift_psi_weight: float = 0.18
    drift_ks_weight: float = 0.14
    drift_wasserstein_weight: float = 0.14
    drift_mahalanobis_weight: float = 0.10
    drift_prediction_weight: float = 0.14
    drift_covariance_weight: float = 0.12
    drift_contribution_weight: float = 0.12
    drift_other_weight: float = 0.06
    max_window_diagnostics: int = 50
    top_eigenvalues: int = 5
    asset_return_col: Optional[str] = None

    ae_enabled: bool = True
    ae_min_windows: int = 50
    ae_max_windows: int = 5000
    ae_latent_dim: int = 4
    ae_max_iter: int = 50
    ae_input_noise: float = 0.02
    day_similarity_min_rows: int = 24
    day_similarity_strength: float = 0.50
    feature_engineering_enabled: bool = False
    feature_engineering_max_final_features: int = 40
    feature_engineering_max_pair_candidates: int = 2500
    feature_engineering_univariate_subsample_per_class: int = 8000
    feature_engineering_lgbm_enabled: bool = True
    feature_engineering_elasticnet_enabled: bool = True
    feature_engineering_grouped_cv_folds: int = 5
    feature_engineering_grouped_cv_repeats: int = 3
    feature_engineering_permutation_repeats: int = 2
    feature_engineering_max_permutation_features: int = 80
    feature_engineering_max_permutation_rows: int = 4000
    feature_engineering_max_shap_rows: int = 4000
    feature_engineering_drift_window_days: float = 28.0
    feature_engineering_max_drift_raw_features: int = 80
    feature_engineering_drift_window_max_rows: int = 20000
    feature_engineering_drift_knn_max_rows: int = 4000
    feature_engineering_drift_knn_chunk_pairs: int = 2_000_000
    feature_engineering_domain_score_smoothing_enabled: bool = True
    feature_engineering_domain_score_ewma_half_life_days: float = 1.0
    feature_engineering_domain_score_ewma_max_days: float = 4.0
    feature_engineering_diagnostics_enabled: bool = True
    feature_engineering_run_validation_diagnostics: bool = False
    random_state: int = 42

    min_candidate_rows: int = 24
    min_current_rows: int = 24
    eps: float = 1e-12


@dataclass
class SpecialistWeightConfig:
    current_gamma: float = 1.0
    analogue_gamma: float = 2.0
    replay_gamma: float = 2.0
    recency_power: float = 0.5
    # Legacy per-band knobs retained for config compatibility; replay rows are
    # allocated as one continuous normal+irrelevant bucket via replay_gamma.
    normal_gamma: float = 1.0
    irrelevant_gamma: float = 2.0

    tau_current: float = 20_000.0
    tau_analogue: float = 20_000.0
    tau_normal: float = 50_000.0
    tau_irrelevant: float = 50_000.0

    current_prior: float = 1.00
    analogue_prior: float = 0.90
    normal_prior: float = 0.20
    irrelevant_prior: float = 0.03

    # Legacy replay-strength knobs retained for config compatibility. Active
    # replay mass is governed by n_eff reliability and less_interesting_min/max.
    replay_min: float = 0.10
    replay_max: float = 0.30

    min_current_plus_analogue_mass: float = 0.50
    less_interesting_min_mass: float = 0.10
    less_interesting_max_mass: float = 0.50
    # Legacy individual caps retained for config compatibility; active cap is
    # less_interesting_min/max_mass over normal+irrelevant combined.
    max_normal_mass: float = 0.25
    max_irrelevant_mass: float = 0.05
    min_adaptive_reliability_to_train: float = 0.20

    min_weight: float = 0.05
    max_weight: float = 20.0

    eps: float = 1e-12


def _safe_numeric_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for col in columns:
        if col not in frame.columns:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if vals.notna().any():
            data[str(col)] = vals.to_numpy(dtype=np.float32, copy=False)
    return pd.DataFrame(data, index=frame.index)


def _timestamp_series(frame: pd.DataFrame, timestamp_col: str) -> pd.Series:
    if timestamp_col not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")


def current_regime_recency_weights(
    timestamps: Sequence[Any],
    *,
    current_end: Any | None = None,
    decay_per_week: float = 0.67,
    eps: float = 1e-12,
) -> pd.Series:
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    if current_end is None:
        end = ts.max()
    else:
        end = pd.to_datetime(current_end, utc=True, errors="coerce")
    if pd.isna(end) or not ts.notna().any():
        weights = pd.Series(1.0, index=ts.index, dtype=np.float64)
        return weights / max(float(weights.sum()), eps)
    age_weeks = ((end - ts).dt.total_seconds() / (7.0 * 24.0 * 3600.0)).clip(lower=0.0)
    weights = np.power(float(decay_per_week), age_weeks.to_numpy(dtype=np.float64))
    weights = np.where(np.isfinite(weights), weights, 0.0)
    total = float(np.sum(weights))
    if total <= eps:
        weights = np.ones(len(ts), dtype=np.float64)
        total = float(len(ts))
    return pd.Series(weights / max(total, eps), index=ts.index, dtype=np.float64)


def _timestamp_ns(ts: pd.Series) -> np.ndarray:
    values = pd.to_datetime(ts, utc=True, errors="coerce").astype("int64").to_numpy(
        dtype=np.int64,
        copy=False,
    )
    return np.where(values > 0, values, 0).astype(np.int64, copy=False)


def _position_recency_weights(
    timestamp_ns: np.ndarray,
    positions: np.ndarray,
    *,
    current_end_ns: int | None = None,
    decay_per_week: float,
    eps: float,
) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.int64)
    if pos.size == 0:
        return np.zeros(0, dtype=np.float64)
    ts = np.asarray(timestamp_ns, dtype=np.int64)[pos]
    valid = ts > 0
    if current_end_ns is None or int(current_end_ns) <= 0:
        end_ns = int(np.max(ts[valid])) if bool(valid.any()) else 0
    else:
        end_ns = int(current_end_ns)
    if end_ns <= 0 or not bool(valid.any()):
        weights = np.ones(len(pos), dtype=np.float64)
        return weights / max(float(weights.sum()), eps)
    week_ns = 7.0 * 24.0 * 3600.0 * 1e9
    age_weeks = np.clip((float(end_ns) - ts.astype(np.float64)) / week_ns, 0.0, None)
    weights = np.power(float(decay_per_week), age_weeks)
    weights = np.where(valid & np.isfinite(weights), weights, 0.0)
    total = float(np.sum(weights))
    if total <= eps:
        weights = np.ones(len(pos), dtype=np.float64)
        total = float(len(pos))
    return weights / max(total, eps)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not bool(mask.any()):
        return 0.0
    return float(np.average(values[mask], weights=weights[mask]))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not bool(mask.any()):
        return 0.0
    vals = values[mask].astype(np.float64, copy=False)
    w = weights[mask].astype(np.float64, copy=False)
    order = np.argsort(vals, kind="mergesort")
    vals = vals[order]
    w = w[order]
    cdf = np.cumsum(w)
    cutoff = 0.5 * float(cdf[-1])
    return float(vals[int(np.searchsorted(cdf, cutoff, side="left"))])


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not bool(mask.any()):
        return 0.0
    vals = values[mask].astype(np.float64, copy=False)
    w = weights[mask].astype(np.float64, copy=False)
    order = np.argsort(vals, kind="mergesort")
    vals = vals[order]
    w = w[order]
    cdf = np.cumsum(w)
    cutoff = float(np.clip(quantile, 0.0, 1.0)) * float(cdf[-1])
    return float(vals[int(np.searchsorted(cdf, cutoff, side="left"))])


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Sequence[float],
) -> list[float]:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not bool(mask.any()):
        return [0.0 for _ in quantiles]
    vals = values[mask].astype(np.float64, copy=False)
    w = weights[mask].astype(np.float64, copy=False)
    order = np.argsort(vals, kind="mergesort")
    vals = vals[order]
    w = w[order]
    cdf = np.cumsum(w)
    total = float(cdf[-1])
    return [
        float(vals[int(np.searchsorted(cdf, float(np.clip(q, 0.0, 1.0)) * total, side="left"))])
        for q in quantiles
    ]


def _weighted_fraction(mask_values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.asarray(mask_values, dtype=bool)
    w = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(w) & (w > 0.0)
    if not bool(valid.any()):
        return 0.0
    return float(np.average(mask[valid].astype(np.float64), weights=w[valid]))


def _weighted_slope_ns(values: np.ndarray, timestamp_ns: np.ndarray, weights: np.ndarray) -> float:
    y = np.asarray(values, dtype=np.float64)
    x = np.asarray(timestamp_ns, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(x) & (x > 0.0) & np.isfinite(w) & (w > 0.0)
    if int(np.sum(mask)) < 3:
        return 0.0
    x = x[mask]
    y = y[mask]
    w = w[mask]
    x = (x - float(np.min(x))) / max(float(np.max(x) - np.min(x)), 1.0)
    x_mean = float(np.average(x, weights=w))
    y_mean = float(np.average(y, weights=w))
    denom = float(np.average((x - x_mean) ** 2, weights=w))
    if denom <= 1e-12:
        return 0.0
    return float(np.average((x - x_mean) * (y - y_mean), weights=w) / denom)


def _weighted_slope(values: np.ndarray, timestamps: pd.Series, weights: np.ndarray) -> float:
    y = np.asarray(values, dtype=np.float64)
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    x = ts.astype("int64").to_numpy(dtype=np.float64)
    finite_x = np.isfinite(x) & (x > 0)
    mask = np.isfinite(y) & np.isfinite(weights) & (weights > 0.0) & finite_x
    if int(np.sum(mask)) < 3:
        return 0.0
    x = x[mask]
    y = y[mask]
    w = weights[mask].astype(np.float64, copy=False)
    x = (x - float(np.min(x))) / max(float(np.max(x) - np.min(x)), 1.0)
    x_mean = float(np.average(x, weights=w))
    y_mean = float(np.average(y, weights=w))
    denom = float(np.average((x - x_mean) ** 2, weights=w))
    if denom <= 1e-12:
        return 0.0
    return float(np.average((x - x_mean) * (y - y_mean), weights=w) / denom)


def _robust_scale(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not bool(finite.any()):
        return np.zeros_like(arr, dtype=np.float32), 0.0, 1.0
    center = float(np.nanmedian(arr[finite]))
    mad = float(np.nanmedian(np.abs(arr[finite] - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(np.nanstd(arr[finite]))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    z = (np.where(finite, arr, center) - center) / scale
    return np.clip(z, -8.0, 8.0).astype(np.float32), center, scale


def _fit_robust_column_scaler(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> dict[str, tuple[float, float]]:
    numeric = _safe_numeric_frame(frame, columns)
    scaler: dict[str, tuple[float, float]] = {}
    for col in numeric.columns:
        _z, center, scale = _robust_scale(numeric[col].to_numpy(dtype=np.float64))
        scaler[str(col)] = (float(center), float(scale))
    return scaler


def _scaled_numeric_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
    scaler: Mapping[str, tuple[float, float]],
) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for col in columns:
        col_s = str(col)
        if col_s not in frame.columns or col_s not in scaler:
            continue
        center, scale = scaler[col_s]
        vals = pd.to_numeric(frame[col_s], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        arr = vals.to_numpy(dtype=np.float64, copy=False)
        filled = np.where(np.isfinite(arr), arr, float(center))
        data[col_s] = np.clip(
            (filled - float(center)) / max(float(scale), 1e-9),
            -8.0,
            8.0,
        ).astype(np.float32)
    return pd.DataFrame(data, index=frame.index)


def _per_asset_robust_z(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    symbol_col: str,
    fit_frame: pd.DataFrame | None = None,
    min_symbol_fit_rows: int = 12,
) -> pd.DataFrame:
    numeric = _safe_numeric_frame(frame, columns)
    if numeric.empty:
        return numeric
    fit = fit_frame if fit_frame is not None and not fit_frame.empty else frame
    cols = [str(col) for col in numeric.columns]
    values = numeric.loc[:, cols].to_numpy(dtype=np.float32, copy=True)
    fit_numeric = _safe_numeric_frame(fit, cols).reindex(columns=cols)
    fit_values = fit_numeric.to_numpy(dtype=np.float32, copy=True)

    global_centers = np.zeros(len(cols), dtype=np.float32)
    global_scales = np.ones(len(cols), dtype=np.float32)
    for j in range(len(cols)):
        fit_col = fit_values[:, j] if fit_values.shape[0] else values[:, j]
        if not bool(np.isfinite(fit_col).any()):
            fit_col = values[:, j]
        _fit_z, center, scale = _robust_scale(fit_col)
        global_centers[j] = float(center)
        global_scales[j] = max(float(scale), 1e-9)

    out_values = np.clip(
        (
            np.where(np.isfinite(values), values, global_centers.reshape(1, -1))
            - global_centers.reshape(1, -1)
        )
        / np.maximum(global_scales.reshape(1, -1), 1e-9),
        -8.0,
        8.0,
    ).astype(np.float32)

    if symbol_col not in frame.columns:
        return pd.DataFrame(out_values, index=frame.index, columns=cols, dtype=np.float32)

    def _position_groups(raw: Sequence[Any]) -> dict[str, np.ndarray]:
        symbols_arr = pd.Series(raw).astype("string").fillna("").to_numpy(dtype=str)
        if symbols_arr.size == 0:
            return {}
        order = np.argsort(symbols_arr, kind="mergesort")
        sorted_symbols = symbols_arr[order]
        starts = np.r_[0, np.flatnonzero(sorted_symbols[1:] != sorted_symbols[:-1]) + 1]
        ends = np.r_[starts[1:], len(sorted_symbols)]
        return {
            str(sorted_symbols[start]): order[start:end].astype(np.int64, copy=False)
            for start, end in zip(starts, ends)
        }

    frame_groups = _position_groups(frame[symbol_col].to_numpy())
    fit_groups = (
        _position_groups(fit[symbol_col].to_numpy())
        if symbol_col in fit.columns
        else {}
    )
    min_rows = max(1, int(min_symbol_fit_rows))
    for sym, frame_pos in frame_groups.items():
        fit_pos = fit_groups.get(str(sym))
        if fit_pos is None or fit_pos.size == 0:
            continue
        fit_sub = fit_values[fit_pos]
        finite_counts = np.isfinite(fit_sub).sum(axis=0)
        eligible = finite_counts >= min_rows
        if not bool(eligible.any()):
            continue
        eligible_idx = np.flatnonzero(eligible)
        sub = fit_sub[:, eligible_idx].astype(np.float64, copy=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            centers = np.nanmedian(np.where(np.isfinite(sub), sub, np.nan), axis=0)
            mad = np.nanmedian(np.abs(sub - centers.reshape(1, -1)), axis=0)
            scales = 1.4826 * mad
            bad_scale = ~np.isfinite(scales) | (scales <= 1e-9)
            if bool(bad_scale.any()):
                std = np.nanstd(np.where(np.isfinite(sub), sub, np.nan), axis=0)
                scales = np.where(bad_scale & np.isfinite(std) & (std > 1e-9), std, scales)
        bad_scale = ~np.isfinite(scales) | (scales <= 1e-9)
        centers = np.where(
            np.isfinite(centers),
            centers,
            global_centers[eligible_idx].astype(np.float64, copy=False),
        )
        centers = np.where(bad_scale, global_centers[eligible_idx], centers)
        scales = np.where(bad_scale, global_scales[eligible_idx], scales)
        vals = values[np.ix_(frame_pos, eligible_idx)].astype(np.float64, copy=False)
        out_values[np.ix_(frame_pos, eligible_idx)] = np.clip(
            (np.where(np.isfinite(vals), vals, centers.reshape(1, -1)) - centers.reshape(1, -1))
            / np.maximum(scales.reshape(1, -1), 1e-9),
            -8.0,
            8.0,
        ).astype(np.float32)
    return pd.DataFrame(out_values, index=frame.index, columns=cols, dtype=np.float32)


def _matches_any_token(name: str, tokens: Sequence[str]) -> bool:
    low = str(name).lower()
    return any(str(token).lower() in low for token in tokens)


def _is_excluded_feature(name: str) -> bool:
    low = str(name).lower()
    return low in {"timestamp", "symbol", "asset", "strategy_id", "side"} or _matches_any_token(
        low, DEFAULT_EXCLUDE_TOKENS
    )


def _stable_unique_strings(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = str(value)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _feature_engineering_knn_unsafe_reason(column: str) -> str | None:
    low = str(column).lower()
    unsafe_substrings = (
        "knn",
        "nearest",
        "neighbor",
        "distance",
        "mahalanobis",
        "reconstruction",
        "anomaly",
        "rarity",
        "drift",
        "wasserstein",
        "score",
        "uncertainty",
    )
    for token in unsafe_substrings:
        if token in low:
            return token
    parts = {part for part in re.split(r"[^a-z0-9]+", low) if part}
    if "psi" in parts:
        return "psi"
    if "ks" in parts or "kolmogorov" in parts:
        return "ks"
    return None


def _feature_engineering_knn_safe_columns(columns: Sequence[str]) -> list[str]:
    safe: list[str] = []
    for col in columns:
        if _feature_engineering_knn_unsafe_reason(str(col)) is not None:
            continue
        safe.append(str(col))
    return safe


def infer_regime_specialist_columns(
    frame: pd.DataFrame,
    *,
    selected_feature_columns: Sequence[str] | None = None,
    market_columns: Sequence[str] | None = None,
    drift_columns: Sequence[str] | None = None,
    covariance_columns: Sequence[str] | None = None,
    knn_columns: Sequence[str] | None = None,
    config: RegimeSimilarityConfig = RegimeSimilarityConfig(),
) -> dict[str, list[str]]:
    numeric_cols = [
        str(c)
        for c in frame.columns
        if not _is_excluded_feature(str(c)) and pd.api.types.is_numeric_dtype(frame[c])
    ]
    selected = [str(c) for c in (selected_feature_columns or []) if str(c) in numeric_cols]
    market = (
        [str(c) for c in market_columns if str(c) in numeric_cols]
        if market_columns is not None
        else [c for c in numeric_cols if _matches_any_token(c, DEFAULT_MARKET_REGIME_TOKENS)]
    )
    drift = (
        [str(c) for c in drift_columns if str(c) in numeric_cols]
        if drift_columns is not None
        else [c for c in numeric_cols if _matches_any_token(c, DEFAULT_DRIFT_TOKENS)]
    )
    cov = (
        [str(c) for c in covariance_columns if str(c) in numeric_cols]
        if covariance_columns is not None
        else [
            c
            for c in (selected or numeric_cols)
            if _matches_any_token(c, DEFAULT_COVARIANCE_TOKENS)
        ]
    )
    if not cov:
        cov = selected[: int(config.max_covariance_features)] or numeric_cols[: int(config.max_covariance_features)]
    cov = list(dict.fromkeys(cov))[: int(config.max_covariance_features)]
    knn = (
        [str(c) for c in knn_columns if str(c) in numeric_cols]
        if knn_columns is not None
        else list(dict.fromkeys(market + drift + cov))[: int(config.max_covariance_features)]
    )
    return {
        "market": list(dict.fromkeys(market)),
        "drift": list(dict.fromkeys(drift)),
        "covariance": list(dict.fromkeys(cov)),
        "knn": list(dict.fromkeys(knn)),
    }


def _column_selection_diagnostics(
    frame: pd.DataFrame,
    resolved: Mapping[str, Sequence[str]],
    *,
    market_columns: Sequence[str] | None,
    drift_columns: Sequence[str] | None,
    covariance_columns: Sequence[str] | None,
    knn_columns: Sequence[str] | None,
) -> dict[str, Any]:
    requested = {
        "market": market_columns,
        "drift": drift_columns,
        "covariance": covariance_columns,
        "knn": knn_columns,
    }
    out: dict[str, Any] = {}
    frame_cols = {str(c) for c in frame.columns}
    for group, req in requested.items():
        req_list = [str(c) for c in req] if req is not None else []
        resolved_list = [str(c) for c in resolved.get(group, [])]
        out[group] = {
            "source": "explicit" if req is not None else "token_inferred",
            "requested_count": int(len(req_list)),
            "resolved_count": int(len(resolved_list)),
            "missing_requested": [c for c in req_list if c not in frame_cols],
        }
    return out


def _window_weights(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    decay_per_week: float,
) -> np.ndarray:
    ts = _timestamp_series(frame, timestamp_col)
    return current_regime_recency_weights(
        ts,
        current_end=ts.max(),
        decay_per_week=decay_per_week,
    ).to_numpy(dtype=np.float64)


def _market_fingerprint(
    frame: pd.DataFrame,
    market_z: pd.DataFrame,
    columns: Sequence[str],
    *,
    timestamp_col: str,
    weights: np.ndarray,
) -> np.ndarray:
    if not columns or market_z.empty or frame.empty:
        return np.zeros(0, dtype=np.float32)
    ts = _timestamp_series(frame, timestamp_col)
    feats: list[float] = []
    for col in columns:
        if col not in market_z.columns:
            continue
        vals = market_z.loc[frame.index, col].to_numpy(dtype=np.float64)
        raw = (
            pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if col in frame.columns
            else pd.Series(np.nan, index=frame.index)
        )
        raw_arr = raw.to_numpy(dtype=np.float64, copy=False)
        p10 = _weighted_quantile(vals, weights, 0.10)
        p90 = _weighted_quantile(vals, weights, 0.90)
        feats.append(_weighted_mean(vals, weights))
        feats.append(_weighted_median(vals, weights))
        feats.append(_weighted_slope(vals, ts, weights))
        feats.append(float(p90 - p10))
        feats.append(p10)
        feats.append(p90)
        feats.append(_weighted_fraction(vals > 1.0, weights))
        feats.append(_weighted_fraction(vals < -1.0, weights))
        feats.append(_weighted_fraction(np.abs(vals) > 2.0, weights))
        feats.append(_weighted_fraction(~np.isfinite(raw_arr), weights))
    return np.asarray(feats, dtype=np.float32)


def _numeric_matrix(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    scaler: Mapping[str, tuple[float, float]] | None = None,
    fill_scaled_missing: bool = False,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    candidate_cols = [
        str(col)
        for col in columns
        if str(col) in frame.columns and (scaler is None or str(col) in scaler)
    ]
    if not candidate_cols:
        return [], np.zeros((len(frame), 0), dtype=np.float32), np.zeros((len(frame), 0), dtype=bool)
    values = np.empty((len(frame), len(candidate_cols)), dtype=np.float32)
    missing_values = np.empty((len(frame), len(candidate_cols)), dtype=bool)
    cols: list[str] = []
    out_j = 0
    for col_s in candidate_cols:
        arr = pd.to_numeric(frame[col_s], errors="coerce").to_numpy(dtype=np.float32, copy=True)
        arr[~np.isfinite(arr)] = np.nan
        missing = ~np.isfinite(arr)
        if scaler is not None:
            center, scale = scaler[col_s]
            filled = np.where(missing, float(center), arr.astype(np.float64, copy=False))
            arr = np.clip(
                (filled - float(center)) / max(float(scale), 1e-9),
                -8.0,
                8.0,
            ).astype(np.float32)
        elif fill_scaled_missing:
            z, _center, _scale = _robust_scale(arr)
            arr = z.astype(np.float32, copy=False)
        cols.append(col_s)
        values[:, out_j] = arr.astype(np.float32, copy=False)
        missing_values[:, out_j] = missing.astype(bool, copy=False)
        out_j += 1
    if out_j != values.shape[1]:
        values = values[:, :out_j]
        missing_values = missing_values[:, :out_j]
    return cols, values, missing_values


def _matrix_from_frame(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[list[str], np.ndarray]:
    cols = [str(c) for c in columns if str(c) in frame.columns]
    if not cols:
        return [], np.zeros((len(frame), 0), dtype=np.float32)
    return cols, frame.loc[:, cols].to_numpy(dtype=np.float32, copy=False)


def _market_fingerprint_array(
    market_values: np.ndarray,
    raw_missing: np.ndarray,
    timestamp_ns: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.int64)
    if market_values.ndim != 2 or market_values.shape[1] == 0 or pos.size == 0:
        return np.zeros(0, dtype=np.float32)
    ts = np.asarray(timestamp_ns, dtype=np.int64)[pos]
    feats: list[float] = []
    for j in range(market_values.shape[1]):
        vals = market_values[pos, j].astype(np.float64, copy=False)
        miss = raw_missing[pos, j] if raw_missing.ndim == 2 and raw_missing.shape[1] > j else ~np.isfinite(vals)
        p10, p50, p90 = _weighted_quantiles(vals, weights, (0.10, 0.50, 0.90))
        feats.extend(
            [
                _weighted_mean(vals, weights),
                p50,
                _weighted_slope_ns(vals, ts, weights),
                float(p90 - p10),
                p10,
                p90,
                _weighted_fraction(vals > 1.0, weights),
                _weighted_fraction(vals < -1.0, weights),
                _weighted_fraction(np.abs(vals) > 2.0, weights),
                _weighted_fraction(miss, weights),
            ]
        )
    return np.asarray(feats, dtype=np.float32)


def _weighted_covariance(x: np.ndarray, weights: np.ndarray, eps: float) -> np.ndarray:
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2:
        return np.zeros((0, 0), dtype=np.float32)
    w = np.asarray(weights, dtype=np.float64)
    valid_rows = np.isfinite(x).any(axis=1) & np.isfinite(w) & (w > 0.0)
    if int(np.sum(valid_rows)) < 3:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(x[valid_rows], dtype=np.float64)
    w = w[valid_rows]
    fill = np.nanmedian(np.where(np.isfinite(arr), arr, np.nan), axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    arr = np.where(np.isfinite(arr), arr, fill.reshape(1, -1))
    w = w / max(float(np.sum(w)), eps)
    mean = np.sum(arr * w.reshape(-1, 1), axis=0)
    centered = arr - mean.reshape(1, -1)
    cov = (centered * w.reshape(-1, 1)).T @ centered
    return np.asarray(cov, dtype=np.float32)


def _covariance_fingerprint(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    weights: np.ndarray,
    top_eigenvalues: int,
    eps: float,
    scaler: Mapping[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    numeric = (
        _scaled_numeric_frame(frame, columns, scaler)
        if scaler is not None
        else _safe_numeric_frame(frame, columns)
    )
    if numeric.shape[1] < 2 or len(numeric) < 3:
        return np.zeros(int(top_eigenvalues) * 2 + 5, dtype=np.float32)
    arr = numeric.to_numpy(dtype=np.float32, copy=True)
    if scaler is None:
        for j in range(arr.shape[1]):
            arr[:, j], _c, _s = _robust_scale(arr[:, j])
    cov = _weighted_covariance(arr, weights, eps)
    if cov.size == 0:
        return np.zeros(int(top_eigenvalues) * 2 + 5, dtype=np.float32)
    diag = np.sqrt(np.maximum(np.diag(cov), eps))
    corr = cov / np.maximum(np.outer(diag, diag), eps)
    corr = np.clip(corr, -1.0, 1.0)
    cov_eig = np.sort(np.linalg.eigvalsh(cov).astype(np.float64))[::-1]
    corr_eig = np.sort(np.linalg.eigvalsh(corr).astype(np.float64))[::-1]
    cov_eig = np.maximum(cov_eig, 0.0)
    corr_eig = np.maximum(corr_eig, 0.0)
    k = int(top_eigenvalues)
    cov_top = np.zeros(k, dtype=np.float64)
    corr_top = np.zeros(k, dtype=np.float64)
    cov_top[: min(k, len(cov_eig))] = cov_eig[: min(k, len(cov_eig))]
    corr_top[: min(k, len(corr_eig))] = corr_eig[: min(k, len(corr_eig))]
    eig_sum = float(np.sum(corr_eig))
    p = corr_eig / max(eig_sum, eps)
    p = p[p > eps]
    effective_rank = float(np.exp(-np.sum(p * np.log(p)))) if p.size else 0.0
    pc1_concentration = float(corr_eig[0] / max(eig_sum, eps)) if corr_eig.size else 0.0
    cov_sum = float(np.sum(cov_eig))
    cov_concentration = float(cov_eig[0] / max(cov_sum, eps)) if cov_eig.size else 0.0
    offdiag = corr[np.triu_indices_from(corr, k=1)] if corr.shape[0] > 1 else np.asarray([])
    avg_pairwise_corr = float(np.nanmean(offdiag)) if offdiag.size else 0.0
    abs_avg_pairwise_corr = float(np.nanmean(np.abs(offdiag))) if offdiag.size else 0.0
    return np.asarray(
        [
            *cov_top.tolist(),
            *corr_top.tolist(),
            cov_concentration,
            pc1_concentration,
            effective_rank,
            avg_pairwise_corr,
            abs_avg_pairwise_corr,
        ],
        dtype=np.float32,
    )


def _covariance_fingerprint_array(
    matrix: np.ndarray,
    positions: np.ndarray,
    *,
    weights: np.ndarray,
    top_eigenvalues: int,
    eps: float,
) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[1] < 2 or pos.size < 3:
        return np.zeros(int(top_eigenvalues) * 2 + 5, dtype=np.float32)
    arr = matrix[pos].astype(np.float32, copy=True)
    cov = _weighted_covariance(arr, weights, eps)
    if cov.size == 0:
        return np.zeros(int(top_eigenvalues) * 2 + 5, dtype=np.float32)
    diag = np.sqrt(np.maximum(np.diag(cov), eps))
    corr = cov / np.maximum(np.outer(diag, diag), eps)
    corr = np.clip(corr, -1.0, 1.0)
    cov_eig = np.maximum(np.sort(np.linalg.eigvalsh(cov).astype(np.float64))[::-1], 0.0)
    corr_eig = np.maximum(np.sort(np.linalg.eigvalsh(corr).astype(np.float64))[::-1], 0.0)
    k = int(top_eigenvalues)
    cov_top = np.zeros(k, dtype=np.float64)
    corr_top = np.zeros(k, dtype=np.float64)
    cov_top[: min(k, len(cov_eig))] = cov_eig[: min(k, len(cov_eig))]
    corr_top[: min(k, len(corr_eig))] = corr_eig[: min(k, len(corr_eig))]
    eig_sum = float(np.sum(corr_eig))
    p = corr_eig / max(eig_sum, eps)
    p = p[p > eps]
    effective_rank = float(np.exp(-np.sum(p * np.log(p)))) if p.size else 0.0
    pc1_concentration = float(corr_eig[0] / max(eig_sum, eps)) if corr_eig.size else 0.0
    cov_sum = float(np.sum(cov_eig))
    cov_concentration = float(cov_eig[0] / max(cov_sum, eps)) if cov_eig.size else 0.0
    offdiag = corr[np.triu_indices_from(corr, k=1)] if corr.shape[0] > 1 else np.asarray([])
    avg_pairwise_corr = float(np.nanmean(offdiag)) if offdiag.size else 0.0
    abs_avg_pairwise_corr = float(np.nanmean(np.abs(offdiag))) if offdiag.size else 0.0
    return np.asarray(
        [
            *cov_top.tolist(),
            *corr_top.tolist(),
            cov_concentration,
            pc1_concentration,
            effective_rank,
            avg_pairwise_corr,
            abs_avg_pairwise_corr,
        ],
        dtype=np.float32,
    )


def _infer_asset_return_col(frame: pd.DataFrame, requested: str | None = None) -> str | None:
    if requested is not None and str(requested) in frame.columns and not _is_excluded_feature(str(requested)):
        return str(requested)
    for candidate in DEFAULT_ASSET_RETURN_CANDIDATES:
        if candidate in frame.columns and not _is_excluded_feature(candidate):
            return candidate
    return None


def _asset_covariance_fingerprint(
    frame: pd.DataFrame,
    *,
    return_col: str | None,
    timestamp_col: str,
    symbol_col: str,
    weights: np.ndarray,
    top_eigenvalues: int,
    eps: float,
) -> np.ndarray:
    k = int(top_eigenvalues)
    empty = np.zeros(k * 2 + 5, dtype=np.float32)
    if (
        return_col is None
        or return_col not in frame.columns
        or timestamp_col not in frame.columns
        or symbol_col not in frame.columns
        or frame.empty
    ):
        return empty
    ts = _timestamp_series(frame, timestamp_col)
    vals = pd.to_numeric(frame[return_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    work = pd.DataFrame(
        {
            "_ts": ts,
            "_symbol": frame[symbol_col].astype(str),
            "_return": vals,
            "_weight": np.asarray(weights, dtype=np.float64),
        },
        index=frame.index,
    )
    work = work.dropna(subset=["_ts", "_return"])
    if work["_ts"].nunique() < 3 or work["_symbol"].nunique() < 2:
        return empty
    pivot = work.pivot_table(index="_ts", columns="_symbol", values="_return", aggfunc="mean")
    if pivot.shape[0] < 3 or pivot.shape[1] < 2:
        return empty
    time_weights = work.groupby("_ts", sort=False)["_weight"].mean().reindex(pivot.index)
    w = time_weights.fillna(0.0).to_numpy(dtype=np.float64)
    arr = pivot.to_numpy(dtype=np.float64, copy=True)
    fill = np.nanmedian(np.where(np.isfinite(arr), arr, np.nan), axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    arr = np.where(np.isfinite(arr), arr, fill.reshape(1, -1))
    cov = _weighted_covariance(arr, w, eps)
    if cov.size == 0:
        return empty
    diag = np.sqrt(np.maximum(np.diag(cov), eps))
    corr = cov / np.maximum(np.outer(diag, diag), eps)
    corr = np.clip(corr, -1.0, 1.0)
    cov_eig = np.maximum(np.sort(np.linalg.eigvalsh(cov).astype(np.float64))[::-1], 0.0)
    corr_eig = np.maximum(np.sort(np.linalg.eigvalsh(corr).astype(np.float64))[::-1], 0.0)
    cov_top = np.zeros(k, dtype=np.float64)
    corr_top = np.zeros(k, dtype=np.float64)
    cov_top[: min(k, len(cov_eig))] = cov_eig[: min(k, len(cov_eig))]
    corr_top[: min(k, len(corr_eig))] = corr_eig[: min(k, len(corr_eig))]
    corr_sum = float(np.sum(corr_eig))
    p = corr_eig / max(corr_sum, eps)
    p = p[p > eps]
    effective_rank = float(np.exp(-np.sum(p * np.log(p)))) if p.size else 0.0
    pc1_concentration = float(corr_eig[0] / max(corr_sum, eps)) if corr_eig.size else 0.0
    cov_sum = float(np.sum(cov_eig))
    cov_concentration = float(cov_eig[0] / max(cov_sum, eps)) if cov_eig.size else 0.0
    offdiag = corr[np.triu_indices_from(corr, k=1)] if corr.shape[0] > 1 else np.asarray([])
    avg_pairwise_corr = float(np.nanmean(offdiag)) if offdiag.size else 0.0
    abs_avg_pairwise_corr = float(np.nanmean(np.abs(offdiag))) if offdiag.size else 0.0
    return np.asarray(
        [
            *cov_top.tolist(),
            *corr_top.tolist(),
            cov_concentration,
            pc1_concentration,
            effective_rank,
            avg_pairwise_corr,
            abs_avg_pairwise_corr,
        ],
        dtype=np.float32,
    )


@dataclass
class _AssetReturnCache:
    enabled: bool
    matrix: np.ndarray
    time_ns: np.ndarray
    row_time_pos: np.ndarray
    return_col: str | None = None


def _build_asset_return_cache(
    frame: pd.DataFrame,
    *,
    return_col: str | None,
    timestamp_col: str,
    symbol_col: str,
    max_assets: int,
    max_time_rows: int,
    min_observation_fraction: float,
) -> _AssetReturnCache:
    disabled = _AssetReturnCache(
        enabled=False,
        matrix=np.zeros((0, 0), dtype=np.float32),
        time_ns=np.zeros(0, dtype=np.int64),
        row_time_pos=np.full(len(frame), -1, dtype=np.int64),
        return_col=return_col,
    )
    if (
        return_col is None
        or return_col not in frame.columns
        or timestamp_col not in frame.columns
        or symbol_col not in frame.columns
        or frame.empty
    ):
        return disabled
    ts = _timestamp_series(frame, timestamp_col)
    vals = pd.to_numeric(frame[return_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    work = pd.DataFrame(
        {
            "_ts": ts,
            "_symbol": frame[symbol_col].astype(str),
            "_return": vals,
        },
        index=frame.index,
    ).dropna(subset=["_ts", "_return"])
    if work["_ts"].nunique() < 3 or work["_symbol"].nunique() < 2:
        return disabled
    pivot = work.pivot_table(index="_ts", columns="_symbol", values="_return", aggfunc="mean")
    if pivot.shape[0] < 3 or pivot.shape[1] < 2:
        return disabled
    coverage = pivot.notna().mean(axis=0).sort_values(ascending=False)
    min_obs = float(np.clip(min_observation_fraction, 0.0, 1.0))
    keep = list(coverage[coverage >= min_obs].index)
    if len(keep) < 2:
        keep = list(coverage.head(max(2, min(len(coverage), int(max_assets or len(coverage))))).index)
    if int(max_assets) > 0 and len(keep) > int(max_assets):
        keep = keep[: int(max_assets)]
    pivot = pivot.reindex(columns=keep)
    if pivot.shape[0] < 3 or pivot.shape[1] < 2:
        return disabled
    time_ns = _timestamp_ns(pd.Series(pivot.index))
    order = np.argsort(time_ns, kind="mergesort")
    time_ns = time_ns[order]
    matrix = pivot.to_numpy(dtype=np.float32, copy=True)[order]
    if int(max_time_rows) > 0 and len(time_ns) > int(max_time_rows):
        keep_pos = _subsample_positions(np.arange(len(time_ns), dtype=np.int64), max_rows=int(max_time_rows))
        time_ns = time_ns[keep_pos]
        matrix = matrix[keep_pos]
    row_ts_ns = _timestamp_ns(ts)
    row_time_pos = np.searchsorted(time_ns, row_ts_ns)
    valid = (row_time_pos >= 0) & (row_time_pos < len(time_ns)) & (time_ns[np.clip(row_time_pos, 0, max(len(time_ns) - 1, 0))] == row_ts_ns)
    row_time_pos = np.where(valid, row_time_pos, -1).astype(np.int64)
    return _AssetReturnCache(
        enabled=True,
        matrix=matrix,
        time_ns=time_ns.astype(np.int64, copy=False),
        row_time_pos=row_time_pos,
        return_col=str(return_col),
    )


def _asset_covariance_fingerprint_from_cache(
    cache: _AssetReturnCache,
    positions: np.ndarray,
    weights: np.ndarray,
    *,
    top_eigenvalues: int,
    shrinkage: float,
    eps: float,
) -> np.ndarray:
    k = int(top_eigenvalues)
    empty = np.zeros(k * 2 + 5, dtype=np.float32)
    if not cache.enabled or cache.matrix.shape[0] < 3 or cache.matrix.shape[1] < 2:
        return empty
    pos = np.asarray(positions, dtype=np.int64)
    if pos.size < 3:
        return empty
    row_time_pos = cache.row_time_pos[pos]
    valid = row_time_pos >= 0
    if int(np.sum(valid)) < 3:
        return empty
    time_pos = row_time_pos[valid]
    row_w = np.asarray(weights, dtype=np.float64)[valid]
    unique_time_pos, inverse = np.unique(time_pos, return_inverse=True)
    if unique_time_pos.size < 3:
        return empty
    weight_sum = np.bincount(inverse, weights=row_w, minlength=len(unique_time_pos))
    counts = np.bincount(inverse, minlength=len(unique_time_pos))
    time_weights = weight_sum / np.maximum(counts, 1)
    arr = cache.matrix[unique_time_pos].astype(np.float64, copy=True)
    fill = np.nanmedian(np.where(np.isfinite(arr), arr, np.nan), axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    arr = np.where(np.isfinite(arr), arr, fill.reshape(1, -1))
    cov = _weighted_covariance(arr, time_weights, eps)
    if cov.size == 0:
        return empty
    shrink = float(np.clip(shrinkage, 0.0, 1.0))
    if shrink > 0.0:
        cov = ((1.0 - shrink) * cov + shrink * np.diag(np.diag(cov))).astype(np.float32)
    diag = np.sqrt(np.maximum(np.diag(cov), eps))
    corr = cov / np.maximum(np.outer(diag, diag), eps)
    corr = np.clip(corr, -1.0, 1.0)
    cov_eig = np.maximum(np.sort(np.linalg.eigvalsh(cov).astype(np.float64))[::-1], 0.0)
    corr_eig = np.maximum(np.sort(np.linalg.eigvalsh(corr).astype(np.float64))[::-1], 0.0)
    cov_top = np.zeros(k, dtype=np.float64)
    corr_top = np.zeros(k, dtype=np.float64)
    cov_top[: min(k, len(cov_eig))] = cov_eig[: min(k, len(cov_eig))]
    corr_top[: min(k, len(corr_eig))] = corr_eig[: min(k, len(corr_eig))]
    corr_sum = float(np.sum(corr_eig))
    p = corr_eig / max(corr_sum, eps)
    p = p[p > eps]
    effective_rank = float(np.exp(-np.sum(p * np.log(p)))) if p.size else 0.0
    pc1_concentration = float(corr_eig[0] / max(corr_sum, eps)) if corr_eig.size else 0.0
    cov_sum = float(np.sum(cov_eig))
    cov_concentration = float(cov_eig[0] / max(cov_sum, eps)) if cov_eig.size else 0.0
    offdiag = corr[np.triu_indices_from(corr, k=1)] if corr.shape[0] > 1 else np.asarray([])
    avg_pairwise_corr = float(np.nanmean(offdiag)) if offdiag.size else 0.0
    abs_avg_pairwise_corr = float(np.nanmean(np.abs(offdiag))) if offdiag.size else 0.0
    return np.asarray(
        [
            *cov_top.tolist(),
            *corr_top.tolist(),
            cov_concentration,
            pc1_concentration,
            effective_rank,
            avg_pairwise_corr,
            abs_avg_pairwise_corr,
        ],
        dtype=np.float32,
    )


def _top_mean(vals: np.ndarray, frac: float) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    arr = np.sort(np.abs(arr))[::-1]
    k = max(1, int(math.ceil(float(frac) * arr.size)))
    return float(np.mean(arr[:k]))


def _drift_family(name: str) -> str:
    low = str(name).lower()
    if "wasserstein" in low:
        return "wasserstein"
    if (
        low == "psi"
        or low.startswith("psi_")
        or low.endswith("_psi")
        or "_psi_" in low
    ):
        return "psi"
    if (
        low == "ks"
        or low.startswith("ks_")
        or low.endswith("_ks")
        or "_ks_" in low
        or "kolmogorov" in low
    ):
        return "ks"
    if "mahalanobis" in low:
        return "mahalanobis"
    if low.startswith("base_lgbm_") or low.startswith("base_") or "base_model" in low:
        return "base_model"
    if low.startswith("meta_lgbm_") or low.startswith("meta_") or "meta_model" in low:
        return "meta_model"
    if "prediction_distribution" in low or "pred_distribution" in low or "pred_dist" in low:
        return "prediction_distribution"
    if "cov" in low or "frobenius" in low or "corr_shift" in low:
        return "covariance"
    if "contribution" in low or "contrib" in low:
        return "contribution"
    if low.startswith("row_drift") or "row_drift" in low:
        return "row_drift"
    if "raw_state" in low or "state_" in low:
        return "raw_state"
    return "other"


def _recent_minus_prior_abs_mean(
    frame: pd.DataFrame,
    numeric: pd.DataFrame,
    columns: Sequence[str],
    *,
    timestamp_col: str,
    weights: np.ndarray,
    lookback_days: float = 7.0,
) -> float:
    if not columns or numeric.empty or timestamp_col not in frame.columns:
        return 0.0
    ts = _timestamp_series(frame, timestamp_col)
    if not ts.notna().any():
        return 0.0
    end = ts.max()
    split = end - pd.Timedelta(days=float(lookback_days))
    recent_mask = (ts >= split).to_numpy(dtype=bool)
    prior_mask = (ts < split).to_numpy(dtype=bool)
    if not bool(recent_mask.any()) or not bool(prior_mask.any()):
        return 0.0
    recent_vals: list[float] = []
    prior_vals: list[float] = []
    w = np.asarray(weights, dtype=np.float64)
    for col in columns:
        if col not in numeric.columns:
            continue
        vals = np.abs(numeric[col].to_numpy(dtype=np.float64))
        recent_vals.append(_weighted_mean(vals[recent_mask], w[recent_mask]))
        prior_vals.append(_weighted_mean(vals[prior_mask], w[prior_mask]))
    if not recent_vals or not prior_vals:
        return 0.0
    return float(np.nanmean(recent_vals) - np.nanmean(prior_vals))


def _recent_minus_prior_abs_mean_array(
    values: np.ndarray,
    columns: Sequence[int],
    timestamp_ns: np.ndarray,
    weights: np.ndarray,
    *,
    lookback_days: float = 7.0,
) -> float:
    idx = list(columns)
    if not idx or values.ndim != 2 or values.shape[0] == 0:
        return 0.0
    ts = np.asarray(timestamp_ns, dtype=np.int64)
    valid_ts = ts > 0
    if not bool(valid_ts.any()):
        return 0.0
    end_ns = int(np.max(ts[valid_ts]))
    split_ns = end_ns - int(float(lookback_days) * 24.0 * 3600.0 * 1e9)
    recent = ts >= split_ns
    prior = (ts > 0) & (ts < split_ns)
    if not bool(recent.any()) or not bool(prior.any()):
        return 0.0
    recent_vals: list[float] = []
    prior_vals: list[float] = []
    for j in idx:
        if j >= values.shape[1]:
            continue
        vals = np.abs(values[:, j].astype(np.float64, copy=False))
        recent_vals.append(_weighted_mean(vals[recent], weights[recent]))
        prior_vals.append(_weighted_mean(vals[prior], weights[prior]))
    if not recent_vals or not prior_vals:
        return 0.0
    return float(np.nanmean(recent_vals) - np.nanmean(prior_vals))


def _drift_fingerprint(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    weights: np.ndarray,
    eps: float,
    timestamp_col: str,
) -> np.ndarray:
    numeric = _safe_numeric_frame(frame, columns)
    if numeric.empty:
        return np.zeros(DRIFT_GLOBAL_METRIC_COUNT + len(DRIFT_FAMILY_ORDER) * DRIFT_FAMILY_METRIC_COUNT, dtype=np.float32)
    ts = _timestamp_series(frame, timestamp_col)
    per_feature_abs = []
    per_feature_signed = []
    per_feature_abs_slope = []
    per_feature_signed_slope = []
    per_feature_missing = []
    cols = list(numeric.columns)
    for col in numeric.columns:
        raw = (
            pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if col in frame.columns
            else pd.Series(np.nan, index=frame.index)
        )
        vals = numeric[col].to_numpy(dtype=np.float64)
        abs_vals = np.abs(vals)
        per_feature_abs.append(_weighted_mean(abs_vals, weights))
        per_feature_signed.append(_weighted_mean(vals, weights))
        per_feature_abs_slope.append(_weighted_slope(abs_vals, ts, weights))
        per_feature_signed_slope.append(_weighted_slope(vals, ts, weights))
        per_feature_missing.append(
            _weighted_fraction(~np.isfinite(raw.to_numpy(dtype=np.float64, copy=False)), weights),
        )
    abs_arr = np.asarray(per_feature_abs, dtype=np.float64)
    signed_arr = np.asarray(per_feature_signed, dtype=np.float64)
    abs_slope_arr = np.asarray(per_feature_abs_slope, dtype=np.float64)
    signed_slope_arr = np.asarray(per_feature_signed_slope, dtype=np.float64)
    missing_arr = np.asarray(per_feature_missing, dtype=np.float64)
    finite_abs = abs_arr[np.isfinite(abs_arr)]
    if finite_abs.size == 0:
        return np.zeros(DRIFT_GLOBAL_METRIC_COUNT + len(DRIFT_FAMILY_ORDER) * DRIFT_FAMILY_METRIC_COUNT, dtype=np.float32)
    total = float(np.sum(np.abs(finite_abs)))
    top10 = _top_mean(finite_abs, 0.10)
    feats: list[float] = [
        float(np.mean(finite_abs)),
        float(np.median(finite_abs)),
        float(np.max(finite_abs)),
        top10,
        _top_mean(finite_abs, 0.25),
        float(top10 / max(total / max(finite_abs.size, 1), eps)),
        float(np.nanmean(signed_arr)) if np.isfinite(signed_arr).any() else 0.0,
        float(abs(np.nanmean(signed_arr)) / max(np.nanmean(finite_abs), eps)),
        float(np.nanmean(abs_slope_arr)) if np.isfinite(abs_slope_arr).any() else 0.0,
        float(np.nanmean(signed_slope_arr)) if np.isfinite(signed_slope_arr).any() else 0.0,
        float(np.nanmean(missing_arr)) if np.isfinite(missing_arr).any() else 0.0,
    ]
    family_by_col = {str(col): _drift_family(str(col)) for col in cols}
    for family in DRIFT_FAMILY_ORDER:
        idx = [i for i, col in enumerate(cols) if family_by_col[str(col)] == family]
        if not idx:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        fam_abs = abs_arr[idx]
        fam_signed = signed_arr[idx]
        fam_abs_slope = abs_slope_arr[idx]
        fam_cols = [cols[i] for i in idx]
        feats.extend(
            [
                float(np.nanmean(fam_abs)) if np.isfinite(fam_abs).any() else 0.0,
                float(np.nanmax(fam_abs)) if np.isfinite(fam_abs).any() else 0.0,
                float(np.nanmean(fam_signed)) if np.isfinite(fam_signed).any() else 0.0,
                float(np.nanmean(fam_abs_slope)) if np.isfinite(fam_abs_slope).any() else 0.0,
                _recent_minus_prior_abs_mean(
                    frame,
                    numeric,
                    fam_cols,
                    timestamp_col=timestamp_col,
                    weights=weights,
                ),
            ]
        )
    return np.asarray(feats, dtype=np.float32)


def _drift_fingerprint_array(
    drift_values: np.ndarray,
    missing: np.ndarray,
    families: Sequence[str],
    timestamp_ns: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.int64)
    if drift_values.ndim != 2 or drift_values.shape[1] == 0 or pos.size == 0:
        return np.zeros(DRIFT_GLOBAL_METRIC_COUNT + len(DRIFT_FAMILY_ORDER) * DRIFT_FAMILY_METRIC_COUNT, dtype=np.float32)
    sub = drift_values[pos].astype(np.float64, copy=False)
    miss = missing[pos] if missing.ndim == 2 and missing.shape[1] == sub.shape[1] else ~np.isfinite(sub)
    ts = np.asarray(timestamp_ns, dtype=np.int64)[pos]
    per_feature_abs = []
    per_feature_signed = []
    per_feature_abs_slope = []
    per_feature_signed_slope = []
    per_feature_missing = []
    for j in range(sub.shape[1]):
        vals = sub[:, j]
        abs_vals = np.abs(vals)
        per_feature_abs.append(_weighted_mean(abs_vals, weights))
        per_feature_signed.append(_weighted_mean(vals, weights))
        per_feature_abs_slope.append(_weighted_slope_ns(abs_vals, ts, weights))
        per_feature_signed_slope.append(_weighted_slope_ns(vals, ts, weights))
        per_feature_missing.append(_weighted_fraction(miss[:, j], weights))
    abs_arr = np.asarray(per_feature_abs, dtype=np.float64)
    signed_arr = np.asarray(per_feature_signed, dtype=np.float64)
    abs_slope_arr = np.asarray(per_feature_abs_slope, dtype=np.float64)
    signed_slope_arr = np.asarray(per_feature_signed_slope, dtype=np.float64)
    missing_arr = np.asarray(per_feature_missing, dtype=np.float64)
    finite_abs = abs_arr[np.isfinite(abs_arr)]
    if finite_abs.size == 0:
        return np.zeros(DRIFT_GLOBAL_METRIC_COUNT + len(DRIFT_FAMILY_ORDER) * DRIFT_FAMILY_METRIC_COUNT, dtype=np.float32)
    total = float(np.sum(np.abs(finite_abs)))
    top10 = _top_mean(finite_abs, 0.10)
    feats: list[float] = [
        float(np.mean(finite_abs)),
        float(np.median(finite_abs)),
        float(np.max(finite_abs)),
        top10,
        _top_mean(finite_abs, 0.25),
        float(top10 / max(total / max(finite_abs.size, 1), eps)),
        float(np.nanmean(signed_arr)) if np.isfinite(signed_arr).any() else 0.0,
        float(abs(np.nanmean(signed_arr)) / max(np.nanmean(finite_abs), eps)),
        float(np.nanmean(abs_slope_arr)) if np.isfinite(abs_slope_arr).any() else 0.0,
        float(np.nanmean(signed_slope_arr)) if np.isfinite(signed_slope_arr).any() else 0.0,
        float(np.nanmean(missing_arr)) if np.isfinite(missing_arr).any() else 0.0,
    ]
    family_to_idx: dict[str, list[int]] = {family: [] for family in DRIFT_FAMILY_ORDER}
    for i, family in enumerate(families):
        family_to_idx.setdefault(str(family), []).append(i)
    for family in DRIFT_FAMILY_ORDER:
        idx = family_to_idx.get(family, [])
        if not idx:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        fam_abs = abs_arr[idx]
        fam_signed = signed_arr[idx]
        fam_abs_slope = abs_slope_arr[idx]
        feats.extend(
            [
                float(np.nanmean(fam_abs)) if np.isfinite(fam_abs).any() else 0.0,
                float(np.nanmax(fam_abs)) if np.isfinite(fam_abs).any() else 0.0,
                float(np.nanmean(fam_signed)) if np.isfinite(fam_signed).any() else 0.0,
                float(np.nanmean(fam_abs_slope)) if np.isfinite(fam_abs_slope).any() else 0.0,
                _recent_minus_prior_abs_mean_array(
                    sub,
                    idx,
                    ts,
                    weights,
                ),
            ]
        )
    return np.asarray(feats, dtype=np.float32)


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    n = min(aa.size, bb.size)
    if n == 0:
        return 0.0
    diff = aa[:n] - bb[:n]
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(diff * diff)))


def _normalise_distances(values: np.ndarray, eps: float) -> np.ndarray:
    norm, _scale = _normalise_distances_with_scale(values, eps)
    return norm


def _normalise_distances_with_scale(values: np.ndarray, eps: float) -> tuple[np.ndarray, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not bool(finite.any()):
        return np.ones_like(arr, dtype=np.float64), 1.0
    med = float(np.nanmedian(arr[finite]))
    if not np.isfinite(med) or med <= eps:
        med = float(np.nanmean(arr[finite]))
    if not np.isfinite(med) or med <= eps:
        med = 1.0
    return (
        np.nan_to_num(arr / max(med, eps), nan=1.0, posinf=10.0, neginf=10.0),
        float(med),
    )


def _component_scales(matrix: np.ndarray, current: np.ndarray, eps: float) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64).reshape(1, -1)
    if mat.ndim != 2 or mat.shape[1] == 0:
        return np.ones(0, dtype=np.float64)
    n = min(mat.shape[1], cur.shape[1])
    stack = np.vstack([mat[:, :n], cur[:, :n]])
    scales = np.ones(n, dtype=np.float64)
    for j in range(n):
        vals = stack[:, j]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        q25, q75 = np.nanpercentile(vals, [25.0, 75.0])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= eps:
            med = float(np.nanmedian(vals))
            scale = float(np.nanmedian(np.abs(vals - med)))
        if not np.isfinite(scale) or scale <= eps:
            scale = float(np.nanmedian(np.abs(vals)))
        scales[j] = scale if np.isfinite(scale) and scale > eps else 1.0
    return scales


def _scaled_euclidean_by_indices(
    matrix: np.ndarray,
    current: np.ndarray,
    indices: Sequence[int],
    eps: float,
) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] == 0 or cur.size == 0:
        return np.zeros(mat.shape[0] if mat.ndim == 2 else 0, dtype=np.float64)
    idx = np.asarray([int(i) for i in indices if 0 <= int(i) < mat.shape[1] and int(i) < cur.size], dtype=np.int64)
    if idx.size == 0:
        return np.zeros(mat.shape[0], dtype=np.float64)
    block = mat[:, idx]
    cur_block = cur[idx]
    scales = _component_scales(block, cur_block, eps)
    diff = (block - cur_block.reshape(1, -1)) / np.maximum(scales.reshape(1, -1), eps)
    diff = np.where(np.isfinite(diff), diff, 0.0)
    return np.sqrt(np.mean(diff * diff, axis=1))


def _normalize_weight_map(weights: Mapping[str, float], eps: float) -> dict[str, float]:
    clean = {
        str(key): max(float(value), 0.0)
        for key, value in weights.items()
        if np.isfinite(float(value)) and float(value) > 0.0
    }
    total = float(sum(clean.values()))
    if total <= eps:
        n = max(len(weights), 1)
        return {str(key): 1.0 / n for key in weights}
    return {key: value / total for key, value in clean.items()}


def _combine_block_distances(
    matrix: np.ndarray,
    current: np.ndarray,
    blocks: Mapping[str, Sequence[int]],
    weights: Mapping[str, float],
    eps: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        mat = np.zeros((0, 0), dtype=np.float64)
    norm_weights = _normalize_weight_map(weights, eps)
    block_distances: dict[str, np.ndarray] = {}
    total = np.zeros(mat.shape[0], dtype=np.float64)
    diagnostics: dict[str, Any] = {"weights": norm_weights, "blocks": {}}
    for name, idx in blocks.items():
        dist = _scaled_euclidean_by_indices(mat, current, idx, eps)
        block_distances[str(name)] = dist
        total += float(norm_weights.get(str(name), 0.0)) * dist
        finite = dist[np.isfinite(dist)]
        diagnostics["blocks"][str(name)] = {
            "feature_count": int(len([i for i in idx if 0 <= int(i) < mat.shape[1]])),
            "median_distance": float(np.nanmedian(finite)) if finite.size else 0.0,
        }
    return total.astype(np.float64), block_distances, diagnostics


def _covariance_block_indices(top_eigenvalues: int) -> dict[str, np.ndarray]:
    k = int(top_eigenvalues)
    feature_offset = 0
    asset_offset = k * 2 + 5
    return {
        "feature_cov_eig": np.arange(feature_offset, feature_offset + k),
        "feature_corr_eig": np.arange(feature_offset + k, feature_offset + 2 * k),
        "feature_concentration": np.arange(feature_offset + 2 * k, feature_offset + 2 * k + 5),
        "asset_cov_eig": np.arange(asset_offset, asset_offset + k),
        "asset_corr_eig": np.arange(asset_offset + k, asset_offset + 2 * k),
        "asset_concentration": np.arange(asset_offset + 2 * k, asset_offset + 2 * k + 5),
    }


def _covariance_block_distances(
    matrix: np.ndarray,
    current: np.ndarray,
    *,
    top_eigenvalues: int,
    config: RegimeSimilarityConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    weights = {
        "feature_cov_eig": config.cov_feature_cov_eig_weight,
        "feature_corr_eig": config.cov_feature_corr_eig_weight,
        "feature_concentration": config.cov_feature_concentration_weight,
        "asset_cov_eig": config.cov_asset_cov_eig_weight,
        "asset_corr_eig": config.cov_asset_corr_eig_weight,
        "asset_concentration": config.cov_asset_concentration_weight,
    }
    return _combine_block_distances(
        matrix,
        current,
        _covariance_block_indices(top_eigenvalues),
        weights,
        float(config.eps),
    )


def _drift_block_indices() -> dict[str, np.ndarray]:
    family_slices: dict[str, np.ndarray] = {}
    for i, family in enumerate(DRIFT_FAMILY_ORDER):
        start = DRIFT_GLOBAL_METRIC_COUNT + i * DRIFT_FAMILY_METRIC_COUNT
        family_slices[family] = np.arange(start, start + DRIFT_FAMILY_METRIC_COUNT)
    other_parts = [
        np.arange(0, DRIFT_GLOBAL_METRIC_COUNT),
        family_slices.get("base_model", np.zeros(0, dtype=np.int64)),
        family_slices.get("meta_model", np.zeros(0, dtype=np.int64)),
        family_slices.get("row_drift", np.zeros(0, dtype=np.int64)),
        family_slices.get("raw_state", np.zeros(0, dtype=np.int64)),
        family_slices.get("other", np.zeros(0, dtype=np.int64)),
    ]
    return {
        "psi": family_slices.get("psi", np.zeros(0, dtype=np.int64)),
        "ks": family_slices.get("ks", np.zeros(0, dtype=np.int64)),
        "wasserstein": family_slices.get("wasserstein", np.zeros(0, dtype=np.int64)),
        "mahalanobis": family_slices.get("mahalanobis", np.zeros(0, dtype=np.int64)),
        "prediction": family_slices.get("prediction_distribution", np.zeros(0, dtype=np.int64)),
        "covariance": family_slices.get("covariance", np.zeros(0, dtype=np.int64)),
        "contribution": family_slices.get("contribution", np.zeros(0, dtype=np.int64)),
        "other": np.concatenate(other_parts).astype(np.int64),
    }


def _drift_block_distances(
    matrix: np.ndarray,
    current: np.ndarray,
    *,
    config: RegimeSimilarityConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    weights = {
        "psi": config.drift_psi_weight,
        "ks": config.drift_ks_weight,
        "wasserstein": config.drift_wasserstein_weight,
        "mahalanobis": config.drift_mahalanobis_weight,
        "prediction": config.drift_prediction_weight,
        "covariance": config.drift_covariance_weight,
        "contribution": config.drift_contribution_weight,
        "other": config.drift_other_weight,
    }
    return _combine_block_distances(
        matrix,
        current,
        _drift_block_indices(),
        weights,
        float(config.eps),
    )


def _historical_iqr_scale(values: np.ndarray, historical_mask: np.ndarray, eps: float) -> float:
    vals = np.asarray(values, dtype=np.float64)
    hist = np.asarray(historical_mask, dtype=bool)
    if hist.size != vals.size:
        hist = np.ones(vals.size, dtype=bool)
    finite = vals[hist & np.isfinite(vals)]
    if finite.size == 0:
        finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return 1.0
    q25, q75 = np.nanpercentile(finite, [25.0, 75.0])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= eps:
        scale = float(np.nanmedian(np.abs(finite)))
    return scale if np.isfinite(scale) and scale > eps else 1.0


def _baseline_covariance_norm(matrix: np.ndarray, historical_mask: np.ndarray, eps: float) -> float:
    arr = np.asarray(matrix, dtype=np.float64)
    hist = np.asarray(historical_mask, dtype=bool)
    if arr.ndim != 2 or arr.shape[1] < 2 or hist.size != arr.shape[0] or not bool(hist.any()):
        return 1.0
    sub = arr[hist]
    if sub.shape[0] < 3:
        return 1.0
    weights = np.ones(sub.shape[0], dtype=np.float64)
    cov = _weighted_covariance(sub, weights, eps)
    if cov.size == 0:
        return 1.0
    norm = float(np.linalg.norm(cov.astype(np.float64), ord="fro"))
    return norm if np.isfinite(norm) and norm > eps else 1.0


def _normalize_drift_values_by_family(
    drift_values: np.ndarray,
    columns: Sequence[str],
    historical_mask: np.ndarray,
    *,
    baseline_covariance_norm: float,
    config: RegimeSimilarityConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(drift_values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return arr.astype(np.float32, copy=False), {"enabled": False, "reason": "empty_matrix"}
    out = arr.copy()
    eps = float(config.eps)
    psi_scale = max(float(config.drift_psi_scale), eps)
    feature_dim = max(int(arr.shape[1]), 1)
    counts: dict[str, int] = {}
    scale_values: dict[str, list[float]] = {}
    for j, col in enumerate(columns):
        family = _drift_family(str(col))
        counts[family] = counts.get(family, 0) + 1
        vals = arr[:, j]
        if family == "psi":
            out[:, j] = np.log1p(np.maximum(vals, 0.0)) / psi_scale
            scale = psi_scale
        elif family == "ks":
            out[:, j] = vals
            scale = 1.0
        elif family == "mahalanobis":
            out[:, j] = np.sqrt(np.maximum(vals, 0.0)) / math.sqrt(float(feature_dim))
            scale = math.sqrt(float(feature_dim))
        elif family == "covariance":
            iqr_scale = _historical_iqr_scale(vals, historical_mask, eps)
            scale = max(float(baseline_covariance_norm), iqr_scale, eps)
            out[:, j] = vals / max(scale, eps)
        else:
            scale = _historical_iqr_scale(vals, historical_mask, eps)
            out[:, j] = vals / max(scale, eps)
        scale_values.setdefault(family, []).append(float(scale))
    scale_summary = {
        family: float(np.nanmedian(values)) if values else 1.0
        for family, values in scale_values.items()
    }
    return out.astype(np.float32, copy=False), {
        "enabled": True,
        "counts": counts,
        "scale_median_by_family": scale_summary,
        "psi_scale": float(psi_scale),
        "mahalanobis_feature_dim": int(feature_dim),
        "baseline_covariance_norm": float(baseline_covariance_norm),
    }


def _window_ids(ts: pd.Series, *, anchor: pd.Timestamp, window_days: float) -> np.ndarray:
    seconds = (ts - anchor).dt.total_seconds().to_numpy(dtype=np.float64)
    seconds = np.nan_to_num(seconds, nan=-1.0, neginf=-1.0, posinf=-1.0)
    window_seconds = max(float(window_days) * 24.0 * 3600.0, 1.0)
    return np.floor(np.maximum(seconds, 0.0) / window_seconds).astype(np.int64)


def _subsample_positions(
    positions: np.ndarray,
    *,
    max_rows: int,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    if max_rows <= 0 or len(positions) <= max_rows:
        return positions
    idx = np.linspace(0, len(positions) - 1, int(max_rows)).round().astype(int)
    return positions[np.unique(idx)]


def _subsample_positions_by_window(
    positions: np.ndarray,
    window_ids: np.ndarray,
    *,
    max_rows_per_window: int,
) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.int64)
    if max_rows_per_window <= 0 or pos.size == 0:
        return pos
    ids = np.asarray(window_ids, dtype=np.int64)[pos]
    order = np.argsort(ids, kind="mergesort")
    sorted_pos = pos[order]
    sorted_ids = ids[order]
    splits = np.flatnonzero(np.diff(sorted_ids)) + 1
    groups = np.split(sorted_pos, splits)
    capped = [
        _subsample_positions(group, max_rows=int(max_rows_per_window))
        for group in groups
        if len(group)
    ]
    if not capped:
        return np.zeros(0, dtype=np.int64)
    return np.sort(np.concatenate(capped).astype(np.int64))


def _knn_mean_scaled_distances(
    block: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    if block.size == 0 or reference.size == 0:
        return np.zeros((len(block), len(reference)), dtype=np.float32)
    block64 = np.asarray(block, dtype=np.float64)
    ref64 = np.asarray(reference, dtype=np.float64)
    dim = max(int(block64.shape[1]), 1)
    block_sq = np.sum(block64 * block64, axis=1, keepdims=True)
    ref_sq = np.sum(ref64 * ref64, axis=1, keepdims=True).T
    dist_sq = np.maximum((block_sq + ref_sq - 2.0 * (block64 @ ref64.T)) / float(dim), 0.0)
    return np.sqrt(dist_sq).astype(np.float32, copy=False)


def _knn_fallback_block_size(reference_rows: int, *, chunk_pairs: int, default: int = 256) -> int:
    if int(reference_rows) <= 0:
        return 1
    if int(chunk_pairs) <= 0:
        return max(1, int(default))
    return max(1, min(int(default), int(chunk_pairs) // max(int(reference_rows), 1)))


def _knn_window_distance(
    scaled_knn: np.ndarray,
    current_pos: np.ndarray,
    candidate_pos: np.ndarray,
    current_weights: np.ndarray,
    *,
    k: int,
    max_current_rows: int,
    max_candidate_rows: int,
    eps: float,
    fallback_chunk_pairs: int = 2_000_000,
) -> float:
    if scaled_knn.size == 0 or len(current_pos) == 0 or len(candidate_pos) == 0:
        return 1.0
    current_pos = np.asarray(current_pos, dtype=np.int64)
    if max_current_rows > 0 and len(current_pos) > max_current_rows:
        current_local_idx = np.linspace(
            0,
            len(current_pos) - 1,
            int(max_current_rows),
        ).round().astype(int)
        current_local_idx = np.unique(current_local_idx)
        cur_pos = current_pos[current_local_idx]
    else:
        current_local_idx = np.arange(len(current_pos), dtype=np.int64)
        cur_pos = current_pos
    cand_pos = _subsample_positions(candidate_pos, max_rows=max_candidate_rows)
    cur = scaled_knn[cur_pos]
    cand = scaled_knn[cand_pos]
    if cur.ndim != 2 or cand.ndim != 2 or cur.shape[1] == 0 or cand.shape[1] == 0:
        return 1.0
    kk = max(1, min(int(k), len(cand)))
    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=kk, metric="euclidean")
        nn.fit(cand)
        distances, _idx = nn.kneighbors(cur, return_distance=True)
        row_dist = np.mean(distances, axis=1)
    except Exception:
        row_dist_chunks: list[np.ndarray] = []
        block_size = _knn_fallback_block_size(len(cand), chunk_pairs=int(fallback_chunk_pairs))
        for start in range(0, len(cur), block_size):
            block = cur[start : start + block_size]
            d = _knn_mean_scaled_distances(block, cand)
            d.sort(axis=1)
            row_dist_chunks.append(d[:, :kk].mean(axis=1))
        row_dist = np.concatenate(row_dist_chunks) if row_dist_chunks else np.asarray([1.0])
    w_all = np.asarray(current_weights, dtype=np.float64)
    w = (
        w_all[current_local_idx]
        if len(w_all) >= int(np.max(current_local_idx, initial=-1)) + 1
        else w_all[: len(cur_pos)]
    )
    if len(w) != len(row_dist) or not np.isfinite(w).any() or float(np.sum(w)) <= eps:
        return float(np.nanmean(row_dist))
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.average(row_dist, weights=np.maximum(w, eps)))


def _knn_window_distances_global(
    scaled_knn: np.ndarray,
    current_pos: np.ndarray,
    historical_pos: np.ndarray,
    window_id_full: np.ndarray,
    current_weights: np.ndarray,
    *,
    k: int,
    max_current_rows: int,
    max_candidate_rows: int,
    max_historical_rows: int,
    fallback_chunk_pairs: int,
    eps: float,
) -> tuple[dict[int, float], float, dict[str, Any]]:
    if (
        scaled_knn.size == 0
        or len(current_pos) == 0
        or len(historical_pos) == 0
        or scaled_knn.ndim != 2
        or scaled_knn.shape[1] == 0
    ):
        return {}, 1.0, {"mode": "global_knn", "enabled": False, "reason": "empty_matrix"}
    current_pos = np.asarray(current_pos, dtype=np.int64)
    historical_pos = np.asarray(historical_pos, dtype=np.int64)
    if max_current_rows > 0 and len(current_pos) > max_current_rows:
        cur_local = np.linspace(0, len(current_pos) - 1, int(max_current_rows)).round().astype(int)
        cur_local = np.unique(cur_local)
        cur_pos = current_pos[cur_local]
    else:
        cur_local = np.arange(len(current_pos), dtype=np.int64)
        cur_pos = current_pos
    hist_pos = _subsample_positions_by_window(
        historical_pos,
        window_id_full,
        max_rows_per_window=int(max_candidate_rows),
    )
    hist_pos = _subsample_positions(hist_pos, max_rows=int(max_historical_rows))
    cur = scaled_knn[cur_pos]
    hist = scaled_knn[hist_pos]
    if cur.ndim != 2 or hist.ndim != 2 or len(hist) == 0:
        return {}, 1.0, {"mode": "global_knn", "enabled": False, "reason": "empty_subsample"}
    neighbor_count = max(int(k), min(len(hist), max(int(k) * 8, 64)))
    neighbor_count = max(1, min(neighbor_count, len(hist)))
    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
        nn.fit(hist)
        distances, indices = nn.kneighbors(cur, return_distance=True)
    except Exception:
        row_dist_chunks: list[np.ndarray] = []
        row_idx_chunks: list[np.ndarray] = []
        block_size = _knn_fallback_block_size(len(hist), chunk_pairs=int(fallback_chunk_pairs))
        for start in range(0, len(cur), block_size):
            block = cur[start : start + block_size]
            d = _knn_mean_scaled_distances(block, hist)
            idx = np.argsort(d, axis=1)[:, :neighbor_count]
            row_idx_chunks.append(idx)
            row_dist_chunks.append(np.take_along_axis(d, idx, axis=1))
        if not row_dist_chunks:
            return {}, 1.0, {"mode": "global_knn", "enabled": False, "reason": "distance_failed"}
        distances = np.vstack(row_dist_chunks)
        indices = np.vstack(row_idx_chunks)
    w_all = np.asarray(current_weights, dtype=np.float64)
    row_weights = (
        w_all[cur_local]
        if len(w_all) >= int(np.max(cur_local, initial=-1)) + 1
        else w_all[: len(cur_pos)]
    )
    if len(row_weights) != len(distances) or float(np.sum(np.maximum(row_weights, 0.0))) <= eps:
        row_weights = np.ones(len(distances), dtype=np.float64) / max(len(distances), 1)
    hist_window_ids = window_id_full[hist_pos]
    neighbor_windows = hist_window_ids[indices]
    valid = neighbor_windows >= 0
    weighted_sum: dict[int, float] = {}
    weight_count: dict[int, float] = {}
    for row_i in range(distances.shape[0]):
        row_valid = valid[row_i]
        if not bool(row_valid.any()):
            continue
        wins = neighbor_windows[row_i, row_valid]
        dists = distances[row_i, row_valid]
        row_w = float(max(row_weights[row_i], eps))
        for win in np.unique(wins):
            mask = wins == int(win)
            mean_dist = float(np.nanmean(dists[mask])) if bool(mask.any()) else 1.0
            weighted_sum[int(win)] = weighted_sum.get(int(win), 0.0) + row_w * mean_dist
            weight_count[int(win)] = weight_count.get(int(win), 0.0) + row_w
    distances_by_window = {
        win: weighted_sum[win] / max(weight_count.get(win, 0.0), eps)
        for win in weighted_sum
    }
    finite = np.asarray(list(distances_by_window.values()), dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    fallback = float(np.nanmax(finite) * 1.5) if finite.size else 1.0
    fallback = max(fallback, 1.0)
    return distances_by_window, fallback, {
        "mode": "global_knn",
        "enabled": True,
        "current_rows_used": int(len(cur_pos)),
        "historical_rows_used": int(len(hist_pos)),
        "max_candidate_rows_per_window": int(max_candidate_rows),
        "fallback_chunk_pairs": int(fallback_chunk_pairs),
        "fallback_block_size": int(_knn_fallback_block_size(len(hist), chunk_pairs=int(fallback_chunk_pairs))),
        "neighbors_per_current_row": int(neighbor_count),
        "windows_with_neighbor_support": int(len(distances_by_window)),
    }


def _scale_knn_matrix(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    scaler: Mapping[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    numeric = (
        _scaled_numeric_frame(frame, columns, scaler)
        if scaler is not None
        else _safe_numeric_frame(frame, columns)
    )
    if numeric.empty:
        return np.zeros((len(frame), 0), dtype=np.float32)
    arr = numeric.to_numpy(dtype=np.float32, copy=True)
    if scaler is None:
        for j in range(arr.shape[1]):
            arr[:, j], _center, _scale = _robust_scale(arr[:, j])
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _activation(x: np.ndarray, name: str) -> np.ndarray:
    if name == "tanh":
        return np.tanh(x)
    if name == "logistic":
        return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))
    if name == "identity":
        return x
    return np.maximum(x, 0.0)


def _mlp_forward(
    coefs: Sequence[np.ndarray],
    intercepts: Sequence[np.ndarray],
    x: np.ndarray,
    *,
    activation: str,
    bottleneck_layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(x, dtype=np.float32)
    latent = None
    for i, (w, b) in enumerate(zip(coefs, intercepts)):
        h = h @ np.asarray(w, dtype=np.float32) + np.asarray(b, dtype=np.float32).reshape(1, -1)
        if i < len(coefs) - 1:
            h = _activation(h, activation).astype(np.float32, copy=False)
            if i == bottleneck_layer:
                latent = h.copy()
    if latent is None:
        latent = np.zeros((x.shape[0], 0), dtype=np.float32)
    return latent.astype(np.float32), h.astype(np.float32)


def _ae_similarity(
    window_matrix: np.ndarray,
    current_vector: np.ndarray,
    *,
    config: RegimeSimilarityConfig,
) -> np.ndarray:
    n = int(window_matrix.shape[0])
    if (
        not bool(config.ae_enabled)
        or n < 4
        or window_matrix.ndim != 2
        or window_matrix.shape[1] < 2
    ):
        return np.zeros(n, dtype=np.float32)
    all_x = np.vstack([window_matrix, current_vector.reshape(1, -1)]).astype(np.float32)
    max_windows = int(config.ae_max_windows)
    if max_windows > 0 and n > max_windows:
        fit_window_idx = np.linspace(0, n - 1, max_windows).round().astype(np.int64)
        fit_window_idx = np.unique(fit_window_idx)
        fit_idx = np.concatenate([fit_window_idx, np.asarray([n], dtype=np.int64)])
    else:
        fit_idx = np.arange(n + 1, dtype=np.int64)
    fit_idx = np.unique(fit_idx)
    fit_x = all_x[fit_idx]
    centers = np.zeros(all_x.shape[1], dtype=np.float32)
    scales = np.ones(all_x.shape[1], dtype=np.float32)
    for j in range(all_x.shape[1]):
        _fit_z, center, scale = _robust_scale(fit_x[:, j])
        centers[j] = float(center)
        scales[j] = max(float(scale), 1e-9)
    all_x = np.clip(
        (np.where(np.isfinite(all_x), all_x, centers.reshape(1, -1)) - centers.reshape(1, -1))
        / np.maximum(scales.reshape(1, -1), 1e-9),
        -8.0,
        8.0,
    ).astype(np.float32)
    train_x = all_x[fit_idx]
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPRegressor

        rng = np.random.default_rng(int(config.random_state))
        noisy = np.clip(
            train_x + rng.normal(0.0, float(config.ae_input_noise), size=train_x.shape).astype(np.float32),
            -8.0,
            8.0,
        )
        model = MLPRegressor(
            hidden_layer_sizes=(64, 16, int(config.ae_latent_dim), 16, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=max(1, int(config.ae_max_iter)),
            batch_size=min(1024, max(1, len(train_x))),
            random_state=int(config.random_state),
            early_stopping=False,
            verbose=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(noisy, train_x)
        coefs = [np.asarray(w, dtype=np.float32) for w in model.coefs_]
        intercepts = [np.asarray(b, dtype=np.float32) for b in model.intercepts_]
        latent, recon = _mlp_forward(
            coefs,
            intercepts,
            all_x,
            activation=str(getattr(model, "activation", "relu")),
            bottleneck_layer=2,
        )
    except Exception:
        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=min(int(config.ae_latent_dim), train_x.shape[0], train_x.shape[1]))
            pca.fit(train_x)
            latent = pca.transform(all_x).astype(np.float32)
            recon = pca.inverse_transform(latent).astype(np.float32)
        except Exception:
            return np.zeros(n, dtype=np.float32)
    err = np.mean(np.abs(recon - all_x), axis=1)
    cur_err = float(err[-1])
    err_dist = np.abs(err[:n] - cur_err)
    err_scale = float(np.nanmedian(err_dist[np.isfinite(err_dist)])) if np.isfinite(err_dist).any() else 1.0
    err_scale = max(err_scale, 1e-6)
    sim_recon = np.exp(-err_dist / err_scale)
    cur_latent = latent[-1]
    latent_dist = np.sqrt(np.mean((latent[:n] - cur_latent.reshape(1, -1)) ** 2, axis=1))
    latent_scale = float(np.nanmedian(latent_dist[np.isfinite(latent_dist)])) if np.isfinite(latent_dist).any() else 1.0
    latent_scale = max(latent_scale, 1e-6)
    sim_latent = np.exp(-latent_dist / latent_scale)
    return np.clip(0.5 * sim_recon + 0.5 * sim_latent, 0.0, 1.0).astype(np.float32)


def _concat_fingerprint(
    market: np.ndarray,
    covariance: np.ndarray,
    drift: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(market, dtype=np.float32).ravel(),
            np.asarray(covariance, dtype=np.float32).ravel(),
            np.asarray(drift, dtype=np.float32).ravel(),
        ]
    ).astype(np.float32)


def _finalize_similarity_output(
    out: pd.DataFrame,
    *,
    original_index: pd.Index,
    future_index: pd.Index,
    diagnostics: Dict[str, Any],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if not out.index.equals(original_index):
        out = out.reindex(original_index)
    for col in (
        "window_similarity",
        "day_similarity",
        "day_similarity_available",
        "similarity_to_current",
        "current_regime_recency_weight",
    ):
        if col not in out.columns:
            out[col] = np.nan
    if "regime_specialist_bucket" not in out.columns:
        out["regime_specialist_bucket"] = "irrelevant"
    else:
        out["regime_specialist_bucket"] = (
            out["regime_specialist_bucket"]
            .astype("object")
            .where(out["regime_specialist_bucket"].notna(), "irrelevant")
        )
    if len(future_index) > 0:
        out.loc[future_index, "window_similarity"] = 0.0
        out.loc[future_index, "day_similarity"] = 0.0
        out.loc[future_index, "day_similarity_available"] = False
        out.loc[future_index, "similarity_to_current"] = 0.0
        out.loc[future_index, "current_regime_recency_weight"] = 0.0
        out.loc[future_index, "regime_specialist_bucket"] = "future_excluded"
    out["window_similarity"] = pd.to_numeric(
        out["window_similarity"],
        errors="coerce",
    ).fillna(0.0).astype(np.float32)
    out["day_similarity"] = pd.to_numeric(
        out["day_similarity"],
        errors="coerce",
    ).fillna(0.0).astype(np.float32)
    out["day_similarity_available"] = (
        out["day_similarity_available"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    )
    out["similarity_to_current"] = pd.to_numeric(
        out["similarity_to_current"],
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0).astype(np.float32)
    out["current_regime_recency_weight"] = pd.to_numeric(
        out["current_regime_recency_weight"],
        errors="coerce",
    ).fillna(0.0).astype(np.float32)
    diagnostics = dict(diagnostics)
    diagnostics["future_excluded_rows"] = int(len(future_index))
    diagnostics["asof_rows"] = int(len(original_index) - len(future_index))
    return out, diagnostics


_SIMILARITY_RESULT_COLUMNS: tuple[str, ...] = (
    "window_similarity",
    "day_similarity",
    "day_similarity_available",
    "similarity_to_current",
    "current_regime_recency_weight",
)


def _local_label_overlap_mask(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    end: pd.Timestamp,
    config: RegimeSimilarityConfig,
) -> tuple[pd.Series, str]:
    ts = _timestamp_series(frame, timestamp_col)
    valid_ts = ts.notna()
    asof_mask = valid_ts & (ts <= end)
    label_end = None
    source = "none"
    if config.label_end_col is not None and str(config.label_end_col) in frame.columns:
        label_end = pd.to_datetime(frame[str(config.label_end_col)], utc=True, errors="coerce")
        source = str(config.label_end_col)
    elif float(config.label_horizon_hours or 0.0) > 0.0:
        label_end = ts + pd.Timedelta(hours=float(config.label_horizon_hours))
        source = "label_horizon_hours"
    if label_end is None:
        return pd.Series(False, index=frame.index), source
    label_complete = label_end.notna() & (label_end <= end)
    return asof_mask & ~label_complete, source


def _derive_local_buckets_from_similarity(
    frame: pd.DataFrame,
    similarity: np.ndarray,
    *,
    timestamp_col: str,
    end: pd.Timestamp,
    config: RegimeSimilarityConfig,
    label_overlap_mask: pd.Series | None = None,
) -> np.ndarray:
    ts = _timestamp_series(frame, timestamp_col)
    valid_ts = ts.notna()
    future = (valid_ts & (ts > end)).to_numpy(dtype=bool)
    current_start = end - pd.Timedelta(days=float(config.current_window_days))
    current = (valid_ts & (ts >= current_start) & (ts <= end)).to_numpy(dtype=bool)
    sim = np.nan_to_num(np.asarray(similarity, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    bucket = np.where(
        sim >= float(config.analogue_threshold),
        "analogue",
        np.where(sim >= float(config.normal_threshold), "normal", "irrelevant"),
    ).astype(object)
    bucket[current] = "current"
    if label_overlap_mask is not None:
        overlap = label_overlap_mask.reindex(frame.index).fillna(False).to_numpy(dtype=bool)
        bucket[overlap] = "irrelevant"
    bucket[future] = "future_excluded"
    return bucket


def _aggregate_similarity_by_keys(
    assessment_similarity: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    value_cols = [
        col
        for col in _SIMILARITY_RESULT_COLUMNS
        if col in assessment_similarity.columns
    ]
    if not value_cols or not all(key in assessment_similarity.columns for key in keys):
        return pd.DataFrame()
    work = assessment_similarity[keys + value_cols].copy(deep=False)
    if "day_similarity_available" in work.columns:
        work["day_similarity_available"] = work["day_similarity_available"].astype(float)
    grouped = work.groupby(keys, sort=False, dropna=False)[value_cols].mean(numeric_only=True)
    return grouped.reset_index()


def _index_alignment_matches_keys(
    frame: pd.DataFrame,
    aligned: pd.DataFrame,
    *,
    timestamp_col: str,
    symbol_col: str,
) -> bool:
    checked = False
    if timestamp_col in frame.columns and timestamp_col in aligned.columns:
        left_ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
        right_ts = pd.to_datetime(aligned[timestamp_col], utc=True, errors="coerce")
        valid = left_ts.notna() & right_ts.notna()
        if not bool(valid.any()):
            return False
        checked = True
        if not bool((left_ts[valid].to_numpy() == right_ts[valid].to_numpy()).all()):
            return False
    if symbol_col in frame.columns and symbol_col in aligned.columns:
        left_symbol = frame[symbol_col].astype("string")
        right_symbol = aligned[symbol_col].astype("string")
        valid = left_symbol.notna() & right_symbol.notna()
        if not bool(valid.any()):
            return False
        checked = True
        if not bool((left_symbol[valid].to_numpy() == right_symbol[valid].to_numpy()).all()):
            return False
    return bool(checked)


def _align_global_assessment_similarity(
    frame: pd.DataFrame,
    assessment_similarity: pd.DataFrame,
    *,
    timestamp_col: str,
    symbol_col: str,
    current_end: pd.Timestamp,
    config: RegimeSimilarityConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    original_index = frame.index
    out = pd.DataFrame(index=original_index)
    value_cols = [
        col
        for col in _SIMILARITY_RESULT_COLUMNS
        if col in assessment_similarity.columns
    ]
    alignment = "none"
    left = frame.copy(deep=False)
    left["_local_index"] = np.arange(len(frame), dtype=np.int64)
    if timestamp_col in left.columns:
        left["_align_timestamp"] = pd.to_datetime(left[timestamp_col], utc=True, errors="coerce")
    if symbol_col in left.columns:
        left["_align_symbol"] = left[symbol_col].astype("string")
    right_base = assessment_similarity.copy(deep=False)
    if timestamp_col in right_base.columns:
        right_base = right_base.copy(deep=False)
        right_base["_align_timestamp"] = pd.to_datetime(right_base[timestamp_col], utc=True, errors="coerce")
    if symbol_col in right_base.columns:
        right_base = right_base.copy(deep=False)
        right_base["_align_symbol"] = right_base[symbol_col].astype("string")
    key_sets: list[tuple[str, list[str]]] = []
    if "_align_timestamp" in left.columns and "_align_timestamp" in right_base.columns:
        if "_align_symbol" in left.columns and "_align_symbol" in right_base.columns:
            key_sets.append(("timestamp_symbol", ["_align_timestamp", "_align_symbol"]))
        if bool(config.assessment_allow_timestamp_only_alignment):
            key_sets.append(("timestamp", ["_align_timestamp"]))
    for name, keys in key_sets:
        if not value_cols:
            break
        if not all(key in left.columns for key in keys) or not all(key in right_base.columns for key in keys):
            continue
        aggregated = _aggregate_similarity_by_keys(right_base, keys)
        if aggregated.empty:
            continue
        merged = left[["_local_index"] + keys].merge(
            aggregated,
            on=keys,
            how="left",
            sort=False,
        )
        merged = merged.sort_values("_local_index")
        if int(merged[value_cols].notna().any(axis=1).sum()) > 0:
            for col in value_cols:
                out[col] = merged[col].to_numpy()
            alignment = name
            break
    if (
        alignment == "none"
        and value_cols
        and assessment_similarity.index.is_unique
        and original_index.isin(assessment_similarity.index).all()
    ):
        aligned = assessment_similarity.reindex(original_index)
        if _index_alignment_matches_keys(
            frame,
            aligned,
            timestamp_col=timestamp_col,
            symbol_col=symbol_col,
        ):
            for col in value_cols:
                out[col] = aligned[col]
            alignment = "index"
        elif not any(col in assessment_similarity.columns for col in (timestamp_col, symbol_col)):
            for col in value_cols:
                out[col] = aligned[col]
            alignment = "index_unverified"
    if (
        alignment == "none"
        and value_cols
        and bool(config.assessment_allow_timestamp_only_alignment)
        and "_align_timestamp" in left.columns
        and "_align_timestamp" in right_base.columns
    ):
        aggregated = _aggregate_similarity_by_keys(right_base, ["_align_timestamp"])
        if not aggregated.empty:
            merged = left[["_local_index", "_align_timestamp"]].merge(
                aggregated,
                on="_align_timestamp",
                how="left",
                sort=False,
            )
            merged = merged.sort_values("_local_index")
            if int(merged[value_cols].notna().any(axis=1).sum()) > 0:
                for col in value_cols:
                    out[col] = merged[col].to_numpy()
                alignment = "timestamp"
    ts = _timestamp_series(frame, timestamp_col)
    valid_ts = ts.notna()
    future = valid_ts & (ts > current_end)
    current_start = current_end - pd.Timedelta(days=float(config.current_window_days))
    current = valid_ts & (ts >= current_start) & (ts <= current_end)
    label_overlap_mask, label_source = _local_label_overlap_mask(
        frame,
        timestamp_col=timestamp_col,
        end=current_end,
        config=config,
    )
    for col in _SIMILARITY_RESULT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out["similarity_to_current"] = pd.to_numeric(out["similarity_to_current"], errors="coerce")
    out["window_similarity"] = pd.to_numeric(out["window_similarity"], errors="coerce")
    out["day_similarity"] = pd.to_numeric(out["day_similarity"], errors="coerce")
    out["current_regime_recency_weight"] = pd.to_numeric(
        out["current_regime_recency_weight"],
        errors="coerce",
    )
    matched_rows = int(out["similarity_to_current"].notna().sum())
    local_recency = current_regime_recency_weights(
        ts,
        current_end=current_end,
        decay_per_week=float(config.recency_decay_per_week),
    )
    missing_similarity = out["similarity_to_current"].isna()
    out.loc[missing_similarity & current, "similarity_to_current"] = 1.0
    out.loc[missing_similarity & ~current, "similarity_to_current"] = 0.0
    out["similarity_to_current"] = out["similarity_to_current"].fillna(0.0).clip(0.0, 1.0)
    out["window_similarity"] = out["window_similarity"].fillna(out["similarity_to_current"]).clip(0.0, 1.0)
    out["day_similarity_available"] = out["day_similarity_available"].astype("boolean").fillna(False)
    out.loc[current, "day_similarity_available"] = True
    out["day_similarity"] = out["day_similarity"].fillna(0.0).clip(0.0, 1.0)
    out.loc[current, "day_similarity"] = 1.0
    out["current_regime_recency_weight"] = out["current_regime_recency_weight"].fillna(
        pd.Series(local_recency.to_numpy(dtype=np.float32), index=original_index)
    )
    out.loc[~current, "current_regime_recency_weight"] = 0.0
    out.loc[future, list(_SIMILARITY_RESULT_COLUMNS)] = [0.0, 0.0, False, 0.0, 0.0]
    out.loc[label_overlap_mask, ["window_similarity", "similarity_to_current"]] = 0.0
    out.loc[label_overlap_mask, "current_regime_recency_weight"] = 0.0
    out["regime_specialist_bucket"] = _derive_local_buckets_from_similarity(
        frame,
        out["similarity_to_current"].to_numpy(dtype=np.float32),
        timestamp_col=timestamp_col,
        end=current_end,
        config=config,
        label_overlap_mask=label_overlap_mask,
    )
    diagnostics = {
        "alignment": alignment,
        "aligned_rows": matched_rows,
        "aligned_fraction": float(matched_rows / max(len(frame), 1)),
        "local_rows": int(len(frame)),
        "local_current_rows": int(current.sum()),
        "local_future_rows": int(future.sum()),
        "local_label_end_source": label_source,
        "local_label_overlap_excluded_rows": int(label_overlap_mask.sum()),
    }
    return out, diagnostics


def compute_regime_similarity_to_current(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    selected_feature_columns: Sequence[str] | None = None,
    current_end: Any | None = None,
    config: RegimeSimilarityConfig = RegimeSimilarityConfig(),
    market_columns: Sequence[str] | None = None,
    drift_columns: Sequence[str] | None = None,
    covariance_columns: Sequence[str] | None = None,
    knn_columns: Sequence[str] | None = None,
    asset_return_col: str | None = None,
    assessment_frame: pd.DataFrame | None = None,
    unsupervised_regime_artifact: Any | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if frame is None or frame.empty:
        out = pd.DataFrame(index=getattr(frame, "index", None))
        return out, {
            "schema_version": REGIME_SPECIALIST_SCHEMA_VERSION,
            "enabled": False,
            "reason": "empty_frame",
        }
    if (
        assessment_frame is not None
        and not assessment_frame.empty
        and not (
            assessment_frame is frame
            or (
                len(assessment_frame) == len(frame)
                and assessment_frame.index.equals(frame.index)
            )
        )
    ):
        assessment_ts = _timestamp_series(assessment_frame, timestamp_col)
        assessment_end = (
            pd.to_datetime(current_end, utc=True, errors="coerce")
            if current_end is not None
            else assessment_ts.max()
        )
        if pd.isna(assessment_end):
            assessment_end = _timestamp_series(frame, timestamp_col).max()
        assessment_similarity, assessment_diag = compute_regime_similarity_to_current(
            assessment_frame,
            timestamp_col=timestamp_col,
            symbol_col=symbol_col,
            selected_feature_columns=selected_feature_columns,
            current_end=assessment_end,
            config=config,
            market_columns=market_columns,
            drift_columns=drift_columns,
            covariance_columns=covariance_columns,
            knn_columns=knn_columns,
            asset_return_col=asset_return_col,
            assessment_frame=None,
            unsupervised_regime_artifact=unsupervised_regime_artifact,
        )
        for key_col in (timestamp_col, symbol_col):
            if key_col in assessment_frame.columns and key_col not in assessment_similarity.columns:
                assessment_similarity = assessment_similarity.copy(deep=False)
                assessment_similarity[key_col] = assessment_frame[key_col].to_numpy(copy=False)
        aligned, alignment_diag = _align_global_assessment_similarity(
            frame,
            assessment_similarity,
            timestamp_col=timestamp_col,
            symbol_col=symbol_col,
            current_end=assessment_end,
            config=config,
        )
        future_index = aligned.index[
            aligned["regime_specialist_bucket"].astype(str).str.lower().eq("future_excluded").to_numpy(dtype=bool)
        ]
        aligned_fraction = float(alignment_diag.get("aligned_fraction", 0.0) or 0.0)
        min_aligned_fraction = float(np.clip(config.assessment_min_aligned_fraction, 0.0, 1.0))
        alignment_ok = aligned_fraction >= min_aligned_fraction
        assessment_enabled = bool(assessment_diag.get("enabled", False))
        reason = str(assessment_diag.get("reason", ""))
        if not alignment_ok:
            reason = (
                "global_alignment_insufficient:"
                f"{int(alignment_diag.get('aligned_rows', 0))}/{int(len(frame))}"
            )
        diagnostics = {
            "schema_version": REGIME_SPECIALIST_SCHEMA_VERSION,
            "enabled": bool(assessment_enabled and alignment_ok),
            "reason": reason,
            "assessment_scope": {
                "mode": "global_assessment_local_training",
                "assessment_rows": int(len(assessment_frame)),
                "local_training_rows": int(len(frame)),
                "alignment": alignment_diag.get("alignment", "none"),
                "aligned_rows": int(alignment_diag.get("aligned_rows", 0)),
                "aligned_fraction": aligned_fraction,
                "min_aligned_fraction": min_aligned_fraction,
                "alignment_ok": bool(alignment_ok),
            },
            "assessment_diagnostics": assessment_diag,
            "local_alignment": alignment_diag,
        }
        return _finalize_similarity_output(
            aligned,
            original_index=frame.index,
            future_index=future_index,
            diagnostics=diagnostics,
        )
    original_index = frame.index
    full_ts = _timestamp_series(frame, timestamp_col)
    valid_ts_full = full_ts.notna()
    if current_end is None:
        end = full_ts.max()
    else:
        end = pd.to_datetime(current_end, utc=True, errors="coerce")
    if pd.isna(end):
        end = full_ts.max()
    future_mask = valid_ts_full & (full_ts > end)
    asof_mask = valid_ts_full & (full_ts <= end)
    label_overlap_mask = pd.Series(False, index=frame.index)
    label_end = None
    label_end_source = "none"
    if config.label_end_col is not None and str(config.label_end_col) in frame.columns:
        label_end = pd.to_datetime(frame[str(config.label_end_col)], utc=True, errors="coerce")
        label_end_source = str(config.label_end_col)
    elif float(config.label_horizon_hours or 0.0) > 0.0:
        label_end = full_ts + pd.Timedelta(hours=float(config.label_horizon_hours))
        label_end_source = "label_horizon_hours"
    if label_end is not None:
        label_complete = label_end.notna() & (label_end <= end)
        label_overlap_mask = asof_mask & ~label_complete
        asof_mask = asof_mask & label_complete
    if not bool(asof_mask.any()):
        out = pd.DataFrame(index=original_index)
        out["similarity_to_current"] = 0.0
        out["window_similarity"] = 0.0
        out["day_similarity"] = 0.0
        out["day_similarity_available"] = False
        out["current_regime_recency_weight"] = 0.0
        out["regime_specialist_bucket"] = np.where(
            future_mask.to_numpy(dtype=bool),
            "future_excluded",
            "irrelevant",
        )
        return _finalize_similarity_output(
            out,
            original_index=original_index,
            future_index=original_index[future_mask.to_numpy(dtype=bool)],
            diagnostics={
                "schema_version": REGIME_SPECIALIST_SCHEMA_VERSION,
                "enabled": False,
                "reason": "no_asof_rows",
                "similarity_unavailable": True,
                "label_end_source": label_end_source,
                "label_overlap_excluded_rows": int(label_overlap_mask.sum()),
            },
        )
    work = frame.loc[asof_mask].copy(deep=False)
    unsupervised_regime_diag: dict[str, Any] = {
        "enabled": bool(unsupervised_regime_artifact is not None),
        "used": False,
    }
    if unsupervised_regime_artifact is not None:
        try:
            from .unsupervised_regime_learning.regime_models import (
                regime_artifact_assessment_summary,
            )

            unsupervised_regime_diag = regime_artifact_assessment_summary(
                unsupervised_regime_artifact
            )
        except Exception as exc:
            unsupervised_regime_diag = {
                "enabled": True,
                "used": False,
                "reason": f"artifact_assessment_failed:{type(exc).__name__}",
            }
    ts = full_ts.loc[asof_mask]
    valid_ts = ts.notna()
    if not bool(valid_ts.any()):
        out = pd.DataFrame(
            {
                "similarity_to_current": np.zeros(len(work), dtype=np.float32),
                "window_similarity": np.zeros(len(work), dtype=np.float32),
                "day_similarity": np.zeros(len(work), dtype=np.float32),
                "day_similarity_available": np.zeros(len(work), dtype=bool),
                "current_regime_recency_weight": np.zeros(len(work), dtype=np.float32),
                "regime_specialist_bucket": np.repeat("irrelevant", len(work)),
            },
            index=work.index,
        )
        return _finalize_similarity_output(
            out,
            original_index=original_index,
            future_index=original_index[future_mask.to_numpy(dtype=bool)],
            diagnostics={
                "schema_version": REGIME_SPECIALIST_SCHEMA_VERSION,
                "enabled": False,
                "reason": "missing_valid_timestamps",
                "similarity_unavailable": True,
                "label_end_source": label_end_source,
                "label_overlap_excluded_rows": int(label_overlap_mask.sum()),
            },
        )
    current_start = end - pd.Timedelta(days=float(config.current_window_days))
    current_mask = (ts >= current_start) & (ts <= end)
    if int(current_mask.sum()) < int(config.min_current_rows):
        out = pd.DataFrame(
            {
                "similarity_to_current": np.where(current_mask, 1.0, 0.0).astype(np.float32),
                "regime_specialist_bucket": np.where(current_mask, "current", "normal"),
                "window_similarity": np.where(current_mask, 1.0, 0.0).astype(np.float32),
                "day_similarity": np.where(current_mask, 1.0, 0.0).astype(np.float32),
                "day_similarity_available": current_mask.to_numpy(dtype=bool),
                "current_regime_recency_weight": np.zeros(len(work), dtype=np.float32),
            },
            index=work.index,
        )
        return _finalize_similarity_output(
            out,
            original_index=original_index,
            future_index=original_index[future_mask.to_numpy(dtype=bool)],
            diagnostics={
                "schema_version": REGIME_SPECIALIST_SCHEMA_VERSION,
                "enabled": False,
                "reason": "insufficient_current_rows",
                "similarity_unavailable": True,
                "current_rows": int(current_mask.sum()),
                "min_current_rows": int(config.min_current_rows),
                "label_end_source": label_end_source,
                "label_overlap_excluded_rows": int(label_overlap_mask.sum()),
            },
        )
    columns = infer_regime_specialist_columns(
        work,
        selected_feature_columns=selected_feature_columns,
        market_columns=market_columns,
        drift_columns=drift_columns,
        covariance_columns=covariance_columns,
        knn_columns=knn_columns,
        config=config,
    )
    column_selection = _column_selection_diagnostics(
        work,
        columns,
        market_columns=market_columns,
        drift_columns=drift_columns,
        covariance_columns=covariance_columns,
        knn_columns=knn_columns,
    )
    historical_cutoff = current_start - pd.Timedelta(days=max(0.0, float(config.embargo_days or 0.0)))
    historical_mask = valid_ts & (ts < historical_cutoff)
    feature_engineering_diag: dict[str, Any] = {
        "enabled": bool(config.feature_engineering_enabled),
        "used": False,
        "reason": "disabled",
    }
    feature_engineering_scores: np.ndarray | None = None
    if bool(config.feature_engineering_enabled):
        try:
            from .regime_specialist_feature_engineering import (
                RegimeFeatureEngineeringConfig,
                build_regime_specialist_feature_engineering_artifact,
            )

            fe_candidate_features = _stable_unique_strings(
                list(selected_feature_columns or [])
                + list(columns.get("market", []))
                + list(columns.get("covariance", []))
                + list(columns.get("knn", []))
            )
            fe_artifact = build_regime_specialist_feature_engineering_artifact(
                work,
                timestamp_col=timestamp_col,
                symbol_col=symbol_col,
                candidate_features=fe_candidate_features,
                unsupervised_regime_artifact=unsupervised_regime_artifact,
                current_mask=current_mask.to_numpy(dtype=bool),
                historical_mask=historical_mask.to_numpy(dtype=bool),
                config=RegimeFeatureEngineeringConfig(
                    max_final_features=int(config.feature_engineering_max_final_features),
                    max_pair_candidates=int(config.feature_engineering_max_pair_candidates),
                    univariate_subsample_per_class=int(config.feature_engineering_univariate_subsample_per_class),
                    lgbm_enabled=bool(config.feature_engineering_lgbm_enabled),
                    elasticnet_enabled=bool(config.feature_engineering_elasticnet_enabled),
                    grouped_cv_folds=int(config.feature_engineering_grouped_cv_folds),
                    grouped_cv_repeats=int(config.feature_engineering_grouped_cv_repeats),
                    permutation_repeats=int(config.feature_engineering_permutation_repeats),
                    max_permutation_features=int(config.feature_engineering_max_permutation_features),
                    max_permutation_rows=int(config.feature_engineering_max_permutation_rows),
                    max_shap_rows=int(config.feature_engineering_max_shap_rows),
                    drift_window_days=float(config.feature_engineering_drift_window_days),
                    max_drift_raw_features=int(config.feature_engineering_max_drift_raw_features),
                    drift_window_max_rows=int(config.feature_engineering_drift_window_max_rows),
                    drift_knn_max_rows=int(config.feature_engineering_drift_knn_max_rows),
                    drift_knn_chunk_pairs=int(config.feature_engineering_drift_knn_chunk_pairs),
                    domain_score_smoothing_enabled=bool(config.feature_engineering_domain_score_smoothing_enabled),
                    domain_score_ewma_half_life_days=float(config.feature_engineering_domain_score_ewma_half_life_days),
                    domain_score_ewma_max_days=float(config.feature_engineering_domain_score_ewma_max_days),
                    run_validation_diagnostics=bool(config.feature_engineering_run_validation_diagnostics),
                    random_state=int(config.random_state),
                    eps=float(config.eps),
                ),
            )
            fe_materialized = getattr(fe_artifact, "materialized_features", pd.DataFrame(index=work.index))
            fe_groups = getattr(fe_artifact, "materialized_feature_groups", {}) or {}
            if isinstance(fe_materialized, pd.DataFrame) and not fe_materialized.empty:
                aligned_materialized = fe_materialized.reindex(work.index)
                for col in aligned_materialized.columns:
                    if col not in work.columns:
                        work[col] = pd.to_numeric(aligned_materialized[col], errors="coerce").astype(np.float32)
            selected_raw_fe = [
                feature
                for feature in getattr(fe_artifact, "selected_raw_features", fe_artifact.selected_features)
                if feature in work.columns
            ]
            raw_materialized_fe = [
                feature
                for feature in list(fe_groups.get("raw_state", []))
                if feature in work.columns
            ]
            pair_materialized_fe = [
                feature
                for feature in list(fe_groups.get("pair_geometry", []))
                if feature in work.columns
            ]
            drift_materialized_fe = [
                feature
                for feature in list(fe_groups.get("generated_drift", []))
                if feature in work.columns
            ]
            raw_state_route_fe = raw_materialized_fe if raw_materialized_fe else selected_raw_fe
            raw_state_safe_fe = _feature_engineering_knn_safe_columns(raw_state_route_fe)
            pair_safe_fe = _feature_engineering_knn_safe_columns(pair_materialized_fe)
            if raw_state_route_fe:
                columns["covariance"] = raw_state_safe_fe
                columns["knn"] = raw_state_safe_fe
                columns["market"] = _stable_unique_strings(list(columns.get("market", [])) + raw_state_safe_fe)
            if pair_safe_fe:
                columns["covariance"] = _stable_unique_strings(list(columns.get("covariance", [])) + pair_safe_fe)
                columns["knn"] = _stable_unique_strings(list(columns.get("knn", [])) + pair_safe_fe)
            if drift_materialized_fe:
                columns["drift"] = _stable_unique_strings(list(columns.get("drift", [])) + drift_materialized_fe)
            raw_state_excluded_from_knn = [
                feature
                for feature in raw_state_route_fe
                if _feature_engineering_knn_unsafe_reason(feature) is not None
            ]
            pair_geometry_excluded_from_knn = [
                feature
                for feature in pair_materialized_fe
                if _feature_engineering_knn_unsafe_reason(feature) is not None
            ]
            score_materialized_fe = [str(feature) for feature in list(fe_groups.get("score", []))]
            excluded_from_knn = _stable_unique_strings(
                raw_state_excluded_from_knn
                + pair_geometry_excluded_from_knn
                + drift_materialized_fe
                + score_materialized_fe
            )
            lgbm_score_used = bool(fe_artifact.diagnostics.get("lgbm", {}).get("enabled", False))
            elastic_score_used = bool(fe_artifact.diagnostics.get("elasticnet", {}).get("enabled", False))
            if lgbm_score_used or elastic_score_used:
                score_series = fe_artifact.row_scores["regime_domain_current_likeness"].reindex(work.index)
                feature_engineering_scores = pd.to_numeric(score_series, errors="coerce").fillna(0.5).to_numpy(dtype=np.float32)
            if selected_raw_fe or raw_materialized_fe or pair_materialized_fe or drift_materialized_fe or feature_engineering_scores is not None:
                feature_engineering_diag = {
                    "enabled": True,
                    "used": True,
                    "classifier_score_used": bool(feature_engineering_scores is not None),
                    "selected_features": getattr(fe_artifact, "selected_features", []),
                    "selected_raw_features": selected_raw_fe,
                    "selected_pair_features": getattr(fe_artifact, "selected_pair_features", []),
                    "selected_drift_features": getattr(fe_artifact, "selected_drift_features", []),
                    "selected_feature_count": int(len(getattr(fe_artifact, "selected_features", []))),
                    "selected_raw_feature_count": int(len(selected_raw_fe)),
                    "materialized_feature_groups": fe_groups,
                    "materialized_feature_usage": {
                        "raw_state_to_market_covariance": raw_state_safe_fe,
                        "raw_state_to_knn": raw_state_safe_fe,
                        "raw_state_to_market_covariance_knn": raw_state_safe_fe,
                        "raw_state_fallback_to_original_columns": selected_raw_fe if not raw_materialized_fe else [],
                        "raw_state_excluded_from_knn": raw_state_excluded_from_knn,
                        "raw_state_excluded_from_state_blocks": raw_state_excluded_from_knn,
                        "pair_geometry_to_covariance": pair_safe_fe,
                        "pair_geometry_to_knn": pair_safe_fe,
                        "pair_geometry_to_covariance_knn": pair_safe_fe,
                        "pair_geometry_excluded_from_knn": pair_geometry_excluded_from_knn,
                        "pair_geometry_excluded_from_state_blocks": pair_geometry_excluded_from_knn,
                        "generated_drift_to_drift_only": drift_materialized_fe,
                        "score_to_domain_classifier_only": score_materialized_fe,
                        "excluded_from_knn": excluded_from_knn,
                    },
                    "lgbm_features": fe_artifact.lgbm_features,
                    "elasticnet_features": fe_artifact.elasticnet_features,
                    "diagnostics": fe_artifact.diagnostics,
                }
            else:
                feature_engineering_diag = {
                    "enabled": True,
                    "used": False,
                    "reason": "no_selected_features",
                    "diagnostics": fe_artifact.diagnostics,
                }
        except Exception as exc:
            feature_engineering_diag = {
                "enabled": True,
                "used": False,
                "reason": f"failed: {exc}",
            }
    timestamp_ns = _timestamp_ns(ts)
    current_pos = np.flatnonzero(current_mask.to_numpy(dtype=bool))
    end_ns = int(pd.Timestamp(end).value) if not pd.isna(end) else None
    current_weights = _position_recency_weights(
        timestamp_ns,
        current_pos,
        current_end_ns=end_ns,
        decay_per_week=float(config.recency_decay_per_week),
        eps=float(config.eps),
    )
    out = pd.DataFrame(index=work.index)
    out["window_similarity"] = np.where(current_mask, 1.0, 0.0).astype(np.float32)
    out["day_similarity"] = np.where(current_mask, 1.0, np.nan).astype(np.float32)
    out["day_similarity_available"] = current_mask.to_numpy(dtype=bool)
    out["similarity_to_current"] = np.where(current_mask, 1.0, 0.0).astype(np.float32)
    cur_weight_full = np.zeros(len(work), dtype=np.float32)
    cur_weight_full[np.flatnonzero(current_mask.to_numpy(dtype=bool))] = current_weights.astype(np.float32)
    out["current_regime_recency_weight"] = cur_weight_full

    if int(historical_mask.sum()) < int(config.min_candidate_rows):
        out["regime_specialist_bucket"] = np.where(current_mask, "current", "normal")
        return _finalize_similarity_output(
            out,
            original_index=original_index,
            future_index=original_index[future_mask.to_numpy(dtype=bool)],
            diagnostics={
                "schema_version": REGIME_SPECIALIST_SCHEMA_VERSION,
                "enabled": False,
                "reason": "insufficient_historical_candidate_rows",
                "current_rows": int(current_mask.sum()),
                "historical_rows": int(historical_mask.sum()),
                "historical_cutoff": str(historical_cutoff),
                "embargo_days": float(config.embargo_days),
                "label_end_source": label_end_source,
                "label_overlap_excluded_rows": int(label_overlap_mask.sum()),
                "columns": columns,
                "column_selection": column_selection,
            },
        )

    scale_fit_frame = work.loc[historical_mask]
    scaling_source = "pre_current_history"
    if scale_fit_frame.empty:
        scale_fit_frame = work
        scaling_source = "asof_fallback"
    market_z = _per_asset_robust_z(
        work,
        columns["market"],
        symbol_col=symbol_col,
        fit_frame=scale_fit_frame,
    )
    covariance_scaler = _fit_robust_column_scaler(scale_fit_frame, columns["covariance"])
    knn_scaler = _fit_robust_column_scaler(scale_fit_frame, columns["knn"])
    resolved_asset_return_col = _infer_asset_return_col(
        work,
        asset_return_col if asset_return_col is not None else config.asset_return_col,
    )
    market_cols, market_arr = _matrix_from_frame(market_z, columns["market"])
    _market_raw_cols, _market_raw_arr, market_missing = _numeric_matrix(work, market_cols)
    cov_cols, cov_arr, _cov_missing = _numeric_matrix(
        work,
        columns["covariance"],
        scaler=covariance_scaler,
    )
    drift_cols, drift_arr, drift_missing = _numeric_matrix(work, columns["drift"])
    drift_families = [_drift_family(col) for col in drift_cols]
    historical_mask_arr = historical_mask.to_numpy(dtype=bool)
    baseline_cov_norm = _baseline_covariance_norm(
        cov_arr,
        historical_mask_arr,
        float(config.eps),
    )
    drift_arr, drift_normalization = _normalize_drift_values_by_family(
        drift_arr,
        drift_cols,
        historical_mask_arr,
        baseline_covariance_norm=baseline_cov_norm,
        config=config,
    )
    asset_return_cache = _build_asset_return_cache(
        work,
        return_col=resolved_asset_return_col,
        timestamp_col=timestamp_col,
        symbol_col=symbol_col,
        max_assets=int(config.max_asset_covariance_assets),
        max_time_rows=int(config.max_asset_covariance_time_rows),
        min_observation_fraction=float(config.min_asset_observation_fraction),
    )
    current_fp_pos = _subsample_positions(
        current_pos,
        max_rows=int(config.max_fingerprint_rows_per_window),
    )
    current_fp_weights = _position_recency_weights(
        timestamp_ns,
        current_fp_pos,
        current_end_ns=end_ns,
        decay_per_week=float(config.recency_decay_per_week),
        eps=float(config.eps),
    )
    current_market = _market_fingerprint_array(
        market_arr,
        market_missing,
        timestamp_ns,
        current_fp_pos,
        current_fp_weights,
    )
    current_feature_cov = _covariance_fingerprint_array(
        cov_arr,
        current_fp_pos,
        weights=current_fp_weights,
        top_eigenvalues=int(config.top_eigenvalues),
        eps=float(config.eps),
    )
    current_asset_cov = _asset_covariance_fingerprint_from_cache(
        asset_return_cache,
        current_fp_pos,
        current_fp_weights,
        top_eigenvalues=int(config.top_eigenvalues),
        shrinkage=float(config.asset_covariance_shrinkage),
        eps=float(config.eps),
    )
    current_cov = np.concatenate([current_feature_cov, current_asset_cov]).astype(np.float32)
    current_drift = _drift_fingerprint_array(
        drift_arr,
        drift_missing,
        drift_families,
        timestamp_ns,
        current_fp_pos,
        current_fp_weights,
        eps=float(config.eps),
    )
    current_fingerprint = _concat_fingerprint(current_market, current_cov, current_drift)

    hist_ts = ts.loc[historical_mask]
    anchor = hist_ts.min()
    window_id_full = np.full(len(work), -1, dtype=np.int64)
    window_id_full[np.flatnonzero(historical_mask.to_numpy(dtype=bool))] = _window_ids(
        hist_ts,
        anchor=anchor,
        window_days=float(config.candidate_window_days),
    )
    scaled_knn = _scale_knn_matrix(work, columns["knn"], scaler=knn_scaler)
    historical_pos = np.flatnonzero(historical_mask.to_numpy(dtype=bool))
    knn_distance_map, knn_distance_fallback, knn_diagnostics = _knn_window_distances_global(
        scaled_knn,
        current_pos,
        historical_pos,
        window_id_full,
        current_weights,
        k=int(config.knn_k),
        max_current_rows=int(config.max_knn_current_rows),
        max_candidate_rows=int(config.max_knn_candidate_rows),
        max_historical_rows=int(config.max_knn_historical_rows),
        fallback_chunk_pairs=int(config.knn_fallback_chunk_pairs),
        eps=float(config.eps),
    )
    window_rows: list[dict[str, Any]] = []
    fingerprints: list[np.ndarray] = []
    market_fingerprints: list[np.ndarray] = []
    covariance_fingerprints: list[np.ndarray] = []
    drift_fingerprints: list[np.ndarray] = []
    for window_id in sorted(set(window_id_full[window_id_full >= 0])):
        pos = np.flatnonzero(window_id_full == int(window_id))
        if len(pos) < int(config.min_candidate_rows):
            continue
        weights_all = _position_recency_weights(
            timestamp_ns,
            pos,
            current_end_ns=None,
            decay_per_week=float(config.recency_decay_per_week),
            eps=float(config.eps),
        )
        fp_pos = _subsample_positions(
            pos,
            max_rows=int(config.max_fingerprint_rows_per_window),
        )
        weights = _position_recency_weights(
            timestamp_ns,
            fp_pos,
            current_end_ns=None,
            decay_per_week=float(config.recency_decay_per_week),
            eps=float(config.eps),
        )
        market_fp = _market_fingerprint_array(
            market_arr,
            market_missing,
            timestamp_ns,
            fp_pos,
            weights,
        )
        feature_cov_fp = _covariance_fingerprint_array(
            cov_arr,
            fp_pos,
            weights=weights,
            top_eigenvalues=int(config.top_eigenvalues),
            eps=float(config.eps),
        )
        asset_cov_fp = _asset_covariance_fingerprint_from_cache(
            asset_return_cache,
            fp_pos,
            weights=weights,
            top_eigenvalues=int(config.top_eigenvalues),
            shrinkage=float(config.asset_covariance_shrinkage),
            eps=float(config.eps),
        )
        cov_fp = np.concatenate([feature_cov_fp, asset_cov_fp]).astype(np.float32)
        drift_fp = _drift_fingerprint_array(
            drift_arr,
            drift_missing,
            drift_families,
            timestamp_ns,
            fp_pos,
            weights,
            eps=float(config.eps),
        )
        knn_distance = knn_distance_map.get(int(window_id), float(knn_distance_fallback))
        domain_similarity = 1.0
        if feature_engineering_scores is not None and len(feature_engineering_scores) == len(work):
            domain_similarity = _weighted_mean(
                np.asarray(feature_engineering_scores, dtype=np.float64)[pos],
                weights_all,
            )
        row = {
            "window_id": int(window_id),
            "start": str(ts.iloc[pos].min()),
            "end": str(ts.iloc[pos].max()),
            "rows": int(len(pos)),
            "fingerprint_rows": int(len(fp_pos)),
            "d_knn": float(knn_distance),
            "domain_classifier_similarity": float(np.clip(domain_similarity, 0.0, 1.0)),
            "d_domain_classifier": float(1.0 - np.clip(domain_similarity, 0.0, 1.0)),
        }
        window_rows.append(row)
        fingerprints.append(_concat_fingerprint(market_fp, cov_fp, drift_fp))
        market_fingerprints.append(market_fp)
        covariance_fingerprints.append(cov_fp)
        drift_fingerprints.append(drift_fp)

    if not window_rows:
        out["regime_specialist_bucket"] = np.where(current_mask, "current", "normal")
        return _finalize_similarity_output(
            out,
            original_index=original_index,
            future_index=original_index[future_mask.to_numpy(dtype=bool)],
            diagnostics={
                "schema_version": REGIME_SPECIALIST_SCHEMA_VERSION,
                "enabled": False,
                "reason": "no_candidate_windows",
                "current_rows": int(current_mask.sum()),
                "historical_rows": int(historical_mask.sum()),
                "historical_cutoff": str(historical_cutoff),
                "embargo_days": float(config.embargo_days),
                "label_end_source": label_end_source,
                "label_overlap_excluded_rows": int(label_overlap_mask.sum()),
                "columns": columns,
                "column_selection": column_selection,
            },
        )

    market_fp_matrix = np.vstack(market_fingerprints).astype(np.float32)
    cov_fp_matrix = np.vstack(covariance_fingerprints).astype(np.float32)
    drift_fp_matrix = np.vstack(drift_fingerprints).astype(np.float32)
    d_regime = _scaled_euclidean_by_indices(
        market_fp_matrix,
        current_market,
        np.arange(market_fp_matrix.shape[1]),
        float(config.eps),
    )
    d_cov, cov_block_distances, cov_block_scaling = _covariance_block_distances(
        cov_fp_matrix,
        current_cov,
        top_eigenvalues=int(config.top_eigenvalues),
        config=config,
    )
    d_drift, drift_block_distances, drift_block_scaling = _drift_block_distances(
        drift_fp_matrix,
        current_drift,
        config=config,
    )
    for i, row in enumerate(window_rows):
        row["d_regime"] = float(d_regime[i])
        row["d_cov"] = float(d_cov[i])
        row["d_drift"] = float(d_drift[i])
        for block_name, values in cov_block_distances.items():
            row[f"d_cov_{block_name}"] = float(values[i])
        for block_name, values in drift_block_distances.items():
            row[f"d_drift_{block_name}"] = float(values[i])

    d_regime_norm, d_regime_scale = _normalise_distances_with_scale(
        d_regime,
        float(config.eps),
    )
    d_cov_norm, d_cov_scale = _normalise_distances_with_scale(
        d_cov,
        float(config.eps),
    )
    d_drift_norm, d_drift_scale = _normalise_distances_with_scale(
        d_drift,
        float(config.eps),
    )
    d_knn_norm, d_knn_scale = _normalise_distances_with_scale(
        np.asarray([r["d_knn"] for r in window_rows]),
        float(config.eps),
    )
    d_domain_norm, d_domain_scale = _normalise_distances_with_scale(
        np.asarray([r["d_domain_classifier"] for r in window_rows]),
        float(config.eps),
    )
    effective_domain_classifier_weight = (
        float(config.domain_classifier_weight)
        if feature_engineering_scores is not None
        else 0.0
    )
    fp_matrix = np.vstack(fingerprints).astype(np.float32)
    ae_used = bool(config.ae_enabled) and len(window_rows) >= int(config.ae_min_windows)
    sim_ae = (
        _ae_similarity(fp_matrix, current_fingerprint, config=config)
        if ae_used
        else np.zeros(len(window_rows), dtype=np.float32)
    )
    ae_reason = "used" if ae_used else (
        "disabled" if not bool(config.ae_enabled) else "insufficient_candidate_windows"
    )
    composite_distance = (
        float(config.drift_weight) * d_drift_norm
        + float(config.covariance_weight) * d_cov_norm
        + float(config.regime_weight) * d_regime_norm
        + float(config.knn_weight) * d_knn_norm
        + effective_domain_classifier_weight * d_domain_norm
    )
    tau = (
        float(config.tau)
        if config.tau is not None and np.isfinite(float(config.tau)) and float(config.tau) > 0.0
        else float(np.nanmedian(composite_distance[np.isfinite(composite_distance)]))
    )
    tau = max(tau, float(config.eps))
    base_similarity = np.exp(
        -np.power(np.maximum(composite_distance, 0.0) / tau, float(config.alpha))
    )
    ae_weight = float(np.clip(config.ae_weight, 0.0, 1.0))
    if ae_used and ae_weight > 0.0:
        analogue_quality = np.clip(
            (1.0 - ae_weight) * base_similarity + ae_weight * sim_ae,
            0.0,
            1.0,
        )
        ae_blend_mode = "convex_blend"
    else:
        analogue_quality = np.clip(base_similarity, 0.0, 1.0)
        ae_blend_mode = "disabled_or_unavailable"
    window_similarity_map: dict[int, float] = {}
    for i, row in enumerate(window_rows):
        row.update(
            {
                "d_regime_norm": float(d_regime_norm[i]),
                "d_cov_norm": float(d_cov_norm[i]),
                "d_drift_norm": float(d_drift_norm[i]),
                "d_knn_norm": float(d_knn_norm[i]),
                "d_domain_classifier_norm": float(d_domain_norm[i]),
                "sim_ae": float(sim_ae[i]),
                "composite_distance": float(composite_distance[i]),
                "analogue_quality": float(analogue_quality[i]),
            }
        )
        window_similarity_map[int(row["window_id"])] = float(analogue_quality[i])
    for window_id, sim in window_similarity_map.items():
        out.loc[window_id_full == int(window_id), "window_similarity"] = float(sim)

    day_similarity = _compute_day_similarity(
        work,
        ts,
        historical_mask=historical_mask,
        current_market=current_market,
        current_cov=current_cov,
        current_drift=current_drift,
        market_arr=market_arr,
        market_missing=market_missing,
        cov_arr=cov_arr,
        asset_return_cache=asset_return_cache,
        drift_arr=drift_arr,
        drift_missing=drift_missing,
        drift_families=drift_families,
        timestamp_ns=timestamp_ns,
        config=config,
    )
    day_available = day_similarity.notna()
    current_mask_work = current_mask.reindex(day_similarity.index).fillna(False)
    day_output = day_similarity.copy()
    day_output.loc[current_mask_work] = 1.0
    day_available_output = day_available | current_mask_work
    out.loc[day_similarity.index, "day_similarity_available"] = day_available_output.to_numpy(dtype=bool)
    out.loc[day_similarity.index, "day_similarity"] = day_output.fillna(0.0).to_numpy(dtype=np.float32)
    hist_idx = historical_mask.to_numpy(dtype=bool)
    day_strength = float(np.clip(config.day_similarity_strength, 0.0, 1.0))
    day_for_multiplier = day_similarity.reindex(out.index).fillna(1.0).to_numpy(dtype=np.float32)
    day_multiplier = (
        (1.0 - day_strength)
        + day_strength * day_for_multiplier
    ).astype(np.float32)
    row_similarity = (
        out["window_similarity"].to_numpy(dtype=np.float32)
        * day_multiplier
    )
    out.loc[hist_idx, "similarity_to_current"] = np.clip(row_similarity[hist_idx], 0.0, 1.0)
    buckets = np.repeat("irrelevant", len(work)).astype(object)
    buckets[current_mask.to_numpy(dtype=bool)] = "current"
    sim_vals = out["similarity_to_current"].to_numpy(dtype=np.float32)
    analogue_mask = hist_idx & (sim_vals >= float(config.analogue_threshold))
    normal_mask = hist_idx & ~analogue_mask & (sim_vals >= float(config.normal_threshold))
    buckets[analogue_mask] = "analogue"
    buckets[normal_mask] = "normal"
    out["regime_specialist_bucket"] = buckets
    diagnostics = {
        "schema_version": REGIME_SPECIALIST_SCHEMA_VERSION,
        "enabled": True,
        "current_start": str(current_start),
        "current_end": str(end),
        "historical_cutoff": str(historical_cutoff),
        "embargo_days": float(config.embargo_days),
        "label_end_source": label_end_source,
        "label_overlap_excluded_rows": int(label_overlap_mask.sum()),
        "current_rows": int(current_mask.sum()),
        "current_fingerprint_rows": int(len(current_fp_pos)),
        "historical_rows": int(historical_mask.sum()),
        "candidate_window_count": int(len(window_rows)),
        "fingerprint_limits": {
            "max_rows_per_window": int(config.max_fingerprint_rows_per_window),
            "max_day_rows": int(config.max_day_fingerprint_rows),
        },
        "columns": columns,
        "column_selection": column_selection,
        "scaling": {
            "source": scaling_source,
            "fit_rows": int(len(scale_fit_frame)),
            "market_columns_scaled": int(len(market_cols)),
            "covariance_columns_scaled": int(len(cov_cols)),
            "knn_columns_scaled": int(len(knn_scaler)),
        },
        "feature_engineering": feature_engineering_diag,
        "unsupervised_regime_learning": unsupervised_regime_diag,
        "drift_normalization": drift_normalization,
        "asset_covariance": {
            "enabled": bool(asset_return_cache.enabled),
            "return_col": asset_return_cache.return_col,
            "feature_count": int(len(current_asset_cov)),
            "time_rows": int(len(asset_return_cache.time_ns)),
            "asset_count": int(asset_return_cache.matrix.shape[1]) if asset_return_cache.matrix.ndim == 2 else 0,
            "max_assets": int(config.max_asset_covariance_assets),
            "max_time_rows": int(config.max_asset_covariance_time_rows),
            "min_observation_fraction": float(config.min_asset_observation_fraction),
            "shrinkage": float(config.asset_covariance_shrinkage),
        },
        "knn": knn_diagnostics,
        "autoencoder": {
            "enabled": bool(config.ae_enabled),
            "used": bool(ae_used),
            "reason": ae_reason,
            "min_windows": int(config.ae_min_windows),
            "max_windows": int(config.ae_max_windows),
            "blend_mode": ae_blend_mode,
        },
        "day_similarity": {
            "strength": day_strength,
            "min_rows": int(config.day_similarity_min_rows),
            "available_rows": int(day_available_output.sum()),
            "unavailable_policy": "neutral_multiplier_output_zero",
        },
        "weights": {
            "feature_drift_distance": float(config.drift_weight),
            "covariance_distance": float(config.covariance_weight),
            "regime_state_distance": float(config.regime_weight),
            "knn_distance": float(config.knn_weight),
            "domain_classifier_distance": float(effective_domain_classifier_weight),
            "domain_classifier_distance_configured": float(config.domain_classifier_weight),
            "ae_similarity": float(config.ae_weight),
        },
        "block_scaling": {
            "combined_from_normalized_distances": True,
            "internal_distance_scaling": "component_robust_scale_within_weighted_blocks",
            "regime_distance_median": float(d_regime_scale),
            "covariance_distance_median": float(d_cov_scale),
            "drift_distance_median": float(d_drift_scale),
            "knn_distance_median": float(d_knn_scale),
            "domain_classifier_distance_median": float(d_domain_scale),
            "covariance_subblocks": cov_block_scaling,
            "drift_subblocks": drift_block_scaling,
            "tau": float(tau),
            "alpha": float(config.alpha),
            "threshold_calibration": "relative_candidate_distribution",
        },
        "window_diagnostics": window_rows[: max(0, int(config.max_window_diagnostics))],
        "window_diagnostics_count": int(len(window_rows)),
        "bucket_counts": {
            str(k): int(v)
            for k, v in pd.Series(buckets).value_counts(dropna=False).to_dict().items()
        },
    }
    return _finalize_similarity_output(
        out.astype(
            {
                "window_similarity": np.float32,
                "day_similarity": np.float32,
                "day_similarity_available": bool,
                "similarity_to_current": np.float32,
                "current_regime_recency_weight": np.float32,
            },
            copy=False,
        ),
        original_index=original_index,
        future_index=original_index[future_mask.to_numpy(dtype=bool)],
        diagnostics=diagnostics,
    )


def _compute_day_similarity(
    work: pd.DataFrame,
    ts: pd.Series,
    *,
    historical_mask: pd.Series,
    current_market: np.ndarray,
    current_cov: np.ndarray,
    current_drift: np.ndarray,
    market_arr: np.ndarray,
    market_missing: np.ndarray,
    cov_arr: np.ndarray,
    asset_return_cache: _AssetReturnCache,
    drift_arr: np.ndarray,
    drift_missing: np.ndarray,
    drift_families: Sequence[str],
    timestamp_ns: np.ndarray,
    config: RegimeSimilarityConfig,
) -> pd.Series:
    out = pd.Series(np.nan, index=work.index, dtype=np.float32)
    hist_ts = ts.loc[historical_mask]
    if hist_ts.empty:
        return out
    anchor = hist_ts.min()
    day_ids = np.full(len(work), -1, dtype=np.int64)
    day_ids[np.flatnonzero(historical_mask.to_numpy(dtype=bool))] = _window_ids(
        hist_ts,
        anchor=anchor,
        window_days=float(config.day_window_days),
    )
    ids = []
    day_positions: list[np.ndarray] = []
    market_fingerprints: list[np.ndarray] = []
    covariance_fingerprints: list[np.ndarray] = []
    drift_fingerprints: list[np.ndarray] = []
    for day_id in sorted(set(day_ids[day_ids >= 0])):
        pos = np.flatnonzero(day_ids == int(day_id))
        if len(pos) < int(config.day_similarity_min_rows):
            continue
        fp_pos = _subsample_positions(
            pos,
            max_rows=int(config.max_day_fingerprint_rows),
        )
        weights = _position_recency_weights(
            timestamp_ns,
            fp_pos,
            current_end_ns=None,
            decay_per_week=float(config.recency_decay_per_week),
            eps=float(config.eps),
        )
        market_fp = _market_fingerprint_array(
            market_arr,
            market_missing,
            timestamp_ns,
            fp_pos,
            weights,
        )
        feature_cov_fp = _covariance_fingerprint_array(
            cov_arr,
            fp_pos,
            weights=weights,
            top_eigenvalues=int(config.top_eigenvalues),
            eps=float(config.eps),
        )
        asset_cov_fp = _asset_covariance_fingerprint_from_cache(
            asset_return_cache,
            fp_pos,
            weights=weights,
            top_eigenvalues=int(config.top_eigenvalues),
            shrinkage=float(config.asset_covariance_shrinkage),
            eps=float(config.eps),
        )
        cov_fp = np.concatenate([feature_cov_fp, asset_cov_fp]).astype(np.float32)
        drift_fp = _drift_fingerprint_array(
            drift_arr,
            drift_missing,
            drift_families,
            timestamp_ns,
            fp_pos,
            weights,
            eps=float(config.eps),
        )
        market_fingerprints.append(market_fp)
        covariance_fingerprints.append(cov_fp)
        drift_fingerprints.append(drift_fp)
        ids.append(int(day_id))
        day_positions.append(pos)
    if not ids:
        return out
    market_fp_matrix = np.vstack(market_fingerprints).astype(np.float32)
    cov_fp_matrix = np.vstack(covariance_fingerprints).astype(np.float32)
    drift_fp_matrix = np.vstack(drift_fingerprints).astype(np.float32)
    regime_d = _scaled_euclidean_by_indices(
        market_fp_matrix,
        current_market,
        np.arange(market_fp_matrix.shape[1]),
        float(config.eps),
    )
    cov_d, _cov_blocks, _cov_diag = _covariance_block_distances(
        cov_fp_matrix,
        current_cov,
        top_eigenvalues=int(config.top_eigenvalues),
        config=config,
    )
    drift_d, _drift_blocks, _drift_diag = _drift_block_distances(
        drift_fp_matrix,
        current_drift,
        config=config,
    )
    regime_n = _normalise_distances(regime_d, float(config.eps))
    cov_n = _normalise_distances(cov_d, float(config.eps))
    drift_n = _normalise_distances(drift_d, float(config.eps))
    d = (
        float(config.drift_weight) * drift_n
        + float(config.covariance_weight) * cov_n
        + float(config.regime_weight) * regime_n
    )
    sim = np.exp(-np.power(np.maximum(d, 0.0), float(config.alpha)))
    out_values = out.to_numpy(dtype=np.float32, copy=True)
    for pos, value in zip(day_positions, sim):
        out_values[pos] = float(np.clip(value, 0.0, 1.0))
    out = pd.Series(out_values, index=work.index, dtype=np.float32)
    return out.astype(np.float32, copy=False)


def _saturating_reliability(effective_count: float, tau: float) -> float:
    effective_count = max(float(effective_count), 0.0)
    tau = max(float(tau), 1e-12)
    return 1.0 - math.exp(-effective_count / tau)


def _less_interesting_mass_bounds(
    config: SpecialistWeightConfig,
) -> tuple[float, float]:
    lo = float(np.clip(config.less_interesting_min_mass, 0.0, 1.0))
    hi = float(np.clip(config.less_interesting_max_mass, 0.0, 1.0))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _less_interesting_mass_cap(
    config: SpecialistWeightConfig,
    adaptive_n_eff_reliability: float,
    replay_n_eff_reliability: float,
) -> float:
    lo, hi = _less_interesting_mass_bounds(config)
    adaptive_rel = float(np.clip(adaptive_n_eff_reliability, 0.0, 1.0))
    replay_rel = float(np.clip(replay_n_eff_reliability, 0.0, 1.0))
    replay_need = 1.0 - adaptive_rel
    cap = lo + (hi - lo) * replay_need * replay_rel
    max_replay_from_adaptive_floor = 1.0 - float(
        np.clip(config.min_current_plus_analogue_mass, 0.0, 1.0),
    )
    upper = max(0.0, min(hi, max_replay_from_adaptive_floor))
    if upper < lo:
        return float(upper)
    return float(np.clip(cap, lo, upper))


def _normalize_bucket_mass(mass: Dict[str, float], eps: float) -> Dict[str, float]:
    total = float(sum(mass.values()))
    if total > eps:
        return {k: float(v) / total for k, v in mass.items()}
    return {k: float(v) for k, v in mass.items()}


def _cap_bucket_masses(
    bucket_mass: Dict[str, float],
    config: SpecialistWeightConfig,
    *,
    adaptive_n_eff: float,
    replay_n_eff: float,
    adaptive_n_eff_reliability: float,
    replay_n_eff_reliability: float,
) -> tuple[Dict[str, float], Dict[str, float | bool | str]]:
    mass = dict(bucket_mass)
    replay_need = 1.0 - float(np.clip(adaptive_n_eff_reliability, 0.0, 1.0))
    cap_diag: dict[str, float | bool | str] = {
        "bucket_mass_caps_enforced": False,
        "bucket_mass_cap_reason": "no_current_or_analogue_rows",
        "bucket_mass_basis": "n_eff_reliability",
        "less_interesting_mass_before_caps": float(
            mass.get("normal", 0.0) + mass.get("irrelevant", 0.0)
        ),
        "less_interesting_mass_cap": float(config.less_interesting_max_mass),
        "less_interesting_mass_min": float(config.less_interesting_min_mass),
        "less_interesting_mass_max": float(config.less_interesting_max_mass),
        "adaptive_n_eff": float(adaptive_n_eff),
        "replay_n_eff": float(replay_n_eff),
        "adaptive_n_eff_reliability": float(adaptive_n_eff_reliability),
        "replay_n_eff_reliability": float(replay_n_eff_reliability),
        "replay_need": float(replay_need),
    }
    adaptive_total = mass["current"] + mass["analogue"]
    if adaptive_total > config.eps and adaptive_total < config.min_current_plus_analogue_mass:
        needed = config.min_current_plus_analogue_mass - adaptive_total
        replay_total = mass["normal"] + mass["irrelevant"]
        transfer = min(needed, replay_total)
        if replay_total > config.eps and transfer > 0:
            normal_take = transfer * mass["normal"] / replay_total
            irrelevant_take = transfer * mass["irrelevant"] / replay_total
            mass["normal"] -= normal_take
            mass["irrelevant"] -= irrelevant_take
            mass["current"] += transfer * mass["current"] / adaptive_total
            mass["analogue"] += transfer * mass["analogue"] / adaptive_total
    mass = _normalize_bucket_mass(mass, float(config.eps))
    adaptive_total = float(mass["current"] + mass["analogue"])
    less_total = float(mass["normal"] + mass["irrelevant"])
    cap_diag["bucket_mass_caps_enforced"] = bool(adaptive_total > config.eps)
    if adaptive_total > config.eps:
        cap_diag["bucket_mass_cap_reason"] = "ok"
        less_min, less_max = _less_interesting_mass_bounds(config)
        less_cap = _less_interesting_mass_cap(
            config,
            adaptive_n_eff_reliability,
            replay_n_eff_reliability,
        )
        cap_diag["less_interesting_mass_cap"] = float(less_cap)
        cap_diag["less_interesting_mass_min"] = float(less_min)
        cap_diag["less_interesting_mass_max"] = float(less_max)
        if less_total > less_cap + config.eps:
            excess = less_total - less_cap
            if less_total > config.eps:
                mass["normal"] -= excess * mass["normal"] / less_total
                mass["irrelevant"] -= excess * mass["irrelevant"] / less_total
            adaptive_total = mass["current"] + mass["analogue"]
            mass["current"] += excess * mass["current"] / max(adaptive_total, config.eps)
            mass["analogue"] += excess * mass["analogue"] / max(adaptive_total, config.eps)
        elif less_total > config.eps and less_total < less_min - config.eps:
            transfer = min(less_min - less_total, adaptive_total)
            adaptive_total = mass["current"] + mass["analogue"]
            if transfer > 0.0 and adaptive_total > config.eps:
                mass["current"] -= transfer * mass["current"] / adaptive_total
                mass["analogue"] -= transfer * mass["analogue"] / adaptive_total
                less_total = mass["normal"] + mass["irrelevant"]
                if less_total > config.eps:
                    mass["normal"] += transfer * mass["normal"] / less_total
                    mass["irrelevant"] += transfer * mass["irrelevant"] / less_total
                else:
                    normal_share = float(config.normal_prior) / max(
                        float(config.normal_prior + config.irrelevant_prior),
                        config.eps,
                    )
                    mass["normal"] += transfer * normal_share
                    mass["irrelevant"] += transfer * (1.0 - normal_share)
    mass = _normalize_bucket_mass(mass, float(config.eps))
    cap_diag["less_interesting_mass_after_caps"] = float(mass["normal"] + mass["irrelevant"])
    return mass, cap_diag


def _normalize_with_bounds(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
    target_mean: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    w = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=target_mean, posinf=upper, neginf=lower)
    if w.size == 0:
        return w.astype(np.float32)
    lo = float(min(lower, upper))
    hi = float(max(lower, upper))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0 or hi < lo:
        lo, hi = 0.05, 20.0
    target_sum = float(target_mean) * float(len(w))
    w = np.clip(w, lo, hi)
    free = np.ones(len(w), dtype=bool)
    for _ in range(20):
        fixed_sum = float(np.sum(w[~free]))
        free_sum = float(np.sum(w[free]))
        if free_sum <= eps or not bool(free.any()):
            break
        scale = (target_sum - fixed_sum) / max(free_sum, eps)
        w[free] *= scale
        below = free & (w < lo)
        above = free & (w > hi)
        if not bool(below.any()) and not bool(above.any()):
            break
        w[below] = lo
        w[above] = hi
        free[below | above] = False
    return np.clip(w, lo, hi).astype(np.float32)


def _normalize_group_sum_with_bounds(
    values: np.ndarray,
    *,
    target_sum: float,
    lower: float,
    upper: float,
    eps: float = 1e-12,
) -> np.ndarray:
    w = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=upper, neginf=lower)
    if w.size == 0:
        return w.astype(np.float32)
    lo = float(min(lower, upper))
    hi = float(max(lower, upper))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0 or hi < lo:
        lo, hi = 0.05, 20.0
    target = float(np.clip(target_sum, lo * len(w), hi * len(w)))
    w = np.clip(w, lo, hi)
    free = np.ones(len(w), dtype=bool)
    for _ in range(20):
        fixed_sum = float(np.sum(w[~free]))
        free_sum = float(np.sum(w[free]))
        if free_sum <= eps or not bool(free.any()):
            break
        scale = (target - fixed_sum) / max(free_sum, eps)
        w[free] *= scale
        below = free & (w < lo)
        above = free & (w > hi)
        if not bool(below.any()) and not bool(above.any()):
            break
        w[below] = lo
        w[above] = hi
        free[below | above] = False
    return np.clip(w, lo, hi).astype(np.float32)


def compute_specialist_sample_weights(
    df: pd.DataFrame,
    bucket_col: str = "regime_specialist_bucket",
    similarity_col: str = "similarity_to_current",
    recency_col: Optional[str] = None,
    config: SpecialistWeightConfig = SpecialistWeightConfig(),
) -> Tuple[pd.Series, Dict[str, Any]]:
    allowed = {"current", "analogue", "normal", "irrelevant", "future_excluded"}
    buckets = df[bucket_col].astype(str).str.lower()
    bad = set(buckets.unique()) - allowed
    if bad:
        raise ValueError(f"Unknown bucket values: {bad}. Expected {allowed}.")
    excluded = buckets == "future_excluded"
    active = ~excluded
    if not bool(active.any()):
        weights = pd.Series(0.0, index=df.index, name="sample_weight", dtype=np.float32)
        return weights, {
            "adaptive_reliability": 0.0,
            "should_train_specialist": False,
            "current_mass": 0.0,
            "analogue_mass": 0.0,
            "normal_mass": 0.0,
            "irrelevant_mass": 0.0,
            "less_interesting_mass": 0.0,
            "less_interesting_mass_cap": float(config.less_interesting_max_mass),
            "less_interesting_mass_min": float(config.less_interesting_min_mass),
            "less_interesting_mass_max": float(config.less_interesting_max_mass),
            "min_current_plus_analogue_mass": float(config.min_current_plus_analogue_mass),
            "actual_less_interesting_weight_mass": 0.0,
            "future_excluded_rows": int(excluded.sum()),
        }
    sim = df[similarity_col].astype(float).clip(0.0, 1.0)
    if recency_col is None:
        recency = pd.Series(1.0, index=df.index)
    else:
        recency = df[recency_col].astype(float).clip(lower=0.0)
    recency_power = float(config.recency_power)
    if not np.isfinite(recency_power):
        recency_power = 0.5
    recency_factor = recency.pow(max(recency_power, 0.0))
    masks = {
        "current": buckets == "current",
        "analogue": buckets == "analogue",
        "normal": buckets == "normal",
        "irrelevant": buckets == "irrelevant",
    }
    replay_mask = masks["normal"] | masks["irrelevant"]
    row_score = pd.Series(0.0, index=df.index)
    row_score.loc[masks["current"]] = (
        sim.loc[masks["current"]].pow(config.current_gamma)
        * recency_factor.loc[masks["current"]]
    )
    row_score.loc[masks["analogue"]] = (
        sim.loc[masks["analogue"]].pow(config.analogue_gamma)
        * recency_factor.loc[masks["analogue"]]
    )
    row_score.loc[replay_mask] = (
        sim.loc[replay_mask].pow(config.replay_gamma)
        * recency_factor.loc[replay_mask]
    )
    row_score = row_score.clip(lower=config.eps)
    eff = {name: float(row_score.loc[mask].sum()) for name, mask in masks.items()}
    adaptive_n_eff = float(eff["current"] + eff["analogue"])
    replay_n_eff = float(eff["normal"] + eff["irrelevant"])
    reliability = {
        "current": _saturating_reliability(eff["current"], config.tau_current),
        "analogue": _saturating_reliability(eff["analogue"], config.tau_analogue),
        "normal": _saturating_reliability(eff["normal"], config.tau_normal),
        "irrelevant": _saturating_reliability(eff["irrelevant"], config.tau_irrelevant),
    }
    adaptive_reliability = 1.0 - (1.0 - reliability["current"]) * (1.0 - reliability["analogue"])
    adaptive_tau = min(float(config.tau_current), float(config.tau_analogue))
    replay_tau = min(float(config.tau_normal), float(config.tau_irrelevant))
    adaptive_n_eff_reliability = _saturating_reliability(adaptive_n_eff, adaptive_tau)
    replay_n_eff_reliability = _saturating_reliability(replay_n_eff, replay_tau)
    replay_score = (
        replay_n_eff_reliability
        if bool(replay_mask.any())
        else 0.0
    )
    bucket_score = {
        "current": config.current_prior * reliability["current"],
        "analogue": config.analogue_prior * reliability["analogue"],
        "normal": 0.0,
        "irrelevant": 0.0,
    }
    replay_normal_share = eff["normal"] / max(replay_n_eff, config.eps)
    bucket_score["normal"] = replay_score * replay_normal_share
    bucket_score["irrelevant"] = replay_score * (1.0 - replay_normal_share)
    for name, mask in {"current": masks["current"], "analogue": masks["analogue"]}.items():
        if not bool(mask.any()):
            bucket_score[name] = 0.0
    if not bool(replay_mask.any()):
        bucket_score["normal"] = 0.0
        bucket_score["irrelevant"] = 0.0
    total_score = sum(bucket_score.values())
    if total_score <= config.eps:
        weights = pd.Series(1.0, index=df.index, name="sample_weight")
        less_min, less_max = _less_interesting_mass_bounds(config)
        diagnostics = {
            "adaptive_reliability": 0.0,
            "should_train_specialist": False,
            "current_mass": 0.0,
            "analogue_mass": 0.0,
            "normal_mass": 0.0,
            "irrelevant_mass": 0.0,
            "less_interesting_mass": 0.0,
            "less_interesting_mass_cap": float(less_max),
            "less_interesting_mass_min": float(less_min),
            "less_interesting_mass_max": float(less_max),
            "min_current_plus_analogue_mass": float(config.min_current_plus_analogue_mass),
        }
        return weights, diagnostics
    bucket_mass = {k: v / total_score for k, v in bucket_score.items()}
    adaptive_mass_before_caps = float(bucket_mass.get("current", 0.0) + bucket_mass.get("analogue", 0.0))
    bucket_mass, cap_diag = _cap_bucket_masses(
        bucket_mass,
        config,
        adaptive_n_eff=adaptive_n_eff,
        replay_n_eff=replay_n_eff,
        adaptive_n_eff_reliability=adaptive_n_eff_reliability,
        replay_n_eff_reliability=replay_n_eff_reliability,
    )
    raw_weight = pd.Series(0.0, index=df.index)
    for bucket_name in ("current", "analogue"):
        mask = masks[bucket_name]
        if bool(mask.any()):
            score_sum = float(row_score.loc[mask].sum())
            raw_weight.loc[mask] = bucket_mass[bucket_name] * row_score.loc[mask] / max(score_sum, config.eps)
    if bool(replay_mask.any()):
        replay_mass = float(bucket_mass["normal"] + bucket_mass["irrelevant"])
        replay_score_sum = float(row_score.loc[replay_mask].sum())
        raw_weight.loc[replay_mask] = replay_mass * row_score.loc[replay_mask] / max(replay_score_sum, config.eps)
    weights = raw_weight * int(active.sum())
    weights.loc[active] = _normalize_with_bounds(
        weights.loc[active].to_numpy(dtype=np.float64),
        lower=float(config.min_weight),
        upper=float(config.max_weight),
        target_mean=1.0,
        eps=float(config.eps),
    )
    less_interesting_active = active & (masks["normal"] | masks["irrelevant"])
    adaptive_active = active & (masks["current"] | masks["analogue"])
    less_cap = float(cap_diag.get("less_interesting_mass_cap", config.less_interesting_max_mass))
    if bool(less_interesting_active.any()) and bool(adaptive_active.any()):
        target_total = float(int(active.sum()))
        current_less_sum = float(np.nansum(weights.loc[less_interesting_active].to_numpy(dtype=np.float64)))
        current_total = float(np.nansum(weights.loc[active].to_numpy(dtype=np.float64)))
        current_less_mass = current_less_sum / max(current_total, config.eps)
        if current_less_mass > less_cap + config.eps:
            target_less_sum = float(less_cap * target_total)
            target_less_sum = float(
                np.clip(
                    target_less_sum,
                    float(config.min_weight) * int(less_interesting_active.sum()),
                    float(config.max_weight) * int(less_interesting_active.sum()),
                )
            )
            target_adaptive_sum = target_total - target_less_sum
            target_adaptive_sum = float(
                np.clip(
                    target_adaptive_sum,
                    float(config.min_weight) * int(adaptive_active.sum()),
                    float(config.max_weight) * int(adaptive_active.sum()),
                )
            )
            target_less_sum = target_total - target_adaptive_sum
            weights.loc[less_interesting_active] = _normalize_group_sum_with_bounds(
                weights.loc[less_interesting_active].to_numpy(dtype=np.float64),
                target_sum=target_less_sum,
                lower=float(config.min_weight),
                upper=float(config.max_weight),
                eps=float(config.eps),
            )
            weights.loc[adaptive_active] = _normalize_group_sum_with_bounds(
                weights.loc[adaptive_active].to_numpy(dtype=np.float64),
                target_sum=target_adaptive_sum,
                lower=float(config.min_weight),
                upper=float(config.max_weight),
                eps=float(config.eps),
            )
    weights.loc[excluded] = 0.0
    weights = weights.rename("sample_weight")
    active_weights = weights.loc[active].to_numpy(dtype=np.float64)
    active_weight_total = float(np.nansum(active_weights))
    actual_bucket_mass: dict[str, float] = {}
    if active_weight_total > config.eps:
        for name, mask in masks.items():
            actual_bucket_mass[name] = float(
                np.nansum(weights.loc[mask & active].to_numpy(dtype=np.float64))
                / active_weight_total
            )
    else:
        actual_bucket_mass = {name: 0.0 for name in masks}
    actual_less_interesting_mass = float(
        actual_bucket_mass.get("normal", 0.0)
        + actual_bucket_mass.get("irrelevant", 0.0)
    )
    diagnostics = {
        "adaptive_reliability": adaptive_reliability,
        "should_train_specialist": adaptive_reliability >= config.min_adaptive_reliability_to_train,
        "current_mass": bucket_mass["current"],
        "analogue_mass": bucket_mass["analogue"],
        "normal_mass": bucket_mass["normal"],
        "irrelevant_mass": bucket_mass["irrelevant"],
        "less_interesting_mass": float(bucket_mass["normal"] + bucket_mass["irrelevant"]),
        "min_current_plus_analogue_mass": float(config.min_current_plus_analogue_mass),
        "actual_current_weight_mass": actual_bucket_mass.get("current", 0.0),
        "actual_analogue_weight_mass": actual_bucket_mass.get("analogue", 0.0),
        "actual_normal_weight_mass": actual_bucket_mass.get("normal", 0.0),
        "actual_irrelevant_weight_mass": actual_bucket_mass.get("irrelevant", 0.0),
        "actual_less_interesting_weight_mass": actual_less_interesting_mass,
        "actual_less_interesting_weight_cap_ok": bool(
            actual_less_interesting_mass
            <= float(cap_diag.get("less_interesting_mass_cap", config.less_interesting_max_mass))
            + 1e-9
        ),
        "effective_current": eff["current"],
        "effective_analogue": eff["analogue"],
        "effective_normal": eff["normal"],
        "effective_irrelevant": eff["irrelevant"],
        "recency_power": float(max(recency_power, 0.0)),
        "future_excluded_rows": int(excluded.sum()),
        "bucket_mass_caps_enforced": bool(cap_diag.get("bucket_mass_caps_enforced", False)),
        "bucket_mass_cap_reason": str(cap_diag.get("bucket_mass_cap_reason", "")),
        "adaptive_mass_before_caps": adaptive_mass_before_caps,
        "weight_min": float(np.nanmin(active_weights)) if active_weights.size else 0.0,
        "weight_mean": float(np.nanmean(active_weights)) if active_weights.size else 0.0,
        "weight_max": float(np.nanmax(active_weights)) if active_weights.size else 0.0,
        "weight_bounds": [float(config.min_weight), float(config.max_weight)],
    }
    diagnostics.update(cap_diag)
    return weights.astype(np.float32), diagnostics


def build_regime_specialist_training_frame(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    selected_feature_columns: Sequence[str] | None = None,
    sample_recency_col: str | None = None,
    current_end: Any | None = None,
    similarity_config: RegimeSimilarityConfig = RegimeSimilarityConfig(),
    weight_config: SpecialistWeightConfig = SpecialistWeightConfig(),
    market_columns: Sequence[str] | None = None,
    drift_columns: Sequence[str] | None = None,
    covariance_columns: Sequence[str] | None = None,
    knn_columns: Sequence[str] | None = None,
    asset_return_col: str | None = None,
    include_input_columns: bool = True,
    assessment_frame: pd.DataFrame | None = None,
    unsupervised_regime_artifact: Any | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    similarity, sim_diag = compute_regime_similarity_to_current(
        frame,
        timestamp_col=timestamp_col,
        symbol_col=symbol_col,
        selected_feature_columns=selected_feature_columns,
        current_end=current_end,
        config=similarity_config,
        market_columns=market_columns,
        drift_columns=drift_columns,
        covariance_columns=covariance_columns,
        knn_columns=knn_columns,
        asset_return_col=asset_return_col,
        assessment_frame=assessment_frame,
        unsupervised_regime_artifact=unsupervised_regime_artifact,
    )
    out = frame.copy(deep=False) if bool(include_input_columns) else pd.DataFrame(index=frame.index)
    if unsupervised_regime_artifact is not None:
        try:
            from .unsupervised_regime_learning.regime_models import (
                regime_artifact_assessment_summary,
            )

            _regime_diag = regime_artifact_assessment_summary(unsupervised_regime_artifact)
        except Exception:
            pass
    for col in similarity.columns:
        out[col] = similarity[col].reindex(out.index)
    drift_cols = list(
        ((sim_diag or {}).get("columns", {}) or {}).get("drift", [])
    )
    if not drift_cols:
        try:
            drift_cols = infer_regime_specialist_columns(
                frame,
                selected_feature_columns=selected_feature_columns,
                market_columns=market_columns,
                drift_columns=drift_columns,
                covariance_columns=covariance_columns,
                knn_columns=knn_columns,
                config=similarity_config,
            ).get("drift", [])
        except Exception:
            drift_cols = []
    drift_baseline_frame = frame.reindex(index=similarity.index).copy(deep=False)
    if "current_regime_recency_weight" in similarity.columns:
        drift_baseline_frame["current_regime_recency_weight"] = pd.to_numeric(
            similarity["current_regime_recency_weight"],
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=np.float32)
    drift_baseline = weighted_drift_baseline(
        drift_baseline_frame,
        drift_columns=drift_cols,
    )
    if not bool((sim_diag or {}).get("enabled", False)):
        future = (
            out.get(
                "regime_specialist_bucket",
                pd.Series("", index=out.index),
            )
            .astype(str)
            .str.lower()
            == "future_excluded"
        )
        weights = pd.Series(1.0, index=out.index, name="sample_weight", dtype=np.float32)
        weights.loc[future] = 0.0
        out["regime_specialist_sample_weight"] = weights.to_numpy(dtype=np.float32)
        diag = {
            "schema_version": REGIME_SPECIALIST_SCHEMA_VERSION,
            "similarity": sim_diag,
            "weighted_drift_baseline": drift_baseline,
            "sample_weight": {
                "adaptive_reliability": 0.0,
                "should_train_specialist": False,
                "reason": str((sim_diag or {}).get("reason", "similarity_disabled")),
                "future_excluded_rows": int(future.sum()),
            },
        }
        return out, diag
    recency_col = (
        str(sample_recency_col)
        if sample_recency_col is not None and str(sample_recency_col) in out.columns
        else None
    )
    weights, weight_diag = compute_specialist_sample_weights(
        out,
        bucket_col="regime_specialist_bucket",
        similarity_col="similarity_to_current",
        recency_col=recency_col,
        config=weight_config,
    )
    out["regime_specialist_sample_weight"] = weights.to_numpy(dtype=np.float32)
    diag = {
        "schema_version": REGIME_SPECIALIST_SCHEMA_VERSION,
        "similarity": sim_diag,
        "weighted_drift_baseline": drift_baseline,
        "sample_weight": weight_diag,
    }
    return out, diag


def shrink_self_distillation_towards_one(
    distillation_weight: Sequence[float],
    similarity_to_current: Sequence[float],
    *,
    power: float = 1.0,
    min_similarity: float = 0.0,
) -> np.ndarray:
    weights = np.asarray(distillation_weight, dtype=np.float64)
    sim = np.clip(np.asarray(similarity_to_current, dtype=np.float64), 0.0, 1.0)
    if len(sim) != len(weights):
        raise ValueError("similarity_to_current length must match distillation_weight")
    factor = np.power(np.maximum(sim, float(min_similarity)), float(power))
    adjusted = 1.0 + (weights - 1.0) * factor
    return np.nan_to_num(adjusted, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)


def weighted_drift_baseline(
    frame: pd.DataFrame,
    *,
    drift_columns: Sequence[str],
    weight_col: str = "current_regime_recency_weight",
) -> dict[str, Any]:
    if frame is None or frame.empty or weight_col not in frame.columns:
        return {"enabled": False, "reason": "missing_frame_or_weight_col"}
    numeric = _safe_numeric_frame(frame, drift_columns)
    if numeric.empty:
        return {"enabled": False, "reason": "no_drift_columns"}
    w = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
    if float(np.sum(w)) <= 1e-12:
        return {"enabled": False, "reason": "zero_weight_sum"}
    w = w / float(np.sum(w))
    stats = {}
    for col in numeric.columns:
        vals = numeric[col].to_numpy(dtype=np.float64)
        stats[str(col)] = {
            "weighted_mean": _weighted_mean(vals, w),
            "weighted_median": _weighted_median(vals, w),
        }
    return {
        "enabled": True,
        "weight_col": str(weight_col),
        "weight_sum": float(np.sum(w)),
        "feature_count": int(len(stats)),
        "stats": stats,
    }
