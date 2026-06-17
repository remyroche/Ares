"""Current-regime analogue scoring and specialist sample weights.

This module is intentionally standalone. It computes a current-regime
fingerprint, finds historical analogue windows, assigns
``similarity_to_current`` to rows, and builds sample-weight multipliers for a
shadow current-regime specialist. It does not mutate base/meta training by
itself.
"""

from __future__ import annotations

import math
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
    "base_model",
    "meta_model",
    "psi_ks",
    "prediction_distribution",
    "covariance",
    "contribution",
    "row_drift",
    "raw_state",
    "other",
)

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
    current_window_days: float = 21.0
    candidate_window_days: float = 21.0
    day_window_days: float = 1.0
    recency_decay_per_week: float = 0.67

    drift_weight: float = 0.40
    covariance_weight: float = 0.35
    regime_weight: float = 0.15
    knn_weight: float = 0.10
    ae_weight: float = 0.10
    alpha: float = 1.5
    tau: Optional[float] = None

    analogue_threshold: float = 0.55
    normal_threshold: float = 0.15

    knn_k: int = 25
    max_knn_current_rows: int = 2000
    max_knn_candidate_rows: int = 5000
    max_knn_historical_rows: int = 50000
    max_covariance_features: int = 48
    max_window_diagnostics: int = 50
    top_eigenvalues: int = 5
    asset_return_col: Optional[str] = None

    ae_enabled: bool = True
    ae_min_windows: int = 50
    ae_latent_dim: int = 4
    ae_max_iter: int = 50
    ae_input_noise: float = 0.02
    day_similarity_min_rows: int = 24
    day_similarity_strength: float = 0.50
    random_state: int = 42

    min_candidate_rows: int = 24
    min_current_rows: int = 24
    eps: float = 1e-12


@dataclass
class SpecialistWeightConfig:
    current_gamma: float = 1.0
    analogue_gamma: float = 2.0
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

    replay_min: float = 0.10
    replay_max: float = 0.30

    min_current_plus_analogue_mass: float = 0.70
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
    fit_numeric = _safe_numeric_frame(fit, columns)
    out = pd.DataFrame(index=frame.index)
    if symbol_col not in frame.columns:
        for col in numeric.columns:
            fit_vals = (
                fit_numeric[col].to_numpy(dtype=np.float64)
                if col in fit_numeric.columns
                else numeric[col].to_numpy(dtype=np.float64)
            )
            _fit_z, center, scale = _robust_scale(fit_vals)
            vals = numeric[col].to_numpy(dtype=np.float64)
            filled = np.where(np.isfinite(vals), vals, center)
            out[col] = np.clip((filled - center) / max(scale, 1e-9), -8.0, 8.0).astype(
                np.float32,
            )
        return out.astype(np.float32, copy=False)
    symbols = frame[symbol_col].astype(str)
    fit_symbols = fit[symbol_col].astype(str) if symbol_col in fit.columns else pd.Series("", index=fit.index)
    frame_symbol_groups = {
        str(sym): idx for sym, idx in symbols.groupby(symbols, sort=False).groups.items()
    }
    fit_symbol_groups = {
        str(sym): idx for sym, idx in fit_symbols.groupby(fit_symbols, sort=False).groups.items()
    }
    for col in numeric.columns:
        series = numeric[col]
        fit_series = (
            fit_numeric[col]
            if col in fit_numeric.columns
            else pd.Series(np.nan, index=fit.index, dtype=np.float64)
        )
        _global_z, global_center, global_scale = _robust_scale(
            fit_series.to_numpy(dtype=np.float64),
        )
        vals_all = series.to_numpy(dtype=np.float64)
        z = pd.Series(
            np.clip(
                (np.where(np.isfinite(vals_all), vals_all, global_center) - global_center)
                / max(global_scale, 1e-9),
                -8.0,
                8.0,
            ).astype(np.float32),
            index=frame.index,
            dtype=np.float32,
        )
        for _sym, idx in frame_symbol_groups.items():
            fit_idx = fit_symbol_groups.get(str(_sym), [])
            fit_vals = (
                fit_series.loc[fit_idx].to_numpy(dtype=np.float64)
                if len(fit_idx) > 0
                else np.asarray([], dtype=np.float64)
            )
            finite_fit = np.isfinite(fit_vals)
            if int(np.sum(finite_fit)) < int(min_symbol_fit_rows):
                continue
            center = float(np.nanmedian(fit_vals[finite_fit]))
            mad = float(np.nanmedian(np.abs(fit_vals[finite_fit] - center)))
            scale = 1.4826 * mad
            if not np.isfinite(scale) or scale <= 1e-9:
                scale = float(np.nanstd(fit_vals[finite_fit]))
            if not np.isfinite(scale) or scale <= 1e-9:
                scale = global_scale
                center = global_center
            vals = series.loc[idx].to_numpy(dtype=np.float64)
            finite = np.isfinite(vals)
            z.loc[idx] = np.clip(
                (np.where(finite, vals, center) - center) / max(scale, 1e-9),
                -8.0,
                8.0,
            ).astype(np.float32)
        out[col] = z.to_numpy(dtype=np.float32, copy=False)
    return out.astype(np.float32, copy=False)


def _matches_any_token(name: str, tokens: Sequence[str]) -> bool:
    low = str(name).lower()
    return any(str(token).lower() in low for token in tokens)


def _is_excluded_feature(name: str) -> bool:
    low = str(name).lower()
    return low in {"timestamp", "symbol", "asset", "strategy_id", "side"} or _matches_any_token(
        low, DEFAULT_EXCLUDE_TOKENS
    )


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
    cols: list[str] = []
    arrays: list[np.ndarray] = []
    missing_arrays: list[np.ndarray] = []
    for col in columns:
        col_s = str(col)
        if col_s not in frame.columns:
            continue
        vals = pd.to_numeric(frame[col_s], errors="coerce").replace([np.inf, -np.inf], np.nan)
        arr = vals.to_numpy(dtype=np.float32, copy=False)
        missing = ~np.isfinite(arr)
        if scaler is not None:
            if col_s not in scaler:
                continue
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
        arrays.append(arr.astype(np.float32, copy=False))
        missing_arrays.append(missing.astype(bool, copy=False))
    if not arrays:
        return [], np.zeros((len(frame), 0), dtype=np.float32), np.zeros((len(frame), 0), dtype=bool)
    return (
        cols,
        np.column_stack(arrays).astype(np.float32, copy=False),
        np.column_stack(missing_arrays).astype(bool, copy=False),
    )


def _matrix_from_frame(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[list[str], np.ndarray]:
    cols = [str(c) for c in columns if str(c) in frame.columns]
    if not cols:
        return [], np.zeros((len(frame), 0), dtype=np.float32)
    return cols, frame.loc[:, cols].to_numpy(dtype=np.float32, copy=True)


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
    time_ns = _timestamp_ns(pd.Series(pivot.index))
    order = np.argsort(time_ns, kind="mergesort")
    time_ns = time_ns[order]
    matrix = pivot.to_numpy(dtype=np.float32, copy=True)[order]
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
    if low.startswith("base_lgbm_") or low.startswith("base_") or "base_model" in low:
        return "base_model"
    if low.startswith("meta_lgbm_") or low.startswith("meta_") or "meta_model" in low:
        return "meta_model"
    if "psi" in low or "_ks" in low or "ks_" in low or "wasserstein" in low:
        return "psi_ks"
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
        return np.zeros(11 + len(DRIFT_FAMILY_ORDER) * 5, dtype=np.float32)
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
        return np.zeros(11 + len(DRIFT_FAMILY_ORDER) * 5, dtype=np.float32)
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
        return np.zeros(11 + len(DRIFT_FAMILY_ORDER) * 5, dtype=np.float32)
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
        return np.zeros(11 + len(DRIFT_FAMILY_ORDER) * 5, dtype=np.float32)
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
        for start in range(0, len(cur), 256):
            block = cur[start : start + 256]
            d = np.sqrt(np.maximum(((block[:, None, :] - cand[None, :, :]) ** 2).mean(axis=2), 0.0))
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
    max_historical_rows: int,
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
    hist_pos = _subsample_positions(historical_pos, max_rows=int(max_historical_rows))
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
        for start in range(0, len(cur), 256):
            block = cur[start : start + 256]
            d = np.sqrt(np.maximum(((block[:, None, :] - hist[None, :, :]) ** 2).mean(axis=2), 0.0))
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
    for j in range(all_x.shape[1]):
        all_x[:, j], _center, _scale = _robust_scale(all_x[:, j])
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPRegressor

        rng = np.random.default_rng(int(config.random_state))
        noisy = np.clip(
            all_x + rng.normal(0.0, float(config.ae_input_noise), size=all_x.shape).astype(np.float32),
            -8.0,
            8.0,
        )
        model = MLPRegressor(
            hidden_layer_sizes=(64, 16, int(config.ae_latent_dim), 16, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=max(1, int(config.ae_max_iter)),
            batch_size=min(1024, max(1, len(all_x))),
            random_state=int(config.random_state),
            early_stopping=False,
            verbose=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(noisy, all_x)
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

            pca = PCA(n_components=min(int(config.ae_latent_dim), all_x.shape[0], all_x.shape[1]))
            latent = pca.fit_transform(all_x).astype(np.float32)
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
        "similarity_to_current",
        "current_regime_recency_weight",
    ):
        if col not in out.columns:
            out[col] = np.nan
    if "regime_specialist_bucket" not in out.columns:
        out["regime_specialist_bucket"] = "irrelevant"
    if len(future_index) > 0:
        out.loc[future_index, "window_similarity"] = 0.0
        out.loc[future_index, "day_similarity"] = 0.0
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
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if frame is None or frame.empty:
        out = pd.DataFrame(index=getattr(frame, "index", None))
        return out, {
            "schema_version": REGIME_SPECIALIST_SCHEMA_VERSION,
            "enabled": False,
            "reason": "empty_frame",
        }
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
    if not bool(asof_mask.any()):
        out = pd.DataFrame(index=original_index)
        out["similarity_to_current"] = 0.0
        out["window_similarity"] = 0.0
        out["day_similarity"] = 0.0
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
            },
        )
    work = frame.loc[asof_mask].copy()
    ts = full_ts.loc[asof_mask]
    valid_ts = ts.notna()
    if not bool(valid_ts.any()):
        out = pd.DataFrame(
            {
                "similarity_to_current": np.ones(len(work), dtype=np.float32),
                "regime_specialist_bucket": np.repeat("current", len(work)),
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
            },
        )
    current_start = end - pd.Timedelta(days=float(config.current_window_days))
    current_mask = (ts >= current_start) & (ts <= end)
    if int(current_mask.sum()) < int(config.min_current_rows):
        out = pd.DataFrame(
            {
                "similarity_to_current": np.ones(len(work), dtype=np.float32),
                "regime_specialist_bucket": np.where(current_mask, "current", "normal"),
                "window_similarity": np.ones(len(work), dtype=np.float32),
                "day_similarity": np.ones(len(work), dtype=np.float32),
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
                "current_rows": int(current_mask.sum()),
                "min_current_rows": int(config.min_current_rows),
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
    historical_mask = valid_ts & (ts < current_start)
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
    out["day_similarity"] = np.where(current_mask, 1.0, 1.0).astype(np.float32)
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
    asset_return_cache = _build_asset_return_cache(
        work,
        return_col=resolved_asset_return_col,
        timestamp_col=timestamp_col,
        symbol_col=symbol_col,
    )
    current_market = _market_fingerprint_array(
        market_arr,
        market_missing,
        timestamp_ns,
        current_pos,
        current_weights,
    )
    current_feature_cov = _covariance_fingerprint_array(
        cov_arr,
        current_pos,
        weights=current_weights,
        top_eigenvalues=int(config.top_eigenvalues),
        eps=float(config.eps),
    )
    current_asset_cov = _asset_covariance_fingerprint_from_cache(
        asset_return_cache,
        current_pos,
        current_weights,
        top_eigenvalues=int(config.top_eigenvalues),
        eps=float(config.eps),
    )
    current_cov = np.concatenate([current_feature_cov, current_asset_cov]).astype(np.float32)
    current_drift = _drift_fingerprint_array(
        drift_arr,
        drift_missing,
        drift_families,
        timestamp_ns,
        current_pos,
        current_weights,
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
        max_historical_rows=int(config.max_knn_historical_rows),
        eps=float(config.eps),
    )
    window_rows: list[dict[str, Any]] = []
    fingerprints: list[np.ndarray] = []
    for window_id in sorted(set(window_id_full[window_id_full >= 0])):
        pos = np.flatnonzero(window_id_full == int(window_id))
        if len(pos) < int(config.min_candidate_rows):
            continue
        weights = _position_recency_weights(
            timestamp_ns,
            pos,
            current_end_ns=None,
            decay_per_week=float(config.recency_decay_per_week),
            eps=float(config.eps),
        )
        market_fp = _market_fingerprint_array(
            market_arr,
            market_missing,
            timestamp_ns,
            pos,
            weights,
        )
        feature_cov_fp = _covariance_fingerprint_array(
            cov_arr,
            pos,
            weights=weights,
            top_eigenvalues=int(config.top_eigenvalues),
            eps=float(config.eps),
        )
        asset_cov_fp = _asset_covariance_fingerprint_from_cache(
            asset_return_cache,
            pos,
            weights=weights,
            top_eigenvalues=int(config.top_eigenvalues),
            eps=float(config.eps),
        )
        cov_fp = np.concatenate([feature_cov_fp, asset_cov_fp]).astype(np.float32)
        drift_fp = _drift_fingerprint_array(
            drift_arr,
            drift_missing,
            drift_families,
            timestamp_ns,
            pos,
            weights,
            eps=float(config.eps),
        )
        knn_distance = knn_distance_map.get(int(window_id), float(knn_distance_fallback))
        row = {
            "window_id": int(window_id),
            "start": str(ts.iloc[pos].min()),
            "end": str(ts.iloc[pos].max()),
            "rows": int(len(pos)),
            "d_regime": _euclidean(market_fp, current_market),
            "d_cov": _euclidean(cov_fp, current_cov),
            "d_drift": _euclidean(drift_fp, current_drift),
            "d_knn": float(knn_distance),
        }
        window_rows.append(row)
        fingerprints.append(_concat_fingerprint(market_fp, cov_fp, drift_fp))

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
                "columns": columns,
                "column_selection": column_selection,
            },
        )

    d_regime_norm, d_regime_scale = _normalise_distances_with_scale(
        np.asarray([r["d_regime"] for r in window_rows]),
        float(config.eps),
    )
    d_cov_norm, d_cov_scale = _normalise_distances_with_scale(
        np.asarray([r["d_cov"] for r in window_rows]),
        float(config.eps),
    )
    d_drift_norm, d_drift_scale = _normalise_distances_with_scale(
        np.asarray([r["d_drift"] for r in window_rows]),
        float(config.eps),
    )
    d_knn_norm, d_knn_scale = _normalise_distances_with_scale(
        np.asarray([r["d_knn"] for r in window_rows]),
        float(config.eps),
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
    analogue_quality = np.clip(base_similarity + float(config.ae_weight) * sim_ae, 0.0, 1.0)
    window_similarity_map: dict[int, float] = {}
    for i, row in enumerate(window_rows):
        row.update(
            {
                "d_regime_norm": float(d_regime_norm[i]),
                "d_cov_norm": float(d_cov_norm[i]),
                "d_drift_norm": float(d_drift_norm[i]),
                "d_knn_norm": float(d_knn_norm[i]),
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
    out.loc[day_similarity.index, "day_similarity"] = day_similarity.to_numpy(dtype=np.float32)
    hist_idx = historical_mask.to_numpy(dtype=bool)
    day_strength = float(np.clip(config.day_similarity_strength, 0.0, 1.0))
    day_multiplier = (
        (1.0 - day_strength)
        + day_strength * out["day_similarity"].to_numpy(dtype=np.float32)
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
        "current_rows": int(current_mask.sum()),
        "historical_rows": int(historical_mask.sum()),
        "candidate_window_count": int(len(window_rows)),
        "columns": columns,
        "column_selection": column_selection,
        "scaling": {
            "source": scaling_source,
            "fit_rows": int(len(scale_fit_frame)),
            "market_columns_scaled": int(len(market_cols)),
            "covariance_columns_scaled": int(len(cov_cols)),
            "knn_columns_scaled": int(len(knn_scaler)),
        },
        "asset_covariance": {
            "enabled": bool(asset_return_cache.enabled),
            "return_col": asset_return_cache.return_col,
            "feature_count": int(len(current_asset_cov)),
            "time_rows": int(len(asset_return_cache.time_ns)),
            "asset_count": int(asset_return_cache.matrix.shape[1]) if asset_return_cache.matrix.ndim == 2 else 0,
        },
        "knn": knn_diagnostics,
        "autoencoder": {
            "enabled": bool(config.ae_enabled),
            "used": bool(ae_used),
            "reason": ae_reason,
            "min_windows": int(config.ae_min_windows),
        },
        "day_similarity": {
            "strength": day_strength,
            "min_rows": int(config.day_similarity_min_rows),
        },
        "weights": {
            "feature_drift_distance": float(config.drift_weight),
            "covariance_distance": float(config.covariance_weight),
            "regime_state_distance": float(config.regime_weight),
            "knn_distance": float(config.knn_weight),
            "ae_similarity": float(config.ae_weight),
        },
        "block_scaling": {
            "combined_from_normalized_distances": True,
            "regime_distance_median": float(d_regime_scale),
            "covariance_distance_median": float(d_cov_scale),
            "drift_distance_median": float(d_drift_scale),
            "knn_distance_median": float(d_knn_scale),
            "tau": float(tau),
            "alpha": float(config.alpha),
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
    out = pd.Series(1.0, index=work.index, dtype=np.float32)
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
    rows = []
    ids = []
    for day_id in sorted(set(day_ids[day_ids >= 0])):
        pos = np.flatnonzero(day_ids == int(day_id))
        if len(pos) < int(config.day_similarity_min_rows):
            continue
        weights = _position_recency_weights(
            timestamp_ns,
            pos,
            current_end_ns=None,
            decay_per_week=float(config.recency_decay_per_week),
            eps=float(config.eps),
        )
        market_fp = _market_fingerprint_array(
            market_arr,
            market_missing,
            timestamp_ns,
            pos,
            weights,
        )
        feature_cov_fp = _covariance_fingerprint_array(
            cov_arr,
            pos,
            weights=weights,
            top_eigenvalues=int(config.top_eigenvalues),
            eps=float(config.eps),
        )
        asset_cov_fp = _asset_covariance_fingerprint_from_cache(
            asset_return_cache,
            pos,
            weights=weights,
            top_eigenvalues=int(config.top_eigenvalues),
            eps=float(config.eps),
        )
        cov_fp = np.concatenate([feature_cov_fp, asset_cov_fp]).astype(np.float32)
        drift_fp = _drift_fingerprint_array(
            drift_arr,
            drift_missing,
            drift_families,
            timestamp_ns,
            pos,
            weights,
            eps=float(config.eps),
        )
        rows.append(
            [
                _euclidean(market_fp, current_market),
                _euclidean(cov_fp, current_cov),
                _euclidean(drift_fp, current_drift),
            ]
        )
        ids.append(int(day_id))
    if not rows:
        return out
    arr = np.asarray(rows, dtype=np.float64)
    regime_n = _normalise_distances(arr[:, 0], float(config.eps))
    cov_n = _normalise_distances(arr[:, 1], float(config.eps))
    drift_n = _normalise_distances(arr[:, 2], float(config.eps))
    d = (
        float(config.drift_weight) * drift_n
        + float(config.covariance_weight) * cov_n
        + float(config.regime_weight) * regime_n
    )
    sim = np.exp(-np.power(np.maximum(d, 0.0), float(config.alpha)))
    sim_by_day = {day_id: float(np.clip(value, 0.0, 1.0)) for day_id, value in zip(ids, sim)}
    for day_id, value in sim_by_day.items():
        out.iloc[np.flatnonzero(day_ids == int(day_id))] = float(value)
    return out.astype(np.float32, copy=False)


def _saturating_reliability(effective_count: float, tau: float) -> float:
    effective_count = max(float(effective_count), 0.0)
    tau = max(float(tau), 1e-12)
    return 1.0 - math.exp(-effective_count / tau)


def _cap_bucket_masses(
    bucket_mass: Dict[str, float],
    config: SpecialistWeightConfig,
) -> Dict[str, float]:
    mass = dict(bucket_mass)
    if mass["irrelevant"] > config.max_irrelevant_mass:
        excess = mass["irrelevant"] - config.max_irrelevant_mass
        mass["irrelevant"] = config.max_irrelevant_mass
        mass["normal"] += excess
    if mass["normal"] > config.max_normal_mass:
        excess = mass["normal"] - config.max_normal_mass
        mass["normal"] = config.max_normal_mass
        adaptive_total = mass["current"] + mass["analogue"]
        if adaptive_total > config.eps:
            mass["current"] += excess * mass["current"] / adaptive_total
            mass["analogue"] += excess * mass["analogue"] / adaptive_total
        else:
            mass["normal"] += excess
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
    total = sum(mass.values())
    if total > config.eps:
        mass = {k: v / total for k, v in mass.items()}
    return mass


def compute_specialist_sample_weights(
    df: pd.DataFrame,
    bucket_col: str = "regime_specialist_bucket",
    similarity_col: str = "similarity_to_current",
    recency_col: Optional[str] = None,
    config: SpecialistWeightConfig = SpecialistWeightConfig(),
) -> Tuple[pd.Series, Dict[str, float]]:
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
            "future_excluded_rows": int(excluded.sum()),
        }
    sim = df[similarity_col].astype(float).clip(0.0, 1.0)
    if recency_col is None:
        recency = pd.Series(1.0, index=df.index)
    else:
        recency = df[recency_col].astype(float).clip(lower=0.0)
    masks = {
        "current": buckets == "current",
        "analogue": buckets == "analogue",
        "normal": buckets == "normal",
        "irrelevant": buckets == "irrelevant",
    }
    row_score = pd.Series(0.0, index=df.index)
    row_score.loc[masks["current"]] = sim.loc[masks["current"]].pow(config.current_gamma) * recency.loc[masks["current"]]
    row_score.loc[masks["analogue"]] = sim.loc[masks["analogue"]].pow(config.analogue_gamma) * recency.loc[masks["analogue"]]
    row_score.loc[masks["normal"]] = sim.loc[masks["normal"]].pow(config.normal_gamma) * recency.loc[masks["normal"]]
    row_score.loc[masks["irrelevant"]] = sim.loc[masks["irrelevant"]].pow(config.irrelevant_gamma) * recency.loc[masks["irrelevant"]]
    row_score = row_score.clip(lower=config.eps)
    eff = {name: float(row_score.loc[mask].sum()) for name, mask in masks.items()}
    reliability = {
        "current": _saturating_reliability(eff["current"], config.tau_current),
        "analogue": _saturating_reliability(eff["analogue"], config.tau_analogue),
        "normal": _saturating_reliability(eff["normal"], config.tau_normal),
        "irrelevant": _saturating_reliability(eff["irrelevant"], config.tau_irrelevant),
    }
    adaptive_reliability = 1.0 - (1.0 - reliability["current"]) * (1.0 - reliability["analogue"])
    replay_strength = config.replay_min + (config.replay_max - config.replay_min) * (1.0 - adaptive_reliability)
    adaptive_strength = 1.0 - replay_strength
    bucket_score = {
        "current": config.current_prior * adaptive_strength * reliability["current"],
        "analogue": config.analogue_prior * adaptive_strength * reliability["analogue"],
        "normal": config.normal_prior * replay_strength * reliability["normal"],
        "irrelevant": config.irrelevant_prior * replay_strength * reliability["irrelevant"],
    }
    for name, mask in masks.items():
        if not bool(mask.any()):
            bucket_score[name] = 0.0
    total_score = sum(bucket_score.values())
    if total_score <= config.eps:
        weights = pd.Series(1.0, index=df.index, name="sample_weight")
        diagnostics = {
            "adaptive_reliability": 0.0,
            "should_train_specialist": False,
            "current_mass": 0.0,
            "analogue_mass": 0.0,
            "normal_mass": 0.0,
            "irrelevant_mass": 0.0,
        }
        return weights, diagnostics
    bucket_mass = {k: v / total_score for k, v in bucket_score.items()}
    bucket_mass = _cap_bucket_masses(bucket_mass, config)
    raw_weight = pd.Series(0.0, index=df.index)
    for bucket_name, mask in masks.items():
        if not bool(mask.any()):
            continue
        score_sum = float(row_score.loc[mask].sum())
        raw_weight.loc[mask] = bucket_mass[bucket_name] * row_score.loc[mask] / max(score_sum, config.eps)
    weights = raw_weight * int(active.sum())
    weights.loc[active] = weights.loc[active].clip(config.min_weight, config.max_weight)
    weights.loc[excluded] = 0.0
    weights.loc[active] = weights.loc[active] / max(
        float(weights.loc[active].mean()),
        config.eps,
    )
    weights = weights.rename("sample_weight")
    diagnostics = {
        "adaptive_reliability": adaptive_reliability,
        "should_train_specialist": adaptive_reliability >= config.min_adaptive_reliability_to_train,
        "current_mass": bucket_mass["current"],
        "analogue_mass": bucket_mass["analogue"],
        "normal_mass": bucket_mass["normal"],
        "irrelevant_mass": bucket_mass["irrelevant"],
        "effective_current": eff["current"],
        "effective_analogue": eff["analogue"],
        "effective_normal": eff["normal"],
        "effective_irrelevant": eff["irrelevant"],
        "future_excluded_rows": int(excluded.sum()),
    }
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
    )
    out = frame.copy(deep=False) if bool(include_input_columns) else pd.DataFrame(index=frame.index)
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
