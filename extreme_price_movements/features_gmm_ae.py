from __future__ import annotations

import math
import os
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

try:  # pragma: no cover - fallback is exercised only when numba is unavailable.
    from numba import njit as _numba_njit
except Exception:  # pragma: no cover
    _numba_njit = None

from .features_denoising_ae import (
    fit_denoising_autoencoder_state,
    transform_denoising_autoencoder_features,
)


def _env_int_tuple(name: str, default: Sequence[int], *, min_value: int, max_value: int) -> tuple[int, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return tuple(int(v) for v in default)
    out: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError:
            continue
        if int(min_value) <= value <= int(max_value) and value not in out:
            out.append(value)
    return tuple(out) if out else tuple(int(v) for v in default)


def _env_float_tuple(
    name: str,
    default: Sequence[float],
    *,
    min_value: float,
    max_value: float,
) -> tuple[float, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return tuple(float(v) for v in default)
    out: list[float] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            continue
        if np.isfinite(value) and float(min_value) <= value <= float(max_value) and value not in out:
            out.append(value)
    return tuple(out) if out else tuple(float(v) for v in default)


AE_GMM_MAX_COMPONENTS = 7
AE_GMM_CLUSTER_CANDIDATES = _env_int_tuple(
    "EPM_AE_GMM_CLUSTER_CANDIDATES",
    (2, 3, 4, 5, 6),
    min_value=2,
    max_value=AE_GMM_MAX_COMPONENTS,
)
AE_GMM_REG_COVAR_CANDIDATES = _env_float_tuple(
    "EPM_AE_GMM_REG_COVAR_CANDIDATES",
    (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3),
    min_value=1e-8,
    max_value=1.0,
)
AE_GMM_SMOOTH_LAMBDA_CANDIDATES = _env_float_tuple(
    "EPM_AE_GMM_SMOOTH_LAMBDA_CANDIDATES",
    (0.50, 0.80, 0.925, 0.97),
    min_value=0.0,
    max_value=0.999,
)
AE_GMM_SMOOTH_LAMBDA = 0.925
AE_GMM_LATENT_DIM = 16
AE_GMM_MIN_OCCUPANCY = 0.03
AE_GMM_MAX_OCCUPANCY = 0.75
AE_GMM_PATH_AWARE_HPO = os.environ.get("EPM_AE_GMM_PATH_AWARE_HPO", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "n",
    "off",
}
AE_GMM_TEMPORAL_CONCENTRATION_HPO = os.environ.get(
    "EPM_AE_GMM_TEMPORAL_CONCENTRATION_HPO",
    "1",
).strip().lower() not in {
    "0",
    "false",
    "no",
    "n",
    "off",
}

AE_GMM_LATENT_FEATURE_COLUMNS = tuple(
    f"dae_b16_{i:02d}" for i in range(AE_GMM_LATENT_DIM)
)
AE_GMM_CLUSTER_FEATURE_COLUMNS = tuple(
    [f"gmm_prob_{i}" for i in range(AE_GMM_MAX_COMPONENTS)]
    + [f"gmm_cluster_posterior_{i}" for i in range(AE_GMM_MAX_COMPONENTS)]
    + [f"gmm_dist_center_{i}" for i in range(AE_GMM_MAX_COMPONENTS)]
    + [f"gmm_mahal_{i}" for i in range(AE_GMM_MAX_COMPONENTS)]
    + [
        "gmm_cluster_id",
        "gmm_posterior_max",
        "gmm_posterior_margin",
        "gmm_posterior_delta_1",
        "gmm_posterior_accel_1",
        "gmm_entropy",
        "cluster_entropy",
        "cluster_entropy_norm",
        "cluster_entropy_delta_1",
        "cluster_entropy_accel_1",
        "mahalanobis_distance",
        "min_mahalanobis",
        "min_mahalanobis_delta_1",
        "expected_mahalanobis",
        "expected_mahalanobis_delta_1",
        "expected_mahalanobis_accel_1",
        "cluster_t",
        "cluster_speed",
        "cluster_acceleration",
        "time_since_cluster_change",
        "rolling_cluster_stability",
        "cluster_flip_count_20",
        "AE_reconstruction_error",
        "ae_reconstruction_error",
        "dae_reconstruction_error",
        "dae_reconstruction_error_zscore",
        "dae_reconstruction_error_delta_1",
        "dae_reconstruction_error_accel_1",
        "latent_mahalanobis_drift",
        "latent_speed",
        "latent_acceleration",
    ]
)
AE_GMM_FEATURE_COLUMNS = AE_GMM_LATENT_FEATURE_COLUMNS + AE_GMM_CLUSTER_FEATURE_COLUMNS


def ae_gmm_feature_columns(prefix: str = "") -> list[str]:
    return [f"{prefix}{name}" for name in AE_GMM_FEATURE_COLUMNS]


def _as_float_frame(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        frame = x.copy()
    else:
        frame = pd.DataFrame(x)
    frame.columns = [str(c) for c in frame.columns]
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan)


def _robust_scale_fit(x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    arr = x.to_numpy(dtype=np.float32, copy=True)
    med = np.nanmedian(arr, axis=0).astype(np.float32)
    q25 = np.nanpercentile(arr, 25.0, axis=0).astype(np.float32)
    q75 = np.nanpercentile(arr, 75.0, axis=0).astype(np.float32)
    scale = (q75 - q25).astype(np.float32)
    fallback = np.nanstd(arr, axis=0).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    return med, scale


def _robust_scale_apply(x: pd.DataFrame, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    arr = x.to_numpy(dtype=np.float32, copy=True)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    out = (arr - center.reshape(1, -1)) / scale.reshape(1, -1)
    return np.clip(out, -8.0, 8.0).astype(np.float32)


def _softmax_log(logp: np.ndarray) -> np.ndarray:
    z = logp - np.nanmax(logp, axis=1, keepdims=True)
    p = np.exp(np.clip(z, -80.0, 0.0))
    denom = np.sum(p, axis=1, keepdims=True)
    denom = np.where(denom > 0.0, denom, 1.0)
    return (p / denom).astype(np.float32)


def _diag_gmm_predict_proba(z: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    means = np.asarray(state.get("gmm_means", []), dtype=np.float32)
    covars = np.asarray(state.get("gmm_covariances", []), dtype=np.float32)
    weights = np.asarray(state.get("gmm_weights", []), dtype=np.float32)
    if means.ndim != 2 or covars.ndim != 2 or len(means) == 0:
        return np.zeros((len(z), 0), dtype=np.float32)
    covars = np.maximum(covars, 1e-8)
    weights = np.maximum(weights, 1e-12)
    diff = z[:, None, :] - means[None, :, :]
    quad = np.sum((diff * diff) / covars[None, :, :], axis=2)
    log_det = np.sum(np.log(covars), axis=1)
    dim = float(means.shape[1])
    logp = (
        np.log(weights)[None, :]
        - 0.5 * (quad + log_det[None, :] + dim * np.log(2.0 * np.pi))
    )
    return _softmax_log(logp)


def _gmm_distances(z: np.ndarray, state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(state.get("gmm_means", []), dtype=np.float32)
    covars = np.asarray(state.get("gmm_covariances", []), dtype=np.float32)
    if means.ndim != 2 or covars.ndim != 2 or len(means) == 0:
        empty = np.zeros((len(z), 0), dtype=np.float32)
        return empty, empty
    diff = z[:, None, :] - means[None, :, :]
    dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0)).astype(np.float32)
    covars = np.maximum(covars, 1e-8)
    mahal = np.sqrt(
        np.maximum(np.sum((diff * diff) / covars[None, :, :], axis=2), 0.0)
    ).astype(np.float32)
    return dist, mahal


if _numba_njit is not None:

    @_numba_njit(cache=True)
    def _smooth_probabilities_numba(prob_arr: np.ndarray, lam32: np.float32) -> np.ndarray:
        n_rows = prob_arr.shape[0]
        n_cols = prob_arr.shape[1] if prob_arr.ndim == 2 else 0
        out = np.empty_like(prob_arr, dtype=np.float32)
        if n_rows == 0 or n_cols == 0:
            return out
        prev = np.empty(n_cols, dtype=np.float32)
        for j in range(n_cols):
            prev[j] = prob_arr[0, j]
            out[0, j] = prev[j]
        one_minus = np.float32(1.0) - lam32
        for i in range(1, n_rows):
            total = np.float32(0.0)
            for j in range(n_cols):
                prev[j] = prev[j] * lam32 + one_minus * prob_arr[i, j]
                total += prev[j]
            if total > np.float32(0.0):
                inv_total = np.float32(1.0) / total
                for j in range(n_cols):
                    prev[j] *= inv_total
                    out[i, j] = prev[j]
            else:
                for j in range(n_cols):
                    out[i, j] = prev[j]
        return out

else:
    _smooth_probabilities_numba = None


def _smooth_probabilities_python(prob_arr: np.ndarray, lam: float) -> np.ndarray:
    prob_arr = np.asarray(prob_arr, dtype=np.float32)
    if len(prob_arr) == 0:
        return prob_arr
    out = np.empty_like(prob_arr, dtype=np.float32)
    prev = prob_arr[0].copy()
    out[0] = prev
    lam32 = np.float32(lam)
    one_minus = np.float32(1.0 - float(lam))
    for i in range(1, len(prob_arr)):
        prev *= lam32
        prev += one_minus * prob_arr[i]
        total = float(prev.sum(dtype=np.float32))
        if total > 0.0:
            prev /= np.float32(total)
        out[i] = prev
    return out


def _smooth_probabilities(prob: np.ndarray, lam: float) -> np.ndarray:
    prob_arr = np.asarray(prob, dtype=np.float32)
    if len(prob_arr) == 0:
        return prob_arr
    if _smooth_probabilities_numba is not None and prob_arr.ndim == 2:
        return _smooth_probabilities_numba(prob_arr, np.float32(lam))
    return _smooth_probabilities_python(prob_arr, lam)


def _cluster_stability(labels: np.ndarray, window: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(labels))
    if n <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty
    labels_arr = np.asarray(labels)
    window_i = max(int(window), 1)
    idx = np.arange(n, dtype=np.int64)

    change = np.zeros(n, dtype=bool)
    if n > 1:
        change[1:] = labels_arr[1:] != labels_arr[:-1]
    last_change_idx = np.where(change, idx, 0)
    last_change_idx = np.maximum.accumulate(last_change_idx)
    age = (idx - last_change_idx).astype(np.float32)

    _unique, inv = np.unique(labels_arr, return_inverse=True)
    one_hot = np.eye(int(inv.max()) + 1, dtype=np.int16)[inv]
    csum = np.vstack(
        [
            np.zeros((1, one_hot.shape[1]), dtype=np.int32),
            np.cumsum(one_hot, axis=0, dtype=np.int32),
        ]
    )
    starts = np.maximum(0, idx - window_i + 1)
    counts = csum[idx + 1, inv] - csum[starts, inv]
    win_len = (idx - starts + 1).astype(np.float32)
    stability = (counts.astype(np.float32) / np.maximum(win_len, 1.0)).astype(np.float32)

    flip_step = np.zeros(n, dtype=np.int16)
    if n > 1:
        flip_step[1:] = change[1:].astype(np.int16)
    flip_csum = np.concatenate(
        [
            np.zeros(1, dtype=np.int32),
            np.cumsum(flip_step, dtype=np.int32),
        ]
    )
    flips = (flip_csum[idx + 1] - flip_csum[starts + 1]).astype(np.float32)
    return age, stability, flips


def _economic_separation(labels: np.ndarray, targets: dict[str, Any] | None) -> float:
    if not targets:
        return 0.0
    vals: list[float] = []
    for name, arr_like in targets.items():
        if str(name) in {"side", "time_bucket", "month", "__time_bucket__"} or str(name).startswith("_"):
            continue
        arr = np.asarray(arr_like, dtype=np.float32)
        if len(arr) != len(labels):
            continue
        if not np.isfinite(arr).any():
            continue
        scale = float(np.nanstd(arr) + 1e-6)
        means = []
        for k in np.unique(labels):
            mask = labels == k
            if int(np.sum(mask)) < 10:
                continue
            means.append(float(np.nanmean(arr[mask])))
        if len(means) >= 2:
            vals.append(
                float(
                    (np.nanpercentile(means, 90.0) - np.nanpercentile(means, 10.0))
                    / scale
                )
            )
    if not vals:
        return 0.0
    return float(np.nanmean(vals))


def _cluster_target_signature_score(
    labels: np.ndarray,
    targets: dict[str, Any] | None,
) -> dict[str, Any]:
    """Score whether cluster target signs are stable across early/mid/late slices."""
    if not targets or len(labels) == 0:
        return {
            "score": 0.0,
            "sign_stability": 0.0,
            "contrast": 0.0,
            "target_count": 0,
            "rows": [],
        }
    labels_arr = np.asarray(labels)
    n = int(len(labels_arr))
    thirds = [np.asarray(b, dtype=np.int32) for b in np.array_split(np.arange(n, dtype=np.int32), min(3, n))]
    rows: list[dict[str, Any]] = []
    target_scores: list[float] = []
    sign_scores: list[float] = []
    contrast_scores: list[float] = []
    for name, arr_like in targets.items():
        if str(name) in {"side", "time_bucket", "month", "__time_bucket__"} or str(name).startswith("_"):
            continue
        arr = np.asarray(arr_like, dtype=np.float32)
        if len(arr) != n:
            continue
        finite = np.isfinite(arr)
        if int(np.sum(finite)) < 50:
            continue
        scale = float(np.nanstd(arr[finite]) + 1e-6)
        global_mean = float(np.nanmean(arr[finite]))
        cluster_means: list[float] = []
        cluster_rows: list[dict[str, Any]] = []
        stable_votes = 0
        total_votes = 0
        for cluster in sorted(int(v) for v in np.unique(labels_arr)):
            mask = (labels_arr == cluster) & finite
            count = int(np.sum(mask))
            if count < 10:
                continue
            full_delta = float(np.nanmean(arr[mask]) - global_mean)
            full_sign = 1 if full_delta >= 0.0 else -1
            cluster_means.append(full_delta)
            slice_signs: list[int] = []
            slice_counts: list[int] = []
            for band in thirds:
                if len(band) == 0:
                    continue
                band_mask = mask[band]
                band_count = int(np.sum(band_mask))
                slice_counts.append(band_count)
                if band_count < 5:
                    continue
                band_delta = float(np.nanmean(arr[band][band_mask]) - global_mean)
                slice_sign = 1 if band_delta >= 0.0 else -1
                slice_signs.append(slice_sign)
                stable_votes += int(slice_sign == full_sign)
                total_votes += 1
            cluster_rows.append(
                {
                    "target": str(name),
                    "cluster": int(cluster),
                    "rows": count,
                    "delta": full_delta,
                    "sign": int(full_sign),
                    "time_slice_counts": slice_counts,
                    "time_slice_signs": slice_signs,
                }
            )
        if len(cluster_means) < 2:
            continue
        contrast = float(
            (np.nanpercentile(cluster_means, 90.0) - np.nanpercentile(cluster_means, 10.0))
            / scale
        )
        sign_stability = float(stable_votes / max(total_votes, 1))
        target_score = float(np.clip(0.60 * sign_stability + 0.40 * np.tanh(max(contrast, 0.0)), 0.0, 1.0))
        rows.extend(cluster_rows)
        sign_scores.append(sign_stability)
        contrast_scores.append(contrast)
        target_scores.append(target_score)
    if not target_scores:
        return {
            "score": 0.0,
            "sign_stability": 0.0,
            "contrast": 0.0,
            "target_count": 0,
            "rows": [],
        }
    return {
        "score": float(np.nanmean(target_scores)),
        "sign_stability": float(np.nanmean(sign_scores)),
        "contrast": float(np.nanmean(contrast_scores)),
        "target_count": int(len(target_scores)),
        "rows": rows[:60],
    }


def _first_finite_target(
    targets: dict[str, Any] | None,
    names: tuple[str, ...],
    n: int,
) -> np.ndarray | None:
    if not targets:
        return None
    for name in names:
        if name not in targets:
            continue
        arr = np.asarray(targets.get(name), dtype=np.float32)
        if len(arr) != n:
            continue
        if np.isfinite(arr).any():
            return arr
    return None


def _path_cleanliness_separation(
    labels: np.ndarray,
    targets: dict[str, Any] | None,
) -> dict[str, Any]:
    """Score whether clusters separate clean executable positives from dirty ones."""
    if not targets or len(labels) == 0:
        return {
            "score": 0.0,
            "clean_positive_contrast": 0.0,
            "dirty_positive_contrast": 0.0,
            "bad_mae_contrast": 0.0,
            "timeout_contrast": 0.0,
            "clean_dirty_overlap_penalty": 0.0,
            "rows": [],
        }
    labels_arr = np.asarray(labels)
    n = int(len(labels_arr))
    utility = _first_finite_target(
        targets,
        ("clean_utility", "returns", "utility", "target"),
        n,
    )
    if utility is None:
        return {
            "score": 0.0,
            "clean_positive_contrast": 0.0,
            "dirty_positive_contrast": 0.0,
            "bad_mae_contrast": 0.0,
            "timeout_contrast": 0.0,
            "clean_dirty_overlap_penalty": 0.0,
            "rows": [],
        }
    bad_mae = _first_finite_target(targets, ("bad_mae_1r", "bad_mae", "bad_MAE"), n)
    timeout = _first_finite_target(targets, ("timeout", "is_timeout"), n)
    full_stop = _first_finite_target(targets, ("full_stop_loss", "full_sl", "stop_loss"), n)
    clean_positive_arr = _first_finite_target(targets, ("clean_positive", "clean_executable_positive"), n)
    dirty_positive_arr = _first_finite_target(targets, ("dirty_positive", "dirty_executable_positive"), n)

    finite = np.isfinite(utility)
    if bad_mae is None:
        bad_mae = np.zeros(n, dtype=np.float32)
    if timeout is None:
        timeout = np.zeros(n, dtype=np.float32)
    if full_stop is None:
        full_stop = np.zeros(n, dtype=np.float32)
    bad_flag = np.asarray(np.nan_to_num(bad_mae, nan=0.0) >= 0.5, dtype=bool)
    timeout_flag = np.asarray(np.nan_to_num(timeout, nan=0.0) >= 0.5, dtype=bool)
    stop_flag = np.asarray(np.nan_to_num(full_stop, nan=0.0) >= 0.5, dtype=bool)
    positive = np.asarray(utility > 0.0, dtype=bool) & finite
    if clean_positive_arr is None:
        clean_positive = positive & ~bad_flag & ~timeout_flag & ~stop_flag
    else:
        clean_positive = np.asarray(np.nan_to_num(clean_positive_arr, nan=0.0) >= 0.5, dtype=bool) & finite
    if dirty_positive_arr is None:
        dirty_positive = positive & (bad_flag | timeout_flag | stop_flag)
    else:
        dirty_positive = np.asarray(np.nan_to_num(dirty_positive_arr, nan=0.0) >= 0.5, dtype=bool) & finite

    if int(np.sum(finite)) < 50 or int(np.sum(positive)) < 10:
        return {
            "score": 0.0,
            "clean_positive_contrast": 0.0,
            "dirty_positive_contrast": 0.0,
            "bad_mae_contrast": 0.0,
            "timeout_contrast": 0.0,
            "clean_dirty_overlap_penalty": 0.0,
            "rows": [],
        }

    cluster_rows: list[dict[str, Any]] = []
    clean_rates: list[float] = []
    dirty_rates: list[float] = []
    bad_rates: list[float] = []
    timeout_rates: list[float] = []
    risk_rates: list[float] = []
    weights: list[float] = []
    for cluster in sorted(int(v) for v in np.unique(labels_arr)):
        mask = (labels_arr == cluster) & finite
        count = int(np.sum(mask))
        if count < 10:
            continue
        clean_rate = float(np.mean(clean_positive[mask]))
        dirty_rate = float(np.mean(dirty_positive[mask]))
        bad_rate = float(np.mean(bad_flag[mask]))
        timeout_rate = float(np.mean(timeout_flag[mask]))
        risk_rate = float(np.mean((bad_flag | timeout_flag | stop_flag)[mask]))
        mean_u = float(np.nanmean(utility[mask]))
        clean_rates.append(clean_rate)
        dirty_rates.append(dirty_rate)
        bad_rates.append(bad_rate)
        timeout_rates.append(timeout_rate)
        risk_rates.append(risk_rate)
        weights.append(float(count))
        cluster_rows.append(
            {
                "cluster": int(cluster),
                "rows": count,
                "mean_u": mean_u,
                "positive_rate": float(np.mean(positive[mask])),
                "clean_positive_rate": clean_rate,
                "dirty_positive_rate": dirty_rate,
                "bad_mae_rate": bad_rate,
                "timeout_rate": timeout_rate,
                "path_risk_rate": risk_rate,
            }
        )

    if len(clean_rates) < 2:
        return {
            "score": 0.0,
            "clean_positive_contrast": 0.0,
            "dirty_positive_contrast": 0.0,
            "bad_mae_contrast": 0.0,
            "timeout_contrast": 0.0,
            "clean_dirty_overlap_penalty": 0.0,
            "rows": cluster_rows,
        }

    clean_arr = np.asarray(clean_rates, dtype=np.float32)
    dirty_arr = np.asarray(dirty_rates, dtype=np.float32)
    bad_arr = np.asarray(bad_rates, dtype=np.float32)
    timeout_arr = np.asarray(timeout_rates, dtype=np.float32)
    risk_arr = np.asarray(risk_rates, dtype=np.float32)
    weight_arr = np.asarray(weights, dtype=np.float32)

    def contrast(arr: np.ndarray) -> float:
        return float(np.nanpercentile(arr, 90.0) - np.nanpercentile(arr, 10.0))

    clean_contrast = contrast(clean_arr)
    dirty_contrast = contrast(dirty_arr)
    bad_contrast = contrast(bad_arr)
    timeout_contrast = contrast(timeout_arr)
    clean_std = float(np.nanstd(clean_arr))
    risk_std = float(np.nanstd(risk_arr))
    corr_penalty = 0.0
    if clean_std > 1e-6 and risk_std > 1e-6:
        corr = float(np.corrcoef(clean_arr, risk_arr)[0, 1])
        corr_penalty = max(corr, 0.0) if math.isfinite(corr) else 0.0
    global_risk = float(np.average(risk_arr, weights=weight_arr))
    top_clean = clean_arr >= float(np.nanpercentile(clean_arr, 75.0))
    top_clean_risk = float(np.average(risk_arr[top_clean], weights=weight_arr[top_clean])) if np.any(top_clean) else global_risk
    excess_penalty = max(top_clean_risk - global_risk, 0.0) / max(1.0 - global_risk, 1e-6)
    overlap_penalty = float(np.clip(max(corr_penalty, excess_penalty), 0.0, 1.0))
    raw_score = float(
        0.36 * np.tanh(clean_contrast / 0.08)
        + 0.22 * np.tanh(dirty_contrast / 0.08)
        + 0.20 * np.tanh(bad_contrast / 0.08)
        + 0.12 * np.tanh(timeout_contrast / 0.06)
        + 0.10 * np.tanh(abs(float(np.average(clean_arr, weights=weight_arr)) - float(np.average(dirty_arr, weights=weight_arr))) / 0.08)
    )
    score = float(np.clip(raw_score * (1.0 - 0.45 * overlap_penalty), 0.0, 1.0))
    return {
        "score": score,
        "clean_positive_contrast": float(clean_contrast),
        "dirty_positive_contrast": float(dirty_contrast),
        "bad_mae_contrast": float(bad_contrast),
        "timeout_contrast": float(timeout_contrast),
        "clean_dirty_overlap_penalty": overlap_penalty,
        "top_clean_path_risk_rate": float(top_clean_risk),
        "global_path_risk_rate": float(global_risk),
        "rows": cluster_rows[:60],
    }


def _temporal_concentration_score(
    labels: np.ndarray,
    targets: dict[str, Any] | None,
) -> dict[str, Any]:
    if not targets or len(labels) == 0:
        return {
            "score": 1.0,
            "max_cluster_time_bucket_share": 0.0,
            "min_cluster_time_bucket_coverage": 1.0,
            "rows": [],
        }
    time_arr = _first_finite_target(targets, ("time_bucket", "month", "__time_bucket__", "_time_bucket"), len(labels))
    if time_arr is None:
        return {
            "score": 1.0,
            "max_cluster_time_bucket_share": 0.0,
            "min_cluster_time_bucket_coverage": 1.0,
            "rows": [],
        }
    labels_arr = np.asarray(labels)
    time = np.asarray(time_arr)
    finite = np.isfinite(time)
    if int(np.sum(finite)) < 50:
        return {
            "score": 1.0,
            "max_cluster_time_bucket_share": 0.0,
            "min_cluster_time_bucket_coverage": 1.0,
            "rows": [],
        }
    unique_buckets = np.unique(time[finite])
    if len(unique_buckets) <= 1:
        return {
            "score": 1.0,
            "max_cluster_time_bucket_share": 1.0,
            "min_cluster_time_bucket_coverage": 1.0,
            "rows": [],
        }
    rows: list[dict[str, Any]] = []
    max_shares: list[float] = []
    coverage: list[float] = []
    weights: list[float] = []
    for cluster in sorted(int(v) for v in np.unique(labels_arr)):
        mask = (labels_arr == cluster) & finite
        count = int(np.sum(mask))
        if count < 10:
            continue
        bucket_counts = np.asarray([np.sum(time[mask] == bucket) for bucket in unique_buckets], dtype=np.float32)
        max_share = float(np.max(bucket_counts) / max(float(np.sum(bucket_counts)), 1.0))
        covered = float(np.mean(bucket_counts > 0.0))
        max_shares.append(max_share)
        coverage.append(covered)
        weights.append(float(count))
        rows.append(
            {
                "cluster": int(cluster),
                "rows": count,
                "max_time_bucket_share": max_share,
                "time_bucket_coverage": covered,
            }
        )
    if not max_shares:
        return {
            "score": 1.0,
            "max_cluster_time_bucket_share": 0.0,
            "min_cluster_time_bucket_coverage": 1.0,
            "rows": rows,
        }
    max_arr = np.asarray(max_shares, dtype=np.float32)
    cov_arr = np.asarray(coverage, dtype=np.float32)
    w_arr = np.asarray(weights, dtype=np.float32)
    weighted_max = float(np.average(max_arr, weights=w_arr))
    weighted_cov = float(np.average(cov_arr, weights=w_arr))
    score = float(np.clip(0.55 * (1.0 - weighted_max) + 0.45 * weighted_cov, 0.0, 1.0))
    return {
        "score": score,
        "max_cluster_time_bucket_share": float(np.max(max_arr)),
        "weighted_max_time_bucket_share": weighted_max,
        "min_cluster_time_bucket_coverage": float(np.min(cov_arr)),
        "weighted_time_bucket_coverage": weighted_cov,
        "rows": rows,
    }


def _temporal_stability_score(labels: np.ndarray) -> dict[str, float]:
    if len(labels) <= 1:
        return {
            "switch_rate": 0.0,
            "stability_20_mean": 1.0,
            "avg_duration": float(len(labels)),
            "score": 1.0,
        }
    switch_rate = float(np.mean(labels[1:] != labels[:-1]))
    _age, stability, _flips = _cluster_stability(labels, window=20)
    changes = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
    durations = np.diff(changes).astype(np.float32)
    avg_duration = float(np.nanmean(durations)) if len(durations) else float(len(labels))
    switch_score = 1.0 - min(abs(switch_rate - 0.05) / 0.20, 1.0)
    duration_score = min(avg_duration / 20.0, 1.0)
    stability_score = float(np.nanmean(stability)) if len(stability) else 1.0
    score = float(np.clip(0.45 * stability_score + 0.35 * switch_score + 0.20 * duration_score, 0.0, 1.0))
    return {
        "switch_rate": switch_rate,
        "stability_20_mean": stability_score,
        "avg_duration": avg_duration,
        "score": score,
    }


def _diff1(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) <= 1:
        return np.zeros_like(arr, dtype=np.float32)
    out = np.empty_like(arr, dtype=np.float32)
    out[0] = 0.0
    out[1:] = arr[1:] - arr[:-1]
    return out.astype(np.float32, copy=False)


def _row_speed(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or len(arr) <= 1:
        return np.zeros(arr.shape[0] if arr.ndim == 2 else 0, dtype=np.float32)
    diff = np.zeros_like(arr, dtype=np.float32)
    diff[1:] = arr[1:] - arr[:-1]
    return np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)


def _candidate_ints(values: Any, default: tuple[int, ...], *, min_value: int, max_value: int) -> tuple[int, ...]:
    if values is None:
        raw = list(default)
    elif isinstance(values, str):
        raw = [v.strip() for v in values.split(",") if v.strip()]
    else:
        raw = list(values)
    out: list[int] = []
    for value in raw:
        try:
            item = int(value)
        except Exception:
            continue
        if min_value <= item <= max_value and item not in out:
            out.append(item)
    return tuple(out or default)


def _candidate_floats(
    values: Any,
    default: tuple[float, ...],
    *,
    min_value: float,
    max_value: float,
) -> tuple[float, ...]:
    if values is None:
        raw = list(default)
    elif isinstance(values, str):
        raw = [v.strip() for v in values.split(",") if v.strip()]
    else:
        raw = list(values)
    out: list[float] = []
    for value in raw:
        try:
            item = float(value)
        except Exception:
            continue
        if not np.isfinite(item):
            continue
        if min_value <= item <= max_value and item not in out:
            out.append(item)
    return tuple(out or default)


def _side_balance_report(
    labels: np.ndarray,
    side_values: Any,
    *,
    min_side_cluster_frac: float,
    min_side_cluster_rows: int,
) -> dict[str, Any]:
    side = np.asarray(side_values, dtype=np.float32)
    if len(side) != len(labels):
        return {
            "side_available": False,
            "side_coverage_ok": False,
            "side_balance_score": 0.0,
            "min_cluster_long_share": float("nan"),
            "min_cluster_short_share": float("nan"),
            "cluster_side_counts": [],
        }
    side = np.where(side < 0.0, -1, 1).astype(np.int8)
    clusters = sorted(int(v) for v in np.unique(labels))
    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    coverage_ok = True
    min_long_share = 1.0
    min_short_share = 1.0
    for cluster in clusters:
        mask = labels == cluster
        total = int(np.sum(mask))
        long_count = int(np.sum(side[mask] > 0))
        short_count = int(np.sum(side[mask] < 0))
        long_share = float(long_count / max(total, 1))
        short_share = float(short_count / max(total, 1))
        min_long_share = min(min_long_share, long_share)
        min_short_share = min(min_short_share, short_share)
        scores.append(float(1.0 - min(abs(long_share - short_share), 1.0)))
        cluster_ok = (
            long_count >= int(min_side_cluster_rows)
            and short_count >= int(min_side_cluster_rows)
            and min(long_share, short_share) >= float(min_side_cluster_frac)
        )
        coverage_ok = bool(coverage_ok and cluster_ok)
        rows.append(
            {
                "cluster": int(cluster),
                "rows": total,
                "long_count": long_count,
                "short_count": short_count,
                "long_share": long_share,
                "short_share": short_share,
                "side_coverage_ok": bool(cluster_ok),
            }
        )
    return {
        "side_available": True,
        "side_coverage_ok": bool(coverage_ok),
        "side_balance_score": float(np.mean(scores)) if scores else 0.0,
        "min_cluster_long_share": float(min_long_share) if rows else float("nan"),
        "min_cluster_short_share": float(min_short_share) if rows else float("nan"),
        "cluster_side_counts": rows,
    }


def _rank01(values: list[float], *, higher_is_better: bool = True) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    if not np.isfinite(arr).any():
        return [0.5 for _ in values]
    fill = float(np.nanmedian(arr[np.isfinite(arr)]))
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=np.float64)
    ranks[order] = np.arange(len(arr), dtype=np.float64)
    if len(arr) > 1:
        ranks /= float(len(arr) - 1)
    else:
        ranks[:] = 1.0
    if not higher_is_better:
        ranks = 1.0 - ranks
    return [float(x) for x in ranks]


def fit_ae_gmm_state(
    x_reference: Any,
    *,
    economic_targets: dict[str, Any] | None = None,
    random_state: int = 42,
    max_train_rows: int = 5000,
    ae_max_iter: int = 80,
    cluster_candidates: Any = None,
    reg_covar_candidates: Any = None,
    smooth_lambda_candidates: Any = None,
    min_occupancy: float = AE_GMM_MIN_OCCUPANCY,
    max_occupancy: float = AE_GMM_MAX_OCCUPANCY,
    require_both_sides: bool = False,
    min_side_cluster_frac: float = 0.02,
    min_side_cluster_rows: int = 10,
    path_aware_hpo: bool | None = None,
    temporal_concentration_hpo: bool | None = None,
) -> dict[str, Any]:
    x_df = _as_float_frame(x_reference)
    if len(x_df) < 200 or x_df.shape[1] < 2:
        return {
            "enabled": False,
            "reason": "insufficient_rows_or_features",
            "feature_columns": list(x_df.columns),
        }
    if max_train_rows > 0 and len(x_df) > int(max_train_rows):
        idx = np.linspace(0, len(x_df) - 1, int(max_train_rows), dtype=int)
        fit_df = x_df.iloc[idx].reset_index(drop=True)
        economic_targets_fit = {
            k: np.asarray(v)[idx]
            for k, v in (economic_targets or {}).items()
            if len(np.asarray(v)) == len(x_df)
        }
    else:
        fit_df = x_df.reset_index(drop=True)
        economic_targets_fit = economic_targets or {}
    center, scale = _robust_scale_fit(fit_df)
    x_scaled = _robust_scale_apply(fit_df, center, scale)
    ae_state = fit_denoising_autoencoder_state(
        x_scaled,
        random_state=random_state,
        max_train_rows=max_train_rows,
        max_iter=ae_max_iter,
    )
    ae_features = transform_denoising_autoencoder_features(
        x_scaled,
        ae_state,
        index=pd.RangeIndex(len(x_scaled)),
    )
    latent_cols = [f"ae_b16_{i:02d}" for i in range(AE_GMM_LATENT_DIM)]
    if not set(latent_cols).issubset(ae_features.columns):
        return {
            "enabled": False,
            "reason": "ae_b16_unavailable",
            "feature_columns": list(x_df.columns),
            "ae_state": ae_state,
        }
    z = ae_features[latent_cols].to_numpy(dtype=np.float32, copy=False)
    recon = pd.to_numeric(
        ae_features.get("ae_b16_reconstruction_error", 0.0),
        errors="coerce",
    ).to_numpy(dtype=np.float32, copy=False)
    recon_mean = float(np.nanmean(recon)) if np.isfinite(recon).any() else 0.0
    recon_std = float(np.nanstd(recon) + 1e-6) if np.isfinite(recon).any() else 1.0
    split = max(1, int(round(0.80 * len(z))))
    if len(z) - split < max(50, int(0.05 * len(z))):
        split = max(1, len(z) - max(50, int(0.10 * len(z))))
    z_train = z[:split]
    z_valid = z[split:] if split < len(z) else z_train
    reports: list[dict[str, Any]] = []
    fitted: list[tuple[dict[str, Any], GaussianMixture]] = []
    clusters_to_try = _candidate_ints(
        cluster_candidates,
        AE_GMM_CLUSTER_CANDIDATES,
        min_value=2,
        max_value=AE_GMM_MAX_COMPONENTS,
    )
    regs_to_try = _candidate_floats(
        reg_covar_candidates,
        AE_GMM_REG_COVAR_CANDIDATES,
        min_value=1e-8,
        max_value=1.0,
    )
    smooth_to_try = _candidate_floats(
        smooth_lambda_candidates,
        AE_GMM_SMOOTH_LAMBDA_CANDIDATES,
        min_value=0.0,
        max_value=0.999,
    )
    side_values = economic_targets_fit.get("side") if isinstance(economic_targets_fit, dict) else None
    use_path_aware_hpo = bool(AE_GMM_PATH_AWARE_HPO if path_aware_hpo is None else path_aware_hpo)
    use_temporal_concentration_hpo = bool(
        AE_GMM_TEMPORAL_CONCENTRATION_HPO
        if temporal_concentration_hpo is None
        else temporal_concentration_hpo
    )
    for k in clusters_to_try:
        if len(z_train) < k * 20:
            continue
        for reg_covar in regs_to_try:
            try:
                gmm = GaussianMixture(
                    n_components=int(k),
                    covariance_type="diag",
                    reg_covar=float(reg_covar),
                    random_state=int(random_state + k * 17 + int(reg_covar * 1e6)),
                    max_iter=200,
                )
                labels_train = gmm.fit_predict(z_train)
                prob_all = gmm.predict_proba(z)
                ll_valid = float(gmm.score(z_valid)) if len(z_valid) else float(gmm.score(z_train))
                latent_quality = float(
                    1.0 / (1.0 + float((ae_state.get("models") or {}).get("b16", {}).get("selected_score", 0.0)))
                )
                for smooth_lambda in smooth_to_try:
                    prob_eval = _smooth_probabilities(prob_all, float(smooth_lambda))
                    labels_all = np.argmax(prob_eval, axis=1).astype(np.int32)
                    counts = np.bincount(labels_all, minlength=int(k)).astype(np.float64)
                    occupancy = counts / max(float(np.sum(counts)), 1.0)
                    occupancy_ok = bool(
                        float(np.min(occupancy)) >= float(min_occupancy)
                        and float(np.max(occupancy)) <= float(max_occupancy)
                    )
                    economic = _economic_separation(labels_all, economic_targets_fit)
                    signature = _cluster_target_signature_score(
                        labels_all,
                        economic_targets_fit,
                    )
                    path_cleanliness = _path_cleanliness_separation(
                        labels_all,
                        economic_targets_fit,
                    )
                    temporal_concentration = _temporal_concentration_score(
                        labels_all,
                        economic_targets_fit,
                    )
                    stability = _temporal_stability_score(labels_all)
                    side_report = _side_balance_report(
                        labels_all,
                        side_values,
                        min_side_cluster_frac=float(min_side_cluster_frac),
                        min_side_cluster_rows=int(min_side_cluster_rows),
                    ) if side_values is not None else {
                        "side_available": False,
                        "side_coverage_ok": not bool(require_both_sides),
                        "side_balance_score": 0.0,
                        "min_cluster_long_share": float("nan"),
                        "min_cluster_short_share": float("nan"),
                        "cluster_side_counts": [],
                    }
                    side_ok = bool(side_report.get("side_coverage_ok", False)) or not bool(require_both_sides)
                    occupancy_balance_score = float(1.0 - min(float(np.max(occupancy) - np.min(occupancy)), 1.0))
                    report = {
                        "n_components": int(k),
                        "reg_covar": float(reg_covar),
                        "smooth_lambda": float(smooth_lambda),
                        "validation_log_likelihood": ll_valid,
                        "economic_regime_separation": float(economic),
                        "target_signature_score": float(signature["score"]),
                        "target_signature_stability": float(signature["sign_stability"]),
                        "target_signature_contrast": float(signature["contrast"]),
                        "target_signature_target_count": int(signature["target_count"]),
                        "cluster_target_signatures": signature.get("rows", []),
                        "path_aware_hpo": bool(use_path_aware_hpo),
                        "path_cleanliness_score": float(path_cleanliness["score"]),
                        "clean_positive_contrast": float(path_cleanliness["clean_positive_contrast"]),
                        "dirty_positive_contrast": float(path_cleanliness["dirty_positive_contrast"]),
                        "bad_mae_contrast": float(path_cleanliness["bad_mae_contrast"]),
                        "timeout_contrast": float(path_cleanliness["timeout_contrast"]),
                        "clean_dirty_overlap_penalty": float(path_cleanliness["clean_dirty_overlap_penalty"]),
                        "top_clean_path_risk_rate": path_cleanliness.get("top_clean_path_risk_rate"),
                        "global_path_risk_rate": path_cleanliness.get("global_path_risk_rate"),
                        "cluster_path_cleanliness": path_cleanliness.get("rows", []),
                        "temporal_concentration_hpo": bool(use_temporal_concentration_hpo),
                        "temporal_concentration_score": float(temporal_concentration["score"]),
                        "max_cluster_time_bucket_share": float(
                            temporal_concentration["max_cluster_time_bucket_share"]
                        ),
                        "min_cluster_time_bucket_coverage": float(
                            temporal_concentration["min_cluster_time_bucket_coverage"]
                        ),
                        "cluster_temporal_concentration": temporal_concentration.get("rows", []),
                        "temporal_stability_score": float(stability["score"]),
                        "switch_rate": float(stability["switch_rate"]),
                        "stability_20_mean": float(stability["stability_20_mean"]),
                        "avg_duration": float(stability["avg_duration"]),
                        "latent_quality_score": float(latent_quality),
                        "min_occupancy": float(np.min(occupancy)),
                        "max_occupancy": float(np.max(occupancy)),
                        "occupancy_balance_score": occupancy_balance_score,
                        "occupancy": [float(x) for x in occupancy],
                        "occupancy_ok": occupancy_ok,
                        "side_available": bool(side_report.get("side_available", False)),
                        "side_coverage_ok": bool(side_report.get("side_coverage_ok", False)),
                        "side_balance_score": float(side_report.get("side_balance_score", 0.0)),
                        "min_cluster_long_share": side_report.get("min_cluster_long_share"),
                        "min_cluster_short_share": side_report.get("min_cluster_short_share"),
                        "cluster_side_counts": side_report.get("cluster_side_counts", []),
                        "converged": bool(getattr(gmm, "converged_", False)),
                    }
                    reports.append(report)
                    if occupancy_ok and side_ok:
                        fitted.append((report, gmm))
            except Exception as exc:
                reports.append(
                    {
                        "n_components": int(k),
                        "reg_covar": float(reg_covar),
                        "error": str(exc),
                        "occupancy_ok": False,
                        "side_coverage_ok": False,
                    }
                )
    valid_reports = [r for r, _g in fitted]
    if not valid_reports:
        return {
            "enabled": False,
            "reason": "no_valid_gmm_config",
            "feature_columns": list(x_df.columns),
            "center": center.astype(float).tolist(),
            "scale": scale.astype(float).tolist(),
            "ae_state": ae_state,
            "hpo_grid": {
                "cluster_candidates": [int(v) for v in clusters_to_try],
                "reg_covar_candidates": [float(v) for v in regs_to_try],
                "smooth_lambda_candidates": [float(v) for v in smooth_to_try],
                "require_both_sides": bool(require_both_sides),
                "path_aware_hpo": bool(use_path_aware_hpo),
                "temporal_concentration_hpo": bool(use_temporal_concentration_hpo),
                "min_side_cluster_frac": float(min_side_cluster_frac),
                "min_side_cluster_rows": int(min_side_cluster_rows),
            },
            "reports": reports[:12],
        }
    econ_rank = _rank01([r["economic_regime_separation"] for r in valid_reports])
    signature_rank = _rank01([r.get("target_signature_score", 0.0) for r in valid_reports])
    path_rank = _rank01([r.get("path_cleanliness_score", 0.0) for r in valid_reports])
    concentration_rank = _rank01([r.get("temporal_concentration_score", 1.0) for r in valid_reports])
    stability_rank = _rank01([r["temporal_stability_score"] for r in valid_reports])
    ll_rank = _rank01([r["validation_log_likelihood"] for r in valid_reports])
    latent_rank = _rank01([r["latent_quality_score"] for r in valid_reports])
    side_rank = _rank01([r.get("side_balance_score", 0.0) for r in valid_reports])
    occupancy_rank = _rank01([r.get("occupancy_balance_score", 0.0) for r in valid_reports])
    for i, r in enumerate(valid_reports):
        path_weight = 0.18 if use_path_aware_hpo else 0.0
        concentration_weight = 0.08 if use_temporal_concentration_hpo else 0.0
        stability_weight = 0.13 + (0.18 - path_weight) * 0.45 + (0.08 - concentration_weight) * 0.35
        side_weight = 0.13 + (0.18 - path_weight) * 0.30
        occupancy_weight = 1.0 - (
            0.18
            + 0.16
            + path_weight
            + concentration_weight
            + stability_weight
            + 0.05
            + 0.05
            + side_weight
        )
        r["final_score"] = float(
            0.18 * econ_rank[i]
            + 0.16 * signature_rank[i]
            + path_weight * path_rank[i]
            + concentration_weight * concentration_rank[i]
            + stability_weight * stability_rank[i]
            + 0.05 * ll_rank[i]
            + 0.05 * latent_rank[i]
            + side_weight * side_rank[i]
            + occupancy_weight * occupancy_rank[i]
        )
    best_i = int(np.argmax([r["final_score"] for r in valid_reports]))
    best_report, best_gmm = fitted[best_i]
    reports_sorted = sorted(
        reports,
        key=lambda r: float(r.get("final_score", -1e9)),
        reverse=True,
    )[:6]
    return {
        "enabled": True,
        "schema_version": "ae_gmm_v1",
        "feature_columns": list(x_df.columns),
        "center": center.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
        "clip": [-8.0, 8.0],
        "ae_state": ae_state,
        "latent_columns": list(AE_GMM_LATENT_FEATURE_COLUMNS),
        "gmm_n_components": int(best_gmm.n_components),
        "gmm_covariance_type": "diag",
        "gmm_reg_covar": float(best_gmm.reg_covar),
        "gmm_weights": best_gmm.weights_.astype(float).tolist(),
        "gmm_means": best_gmm.means_.astype(float).tolist(),
        "gmm_covariances": best_gmm.covariances_.astype(float).tolist(),
        "smooth_lambda": float(best_report.get("smooth_lambda", AE_GMM_SMOOTH_LAMBDA)),
        "max_components": int(AE_GMM_MAX_COMPONENTS),
        "reconstruction_error_mean": float(recon_mean),
        "reconstruction_error_std": float(recon_std),
        "selected_config": dict(best_report),
        "top_configs": reports_sorted,
        "hpo_grid": {
            "cluster_candidates": [int(v) for v in clusters_to_try],
            "reg_covar_candidates": [float(v) for v in regs_to_try],
            "smooth_lambda_candidates": [float(v) for v in smooth_to_try],
            "require_both_sides": bool(require_both_sides),
            "path_aware_hpo": bool(use_path_aware_hpo),
            "temporal_concentration_hpo": bool(use_temporal_concentration_hpo),
            "min_occupancy": float(min_occupancy),
            "max_occupancy": float(max_occupancy),
            "min_side_cluster_frac": float(min_side_cluster_frac),
            "min_side_cluster_rows": int(min_side_cluster_rows),
        },
        "hpo_report_count": int(len(reports)),
        "hpo_reports": sorted(
            reports,
            key=lambda r: float(r.get("final_score", -1e9)),
            reverse=True,
        ),
        "report": dict(best_report),
    }


def transform_ae_gmm_features(
    x: Any,
    state: dict[str, Any] | None,
    *,
    index: Any = None,
    prefix: str = "",
) -> pd.DataFrame:
    x_df = _as_float_frame(x)
    out_columns = ae_gmm_feature_columns(prefix)
    idx = x_df.index if index is None else index
    if not state or not bool(state.get("enabled", False)):
        return pd.DataFrame(0.0, index=idx, columns=out_columns, dtype=np.float32)
    feature_columns = [str(c) for c in state.get("feature_columns", list(x_df.columns))]
    x_aligned = x_df.reindex(columns=feature_columns, fill_value=0.0)
    center = np.asarray(state.get("center", np.zeros(len(feature_columns))), dtype=np.float32)
    scale = np.asarray(state.get("scale", np.ones(len(feature_columns))), dtype=np.float32)
    x_scaled = _robust_scale_apply(x_aligned, center, scale)
    ae_features = transform_denoising_autoencoder_features(
        x_scaled,
        state.get("ae_state", {}),
        index=idx,
    )
    latent_source_cols = [f"ae_b16_{i:02d}" for i in range(AE_GMM_LATENT_DIM)]
    z = ae_features.reindex(columns=latent_source_cols, fill_value=0.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    prob = _diag_gmm_predict_proba(z, state)
    prob_smooth = _smooth_probabilities(
        prob,
        float(state.get("smooth_lambda", AE_GMM_SMOOTH_LAMBDA)),
    )
    dist, mahal = _gmm_distances(z, state)
    k = prob.shape[1]
    labels = (
        np.argmax(prob_smooth, axis=1).astype(np.int32)
        if k > 0 and len(prob_smooth)
        else np.zeros(len(x_df), dtype=np.int32)
    )
    age, stability, flips = _cluster_stability(labels, window=20)
    entropy = -np.sum(prob_smooth * np.log(np.maximum(prob_smooth, 1e-12)), axis=1) if k > 0 else np.zeros(len(x_df), dtype=np.float32)
    entropy_norm = entropy / max(float(np.log(max(k, 2))), 1e-6)
    min_mahal = np.min(mahal, axis=1) if k > 0 else np.zeros(len(x_df), dtype=np.float32)
    expected_mahal = np.sum(prob_smooth * mahal, axis=1) if k > 0 else np.zeros(len(x_df), dtype=np.float32)
    posterior_max = np.max(prob_smooth, axis=1) if k > 0 else np.zeros(len(x_df), dtype=np.float32)
    if k > 1:
        posterior_top2 = np.sort(prob_smooth, axis=1)[:, -2:]
        posterior_margin = posterior_top2[:, 1] - posterior_top2[:, 0]
    else:
        posterior_margin = posterior_max.copy()
    posterior_delta = _diff1(posterior_max)
    posterior_accel = _diff1(posterior_delta)
    entropy_delta = _diff1(entropy)
    entropy_accel = _diff1(entropy_delta)
    min_mahal_delta = _diff1(min_mahal)
    expected_mahal_delta = _diff1(expected_mahal)
    expected_mahal_accel = _diff1(expected_mahal_delta)
    recon = pd.to_numeric(
        ae_features.get("ae_b16_reconstruction_error", 0.0),
        errors="coerce",
    ).to_numpy(dtype=np.float32, copy=False)
    recon = np.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0)
    recon_z = (
        (recon - float(state.get("reconstruction_error_mean", 0.0)))
        / max(float(state.get("reconstruction_error_std", 1.0)), 1e-6)
    ).astype(np.float32)
    recon_delta = _diff1(recon)
    recon_accel = _diff1(recon_delta)
    latent_speed = _row_speed(z)
    latent_acceleration = _diff1(latent_speed)
    cluster_speed = _row_speed(prob_smooth) if k > 0 else np.zeros(len(x_df), dtype=np.float32)
    cluster_acceleration = _diff1(cluster_speed)
    data: dict[str, np.ndarray] = {}
    for i in range(AE_GMM_LATENT_DIM):
        data[f"{prefix}dae_b16_{i:02d}"] = z[:, i].astype(np.float32)
    for i in range(AE_GMM_MAX_COMPONENTS):
        data[f"{prefix}gmm_prob_{i}"] = (
            prob_smooth[:, i].astype(np.float32) if i < k else np.zeros(len(x_df), dtype=np.float32)
        )
    for i in range(AE_GMM_MAX_COMPONENTS):
        data[f"{prefix}gmm_cluster_posterior_{i}"] = data[f"{prefix}gmm_prob_{i}"]
    for i in range(AE_GMM_MAX_COMPONENTS):
        data[f"{prefix}gmm_dist_center_{i}"] = (
            dist[:, i].astype(np.float32) if i < k else np.zeros(len(x_df), dtype=np.float32)
        )
    for i in range(AE_GMM_MAX_COMPONENTS):
        data[f"{prefix}gmm_mahal_{i}"] = (
            mahal[:, i].astype(np.float32) if i < k else np.zeros(len(x_df), dtype=np.float32)
        )
    data[f"{prefix}gmm_cluster_id"] = labels.astype(np.float32)
    data[f"{prefix}gmm_posterior_max"] = posterior_max.astype(np.float32)
    data[f"{prefix}gmm_posterior_margin"] = posterior_margin.astype(np.float32)
    data[f"{prefix}gmm_posterior_delta_1"] = posterior_delta.astype(np.float32)
    data[f"{prefix}gmm_posterior_accel_1"] = posterior_accel.astype(np.float32)
    data[f"{prefix}gmm_entropy"] = entropy.astype(np.float32)
    data[f"{prefix}cluster_entropy"] = entropy.astype(np.float32)
    data[f"{prefix}cluster_entropy_norm"] = entropy_norm.astype(np.float32)
    data[f"{prefix}cluster_entropy_delta_1"] = entropy_delta.astype(np.float32)
    data[f"{prefix}cluster_entropy_accel_1"] = entropy_accel.astype(np.float32)
    data[f"{prefix}mahalanobis_distance"] = min_mahal.astype(np.float32)
    data[f"{prefix}min_mahalanobis"] = min_mahal.astype(np.float32)
    data[f"{prefix}min_mahalanobis_delta_1"] = min_mahal_delta.astype(np.float32)
    data[f"{prefix}expected_mahalanobis"] = expected_mahal.astype(np.float32)
    data[f"{prefix}expected_mahalanobis_delta_1"] = expected_mahal_delta.astype(np.float32)
    data[f"{prefix}expected_mahalanobis_accel_1"] = expected_mahal_accel.astype(np.float32)
    data[f"{prefix}cluster_t"] = labels.astype(np.float32)
    data[f"{prefix}cluster_speed"] = cluster_speed.astype(np.float32)
    data[f"{prefix}cluster_acceleration"] = cluster_acceleration.astype(np.float32)
    data[f"{prefix}time_since_cluster_change"] = age.astype(np.float32)
    data[f"{prefix}rolling_cluster_stability"] = stability.astype(np.float32)
    data[f"{prefix}cluster_flip_count_20"] = flips.astype(np.float32)
    data[f"{prefix}AE_reconstruction_error"] = recon.astype(np.float32)
    data[f"{prefix}ae_reconstruction_error"] = recon.astype(np.float32)
    data[f"{prefix}dae_reconstruction_error"] = recon.astype(np.float32)
    data[f"{prefix}dae_reconstruction_error_zscore"] = recon_z.astype(np.float32)
    data[f"{prefix}dae_reconstruction_error_delta_1"] = recon_delta.astype(np.float32)
    data[f"{prefix}dae_reconstruction_error_accel_1"] = recon_accel.astype(np.float32)
    data[f"{prefix}latent_mahalanobis_drift"] = min_mahal.astype(np.float32)
    data[f"{prefix}latent_speed"] = latent_speed.astype(np.float32)
    data[f"{prefix}latent_acceleration"] = latent_acceleration.astype(np.float32)
    return pd.DataFrame(data, index=idx).reindex(columns=out_columns, fill_value=0.0).astype(np.float32)


def fit_transform_ae_gmm_features(
    x_reference: Any,
    *,
    economic_targets: dict[str, Any] | None = None,
    random_state: int = 42,
    max_train_rows: int = 5000,
    ae_max_iter: int = 80,
    cluster_candidates: Any = None,
    reg_covar_candidates: Any = None,
    smooth_lambda_candidates: Any = None,
    require_both_sides: bool = False,
    min_side_cluster_frac: float = 0.02,
    min_side_cluster_rows: int = 10,
    index: Any = None,
    prefix: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    state = fit_ae_gmm_state(
        x_reference,
        economic_targets=economic_targets,
        random_state=random_state,
        max_train_rows=max_train_rows,
        ae_max_iter=ae_max_iter,
        cluster_candidates=cluster_candidates,
        reg_covar_candidates=reg_covar_candidates,
        smooth_lambda_candidates=smooth_lambda_candidates,
        require_both_sides=bool(require_both_sides),
        min_side_cluster_frac=float(min_side_cluster_frac),
        min_side_cluster_rows=int(min_side_cluster_rows),
    )
    return transform_ae_gmm_features(x_reference, state, index=index, prefix=prefix), state
