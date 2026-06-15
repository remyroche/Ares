from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .features_denoising_ae import (
    fit_denoising_autoencoder_state,
    transform_denoising_autoencoder_features,
)


AE_GMM_MAX_COMPONENTS = 6
AE_GMM_CLUSTER_CANDIDATES = (4, 5, 6)
AE_GMM_REG_COVAR_CANDIDATES = (1e-4, 1e-3)
AE_GMM_SMOOTH_LAMBDA = 0.925
AE_GMM_LATENT_DIM = 16
AE_GMM_MIN_OCCUPANCY = 0.03
AE_GMM_MAX_OCCUPANCY = 0.75

AE_GMM_LATENT_FEATURE_COLUMNS = tuple(
    f"dae_b16_{i:02d}" for i in range(AE_GMM_LATENT_DIM)
)
AE_GMM_CLUSTER_FEATURE_COLUMNS = tuple(
    [f"gmm_prob_{i}" for i in range(AE_GMM_MAX_COMPONENTS)]
    + [f"gmm_dist_center_{i}" for i in range(AE_GMM_MAX_COMPONENTS)]
    + [f"gmm_mahal_{i}" for i in range(AE_GMM_MAX_COMPONENTS)]
    + [
        "cluster_entropy",
        "cluster_entropy_norm",
        "min_mahalanobis",
        "expected_mahalanobis",
        "cluster_t",
        "time_since_cluster_change",
        "rolling_cluster_stability",
        "cluster_flip_count_20",
        "dae_reconstruction_error",
        "dae_reconstruction_error_zscore",
        "latent_mahalanobis_drift",
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


def _smooth_probabilities(prob: np.ndarray, lam: float) -> np.ndarray:
    if len(prob) == 0:
        return prob.astype(np.float32)
    out = np.empty_like(prob, dtype=np.float32)
    prev = prob[0].astype(np.float32)
    out[0] = prev
    for i in range(1, len(prob)):
        prev = (float(lam) * prev) + ((1.0 - float(lam)) * prob[i])
        total = float(np.sum(prev))
        if total > 0.0:
            prev = prev / total
        out[i] = prev.astype(np.float32)
    return out


def _cluster_stability(labels: np.ndarray, window: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(labels))
    age = np.zeros(n, dtype=np.float32)
    stability = np.ones(n, dtype=np.float32)
    flips = np.zeros(n, dtype=np.float32)
    last_change = 0
    for i in range(n):
        if i > 0 and labels[i] != labels[i - 1]:
            last_change = i
        start = max(0, i - int(window) + 1)
        recent = labels[start : i + 1]
        age[i] = float(i - last_change)
        stability[i] = float(np.mean(recent == labels[i])) if len(recent) else 1.0
        flips[i] = float(np.sum(recent[1:] != recent[:-1])) if len(recent) > 1 else 0.0
    return age, stability, flips


def _economic_separation(labels: np.ndarray, targets: dict[str, Any] | None) -> float:
    if not targets:
        return 0.0
    vals: list[float] = []
    for arr_like in targets.values():
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
    for k in AE_GMM_CLUSTER_CANDIDATES:
        if len(z_train) < k * 20:
            continue
        for reg_covar in AE_GMM_REG_COVAR_CANDIDATES:
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
                labels_all = np.argmax(prob_all, axis=1).astype(np.int32)
                counts = np.bincount(labels_all, minlength=int(k)).astype(np.float64)
                occupancy = counts / max(float(np.sum(counts)), 1.0)
                occupancy_ok = bool(
                    float(np.min(occupancy)) >= AE_GMM_MIN_OCCUPANCY
                    and float(np.max(occupancy)) <= AE_GMM_MAX_OCCUPANCY
                )
                ll_valid = float(gmm.score(z_valid)) if len(z_valid) else float(gmm.score(z_train))
                economic = _economic_separation(labels_all, economic_targets_fit)
                stability = _temporal_stability_score(labels_all)
                latent_quality = float(
                    1.0 / (1.0 + float((ae_state.get("models") or {}).get("b16", {}).get("selected_score", 0.0)))
                )
                report = {
                    "n_components": int(k),
                    "reg_covar": float(reg_covar),
                    "validation_log_likelihood": ll_valid,
                    "economic_regime_separation": float(economic),
                    "temporal_stability_score": float(stability["score"]),
                    "switch_rate": float(stability["switch_rate"]),
                    "stability_20_mean": float(stability["stability_20_mean"]),
                    "avg_duration": float(stability["avg_duration"]),
                    "latent_quality_score": float(latent_quality),
                    "min_occupancy": float(np.min(occupancy)),
                    "max_occupancy": float(np.max(occupancy)),
                    "occupancy": [float(x) for x in occupancy],
                    "occupancy_ok": occupancy_ok,
                    "converged": bool(getattr(gmm, "converged_", False)),
                }
                reports.append(report)
                if occupancy_ok:
                    fitted.append((report, gmm))
            except Exception as exc:
                reports.append(
                    {
                        "n_components": int(k),
                        "reg_covar": float(reg_covar),
                        "error": str(exc),
                        "occupancy_ok": False,
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
            "reports": reports[:6],
        }
    econ_rank = _rank01([r["economic_regime_separation"] for r in valid_reports])
    stability_rank = _rank01([r["temporal_stability_score"] for r in valid_reports])
    ll_rank = _rank01([r["validation_log_likelihood"] for r in valid_reports])
    latent_rank = _rank01([r["latent_quality_score"] for r in valid_reports])
    for i, r in enumerate(valid_reports):
        r["final_score"] = float(
            0.20 * econ_rank[i]
            + 0.20 * stability_rank[i]
            + 0.10 * ll_rank[i]
            + 0.10 * latent_rank[i]
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
        "smooth_lambda": float(AE_GMM_SMOOTH_LAMBDA),
        "max_components": int(AE_GMM_MAX_COMPONENTS),
        "reconstruction_error_mean": float(recon_mean),
        "reconstruction_error_std": float(recon_std),
        "selected_config": dict(best_report),
        "top_configs": reports_sorted,
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
    recon = pd.to_numeric(
        ae_features.get("ae_b16_reconstruction_error", 0.0),
        errors="coerce",
    ).to_numpy(dtype=np.float32, copy=False)
    recon = np.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0)
    recon_z = (
        (recon - float(state.get("reconstruction_error_mean", 0.0)))
        / max(float(state.get("reconstruction_error_std", 1.0)), 1e-6)
    ).astype(np.float32)
    data: dict[str, np.ndarray] = {}
    for i in range(AE_GMM_LATENT_DIM):
        data[f"{prefix}dae_b16_{i:02d}"] = z[:, i].astype(np.float32)
    for i in range(AE_GMM_MAX_COMPONENTS):
        data[f"{prefix}gmm_prob_{i}"] = (
            prob_smooth[:, i].astype(np.float32) if i < k else np.zeros(len(x_df), dtype=np.float32)
        )
    for i in range(AE_GMM_MAX_COMPONENTS):
        data[f"{prefix}gmm_dist_center_{i}"] = (
            dist[:, i].astype(np.float32) if i < k else np.zeros(len(x_df), dtype=np.float32)
        )
    for i in range(AE_GMM_MAX_COMPONENTS):
        data[f"{prefix}gmm_mahal_{i}"] = (
            mahal[:, i].astype(np.float32) if i < k else np.zeros(len(x_df), dtype=np.float32)
        )
    data[f"{prefix}cluster_entropy"] = entropy.astype(np.float32)
    data[f"{prefix}cluster_entropy_norm"] = entropy_norm.astype(np.float32)
    data[f"{prefix}min_mahalanobis"] = min_mahal.astype(np.float32)
    data[f"{prefix}expected_mahalanobis"] = expected_mahal.astype(np.float32)
    data[f"{prefix}cluster_t"] = labels.astype(np.float32)
    data[f"{prefix}time_since_cluster_change"] = age.astype(np.float32)
    data[f"{prefix}rolling_cluster_stability"] = stability.astype(np.float32)
    data[f"{prefix}cluster_flip_count_20"] = flips.astype(np.float32)
    data[f"{prefix}dae_reconstruction_error"] = recon.astype(np.float32)
    data[f"{prefix}dae_reconstruction_error_zscore"] = recon_z.astype(np.float32)
    data[f"{prefix}latent_mahalanobis_drift"] = min_mahal.astype(np.float32)
    return pd.DataFrame(data, index=idx).reindex(columns=out_columns, fill_value=0.0).astype(np.float32)


def fit_transform_ae_gmm_features(
    x_reference: Any,
    *,
    economic_targets: dict[str, Any] | None = None,
    random_state: int = 42,
    max_train_rows: int = 5000,
    ae_max_iter: int = 80,
    index: Any = None,
    prefix: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    state = fit_ae_gmm_state(
        x_reference,
        economic_targets=economic_targets,
        random_state=random_state,
        max_train_rows=max_train_rows,
        ae_max_iter=ae_max_iter,
    )
    return transform_ae_gmm_features(x_reference, state, index=index, prefix=prefix), state
