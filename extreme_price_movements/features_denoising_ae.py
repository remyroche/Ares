"""Denoising autoencoder bottleneck features for compact state embeddings."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.neural_network import MLPRegressor
except Exception:  # pragma: no cover - optional dependency guard.
    MLPRegressor = None


AE_BOTTLENECKS: tuple[int, ...] = (8, 16)
AE_NOISE_LEVELS: tuple[float, ...] = (0.05, 0.10, 0.15)
AE_WIDTHS: tuple[str, ...] = ("small", "medium")
AE_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    [f"ae_b8_{i:02d}" for i in range(8)]
    + ["ae_b8_reconstruction_error"]
    + [f"ae_b16_{i:02d}" for i in range(16)]
    + ["ae_b16_reconstruction_error"]
)


def _width_layers(bottleneck: int, width: str) -> tuple[int, int, int]:
    b = int(max(1, bottleneck))
    if str(width) == "medium":
        w = max(64, 8 * b)
    else:
        w = max(32, 4 * b)
    return (w, b, w)


def _activation(x: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return x
    if name == "tanh":
        return np.tanh(x)
    if name == "logistic":
        return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))
    return np.maximum(x, 0.0)


def _model_to_spec(model: Any, *, bottleneck: int, width: str, noise: float, metrics: Mapping[str, float]) -> dict[str, Any]:
    return {
        "bottleneck": int(bottleneck),
        "width": str(width),
        "noise": float(noise),
        "activation": str(getattr(model, "activation", "relu")),
        "coefs": [np.asarray(w, dtype=np.float32).tolist() for w in getattr(model, "coefs_", [])],
        "intercepts": [np.asarray(b, dtype=np.float32).tolist() for b in getattr(model, "intercepts_", [])],
        "metrics": {str(k): float(v) for k, v in dict(metrics).items()},
    }


def _forward(spec: Mapping[str, Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coefs = [np.asarray(w, dtype=np.float32) for w in spec.get("coefs", [])]
    intercepts = [np.asarray(b, dtype=np.float32) for b in spec.get("intercepts", [])]
    if not coefs or len(coefs) != len(intercepts):
        n = int(x.shape[0]) if x.ndim == 2 else 0
        b = int(spec.get("bottleneck", 1) or 1)
        return np.zeros((n, b), dtype=np.float32), np.zeros_like(x, dtype=np.float32)
    act_name = str(spec.get("activation", "relu"))
    h = np.asarray(x, dtype=np.float32)
    latent = None
    bottleneck_layer = 1
    for i, (w, b) in enumerate(zip(coefs, intercepts)):
        h = h @ w + b.reshape(1, -1)
        if i < len(coefs) - 1:
            h = _activation(h, act_name).astype(np.float32, copy=False)
            if i == bottleneck_layer:
                latent = h.copy()
    if latent is None:
        b = int(spec.get("bottleneck", 1) or 1)
        latent = np.zeros((x.shape[0], b), dtype=np.float32)
    return latent.astype(np.float32, copy=False), h.astype(np.float32, copy=False)


def _effective_rank(latent: np.ndarray) -> float:
    z = np.asarray(latent, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 3 or z.shape[1] == 0:
        return 0.0
    z = z - np.nanmean(z, axis=0, keepdims=True)
    cov = np.cov(np.nan_to_num(z, nan=0.0), rowvar=False)
    vals = np.linalg.eigvalsh(np.atleast_2d(cov))
    vals = np.clip(np.asarray(vals, dtype=np.float64), 0.0, None)
    total = float(vals.sum())
    if total <= 1e-12:
        return 0.0
    p = vals / total
    entropy = -float(np.sum(p[p > 0.0] * np.log(p[p > 0.0])))
    return float(np.exp(entropy))


def _rank_penalty(effective_rank: float, bottleneck: int) -> float:
    if int(bottleneck) >= 16:
        lo, hi = 8.0, 14.0
    else:
        lo, hi = 4.0, 8.0
    if effective_rank < lo:
        return float((lo - effective_rank) / max(lo, 1.0))
    if effective_rank > hi:
        return float((effective_rank - hi) / max(hi, 1.0))
    return 0.0


def _evaluate_spec(
    spec: Mapping[str, Any],
    x_val: np.ndarray,
    *,
    noise: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    latent, recon = _forward(spec, x_val)
    row_err = np.mean(np.square(recon - x_val), axis=1)
    val_recon = float(np.nanmean(row_err)) if row_err.size else 0.0
    noisy = np.clip(
        x_val + rng.normal(0.0, float(noise), size=x_val.shape).astype(np.float32),
        -12.0,
        12.0,
    )
    latent_noisy, recon_noisy = _forward(spec, noisy)
    sensitivity = float(np.nanmean(np.sqrt(np.sum(np.square(latent_noisy - latent), axis=1)))) if len(latent) else 0.0
    normal_mask = np.sqrt(np.sum(np.square(x_val), axis=1)) <= np.nanpercentile(
        np.sqrt(np.sum(np.square(x_val), axis=1)),
        60,
    )
    normal_err = row_err[normal_mask] if normal_mask.any() else row_err
    normal_vol = float(np.nanstd(normal_err)) if normal_err.size else 0.0
    radius = np.sqrt(np.sum(np.square(x_val), axis=1))
    ood = radius >= np.nanpercentile(radius, 90) if radius.size else np.zeros(0, dtype=bool)
    normal = radius <= np.nanpercentile(radius, 60) if radius.size else np.zeros(0, dtype=bool)
    ood_sep = 0.0
    if ood.any() and normal.any():
        ood_sep = float(np.nanmean(row_err[ood]) - np.nanmean(row_err[normal]))
    curve_errors: list[float] = []
    for curve_noise in (0.025, 0.05, 0.10, 0.15, 0.20):
        pert = np.clip(
            x_val + rng.normal(0.0, curve_noise, size=x_val.shape).astype(np.float32),
            -12.0,
            12.0,
        )
        _, curve_recon = _forward(spec, pert)
        curve_errors.append(float(np.nanmean(np.mean(np.square(curve_recon - x_val), axis=1))))
    curve_diffs = np.diff(np.asarray(curve_errors, dtype=np.float64))
    smooth_penalty = float(np.nanstd(curve_diffs)) if curve_diffs.size else 0.0
    temporal_consistency = (
        float(np.nanmean(np.sqrt(np.sum(np.square(np.diff(latent, axis=0)), axis=1))))
        if len(latent) > 1
        else 0.0
    )
    erank = _effective_rank(latent)
    return {
        "val_recon_loss": val_recon,
        "latent_noise_sensitivity": sensitivity,
        "normal_recon_error_volatility": normal_vol,
        "ood_separation": ood_sep,
        "effective_rank": erank,
        "rank_penalty": _rank_penalty(erank, int(spec.get("bottleneck", 0) or 0)),
        "noise_curve_smoothness_penalty": smooth_penalty,
        "temporal_consistency": temporal_consistency,
        "noisy_recon_loss": float(np.nanmean(np.mean(np.square(recon_noisy - x_val), axis=1))) if len(x_val) else 0.0,
    }


def _fit_one(
    x_train: np.ndarray,
    x_val: np.ndarray,
    *,
    bottleneck: int,
    width: str,
    noise: float,
    random_state: int,
    max_iter: int,
) -> dict[str, Any] | None:
    if MLPRegressor is None:
        return None
    rng = np.random.default_rng(int(random_state))
    noisy = np.clip(
        x_train + rng.normal(0.0, float(noise), size=x_train.shape).astype(np.float32),
        -12.0,
        12.0,
    )
    model = MLPRegressor(
        hidden_layer_sizes=_width_layers(int(bottleneck), str(width)),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=max(1, min(512, len(x_train))),
        learning_rate_init=1e-3,
        max_iter=int(max_iter),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=8,
        random_state=int(random_state),
        verbose=False,
    )
    try:
        model.fit(noisy, x_train)
    except Exception:
        return None
    spec = _model_to_spec(model, bottleneck=bottleneck, width=width, noise=noise, metrics={})
    metrics = _evaluate_spec(spec, x_val, noise=noise, rng=np.random.default_rng(int(random_state) + 17))
    spec["metrics"] = metrics
    return spec


def _score_width(metrics: Mapping[str, float]) -> float:
    return (
        0.50 * float(metrics.get("val_recon_loss", 0.0))
        + 0.30 * float(metrics.get("latent_noise_sensitivity", 0.0))
        + 0.20 * float(metrics.get("normal_recon_error_volatility", 0.0))
    )


def _score_final(metrics: Mapping[str, float]) -> float:
    return (
        0.35 * float(metrics.get("val_recon_loss", 0.0))
        + 0.20 * float(metrics.get("latent_noise_sensitivity", 0.0))
        + 0.15 * float(metrics.get("normal_recon_error_volatility", 0.0))
        - 0.20 * float(metrics.get("ood_separation", 0.0))
        + 0.10 * float(metrics.get("rank_penalty", 0.0))
    )


def fit_denoising_autoencoder_state(
    x_reference: np.ndarray,
    *,
    random_state: int = 42,
    max_train_rows: int = 5000,
    max_iter: int = 80,
) -> dict[str, Any]:
    """Fit bottleneck-8 and bottleneck-16 denoising AEs on compact numeric rows."""
    x = np.asarray(x_reference, dtype=np.float32)
    if MLPRegressor is None:
        return {"enabled": False, "reason": "sklearn_mlp_unavailable", "schema_version": "denoising_ae_v1"}
    if x.ndim != 2 or x.shape[0] < 200 or x.shape[1] < 2:
        return {"enabled": False, "reason": "insufficient_rows_or_features", "schema_version": "denoising_ae_v1"}
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if len(x) > int(max_train_rows):
        idx = np.linspace(0, len(x) - 1, int(max_train_rows)).round().astype(int)
        x = x[np.unique(idx)]
    split = int(max(50, min(len(x) - 50, math_floor(0.80 * len(x)))))
    if split <= 0 or split >= len(x):
        split = max(1, len(x) - max(1, len(x) // 5))
    x_train = x[:split]
    x_val = x[split:]
    if len(x_val) < 20:
        x_val = x_train[-min(len(x_train), 100):]
    models: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for bottleneck in AE_BOTTLENECKS:
        width_trials: list[dict[str, Any]] = []
        best_width = None
        best_width_score = float("inf")
        for width in AE_WIDTHS:
            spec = _fit_one(
                x_train,
                x_val,
                bottleneck=int(bottleneck),
                width=str(width),
                noise=0.10,
                random_state=int(random_state) + int(bottleneck) * 11 + len(str(width)),
                max_iter=int(max_iter),
            )
            if spec is None:
                continue
            score = _score_width(spec.get("metrics", {}))
            width_trials.append({"width": str(width), "noise": 0.10, "score": float(score), "metrics": spec.get("metrics", {})})
            if score < best_width_score:
                best_width_score = float(score)
                best_width = str(width)
        if best_width is None:
            reports[f"b{int(bottleneck)}"] = {"enabled": False, "reason": "no_width_model_fit"}
            continue
        noise_trials: list[dict[str, Any]] = []
        best_spec = None
        best_final_score = float("inf")
        for noise in AE_NOISE_LEVELS:
            spec = _fit_one(
                x_train,
                x_val,
                bottleneck=int(bottleneck),
                width=best_width,
                noise=float(noise),
                random_state=int(random_state) + int(bottleneck) * 37 + int(round(noise * 1000)),
                max_iter=int(max_iter),
            )
            if spec is None:
                continue
            score = _score_final(spec.get("metrics", {}))
            noise_trials.append({"width": best_width, "noise": float(noise), "score": float(score), "metrics": spec.get("metrics", {})})
            if score < best_final_score:
                best_final_score = float(score)
                best_spec = spec
        if best_spec is not None:
            models[f"b{int(bottleneck)}"] = best_spec
            reports[f"b{int(bottleneck)}"] = {
                "enabled": True,
                "best_width": best_width,
                "best_noise": float(best_spec.get("noise", 0.0)),
                "best_score": float(best_final_score),
                "width_trials": width_trials,
                "noise_trials": noise_trials,
                "metrics": best_spec.get("metrics", {}),
            }
    return {
        "enabled": bool(models),
        "schema_version": "denoising_ae_v1",
        "models": models,
        "report": reports,
        "input_dim": int(x_reference.shape[1]) if np.asarray(x_reference).ndim == 2 else 0,
    }


def math_floor(value: float) -> int:
    return int(np.floor(float(value)))


def transform_denoising_autoencoder_features(
    x: np.ndarray,
    state: Mapping[str, Any] | None,
    *,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        arr = np.zeros((0, 0), dtype=np.float32)
    idx = index if index is not None else pd.RangeIndex(arr.shape[0])
    out = pd.DataFrame(index=idx)
    for col in AE_FEATURE_COLUMNS:
        out[col] = np.zeros(arr.shape[0], dtype=np.float32)
    if not isinstance(state, Mapping) or not bool(state.get("enabled", False)):
        return out.astype(np.float32)
    models = state.get("models", {}) or {}
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    for bottleneck in AE_BOTTLENECKS:
        key = f"b{int(bottleneck)}"
        spec = models.get(key, {}) if isinstance(models, Mapping) else {}
        if not isinstance(spec, Mapping):
            continue
        latent, recon = _forward(spec, clean)
        if latent.shape[0] != len(out):
            continue
        dim = min(int(bottleneck), latent.shape[1])
        for i in range(dim):
            out[f"ae_b{int(bottleneck)}_{i:02d}"] = latent[:, i].astype(np.float32)
        if recon.shape == clean.shape:
            out[f"ae_b{int(bottleneck)}_reconstruction_error"] = np.mean(
                np.square(recon - clean),
                axis=1,
            ).astype(np.float32)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
