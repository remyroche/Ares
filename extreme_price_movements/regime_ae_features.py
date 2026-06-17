"""Current-regime autoencoder features for the RegimeAdaptor only."""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


CURRENT_REGIME_AE_LATENT_DIM = 8
CURRENT_REGIME_AE_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    [f"z_ae_{i}" for i in range(1, CURRENT_REGIME_AE_LATENT_DIM + 1)]
    + [
        "ae_reconstruction_error",
        "ae_reconstruction_error_percentile",
        "ae_latent_norm",
        "ae_latent_norm_percentile",
        "ae_latent_distance",
        "ae_latent_distance_percentile",
    ]
)


def _to_numeric_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    cols = [str(c) for c in columns if str(c) in frame.columns]
    out = pd.DataFrame(index=frame.index)
    for col in cols:
        out[col] = pd.to_numeric(frame[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _recent_window_mask(
    frame: pd.DataFrame,
    *,
    lookback_days: int,
) -> np.ndarray:
    n = len(frame)
    if n == 0 or "timestamp" not in frame.columns:
        return np.ones(n, dtype=bool)
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if not ts.notna().any():
        return np.ones(n, dtype=bool)
    end = ts.max()
    start = end - pd.Timedelta(days=max(1, int(lookback_days)))
    mask = (ts >= start) & (ts <= end)
    return mask.fillna(False).to_numpy(dtype=bool)


def _select_source_columns(
    reference: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    min_features: int,
    max_features: int,
) -> list[str]:
    requested = [str(c) for c in feature_columns if str(c)]
    seen: set[str] = set()
    cols: list[str] = []
    for col in requested:
        if col in seen or col not in reference.columns:
            continue
        values = pd.to_numeric(reference[col], errors="coerce")
        if values.notna().sum() < 3:
            continue
        if float(values.replace([np.inf, -np.inf], np.nan).std(skipna=True) or 0.0) <= 1e-12:
            continue
        seen.add(col)
        cols.append(col)
    if len(cols) <= max_features:
        return cols
    scored: list[tuple[float, int, str]] = []
    for i, col in enumerate(cols):
        values = pd.to_numeric(reference[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        missing = float(values.isna().mean())
        variance = float(values.var(skipna=True) or 0.0)
        scored.append((missing - min(np.log1p(max(variance, 0.0)), 5.0) * 1e-3, i, col))
    scored.sort()
    selected = [col for _score, _i, col in scored[:max_features]]
    selected_set = set(selected)
    return [col for col in cols if col in selected_set][: max(max_features, min_features)]


def _fit_preprocessor(x: pd.DataFrame, *, clip: tuple[float, float]) -> dict[str, Any]:
    arr = x.to_numpy(dtype=np.float32, copy=True)
    center = np.nanmedian(arr, axis=0).astype(np.float32)
    center = np.where(np.isfinite(center), center, 0.0).astype(np.float32)
    centered = np.abs(arr - center.reshape(1, -1))
    mad = np.nanmedian(centered, axis=0).astype(np.float32)
    scale = (1.4826 * mad).astype(np.float32)
    q25 = np.nanpercentile(arr, 25.0, axis=0).astype(np.float32)
    q75 = np.nanpercentile(arr, 75.0, axis=0).astype(np.float32)
    iqr_scale = ((q75 - q25) / 1.349).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, iqr_scale)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
    missing_rates = np.mean(~np.isfinite(arr), axis=0).astype(np.float32)
    missing_indicator_columns = [
        str(col) for col, rate in zip(x.columns, missing_rates) if float(rate) > 0.0
    ]
    return {
        "center": center.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
        "clip": [float(clip[0]), float(clip[1])],
        "missing_indicator_columns": missing_indicator_columns,
        "missing_rates": {
            str(col): float(rate) for col, rate in zip(x.columns, missing_rates)
        },
    }


def _apply_preprocessor(
    x: pd.DataFrame,
    state: Mapping[str, Any],
) -> np.ndarray:
    feature_columns = [str(c) for c in state.get("feature_columns", [])]
    x_aligned = _to_numeric_frame(x, feature_columns).reindex(
        columns=feature_columns,
        fill_value=np.nan,
    )
    arr = x_aligned.to_numpy(dtype=np.float32, copy=True)
    center = np.asarray(state.get("center", np.zeros(len(feature_columns))), dtype=np.float32)
    scale = np.asarray(state.get("scale", np.ones(len(feature_columns))), dtype=np.float32)
    if center.size != len(feature_columns):
        center = np.zeros(len(feature_columns), dtype=np.float32)
    if scale.size != len(feature_columns):
        scale = np.ones(len(feature_columns), dtype=np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
    missing = ~np.isfinite(arr)
    filled = np.where(missing, center.reshape(1, -1), arr).astype(np.float32)
    z = (filled - center.reshape(1, -1)) / scale.reshape(1, -1)
    lo, hi = state.get("clip", [-5.0, 5.0])
    z = np.clip(z, float(lo), float(hi)).astype(np.float32)
    missing_indicator_columns = [str(c) for c in state.get("missing_indicator_columns", [])]
    if missing_indicator_columns:
        col_pos = {col: i for i, col in enumerate(feature_columns)}
        indicators = [
            missing[:, col_pos[col]].astype(np.float32)
            for col in missing_indicator_columns
            if col in col_pos
        ]
        if indicators:
            z = np.concatenate([z, np.column_stack(indicators).astype(np.float32)], axis=1)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _huber_row_error(recon: np.ndarray, target: np.ndarray, delta: float = 1.0) -> np.ndarray:
    diff = np.asarray(recon, dtype=np.float32) - np.asarray(target, dtype=np.float32)
    abs_diff = np.abs(diff)
    loss = np.where(
        abs_diff <= float(delta),
        0.5 * diff * diff,
        float(delta) * (abs_diff - 0.5 * float(delta)),
    )
    return np.mean(loss, axis=1).astype(np.float32)


def _quantiles(values: np.ndarray) -> list[float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return [0.0, 1.0]
    q = np.quantile(vals, np.linspace(0.0, 1.0, 101))
    q = np.maximum.accumulate(np.asarray(q, dtype=np.float64))
    return q.astype(float).tolist()


def _percentile_from_quantiles(values: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    q = np.asarray(list(quantiles or [0.0, 1.0]), dtype=np.float64)
    if q.size < 2 or not np.isfinite(q).any():
        return np.full(len(values), 0.5, dtype=np.float32)
    q = np.maximum.accumulate(np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0))
    grid = np.linspace(0.0, 1.0, q.size)
    vals = np.asarray(values, dtype=np.float64)
    return np.interp(vals, q, grid, left=0.0, right=1.0).astype(np.float32)


def _fit_pca_backend(x: np.ndarray) -> dict[str, Any]:
    try:
        from sklearn.decomposition import PCA
    except Exception:
        return {"enabled": False, "backend": "none", "reason": "pca_unavailable"}
    n_components = min(CURRENT_REGIME_AE_LATENT_DIM, x.shape[0], x.shape[1])
    if n_components < 1:
        return {"enabled": False, "backend": "none", "reason": "empty_matrix"}
    pca = PCA(n_components=n_components, random_state=42)
    z = pca.fit_transform(x)
    recon = pca.inverse_transform(z)
    row_err = _huber_row_error(np.asarray(recon, dtype=np.float32), x)
    return {
        "enabled": True,
        "backend": "pca_fallback",
        "mean": np.asarray(pca.mean_, dtype=float).tolist(),
        "components": np.asarray(pca.components_, dtype=float).tolist(),
        "explained_variance_ratio": np.asarray(
            pca.explained_variance_ratio_, dtype=float
        ).tolist(),
        "latent_dim": int(n_components),
        "_fit_latent": np.asarray(z, dtype=np.float32),
        "_fit_row_error": np.asarray(row_err, dtype=np.float32),
    }


def _sklearn_activation(x: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return x
    if name == "tanh":
        return np.tanh(x)
    if name == "logistic":
        return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))
    return np.maximum(x, 0.0)


def _sklearn_forward(state: Mapping[str, Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coefs = [np.asarray(w, dtype=np.float32) for w in state.get("coefs", [])]
    intercepts = [np.asarray(b, dtype=np.float32) for b in state.get("intercepts", [])]
    d = int(state.get("input_dim", x.shape[1]) or x.shape[1])
    if not coefs or len(coefs) != len(intercepts):
        z = np.zeros((x.shape[0], CURRENT_REGIME_AE_LATENT_DIM), dtype=np.float32)
        return z, np.zeros((x.shape[0], d), dtype=np.float32)
    h = np.asarray(x, dtype=np.float32)
    latent = None
    activation = str(state.get("activation", "relu"))
    bottleneck_layer = int(state.get("bottleneck_layer", 2) or 2)
    for i, (w, b) in enumerate(zip(coefs, intercepts)):
        h = h @ w + b.reshape(1, -1)
        if i < len(coefs) - 1:
            h = _sklearn_activation(h, activation).astype(np.float32, copy=False)
            if i == bottleneck_layer:
                latent = h.copy()
    if latent is None:
        latent = np.zeros((x.shape[0], CURRENT_REGIME_AE_LATENT_DIM), dtype=np.float32)
    z = np.zeros((x.shape[0], CURRENT_REGIME_AE_LATENT_DIM), dtype=np.float32)
    dim = min(z.shape[1], latent.shape[1])
    z[:, :dim] = latent[:, :dim].astype(np.float32)
    recon = h[:, :d] if h.ndim == 2 and h.shape[1] >= d else np.zeros((x.shape[0], d), dtype=np.float32)
    return z.astype(np.float32), recon.astype(np.float32)


def _fit_sklearn_mlp_backend(
    x: np.ndarray,
    *,
    score_target: np.ndarray | None,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    input_noise_std: float,
    supervised_weight: float,
    rank_weight: float,
    weight_decay: float,
    random_state: int,
) -> dict[str, Any]:
    try:
        from sklearn.neural_network import MLPRegressor
        try:
            from sklearn.exceptions import ConvergenceWarning
        except Exception:  # pragma: no cover - old sklearn fallback
            ConvergenceWarning = Warning
    except Exception as exc:
        out = _fit_pca_backend(x)
        out["sklearn_error"] = str(exc)
        return out
    rng = np.random.default_rng(int(random_state))
    target = np.asarray(x, dtype=np.float32)
    supervised_enabled = (
        score_target is not None
        and len(score_target) == len(x)
        and np.isfinite(score_target).any()
        and (float(supervised_weight) > 0.0 or float(rank_weight) > 0.0)
    )
    if supervised_enabled:
        y = np.asarray(score_target, dtype=np.float32)
        fill = float(np.nanmedian(y[np.isfinite(y)]))
        y = np.clip(np.nan_to_num(y, nan=fill, posinf=fill, neginf=fill), 0.0, 1.0)
        order = np.argsort(y)
        ranks = np.empty_like(y, dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, len(y), dtype=np.float32)
        extras = []
        if float(supervised_weight) > 0.0:
            extras.append(np.sqrt(float(supervised_weight)) * y.reshape(-1, 1))
        if float(rank_weight) > 0.0:
            extras.append(np.sqrt(float(rank_weight)) * ranks.reshape(-1, 1))
        if extras:
            target = np.concatenate([target, *extras], axis=1).astype(np.float32)
    n = int(x.shape[0])
    split = int(np.floor(0.85 * n))
    split = min(max(split, 1), n - 1) if n > 1 else n
    has_validation = bool(0 < split < n)
    x_train = x[:split] if has_validation else x
    target_train = target[:split] if has_validation else target
    x_valid = x[split:] if has_validation else x
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, CURRENT_REGIME_AE_LATENT_DIM, 64, 128),
        activation="relu",
        solver="adam",
        alpha=float(weight_decay),
        batch_size=max(1, min(int(batch_size), len(x_train))),
        learning_rate_init=float(learning_rate),
        max_iter=1,
        warm_start=True,
        early_stopping=False,
        shuffle=True,
        n_iter_no_change=max(2, int(max_epochs) + 1),
        random_state=int(random_state),
        verbose=False,
    )
    best_state = None
    best_validation_loss = float("inf")
    stale = 0
    epochs_done = 0
    try:
        for epoch in range(max(1, int(max_epochs))):
            noisy_train = np.clip(
                x_train
                + rng.normal(0.0, float(input_noise_std), size=x_train.shape).astype(
                    np.float32
                ),
                -5.0,
                5.0,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(noisy_train, target_train)
            tmp_state = {
                "enabled": True,
                "backend": "sklearn_mlp",
                "input_dim": int(x.shape[1]),
                "latent_dim": CURRENT_REGIME_AE_LATENT_DIM,
                "coefs": [
                    np.asarray(w, dtype=np.float32).astype(float).tolist()
                    for w in model.coefs_
                ],
                "intercepts": [
                    np.asarray(b, dtype=np.float32).astype(float).tolist()
                    for b in model.intercepts_
                ],
                "activation": str(getattr(model, "activation", "relu")),
                "bottleneck_layer": 2,
            }
            _z_val, recon_val = _sklearn_forward(tmp_state, x_valid)
            val_loss = float(np.nanmean(_huber_row_error(recon_val, x_valid)))
            epochs_done = epoch + 1
            if val_loss + 1e-7 < best_validation_loss:
                best_validation_loss = val_loss
                best_state = {
                    "coefs": tmp_state["coefs"],
                    "intercepts": tmp_state["intercepts"],
                }
                stale = 0
            else:
                stale += 1
            if stale >= 5:
                break
    except Exception as exc:
        out = _fit_pca_backend(x)
        out["sklearn_error"] = str(exc)
        return out
    coefs = best_state["coefs"] if best_state is not None else [
        np.asarray(w, dtype=np.float32).astype(float).tolist()
        for w in model.coefs_
    ]
    intercepts = best_state["intercepts"] if best_state is not None else [
        np.asarray(b, dtype=np.float32).astype(float).tolist()
        for b in model.intercepts_
    ]
    state = {
        "enabled": True,
        "backend": "sklearn_mlp",
        "backend_family": "bottleneck_mlp_autoencoder_approximation",
        "input_dim": int(x.shape[1]),
        "latent_dim": CURRENT_REGIME_AE_LATENT_DIM,
        "coefs": coefs,
        "intercepts": intercepts,
        "activation": str(getattr(model, "activation", "relu")),
        "bottleneck_layer": 2,
        "epochs": int(epochs_done),
        "best_validation_reconstruction_loss": float(best_validation_loss),
        "validation_metric_name": "time_ordered_huber_reconstruction_loss",
        "validation_policy": "last_15_percent_by_time",
        "supervised_head_enabled": bool(supervised_enabled),
    }
    z, recon = _sklearn_forward(state, x)
    row_err = _huber_row_error(recon, x) if recon.shape == x.shape else np.zeros(len(x), dtype=np.float32)
    state["_fit_latent"] = z
    state["_fit_row_error"] = np.asarray(row_err, dtype=np.float32)
    return state


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _fit_torch_backend(
    x: np.ndarray,
    *,
    score_target: np.ndarray | None,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    min_learning_rate: float,
    input_noise_std: float,
    supervised_weight: float,
    rank_weight: float,
    latent_l1: float,
    latent_stability: float,
    weight_decay: float,
    random_state: int,
) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        out = _fit_pca_backend(x)
        out["torch_error"] = str(exc)
        return out

    torch.manual_seed(int(random_state))
    device = torch.device("cpu")
    d = int(x.shape[1])
    latent_dim = CURRENT_REGIME_AE_LATENT_DIM

    class _RegimeAE(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.GELU(),
                nn.Dropout(0.20),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(64, latent_dim),
            )
            self.latent_dropout = nn.Dropout(0.05)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(64, 128),
                nn.GELU(),
                nn.Linear(128, input_dim),
            )
            self.head = nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )

        def forward(self, batch):
            z = self.encoder(batch)
            recon = self.decoder(self.latent_dropout(z))
            pred = self.head(z).reshape(-1)
            return z, recon, pred

    n = int(x.shape[0])
    split = int(np.floor(0.85 * n))
    split = min(max(split, max(1, n - max(1, int(0.15 * n)))), n - 1)
    if split <= 0 or split >= n:
        split = max(1, n - 1)
    x_train = x[:split]
    x_valid = x[split:]
    y_train = None
    y_valid = None
    rank_train = None
    rank_valid = None
    if score_target is not None and len(score_target) == n and np.isfinite(score_target).any():
        y = np.asarray(score_target, dtype=np.float32)
        fill = float(np.nanmedian(y[np.isfinite(y)]))
        y = np.nan_to_num(y, nan=fill, posinf=fill, neginf=fill)
        y = np.clip(y, 0.0, 1.0).astype(np.float32)
        order = np.argsort(y)
        ranks = np.empty_like(y, dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, len(y), dtype=np.float32)
        y_train, y_valid = y[:split], y[split:]
        rank_train, rank_valid = ranks[:split], ranks[split:]

    train_tensors = [torch.as_tensor(x_train, dtype=torch.float32)]
    if y_train is not None and rank_train is not None:
        train_tensors.extend(
            [
                torch.as_tensor(y_train, dtype=torch.float32),
                torch.as_tensor(rank_train, dtype=torch.float32),
            ]
        )
    else:
        train_tensors.extend(
            [
                torch.zeros(len(x_train), dtype=torch.float32),
                torch.zeros(len(x_train), dtype=torch.float32),
            ]
        )
    loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=max(1, min(int(batch_size), len(x_train))),
        shuffle=True,
        drop_last=False,
    )
    model = _RegimeAE(d).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=float(min_learning_rate),
    )
    huber = nn.HuberLoss(delta=1.0, reduction="mean")
    x_valid_tensor = torch.as_tensor(x_valid, dtype=torch.float32, device=device)
    y_valid_tensor = (
        torch.as_tensor(y_valid, dtype=torch.float32, device=device)
        if y_valid is not None
        else None
    )
    rank_valid_tensor = (
        torch.as_tensor(rank_valid, dtype=torch.float32, device=device)
        if rank_valid is not None
        else None
    )
    best_state = None
    best_loss = float("inf")
    patience = 5
    stale = 0
    epochs_done = 0
    supervised_enabled = y_train is not None and float(supervised_weight) > 0.0
    for epoch in range(max(1, int(max_epochs))):
        model.train()
        for xb, yb, rb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            rb = rb.to(device)
            noisy = xb + torch.randn_like(xb) * float(input_noise_std)
            noisy = torch.clamp(noisy, -5.0, 5.0)
            z, recon, pred = model(noisy)
            loss = huber(recon, xb)
            if supervised_enabled:
                loss = loss + float(supervised_weight) * huber(pred, yb)
                loss = loss + float(rank_weight) * huber(pred, rb)
            loss = loss + float(latent_l1) * torch.mean(torch.abs(z))
            if z.shape[0] > 1:
                loss = loss + float(latent_stability) * torch.mean((z[1:] - z[:-1]) ** 2)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        model.eval()
        with torch.no_grad():
            z_val, recon_val, pred_val = model(x_valid_tensor)
            val_loss = huber(recon_val, x_valid_tensor)
            if supervised_enabled and y_valid_tensor is not None and rank_valid_tensor is not None:
                val_loss = val_loss + float(supervised_weight) * huber(pred_val, y_valid_tensor)
                val_loss = val_loss + float(rank_weight) * huber(pred_val, rank_valid_tensor)
            val = float(val_loss.detach().cpu().item())
        scheduler.step(val)
        epochs_done = epoch + 1
        if val + 1e-7 < best_loss:
            best_loss = val
            best_state = {
                k: v.detach().cpu().numpy().astype(np.float32)
                for k, v in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict({k: torch.as_tensor(v) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        z_all, recon_all, _pred_all = model(x_tensor)
    z_np = z_all.detach().cpu().numpy().astype(np.float32)
    recon_np = recon_all.detach().cpu().numpy().astype(np.float32)
    row_err = _huber_row_error(recon_np, x) if recon_np.shape == x.shape else np.zeros(len(x), dtype=np.float32)
    del recon_np
    state_dict = {
        k: np.asarray(v, dtype=np.float32).astype(float).tolist()
        for k, v in (best_state or model.state_dict()).items()
    }
    return {
        "enabled": True,
        "backend": "torch_mlp",
        "input_dim": d,
        "latent_dim": latent_dim,
        "state_dict": state_dict,
        "best_validation_loss": float(best_loss),
        "validation_metric_name": "time_ordered_huber_loss",
        "validation_policy": "last_15_percent_by_time",
        "epochs": int(epochs_done),
        "supervised_head_enabled": bool(supervised_enabled),
        "_fit_latent": z_np,
        "_fit_row_error": np.asarray(row_err, dtype=np.float32),
    }


def _torch_forward(state: Mapping[str, Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import torch
        from torch import nn
    except Exception:
        return None
    d = int(state.get("input_dim", x.shape[1]) or x.shape[1])
    latent_dim = int(state.get("latent_dim", CURRENT_REGIME_AE_LATENT_DIM) or CURRENT_REGIME_AE_LATENT_DIM)

    class _RegimeAE(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.GELU(),
                nn.Dropout(0.20),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(64, latent_dim),
            )
            self.latent_dropout = nn.Dropout(0.05)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(64, 128),
                nn.GELU(),
                nn.Linear(128, input_dim),
            )
            self.head = nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )

        def forward(self, batch):
            z = self.encoder(batch)
            return z, self.decoder(z)

    model = _RegimeAE(d)
    raw_state = {}
    for key, value in dict(state.get("state_dict", {}) or {}).items():
        raw_state[str(key)] = torch.as_tensor(np.asarray(value, dtype=np.float32))
    try:
        model.load_state_dict(raw_state, strict=False)
    except Exception:
        return None
    model.eval()
    with torch.no_grad():
        z, recon = model(torch.as_tensor(x, dtype=torch.float32))
    return z.detach().cpu().numpy().astype(np.float32), recon.detach().cpu().numpy().astype(np.float32)


def _pca_forward(state: Mapping[str, Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(state.get("mean", np.zeros(x.shape[1])), dtype=np.float32)
    components = np.asarray(state.get("components", []), dtype=np.float32)
    if mean.size != x.shape[1] or components.ndim != 2 or components.shape[1] != x.shape[1]:
        z = np.zeros((x.shape[0], CURRENT_REGIME_AE_LATENT_DIM), dtype=np.float32)
        return z, np.zeros_like(x, dtype=np.float32)
    centered = x - mean.reshape(1, -1)
    z_raw = centered @ components.T
    recon = z_raw @ components + mean.reshape(1, -1)
    z = np.zeros((x.shape[0], CURRENT_REGIME_AE_LATENT_DIM), dtype=np.float32)
    dim = min(z.shape[1], z_raw.shape[1])
    z[:, :dim] = z_raw[:, :dim].astype(np.float32)
    return z.astype(np.float32), recon.astype(np.float32)


def fit_current_regime_ae_state(
    reference_frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    score_target: Sequence[float] | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(cfg or {})
    if reference_frame is None or reference_frame.empty:
        return {"enabled": False, "schema_version": "ae_current_regime_v1", "reason": "empty_reference_frame"}
    lookback_days = int(cfg.get("regime_ae_lookback_days", 92) or 92)
    min_rows = int(cfg.get("regime_ae_min_rows", 200) or 200)
    max_rows = int(cfg.get("regime_ae_max_train_rows", 30000) or 30000)
    min_features = int(cfg.get("regime_ae_min_features", 8) or 8)
    max_features = int(cfg.get("regime_ae_max_features", 180) or 180)
    clip = tuple(cfg.get("regime_ae_clip", [-5.0, 5.0]) or [-5.0, 5.0])
    allow_full_history_fallback = bool(
        cfg.get("regime_ae_allow_full_history_fallback", False)
    )
    recent_mask = _recent_window_mask(reference_frame, lookback_days=lookback_days)
    fit_frame = reference_frame.loc[recent_mask].copy()
    target = None
    if score_target is not None and len(score_target) == len(reference_frame):
        target = np.asarray(score_target, dtype=np.float32)[recent_mask]
    if len(fit_frame) < min_rows:
        if not allow_full_history_fallback:
            return {
                "enabled": False,
                "schema_version": "ae_current_regime_v1",
                "reason": "insufficient_recent_rows",
                "recent_rows": int(len(fit_frame)),
                "min_rows": int(min_rows),
                "lookback_days": int(lookback_days),
                "requested_feature_count": int(len(list(feature_columns or []))),
            }
        fit_frame = reference_frame.copy()
        if score_target is not None and len(score_target) == len(reference_frame):
            target = np.asarray(score_target, dtype=np.float32)
    cols = _select_source_columns(
        fit_frame,
        feature_columns,
        min_features=min_features,
        max_features=max_features,
    )
    min_required_features = max(2, int(min_features))
    if len(fit_frame) < max(20, min_rows) or len(cols) < min_required_features:
        return {
            "enabled": False,
            "schema_version": "ae_current_regime_v1",
            "reason": "insufficient_rows_or_features",
            "rows": int(len(fit_frame)),
            "feature_count": int(len(cols)),
            "min_features": int(min_required_features),
            "requested_feature_count": int(len(list(feature_columns or []))),
        }
    if "timestamp" in fit_frame.columns:
        fit_ts = pd.to_datetime(fit_frame["timestamp"], utc=True, errors="coerce")
        sort_key = fit_ts.fillna(pd.Timestamp.min.tz_localize("UTC")).to_numpy()
        order = np.argsort(sort_key, kind="mergesort")
        fit_frame = fit_frame.iloc[order].copy()
        if target is not None and len(target) == len(order):
            target = np.asarray(target, dtype=np.float32)[order]
    if max_rows > 0 and len(fit_frame) > max_rows:
        idx = np.linspace(0, len(fit_frame) - 1, max_rows).round().astype(int)
        idx = np.unique(idx)
        fit_frame = fit_frame.iloc[idx].copy()
        if target is not None and len(target) >= int(np.max(idx)) + 1:
            target = target[idx]
    x_fit = _to_numeric_frame(fit_frame, cols)
    pre = _fit_preprocessor(x_fit, clip=(float(clip[0]), float(clip[1])))
    state: dict[str, Any] = {
        "enabled": True,
        "schema_version": "ae_current_regime_v1",
        "feature_columns": cols,
        "source_feature_count": int(len(cols)),
        "lookback_days": int(lookback_days),
        "fit_rows": int(len(fit_frame)),
        "preprocessing": {
            "scaling": "robust_z_score",
            "center": "fit_window_median",
            "scale": "fit_window_MAD_with_iqr_std_fallback",
            "missing_values": "median_impute_plus_missing_indicator",
        },
        **pre,
    }
    x_proc = _apply_preprocessor(x_fit, state)
    if target is not None and len(target) != len(fit_frame):
        target = target[: len(fit_frame)]
    backend_name = str(cfg.get("regime_ae_backend", "sklearn_mlp") or "sklearn_mlp")
    common_backend_kwargs = {
        "score_target": target,
        "max_epochs": int(cfg.get("regime_ae_max_epochs", 50) or 50),
        "batch_size": int(cfg.get("regime_ae_batch_size", 8192) or 8192),
        "learning_rate": float(cfg.get("regime_ae_learning_rate", 1e-3) or 1e-3),
        "input_noise_std": float(cfg.get("regime_ae_input_noise_std", 0.03) or 0.03),
        "supervised_weight": float(cfg.get("regime_ae_alpha_loss_weight", 0.25) or 0.25),
        "rank_weight": float(cfg.get("regime_ae_rank_loss_weight", 0.25) or 0.25),
        "weight_decay": float(cfg.get("regime_ae_weight_decay", 1e-4) or 1e-4),
        "random_state": int(cfg.get("regime_ae_random_state", 42) or 42),
    }
    if backend_name == "torch_mlp":
        backend = _fit_torch_backend(
            x_proc,
            min_learning_rate=float(
                cfg.get("regime_ae_min_learning_rate", 1e-5) or 1e-5
            ),
            latent_l1=float(cfg.get("regime_ae_latent_l1", 1e-4) or 1e-4),
            latent_stability=float(
                cfg.get("regime_ae_latent_stability", 0.005) or 0.005
            ),
            **common_backend_kwargs,
        )
    else:
        backend = _fit_sklearn_mlp_backend(x_proc, **common_backend_kwargs)
    latent = np.asarray(backend.pop("_fit_latent", np.zeros((len(x_proc), CURRENT_REGIME_AE_LATENT_DIM))), dtype=np.float32)
    row_err_raw = backend.pop("_fit_row_error", None)
    recon = backend.pop("_fit_recon", None)
    state.update({k: v for k, v in backend.items() if not str(k).startswith("_")})
    if latent.ndim != 2 or latent.shape[1] < CURRENT_REGIME_AE_LATENT_DIM:
        padded = np.zeros((len(x_proc), CURRENT_REGIME_AE_LATENT_DIM), dtype=np.float32)
        if latent.ndim == 2 and latent.shape[0] == len(x_proc):
            padded[:, : min(padded.shape[1], latent.shape[1])] = latent[:, : min(padded.shape[1], latent.shape[1])]
        latent = padded
    latent = latent[:, :CURRENT_REGIME_AE_LATENT_DIM].astype(np.float32)
    if row_err_raw is not None:
        row_err = np.asarray(row_err_raw, dtype=np.float32)
        if row_err.shape[0] != len(x_proc):
            row_err = np.zeros(len(x_proc), dtype=np.float32)
    else:
        recon_arr = np.asarray(recon, dtype=np.float32) if recon is not None else np.zeros((0, 0), dtype=np.float32)
        row_err = _huber_row_error(recon_arr, x_proc) if recon_arr.shape == x_proc.shape else np.zeros(len(x_proc), dtype=np.float32)
        del recon_arr
    del recon
    latent_norm = np.sqrt(np.sum(latent * latent, axis=1)).astype(np.float32)
    latent_center = np.nanmedian(latent, axis=0).astype(np.float32)
    latent_scale = 1.4826 * np.nanmedian(np.abs(latent - latent_center.reshape(1, -1)), axis=0).astype(np.float32)
    latent_scale = np.where(np.isfinite(latent_scale) & (latent_scale > 1e-6), latent_scale, 1.0).astype(np.float32)
    latent_distance = np.sqrt(
        np.sum(np.square((latent - latent_center.reshape(1, -1)) / latent_scale.reshape(1, -1)), axis=1)
    ).astype(np.float32)
    state.update(
        {
            "latent_center": latent_center.astype(float).tolist(),
            "latent_scale": latent_scale.astype(float).tolist(),
            "reconstruction_error_quantiles": _quantiles(row_err),
            "latent_norm_quantiles": _quantiles(latent_norm),
            "latent_distance_quantiles": _quantiles(latent_distance),
            "feature_columns_after_missing_indicators": int(x_proc.shape[1]),
            "training_summary": {
                "backend": str(state.get("backend", "")),
                "backend_family": str(state.get("backend_family", "")),
                "fit_rows": int(len(x_proc)),
                "source_feature_count": int(len(cols)),
                "input_dim": int(x_proc.shape[1]),
                "reconstruction_error_mean": float(np.nanmean(row_err)),
                "latent_norm_mean": float(np.nanmean(latent_norm)),
                "latent_distance_mean": float(np.nanmean(latent_distance)),
                "torch_available": bool(_torch_available()),
            },
        }
    )
    return state


def transform_current_regime_ae_features(
    frame: pd.DataFrame,
    state: Mapping[str, Any] | None,
    *,
    index: Any = None,
) -> pd.DataFrame:
    idx = frame.index if index is None else index
    out = pd.DataFrame(0.0, index=idx, columns=CURRENT_REGIME_AE_FEATURE_COLUMNS, dtype=np.float32)
    if not isinstance(state, Mapping) or not bool(state.get("enabled", False)):
        return out
    x_proc = _apply_preprocessor(frame, state)
    backend = str(state.get("backend", "") or "")
    if backend == "torch_mlp":
        forwarded = _torch_forward(state, x_proc)
        if forwarded is None:
            z, recon = _pca_forward(state, x_proc)
        else:
            z, recon = forwarded
    elif backend == "sklearn_mlp":
        z, recon = _sklearn_forward(state, x_proc)
    else:
        z, recon = _pca_forward(state, x_proc)
    if z.ndim != 2:
        z = np.zeros((len(frame), CURRENT_REGIME_AE_LATENT_DIM), dtype=np.float32)
    if z.shape[1] < CURRENT_REGIME_AE_LATENT_DIM:
        padded = np.zeros((z.shape[0], CURRENT_REGIME_AE_LATENT_DIM), dtype=np.float32)
        padded[:, : z.shape[1]] = z
        z = padded
    z = z[:, :CURRENT_REGIME_AE_LATENT_DIM].astype(np.float32)
    for i in range(CURRENT_REGIME_AE_LATENT_DIM):
        out[f"z_ae_{i + 1}"] = z[:, i].astype(np.float32)
    row_err = (
        _huber_row_error(recon, x_proc)
        if np.asarray(recon).shape == x_proc.shape
        else np.zeros(len(frame), dtype=np.float32)
    )
    latent_norm = np.sqrt(np.sum(z * z, axis=1)).astype(np.float32)
    latent_center = np.asarray(state.get("latent_center", np.zeros(CURRENT_REGIME_AE_LATENT_DIM)), dtype=np.float32)
    latent_scale = np.asarray(state.get("latent_scale", np.ones(CURRENT_REGIME_AE_LATENT_DIM)), dtype=np.float32)
    if latent_center.size != CURRENT_REGIME_AE_LATENT_DIM:
        latent_center = np.zeros(CURRENT_REGIME_AE_LATENT_DIM, dtype=np.float32)
    if latent_scale.size != CURRENT_REGIME_AE_LATENT_DIM:
        latent_scale = np.ones(CURRENT_REGIME_AE_LATENT_DIM, dtype=np.float32)
    latent_scale = np.where(np.isfinite(latent_scale) & (latent_scale > 1e-6), latent_scale, 1.0).astype(np.float32)
    latent_distance = np.sqrt(
        np.sum(np.square((z - latent_center.reshape(1, -1)) / latent_scale.reshape(1, -1)), axis=1)
    ).astype(np.float32)
    out["ae_reconstruction_error"] = row_err.astype(np.float32)
    out["ae_reconstruction_error_percentile"] = _percentile_from_quantiles(
        row_err,
        state.get("reconstruction_error_quantiles", []),
    )
    out["ae_latent_norm"] = latent_norm.astype(np.float32)
    out["ae_latent_norm_percentile"] = _percentile_from_quantiles(
        latent_norm,
        state.get("latent_norm_quantiles", []),
    )
    out["ae_latent_distance"] = latent_distance.astype(np.float32)
    out["ae_latent_distance_percentile"] = _percentile_from_quantiles(
        latent_distance,
        state.get("latent_distance_quantiles", []),
    )
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def fit_transform_current_regime_ae_features(
    reference_frame: pd.DataFrame,
    transform_frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    score_target: Sequence[float] | None = None,
    cfg: Mapping[str, Any] | None = None,
    index: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    state = fit_current_regime_ae_state(
        reference_frame,
        feature_columns=feature_columns,
        score_target=score_target,
        cfg=cfg,
    )
    return transform_current_regime_ae_features(
        transform_frame,
        state,
        index=index,
    ), state


def fit_transform_current_regime_ae_features_walk_forward(
    reference_frame: pd.DataFrame,
    transform_frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    score_target: Sequence[float] | None = None,
    cfg: Mapping[str, Any] | None = None,
    index: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = dict(cfg or {})
    idx = transform_frame.index if index is None else index
    out = pd.DataFrame(
        0.0,
        index=idx,
        columns=CURRENT_REGIME_AE_FEATURE_COLUMNS,
        dtype=np.float32,
    )
    final_state = fit_current_regime_ae_state(
        reference_frame,
        feature_columns=feature_columns,
        score_target=score_target,
        cfg=cfg,
    )
    diagnostics: dict[str, Any] = {
        "mode": "walk_forward_prior_only",
        "enabled_blocks": 0,
        "disabled_blocks": 0,
        "transformed_rows": int(len(transform_frame)),
        "reference_rows": int(len(reference_frame)),
    }
    if (
        reference_frame is None
        or reference_frame.empty
        or transform_frame is None
        or transform_frame.empty
        or "timestamp" not in reference_frame.columns
        or "timestamp" not in transform_frame.columns
    ):
        diagnostics["reason"] = "missing_timestamp_for_walk_forward"
        final_state = dict(final_state or {})
        final_state["candidate_generation"] = diagnostics
        return out, final_state

    ref_ts = pd.to_datetime(reference_frame["timestamp"], utc=True, errors="coerce")
    trans_ts = pd.to_datetime(transform_frame["timestamp"], utc=True, errors="coerce")
    valid_trans = trans_ts.notna().to_numpy(dtype=bool)
    if not bool(valid_trans.any()):
        diagnostics["reason"] = "no_valid_transform_timestamps"
        final_state = dict(final_state or {})
        final_state["candidate_generation"] = diagnostics
        return out, final_state

    block_hours = float(cfg.get("regime_ae_oof_block_hours", 168.0) or 168.0)
    block_hours = max(1.0, block_hours)
    min_prior_rows = int(
        cfg.get("regime_ae_walk_forward_min_prior_rows", cfg.get("regime_ae_min_rows", 200))
        or cfg.get("regime_ae_min_rows", 200)
        or 200
    )
    trans_min = trans_ts[valid_trans].min()
    hours_from_start = (
        (trans_ts - trans_min).dt.total_seconds().to_numpy(dtype=np.float64) / 3600.0
    )
    hours_from_start = np.nan_to_num(hours_from_start, nan=-1.0, posinf=-1.0, neginf=-1.0)
    block_ids = np.floor(np.maximum(hours_from_start, 0.0) / block_hours).astype(int)
    score_arr = (
        np.asarray(score_target, dtype=np.float32)
        if score_target is not None and len(score_target) == len(reference_frame)
        else None
    )
    block_reports: list[dict[str, Any]] = []
    for block_id in sorted(set(block_ids[valid_trans])):
        block_mask = valid_trans & (block_ids == int(block_id))
        if not bool(block_mask.any()):
            continue
        block_start = trans_ts[block_mask].min()
        fit_mask = (ref_ts.notna() & (ref_ts < block_start)).to_numpy(dtype=bool)
        prior_rows = int(np.sum(fit_mask))
        if prior_rows < min_prior_rows:
            diagnostics["disabled_blocks"] += 1
            block_reports.append(
                {
                    "block_id": int(block_id),
                    "rows": int(np.sum(block_mask)),
                    "prior_rows": prior_rows,
                    "enabled": False,
                    "reason": "insufficient_prior_rows",
                }
            )
            continue
        ref_block = reference_frame.loc[fit_mask].copy()
        target_block = score_arr[fit_mask] if score_arr is not None else None
        state_block = fit_current_regime_ae_state(
            ref_block,
            feature_columns=feature_columns,
            score_target=target_block,
            cfg=cfg,
        )
        if not bool(state_block.get("enabled", False)):
            diagnostics["disabled_blocks"] += 1
            block_reports.append(
                {
                    "block_id": int(block_id),
                    "rows": int(np.sum(block_mask)),
                    "prior_rows": prior_rows,
                    "enabled": False,
                    "reason": str(state_block.get("reason", "fit_disabled")),
                }
            )
            continue
        transformed = transform_current_regime_ae_features(
            transform_frame.loc[block_mask],
            state_block,
            index=np.asarray(idx)[block_mask] if len(idx) == len(transform_frame) else None,
        )
        out.loc[transformed.index, CURRENT_REGIME_AE_FEATURE_COLUMNS] = transformed[
            list(CURRENT_REGIME_AE_FEATURE_COLUMNS)
        ].to_numpy(dtype=np.float32)
        diagnostics["enabled_blocks"] += 1
        block_reports.append(
            {
                "block_id": int(block_id),
                "rows": int(np.sum(block_mask)),
                "prior_rows": prior_rows,
                "enabled": True,
                "fit_rows": int(state_block.get("fit_rows", 0) or 0),
                "source_feature_count": int(
                    state_block.get("source_feature_count", 0) or 0
                ),
            }
        )
    diagnostics["blocks"] = block_reports[:20]
    diagnostics["block_count"] = int(len(block_reports))
    final_state = dict(final_state or {})
    final_state["candidate_generation"] = diagnostics
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32), final_state
