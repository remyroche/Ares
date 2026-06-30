"""Causal spectral-position features for market-state geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketSpectralPositionConfig:
    lookback: int = 48
    min_periods: int = 24
    top_k: int = 3
    max_features: int = 64
    shrinkage: float = 0.10
    eps: float = 1e-8
    prefix: str = "state_spectral_"


MARKET_SPECTRAL_POSITION_BASE_FEATURES: tuple[str, ...] = (
    "eig_lambda1_share",
    "eig_top3_share",
    "eig_effective_rank",
    "eig_entropy",
    "eig_gap_1_2",
    "eig_gap_ratio_1_2",
    "eig_condition",
    "pc1_score",
    "pc2_score",
    "pc3_score",
    "pc1_z",
    "pc2_z",
    "pc3_z",
    "abs_pc1_z",
    "abs_pc2_z",
    "abs_pc3_z",
    "sum_abs_top3_pc_z",
    "projection_norm_top3",
    "top3_reconstruction_error",
    "top3_reconstruction_ratio",
    "top3_mahalanobis",
)


def market_spectral_position_feature_names(prefix: str = "state_spectral_") -> tuple[str, ...]:
    return tuple(f"{prefix}{name}" for name in MARKET_SPECTRAL_POSITION_BASE_FEATURES)


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _safe_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        out = int(value)
    except Exception:
        return int(default)
    return max(int(minimum), out)


def _numeric_columns(frame: pd.DataFrame, timestamp_col: str) -> list[str]:
    return [
        str(col)
        for col in frame.columns
        if str(col) != timestamp_col and pd.api.types.is_numeric_dtype(frame[col])
    ]


def _fit_robust_refs(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, dict[str, float]]:
    refs: dict[str, dict[str, float]] = {}
    for col in columns:
        values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = values.dropna()
        if finite.empty:
            continue
        med = float(finite.median())
        q25 = float(finite.quantile(0.25))
        q75 = float(finite.quantile(0.75))
        scale = (q75 - q25) / 1.349
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(finite.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        refs[str(col)] = {
            "median": med,
            "scale": float(scale),
            "finite_share": float(values.notna().mean()),
            "variance": float(finite.var(ddof=0)) if len(finite) > 1 else 0.0,
        }
    return refs


def _select_spectral_columns(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    feature_columns: Sequence[str] | None,
    max_features: int,
) -> list[str]:
    candidates = (
        [str(c) for c in feature_columns if str(c) in frame.columns]
        if feature_columns is not None
        else _numeric_columns(frame, timestamp_col)
    )
    rows: list[tuple[float, float, str]] = []
    for col in dict.fromkeys(candidates):
        if col == timestamp_col or not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite_share = float(values.notna().mean())
        var = float(values.var(ddof=0)) if values.notna().sum() > 1 else 0.0
        if finite_share <= 0.0 or not np.isfinite(var) or var <= 1e-12:
            continue
        rows.append((finite_share, var, str(col)))
    rows.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [col for _, _, col in rows[: max(1, int(max_features))]]


def _standardized_matrix(
    frame: pd.DataFrame,
    columns: Sequence[str],
    refs: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    arr = np.zeros((len(frame), len(columns)), dtype=np.float64)
    for j, col in enumerate(columns):
        if col not in frame.columns:
            continue
        ref = refs.get(str(col), {})
        med = _safe_float(ref.get("median"), 0.0)
        scale = max(_safe_float(ref.get("scale"), 1.0), 1e-8)
        values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        z = ((values.to_numpy(dtype=np.float64, copy=False) - med) / scale)
        arr[:, j] = np.nan_to_num(np.clip(z, -8.0, 8.0), nan=0.0, posinf=8.0, neginf=-8.0)
    return arr


def fit_market_spectral_position_encoder(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    feature_columns: Sequence[str] | None = None,
    config: MarketSpectralPositionConfig | None = None,
) -> dict[str, Any]:
    """Fit a train-only spectral-position artifact.

    The fitted artifact stores robust train references and a small history tail.
    ``transform_market_spectral_position`` uses that tail only when scoring
    timestamps strictly after the fit period, so validation/live rows can build
    rolling covariance matrices from past data without refitting.
    """

    cfg = config or MarketSpectralPositionConfig()
    if timestamp_col not in frame.columns:
        raise KeyError(f"Missing timestamp column: {timestamp_col}")
    work = frame.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], utc=True, errors="coerce")
    work = work.loc[work[timestamp_col].notna()].sort_values(timestamp_col, kind="mergesort")
    features = _select_spectral_columns(
        work,
        timestamp_col=timestamp_col,
        feature_columns=feature_columns,
        max_features=cfg.max_features,
    )
    refs = _fit_robust_refs(work, features)
    features = [col for col in features if col in refs]
    history_cols = [timestamp_col, *features]
    history_tail = work.loc[:, history_cols].tail(max(cfg.lookback, cfg.min_periods)).copy()
    return {
        "mode": "market_spectral_position_v1",
        "config": asdict(cfg),
        "timestamp_col": str(timestamp_col),
        "feature_columns": features,
        "column_refs": refs,
        "history_tail": history_tail,
        "fit_rows": int(len(work)),
        "fit_timestamp_min": work[timestamp_col].min() if len(work) else None,
        "fit_timestamp_max": work[timestamp_col].max() if len(work) else None,
        "feature_count": int(len(features)),
        "contract": (
            "Rolling spectral features use robust train scaling and covariance "
            "from rows strictly before the scored timestamp."
        ),
    }


def _empty_spectral_frame(timestamps: pd.Series, prefix: str) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": pd.to_datetime(timestamps, utc=True, errors="coerce")})


def transform_market_spectral_position(
    frame: pd.DataFrame,
    encoder: Mapping[str, Any],
) -> pd.DataFrame:
    """Transform timestamp rows into causal spectral state features."""

    timestamp_col = str(encoder.get("timestamp_col") or "timestamp")
    if timestamp_col not in frame.columns:
        raise KeyError(f"Missing timestamp column: {timestamp_col}")
    cfg_raw = dict(encoder.get("config") or {})
    cfg = MarketSpectralPositionConfig(
        lookback=_safe_int(cfg_raw.get("lookback"), 48),
        min_periods=_safe_int(cfg_raw.get("min_periods"), 24),
        top_k=_safe_int(cfg_raw.get("top_k"), 3),
        max_features=_safe_int(cfg_raw.get("max_features"), 64),
        shrinkage=float(np.clip(_safe_float(cfg_raw.get("shrinkage"), 0.10), 0.0, 1.0)),
        eps=max(_safe_float(cfg_raw.get("eps"), 1e-8), 1e-12),
        prefix=str(cfg_raw.get("prefix") or "state_spectral_"),
    )
    features = [str(c) for c in encoder.get("feature_columns", []) if str(c)]
    target = frame.copy()
    target[timestamp_col] = pd.to_datetime(target[timestamp_col], utc=True, errors="coerce")
    target = target.loc[target[timestamp_col].notna()].copy()
    if not features or target.empty:
        return _empty_spectral_frame(frame[timestamp_col], cfg.prefix)

    fit_max = pd.to_datetime(encoder.get("fit_timestamp_max"), utc=True, errors="coerce")
    history_tail = encoder.get("history_tail")
    use_history = (
        isinstance(history_tail, pd.DataFrame)
        and not history_tail.empty
        and pd.notna(fit_max)
        and bool(target[timestamp_col].min() > fit_max)
    )
    parts: list[pd.DataFrame] = []
    if use_history:
        hist = history_tail.copy()
        hist[timestamp_col] = pd.to_datetime(hist[timestamp_col], utc=True, errors="coerce")
        hist = hist.loc[hist[timestamp_col].notna(), [timestamp_col, *features]]
        hist["__target__"] = False
        parts.append(hist)
    target["__target__"] = True
    target["__input_order__"] = np.arange(len(target), dtype=np.int64)
    parts.append(target[[timestamp_col, *[c for c in features if c in target.columns], "__target__", "__input_order__"]])
    work = pd.concat(parts, ignore_index=True, sort=False)
    for col in features:
        if col not in work.columns:
            work[col] = np.nan
    work = work.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
    z = _standardized_matrix(work, features, dict(encoder.get("column_refs") or {}))
    n_rows, n_features = z.shape
    k = min(cfg.top_k, n_features)

    out_cols = list(MARKET_SPECTRAL_POSITION_BASE_FEATURES)
    values = np.zeros((n_rows, len(out_cols)), dtype=np.float32)
    prev_vecs: np.ndarray | None = None
    eye = np.eye(n_features, dtype=np.float64)
    for i in range(n_rows):
        start = max(0, i - cfg.lookback)
        if i - start < cfg.min_periods:
            continue
        window = z[start:i]
        if window.shape[0] < 2:
            continue
        cov = np.cov(window, rowvar=False, ddof=1)
        if np.ndim(cov) == 0:
            cov = np.array([[float(cov)]], dtype=np.float64)
        cov = np.nan_to_num(np.asarray(cov, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        cov = 0.5 * (cov + cov.T)
        cov = (1.0 - cfg.shrinkage) * cov + cfg.shrinkage * eye
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], cfg.eps)
        eigvecs = eigvecs[:, order]
        if prev_vecs is not None and prev_vecs.shape == eigvecs.shape:
            for comp in range(min(k, prev_vecs.shape[1])):
                if float(np.dot(eigvecs[:, comp], prev_vecs[:, comp])) < 0.0:
                    eigvecs[:, comp] *= -1.0
        prev_vecs = eigvecs.copy()
        lam_sum = float(np.sum(eigvals))
        if lam_sum <= cfg.eps:
            continue
        p = eigvals / lam_sum
        entropy = float(-np.sum(p * np.log(np.maximum(p, cfg.eps))))
        x = z[i]
        top_vecs = eigvecs[:, :k]
        scores = top_vecs.T @ x
        z_scores = scores / np.sqrt(eigvals[:k] + cfg.eps)
        x_hat = top_vecs @ scores
        residual = x - x_hat
        x_norm = float(np.linalg.norm(x))
        recon_error = float(np.linalg.norm(residual))
        top3 = min(3, len(eigvals))
        scores3 = np.zeros(3, dtype=np.float64)
        zscores3 = np.zeros(3, dtype=np.float64)
        scores3[: min(3, len(scores))] = scores[: min(3, len(scores))]
        zscores3[: min(3, len(z_scores))] = z_scores[: min(3, len(z_scores))]
        values[i, :] = np.asarray(
            [
                eigvals[0] / lam_sum,
                np.sum(eigvals[:top3]) / lam_sum,
                np.exp(entropy),
                entropy,
                eigvals[0] - eigvals[1] if len(eigvals) > 1 else 0.0,
                eigvals[0] / max(eigvals[1] if len(eigvals) > 1 else cfg.eps, cfg.eps),
                eigvals[0] / max(eigvals[-1], cfg.eps),
                scores3[0],
                scores3[1],
                scores3[2],
                zscores3[0],
                zscores3[1],
                zscores3[2],
                abs(zscores3[0]),
                abs(zscores3[1]),
                abs(zscores3[2]),
                np.sum(np.abs(zscores3[:top3])),
                np.linalg.norm(scores[:k]),
                recon_error,
                recon_error / max(x_norm, cfg.eps),
                np.sum((scores[:k] ** 2) / (eigvals[:k] + cfg.eps)),
            ],
            dtype=np.float32,
        )

    out = pd.DataFrame(values, columns=[f"{cfg.prefix}{name}" for name in out_cols])
    out.insert(0, "timestamp", work[timestamp_col].to_numpy())
    out["__target__"] = work["__target__"].to_numpy(dtype=bool)
    if "__input_order__" in work.columns:
        out["__input_order__"] = pd.to_numeric(work["__input_order__"], errors="coerce")
    out = out.loc[out["__target__"]].copy()
    out = out.sort_values("__input_order__", kind="mergesort") if "__input_order__" in out.columns else out
    out = out.drop(columns=["__target__", "__input_order__"], errors="ignore")
    out = out.reset_index(drop=True)
    for col in out.columns:
        if col != "timestamp":
            out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return out
