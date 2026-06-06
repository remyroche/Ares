"""Model-local feature drift diagnostics for OOF and live inference.

The helpers in this module deliberately keep the transform deterministic and
artifact-backed: fit a compact reference state on the model training matrix, then
reuse that state to generate the same drift/context columns for OOF rows and
live inference rows.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


MODEL_DRIFT_FEATURE_KEYS: tuple[str, ...] = (
    "regime_centroid_similarity_train",
    "regime_centroid_similarity_train_pc0",
    "regime_centroid_similarity_train_pc1",
    "regime_centroid_similarity_train_pc2",
    "regime_centroid_similarity_train_window_mean",
    "regime_centroid_similarity_train_window_p10",
    "feature_drift_psi_core_50",
    "feature_drift_psi_core_80",
    "mahalanobis_mean_shift",
    "frobenius_corr_shift",
    "feature_drift_cov_shift",
    "inference_drift_score",
    "uncertainty_score",
    "rare_leaf_low_support_score",
    "contribution_drift_score",
)


def _as_numeric_frame(
    x: Any,
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        frame = x.copy()
        if feature_columns:
            frame = frame.reindex(columns=[str(c) for c in feature_columns])
    else:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols = [str(c) for c in (feature_columns or [f"f{i}" for i in range(arr.shape[1])])]
        frame = pd.DataFrame(arr, columns=cols[: arr.shape[1]])
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan)


def _model_predict_contrib(model: Any, x: pd.DataFrame) -> np.ndarray | None:
    if model is None or x.empty:
        return None
    candidates = [model, getattr(model, "best_model", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        estimator = getattr(candidate, "estimator", candidate)
        booster = getattr(estimator, "booster_", None)
        if booster is not None:
            try:
                return np.asarray(booster.predict(x, pred_contrib=True), dtype=np.float32)
            except Exception:
                pass
        native_booster = getattr(estimator, "booster", None)
        if native_booster is not None:
            try:
                return np.asarray(native_booster.predict(x, pred_contrib=True), dtype=np.float32)
            except Exception:
                pass
        get_booster = getattr(estimator, "get_booster", None)
        if callable(get_booster):
            try:
                return np.asarray(
                    get_booster().predict(
                        getattr(estimator, "_validate_data", lambda z, **_: z)(x, reset=False),
                        pred_contribs=True,
                    ),
                    dtype=np.float32,
                )
            except Exception:
                pass
    return None


def fit_model_drift_state(
    x: Any,
    *,
    feature_columns: Sequence[str] | None = None,
    model: Any = None,
    max_core_features: int = 80,
    max_pca_components: int = 3,
    window: int = 240,
) -> dict[str, Any]:
    frame = _as_numeric_frame(x, feature_columns)
    if frame.empty:
        return {"enabled": False, "reason": "empty_feature_frame"}
    finite_share = frame.notna().mean(axis=0)
    variances = frame.var(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    keep = [
        str(c)
        for c in variances.sort_values(ascending=False).index
        if float(finite_share.get(c, 0.0)) >= 0.5 and float(variances.get(c, 0.0)) > 1e-12
    ][: int(max_core_features)]
    if not keep:
        return {"enabled": False, "reason": "no_finite_variable_features"}
    core = frame.reindex(columns=keep)
    med = core.median(axis=0, skipna=True).fillna(0.0)
    q25 = core.quantile(0.25).fillna(med)
    q75 = core.quantile(0.75).fillna(med)
    scale = (q75 - q25).abs().replace(0.0, np.nan).fillna(core.std(axis=0, skipna=True))
    scale = scale.replace(0.0, 1.0).fillna(1.0)
    z = ((core.fillna(med) - med) / scale).clip(-12.0, 12.0).to_numpy(dtype=np.float32)
    z_abs = np.abs(z)
    q50 = np.nanquantile(z_abs, 0.50, axis=0).astype(np.float32)
    q80 = np.nanquantile(z_abs, 0.80, axis=0).astype(np.float32)
    n_comp = int(max(1, min(max_pca_components, z.shape[1], max(1, z.shape[0] - 1))))
    pca = PCA(n_components=n_comp, random_state=42)
    scores = pca.fit_transform(z)
    centroid = np.nanmean(scores, axis=0).astype(np.float32)
    pca_scale = np.nanstd(scores, axis=0).astype(np.float32)
    pca_scale = np.where(pca_scale > 1e-6, pca_scale, 1.0).astype(np.float32)
    corr = np.corrcoef(z, rowvar=False) if z.shape[0] >= 3 and z.shape[1] >= 2 else np.eye(z.shape[1])
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    contrib = _model_predict_contrib(model, core)
    contrib_abs_mean = None
    if contrib is not None and contrib.ndim == 2 and contrib.shape[0] == len(core):
        contrib_abs_mean = np.nanmean(np.abs(contrib), axis=0).astype(np.float32).tolist()
    return {
        "enabled": True,
        "version": 1,
        "feature_columns": keep,
        "median": med.astype(float).to_dict(),
        "scale": scale.astype(float).to_dict(),
        "z_abs_q50": q50.tolist(),
        "z_abs_q80": q80.tolist(),
        "pca_components": pca.components_.astype(np.float32).tolist(),
        "pca_mean": pca.mean_.astype(np.float32).tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32).tolist(),
        "pca_centroid": centroid.tolist(),
        "pca_scale": pca_scale.tolist(),
        "corr_reference": corr.tolist(),
        "contrib_abs_mean": contrib_abs_mean,
        "window": int(window),
    }


def transform_model_drift_features(
    x: Any,
    state: Mapping[str, Any] | None,
    *,
    model: Any = None,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    if not isinstance(state, Mapping) or not bool(state.get("enabled", False)):
        n = len(x) if hasattr(x, "__len__") else 0
        return pd.DataFrame(index=index if index is not None else range(n))
    cols = [str(c) for c in (state.get("feature_columns") or []) if str(c)]
    frame = _as_numeric_frame(x, cols)
    out_index = index if index is not None else frame.index
    if frame.empty or not cols:
        return pd.DataFrame(index=out_index)
    med = pd.Series(state.get("median", {}), dtype=float).reindex(cols).fillna(0.0)
    scale = pd.Series(state.get("scale", {}), dtype=float).reindex(cols).replace(0.0, np.nan).fillna(1.0)
    z = ((frame.fillna(med) - med) / scale).clip(-12.0, 12.0).to_numpy(dtype=np.float32)
    z_abs = np.abs(z)
    q50 = np.asarray(state.get("z_abs_q50", np.ones(len(cols))), dtype=np.float32)
    q80 = np.asarray(state.get("z_abs_q80", np.ones(len(cols))), dtype=np.float32)
    if len(q50) != len(cols):
        q50 = np.ones(len(cols), dtype=np.float32)
    if len(q80) != len(cols):
        q80 = np.ones(len(cols), dtype=np.float32)
    psi50 = np.mean(z_abs > np.maximum(q50, 1e-6), axis=1).astype(np.float32)
    psi80 = np.mean(z_abs > np.maximum(q80, 1e-6), axis=1).astype(np.float32)
    comps = np.asarray(state.get("pca_components", []), dtype=np.float32)
    pca_mean = np.asarray(state.get("pca_mean", np.zeros(len(cols))), dtype=np.float32)
    if comps.ndim != 2 or comps.shape[1] != len(cols):
        scores = np.zeros((len(frame), 1), dtype=np.float32)
    else:
        scores = (z - pca_mean.reshape(1, -1)) @ comps.T
    centroid = np.asarray(state.get("pca_centroid", np.zeros(scores.shape[1])), dtype=np.float32)
    if len(centroid) != scores.shape[1]:
        centroid = np.zeros(scores.shape[1], dtype=np.float32)
    pca_scale = np.asarray(state.get("pca_scale", np.ones(scores.shape[1])), dtype=np.float32)
    if len(pca_scale) != scores.shape[1]:
        pca_scale = np.ones(scores.shape[1], dtype=np.float32)
    dist = np.sqrt(np.mean(np.square((scores - centroid.reshape(1, -1)) / np.maximum(pca_scale, 1e-6)), axis=1))
    sim = (1.0 / (1.0 + dist)).astype(np.float32)
    maha = np.sqrt(np.mean(np.square(z), axis=1)).astype(np.float32)
    corr_ref = np.asarray(state.get("corr_reference", []), dtype=np.float32)
    # A live scoring batch is an arbitrary set of candidates, not a causal
    # covariance window. Keep per-row drift diagnostics invariant to batch
    # composition; covariance-window drift belongs in a separately persisted
    # causal state, not in this row-local artifact transform.
    frob = np.zeros(len(frame), dtype=np.float32)
    cov_shift = (maha + frob).astype(np.float32)
    sim_s = pd.Series(sim, index=out_index)
    window = int(state.get("window", 240) or 240)
    contrib = _model_predict_contrib(model, frame)
    if contrib is not None and contrib.ndim == 2 and contrib.shape[0] == len(frame):
        ref = np.asarray(state.get("contrib_abs_mean") or [], dtype=np.float32)
        cur = np.abs(contrib).astype(np.float32)
        if ref.size == cur.shape[1]:
            contribution_drift = np.mean(np.abs(cur - ref.reshape(1, -1)) / (ref.reshape(1, -1) + 1e-6), axis=1)
        else:
            contribution_drift = np.mean(cur, axis=1)
    else:
        contribution_drift = np.mean(z_abs, axis=1)
    uncertainty_score = np.clip(np.mean(z_abs, axis=1) / 3.0, 0.0, 1.0).astype(np.float32)
    rare_leaf = np.clip(psi80 + 0.25 * psi50, 0.0, 1.0).astype(np.float32)
    inference_drift = np.clip(
        0.35 * (1.0 - sim) + 0.25 * psi80 + 0.25 * np.tanh(cov_shift / 3.0) + 0.15 * np.tanh(contribution_drift / 3.0),
        0.0,
        1.0,
    ).astype(np.float32)
    out = pd.DataFrame(index=out_index)
    out["regime_centroid_similarity_train"] = sim
    for i in range(3):
        if i < scores.shape[1]:
            comp_dist = np.abs((scores[:, i] - centroid[i]) / max(float(pca_scale[i]), 1e-6))
            out[f"regime_centroid_similarity_train_pc{i}"] = (1.0 / (1.0 + comp_dist)).astype(np.float32)
        else:
            out[f"regime_centroid_similarity_train_pc{i}"] = np.float32(1.0)
    out["regime_centroid_similarity_train_window_mean"] = sim_s.to_numpy(dtype=np.float32)
    out["regime_centroid_similarity_train_window_p10"] = sim_s.to_numpy(dtype=np.float32)
    out["feature_drift_psi_core_50"] = psi50
    out["feature_drift_psi_core_80"] = psi80
    out["mahalanobis_mean_shift"] = maha
    out["frobenius_corr_shift"] = frob
    out["feature_drift_cov_shift"] = cov_shift
    out["inference_drift_score"] = inference_drift
    out["uncertainty_score"] = uncertainty_score
    out["rare_leaf_low_support_score"] = rare_leaf
    out["contribution_drift_score"] = np.asarray(contribution_drift, dtype=np.float32)
    return out.reindex(columns=list(MODEL_DRIFT_FEATURE_KEYS)).astype(np.float32)
