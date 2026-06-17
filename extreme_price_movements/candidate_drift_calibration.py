"""Artifact-backed policy-candidate drift and local utility calibration."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.features_denoising_ae import (
    AE_FEATURE_COLUMNS,
    fit_denoising_autoencoder_state,
    transform_denoising_autoencoder_features,
)

try:  # sklearn is already used elsewhere, but keep this module import-safe.
    from sklearn.cluster import MiniBatchKMeans
except Exception:  # pragma: no cover - only relevant in minimal environments.
    MiniBatchKMeans = None


CANDIDATE_KNN_KS: tuple[int, ...] = (25, 50, 100)
CANDIDATE_DRIFT_SCHEMA_VERSION = "candidate_drift_calibration_v3"
CANDIDATE_DRIFT_ARRAY_SIDECAR_VERSION = "candidate_drift_array_sidecar_v1"
CANDIDATE_DRIFT_FEATURE_COLUMNS: tuple[str, ...] = (
    "knn_dist_k25",
    "knn_dist_k50",
    "knn_dist_k100",
    "knn_dist_pct_k25",
    "knn_dist_pct_k50",
    "knn_dist_pct_k100",
    "local_ev_k25",
    "local_ev_k50",
    "local_ev_k100",
    "local_ev_shrunk_k50",
    "local_hit_rate_k50",
    "local_gross_ev_k50",
    "local_downside_p25_k50",
    "local_sample_n_k50",
    "local_effective_n_k50",
    "local_same_symbol_n_k50",
    "local_median_neighbor_age_days_k50",
    "max_abs_zscore_pct",
    "mean_abs_zscore_pct",
    "p95_PSI_pct",
    "mean_PSI_pct",
    "pca_reconstruction_error_pct",
    "missing_count_pct",
    "stale_feature_count_pct",
    "distribution_ood_score",
    "prediction_disagreement_score",
    "recent_calibration_risk_score",
    "ood_risk_score",
    "similarity_support_score",
    "feature_drift_norm",
    "feature_drift_bad_cosine",
    "feature_drift_good_cosine",
    "feature_drift_fp_cosine",
    "feature_drift_bad_minus_good_projection",
    "feature_drift_pc1_signed",
    "feature_drift_pc2_signed",
    "contrib_bad_closeness_score",
    "contrib_good_closeness_score",
    "contrib_bad_minus_good_closeness",
    "distance_to_nearest_bad_archetype",
    "distance_to_nearest_good_archetype",
    "nearest_archetype_bad_rate_lift",
    "directional_local_ev_shrunk_k50",
    "directional_local_hit_rate_k50",
    "directional_ev_spread_k50",
    "directional_effective_n_k50",
    "unknown_direction_score",
    "unknown_unsupported_score",
    "nearest_regime_distance",
    "nearest_regime_distance_pct_global",
    "nearest_regime_distance_pct_local",
    "regime_membership_entropy",
    "top2_regime_margin",
    "regime_transition_score",
    "inter_regime_bridge_score",
    "local_bad_direction_alignment",
    "local_good_direction_alignment",
    "local_false_positive_alignment",
    "local_bad_minus_good_projection",
    "local_directional_ev",
    "local_directional_ev_shrunk",
    "local_directional_hit_rate",
    "local_directional_effective_n",
    "local_directional_ev_spread",
    "local_regime_policy_ev",
    "local_regime_hit_rate",
    "local_regime_support_n",
    "local_regime_score_calibration_error",
    "cluster_time_span_days",
    "cluster_asset_count",
    "local_neighbor_age_days",
    "membership_concentration",
    "atlas_support_quality",
    "local_unknown_direction_score",
    "local_unknown_unsupported_score",
    *AE_FEATURE_COLUMNS,
)
CANDIDATE_DRIFT_LEGACY_ALIAS_COLUMNS: tuple[str, ...] = (
    "contrib_bad_archetype_cosine",
    "contrib_good_archetype_cosine",
    "contrib_bad_minus_good_projection",
)
CANDIDATE_DRIFT_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "nearest_regime_id",
    "nearest_regime_archetype",
    "medoid_reference_index",
)

DISTRIBUTION_SOURCE_COLUMNS: tuple[str, ...] = (
    "max_abs_zscore",
    "mean_abs_zscore",
    "p95_PSI",
    "mean_PSI",
    "pca_reconstruction_error",
    "missing_count",
    "stale_feature_count",
)

DRIFT_SOURCE_COLUMNS: tuple[str, ...] = (
    "inference_drift_score",
    "feature_drift_psi_core_80",
    "feature_drift_ks_bin_mean",
    "mahalanobis_mean_shift",
    "uncertainty_score",
    "rare_leaf_low_support_score",
    "contribution_drift_score",
    "raw_state_knn_distance",
    "raw_state_reconstruction_error",
)

_BLOCKED_FEATURE_EXACT = {
    "timestamp",
    "anchor_date",
    "symbol",
    "strategy_id",
    "side",
    "net_return",
    "gross_return",
    "target",
    "label",
}

_BLOCKED_FEATURE_PREFIXES = (
    "future_",
    "target_",
    "label_",
    "knn_dist_",
    "local_ev_",
    "local_hit_rate_",
    "local_gross_ev_",
    "local_downside_",
    "local_sample_",
    "local_effective_",
    "local_same_symbol_",
    "local_median_neighbor_",
    "feature_drift_bad_",
    "feature_drift_good_",
    "feature_drift_fp_",
    "feature_drift_pc",
    "feature_drift_norm",
    "contrib_bad_archetype_",
    "contrib_good_archetype_",
    "contrib_bad_minus_good_",
    "contrib_bad_closeness_",
    "contrib_good_closeness_",
    "nearest_archetype_bad_rate_lift",
    "directional_local_",
    "directional_ev_",
    "directional_effective_",
    "unknown_direction_",
    "unknown_unsupported_",
    "nearest_regime_",
    "regime_membership_",
    "top2_regime_",
    "regime_transition_",
    "inter_regime_",
    "local_bad_",
    "local_good_",
    "local_false_positive_",
    "local_directional_",
    "local_regime_",
    "cluster_time_span_",
    "cluster_asset_count",
    "local_neighbor_age_",
    "membership_concentration",
    "atlas_support_quality",
    "local_unknown_",
    "ae_b",
)

_TOP_LEVEL_ARRAY_SIDECAR_DTYPES: dict[str, Any] = {
    "reference_matrix": np.float32,
    "reference_net_return": np.float32,
    "reference_gross_return": np.float32,
    "reference_timestamp_ns": np.int64,
    "reference_symbol": str,
    "reference_side": str,
    "reference_strategy_id": str,
    "reference_rank_bucket": np.int16,
}
_ATLAS_ARRAY_SIDECAR_DTYPES: dict[str, Any] = {
    "reference_embedding": np.float32,
    "reference_regime_id": np.int16,
}


def _as_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for col in frame.columns:
        col_s = str(col)
        if col_s in out.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().any():
            out[col_s] = values.astype(np.float32)
    return out.replace([np.inf, -np.inf], np.nan)


def _select_feature_columns(
    frame: pd.DataFrame,
    *,
    max_features: int,
    min_finite_share: float = 0.70,
) -> list[str]:
    numeric = _as_numeric_frame(frame)
    if numeric.empty:
        return []
    finite_share = numeric.notna().mean(axis=0)
    variances = numeric.var(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    candidates: list[tuple[int, float, str]] = []
    priority_tokens = (
        "drift",
        "psi",
        "ks",
        "knn",
        "mahalanobis",
        "uncert",
        "rare_leaf",
        "contrib",
        "raw_state",
        "reconstruction",
        "calibration",
        "brier",
        "ece",
        "disagreement",
        "rank",
        "score",
    )
    for col in numeric.columns:
        low = str(col).lower()
        if low in _BLOCKED_FEATURE_EXACT:
            continue
        if any(low.startswith(prefix) for prefix in _BLOCKED_FEATURE_PREFIXES):
            continue
        if float(finite_share.get(col, 0.0)) < float(min_finite_share):
            continue
        var = float(variances.get(col, 0.0))
        if not np.isfinite(var) or var <= 1e-12:
            continue
        priority = 0 if any(token in low for token in priority_tokens) else 1
        candidates.append((priority, -var, str(col)))
    candidates.sort()
    return [col for _, _, col in candidates[: max(1, int(max_features))]]


def _timestamp_ns(values: Sequence[Any] | None, n: int) -> np.ndarray:
    if values is None or len(values) < n:
        return np.full(n, np.iinfo(np.int64).min, dtype=np.int64)
    ts = pd.to_datetime(np.asarray(values)[:n], utc=True, errors="coerce")
    arr = np.asarray(pd.DatetimeIndex(ts).asi8, dtype=np.int64)
    return np.where(arr == np.iinfo(np.int64).min, np.iinfo(np.int64).min, arr)


def _metadata_array(
    frame: pd.DataFrame,
    name: str,
    n: int,
    *,
    default: str = "",
) -> np.ndarray:
    if name in frame.columns:
        return frame[name].astype(str).fillna(default).to_numpy()[:n]
    return np.repeat(default, n).astype(str)


def _rank_values(frame: pd.DataFrame, n: int) -> np.ndarray:
    for col in ("normalized_rank_score", "auction_rank_score", "strategy_rank_pct", "rank_pct", "calibrated_score"):
        if col in frame.columns:
            vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)[:n]
            if np.isfinite(vals).any():
                finite = vals[np.isfinite(vals)]
                fill = float(np.nanmedian(finite)) if len(finite) else 0.5
                return np.clip(np.nan_to_num(vals, nan=fill), 0.0, 1.0)
    return np.full(n, 0.5, dtype=np.float64)


def _rank_bucket(values: Sequence[float]) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.clip(np.floor(arr * 10.0).astype(np.int16), 0, 9)


def _prepare_matrix(
    frame: pd.DataFrame,
    columns: Sequence[str],
    medians: Mapping[str, float],
    scales: Mapping[str, float],
) -> np.ndarray:
    if not columns:
        return np.zeros((len(frame), 0), dtype=np.float32)
    aligned = _as_numeric_frame(frame).reindex(columns=list(columns))
    med = pd.Series({c: float(medians.get(c, 0.0)) for c in columns}, dtype=np.float64)
    scale = pd.Series({c: float(scales.get(c, 1.0)) for c in columns}, dtype=np.float64)
    scale = scale.replace(0.0, np.nan).fillna(1.0)
    z = ((aligned.fillna(med) - med) / scale).clip(-12.0, 12.0)
    return z.to_numpy(dtype=np.float32, copy=False)


def _fit_scaler(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[dict[str, float], dict[str, float]]:
    aligned = _as_numeric_frame(frame).reindex(columns=list(columns))
    med = aligned.median(axis=0, skipna=True).fillna(0.0)
    q25 = aligned.quantile(0.25).fillna(med)
    q75 = aligned.quantile(0.75).fillna(med)
    scale = (q75 - q25).abs().replace(0.0, np.nan)
    scale = scale.fillna(aligned.std(axis=0, skipna=True)).replace(0.0, 1.0).fillna(1.0)
    return med.astype(float).to_dict(), scale.astype(float).to_dict()


def _sample_reference_indices(
    n: int,
    ts_ns: np.ndarray,
    ranks: np.ndarray,
    *,
    max_rows: int,
) -> np.ndarray:
    if n <= max_rows:
        return np.arange(n, dtype=np.int64)
    finite_ts = np.isfinite(ts_ns.astype(np.float64)) & (ts_ns != np.iinfo(np.int64).min)
    order = np.argsort(np.where(finite_ts, ts_ns, np.arange(n, dtype=np.int64)))
    recent_n = max_rows // 2
    recent = order[-recent_n:]
    remaining = np.setdiff1d(np.arange(n, dtype=np.int64), recent, assume_unique=False)
    rest_n = max_rows - len(recent)
    if rest_n <= 0 or len(remaining) == 0:
        return np.sort(recent.astype(np.int64))
    sort_key = np.lexsort((np.arange(len(remaining)), np.asarray(ranks)[remaining]))
    rem_sorted = remaining[sort_key]
    pick_pos = np.linspace(0, len(rem_sorted) - 1, rest_n).round().astype(int)
    picked = rem_sorted[np.unique(pick_pos)]
    combined = np.unique(np.concatenate([recent, picked])).astype(np.int64)
    if len(combined) > max_rows:
        combined = combined[-max_rows:]
    return np.sort(combined)


def _fit_percentile(values: Sequence[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return []
    return np.sort(arr).astype(float).tolist()


def _apply_percentile(values: Sequence[float], ref: Sequence[float], *, missing: float = 0.5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(len(arr), float(missing), dtype=np.float32)
    ref_arr = np.asarray(ref, dtype=np.float64)
    ref_arr = np.sort(ref_arr[np.isfinite(ref_arr)])
    finite = np.isfinite(arr)
    if len(ref_arr) == 0:
        return out
    out[finite] = (
        np.searchsorted(ref_arr, arr[finite], side="right") / max(len(ref_arr), 1)
    ).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def _sidecar_array(value: Any, dtype: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if dtype is str:
        arr = arr.astype(str)
    return arr


def compact_candidate_drift_calibrator_state(
    state: Mapping[str, Any] | None,
    sidecar_path: str | Path,
) -> dict[str, Any]:
    """Move large calibrator arrays into a compressed sidecar.

    The returned dict is JSON-friendly and can be hydrated with
    :func:`hydrate_candidate_drift_calibrator_state`. Existing inline states
    remain supported because callers only compact explicitly at artifact save
    time.
    """
    if not isinstance(state, Mapping) or not state:
        return dict(state or {})
    out: dict[str, Any] = copy.deepcopy(dict(state))
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []

    def _pop(container: dict[str, Any], key: str, scope: str, dtype: Any) -> None:
        if key not in container:
            return
        value = container.pop(key)
        if value is None:
            return
        array_key = f"{scope}__{key}"
        try:
            arr = _sidecar_array(value, dtype)
        except Exception:
            container[key] = value
            return
        arrays[array_key] = arr
        records.append(
            {
                "scope": scope,
                "key": key,
                "array_key": array_key,
                "dtype": str(arr.dtype),
                "shape": [int(v) for v in arr.shape],
            }
        )

    for key, dtype in _TOP_LEVEL_ARRAY_SIDECAR_DTYPES.items():
        _pop(out, key, "state", dtype)
    atlas = out.get("calibration_atlas")
    if isinstance(atlas, dict):
        for key, dtype in _ATLAS_ARRAY_SIDECAR_DTYPES.items():
            _pop(atlas, key, "calibration_atlas", dtype)
    if not arrays:
        return out

    path = Path(sidecar_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    out["compact_array_sidecar"] = {
        "schema_version": CANDIDATE_DRIFT_ARRAY_SIDECAR_VERSION,
        "path": path.name,
        "array_records": records,
    }
    return out


def hydrate_candidate_drift_calibrator_state(
    state: Mapping[str, Any] | None,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load compact sidecar arrays back into a candidate drift state."""
    if not isinstance(state, Mapping) or not state:
        return dict(state or {})
    out: dict[str, Any] = copy.deepcopy(dict(state))
    sidecar = out.get("compact_array_sidecar")
    if not isinstance(sidecar, Mapping):
        return out
    raw_path = sidecar.get("path")
    if not raw_path:
        return out
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = Path(base_dir or ".") / path
    records = list(sidecar.get("array_records", []) or [])
    if not records or not path.exists():
        out["compact_array_sidecar_error"] = (
            "missing_array_records" if not records else f"sidecar_not_found:{path}"
        )
        return out
    try:
        with np.load(path, allow_pickle=False) as data:
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                array_key = str(record.get("array_key") or "")
                key = str(record.get("key") or "")
                scope = str(record.get("scope") or "state")
                if not array_key or not key or array_key not in data:
                    continue
                values = data[array_key]
                if scope == "calibration_atlas":
                    atlas = out.setdefault("calibration_atlas", {})
                    if isinstance(atlas, dict):
                        atlas[key] = values.tolist()
                else:
                    out[key] = values.tolist()
    except Exception as exc:
        out["compact_array_sidecar_error"] = str(exc)
    return out


def _cohort_key(strategy: str, symbol: str, side: str, bucket: int, level: int) -> str:
    strategy = str(strategy or "")
    symbol = str(symbol or "")
    side = str(side or "")
    bucket_s = str(int(bucket))
    if level == 0:
        return f"{strategy}|{symbol}|{side}|{bucket_s}"
    if level == 1:
        return f"{strategy}|{side}|{bucket_s}"
    if level == 2:
        return f"{side}|{bucket_s}"
    return "__global__"


def _fit_cohorts(
    net_return: np.ndarray,
    gross_return: np.ndarray,
    strategy: np.ndarray,
    symbol: np.ndarray,
    side: np.ndarray,
    rank_bucket: np.ndarray,
) -> dict[str, dict[str, float]]:
    rows: dict[str, list[int]] = {}
    n = len(net_return)
    for i in range(n):
        if not np.isfinite(net_return[i]):
            continue
        for level in range(4):
            rows.setdefault(
                _cohort_key(strategy[i], symbol[i], side[i], int(rank_bucket[i]), level),
                [],
            ).append(i)
    out: dict[str, dict[str, float]] = {}
    for key, idxs in rows.items():
        idx = np.asarray(idxs, dtype=np.int64)
        net = net_return[idx]
        gross = gross_return[idx]
        out[key] = {
            "n": float(len(idx)),
            "ev": float(np.nanmean(net)) if len(net) else 0.0,
            "gross_ev": float(np.nanmean(gross)) if len(gross) else 0.0,
            "hit_rate": float(np.nanmean(net > 0.0)) if len(net) else 0.5,
        }
    if "__global__" not in out:
        finite = np.isfinite(net_return)
        out["__global__"] = {
            "n": float(np.sum(finite)),
            "ev": float(np.nanmean(net_return[finite])) if finite.any() else 0.0,
            "gross_ev": float(np.nanmean(gross_return[finite])) if finite.any() else 0.0,
            "hit_rate": float(np.nanmean(net_return[finite] > 0.0)) if finite.any() else 0.5,
        }
    return out


def _lookup_cohort(
    cohorts: Mapping[str, Mapping[str, float]],
    strategy: str,
    symbol: str,
    side: str,
    bucket: int,
) -> Mapping[str, float]:
    for level in range(4):
        key = _cohort_key(strategy, symbol, side, bucket, level)
        row = cohorts.get(key)
        if row:
            return row
    return cohorts.get("__global__", {"n": 0.0, "ev": 0.0, "gross_ev": 0.0, "hit_rate": 0.5})


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not finite.any():
        return float("nan")
    v = values[finite]
    w = weights[finite]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w)
    total = float(cdf[-1])
    if total <= 0.0:
        return float("nan")
    return float(v[np.searchsorted(cdf, q * total, side="left")])


def _weighted_top_neighbor_stats(
    weights: np.ndarray,
    net_return: np.ndarray,
    *,
    k: int = 50,
) -> tuple[float, float, float, float]:
    weights = np.asarray(weights, dtype=np.float64)
    net = np.asarray(net_return, dtype=np.float64)
    valid = np.isfinite(weights) & np.isfinite(net) & (weights > 1e-12)
    if not valid.any():
        return float("nan"), float("nan"), 0.0, 0.0
    idx = np.flatnonzero(valid)
    if len(idx) > int(k):
        top = np.argpartition(weights[idx], kth=len(idx) - int(k))[-int(k):]
        idx = idx[top]
    w = weights[idx]
    y = net[idx]
    total = float(np.sum(w))
    if total <= 0.0:
        return float("nan"), float("nan"), 0.0, float(len(idx))
    ev = float(np.average(y, weights=w))
    hit = float(np.average(y > 0.0, weights=w))
    eff = float((total * total) / max(float(np.sum(w * w)), 1e-12))
    return ev, hit, eff, float(len(idx))


def _normalise_vector(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    norm = float(np.sqrt(np.sum(arr * arr)))
    if norm <= 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / norm


def _centroid_axis(
    z: np.ndarray,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    *,
    min_rows: int = 20,
) -> np.ndarray:
    pos = np.asarray(pos_mask, dtype=bool)
    neg = np.asarray(neg_mask, dtype=bool)
    if int(pos.sum()) < int(min_rows) or int(neg.sum()) < int(min_rows):
        return np.zeros(z.shape[1], dtype=np.float64)
    pos_centroid = np.nanmean(z[pos], axis=0)
    neg_centroid = np.nanmean(z[neg], axis=0)
    return _normalise_vector(pos_centroid - neg_centroid)


def _fit_pca_components(z: np.ndarray, center: np.ndarray, *, n_components: int = 2) -> list[list[float]]:
    if z.ndim != 2 or z.shape[0] < 20 or z.shape[1] == 0:
        return []
    x = np.asarray(z, dtype=np.float64) - np.asarray(center, dtype=np.float64).reshape(1, -1)
    x = np.where(np.isfinite(x), x, 0.0)
    try:
        _, _, vt = np.linalg.svd(x, full_matrices=False)
    except Exception:
        return []
    comps: list[list[float]] = []
    for comp in vt[: max(0, int(n_components))]:
        vec = _normalise_vector(comp)
        if vec.size:
            anchor = int(np.argmax(np.abs(vec)))
            if vec[anchor] < 0.0:
                vec = -vec
        comps.append(vec.astype(float).tolist())
    return comps


def _embed_matrix(z: np.ndarray, center: Sequence[float], components: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(z, dtype=np.float64)
    ctr = np.asarray(center, dtype=np.float64)
    comps = np.asarray(components, dtype=np.float64)
    if arr.ndim != 2 or len(ctr) != arr.shape[1] or comps.ndim != 2 or comps.shape[1] != arr.shape[1]:
        return np.zeros((arr.shape[0] if arr.ndim == 2 else 0, 0), dtype=np.float32)
    x = np.clip(arr - ctr.reshape(1, -1), -8.0, 8.0)
    x = np.where(np.isfinite(x), x, 0.0)
    return (x @ comps.T).astype(np.float32)


def _score_calibration_error(scores: np.ndarray, outcomes: np.ndarray) -> float:
    score = np.asarray(scores, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    finite = np.isfinite(score) & np.isfinite(y)
    if int(finite.sum()) < 20:
        return 0.0
    score = np.clip(score[finite], 0.0, 1.0)
    hit = (y[finite] > 0.0).astype(np.float64)
    return float(abs(np.nanmean(score) - np.nanmean(hit)))


def _axis_from_embedded_rows(
    emb: np.ndarray,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    *,
    min_rows: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(pos_mask, dtype=bool)
    neg = np.asarray(neg_mask, dtype=bool)
    dim = int(emb.shape[1]) if emb.ndim == 2 else 0
    if int(pos.sum()) < int(min_rows) or int(neg.sum()) < int(min_rows) or dim == 0:
        zero = np.zeros(dim, dtype=np.float64)
        return zero, zero, zero
    pos_centroid = np.nanmean(emb[pos], axis=0)
    neg_centroid = np.nanmean(emb[neg], axis=0)
    return (
        np.where(np.isfinite(pos_centroid), pos_centroid, 0.0),
        np.where(np.isfinite(neg_centroid), neg_centroid, 0.0),
        _normalise_vector(neg_centroid - pos_centroid),
    )


def _cluster_time_span_days(ts_ns: np.ndarray) -> float:
    vals = np.asarray(ts_ns, dtype=np.int64)
    valid = vals[vals != np.iinfo(np.int64).min]
    if len(valid) < 2:
        return 0.0
    return float((np.max(valid) - np.min(valid)) / (86400.0 * 1e9))


def _cluster_archetype(local_ev: float, ev_p10: float, ev_p90: float) -> str:
    if np.isfinite(local_ev) and np.isfinite(ev_p90) and local_ev >= ev_p90:
        return "favorable"
    if np.isfinite(local_ev) and np.isfinite(ev_p10) and local_ev <= ev_p10:
        return "unfavorable"
    return "neutral"


def _fit_calibration_atlas(
    z_ref: np.ndarray,
    net_ref: np.ndarray,
    rank_ref: np.ndarray,
    ts_ref: np.ndarray,
    symbol_ref: np.ndarray,
    *,
    source_frame: pd.DataFrame,
    ref_idx: np.ndarray,
) -> dict[str, Any]:
    if MiniBatchKMeans is None:
        return {"enabled": False, "reason": "sklearn_minibatchkmeans_unavailable"}
    if z_ref.ndim != 2 or z_ref.shape[0] < 200 or z_ref.shape[1] == 0:
        return {"enabled": False, "reason": "insufficient_atlas_reference"}
    finite_net = np.isfinite(net_ref)
    if int(finite_net.sum()) < 200 or len(np.unique(net_ref[finite_net] > 0.0)) < 2:
        return {"enabled": False, "reason": "insufficient_atlas_outcome_support"}
    n_ref = int(z_ref.shape[0])
    min_cluster_size = int(max(100, math.ceil(0.02 * n_ref)))
    max_valid_k = max(1, n_ref // max(min_cluster_size, 1))
    k_candidates = [k for k in (8, 12, 16, 24, 32) if k <= max_valid_k]
    if not k_candidates:
        return {
            "enabled": False,
            "reason": "insufficient_rows_for_min_cluster_size",
            "reference_rows": n_ref,
            "min_cluster_size": min_cluster_size,
        }
    embedding_center = np.nanmedian(z_ref, axis=0)
    embedding_center = np.where(np.isfinite(embedding_center), embedding_center, 0.0)
    n_components = int(min(16, z_ref.shape[1], max(1, n_ref - 1)))
    components = _fit_pca_components(z_ref, embedding_center, n_components=n_components)
    emb_ref = _embed_matrix(z_ref, embedding_center, components)
    if emb_ref.shape[1] == 0:
        return {"enabled": False, "reason": "empty_atlas_embedding"}
    high_rank_cut = float(np.nanquantile(rank_ref[np.isfinite(rank_ref)], 0.70)) if np.isfinite(rank_ref).any() else 0.70
    best: dict[str, Any] | None = None
    trial_rows: list[dict[str, Any]] = []
    for k in k_candidates:
        try:
            km = MiniBatchKMeans(
                n_clusters=int(k),
                random_state=42,
                batch_size=min(4096, max(256, n_ref)),
                n_init=10,
            )
            labels = km.fit_predict(emb_ref)
        except Exception as exc:
            trial_rows.append({"k": int(k), "accepted": False, "reason": f"kmeans_failed:{exc}"})
            continue
        counts = np.bincount(labels, minlength=int(k))
        if int(np.min(counts)) < min_cluster_size:
            trial_rows.append(
                {
                    "k": int(k),
                    "accepted": False,
                    "reason": "min_cluster_size_failed",
                    "min_count": int(np.min(counts)),
                }
            )
            continue
        ev_by_cluster = np.asarray(
            [
                float(np.nanmean(net_ref[labels == c])) if np.any(labels == c) else 0.0
                for c in range(int(k))
            ],
            dtype=np.float64,
        )
        ev_spread = float(np.nanpercentile(ev_by_cluster, 90) - np.nanpercentile(ev_by_cluster, 10))
        top_mask = np.asarray(rank_ref, dtype=np.float64) >= high_rank_cut
        top_coverage = float(np.mean(top_mask)) if top_mask.size else 0.0
        score = float(ev_spread + 0.02 * top_coverage - 0.01 * int(k))
        row = {
            "k": int(k),
            "accepted": True,
            "score": score,
            "ev_spread": ev_spread,
            "top_ranked_share": top_coverage,
            "min_count": int(np.min(counts)),
            "max_count": int(np.max(counts)),
        }
        trial_rows.append(row)
        if best is None or score > float(best.get("score", -np.inf)):
            best = {
                "k": int(k),
                "score": score,
                "model": km,
                "labels": labels,
                "counts": counts,
                "ev_by_cluster": ev_by_cluster,
            }
    if best is None:
        return {
            "enabled": False,
            "reason": "no_valid_atlas_k",
            "reference_rows": n_ref,
            "min_cluster_size": min_cluster_size,
            "candidate_k_trials": trial_rows,
        }
    labels = np.asarray(best["labels"], dtype=np.int32)
    centers = np.asarray(best["model"].cluster_centers_, dtype=np.float64)
    nearest_dist = np.sqrt(np.sum(np.square(emb_ref - centers[labels]), axis=1))
    ev_by_cluster = np.asarray(best["ev_by_cluster"], dtype=np.float64)
    ev_p10 = float(np.nanpercentile(ev_by_cluster, 10))
    ev_p90 = float(np.nanpercentile(ev_by_cluster, 90))
    source_numeric = _as_numeric_frame(source_frame)
    context_cols = [
        str(c)
        for c in source_numeric.columns
        if ("archetype" in str(c).lower() or "leaf" in str(c).lower())
    ][:16]
    anchors: list[dict[str, Any]] = []
    for cid in range(int(best["k"])):
        mask = labels == cid
        local_idx = np.flatnonzero(mask)
        local_emb = emb_ref[mask]
        local_net = net_ref[mask]
        local_rank = rank_ref[mask]
        local_ts = ts_ref[mask]
        local_symbol = symbol_ref[mask]
        local_center = centers[cid]
        d = np.sqrt(np.sum(np.square(local_emb - local_center.reshape(1, -1)), axis=1))
        medoid_pos = int(local_idx[int(np.argmin(d))]) if len(local_idx) else -1
        good_mask = local_net > 0.0
        bad_mask = np.isfinite(local_net) & (local_net <= 0.0)
        good_centroid, bad_centroid, bad_axis = _axis_from_embedded_rows(
            local_emb,
            good_mask,
            bad_mask,
            min_rows=10,
        )
        high_score = local_rank >= high_rank_cut
        _, _, fp_axis = _axis_from_embedded_rows(
            local_emb,
            high_score & (local_net > 0.0),
            high_score & (np.isfinite(local_net) & (local_net <= 0.0)),
            min_rows=5,
        )
        context_summary: dict[str, float] = {}
        if context_cols and len(local_idx):
            raw_idx = ref_idx[local_idx]
            for col in context_cols:
                vals = pd.to_numeric(source_numeric.iloc[raw_idx][col], errors="coerce")
                if vals.notna().any():
                    context_summary[col] = float(vals.mean())
        support_n = int(mask.sum())
        policy_ev = float(np.nanmean(local_net)) if support_n else 0.0
        hit_rate = float(np.nanmean(local_net > 0.0)) if support_n else 0.5
        time_span = _cluster_time_span_days(local_ts)
        asset_count = int(pd.Series(local_symbol.astype(str)).nunique()) if support_n else 0
        anchors.append(
            {
                "cluster_id": int(cid),
                "archetype": _cluster_archetype(policy_ev, ev_p10, ev_p90),
                "support_n": support_n,
                "policy_ev": policy_ev,
                "hit_rate": hit_rate,
                "bad_rate": float(np.nanmean(local_net <= 0.0)) if support_n else 0.5,
                "mean_score": float(np.nanmean(local_rank)) if np.isfinite(local_rank).any() else 0.5,
                "score_calibration_error": _score_calibration_error(local_rank, local_net),
                "time_span_days": time_span,
                "asset_count": asset_count,
                "center": local_center.astype(float).tolist(),
                "good_centroid": good_centroid.astype(float).tolist(),
                "bad_centroid": bad_centroid.astype(float).tolist(),
                "bad_axis": bad_axis.astype(float).tolist(),
                "good_axis": (-bad_axis).astype(float).tolist(),
                "false_positive_axis": fp_axis.astype(float).tolist(),
                "medoid_reference_index": int(ref_idx[medoid_pos]) if medoid_pos >= 0 and len(ref_idx) > medoid_pos else -1,
                "distance_ref": _fit_percentile(nearest_dist[mask]),
                "embedding_p05": np.nanpercentile(local_emb, 5, axis=0).astype(float).tolist(),
                "embedding_p95": np.nanpercentile(local_emb, 95, axis=0).astype(float).tolist(),
                "context_summary": context_summary,
            }
        )
    dist_tau = float(np.nanmedian(nearest_dist[np.isfinite(nearest_dist)])) if np.isfinite(nearest_dist).any() else 1.0
    if not np.isfinite(dist_tau) or dist_tau <= 1e-6:
        dist_tau = 1.0
    return {
        "enabled": True,
        "schema_version": "calibration_atlas_v1",
        "k": int(best["k"]),
        "score": float(best["score"]),
        "min_cluster_size": min_cluster_size,
        "reference_rows": n_ref,
        "embedding_center": np.asarray(embedding_center, dtype=float).tolist(),
        "embedding_components": components,
        "centers": centers.astype(float).tolist(),
        "anchors": anchors,
        "reference_embedding": emb_ref.astype(np.float32).tolist(),
        "reference_regime_id": labels.astype(int).tolist(),
        "global_distance_ref": _fit_percentile(nearest_dist),
        "distance_tau": dist_tau,
        "high_rank_cut": high_rank_cut,
        "candidate_k_trials": trial_rows,
    }


def _fit_directional_drift_spec(
    z_all: np.ndarray,
    z_ref: np.ndarray,
    net_return: np.ndarray,
    ranks: np.ndarray,
    ref_idx: np.ndarray,
) -> dict[str, Any]:
    if z_ref.ndim != 2 or z_ref.shape[0] < 50 or z_ref.shape[1] == 0:
        return {"enabled": False, "reason": "insufficient_directional_reference"}
    center = np.nanmedian(z_ref, axis=0)
    center = np.where(np.isfinite(center), center, 0.0).astype(np.float64)
    z_delta = np.clip(np.asarray(z_all, dtype=np.float64) - center.reshape(1, -1), -8.0, 8.0)
    net = np.asarray(net_return, dtype=np.float64)
    finite = np.isfinite(net)
    good = finite & (net > 0.0)
    bad = finite & (net <= 0.0)
    bad_policy_axis = _centroid_axis(z_delta, bad, good)
    high_score = np.asarray(ranks, dtype=np.float64) >= float(np.nanquantile(ranks[np.isfinite(ranks)], 0.70)) if np.isfinite(ranks).any() else np.zeros(len(ranks), dtype=bool)
    fp_axis = _centroid_axis(z_delta, high_score & bad, high_score & good, min_rows=10)
    ref_delta = np.clip(np.asarray(z_ref, dtype=np.float64) - center.reshape(1, -1), -8.0, 8.0)
    ref_radius = np.sqrt(np.sum(ref_delta * ref_delta, axis=1))
    radius_tau = float(np.nanmedian(ref_radius[np.isfinite(ref_radius)])) if np.isfinite(ref_radius).any() else 1.0
    if not np.isfinite(radius_tau) or radius_tau <= 1e-6:
        radius_tau = 1.0
    return {
        "enabled": True,
        "schema_version": "directional_drift_v1",
        "center": center.astype(float).tolist(),
        "bad_policy_axis": bad_policy_axis.astype(float).tolist(),
        "good_policy_axis": (-bad_policy_axis).astype(float).tolist(),
        "false_positive_axis": fp_axis.astype(float).tolist(),
        "pca_components": _fit_pca_components(z_ref, center, n_components=2),
        "radius_tau": radius_tau,
        "cosine_tau": 0.20,
        "reference_index_count": int(len(ref_idx)),
    }


def _directional_axis_features(
    z: np.ndarray,
    state: Mapping[str, Any],
    index: pd.Index,
) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    n = len(index)
    defaults = {
        "feature_drift_norm": 0.0,
        "feature_drift_bad_cosine": 0.0,
        "feature_drift_good_cosine": 0.0,
        "feature_drift_fp_cosine": 0.0,
        "feature_drift_bad_minus_good_projection": 0.0,
        "feature_drift_pc1_signed": 0.0,
        "feature_drift_pc2_signed": 0.0,
    }
    spec = state.get("directional_drift_spec", {}) or {}
    if z.ndim != 2 or z.shape[1] == 0 or not bool(spec.get("enabled", False)):
        for col, val in defaults.items():
            out[col] = np.full(n, val, dtype=np.float32)
        return out
    center = np.asarray(spec.get("center", []), dtype=np.float64)
    if len(center) != z.shape[1]:
        for col, val in defaults.items():
            out[col] = np.full(n, val, dtype=np.float32)
        return out
    delta = np.clip(np.asarray(z, dtype=np.float64) - center.reshape(1, -1), -8.0, 8.0)
    norm = np.sqrt(np.sum(delta * delta, axis=1))

    def _projection_and_cosine(axis_key: str) -> tuple[np.ndarray, np.ndarray]:
        axis = _normalise_vector(spec.get(axis_key, []))
        if len(axis) != delta.shape[1] or not np.any(axis):
            zeros = np.zeros(len(delta), dtype=np.float64)
            return zeros, zeros
        proj = delta @ axis
        cos = proj / np.maximum(norm, 1e-12)
        return proj, np.clip(cos, -1.0, 1.0)

    bad_proj, bad_cos = _projection_and_cosine("bad_policy_axis")
    good_proj, good_cos = _projection_and_cosine("good_policy_axis")
    _, fp_cos = _projection_and_cosine("false_positive_axis")
    out["feature_drift_norm"] = norm.astype(np.float32)
    out["feature_drift_bad_cosine"] = bad_cos.astype(np.float32)
    out["feature_drift_good_cosine"] = good_cos.astype(np.float32)
    out["feature_drift_fp_cosine"] = fp_cos.astype(np.float32)
    out["feature_drift_bad_minus_good_projection"] = (bad_proj - good_proj).astype(np.float32)
    pcs = list(spec.get("pca_components", []) or [])
    for idx, name in enumerate(("feature_drift_pc1_signed", "feature_drift_pc2_signed")):
        if idx < len(pcs):
            pc = _normalise_vector(pcs[idx])
            if len(pc) == delta.shape[1] and np.any(pc):
                out[name] = (delta @ pc).astype(np.float32)
            else:
                out[name] = np.zeros(n, dtype=np.float32)
        else:
            out[name] = np.zeros(n, dtype=np.float32)
    return out.astype(np.float32)


def _neutral_atlas_features(index: pd.Index) -> pd.DataFrame:
    n = len(index)
    defaults: dict[str, float] = {
        "nearest_regime_distance": 0.0,
        "nearest_regime_distance_pct_global": 0.5,
        "nearest_regime_distance_pct_local": 0.5,
        "regime_membership_entropy": 0.0,
        "top2_regime_margin": 0.0,
        "regime_transition_score": 0.0,
        "inter_regime_bridge_score": 0.0,
        "local_bad_direction_alignment": 0.0,
        "local_good_direction_alignment": 0.0,
        "local_false_positive_alignment": 0.0,
        "local_bad_minus_good_projection": 0.0,
        "local_directional_ev": 0.0,
        "local_directional_ev_shrunk": 0.0,
        "local_directional_hit_rate": 0.5,
        "local_directional_effective_n": 0.0,
        "local_directional_ev_spread": 0.0,
        "local_regime_policy_ev": 0.0,
        "local_regime_hit_rate": 0.5,
        "local_regime_support_n": 0.0,
        "local_regime_score_calibration_error": 0.0,
        "cluster_time_span_days": 0.0,
        "cluster_asset_count": 0.0,
        "local_neighbor_age_days": 365.0,
        "membership_concentration": 0.0,
        "atlas_support_quality": 0.0,
        "local_unknown_direction_score": 0.0,
        "local_unknown_unsupported_score": 0.0,
        "nearest_regime_id": -1.0,
        "nearest_regime_archetype": 0.0,
        "medoid_reference_index": -1.0,
    }
    return pd.DataFrame({k: np.full(n, v, dtype=np.float32) for k, v in defaults.items()}, index=index)


def _archetype_code(value: Any) -> float:
    text = str(value or "").lower()
    if text == "favorable":
        return 1.0
    if text == "unfavorable":
        return -1.0
    return 0.0


def _atlas_support_quality(
    *,
    support_n: float,
    effective_n: float,
    time_span_days: float,
    asset_count: float,
    neighbor_age_days: float,
    membership_concentration: float,
    k: int,
) -> float:
    support_q = np.clip(float(support_n) / 500.0, 0.0, 1.0)
    eff_q = np.clip(float(effective_n) / 25.0, 0.0, 1.0)
    span_q = np.clip(float(time_span_days) / 90.0, 0.0, 1.0)
    asset_q = np.clip(float(asset_count) / 10.0, 0.0, 1.0)
    age = float(neighbor_age_days)
    if not np.isfinite(age):
        age = 365.0
    age_q = 1.0 / (1.0 + np.clip(age, 0.0, 365.0) / 60.0)
    if int(k) > 1:
        base = 1.0 / float(k)
        conc_q = np.clip((float(membership_concentration) - base) / max(1.0 - base, 1e-9), 0.0, 1.0)
    else:
        conc_q = 0.0
    return float(
        np.clip(
            0.25 * support_q
            + 0.25 * eff_q
            + 0.15 * span_q
            + 0.15 * asset_q
            + 0.10 * age_q
            + 0.10 * conc_q,
            0.0,
            1.0,
        )
    )


def _transform_calibration_atlas_features(
    z: np.ndarray,
    state: Mapping[str, Any],
    *,
    metadata: pd.DataFrame,
    timestamps: Sequence[Any] | None,
    training_mode: bool,
    index: pd.Index,
) -> pd.DataFrame:
    out = _neutral_atlas_features(index)
    atlas = state.get("calibration_atlas", {}) or {}
    if z.ndim != 2 or z.shape[1] == 0 or not bool(atlas.get("enabled", False)):
        return out
    emb = _embed_matrix(
        z,
        atlas.get("embedding_center", []) or [],
        atlas.get("embedding_components", []) or [],
    )
    centers = np.asarray(atlas.get("centers", []), dtype=np.float64)
    anchors = list(atlas.get("anchors", []) or [])
    if emb.ndim != 2 or emb.shape[1] == 0 or centers.ndim != 2 or centers.shape[1] != emb.shape[1] or not anchors:
        return out
    k = int(min(len(anchors), centers.shape[0]))
    if k <= 0:
        return out
    centers = centers[:k]
    distances = np.sqrt(np.sum(np.square(emb[:, None, :] - centers[None, :, :]), axis=2))
    nearest = np.argmin(distances, axis=1)
    nearest_dist = distances[np.arange(len(emb)), nearest]
    tau = float(atlas.get("distance_tau", 1.0) or 1.0)
    if not np.isfinite(tau) or tau <= 1e-6:
        tau = 1.0
    raw_weights = np.exp(-distances / tau)
    weight_sum = np.maximum(raw_weights.sum(axis=1, keepdims=True), 1e-12)
    weights = raw_weights / weight_sum
    sorted_weights = np.sort(weights, axis=1)[:, ::-1]
    concentration = sorted_weights[:, 0]
    top2_margin = (
        sorted_weights[:, 0] - sorted_weights[:, 1] if sorted_weights.shape[1] > 1 else sorted_weights[:, 0]
    )
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=1)
    entropy = entropy / max(math.log(max(k, 2)), 1e-12)
    global_pct = _apply_percentile(
        nearest_dist,
        atlas.get("global_distance_ref", []) or [],
        missing=1.0,
    )
    local_pct = np.full(len(emb), 0.5, dtype=np.float32)
    for cid in range(k):
        mask = nearest == cid
        if not mask.any():
            continue
        ref = (anchors[cid] or {}).get("distance_ref", []) or []
        local_pct[mask] = _apply_percentile(nearest_dist[mask], ref, missing=1.0 if not ref else 0.5)
    out["nearest_regime_id"] = nearest.astype(np.float32)
    out["nearest_regime_distance"] = nearest_dist.astype(np.float32)
    out["nearest_regime_distance_pct_global"] = global_pct.astype(np.float32)
    out["nearest_regime_distance_pct_local"] = local_pct.astype(np.float32)
    out["regime_membership_entropy"] = np.clip(entropy, 0.0, 1.0).astype(np.float32)
    out["top2_regime_margin"] = np.clip(top2_margin, 0.0, 1.0).astype(np.float32)
    out["membership_concentration"] = np.clip(concentration, 0.0, 1.0).astype(np.float32)
    out["regime_transition_score"] = np.clip(entropy * (1.0 - np.clip(top2_margin, 0.0, 1.0)), 0.0, 1.0).astype(np.float32)
    out["inter_regime_bridge_score"] = np.clip(entropy * (1.0 - global_pct), 0.0, 1.0).astype(np.float32)

    ref_emb = np.asarray(atlas.get("reference_embedding", []), dtype=np.float64)
    ref_regime = np.asarray(atlas.get("reference_regime_id", []), dtype=np.int32)
    ref_net = np.asarray(state.get("reference_net_return", []), dtype=np.float64)
    ref_ts = np.asarray(state.get("reference_timestamp_ns", []), dtype=np.int64)
    q_ts = _timestamp_ns(timestamps, len(emb))
    for row_i in range(len(emb)):
        cid = int(nearest[row_i])
        anchor = anchors[cid] if cid < len(anchors) and isinstance(anchors[cid], Mapping) else {}
        center = np.asarray(anchor.get("center", centers[cid]), dtype=np.float64)
        delta = emb[row_i].astype(np.float64) - center
        delta_norm = float(np.sqrt(np.sum(delta * delta)))

        def _axis_score(axis_key: str) -> tuple[float, float]:
            axis = _normalise_vector(anchor.get(axis_key, []))
            if len(axis) != len(delta) or not np.any(axis):
                return 0.0, 0.0
            proj = float(delta @ axis)
            cos = proj / max(delta_norm, 1e-12)
            return proj, float(np.clip(cos, -1.0, 1.0))

        bad_proj, bad_cos = _axis_score("bad_axis")
        good_proj, good_cos = _axis_score("good_axis")
        _, fp_cos = _axis_score("false_positive_axis")
        out.iat[row_i, out.columns.get_loc("local_bad_direction_alignment")] = bad_cos
        out.iat[row_i, out.columns.get_loc("local_good_direction_alignment")] = good_cos
        out.iat[row_i, out.columns.get_loc("local_false_positive_alignment")] = fp_cos
        out.iat[row_i, out.columns.get_loc("local_bad_minus_good_projection")] = bad_proj - good_proj
        out.iat[row_i, out.columns.get_loc("nearest_regime_archetype")] = _archetype_code(anchor.get("archetype", "neutral"))
        out.iat[row_i, out.columns.get_loc("medoid_reference_index")] = float(anchor.get("medoid_reference_index", -1) or -1)
        out.iat[row_i, out.columns.get_loc("local_regime_policy_ev")] = float(anchor.get("policy_ev", 0.0) or 0.0)
        out.iat[row_i, out.columns.get_loc("local_regime_hit_rate")] = float(anchor.get("hit_rate", 0.5) or 0.5)
        out.iat[row_i, out.columns.get_loc("local_regime_support_n")] = float(anchor.get("support_n", 0.0) or 0.0)
        out.iat[row_i, out.columns.get_loc("local_regime_score_calibration_error")] = float(anchor.get("score_calibration_error", 0.0) or 0.0)
        out.iat[row_i, out.columns.get_loc("cluster_time_span_days")] = float(anchor.get("time_span_days", 0.0) or 0.0)
        out.iat[row_i, out.columns.get_loc("cluster_asset_count")] = float(anchor.get("asset_count", 0.0) or 0.0)

        same_ev = float(anchor.get("policy_ev", 0.0) or 0.0)
        same_hit = float(anchor.get("hit_rate", 0.5) or 0.5)
        same_eff = 0.0
        ev_spread = 0.0
        age_days = 365.0
        if (
            ref_emb.ndim == 2
            and len(ref_regime) == ref_emb.shape[0]
            and len(ref_net) >= ref_emb.shape[0]
            and ref_emb.shape[1] == emb.shape[1]
        ):
            mask = ref_regime == cid
            if training_mode and len(ref_ts) >= ref_emb.shape[0]:
                mask = mask & (ref_ts[: ref_emb.shape[0]] < q_ts[row_i])
            idx = np.flatnonzero(mask)
            if len(idx):
                ref_delta = ref_emb[idx] - center.reshape(1, -1)
                ref_norm = np.sqrt(np.sum(ref_delta * ref_delta, axis=1))
                denom = np.maximum(delta_norm * ref_norm, 1e-12)
                cos_sim = np.clip((ref_delta @ delta) / denom, -1.0, 1.0)
                radius_gap = np.abs(delta_norm - ref_norm)
                radius_tau = max(float(np.nanmedian(ref_norm[np.isfinite(ref_norm)])) if np.isfinite(ref_norm).any() else 1.0, 1e-6)
                same_weights = np.exp((cos_sim - 1.0) / 0.20) * np.exp(-radius_gap / radius_tau)
                opp_weights = np.exp((-cos_sim - 1.0) / 0.20) * np.exp(-radius_gap / radius_tau)
                same_ev_raw, same_hit_raw, same_eff_raw, _ = _weighted_top_neighbor_stats(
                    same_weights,
                    ref_net[idx],
                    k=50,
                )
                opp_ev, _, _, _ = _weighted_top_neighbor_stats(opp_weights, ref_net[idx], k=50)
                if np.isfinite(same_ev_raw):
                    same_ev = same_ev_raw
                if np.isfinite(same_hit_raw):
                    same_hit = same_hit_raw
                same_eff = same_eff_raw
                if np.isfinite(opp_ev):
                    ev_spread = float(same_ev - opp_ev)
                if len(ref_ts) >= ref_emb.shape[0]:
                    top = np.flatnonzero(np.isfinite(same_weights) & (same_weights > 0.0))
                    if len(top) > 50:
                        top = top[np.argpartition(same_weights[top], kth=len(top) - 50)[-50:]]
                    ages = (q_ts[row_i] - ref_ts[idx[top]]).astype(np.float64) / (86400.0 * 1e9) if len(top) else np.array([], dtype=np.float64)
                    if np.isfinite(ages).any():
                        age_days = float(np.nanmedian(ages))
        shrink = same_eff / (same_eff + 50.0) if same_eff > 0.0 else 0.0
        local_ev = float(anchor.get("policy_ev", 0.0) or 0.0)
        out.iat[row_i, out.columns.get_loc("local_directional_ev")] = same_ev
        out.iat[row_i, out.columns.get_loc("local_directional_ev_shrunk")] = float(
            shrink * same_ev + (1.0 - shrink) * local_ev
        )
        out.iat[row_i, out.columns.get_loc("local_directional_hit_rate")] = same_hit
        out.iat[row_i, out.columns.get_loc("local_directional_effective_n")] = same_eff
        out.iat[row_i, out.columns.get_loc("local_directional_ev_spread")] = ev_spread
        out.iat[row_i, out.columns.get_loc("local_neighbor_age_days")] = age_days
        known_alignment = max(abs(bad_cos), abs(good_cos), abs(fp_cos))
        local_unknown = float(np.clip(1.0 - known_alignment, 0.0, 1.0))
        out.iat[row_i, out.columns.get_loc("local_unknown_direction_score")] = local_unknown
        out.iat[row_i, out.columns.get_loc("local_unknown_unsupported_score")] = float(
            np.clip(local_unknown * global_pct[row_i], 0.0, 1.0)
        )
        out.iat[row_i, out.columns.get_loc("atlas_support_quality")] = _atlas_support_quality(
            support_n=float(anchor.get("support_n", 0.0) or 0.0),
            effective_n=same_eff,
            time_span_days=float(anchor.get("time_span_days", 0.0) or 0.0),
            asset_count=float(anchor.get("asset_count", 0.0) or 0.0),
            neighbor_age_days=age_days,
            membership_concentration=float(concentration[row_i]),
            k=k,
        )
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _group_percentile_score(
    frame: pd.DataFrame,
    state: Mapping[str, Any],
    group_name: str,
    *,
    abs_values: bool = False,
) -> np.ndarray:
    n = len(frame)
    refs = (state.get("group_percentile_refs", {}) or {}).get(group_name, {}) or {}
    if not refs:
        return np.full(n, 0.5, dtype=np.float32)
    parts: list[np.ndarray] = []
    for col, ref in refs.items():
        if col not in frame.columns:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
        if abs_values:
            vals = np.abs(vals)
        parts.append(_apply_percentile(vals, ref))
    if not parts:
        return np.full(n, 0.5, dtype=np.float32)
    return np.nanmean(np.column_stack(parts), axis=1).astype(np.float32)


def _source_columns_for_group(frame: pd.DataFrame, group_name: str) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        low = str(col).lower()
        if group_name == "prediction_disagreement":
            if "disagreement" in low or "base_meta_diff" in low or "pred_std" in low:
                cols.append(str(col))
        elif group_name == "recent_calibration_risk":
            if (
                "calibration_error" in low
                or "cal_error" in low
                or "brier" in low
                or "ece" in low
                or "hit_rate_surprise" in low
                or "confidence_surprise" in low
            ):
                cols.append(str(col))
    return sorted(dict.fromkeys(cols))[:32]


def _neighbor_features(
    z: np.ndarray,
    state: Mapping[str, Any],
    *,
    metadata: pd.DataFrame,
    timestamps: Sequence[Any] | None,
    training_mode: bool,
    chunk_size: int = 512,
) -> pd.DataFrame:
    n = int(z.shape[0])
    out = pd.DataFrame(index=metadata.index)
    for k in CANDIDATE_KNN_KS:
        out[f"knn_dist_k{k}"] = np.nan
        out[f"local_ev_k{k}"] = np.nan
    out["local_hit_rate_k50"] = np.nan
    out["local_gross_ev_k50"] = np.nan
    out["local_downside_p25_k50"] = np.nan
    out["local_sample_n_k50"] = 0.0
    out["local_effective_n_k50"] = 0.0
    out["local_same_symbol_n_k50"] = 0.0
    out["local_median_neighbor_age_days_k50"] = np.nan
    out["directional_local_ev_shrunk_k50"] = np.nan
    out["directional_local_hit_rate_k50"] = np.nan
    out["directional_ev_spread_k50"] = np.nan
    out["directional_effective_n_k50"] = 0.0

    ref = np.asarray(state.get("reference_matrix", []), dtype=np.float32)
    if ref.ndim != 2 or ref.shape[0] == 0 or z.shape[1] == 0:
        return out.astype(np.float32)
    ref_net = np.asarray(state.get("reference_net_return", []), dtype=np.float64)
    ref_gross = np.asarray(state.get("reference_gross_return", []), dtype=np.float64)
    ref_ts = np.asarray(state.get("reference_timestamp_ns", []), dtype=np.int64)
    ref_symbol = np.asarray(state.get("reference_symbol", []), dtype=str)
    q_ts = _timestamp_ns(timestamps, n)
    q_symbol = _metadata_array(metadata, "symbol", n)
    max_k = min(max(CANDIDATE_KNN_KS), ref.shape[0])
    if max_k <= 0:
        return out.astype(np.float32)
    ref_norm = np.sum(ref * ref, axis=1).reshape(1, -1)
    directional_spec = state.get("directional_drift_spec", {}) or {}
    center = np.asarray(directional_spec.get("center", []), dtype=np.float64)
    directional_enabled = (
        bool(directional_spec.get("enabled", False))
        and len(center) == ref.shape[1]
        and len(ref_net) >= ref.shape[0]
    )
    if directional_enabled:
        ref_delta = np.clip(ref.astype(np.float64) - center.reshape(1, -1), -8.0, 8.0)
        ref_delta_norm = np.sqrt(np.sum(ref_delta * ref_delta, axis=1))
        radius_tau = float(directional_spec.get("radius_tau", 1.0) or 1.0)
        if not np.isfinite(radius_tau) or radius_tau <= 1e-6:
            radius_tau = 1.0
        cosine_tau = float(directional_spec.get("cosine_tau", 0.20) or 0.20)
        if not np.isfinite(cosine_tau) or cosine_tau <= 1e-6:
            cosine_tau = 0.20
    else:
        ref_delta = np.zeros((ref.shape[0], ref.shape[1]), dtype=np.float64)
        ref_delta_norm = np.zeros(ref.shape[0], dtype=np.float64)
        radius_tau = 1.0
        cosine_tau = 0.20
    cohorts = state.get("cohorts", {}) or {}
    strategy = _metadata_array(metadata, "strategy_id", n)
    side = _metadata_array(metadata, "side", n)
    buckets = _rank_bucket(_rank_values(metadata, n))
    for start in range(0, n, max(1, int(chunk_size))):
        end = min(n, start + max(1, int(chunk_size)))
        q = z[start:end].astype(np.float32, copy=False)
        dist2 = np.sum(q * q, axis=1).reshape(-1, 1) + ref_norm - 2.0 * (q @ ref.T)
        dist = np.sqrt(np.maximum(dist2, 0.0)).astype(np.float64)
        allowed_mask = None
        if training_mode and len(ref_ts) == ref.shape[0]:
            allowed_mask = ref_ts.reshape(1, -1) < q_ts[start:end].reshape(-1, 1)
            dist = np.where(allowed_mask, dist, np.inf)
        if directional_enabled:
            q_delta = np.clip(q.astype(np.float64) - center.reshape(1, -1), -8.0, 8.0)
            q_delta_norm = np.sqrt(np.sum(q_delta * q_delta, axis=1))
            denom = np.maximum(q_delta_norm.reshape(-1, 1) * ref_delta_norm.reshape(1, -1), 1e-12)
            cos_sim = np.clip((q_delta @ ref_delta.T) / denom, -1.0, 1.0)
            radius_gap = np.abs(q_delta_norm.reshape(-1, 1) - ref_delta_norm.reshape(1, -1))
            same_weights = np.exp((cos_sim - 1.0) / cosine_tau) * np.exp(-radius_gap / radius_tau)
            opp_weights = np.exp((-cos_sim - 1.0) / cosine_tau) * np.exp(-radius_gap / radius_tau)
            if allowed_mask is not None:
                same_weights = np.where(allowed_mask, same_weights, 0.0)
                opp_weights = np.where(allowed_mask, opp_weights, 0.0)
        else:
            same_weights = None
            opp_weights = None
        kth = min(max_k - 1, dist.shape[1] - 1)
        top_idx = np.argpartition(dist, kth=kth, axis=1)[:, :max_k]
        top_dist = np.take_along_axis(dist, top_idx, axis=1)
        order = np.argsort(top_dist, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_dist = np.take_along_axis(top_dist, order, axis=1)
        for local_i in range(end - start):
            row_i = start + local_i
            row_dist = top_dist[local_i]
            row_idx = top_idx[local_i]
            finite = np.isfinite(row_dist)
            row_dist = row_dist[finite]
            row_idx = row_idx[finite]
            if len(row_idx) == 0:
                continue
            for k in CANDIDATE_KNN_KS:
                kk = min(k, len(row_idx))
                if kk <= 0:
                    continue
                idx = row_idx[:kk]
                d = row_dist[:kk]
                w = 1.0 / np.maximum(d, 1e-6)
                w = np.where(np.isfinite(w), w, 0.0)
                if float(np.sum(w)) <= 0.0:
                    w = np.ones(kk, dtype=np.float64)
                net = ref_net[idx] if len(ref_net) >= ref.shape[0] else np.zeros(kk, dtype=np.float64)
                gross = ref_gross[idx] if len(ref_gross) >= ref.shape[0] else net
                out.iat[row_i, out.columns.get_loc(f"knn_dist_k{k}")] = float(np.nanmean(d))
                out.iat[row_i, out.columns.get_loc(f"local_ev_k{k}")] = float(np.average(net, weights=w))
                if k == 50:
                    eff = float((np.sum(w) ** 2.0) / max(np.sum(w * w), 1e-12))
                    same_symbol_n = int(np.sum(ref_symbol[idx].astype(str) == str(q_symbol[row_i]))) if len(ref_symbol) >= ref.shape[0] else 0
                    ages = (q_ts[row_i] - ref_ts[idx]).astype(np.float64) / (86400.0 * 1e9) if len(ref_ts) >= ref.shape[0] else np.full(kk, np.nan)
                    out.iat[row_i, out.columns.get_loc("local_hit_rate_k50")] = float(np.average(net > 0.0, weights=w))
                    out.iat[row_i, out.columns.get_loc("local_gross_ev_k50")] = float(np.average(gross, weights=w))
                    out.iat[row_i, out.columns.get_loc("local_downside_p25_k50")] = _weighted_quantile(net, w, 0.25)
                    out.iat[row_i, out.columns.get_loc("local_sample_n_k50")] = float(kk)
                    out.iat[row_i, out.columns.get_loc("local_effective_n_k50")] = eff
                    out.iat[row_i, out.columns.get_loc("local_same_symbol_n_k50")] = float(same_symbol_n)
                    out.iat[row_i, out.columns.get_loc("local_median_neighbor_age_days_k50")] = float(np.nanmedian(ages)) if np.isfinite(ages).any() else float("nan")
            if directional_enabled and same_weights is not None and opp_weights is not None:
                same_ev, same_hit, same_eff, _ = _weighted_top_neighbor_stats(
                    same_weights[local_i],
                    ref_net,
                    k=50,
                )
                opp_ev, _, _, _ = _weighted_top_neighbor_stats(
                    opp_weights[local_i],
                    ref_net,
                    k=50,
                )
                if np.isfinite(same_ev):
                    out.iat[row_i, out.columns.get_loc("directional_local_ev_shrunk_k50")] = same_ev
                if np.isfinite(same_hit):
                    out.iat[row_i, out.columns.get_loc("directional_local_hit_rate_k50")] = same_hit
                out.iat[row_i, out.columns.get_loc("directional_effective_n_k50")] = same_eff
                if np.isfinite(same_ev) and np.isfinite(opp_ev):
                    out.iat[row_i, out.columns.get_loc("directional_ev_spread_k50")] = float(same_ev - opp_ev)
    for i in range(n):
        fallback = _lookup_cohort(cohorts, strategy[i], q_symbol[i], side[i], int(buckets[i]))
        ev = out.at[out.index[i], "local_ev_k50"]
        eff = float(out.at[out.index[i], "local_effective_n_k50"] or 0.0)
        if not np.isfinite(ev):
            ev = float(fallback.get("ev", 0.0))
            out.at[out.index[i], "local_ev_k50"] = ev
        shrink = eff / (eff + 50.0) if eff > 0.0 else 0.0
        out.at[out.index[i], "local_ev_shrunk_k50"] = float(
            shrink * float(ev) + (1.0 - shrink) * float(fallback.get("ev", 0.0))
        )
        for k in (25, 100):
            col = f"local_ev_k{k}"
            if not np.isfinite(out.at[out.index[i], col]):
                out.at[out.index[i], col] = float(fallback.get("ev", 0.0))
        if not np.isfinite(out.at[out.index[i], "local_hit_rate_k50"]):
            out.at[out.index[i], "local_hit_rate_k50"] = float(fallback.get("hit_rate", 0.5))
        if not np.isfinite(out.at[out.index[i], "local_gross_ev_k50"]):
            out.at[out.index[i], "local_gross_ev_k50"] = float(fallback.get("gross_ev", 0.0))
        dir_ev = out.at[out.index[i], "directional_local_ev_shrunk_k50"]
        dir_eff = float(out.at[out.index[i], "directional_effective_n_k50"] or 0.0)
        if not np.isfinite(dir_ev):
            dir_ev = float(fallback.get("ev", 0.0))
        dir_shrink = dir_eff / (dir_eff + 50.0) if dir_eff > 0.0 else 0.0
        out.at[out.index[i], "directional_local_ev_shrunk_k50"] = float(
            dir_shrink * float(dir_ev) + (1.0 - dir_shrink) * float(fallback.get("ev", 0.0))
        )
        if not np.isfinite(out.at[out.index[i], "directional_local_hit_rate_k50"]):
            out.at[out.index[i], "directional_local_hit_rate_k50"] = float(fallback.get("hit_rate", 0.5))
        if not np.isfinite(out.at[out.index[i], "directional_ev_spread_k50"]):
            out.at[out.index[i], "directional_ev_spread_k50"] = 0.0
    return out.astype(np.float32)


def transform_candidate_drift_features(
    feature_frame: pd.DataFrame,
    state: Mapping[str, Any] | None,
    *,
    candidate_frame: pd.DataFrame | None = None,
    timestamps: Sequence[Any] | None = None,
    training_mode: bool = False,
) -> pd.DataFrame:
    n = len(feature_frame)
    if not isinstance(state, Mapping) or not bool(state.get("enabled", False)):
        return pd.DataFrame(index=feature_frame.index, columns=list(CANDIDATE_DRIFT_FEATURE_COLUMNS), dtype=np.float32).fillna(0.0)
    metadata = candidate_frame.iloc[:n].copy() if isinstance(candidate_frame, pd.DataFrame) and len(candidate_frame) >= n else pd.DataFrame(index=feature_frame.index)
    metadata.index = feature_frame.index
    columns = [str(c) for c in state.get("feature_columns", []) if str(c)]
    z = _prepare_matrix(
        feature_frame,
        columns,
        state.get("medians", {}) or {},
        state.get("scales", {}) or {},
    )
    out = _neighbor_features(
        z,
        state,
        metadata=metadata,
        timestamps=timestamps,
        training_mode=training_mode,
    )
    axis_features = _directional_axis_features(z, state, feature_frame.index)
    out = pd.concat([out, axis_features], axis=1, copy=False)
    atlas_features = _transform_calibration_atlas_features(
        z,
        state,
        metadata=metadata,
        timestamps=timestamps,
        training_mode=training_mode,
        index=feature_frame.index,
    )
    out = pd.concat([out, atlas_features], axis=1, copy=False)
    ae_features = transform_denoising_autoencoder_features(
        z,
        state.get("denoising_autoencoder", {}) or {},
        index=feature_frame.index,
    )
    out = pd.concat([out, ae_features], axis=1, copy=False)

    bad_dist_src = None
    for col in ("distance_to_nearest_bad_archetype", "distance_to_bad_archetype"):
        if col in feature_frame.columns:
            bad_dist_src = pd.to_numeric(feature_frame[col], errors="coerce").to_numpy(dtype=np.float64)
            break
    good_dist_src = None
    for col in ("distance_to_nearest_good_archetype", "distance_to_good_archetype"):
        if col in feature_frame.columns:
            good_dist_src = pd.to_numeric(feature_frame[col], errors="coerce").to_numpy(dtype=np.float64)
            break
    if bad_dist_src is None:
        bad_dist_src = np.zeros(n, dtype=np.float64)
    if good_dist_src is None:
        good_dist_src = np.zeros(n, dtype=np.float64)
    bad_dist_src = np.nan_to_num(bad_dist_src[:n], nan=0.0, posinf=0.0, neginf=0.0)
    good_dist_src = np.nan_to_num(good_dist_src[:n], nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.maximum(np.abs(bad_dist_src) + np.abs(good_dist_src), 1e-9)
    contrib_bad_closeness = np.clip((good_dist_src - bad_dist_src) / denom, -1.0, 1.0)
    out["contrib_bad_closeness_score"] = contrib_bad_closeness.astype(np.float32)
    out["contrib_good_closeness_score"] = (-contrib_bad_closeness).astype(np.float32)
    out["contrib_bad_minus_good_closeness"] = (2.0 * contrib_bad_closeness).astype(np.float32)
    # Legacy aliases are kept for loading older regime-adaptor models. New fits
    # exclude these aliases before model training.
    out["contrib_bad_archetype_cosine"] = out["contrib_bad_closeness_score"]
    out["contrib_good_archetype_cosine"] = out["contrib_good_closeness_score"]
    out["contrib_bad_minus_good_projection"] = out["contrib_bad_minus_good_closeness"]
    out["distance_to_nearest_bad_archetype"] = bad_dist_src.astype(np.float32)
    out["distance_to_nearest_good_archetype"] = good_dist_src.astype(np.float32)
    lift_src = None
    for col in ("nearest_archetype_bad_rate_lift", "archetype_oof_bad_rate_lift"):
        if col in feature_frame.columns:
            lift_src = pd.to_numeric(feature_frame[col], errors="coerce").to_numpy(dtype=np.float64)
            break
    if lift_src is None:
        lift_src = np.ones(n, dtype=np.float64)
    out["nearest_archetype_bad_rate_lift"] = np.nan_to_num(
        lift_src[:n],
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    ).astype(np.float32)

    percentile_refs = state.get("percentile_refs", {}) or {}
    for k in CANDIDATE_KNN_KS:
        raw_col = f"knn_dist_k{k}"
        pct_col = f"knn_dist_pct_k{k}"
        ref = percentile_refs.get(raw_col, [])
        raw = pd.to_numeric(out[raw_col], errors="coerce").to_numpy(dtype=np.float64)
        if len(ref):
            fill = float(np.nanpercentile(np.asarray(ref, dtype=np.float64), 90))
        else:
            fill = 0.0
        raw = np.nan_to_num(raw, nan=fill, posinf=fill, neginf=fill)
        out[raw_col] = raw.astype(np.float32)
        out[pct_col] = _apply_percentile(raw, ref, missing=1.0 if not len(ref) else 0.5)

    distribution_parts: list[np.ndarray] = []
    for src in DISTRIBUTION_SOURCE_COLUMNS:
        out_col = f"{src}_pct"
        if src in feature_frame.columns:
            vals = pd.to_numeric(feature_frame[src], errors="coerce").to_numpy(dtype=np.float64)
            out[out_col] = _apply_percentile(vals, percentile_refs.get(src, []))
        else:
            out[out_col] = np.full(n, 0.5, dtype=np.float32)
        distribution_parts.append(out[out_col].to_numpy(dtype=np.float32))
    out["distribution_ood_score"] = np.nanmean(np.column_stack(distribution_parts), axis=1).astype(np.float32)

    for src in DRIFT_SOURCE_COLUMNS:
        out_col = f"{src}_pct"
        if src in feature_frame.columns:
            vals = pd.to_numeric(feature_frame[src], errors="coerce").to_numpy(dtype=np.float64)
            out[out_col] = _apply_percentile(vals, percentile_refs.get(src, []))

    out["prediction_disagreement_score"] = _group_percentile_score(
        feature_frame,
        state,
        "prediction_disagreement",
        abs_values=True,
    )
    out["recent_calibration_risk_score"] = _group_percentile_score(
        feature_frame,
        state,
        "recent_calibration_risk",
        abs_values=True,
    )
    out["ood_risk_score"] = np.nanmean(
        np.column_stack(
            [
                out["knn_dist_pct_k50"].to_numpy(dtype=np.float32),
                out["distribution_ood_score"].to_numpy(dtype=np.float32),
                out["prediction_disagreement_score"].to_numpy(dtype=np.float32),
                out["recent_calibration_risk_score"].to_numpy(dtype=np.float32),
            ]
        ),
        axis=1,
    ).astype(np.float32)
    known_alignment = np.nanmax(
        np.column_stack(
            [
                np.abs(out["feature_drift_bad_cosine"].to_numpy(dtype=np.float32)),
                np.abs(out["feature_drift_good_cosine"].to_numpy(dtype=np.float32)),
                np.abs(out["feature_drift_fp_cosine"].to_numpy(dtype=np.float32)),
                np.abs(out["contrib_bad_closeness_score"].to_numpy(dtype=np.float32)),
                np.abs(out["contrib_good_closeness_score"].to_numpy(dtype=np.float32)),
            ]
        ),
        axis=1,
    )
    out["unknown_direction_score"] = np.clip(1.0 - known_alignment, 0.0, 1.0).astype(np.float32)
    out["unknown_unsupported_score"] = np.clip(
        out["unknown_direction_score"].to_numpy(dtype=np.float32)
        * out["knn_dist_pct_k50"].to_numpy(dtype=np.float32),
        0.0,
        1.0,
    ).astype(np.float32)
    eff = pd.to_numeric(out["local_effective_n_k50"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    same = pd.to_numeric(out["local_same_symbol_n_k50"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    age = pd.to_numeric(out["local_median_neighbor_age_days_k50"], errors="coerce").fillna(365.0).to_numpy(dtype=np.float32)
    out["similarity_support_score"] = np.clip(
        0.45 * (1.0 - out["knn_dist_pct_k50"].to_numpy(dtype=np.float32))
        + 0.35 * np.clip(eff / 25.0, 0.0, 1.0)
        + 0.10 * np.clip(same / 10.0, 0.0, 1.0)
        + 0.10 * (1.0 / (1.0 + np.clip(age, 0.0, 365.0) / 60.0)),
        0.0,
        1.0,
    ).astype(np.float32)
    for col in CANDIDATE_DRIFT_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = np.float32(0.0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def fit_transform_candidate_drift_calibrator(
    feature_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    *,
    timestamps: Sequence[Any] | None = None,
    max_features: int = 96,
    max_reference_rows: int = 5000,
    enable_denoising_ae: bool = True,
    denoising_ae_max_iter: int = 80,
    include_forward_oos_report: bool = False,
    forward_oos_folds: int = 3,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    n = min(len(feature_frame), len(candidate_frame))
    if n <= 0:
        state = {"enabled": False, "reason": "empty_candidate_frame", "schema_version": CANDIDATE_DRIFT_SCHEMA_VERSION}
        return state, pd.DataFrame(index=feature_frame.index), {"enabled": False, "reason": "empty_candidate_frame"}
    frame = feature_frame.iloc[:n].copy()
    candidates = candidate_frame.iloc[:n].copy()
    columns = _select_feature_columns(frame, max_features=max_features)
    if not columns:
        state = {"enabled": False, "reason": "no_candidate_calibration_features", "schema_version": CANDIDATE_DRIFT_SCHEMA_VERSION}
        return state, pd.DataFrame(index=frame.index), {"enabled": False, "reason": "no_candidate_calibration_features", "rows": int(n)}
    if "net_return" not in candidates.columns:
        state = {"enabled": False, "reason": "missing_realized_net_return", "schema_version": CANDIDATE_DRIFT_SCHEMA_VERSION}
        return state, pd.DataFrame(index=frame.index), {"enabled": False, "reason": "missing_realized_net_return", "rows": int(n)}
    net_return = pd.to_numeric(candidates["net_return"], errors="coerce").to_numpy(dtype=np.float64)
    if len(net_return) != n or not np.isfinite(net_return).any():
        state = {"enabled": False, "reason": "missing_realized_net_return", "schema_version": CANDIDATE_DRIFT_SCHEMA_VERSION}
        return state, pd.DataFrame(index=frame.index), {"enabled": False, "reason": "missing_realized_net_return", "rows": int(n)}
    gross_return = pd.to_numeric(candidates.get("gross_return", candidates.get("net_return")), errors="coerce").to_numpy(dtype=np.float64)
    ts_ns = _timestamp_ns(timestamps if timestamps is not None else candidates.get("timestamp"), n)
    ranks = _rank_values(candidates, n)
    ref_idx = _sample_reference_indices(
        n,
        ts_ns,
        ranks,
        max_rows=max(100, int(max_reference_rows)),
    )
    medians, scales = _fit_scaler(frame.iloc[ref_idx], columns)
    z_ref = _prepare_matrix(frame.iloc[ref_idx], columns, medians, scales)
    strategy = _metadata_array(candidates, "strategy_id", n)
    symbol = _metadata_array(candidates, "symbol", n)
    side = _metadata_array(candidates, "side", n)
    buckets = _rank_bucket(ranks)
    z_all = _prepare_matrix(frame, columns, medians, scales)
    directional_drift_spec = _fit_directional_drift_spec(
        z_all,
        z_ref,
        net_return,
        ranks,
        ref_idx,
    )
    calibration_atlas = _fit_calibration_atlas(
        z_ref,
        np.nan_to_num(net_return[ref_idx], nan=0.0),
        np.nan_to_num(ranks[ref_idx], nan=0.5),
        ts_ns[ref_idx],
        symbol[ref_idx].astype(str),
        source_frame=frame,
        ref_idx=ref_idx,
    )
    if bool(enable_denoising_ae):
        denoising_autoencoder = fit_denoising_autoencoder_state(
            z_ref,
            random_state=42,
            max_train_rows=min(5000, max(200, int(max_reference_rows))),
            max_iter=int(max(20, denoising_ae_max_iter)),
        )
    else:
        denoising_autoencoder = {
            "enabled": False,
            "reason": "disabled_for_diagnostic_subfit",
            "schema_version": "denoising_ae_v1",
        }
    state: dict[str, Any] = {
        "enabled": True,
        "schema_version": CANDIDATE_DRIFT_SCHEMA_VERSION,
        "feature_columns": list(columns),
        "medians": {str(k): float(v) for k, v in medians.items()},
        "scales": {str(k): float(v) for k, v in scales.items()},
        "reference_matrix": z_ref.astype(np.float32).tolist(),
        "reference_net_return": np.nan_to_num(net_return[ref_idx], nan=0.0).astype(float).tolist(),
        "reference_gross_return": np.nan_to_num(gross_return[ref_idx], nan=0.0).astype(float).tolist(),
        "reference_timestamp_ns": ts_ns[ref_idx].astype(np.int64).tolist(),
        "reference_symbol": symbol[ref_idx].astype(str).tolist(),
        "reference_side": side[ref_idx].astype(str).tolist(),
        "reference_strategy_id": strategy[ref_idx].astype(str).tolist(),
        "reference_rank_bucket": buckets[ref_idx].astype(int).tolist(),
        "cohorts": _fit_cohorts(
            net_return[ref_idx],
            gross_return[ref_idx],
            strategy[ref_idx],
            symbol[ref_idx],
            side[ref_idx],
            buckets[ref_idx],
        ),
        "source_rows": int(n),
        "reference_rows": int(len(ref_idx)),
        "max_features": int(max_features),
        "max_reference_rows": int(max_reference_rows),
        "directional_drift_spec": directional_drift_spec,
        "calibration_atlas": calibration_atlas,
        "denoising_autoencoder": denoising_autoencoder,
        "asof_ts": pd.Timestamp.utcnow().isoformat(),
    }
    raw_features = transform_candidate_drift_features(
        frame,
        state,
        candidate_frame=candidates,
        timestamps=timestamps if timestamps is not None else candidates.get("timestamp"),
        training_mode=True,
    )
    percentile_refs: dict[str, list[float]] = {}
    for k in CANDIDATE_KNN_KS:
        percentile_refs[f"knn_dist_k{k}"] = _fit_percentile(raw_features[f"knn_dist_k{k}"])
    for src in DISTRIBUTION_SOURCE_COLUMNS + DRIFT_SOURCE_COLUMNS:
        if src in frame.columns:
            percentile_refs[src] = _fit_percentile(pd.to_numeric(frame[src], errors="coerce"))
    group_refs: dict[str, dict[str, list[float]]] = {}
    for group in ("prediction_disagreement", "recent_calibration_risk"):
        group_refs[group] = {}
        for col in _source_columns_for_group(frame, group):
            vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
            group_refs[group][col] = _fit_percentile(np.abs(vals))
    state["percentile_refs"] = percentile_refs
    state["group_percentile_refs"] = group_refs
    features = transform_candidate_drift_features(
        frame,
        state,
        candidate_frame=candidates,
        timestamps=timestamps if timestamps is not None else candidates.get("timestamp"),
        training_mode=True,
    )
    report = candidate_drift_report(features, candidates, state)
    if include_forward_oos_report:
        report["forward_oos"] = candidate_drift_forward_oos_report(
            frame,
            candidates,
            timestamps=timestamps if timestamps is not None else candidates.get("timestamp"),
            max_features=max_features,
            max_reference_rows=max_reference_rows,
            enable_denoising_ae=False,
            n_folds=forward_oos_folds,
        )
    return state, features, report


def candidate_drift_report(
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "enabled": bool(state.get("enabled", False)),
        "schema_version": str(state.get("schema_version", CANDIDATE_DRIFT_SCHEMA_VERSION)),
        "rows": int(len(features)),
        "source_rows": int(state.get("source_rows", len(features))),
        "reference_rows": int(state.get("reference_rows", 0)),
        "selected_feature_count": int(len(state.get("feature_columns", []) or [])),
        "selected_features_preview": list(state.get("feature_columns", []) or [])[:50],
        "directional_drift_enabled": bool(
            (state.get("directional_drift_spec", {}) or {}).get("enabled", False)
        ),
    }
    ae_state = state.get("denoising_autoencoder", {}) or {}
    out["denoising_autoencoder_enabled"] = bool(ae_state.get("enabled", False))
    out["denoising_autoencoder_schema_version"] = str(
        ae_state.get("schema_version", "")
    )
    if isinstance(ae_state.get("report"), Mapping):
        out["denoising_autoencoder_report"] = ae_state.get("report", {})
    atlas = state.get("calibration_atlas", {}) or {}
    out["calibration_atlas_enabled"] = bool(atlas.get("enabled", False))
    if atlas:
        out["calibration_atlas_schema_version"] = str(atlas.get("schema_version", ""))
        out["calibration_atlas_reason"] = str(atlas.get("reason", ""))
        out["calibration_atlas_k"] = int(atlas.get("k", 0) or 0)
        out["calibration_atlas_score"] = float(atlas.get("score", 0.0) or 0.0)
        out["calibration_atlas_min_cluster_size"] = int(atlas.get("min_cluster_size", 0) or 0)
        anchors = list(atlas.get("anchors", []) or [])
        out["calibration_atlas_anchor_count"] = int(len(anchors))
        out["calibration_atlas_candidate_k_trials"] = atlas.get("candidate_k_trials", [])
        if anchors:
            out["calibration_atlas_anchor_summary"] = [
                {
                    "cluster_id": int(a.get("cluster_id", -1)),
                    "archetype": str(a.get("archetype", "neutral")),
                    "support_n": int(a.get("support_n", 0) or 0),
                    "policy_ev": float(a.get("policy_ev", 0.0) or 0.0),
                    "hit_rate": float(a.get("hit_rate", 0.5) or 0.5),
                    "time_span_days": float(a.get("time_span_days", 0.0) or 0.0),
                    "asset_count": int(a.get("asset_count", 0) or 0),
                }
                for a in anchors[:64]
                if isinstance(a, Mapping)
            ]
    if "timestamp" in candidates.columns:
        ts = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce").dropna()
        if not ts.empty:
            out["start_ts"] = ts.min().isoformat()
            out["end_ts"] = ts.max().isoformat()
    for col in (
        "knn_dist_pct_k50",
        "local_ev_shrunk_k50",
        "local_hit_rate_k50",
        "local_effective_n_k50",
        "ood_risk_score",
        "similarity_support_score",
        "feature_drift_bad_cosine",
        "feature_drift_good_cosine",
        "feature_drift_fp_cosine",
        "feature_drift_bad_minus_good_projection",
        "contrib_bad_closeness_score",
        "contrib_good_closeness_score",
        "contrib_bad_minus_good_closeness",
        "directional_local_ev_shrunk_k50",
        "directional_local_hit_rate_k50",
        "directional_ev_spread_k50",
        "directional_effective_n_k50",
        "unknown_direction_score",
        "unknown_unsupported_score",
        "nearest_regime_distance_pct_global",
        "nearest_regime_distance_pct_local",
        "regime_membership_entropy",
        "top2_regime_margin",
        "regime_transition_score",
        "inter_regime_bridge_score",
        "local_directional_ev_shrunk",
        "local_directional_ev_spread",
        "local_directional_effective_n",
        "local_regime_policy_ev",
        "atlas_support_quality",
        "local_unknown_unsupported_score",
    ):
        if col in features.columns:
            vals = pd.to_numeric(features[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            out[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else None
            out[f"{col}_p90"] = float(vals.quantile(0.90)) if vals.notna().any() else None
    net = pd.to_numeric(candidates.get("net_return"), errors="coerce") if "net_return" in candidates.columns else None
    if net is not None and net.notna().any():
        for col in (
            "knn_dist_pct_k50",
            "local_ev_shrunk_k50",
            "ood_risk_score",
            "similarity_support_score",
            "directional_ev_spread_k50",
            "feature_drift_bad_cosine",
            "contrib_bad_closeness_score",
            "unknown_direction_score",
            "nearest_regime_distance_pct_global",
            "nearest_regime_distance_pct_local",
            "regime_membership_entropy",
            "local_directional_ev_spread",
            "atlas_support_quality",
        ):
            if col not in features.columns:
                continue
            vals = pd.to_numeric(features[col], errors="coerce")
            finite = vals.notna() & net.notna()
            if int(finite.sum()) < 10:
                continue
            try:
                buckets = pd.qcut(vals[finite].rank(method="first"), 5, labels=False, duplicates="drop")
                grouped = net[finite].groupby(buckets).mean()
                out[f"{col}_net_return_by_quintile"] = {
                    str(int(k)): float(v) for k, v in grouped.items()
                }
            except Exception:
                pass
            try:
                x_corr = vals[finite].to_numpy(dtype=np.float64)
                y_corr = net[finite].to_numpy(dtype=np.float64)
                if float(np.nanstd(x_corr)) > 1e-12 and float(np.nanstd(y_corr)) > 1e-12:
                    out[f"{col}_net_return_corr"] = float(np.corrcoef(x_corr, y_corr)[0, 1])
            except Exception:
                pass
        if "knn_dist_pct_k50" in features.columns and "directional_ev_spread_k50" in features.columns:
            x = pd.to_numeric(features["knn_dist_pct_k50"], errors="coerce")
            y = pd.to_numeric(features["directional_ev_spread_k50"], errors="coerce")
            finite = x.notna() & y.notna() & net.notna()
            if int(finite.sum()) >= 30:
                try:
                    xb = pd.qcut(x[finite].rank(method="first"), 3, labels=False, duplicates="drop")
                    yb = pd.qcut(y[finite].rank(method="first"), 3, labels=False, duplicates="drop")
                    grid: dict[str, dict[str, float]] = {}
                    for bx in sorted(pd.Series(xb).dropna().unique()):
                        for by in sorted(pd.Series(yb).dropna().unique()):
                            mask = (xb == bx) & (yb == by)
                            if int(mask.sum()) == 0:
                                continue
                            cell_net = net[finite].to_numpy(dtype=np.float64)[np.asarray(mask, dtype=bool)]
                            grid[f"knn{int(bx)}_dir{int(by)}"] = {
                                "n": float(len(cell_net)),
                                "mean_net_return": float(np.nanmean(cell_net)),
                                "hit_rate": float(np.nanmean(cell_net > 0.0)),
                            }
                    out["knn_pct_x_directional_ev_spread_grid"] = grid
                except Exception:
                    pass
        if "nearest_regime_distance_pct_global" in features.columns and "regime_membership_entropy" in features.columns:
            x = pd.to_numeric(features["nearest_regime_distance_pct_global"], errors="coerce")
            y = pd.to_numeric(features["regime_membership_entropy"], errors="coerce")
            finite = x.notna() & y.notna() & net.notna()
            if int(finite.sum()) >= 30:
                try:
                    xb = pd.qcut(x[finite].rank(method="first"), 3, labels=False, duplicates="drop")
                    yb = pd.qcut(y[finite].rank(method="first"), 3, labels=False, duplicates="drop")
                    grid: dict[str, dict[str, float]] = {}
                    for bx in sorted(pd.Series(xb).dropna().unique()):
                        for by in sorted(pd.Series(yb).dropna().unique()):
                            mask = (xb == bx) & (yb == by)
                            if int(mask.sum()) == 0:
                                continue
                            cell_net = net[finite].to_numpy(dtype=np.float64)[np.asarray(mask, dtype=bool)]
                            grid[f"dist{int(bx)}_entropy{int(by)}"] = {
                                "n": float(len(cell_net)),
                                "mean_net_return": float(np.nanmean(cell_net)),
                                "hit_rate": float(np.nanmean(cell_net > 0.0)),
                            }
                    out["atlas_distance_x_entropy_grid"] = grid
                except Exception:
                    pass
    if "symbol" in candidates.columns:
        out["symbol_count"] = int(candidates["symbol"].astype(str).nunique())
    if "strategy_id" in candidates.columns:
        out["strategy_count"] = int(candidates["strategy_id"].astype(str).nunique())
    return out


def _candidate_drift_outcome_diagnostics(
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    columns: Sequence[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "net_return" not in candidates.columns or features.empty:
        return out
    net = pd.to_numeric(candidates["net_return"], errors="coerce")
    for col in columns:
        if col not in features.columns:
            continue
        vals = pd.to_numeric(features[col], errors="coerce")
        finite = vals.notna() & net.notna()
        if int(finite.sum()) < 10:
            continue
        row: dict[str, Any] = {
            "rows": int(finite.sum()),
            "mean": float(vals[finite].mean()),
            "p90": float(vals[finite].quantile(0.90)),
        }
        x_corr = vals[finite].to_numpy(dtype=np.float64)
        y_corr = net[finite].to_numpy(dtype=np.float64)
        if float(np.nanstd(x_corr)) > 1e-12 and float(np.nanstd(y_corr)) > 1e-12:
            row["net_return_corr"] = float(np.corrcoef(x_corr, y_corr)[0, 1])
        try:
            buckets = pd.qcut(
                vals[finite].rank(method="first"),
                5,
                labels=False,
                duplicates="drop",
            )
            grouped = net[finite].groupby(buckets)
            row["net_return_by_quintile"] = {
                str(int(k)): float(v) for k, v in grouped.mean().items()
            }
            row["hit_rate_by_quintile"] = {
                str(int(k)): float(v)
                for k, v in grouped.apply(lambda s: float(np.nanmean(s > 0.0))).items()
            }
            row["n_by_quintile"] = {
                str(int(k)): int(v) for k, v in grouped.size().items()
            }
        except Exception:
            pass
        out[str(col)] = row
    return out


def _forward_oos_splits(
    ts_ns: np.ndarray,
    n: int,
    *,
    n_folds: int,
    min_train_rows: int,
    min_validation_rows: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    if n <= 0:
        return [], "empty"
    has_ts = np.any(np.asarray(ts_ns, dtype=np.int64) != np.iinfo(np.int64).min)
    order = (
        np.argsort(np.where(ts_ns != np.iinfo(np.int64).min, ts_ns, np.arange(n)))
        if has_ts
        else np.arange(n, dtype=np.int64)
    )
    split_basis = "timestamp" if has_ts else "row_order_no_timestamp"
    start = max(int(min_train_rows), int(math.floor(0.40 * n)))
    if n - start < int(min_validation_rows):
        start = max(int(min_train_rows), n - int(min_validation_rows))
    if start <= 0 or start >= n:
        return [], split_basis
    validation_order = order[start:]
    raw_parts = np.array_split(validation_order, max(1, int(n_folds)))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for part in raw_parts:
        val_idx = np.asarray(part, dtype=np.int64)
        if len(val_idx) < int(min_validation_rows):
            continue
        first_pos = int(np.flatnonzero(order == val_idx[0])[0])
        train_idx = np.asarray(order[:first_pos], dtype=np.int64)
        if len(train_idx) < int(min_train_rows):
            continue
        splits.append((train_idx, val_idx))
    return splits, split_basis


def candidate_drift_forward_oos_feature_frame(
    feature_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    *,
    timestamps: Sequence[Any] | None = None,
    max_features: int = 96,
    max_reference_rows: int = 5000,
    enable_denoising_ae: bool = False,
    denoising_ae_max_iter: int = 80,
    n_folds: int = 3,
    min_train_rows: int = 100,
    min_validation_rows: int = 30,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return past-fit/future-transform candidate drift features for training.

    The fitted calibrator artifact can use the full training sample for live
    inference, but validation evidence should not let a row's realized PnL shape
    its own good/bad centroid or atlas assignment. This helper builds a compact
    expanding-window OOS feature frame and leaves rows without sufficient past
    support at neutral defaults.
    """
    n = min(len(feature_frame), len(candidate_frame))
    all_cols = list(
        dict.fromkeys(
            [
                *CANDIDATE_DRIFT_FEATURE_COLUMNS,
                *CANDIDATE_DRIFT_LEGACY_ALIAS_COLUMNS,
                *CANDIDATE_DRIFT_DIAGNOSTIC_COLUMNS,
            ]
        )
    )
    index = feature_frame.index[:n]
    out = pd.DataFrame(
        {col: np.zeros(n, dtype=np.float32) for col in all_cols},
        index=index,
    )
    neutral_defaults = {
        "local_hit_rate_k50": 0.5,
        "directional_local_hit_rate_k50": 0.5,
        "local_directional_hit_rate": 0.5,
        "local_regime_hit_rate": 0.5,
        "nearest_archetype_bad_rate_lift": 1.0,
        "local_neighbor_age_days": 365.0,
        "nearest_regime_id": -1.0,
        "medoid_reference_index": -1.0,
        "distribution_ood_score": 0.5,
        "prediction_disagreement_score": 0.5,
        "recent_calibration_risk_score": 0.5,
        "ood_risk_score": 0.5,
        "unknown_direction_score": 1.0,
        "unknown_unsupported_score": 1.0,
        "local_unknown_direction_score": 1.0,
        "local_unknown_unsupported_score": 1.0,
    }
    for col in all_cols:
        if col.endswith("_pct") or col in {
            "nearest_regime_distance_pct_global",
            "nearest_regime_distance_pct_local",
        }:
            neutral_defaults.setdefault(col, 0.5)
    for col, value in neutral_defaults.items():
        if col in out.columns:
            out[col] = np.full(n, float(value), dtype=np.float32)
    if n <= 0:
        return out, {"enabled": False, "reason": "empty_candidate_frame"}
    if "net_return" not in candidate_frame.columns:
        return out, {
            "enabled": False,
            "reason": "missing_realized_net_return",
            "rows": int(n),
        }
    frame = feature_frame.iloc[:n].copy()
    candidates = candidate_frame.iloc[:n].copy()
    ts_values = timestamps if timestamps is not None else candidates.get("timestamp")
    ts_ns = _timestamp_ns(ts_values, n)
    splits, split_basis = _forward_oos_splits(
        ts_ns,
        n,
        n_folds=max(1, int(n_folds)),
        min_train_rows=max(20, int(min_train_rows)),
        min_validation_rows=max(10, int(min_validation_rows)),
    )
    if not splits:
        return out, {
            "enabled": False,
            "reason": "insufficient_rows_for_forward_oos_splits",
            "rows": int(n),
            "split_basis": split_basis,
            "requested_folds": int(n_folds),
            "neutral_rows": int(n),
        }

    fold_rows: list[dict[str, Any]] = []
    covered = np.zeros(n, dtype=bool)
    ts_array = np.asarray(ts_values) if ts_values is not None and len(ts_values) >= n else None
    for fold_no, (train_idx, valid_idx) in enumerate(splits, start=1):
        train_frame = frame.iloc[train_idx].copy()
        train_candidates = candidates.iloc[train_idx].copy()
        valid_frame = frame.iloc[valid_idx].copy()
        valid_candidates = candidates.iloc[valid_idx].copy()
        train_ts = ts_array[train_idx] if ts_array is not None else None
        valid_ts = ts_array[valid_idx] if ts_array is not None else None
        state, _, train_report = fit_transform_candidate_drift_calibrator(
            train_frame,
            train_candidates,
            timestamps=train_ts,
            max_features=max_features,
            max_reference_rows=min(max(100, int(max_reference_rows)), max(100, len(train_frame))),
            enable_denoising_ae=bool(enable_denoising_ae),
            denoising_ae_max_iter=int(denoising_ae_max_iter),
            include_forward_oos_report=False,
        )
        row: dict[str, Any] = {
            "fold": int(fold_no),
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(valid_frame)),
            "enabled": bool(state.get("enabled", False)),
            "reason": str(state.get("reason", "")),
            "selected_feature_count": int(len(state.get("feature_columns", []) or [])),
        }
        if bool(state.get("enabled", False)):
            part = transform_candidate_drift_features(
                valid_frame,
                state,
                candidate_frame=valid_candidates,
                timestamps=valid_ts,
                training_mode=False,
            )
            valid_cols = [c for c in all_cols if c in part.columns]
            if valid_cols:
                out.iloc[valid_idx, out.columns.get_indexer(valid_cols)] = part[valid_cols].to_numpy(
                    dtype=np.float32,
                    copy=False,
                )
                covered[valid_idx] = True
            row["validation_feature_rows"] = int(len(part))
            row["calibration_atlas_enabled"] = bool(
                (state.get("calibration_atlas", {}) or {}).get("enabled", False)
            )
        else:
            row["train_report"] = train_report
        for key, arr in (("train", ts_ns[train_idx]), ("validation", ts_ns[valid_idx])):
            arr = arr[arr != np.iinfo(np.int64).min]
            if len(arr):
                row[f"{key}_start_ts"] = pd.Timestamp(int(np.min(arr)), unit="ns", tz="UTC").isoformat()
                row[f"{key}_end_ts"] = pd.Timestamp(int(np.max(arr)), unit="ns", tz="UTC").isoformat()
        fold_rows.append(row)

    report = {
        "enabled": bool(np.any(covered)),
        "schema_version": "candidate_drift_forward_oos_feature_frame_v1",
        "split": "fit_past_transform_future",
        "split_basis": split_basis,
        "rows": int(n),
        "covered_rows": int(np.sum(covered)),
        "neutral_rows": int(n - np.sum(covered)),
        "coverage": float(np.mean(covered)) if n else 0.0,
        "folds_completed": int(sum(1 for row in fold_rows if row.get("enabled"))),
        "folds": fold_rows,
    }
    if not np.any(covered):
        report["reason"] = "no_enabled_forward_oos_folds"
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32), report


def candidate_drift_forward_oos_report(
    feature_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    *,
    timestamps: Sequence[Any] | None = None,
    max_features: int = 96,
    max_reference_rows: int = 5000,
    enable_denoising_ae: bool = False,
    denoising_ae_max_iter: int = 80,
    n_folds: int = 3,
    min_train_rows: int = 100,
    min_validation_rows: int = 30,
) -> dict[str, Any]:
    """Fit on past candidate rows, transform future rows, and summarize utility.

    This is diagnostic only. It avoids the in-sample optimism of the fitted
    calibrator report and is not consumed by live inference.
    """
    n = min(len(feature_frame), len(candidate_frame))
    if n <= 0:
        return {"enabled": False, "reason": "empty_candidate_frame"}
    if "net_return" not in candidate_frame.columns:
        return {"enabled": False, "reason": "missing_realized_net_return", "rows": int(n)}
    frame = feature_frame.iloc[:n].copy()
    candidates = candidate_frame.iloc[:n].copy()
    ts_values = timestamps if timestamps is not None else candidates.get("timestamp")
    ts_ns = _timestamp_ns(ts_values, n)
    splits, split_basis = _forward_oos_splits(
        ts_ns,
        n,
        n_folds=max(1, int(n_folds)),
        min_train_rows=max(20, int(min_train_rows)),
        min_validation_rows=max(10, int(min_validation_rows)),
    )
    if not splits:
        return {
            "enabled": False,
            "reason": "insufficient_rows_for_forward_oos_splits",
            "rows": int(n),
            "split_basis": split_basis,
            "requested_folds": int(n_folds),
            "min_train_rows": int(min_train_rows),
            "min_validation_rows": int(min_validation_rows),
        }

    feature_parts: list[pd.DataFrame] = []
    candidate_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_no, (train_idx, valid_idx) in enumerate(splits, start=1):
        train_frame = frame.iloc[train_idx].copy()
        train_candidates = candidates.iloc[train_idx].copy()
        valid_frame = frame.iloc[valid_idx].copy()
        valid_candidates = candidates.iloc[valid_idx].copy()
        train_ts = np.asarray(ts_values)[train_idx] if ts_values is not None and len(ts_values) >= n else None
        valid_ts = np.asarray(ts_values)[valid_idx] if ts_values is not None and len(ts_values) >= n else None
        state, _, train_report = fit_transform_candidate_drift_calibrator(
            train_frame,
            train_candidates,
            timestamps=train_ts,
            max_features=max_features,
            max_reference_rows=min(max(100, int(max_reference_rows)), max(100, len(train_frame))),
            enable_denoising_ae=bool(enable_denoising_ae),
            denoising_ae_max_iter=int(denoising_ae_max_iter),
            include_forward_oos_report=False,
        )
        row: dict[str, Any] = {
            "fold": int(fold_no),
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(valid_frame)),
            "enabled": bool(state.get("enabled", False)),
            "reason": str(state.get("reason", "")),
            "selected_feature_count": int(len(state.get("feature_columns", []) or [])),
        }
        if bool(state.get("enabled", False)):
            valid_features = transform_candidate_drift_features(
                valid_frame,
                state,
                candidate_frame=valid_candidates,
                timestamps=valid_ts,
                training_mode=False,
            )
            feature_parts.append(valid_features)
            candidate_parts.append(valid_candidates)
            row["validation_feature_rows"] = int(len(valid_features))
            row["calibration_atlas_enabled"] = bool(
                (state.get("calibration_atlas", {}) or {}).get("enabled", False)
            )
        else:
            row["train_report"] = train_report
        train_ts_valid = ts_ns[train_idx]
        valid_ts_valid = ts_ns[valid_idx]
        for key, arr in (
            ("train", train_ts_valid),
            ("validation", valid_ts_valid),
        ):
            arr = arr[arr != np.iinfo(np.int64).min]
            if len(arr):
                row[f"{key}_start_ts"] = pd.Timestamp(int(np.min(arr)), unit="ns", tz="UTC").isoformat()
                row[f"{key}_end_ts"] = pd.Timestamp(int(np.max(arr)), unit="ns", tz="UTC").isoformat()
        fold_rows.append(row)

    if not feature_parts:
        return {
            "enabled": False,
            "reason": "no_enabled_forward_oos_folds",
            "rows": int(n),
            "split_basis": split_basis,
            "folds": fold_rows,
        }
    oos_features = pd.concat(feature_parts, axis=0).reset_index(drop=True)
    oos_candidates = pd.concat(candidate_parts, axis=0).reset_index(drop=True)
    net = pd.to_numeric(oos_candidates["net_return"], errors="coerce")
    diagnostic_cols = (
        "knn_dist_pct_k50",
        "local_ev_shrunk_k50",
        "directional_ev_spread_k50",
        "ood_risk_score",
        "similarity_support_score",
        "feature_drift_bad_cosine",
        "contrib_bad_closeness_score",
        "unknown_direction_score",
        "nearest_regime_distance_pct_global",
        "regime_membership_entropy",
        "local_directional_ev_spread",
        "atlas_support_quality",
    )
    return {
        "enabled": True,
        "schema_version": "candidate_drift_forward_oos_v1",
        "split": "fit_past_transform_future",
        "split_basis": split_basis,
        "rows": int(len(oos_features)),
        "source_rows": int(n),
        "folds_completed": int(len(feature_parts)),
        "folds": fold_rows,
        "mean_net_return": float(net.mean()) if net.notna().any() else None,
        "hit_rate": float((net > 0.0).mean()) if net.notna().any() else None,
        "feature_outcome_diagnostics": _candidate_drift_outcome_diagnostics(
            oos_features,
            oos_candidates,
            diagnostic_cols,
        ),
    }
