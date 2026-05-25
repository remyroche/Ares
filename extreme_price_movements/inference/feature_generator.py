"""
Feature Generator for Inference.

This module generates features for inference:
- Uses compute_market_features from features.py
- Uses add_regime_gates for regime features
- Computes per-symbol features needed by candidate selector
"""

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

from extreme_price_movements.data_store import load_features_selected
from extreme_price_movements.feature_transform_contract import FeatureTransformContract
from extreme_price_movements.config import is_non_portable_feature_key
from extreme_price_movements.features import (
    add_regime_gates,
    atr_percent,
    compute_features_hourly,
    compute_market_features,
)
from extreme_price_movements.inference.parity import (
    LIVE_UNAVAILABLE_FEATURES,
    strategy_id_matches,
)
from extreme_price_movements.regime_adaptor import regime_adaptor_inference_enabled
from extreme_price_movements.utils import tprint


def get_market_data(
    panel: Dict[str, pd.DataFrame],
    symbol: str,
) -> pd.DataFrame:
    """Get market data for a specific symbol from the panel.

    Args:
        panel: Dictionary of symbol -> OHLCV DataFrames
        symbol: Trading symbol to get data for

    Returns:
        DataFrame with OHLCV data for the symbol
    """
    if symbol in panel:
        return panel[symbol]
    return pd.DataFrame()


# Default feature generation parameters
DEFAULT_TREND_SMA_HOURS = 24 * 14  # 14 days
DEFAULT_GATE_VOL_LOOKBACK_HOURS = 24 * 7  # 7 days
DEFAULT_GATE_TREND_THR = 0.0
DEFAULT_CAUSAL_TRANSFORM_ROLL_WINDOW_HOURS = 24 * 30
DEFAULT_IDENTITY_EWMA_WARMUP_HOURS = 24 * 60 * 5
DEFAULT_TAIL_WARMUP_BUFFER_HOURS = 72
LIVE_FEATURE_CACHE_VERSION = 13
_LIVE_FEATURE_MEMORY_CACHE: Dict[str, Dict[str, Any]] = {}
_TRAINING_FEATURE_VARIATION_CACHE: Dict[tuple[str, str], Dict[str, bool]] = {}
MODEL_DERIVED_FEATURE_RE = re.compile(
    r"^(base_H\d+_|pred_H\d+|pred_.*_H\d+|pred_logit$|oof_ebm_unc_|base_med_|base_prob_|base_model_|regime_centroid_similarity_train$|feature_drift_|prob_error|recent_prob_error_|recent_hit_rate_|recent_global_|recent_side_horizon_|recent_bucket_|recent_regime_|recent_meta_|recent_base_meta_disagreement_|recent_base_internal_disagreement_|recent_prediction_disagreement_available_|recent_effectiveness_available$)"
)


class _StageTimer:
    """Small timing helper for live inference feature stages."""

    def __init__(self, label: str):
        self.label = label
        self.start = time.perf_counter()
        self.last = self.start

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        tprint(
            f"[Timing] {self.label}.{stage}: "
            f"stage={now - self.last:.3f}s total={now - self.start:.3f}s"
        )
        self.last = now


def _hash_values(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(v) for v in values if str(v))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _feature_runtime_cfg_hash(cfg: Optional[Dict[str, Any]]) -> str:
    """Return a stable hash for feature-generation-relevant runtime config.

    Live feature snapshots must not be reused across different runtime configs:
    the feature pipeline reads optimizer/runtime fields from ``cfg``.  Omitting
    this from the cache key can make a later live/replay run load a finite but
    wrong transformed feature frame.
    """

    def _normalise(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [_normalise(v) for v in value]
        if isinstance(value, dict):
            return {
                str(k): _normalise(v)
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
                if str(k)
                not in {
                    "full_state",
                    "model_bundle",
                    "bundle",
                    "runtime_cfg",
                }
            }
        return str(value)

    payload = json.dumps(_normalise(cfg or {}), sort_keys=True, default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def _live_feature_cache_key(
    *,
    run_id: str,
    symbols: List[str],
    required_feature_keys: Set[str],
    lookback_hours: int,
    cfg: Optional[Dict[str, Any]] = None,
    data_root: Optional[str] = None,
) -> str:
    return "|".join(
        [
            str(LIVE_FEATURE_CACHE_VERSION),
            str(run_id),
            str(int(lookback_hours)),
            str(data_root or ""),
            _hash_values(symbols),
            _hash_values(required_feature_keys),
            _feature_runtime_cfg_hash(cfg),
        ]
    )


def _feature_snapshot_dir(cfg: Dict[str, Any], run_id: str, cache_key: str) -> Path:
    root = Path(
        cfg.get(
            "live_feature_snapshot_cache_dir",
            f"cache/inference_live_features/{run_id}",
        )
    )
    safe = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:24]
    return root / safe


def _feature_transform_contract_hash_from_cfg(
    cfg: Optional[Dict[str, Any]],
) -> str | None:
    if not isinstance(cfg, dict):
        return None
    explicit = cfg.get("feature_transform_contract_hash")
    if explicit:
        return str(explicit)
    contract = cfg.get("feature_transform_contract")
    if isinstance(contract, FeatureTransformContract) and contract.contract_hash:
        return str(contract.contract_hash)
    bundle = cfg.get("bundle")
    if isinstance(bundle, dict):
        bundled_hash = bundle.get("feature_transform_contract_hash")
        if bundled_hash:
            return str(bundled_hash)
        bundled_contract = bundle.get("feature_transform_contract")
        if (
            isinstance(bundled_contract, FeatureTransformContract)
            and bundled_contract.contract_hash
        ):
            return str(bundled_contract.contract_hash)
    return None


def _latest_feature_matrix(
    feats: Dict[str, pd.DataFrame],
    symbols: List[str],
    end_ts: pd.Timestamp,
    required_feature_keys: Set[str],
) -> pd.DataFrame:
    rows: Dict[str, Dict[str, float]] = {sym: {} for sym in symbols}
    keys = required_feature_keys or set(feats.keys())
    for key in keys:
        df = feats.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        available = [sym for sym in symbols if sym in df.columns]
        if not available:
            continue
        if end_ts in df.index:
            values = df.loc[end_ts, available]
        elif _is_live_stale_sensitive_feature_key(key):
            continue
        else:
            values = df.loc[:, available].ffill().iloc[-1]
        for sym, value in values.items():
            try:
                rows[str(sym)][str(key)] = float(value)
            except (TypeError, ValueError):
                rows[str(sym)][str(key)] = np.nan
    return pd.DataFrame.from_dict(rows, orient="index").astype(np.float32)


def _matrix_to_feature_dict(
    matrix: pd.DataFrame,
    end_ts: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if matrix is None or matrix.empty:
        return out
    ts_index = pd.DatetimeIndex([pd.Timestamp(end_ts)])
    for key in matrix.columns:
        values = matrix[key].to_numpy(dtype=np.float32, copy=False)[None, :]
        out[str(key)] = pd.DataFrame(values, index=ts_index, columns=matrix.index)
    return out


def _feature_history_matrix(
    feats: Dict[str, pd.DataFrame],
    *,
    symbols: List[str],
    required_feature_keys: Set[str],
    start_ts: Optional[pd.Timestamp],
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Convert feature frames to a compact timestamp/symbol matrix.

    The rolling live cache stores only transformed rows that extend beyond the
    offline feature cache.  It intentionally uses a row MultiIndex so hourly
    appends are small and restart-safe without rewriting the full historical
    per-symbol feature store.
    """
    keys = sorted(required_feature_keys or set(feats.keys()))
    frames: Dict[str, pd.Series] = {}
    idx_union: Optional[pd.DatetimeIndex] = None
    end = pd.Timestamp(end_ts)
    start = pd.Timestamp(start_ts) if start_ts is not None else None
    for key in keys:
        df = feats.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        available = [sym for sym in symbols if sym in df.columns]
        if not available:
            continue
        sliced = df.loc[:, available]
        if start is not None:
            sliced = sliced[sliced.index > start]
        sliced = sliced[sliced.index <= end]
        if sliced.empty:
            continue
        if idx_union is None:
            idx_union = pd.DatetimeIndex(sliced.index)
        else:
            idx_union = idx_union.union(pd.DatetimeIndex(sliced.index))
        frames[str(key)] = sliced.stack(dropna=False)

    if not frames or idx_union is None or idx_union.empty:
        return pd.DataFrame()

    row_index = pd.MultiIndex.from_product(
        [idx_union.sort_values(), symbols],
        names=["timestamp", "symbol"],
    )
    matrix = pd.DataFrame(index=row_index)
    for key, series in frames.items():
        series.index = series.index.set_names(["timestamp", "symbol"])
        matrix[key] = series.reindex(row_index).astype(np.float32)
    return matrix


def _history_matrix_to_feature_dict(
    matrix: pd.DataFrame,
    *,
    symbols: List[str],
) -> Dict[str, pd.DataFrame]:
    if matrix is None or matrix.empty or not isinstance(matrix.index, pd.MultiIndex):
        return {}
    out: Dict[str, pd.DataFrame] = {}
    matrix = matrix.copy()
    matrix.index = matrix.index.set_names(["timestamp", "symbol"])
    for key in matrix.columns:
        series = matrix[key]
        try:
            df = series.unstack("symbol").reindex(columns=symbols)
        except Exception:
            continue
        if isinstance(df, pd.DataFrame) and not df.empty:
            out[str(key)] = df.astype(np.float32)
    return out


def _load_live_feature_rolling_cache(
    *,
    cfg: Dict[str, Any],
    run_id: str,
    cache_key: str,
    symbols: List[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    required_feature_keys: Set[str],
) -> Dict[str, pd.DataFrame]:
    if not bool(cfg.get("live_feature_rolling_cache_enabled", True)):
        return {}
    cache_dir = _feature_snapshot_dir(cfg, run_id, cache_key)
    meta_path = cache_dir / "rolling_meta.json"
    data_path = cache_dir / "rolling_history.parquet"
    if not (meta_path.exists() and data_path.exists()):
        return {}
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("version") != LIVE_FEATURE_CACHE_VERSION:
            return {}
        if meta.get("cache_key") != cache_key:
            return {}
        if meta.get("symbols_hash") != _hash_values(symbols):
            return {}
        if meta.get("required_hash") != _hash_values(required_feature_keys):
            return {}
        expected_contract_hash = _feature_transform_contract_hash_from_cfg(cfg)
        if (
            expected_contract_hash
            and meta.get("contract_hash") != expected_contract_hash
        ):
            return {}
        matrix = pd.read_parquet(data_path)
    except Exception:
        return {}
    if matrix.empty or not isinstance(matrix.index, pd.MultiIndex):
        return {}
    missing = required_feature_keys.difference(str(c) for c in matrix.columns)
    if missing:
        return {}
    try:
        ts = pd.to_datetime(matrix.index.get_level_values("timestamp"), utc=True)
        mask = (ts >= pd.Timestamp(start_ts)) & (ts <= pd.Timestamp(end_ts))
        matrix = matrix.loc[mask]
    except Exception:
        return {}
    if matrix.empty:
        return {}
    tprint(
        "Loaded rolling live transformed feature cache: "
        f"rows={len(matrix.index)} features={len(matrix.columns)} "
        f"end_ts={meta.get('end_ts')}"
    )
    return _history_matrix_to_feature_dict(matrix, symbols=symbols)


def _write_live_feature_rolling_cache(
    *,
    cfg: Dict[str, Any],
    run_id: str,
    cache_key: str,
    feats: Dict[str, pd.DataFrame],
    symbols: List[str],
    end_ts: pd.Timestamp,
    required_feature_keys: Set[str],
    append_after_ts: Optional[pd.Timestamp],
    keep_start_ts: pd.Timestamp,
) -> None:
    if not bool(cfg.get("live_feature_rolling_cache_enabled", True)):
        return
    try:
        cache_dir = _feature_snapshot_dir(cfg, run_id, cache_key)
        cache_dir.mkdir(parents=True, exist_ok=True)
        new_matrix = _feature_history_matrix(
            feats,
            symbols=symbols,
            required_feature_keys=required_feature_keys,
            start_ts=append_after_ts,
            end_ts=end_ts,
        )
        if new_matrix.empty:
            return
        data_path = cache_dir / "rolling_history.parquet"
        if data_path.exists():
            try:
                old_matrix = pd.read_parquet(data_path)
                if isinstance(old_matrix.index, pd.MultiIndex) and not old_matrix.empty:
                    matrix = pd.concat([old_matrix, new_matrix], axis=0)
                    matrix = matrix[~matrix.index.duplicated(keep="last")]
                else:
                    matrix = new_matrix
            except Exception:
                matrix = new_matrix
        else:
            matrix = new_matrix
        try:
            ts = pd.to_datetime(matrix.index.get_level_values("timestamp"), utc=True)
            matrix = matrix.loc[ts >= pd.Timestamp(keep_start_ts)]
        except Exception:
            pass
        matrix = matrix.sort_index()
        tmp_data = cache_dir / "rolling_history.tmp.parquet"
        matrix.to_parquet(tmp_data)
        tmp_data.replace(data_path)
        meta = {
            "version": LIVE_FEATURE_CACHE_VERSION,
            "cache_key": cache_key,
            "feature_runtime_cfg_hash": _feature_runtime_cfg_hash(cfg),
            "contract_hash": _feature_transform_contract_hash_from_cfg(cfg),
            "symbols_hash": _hash_values(symbols),
            "required_hash": _hash_values(required_feature_keys),
            "end_ts": pd.Timestamp(end_ts).isoformat(),
            "append_after_ts": (
                None
                if append_after_ts is None
                else pd.Timestamp(append_after_ts).isoformat()
            ),
            "keep_start_ts": pd.Timestamp(keep_start_ts).isoformat(),
            "rows": int(len(matrix.index)),
            "features": list(matrix.columns),
            "symbols": list(symbols),
        }
        tmp_meta = cache_dir / "rolling_meta.tmp.json"
        tmp_meta.write_text(json.dumps(meta))
        tmp_meta.replace(cache_dir / "rolling_meta.json")
        tprint(
            "Persisted rolling live transformed feature cache: "
            f"new_rows={len(new_matrix.index)} total_rows={len(matrix.index)} "
            f"features={len(matrix.columns)} end_ts={end_ts}"
        )
    except Exception as exc:
        tprint(f"Warning: failed to persist rolling live feature cache: {exc}")


def _load_live_feature_snapshot(
    *,
    cfg: Dict[str, Any],
    run_id: str,
    cache_key: str,
    symbols: List[str],
    end_ts: pd.Timestamp,
    required_feature_keys: Set[str],
) -> Dict[str, pd.DataFrame]:
    if not bool(cfg.get("live_feature_snapshot_cache_enabled", True)):
        return {}
    cache_dir = _feature_snapshot_dir(cfg, run_id, cache_key)
    meta_path = cache_dir / "meta.json"
    data_path = cache_dir / "latest.parquet"
    if not (meta_path.exists() and data_path.exists()):
        return {}
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("version") != LIVE_FEATURE_CACHE_VERSION:
            return {}
        if meta.get("cache_key") != cache_key:
            return {}
        if pd.Timestamp(meta.get("end_ts")) != pd.Timestamp(end_ts):
            return {}
        if meta.get("symbols_hash") != _hash_values(symbols):
            return {}
        if meta.get("required_hash") != _hash_values(required_feature_keys):
            return {}
        expected_contract_hash = _feature_transform_contract_hash_from_cfg(cfg)
        if (
            expected_contract_hash
            and meta.get("contract_hash") != expected_contract_hash
        ):
            return {}
        matrix = pd.read_parquet(data_path)
    except Exception:
        return {}
    missing = required_feature_keys.difference(str(c) for c in matrix.columns)
    if missing:
        return {}
    tprint(
        "Loaded persisted live transformed feature snapshot: "
        f"symbols={len(matrix.index)} features={len(matrix.columns)} end_ts={end_ts}"
    )
    return _matrix_to_feature_dict(matrix, end_ts=end_ts)


def _write_live_feature_snapshot(
    *,
    cfg: Dict[str, Any],
    run_id: str,
    cache_key: str,
    feats: Dict[str, pd.DataFrame],
    raw_panel: Optional[Dict[str, pd.DataFrame]] = None,
    raw_start_ts: Optional[pd.Timestamp] = None,
    symbols: List[str],
    end_ts: pd.Timestamp,
    required_feature_keys: Set[str],
) -> None:
    if not bool(cfg.get("live_feature_snapshot_cache_enabled", True)):
        return
    try:
        cache_dir = _feature_snapshot_dir(cfg, run_id, cache_key)
        cache_dir.mkdir(parents=True, exist_ok=True)
        matrix = _latest_feature_matrix(feats, symbols, end_ts, required_feature_keys)
        if matrix.empty:
            return
        tmp_data = cache_dir / "latest.tmp.parquet"
        data_path = cache_dir / "latest.parquet"
        matrix.to_parquet(tmp_data)
        tmp_data.replace(data_path)
        raw_panel_fields: List[str] = []
        raw_panel_rows = 0
        if bool(cfg.get("live_feature_snapshot_raw_panel_enabled", True)) and raw_panel:
            raw_matrix_parts: List[pd.DataFrame] = []
            for field, frame in raw_panel.items():
                if not isinstance(frame, pd.DataFrame) or frame.empty:
                    continue
                cols = [sym for sym in symbols if sym in frame.columns]
                if not cols:
                    continue
                field_frame = frame.loc[:, cols].copy()
                field_frame.index = pd.to_datetime(
                    field_frame.index, utc=True, errors="coerce"
                )
                field_frame = field_frame[~pd.isna(field_frame.index)]
                if field_frame.empty:
                    continue
                if raw_start_ts is not None:
                    field_frame = field_frame[
                        field_frame.index >= pd.Timestamp(raw_start_ts)
                    ]
                field_frame = field_frame[field_frame.index <= pd.Timestamp(end_ts)]
                field_frame = field_frame.sort_index()
                if field_frame.empty:
                    continue
                field_frame = field_frame.stack(dropna=False).rename(str(field))
                raw_matrix_parts.append(field_frame.to_frame())
                raw_panel_fields.append(str(field))
            if raw_matrix_parts:
                raw_matrix = pd.concat(raw_matrix_parts, axis=1).sort_index()
                raw_matrix.index = raw_matrix.index.set_names(["timestamp", "symbol"])
                raw_matrix = raw_matrix[~raw_matrix.index.duplicated(keep="last")]
                raw_panel_rows = int(len(raw_matrix.index))
                tmp_raw = cache_dir / "raw_panel.tmp.parquet"
                raw_path = cache_dir / "raw_panel.parquet"
                raw_matrix.to_parquet(tmp_raw)
                tmp_raw.replace(raw_path)
        meta = {
            "version": LIVE_FEATURE_CACHE_VERSION,
            "cache_key": cache_key,
            "feature_runtime_cfg_hash": _feature_runtime_cfg_hash(cfg),
            "contract_hash": _feature_transform_contract_hash_from_cfg(cfg),
            "symbols_hash": _hash_values(symbols),
            "required_hash": _hash_values(required_feature_keys),
            "end_ts": pd.Timestamp(end_ts).isoformat(),
            "features": list(matrix.columns),
            "symbols": list(matrix.index),
            "raw_panel_path": "raw_panel.parquet" if raw_panel_rows else None,
            "raw_panel_fields": raw_panel_fields,
            "raw_panel_rows": raw_panel_rows,
            "raw_panel_start_ts": (
                None if raw_start_ts is None else pd.Timestamp(raw_start_ts).isoformat()
            ),
        }
        tmp_meta = cache_dir / "meta.tmp.json"
        tmp_meta.write_text(json.dumps(meta))
        tmp_meta.replace(cache_dir / "meta.json")
        tprint(
            "Persisted live transformed feature snapshot: "
            f"symbols={len(matrix.index)} features={len(matrix.columns)} end_ts={end_ts}"
        )
    except Exception as exc:
        tprint(f"Warning: failed to persist live feature snapshot: {exc}")


def is_model_derived_feature_key(key: str) -> bool:
    """Return True for features generated by model inference, not raw OHLCV."""
    return bool(isinstance(key, str) and MODEL_DERIVED_FEATURE_RE.match(key))


DELETED_INFERENCE_FEATURE_KEYS: Set[str] = {
    "p_exh_lag1",
    "retest_accept",
    "vol_price_diverge",
    "vortex_diff_14",
    "vortex_diff_21",
    "vortex_diff_34",
    "z_breakout_dn_24",
    "z_breakout_up_24",
    "z_slope_change_24",
    "z_sm_momentum_24",
}


def raw_required_feature_keys(
    required_feature_keys: Optional[Iterable[str]],
) -> Set[str]:
    """Filter a full inference contract down to raw/live-computable features."""
    return {
        str(key)
        for key in (required_feature_keys or set())
        if str(key)
        and str(key) not in DELETED_INFERENCE_FEATURE_KEYS
        and not is_model_derived_feature_key(str(key))
    }


def _raw_feature_compute_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return feature-compute cfg for raw panels without refitting transforms."""
    out = dict(cfg or {})
    if out.get("feature_transform_contract") is not None:
        out["feature_transform_contract_raw_mode"] = True
    # Live inference must reproduce the saved model contract. Current config
    # portability policy may be stricter than the artifact that was trained and
    # deployed; strict source-panel and final-matrix gates below still reject
    # genuinely unavailable, non-finite, or missing model inputs.
    out["feature_portability_mode"] = "legacy"
    out["feature_portability_strict"] = False
    return out


def _transform_feature_panels_for_inference(
    feats: Dict[str, pd.DataFrame],
    cfg: Optional[Dict[str, Any]],
    *,
    strict: bool = True,
    label: str = "inference",
) -> Dict[str, pd.DataFrame]:
    """Apply the fitted training transform contract to live feature panels."""
    cfg = cfg or {}
    contract = cfg.get("feature_transform_contract")
    if not isinstance(contract, FeatureTransformContract):
        bundle = cfg.get("bundle")
        if isinstance(bundle, dict):
            contract = bundle.get("feature_transform_contract")
    if not isinstance(contract, FeatureTransformContract):
        if strict:
            raise RuntimeError(f"{label}: missing feature_transform_contract")
        return feats
    expected_contract_hash = _feature_transform_contract_hash_from_cfg(cfg)
    if expected_contract_hash and expected_contract_hash != contract.contract_hash:
        raise RuntimeError(
            f"{label}: feature transform contract hash mismatch: "
            f"{contract.contract_hash} != {expected_contract_hash}"
        )
    return contract.transform_panels(feats, strict=strict)


def _requires_gated_feature_generation(
    required_feature_keys: Optional[Set[str]],
) -> bool:
    """Return True when the requested feature set needs gated feature families.

    The alpha bundles for some strategies include gate-conditioned columns such
    as ``*_G_VOL_0`` and ``*_G_VOL_1``. Those are only generated when gated
    feature construction is enabled in the shared feature pipeline.
    """
    if not required_feature_keys:
        return False

    for key in required_feature_keys:
        if not isinstance(key, str) or not key:
            continue
        if key in {"G_VOL", "G_TREND"}:
            return True
        if "_G_VOL_" in key or "_G_TREND_" in key:
            return True
    return False


def _required_tail_warmup_hours(
    lookback_hours: int,
    trend_sma_hours: int,
    gate_vol_lookback_hours: int,
    tail_compute_hours: Optional[int] = None,
) -> int:
    """Choose the smallest safe warmup window for incremental inference backfills."""
    if tail_compute_hours is not None:
        return int(tail_compute_hours)
    # The lookback window is already covered by cached stored features. For
    # incremental backfill we still need enough history to stabilize both
    # raw rolling/gated features and CausalFeatureTransformer's rolling
    # standardization.  Using less than the transform window creates a live-vs-
    # replay parity break around the cache/tail seam.
    base_hours = max(
        int(trend_sma_hours),
        int(gate_vol_lookback_hours),
        int(DEFAULT_CAUSAL_TRANSFORM_ROLL_WINDOW_HOURS),
        int(DEFAULT_IDENTITY_EWMA_WARMUP_HOURS),
        24 * 7,
    )
    return base_hours + DEFAULT_TAIL_WARMUP_BUFFER_HOURS


def _slice_feature_window(
    feats: Dict[str, pd.DataFrame],
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for key, df in feats.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        sliced = df
        if start_ts is not None:
            sliced = sliced[sliced.index >= start_ts]
        if end_ts is not None:
            sliced = sliced[sliced.index <= end_ts]
        out[key] = sliced
    return out


def _merge_feature_dicts(
    cached_feats: Dict[str, pd.DataFrame],
    new_feats: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    merged: Dict[str, pd.DataFrame] = {}
    all_keys = sorted(set(cached_feats.keys()) | set(new_feats.keys()))
    for key in all_keys:
        left = cached_feats.get(key)
        right = new_feats.get(key)
        if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
            df = pd.concat([left, right])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            merged[key] = df
        elif isinstance(left, pd.DataFrame):
            merged[key] = left
        elif isinstance(right, pd.DataFrame):
            merged[key] = right
    return merged


def _merge_feature_dicts_preserve_cached(
    cached_feats: Dict[str, pd.DataFrame],
    new_feats: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Merge feature dictionaries while preserving cached training values."""
    merged: Dict[str, pd.DataFrame] = {}
    all_keys = sorted(set(cached_feats.keys()) | set(new_feats.keys()))
    for key in all_keys:
        left = cached_feats.get(key)
        right = new_feats.get(key)
        if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
            df = left.combine_first(right).sort_index()
            merged[key] = df
        elif isinstance(left, pd.DataFrame):
            merged[key] = left
        elif isinstance(right, pd.DataFrame):
            merged[key] = right
    return merged


def _merge_missing_feature_dicts(
    cached_feats: Dict[str, pd.DataFrame],
    new_feats: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Merge feature dictionaries without replacing already-cached feature keys."""
    if not new_feats:
        return dict(cached_feats or {})
    merged = dict(cached_feats or {})
    for key, value in new_feats.items():
        if key not in merged:
            merged[key] = value
    return merged


def _cached_feature_coverage_end_ts(
    cached_feats: Dict[str, pd.DataFrame],
    required_feature_keys: Optional[Set[str]] = None,
) -> Optional[pd.Timestamp]:
    """Return the deterministic timestamp through which cached features cover.

    The tail-backfill seam must not depend on dictionary insertion/hash order.
    Use the minimum latest timestamp across cached required feature frames so
    any stale cached feature triggers a tail recompute from a stable point.
    """
    if not cached_feats:
        return None

    required = set(required_feature_keys or set())
    candidates: List[pd.Timestamp] = []
    for key in sorted(cached_feats):
        if required and key not in required:
            continue
        df = cached_feats.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        try:
            idx = pd.DatetimeIndex(df.index)
            if idx.empty:
                continue
            candidates.append(pd.Timestamp(idx.max()))
        except Exception:
            continue

    # If none of the requested frames are present in the cache, fall back to the
    # whole cache so missing requested keys can still be backfilled later.
    if not candidates and required:
        return _cached_feature_coverage_end_ts(cached_feats, None)
    if not candidates:
        return None
    return min(candidates)


def _is_live_stale_sensitive_feature_key(key: str) -> bool:
    """Return True for features that must never be forward-filled indefinitely.

    Orderbook and microstructure features are valid only when the live panel can
    regenerate them for the current signal timestamp. If an old selected-feature
    cache has these keys but stops before the live timestamp, carrying the last
    finite value forward creates a train/inference parity break.
    """
    key_l = str(key or "").lower()
    if key_l.startswith(("ob_", "obw_")):
        return True
    return "orderbook" in key_l or (key_l.startswith("xasset_") and "ob" in key_l)


def _training_feature_has_variation(
    *,
    data_root: str,
    run_id: str,
    feature_key: str,
) -> bool:
    """Return whether the deployed training artifact saw this feature vary.

    Live must not inject a varying model feature when the OOF/OOS artifact was
    trained with that feature constant/neutral.  Liquidity checks still use
    live orderbooks separately; this guard only controls ML model inputs.
    """
    cache_key = (str(data_root), str(run_id))
    cached = _TRAINING_FEATURE_VARIATION_CACHE.get(cache_key)
    if cached is None:
        cached = {}
        path = (
            Path(data_root)
            / "artifacts"
            / str(run_id)
            / "features"
            / "feature_health_feature_detail.csv"
        )
        if path.exists():
            try:
                health = pd.read_csv(
                    path,
                    usecols=["feature", "is_all_nan", "is_constant_non_nan"],
                )
                health["feature"] = health["feature"].astype(str)
                for feat, grp in health.groupby("feature", sort=False):
                    all_nan = grp["is_all_nan"].astype(bool)
                    constant = grp["is_constant_non_nan"].astype(bool)
                    cached[str(feat)] = bool((~all_nan & ~constant).any())
            except Exception as exc:
                tprint(
                    "WARNING: Failed to load training feature variation contract "
                    f"from {path}: {exc}; allowing live feature materialization."
                )
        _TRAINING_FEATURE_VARIATION_CACHE[cache_key] = cached
    if not cached:
        return True
    return bool(cached.get(str(feature_key), True))


def _drop_stale_live_sensitive_features(
    feats: Dict[str, pd.DataFrame],
    *,
    end_ts: pd.Timestamp,
    required_feature_keys: Optional[Set[str]],
) -> Dict[str, pd.DataFrame]:
    """Drop required live-sensitive features that do not reach ``end_ts``.

    Dropped keys are later reconstructed from current live panels. Strict
    portability modes reject missing orderbook/funding features rather than
    neutralizing them.
    """
    if not feats or not required_feature_keys:
        return feats

    end = pd.Timestamp(end_ts)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")

    out: Dict[str, pd.DataFrame] = {}
    dropped: Dict[str, str] = {}
    for key, df in feats.items():
        if (
            key in required_feature_keys
            and _is_live_stale_sensitive_feature_key(key)
            and isinstance(df, pd.DataFrame)
            and not df.empty
        ):
            try:
                idx = pd.to_datetime(df.index, utc=True, errors="coerce")
                finite_idx = idx[~pd.isna(idx)]
                max_ts = finite_idx.max() if len(finite_idx) else None
            except Exception:
                max_ts = None
            if max_ts is None or pd.Timestamp(max_ts) < end:
                dropped[str(key)] = (
                    "missing_timestamp"
                    if max_ts is None
                    else pd.Timestamp(max_ts).isoformat()
                )
                continue
        out[key] = df

    if dropped:
        sample = dict(list(sorted(dropped.items()))[:12])
        tprint(
            "Dropped stale live-sensitive orderbook features before inference "
            f"materialization: n={len(dropped)} end_ts={end.isoformat()} sample={sample}"
        )
    return out


def _compute_policy_barrier_pct(
    panel: Dict[str, pd.DataFrame],
    basket_syms: List[str],
    cfg: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """Compute the raw optimiser barrier fraction from raw OHLCV.

    The policy optimiser consumes label ``barrier_pct`` values generated from
    raw ATR% before CausalTransform. Live stop placement must not use the
    transformed ``atr_pct``/``atr_pct_base`` model features, because those may
    be standardized or otherwise unsuitable as price-distance fractions.
    """
    high = panel.get("high")
    low = panel.get("low")
    close = panel.get("close")
    if (
        not isinstance(high, pd.DataFrame)
        or high.empty
        or not isinstance(low, pd.DataFrame)
        or low.empty
        or not isinstance(close, pd.DataFrame)
        or close.empty
    ):
        return None

    cols = [
        sym
        for sym in basket_syms
        if sym in high.columns and sym in low.columns and sym in close.columns
    ]
    if not cols:
        return None

    atr_n = max(1, int((cfg or {}).get("atr_n", 14)))
    high_raw = high.loc[:, cols].astype(np.float32)
    low_raw = low.loc[:, cols].astype(np.float32)
    close_raw = close.loc[:, cols].astype(np.float32)
    barrier = (
        atr_percent(high_raw, low_raw, close_raw, atr_n) / (close_raw + 1e-12)
    ).astype(np.float32)
    barrier = barrier.replace([np.inf, -np.inf], np.nan)
    return barrier.clip(lower=np.float32(0.005))


def _backfill_missing_requested_keys(
    panel: Dict[str, pd.DataFrame],
    basket_syms: List[str],
    cfg: Dict[str, Any],
    merged_feats: Dict[str, pd.DataFrame],
    missing_keys: Set[str],
) -> Dict[str, pd.DataFrame]:
    """Compute and merge any requested feature keys that are still missing.

    This is the self-corrective step for inference: if the cached feature cache
    and the lightweight selector cache do not satisfy the exact model contract,
    we recompute the missing feature family directly from the panel.
    """
    if not missing_keys:
        return merged_feats

    # Live-sensitive orderbook/microstructure keys are handled by
    # _synthesize_live_safe_feature_keys(), which materializes current live
    # summaries where possible and neutralizes unavailable flow-only fields.
    # Sending those keys through compute_features_hourly() can trigger a full
    # historical feature rebuild during live inference, delaying entries long
    # enough to make the signal stale.
    compute_missing_keys = {
        key for key in missing_keys if not _is_live_stale_sensitive_feature_key(key)
    }
    if not compute_missing_keys:
        return merged_feats

    compute_panel: Dict[str, pd.DataFrame] = {}
    for key, df in panel.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            compute_panel[key] = df.copy()
    if not compute_panel:
        return merged_feats

    local_cfg = _raw_feature_compute_cfg(cfg)
    if _requires_gated_feature_generation(compute_missing_keys):
        local_cfg["enable_gated_features"] = True

    # Compute only the missing keys, then merge them into the existing feature map.
    mkt_df = compute_market_features(
        compute_panel, basket_syms, trend_sma_hours=DEFAULT_TREND_SMA_HOURS
    )
    mkt_gates = add_regime_gates(
        mkt_df,
        gate_vol_lookback_hours=DEFAULT_GATE_VOL_LOOKBACK_HOURS,
        gate_trend_thr=DEFAULT_GATE_TREND_THR,
    )
    missing_feats, missing_index, missing_columns = compute_features_hourly(
        compute_panel,
        mkt_gates,
        local_cfg,
        requested_feature_keys=sorted(compute_missing_keys),
    )

    if not missing_feats:
        return merged_feats

    ref_index = None
    for df in compute_panel.values():
        if isinstance(df, pd.DataFrame) and not df.empty:
            ref_index = df.index
            break
    if ref_index is None:
        for df in merged_feats.values():
            if isinstance(df, pd.DataFrame) and not df.empty:
                ref_index = df.index
                break
    if ref_index is None:
        return merged_feats

    missing_frames: Dict[str, pd.DataFrame] = {}
    for feat_name, feat_value in missing_feats.items():
        if isinstance(feat_value, pd.DataFrame):
            missing_frames[feat_name] = feat_value
            continue
        arr = np.asarray(feat_value)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        value_index = ref_index
        if (
            missing_index is not None
            and hasattr(missing_index, "__len__")
            and len(missing_index) == arr.shape[0]
        ):
            value_index = missing_index
        missing_frames[feat_name] = pd.DataFrame(
            arr,
            index=value_index,
            columns=(
                missing_columns
                if missing_columns is not None and len(missing_columns) == arr.shape[1]
                else (
                    basket_syms[: arr.shape[1]]
                    if arr.shape[1] <= len(basket_syms)
                    else None
                )
            ),
        )

    if not missing_frames:
        return merged_feats

    return _merge_feature_dicts(merged_feats, missing_frames)


def load_cached_features_for_inference(
    run_id: str,
    data_root: str,
    symbols: List[str],
    feature_keys: Optional[Set[str]] = None,
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
) -> Dict[str, pd.DataFrame]:
    ts = pd.to_datetime(run_id, format="%Y%m%d_%H%M%S", utc=True)
    # load_features_selected uses half-open parquet pushdown filters. Live and
    # historical inference callers treat end_ts as inclusive, so pad the query
    # boundary and keep the explicit inclusive slice below.
    query_end_ts = (
        pd.Timestamp(end_ts) + pd.Timedelta(microseconds=1)
        if end_ts is not None
        else None
    )
    feats = load_features_selected(
        ts,
        data_root,
        feature_keys=sorted(feature_keys) if feature_keys else None,
        symbols=symbols,
        start_ts=start_ts,
        end_ts=query_end_ts,
    )
    if not hasattr(feats, "items"):
        return {}
    return _slice_feature_window(dict(feats), start_ts=start_ts, end_ts=end_ts)


def load_or_compute_features(
    panel: Dict[str, pd.DataFrame],
    basket_syms: List[str],
    run_id: str,
    data_root: str,
    cfg: Dict[str, Any],
    lookback_hours: int,
    required_feature_keys: Optional[Set[str]] = None,
    trend_sma_hours: int = DEFAULT_TREND_SMA_HOURS,
    gate_vol_lookback_hours: int = DEFAULT_GATE_VOL_LOOKBACK_HOURS,
    gate_trend_thr: float = DEFAULT_GATE_TREND_THR,
    tail_compute_hours: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    timer = _StageTimer("load_or_compute_features")
    close = panel.get("close")
    if not isinstance(close, pd.DataFrame) or close.empty:
        return {}

    required_feature_keys = raw_required_feature_keys(required_feature_keys)
    outer_cfg = cfg if isinstance(cfg, dict) else {}
    cfg_source = outer_cfg.get("runtime_cfg", outer_cfg)
    cfg = dict(cfg_source or {})
    # Runtime config is the primary feature-generation config, but model
    # artifacts such as the fitted training transform contract often live on
    # the outer inference config/full state.  Preserve them before tail
    # recomputation so live/replay feature values use the same fitted
    # parameters as training.
    for key in (
        "feature_transform_contract",
        "feature_transform_contract_hash",
        "feature_transform_manifest",
    ):
        if key in outer_cfg and key not in cfg:
            cfg[key] = outer_cfg[key]
    for bundle_key in ("bundle", "model_bundle", "full_state"):
        bundle_value = outer_cfg.get(bundle_key)
        if isinstance(bundle_value, dict) and "bundle" not in cfg:
            cfg["bundle"] = bundle_value.get("bundle", bundle_value)
        if isinstance(bundle_value, dict):
            for key in (
                "feature_transform_contract",
                "feature_transform_contract_hash",
                "feature_transform_manifest",
            ):
                if key in bundle_value and key not in cfg:
                    cfg[key] = bundle_value[key]
            inner_bundle = bundle_value.get("bundle")
            if isinstance(inner_bundle, dict):
                for key in (
                    "feature_transform_contract",
                    "feature_transform_contract_hash",
                    "feature_transform_manifest",
                ):
                    if key in inner_bundle and key not in cfg:
                        cfg[key] = inner_bundle[key]
    if _requires_gated_feature_generation(required_feature_keys):
        cfg["enable_gated_features"] = True
    cfg.setdefault("feature_transform_cache_enabled", False)

    end_ts = close.index.max()
    start_ts = end_ts - pd.Timedelta(hours=lookback_hours)
    cache_key = _live_feature_cache_key(
        run_id=run_id,
        symbols=basket_syms,
        required_feature_keys=required_feature_keys,
        lookback_hours=lookback_hours,
        cfg=cfg,
        data_root=data_root,
    )
    if bool(cfg.get("live_feature_memory_cache_enabled", True)):
        memory_entry = _LIVE_FEATURE_MEMORY_CACHE.get(cache_key)
        if isinstance(memory_entry, dict):
            memory_end = memory_entry.get("end_ts")
            memory_feats = memory_entry.get("feats")
            if (
                memory_end is not None
                and pd.Timestamp(memory_end) == pd.Timestamp(end_ts)
                and isinstance(memory_feats, dict)
            ):
                tprint(
                    "Loaded in-memory live transformed feature cache: "
                    f"features={len(memory_feats)} end_ts={end_ts}"
                )
                return memory_feats

    snapshot_feats = _load_live_feature_snapshot(
        cfg=cfg,
        run_id=run_id,
        cache_key=cache_key,
        symbols=basket_syms,
        end_ts=end_ts,
        required_feature_keys=required_feature_keys,
    )
    if snapshot_feats:
        if bool(cfg.get("live_feature_memory_cache_enabled", True)):
            _LIVE_FEATURE_MEMORY_CACHE[cache_key] = {
                "end_ts": end_ts,
                "feats": snapshot_feats,
            }
        timer.mark("snapshot_cache_hit")
        return snapshot_feats

    cached_feats = {}
    memory_entry = _LIVE_FEATURE_MEMORY_CACHE.get(cache_key)
    if isinstance(memory_entry, dict) and isinstance(memory_entry.get("feats"), dict):
        cached_feats = _slice_feature_window(
            memory_entry["feats"], start_ts=start_ts, end_ts=end_ts
        )
        if cached_feats:
            tprint(
                "Using in-memory live feature history as tail base: "
                f"features={len(cached_feats)}"
            )
    offline_feature_data_root = str(cfg.get("offline_feature_data_root") or data_root)
    if not cached_feats and bool(cfg.get("live_feature_offline_cache_enabled", True)):
        cached_feats = load_cached_features_for_inference(
            run_id=run_id,
            data_root=offline_feature_data_root,
            symbols=basket_syms,
            feature_keys=required_feature_keys,
            start_ts=start_ts,
            end_ts=end_ts,
        )
    offline_cached_last_ts = _cached_feature_coverage_end_ts(
        cached_feats,
        required_feature_keys=required_feature_keys,
    )
    rolling_feats = _load_live_feature_rolling_cache(
        cfg=cfg,
        run_id=run_id,
        cache_key=cache_key,
        symbols=basket_syms,
        start_ts=start_ts,
        end_ts=end_ts,
        required_feature_keys=required_feature_keys,
    )
    if rolling_feats:
        cached_feats = _merge_feature_dicts(cached_feats, rolling_feats)
    timer.mark("load_cached_transformed_features")

    cached_last_ts = _cached_feature_coverage_end_ts(
        cached_feats,
        required_feature_keys=required_feature_keys,
    )

    need_tail_backfill = cached_last_ts is None or end_ts > cached_last_ts
    # Always layer the lightweight candidate-selector features on top of the
    # stored offline feature cache, because the offline cache does not
    # necessarily contain keys like ret24h/range_12h_pct used by inference.
    selector_feats = _compute_per_symbol_features(panel, basket_syms)
    policy_barrier = _compute_policy_barrier_pct(panel, basket_syms, cfg)
    if isinstance(policy_barrier, pd.DataFrame) and not policy_barrier.empty:
        selector_feats["barrier_pct"] = policy_barrier
    timer.mark("compute_selector_features")

    if not need_tail_backfill:
        tprint(
            f"Loaded stored inference features for {len(basket_syms)} symbols through {cached_last_ts}"
        )
        merged = _merge_missing_feature_dicts(cached_feats, selector_feats)
        merged = _drop_stale_live_sensitive_features(
            merged,
            end_ts=end_ts,
            required_feature_keys=required_feature_keys,
        )
        if required_feature_keys:
            missing = {k for k in required_feature_keys if k not in merged}
            merged = _backfill_missing_requested_keys(
                panel=panel,
                basket_syms=basket_syms,
                cfg=cfg,
                merged_feats=merged,
                missing_keys=missing,
            )
            merged = _synthesize_gated_feature_keys(
                merged, panel, basket_syms, required_feature_keys
            )
            merged = _synthesize_live_safe_feature_keys(
                merged,
                panel,
                basket_syms,
                required_feature_keys,
                data_root=data_root,
                run_id=run_id,
                cfg=cfg,
            )
        merged = _slice_feature_window(merged, start_ts=start_ts, end_ts=end_ts)
        if bool(cfg.get("live_feature_memory_cache_enabled", True)):
            _LIVE_FEATURE_MEMORY_CACHE[cache_key] = {"end_ts": end_ts, "feats": merged}
        _write_live_feature_snapshot(
            cfg=cfg,
            run_id=run_id,
            cache_key=cache_key,
            feats=merged,
            raw_panel=panel,
            raw_start_ts=start_ts,
            symbols=basket_syms,
            end_ts=end_ts,
            required_feature_keys=required_feature_keys,
        )
        _write_live_feature_rolling_cache(
            cfg=cfg,
            run_id=run_id,
            cache_key=cache_key,
            feats=merged,
            symbols=basket_syms,
            end_ts=end_ts,
            required_feature_keys=required_feature_keys,
            append_after_ts=(
                offline_cached_last_ts
                if offline_cached_last_ts is not None
                else cached_last_ts
            ),
            keep_start_ts=start_ts,
        )
        timer.mark("feature_cache_ready_no_tail")
        return merged

    tail_warmup_hours = _required_tail_warmup_hours(
        lookback_hours=lookback_hours,
        trend_sma_hours=trend_sma_hours,
        gate_vol_lookback_hours=gate_vol_lookback_hours,
        tail_compute_hours=tail_compute_hours,
    )
    tail_start_ts = max(
        close.index.min(),
        (
            (cached_last_ts - pd.Timedelta(hours=tail_warmup_hours))
            if cached_last_ts is not None
            else start_ts
        ),
    )
    panel_tail: Dict[str, pd.DataFrame] = {}
    for key, df in panel.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            panel_tail[key] = df[df.index >= tail_start_ts]

    tprint(
        "Stored features missing latest timestamps; computing tail-only feature backfill "
        f"from {tail_start_ts} to {end_ts}"
    )
    tprint(
        "Live feature parity: tail backfill uses shared training/backtest "
        "compute_features_hourly(); fitted training transform parameters are "
        "then applied to available raw feature panels."
    )
    mkt_df = compute_market_features(
        panel_tail, basket_syms, trend_sma_hours=trend_sma_hours
    )
    timer.mark("compute_market_features_tail")
    mkt_gates = add_regime_gates(
        mkt_df,
        gate_vol_lookback_hours=gate_vol_lookback_hours,
        gate_trend_thr=gate_trend_thr,
    )
    timer.mark("compute_regime_gates_tail")
    full_tail_feats, _, _ = compute_features_hourly(
        panel_tail,
        mkt_gates,
        _raw_feature_compute_cfg(cfg),
        requested_feature_keys=(
            sorted(required_feature_keys) if required_feature_keys else None
        ),
    )
    if _feature_transform_contract_hash_from_cfg(cfg):
        full_tail_feats = _transform_feature_panels_for_inference(
            full_tail_feats,
            cfg,
            strict=False,
            label="live_tail_backfill",
        )
    timer.mark("compute_features_hourly_tail")
    if cached_last_ts is not None:
        full_tail_feats = {
            key: df[df.index > cached_last_ts]
            for key, df in full_tail_feats.items()
            if isinstance(df, pd.DataFrame) and not df.empty
        }

    preserve_cached = bool(
        cfg.get("historical_inference_parity_preserve_cached_features", False)
    )
    if preserve_cached:
        merged_feats = _merge_feature_dicts_preserve_cached(cached_feats, full_tail_feats)
    else:
        merged_feats = _merge_feature_dicts(cached_feats, full_tail_feats)
    merged_feats = _merge_missing_feature_dicts(merged_feats, selector_feats)
    merged_feats = _slice_feature_window(merged_feats, start_ts=start_ts, end_ts=end_ts)
    if not preserve_cached:
        merged_feats = _drop_stale_live_sensitive_features(
            merged_feats,
            end_ts=end_ts,
            required_feature_keys=required_feature_keys,
        )
    if required_feature_keys:
        missing = {k for k in required_feature_keys if k not in merged_feats}
        merged_feats = _backfill_missing_requested_keys(
            panel=panel,
            basket_syms=basket_syms,
            cfg=cfg,
            merged_feats=merged_feats,
            missing_keys=missing,
        )
        merged_feats = _synthesize_gated_feature_keys(
            merged_feats, panel, basket_syms, required_feature_keys
        )
        merged_feats = _synthesize_live_safe_feature_keys(
            merged_feats,
            panel,
            basket_syms,
            required_feature_keys,
            data_root=data_root,
            run_id=run_id,
            cfg=cfg,
        )
    if bool(cfg.get("live_feature_memory_cache_enabled", True)):
        _LIVE_FEATURE_MEMORY_CACHE[cache_key] = {
            "end_ts": end_ts,
            "feats": merged_feats,
        }
    _write_live_feature_snapshot(
        cfg=cfg,
        run_id=run_id,
        cache_key=cache_key,
        feats=merged_feats,
        raw_panel=panel,
        raw_start_ts=start_ts,
        symbols=basket_syms,
        end_ts=end_ts,
        required_feature_keys=required_feature_keys,
    )
    rolling_seed_hours = int(cfg.get("live_feature_rolling_cache_seed_hours", 24 * 14))
    rolling_append_after_ts = cached_last_ts
    if rolling_append_after_ts is None:
        rolling_append_after_ts = max(
            pd.Timestamp(start_ts),
            pd.Timestamp(end_ts) - pd.Timedelta(hours=rolling_seed_hours),
        )
    _write_live_feature_rolling_cache(
        cfg=cfg,
        run_id=run_id,
        cache_key=cache_key,
        feats=merged_feats,
        symbols=basket_syms,
        end_ts=end_ts,
        required_feature_keys=required_feature_keys,
        append_after_ts=rolling_append_after_ts,
        keep_start_ts=start_ts,
    )
    new_rows = 0
    if full_tail_feats:
        for df in full_tail_feats.values():
            if isinstance(df, pd.DataFrame):
                new_rows = len(df.index)
                break
    tprint(f"Loaded cached features and backfilled {new_rows} new timestamps")
    timer.mark("merge_and_cache_features")
    return merged_feats


def _model_key_matches_allowed(
    model_key: str,
    allowed: Optional[Set[str]],
) -> bool:
    if allowed is None:
        return True
    key = str(model_key or "")
    if strategy_id_matches(key, allowed):
        return True
    return any(
        key == allowed_key
        or key.startswith(f"{allowed_key}_")
        or allowed_key.startswith(f"{key}_")
        for allowed_key in allowed
    )


def _meta_feature_columns(meta: Any) -> List[str]:
    out: List[str] = []
    for source in (meta, getattr(meta, "best_model", None)):
        if source is None:
            continue
        raw = getattr(source, "meta_feature_columns_", None)
        if raw:
            out.extend(str(v) for v in raw if str(v))
    return list(dict.fromkeys(out))


def _effective_meta_feature_columns(meta: Any) -> List[str]:
    """Return the actual selected meta-model input contract, when available.

    Meta wrappers also carry the broad candidate feature matrix columns used
    during training. Live source gating must not treat that broad matrix as the
    prediction contract when the persisted winner selected a narrower named
    subset.
    """
    for source in (getattr(meta, "best_model", None), meta):
        if source is None:
            continue
        selected = [str(v) for v in (getattr(source, "selected_features", []) or [])]
        if not selected:
            continue
        input_features = [
            str(v) for v in (getattr(source, "input_feature_names", []) or [])
        ]
        if input_features and len(input_features) == len(selected):
            return input_features
        if all(re.fullmatch(r"f\d+", name) is not None for name in selected):
            continue
        return selected
    return []


def _meta_model_derived_raw_dependencies(feature_cols: Iterable[str]) -> Set[str]:
    """Return live-computable source keys needed for derived meta inputs."""
    deps: Set[str] = set()
    for raw in feature_cols or []:
        col = str(raw)
        if col == "base_med_x_side_aligned_trend":
            deps.update({"trend_slope_24h", "trend_t", "trend_pct", "trend_slope_72h", "trend_slope_48h"})
        elif col == "base_med_x_vol_z":
            deps.update({"vol_z24", "vol_z_4h", "vol_z", "volatility_zscore"})
        elif col == "base_med_x_efficiency_ratio":
            deps.update({"efficiency_ratio_20", "path_efficiency_24"})
        elif col == "base_med_x_compression_score":
            deps.add("compression_score")
        elif col == "base_med_x_compression_x_vol_z":
            deps.update({"compression_score", "vol_z24", "vol_z_4h", "vol_z", "volatility_zscore"})
        elif col == "base_med_x_side_trend_x_vol_z":
            deps.update(
                {
                    "trend_slope_24h",
                    "trend_t",
                    "trend_pct",
                    "trend_slope_72h",
                    "trend_slope_48h",
                    "vol_z24",
                    "vol_z_4h",
                    "vol_z",
                    "volatility_zscore",
                }
            )
        elif col == "base_med_x_side_trend_x_efficiency":
            deps.update(
                {
                    "trend_slope_24h",
                    "trend_t",
                    "trend_pct",
                    "trend_slope_72h",
                    "trend_slope_48h",
                    "efficiency_ratio_20",
                    "path_efficiency_24",
                }
            )
        elif col == "base_med_x_trend_24h_x_trend_72h":
            deps.update({"trend_slope_24h", "trend_t", "trend_pct", "trend_slope_72h", "trend_slope_48h"})
        elif col == "base_med_x_vol_z_24h_minus_96h":
            deps.update({"vol_z24", "vol_z_4h", "vol_z", "volatility_zscore"})
        elif col == "base_prob_x_vol_regime":
            deps.update({"regime_vol_score", "asset_vol_level"})
        elif col == "base_prob_x_entropy":
            deps.add("regime_transition_entropy_12h")
        elif col.startswith("base_prob_x_"):
            deps.add(col.removeprefix("base_prob_x_"))
        elif col.startswith("base_med_x_"):
            deps.add(col.removeprefix("base_med_x_"))
    return {dep for dep in deps if dep and dep not in DELETED_INFERENCE_FEATURE_KEYS}


def get_inference_required_feature_keys(
    model_bundle: Dict[str, Any],
    accepted_strategies: Optional[Iterable[str]] = None,
) -> Set[str]:
    """Extract the union of raw feature keys needed by live inference models."""
    required: Set[str] = set()
    allowed = (
        {str(strategy) for strategy in accepted_strategies if str(strategy)}
        if accepted_strategies is not None
        else None
    )
    bundle = (
        model_bundle.get("bundle", model_bundle)
        if isinstance(model_bundle, dict)
        else {}
    )

    def _effective_alpha_feature_cols(model_info: Dict[str, Any]) -> List[str]:
        feat_cols = [
            str(k)
            for k in (model_info.get("feat_cols", []) or [])
            if str(k) not in DELETED_INFERENCE_FEATURE_KEYS
        ]
        model = model_info.get("model")
        inner = getattr(model, "best_model", model)
        selected = [str(k) for k in (getattr(inner, "selected_features", []) or [])]
        input_features = [
            str(k) for k in (getattr(inner, "input_feature_names", []) or [])
        ]
        if selected:
            if input_features and len(input_features) == len(selected):
                return [
                    k for k in input_features if k not in DELETED_INFERENCE_FEATURE_KEYS
                ]
            if all(re.fullmatch(r"f\d+", name) is not None for name in selected):
                return feat_cols
            return [k for k in selected if k not in DELETED_INFERENCE_FEATURE_KEYS]
        return feat_cols

    alpha_models = bundle.get("alpha_models", {}) if isinstance(bundle, dict) else {}
    for key, value in alpha_models.items():
        if not _model_key_matches_allowed(str(key), allowed):
            continue
        if not isinstance(value, dict):
            continue
        if "feat_cols" in value:
            required.update(_effective_alpha_feature_cols(value))
            continue
        for nested_key, model_info in value.items():
            if not _model_key_matches_allowed(f"{key}_{nested_key}", allowed):
                continue
            if isinstance(model_info, dict):
                required.update(_effective_alpha_feature_cols(model_info))

    meta_models = bundle.get("meta_models", {}) if isinstance(bundle, dict) else {}
    for key, meta in meta_models.items():
        if not _model_key_matches_allowed(str(key), allowed):
            continue
        meta_cols = _effective_meta_feature_columns(meta)
        if not meta_cols:
            meta_cols = _meta_feature_columns(meta)
        if meta_cols:
            required.update(
                k for k in meta_cols if str(k) not in DELETED_INFERENCE_FEATURE_KEYS
            )
            required.update(_meta_model_derived_raw_dependencies(meta_cols))
            continue
        selected = getattr(meta, "selected_features", None)
        if selected:
            required.update(
                str(v)
                for v in selected
                if str(v)
                and str(v) not in DELETED_INFERENCE_FEATURE_KEYS
                and not re.fullmatch(r"f\d+", str(v))
            )

    ridge_sizer = (
        model_bundle.get("ridge_sizer") if isinstance(model_bundle, dict) else None
    )
    if allowed is None:
        for attr in ("model_names_", "model_names_ridge_", "limit_offset_features_"):
            vals = getattr(ridge_sizer, attr, None)
            if vals:
                required.update(
                    [
                        v
                        for v in vals
                        if v != "sizer_score_oof"
                        and str(v) not in DELETED_INFERENCE_FEATURE_KEYS
                    ]
                )

    booster_bundles = (
        model_bundle.get("booster_bundles", {})
        if isinstance(model_bundle, dict)
        else {}
    )
    if isinstance(booster_bundles, dict):
        for key, booster in booster_bundles.items():
            if not _model_key_matches_allowed(str(key), allowed):
                continue
            if isinstance(booster, dict):
                required.update(
                    k
                    for k in (booster.get("feature_keys", []) or [])
                    if str(k) not in DELETED_INFERENCE_FEATURE_KEYS
                )

    regime_adaptors = (
        model_bundle.get("regime_adaptors", {})
        if isinstance(model_bundle, dict)
        else {}
    )
    if isinstance(regime_adaptors, dict):
        for key, adaptor in regime_adaptors.items():
            if not _model_key_matches_allowed(str(key), allowed):
                continue
            if isinstance(adaptor, dict) and regime_adaptor_inference_enabled(
                artifact=adaptor
            ):
                mapping = adaptor.get("feature_mapping", {}) or {}
                for value in mapping.values():
                    if isinstance(value, str):
                        if value not in DELETED_INFERENCE_FEATURE_KEYS:
                            required.add(value)
                    elif isinstance(value, dict):
                        required.update(
                            str(v)
                            for v in value.values()
                            if isinstance(v, str)
                            and str(v) not in DELETED_INFERENCE_FEATURE_KEYS
                        )

    ridge_weights = bundle.get("ridge_weights", {}) if isinstance(bundle, dict) else {}
    params_per_bucket = (
        ridge_weights.get("params_per_bucket", {})
        if isinstance(ridge_weights, dict)
        else {}
    )
    for key, bucket_cfg in params_per_bucket.items():
        if not _model_key_matches_allowed(str(key), allowed):
            continue
        if isinstance(bucket_cfg, dict):
            required.update(
                k
                for k in (bucket_cfg.get("feature_names", []) or [])
                if str(k) not in DELETED_INFERENCE_FEATURE_KEYS
            )

    # Keep a small set of always-needed raw features used across inference glue.
    required.update(
        {
            "barrier_pct",
            "volatility_zscore",
            "range_12h_pct",
            "range_24h_pct",
            "ret12h",
            "ret24h",
            "ret1h",
            "z_r_12",
            "z_r_24",
        }
    )
    return {
        k
        for k in required
        if isinstance(k, str)
        and k
        and k not in LIVE_UNAVAILABLE_FEATURES
        and k not in DELETED_INFERENCE_FEATURE_KEYS
    }


def _compute_per_symbol_features(
    panel: Dict[str, pd.DataFrame],
    basket_syms: List[str],
) -> Dict[str, pd.DataFrame]:
    """Compute per-symbol features needed by candidate selector.

    This computes the minimum required features for candidate selection:
    - ret24h: 24-hour returns
    - range_12h_pct: 12-hour high-low range
    - volatility_zscore: Volatility z-score
    - chop_score: Choppiness index

    Args:
        panel: Price panel with open, high, low, close, volume DataFrames
        basket_syms: List of symbols to compute features for

    Returns:
        Dictionary of feature DataFrames (feature_name -> DataFrame with symbols as columns)
    """
    feats: Dict[str, pd.DataFrame] = {}

    # Get price data
    close = panel.get("close")
    high = panel.get("high")
    low = panel.get("low")
    volume = panel.get("volume")

    # Safely check for empty - handle case where close might be a string or other type
    try:
        is_empty = (
            close is None
            or not isinstance(close, (pd.DataFrame, pd.Series))
            or (hasattr(close, "empty") and close.empty)
        )
    except Exception as e:
        tprint(f"Error checking close.empty: {e}, type: {type(close)}")
        is_empty = True

    if is_empty:
        return feats

    # Filter to basket symbols
    valid_syms = [s for s in basket_syms if s in close.columns]
    if not valid_syms:
        return feats

    c = close[valid_syms]
    h = high[valid_syms] if high is not None else c
    l = low[valid_syms] if low is not None else c
    v = (
        volume[valid_syms]
        if volume is not None
        else pd.DataFrame(1.0, index=c.index, columns=c.columns)
    )

    # Compute ret24h (24-hour returns)
    ret24h = c / c.shift(24) - 1.0
    feats["ret24h"] = ret24h.astype(np.float32)

    # Compute ret12h (12-hour returns)
    ret12h = c / c.shift(12) - 1.0
    feats["ret12h"] = ret12h.astype(np.float32)

    # Compute ret6h (6-hour returns)
    ret6h = c / c.shift(6) - 1.0
    feats["ret6h"] = ret6h.astype(np.float32)

    # Compute ret1h (1-hour returns)
    ret1h = c / c.shift(1) - 1.0
    feats["ret1h"] = ret1h.astype(np.float32)

    # Compute range_12h_pct (12-hour high-low range)
    h_12 = h.rolling(12).max()
    l_12 = l.rolling(12).min()
    range_12h_pct = (h_12 - l_12) / (c + 1e-12)
    feats["range_12h_pct"] = range_12h_pct.astype(np.float32)

    # LGBM deployment masks are saved as raw-feature predicates. Keep this
    # selector-side subset aligned with the feature names used in those masks.
    h_24 = h.rolling(24, min_periods=1).max()
    l_24 = l.rolling(24, min_periods=1).min()
    range_24h_pct = (h_24 - l_24) / (c + 1e-12)
    feats["range_24h_pct"] = range_24h_pct.astype(np.float32)

    prev_low_24 = l_24.shift(1)
    dist_prior_day_low = (c - prev_low_24) / (c + 1e-12)
    feats["dist_prior_day_low"] = dist_prior_day_low.fillna(0.0).astype(np.float32)

    tr_components = [
        (h - l).abs(),
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ]
    tr = pd.concat(tr_components, axis=0).groupby(level=0).max()
    atr = tr.rolling(14, min_periods=1).mean().replace(0.0, np.nan)
    ema_fast = c.ewm(span=10, adjust=False, min_periods=1).mean()
    dist_ema_fast = (c - ema_fast) / (atr + 1e-12)
    feats["dist_ema_fast"] = (
        dist_ema_fast.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    )

    delta = c.diff()
    gain = delta.clip(lower=0.0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    feats["rsi_slope"] = rsi.diff(3).fillna(0.0).astype(np.float32)

    rolling_notional = (c * v).rolling(48, min_periods=1).sum()
    rolling_volume = v.rolling(48, min_periods=1).sum()
    vwap_48 = rolling_notional / (rolling_volume + 1e-12)
    session_stdev_48 = c.rolling(48, min_periods=2).std()
    loc_vwap_dev_z_48 = (c - vwap_48) / (
        np.maximum(session_stdev_48, atr * 0.5) + 1e-12
    )
    feats["loc_vwap_dev_z_48"] = (
        loc_vwap_dev_z_48.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )

    # Compute volatility (24-hour rolling std of returns)
    rv_24h = ret1h.rolling(24).std()

    # Compute volatility z-score (relative to 90-day rolling window)
    rv_24h_mean = rv_24h.rolling(24 * 90, min_periods=100).mean()
    rv_24h_std = rv_24h.rolling(24 * 90, min_periods=100).std()
    volatility_zscore = (rv_24h - rv_24h_mean) / (rv_24h_std + 1e-12)
    feats["volatility_zscore"] = volatility_zscore.astype(np.float32)

    # Compute choppiness index (simplified version)
    # Uses 24-hour rolling sum of absolute returns / rolling max-min
    sum_abs_ret = ret1h.abs().rolling(24).sum()
    high_low_range = h.rolling(24).max() - l.rolling(24).min()
    chop_score = sum_abs_ret / (np.log(high_low_range + 1e-12) + 1e-12)
    # Normalize to 0-1 range (approximate)
    chop_score = 1 - np.clip(chop_score / 50, 0, 1)
    feats["chop_score"] = chop_score.astype(np.float32)

    # Compute mkt_rv_24h (market realized volatility - average across symbols)
    mkt_rv_24h = rv_24h.mean(axis=1)
    feats["mkt_rv_24h"] = mkt_rv_24h.astype(np.float32)

    tprint(f"Computed {len(feats)} per-symbol features for {len(valid_syms)} symbols")
    return feats


def compute_selector_features(
    panel: Dict[str, pd.DataFrame],
    basket_syms: List[str],
) -> Dict[str, pd.DataFrame]:
    """Public helper for the lightweight candidate-selection feature set."""
    return _compute_per_symbol_features(panel, basket_syms)


def _broadcast_series_to_symbols(
    series: pd.Series,
    columns: List[str],
) -> pd.DataFrame:
    values = np.repeat(series.to_numpy(dtype=np.float32)[:, None], len(columns), axis=1)
    return pd.DataFrame(values, index=series.index, columns=columns)


def _zero_frame_like_panel(
    panel: Dict[str, pd.DataFrame],
    basket_syms: List[str],
) -> Optional[pd.DataFrame]:
    close = panel.get("close")
    if not isinstance(close, pd.DataFrame) or close.empty:
        return None
    valid_syms = [sym for sym in basket_syms if sym in close.columns]
    if not valid_syms:
        return None
    return pd.DataFrame(0.0, index=close.index, columns=valid_syms, dtype=np.float32)


def _rolling_zscore_frame(df: pd.DataFrame, window: int) -> pd.DataFrame:
    values = df.astype(np.float32)
    mean = values.rolling(window, min_periods=max(4, min(window, 24))).mean()
    std = values.rolling(window, min_periods=max(4, min(window, 24))).std(ddof=0)
    return (
        ((values - mean) / (std + 1e-6))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )


def _aligned_orderbook_panel_field(
    panel: Dict[str, pd.DataFrame],
    field_name: str,
    index: pd.Index,
    columns: List[str],
    shift_bars: int,
) -> Optional[pd.DataFrame]:
    frame = panel.get(f"orderbook_{field_name}")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    return (
        frame.reindex(index=index, columns=columns)
        .ffill()
        .shift(max(int(shift_bars), 0))
        .astype(np.float32)
    )


def _materialize_live_orderbook_summary_features(
    feats: Dict[str, pd.DataFrame],
    panel: Dict[str, pd.DataFrame],
    basket_syms: List[str],
    required_feature_keys: Set[str],
    cfg: Optional[Dict[str, Any]] = None,
    data_root: str | None = None,
    run_id: str | None = None,
) -> Dict[str, pd.DataFrame]:
    """Compute live orderbook features from saved hourly summary panels."""
    if not any(k.startswith(("ob_", "obw_")) for k in required_feature_keys):
        return feats
    if cfg is not None and not bool(
        cfg.get("live_materialize_orderbook_model_features", True)
    ):
        return feats

    zero_frame = _zero_frame_like_panel(panel, basket_syms)
    close = panel.get("close")
    volume = panel.get("volume")
    if (
        zero_frame is None
        or not isinstance(close, pd.DataFrame)
        or close.empty
        or not isinstance(volume, pd.DataFrame)
        or volume.empty
    ):
        return feats

    out = dict(feats)
    idx = zero_frame.index
    cols = list(zero_frame.columns)
    shift_bars = int((cfg or {}).get("microstructure_shift_bars", 1))
    eps = 1e-12
    preserve_cached = bool(
        (cfg or {}).get("historical_inference_parity_preserve_cached_features", False)
    )
    broadcast_feature_keys = {
        "xasset_mkt_spread_bps",
        "xasset_mkt_depth_z",
        "xasset_mkt_spread_bps_z_24h",
        "xasset_mkt_depth_to_qv_z",
        "xasset_mkt_ob_stress_z_24h",
        "xasset_ob_stress_basket_z_24h",
        "xasset_ob_stress_basket",
        "xasset_mkt_ob_stress",
        "median_spread_bps",
        "pct_assets_wide_spread",
    }

    def complete_broadcast_columns(value: pd.DataFrame) -> pd.DataFrame:
        frame = value.reindex(index=idx)
        missing_cols = [col for col in cols if col not in frame.columns]
        if missing_cols:
            row_value = frame.ffill(axis=1).bfill(axis=1).iloc[:, 0]
            for col in missing_cols:
                frame[col] = row_value
        return frame.reindex(index=idx, columns=cols).astype(np.float32)

    best_bid = _aligned_orderbook_panel_field(panel, "best_bid", idx, cols, shift_bars)
    best_ask = _aligned_orderbook_panel_field(panel, "best_ask", idx, cols, shift_bars)
    mid = _aligned_orderbook_panel_field(panel, "mid", idx, cols, shift_bars)
    bid_qty_1 = _aligned_orderbook_panel_field(
        panel, "bid_qty_1", idx, cols, shift_bars
    )
    ask_qty_1 = _aligned_orderbook_panel_field(
        panel, "ask_qty_1", idx, cols, shift_bars
    )
    bid_qty_l10 = _aligned_orderbook_panel_field(
        panel, "cum_bid_qty_l10", idx, cols, shift_bars
    )
    ask_qty_l10 = _aligned_orderbook_panel_field(
        panel, "cum_ask_qty_l10", idx, cols, shift_bars
    )
    bid_qty_l20 = _aligned_orderbook_panel_field(
        panel, "cum_bid_qty_l20", idx, cols, shift_bars
    )
    ask_qty_l20 = _aligned_orderbook_panel_field(
        panel, "cum_ask_qty_l20", idx, cols, shift_bars
    )

    if best_bid is None or best_ask is None:
        return feats
    if mid is None:
        mid = ((best_bid + best_ask) * 0.5).astype(np.float32)

    close_aligned = close.reindex(index=idx, columns=cols).astype(np.float32)
    volume_aligned = volume.reindex(index=idx, columns=cols).astype(np.float32)
    quote_volume_24h = (
        (close_aligned * volume_aligned).rolling(24, min_periods=6).sum().shift(1)
    )
    available = (best_bid.notna() & best_ask.notna() & mid.notna()).astype(np.float32)
    def put(name: str, value: pd.DataFrame) -> None:
        if name in required_feature_keys or name in out:
            existing = out.get(name)
            if (
                preserve_cached
                and isinstance(existing, pd.DataFrame)
                and not existing.empty
            ):
                if name in broadcast_feature_keys:
                    out[name] = complete_broadcast_columns(existing)
                return
            out[name] = (
                value.reindex(index=idx, columns=cols)
                .replace([np.inf, -np.inf], np.nan)
                .astype(np.float32)
            )

    def broadcast_to_symbols(value: pd.Series) -> pd.DataFrame:
        ser = pd.to_numeric(value, errors="coerce").reindex(idx)
        return pd.DataFrame(
            np.repeat(ser.to_numpy(dtype=np.float32)[:, None], len(cols), axis=1),
            index=idx,
            columns=cols,
            dtype=np.float32,
        )

    spread_bps = (((best_ask - best_bid) / (mid.abs() + eps)) * 1e4).clip(0, 1000)
    put("ob_available", available)
    put("ob_stale_flag", 1.0 - available)
    put("ob_update_gap_flag", 1.0 - available)
    put("ob_spread_bps", spread_bps)
    put("ob_spread_z_24h", _rolling_zscore_frame(spread_bps, 24))
    mid_close_gap = (((mid - close_aligned) / (close_aligned.abs() + eps)) * 1e4).clip(
        -1000, 1000
    )
    put("ob_mid_close_dislocation_bps", mid_close_gap)
    put("ob_mid_vs_close_bps", mid_close_gap)
    put("ob_mid_close_dislocation_bps_z_24h", _rolling_zscore_frame(mid_close_gap, 24))

    if bid_qty_1 is not None and ask_qty_1 is not None:
        l1_imb = ((bid_qty_1 - ask_qty_1) / (bid_qty_1 + ask_qty_1 + eps)).clip(-1, 1)
        microprice = (best_ask * bid_qty_1 + best_bid * ask_qty_1) / (
            bid_qty_1 + ask_qty_1 + eps
        )
        microprice_bps = (((microprice - mid) / (mid.abs() + eps)) * 1e4).clip(
            -1000, 1000
        )
        put("ob_l1_imbalance", l1_imb)
        put("ob_imb_l1", l1_imb)
        put("ob_microprice_premium_bps", microprice_bps)
        put("ob_microprice_dev_bps", microprice_bps)
        put("ob_microprice_dev_bps_z_24h", _rolling_zscore_frame(microprice_bps, 24))
        top_liq = (mid * (bid_qty_1 + ask_qty_1)).clip(lower=0.0)
        put("ob_top_liquidity_usd", np.log1p(top_liq))
        put(
            "ob_top_liquidity_to_qv_24h",
            (top_liq / (quote_volume_24h + eps)).clip(0, 100),
        )

    l10_imb = None
    if bid_qty_l10 is not None and ask_qty_l10 is not None:
        l10_imb = (
            (bid_qty_l10 - ask_qty_l10) / (bid_qty_l10 + ask_qty_l10 + eps)
        ).clip(-1, 1)
        depth_l10 = (mid * (bid_qty_l10 + ask_qty_l10)).clip(lower=0.0)
        put("ob_l10_imbalance", l10_imb)
        put("ob_imb_l10", l10_imb)
        put("ob_wimb_l10", l10_imb)
        put("ob_depth_usd_l10", np.log1p(depth_l10))
        put("ob_depth_usd_l10_z", _rolling_zscore_frame(np.log1p(depth_l10), 24 * 7))
        put(
            "ob_depth_l10_to_qv_24h",
            (depth_l10 / (quote_volume_24h + eps)).clip(0, 100),
        )

    l20_imb = None
    depth_l20 = None
    if bid_qty_l20 is not None and ask_qty_l20 is not None:
        l20_imb = (
            (bid_qty_l20 - ask_qty_l20) / (bid_qty_l20 + ask_qty_l20 + eps)
        ).clip(-1, 1)
        depth_l20 = (mid * (bid_qty_l20 + ask_qty_l20)).clip(lower=0.0)
        put("ob_l20_imbalance", l20_imb)
        put("ob_imb_l20", l20_imb)
        put("ob_wimb_l20", l20_imb)
        put("ob_wall_imb_l20", l20_imb)
        put("ob_depth_usd_l20", np.log1p(depth_l20))
        depth_l20_z = _rolling_zscore_frame(np.log1p(depth_l20), 24 * 7)
        put("ob_depth_usd_l20_z", depth_l20_z)
        put("ob_depth_usd_z_24h", depth_l20_z)
        depth_l20_to_qv = (depth_l20 / (quote_volume_24h + eps)).clip(0, 100)
        put("ob_depth_l20_to_qv_24h", depth_l20_to_qv)
        put("ob_depth_l20_to_qv_z_7d", _rolling_zscore_frame(depth_l20_to_qv, 24 * 7))
        if bid_qty_1 is not None:
            put("ob_bid_depth_decay_l20", (bid_qty_1 / (bid_qty_l20 + eps)).clip(0, 1))
        if ask_qty_1 is not None:
            put("ob_ask_depth_decay_l20", (ask_qty_1 / (ask_qty_l20 + eps)).clip(0, 1))
        if bid_qty_1 is not None and ask_qty_1 is not None:
            top_depth = mid * (bid_qty_1 + ask_qty_1)
            put("ob_depth_ratio_l1_l20", (top_depth / (depth_l20 + eps)).clip(0, 1))
        if l10_imb is not None:
            put("ob_flow_vs_book_l10", (-l10_imb).clip(-2, 2))
            put("ob_imb_near_far_delta", (l10_imb - l20_imb).clip(-2, 2))
        put("ob_flow_vs_book_l20", (-l20_imb).clip(-2, 2))
        put("ob_abs_flow_vs_book_l20", l20_imb.abs().clip(0, 2))
        put(
            "ob_book_pressure_l10",
            (l10_imb if l10_imb is not None else l20_imb) * spread_bps,
        )
        put("ob_book_absorption_score", (-l20_imb.abs()).clip(-2, 2))
        put(
            "ob_liquidity_shock_z",
            _rolling_zscore_frame(depth_l20.diff().fillna(0.0), 24 * 7),
        )

    trade_count = _aligned_orderbook_panel_field(
        panel, "trade_count_1h", idx, cols, shift_bars
    )
    buy_qty = _aligned_orderbook_panel_field(panel, "buy_qty_1h", idx, cols, shift_bars)
    sell_qty = _aligned_orderbook_panel_field(
        panel, "sell_qty_1h", idx, cols, shift_bars
    )
    notional = _aligned_orderbook_panel_field(
        panel, "notional_1h", idx, cols, shift_bars
    )
    buy_notional = _aligned_orderbook_panel_field(
        panel, "buy_notional_1h", idx, cols, shift_bars
    )
    sell_notional = _aligned_orderbook_panel_field(
        panel, "sell_notional_1h", idx, cols, shift_bars
    )
    vwap = _aligned_orderbook_panel_field(panel, "vwap_1h", idx, cols, shift_bars)
    mean_trade_qty = _aligned_orderbook_panel_field(
        panel, "mean_trade_qty_1h", idx, cols, shift_bars
    )
    signed_flow = _aligned_orderbook_panel_field(
        panel, "signed_flow_imbalance_1h", idx, cols, shift_bars
    )

    if trade_count is not None:
        put(
            "ob_trade_count_z_24h",
            _rolling_zscore_frame(np.log1p(trade_count.clip(lower=0.0)), 24 * 7),
        )
    if notional is not None:
        put(
            "ob_notional_z_24h",
            _rolling_zscore_frame(np.log1p(notional.clip(lower=0.0)), 24 * 7),
        )
        if depth_l20 is not None:
            notional_to_depth = (notional / (depth_l20 + eps)).clip(0, 100)
            put("ob_notional_to_depth_l20", notional_to_depth)
            put(
                "ob_notional_to_depth_l20_z_24h",
                _rolling_zscore_frame(notional_to_depth, 24),
            )
            if signed_flow is not None:
                put(
                    "ob_flow_toxicity_1h",
                    (signed_flow.abs() * notional_to_depth).clip(0, 100),
                )
    if buy_notional is not None:
        put(
            "ob_buy_notional_z_24h",
            _rolling_zscore_frame(np.log1p(buy_notional.clip(lower=0.0)), 24 * 7),
        )
    if sell_notional is not None:
        put(
            "ob_sell_notional_z_24h",
            _rolling_zscore_frame(np.log1p(sell_notional.clip(lower=0.0)), 24 * 7),
        )
    if buy_notional is not None and sell_notional is not None:
        flow_notional_imb = (
            (buy_notional - sell_notional) / (buy_notional + sell_notional + eps)
        ).clip(-1, 1)
        put("ob_flow_notional_imbalance_1h", flow_notional_imb)
        put(
            "ob_flow_notional_skew_z_24h",
            _rolling_zscore_frame(buy_notional - sell_notional, 24 * 7),
        )
    if buy_qty is not None and sell_qty is not None:
        flow_qty_imb = ((buy_qty - sell_qty) / (buy_qty + sell_qty + eps)).clip(-1, 1)
        put("ob_flow_qty_imbalance_1h", flow_qty_imb)
    if signed_flow is not None:
        put("ob_trade_flow_imbalance_1h", signed_flow.clip(-1, 1))
    if vwap is not None:
        vwap_gap = (((vwap - mid) / (mid.abs() + eps)) * 1e4).clip(-1000, 1000)
        put("ob_vwap_mid_gap_bps", vwap_gap)
        if notional is not None and depth_l20 is not None:
            kyle = (vwap_gap.abs() / ((notional / (depth_l20 + eps)).abs() + eps)).clip(
                0, 1000
            )
            put("ob_kyle_lambda_1h", kyle)
    if mean_trade_qty is not None:
        put(
            "ob_mean_trade_qty_z_24h",
            _rolling_zscore_frame(np.log1p(mean_trade_qty.clip(lower=0.0)), 24 * 7),
        )
        if bid_qty_1 is not None and ask_qty_1 is not None:
            trade_size_to_l1 = (
                (mean_trade_qty * close_aligned)
                / (mid * (bid_qty_1 + ask_qty_1) + eps)
            ).clip(0, 100)
            put("ob_trade_size_to_l1_depth", trade_size_to_l1)
            put(
                "ob_trade_size_to_l1_depth_z_24h",
                _rolling_zscore_frame(trade_size_to_l1, 24),
            )

    depth_norm_z = out.get("ob_depth_l20_to_qv_z_7d")
    spread_z = out.get("ob_spread_z_24h")
    depth_usd_z = out.get("ob_depth_usd_l20_z")
    available_spread = [s for s in basket_syms if s in spread_bps.columns]
    basket_spread_bps = (
        spread_bps[available_spread].mean(axis=1)
        if available_spread
        else spread_bps.mean(axis=1)
    )
    put("xasset_mkt_spread_bps", broadcast_to_symbols(basket_spread_bps))
    if isinstance(depth_usd_z, pd.DataFrame):
        available_depth_usd = [s for s in basket_syms if s in depth_usd_z.columns]
        basket_depth_usd_z = (
            depth_usd_z[available_depth_usd].mean(axis=1)
            if available_depth_usd
            else depth_usd_z.mean(axis=1)
        )
        put("xasset_mkt_depth_z", broadcast_to_symbols(basket_depth_usd_z))
    if isinstance(depth_norm_z, pd.DataFrame) and isinstance(spread_z, pd.DataFrame):
        available_basket = [s for s in basket_syms if s in depth_norm_z.columns]
        basket_depth_z = (
            depth_norm_z[available_basket].mean(axis=1)
            if available_basket
            else depth_norm_z.mean(axis=1)
        )
        basket_spread_z = (
            spread_z[available_basket].mean(axis=1)
            if available_basket
            else spread_z.mean(axis=1)
        )
        put("xasset_mkt_spread_bps_z_24h", broadcast_to_symbols(basket_spread_z))
        put("xasset_mkt_depth_to_qv_z", broadcast_to_symbols(basket_depth_z))
        stress = _rolling_zscore_frame(
            broadcast_to_symbols((basket_spread_z - basket_depth_z).clip(-10, 10)),
            24,
        )
        put("xasset_mkt_ob_stress_z_24h", stress)
        put("xasset_ob_stress_basket_z_24h", stress)
        put(
            "xasset_ob_liquidity_divergence_z_24h",
            _rolling_zscore_frame(depth_norm_z.sub(basket_depth_z, axis=0), 24),
        )

    if l20_imb is not None:
        depth_proxy = out.get("ob_depth_usd_l20_z", zero_frame)
        bid_wall = l20_imb.clip(lower=0.0).abs() * (depth_proxy.abs() + 1.0)
        ask_wall = (-l20_imb).clip(lower=0.0).abs() * (depth_proxy.abs() + 1.0)
        for band in ("r005", "r010", "r020", "r030", "a05", "a10", "a20", "a30"):
            put(f"obw_wall_skew_book_{band}", l20_imb)
            put(f"obw_wall_skew_vol_{band}", (l20_imb * depth_proxy).clip(-6, 6))
            put(
                f"obw_wall_pressure_skew_{band}",
                (l20_imb * (spread_bps + 1.0)).clip(-100, 100),
            )
            put(f"obw_band_depth_skew_vol_{band}", (l20_imb * depth_proxy).clip(-6, 6))
            put(f"obw_wall_concentration_skew_{band}", l20_imb.abs().clip(0, 1))
            put(f"obw_blocking_wall_to_vol_{band}", ask_wall)
            put(f"obw_support_wall_to_vol_{band}", bid_wall)
            put(
                f"obw_blocking_minus_support_wall_{band}",
                ((ask_wall - bid_wall) / (ask_wall + bid_wall + eps)).clip(-1, 1),
            )
            put(f"obw_blocking_wall_pressure_{band}", ask_wall * (spread_bps + 1.0))
            put(f"obw_blocking_wall_distance_{band}", (spread_bps / 100.0).clip(0, 1))
            put(f"obw_path_depth_to_target_{band}", ask_wall)
        put("obw_nearest_bid_wall_to_vol", bid_wall)
        put("obw_nearest_ask_wall_to_vol", ask_wall)
        put(
            "obw_nearest_wall_skew_vol",
            ((bid_wall - ask_wall) / (bid_wall + ask_wall + eps)).clip(-1, 1),
        )
        put("obw_nearest_wall_distance_skew", (spread_bps / 100.0).clip(0, 1))

    produced = sorted(
        k for k in required_feature_keys if k in out and k.startswith(("ob_", "obw_"))
    )
    if produced:
        tprint(
            "Materialized live orderbook summary features from hourly panels: "
            f"{len(produced)} required keys available"
        )
    return out


def _synthesize_live_safe_feature_keys(
    feats: Dict[str, pd.DataFrame],
    panel: Dict[str, pd.DataFrame],
    basket_syms: List[str],
    required_feature_keys: Optional[Set[str]],
    *,
    data_root: str | None = None,
    run_id: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Materialize deterministic derived live features.

    Missing source-dependent model inputs are hard errors.  This function may
    derive a feature from an available portable source, but it must not create
    neutral or zero fallback columns.
    """
    if not required_feature_keys:
        return feats

    required = set(required_feature_keys)
    non_portable_required = sorted(k for k in required if is_non_portable_feature_key(k))
    if non_portable_required:
        sample = ", ".join(non_portable_required[:20])
        extra = (
            ""
            if len(non_portable_required) <= 20
            else f" (+{len(non_portable_required) - 20} more)"
        )
        tprint(
            "Live inference artifact requests source-dependent feature keys; "
            "requiring live source panels plus strict finite model-matrix parity: "
            f"{sample}{extra}"
        )
    feats = _materialize_live_orderbook_summary_features(
        feats,
        panel,
        basket_syms,
        required,
        cfg=cfg,
        data_root=data_root,
        run_id=run_id,
    )
    missing: List[str] = []
    barrier_missing = "barrier_pct" in required and "barrier_pct" not in feats
    calendar_missing = sorted(
        key
        for key in required
        if key not in feats and re.fullmatch(r"timestamp\.dayofweek>=\d+", key)
    )
    rolling_missing = sorted(
        key
        for key in required
        if key not in feats and re.fullmatch(r"rolling30d\([^)]+\)", key)
    )
    # Several historical orderbook features were trained from richer L2/aggtrade
    # data. Live inference may only synthesize aliases from an equivalent source;
    # unavailable orderbook fields are rejected.
    orderbook_aliases = {
        "ob_l1_imbalance": "ob_imb_l1",
        "ob_l10_imbalance": "ob_imb_l10",
        "ob_l20_imbalance": "ob_imb_l20",
        "ob_microprice_premium_bps": "ob_microprice_dev_bps",
        "ob_mid_vs_close_bps": "ob_mid_close_dislocation_bps",
    }
    orderbook_prefix_zero_missing = sorted(
        key
        for key in required
        if key not in feats
        and key.startswith(("ob_", "obw_"))
        and key not in orderbook_aliases
    )
    orderbook_alias_missing = sorted(
        key for key in orderbook_aliases if key in required and key not in feats
    )
    orderbook_zero_missing: List[str] = []
    stale_sensitive_zero_missing = sorted(
        key
        for key in required
        if key not in feats
        and _is_live_stale_sensitive_feature_key(key)
        and not key.startswith(("ob_", "obw_"))
    )
    source_missing = (
        missing
        + orderbook_zero_missing
        + orderbook_prefix_zero_missing
        + stale_sensitive_zero_missing
    )
    allow_missing_live_sources = bool(
        (cfg or {}).get("historical_inference_parity_allow_missing_live_sources", False)
    )
    if source_missing and not allow_missing_live_sources:
        sample = ", ".join(sorted(source_missing)[:20])
        extra = "" if len(source_missing) <= 20 else f" (+{len(source_missing) - 20} more)"
        raise ValueError(
            "Live inference requires features that cannot be materialized from "
            f"portable live sources: {sample}{extra}"
        )
    if (
        not missing
        and not barrier_missing
        and not calendar_missing
        and not rolling_missing
        and not orderbook_alias_missing
        and not orderbook_zero_missing
        and not orderbook_prefix_zero_missing
        and not stale_sensitive_zero_missing
    ):
        return _ensure_required_symbol_columns(feats, panel, basket_syms, required)

    zero_frame = _zero_frame_like_panel(panel, basket_syms)
    if zero_frame is None:
        raise ValueError("Cannot materialize derived live features without a close panel")

    out = dict(feats)
    close = panel.get("close")
    if barrier_missing:
        policy_barrier = _compute_policy_barrier_pct(panel, basket_syms, {})
        if isinstance(policy_barrier, pd.DataFrame) and not policy_barrier.empty:
            out["barrier_pct"] = policy_barrier.reindex(
                index=zero_frame.index,
                columns=zero_frame.columns,
            ).astype(np.float32)
        else:
            raise ValueError("barrier_pct is required but could not be computed")
    for key in rolling_missing:
        match = re.fullmatch(r"rolling30d\(([^)]+)\)", key)
        source = match.group(1) if match else ""
        source_df = out.get(source)
        if isinstance(source_df, pd.DataFrame) and not source_df.empty:
            out[key] = (
                source_df.rolling(24 * 30, min_periods=24).mean().astype(np.float32)
            )
        else:
            raise ValueError(f"{key} is required but source feature {source!r} is unavailable")
    for key in calendar_missing:
        match = re.fullmatch(r"timestamp\.dayofweek>=(\d+)", key)
        if not (
            match
            and isinstance(close, pd.DataFrame)
            and isinstance(close.index, pd.DatetimeIndex)
        ):
            raise ValueError(f"{key} is required but timestamp calendar source is unavailable")
        threshold = int(match.group(1))
        idx = close.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        weekend = pd.Series(
            (idx.dayofweek >= threshold).astype(np.float32),
            index=close.index,
        )
        out[key] = _broadcast_series_to_symbols(
            weekend,
            list(zero_frame.columns),
        )
    for key in orderbook_alias_missing:
        source = orderbook_aliases[key]
        source_df = out.get(source)
        if isinstance(source_df, pd.DataFrame) and not source_df.empty:
            out[key] = (
                source_df.reindex(
                    index=zero_frame.index,
                    columns=zero_frame.columns,
                )
                .astype(np.float32)
            )
        else:
            raise ValueError(
                f"{key} is required but equivalent source feature {source!r} is unavailable"
            )
    out = _ensure_required_symbol_columns(out, panel, basket_syms, required)
    tprint(
        "Materialized derived live feature keys for inference: "
        f"{(['barrier_pct'] if barrier_missing else []) + calendar_missing + rolling_missing + orderbook_alias_missing}"
    )
    return out


def _ensure_required_symbol_columns(
    feats: Dict[str, pd.DataFrame],
    panel: Dict[str, pd.DataFrame],
    basket_syms: List[str],
    required_feature_keys: Set[str],
) -> Dict[str, pd.DataFrame]:
    """Validate that required feature frames cover the current live symbols."""
    zero_frame = _zero_frame_like_panel(panel, basket_syms)
    if zero_frame is None:
        raise ValueError("Cannot validate live feature symbols without a close panel")
    out = dict(feats)
    added: Dict[str, int] = {}
    for key in required_feature_keys:
        value = out.get(key)
        if not isinstance(value, pd.DataFrame) or value.empty:
            continue
        missing_cols = [sym for sym in zero_frame.columns if sym not in value.columns]
        if not missing_cols:
            continue
        added[key] = len(missing_cols)
    if added:
        sample = dict(list(sorted(added.items()))[:10])
        raise ValueError(
            "Required inference features are missing live symbol columns; "
            f"refusing neutral fallback columns: {sample}"
        )
    return out


def _synthesize_gated_feature_keys(
    feats: Dict[str, pd.DataFrame],
    panel: Dict[str, pd.DataFrame],
    basket_syms: List[str],
    required_feature_keys: Optional[Set[str]],
) -> Dict[str, pd.DataFrame]:
    """Materialize live-safe gate-conditioned feature columns required by models.

    Some alpha contracts include columns such as ``ret24h_G_VOL_1``. Training
    generates those as base-feature times a binary regime gate. The cached
    selected feature store may omit the explicit expanded columns, so inference
    recreates them from the base feature and a causal volatility gate.
    """
    if not required_feature_keys:
        return feats

    needed = {
        key
        for key in required_feature_keys
        if key == "G_VOL" or "_G_VOL_" in key or key == "G_TREND" or "_G_TREND_" in key
    }
    if not needed:
        return feats

    close = panel.get("close")
    if not isinstance(close, pd.DataFrame) or close.empty:
        raise ValueError("Gate-conditioned features are required but close panel is unavailable")

    valid_syms = [s for s in basket_syms if s in close.columns]
    if not valid_syms:
        raise ValueError("Gate-conditioned features are required but no live symbols are available")

    out = dict(feats)
    close = close[valid_syms]
    ret1h = close.pct_change()
    rv = ret1h.rolling(24, min_periods=12).std().mean(axis=1)
    rv_med = rv.rolling(24 * 7, min_periods=48).median()
    g_vol_series = (rv > rv_med).fillna(False).astype(np.float32)

    ret24 = close.pct_change(24).abs().mean(axis=1)
    trend_thr = np.maximum(rv.fillna(0.0) * np.sqrt(24.0) * 1.5, 0.005)
    g_trend_series = (ret24 > trend_thr).fillna(False).astype(np.float32)

    requires_g_vol = "G_VOL" in needed or any("_G_VOL_" in key for key in needed)
    requires_g_trend = "G_TREND" in needed or any("_G_TREND_" in key for key in needed)

    if requires_g_vol and "G_VOL" not in out:
        out["G_VOL"] = _broadcast_series_to_symbols(g_vol_series, valid_syms)
    if requires_g_trend and "G_TREND" not in out:
        out["G_TREND"] = _broadcast_series_to_symbols(g_trend_series, valid_syms)

    gates = {
        "G_VOL": out.get("G_VOL"),
        "G_TREND": out.get("G_TREND"),
    }
    for feat_name in sorted(needed):
        if feat_name in out or "_" not in feat_name:
            continue
        gate_name = "G_VOL" if "_G_VOL_" in feat_name else "G_TREND"
        marker = f"_{gate_name}_"
        if marker not in feat_name:
            continue
        base_name, state = feat_name.rsplit(marker, 1)
        if state not in {"0", "1"}:
            raise ValueError(f"Unsupported gate-conditioned feature state in {feat_name}")
        if base_name not in out:
            raise ValueError(
                f"{feat_name} is required but base feature {base_name!r} is unavailable"
            )
        gate_df = gates.get(gate_name)
        base_df = out.get(base_name)
        if not isinstance(gate_df, pd.DataFrame) or not isinstance(
            base_df, pd.DataFrame
        ):
            raise ValueError(f"{feat_name} cannot be materialized from portable gate inputs")
        gate_aligned = gate_df.reindex(index=base_df.index, columns=base_df.columns)
        if state == "1":
            out[feat_name] = (base_df.astype(np.float32) * gate_aligned).astype(
                np.float32
            )
        else:
            out[feat_name] = (base_df.astype(np.float32) * (1.0 - gate_aligned)).astype(
                np.float32
            )
    return out


def generate_features(
    panel: Dict[str, pd.DataFrame],
    basket_syms: Optional[List[str]] = None,
    trend_sma_hours: int = DEFAULT_TREND_SMA_HOURS,
    gate_vol_lookback_hours: int = DEFAULT_GATE_VOL_LOOKBACK_HOURS,
    gate_trend_thr: float = DEFAULT_GATE_TREND_THR,
) -> Dict[str, pd.DataFrame]:
    """Generate market features for inference.

    Computes the full set of market features needed for model inference:
    - Price-based features (returns, ranges, volatility)
    - Market-wide features (correlations, market returns)
    - Regime features (volatility regime, trend regime)

    Args:
        panel: Price panel with open, high, low, close, volume DataFrames
        basket_syms: List of symbols to include in basket features.
                     If None, uses all symbols in panel
        trend_sma_hours: Hours for trend SMA calculation
        gate_vol_lookback_hours: Hours for volatility regime lookback
        gate_trend_thr: Threshold for trend regime

    Returns:
        Dictionary of feature DataFrames (feature_name -> DataFrame with symbols as columns)
    """
    tprint("Generating market features for inference")

    # If no basket_syms provided, use all symbols from panel
    if basket_syms is None:
        close = panel.get("close")
        if close is not None:
            basket_syms = list(close.columns)

    if not basket_syms:
        tprint("Warning: No symbols provided for feature generation")
        return {}

    # Start with empty feature dictionary
    feats: Dict[str, pd.DataFrame] = {}

    # Compute per-symbol features (required by candidate selector)
    per_symbol_feats = _compute_per_symbol_features(panel, basket_syms)
    feats.update(per_symbol_feats)

    # Compute market-level features
    mkt_features = compute_market_features(
        panel=panel,
        basket_syms=basket_syms,
        trend_sma_hours=trend_sma_hours,
    )

    # Add market features with 'mkt_' prefix
    if isinstance(mkt_features, pd.DataFrame) and not mkt_features.empty:
        for col in mkt_features.columns:
            feats[f"mkt_{col}"] = mkt_features[col].astype(np.float32)

    # Add regime gates - pass computed market features, not raw close
    if isinstance(mkt_features, pd.DataFrame) and not mkt_features.empty:
        regime_features = add_regime_gates(
            mkt_df=mkt_features,
            gate_vol_lookback_hours=gate_vol_lookback_hours,
            gate_trend_thr=gate_trend_thr,
        )

        # Add regime features with 'reg_' prefix
        if isinstance(regime_features, pd.DataFrame) and not regime_features.empty:
            for col in regime_features.columns:
                if col not in feats:  # Don't overwrite existing features
                    feats[f"reg_{col}"] = regime_features[col].astype(np.float32)

    tprint(f"Generated {len(feats)} feature sets")

    # DEBUG: Log the feature structure
    tprint(f"DEBUG: feats keys: {list(feats.keys())}")
    for k, v in list(feats.items())[:3]:
        if isinstance(v, pd.DataFrame):
            tprint(f"DEBUG: feats[{k}] shape: {v.shape}, type: DataFrame")
        else:
            tprint(f"DEBUG: feats[{k}] type: {type(v)}")

    return feats


def generate_features_for_timestamp(
    panel: Dict[str, pd.DataFrame],
    ts: pd.Timestamp,
    basket_syms: Optional[List[str]] = None,
    lookback_hours: int = 48,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """Generate features up to a specific timestamp.

    Like generate_features but ensures all data is available up to
    the specified timestamp for inference.

    Args:
        panel: Full price panel
        ts: Target timestamp
        basket_syms: Symbols to include
        lookback_hours: Hours to include in lookback
        **kwargs: Additional args for generate_features

    Returns:
        Feature dictionary
    """
    # Filter panel to include only data up to ts
    filtered_panel = {}

    for key, df in panel.items():
        # Safely check for empty - handle case where df might be a string or other type
        try:
            is_empty = not isinstance(df, (pd.DataFrame, pd.Series)) or (
                hasattr(df, "empty") and df.empty
            )
        except Exception:
            is_empty = True

        if is_empty:
            filtered_panel[key] = df
            continue

        # Filter to timestamps <= ts
        mask = df.index <= ts
        filtered_df = df[mask]

        # Also take lookback_hours of data before ts
        if len(filtered_df) > lookback_hours:
            filtered_df = filtered_df.iloc[-lookback_hours:]

        filtered_panel[key] = filtered_df

    return generate_features(filtered_panel, basket_syms, **kwargs)


def get_feature_for_symbol(
    feats: Dict[str, pd.DataFrame],
    symbol: str,
    feature_name: str,
    ts: Optional[pd.Timestamp] = None,
) -> Optional[pd.Series]:
    """Get a specific feature for a symbol.

    Args:
        feats: Feature dictionary
        symbol: Symbol to get feature for
        feature_name: Name of feature
        ts: Specific timestamp (if None, gets latest)

    Returns:
        Series of feature values, or None if not found
    """
    if feature_name not in feats:
        return None

    feat_df = feats[feature_name]

    if symbol not in feat_df.columns:
        return None

    series = feat_df[symbol]

    if ts is not None and ts in series.index:
        return series.loc[ts]

    # Return latest value
    # Safely check for empty
    try:
        dropped = series.dropna()
        is_empty = not isinstance(dropped, (pd.DataFrame, pd.Series)) or (
            hasattr(dropped, "empty") and dropped.empty
        )
    except Exception:
        is_empty = True

    return dropped.iloc[-1] if not is_empty else None


def get_features_for_candidates(
    feats: Dict[str, pd.DataFrame],
    candidates: List[str],
    ts: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Get feature matrix for candidate symbols.

    Args:
        feats: Feature dictionary
        candidates: List of candidate symbols
        ts: Specific timestamp

    Returns:
        DataFrame with candidates as rows, features as columns
    """
    if not candidates:
        return pd.DataFrame()

    # Collect features for all candidates at timestamp
    feature_rows = []

    for sym in candidates:
        row = {}
        for feat_name, feat_df in feats.items():
            # Skip if feat_df is not a DataFrame
            if not isinstance(feat_df, pd.DataFrame):
                continue
            if sym in feat_df.columns:
                series = feat_df[sym]
                # Skip if series is not a proper Series
                if not isinstance(series, pd.Series):
                    continue
                if ts is not None:
                    ts_utc = pd.Timestamp(ts)
                    if ts_utc.tzinfo is None:
                        ts_utc = ts_utc.tz_localize("UTC")
                    else:
                        ts_utc = ts_utc.tz_convert("UTC")
                    series_index = pd.to_datetime(
                        series.index, utc=True, errors="coerce"
                    )
                    series_at_or_before = series.loc[series_index <= ts_utc]
                    if series_at_or_before.empty:
                        continue
                    if _is_live_stale_sensitive_feature_key(feat_name):
                        latest_ts = pd.to_datetime(
                            series_at_or_before.index[-1], utc=True, errors="coerce"
                        )
                        if pd.isna(latest_ts) or pd.Timestamp(latest_ts) < ts_utc:
                            continue
                    # Preserve timestamped NaNs. LightGBM consumes missing values
                    # natively, and dropping a NaN feature column turns a valid
                    # training-time missing value into a contract mismatch.
                    row[feat_name] = series_at_or_before.iloc[-1]
                elif isinstance(series, (pd.DataFrame, pd.Series)) and not series.empty:
                    finite_series = series.dropna()
                    if finite_series.empty:
                        continue
                    row[feat_name] = finite_series.iloc[-1]

        if row:
            row["symbol"] = sym
            feature_rows.append(row)

    if not feature_rows:
        return pd.DataFrame()

    return pd.DataFrame(feature_rows).set_index("symbol")
