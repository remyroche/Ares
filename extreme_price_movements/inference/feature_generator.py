"""
Feature Generator for Inference.

This module generates features for inference:
- Uses compute_market_features from features.py
- Uses add_regime_gates for regime features
- Computes per-symbol features needed by candidate selector
"""

from typing import Dict, List, Any, Optional, Set

import pandas as pd
import numpy as np

from extreme_price_movements.features import (
    compute_market_features,
    add_regime_gates,
    compute_features_hourly,
)
from extreme_price_movements.data_store import load_features_selected
from extreme_price_movements.utils import tprint, log_error


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
DEFAULT_TAIL_WARMUP_BUFFER_HOURS = 72


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
    # incremental backfill we only need enough history to stabilize the
    # rolling/gated feature computations for newly missing timestamps.
    base_hours = max(int(trend_sma_hours), int(gate_vol_lookback_hours), 24 * 7)
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
    all_keys = set(cached_feats.keys()) | set(new_feats.keys())
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

    compute_panel: Dict[str, pd.DataFrame] = {}
    for key, df in panel.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            compute_panel[key] = df.copy()
    if not compute_panel:
        return merged_feats

    local_cfg = dict(cfg or {})
    if _requires_gated_feature_generation(missing_keys):
        local_cfg["enable_gated_features"] = True

    # Compute only the missing keys, then merge them into the existing feature map.
    mkt_df = compute_market_features(compute_panel, basket_syms, trend_sma_hours=DEFAULT_TREND_SMA_HOURS)
    mkt_gates = add_regime_gates(
        mkt_df,
        gate_vol_lookback_hours=DEFAULT_GATE_VOL_LOOKBACK_HOURS,
        gate_trend_thr=DEFAULT_GATE_TREND_THR,
    )
    missing_feats, missing_index, missing_columns = compute_features_hourly(
        compute_panel,
        mkt_gates,
        local_cfg,
        requested_feature_keys=sorted(missing_keys),
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
                else (basket_syms[: arr.shape[1]] if arr.shape[1] <= len(basket_syms) else None)
            ),
        )

    if not missing_frames:
        return merged_feats

    return _merge_feature_dicts(merged_feats, missing_frames)


def load_cached_features_for_inference(
    run_id: str,
    data_root: str,
    symbols: List[str],
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
) -> Dict[str, pd.DataFrame]:
    ts = pd.to_datetime(run_id, format="%Y%m%d_%H%M%S", utc=True)
    feats = load_features_selected(ts, data_root, symbols=symbols)
    if not isinstance(feats, dict):
        return {}
    return _slice_feature_window(feats, start_ts=start_ts, end_ts=end_ts)


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
    close = panel.get("close")
    if not isinstance(close, pd.DataFrame) or close.empty:
        return {}

    cfg = dict(cfg or {})
    if _requires_gated_feature_generation(required_feature_keys):
        cfg["enable_gated_features"] = True

    end_ts = close.index.max()
    start_ts = end_ts - pd.Timedelta(hours=lookback_hours)
    cached_feats = load_cached_features_for_inference(
        run_id=run_id,
        data_root=data_root,
        symbols=basket_syms,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    cached_last_ts = None
    if cached_feats:
        for df in cached_feats.values():
            if isinstance(df, pd.DataFrame) and not df.empty:
                cached_last_ts = df.index.max()
                break

    need_tail_backfill = cached_last_ts is None or end_ts > cached_last_ts
    # Always layer the lightweight candidate-selector features on top of the
    # stored offline feature cache, because the offline cache does not
    # necessarily contain keys like ret24h/range_12h_pct used by inference.
    selector_feats = _compute_per_symbol_features(panel, basket_syms)

    if not need_tail_backfill:
        tprint(
            f"Loaded stored inference features for {len(basket_syms)} symbols through {cached_last_ts}"
        )
        merged = _merge_feature_dicts(cached_feats, selector_feats)
        if required_feature_keys:
            missing = {k for k in required_feature_keys if k not in merged}
            merged = _backfill_missing_requested_keys(
                panel=panel,
                basket_syms=basket_syms,
                cfg=cfg,
                merged_feats=merged,
                missing_keys=missing,
            )
        return merged

    tail_warmup_hours = _required_tail_warmup_hours(
        lookback_hours=lookback_hours,
        trend_sma_hours=trend_sma_hours,
        gate_vol_lookback_hours=gate_vol_lookback_hours,
        tail_compute_hours=tail_compute_hours,
    )
    tail_start_ts = max(
        close.index.min(),
        (cached_last_ts - pd.Timedelta(hours=tail_warmup_hours)) if cached_last_ts is not None else start_ts,
    )
    panel_tail: Dict[str, pd.DataFrame] = {}
    for key, df in panel.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            panel_tail[key] = df[df.index >= tail_start_ts]

    tprint(
        "Stored features missing latest timestamps; computing tail-only feature backfill "
        f"from {tail_start_ts} to {end_ts}"
    )
    mkt_df = compute_market_features(panel_tail, basket_syms, trend_sma_hours=trend_sma_hours)
    mkt_gates = add_regime_gates(
        mkt_df,
        gate_vol_lookback_hours=gate_vol_lookback_hours,
        gate_trend_thr=gate_trend_thr,
    )
    full_tail_feats, _, _ = compute_features_hourly(
        panel_tail,
        mkt_gates,
        cfg,
        requested_feature_keys=sorted(required_feature_keys) if required_feature_keys else None,
    )
    if cached_last_ts is not None:
        full_tail_feats = {
            key: df[df.index > cached_last_ts]
            for key, df in full_tail_feats.items()
            if isinstance(df, pd.DataFrame) and not df.empty
        }

    merged_feats = _merge_feature_dicts(cached_feats, full_tail_feats)
    merged_feats = _merge_feature_dicts(merged_feats, selector_feats)
    merged_feats = _slice_feature_window(merged_feats, start_ts=start_ts, end_ts=end_ts)
    if required_feature_keys:
        missing = {k for k in required_feature_keys if k not in merged_feats}
        merged_feats = _backfill_missing_requested_keys(
            panel=panel,
            basket_syms=basket_syms,
            cfg=cfg,
            merged_feats=merged_feats,
            missing_keys=missing,
        )
    new_rows = 0
    if full_tail_feats:
        for df in full_tail_feats.values():
            if isinstance(df, pd.DataFrame):
                new_rows = len(df.index)
                break
    tprint(
        f"Loaded cached features and backfilled {new_rows} new timestamps"
    )
    return merged_feats


def get_inference_required_feature_keys(model_bundle: Dict[str, Any]) -> Set[str]:
    """Extract the union of raw feature keys needed by live inference models."""
    required: Set[str] = set()
    bundle = model_bundle.get("bundle", model_bundle) if isinstance(model_bundle, dict) else {}

    alpha_models = bundle.get("alpha_models", {}) if isinstance(bundle, dict) else {}
    for side_models in alpha_models.values():
        if not isinstance(side_models, dict):
            continue
        for model_info in side_models.values():
            if isinstance(model_info, dict):
                required.update(model_info.get("feat_cols", []) or [])

    meta_models = bundle.get("meta_models", {}) if isinstance(bundle, dict) else {}
    for meta in meta_models.values():
        selected = getattr(meta, "selected_features", None)
        if selected:
            required.update(selected)

    ridge_sizer = model_bundle.get("ridge_sizer") if isinstance(model_bundle, dict) else None
    for attr in ("model_names_", "model_names_ridge_", "model_names_tree_", "limit_offset_features_"):
        vals = getattr(ridge_sizer, attr, None)
        if vals:
            required.update([v for v in vals if v != "sizer_score_oof"])

    ridge_weights = bundle.get("ridge_weights", {}) if isinstance(bundle, dict) else {}
    params_per_bucket = ridge_weights.get("params_per_bucket", {}) if isinstance(ridge_weights, dict) else {}
    for bucket_cfg in params_per_bucket.values():
        if isinstance(bucket_cfg, dict):
            required.update(bucket_cfg.get("feature_names", []) or [])

    # Keep a small set of always-needed raw features used across inference glue.
    required.update(
        {
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
    return {k for k in required if isinstance(k, str) and k}


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
        is_empty = close is None or not isinstance(close, (pd.DataFrame, pd.Series)) or (hasattr(close, 'empty') and close.empty)
    except Exception as e:
        log_error(f"Error checking close.empty: {e}, type: {type(close)}", exc=e)
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
    v = volume[valid_syms] if volume is not None else pd.DataFrame(1.0, index=c.index, columns=c.columns)
    
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
            is_empty = not isinstance(df, (pd.DataFrame, pd.Series)) or (hasattr(df, 'empty') and df.empty)
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
        is_empty = not isinstance(dropped, (pd.DataFrame, pd.Series)) or (hasattr(dropped, 'empty') and dropped.empty)
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
                if ts is not None and ts in series.index:
                    row[feat_name] = series.loc[ts]
                elif isinstance(series, (pd.DataFrame, pd.Series)) and not series.empty:
                    row[feat_name] = series.dropna().iloc[-1]
        
        if row:
            row["symbol"] = sym
            feature_rows.append(row)
    
    if not feature_rows:
        return pd.DataFrame()
    
    return pd.DataFrame(feature_rows).set_index("symbol")
