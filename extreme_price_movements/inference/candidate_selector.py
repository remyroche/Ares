"""
Candidate Selector for Inference.

This module applies candidate thresholds to select trade candidates:
- Uses select_trade_candidates_vectorized from candidates.py
- Applies thresholds: extreme_pct=0.05, min_range_pct=0.06, min_vol_zscore=1.5
- Returns long_candidates and short_candidates
"""

from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np

from extreme_price_movements.inference.config import _resolve_runtime_cfg
from extreme_price_movements.utils import tprint


def _build_mask_for_mode(
    panel: Dict[str, pd.DataFrame],
    feats: Dict[str, pd.DataFrame],
    mask_cfg: Dict[str, Any],
) -> pd.DataFrame:
    from extreme_price_movements.lgbm_based_mask_generation import FeatureProcessor, CanonicalRuleMaskResolver
    from extreme_price_movements.mask_optimiser import _compute_z_cache, _generate_event_masks, _generate_event_masks_fast

    close_df = panel['close']
    n_ts, n_syms = close_df.shape

    idx_flat = np.repeat(close_df.index.to_numpy(), n_syms)
    sym_flat = np.tile(close_df.columns.to_numpy(), n_ts)

    feats_1d = {}
    for k, v in feats.items():
        if hasattr(v, 'to_numpy'):
            feats_1d[k] = v.to_numpy(dtype=np.float32).ravel()
        else:
            feats_1d[k] = np.asarray(v, dtype=np.float32).ravel()

    fp = FeatureProcessor()
    X, metadata, _ = fp.prepare_features(
        feats_1d, idx_flat, sym_flat, mask_cfg
    )
    resolver = CanonicalRuleMaskResolver(X, metadata)
    tprint(f"candidate_selector: CanonicalRuleMaskResolver initialized for mask_cfg name {mask_cfg.get('name')}")

    def _normalize_legacy_base_event(base_event: str) -> str:
        base_event = str(base_event or "").strip()
        if (
            base_event
            and "|" not in base_event
            and base_event.startswith("price_")
        ):
            return f"({base_event}==1)|(*)|(*)"
        return base_event

    base = _normalize_legacy_base_event(mask_cfg.get("base_event_trigger", ""))
    if base:
        try:
            mask_1d = resolver.get_mask(base)
            mask_2d = mask_1d.reshape((n_ts, n_syms))
            return pd.DataFrame(mask_2d, index=close_df.index, columns=close_df.columns, dtype=bool)
        except Exception:
            pass

    family = str(mask_cfg.get("family", "")).strip()
    if not family:
        return pd.DataFrame(False, index=close_df.index, columns=close_df.columns, dtype=bool)

    idx = close_df.index
    if len(idx) >= 2:
        delta_seconds = max((idx[1] - idx[0]).total_seconds(), 1.0)
        bph = max(int(round(3600.0 / delta_seconds)), 1)
    else:
        bph = 1

    close_arr = close_df.to_numpy(dtype=np.float32, copy=False).ravel()
    high_arr = panel["high"].reindex(index=idx, columns=close_df.columns).to_numpy(dtype=np.float32, copy=False).ravel()
    low_arr = panel["low"].reindex(index=idx, columns=close_df.columns).to_numpy(dtype=np.float32, copy=False).ravel()
    volume_df = panel.get("volume")
    volume_arr = None
    if isinstance(volume_df, pd.DataFrame):
        volume_arr = volume_df.reindex(index=idx, columns=close_df.columns).to_numpy(dtype=np.float32, copy=False).ravel()

    ret1_df = feats.get("ret1h")
    if isinstance(ret1_df, pd.DataFrame):
        ret_1 = ret1_df.reindex(index=idx, columns=close_df.columns).to_numpy(dtype=np.float32, copy=False).ravel()
    else:
        ret_1 = close_df.pct_change().fillna(0.0).to_numpy(dtype=np.float32, copy=False).ravel()

    vol_df = feats.get("atr_pct_base")
    if not isinstance(vol_df, pd.DataFrame):
        vol_df = feats.get("atr_pct")
    if isinstance(vol_df, pd.DataFrame):
        vol_g = vol_df.reindex(index=idx, columns=close_df.columns).to_numpy(dtype=np.float32, copy=False).ravel()
    else:
        close_safe = np.maximum(close_df.to_numpy(dtype=np.float32, copy=False), 1e-6)
        vol_g = ((panel["high"].reindex(index=idx, columns=close_df.columns).to_numpy(dtype=np.float32, copy=False) - panel["low"].reindex(index=idx, columns=close_df.columns).to_numpy(dtype=np.float32, copy=False)) / close_safe).astype(np.float32).ravel()

    asset_groups = {
        int(i): np.arange(i, n_ts * n_syms, n_syms, dtype=np.int32)
        for i in range(n_syms)
    }

    z_hours = float(mask_cfg.get("z_hours", 1.0) or 1.0)
    duration_hours = float(mask_cfg.get("duration_hours", 1.0) or 1.0)
    z_bars = max(int(round(z_hours * bph)), 1)
    duration_bars = max(int(round(duration_hours * bph)), 1)
    tprint("candidate_selector: calling _compute_z_cache...")
    if not hasattr(_build_mask_for_mode, "_zc_cache"):
        _build_mask_for_mode._zc_cache = {}
    _zc_cache = _build_mask_for_mode._zc_cache
    _zc_key = int(z_bars)
    if _zc_key in _zc_cache:
        zc = _zc_cache[_zc_key]
        tprint("candidate_selector: _compute_z_cache complete (cached).")
    else:
        zc = _compute_z_cache(
            high=high_arr,
            low=low_arr,
            close=close_arr,
            ret_1=ret_1,
            vol_g=vol_g,
            asset_groups=asset_groups,
            z=z_bars,
            bph=bph,
            volume=volume_arr,
            precomputed=feats_1d,
        )
        _zc_cache[_zc_key] = zc
        tprint("candidate_selector: _compute_z_cache complete.")

    name = str(mask_cfg.get("name", "") or "")
    feature_base = str(mask_cfg.get("feature_base", "") or "")
    param_token = None
    if name and "|p=" in name:
        param_token = name.split("|p=", 1)[1].split("|", 1)[0]

    candidate = None
    parsed_token = param_token or str(mask_cfg.get("param", "") or "")
    if parsed_token:
        if "_gt_" in parsed_token:
            parsed_feature_base, parsed_threshold = parsed_token.rsplit("_gt_", 1)
            candidate = {
                "family": family,
                "feature_base": feature_base or parsed_feature_base,
                "direction": "gt",
                "threshold": float(parsed_threshold),
            }
        elif "_lt_" in parsed_token:
            parsed_feature_base, parsed_threshold = parsed_token.rsplit("_lt_", 1)
            candidate = {
                "family": family,
                "feature_base": feature_base or parsed_feature_base,
                "direction": "lt",
                "threshold": float(parsed_threshold),
            }

    if candidate is None and family in {"std_threshold", "abs_move_threshold", "std_plus_abs"}:
        move_df = feats.get("ret12h")
        if not isinstance(move_df, pd.DataFrame):
            move_df = close_df.pct_change(12).fillna(0.0)
        move_df = move_df.reindex(index=idx, columns=close_df.columns).fillna(0.0)
        if family == "std_threshold":
            threshold_df = move_df.rolling(24 * 30, min_periods=2).std().fillna(0.0) * float(mask_cfg.get("param", 0.0) or 0.0)
            mask_h_df = move_df >= threshold_df
            mask_l_df = (-move_df) >= threshold_df
        elif family == "abs_move_threshold":
            threshold = float(mask_cfg.get("param", 0.0) or 0.0) / 100.0
            mask_h_df = move_df >= threshold
            mask_l_df = (-move_df) >= threshold
        else:
            param_val = mask_cfg.get("param", (0.0, 0.0))
            if isinstance(param_val, (list, tuple)) and len(param_val) >= 2:
                std_val = float(param_val[0])
                abs_val = float(param_val[1]) / 100.0
            else:
                std_val = 0.0
                abs_val = float(param_val or 0.0) / 100.0
            threshold_df = move_df.rolling(24 * 30, min_periods=2).std().fillna(0.0) * std_val
            mask_h_df = (move_df >= threshold_df) & (move_df >= abs_val)
            mask_l_df = ((-move_df) >= threshold_df) & ((-move_df) >= abs_val)

        mask_df = (mask_h_df | mask_l_df).fillna(False).astype(bool)
        if duration_bars > 1:
            for lag in range(1, duration_bars):
                mask_df = mask_df | mask_df.shift(lag, fill_value=False)
        return mask_df.astype(bool)

    try:
        if candidate is not None:
            tprint("candidate_selector: Calling _generate_event_masks_fast...")
            mask_h, mask_l = _generate_event_masks_fast(candidate=candidate, zc=zc)
            tprint("candidate_selector: _generate_event_masks_fast complete.")
        else:
            param_val = mask_cfg.get("param")
            if param_val is None:
                return pd.DataFrame(False, index=close_df.index, columns=close_df.columns, dtype=bool)
            mask_h, mask_l = _generate_event_masks(
                family=family,
                param_val=param_val,
                up_move=zc["up"],
                dn_move=zc["dn"],
                rolling_std_up=zc["std_up"],
                rolling_std_dn=zc["std_dn"],
                asset_groups=asset_groups,
                duration_bars=duration_bars,
            )
            tprint("candidate_selector: _generate_event_masks complete.")
        mask_2d = (mask_h | mask_l).reshape((n_ts, n_syms))
        return pd.DataFrame(mask_2d, index=close_df.index, columns=close_df.columns, dtype=bool)
    except Exception:
        return pd.DataFrame(False, index=close_df.index, columns=close_df.columns, dtype=bool)


def _up_down_zones(feats: Dict[str, pd.DataFrame], panel: Dict[str, pd.DataFrame], metric: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if metric in feats:
        metric_df = feats[metric]
    else:
        metric_df = panel["close"].pct_change(24).fillna(0.0)
    ranks = metric_df.rank(axis=1, method="first", na_option="keep", pct=True)
    up_zone = (ranks > 0.5).fillna(False).astype(bool)
    down_zone = (ranks <= 0.5).fillna(False).astype(bool)
    return up_zone, down_zone


def _require_mode_cfg(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    mode_cfg = dict(cfg.get("candidate_mask_params_by_mode", {}) or {})
    required = ["price_up_tf", "price_up_mr", "price_down_tf", "price_down_mr"]
    missing = [m for m in required if m not in mode_cfg]
    if missing:
        raise ValueError(
            "Per-mode mask params missing; refusing legacy fallback. "
            f"missing={missing} available={sorted(mode_cfg.keys())}"
        )
    return mode_cfg


def select_candidates(
    panel: Dict[str, pd.DataFrame],
    feats: Dict[str, pd.DataFrame],
    extreme_pct: Optional[float] = None,
    min_move_12h_pct: Optional[float] = None,
    min_range_pct: Optional[float] = None,
    min_vol_zscore: Optional[float] = None,
    metric: str = "ret12h",
    chop_thr: float = 0.5,
) -> Tuple[List[str], List[str]]:
    """Select trade candidates using mask optimiser logic.
    
    Applies the candidate selection algorithm with optimized parameters from
    mask_optimiser.py instead of the legacy threshold selection.
    
    Args:
        panel: Price panel with open, high, low, close, volume DataFrames
        feats: Feature dictionary with computed market features
        extreme_pct: Deprecated and unsupported (raises)
        min_move_12h_pct: Deprecated and unsupported (raises)
        min_range_pct: Deprecated and unsupported (raises)
        min_vol_zscore: Deprecated and unsupported (raises)
        metric: Performance metric to rank by
        chop_thr: Maximum choppiness score threshold
        
    Returns:
        Tuple of (long_candidates, short_candidates) - lists of symbol strings
    """
    cfg = _resolve_runtime_cfg()
    if any(v is not None for v in (extreme_pct, min_move_12h_pct, min_range_pct, min_vol_zscore)):
        raise ValueError(
            "Legacy threshold overrides are not supported after per-mode mask migration. "
            "Use persisted candidate_mask_params_by_mode instead."
        )
    mode_cfg = _require_mode_cfg(cfg)
    default_cfg = {
        "family": cfg.get("family", "top_movers"),
        "param": cfg.get("param", 5.0),
        "z_hours": cfg.get("z_hours", 12.0),
        "duration_hours": cfg.get("duration_hours", 1.0),
    }
    
    try:
        up_zone, down_zone = _up_down_zones(feats, panel, metric=metric)
        m_up_tf = _build_mask_for_mode(panel, feats, mode_cfg.get("price_up_tf", default_cfg))
        m_up_mr = _build_mask_for_mode(panel, feats, mode_cfg.get("price_up_mr", default_cfg))
        m_down_tf = _build_mask_for_mode(panel, feats, mode_cfg.get("price_down_tf", default_cfg))
        m_down_mr = _build_mask_for_mode(panel, feats, mode_cfg.get("price_down_mr", default_cfg))

        long_mask = (up_zone & m_up_tf) | (down_zone & m_down_mr)
        short_mask = (up_zone & m_up_mr) | (down_zone & m_down_tf)

    except Exception as e:
        raise RuntimeError(f"Per-mode candidate mask generation failed: {e}") from e
    
    if long_mask.empty and short_mask.empty:
        tprint("No candidates found - candidate masks are empty")
        return [], []
    
    latest_ts = long_mask.index[-1]
    latest_long = long_mask.loc[latest_ts]
    latest_short = short_mask.loc[latest_ts]
    long_candidates = latest_long[latest_long].index.tolist()
    short_candidates = latest_short[latest_short].index.tolist()
    
    tprint(f"Selected {len(long_candidates)} long candidates, "
           f"{len(short_candidates)} short candidates")
    
    return long_candidates, short_candidates


def select_candidates_at_timestamp(
    panel: Dict[str, pd.DataFrame],
    feats: Dict[str, pd.DataFrame],
    ts: pd.Timestamp,
    extreme_pct: Optional[float] = None,
    min_move_12h_pct: Optional[float] = None,
    min_range_pct: Optional[float] = None,
    min_vol_zscore: Optional[float] = None,
    metric: str = "ret12h",
    chop_thr: float = 0.5,
) -> Tuple[List[str], List[str]]:
    """Select candidates at a specific timestamp using mask_optimiser logic."""
    cfg = _resolve_runtime_cfg()
    if any(v is not None for v in (extreme_pct, min_move_12h_pct, min_range_pct, min_vol_zscore)):
        raise ValueError(
            "Legacy threshold overrides are not supported after per-mode mask migration. "
            "Use persisted candidate_mask_params_by_mode instead."
        )
    mode_cfg = _require_mode_cfg(cfg)
    default_cfg = {
        "family": cfg.get("family", "top_movers"),
        "param": cfg.get("param", 5.0),
        "z_hours": cfg.get("z_hours", 12.0),
        "duration_hours": cfg.get("duration_hours", 1.0),
    }

    try:
        up_zone, down_zone = _up_down_zones(feats, panel, metric=metric)
        m_up_tf = _build_mask_for_mode(panel, feats, mode_cfg.get("price_up_tf", default_cfg))
        m_up_mr = _build_mask_for_mode(panel, feats, mode_cfg.get("price_up_mr", default_cfg))
        m_down_tf = _build_mask_for_mode(panel, feats, mode_cfg.get("price_down_tf", default_cfg))
        m_down_mr = _build_mask_for_mode(panel, feats, mode_cfg.get("price_down_mr", default_cfg))

        long_mask = (up_zone & m_up_tf) | (down_zone & m_down_mr)
        short_mask = (up_zone & m_up_mr) | (down_zone & m_down_tf)

    except Exception as e:
        raise RuntimeError(f"Per-mode candidate mask generation at timestamp failed: {e}") from e
    
    if long_mask.empty and short_mask.empty:
        return [], []
    
    # Check if requested timestamp exists
    if ts not in long_mask.index:
        # Find nearest timestamp
        tprint(f"Timestamp {ts} not in mask, using nearest")
        ts = long_mask.index[np.abs(long_mask.index - ts).argmin()]
    
    long_candidates = long_mask.loc[ts]
    short_candidates = short_mask.loc[ts]
    long_candidates = long_candidates[long_candidates].index.tolist()
    short_candidates = short_candidates[short_candidates].index.tolist()
    
    return long_candidates, short_candidates


def filter_candidates_by_direction(
    candidates: List[str],
    panel: Dict[str, pd.DataFrame],
    side: str,
    lookback_hours: int = 24,
) -> List[str]:
    """Filter candidates based on price direction.
    
    Args:
        candidates: List of candidate symbols
        panel: Price panel
        side: "long" or "short"
        lookback_hours: Hours to look back for direction
        
    Returns:
        Filtered list of candidates
    """
    if not candidates:
        return []
    
    close = panel.get("close")
    # Safely check for empty - handle case where close might be a string or other type
    try:
        is_empty = close is None or not isinstance(close, (pd.DataFrame, pd.Series)) or (hasattr(close, 'empty') and close.empty)
    except Exception as e:
        tprint(f"Error checking close.empty: {e}, type: {type(close)}")
        is_empty = True
    
    if is_empty:
        return candidates
    
    filtered = []
    for sym in candidates:
        if sym not in close.columns:
            continue
        
        try:
            # Get recent prices
            recent_prices = close[sym].dropna()
            if len(recent_prices) < 2:
                continue
            
            # Calculate return over lookback period
            current_price = recent_prices.iloc[-1]
            past_price = recent_prices.iloc[-min(lookback_hours, len(recent_prices))]
            
            if past_price > 0:
                ret = (current_price / past_price) - 1
                
                if side == "long" and ret > 0:
                    filtered.append(sym)
                elif side == "short" and ret < 0:
                    filtered.append(sym)
        except (KeyError, IndexError, ZeroDivisionError):
            continue
    
    return filtered
