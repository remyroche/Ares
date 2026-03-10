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
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]

    family = str(mask_cfg.get("family", "top_movers"))
    raw_param = mask_cfg.get("param", 5.0)
    z_hr = float(mask_cfg.get("z_hours", 12.0))
    duration_hr = float(mask_cfg.get("duration_hours", 1.0))
    z_bars = max(1, int(z_hr * 4))

    roll_h = h.rolling(z_bars, min_periods=1).max()
    roll_l = l.rolling(z_bars, min_periods=1).min()
    st_px = c.shift(z_bars).bfill()

    up_move = ((roll_h - st_px) / (st_px + 1e-9)).fillna(0.0)
    dn_move = ((st_px - roll_l) / (st_px + 1e-9)).fillna(0.0)
    ast_ret = feats.get("ret15m", c.pct_change())
    std_up = ast_ret.rolling(24 * 4, min_periods=1).std().fillna(0.0)
    std_dn = std_up

    mask_h_df = pd.DataFrame(False, index=c.index, columns=c.columns)
    mask_l_df = pd.DataFrame(False, index=c.index, columns=c.columns)

    if family == "top_movers":
        param = float(raw_param)
        for ts, row in up_move.iterrows():
            q = row.quantile(1.0 - param / 100.0)
            mask_h_df.loc[ts] = row >= q
        for ts, row in dn_move.iterrows():
            q = row.quantile(1.0 - param / 100.0)
            mask_l_df.loc[ts] = row >= q
    elif family == "std_threshold":
        param = float(raw_param)
        mask_h_df = up_move >= (param * std_up)
        mask_l_df = dn_move >= (param * std_dn)
    elif family == "abs_move_threshold":
        y_move = float(raw_param) / 100.0
        mask_h_df = up_move >= y_move
        mask_l_df = dn_move >= y_move
    elif family == "std_plus_abs":
        if isinstance(raw_param, str):
            import ast as python_ast

            std_v, abs_v = python_ast.literal_eval(raw_param)
        elif isinstance(raw_param, (list, tuple)):
            std_v, abs_v = raw_param
        else:
            std_v, abs_v = float(raw_param), 6.0
        y_move = float(abs_v) / 100.0
        mask_h_df = (up_move >= (float(std_v) * std_up)) & (up_move >= y_move)
        mask_l_df = (dn_move >= (float(std_v) * std_dn)) & (dn_move >= y_move)

    if duration_hr > 1.0:
        d_bars = max(1, int(duration_hr * 4))
        mask_h_df = mask_h_df.rolling(d_bars, min_periods=1).max().astype(bool)
        mask_l_df = mask_l_df.rolling(d_bars, min_periods=1).max().astype(bool)

    return (mask_h_df | mask_l_df).astype(bool)


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
