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

from extreme_price_movements.candidates import select_trade_candidates_vectorized
from extreme_price_movements.mask_optimiser import _generate_event_masks
from extreme_price_movements.inference.config import _resolve_runtime_cfg
from extreme_price_movements.utils import tprint


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
        extreme_pct: Kept for compatibility
        min_move_12h_pct: Kept for compatibility
        min_range_pct: Kept for compatibility
        min_vol_zscore: Kept for compatibility
        metric: Performance metric to rank by
        chop_thr: Maximum choppiness score threshold
        
    Returns:
        Tuple of (long_candidates, short_candidates) - lists of symbol strings
    """
    cfg = _resolve_runtime_cfg()
    family = cfg.get("family", "top_movers")
    param_val = float(cfg.get("param", 5.0))
    z_hr = float(cfg.get("z_hours", 12.0))

    duration_hr = float(cfg.get("duration_hours", 1.0))

    tprint(
        f"Selecting candidates using mask_optimiser logic: family={family}, "
        f"param={param_val}, z_hours={z_hr}, duration_hours={duration_hr}"
    )
    
    try:
        # Replicate _generate_event_masks inputs
        c = panel["close"]
        h = panel["high"]
        l = panel["low"]

        # Calculate moving metrics manually (like inside rolling_max_index_nb)
        z_bars = int(z_hr * 4) # approx 15m bars

        roll_h = h.rolling(z_bars, min_periods=1).max()
        roll_l = l.rolling(z_bars, min_periods=1).min()

        st_px = c.shift(z_bars).bfill()

        up_move = ((roll_h - st_px) / (st_px + 1e-9)).fillna(0.0)
        dn_move = ((st_px - roll_l) / (st_px + 1e-9)).fillna(0.0)

        # For std_threshold
        if "ret15m" not in feats:
            ast_ret = c.pct_change()
        else:
            ast_ret = feats["ret15m"]

        std_up = ast_ret.rolling(24 * 4, min_periods=1).std().fillna(0.0)
        std_dn = std_up # Simplified

        # We need an array-like timestamp for _generate_event_masks if we used it directly,
        # but since we have a dataframe, we can apply the logic manually for the top_movers here
        # or use it iteratively. Let's just implement the logic for DataFrame:

        mask_h_df = pd.DataFrame(False, index=c.index, columns=c.columns)
        mask_l_df = pd.DataFrame(False, index=c.index, columns=c.columns)

        if family == "top_movers":
            for ts, row in up_move.iterrows():
                q = row.quantile(1.0 - param_val/100.0)
                mask_h_df.loc[ts] = row >= q

            for ts, row in dn_move.iterrows():
                q = row.quantile(1.0 - param_val/100.0)
                mask_l_df.loc[ts] = row >= q

        elif family == "std_threshold":
            mask_h_df = up_move >= (param_val * std_up)
            mask_l_df = dn_move >= (param_val * std_dn)

        elif family == "abs_move_threshold":
            y_move = param_val / 100.0
            mask_h_df = up_move >= y_move
            mask_l_df = dn_move >= y_move

        elif family == "std_plus_abs":
            # param_val can be a string "(std, abs)" from CSV or a tuple
            if isinstance(param_val, str):
                import ast as python_ast
                std_v, abs_v = python_ast.literal_eval(param_val)
            elif isinstance(param_val, (list, tuple)):
                std_v, abs_v = param_val
            else:
                # Fallback if param_val is just a float (shouldn't happen with std_plus_abs)
                std_v, abs_v = param_val, 6.0
                
            y_move = abs_v / 100.0
            mask_h_df = (up_move >= (std_v * std_up)) & (up_move >= y_move)
            mask_l_df = (dn_move >= (std_v * std_dn)) & (dn_move >= y_move)

        # Apply duration dilation
        if duration_hr > 1.0:
            d_bars = int(duration_hr * 4)
            mask_h_df = mask_h_df.rolling(d_bars, min_periods=1).max().astype(bool)
            mask_l_df = mask_l_df.rolling(d_bars, min_periods=1).max().astype(bool)

        # Combine masks
        candidate_mask = mask_h_df | mask_l_df

    except Exception as e:
        tprint(f"Error in mask_optimiser candidate generation: {e}")
        import traceback
        tprint(f"Traceback: {traceback.format_exc()}")
        return [], []
    
    # Safely check for empty - handle case where candidate_mask might be a string or other type
    try:
        is_empty = candidate_mask is None or (hasattr(candidate_mask, 'empty') and candidate_mask.empty)
    except Exception as e:
        tprint(f"Error checking candidate_mask.empty: {e}, type: {type(candidate_mask)}")
        is_empty = True
    
    if is_empty:
        tprint("No candidates found - candidate mask is empty")
        return [], []
    
    # Get the latest timestamp from the mask
    latest_ts = candidate_mask.index[-1]
    
    # Extract symbols that are candidates at the latest timestamp
    latest_mask = candidate_mask.loc[latest_ts]
    
    # Long candidates are those marked as True (top performers for long)
    # Short candidates would need separate logic - for now we treat all True as candidates
    # and distinguish by side based on return direction
    
    # Get all candidates at latest timestamp
    candidate_symbols = latest_mask[latest_mask].index.tolist()
    
    # For each candidate, determine if it's long or short based on the metric
    # If ret24h > 0, it's a long candidate; if < 0, it's a short candidate
    long_candidates = []
    short_candidates = []
    
    if metric in feats:
        metric_series = feats[metric].loc[latest_ts]
        for sym in candidate_symbols:
            try:
                val = metric_series[sym]
                if pd.notna(val):
                    if val > 0:
                        long_candidates.append(sym)
                    else:
                        short_candidates.append(sym)
            except (KeyError, TypeError):
                # If we can't determine direction, include as both or skip
                continue
    
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
    family = cfg.get("family", "top_movers")
    param_val = float(cfg.get("param", 5.0))
    z_hr = float(cfg.get("z_hours", 12.0))

    try:
        c = panel["close"]
        h = panel["high"]
        l = panel["low"]
        
        z_bars = int(z_hr * 4)
        roll_h = h.rolling(z_bars, min_periods=1).max()
        roll_l = l.rolling(z_bars, min_periods=1).min()
        st_px = c.shift(z_bars).bfill()

        up_move = ((roll_h - st_px) / (st_px + 1e-9)).fillna(0.0)
        dn_move = ((st_px - roll_l) / (st_px + 1e-9)).fillna(0.0)

        if "ret15m" not in feats:
            ast_ret = c.pct_change()
        else:
            ast_ret = feats["ret15m"]

        std_up = ast_ret.rolling(24 * 4, min_periods=1).std().fillna(0.0)
        std_dn = std_up

        mask_h_df = pd.DataFrame(False, index=c.index, columns=c.columns)
        mask_l_df = pd.DataFrame(False, index=c.index, columns=c.columns)

        if family == "top_movers":
            for _ts, row in up_move.iterrows():
                q = row.quantile(1.0 - param_val/100.0)
                mask_h_df.loc[_ts] = row >= q
            for _ts, row in dn_move.iterrows():
                q = row.quantile(1.0 - param_val/100.0)
                mask_l_df.loc[_ts] = row >= q
        elif family == "std_threshold":
            mask_h_df = up_move >= (param_val * std_up)
            mask_l_df = dn_move >= (param_val * std_dn)
        elif family == "abs_move_threshold":
            y_move = param_val / 100.0
            mask_h_df = up_move >= y_move
            mask_l_df = dn_move >= y_move

        elif family == "std_plus_abs":
            if isinstance(param_val, str):
                import ast as python_ast
                std_v, abs_v = python_ast.literal_eval(param_val)
            elif isinstance(param_val, (list, tuple)):
                std_v, abs_v = param_val
            else:
                std_v, abs_v = param_val, 6.0
            y_move = abs_v / 100.0
            mask_h_df = (up_move >= (std_v * std_up)) & (up_move >= y_move)
            mask_l_df = (dn_move >= (std_v * std_dn)) & (dn_move >= y_move)

        if duration_hr > 1.0:
            d_bars = int(duration_hr * 4)
            mask_h_df = mask_h_df.rolling(d_bars, min_periods=1).max().astype(bool)
            mask_l_df = mask_l_df.rolling(d_bars, min_periods=1).max().astype(bool)

        candidate_mask = mask_h_df | mask_l_df

    except Exception as e:
        tprint(f"Error in select_candidates_at_timestamp (mask_optimiser logic): {e}")
        return [], []
    
    # Safely check for empty - handle case where candidate_mask might be a string or other type
    try:
        is_empty = candidate_mask is None or not isinstance(candidate_mask, (pd.DataFrame, pd.Series)) or (hasattr(candidate_mask, 'empty') and candidate_mask.empty)
    except Exception as e:
        tprint(f"Error checking candidate_mask.empty: {e}, type: {type(candidate_mask)}")
        is_empty = True
    
    if is_empty:
        return [], []
    
    # Check if requested timestamp exists
    if ts not in candidate_mask.index:
        # Find nearest timestamp
        tprint(f"Timestamp {ts} not in mask, using nearest")
        ts = candidate_mask.index[np.abs(candidate_mask.index - ts).argmin()]
    
    # Get candidates at this timestamp
    ts_mask = candidate_mask.loc[ts]
    candidate_symbols = ts_mask[ts_mask].index.tolist()
    
    # Determine long/short based on metric
    long_candidates = []
    short_candidates = []
    
    if metric in feats:
        metric_series = feats[metric].loc[ts]
        for sym in candidate_symbols:
            try:
                val = metric_series[sym]
                if pd.notna(val):
                    if val > 0:
                        long_candidates.append(sym)
                    else:
                        short_candidates.append(sym)
            except (KeyError, TypeError):
                continue
    
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
