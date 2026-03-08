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
from extreme_price_movements.utils import tprint


# Default thresholds (from offline optimization)
DEFAULT_EXTREME_PCT = 0.05
DEFAULT_MIN_MOVE_12H_PCT = 0.06
DEFAULT_MIN_RANGE_PCT = 0.06
DEFAULT_MIN_VOL_ZSCORE = 1.5


def select_candidates(
    panel: Dict[str, pd.DataFrame],
    feats: Dict[str, pd.DataFrame],
    extreme_pct: float = DEFAULT_EXTREME_PCT,
    min_move_12h_pct: float = DEFAULT_MIN_MOVE_12H_PCT,
    min_range_pct: float = DEFAULT_MIN_RANGE_PCT,
    min_vol_zscore: float = DEFAULT_MIN_VOL_ZSCORE,
    metric: str = "ret12h",
    chop_thr: float = 0.5,
) -> Tuple[List[str], List[str]]:
    """Select trade candidates using vectorized candidate selection.
    
    Applies the candidate selection algorithm with configured thresholds
    to identify potential long and short opportunities.
    
    Args:
        panel: Price panel with open, high, low, close, volume DataFrames
        feats: Feature dictionary with computed market features
        extreme_pct: Percentage of top/bottom performers to consider (default: 0.05)
        min_range_pct: Minimum 12h high/low range percentage (default: 0.06)
        min_vol_zscore: Minimum volatility z-score threshold (default: 1.5)
        metric: Performance metric to rank by (default: "ret24h")
        chop_thr: Maximum choppiness score threshold (default: 0.5)
        
    Returns:
        Tuple of (long_candidates, short_candidates) - lists of symbol strings
    """
    tprint(
        f"Selecting candidates with extreme_pct={extreme_pct}, "
        f"min_move_12h_pct={min_move_12h_pct}, min_vol_zscore={min_vol_zscore}"
    )
    
    # Apply vectorized candidate selection
    # Returns a boolean mask DataFrame
    try:
        candidate_mask = select_trade_candidates_vectorized(
            panel=panel,
            feats=feats,
            pct=extreme_pct,
            metric=metric,
            min_move_12h_pct=min_move_12h_pct,
            min_range_pct=min_range_pct,
            min_vol_zscore=min_vol_zscore,
            chop_thr=chop_thr,
        )
    except Exception as e:
        tprint(f"Error in select_trade_candidates_vectorized: {e}")
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
    extreme_pct: float = DEFAULT_EXTREME_PCT,
    min_move_12h_pct: float = DEFAULT_MIN_MOVE_12H_PCT,
    min_range_pct: float = DEFAULT_MIN_RANGE_PCT,
    min_vol_zscore: float = DEFAULT_MIN_VOL_ZSCORE,
    metric: str = "ret12h",
    chop_thr: float = 0.5,
) -> Tuple[List[str], List[str]]:
    """Select candidates at a specific timestamp.
    
    Similar to select_candidates but operates at a specific timestamp
    rather than the latest available timestamp.
    
    Args:
        panel: Price panel
        feats: Feature dictionary
        ts: Specific timestamp to evaluate
        extreme_pct: Percentage of top/bottom performers
        min_range_pct: Minimum range percentage
        min_vol_zscore: Minimum volatility z-score
        metric: Performance metric
        chop_thr: Choppiness threshold
        
    Returns:
        Tuple of (long_candidates, short_candidates)
    """
    # Get candidate mask for all timestamps
    candidate_mask = select_trade_candidates_vectorized(
        panel=panel,
        feats=feats,
        pct=extreme_pct,
        metric=metric,
        min_move_12h_pct=min_move_12h_pct,
        min_range_pct=min_range_pct,
        min_vol_zscore=min_vol_zscore,
        chop_thr=chop_thr,
    )
    
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
