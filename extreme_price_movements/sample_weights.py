"""
Sample Uniqueness Weighting (AFML Chapter 4)

Addresses label overlap and information leakage by computing sample uniqueness
based on label time ranges and overlaps.
"""
import numpy as np
import pandas as pd
from typing import Union


def compute_avg_uniqueness(
    label_times: pd.DataFrame,
    num_concurrent_labels: Union[pd.Series, None] = None
) -> pd.Series:
    """
    Compute average uniqueness of samples based on label overlap.
    
    Args:
        label_times: DataFrame with columns ['t_start', 't_end'] indexed by sample
        num_concurrent_labels: Optional pre-computed concurrent label counts
        
    Returns:
        Series of uniqueness weights (0-1) indexed by sample
        
    AFML Chapter 4: u[i] = 1 / (# concurrent labels at each timestamp in label[i])
    averaged over the label's time span.
    """
    if label_times.empty:
        return pd.Series(dtype=float)
    
    # Ensure datetime index
    if not isinstance(label_times.index, pd.DatetimeIndex):
        label_times.index = pd.to_datetime(label_times.index)
    
    # Build event index (all unique timestamps where labels start or end)
    t_starts = label_times['t_start'].dropna()
    t_ends = label_times['t_end'].dropna()
    
    all_times = pd.DatetimeIndex(
        sorted(set(t_starts.values).union(set(t_ends.values)))
    )
    
    # Count concurrent labels at each timestamp
    if num_concurrent_labels is None:
        num_concurrent_labels = pd.Series(0, index=all_times, dtype=int)
        
        for ts in all_times:
            # Count how many labels are active at timestamp ts
            concurrent = (
                (label_times['t_start'] <= ts) & 
                (label_times['t_end'] >= ts)
            ).sum()
            num_concurrent_labels.loc[ts] = concurrent
    
    # Compute uniqueness for each label
    uniqueness = pd.Series(0.0, index=label_times.index)
    
    for idx in label_times.index:
        t_start = label_times.loc[idx, 't_start']
        t_end = label_times.loc[idx, 't_end']
        
        if pd.isna(t_start) or pd.isna(t_end):
            uniqueness.loc[idx] = 0.0
            continue
        
        # Get timestamps within this label's range
        mask = (all_times >= t_start) & (all_times <= t_end)
        active_times = all_times[mask]
        
        if len(active_times) == 0:
            uniqueness.loc[idx] = 1.0
            continue
        
        # Average uniqueness = 1 / avg(concurrent labels)
        concurrent_at_times = num_concurrent_labels.loc[active_times]
        avg_concurrent = concurrent_at_times.mean()
        
        uniqueness.loc[idx] = 1.0 / max(avg_concurrent, 1.0)
    
    return uniqueness


def compute_sample_weights_with_uniqueness(
    label_times: pd.DataFrame,
    returns: Union[pd.Series, np.ndarray],
    base_weights: Union[pd.Series, np.ndarray, None] = None
) -> np.ndarray:
    """
    Compute sample weights combining uniqueness and return magnitude.
    
    Args:
        label_times: DataFrame with ['t_start', 't_end'] columns
        returns: Realized returns for weighting by magnitude
        base_weights: Optional base weights (e.g., from exhaustion probability)
        
    Returns:
        Combined sample weights as numpy array
    """
    # Compute uniqueness
    uniqueness = compute_avg_uniqueness(label_times)
    
    # Align indices
    if isinstance(returns, pd.Series):
        returns = returns.reindex(uniqueness.index).fillna(0.0)
    else:
        returns = pd.Series(returns, index=uniqueness.index)
    
    # Return magnitude weighting (log scale)
    magnitude_weight = np.log1p(np.abs(returns))
    
    # Combine: w = uniqueness * magnitude * base_weight
    combined = uniqueness * magnitude_weight
    
    if base_weights is not None:
        if isinstance(base_weights, pd.Series):
            base_weights = base_weights.reindex(uniqueness.index).fillna(1.0)
        else:
            base_weights = pd.Series(base_weights, index=uniqueness.index)
        combined = combined * base_weights
    
    # Normalize to sum to N (standard practice)
    weights = combined.values
    weights = weights * len(weights) / (weights.sum() + 1e-12)
    
    # Clip extremes
    weights = np.clip(weights, 0.1, 10.0)
    
    return weights


def build_label_time_ranges(
    entry_times: pd.DatetimeIndex,
    exit_times: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Build label time ranges DataFrame from entry/exit timestamps.
    
    Args:
        entry_times: Timestamps when labels start
        exit_times: Timestamps when labels end
        
    Returns:
        DataFrame with ['t_start', 't_end'] indexed by sample
    """
    if len(entry_times) != len(exit_times):
        raise ValueError("entry_times and exit_times must have same length")
    
    return pd.DataFrame({
        't_start': entry_times,
        't_end': exit_times
    }, index=range(len(entry_times)))
