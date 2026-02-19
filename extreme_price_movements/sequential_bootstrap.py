"""
Sequential Bootstrap (AFML Chapter 4.5)

Generates bootstrapped samples respecting sample uniqueness and temporal structure.
Optimized to avoid dense matrix construction (O(N*T)) and O(N^2) updates.
"""
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from numba import jit

def get_label_intervals(label_times: pd.DataFrame, price_times: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map label timestamps to integer indices in the price_times array.
    
    Args:
        label_times: DataFrame with ['t_start', 't_end'] indexed by sample
        price_times: DatetimeIndex of all available timestamps
        
    Returns:
        starts: Start index for each label (int array)
        ends: End index for each label (int array)
        valid: Boolean mask of valid labels (within price_times range)
    """
    pt = price_times.values
    starts_ts = label_times["t_start"].values.astype("datetime64[ns]")
    ends_ts = label_times["t_end"].values.astype("datetime64[ns]")

    # Find insertion points
    # side='left': a[i-1] < v <= a[i]
    i0 = np.searchsorted(pt, starts_ts, side="left")
    
    # side='right': a[i-1] <= v < a[i]
    # We want inclusive end. If t_end matches exactly, searchsorted right gives index+1.
    # So subtracting 1 gives the index of t_end.
    i1 = np.searchsorted(pt, ends_ts, side="right") - 1

    # Validate
    # Valid if t_start and t_end are not NaT, and i0 <= i1
    valid = (~pd.isna(starts_ts)) & (~pd.isna(ends_ts)) & (i0 <= i1)

    # Clip to bounds to avoid index errors, but use valid mask to filter later
    n_prices = len(pt)
    i0 = np.clip(i0, 0, n_prices - 1)
    i1 = np.clip(i1, 0, n_prices - 1)
    
    if n_prices > 0:
        min_ts = pt[0]
        max_ts = pt[-1]
        in_bounds = (starts_ts <= max_ts) & (ends_ts >= min_ts)
        valid = valid & in_bounds

    return i0, i1, valid

@jit(nopython=True, nogil=True, cache=True)
def _seq_bootstrap_numba(
    starts: np.ndarray,
    ends: np.ndarray,
    valid: np.ndarray,
    num_bars: int,
    num_samples: int,
    random_state: int
) -> np.ndarray:
    """
    Core Sequential Bootstrap logic optimized with Numba.
    
    Algorithm:
    1. Init concurrent counts (of selected samples) to 0.
    2. Init uniqueness (phi) to 1.0.
    3. Loop num_samples:
       a. Draw k with prob ~ phi.
       b. Update concurrent counts on k's interval.
       c. Recompute phi for OVERLAPPING labels only.
          phi[j] = avg(1 / (1 + concurrent)) over j's interval.
    """
    np.random.seed(random_state)
    
    n_labels = len(starts)
    selected = np.empty(num_samples, dtype=np.int64)
    count = 0
    
    # State
    concurrent = np.zeros(num_bars, dtype=np.int32)
    
    # Uniqueness (phi).
    # Initially concurrent=0, so 1/(1+0) = 1. Average is 1.
    phi = np.ones(n_labels, dtype=np.float64)

    # Set invalid to 0 prob
    for i in range(n_labels):
        if not valid[i]:
            phi[i] = 0.0

    # Track available explicitly to avoid picking same index?
    # Standard Seq Bootstrap allows replacement, but implementation usually removes it
    # or sets phi to 0 to prevent re-picking identical sample if that's the goal.

    for step in range(num_samples):
        # 1. Normalize probabilities
        sum_phi = np.sum(phi)
        if sum_phi <= 0:
             # Fallback if no valid candidates left
             break

        # Weighted Choice
        r = np.random.random() * sum_phi
        curr_sum = 0.0
        choice = -1

        # Linear scan is fast in Numba for typical N ~ 50k
        for i in range(n_labels):
            if phi[i] > 0:
                curr_sum += phi[i]
                if curr_sum >= r:
                    choice = i
                    break

        if choice == -1:
            # Rounding error edge case, pick last valid
            for i in range(n_labels - 1, -1, -1):
                if phi[i] > 0:
                    choice = i
                    break

        if choice == -1:
            # Should not happen if sum_phi > 0
            break

        selected[count] = choice
        count += 1

        # 2. Update concurrent
        s_k = starts[choice]
        e_k = ends[choice]

        # Vectorized update on concurrent array
        concurrent[s_k : e_k + 1] += 1

        # 3. Mark selected as unavailable
        phi[choice] = 0.0

        # 4. Update phi for OVERLAPPING labels
        # Overlap condition: start[j] <= e_k AND end[j] >= s_k
        # And phi[j] > 0 (still available)

        for j in range(n_labels):
            if phi[j] > 0: # Check if available
                s_j = starts[j]
                e_j = ends[j]

                # Check Overlap
                if s_j <= e_k and e_j >= s_k:
                    # Recompute uniqueness
                    # avg( 1 / (1 + concurrent[t]) ) for t in [s_j, e_j]

                    # Compute sum of inverse concurrent
                    inv_sum = 0.0
                    inv_count = 0
                    for t in range(s_j, e_j + 1):
                        c = concurrent[t]
                        inv_sum += 1.0 / (1.0 + c)
                        inv_count += 1

                    if inv_count > 0:
                        phi[j] = inv_sum / inv_count
                    else:
                        phi[j] = 0.0 # Should not happen if valid

    return selected[:count]

def seq_bootstrap(
    ind_matrix: Optional[pd.DataFrame] = None, # Deprecated
    sample_length: Optional[int] = None,
    random_state: Optional[int] = None,
    # New Arguments
    label_times: Optional[pd.DataFrame] = None,
    price_times: Optional[pd.DatetimeIndex] = None
) -> List[int]:
    """
    Sequential bootstrap respecting sample uniqueness.
    
    Args:
        ind_matrix: DEPRECATED.
        sample_length: Number of samples to draw.
        random_state: Random seed.
        label_times: DataFrame with ['t_start', 't_end'].
        price_times: DatetimeIndex of all available timestamps.
        
    Returns:
        List of selected sample indices.
    """
    # Adapter for legacy calls passing ind_matrix (if any)
    # If label_times/price_times are provided, use optimized path.
    if label_times is not None and price_times is not None:
        starts, ends, valid = get_label_intervals(label_times, price_times)
        num_bars = len(price_times)
        
        n_samples = sample_length if sample_length is not None else len(starts)
        rs = random_state if random_state is not None else 42
        
        selected_arr = _seq_bootstrap_numba(
            starts, ends, valid, num_bars, n_samples, rs
        )
        return selected_arr.tolist()
        
    raise ValueError("Must provide label_times and price_times for optimized sequential bootstrap.")

def get_sequential_bootstrap_samples(
    label_times: pd.DataFrame,
    price_times: pd.DatetimeIndex,
    n_samples: int,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Convenience function to get sequential bootstrap samples.
    """
    return np.array(seq_bootstrap(
        sample_length=n_samples,
        random_state=random_state,
        label_times=label_times,
        price_times=price_times
    ))

# Deprecated functions kept for compatibility if needed,
# but modified to warn or fail if used unexpectedly.
def get_ind_matrix(label_times, price_times):
    raise DeprecationWarning("get_ind_matrix is deprecated. Use get_label_intervals logic.")

def compute_avg_uniqueness_fast(ind_matrix):
    raise DeprecationWarning("compute_avg_uniqueness_fast is deprecated.")
