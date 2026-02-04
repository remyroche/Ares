"""
Sequential Bootstrap (AFML Chapter 4.5)

Generates bootstrapped samples respecting sample uniqueness and temporal structure.
"""
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple


def get_ind_matrix(label_times: pd.DataFrame, price_times: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build indicator matrix showing which price bars each label touches.
    
    Args:
        label_times: DataFrame with ['t_start', 't_end'] indexed by sample
        price_times: DatetimeIndex of all available timestamps
        
    Returns:
        Binary matrix (labels × timestamps) indicating overlap
    """
    ind_matrix = pd.DataFrame(0, index=label_times.index, columns=price_times, dtype=np.int8)
    
    for idx in label_times.index:
        t_start = label_times.loc[idx, 't_start']
        t_end = label_times.loc[idx, 't_end']
        
        if pd.isna(t_start) or pd.isna(t_end):
            continue
        
        # Mark all timestamps within [t_start, t_end] as 1
        mask = (price_times >= t_start) & (price_times <= t_end)
        ind_matrix.loc[idx, mask] = 1
    
    return ind_matrix


def compute_avg_uniqueness_fast(ind_matrix: pd.DataFrame) -> pd.Series:
    """
    Fast computation of average uniqueness from indicator matrix.
    
    Args:
        ind_matrix: Binary matrix from get_ind_matrix
        
    Returns:
        Series of uniqueness values per label
    """
    # Concurrent labels per timestamp
    concurrent = ind_matrix.sum(axis=0)
    
    # For each label, compute average 1/concurrent over its active timestamps
    uniqueness = pd.Series(0.0, index=ind_matrix.index)
    
    for idx in ind_matrix.index:
        active_mask = ind_matrix.loc[idx] == 1
        if active_mask.sum() == 0:
            uniqueness.loc[idx] = 1.0
            continue
        
        # Average uniqueness = mean(1 / concurrent) over active timestamps
        concurrent_at_label = concurrent[active_mask]
        inv_concurrent = 1.0 / concurrent_at_label.replace(0, 1)
        uniqueness.loc[idx] = inv_concurrent.mean()
    
    return uniqueness


def seq_bootstrap(
    ind_matrix: pd.DataFrame,
    sample_length: Optional[int] = None,
    random_state: Optional[int] = None
) -> List[int]:
    """
    Sequential bootstrap respecting sample uniqueness.
    
    Algorithm (AFML 4.5.2):
    1. Compute average uniqueness for all samples
    2. Draw samples with probability proportional to uniqueness
    3. After each draw, reduce uniqueness of overlapping samples
    4. Repeat until desired sample size reached
    
    Args:
        ind_matrix: Binary indicator matrix (labels × timestamps)
        sample_length: Number of samples to draw (default: len(ind_matrix))
        random_state: Random seed for reproducibility
        
    Returns:
        List of selected sample indices
    """
    if sample_length is None:
        sample_length = len(ind_matrix)
    
    rng = np.random.RandomState(random_state)
    
    # Initial average uniqueness
    phi = compute_avg_uniqueness_fast(ind_matrix)
    
    selected_indices = []
    available_indices = list(ind_matrix.index)
    
    for _ in range(sample_length):
        if len(available_indices) == 0:
            break
        
        # Normalize probabilities
        phi_available = phi.loc[available_indices]
        if phi_available.sum() == 0:
            # Fallback to uniform
            prob = np.ones(len(available_indices)) / len(available_indices)
        else:
            prob = phi_available / phi_available.sum()
        
        # Draw sample
        choice = rng.choice(available_indices, p=prob.values)
        selected_indices.append(choice)
        
        # Reduce uniqueness of overlapping samples
        # Get timestamps touched by selected sample
        touched_times = ind_matrix.columns[ind_matrix.loc[choice] == 1]
        
        # For each remaining sample, reduce its uniqueness if it overlaps
        for idx in available_indices:
            if idx == choice:
                continue
            
            # Check if this sample overlaps with selected sample
            overlap_mask = ind_matrix.loc[idx, touched_times] == 1
            if overlap_mask.any():
                # Reduce uniqueness proportional to overlap
                overlap_fraction = overlap_mask.sum() / (ind_matrix.loc[idx] == 1).sum()
                phi.loc[idx] *= (1.0 - overlap_fraction)
        
        # Remove selected sample from available pool
        available_indices.remove(choice)
        phi.loc[choice] = 0.0
    
    return selected_indices


def get_sequential_bootstrap_samples(
    label_times: pd.DataFrame,
    price_times: pd.DatetimeIndex,
    n_samples: int,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Convenience function to get sequential bootstrap samples.
    
    Args:
        label_times: DataFrame with ['t_start', 't_end']
        price_times: All available timestamps
        n_samples: Number of samples to draw
        random_state: Random seed
        
    Returns:
        Array of selected indices
    """
    ind_matrix = get_ind_matrix(label_times, price_times)
    selected = seq_bootstrap(ind_matrix, sample_length=n_samples, random_state=random_state)
    return np.array(selected)
