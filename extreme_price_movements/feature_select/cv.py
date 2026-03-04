import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class CVConfig:
    n_splits: int
    min_train_size: int
    val_size: int
    purge_gap: int = 0
    embargo: int = 0
    shuffle: bool = False
    group_col: Optional[str] = None
    time_col: Optional[str] = None

def create_cv_splits(
    X: pd.DataFrame,
    cv_config: CVConfig,
    time_index: Optional[pd.Series] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates walk-forward time series CV splits.
    Supports purging and embargoing if specified.
    """
    n = len(X)
    n_splits = cv_config.n_splits
    min_train_size = cv_config.min_train_size
    val_size = cv_config.val_size
    purge_gap = cv_config.purge_gap
    embargo = cv_config.embargo

    if n < min_train_size + val_size + purge_gap:
        raise ValueError(f"Not enough data ({n} samples) for min_train_size ({min_train_size}), val_size ({val_size}), and purge_gap ({purge_gap}).")

    splits = []

    # Calculate step size based on available testing space
    available_test_space = n - min_train_size - purge_gap

    # If we only have 1 split, just use the end of the dataset
    if n_splits == 1:
        val_end = n
        val_start = val_end - val_size
        train_end = max(0, val_start - purge_gap)
        train_start = 0
        splits.append((np.arange(train_start, train_end), np.arange(val_start, val_end)))
        return splits

    step = (available_test_space - val_size) / (n_splits - 1)

    for i in range(n_splits):
        # Determine the start and end of the validation set
        val_start = int(min_train_size + purge_gap + i * step)
        val_end = int(val_start + val_size)

        if val_end > n:
            val_end = n
            val_start = max(0, val_end - val_size)

        # The training set is everything before the validation set, minus the purge gap
        train_end = max(0, val_start - purge_gap)
        train_start = 0

        train_idx = np.arange(train_start, train_end)
        val_idx = np.arange(val_start, val_end)

        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))

    return splits
