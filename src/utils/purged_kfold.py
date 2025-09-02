"""
Purged and Embargoed K-Fold for DatetimeIndex time series.

- Splits data into sequential folds by time order.
- For each validation fold, removes from the training set any samples whose
  timestamps fall within [val_start - purge, val_end + embargo].
- If index is not DatetimeIndex, falls back to sample-count-based purge/embargo
  interpreted as number of rows.
"""

from collections.abc import Iterator
from dataclasses import dataclass
import logging
from typing import Any, Union, Optional

# Try to import numpy and pandas, with fallbacks
try:
    import numpy as np
    import pandas as pd
    _NUMPY_AVAILABLE = True
    _PANDAS_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    _PANDAS_AVAILABLE = False
    # Create mock classes for type hints
    class np:
        @staticmethod
        def full(shape, fill_value, dtype=None):
            return [fill_value] * shape[0]
        
        @staticmethod
        def ones(shape, dtype=None):
            return [True] * shape[0]
        
        @staticmethod
        def nonzero(array):
            return [i for i, val in enumerate(array) if val]
        
        @staticmethod
        def arange(start, stop):
            return list(range(start, stop))
    
    class pd:
        class Timedelta:
            def __init__(self, minutes=0):
                self.minutes = minutes

# Get system logger
try:
    from src.utils.logger import system_logger
except ImportError:
    system_logger = logging.getLogger(__name__)


@dataclass
class PurgedKFoldTime:
    """Purged and Embargoed K-Fold for time series data."""

    n_splits: int = 5
    purge: Union[pd.Timedelta, int] = pd.Timedelta(minutes=30) if _PANDAS_AVAILABLE else 30
    embargo: Union[pd.Timedelta, int] = pd.Timedelta(minutes=15) if _PANDAS_AVAILABLE else 15

    def __post_init__(self) -> None:
        """Initialize logger and validation."""
        self.logger = system_logger.getChild("PurgedKFoldTime")
        
        # Check dependencies
        if not _NUMPY_AVAILABLE:
            self.logger.warning("NumPy not available - using fallback implementation")
        if not _PANDAS_AVAILABLE:
            self.logger.warning("Pandas not available - using fallback implementation")
        
        # Validate parameters
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        
        if isinstance(self.purge, (int, float)) and self.purge < 0:
            raise ValueError("purge must be non-negative")
        
        if isinstance(self.embargo, (int, float)) and self.embargo < 0:
            raise ValueError("embargo must be non-negative")
        
        self.is_initialized = True

    def split(
        self, X, y=None, groups=None
    ) -> Iterator[tuple[list, list]]:
        """Split the data into training and validation sets."""
        if not hasattr(X, 'index'):
            msg = "X must have an index attribute"
            raise ValueError(msg)
        
        index = X.index
        n_samples = len(X)
        
        if self.n_splits < 2 or self.n_splits > n_samples:
            msg = "n_splits must be at least 2 and at most n_samples"
            raise ValueError(msg)

        # Order by index (time)
        sorted_indices = np.arange(n_samples)
        
        # Build fold boundaries
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        folds = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            folds.append((start, stop))
            current = stop

        is_time = hasattr(index, 'to_pydatetime') and _PANDAS_AVAILABLE

        for _i, (val_start_i, val_stop_i) in enumerate(folds):
            val_idx = np.arange(val_start_i, val_stop_i)
            
            if is_time:
                val_start_time = index[val_start_i]
                val_end_time = index[val_stop_i - 1]
                
                purge_delta = (
                    self.purge
                    if isinstance(self.purge, pd.Timedelta)
                    else pd.Timedelta(minutes=int(self.purge))
                )
                embargo_delta = (
                    self.embargo
                    if isinstance(self.embargo, pd.Timedelta)
                    else pd.Timedelta(minutes=int(self.embargo))
                )
                
                # Build boolean mask for training indices
                train_mask = np.ones(n_samples, dtype=bool)
                left_bound_time = val_start_time - purge_delta
                right_bound_time = val_end_time + embargo_delta
                
                # Purge and embargo window
                in_window = (index >= left_bound_time) & (index <= right_bound_time)
                if hasattr(in_window, 'values'):
                    train_mask[in_window.values] = False
                else:
                    # Fallback for non-pandas boolean arrays
                    for i, val in enumerate(in_window):
                        if val:
                            train_mask[i] = False
                
                # Also exclude validation itself
                train_mask[val_idx] = False
                train_idx = np.nonzero(train_mask)[0]
            else:
                # Fallback to sample-count-based purge/embargo
                purge_n = int(self.purge) if isinstance(self.purge, (int, float)) else 0
                embargo_n = (
                    int(self.embargo) if isinstance(self.embargo, (int, float)) else 0
                )
                
                left = max(0, val_start_i - purge_n)
                right = min(n_samples, val_stop_i + embargo_n)
                
                train_mask = np.ones(n_samples, dtype=bool)
                train_mask[left:right] = False
                train_mask[val_idx] = False
                train_idx = np.nonzero(train_mask)[0]

            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get the number of splits.
        
        Args:
            X: Input data (optional, for compatibility with sklearn interface)
            y: Target data (optional, for compatibility with sklearn interface)
            groups: Group labels (optional, for compatibility with sklearn interface)
            
        Returns:
            Number of splits configured for this cross-validator
        """
        return self.n_splits
