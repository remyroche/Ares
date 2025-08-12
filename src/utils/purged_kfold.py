import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterator, Tuple


@dataclass
class PurgedKFoldTime:
    """Purged and Embargoed K-Fold for DatetimeIndex time series.

    - Splits data into sequential folds by time order.
    - For each validation fold, removes from the training set any samples whose
      timestamps fall within [val_start - purge, val_end + embargo].
    - If index is not DatetimeIndex, falls back to sample-count-based purge/embargo
      interpreted as number of rows.
    """

    n_splits: int = 5
    purge: pd.Timedelta | int = pd.Timedelta(minutes=30)
    embargo: pd.Timedelta | int = pd.Timedelta(minutes=15)

    def split(self, X: pd.DataFrame, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame with an index")
        index = X.index
        n_samples = len(X)
        if self.n_splits < 2 or self.n_splits > n_samples:
            raise ValueError("n_splits must be at least 2 and at most n_samples")

        # Order by index (time)
        order = np.argsort(np.arange(n_samples))
        # Build fold boundaries
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        current = 0
        folds = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            folds.append((start, stop))
            current = stop

        is_time = isinstance(index, pd.DatetimeIndex)

        for i, (val_start_i, val_stop_i) in enumerate(folds):
            val_idx = np.arange(val_start_i, val_stop_i)
            if is_time:
                val_start_time = index[val_start_i]
                val_end_time = index[val_stop_i - 1]
                purge_delta = self.purge if isinstance(self.purge, pd.Timedelta) else pd.Timedelta(minutes=int(self.purge))
                embargo_delta = self.embargo if isinstance(self.embargo, pd.Timedelta) else pd.Timedelta(minutes=int(self.embargo))
                # Build boolean mask for training indices
                train_mask = np.ones(n_samples, dtype=bool)
                left_bound_time = val_start_time - purge_delta
                right_bound_time = val_end_time + embargo_delta
                # Purge and embargo window
                in_window = (index >= left_bound_time) & (index <= right_bound_time)
                train_mask[in_window.values] = False
                # Also exclude validation itself
                train_mask[val_idx] = False
                train_idx = np.nonzero(train_mask)[0]
            else:
                purge_n = int(self.purge) if isinstance(self.purge, (int, float)) else 0
                embargo_n = int(self.embargo) if isinstance(self.embargo, (int, float)) else 0
                left = max(0, val_start_i - purge_n)
                right = min(n_samples, val_stop_i + embargo_n)
                train_mask = np.ones(n_samples, dtype=bool)
                train_mask[left:right] = False
                train_mask[val_idx] = False
                train_idx = np.nonzero(train_mask)[0]

            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits