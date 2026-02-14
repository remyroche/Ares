"""
Sklearn-Compatible Purged K-Fold Cross-Validation (De Prado Framework)

Prevents data leakage in time series by purging and embargoing samples
at train/test boundaries.
"""
from typing import Generator, Tuple
import numpy as np
from sklearn.model_selection import BaseCrossValidator


class PurgedKFold(BaseCrossValidator):
    """
    Sklearn-compatible Purged K-Fold cross-validator.
    
    Implements De Prado's purged and embargoed CV to prevent leakage
    in time series with overlapping labels.
    
    Args:
        n_splits: Number of folds
        purge: Number of samples to purge at train/test boundary (in index units),
               or timedelta/seconds if times are provided
        embargo: Number of samples to embargo after test set (in index units),
                 or timedelta/seconds if times are provided
        min_train_size: Minimum training set size (optional)
        times: Array of timestamps for time-based purging (optional)
    
    Note:
        If times are provided, purge and embargo are interpreted as seconds.
        Otherwise, they are interpreted as number of samples.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge: int = 5,
        embargo: int = 0,
        min_train_size: int = None,
        times: np.ndarray = None
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo
        self.min_train_size = min_train_size
        self.times = times
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Returns the number of splitting iterations."""
        return self.n_splits
    
    def split(
        self,
        X,
        y=None,
        groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Training data (array-like or DataFrame)
            y: Target variable (ignored, for compatibility)
            groups: Group labels (ignored, for compatibility)
            
        Yields:
            train_idx, test_idx: Arrays of indices
        """
        if hasattr(X, 'shape'):
            n_samples = X.shape[0]
        else:
            n_samples = len(X)
        
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=np.int32)
        fold_sizes[: n_samples % self.n_splits] += 1
        
        # Time-based purging if times are available
        times = self.times
        if times is not None and hasattr(times, '__len__') and len(times) == n_samples:
            # Convert to numeric (seconds since epoch) for comparison
            import pandas as pd
            if hasattr(times, 'dtype') and np.issubdtype(times.dtype, np.datetime64):
                times_numeric = pd.to_datetime(times).astype(np.int64) // 10**9
            else:
                times_numeric = np.asarray(times, dtype=np.float64)
            
            purge_seconds = float(self.purge)
            embargo_seconds = float(self.embargo)
            
            start = 0
            for i in range(self.n_splits):
                val_start = start
                val_end = start + fold_sizes[i]
                start = val_end

                val_idx = np.arange(val_start, val_end, dtype=np.int32)
                
                # Time-based purge: find all indices whose time is before (test_start_time - purge)
                test_start_time = times_numeric[val_start]
                test_end_time = times_numeric[min(val_end - 1, n_samples - 1)]
                
                purge_time = test_start_time - purge_seconds
                train_mask = times_numeric[:val_start] < purge_time
                
                train_idx = np.where(train_mask)[0].astype(np.int32)
                
                if train_idx.size == 0:
                    continue
                if self.min_train_size is not None and train_idx.size < self.min_train_size:
                    continue

                yield train_idx, val_idx
        else:
            # Index-based purging (original behavior)
            start = 0
            for i in range(self.n_splits):
                val_start = start
                val_end = start + fold_sizes[i]
                start = val_end

                val_idx = np.arange(val_start, val_end, dtype=np.int32)
                train_end = max(0, val_start - self.purge)
                train_idx = np.arange(0, train_end, dtype=np.int32)

                if train_idx.size == 0:
                    continue
                if self.min_train_size is not None and train_idx.size < self.min_train_size:
                    continue

                yield train_idx, val_idx
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"n_splits={self.n_splits}, "
            f"purge={self.purge}, "
            f"embargo={self.embargo})"
        )


class CombinatorialPurgedKFold(BaseCrossValidator):
    """
    Combinatorial Purged K-Fold (CPCV) from AFML Chapter 7.
    
    Generates all possible combinations of train/test splits to maximize
    the number of backtest paths while respecting purging constraints.
    
    This provides more robust performance estimates at the cost of
    computational expense.
    
    Args:
        n_splits: Number of folds
        n_test_splits: Number of consecutive folds to use as test set
        purge: Number of samples to purge at boundaries (in index units)
        embargo: Number of samples to embargo after test set (in index units)
    """
    
    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        purge: int = 5,
        embargo: int = 0
    ):
        if n_splits < 3:
            raise ValueError("n_splits must be >= 3 for CPCV")
        if n_test_splits < 1 or n_test_splits >= n_splits:
            raise ValueError("n_test_splits must be in [1, n_splits)")
        
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge = purge
        self.embargo = embargo
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Returns the number of splitting iterations."""
        # Number of combinations = n_splits - n_test_splits + 1
        return self.n_splits - self.n_test_splits + 1
    
    def split(
        self,
        X,
        y=None,
        groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate combinatorial purged splits.
        
        Args:
            X: Training data
            y: Target variable (ignored)
            groups: Group labels (ignored)
            
        Yields:
            train_idx, test_idx: Arrays of indices
        """
        if hasattr(X, 'shape'):
            n_samples = X.shape[0]
        else:
            n_samples = len(X)
        
        # Calculate fold sizes (vectorized)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=np.int32)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        # Create fold boundaries
        fold_bounds = np.r_[0, fold_sizes.cumsum()]  # shape (n_splits+1,)
        
        # Generate all possible test set positions
        for test_start_fold in range(self.n_splits - self.n_test_splits + 1):
            test_end_fold = test_start_fold + self.n_test_splits
            
            # Test set indices
            test_start_idx = int(fold_bounds[test_start_fold])
            test_end_idx = int(fold_bounds[test_end_fold])
            
            pre_end = max(0, test_start_idx - self.purge)
            post_start = min(n_samples, test_end_idx + self.embargo)
            
            test_idx = np.arange(test_start_idx, test_end_idx, dtype=np.int32)
            # Use np.r_ to efficiently build train indices without lists
            train_idx = np.r_[0:pre_end, post_start:n_samples].astype(np.int32, copy=False)
            
            if train_idx.size > 0 and test_idx.size > 0:
                yield train_idx, test_idx


def cv_score_with_purging(
    model,
    X,
    y,
    cv_splitter,
    sample_weight=None,
    scoring_func=None,
    predict_method="predict"
):
    """
    Cross-validation scoring with purged folds.
    
    Convenience function for running CV with a purged splitter.
    
    Args:
        model: Sklearn-compatible model
        X: Feature matrix
        y: Target variable
        cv_splitter: PurgedKFold or CombinatorialPurgedKFold instance
        sample_weight: Optional sample weights
        scoring_func: Scoring function (default: accuracy for classifiers)
        predict_method: Method to call on estimator (predict, predict_proba, decision_function)
        
    Returns:
        Array of CV scores
    """
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.base import is_classifier, clone
    
    if scoring_func is None:
        scoring_func = accuracy_score if is_classifier(model) else r2_score
    
    scores = []
    
    # Pre-check accessor types to avoid repeated checks inside loop
    is_pandas_X = hasattr(X, "iloc")
    is_pandas_y = hasattr(y, "iloc")
    is_pandas_w = sample_weight is not None and hasattr(sample_weight, "iloc")

    for train_idx, test_idx in cv_splitter.split(X):
        est = clone(model)

        # Handle X
        if is_pandas_X:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        else:
            X_train, X_test = X[train_idx], X[test_idx]
        
        # Handle y
        if is_pandas_y:
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            y_train, y_test = y[train_idx], y[test_idx]
        
        # Handle sample weights
        if sample_weight is not None:
            if is_pandas_w:
                w_train = sample_weight.iloc[train_idx]
            else:
                w_train = sample_weight[train_idx]
            est.fit(X_train, y_train, sample_weight=w_train)
        else:
            est.fit(X_train, y_train)
        
        # Predict and score
        y_pred = getattr(est, predict_method)(X_test)
        score = scoring_func(y_test, y_pred)
        scores.append(score)
    
    return np.array(scores)


class RegimeStratifiedPurgedKFold(BaseCrossValidator):
    """
    Regime-Stratified Purged K-Fold Cross-Validator.
    
    Balances Low/Normal/High Volatility regimes across folds to ensure
    each fold has a representative mix of market regimes. This mitigates
    fold robustness issues where some folds may be dominated by a single
    regime (e.g., all low-vol periods), leading to poor generalization.
    
    Args:
        n_splits: Number of folds (default 3, reduced from 4 for regularization)
        purge: Number of samples to purge at train/test boundary
        embargo: Number of samples to embargo after test set
        regime_col: Name of regime column or array of regime labels
        n_regime_bins: Number of regime bins if regime_col is continuous (default 3: Low/Normal/High)
        min_regime_ratio: Minimum ratio of each regime in each fold (default 0.5)
    
    Example:
        >>> cv = RegimeStratifiedPurgedKFold(n_splits=3, purge=12, regime_col='vol_regime')
        >>> for train_idx, test_idx in cv.split(X, y):
        ...     # Each fold has balanced regime distribution
        ...     model.fit(X[train_idx], y[train_idx])
    """
    
    def __init__(
        self,
        n_splits: int = 3,
        purge: int = 12,
        embargo: int = 0,
        regime_col: str = None,
        n_regime_bins: int = 3,
        min_regime_ratio: float = 0.5
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo
        self.regime_col = regime_col
        self.n_regime_bins = n_regime_bins
        self.min_regime_ratio = min_regime_ratio
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Returns the number of splitting iterations."""
        return self.n_splits
    
    def _get_regime_labels(self, X, y=None) -> np.ndarray:
        """Extract or compute regime labels from input data."""
        if self.regime_col is not None:
            # Try to get regime column from DataFrame
            if hasattr(X, 'columns') and self.regime_col in X.columns:
                regime_values = X[self.regime_col].to_numpy()
            elif isinstance(self.regime_col, np.ndarray):
                regime_values = self.regime_col
            else:
                # Fall back to computing from volatility
                regime_values = self._compute_regime_from_vol(X)
        else:
            # Compute regime from volatility proxy
            regime_values = self._compute_regime_from_vol(X)
        
        # If continuous, bin into n_regime_bins
        if np.issubdtype(regime_values.dtype, np.floating):
            # Use quantile-based binning for robustness
            quantiles = np.linspace(0, 1, self.n_regime_bins + 1)
            thresholds = np.nanquantile(regime_values, quantiles)
            regime_labels = np.digitize(regime_values, thresholds[1:-1])
        else:
            regime_labels = regime_values.astype(np.int32)
        
        return regime_labels
    
    def _compute_regime_from_vol(self, X) -> np.ndarray:
        """Compute regime labels from volatility proxy (rolling std of returns)."""
        if hasattr(X, 'shape'):
            n_samples = X.shape[0]
        else:
            n_samples = len(X)
        
        # Use variance across features as volatility proxy
        if hasattr(X, 'to_numpy'):
            X_arr = X.to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)
        
        # Compute row-wise variance as volatility proxy
        row_var = np.nanvar(X_arr, axis=1)
        
        return row_var
    
    def split(
        self,
        X,
        y=None,
        groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate regime-stratified purged splits.
        
        Each fold is constructed to have a balanced mix of regimes,
        ensuring that training and test sets both contain Low/Normal/High
        volatility periods in representative proportions.
        
        Args:
            X: Training data (array-like or DataFrame)
            y: Target variable (optional, used for regime detection)
            groups: Group labels (ignored, for compatibility)
            
        Yields:
            train_idx, test_idx: Arrays of indices
        """
        if hasattr(X, 'shape'):
            n_samples = X.shape[0]
        else:
            n_samples = len(X)
        
        # Get regime labels
        regime_labels = self._get_regime_labels(X, y)
        unique_regimes = np.unique(regime_labels[~np.isnan(regime_labels)])
        n_regimes = len(unique_regimes)
        
        # Stratify by regime: sort indices within each regime
        regime_indices = {}
        for r in unique_regimes:
            regime_indices[r] = np.where(regime_labels == r)[0]
        
        # For each regime, assign indices to folds in round-robin fashion
        # This ensures each fold gets a balanced mix
        fold_test_indices = [[] for _ in range(self.n_splits)]
        
        for r in unique_regimes:
            r_indices = regime_indices[r]
            n_in_regime = len(r_indices)
            
            # Shuffle within regime for randomness (deterministic with fixed seed)
            rng = np.random.default_rng(42)
            rng.shuffle(r_indices)
            
            # Assign to folds in round-robin
            for i, idx in enumerate(r_indices):
                fold_idx = i % self.n_splits
                fold_test_indices[fold_idx].append(idx)
        
        # Convert to arrays and sort
        fold_test_indices = [np.sort(np.array(f, dtype=np.int32)) for f in fold_test_indices]
        
        # Generate train/test splits with purging
        for fold_idx in range(self.n_splits):
            test_idx = fold_test_indices[fold_idx]
            
            if len(test_idx) == 0:
                continue
            
            # Train = all other folds, with purging
            train_folds = [i for i in range(self.n_splits) if i != fold_idx]
            train_idx_raw = np.concatenate([fold_test_indices[i] for i in train_folds])
            train_idx_raw = np.sort(train_idx_raw)
            
            # Apply purging: remove samples within purge distance of test boundaries
            test_min, test_max = test_idx.min(), test_idx.max()
            
            # Purge from train: remove indices within purge distance of test set
            purge_mask = (train_idx_raw < test_min - self.purge) | (train_idx_raw > test_max + self.purge)
            train_idx = train_idx_raw[purge_mask]
            
            # Apply embargo: remove samples immediately after test set
            if self.embargo > 0:
                embargo_mask = train_idx > test_max + self.embargo
                # Keep samples before test set and after embargo period
                train_idx = train_idx[(train_idx < test_min) | embargo_mask]
            
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            
            # Verify regime balance in this fold
            train_regimes = regime_labels[train_idx]
            test_regimes = regime_labels[test_idx]
            
            # Check if each regime is represented in both train and test
            train_regime_counts = np.bincount(train_regimes[~np.isnan(train_regimes)].astype(int))
            test_regime_counts = np.bincount(test_regimes[~np.isnan(test_regimes)].astype(int))
            
            # Yield the split
            yield train_idx, test_idx
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"n_splits={self.n_splits}, "
            f"purge={self.purge}, "
            f"embargo={self.embargo}, "
            f"n_regime_bins={self.n_regime_bins})"
        )
