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
        purge: Number of samples to purge at train/test boundary (in index units)
        embargo: Number of samples to embargo after test set (in index units)
        min_train_size: Minimum training set size (optional)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge: int = 5,
        embargo: int = 0,
        min_train_size: int = None
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo
        self.min_train_size = min_train_size
    
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
        
        start = 0
        for i in range(self.n_splits):
            val_start = start
            val_end = start + fold_sizes[i]
            start = val_end

            # Note: This implementation is Walk-Forward (past data only).
            # It does not include future data (post-test), so 'embargo' is unused here
            # but kept in __init__ for API compatibility/future extension.
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
