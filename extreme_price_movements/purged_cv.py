"""
Sklearn-Compatible Purged K-Fold Cross-Validation (De Prado Framework)

Prevents data leakage in time series by purging and embargoing samples
at train/test boundaries.
"""
from typing import Generator, Tuple
import numpy as np
from sklearn.model_selection import BaseCrossValidator
from .feature_selection_extreme_events import purged_embargoed_splits


class PurgedKFold(BaseCrossValidator):
    """
    Sklearn-compatible Purged K-Fold cross-validator.
    
    Implements De Prado's purged and embargoed CV to prevent leakage
    in time series with overlapping labels.
    
    Args:
        n_splits: Number of folds
        purge: Number of samples to purge at train/test boundary
        embargo: Number of samples to embargo after test set
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
        # Get number of samples
        if hasattr(X, 'shape'):
            n_samples = X.shape[0]
        else:
            n_samples = len(X)
        
        # Use existing purged_embargoed_splits function
        splits = purged_embargoed_splits(
            n_samples=n_samples,
            n_splits=self.n_splits,
            purge=self.purge,
            embargo=self.embargo,
            min_train_size=self.min_train_size
        )
        
        for train_idx, test_idx in splits:
            yield train_idx, test_idx
    
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
        purge: Number of samples to purge at boundaries
        embargo: Number of samples to embargo after test set
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
        
        indices = np.arange(n_samples)
        
        # Calculate fold sizes
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        # Create fold boundaries
        fold_bounds = [0]
        for size in fold_sizes:
            fold_bounds.append(fold_bounds[-1] + size)
        
        # Generate all possible test set positions
        for test_start_fold in range(self.n_splits - self.n_test_splits + 1):
            test_end_fold = test_start_fold + self.n_test_splits
            
            # Test set indices
            test_start_idx = fold_bounds[test_start_fold]
            test_end_idx = fold_bounds[test_end_fold]
            test_idx = indices[test_start_idx:test_end_idx]
            
            # Training set: all except test set, with purging and embargo
            train_idx = []
            
            # Add pre-test training data (with purging)
            if test_start_fold > 0:
                train_end = max(0, test_start_idx - self.purge)
                train_idx.extend(indices[:train_end])
            
            # Add post-test training data (with embargo)
            if test_end_fold < self.n_splits:
                train_start = min(n_samples, test_end_idx + self.embargo)
                train_idx.extend(indices[train_start:])
            
            train_idx = np.array(train_idx)
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


def cv_score_with_purging(
    model,
    X,
    y,
    cv_splitter,
    sample_weight=None,
    scoring_func=None
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
        
    Returns:
        Array of CV scores
    """
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.base import is_classifier
    
    if scoring_func is None:
        scoring_func = accuracy_score if is_classifier(model) else r2_score
    
    scores = []
    
    for train_idx, test_idx in cv_splitter.split(X):
        # Handle DataFrame/Series
        if hasattr(X, 'iloc'):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        else:
            X_train, X_test = X[train_idx], X[test_idx]
        
        if hasattr(y, 'iloc'):
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            y_train, y_test = y[train_idx], y[test_idx]
        
        # Handle sample weights
        if sample_weight is not None:
            if hasattr(sample_weight, 'iloc'):
                w_train = sample_weight.iloc[train_idx]
            else:
                w_train = sample_weight[train_idx]
            model.fit(X_train, y_train, sample_weight=w_train)
        else:
            model.fit(X_train, y_train)
        
        # Predict and score
        y_pred = model.predict(X_test)
        score = scoring_func(y_test, y_pred)
        scores.append(score)
    
    return np.array(scores)
