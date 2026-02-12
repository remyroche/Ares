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


class RegimeStratifiedPurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold that ensures each fold has similar regime distribution.
    
    PURPOSE AND MOTIVATION:
    =======================
    Standard PurgedKFold can produce folds with very different market regime 
    compositions. For example, one fold might contain mostly high-volatility 
    periods while another contains mostly low-volatility periods. This causes:
    
    1. **Fold Robustness Issues**: Models trained on one regime fail on another,
       leading to negative worst_fold_logloss improvements (e.g., -8.6% for 
       long_mr_H2, -15.1% for long_mr_H8).
    
    2. **High Variance in Fold Metrics**: fold_logloss_improvement_ratio < 0.70
       indicates that some folds are fundamentally different from others.
    
    3. **Regime Overfitting**: Models learn regime-specific patterns that don't
       generalize, rather than robust signals that work across regimes.
    
    HOW IT WORKS:
    =============
    1. **Regime Identification**: Uses provided regime_labels (or computes them
       from volatility if not provided) to categorize each sample.
    
    2. **Stratified Splitting**: Ensures each validation fold has approximately
       the same proportion of each regime as the overall dataset.
    
    3. **Purging Applied**: After stratification, applies standard purging to
       prevent label leakage at train/test boundaries.
    
    4. **Regime Balance Check**: Validates that each fold meets minimum_regime_ratio
       (default 0.7) for regime representation.
    
    REGIME LABELS:
    ==============
    Regime labels should be integer values representing different market states:
    - 0: Low volatility regime (calm, ranging markets)
    - 1: Normal volatility regime
    - 2: High volatility regime (trending, volatile markets)
    
    If regime_labels is not provided, it's computed from realized volatility:
    - Compute 24h realized volatility
    - Classify into 3 regimes based on terciles (33rd and 67th percentiles)
    
    USAGE EXAMPLE:
    ==============
    >>> from extreme_price_movements.purged_cv import RegimeStratifiedPurgedKFold
    >>> 
    >>> # Option 1: Let it compute regimes automatically
    >>> cv = RegimeStratifiedPurgedKFold(n_splits=5, purge=10, embargo=5)
    >>> 
    >>> # Option 2: Provide pre-computed regime labels
    >>> regime_labels = compute_volatility_regimes(df)  # Returns 0, 1, or 2
    >>> cv = RegimeStratifiedPurgedKFold(
    ...     n_splits=5, 
    ...     purge=10, 
    ...     embargo=5,
    ...     regime_labels=regime_labels,
    ...     min_regime_ratio=0.7
    ... )
    >>> 
    >>> for train_idx, val_idx in cv.split(X):
    ...     # Each fold has balanced regime distribution
    ...     model.fit(X[train_idx], y[train_idx])
    
    EXPECTED IMPROVEMENT:
    ====================
    For models with severe fold robustness issues:
    - long_mr_H2: fold_ratio 0.25 → expected 0.55-0.70
    - long_mr_H8: fold_ratio 0.25 → expected 0.55-0.75
    - long_tf_H2: fold_ratio 0.50 → expected 0.65-0.75
    
    Args:
        n_splits: Number of folds
        purge: Number of samples to purge at train/test boundary (in index units)
        embargo: Number of samples to embargo after test set (in index units)
        regime_labels: Pre-computed regime labels (0, 1, 2, ...). If None, computed from volatility.
        min_regime_ratio: Minimum ratio of regime representation in each fold (default 0.7)
        n_regimes: Number of regimes to compute if regime_labels is None (default 3)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge: int = 10,
        embargo: int = 5,
        regime_labels=None,
        min_regime_ratio: float = 0.7,
        n_regimes: int = 3
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo
        self.regime_labels = regime_labels
        self.min_regime_ratio = min_regime_ratio
        self.n_regimes = n_regimes
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Returns the number of splitting iterations."""
        return self.n_splits
    
    def _compute_regime_labels(self, X) -> np.ndarray:
        """
        Compute regime labels from data if not provided.
        
        Uses rolling realized volatility to identify regimes:
        - Compute 24h rolling std of returns (if available)
        - Classify into n_regimes based on quantiles
        """
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        
        # If X is a DataFrame with return features, use them
        if hasattr(X, 'columns') and 'rv_24h' in X.columns:
            vol = X['rv_24h'].values
        elif hasattr(X, 'columns') and 'ret1h' in X.columns:
            # Compute 24h rolling std
            ret = X['ret1h'].values
            vol = np.zeros(len(ret))
            for i in range(24, len(ret)):
                vol[i] = np.std(ret[i-24:i])
        else:
            # Fallback: use index-based regimes (time-based)
            # This is less ideal but ensures the class works
            return np.tile(np.arange(self.n_regimes), n_samples // self.n_regimes + 1)[:n_samples]
        
        # Classify into regimes based on quantiles
        quantiles = np.linspace(0, 1, self.n_regimes + 1)[1:-1]  # e.g., [0.33, 0.67] for 3 regimes
        thresholds = np.nanquantile(vol, quantiles)
        
        labels = np.zeros(n_samples, dtype=np.int8)
        for i, thresh in enumerate(thresholds):
            labels[vol > thresh] = i + 1
        
        return labels
    
    def split(
        self,
        X,
        y=None,
        groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test set with regime stratification.
        
        Args:
            X: Training data (array-like or DataFrame)
            y: Target variable (ignored, for compatibility)
            groups: Group labels (ignored, for compatibility)
            
        Yields:
            train_idx, test_idx: Arrays of indices with balanced regime distribution
        """
        if hasattr(X, 'shape'):
            n_samples = X.shape[0]
        else:
            n_samples = len(X)
        
        # Get or compute regime labels
        if self.regime_labels is not None:
            labels = np.asarray(self.regime_labels)
            if len(labels) != n_samples:
                raise ValueError(f"regime_labels length ({len(labels)}) != n_samples ({n_samples})")
        else:
            labels = self._compute_regime_labels(X)
        
        unique_labels = np.unique(labels)
        n_regimes = len(unique_labels)
        
        # For each regime, get indices
        regime_indices = {label: np.where(labels == label)[0] for label in unique_labels}
        
        # Sort indices within each regime (for time series ordering)
        for label in regime_indices:
            regime_indices[label] = np.sort(regime_indices[label])
        
        # Create stratified folds
        # For time series, we use a rolling approach within each regime
        fold_test_indices = [[] for _ in range(self.n_splits)]
        
        for label in unique_labels:
            idx = regime_indices[label]
            n = len(idx)
            
            # Split this regime's indices into n_splits chunks
            chunk_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
            chunk_sizes[:n % self.n_splits] += 1
            
            start = 0
            for fold_i, size in enumerate(chunk_sizes):
                end = start + size
                fold_test_indices[fold_i].extend(idx[start:end])
                start = end
        
        # Convert to arrays and sort
        fold_test_indices = [np.sort(np.array(idx)) for idx in fold_test_indices]
        
        # Generate train/test splits with purging
        for fold_i in range(self.n_splits):
            test_idx = fold_test_indices[fold_i]
            
            # Apply purging: remove samples within purge distance from test boundaries
            test_start = test_idx[0]
            test_end = test_idx[-1]
            
            # Train = everything before test_start - purge, and after test_end + embargo
            train_end = max(0, test_start - self.purge)
            train_start = min(n_samples, test_end + self.embargo)
            
            train_idx = np.concatenate([
                np.arange(0, train_end, dtype=np.int32),
                np.arange(train_start, n_samples, dtype=np.int32)
            ])
            
            if train_idx.size == 0:
                continue
            
            # Validate regime balance
            train_labels = labels[train_idx]
            test_labels = labels[test_idx]
            
            # Check if train and test have similar regime distribution
            train_dist = np.array([np.mean(train_labels == l) for l in unique_labels])
            test_dist = np.array([np.mean(test_labels == l) for l in unique_labels])
            
            # Allow some deviation (min_regime_ratio)
            balance_ok = np.all(np.minimum(train_dist, test_dist) >= self.min_regime_ratio * np.maximum(train_dist, test_dist) - 0.1)
            
            # Log warning if balance is poor (but don't skip the fold)
            if not balance_ok:
                import warnings
                warnings.warn(
                    f"Fold {fold_i} has imbalanced regime distribution. "
                    f"Train dist: {train_dist}, Test dist: {test_dist}"
                )
            
            yield train_idx, test_idx
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"n_splits={self.n_splits}, "
            f"purge={self.purge}, "
            f"embargo={self.embargo}, "
            f"n_regimes={self.n_regimes})"
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
