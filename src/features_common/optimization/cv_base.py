"""
Base Cross-Validation Splitter

Provides shared cross-validation logic with embargo support for both
feature_generation and feature_engineering_roadmap lookback optimization.
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import logging

logger = logging.getLogger(__name__)


class BaseCVSplitter:
    """
    Base class for time series cross-validation with embargo.
    
    This class provides common CV splitting logic that can be used by both:
    - feature_generation/utils/optimization/lookback_optimizer.py
    - feature_engineering_roadmap/lookback_selection.py
    
    The embargo feature helps prevent data leakage by creating a gap
    between training and validation sets.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        embargo_pct: float = 0.1,
        min_train_size: Optional[int] = None
    ):
        """
        Initialize CV splitter.
        
        Args:
            n_folds: Number of folds for time series split
            embargo_pct: Percentage of validation data to skip as embargo
            min_train_size: Minimum training size (None = use sklearn default)
        """
        self.n_folds = n_folds
        self.embargo_pct = embargo_pct
        self.min_train_size = min_train_size
        
        if not 0 <= embargo_pct <= 0.5:
            raise ValueError("embargo_pct must be between 0 and 0.5")
        
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2")
    
    def split_with_embargo(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Perform time series split with embargo between train/val sets.
        
        Args:
            X: Feature DataFrame with DatetimeIndex
            y: Optional target Series (not used but kept for sklearn compatibility)
            
        Returns:
            List of (train_index, val_index) tuples
            
        Example:
            >>> splitter = BaseCVSplitter(n_folds=3, embargo_pct=0.1)
            >>> for train_idx, val_idx in splitter.split_with_embargo(X):
            ...     X_train, X_val = X.loc[train_idx], X.loc[val_idx]
        """
        if X.empty:
            logger.warning("Empty DataFrame provided to CV splitter")
            return []
        
        n_samples = len(X)
        embargo_size = int(n_samples * self.embargo_pct)
        
        # Create TimeSeriesSplit
        tscv_kwargs = {'n_splits': self.n_folds}
        if self.min_train_size is not None:
            tscv_kwargs['test_size'] = max(1, (n_samples - self.min_train_size) // self.n_folds)
        
        tscv = TimeSeriesSplit(**tscv_kwargs)
        splits = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # Apply embargo: skip first N samples of validation set
            if embargo_size > 0 and len(val_idx) > embargo_size:
                original_val_size = len(val_idx)
                val_idx = val_idx[embargo_size:]
                
                logger.debug(
                    f"Fold {fold_idx + 1}: Applied embargo of {embargo_size} samples "
                    f"(reduced validation from {original_val_size} to {len(val_idx)})"
                )
            
            # Only include fold if validation set is not empty
            if len(val_idx) > 0:
                # Convert to Index objects
                train_index = X.index[train_idx]
                val_index = X.index[val_idx]
                splits.append((train_index, val_index))
            else:
                logger.warning(
                    f"Fold {fold_idx + 1}: Validation set empty after embargo, skipping"
                )
        
        if not splits:
            logger.error("No valid splits generated - all validation sets were empty")
        else:
            logger.info(f"Generated {len(splits)} CV splits with embargo")
        
        return splits
    
    def get_n_splits(self, X: Optional[pd.DataFrame] = None) -> int:
        """
        Get number of splits.
        
        Args:
            X: Optional DataFrame (for sklearn compatibility)
            
        Returns:
            Number of splits (may be less than n_folds if embargo removes some)
        """
        return self.n_folds


class PurgedCVSplitter(BaseCVSplitter):
    """
    Extended CV splitter with purging support.
    
    Purging removes samples from training set that are too close
    in time to the validation set, further preventing data leakage.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        embargo_pct: float = 0.1,
        purge_pct: float = 0.05,
        min_train_size: Optional[int] = None
    ):
        """
        Initialize purged CV splitter.
        
        Args:
            n_folds: Number of folds
            embargo_pct: Percentage of validation to skip (after validation)
            purge_pct: Percentage of training to remove (before validation)
            min_train_size: Minimum training size
        """
        super().__init__(n_folds, embargo_pct, min_train_size)
        self.purge_pct = purge_pct
        
        if not 0 <= purge_pct <= 0.3:
            raise ValueError("purge_pct must be between 0 and 0.3")
    
    def split_with_embargo(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Perform time series split with both purging and embargo.
        
        Args:
            X: Feature DataFrame
            y: Optional target Series
            
        Returns:
            List of (purged_train_index, embargoed_val_index) tuples
        """
        # Get base splits with embargo
        base_splits = super().split_with_embargo(X, y)
        
        if not base_splits:
            return []
        
        n_samples = len(X)
        purge_size = int(n_samples * self.purge_pct)
        
        purged_splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(base_splits):
            # Remove last N samples from training (purge before validation)
            if purge_size > 0 and len(train_idx) > purge_size:
                original_train_size = len(train_idx)
                train_idx = train_idx[:-purge_size]
                
                logger.debug(
                    f"Fold {fold_idx + 1}: Applied purge of {purge_size} samples "
                    f"(reduced training from {original_train_size} to {len(train_idx)})"
                )
            
            # Only include if training set is still sufficient
            if len(train_idx) > 0:
                purged_splits.append((train_idx, val_idx))
            else:
                logger.warning(
                    f"Fold {fold_idx + 1}: Training set empty after purge, skipping"
                )
        
        logger.info(
            f"Generated {len(purged_splits)} purged CV splits "
            f"(purge={self.purge_pct:.1%}, embargo={self.embargo_pct:.1%})"
        )
        
        return purged_splits
