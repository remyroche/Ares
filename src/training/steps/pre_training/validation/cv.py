"""
Cross-validation utilities for pre-training pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, KFold
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WalkForwardFold:
    """Walk-forward validation fold."""
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    fold_index: int
    
    def __post_init__(self):
        """Validate fold parameters."""
        if self.train_start >= self.train_end:
            raise ValueError("train_start must be less than train_end")
        if self.test_start >= self.test_end:
            raise ValueError("test_start must be less than test_end")
        if self.train_end > self.test_start:
            raise ValueError("train_end must be less than or equal to test_start")

def create_time_series_cv(n_splits: int = 5, test_size: Optional[float] = None) -> TimeSeriesSplit:
    """Create time series cross-validation splitter."""
    return TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

def create_kfold_cv(n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = None) -> KFold:
    """Create K-fold cross-validation splitter."""
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

def validate_cv_splits(X: np.ndarray, y: np.ndarray, cv_splitter: Any) -> bool:
    """Validate cross-validation splits."""
    try:
        splits = list(cv_splitter.split(X, y))
        
        # Check if we have the expected number of splits
        expected_splits = getattr(cv_splitter, 'n_splits', len(splits))
        if len(splits) != expected_splits:
            logger.warning(f"Expected {expected_splits} splits, got {len(splits)}")
            return False
        
        # Check each split
        for i, (train_idx, test_idx) in enumerate(splits):
            if len(train_idx) == 0 or len(test_idx) == 0:
                logger.warning(f"Split {i} has empty train or test set")
                return False
            
            if len(set(train_idx) & set(test_idx)) > 0:
                logger.warning(f"Split {i} has overlapping train and test indices")
                return False
        
        logger.info(f"CV validation passed with {len(splits)} splits")
        return True
        
    except Exception as e:
        logger.error(f"Error validating CV splits: {e}")
        return False

def get_cv_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate cross-validation metrics."""
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating CV metrics: {e}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }

def perform_cv_validation(X: np.ndarray, y: np.ndarray, model: Any, cv_splitter: Any) -> Dict[str, Any]:
    """Perform cross-validation validation."""
    try:
        cv_scores = []
        cv_metrics = []
        
        for train_idx, test_idx in cv_splitter.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = get_cv_metrics(y_test, y_pred)
            cv_metrics.append(metrics)
            
            # Store score (using accuracy as default)
            cv_scores.append(metrics['accuracy'])
        
        results = {
            'cv_scores': cv_scores,
            'cv_metrics': cv_metrics,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'n_splits': len(cv_scores)
        }
        
        logger.info(f"CV validation completed: mean_score={results['mean_score']:.4f}, std_score={results['std_score']:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"Error in CV validation: {e}")
        return {
            'cv_scores': [],
            'cv_metrics': [],
            'mean_score': 0.0,
            'std_score': 0.0,
            'n_splits': 0,
            'error': str(e)
        }


def purged_walk_forward_cv(n_splits: int = 5, purge_days: int = 1, embargo_days: int = 1) -> List[WalkForwardFold]:
    """Create purged walk-forward cross-validation folds."""
    try:
        folds = []
        
        # Calculate fold sizes (this is a simplified implementation)
        # In practice, you would use actual data timestamps
        for i in range(n_splits):
            # Calculate fold boundaries
            train_start = i * 100  # Simplified calculation
            train_end = (i + 1) * 100
            test_start = train_end + purge_days
            test_end = test_start + 20  # Simplified test size
            
            fold = WalkForwardFold(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                fold_index=i
            )
            folds.append(fold)
        
        logger.info(f"Created {len(folds)} purged walk-forward CV folds")
        return folds
        
    except Exception as e:
        logger.error(f"Error creating purged walk-forward CV: {e}")
        return []

def validate_cv_no_leakage(cv_splitter: Any, X: np.ndarray, y: np.ndarray) -> bool:
    """Validate that cross-validation splits don't have data leakage."""
    try:
        if cv_splitter is None:
            logger.warning("CV splitter is None, cannot validate leakage")
            return False
        
        # Check if it's a time series splitter
        if hasattr(cv_splitter, 'split'):
            for train_idx, test_idx in cv_splitter.split(X):
                # Check that test indices come after train indices
                if len(train_idx) > 0 and len(test_idx) > 0:
                    if max(train_idx) >= min(test_idx):
                        logger.warning("Data leakage detected: test data overlaps with training data")
                        return False
        
        return True
    except Exception as e:
        logger.warning(f"Error validating CV leakage: {e}")
        return False
