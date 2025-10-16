"""
Compatibility shim: temporal_cross_validation

Provides temporal_cross_validation function via the unified CV API and
exposes commonly referenced legacy names for compatibility.
"""

from typing import Any, Dict, List, Optional, Union

from .unified_cv import temporal_cross_validation as _temporal_cv
from .universal_temporal_validation import (
    UniversalTimeSeriesSplit as TimeSeriesSplit,
    UniversalTemporalCrossValidator as TemporalCrossValidator,
)

def temporal_cross_validation(
    model: Any,
    X: Any,
    y: Any,
    *,
    n_splits: int = 5,
    gap: int = 0,
    test_size: Optional[int] = None,
    scoring: Optional[Union[str, List[str]]] = None,
) -> Dict[str, Any]:
    return _temporal_cv(
        model,
        X,
        y,
        n_splits=n_splits,
        gap=gap,
        test_size=test_size,
        scoring=scoring,
    )

class TemporalValidationPipeline:  # Legacy placeholder for compatibility
    """Legacy temporal validation pipeline for backward compatibility."""

    def __init__(self, n_splits: int = 5, test_size: float = 0.2, gap: int = 0):
        """
        Initialize temporal validation pipeline.

        Args:
            n_splits: Number of splits for cross-validation
            test_size: Size of test set as proportion of total data
            gap: Gap between train and test sets to prevent data leakage
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.logger = logging.getLogger(__name__)

    def split(self, X, y=None, groups=None):
        """
        Generate train/test splits for temporal cross-validation.

        Args:
            X: Feature matrix
            y: Target vector (optional)
            groups: Group labels (optional)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        test_size_samples = int(n_samples * self.test_size)

        for i in range(self.n_splits):
            # Calculate split boundaries
            start_test = n_samples - test_size_samples - (i * test_size_samples // self.n_splits)
            end_test = start_test + test_size_samples

            # Ensure we don't go out of bounds
            start_test = max(0, start_test)
            end_test = min(n_samples, end_test)

            # Add gap if specified
            if self.gap > 0:
                start_test = max(0, start_test - self.gap)

            # Generate indices
            train_indices = list(range(0, start_test))
            test_indices = list(range(start_test, end_test))

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Get number of splits."""
        return self.n_splits

__all__ = [
    'temporal_cross_validation',
    'TemporalCrossValidator',
    'TimeSeriesSplit',
    'TemporalValidationPipeline',
]
