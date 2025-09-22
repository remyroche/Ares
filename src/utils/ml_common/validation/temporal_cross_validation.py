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
    pass


__all__ = [
    'temporal_cross_validation',
    'TemporalCrossValidator',
    'TimeSeriesSplit',
    'TemporalValidationPipeline',
]

