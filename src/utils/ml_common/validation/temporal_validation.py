"""
Compatibility shim: temporal_validation

Re-exports universal temporal validation components under legacy names.
"""

from .universal_temporal_validation import (
    TemporalValidationConfig,
    UniversalTemporalValidator as TemporalValidator,
    UniversalTemporalCrossValidator as TemporalCrossValidator,
    UniversalTimeSeriesSplit as WalkForwardValidator,
)

__all__ = [
    'TemporalValidationConfig',
    'TemporalValidator',
    'TemporalCrossValidator',
    'WalkForwardValidator',
]
