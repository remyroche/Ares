"""
Time Series Cross-Validation Module

Provides leakage-free cross-validation for time series data using
Purged & Embargoed Walk-Forward CV methodology.
"""

from .purged_embargoed_cv import (
    PurgedEmbargoedWalkForwardCV,
    PurgedEmbargoedConfig,
    TimeSeriesSplit,
    TimeSeriesSplitIterator,
    LeakagePreventionUtils,
    create_purged_embargoed_cv,
    validate_time_series_splits
)

__all__ = [
    'PurgedEmbargoedWalkForwardCV',
    'PurgedEmbargoedConfig', 
    'TimeSeriesSplit',
    'TimeSeriesSplitIterator',
    'LeakagePreventionUtils',
    'create_purged_embargoed_cv',
    'validate_time_series_splits'
]