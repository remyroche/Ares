"""
Diagnostics utilities for ML model evaluation.

This module provides tools for assessing model signal-to-noise ratios,
statistical significance, and predictive performance.
"""

from .snr_diagnostics import (
    SNRDiagnostics,
    compute_snr_metrics,
    bootstrap_r2,
    permutation_test,
    cross_val_predictions,
)

__all__ = [
    'SNRDiagnostics',
    'compute_snr_metrics',
    'bootstrap_r2',
    'permutation_test',
    'cross_val_predictions',
]
