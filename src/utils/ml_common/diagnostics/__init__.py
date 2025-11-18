"""
Diagnostics utilities for ML model evaluation.

This module provides comprehensive tools for assessing model performance,
signal quality, and uncertainty. Includes all 5 phases of the SNR diagnostic framework.

Phases:
- Phase 1: Core diagnostics (SNR, R², bootstrap CI, permutation tests)
- Phase 2: Signal-vs-noise attribution (model sweep, ablation, synthetic injection)
- Phase 3: Noise ceiling (ICC, Krippendorff's alpha, replicate analysis)
- Phase 4: Uncertainty decomposition (aleatoric vs epistemic)
- Phase 5: Subgroup & temporal diagnostics
"""

# Phase 1: Core diagnostics
from .snr_diagnostics import (
    SNRDiagnostics,
    SNRMetrics,
    compute_snr_metrics,
    bootstrap_r2,
    permutation_test,
    cross_val_predictions,
)

# Phase 2: Signal attribution
from .phase2_attribution import (
    SignalAttributionExperiments,
    AttributionResults,
)

# Phase 3: Noise ceiling
from .phase3_noise_ceiling import (
    NoiseCeilingAnalysis,
    NoiseCeilingResults,
)

# Phase 4: Uncertainty decomposition
from .phase4_uncertainty import (
    UncertaintyDecomposition,
    UncertaintyResults,
    HeteroscedasticModel,
    MCDropoutModel,
)

# Phase 5: Subgroup & temporal
from .phase5_subgroup_temporal import (
    SubgroupDiagnostics,
    TemporalDiagnostics,
    SubgroupResults,
    TemporalResults,
)

__all__ = [
    # Phase 1
    'SNRDiagnostics',
    'SNRMetrics',
    'compute_snr_metrics',
    'bootstrap_r2',
    'permutation_test',
    'cross_val_predictions',
    # Phase 2
    'SignalAttributionExperiments',
    'AttributionResults',
    # Phase 3
    'NoiseCeilingAnalysis',
    'NoiseCeilingResults',
    # Phase 4
    'UncertaintyDecomposition',
    'UncertaintyResults',
    'HeteroscedasticModel',
    'MCDropoutModel',
    # Phase 5
    'SubgroupDiagnostics',
    'TemporalDiagnostics',
    'SubgroupResults',
    'TemporalResults',
]
