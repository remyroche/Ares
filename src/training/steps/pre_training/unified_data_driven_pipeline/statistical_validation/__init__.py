"""
Statistical Validation Framework for UnifiedDataDrivenPipeline

Provides comprehensive statistical validation to prevent overconfidence and
ensure robust feature selection and model validation.

Key Features:
- Deflated Sharpe ratio calculation
- SPA (Superior Predictive Ability) test
- White's Reality Check
- Model Confidence Set (MCS)
- Multiple testing correction
- Bootstrap validation
"""

from .deflated_sharpe import (
    DeflatedSharpeCalculator,
    DeflatedSharpeConfig,
    DeflatedSharpeResult
)

from .reality_check import (
    RealityCheckFramework,
    RealityCheckConfig,
    RealityCheckResult,
    RealityCheckType
)

from .model_confidence_set import (
    ModelConfidenceSet,
    MCSConfig,
    MCSResult
)

from .multiple_testing_correction import (
    MultipleTestingCorrection,
    MTCConfig,
    MTCResult,
    CorrectionMethod
)

from .bootstrap_validation import (
    BootstrapValidator,
    BootstrapConfig,
    BootstrapResult
)

__all__ = [
    # Deflated Sharpe
    'DeflatedSharpeCalculator',
    'DeflatedSharpeConfig',
    'DeflatedSharpeResult',
    
    # Reality check
    'RealityCheckFramework',
    'RealityCheckConfig',
    'RealityCheckResult',
    'RealityCheckType',
    
    # Model confidence set
    'ModelConfidenceSet',
    'MCSConfig',
    'MCSResult',
    
    # Multiple testing correction
    'MultipleTestingCorrection',
    'MTCConfig',
    'MTCResult',
    'CorrectionMethod',
    
    # Bootstrap validation
    'BootstrapValidator',
    'BootstrapConfig',
    'BootstrapResult'
]