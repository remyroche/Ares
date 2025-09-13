"""
ML Common - Ensembles Module

This module contains all ensemble-related functionality including:
- Ensemble managers
- Stacking ensemble implementations
- Confidence calibration
- Ensemble evaluation
"""

from .ensemble_manager import EnsembleManager, EnsembleType, EnsembleConfig
# Ensemble classes are inner classes of EnsembleManager, not standalone
# from .ensemble_manager import (
#     VotingEnsemble, StackingEnsemble, BlendingEnsemble,
#     WeightedAverageEnsemble, DynamicWeightingEnsemble
# )
from .stacking_ensemble_manager import (
    StackingEnsembleManager, StackingEnsembleConfig, StackingEnsembleResult,
    create_analyst_ensemble, create_tactician_ensemble
)
from .stacking_confidence_calibration import (
    StackingConfidenceCalibrator, StackingCalibrationConfig, StackingCalibrationResult,
    create_analyst_calibrator, create_tactician_calibrator
)

__all__ = [
    # Ensemble Manager
    'EnsembleManager', 'EnsembleType', 'EnsembleConfig',
    
    # Basic Ensembles (available through EnsembleManager)
    # 'VotingEnsemble', 'StackingEnsemble', 'BlendingEnsemble',
    # 'WeightedAverageEnsemble', 'DynamicWeightingEnsemble',
    
    # Stacking Ensemble Manager
    'StackingEnsembleManager', 'StackingEnsembleConfig', 'StackingEnsembleResult',
    'create_analyst_ensemble', 'create_tactician_ensemble',
    
    # Stacking Confidence Calibration
    'StackingConfidenceCalibrator', 'StackingCalibrationConfig', 'StackingCalibrationResult',
    'create_analyst_calibrator', 'create_tactician_calibrator'
]