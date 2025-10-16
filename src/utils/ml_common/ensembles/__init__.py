"""
ML Common - Ensembles Module

This module contains all ensemble-related functionality including:
- Ensemble managers
- Stacking ensemble implementations
- Confidence calibration
- Ensemble evaluation
"""

import logging
from typing import Dict, List, Any, Optional

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

# Global ensemble utilities instance
_ensemble_utils: Optional['EnsembleUtils'] = None

def get_ensemble_utils() -> 'EnsembleUtils':
    """Get or create the global ensemble utilities instance."""
    global _ensemble_utils

    if _ensemble_utils is None:
        _ensemble_utils = EnsembleUtils()
        logging.info("✅ Ensemble utilities initialized")

    return _ensemble_utils

class EnsembleUtils:
    """Ensemble utilities framework for ML common operations."""

    def __init__(self):
        """Initialize ensemble utilities."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_ensemble(self, models: List[Any], method: str = 'voting') -> Any:
        """Create ensemble using available ensemble managers."""
        try:
            if method == 'voting':
                return EnsembleManager._create_voting_ensemble_static(models)
            elif method == 'stacking':
                return EnsembleManager._create_stacking_ensemble_static(models)
            elif method == 'weighted_average':
                return EnsembleManager._create_weighted_average_ensemble_static(models)
            else:
                self.logger.warning(f"Unknown ensemble method: {method}, using voting")
                return EnsembleManager._create_voting_ensemble_static(models)
        except Exception as e:
            self.logger.error(f"Failed to create ensemble: {e}")
            return None

    def get_ensemble_metrics(self) -> Dict[str, Any]:
        """Get ensemble-related metrics."""
        return {
            'ensemble_enabled': True,
            'available_methods': ['voting', 'stacking', 'weighted_average'],
            'ensemble_types': ['classification', 'regression']
        }

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
    'create_analyst_calibrator', 'create_tactician_calibrator',

    # Ensemble Utilities
    'get_ensemble_utils', 'EnsembleUtils'
]
