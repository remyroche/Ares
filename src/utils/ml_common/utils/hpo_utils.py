"""
Hyperparameter Optimization Utilities (Canonical Re-exports)

This module re-exports the canonical implementation in
`src.utils.ml_common.optimization.hpo_utils`. New code should import from the
canonical module, but existing imports from this path will continue to work.
"""

import logging
from typing import Dict, Any, List, Optional, Callable

# Re-export canonical implementation for new code paths
from src.utils.ml_common.optimization.hpo_utils import (
    HyperparameterOptimization as CanonicalHyperparameterOptimization,
    optimize_hyperparameters,
    create_search_space,
    validate_hpo_config,
    create_hpo_config,
)

logger = logging.getLogger(__name__)

# Alias canonical class for external imports
HyperparameterOptimization = CanonicalHyperparameterOptimization

# Export key classes and functions
__all__ = [
    'HyperparameterOptimization',
    'optimize_hyperparameters',
    'create_search_space',
    'validate_hpo_config',
    'create_hpo_config',
]
