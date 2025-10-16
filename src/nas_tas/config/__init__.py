"""
NAS/TAS Shared Configuration Utilities

This module provides unified configuration management for both NAS and TAS systems,
eliminating redundancy and ensuring consistency across architecture search implementations.
"""

from .base_config import (
    UnifiedArchitectureConfig,
    ArchitectureType,
    OptimizationMode,
    SearchStrategy,
    ValidationMethod
)

from .search_config import (
    SearchConfig,
    NASSearchConfig,
    TASSearchConfig,
    BayesianSearchConfig,
    EvolutionarySearchConfig
)

from .validation_config import (
    ValidationConfig,
    FinancialValidationConfig,
    PerformanceValidationConfig,
    RegimeValidationConfig
)

__all__ = [
    # Base configuration
    'UnifiedArchitectureConfig',
    'ArchitectureType',
    'OptimizationMode',
    'SearchStrategy',
    'ValidationMethod',

    # Search configuration
    'SearchConfig',
    'NASSearchConfig',
    'TASSearchConfig',
    'BayesianSearchConfig',
    'EvolutionarySearchConfig',

    # Validation configuration
    'ValidationConfig',
    'FinancialValidationConfig',
    'PerformanceValidationConfig',
    'RegimeValidationConfig'
]
