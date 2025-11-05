"""
Optimization Components for Statsmodels Clustering

This module provides optimization components for statsmodels regime switching models,
including parameter mapping from Pyro configurations.

Key Components:
- PyroToStatsmodelsMapper: Maps Pyro parameters to statsmodels format
"""

from .parameter_mapper import (
    PyroToStatsmodelsMapper,
    ParameterMappingConfig,
    ParameterMappingResult,
    map_pyro_to_statsmodels,
    map_pyro_search_space,
    create_default_mapping_config
)

__all__ = [
    'PyroToStatsmodelsMapper',
    'ParameterMappingConfig',
    'ParameterMappingResult',
    'map_pyro_to_statsmodels',
    'map_pyro_search_space',
    'create_default_mapping_config'
]