"""
Feature Generation Compatibility Package

This package provides compatibility layers for integrating with legacy systems
and external components that expect older interfaces.

Main Components:
- hmm_compatibility.py: HMM process compatibility (primary)
- legacy_adapter.py: Legacy feature generation compatibility (simplified)
"""

# Primary HMM compatibility (main interface)
from .hmm_compatibility import (
    HMMCompatibleFeatureGenerators,
    FeatureGenerators,  # Alias for backward compatibility
    get_hmm_compatible_generators
)

# Simplified legacy adapter (deprecated)
from .legacy_adapter import (
    LegacyFeatureAdapter,
    migrate_legacy_features,
    get_legacy_adapter,
    enable_legacy_compatibility
)

__all__ = [
    # HMM compatibility (primary)
    'HMMCompatibleFeatureGenerators',
    'FeatureGenerators',
    'get_hmm_compatible_generators',
    
    # Legacy adapter (deprecated)
    'LegacyFeatureAdapter',
    'migrate_legacy_features',
    'get_legacy_adapter', 
    'enable_legacy_compatibility'
]
