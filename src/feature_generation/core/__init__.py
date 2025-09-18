"""
Core Feature Generation Framework

This module provides the core framework for the unified feature generation system,
including the main classes and interfaces for feature generation, registration,
and management.
"""

from .feature_bank import FeatureBank
from .feature_generator import FeatureGenerator, FeatureCategory
from .feature_registry import FeatureRegistry
from .factory import (
    get_feature_generator,
    get_feature_bank,
    register_feature_generator,
    list_available_features,
    list_available_categories
)

def _initialize_default_bank():
    """Initialize the default feature bank with standard generators."""
    try:
        bank = get_feature_bank()
        # Basic initialization - the bank will auto-register available generators
        return bank
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to initialize default bank: {e}")
        return None

__all__ = [
    "FeatureBank",
    "FeatureGenerator", 
    "FeatureCategory",
    "FeatureRegistry",
    "get_feature_generator",
    "get_feature_bank",
    "register_feature_generator",
    "list_available_features",
    "list_available_categories",
    "_initialize_default_bank"
]