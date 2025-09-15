"""
Feature Generators Compatibility Module

This module provides compatibility for code that imports FeatureGenerators from
src.feature_engineering.feature_generators, redirecting to the new unified
feature generation system.
"""

import logging
import warnings

logger = logging.getLogger(__name__)

# Issue deprecation warning
warnings.warn(
    "Importing FeatureGenerators from src.feature_engineering.feature_generators is deprecated. "
    "Please use: from src.feature_generation import FeatureGenerators",
    DeprecationWarning,
    stacklevel=2
)

try:
    # Try to import from the new unified feature generation system
    from ...feature_generation import FeatureGenerators as NewFeatureGenerators
    logger.info("✅ Successfully imported FeatureGenerators from new unified system")
    
    # Export the new class as the old name for compatibility
    FeatureGenerators = NewFeatureGenerators
    
except ImportError as e:
    logger.warning(f"⚠️ Failed to import from new system: {e}")
    
    # Try simple compatibility layer
    try:
        from ...feature_generation.compatibility.simple_hmm_compatibility import FeatureGenerators as SimpleFeatureGenerators
        logger.info("✅ Using simple HMM compatibility layer")
        FeatureGenerators = SimpleFeatureGenerators
        
    except ImportError as e2:
        logger.warning(f"⚠️ Simple compatibility layer also failed: {e2}")
        
        # Fallback to original implementation if available
        try:
            from .feature_generators import FeatureGenerators as OriginalFeatureGenerators
            logger.warning("⚠️ Using original FeatureGenerators as fallback")
            FeatureGenerators = OriginalFeatureGenerators
            
        except ImportError as e3:
            logger.error(f"❌ Original FeatureGenerators also not available: {e3}")
            
            # Create a minimal fallback class
            class FeatureGenerators:
                """Minimal fallback FeatureGenerators class."""
                
                def __init__(self):
                    self.logger = logger.getChild('FeatureGenerators')
                    self.logger.warning("⚠️ Using minimal fallback FeatureGenerators")
                
                def generate_features_for_hmm(self, data):
                    """Minimal fallback implementation."""
                    self.logger.info("📊 Minimal fallback: returning data as-is")
                    return data

# Export the class
__all__ = ['FeatureGenerators']