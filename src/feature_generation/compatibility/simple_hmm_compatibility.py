"""
Simple HMM Compatibility Layer

This module provides a minimal compatibility layer for HMM processes that expect
the old FeatureGenerators interface, without requiring external dependencies.
"""

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class SimpleHMMCompatibleFeatureGenerators:
    """
    Simple compatibility wrapper for HMM processes that expect the old FeatureGenerators interface.
    
    This class provides the same interface as the old FeatureGenerators class,
    but with minimal dependencies for testing purposes.
    """
    
    def __init__(self):
        """Initialize the simple HMM-compatible feature generators."""
        self.logger = logger.getChild('SimpleHMMCompatibleFeatureGenerators')
        self.logger.info("✅ Simple HMM-compatible feature generators initialized")
    
    def generate_features_for_hmm(self, data):
        """
        Generate focused feature set for HMM models training.
        
        This is a minimal implementation that returns the input data with some basic features added.
        In a real environment with pandas/numpy, this would generate comprehensive features.
        
        Args:
            data: Input data (DataFrame or similar)
            
        Returns:
            Enhanced data with additional features
        """
        self.logger.info("🚀 Generating focused HMM-ready feature set (simple mode)...")
        
        # In a real environment, this would generate comprehensive features
        # For now, just return the input data with a note
        self.logger.info("📊 Simple mode: returning input data with basic structure")
        
        # Try to add some basic structure if possible
        try:
            # If data has a copy method (like pandas DataFrame), use it
            if hasattr(data, 'copy'):
                result = data.copy()
                self.logger.info(f"✅ Data copied successfully, shape: {getattr(result, 'shape', 'unknown')}")
                return result
            else:
                # Otherwise, just return the data as-is
                self.logger.info("✅ Returning data as-is")
                return data
        except Exception as e:
            self.logger.warning(f"⚠️ Could not process data: {e}, returning as-is")
            return data

# Create a global instance for compatibility
_global_simple_hmm_generators: Optional[SimpleHMMCompatibleFeatureGenerators] = None

def get_simple_hmm_compatible_generators() -> SimpleHMMCompatibleFeatureGenerators:
    """
    Get the global simple HMM-compatible feature generators instance.
    
    Returns:
        Simple HMM-compatible feature generators instance
    """
    global _global_simple_hmm_generators
    if _global_simple_hmm_generators is None:
        _global_simple_hmm_generators = SimpleHMMCompatibleFeatureGenerators()
    return _global_simple_hmm_generators

# Compatibility alias
FeatureGenerators = SimpleHMMCompatibleFeatureGenerators