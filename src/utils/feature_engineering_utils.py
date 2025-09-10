"""
Feature Engineering Utilities

This module provides comprehensive feature engineering utilities that were previously
part of step06. These utilities can be used by any step in the pipeline that needs
advanced feature engineering capabilities.

Features include:
- Technical indicator extraction
- Feature interaction creation
- Temporal validation
- Memory-efficient processing
- Mathematical safety utilities

Note: This module now uses the restored step06 functionality from utils.
"""

# Import the restored step06 utilities
from .step06_enhanced_feature_engineering import EnhancedFeatureEngineering

# Re-export for backward compatibility
FeatureEngineeringUtils = EnhancedFeatureEngineering

# Convenience functions for easy access
def create_feature_engineering_utils(config=None):
    """Create a new instance of FeatureEngineeringUtils."""
    return EnhancedFeatureEngineering(config)

def extract_technical_indicators(market_data, periods_config, config=None):
    """Convenience function to extract technical indicators."""
    utils = EnhancedFeatureEngineering(config)
    return utils.extract_indicators_batch(market_data, periods_config)

def create_feature_interactions(features, current_idx=None, config=None):
    """Convenience function to create feature interactions."""
    utils = EnhancedFeatureEngineering(config)
    return utils.create_sophisticated_interactions(features, current_idx)