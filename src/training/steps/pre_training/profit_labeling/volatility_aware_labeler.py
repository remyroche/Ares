"""
Volatility-Aware Labeler Module

This module provides compatibility functions for the feature generation labeling integration step.
It re-exports the create_enhanced_analyst_labeler function from the volatility_aware_profit_labeler module.
"""

# Import the create_enhanced_analyst_labeler function from the main volatility aware profit labeler
from .volatility_aware_profit_labeler import create_enhanced_analyst_labeler

# Re-export for compatibility
__all__ = ['create_enhanced_analyst_labeler']