"""
Multi-Horizon Profit Probability Labeling - Backward Compatibility Layer

This module provides backward compatibility by re-exporting the enhanced multi-horizon
profit labeler. The enhanced version is a drop-in replacement with additional features.

The enhanced version provides:
- All original functionality
- Enhanced data cleaning and quality assessment
- Trading-aware label definitions (Analyst & Tactician)
- Label stability monitoring and leakage detection
- Full backward compatibility

This file maintains compatibility with existing imports while using the enhanced implementation.
"""

# Re-export everything from the enhanced version
from .enhanced_multi_horizon_labeler import (
    # Main classes
    EnhancedMultiHorizonProfitLabeler as MultiHorizonProfitLabeler,
    EnhancedMultiHorizonConfig as MultiHorizonConfig,
    
    # Result types
    LabelingResult,
    
    # Convenience functions
    create_enhanced_multi_horizon_labeler as create_multi_horizon_labeler,
    create_trading_optimized_multi_horizon_config as create_trading_optimized_config,
    create_research_optimized_multi_horizon_config as create_research_optimized_config,
)

# Maintain backward compatibility
__all__ = [
    'MultiHorizonProfitLabeler',
    'MultiHorizonConfig', 
    'LabelingResult',
    'create_multi_horizon_labeler',
    'create_trading_optimized_config',
    'create_research_optimized_config',
]

# Version info
__version__ = "2.0.0"
__author__ = "Ares Trading System"