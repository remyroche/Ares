"""
Layer 3 Modular Architecture

This module provides a modular, maintainable architecture for Layer 3 meta-modeling,
including geometry systems, feature engineering, model training, and reporting.

Main Components:
- Core orchestration and pipeline management
- Geometry generation and selection systems
- Feature engineering and optimization
- Dual-head model training
- Enhanced weighting systems
- Comprehensive reporting utilities
"""

from .core import layer3_analyst_lgbm
from .geometry_system import generate_geometries_adaptive, select_best_geometries_adaptive
from .feature_engineering import enhance_layer3_features_optimized
from .model_training import train_dual_head_models
from .weighting_system import create_enhanced_weighting_schemes
from .reporting import generate_layer3_reports
from .feature_registry import get_layer3_feature_patterns, validate_layer3_features

__all__ = [
    'layer3_analyst_lgbm',
    'generate_geometries_adaptive',
    'select_best_geometries_adaptive', 
    'enhance_layer3_features_optimized',
    'train_dual_head_models',
    'create_enhanced_weighting_schemes',
    'generate_layer3_reports'
]
