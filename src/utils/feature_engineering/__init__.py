"""
Feature Engineering Utilities

This package provides comprehensive feature engineering utilities including:
- Enhanced feature engineering with 200+ features
- Matrix operations with GPU acceleration
- Feature generation optimization
- Advanced feature interactions and transformations
"""

from .step06_enhanced_feature_engineering import *
from .enhanced_matrix_operations import *

__all__ = [
    # Feature engineering
    'EnhancedFeatureEngineering',
    'FeatureEngineeringConfig',
    'FeatureEngineeringResult',
    
    # Matrix operations
    'EnhancedMatrixOperations',
    'MatrixOperationsConfig',
    'MatrixOperationsResult',
    'GPUError',
    'MemoryError',
    'OptimizationError',
]