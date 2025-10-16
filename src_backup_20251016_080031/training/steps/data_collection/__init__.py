"""
Data Collection Step06 Components

This package contains data collection components for step06 including:
- Feature engineering step
- Data preprocessing
- Feature selection
- Data validation
"""

try:
    from .feature_generation.utils.step06_feature_engineering import FeatureEngineeringStep
    FEATURE_ENGINEERING_STEP_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_STEP_AVAILABLE = False

# Import comprehensive monitoring modules
try:
    from .step01_enhanced_with_monitoring import (
        EnhancedDataCollectionStepWithMonitoring,
        run_enhanced_step01_with_monitoring
    )
    ENHANCED_MONITORING_AVAILABLE = True
except ImportError:
    ENHANCED_MONITORING_AVAILABLE = False

try:
    from .step01_comprehensive_monitoring import (
        Step01ComprehensiveMonitoring,
        run_comprehensive_step01
    )
    COMPREHENSIVE_MONITORING_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_MONITORING_AVAILABLE = False

# Build __all__ list dynamically based on available modules
__all__ = []

# Add feature engineering if available
if FEATURE_ENGINEERING_STEP_AVAILABLE:
    __all__.extend(['FeatureEngineeringStep'])

# Add enhanced monitoring if available
if ENHANCED_MONITORING_AVAILABLE:
    __all__.extend([
        'EnhancedDataCollectionStepWithMonitoring',
        'run_enhanced_step01_with_monitoring'
    ])

# Add comprehensive monitoring if available
if COMPREHENSIVE_MONITORING_AVAILABLE:
    __all__.extend([
        'Step01ComprehensiveMonitoring',
        'run_comprehensive_step01'
    ])