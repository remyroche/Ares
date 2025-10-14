"""
Leakage Prevention Framework for UnifiedDataDrivenPipeline

Provides comprehensive leakage prevention and validation to ensure no future
information is used in feature engineering or model training.

Key Features:
- Label construction validation
- HTF alignment verification
- Future data detection
- Temporal integrity checks
"""

from .label_construction import (
    LabelConstructionValidator,
    LabelConstructionConfig,
    LabelConstructionResult,
    LabelType
)

from .htf_alignment import (
    HTFAlignmentValidator,
    HTFAlignmentConfig,
    HTFAlignmentResult,
    HTFAlignmentError
)

from .temporal_integrity import (
    TemporalIntegrityChecker,
    TemporalIntegrityConfig,
    TemporalIntegrityResult,
    TemporalViolation
)

from .leakage_detector import (
    ComprehensiveLeakageDetector,
    LeakageDetectionConfig,
    LeakageDetectionResult,
    LeakageType
)

__all__ = [
    # Label construction
    'LabelConstructionValidator',
    'LabelConstructionConfig',
    'LabelConstructionResult',
    'LabelType',
    
    # HTF alignment
    'HTFAlignmentValidator',
    'HTFAlignmentConfig',
    'HTFAlignmentResult',
    'HTFAlignmentError',
    
    # Temporal integrity
    'TemporalIntegrityChecker',
    'TemporalIntegrityConfig',
    'TemporalIntegrityResult',
    'TemporalViolation',
    
    # Leakage detection
    'ComprehensiveLeakageDetector',
    'LeakageDetectionConfig',
    'LeakageDetectionResult',
    'LeakageType'
]