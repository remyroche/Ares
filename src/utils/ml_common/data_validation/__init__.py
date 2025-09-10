"""
Data Validation Module

This module provides comprehensive data quality verification for aggtrades and klines data,
with automatic integration into existing pipelines.

Key Features:
- Aggtrades data quality verification
- Klines data quality verification
- Unified quality verification orchestrator
- Pipeline integration hooks
- Automatic quality checks at data collection completion and stage beginnings

Modules:
- aggtrades_quality_verification: Aggtrades-specific quality verification
- klines_quality_verification: Klines-specific quality verification
- unified_quality_verification: Unified orchestrator for all data types
- pipeline_quality_integration: Pipeline integration hooks and decorators
"""

from .aggtrades_quality_verification import (
    AggtradesQualityVerifier,
    verify_aggtrades_quality,
    create_aggtrades_quality_config,
    QualityIssueSeverity,
    QualityAction,
    QualityIssue,
    QualityReport
)

from .klines_quality_verification import (
    KlinesQualityVerifier,
    verify_klines_quality,
    create_klines_quality_config
)

from .unified_quality_verification import (
    UnifiedQualityVerifier,
    DataType,
    VerificationStage,
    UnifiedQualityReport,
    create_unified_quality_verifier,
    verify_data_quality_unified,
    create_pipeline_quality_config
)

from .pipeline_quality_integration import (
    PipelineQualityIntegration,
    get_quality_integration,
    verify_data_collection_quality,
    verify_stage_beginning_quality,
    enforce_quality_gate
)

__all__ = [
    # Aggtrades quality verification
    'AggtradesQualityVerifier',
    'verify_aggtrades_quality',
    'create_aggtrades_quality_config',
    'QualityIssueSeverity',
    'QualityAction',
    'QualityIssue',
    'QualityReport',
    
    # Klines quality verification
    'KlinesQualityVerifier',
    'verify_klines_quality',
    'create_klines_quality_config',
    
    # Unified quality verification
    'UnifiedQualityVerifier',
    'DataType',
    'VerificationStage',
    'UnifiedQualityReport',
    'create_unified_quality_verifier',
    'verify_data_quality_unified',
    'create_pipeline_quality_config',
    
    # Pipeline integration
    'PipelineQualityIntegration',
    'get_quality_integration',
    'verify_data_collection_quality',
    'verify_stage_beginning_quality',
    'enforce_quality_gate'
]