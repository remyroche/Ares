"""
Core Module for Models Training ModularComponent Architecture

This module provides the core ModularComponent architecture specifically
designed for machine learning model training workflows. It includes
comprehensive functionality for configuration management, state management,
performance monitoring, and lifecycle management optimized for ML training scenarios.

Key Components:
- ModularComponent: Abstract base class for all training components
- Migration utilities for converting existing components
- ML-specific state management and performance tracking
- Comprehensive error handling and logging
"""

from .modular_architecture import (
    ModularComponent,
    ExampleModularComponent,
    ValidationLevel,
    ValidationResult,
    ErrorInfo,
    PerformanceMetric,
    MetricType,
    MetricLevel,
    ErrorSeverity,
    ErrorCategory,
    create_modular_component
)

from .migration_utils import (
    ModelsTrainingMigrationUtils,
    ComponentAnalysis,
    MigrationResult,
    analyze_component,
    validate_migration_compatibility,
    create_component_wrapper,
    migrate_component,
    generate_migration_report
)

__all__ = [
    # Core architecture
    'ModularComponent',
    'ExampleModularComponent',
    'ValidationLevel',
    'ValidationResult',
    'ErrorInfo',
    'PerformanceMetric',
    'MetricType',
    'MetricLevel',
    'ErrorSeverity',
    'ErrorCategory',
    'create_modular_component',
    
    # Migration utilities
    'ModelsTrainingMigrationUtils',
    'ComponentAnalysis',
    'MigrationResult',
    'analyze_component',
    'validate_migration_compatibility',
    'create_component_wrapper',
    'migrate_component',
    'generate_migration_report'
]

__version__ = "1.0.0"
__author__ = "Models Training Team"
__description__ = "ModularComponent architecture for models training workflows"