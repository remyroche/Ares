"""
Step06 Enhanced Validation Framework

This package provides comprehensive validation, tracking, and reporting
for all step06 components including:
- Function call validation and logging
- Function-to-function call tracking
- Function completion reports with detailed outcomes
- Performance monitoring and analysis
- Error handling with detailed context
"""

__version__ = "1.0.0"
__author__ = "Step06 Validation Framework"

# Import main validation framework components
try:
    from .step06_enhanced_validation_framework import (  # type: ignore[import-untyped]
        step06_function_validator,
        step06_function_tracker,
        step06_validation_context,
        get_step06_validation_summary,
        reset_step06_validation_tracking,
        ValidationLevel,
        FunctionStatus,
        FunctionCallContext,
        FunctionCallReport,
        Step06Validator,
        Step06Reporter
    )
    VALIDATION_FRAMEWORK_AVAILABLE = True
except ImportError:
    VALIDATION_FRAMEWORK_AVAILABLE = False

# Import validation orchestrator (resilient to any import issues)
try:
    from .step06_validation_orchestrator import (
        Step06ValidationOrchestrator,
        run_step06_comprehensive_validation
    )
    ORCHESTRATOR_AVAILABLE = True
except Exception:
    ORCHESTRATOR_AVAILABLE = False

__all__ = [
    'step06_function_validator',
    'step06_function_tracker',
    'step06_validation_context',
    'get_step06_validation_summary',
    'reset_step06_validation_tracking',
    'ValidationLevel',
    'FunctionStatus',
    'FunctionCallContext',
    'FunctionCallReport',
    'Step06Validator',
    'Step06Reporter',
    'Step06ValidationOrchestrator',
    'run_step06_comprehensive_validation',
    'VALIDATION_FRAMEWORK_AVAILABLE',
    'ORCHESTRATOR_AVAILABLE'
]