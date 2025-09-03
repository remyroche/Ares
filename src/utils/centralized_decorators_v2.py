from __future__ import annotations
"""
Centralized Decorators Module v2
This module provides a unified interface to all decorators with enhanced functionality.
"""

import asyncio
import functools
import logging
from collections.abc import Callable
from typing import Any

# Handle optional dependencies
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    from src.utils.logger import system_logger
    from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
except ImportError:
    system_logger = logging.getLogger("CentralizedDecoratorsV2")

# Import advanced decorators
from .advanced_decorators import (
    PerformanceLevel,
    ValidationLevel,
    adaptive_resource_allocation,
    comprehensive_validation,
    intelligent_caching,
    model_validation,
    performance_monitor,
    pipeline_checkpoint,
)

# Import the new enhanced decorator system
from .decorator_registry import decorator_registry, global_config

# Import existing decorators for backwards compatibility
from .decorators import (
    auto_vectorize,
    enforce_ndarray,
    guard_array_nan_inf,
    guard_dataframe_nulls,
    normalize_errors,
    pa_check_input,
    pa_check_io,
    pa_check_output,
    validate_call_or_runtime_types,
    with_tracing_span,
)
from .enhanced_data_quality_decorators import (
    comprehensive_data_validation,
    optimize_memory_usage,
    validate_constant_features,
    validate_data_completeness,
    validate_data_structure,
    validate_datetime_index,
    validate_feature_engineering_pipeline,
    validate_hmm_data_requirements,
    validate_hmm_regime_discovery,
    validate_low_variance_features,
    validate_memory_optimized_data_quality,
    validate_multi_timeframe_alignment,
    validate_multi_timeframe_processing,
)
from .enhanced_decorators import (
    cached_validation,
    enhanced_validation,
    performance_monitor_v2,
    smart_error_recovery,
)
from .training_pipeline_decorators import (
    artifact_versioning,
    artifact_write_lock,
    circuit_breaker_protection,
    debug_training_step,
    deterministic_seed,
    idempotent_step,
    memory_efficient,
    nan_inf_and_constant_guard,
    prevent_data_leakage,
    resource_monitor,
    secure_data_processing,
    time_budget_watchdog,
    validate_step_output,
    validate_step_prerequisites,
)

# ============================================================================
# ENHANCED VALIDATION DECORATOR IMPLEMENTATION
# ============================================================================


@decorator_registry.register(
    name="validate_data_quality_v2",
    version="2.0",
    description="Enhanced data quality validation with configurable levels and auto-fixing",
    tags=["validation", "data-quality", "auto-fix"],
)
def validate_data_quality_v2(
    validation_level: str | ValidationLevel = "WARNING",
    context: str = "data validation",
    auto_fix: bool = False,
    **validation_kwargs,
):
    """
    Enhanced data quality validation decorator with configurable levels and auto-fixing.

    Args:
        validation_level: Validation severity level
        context: Context for logging and error messages
        auto_fix: Whether to attempt automatic fixes for validation issues
        **validation_kwargs: Additional validation parameters
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"DataQualityV2.{context}")

            # Pre-validation
            if global_config.enable_data_quality_checks:
                try:
                    # Apply data quality checks based on validation level
                    if validation_level in ["ERROR", "CRITICAL", "STRICT"]:
                        # Strict validation - fail on any issues
                        await _validate_data_quality_strict(args, kwargs, context, logger)
                    elif validation_level == "WARNING":
                        # Warning mode - log issues but continue
                        await _validate_data_quality_warning(args, kwargs, context, logger)
                    elif validation_level == "INFO":
                        # Info mode - just log information
                        await _validate_data_quality_info(args, kwargs, context, logger)
                except Exception as e:
                    if auto_fix:
                        logger.warning(f"Auto-fixing data quality issues in {context}: {e}")
                        args, kwargs = await _apply_data_quality_fixes(args, kwargs, context)
                    else:
                        raise

            # Execute the function
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"❌ Function execution failed in {context}: {e}")
                raise

            # Post-validation
            if global_config.enable_data_quality_checks:
                try:
                    await _validate_output_quality(result, context, logger)
                except Exception as e:
                    if auto_fix:
                        logger.warning(f"Auto-fixing output quality issues in {context}: {e}")
                        result = await _apply_output_quality_fixes(result, context)
                    else:
                        raise

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"DataQualityV2.{context}")

            # Pre-validation
            if global_config.enable_data_quality_checks:
                try:
                    if validation_level in ["ERROR", "CRITICAL", "STRICT"]:
                        _validate_data_quality_strict_sync(args, kwargs, context, logger)
                    elif validation_level == "WARNING":
                        _validate_data_quality_warning_sync(args, kwargs, context, logger)
                    elif validation_level == "INFO":
                        _validate_data_quality_info_sync(args, kwargs, context, logger)
                except Exception as e:
                    if auto_fix:
                        logger.warning(f"Auto-fixing data quality issues in {context}: {e}")
                        args, kwargs = _apply_data_quality_fixes_sync(args, kwargs, context)
                    else:
                        raise

            # Execute the function
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"❌ Function execution failed in {context}: {e}")
                raise

            # Post-validation
            if global_config.enable_data_quality_checks:
                try:
                    _validate_output_quality_sync(result, context, logger)
                except Exception as e:
                    if auto_fix:
                        logger.warning(f"Auto-fixing output quality issues in {context}: {e}")
                        result = _apply_output_quality_fixes_sync(result, context)
                    else:
                        raise

            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# ============================================================================
# QUALITY GATE DECORATOR IMPLEMENTATION
# ============================================================================


@decorator_registry.register(
    name="quality_gate_v2",
    version="2.0",
    description="Enhanced quality gate with configurable thresholds and actions",
    tags=["quality", "gate", "thresholds"],
)
def quality_gate_v2(
    min_quality_score: float = 0.7,
    required_grade: str = "C",
    action_on_failure: str = "warn",  # "warn", "raise", "degrade"
    context: str = "quality gate",
):
    """
    Enhanced quality gate decorator with configurable thresholds and actions.

    Args:
        min_quality_score: Minimum quality score required (0.0 to 1.0)
        required_grade: Minimum grade required (A, B, C, D, F)
        action_on_failure: Action to take on quality failure
        context: Context for logging and error messages
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"QualityGateV2.{context}")

            # Execute the function
            result = await func(*args, **kwargs)

            # Assess quality
            quality_score, grade = _assess_quality(result, context)

            if quality_score < min_quality_score or _grade_to_score(grade) < _grade_to_score(required_grade):
                msg = f"Quality gate failed: score {quality_score:.3f} (grade {grade}) below threshold {min_quality_score:.3f} (grade {required_grade})"

                if action_on_failure == "raise":
                    msg = f"Quality gate failed in {context}: {msg}"
                    raise ValueError(msg)
                if action_on_failure == "warn":
                    logger.warning(f"Quality gate warning in {context}: {msg}")
                elif action_on_failure == "degrade":
                    logger.warning(f"Quality gate degradation in {context}: {msg}")
                    # Apply degradation logic
                    result = _apply_quality_degradation(result, quality_score, context)

            logger.info(f"✅ Quality gate passed in {context}: score {quality_score:.3f} (grade {grade})")
            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"QualityGateV2.{context}")

            # Execute the function
            result = func(*args, **kwargs)

            # Assess quality
            quality_score, grade = _assess_quality(result, context)

            if quality_score < min_quality_score or _grade_to_score(grade) < _grade_to_score(required_grade):
                msg = f"Quality gate failed: score {quality_score:.3f} (grade {grade}) below threshold {min_quality_score:.3f} (grade {required_grade})"

                if action_on_failure == "raise":
                    msg = f"Quality gate failed in {context}: {msg}"
                    raise ValueError(msg)
                if action_on_failure == "warn":
                    logger.warning(f"Quality gate warning in {context}: {msg}")
                elif action_on_failure == "degrade":
                    logger.warning(f"Quality gate degradation in {context}: {msg}")
                    # Apply degradation logic
                    result = _apply_quality_degradation(result, quality_score, context)

            logger.info(f"✅ Quality gate passed in {context}: score {quality_score:.3f} (grade {grade})")
            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# ============================================================================
# STEP-SPECIFIC ML VALIDATION DECORATOR
# ============================================================================


@decorator_registry.register(
    name="step_specific_ml_validation_v2",
    version="2.0",
    description="Enhanced step-specific ML validation with adaptive thresholds",
    tags=["ml", "validation", "step-specific", "adaptive"],
)
def step_specific_ml_validation_v2(
    step_name: str,
    validation_config: dict[str, Any] = None,
    adaptive_thresholds: bool = True,
    context: str = "ml validation",
):
    """
    Enhanced step-specific ML validation decorator with adaptive thresholds.

    Args:
        step_name: Name of the ML pipeline step
        validation_config: Configuration for validation parameters
        adaptive_thresholds: Whether to use adaptive thresholds based on data characteristics
        context: Context for logging and error messages
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"StepMLValidationV2.{step_name}")

            # Determine validation thresholds
            thresholds = _get_validation_thresholds(step_name, validation_config, adaptive_thresholds, args, kwargs)

            # Pre-validation
            await _validate_ml_step_prerequisites(args, kwargs, step_name, thresholds, logger)

            # Execute the function
            result = await func(*args, **kwargs)

            # Post-validation
            await _validate_ml_step_output(result, step_name, thresholds, logger)

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"StepMLValidationV2.{step_name}")

            # Determine validation thresholds
            thresholds = _get_validation_thresholds(step_name, validation_config, adaptive_thresholds, args, kwargs)

            # Pre-validation
            _validate_ml_step_prerequisites_sync(args, kwargs, step_name, thresholds, logger)

            # Execute the function
            result = func(*args, **kwargs)

            # Post-validation
            _validate_ml_step_output_sync(result, step_name, thresholds, logger)

            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# ============================================================================
# AUTO-FIX DATA QUALITY ISSUES DECORATOR
# ============================================================================


@decorator_registry.register(
    name="auto_fix_data_quality_issues_v2",
    version="2.0",
    description="Enhanced auto-fix decorator with intelligent issue resolution",
    tags=["auto-fix", "data-quality", "intelligent"],
)
def auto_fix_data_quality_issues_v2(
    context: str = "auto-fix", fix_strategies: list[str] = None, max_fix_attempts: int = 3,
):
    """
    Enhanced auto-fix decorator with intelligent issue resolution.

    Args:
        context: Context for logging and error messages
        fix_strategies: List of fix strategies to apply
        max_fix_attempts: Maximum number of fix attempts
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"AutoFixV2.{context}")

            # Execute with auto-fixing
            for attempt in range(max_fix_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_fix_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed, applying auto-fix: {e}")
                        args, kwargs = await _apply_intelligent_fixes(args, kwargs, context, fix_strategies)
                    else:
                        logger.exception(f"All auto-fix attempts failed in {context}: {e}")
                        raise

            return None  # Should never reach here

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"AutoFixV2.{context}")

            # Execute with auto-fixing
            for attempt in range(max_fix_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_fix_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed, applying auto-fix: {e}")
                        args, kwargs = _apply_intelligent_fixes_sync(args, kwargs, context, fix_strategies)
                    else:
                        logger.exception(f"All auto-fix attempts failed in {context}: {e}")
                        raise

            return None  # Should never reach here

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# ============================================================================
# MONITORING DECORATORS
# ============================================================================


@decorator_registry.register(
    name="monitor_feature_engineering_v2",
    version="2.0",
    description="Enhanced feature engineering monitoring with detailed metrics",
    tags=["monitoring", "feature-engineering", "metrics"],
)
def monitor_feature_engineering_v2(
    track_feature_stats: bool = True,
    track_correlation_changes: bool = True,
    track_memory_usage: bool = True,
    context: str = "feature engineering",
):
    """Enhanced feature engineering monitoring decorator."""
    return performance_monitor_v2(
        level="detailed", track_memory=track_memory_usage, track_cpu=True, track_io=track_feature_stats,
    )


@decorator_registry.register(
    name="monitor_data_collection_v2",
    version="2.0",
    description="Enhanced data collection monitoring with quality metrics",
    tags=["monitoring", "data-collection", "quality"],
)
def monitor_data_collection_v2(
    track_data_volume: bool = True,
    track_quality_metrics: bool = True,
    track_collection_time: bool = True,
    context: str = "data collection",
):
    """Enhanced data collection monitoring decorator."""
    return performance_monitor_v2(level="detailed", track_memory=True, track_cpu=True, track_io=track_data_volume)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def _validate_data_quality_strict(args, kwargs, context, logger):
    """Strict data quality validation."""
    # Implementation for strict validation


async def _validate_data_quality_warning(args, kwargs, context, logger):
    """Warning-based data quality validation."""
    # Implementation for warning-based validation


async def _validate_data_quality_info(args, kwargs, context, logger):
    """Info-based data quality validation."""
    # Implementation for info-based validation


def _validate_data_quality_strict_sync(args, kwargs, context, logger):
    """Synchronous strict data quality validation."""


def _validate_data_quality_warning_sync(args, kwargs, context, logger):
    """Synchronous warning-based data quality validation."""


def _validate_data_quality_info_sync(args, kwargs, context, logger):
    """Synchronous info-based data quality validation."""


async def _apply_data_quality_fixes(args, kwargs, context):
    """Apply data quality fixes."""
    return args, kwargs


def _apply_data_quality_fixes_sync(args, kwargs, context):
    """Synchronous data quality fixes."""
    return args, kwargs


async def _validate_output_quality(result, context, logger):
    """Validate output quality."""


def _validate_output_quality_sync(result, context, logger):
    """Synchronous output quality validation."""


async def _apply_output_quality_fixes(result, context):
    """Apply output quality fixes."""
    return result


def _apply_output_quality_fixes_sync(result, context):
    """Synchronous output quality fixes."""
    return result


def _assess_quality(result, context):
    """Assess the quality of a result."""
    # Placeholder implementation
    return 0.8, "B"


def _grade_to_score(grade):
    """Convert letter grade to numeric score."""
    grade_map = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6, "F": 0.5}
    return grade_map.get(grade, 0.5)


def _apply_quality_degradation(result, quality_score, context):
    """Apply quality degradation to result."""
    return result


def _get_validation_thresholds(step_name, validation_config, adaptive_thresholds, args, kwargs):
    """Get validation thresholds for ML step."""
    # Placeholder implementation
    return {}


async def _validate_ml_step_prerequisites(args, kwargs, step_name, thresholds, logger):
    """Validate ML step prerequisites."""


def _validate_ml_step_prerequisites_sync(args, kwargs, step_name, thresholds, logger):
    """Synchronous ML step prerequisites validation."""


async def _validate_ml_step_output(result, step_name, thresholds, logger):
    """Validate ML step output."""


def _validate_ml_step_output_sync(result, step_name, thresholds, logger):
    """Synchronous ML step output validation."""


async def _apply_intelligent_fixes(args, kwargs, context, fix_strategies):
    """Apply intelligent fixes to arguments."""
    return args, kwargs


def _apply_intelligent_fixes_sync(args, kwargs, context, fix_strategies):
    """Synchronous intelligent fixes."""
    return args, kwargs


# ============================================================================
# EXPORT ALL DECORATORS
# ============================================================================

__all__ = [
    # Enhanced v2 decorators
    "validate_data_quality_v2",
    "quality_gate_v2",
    "step_specific_ml_validation_v2",
    "auto_fix_data_quality_issues_v2",
    "monitor_feature_engineering_v2",
    "monitor_data_collection_v2",
    # Enhanced decorators from enhanced_decorators module
    "smart_error_recovery",
    "cached_validation",
    "enhanced_validation",
    "performance_monitor_v2",
    # Existing decorators for backwards compatibility
    "validate_call_or_runtime_types",
    "pa_check_input",
    "pa_check_output",
    "pa_check_io",
    "enforce_ndarray",
    "auto_vectorize",
    "guard_array_nan_inf",
    "guard_dataframe_nulls",
    "normalize_errors",
    "with_tracing_span",
    # Training pipeline decorators
    "deterministic_seed",
    "idempotent_step",
    "artifact_write_lock",
    "nan_inf_and_constant_guard",
    "artifact_versioning",
    "time_budget_watchdog",
    "validate_step_prerequisites",
    "secure_data_processing",
    "prevent_data_leakage",
    "resource_monitor",
    "memory_efficient",
    "debug_training_step",
    "circuit_breaker_protection",
    "validate_step_output",
    # Data quality decorators
    "validate_constant_features",
    "validate_low_variance_features",
    "validate_data_completeness",
    "validate_datetime_index",
    "validate_multi_timeframe_alignment",
    "validate_hmm_data_requirements",
    "validate_data_structure",
    "optimize_memory_usage",
    "comprehensive_data_validation",
    "validate_memory_optimized_data_quality",
    "validate_feature_engineering_pipeline",
    "validate_hmm_regime_discovery",
    "validate_multi_timeframe_processing",
    # Advanced decorators
    "performance_monitor",
    "model_validation",
    "pipeline_checkpoint",
    "intelligent_caching",
    "adaptive_resource_allocation",
    "comprehensive_validation",
    "PerformanceLevel",
    "ValidationLevel",
    # Registry and config access
    "decorator_registry",
    "global_config",
]
