"""
Enhanced Validation Decorators for ML Data Quality

This module provides advanced decorators that integrate comprehensive ML data quality
validation with quality gates, continuous monitoring, and alert systems.
"""

import os
import sys
import functools
import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.logger import system_logger
from src.utils.advanced_ml_validation import (
    AdvancedMLValidator,
    MLValidationResult,
    QualityScore,
    validate_ml_data_quality,
    detect_data_drift,
    calculate_data_quality_score
)
from src.utils.quality_alert_system import (
    QualityAlertManager,
    create_alert_config,
    Alert
)
from src.utils.validation_decorators import (
    validate_file_operation,
    validate_dataframe_operation
)


def validate_ml_data_quality_decorator(
    target_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    min_quality_score: float = 0.8,
    max_correlation: float = 0.95,
    max_drift_psi: float = 0.25,
    required_grade: str = "B",
    enable_drift_detection: bool = False,
    reference_data: Optional[Any] = None,
    alert_config: Optional[Dict[str, Any]] = None,
    log_level: str = "INFO"
):
    """
    Decorator for comprehensive ML data quality validation.
    
    Args:
        target_col: Target variable column name
        timestamp_col: Timestamp column name for time series validation
        min_quality_score: Minimum acceptable quality score (0.0-1.0)
        max_correlation: Maximum allowed feature correlation
        max_drift_psi: Maximum allowed PSI for drift detection
        required_grade: Minimum required quality grade (A, B, C, D, F)
        enable_drift_detection: Whether to enable drift detection
        reference_data: Reference data for drift detection
        alert_config: Configuration for alert system
        log_level: Logging level for validation messages
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = system_logger.getChild("MLQualityValidator")
            
            # Extract DataFrame from function arguments
            df = _extract_dataframe_from_args(args, kwargs)
            if df is None:
                logger.warning("No DataFrame found in function arguments, skipping ML validation")
                return await func(*args, **kwargs)
            
            # Set up alert manager if configured
            alert_manager = None
            if alert_config:
                alert_config_obj = create_alert_config(**alert_config)
                alert_manager = QualityAlertManager(alert_config_obj)
            
            # Perform ML data quality validation
            logger.info("🔍 Performing ML data quality validation...")
            
            validation_result = validate_ml_data_quality(
                df=df,
                target_col=target_col,
                timestamp_col=timestamp_col,
                config={
                    "detect_drift": enable_drift_detection,
                    "validate_distributions": True,
                    "validate_outliers": True,
                    "validate_time_series": timestamp_col is not None,
                    "validate_financial": True,
                    "validate_correlations": True,
                    "validate_target": target_col is not None
                }
            )
            
            # Check quality gates
            quality_gate_passed = _check_quality_gates(
                validation_result, min_quality_score, max_correlation, 
                max_drift_psi, required_grade
            )
            
            if not quality_gate_passed:
                error_msg = f"Quality gate failed: Score={validation_result.quality_score.overall:.3f}, Grade={validation_result.quality_score.grade}"
                logger.error(f"❌ {error_msg}")
                
                # Send alerts if configured
                if alert_manager:
                    alerts = alert_manager.check_alerts(validation_result)
                    alert_manager.send_alerts(alerts)
                
                raise ValueError(error_msg)
            
            # Log validation results
            logger.info(f"✅ ML quality validation passed: Score={validation_result.quality_score.overall:.3f}, Grade={validation_result.quality_score.grade}")
            
            # Send alerts for warnings if configured
            if alert_manager and not validation_result.is_valid:
                alerts = alert_manager.check_alerts(validation_result)
                alert_manager.send_alerts(alerts)
            
            # Execute the original function
            result = await func(*args, **kwargs)
            
            # Validate output if it's a DataFrame
            output_df = _extract_dataframe_from_result(result)
            if output_df is not None:
                logger.info("🔍 Validating function output...")
                output_validation = validate_ml_data_quality(
                    df=output_df,
                    target_col=target_col,
                    timestamp_col=timestamp_col
                )
                
                if not output_validation.is_valid:
                    logger.warning(f"⚠️ Output validation found issues: {len(output_validation.correlation_issues + output_validation.target_issues)} issues")
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = system_logger.getChild("MLQualityValidator")
            
            # Extract DataFrame from function arguments
            df = _extract_dataframe_from_args(args, kwargs)
            if df is None:
                logger.warning("No DataFrame found in function arguments, skipping ML validation")
                return func(*args, **kwargs)
            
            # Set up alert manager if configured
            alert_manager = None
            if alert_config:
                alert_config_obj = create_alert_config(**alert_config)
                alert_manager = QualityAlertManager(alert_config_obj)
            
            # Perform ML data quality validation
            logger.info("🔍 Performing ML data quality validation...")
            
            validation_result = validate_ml_data_quality(
                df=df,
                target_col=target_col,
                timestamp_col=timestamp_col,
                config={
                    "detect_drift": enable_drift_detection,
                    "validate_distributions": True,
                    "validate_outliers": True,
                    "validate_time_series": timestamp_col is not None,
                    "validate_financial": True,
                    "validate_correlations": True,
                    "validate_target": target_col is not None
                }
            )
            
            # Check quality gates
            quality_gate_passed = _check_quality_gates(
                validation_result, min_quality_score, max_correlation, 
                max_drift_psi, required_grade
            )
            
            if not quality_gate_passed:
                error_msg = f"Quality gate failed: Score={validation_result.quality_score.overall:.3f}, Grade={validation_result.quality_score.grade}"
                logger.error(f"❌ {error_msg}")
                
                # Send alerts if configured
                if alert_manager:
                    alerts = alert_manager.check_alerts(validation_result)
                    alert_manager.send_alerts(alerts)
                
                raise ValueError(error_msg)
            
            # Log validation results
            logger.info(f"✅ ML quality validation passed: Score={validation_result.quality_score.overall:.3f}, Grade={validation_result.quality_score.grade}")
            
            # Send alerts for warnings if configured
            if alert_manager and not validation_result.is_valid:
                alerts = alert_manager.check_alerts(validation_result)
                alert_manager.send_alerts(alerts)
            
            # Execute the original function
            result = func(*args, **kwargs)
            
            # Validate output if it's a DataFrame
            output_df = _extract_dataframe_from_result(result)
            if output_df is not None:
                logger.info("🔍 Validating function output...")
                output_validation = validate_ml_data_quality(
                    df=output_df,
                    target_col=target_col,
                    timestamp_col=timestamp_col
                )
                
                if not output_validation.is_valid:
                    logger.warning(f"⚠️ Output validation found issues: {len(output_validation.correlation_issues + output_validation.target_issues)} issues")
            
            return result
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def quality_gate(
    min_quality_score: float = 0.8,
    max_correlation: float = 0.95,
    max_drift_psi: float = 0.25,
    required_grade: str = "B",
    enable_alerts: bool = True,
    alert_config: Optional[Dict[str, Any]] = None
):
    """
    Quality gate decorator that enforces data quality standards.
    
    Args:
        min_quality_score: Minimum acceptable quality score (0.0-1.0)
        max_correlation: Maximum allowed feature correlation
        max_drift_psi: Maximum allowed PSI for drift detection
        required_grade: Minimum required quality grade (A, B, C, D, F)
        enable_alerts: Whether to enable alert system
        alert_config: Configuration for alert system
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = system_logger.getChild("QualityGate")
            
            # Set up alert manager if enabled
            alert_manager = None
            if enable_alerts and alert_config:
                alert_config_obj = create_alert_config(**alert_config)
                alert_manager = QualityAlertManager(alert_config_obj)
            
            # Execute the original function
            logger.info("🚀 Executing function with quality gate...")
            result = await func(*args, **kwargs)
            
            # Extract DataFrame from result
            df = _extract_dataframe_from_result(result)
            if df is None:
                logger.warning("No DataFrame found in result, skipping quality gate")
                return result
            
            # Perform quality validation
            logger.info("🔍 Applying quality gate validation...")
            validation_result = validate_ml_data_quality(df)
            
            # Check quality gates
            quality_gate_passed = _check_quality_gates(
                validation_result, min_quality_score, max_correlation, 
                max_drift_psi, required_grade
            )
            
            if not quality_gate_passed:
                error_msg = f"Quality gate failed: Score={validation_result.quality_score.overall:.3f}, Grade={validation_result.quality_score.grade}"
                logger.error(f"❌ {error_msg}")
                
                # Send alerts if configured
                if alert_manager:
                    alerts = alert_manager.check_alerts(validation_result)
                    alert_manager.send_alerts(alerts)
                
                raise ValueError(error_msg)
            
            logger.info(f"✅ Quality gate passed: Score={validation_result.quality_score.overall:.3f}, Grade={validation_result.quality_score.grade}")
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = system_logger.getChild("QualityGate")
            
            # Set up alert manager if enabled
            alert_manager = None
            if enable_alerts and alert_config:
                alert_config_obj = create_alert_config(**alert_config)
                alert_manager = QualityAlertManager(alert_config_obj)
            
            # Execute the original function
            logger.info("🚀 Executing function with quality gate...")
            result = func(*args, **kwargs)
            
            # Extract DataFrame from result
            df = _extract_dataframe_from_result(result)
            if df is None:
                logger.warning("No DataFrame found in result, skipping quality gate")
                return result
            
            # Perform quality validation
            logger.info("🔍 Applying quality gate validation...")
            validation_result = validate_ml_data_quality(df)
            
            # Check quality gates
            quality_gate_passed = _check_quality_gates(
                validation_result, min_quality_score, max_correlation, 
                max_drift_psi, required_grade
            )
            
            if not quality_gate_passed:
                error_msg = f"Quality gate failed: Score={validation_result.quality_score.overall:.3f}, Grade={validation_result.quality_score.grade}"
                logger.error(f"❌ {error_msg}")
                
                # Send alerts if configured
                if alert_manager:
                    alerts = alert_manager.check_alerts(validation_result)
                    alert_manager.send_alerts(alerts)
                
                raise ValueError(error_msg)
            
            logger.info(f"✅ Quality gate passed: Score={validation_result.quality_score.overall:.3f}, Grade={validation_result.quality_score.grade}")
            return result
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def continuous_quality_monitoring(
    target_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    monitoring_interval: int = 100,  # Check every N operations
    alert_config: Optional[Dict[str, Any]] = None,
    drift_detection: bool = False,
    reference_data: Optional[Any] = None
):
    """
    Decorator for continuous quality monitoring during data processing.
    
    Args:
        target_col: Target variable column name
        timestamp_col: Timestamp column name
        monitoring_interval: Check quality every N operations
        alert_config: Configuration for alert system
        drift_detection: Whether to enable drift detection
        reference_data: Reference data for drift detection
    """
    def decorator(func: Callable) -> Callable:
        # Initialize monitoring state
        operation_count = 0
        alert_manager = None
        
        if alert_config:
            alert_config_obj = create_alert_config(**alert_config)
            alert_manager = QualityAlertManager(alert_config_obj)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal operation_count, alert_manager
            
            logger = system_logger.getChild("ContinuousMonitoring")
            
            # Execute the original function
            result = await func(*args, **kwargs)
            
            # Increment operation count
            operation_count += 1
            
            # Check if it's time to monitor
            if operation_count % monitoring_interval == 0:
                logger.info(f"📊 Performing continuous quality monitoring (operation #{operation_count})...")
                
                # Extract DataFrame from result
                df = _extract_dataframe_from_result(result)
                if df is not None:
                    # Perform validation
                    validation_result = validate_ml_data_quality(
                        df=df,
                        target_col=target_col,
                        timestamp_col=timestamp_col,
                        config={"detect_drift": drift_detection}
                    )
                    
                    # Send alerts if issues found
                    if alert_manager and not validation_result.is_valid:
                        alerts = alert_manager.check_alerts(validation_result)
                        alert_manager.send_alerts(alerts)
                        
                        logger.warning(f"⚠️ Quality issues detected in operation #{operation_count}")
                    else:
                        logger.info(f"✅ Quality check passed for operation #{operation_count}")
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            nonlocal operation_count, alert_manager
            
            logger = system_logger.getChild("ContinuousMonitoring")
            
            # Execute the original function
            result = func(*args, **kwargs)
            
            # Increment operation count
            operation_count += 1
            
            # Check if it's time to monitor
            if operation_count % monitoring_interval == 0:
                logger.info(f"📊 Performing continuous quality monitoring (operation #{operation_count})...")
                
                # Extract DataFrame from result
                df = _extract_dataframe_from_result(result)
                if df is not None:
                    # Perform validation
                    validation_result = validate_ml_data_quality(
                        df=df,
                        target_col=target_col,
                        timestamp_col=timestamp_col,
                        config={"detect_drift": drift_detection}
                    )
                    
                    # Send alerts if issues found
                    if alert_manager and not validation_result.is_valid:
                        alerts = alert_manager.check_alerts(validation_result)
                        alert_manager.send_alerts(alerts)
                        
                        logger.warning(f"⚠️ Quality issues detected in operation #{operation_count}")
                    else:
                        logger.info(f"✅ Quality check passed for operation #{operation_count}")
            
            return result
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def step_specific_ml_validation(step_name: str, **kwargs):
    """
    Step-specific ML validation decorator with predefined configurations.
    
    Args:
        step_name: Name of the pipeline step
        **kwargs: Additional validation parameters
    """
    # Step-specific configurations
    step_configs = {
        "step1": {
            "target_col": None,
            "timestamp_col": "timestamp",
            "min_quality_score": 0.7,
            "required_grade": "C",
            "validate_financial": True,
            "validate_time_series": True
        },
        "step1_5": {
            "target_col": None,
            "timestamp_col": "timestamp",
            "min_quality_score": 0.75,
            "required_grade": "C",
            "validate_financial": True,
            "validate_time_series": True
        },
        "step2": {
            "target_col": None,
            "timestamp_col": "timestamp",
            "min_quality_score": 0.8,
            "required_grade": "B",
            "validate_correlations": True,
            "validate_distributions": True
        },
        "step4": {
            "target_col": "target",
            "timestamp_col": "timestamp",
            "min_quality_score": 0.85,
            "required_grade": "B",
            "validate_target": True,
            "validate_correlations": True,
            "validate_distributions": True
        }
    }
    
    # Get step configuration
    step_config = step_configs.get(step_name, {})
    
    # Merge with provided kwargs
    config = {**step_config, **kwargs}
    
    return validate_ml_data_quality_decorator(**config)


# Helper functions
def _extract_dataframe_from_args(args: Tuple, kwargs: Dict) -> Optional[Any]:
    """Extract DataFrame from function arguments."""
    import pandas as pd
    
    # Check positional arguments
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            return arg
    
    # Check keyword arguments
    for value in kwargs.values():
        if isinstance(value, pd.DataFrame):
            return value
    
    return None


def _extract_dataframe_from_result(result: Any) -> Optional[Any]:
    """Extract DataFrame from function result."""
    import pandas as pd
    
    if isinstance(result, pd.DataFrame):
        return result
    elif isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, pd.DataFrame):
                return item
    elif isinstance(result, dict):
        for value in result.values():
            if isinstance(value, pd.DataFrame):
                return value
    
    return None


def _check_quality_gates(
    validation_result: MLValidationResult,
    min_quality_score: float,
    max_correlation: float,
    max_drift_psi: float,
    required_grade: str
) -> bool:
    """Check if quality gates are passed."""
    # Check quality score
    if validation_result.quality_score.overall < min_quality_score:
        return False
    
    # Check quality grade
    grade_order = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
    actual_grade_score = grade_order.get(validation_result.quality_score.grade, 0)
    required_grade_score = grade_order.get(required_grade, 0)
    
    if actual_grade_score < required_grade_score:
        return False
    
    # Check correlation issues
    if validation_result.correlation_issues:
        # Extract correlation values from issues
        for issue in validation_result.correlation_issues:
            if "corr=" in issue:
                try:
                    corr_value = float(issue.split("corr=")[1].split()[0])
                    if abs(corr_value) > max_correlation:
                        return False
                except (ValueError, IndexError):
                    continue
    
    # Check drift issues
    if validation_result.drift_report:
        for issue in validation_result.drift_report.issues:
            if "PSI=" in issue:
                try:
                    psi_value = float(issue.split("PSI=")[1].split()[0])
                    if psi_value > max_drift_psi:
                        return False
                except (ValueError, IndexError):
                    continue
    
    return True


# Convenience decorators for specific use cases
def validate_training_data(**kwargs):
    """Decorator specifically for training data validation."""
    return validate_ml_data_quality_decorator(
        target_col="target",
        min_quality_score=0.85,
        required_grade="B",
        validate_target=True,
        validate_correlations=True,
        **kwargs
    )


def validate_inference_data(**kwargs):
    """Decorator specifically for inference data validation."""
    return validate_ml_data_quality_decorator(
        min_quality_score=0.8,
        required_grade="C",
        validate_correlations=True,
        validate_distributions=True,
        **kwargs
    )


def validate_feature_engineering(**kwargs):
    """Decorator specifically for feature engineering validation."""
    return validate_ml_data_quality_decorator(
        min_quality_score=0.8,
        required_grade="B",
        validate_correlations=True,
        validate_distributions=True,
        validate_outliers=True,
        **kwargs
    )


def validate_model_training(**kwargs):
    """Decorator specifically for model training validation."""
    return quality_gate(
        min_quality_score=0.85,
        required_grade="B",
        enable_alerts=True,
        **kwargs
    )