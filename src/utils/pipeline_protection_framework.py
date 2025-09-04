#!/usr/bin/env python3
"""
Pipeline Protection Framework

This module provides comprehensive protection mechanisms for the optimisation pipeline:
- Data validation and formatting
- Error handling and recovery
- Performance monitoring
- Security and access control
- State management and persistence
"""

import asyncio
import functools
import json
import logging
import os
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.common_operations import (
    ensure_directory,
    safe_file_exists,
    safe_json_dump,
    safe_json_load,
    format_datetime,
    get_current_datetime,
)
from src.utils.logger import system_logger


class ValidationLevel(Enum):
    """Validation levels for pipeline operations."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    CRITICAL = "critical"


class OperationType(Enum):
    """Types of operations in the pipeline."""
    DATA_LOADING = "data_loading"
    DATA_PROCESSING = "data_processing"
    MODEL_TRAINING = "model_training"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"
    PERSISTENCE = "persistence"


@dataclass
class PipelineState:
    """Pipeline state management."""
    current_step: str = ""
    step_history: List[str] = field(default_factory=list)
    data_checkpoints: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    error_log: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=get_current_datetime)
    updated_at: datetime = field(default_factory=get_current_datetime)
    
    def update_step(self, step_name: str) -> None:
        """Update current step and history."""
        if self.current_step:
            self.step_history.append(self.current_step)
        self.current_step = step_name
        self.updated_at = get_current_datetime()
    
    def add_checkpoint(self, checkpoint_name: str, data: Any) -> None:
        """Add a data checkpoint."""
        self.data_checkpoints[checkpoint_name] = {
            "data": data,
            "timestamp": get_current_datetime(),
            "step": self.current_step
        }
        self.updated_at = get_current_datetime()
    
    def add_validation_result(self, step_name: str, result: Dict[str, Any]) -> None:
        """Add validation result."""
        self.validation_results[step_name] = result
        self.updated_at = get_current_datetime()
    
    def add_error(self, error: Dict[str, Any]) -> None:
        """Add error to log."""
        error["timestamp"] = get_current_datetime()
        error["step"] = self.current_step
        self.error_log.append(error)
        self.updated_at = get_current_datetime()


@dataclass
class DataIntegrityCheck:
    """Data integrity check result."""
    passed: bool
    checksum: str
    size_bytes: int
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    data_types: Optional[Dict[str, str]] = None
    null_counts: Optional[Dict[str, int]] = None
    timestamp: datetime = field(default_factory=get_current_datetime)


class DataValidator:
    """Comprehensive data validation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("DataValidator")
    
    def validate_dataframe(
        self, 
        df: pd.DataFrame, 
        expected_columns: Optional[List[str]] = None,
        min_rows: int = 1,
        max_null_ratio: float = 0.5
    ) -> DataIntegrityCheck:
        """Validate DataFrame integrity."""
        try:
            # Basic checks
            if df is None or df.empty:
                return DataIntegrityCheck(
                    passed=False,
                    checksum="",
                    size_bytes=0,
                    row_count=0
                )
            
            # Calculate checksum
            checksum = self._calculate_checksum(df)
            
            # Size and shape
            size_bytes = df.memory_usage(deep=True).sum()
            row_count = len(df)
            column_count = len(df.columns)
            
            # Column validation
            if expected_columns:
                missing_cols = set(expected_columns) - set(df.columns)
                if missing_cols:
                    self.logger.error(f"Missing expected columns: {missing_cols}")
                    return DataIntegrityCheck(
                        passed=False,
                        checksum=checksum,
                        size_bytes=size_bytes,
                        row_count=row_count,
                        column_count=column_count
                    )
            
            # Row count validation
            if row_count < min_rows:
                self.logger.error(f"Insufficient rows: {row_count} < {min_rows}")
                return DataIntegrityCheck(
                    passed=False,
                    checksum=checksum,
                    size_bytes=size_bytes,
                    row_count=row_count,
                    column_count=column_count
                )
            
            # Null ratio validation
            null_counts = df.isnull().sum()
            total_cells = row_count * column_count
            null_ratio = null_counts.sum() / total_cells if total_cells > 0 else 0
            
            if null_ratio > max_null_ratio:
                self.logger.error(f"Excessive null ratio: {null_ratio:.3f} > {max_null_ratio}")
                return DataIntegrityCheck(
                    passed=False,
                    checksum=checksum,
                    size_bytes=size_bytes,
                    row_count=row_count,
                    column_count=column_count,
                    null_counts=null_counts.to_dict()
                )
            
            # Data types
            data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
            return DataIntegrityCheck(
                passed=True,
                checksum=checksum,
                size_bytes=size_bytes,
                row_count=row_count,
                column_count=column_count,
                data_types=data_types,
                null_counts=null_counts.to_dict()
            )
            
        except Exception as e:
            self.logger.exception(f"Data validation error: {e}")
            return DataIntegrityCheck(
                passed=False,
                checksum="",
                size_bytes=0
            )
    
    def _calculate_checksum(self, df: pd.DataFrame) -> str:
        """Calculate checksum for DataFrame."""
        try:
            # Convert to string representation for checksum
            df_str = df.to_string()
            return hashlib.md5(df_str.encode()).hexdigest()
        except Exception:
            return ""


class PipelineProtectionDecorator:
    """Decorator for pipeline operation protection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("PipelineProtection")
        self.data_validator = DataValidator(config)
    
    def protect_operation(
        self,
        operation_type: OperationType,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        retry_count: int = 3,
        timeout_seconds: int = 300
    ):
        """Decorator to protect pipeline operations."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                operation_name = f"{func.__module__}.{func.__name__}"
                start_time = time.time()
                
                self.logger.info(f"🛡️ Starting protected operation: {operation_name}")
                
                # Pre-operation validation
                pre_validation = await self._pre_operation_validation(
                    func, args, kwargs, operation_type, validation_level
                )
                
                if not pre_validation["passed"]:
                    self.logger.error(f"❌ Pre-operation validation failed: {pre_validation['error']}")
                    raise ValueError(f"Pre-operation validation failed: {pre_validation['error']}")
                
                # Execute operation with retry logic
                last_exception = None
                for attempt in range(retry_count + 1):
                    try:
                        if attempt > 0:
                            self.logger.info(f"🔄 Retry attempt {attempt}/{retry_count} for {operation_name}")
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        
                        # Execute with timeout
                        result = await asyncio.wait_for(
                            func(*args, **kwargs),
                            timeout=timeout_seconds
                        )
                        
                        # Post-operation validation
                        post_validation = await self._post_operation_validation(
                            result, operation_type, validation_level
                        )
                        
                        if not post_validation["passed"]:
                            self.logger.error(f"❌ Post-operation validation failed: {post_validation['error']}")
                            raise ValueError(f"Post-operation validation failed: {post_validation['error']}")
                        
                        # Log success
                        duration = time.time() - start_time
                        self.logger.info(f"✅ Operation completed successfully: {operation_name} ({duration:.2f}s)")
                        
                        return result
                        
                    except asyncio.TimeoutError:
                        last_exception = asyncio.TimeoutError(f"Operation {operation_name} timed out after {timeout_seconds}s")
                        self.logger.error(f"⏰ Operation timeout: {operation_name}")
                        
                    except Exception as e:
                        last_exception = e
                        self.logger.error(f"❌ Operation failed: {operation_name} - {str(e)}")
                        
                        if attempt == retry_count:
                            # Log full traceback on final failure
                            self.logger.exception(f"Final failure for {operation_name}")
                
                # All retries failed
                duration = time.time() - start_time
                self.logger.error(f"💥 Operation failed after {retry_count} retries: {operation_name} ({duration:.2f}s)")
                raise last_exception
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For synchronous functions, run in event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    async def _pre_operation_validation(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        operation_type: OperationType,
        validation_level: ValidationLevel
    ) -> Dict[str, Any]:
        """Pre-operation validation."""
        try:
            # Basic argument validation
            if not args and not kwargs:
                return {"passed": True, "message": "No arguments to validate"}
            
            # Data validation for DataFrame arguments
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    validation = self.data_validator.validate_dataframe(arg)
                    if not validation.passed:
                        return {
                            "passed": False,
                            "error": f"DataFrame validation failed: {validation}"
                        }
            
            # Check for required kwargs based on operation type
            if operation_type == OperationType.DATA_LOADING:
                required_kwargs = ["symbol", "exchange"]
                missing_kwargs = [kw for kw in required_kwargs if kw not in kwargs]
                if missing_kwargs:
                    return {
                        "passed": False,
                        "error": f"Missing required kwargs for data loading: {missing_kwargs}"
                    }
            
            return {"passed": True, "message": "Pre-operation validation passed"}
            
        except Exception as e:
            return {"passed": False, "error": f"Pre-operation validation error: {str(e)}"}
    
    async def _post_operation_validation(
        self,
        result: Any,
        operation_type: OperationType,
        validation_level: ValidationLevel
    ) -> Dict[str, Any]:
        """Post-operation validation."""
        try:
            # Basic result validation
            if result is None:
                return {"passed": False, "error": "Operation returned None"}
            
            # Type-specific validation
            if isinstance(result, pd.DataFrame):
                validation = self.data_validator.validate_dataframe(result)
                if not validation.passed:
                    return {
                        "passed": False,
                        "error": f"Result DataFrame validation failed: {validation}"
                    }
            
            elif isinstance(result, dict):
                if not result:
                    return {"passed": False, "error": "Operation returned empty dictionary"}
                
                # Check for error indicators
                if "error" in result or "errors" in result:
                    return {"passed": False, "error": "Operation result contains errors"}
            
            return {"passed": True, "message": "Post-operation validation passed"}
            
        except Exception as e:
            return {"passed": False, "error": f"Post-operation validation error: {str(e)}"}


class PipelineStateManager:
    """Pipeline state management with persistence."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("PipelineStateManager")
        self.state_file = Path(config.get("state_file", "data_cache/pipeline_state.json"))
        self._state: Optional[PipelineState] = None
    
    async def load_state(self) -> PipelineState:
        """Load pipeline state from file."""
        try:
            if safe_file_exists(self.state_file):
                state_data = safe_json_load(self.state_file)
                self._state = PipelineState(**state_data)
                self.logger.info(f"✅ Loaded pipeline state from {self.state_file}")
            else:
                self._state = PipelineState()
                self.logger.info("🆕 Created new pipeline state")
            
            return self._state
            
        except Exception as e:
            self.logger.exception(f"Error loading pipeline state: {e}")
            self._state = PipelineState()
            return self._state
    
    async def save_state(self) -> None:
        """Save pipeline state to file."""
        try:
            if self._state is None:
                self.logger.warning("No state to save")
                return
            
            # Ensure directory exists
            ensure_directory(self.state_file.parent)
            
            # Convert to dict for JSON serialization
            state_dict = {
                "current_step": self._state.current_step,
                "step_history": self._state.step_history,
                "data_checkpoints": self._state.data_checkpoints,
                "validation_results": self._state.validation_results,
                "error_log": [
                    {
                        **error,
                        "timestamp": error["timestamp"].isoformat() if isinstance(error.get("timestamp"), datetime) else str(error.get("timestamp", ""))
                    }
                    for error in self._state.error_log
                ],
                "performance_metrics": self._state.performance_metrics,
                "created_at": self._state.created_at.isoformat(),
                "updated_at": self._state.updated_at.isoformat()
            }
            
            safe_json_dump(state_dict, self.state_file, indent=2)
            self.logger.info(f"💾 Saved pipeline state to {self.state_file}")
            
        except Exception as e:
            self.logger.exception(f"Error saving pipeline state: {e}")
    
    def get_state(self) -> Optional[PipelineState]:
        """Get current state."""
        return self._state
    
    def update_state(self, state: PipelineState) -> None:
        """Update current state."""
        self._state = state


class PipelineMonitor:
    """Pipeline monitoring and alerting."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("PipelineMonitor")
        self.metrics: Dict[str, Any] = {}
        self.alerts: List[Dict[str, Any]] = []
    
    def record_metric(self, name: str, value: Any, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric."""
        try:
            if name not in self.metrics:
                self.metrics[name] = []
            
            metric_entry = {
                "value": value,
                "timestamp": get_current_datetime(),
                "tags": tags or {}
            }
            
            self.metrics[name].append(metric_entry)
            
            # Keep only last 1000 entries per metric
            if len(self.metrics[name]) > 1000:
                self.metrics[name] = self.metrics[name][-1000:]
            
        except Exception as e:
            self.logger.exception(f"Error recording metric {name}: {e}")
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []
        
        try:
            # Check for high error rates
            if "error_count" in self.metrics:
                recent_errors = [
                    m for m in self.metrics["error_count"][-10:]  # Last 10 entries
                    if (get_current_datetime() - m["timestamp"]).total_seconds() < 300  # Last 5 minutes
                ]
                
                if len(recent_errors) > 5:
                    alerts.append({
                        "type": "high_error_rate",
                        "message": f"High error rate detected: {len(recent_errors)} errors in last 5 minutes",
                        "severity": "warning",
                        "timestamp": get_current_datetime()
                    })
            
            # Check for performance degradation
            if "operation_duration" in self.metrics:
                recent_durations = [
                    m["value"] for m in self.metrics["operation_duration"][-10:]
                    if (get_current_datetime() - m["timestamp"]).total_seconds() < 300
                ]
                
                if recent_durations:
                    avg_duration = sum(recent_durations) / len(recent_durations)
                    if avg_duration > 60:  # More than 1 minute average
                        alerts.append({
                            "type": "performance_degradation",
                            "message": f"Performance degradation detected: {avg_duration:.2f}s average duration",
                            "severity": "warning",
                            "timestamp": get_current_datetime()
                        })
            
        except Exception as e:
            self.logger.exception(f"Error checking alerts: {e}")
        
        return alerts
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        try:
            summary = {}
            
            for metric_name, metric_data in self.metrics.items():
                if not metric_data:
                    continue
                
                values = [m["value"] for m in metric_data if isinstance(m["value"], (int, float))]
                
                if values:
                    summary[metric_name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "latest": values[-1] if values else None
                    }
            
            return summary
            
        except Exception as e:
            self.logger.exception(f"Error generating metrics summary: {e}")
            return {}


# Global instances
_pipeline_protection: Optional[PipelineProtectionDecorator] = None
_state_manager: Optional[PipelineStateManager] = None
_monitor: Optional[PipelineMonitor] = None


def initialize_pipeline_protection(config: Dict[str, Any]) -> None:
    """Initialize pipeline protection framework."""
    global _pipeline_protection, _state_manager, _monitor
    
    _pipeline_protection = PipelineProtectionDecorator(config)
    _state_manager = PipelineStateManager(config)
    _monitor = PipelineMonitor(config)
    
    system_logger.info("🛡️ Pipeline protection framework initialized")


def get_pipeline_protection() -> PipelineProtectionDecorator:
    """Get pipeline protection decorator."""
    if _pipeline_protection is None:
        raise RuntimeError("Pipeline protection not initialized. Call initialize_pipeline_protection() first.")
    return _pipeline_protection


def get_state_manager() -> PipelineStateManager:
    """Get state manager."""
    if _state_manager is None:
        raise RuntimeError("State manager not initialized. Call initialize_pipeline_protection() first.")
    return _state_manager


def get_monitor() -> PipelineMonitor:
    """Get monitor."""
    if _monitor is None:
        raise RuntimeError("Monitor not initialized. Call initialize_pipeline_protection() first.")
    return _monitor


# Convenience decorators
def protect_data_operation(validation_level: ValidationLevel = ValidationLevel.STANDARD):
    """Decorator for data operations."""
    protection = get_pipeline_protection()
    return protection.protect_operation(
        OperationType.DATA_PROCESSING,
        validation_level=validation_level
    )


def protect_model_operation(validation_level: ValidationLevel = ValidationLevel.CRITICAL):
    """Decorator for model operations."""
    protection = get_pipeline_protection()
    return protection.protect_operation(
        OperationType.MODEL_TRAINING,
        validation_level=validation_level
    )


def protect_optimization_operation(validation_level: ValidationLevel = ValidationLevel.CRITICAL):
    """Decorator for optimization operations."""
    protection = get_pipeline_protection()
    return protection.protect_operation(
        OperationType.OPTIMIZATION,
        validation_level=validation_level
    )