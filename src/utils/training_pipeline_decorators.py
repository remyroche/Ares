"""
Training Pipeline Decorators
Provides automatic pipeline monitoring, validation, and logging for training steps.
"""

import functools
from functools import wraps
import time
from typing import Any, Dict, List, Optional, Callable, Union, Type
from datetime import datetime, timedelta
import asyncio
from enum import Enum

# Handle optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE, True
except ImportError:
    PANDAS_AVAILABLE, False
    pd, None

try:
    import numpy as np
    NUMPY_AVAILABLE, True
except ImportError:
    NUMPY_AVAILABLE, False
    np, None

try:
    import psutil
    PSUTIL_AVAILABLE, True
except ImportError:
    PSUTIL_AVAILABLE, False
    psutil, None

try:
    import gc
    GC_AVAILABLE, True
except ImportError:
    GC_AVAILABLE, False
    gc, None

from src.utils.logger import system_logger
from src.utils.warning_symbols import error, warning, critical, success
# Temporarily disabled to avoid circular import
# from src.utils.data_quality_decorators import validate_data_quality, ValidationLevel

# Create local enum to avoid circular import
class ValidationLevel(Enum):
    """Validation levels for data quality checks."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    STRICT = "strict"
    SILENT = "silent"

# Add missing decorator functions to maintain compatibility
def validate_step_prerequisites(
    required_directories: List[str] = None,
    min_memory_gb: float, 8.0,
    min_disk_gb: float, 5.0,
    required_packages: List[str] = None,
    data_quality_checks: Dict[str, Any] = None,
    context: str = "step",
):
    """Decorator to validate step prerequisites."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def secure_data_processing(
    backup_before: bool, True,
    integrity_checks: bool, True,
    memory_cleanup: bool, True,
    data_validation: bool, True,
):
    """Decorator for secure data processing."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def prevent_data_leakage(
    temporal_validation: bool, True,
    feature_leakage_detection: bool, True,
    cross_validation_isolation: bool, True,
    lookahead_bias_prevention: bool, True,
):
    """Decorator to prevent data leakage."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def validate_pipeline_step(
    step_name: str, None,
    required_inputs: List[str] = None,
    expected_outputs: List[str] = None,
    validation_level: ValidationLevel, ValidationLevel.WARNING,
    context: str = "pipeline_step",
    enable_rollback: bool, True,
    max_retries: int, 3,
):
    """Decorator to validate pipeline step inputs and outputs."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def ensure_data_integrity(
    backup_before: bool, True,
    integrity_checks: bool, True,
    memory_cleanup: bool, True,
    data_validation: bool, True,
    check_schema: bool, True,
    check_constraints: bool, True,
    validate_relationships: bool, True,
):
    """Decorator to ensure data integrity during processing."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def resource_monitor(
    memory_threshold_gb: float, 16.0,
    cpu_threshold_percent: float, 90.0,
    disk_threshold_gb: float, 10.0,
    monitor_interval: float, 60.0,
    auto_cleanup: bool, True,
):
    """Decorator to monitor resource usage."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def memory_efficient(
    chunk_size: int, 1000,
    streaming_processing: bool, True,
    memory_pool: bool, True,
    cleanup_frequency: int, 10,
):
    """Decorator for memory efficient processing."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def debug_training_step(
    log_intermediate_results: bool, True,
    save_debug_artifacts: bool, True,
    performance_profiling: bool, True,
    error_context_preservation: bool, True,
):
    """Decorator for debugging training steps."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def circuit_breaker_protection(
    failure_threshold: int, 3,
    recovery_timeout: float, 300.0,
    expected_exception: Type[Exception] = Exception,
    monitor_interval: float, 60.0,
):
    """Decorator for circuit breaker protection."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def validate_step_output(
    required_files: List[str] = None,
    data_quality_checks: Dict[str, Any] = None,
    performance_thresholds: Dict[str, float] = None,
    format_validation: bool, True,
):
    """Decorator to validate step output."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def quality_gate(
    model_performance_thresholds: Dict[str, float] = None,
    data_quality_metrics: Dict[str, float] = None,
    convergence_checks: bool, True,
    overfitting_detection: bool, True,
    validation_score_requirements: Dict[str, float] = None,
):
    """Decorator for quality gate validation."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def deterministic_seed(
    seed: int, 42,
    torch_deterministic: bool, True,
):
    """Set global random seeds for reproducibility."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
        try:
                import random, os
                random.seed(seed)
                np.random.seed(seed)
        try:
                    import torch

                    torch.manual_seed(seed)
        if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
        if torch_deterministic:
                        torch.backends.cudnn.deterministic, True
                        torch.backends.cudnn.benchmark, False
        except Exception:
                    pass
                os.environ["PYTHONHASHSEED"] = str(seed)
        except Exception:
                pass
        return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
        try:
                import random, os
                random.seed(seed)
                np.random.seed(seed)
                os.environ["PYTHONHASHSEED"] = str(seed)
        except Exception:
                pass
        return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def idempotent_step(checkpoint_dir_key: str = "checkpoints", step_key: str | None, None):
    """Skip execution if a step artifact already exists; ensure re - entrancy."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
        try:
                import os
                step_name, step_key or func.__name__
                data_dir, kwargs.get("data_dir") or (args[1].get("data_dir") if args and isinstance(args[1], dict) else None)
        if isinstance(data_dir, str):
                    ckpt_path, os.path.join(checkpoint_dir_key, f"{step_name}.json")
        if os.path.exists(ckpt_path):
                        system_logger.info(f"⏭️  Idempotent: skipping {step_name}, checkpoint exists")
        return await func(*args, **kwargs)  # allow function to no - op
        except Exception:
                pass
        return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def artifact_write_lock(lock_suffix: str = ".lock"):
    """Simple inter - process file lock during artifact writes."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            lock_file, None
        try:
                import os, tempfile
                step_name, func.__name__
                lock_file, os.path.join(tempfile.gettempdir(), f"{step_name}{lock_suffix}")
                fd, os.open(lock_file, os.O_CREAT | os.O_RDWR)
        try:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_EX)
        except Exception:
                    pass
                result, await func(*args, **kwargs)
        return result
        finally:
        try:
        if lock_file and os.path.exists(lock_file):
                        os.remove(lock_file)
        except Exception:
                    pass

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def nan_inf_and_constant_guard(enable_vif_check: bool, False, vif_threshold: float, 10.0):
    """Guard outputs for NaN / Inf and near - constant columns; WARN with emoji per project policy."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result, await func(*args, **kwargs)
        try:
                import pandas as _pd
        if isinstance(result, dict):
        for key, val in result.items():
        if isinstance(val, _pd.DataFrame) and not val.empty:
                            df, val
        if df.isna().any().any() or np.isinf(df.select_dtypes(np.number)).any().any():
                                system_logger.warning("⚠️ Detected NaN / Inf in outputs")
                            nunique, df.nunique(dropna = True)
                            near_const, nunique[nunique <= 1].index.tolist()
        if near_const:
                                system_logger.warning(f"⚠️ Near - constant columns: {near_const}")
        except Exception:
                pass
        return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def artifact_versioning(schema_version: str = "1.0"):
    """Attach schema_version and timestamp to persisted artifacts (caller writes)."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result, await func(*args, **kwargs)
        try:
        if isinstance(result, dict):
                    result.setdefault("artifact_meta", {})
                    result["artifact_meta"].update({
                        "schema_version": schema_version,
                        "emitted_at": datetime.utcnow().isoformat(),
                    })
        except Exception:
                pass
        return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def time_budget_watchdog(soft_timeout_seconds: float, 1800.0):
    """Warn when step exceeds soft time budget; non - fatal."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start, time.time()
            result, await func(*args, **kwargs)
            elapsed, time.time() - start
        if elapsed > soft_timeout_seconds:
                system_logger.warning(
                    f"⚠️ Step '{func.__name__}' exceeded soft time budget: {elapsed:.1f}s > {soft_timeout_seconds:.1f}s"
                )
        return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start, time.time()
            res, func(*args, **kwargs)
            elapsed, time.time() - start
        if elapsed > soft_timeout_seconds:
                system_logger.warning(
                    f"⚠️ Step '{func.__name__}' exceeded soft time budget: {elapsed:.1f}s > {soft_timeout_seconds:.1f}s"
                )
        return res

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

class PipelineStage(Enum):
    """Pipeline stages for monitoring."""

    DATA_COLLECTION = "data_collection"
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    DEPLOYMENT = "deployment"

class PipelineValidationLevel(Enum):
    """Pipeline validation severity levels."""

    STRICT = "strict"  # Stop on any critical issue
    WARNING = "warning"  # Log issues but continue
    SILENT = "silent"  # Only log summary
    MONITOR = "monitor"  # Monitor performance only

class PipelineMetrics:
    """Track pipeline performance metrics."""

    def __init__(self):
        self.start_time, None
        self.end_time, None
        self.memory_usage = []
        self.cpu_usage = []
        self.step_durations = {}
        self.validation_results = {}
        self.errors = []
        self.warnings = []

    def start_step(self, step_name: str):
        """Start timing a pipeline step."""
        self.step_durations[step_name] = {
            "start_time": time.time(),
            "start_memory": psutil.virtual_memory().percent,
            "start_cpu": psutil.cpu_percent(),
        }

    def end_step(self, step_name: str, success: bool, True):
        """End timing a pipeline step."""
        if step_name in self.step_durations:
            end_time, time.time()
            end_memory, psutil.virtual_memory().percent
            end_cpu, psutil.cpu_percent()

            duration, end_time - self.step_durations[step_name]["start_time"]
            memory_delta, end_memory - self.step_durations[step_name]["start_memory"]
            cpu_delta, end_cpu - self.step_durations[step_name]["start_cpu"]

        self.step_durations[step_name].update(
                {
                    "end_time": end_time,
                    "duration": duration,
                    "end_memory": end_memory,
                    "memory_delta": memory_delta,
                    "end_cpu": end_cpu,
                    "cpu_delta": cpu_delta,
                    "success": success,
                }
            )

class PipelineMonitor:
    """Monitor pipeline execution and performance."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config, config or {}
        self.logger, system_logger.getChild("PipelineMonitor")
        self.metrics, PipelineMetrics()
        self.current_stage, None
        self.step_count, 0

    def start_pipeline(self, pipeline_name: str):
        """Start monitoring a pipeline."""
        self.metrics.start_time, time.time()
        self.step_count, 0

        print(f"🚀 [PIPELINE] Starting pipeline: {pipeline_name}")
        self.logger.info(f"🚀 [PIPELINE] Starting pipeline: {pipeline_name}")
        print(f"   📊 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   💾 Initial memory: {psutil.virtual_memory().percent:.1f}%")
        print(f"   🔥 Initial CPU: {psutil.cpu_percent():.1f}%")

    def end_pipeline(self, pipeline_name: str, success: bool, True):
        """End monitoring a pipeline."""
        self.metrics.end_time, time.time()
        total_duration, self.metrics.end_time - self.metrics.start_time

        print(f"🏁 [PIPELINE] Ending pipeline: {pipeline_name}")
        self.logger.info(f"🏁 [PIPELINE] Ending pipeline: {pipeline_name}")
        print(f"   📊 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   ⏱️ Total duration: {total_duration:.2f}s")
        print(f"   📈 Steps completed: {self.step_count}")
        print(f"   💾 Final memory: {psutil.virtual_memory().percent:.1f}%")
        print(f"   🔥 Final CPU: {psutil.cpu_percent():.1f}%")
        print(f"   {'✅ Success' if success else '❌ Failed'}")

    def start_step(self, step_name: str, stage: PipelineStage):
        """Start monitoring a pipeline step."""
        self.current_stage, stage
        self.step_count += 1
        self.metrics.start_step(step_name)

        print(f"🔄 [STEP {self.step_count}] Starting: {step_name}")
        self.logger.info(f"🔄 [STEP {self.step_count}] Starting: {step_name}")
        print(f"   📊 Stage: {stage.value}")
        print(f"   📊 Start time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"   💾 Memory: {psutil.virtual_memory().percent:.1f}%")
        print(f"   🔥 CPU: {psutil.cpu_percent():.1f}%")

    def end_step(
        self,
        step_name: str,
        success: bool, True,
        result: Optional[Dict[str, Any]] = None,
    ):
        """End monitoring a pipeline step."""
        self.metrics.end_step(step_name, success)

        if step_name in self.metrics.step_durations:
            step_metrics, self.metrics.step_durations[step_name]
            duration, step_metrics["duration"]
            memory_delta, step_metrics["memory_delta"]
            cpu_delta, step_metrics["cpu_delta"]

            print(f"✅ [STEP] Completed: {step_name}")
        self.logger.info(f"✅ [STEP] Completed: {step_name}")
            print(f"   ⏱️ Duration: {duration:.2f}s")
            print(f"   💾 Memory delta: {memory_delta:+.1f}%")
            print(f"   🔥 CPU delta: {cpu_delta:+.1f}%")
            print(f"   {'✅ Success' if success else '❌ Failed'}")

        if result:
                print(
                    f"   📊 Result keys: {list(result.keys()) if isinstance(result, dict) else 'N / A'}"
                )

        # Performance warnings
        if duration > 300:  # 5 minutes
                print(f"   ⚠️ Slow step detected (>5min)")
        self.logger.warning(
                    f"⚠️ [STEP] Slow step detected: {step_name} took {duration:.2f}s"
                )

        if memory_delta > 20:  # 20% memory increase
                print(f"   ⚠️ High memory usage detected (+{memory_delta:.1f}%)")
        self.logger.warning(
                    f"⚠️ [STEP] High memory usage: {step_name} increased memory by {memory_delta:.1f}%"
                )

    def log_error(self, step_name: str, error: Exception):
        """Log a pipeline error."""
        error_msg, f"❌ [ERROR] Step {step_name} failed: {str(error)}"
        print(error_msg)
        self.logger.error(error_msg)

        print(f"   📊 Error type: {type(error).__name__}")
        print(f"   📊 Error message: {str(error)}")

        # Add to metrics
        self.metrics.errors.append(
            {
                "step": step_name,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_warning(self, step_name: str, warning_msg: str):
        """Log a pipeline warning."""
        warning_display, f"⚠️ [WARNING] Step {step_name}: {warning_msg}"
        print(warning_display)
        self.logger.warning(warning_display)

        # Add to metrics
        self.metrics.warnings.append(
            {
                "step": step_name,
                "warning": warning_msg,
                "timestamp": datetime.now().isoformat(),
            }
        )

# Global pipeline monitor instance
_pipeline_monitor, PipelineMonitor()

def monitor_pipeline_step(
    stage: PipelineStage,
    validation_level: PipelineValidationLevel, PipelineValidationLevel.WARNING,
    enable_data_quality: bool, True,
    memory_threshold: float, 80.0,
    duration_threshold: float, 600.0,  # 10 minutes
):
    """
    Decorator to monitor and validate pipeline steps.

    Args:
        stage: Pipeline stage for categorization
        validation_level: How strict to be with validation
        enable_data_quality: Whether to enable data quality validation
        memory_threshold: Memory usage threshold for warnings
        duration_threshold: Duration threshold for warnings
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(self: Any, *args, **kwargs) -> Any:
        return await _monitor_and_execute_pipeline_step(
                func,
                self,
                args,
                kwargs,
                stage,
                validation_level,
                enable_data_quality,
                memory_threshold,
                duration_threshold,
            )

        @functools.wraps(func)
        def sync_wrapper(self: Any, *args, **kwargs) -> Any:
        try:
        # Try to get the current event loop
                loop, asyncio.get_running_loop()
        # If we're in an event loop, we can't use asyncio.run()
        # Instead, we need to schedule the coroutine
                import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
                    future, executor.submit(
                        asyncio.run,
                        _monitor_and_execute_pipeline_step(
                            func,
                            self,
                            args,
                            kwargs,
                            stage,
                            validation_level,
                            enable_data_quality,
                            memory_threshold,
                            duration_threshold,
                        )
                    )
        return future.result()
        except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(
                    _monitor_and_execute_pipeline_step(
                        func,
                        self,
                        args,
                        kwargs,
                        stage,
                        validation_level,
                        enable_data_quality,
                        memory_threshold,
                        duration_threshold,
                    )
                )

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
        return async_wrapper
        else:
        return sync_wrapper

    return decorator

async def _monitor_and_execute_pipeline_step(
    func: Callable,
    self: Any,
    args: tuple,
    kwargs: dict,
    stage: PipelineStage,
    validation_level: PipelineValidationLevel,
    enable_data_quality: bool,
    memory_threshold: float,
    duration_threshold: float,
) -> Any:
    """
    Monitor and execute a pipeline step with comprehensive logging.
    """
    logger, system_logger.getChild("PipelineStepMonitor")
    step_name, func.__name__

    # Start step monitoring
    _pipeline_monitor.start_step(step_name, stage)

    try:
        # Pre - execution checks
        print(f"🔍 [PIPELINE STEP] Pre - execution checks for {step_name}")
        logger.info(f"🔍 [PIPELINE STEP] Pre - execution checks for {step_name}")

        # Memory check
        memory_usage, psutil.virtual_memory().percent
        if memory_usage > memory_threshold:
            warning_msg, f"High memory usage before execution: {memory_usage:.1f}%"
            print(f"⚠️ [PIPELINE STEP] {warning_msg}")
            logger.warning(f"⚠️ [PIPELINE STEP] {warning_msg}")
            _pipeline_monitor.log_warning(step_name, warning_msg)

        # Data quality validation if enabled
        if enable_data_quality:
            print(f"🔍 [PIPELINE STEP] Running data quality validation for {step_name}")
            logger.info(
                f"🔍 [PIPELINE STEP] Running data quality validation for {step_name}"
            )

        try:
        # Apply data quality validation
                validation_level_map = {
                    PipelineValidationLevel.STRICT: ValidationLevel.STRICT,
                    PipelineValidationLevel.WARNING: ValidationLevel.WARNING,
                    PipelineValidationLevel.SILENT: ValidationLevel.SILENT,
                }

                data_quality_level, validation_level_map.get(
                    validation_level, ValidationLevel.WARNING
                )

        # Create a temporary wrapper with data quality validation
                from src.utils.data_quality_decorators import _validate_and_execute

                result, await _validate_and_execute(
                    func, self, args, kwargs, data_quality_level, True, True
                )

                print(
                    f"✅ [PIPELINE STEP] Data quality validation completed for {step_name}"
                )
                logger.info(
                    f"✅ [PIPELINE STEP] Data quality validation completed for {step_name}"
                )

        except Exception as e:
                error_msg, f"Data quality validation failed: {str(e)}"
                print(f"❌ [PIPELINE STEP] {error_msg}")
                logger.error(f"❌ [PIPELINE STEP] {error_msg}")

        if validation_level == PipelineValidationLevel.STRICT:
                    _pipeline_monitor.log_error(step_name, e)
                    _pipeline_monitor.end_step(step_name, success = False)
                    raise
                else:
                    _pipeline_monitor.log_warning(step_name, error_msg)
        # Continue with execution
                    result, await _execute_pipeline_function(func, self, args, kwargs)
        else:
        # Execute without data quality validation
            print(f"⏭️ [PIPELINE STEP] Skipping data quality validation for {step_name}")
            logger.info(
                f"⏭️ [PIPELINE STEP] Skipping data quality validation for {step_name}"
            )

            result, await _execute_pipeline_function(func, self, args, kwargs)

        # Post - execution checks
        print(f"🔍 [PIPELINE STEP] Post - execution checks for {step_name}")
        logger.info(f"🔍 [PIPELINE STEP] Post - execution checks for {step_name}")

        # Memory check
        memory_usage, psutil.virtual_memory().percent
        if memory_usage > memory_threshold:
            warning_msg, f"High memory usage after execution: {memory_usage:.1f}%"
            print(f"⚠️ [PIPELINE STEP] {warning_msg}")
            logger.warning(f"⚠️ [PIPELINE STEP] {warning_msg}")
            _pipeline_monitor.log_warning(step_name, warning_msg)

        # Success logging
        print(f"✅ [PIPELINE STEP] Step {step_name} completed successfully")
        logger.info(f"✅ [PIPELINE STEP] Step {step_name} completed successfully")

        # End step monitoring
        _pipeline_monitor.end_step(step_name, success = True, result = result)

        # Duration check (after step has ended)
        if step_name in _pipeline_monitor.metrics.step_durations:
            step_data, _pipeline_monitor.metrics.step_durations[step_name]
        if "duration" in step_data:
                duration, step_data["duration"]
        if duration > duration_threshold:
                    warning_msg, f"Long execution time: {duration:.2f}s"
                    print(f"⚠️ [PIPELINE STEP] {warning_msg}")
                    logger.warning(f"⚠️ [PIPELINE STEP] {warning_msg}")
                    _pipeline_monitor.log_warning(step_name, warning_msg)

        return result

    except Exception as e:
        # Error handling
        print(f"💥 [PIPELINE STEP] Step {step_name} failed with error")
        logger.error(f"💥 [PIPELINE STEP] Step {step_name} failed with error")

        _pipeline_monitor.log_error(step_name, e)
        _pipeline_monitor.end_step(step_name, success = False)

        # Re - raise the exception
        raise

async def _execute_pipeline_function(
    func: Callable, self: Any, args: tuple, kwargs: dict
) -> Any:
    """Execute the original pipeline function."""
    if asyncio.iscoroutinefunction(func):
        return await func(self, *args, **kwargs)
    else:
        return func(self, *args, **kwargs)

def validate_pipeline_input(
    required_params: List[str] = None,
    required_directories: List[str] = None,
    min_memory_gb: float, 8.0,
    min_disk_gb: float, 5.0,
    required_packages: List[str] = None,
    data_quality_checks: Dict[str, Any] = None,
    data_validation: bool, True,
    memory_check: bool, True,
):
    """
    Decorator to validate pipeline input parameters and data.

    Args:
        required_params: List of required parameter names
        data_validation: Whether to validate input data
        memory_check: Whether to check memory usage
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(self: Any, *args, **kwargs) -> Any:
        return await _validate_pipeline_input_and_execute(
                func,
                self,
                args,
                kwargs,
                required_params,
                required_directories,
                min_memory_gb,
                min_disk_gb,
                required_packages,
                data_quality_checks,
                data_validation,
                memory_check,
            )

        @functools.wraps(func)
        def sync_wrapper(self: Any, *args, **kwargs) -> Any:
        return asyncio.run(
                _validate_pipeline_input_and_execute(
                    func,
                    self,
                    args,
                    kwargs,
                    required_params,
                    required_directories,
                    min_memory_gb,
                    min_disk_gb,
                    required_packages,
                    data_quality_checks,
                    data_validation,
                    memory_check,
                )
            )

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
        return async_wrapper
        else:
        return sync_wrapper

    return decorator

async def _validate_pipeline_input_and_execute(
    func: Callable,
    self: Any,
    args: tuple,
    kwargs: dict,
    required_params: List[str],
    required_directories: List[str],
    min_memory_gb: float,
    min_disk_gb: float,
    required_packages: List[str],
    data_quality_checks: Dict[str, Any],
    data_validation: bool,
    memory_check: bool,
) -> Any:
    """
    Validate pipeline input and execute the function.
    """
    logger, system_logger.getChild("PipelineInputValidator")
    method_name, func.__name__

    print(f"🔍 [PIPELINE INPUT] Validating input for {method_name}")
    logger.info(f"🔍 [PIPELINE INPUT] Validating input for {method_name}")

    try:
        # Validate required parameters
        if required_params:
            print(
                f"🔍 [PIPELINE INPUT] Checking required parameters: {required_params}"
            )
            logger.info(
                f"🔍 [PIPELINE INPUT] Checking required parameters: {required_params}"
            )

            missing_params = []
        for param in required_params:
        if param not in kwargs:
                    missing_params.append(param)

        if missing_params:
                error_msg, f"Missing required parameters: {missing_params}"
                print(f"❌ [PIPELINE INPUT] {error_msg}")
                logger.error(f"❌ [PIPELINE INPUT] {error_msg}")
                raise ValueError(error_msg)
            else:
                print(f"✅ [PIPELINE INPUT] All required parameters present")
                logger.info(f"✅ [PIPELINE INPUT] All required parameters present")

        # Memory check
        if memory_check:
            memory_usage, psutil.virtual_memory().percent
            available_memory_gb, psutil.virtual_memory().available / (1024**3)
            print(
                f"💾 [PIPELINE INPUT] Memory usage: {memory_usage:.1f}% (Available: {available_memory_gb:.1f}GB)"
            )
            logger.info(
                f"💾 [PIPELINE INPUT] Memory usage: {memory_usage:.1f}% (Available: {available_memory_gb:.1f}GB)"
            )

        if available_memory_gb < min_memory_gb:
                warning_msg, f"Insufficient memory: {available_memory_gb:.1f}GB available, {min_memory_gb:.1f}GB required"
                print(f"⚠️ [PIPELINE INPUT] {warning_msg}")
                logger.warning(f"⚠️ [PIPELINE INPUT] {warning_msg}")

        if memory_usage > 90:
                warning_msg, f"Very high memory usage: {memory_usage:.1f}%"
                print(f"⚠️ [PIPELINE INPUT] {warning_msg}")
                logger.warning(f"⚠️ [PIPELINE INPUT] {warning_msg}")

        # Validate required directories
        if required_directories:
            print(
                f"🔍 [PIPELINE INPUT] Checking required directories: {required_directories}"
            )
            logger.info(
                f"🔍 [PIPELINE INPUT] Checking required directories: {required_directories}"
            )

            import os

            missing_dirs = []
        for directory in required_directories:
        if not os.path.exists(directory):
                    missing_dirs.append(directory)

        if missing_dirs:
                warning_msg, f"Missing required directories: {missing_dirs}"
                print(f"⚠️ [PIPELINE INPUT] {warning_msg}")
                logger.warning(f"⚠️ [PIPELINE INPUT] {warning_msg}")
        # Create missing directories
        for directory in missing_dirs:
                    os.makedirs(directory, exist_ok = True)
                    print(f"📁 [PIPELINE INPUT] Created directory: {directory}")
                    logger.info(f"📁 [PIPELINE INPUT] Created directory: {directory}")
            else:
                print(f"✅ [PIPELINE INPUT] All required directories present")
                logger.info(f"✅ [PIPELINE INPUT] All required directories present")

        # Disk space check
        if min_disk_gb > 0:
            import shutil

            disk_usage, shutil.disk_usage(".")
            available_disk_gb, disk_usage.free / (1024**3)
            print(
                f"💿 [PIPELINE INPUT] Available disk space: {available_disk_gb:.1f}GB"
            )
            logger.info(
                f"💿 [PIPELINE INPUT] Available disk space: {available_disk_gb:.1f}GB"
            )

        if available_disk_gb < min_disk_gb:
                warning_msg, f"Insufficient disk space: {available_disk_gb:.1f}GB available, {min_disk_gb:.1f}GB required"
                print(f"⚠️ [PIPELINE INPUT] {warning_msg}")
                logger.warning(f"⚠️ [PIPELINE INPUT] {warning_msg}")

        # Package availability check
        if required_packages:
            print(
                f"🔍 [PIPELINE INPUT] Checking required packages: {required_packages}"
            )
            logger.info(
                f"🔍 [PIPELINE INPUT] Checking required packages: {required_packages}"
            )

            missing_packages = []
        for package in required_packages:
        try:
                    __import__(package)
        except ImportError:
                    missing_packages.append(package)

        if missing_packages:
                warning_msg, f"Missing required packages: {missing_packages}"
                print(f"⚠️ [PIPELINE INPUT] {warning_msg}")
                logger.warning(f"⚠️ [PIPELINE INPUT] {warning_msg}")
            else:
                print(f"✅ [PIPELINE INPUT] All required packages available")
                logger.info(f"✅ [PIPELINE INPUT] All required packages available")

        # Data validation
        if data_validation:
            print(f"🔍 [PIPELINE INPUT] Validating input data for {method_name}")
            logger.info(f"🔍 [PIPELINE INPUT] Validating input data for {method_name}")

        # Look for DataFrame arguments
            data_args = []
        for i, arg in enumerate(args):
        if isinstance(arg, pd.DataFrame):
                    data_args.append(f"arg_{i}")

        for key, value in kwargs.items():
        if isinstance(value, pd.DataFrame):
                    data_args.append(key)

        if data_args:
                print(f"📊 [PIPELINE INPUT] Found data arguments: {data_args}")
                logger.info(f"📊 [PIPELINE INPUT] Found data arguments: {data_args}")

        for data_arg in data_args:
        if data_arg in kwargs:
                        data, kwargs[data_arg]
                    else:
        # Extract from positional args
                        arg_index, int(data_arg.split("_")[1])
                        data, args[arg_index]

                    print(
                        f"📊 [PIPELINE INPUT] Validating {data_arg}: shape={data.shape}, columns={list(data.columns)}"
                    )
                    logger.info(
                        f"📊 [PIPELINE INPUT] Validating {data_arg}: shape={data.shape}, columns={list(data.columns)}"
                    )

        # Basic data validation
        if data.empty:
                        error_msg, f"Empty DataFrame in {data_arg}"
                        print(f"❌ [PIPELINE INPUT] {error_msg}")
                        logger.error(f"❌ [PIPELINE INPUT] {error_msg}")
                        raise ValueError(error_msg)

        if data.isnull().all().all():
                        warning_msg, f"All null values in {data_arg}"
                        print(f"⚠️ [PIPELINE INPUT] {warning_msg}")
                        logger.warning(f"⚠️ [PIPELINE INPUT] {warning_msg}")

                    print(f"✅ [PIPELINE INPUT] {data_arg} validation passed")
                    logger.info(f"✅ [PIPELINE INPUT] {data_arg} validation passed")

        print(f"✅ [PIPELINE INPUT] Input validation completed for {method_name}")
        logger.info(f"✅ [PIPELINE INPUT] Input validation completed for {method_name}")

        # Execute the function
        return await _execute_pipeline_function(func, self, args, kwargs)

    except Exception as e:
        print(
            f"💥 [PIPELINE INPUT] Input validation failed for {method_name}: {str(e)}"
        )
        logger.error(
            f"💥 [PIPELINE INPUT] Input validation failed for {method_name}: {str(e)}"
        )
        raise

def monitor_pipeline_performance(
    enable_memory_tracking: bool, True,
    enable_cpu_tracking: bool, True,
    enable_gc_tracking: bool, True,
    memory_threshold_gb: float, 32.0,
    cpu_threshold_percent: float, 90.0,
):
    """
    Decorator to monitor pipeline performance metrics.

    Args:
        enable_memory_tracking: Whether to track memory usage
        enable_cpu_tracking: Whether to track CPU usage
        enable_gc_tracking: Whether to track garbage collection
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(self: Any, *args, **kwargs) -> Any:
        return await _monitor_performance_and_execute(
                func,
                self,
                args,
                kwargs,
                enable_memory_tracking,
                enable_cpu_tracking,
                enable_gc_tracking,
                memory_threshold_gb,
                cpu_threshold_percent,
            )

        @functools.wraps(func)
        def sync_wrapper(self: Any, *args, **kwargs) -> Any:
        return asyncio.run(
                _monitor_performance_and_execute(
                    func,
                    self,
                    args,
                    kwargs,
                    enable_memory_tracking,
                    enable_cpu_tracking,
                    enable_gc_tracking,
                    memory_threshold_gb,
                    cpu_threshold_percent,
                )
            )

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
        return async_wrapper
        else:
        return sync_wrapper

    return decorator

async def _monitor_performance_and_execute(
    func: Callable,
    self: Any,
    args: tuple,
    kwargs: dict,
    enable_memory_tracking: bool,
    enable_cpu_tracking: bool,
    enable_gc_tracking: bool,
    memory_threshold_gb: float,
    cpu_threshold_percent: float,
) -> Any:
    """
    Monitor performance and execute the function.
    """
    logger, system_logger.getChild("PerformanceMonitor")
    method_name, func.__name__

    print(f"📊 [PERFORMANCE] Starting performance monitoring for {method_name}")
    logger.info(f"📊 [PERFORMANCE] Starting performance monitoring for {method_name}")

    # Pre - execution metrics
    pre_metrics = {}

    if enable_memory_tracking:
        pre_metrics["memory"] = psutil.virtual_memory().percent
        print(f"💾 [PERFORMANCE] Pre - execution memory: {pre_metrics['memory']:.1f}%")
        logger.info(
            f"💾 [PERFORMANCE] Pre - execution memory: {pre_metrics['memory']:.1f}%"
        )

    if enable_cpu_tracking:
        pre_metrics["cpu"] = psutil.cpu_percent()
        print(f"🔥 [PERFORMANCE] Pre - execution CPU: {pre_metrics['cpu']:.1f}%")
        logger.info(f"🔥 [PERFORMANCE] Pre - execution CPU: {pre_metrics['cpu']:.1f}%")

    if enable_gc_tracking:
        pre_metrics["gc_counts"] = gc.get_count()
        print(f"🗑️ [PERFORMANCE] Pre - execution GC counts: {pre_metrics['gc_counts']}")
        logger.info(
            f"🗑️ [PERFORMANCE] Pre - execution GC counts: {pre_metrics['gc_counts']}"
        )

    # Check thresholds
    if enable_memory_tracking:
        available_memory_gb, psutil.virtual_memory().available / (1024**3)
        if available_memory_gb < memory_threshold_gb:
            warning_msg, f"Available memory ({available_memory_gb:.1f}GB) below threshold ({memory_threshold_gb:.1f}GB)"
            print(f"⚠️ [PERFORMANCE] {warning_msg}")
            logger.warning(f"⚠️ [PERFORMANCE] {warning_msg}")

    if enable_cpu_tracking:
        cpu_usage, psutil.cpu_percent()
        if cpu_usage > cpu_threshold_percent:
            warning_msg, f"CPU usage ({cpu_usage:.1f}%) above threshold ({cpu_threshold_percent:.1f}%)"
            print(f"⚠️ [PERFORMANCE] {warning_msg}")
            logger.warning(f"⚠️ [PERFORMANCE] {warning_msg}")

    start_time, time.time()

    try:
        # Execute the function
        result, await _execute_pipeline_function(func, self, args, kwargs)

        # Post - execution metrics
        end_time, time.time()
        duration, end_time - start_time

        print(f"📊 [PERFORMANCE] Performance monitoring completed for {method_name}")
        logger.info(
            f"📊 [PERFORMANCE] Performance monitoring completed for {method_name}"
        )

        print(f"⏱️ [PERFORMANCE] Execution time: {duration:.2f}s")
        logger.info(f"⏱️ [PERFORMANCE] Execution time: {duration:.2f}s")

        if enable_memory_tracking:
            post_memory, psutil.virtual_memory().percent
            memory_delta, post_memory - pre_metrics["memory"]
            print(
                f"💾 [PERFORMANCE] Post - execution memory: {post_memory:.1f}% (delta: {memory_delta:+.1f}%)"
            )
            logger.info(
                f"💾 [PERFORMANCE] Post - execution memory: {post_memory:.1f}% (delta: {memory_delta:+.1f}%)"
            )

        if enable_cpu_tracking:
            post_cpu, psutil.cpu_percent()
            cpu_delta, post_cpu - pre_metrics["cpu"]
            print(
                f"🔥 [PERFORMANCE] Post - execution CPU: {post_cpu:.1f}% (delta: {cpu_delta:+.1f}%)"
            )
            logger.info(
                f"🔥 [PERFORMANCE] Post - execution CPU: {post_cpu:.1f}% (delta: {cpu_delta:+.1f}%)"
            )

        if enable_gc_tracking:
            post_gc_counts, gc.get_count()
            gc_delta, tuple(
                post - pre
        for post, pre in zip(post_gc_counts, pre_metrics["gc_counts"])
            )
            print(
                f"🗑️ [PERFORMANCE] Post - execution GC counts: {post_gc_counts} (delta: {gc_delta})"
            )
            logger.info(
                f"🗑️ [PERFORMANCE] Post - execution GC counts: {post_gc_counts} (delta: {gc_delta})"
            )

        return result

    except Exception as e:
        end_time, time.time()
        duration, end_time - start_time

        print(f"💥 [PERFORMANCE] Performance monitoring failed for {method_name}")
        logger.error(
            f"💥 [PERFORMANCE] Performance monitoring failed for {method_name}"
        )
        print(f"⏱️ [PERFORMANCE] Execution time before error: {duration:.2f}s")
        logger.error(f"⏱️ [PERFORMANCE] Execution time before error: {duration:.2f}s")

        raise

# Utility functions for pipeline monitoring
def start_pipeline_monitoring(pipeline_name: str):
    """Start monitoring a pipeline."""
    _pipeline_monitor.start_pipeline(pipeline_name)

def end_pipeline_monitoring(pipeline_name: str, success: bool, True):
    """End monitoring a pipeline."""
    _pipeline_monitor.end_pipeline(pipeline_name, success)

def get_pipeline_metrics() -> Dict[str, Any]:
    """Get current pipeline metrics."""
    return {
        "step_count": _pipeline_monitor.step_count,
        "current_stage": _pipeline_monitor.current_stage.value
        if _pipeline_monitor.current_stage
        else None,
        "errors": len(_pipeline_monitor.metrics.errors),
        "warnings": len(_pipeline_monitor.metrics.warnings),
        "step_durations": _pipeline_monitor.metrics.step_durations,
    }

def clear_pipeline_metrics():
    """Clear pipeline metrics."""
    _pipeline_monitor.metrics, PipelineMetrics()
    _pipeline_monitor.step_count, 0
    _pipeline_monitor.current_stage, None

    print(f"🗑️ [PIPELINE] Pipeline metrics cleared")
    system_logger.info(f"🗑️ [PIPELINE] Pipeline metrics cleared")

# Convenience decorators for common pipeline stages
def monitor_data_collection(
    validation_level: PipelineValidationLevel, PipelineValidationLevel.WARNING,
):
    """Decorator for data collection steps."""
    return monitor_pipeline_step(
        PipelineStage.DATA_COLLECTION,
        validation_level = validation_level,
        enable_data_quality = True,
    )

def monitor_feature_engineering(
    validation_level: PipelineValidationLevel, PipelineValidationLevel.WARNING,
):
    """Decorator for feature engineering steps."""
    return monitor_pipeline_step(
        PipelineStage.FEATURE_ENGINEERING,
        validation_level = validation_level,
        enable_data_quality = True,
    )

def monitor_model_training(
    validation_level: PipelineValidationLevel, PipelineValidationLevel.WARNING,
):
    """Decorator for model training steps."""
    return monitor_pipeline_step(
        PipelineStage.MODEL_TRAINING,
        validation_level = validation_level,
        enable_data_quality = False,  # Usually no data quality validation needed for training
    )

def monitor_validation(
    validation_level: PipelineValidationLevel, PipelineValidationLevel.WARNING,
):
    """Decorator for validation steps."""
    return monitor_pipeline_step(
        PipelineStage.VALIDATION,
        validation_level = validation_level,
        enable_data_quality = False,
    )

def monitor_optimization(
    validation_level: PipelineValidationLevel, PipelineValidationLevel.WARNING,
):
    """Decorator for optimization steps."""
    return monitor_pipeline_step(
        PipelineStage.OPTIMIZATION,
        validation_level = validation_level,
        enable_data_quality = False,
    )

def monitor_step_execution(
    step_name: str, None,
    enable_timing: bool, True,
    enable_memory_monitoring: bool, True,
    enable_progress_tracking: bool, True,
    log_level: str = "INFO",
):
    """Decorator to monitor step execution."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def secure_step_execution(
    error_handling: bool, True,
    rollback_on_failure: bool, True,
    data_validation: bool, True,
    resource_cleanup: bool, True,
):
    """Decorator to ensure secure step execution."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
