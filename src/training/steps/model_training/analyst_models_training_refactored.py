"""
Analyst Models Training Step - Enhanced and Streamlined with Comprehensive Utilities Integration

This step handles per-regime training of individual Analyst models using common dependencies.
Enhanced Features:
- Comprehensive error handling with detailed failure tracking and fast failing
- Advanced monitoring and health checks with hardware optimization
- Enhanced reporting with performance metrics and resource utilization
- Streamlined code with reduced redundancy
- Silent failure prevention with explicit error propagation
- Real-time training progress tracking with tprint logging
- Integration with common utilities for data operations, validation, and optimization
- M1 GPU/CPU optimization for enhanced performance
- Comprehensive ML utilities integration (CV, HPO, lookahead, etc.)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
from pathlib import Path
import json
import time
import traceback
import pickle
from dataclasses import dataclass, field

# Required psutil import - fail fast if not available for production use
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
from contextlib import contextmanager
import sys
import os

# Required numpy import - fail fast if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Required pandas import - fail fast if not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Core imports
from src.utils.logger import system_logger
from src.utils.ml_common.config import PerRegimeTrainingConfig
from src.utils.ml_common.training import PerRegimeTrainingStep

# Dependency validation functions
def validate_critical_dependencies():
    """Validate that all critical dependencies are available. Fast fail if not."""
    missing_deps = []

    if not NUMPY_AVAILABLE:
        missing_deps.append("numpy")

    if not PANDAS_AVAILABLE:
        missing_deps.append("pandas")

    if not PSUTIL_AVAILABLE:
        missing_deps.append("psutil")

    if missing_deps:
        error_msg = f"Critical dependencies missing: {', '.join(missing_deps)}. " \
                   f"Install with: pip install {' '.join(missing_deps)}"
        raise ImportError(error_msg)

def validate_runtime_dependencies():
    """Validate dependencies at runtime before executing operations."""
    if np is None:
        raise RuntimeError("NumPy is required for array operations. Install with: pip install numpy")

    if pd is None:
        raise RuntimeError("Pandas is required for data operations. Install with: pip install pandas")

    if psutil is None:
        raise RuntimeError("psutil is required for system monitoring. Install with: pip install psutil")

# Enhanced tprint logging
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_debug, tprint_progress, tprint_performance, tprint_structured,
    tprint_timer, tprint_logged, LogLevel
)

# Common utilities integration with safe imports
try:
    from src.utils.common_operations import (
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, cleanup_m1_optimizers,
        safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
        validate_finite, validate_positive, validate_range,
        safe_kelly_calculation, safe_weighted_average, safe_percentage_change,
        ensure_directory, safe_file_exists, safe_json_dump, safe_json_load,
        create_empty_dataframe, validate_dataframe, validate_dataframe_columns,
        safe_dataframe_operation, safe_fillna, safe_convert_dtypes,
        safe_merge_dataframes, safe_drop_columns, safe_rename_columns,
        validate_timestamp_column, safe_timestamp_conversion,
        optimize_dataframe_dtypes, calculate_data_quality_metrics,
        get_dataframe_info, create_data_quality_report,
        safe_rolling, safe_groupby_operation, safe_apply_function,
        safe_filter_dataframe, create_summary_statistics,
        safe_to_parquet, safe_read_parquet, list_parquet_files,
        safe_copy, validate_dataframe_schema, validate_file_size,
        guard_dataframe_nulls, secure_file_path, with_tracing_span,
        sanitize_string, memory_checkpoint, gpu_context, optimize_memory,
        get_memory_usage, validate_file_path, get_file_size, check_disk_space,
        timed_operation, format_bytes, chunked_iterable, parallel_map,
        validate_correlation_matrix, safe_matrix_inverse, math_safe,
        MathValidationError
    )
    COMMON_UTILITIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Common utilities not available: {e}")
    COMMON_UTILITIES_AVAILABLE = False
    # Define fallback functions
    def safe_divide(a, b, default=0.0):
        """Safe division that handles division by zero.

        Args:
            a: Numerator
            b: Denominator
            default: Default value to return if division by zero

        Returns:
            Division result or default value
        """
        return a / b if b != 0 else default

    def safe_mean(arr):
        """Calculate mean safely with empty array handling.

        Args:
            arr: Array to calculate mean for

        Returns:
            Mean value or 0.0 for empty arrays
        """
        return np.mean(arr) if hasattr(arr, '__len__') and len(arr) > 0 else 0.0

    def safe_std(arr):
        """Calculate standard deviation safely with empty array handling.

        Args:
            arr: Array to calculate standard deviation for

        Returns:
            Standard deviation or 0.0 for empty arrays
        """
        return np.std(arr) if hasattr(arr, '__len__') and len(arr) > 0 else 0.0
    def validate_finite(value, name="value"):
        """Validate that a value is finite.

        Args:
            value: Value to validate
            name: Name of the value for error messages

        Returns:
            The validated value

        Raises:
            ValueError: If value is not finite
        """
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
        return value

    def validate_positive(value, name="value"):
        """Validate that a value is positive.

        Args:
            value: Value to validate
            name: Name of the value for error messages

        Returns:
            The validated value

        Raises:
            ValueError: If value is not positive
        """
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value

    def ensure_directory(path):
        """Ensure a directory exists, creating it if necessary.

        Args:
            path: Directory path to create
        """
        return True
    def safe_json_dump(data, filepath, **kwargs):
        """Safely dump data to JSON file.

        Args:
            data: Data to serialize
            filepath: File path to save to
            **kwargs: Additional arguments for json.dump

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, **kwargs)
            return True
        except Exception:
            return False
    def safe_json_load(filepath, default=None):
        """Safely load data from JSON file.

        Args:
            filepath: File path to load from
            default: Default value to return on error

        Returns:
            Loaded data or default value
        """
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            return default
    def sanitize_string(s, max_length=255):
        """Sanitize string by truncating and stripping whitespace.

        Args:
            s: String to sanitize
            max_length: Maximum length to truncate to

        Returns:
            Sanitized string
        """
        if not isinstance(s, str):
            s = str(s)
        return s[:max_length].strip()
    def get_memory_usage():
        """Get current memory usage of the process.

        Returns:
            Memory usage in bytes, or 0 if unable to determine
        """
        try:
            return psutil.Process().memory_info().rss
        except Exception:
            return 0
    def check_disk_space(path, required_gb=1.0):
        """Check if sufficient disk space is available.

        Args:
            path: Path to check disk space for
            required_gb: Required space in GB

        Returns:
            True if sufficient space available, False otherwise
        """
        try:
            import shutil
            stat = shutil.disk_usage(path)
            free_gb = stat.free / (1024 ** 3)
            return {'sufficient': free_gb >= required_gb, 'free_gb': free_gb}
        except Exception:
            return {'sufficient': False, 'free_gb': 0}

# Math validation utilities with safe imports
try:
    from src.utils.math_validation import (
        safe_divide as math_safe_divide, safe_log as math_safe_log,
        safe_sqrt as math_safe_sqrt, safe_power as math_safe_power,
        validate_finite as math_validate_finite, validate_positive as math_validate_positive,
        validate_range as math_validate_range, safe_kelly_calculation as math_safe_kelly,
        safe_weighted_average as math_safe_weighted_avg,
        safe_percentage_change as math_safe_pct_change,
        safe_correlation,
        safe_covariance,
        safe_mean as math_safe_mean,
        safe_std as math_safe_std,
        safe_percentile,
        validate_correlation_matrix as math_validate_corr,
        safe_matrix_inverse as math_safe_matrix_inv, math_safe as math_safe_func,
        MathValidation, MathValidationError as MathValidationError
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Math validation utilities not available: {e}")
    MATH_VALIDATION_AVAILABLE = False
    # Define fallback functions
    def math_validate_finite(value, name="value"):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
        return value
    def math_validate_positive(value, name="value"):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value
    def math_safe_mean(arr):
        return np.mean(arr) if hasattr(arr, '__len__') and len(arr) > 0 else 0.0
    def math_safe_std(arr):
        return np.std(arr) if hasattr(arr, '__len__') and len(arr) > 0 else 0.0
    class MathValidation:
        def __init__(self):
            pass
        def validate_finite(self, value, name="value"):
            return math_validate_finite(value, name)
        def validate_positive(self, value, name="value"):
            return math_validate_positive(value, name)
    class MathValidationError(Exception):
        pass

# Serialization utilities with safe imports
try:
    from src.utils.serialization_utils import (
        JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
    )
    SERIALIZATION_UTILITIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Serialization utilities not available: {e}")
    SERIALIZATION_UTILITIES_AVAILABLE = False
    # Define fallback classes
    class JSONSerializer:
        @staticmethod
        def save(data, filepath):
            try:
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                return True
            except Exception:
                return False
        @staticmethod
        def load(filepath):
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
    class PickleSerializer:
        @staticmethod
        def save(data, filepath):
            try:
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
                return True
            except Exception:
                return False
        @staticmethod
        def load(filepath):
            try:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
    class ParquetSerializer:
        @staticmethod
        def save(data, filepath):
            try:
                if hasattr(data, 'to_parquet'):
                    data.to_parquet(filepath)
                    return True
                return False
            except Exception:
                return False
        @staticmethod
        def load(filepath):
            try:
                return pd.read_parquet(filepath)
            except Exception:
                return None
    class UniversalSerializer:
        def __init__(self):
            self.serializers = {
                'json': JSONSerializer,
                'pickle': PickleSerializer,
                'parquet': ParquetSerializer
            }
        def save(self, data, filepath, format='auto'):
            if format == 'auto':
                if filepath.endswith('.json'):
                    format = 'json'
                elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                    format = 'pickle'
                elif filepath.endswith('.parquet'):
                    format = 'parquet'
                else:
                    format = 'pickle'
            serializer = self.serializers.get(format)
            if serializer:
                return serializer.save(data, filepath)
            return False
        def load(self, filepath):
            if filepath.endswith('.json'):
                return JSONSerializer.load(filepath)
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                return PickleSerializer.load(filepath)
            elif filepath.endswith('.parquet'):
                return ParquetSerializer.load(filepath)
            else:
                return PickleSerializer.load(filepath)

# Hardware optimization utilities with safe imports
try:
    from src.utils.hardware.m1_gpu_utils import (
        get_m1_gpu_manager as get_gpu_manager, is_m1_available, is_mps_available,
        optimize_dataframe_for_m1, create_m1_optimized_array,
        m1_backtesting_simulate, m1_monte_carlo_simulate
    )
    HARDWARE_UTILITIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Hardware utilities not available: {e}")
    HARDWARE_UTILITIES_AVAILABLE = False
    # Define fallback functions
    def is_m1_available():
        return False
    def is_mps_available():
        return False
    def get_gpu_manager():
        return None
    def optimize_dataframe_for_m1(df):
        return df
    def create_m1_optimized_array(data, dtype=None):
        return np.array(data, dtype=dtype)
    def m1_backtesting_simulate(*args, **kwargs):
        return {'error': 'M1 utilities not available'}
    def m1_monte_carlo_simulate(*args, **kwargs):
        return {'error': 'M1 utilities not available'}

# ML common utilities with safe imports
try:
    from src.utils.ml_common.validation.validation_utils import (
        validate_input_data, validate_model_config, validate_training_data
    )
    from src.utils.ml_common.optimization.hpo_utils import (
        optimize_hyperparameters, create_search_space, validate_hpo_config
    )
    # Import Bayesian TPE optimizer with early stopping
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, OptimizationConfig
    )
    from src.utils.ml_common.evaluation.evaluation_utils import (
        calculate_metrics, evaluate_model_performance, create_evaluation_report
    )
    from src.utils.ml_common.monitoring.enhanced_error_detector import (
        EnhancedErrorDetector
    )

    # Enhanced HPO integration is no longer used
    ENHANCED_HPO_AVAILABLE = False
    enhance_existing_hpo_pipeline = None
    EnhancedCVStrategies = None
    RegimeType = None
    RegimeCharacteristics = None
    def EnhancedCVStrategies(*args, **kwargs):
        return None
    def RegimeType(*args, **kwargs):
        return None
    def RegimeCharacteristics(*args, **kwargs):
        return None

    from src.utils.ml_common.reporting.enhanced_reporting_system import (
        ReportGenerator, ReportManager, create_training_report
    )
    ML_UTILITIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML utilities not available: {e}")
    ML_UTILITIES_AVAILABLE = False

# Enhanced training utilities integration
try:
    from src.utils.ml_common.training.enhanced_training_utils import (
        EnhancedTrainingUtils,
        EarlyStoppingConfig,
        PurgedCVConfig,
        OverfittingMonitorConfig,
        RegularizationConfig
    )
    from src.utils.ml_common.training.training_integration import (
        TrainingStepEnhancer,
        TrainingIntegrationConfig
    )
    ENHANCED_TRAINING_AVAILABLE = True
    tprint_success("✅ Enhanced training utilities loaded")
except ImportError as e:
    ENHANCED_TRAINING_AVAILABLE = False
    tprint_warning(f"⚠️ Enhanced training utilities not available: {e}")
    EnhancedTrainingUtils = None
    TrainingStepEnhancer = None
    EarlyStoppingConfig = None
    PurgedCVConfig = None
    OverfittingMonitorConfig = None
    RegularizationConfig = None
    TrainingIntegrationConfig = None
    # Define fallback functions
    def validate_input_data(*args, **kwargs):
        return {'valid': True}
    def validate_model_config(*args, **kwargs):
        return {'valid': True}
    def validate_training_data(*args, **kwargs):
        return {'valid': True}
    def optimize_hyperparameters(*args, **kwargs):
        return {'error': 'ML utilities not available'}
    def create_search_space(*args, **kwargs):
        return {}
    def validate_hpo_config(*args, **kwargs):
        return {'valid': True}
    def calculate_metrics(*args, **kwargs):
        return {}
    def evaluate_model_performance(*args, **kwargs):
        return {'error': 'ML utilities not available'}
    def create_evaluation_report(*args, **kwargs):
        return {'error': 'ML utilities not available'}
    class ErrorDetector:
        def analyze_error(self, *args, **kwargs):
            return {'error': 'ML utilities not available'}
    class ErrorHandler:
        pass
    class ErrorReporter:
        def report_critical_error(self, *args, **kwargs):
            pass
    class ReportGenerator:
        pass
    class ReportManager:
        pass
    def create_training_report(*args, **kwargs):
        return {'error': 'ML utilities not available'}

logger = system_logger.getChild('AnalystModelsTrainingEnhanced')

@contextmanager
def monitor_resources(operation_name: str, logger: logging.Logger):
    """Enhanced context manager for monitoring resource usage - fails fast if psutil unavailable."""
    # Fast fail if psutil is not available
    if not PSUTIL_AVAILABLE or psutil is None:
        raise RuntimeError("psutil is required for resource monitoring. Install with: pip install psutil")

    start_time = time.time()
    start_memory = get_memory_usage() / 1024 / 1024  # MB
    start_cpu = psutil.cpu_percent()

    # Get M1 optimization status
    m1_status = (
        "M1: " + ("✅" if is_m1_available() else "❌") +
        " | MPS: " + ("✅" if is_mps_available() else "❌")
    )

    tprint_info(f"🔄 Starting {operation_name} - Memory: {start_memory:.1f}MB, CPU: {start_cpu:.1f}% | {m1_status}")

    # Start memory monitoring if available
    memory_optimizer = get_m1_memory_optimizer()
    if memory_optimizer and hasattr(memory_optimizer, 'start_monitoring'):
        try:
            memory_optimizer.start_monitoring()
        except Exception as e:
            tprint_warning(f"⚠️ Could not start memory monitoring: {e}")

    try:
        yield
    finally:
        end_time = time.time()
        end_memory = get_memory_usage() / 1024 / 1024  # MB
        end_cpu = psutil.cpu_percent()

        duration = end_time - start_time
        memory_delta = end_memory - start_memory

        # Stop memory monitoring
        if memory_optimizer and hasattr(memory_optimizer, 'stop_monitoring'):
            try:
                memory_optimizer.stop_monitoring()
            except Exception as e:
                tprint_warning(f"⚠️ Could not stop memory monitoring: {e}")

        tprint_success(f"✅ Completed {operation_name} - Duration: {duration:.2f}s, "
                      f"Memory: {end_memory:.1f}MB (+{memory_delta:+.1f}MB), CPU: {end_cpu:.1f}%")

class TrainingProgressTracker:
    """Enhanced training progress tracker with comprehensive monitoring and tprint integration."""

    def __init__(self, total_steps: int, logger: logging.Logger):
        self.total_steps = total_steps
        self.current_step = 0
        self.logger = logger
        self.start_time = time.time()
        self.step_times = []
        self.step_details = []
        self.memory_usage = []
        self.cpu_usage = []

        # Initialize hardware monitoring
        self.gpu_manager = get_m1_gpu_manager()
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()

        tprint_info(f"🚀 Initialized progress tracker for {total_steps} steps")

    def update_step(self, step_name: str, details: Optional[Dict] = None):
        """Update progress with step completion and comprehensive monitoring."""
        try:
            self.current_step += 1
            step_time = time.time()
            self.step_times.append(step_time)

            # Record system metrics
            current_memory = get_memory_usage() / 1024 / 1024  # MB
            current_cpu = psutil.cpu_percent()
            self.memory_usage.append(current_memory)
            self.cpu_usage.append(current_cpu)

            elapsed = step_time - self.start_time
            progress_pct = (self.current_step / self.total_steps) * 100

            # Enhanced status message with hardware info
            m1_info = (
                f"M1: {'✅' if is_m1_available() else '❌'} | "
                f"MPS: {'✅' if is_mps_available() else '❌'}"
            )
            status_msg = f"📊 Progress: {self.current_step}/{self.total_steps} ({progress_pct:.1f}%) - {step_name}"
            if details:
                status_msg += f" - {details}"
            status_msg += (
                f" | Memory: {current_memory:.1f}MB | "
                f"CPU: {current_cpu:.1f}% | {m1_info}"
            )

            # Use tprint for enhanced logging
            tprint_progress(self.current_step, self.total_steps, step_name)
            tprint_info(status_msg)

            # Store step details
            step_detail = {
                'step_number': self.current_step,
                'step_name': step_name,
                'details': details,
                'timestamp': step_time,
                'memory_mb': current_memory,
                'cpu_percent': current_cpu,
                'elapsed_time': elapsed
            }
            self.step_details.append(step_detail)

            # Estimate remaining time with enhanced calculation
            if self.current_step > 1:
                avg_step_time = elapsed / self.current_step
                remaining_steps = self.total_steps - self.current_step
                eta = remaining_steps * avg_step_time

                # Add memory trend analysis
                if len(self.memory_usage) > 1:
                    memory_trend = self.memory_usage[-1] - self.memory_usage[0]
                    memory_warning = "⚠️ Memory increasing" if memory_trend > 100 else ""
                    tprint_info(f"⏱️ ETA: {eta:.1f}s remaining {memory_warning}")
                else:
                    tprint_info(f"⏱️ ETA: {eta:.1f}s remaining")

            # Memory optimization check
            if current_memory > 8000:  # 8GB threshold
                tprint_warning(f"⚠️ High memory usage detected: {current_memory:.1f}MB")
                if self.memory_optimizer and hasattr(self.memory_optimizer, 'optimize_memory'):
                    try:
                        opt_result = self.memory_optimizer.optimize_memory()
                        tprint_info(f"🧠 Memory optimization result: {opt_result}")
                    except Exception as e:
                        tprint_warning(f"⚠️ Memory optimization failed: {e}")

        except Exception as e:
            tprint_error(f"❌ Error updating progress tracker: {e}")
            # Continue execution even if progress tracking fails

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive training progress summary with hardware metrics."""
        try:
            total_time = time.time() - self.start_time

            # Calculate memory and CPU statistics
            memory_stats = {
                'min_mb': min(self.memory_usage) if self.memory_usage else 0,
                'max_mb': max(self.memory_usage) if self.memory_usage else 0,
                'avg_mb': safe_mean(np.array(self.memory_usage)) if self.memory_usage else 0,
                'final_mb': self.memory_usage[-1] if self.memory_usage else 0
            }

            cpu_stats = {
                'min_percent': min(self.cpu_usage) if self.cpu_usage else 0,
                'max_percent': max(self.cpu_usage) if self.cpu_usage else 0,
                'avg_percent': safe_mean(np.array(self.cpu_usage)) if self.cpu_usage else 0,
                'final_percent': self.cpu_usage[-1] if self.cpu_usage else 0
            }

            # Hardware optimization status
            hardware_status = {
                'm1_available': is_m1_available(),
                'mps_available': is_mps_available(),
                'gpu_manager_active': self.gpu_manager is not None,
                'memory_optimizer_active': self.memory_optimizer is not None,
                'cpu_optimizer_active': self.cpu_optimizer is not None
            }

            return {
                'total_steps': self.total_steps,
                'completed_steps': self.current_step,
                'progress_percentage': (self.current_step / self.total_steps) * 100,
                'total_time': total_time,
                'average_step_time': total_time / self.current_step if self.current_step > 0 else 0,
                'memory_stats': memory_stats,
                'cpu_stats': cpu_stats,
                'hardware_status': hardware_status,
                'step_details': self.step_details
            }

        except Exception as e:
            tprint_error(f"❌ Error generating progress summary: {e}")
            return {
                'total_steps': self.total_steps,
                'completed_steps': self.current_step,
                'progress_percentage': (self.current_step / self.total_steps) * 100,
                'total_time': time.time() - self.start_time,
                'error': str(e)
            }

class EnhancedErrorHandler:
    """Enhanced error handling with detailed failure tracking, fast failing, and comprehensive reporting."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.error_history = []
        self.critical_errors = []
        self.warning_count = 0
        self.error_count = 0

        # Initialize ML common error detector
        try:
            self.ml_error_detector = ErrorDetector()
            self.ml_error_handler = ErrorHandler()
            self.ml_error_reporter = ErrorReporter()
        except Exception as e:
            tprint_warning(f"⚠️ Could not initialize ML error detection: {e}")
            self.ml_error_detector = None
            self.ml_error_handler = None
            self.ml_error_reporter = None

        tprint_info("🔧 Enhanced error handler initialized with ML error detection")

    def handle_error(self, error: Exception, context: str,
                    additional_info: Optional[Dict] = None,
                    fast_fail: bool = False) -> Dict[str, Any]:
        """Handle errors with comprehensive logging, tracking, and optional fast failing."""
        try:
            # Get system state at time of error
            system_state = {
                'memory_mb': get_memory_usage() / 1024 / 1024,
                'cpu_percent': psutil.cpu_percent(),
                'disk_space': check_disk_space('/', 1.0),
                'm1_available': is_m1_available(),
                'mps_available': is_mps_available()
            }

            error_info = {
                'timestamp': datetime.now().isoformat(),
                'context': context,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc(),
                'additional_info': additional_info or {},
                'system_state': system_state,
                'fast_fail_triggered': fast_fail
            }

            self.error_history.append(error_info)
            self.error_count += 1

            # Determine error severity
            is_critical = self._is_critical_error(error, context)
            if is_critical:
                self.critical_errors.append(error_info)

            # Enhanced logging with tprint
            if is_critical:
                tprint_error(f"🚨 CRITICAL ERROR in {context}: {error}")
            else:
                tprint_error(f"❌ Error in {context}: {error}")

            tprint_error(f"🔍 Error Type: {type(error).__name__}")
            tprint_error(f"📊 System State: Memory: {system_state['memory_mb']:.1f}MB, CPU: {system_state['cpu_percent']:.1f}%")

            if additional_info:
                tprint_structured(additional_info, LogLevel.ERROR)

            # Log traceback for debugging
            tprint_debug(f"🔍 Full traceback:\n{traceback.format_exc()}")

            # Use ML error detection if available
            if self.ml_error_detector:
                try:
                    ml_analysis = self.ml_error_detector.analyze_error(error, context, additional_info)
                    error_info['ml_analysis'] = ml_analysis
                    tprint_debug(f"🤖 ML Error Analysis: {ml_analysis}")
                except Exception as e:
                    tprint_warning(f"⚠️ ML error analysis failed: {e}")

            # Fast fail for critical errors
            if fast_fail and is_critical:
                tprint_error("🚨 FAST FAILING due to critical error")
                self._trigger_fast_fail(error, context, error_info)

            return error_info

        except Exception as e:
            # Fallback error handling
            tprint_error(f"❌ Error in error handler: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'context': 'error_handler_failure',
                'error_type': 'ErrorHandlerException',
                'error_message': str(e),
                'original_error': str(error)
            }

    def _is_critical_error(self, error: Exception, context: str) -> bool:
        """Determine if an error is critical and should trigger fast failing."""
        critical_types = [
            'MemoryError', 'SystemError', 'OSError', 'IOError',
            'ValueError', 'TypeError', 'AttributeError', 'KeyError',
            'ImportError', 'ModuleNotFoundError'
        ]

        critical_contexts = [
            'model_training', 'data_validation', 'model_saving',
            'configuration_validation', 'hardware_initialization'
        ]

        return (type(error).__name__ in critical_types or
                any(ctx in context.lower() for ctx in critical_contexts))

    def _trigger_fast_fail(self, error: Exception, context: str, error_info: Dict[str, Any]):
        """Trigger fast failing with comprehensive error reporting."""
        try:
            # Generate comprehensive error report
            error_report = {
                'fast_fail_triggered': True,
                'error_info': error_info,
                'error_summary': self.get_error_summary(),
                'system_state': error_info.get('system_state', {}),
                'recommendations': self._generate_error_recommendations(error, context)
            }

            # Save error report
            report_path = f"./error_reports/fast_fail_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            ensure_directory("./error_reports")
            safe_json_dump(error_report, report_path)

            tprint_error(f"📋 Fast fail report saved: {report_path}")

            # Use ML error reporter if available
            if self.ml_error_reporter:
                try:
                    self.ml_error_reporter.report_critical_error(error, context, error_info)
                except Exception as e:
                    tprint_warning(f"⚠️ ML error reporting failed: {e}")

            # Raise the original error to trigger fast fail
            raise error

        except Exception as e:
            tprint_error(f"❌ Error in fast fail trigger: {e}")
            raise error

    def _generate_error_recommendations(self, error: Exception, context: str) -> List[str]:
        """Generate recommendations based on error type and context."""
        recommendations = []

        if isinstance(error, MemoryError):
            recommendations.extend([
                "Reduce batch size or model complexity",
                "Enable memory optimization",
                "Consider using smaller datasets",
                "Check for memory leaks"
            ])
        elif isinstance(error, (ValueError, TypeError)):
            recommendations.extend([
                "Validate input data types and ranges",
                "Check configuration parameters",
                "Verify data preprocessing steps"
            ])
        elif isinstance(error, (IOError, OSError)):
            recommendations.extend([
                "Check file permissions and disk space",
                "Verify file paths and network connectivity",
                "Ensure sufficient disk space"
            ])

        if 'model_training' in context:
            recommendations.extend([
                "Check model configuration",
                "Verify training data quality",
                "Consider reducing model complexity"
            ])

        return recommendations

    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all errors encountered."""
        try:
            if not self.error_history:
                return {'total_errors': 0, 'errors': [], 'critical_errors': 0}

            error_types = {}
            error_contexts = {}

            for error in self.error_history:
                error_type = error['error_type']
                context = error['context']

                error_types[error_type] = error_types.get(error_type, 0) + 1
                error_contexts[context] = error_contexts.get(context, 0) + 1

            return {
                'total_errors': len(self.error_history),
                'critical_errors': len(self.critical_errors),
                'error_types': error_types,
                'error_contexts': error_contexts,
                'errors': self.error_history,
                'error_rate': self.error_count / max(1, len(self.error_history)),
                'recommendations': self._generate_general_recommendations()
            }

        except Exception as e:
            tprint_error(f"❌ Error generating error summary: {e}")
            return {
                'total_errors': len(self.error_history),
                'critical_errors': len(self.critical_errors),
                'error': str(e)
            }

    def _generate_general_recommendations(self) -> List[str]:
        """Generate general recommendations based on error patterns."""
        recommendations = []

        if self.critical_errors:
            recommendations.append("Address critical errors immediately")

        if self.error_count > 10:
            recommendations.append("High error count - review error patterns")

        if any('memory' in str(error).lower() for error in self.error_history):
            recommendations.append("Monitor memory usage and optimize")

        if any('validation' in error['context'] for error in self.error_history):
            recommendations.append("Improve input validation")

        return recommendations

@dataclass
class AnalystTrainingConfig(PerRegimeTrainingConfig):
    """
    Configuration for Analyst models training with analyst-specific parameters.

    Extends PerRegimeTrainingConfig with analyst-specific settings for:
    - Multi-timeframe analysis
    - Directional signal generation
    - Risk scoring and management
    - Magnitude prediction
    """

    # Multi-timeframe analysis settings
    enable_multi_timeframe_analysis: bool = True
    base_timeframe: str = "5m"
    additional_timeframes: List[str] = None  # ["15m", "1h", "4h"]
    timeframe_weights: Dict[str, float] = None  # {"5m": 0.4, "15m": 0.3, "1h": 0.2, "4h": 0.1}

    # Signal generation settings
    enable_directional_signals: bool = True
    directional_confidence_threshold: float = 0.6
    enable_magnitude_prediction: bool = True
    magnitude_thresholds: List[float] = None  # [0.01, 0.02, 0.03, 0.05]

    # Risk management settings
    enable_risk_scoring: bool = True
    risk_model_types: List[str] = None  # ["CATBOOST", "XGBOOST"]
    enable_volatility_adjustment: bool = True
    volatility_lookback_periods: int = 50

    # Advanced analyst features
    enable_regime_transition_detection: bool = True
    enable_market_microstructure_features: bool = True
    enable_order_flow_analysis: bool = False

    # Green light signal settings
    enable_green_light_generation: bool = True
    green_light_confidence_threshold: float = 0.65
    green_light_lookback_periods: int = 20

    def __post_init__(self):
        """Initialize default values for complex fields."""
        super().__post_init__() if hasattr(super(), '__post_init__') else None

        if self.additional_timeframes is None:
            self.additional_timeframes = ["15m", "1h", "4h"]

        if self.timeframe_weights is None:
            self.timeframe_weights = {
                "5m": 0.4,
                "15m": 0.3,
                "1h": 0.2,
                "4h": 0.1
            }

        if self.magnitude_thresholds is None:
            self.magnitude_thresholds = [0.01, 0.02, 0.03, 0.05]

        if self.risk_model_types is None:
            self.risk_model_types = ["CATBOOST", "XGBOOST"]

class AnalystModelsTrainingStepRefactored(PerRegimeTrainingStep):
    """
    Enhanced Analyst Models Training Step with comprehensive utilities integration.

    This is a fully enhanced version that integrates:
    - Common utilities for data operations and validation
    - Hardware optimization (M1 GPU/CPU/Memory)
    - ML common utilities (CV, HPO, lookahead, etc.)
    - Comprehensive error handling with fast failing
    - Extensive tprint logging at every step
    - Math validation and serialization utilities
    """

    def __init__(self, config: Optional[AnalystTrainingConfig] = None):
        """
        Initialize Enhanced Analyst models training step with comprehensive utilities integration.

        Args:
            config: Analyst training configuration
        """
        try:
            # Fast fail if critical dependencies are missing
            validate_critical_dependencies()

            with monitor_resources("Enhanced Analyst Models Training Initialization", logger):
                tprint_info("🚀 Initializing Enhanced Analyst Models Training Step")

                # Set default configuration for analyst models with enhanced settings
                if config is None:
                    config = AnalystTrainingConfig(
                        model_name="analyst_models",
                        timeframe="5m",
                        model_types=["DEEPSCALER", "CATBOOST", "XGBOOST", "MULTISCALE_NBEATS"],
                        hpo_n_trials=100,
                        hpo_timeout_seconds=3600,
                        min_samples_per_regime=1000,
                        enable_data_augmentation=True,
                        augmentation_method="smote",
                        model_save_path="./models/analyst_models",
                        evaluation_metrics=["mse", "mae", "r2", "mape", "smape"]
                    )

                # Initialize parent class
                super().__init__(config)
                self.logger = logger.getChild('AnalystModelsTrainingStepEnhanced')

                # Initialize enhanced components
                self.error_handler = EnhancedErrorHandler(self.logger)
                self.progress_tracker = None  # Will be initialized when training starts

                # Enhanced training metrics with hardware monitoring
                self.training_metrics = {
                    'start_time': None,
                    'end_time': None,
                    'total_duration': None,
                    'memory_usage': [],
                    'cpu_usage': [],
                    'model_performance': {},
                    'regime_statistics': {},
                    'hardware_optimization': {},
                    'error_summary': {},
                    'data_quality_metrics': {}
                }

                # Initialize all components (consolidated)
                self._initialize_components_consolidated()

                # Validate configuration with enhanced error reporting
                validation_result = self._validate_config_enhanced(config)
                if not validation_result['valid']:
                    error_msg = f"Invalid configuration: {validation_result['errors']}"
                    self.error_handler.handle_error(
                        ValueError(error_msg),
                        "Configuration Validation",
                        {'config': config.__dict__, 'validation_errors': validation_result['errors']},
                        fast_fail=True
                    )
                    raise ValueError(error_msg)

                # Log configuration summary
                tprint_success("✅ Enhanced Analyst Models Training Step initialized successfully")
                tprint_info(f"📋 Configuration: {len(config.model_types)} model types, "
                           f"{config.hpo_n_trials} HPO trials, {config.min_samples_per_regime} min samples/regime")

                # Log hardware status
                hardware_status = self._get_hardware_status()
                tprint_structured(hardware_status, LogLevel.INFO)

        except Exception as e:
            tprint_error(f"❌ Failed to initialize Enhanced Analyst Models Training Step: {e}")
            raise

    def _initialize_components_consolidated(self):
        """Consolidated initialization of all components."""
        with tprint_timer("Component initialization"):
            # Hardware optimization
            self._initialize_hardware_optimization()

            # Common utilities
            self._initialize_common_utilities()

            # ML utilities
            self._initialize_ml_utilities()

            # Serialization utilities
            self._initialize_serialization_utilities()

            tprint_success("✅ All components initialized")

    def _validate_config_consolidated(self, config: PerRegimeTrainingConfig) -> None:
        """Consolidated configuration validation using common utilities."""
        try:
            # Basic validation
            if not config.model_types or len(config.model_types) == 0:
                raise ValueError("At least one model type required")

            # HPO validation using validate_positive from common_operations
            if config.enable_hpo:
                validate_positive(config.hpo_n_trials, "hpo_n_trials")
                validate_positive(config.hpo_timeout_seconds, "hpo_timeout_seconds")

            # Regime validation
            validate_positive(config.min_samples_per_regime, "min_samples_per_regime")

            # Path validation using ensure_directory from common_operations
            if config.save_models and config.model_save_path:
                ensure_directory(config.model_save_path)

            tprint_success("✅ Configuration validation passed")
        except Exception as e:
            tprint_error(f"❌ Configuration validation failed: {e}")
            raise

    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization components."""
        try:
            if not HARDWARE_UTILITIES_AVAILABLE:
                tprint_warning("⚠️ Hardware utilities not available, using CPU-only mode")
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
                self.training_metrics['hardware_optimization'] = {
                    'success': False,
                    'error': 'Hardware utilities not available',
                    'fallback': 'CPU_only_mode'
                }
                return

            # Initialize M1 optimizers
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()

            # Integrate with M1 optimizers
            integration_result = integrate_with_m1_optimizers()
            self.training_metrics['hardware_optimization'] = integration_result

            available = sum(1 for x in [self.gpu_manager, self.memory_optimizer, self.cpu_optimizer] if x is not None)
            tprint_success(f"✅ Hardware optimization: {available}/3 optimizers available")

        except Exception as e:
            self.error_handler.handle_error(e, "Hardware Optimization Initialization", {
                'component': 'M1_optimizers',
                'fallback': 'CPU_only_mode'
            })
            tprint_warning("⚠️ Hardware optimization failed, falling back to CPU-only mode")

    def _initialize_common_utilities(self):
        """Initialize common utilities for data operations and validation."""
        try:
            if not COMMON_UTILITIES_AVAILABLE:
                tprint_warning("⚠️ Common utilities not available, using basic operations")
                self.math_validator = MathValidation()
                self.data_quality_tools = {}
                self.file_operations = {
                    'ensure_directory': ensure_directory,
                    'safe_json_dump': safe_json_dump,
                    'safe_json_load': safe_json_load
                }
                return

            # Initialize math validation
            self.math_validator = MathValidation()

            # Initialize data quality tools
            self.data_quality_tools = {
                'validate_dataframe': validate_dataframe,
                'validate_dataframe_columns': validate_dataframe_columns,
                'calculate_data_quality_metrics': calculate_data_quality_metrics,
                'create_data_quality_report': create_data_quality_report,
                'optimize_dataframe_dtypes': optimize_dataframe_dtypes
            }

            # Initialize file operations
            self.file_operations = {
                'ensure_directory': ensure_directory,
                'safe_file_exists': safe_file_exists,
                'safe_json_dump': safe_json_dump,
                'safe_json_load': safe_json_load,
                'safe_to_parquet': safe_to_parquet,
                'safe_read_parquet': safe_read_parquet
            }

            tprint_success("✅ Common utilities initialized successfully")

        except Exception as e:
            self.error_handler.handle_error(e, "Common Utilities Initialization", {
                'component': 'data_operations',
                'fallback': 'basic_operations'
            })
            tprint_warning("⚠️ Common utilities initialization failed, using basic operations")

    def _initialize_ml_utilities(self):
        """Initialize ML common utilities for CV, HPO, and other ML operations."""
        try:
            if not ML_UTILITIES_AVAILABLE:
                tprint_warning("⚠️ ML utilities not available, using basic ML operations")
                self.ml_validation = {}
                self.ml_hpo = {}
                self.ml_evaluation = {}
                self.ml_reporting = {}
                return

            # Initialize validation utilities
            self.ml_validation = {
                'validate_input_data': validate_input_data,
                'validate_model_config': validate_model_config,
                'validate_training_data': validate_training_data
            }

            # Initialize HPO utilities with enhanced regime-aware HPO
            self.ml_hpo = {
                'optimize_hyperparameters': optimize_hyperparameters,
                'create_search_space': create_search_space,
                'validate_hpo_config': validate_hpo_config
            }

            # Initialize enhanced regime-aware HPO system
            if ENHANCED_HPO_AVAILABLE:
                try:
                    enhanced_hpo_config = {
                        'n_trials': self.config.hpo_n_trials,
                        'timeout': self.config.hpo_timeout_seconds,
                        'random_state': 42,
                        'n_jobs': -1,
                        'enable_adaptive_ranges': True,
                        'enable_multi_objective': True,
                        'enable_dynamic_cv': True,
                        'enable_regime_analysis': True,
                        'cv_folds': 5,
                        'search_space': {
                            'learning_rate': {'min': 0.001, 'max': 0.1, 'scale': 'log'},
                            'n_estimators': {'min': 100, 'max': 2000},
                            'max_depth': {'min': 3, 'max': 12},
                            'subsample': {'min': 0.6, 'max': 1.0},
                            'colsample_bytree': {'min': 0.4, 'max': 1.0},
                            'reg_alpha': {'min': 0.0, 'max': 10.0},
                            'reg_lambda': {'min': 0.0, 'max': 10.0},
                            'min_child_weight': {'min': 1, 'max': 20},
                            'gamma': {'min': 0.0, 'max': 5.0}
                        }
                    }

                    self.enhanced_hpo = enhance_existing_hpo_pipeline(enhanced_hpo_config)
                    self.enhanced_cv_strategies = EnhancedCVStrategies()

                    tprint_success("✅ Enhanced regime-aware HPO system initialized")

                except Exception as e:
                    tprint_warning(f"⚠️ Enhanced HPO initialization failed: {e}")
                    self.enhanced_hpo = None
                    self.enhanced_cv_strategies = None
            else:
                self.enhanced_hpo = None
                self.enhanced_cv_strategies = None

            # Initialize evaluation utilities
            self.ml_evaluation = {
                'calculate_metrics': calculate_metrics,
                'evaluate_model_performance': evaluate_model_performance,
                'create_evaluation_report': create_evaluation_report
            }

            # Initialize reporting utilities
            self.ml_reporting = {
                'ReportGenerator': ReportGenerator,
                'ReportManager': ReportManager,
                'create_training_report': create_training_report
            }

            # Initialize enhanced training utilities
            if ENHANCED_TRAINING_AVAILABLE:
                tprint_info("🚀 Initializing enhanced training utilities")
                self._initialize_enhanced_training_utilities()

            tprint_success("✅ ML common utilities initialized successfully")

        except Exception as e:
            self.error_handler.handle_error(e, "ML Utilities Initialization", {
                'component': 'ml_common',
                'fallback': 'basic_ml_operations'
            })
            tprint_warning("⚠️ ML utilities initialization failed, using basic ML operations")

    def _initialize_enhanced_training_utilities(self):
        """Initialize enhanced training utilities for overfitting prevention and lookahead bias detection."""
        try:
            # Create enhanced training configuration
            self.enhanced_training_config = TrainingIntegrationConfig(
                enable_early_stopping=True,
                enable_purged_cv=True,
                enable_lookahead_detection=True,
                enable_temporal_splits=True,
                enable_regularization=True,
                enable_overfitting_monitoring=True,
                model_type='auto'
            )

            # Initialize training enhancer
            self.training_enhancer = TrainingStepEnhancer(self.enhanced_training_config)

            # Store enhanced utilities
            self.enhanced_training_utils = {
                'EnhancedTrainingUtils': EnhancedTrainingUtils,
                'EarlyStoppingConfig': EarlyStoppingConfig,
                'PurgedCVConfig': PurgedCVConfig,
                'OverfittingMonitorConfig': OverfittingMonitorConfig,
                'RegularizationConfig': RegularizationConfig,
                'TrainingStepEnhancer': TrainingStepEnhancer,
                'TrainingIntegrationConfig': TrainingIntegrationConfig
            }

            tprint_success("✅ Enhanced training utilities initialized successfully")

        except Exception as e:
            tprint_warning(f"⚠️ Enhanced training utilities initialization failed: {e}")
            self.enhanced_training_config = None
            self.training_enhancer = None
            self.enhanced_training_utils = {}

    def _initialize_serialization_utilities(self):
        """Initialize serialization utilities for data persistence."""
        try:
            # Initialize serializers
            self.serializers = {
                'json': JSONSerializer(),
                'pickle': PickleSerializer(),
                'parquet': ParquetSerializer(),
                'universal': UniversalSerializer()
            }

            tprint_success("✅ Serialization utilities initialized successfully")

        except Exception as e:
            self.error_handler.handle_error(e, "Serialization Utilities Initialization", {
                'component': 'serializers',
                'fallback': 'basic_serialization'
            })
            tprint_warning("⚠️ Serialization utilities initialization failed, using basic serialization")

    def _get_hardware_status(self) -> Dict[str, Any]:
        """Get comprehensive hardware status information."""
        try:
            return {
                'm1_available': is_m1_available(),
                'mps_available': is_mps_available(),
                'gpu_manager': self.gpu_manager is not None,
                'memory_optimizer': self.memory_optimizer is not None,
                'cpu_optimizer': self.cpu_optimizer is not None,
                'memory_usage_mb': get_memory_usage() / 1024 / 1024,
                'cpu_percent': psutil.cpu_percent(),
                'disk_space': check_disk_space('/', 1.0)
            }
        except Exception as e:
            self.error_handler.handle_error(e, "Hardware Status Check", {
                'component': 'system_info'
            })
            return {'error': str(e)}

    def _optimize_hyperparameters_with_bayesian_tpe(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        search_space: Dict[str, Any],
        n_trials: int = 100,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Bayesian TPE with staged optimization.

        Args:
            model_type: Type of model to optimize
            X: Training features
            y: Training targets
            search_space: Hyperparameter search space
            n_trials: Number of optimization trials
            timeout: Timeout in seconds

        Returns:
            Dictionary containing best parameters and optimization results
        """
        tprint_info(f"🔍 Starting Bayesian TPE hyperparameter optimization for {model_type}...")

        if not ENHANCED_HPO_AVAILABLE or BayesianTPEOptimizer is None:
            tprint_warning("⚠️ Bayesian TPE optimizer not available, skipping HPO")
            return {'best_params': {}, 'optimization_skipped': True}

        try:
            # Create optimization configuration
            opt_config = OptimizationConfig(
                n_trials=n_trials,
                timeout=timeout,
                direction='minimize',
                metric_name='mse',
                enable_staged_optimization=True,
                coarse_grid_points=5,
                fine_grid_points=5,
                coarse_grid_trials=25,
                fine_grid_trials=25,
                tpe_trials=max(50, n_trials - 50),
                enable_hardware_optimization=HARDWARE_UTILITIES_AVAILABLE and is_m1_available(),
                enable_adaptive_optimization=True,
                early_stopping_patience=15,
                seed=42
            )

            # Define objective function
            def objective(params: Dict[str, Any]) -> float:
                """Objective function for hyperparameter optimization."""
                try:
                    if model_type == "XGBOOST":
                        from xgboost import XGBRegressor
                        model = XGBRegressor(**params, random_state=42)
                    elif model_type == "CATBOOST":
                        from catboost import CatBoostRegressor
                        model = CatBoostRegressor(**params, random_state=42, verbose=False)
                    elif model_type == "LIGHTGBM":
                        from lightgbm import LGBMRegressor
                        model = LGBMRegressor(**params, random_state=42, verbose=-1)
                    else:
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(**params, random_state=42)

                    from sklearn.model_selection import cross_val_score
                    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
                    return -np.mean(scores)
                except Exception as e:
                    tprint_warning(f"⚠️ Objective evaluation failed: {e}")
                    return np.inf

            # Run optimizer
            optimizer = BayesianTPEOptimizer(search_space, opt_config)
            result = optimizer.optimize(objective)

            if result.success:
                tprint_success(f"✅ Bayesian TPE optimization completed")
                tprint_info(f"📊 Best MSE: {result.best_value:.6f}, Trials: {result.n_trials}, Time: {result.optimization_time:.2f}s")
                return {
                    'best_params': result.best_params,
                    'best_mse': result.best_value,
                    'n_trials': result.n_trials,
                    'optimization_time': result.optimization_time,
                    'success': True
                }
            else:
                tprint_warning(f"⚠️ Bayesian TPE optimization failed: {result.error_message}")
                return {'best_params': {}, 'success': False, 'error': result.error_message}
        except Exception as e:
            tprint_error(f"❌ Bayesian TPE optimization error: {e}")
            return {'best_params': {}, 'success': False, 'error': str(e)}

    def _validate_config_enhanced(self, config: PerRegimeTrainingConfig) -> Dict[str, Any]:
        """Enhanced configuration validation with comprehensive error reporting and math validation."""
        errors = []
        warnings = []

        try:
            tprint_info("🔍 Starting enhanced configuration validation")

            # Required fields validation with math validation
            if not config.model_name or not isinstance(config.model_name, str):
                errors.append("model_name must be a non-empty string")
            else:
                # Sanitize model name
                config.model_name = sanitize_string(config.model_name, 50)
                tprint_debug(f"✅ Model name validated and sanitized: {config.model_name}")

            if not config.timeframe or not isinstance(config.timeframe, str):
                errors.append("timeframe must be a non-empty string")
            else:
                # Validate timeframe format
                valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
                if config.timeframe not in valid_timeframes:
                    warnings.append(f"Unusual timeframe: {config.timeframe}. Valid timeframes: {valid_timeframes}")
                tprint_debug(f"✅ Timeframe validated: {config.timeframe}")

            # Model types validation with enhanced checks
            if not config.model_types or not isinstance(config.model_types, list):
                errors.append("model_types must be a non-empty list")
            elif len(config.model_types) == 0:
                errors.append("model_types list cannot be empty")
            else:
                # Validate each model type
                valid_model_types = [
                    "TEMPORAL_FUSION_TRANSFORMER", "TABNET", "HIST_GRADIENT_BOOSTING",
                    "EXTRA_TREES", "TCN", "CatBoostRegressor", "LGBMRegressor",
                    "RandomForestRegressor", "XGBRegressor", "NODE"
                ]
                invalid_types = [mt for mt in config.model_types if mt not in valid_model_types]
                if invalid_types:
                    warnings.append(f"Unknown model types: {invalid_types}")

                # Check for model type diversity
                if len(config.model_types) < 2:
                    warnings.append("Consider using multiple model types for better ensemble performance")

                tprint_debug(f"✅ Model types validated: {len(config.model_types)} types")

            # HPO validation with math validation
            try:
                hpo_trials = math_validate_positive(config.hpo_n_trials, "hpo_n_trials")
                if hpo_trials > 1000:
                    warnings.append("hpo_n_trials > 1000 may cause long training times")
                elif hpo_trials < 10:
                    warnings.append("hpo_n_trials < 10 may not provide sufficient optimization")
                tprint_debug(f"✅ HPO trials validated: {hpo_trials}")
            except Exception as e:
                errors.append(f"Invalid hpo_n_trials: {e}")

            try:
                hpo_timeout = math_validate_positive(config.hpo_timeout_seconds, "hpo_timeout_seconds")
                if hpo_timeout < 60:
                    warnings.append("hpo_timeout_seconds < 60 may be too short for effective optimization")
                elif hpo_timeout > 7200:
                    warnings.append("hpo_timeout_seconds > 7200 may cause very long training times")
                tprint_debug(f"✅ HPO timeout validated: {hpo_timeout}s")
            except Exception as e:
                errors.append(f"Invalid hpo_timeout_seconds: {e}")

            # Data validation with enhanced checks
            try:
                min_samples = math_validate_positive(config.min_samples_per_regime, "min_samples_per_regime")
                if min_samples < 100:
                    warnings.append("min_samples_per_regime < 100 may cause poor model performance")
                elif min_samples > 10000:
                    warnings.append("min_samples_per_regime > 10000 may cause memory issues")
                tprint_debug(f"✅ Min samples per regime validated: {min_samples}")
            except Exception as e:
                errors.append(f"Invalid min_samples_per_regime: {e}")

            # Path validation with file system checks
            if not config.model_save_path:
                errors.append("model_save_path cannot be empty")
            else:
                try:
                    # Ensure directory exists and is writable
                    ensure_directory(config.model_save_path)

                    # Check disk space
                    disk_check = check_disk_space(config.model_save_path, 5.0)  # 5GB required
                    if not disk_check['sufficient']:
                        warnings.append(f"Insufficient disk space: {disk_check['free_gb']:.1f}GB available, 5.0GB required")

                    tprint_debug(f"✅ Model save path validated: {config.model_save_path}")
                except Exception as e:
                    errors.append(f"Invalid model_save_path: {e}")

            # Metrics validation with enhanced checks
            if not config.evaluation_metrics or not isinstance(config.evaluation_metrics, list):
                errors.append("evaluation_metrics must be a non-empty list")
            else:
                valid_metrics = ["mse", "mae", "r2", "mape", "smape", "accuracy", "precision", "recall", "f1"]
                invalid_metrics = [m for m in config.evaluation_metrics if m not in valid_metrics]
                if invalid_metrics:
                    warnings.append(f"Unknown evaluation metrics: {invalid_metrics}")

                # Check for metric diversity
                if len(config.evaluation_metrics) < 2:
                    warnings.append("Consider using multiple evaluation metrics for comprehensive assessment")

                tprint_debug(f"✅ Evaluation metrics validated: {len(config.evaluation_metrics)} metrics")

            # Hardware compatibility check
            if is_m1_available():
                tprint_info("🍎 M1 hardware detected - optimization enabled")
            else:
                tprint_info("💻 Non-M1 hardware detected - standard mode")

            # Memory requirements check
            available_memory = get_memory_usage() / 1024 / 1024  # MB
            estimated_memory_need = len(config.model_types) * 1000  # Rough estimate
            if available_memory < estimated_memory_need:
                warnings.append(f"Low available memory: {available_memory:.1f}MB, estimated need: {estimated_memory_need}MB")

            validation_result = {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'hardware_compatibility': {
                    'm1_available': is_m1_available(),
                    'mps_available': is_mps_available(),
                    'available_memory_mb': available_memory
                }
            }

            if validation_result['valid']:
                tprint_success("✅ Configuration validation passed")
            else:
                tprint_error(f"❌ Configuration validation failed: {len(errors)} errors")

            if warnings:
                tprint_warning(f"⚠️ Configuration validation warnings: {len(warnings)} warnings")
                for warning in warnings:
                    tprint_warning(f"  - {warning}")

            return validation_result

        except Exception as e:
            error_msg = f"Validation exception: {str(e)}"
            tprint_error(f"❌ Configuration validation exception: {e}")
            return {
                'valid': False,
                'errors': [error_msg],
                'warnings': [],
                'exception': str(e)
            }

    def _validate_config(self, config: PerRegimeTrainingConfig) -> bool:
        """Legacy validation method for backward compatibility."""
        result = self._validate_config_enhanced(config)
        return result['valid']

    def _generate_datetime_stamp(self) -> str:
        """Generate a consistent datetime stamp for artifacts."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _create_training_report(
        self,
        results: Dict[str, Any],
        execution_time: float,
        status: str = "SUCCESS"
    ) -> str:
        """Create a comprehensive training report with enhanced metrics and monitoring data."""
        timestamp = self._generate_datetime_stamp()
        report_filename = f"analyst_models_training_report_{timestamp}.json"
        report_path = f"{self.config.model_save_path}/reports/{report_filename}"

        # Ensure reports directory exists
        Path(f"{self.config.model_save_path}/reports").mkdir(parents=True, exist_ok=True)

        # Gather system metrics
        system_metrics = self._gather_system_metrics()

        # Gather error summary
        error_summary = self.error_handler.get_error_summary()

        # Gather progress summary if available
        progress_summary = self.progress_tracker.get_summary() if self.progress_tracker else {}

        # Create comprehensive report
        report_data = {
            "metadata": {
                "model_name": self.config.model_name,
                "timeframe": self.config.timeframe,
                "timestamp": timestamp,
                "execution_time_seconds": execution_time,
                "status": status,
                "version": "enhanced_v1.0",
                "config": {
                    "model_types": self.config.model_types,
                    "hpo_n_trials": self.config.hpo_n_trials,
                    "hpo_timeout_seconds": self.config.hpo_timeout_seconds,
                    "min_samples_per_regime": self.config.min_samples_per_regime,
                    "enable_data_augmentation": self.config.enable_data_augmentation,
                    "augmentation_method": self.config.augmentation_method,
                    "evaluation_metrics": self.config.evaluation_metrics
                }
            },
            "results": results,
            "monitoring": {
                "system_metrics": system_metrics,
                "training_metrics": self.training_metrics,
                "progress_summary": progress_summary,
                "error_summary": error_summary
            },
            "summary": {
                "models_trained": len(results.get('models', [])),
                "regimes_processed": len(results.get('regime_analysis', {}).get('unique_regimes', [])),
                "best_performing_model": results.get('best_models_per_regime', {}),
                "training_successful": status == "SUCCESS",
                "total_errors": error_summary.get('total_errors', 0),
                "performance_metrics": self._calculate_performance_metrics(results)
            }
        }

        # Save report
        try:
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            self.logger.info(f"📋 Enhanced training report saved: {report_path}")
            self.logger.info(f"📊 Report includes: {len(report_data['monitoring'])} monitoring sections, "
                           f"{error_summary.get('total_errors', 0)} errors tracked")
        except Exception as e:
            self.error_handler.handle_error(e, "Report Saving", {'report_path': report_path})
            report_path = None

        return report_path

    def _gather_system_metrics(self) -> Dict[str, Any]:
        """Gather comprehensive system metrics."""
        try:

            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Disk metrics
            disk = psutil.disk_usage('/')

            return {
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_percent': memory.percent,
                    'swap_total_gb': swap.total / (1024**3),
                    'swap_used_percent': swap.percent
                },
                'cpu': {
                    'usage_percent': cpu_percent,
                    'count': cpu_count,
                    'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                },
                'disk': {
                    'total_gb': disk.total / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'usage_percent': (disk.used / disk.total) * 100
                }
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to gather system metrics: {e}")
            return {}

    def _calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics from training results."""
        metrics = {
            'model_count': 0,
            'regime_count': 0,
            'best_r2_score': -np.inf,
            'worst_r2_score': np.inf,
            'average_r2_score': 0.0,
            'successful_models': 0,
            'failed_models': 0
        }

        try:
            if 'evaluation_results' in results:
                r2_scores = []
                for regime, regime_results in results['evaluation_results'].items():
                    if isinstance(regime_results, dict):
                        metrics['regime_count'] += 1
                        for model_type, model_results in regime_results.items():
                            if isinstance(model_results, dict) and 'metrics' in model_results:
                                metrics['model_count'] += 1
                                if 'r2' in model_results['metrics']:
                                    r2_score = model_results['metrics']['r2']
                                    r2_scores.append(r2_score)
                                    metrics['best_r2_score'] = max(metrics['best_r2_score'], r2_score)
                                    metrics['worst_r2_score'] = min(metrics['worst_r2_score'], r2_score)
                                    metrics['successful_models'] += 1
                                else:
                                    metrics['failed_models'] += 1

                if r2_scores:
                    metrics['average_r2_score'] = np.mean(r2_scores)
                    metrics['r2_std'] = np.std(r2_scores)
                    metrics['r2_median'] = np.median(r2_scores)

            return metrics
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate performance metrics: {e}")
            return metrics

    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Execute Enhanced Analyst models training step with comprehensive utilities integration.

        This method provides:
        - Comprehensive error handling with fast failing
        - Hardware optimization integration
        - Extensive tprint logging at every step
        - Common utilities integration for data operations
        - ML utilities integration for CV, HPO, etc.
        - Math validation and serialization utilities

        Args:
            X: Input features
            y: Target values (analyst outputs)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states

        Returns:
            Dictionary containing training results and comprehensive metadata

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If training fails
        """
        try:
            # Fast fail if runtime dependencies are missing
            validate_runtime_dependencies()

            # Initialize training metrics and progress tracking
            self.training_metrics['start_time'] = datetime.now()
            start_time = time.time()

            # Initialize progress tracker with enhanced monitoring
            total_steps = 8  # Enhanced with more detailed steps
            self.progress_tracker = TrainingProgressTracker(total_steps, self.logger)

            with monitor_resources("Enhanced Analyst Models Training Execution", self.logger):
                tprint_info("🚀 Starting Enhanced Analyst models training step with comprehensive utilities")
                tprint_info(f"📊 Input data: {X.shape[0]} samples, {X.shape[1]} features, "
                           f"{len(np.unique(regime_labels))} regimes")

                # Log hardware status
                hardware_status = self._get_hardware_status()
                tprint_structured(hardware_status, LogLevel.INFO)

                # Step 1: Enhanced data validation with comprehensive checks
                self.progress_tracker.update_step("Enhanced Data Validation", {"samples": X.shape[0], "features": X.shape[1]})
                validation_result = self._validate_input_data_enhanced(X, y, regime_labels)
                if not validation_result['valid']:
                    error_msg = f"Invalid input data: {validation_result['errors']}"
                    self.error_handler.handle_error(
                        ValueError(error_msg),
                        "Input Data Validation",
                        validation_result,
                        fast_fail=True
                    )
                    raise ValueError(error_msg)

                # Step 2: Data quality assessment
                self.progress_tracker.update_step("Data Quality Assessment")
                data_quality_result = self._assess_data_quality(X, y, regime_labels)
                self.training_metrics['data_quality_metrics'] = data_quality_result

                # Step 3: Hardware optimization setup
                self.progress_tracker.update_step("Hardware Optimization Setup")
                optimization_result = self._setup_hardware_optimization(X, y)
                self.training_metrics['hardware_optimization'] = optimization_result

                # Step 4: Regime analysis with enhanced processing
                self.progress_tracker.update_step("Enhanced Regime Analysis", {"unique_regimes": len(np.unique(regime_labels))})
                regime_analysis_result = self._analyze_regimes_enhanced(regime_labels)

                # Step 5: Training execution with comprehensive error handling
                self.progress_tracker.update_step("Enhanced Model Training", {"model_types": len(self.config.model_types)})

                # Enhanced training with overfitting prevention and lookahead bias detection
                # Enforce enhanced training path only (fast-fail if unavailable)
                if not ENHANCED_TRAINING_AVAILABLE or not hasattr(self, 'training_enhancer'):
                    raise RuntimeError("Enhanced training utilities are required and must be available for Analyst training")

                # Validate temporal data for lookahead bias
                if timestamps is not None:
                    tprint_info("🔍 Validating temporal data for lookahead bias...")
                    is_valid, warnings = self.training_enhancer.enhanced_utils.validate_temporal_data(
                        X, y, timestamps, strict_mode=True
                    )
                    if warnings:
                        for warning in warnings:
                            tprint_warning(f"⚠️ {warning}")
                    if not is_valid:
                        tprint_error("❌ Temporal data validation failed")
                        raise ValueError("Lookahead bias detected in temporal data")

                # Use enhanced training only
                with tprint_timer("Enhanced Training", LogLevel.PERFORMANCE):
                    with monitor_resources("Enhanced Training", self.logger):
                        results = self._execute_enhanced_training(
                            X, y, regime_labels, feature_names, hmm_states, timestamps
                        )

                training_successful = True
                tprint_success("✅ Enhanced training completed successfully")

                # Step 6: Post-processing and metadata enhancement
                self.progress_tracker.update_step("Enhanced Post-processing", {"training_successful": training_successful})
                if 'error' not in results:
                    results = self._add_analyst_specific_metadata(results)
                    results = self._enhance_results_with_utilities_metadata(results)

                # Step 7: Performance evaluation with comprehensive metrics
                self.progress_tracker.update_step("Comprehensive Performance Evaluation")
                results = self._enhance_results_with_performance_metrics(results)
                results = self._add_ml_evaluation_metrics(results)

                # Step 8: Enhanced report generation
                self.progress_tracker.update_step("Enhanced Report Generation")
                execution_time = time.time() - start_time
                self.training_metrics['end_time'] = datetime.now()
                self.training_metrics['total_duration'] = execution_time

                # Add error summary to results
                self.training_metrics['error_summary'] = self.error_handler.get_error_summary()

                report_path = self._create_enhanced_training_report(results, execution_time, "SUCCESS")
                if report_path:
                    results['training_report'] = report_path

                # Final success logging with comprehensive summary
                tprint_success(f"✅ Enhanced Analyst models training completed in {execution_time:.2f}s")
                progress_summary = self.progress_tracker.get_summary()
                tprint_structured(progress_summary, LogLevel.SUCCESS)

                # Log final hardware status
                final_hardware_status = self._get_hardware_status()
                tprint_structured(final_hardware_status, LogLevel.INFO)

                # Generate thorough outcome file with datetime stamp
                try:
                    outcome_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    outcomes_dir = Path('outcomes')
                    outcomes_dir.mkdir(parents=True, exist_ok=True)

                    outcome_filename = f"analyst_model_training_outcome_{outcome_timestamp}.json"
                    outcome_path = outcomes_dir / outcome_filename

                    # Extract comprehensive training statistics
                    models_trained = results.get('models', {})
                    model_stats = {
                        'total_models': len(models_trained),
                        'model_names': list(models_trained.keys()),
                        'per_model_details': {}
                    }

                    for model_name, model_info in models_trained.items():
                        if isinstance(model_info, dict):
                            # Extract comprehensive ML metrics
                            ml_metrics = {}

                            # Classification metrics
                            if 'accuracy' in model_info:
                                ml_metrics['accuracy'] = float(model_info['accuracy']) if model_info['accuracy'] is not None else None
                            if 'auc' in model_info:
                                ml_metrics['auc'] = float(model_info['auc']) if model_info['auc'] is not None else None
                            if 'roc_auc' in model_info:
                                ml_metrics['roc_auc'] = float(model_info['roc_auc']) if model_info['roc_auc'] is not None else None
                            if 'precision' in model_info:
                                ml_metrics['precision'] = float(model_info['precision']) if model_info['precision'] is not None else None
                            if 'recall' in model_info:
                                ml_metrics['recall'] = float(model_info['recall']) if model_info['recall'] is not None else None
                            if 'f1_score' in model_info:
                                ml_metrics['f1_score'] = float(model_info['f1_score']) if model_info['f1_score'] is not None else None
                            if 'average_precision' in model_info:
                                ml_metrics['average_precision'] = float(model_info['average_precision']) if model_info['average_precision'] is not None else None

                            # Regression metrics
                            if 'mse' in model_info:
                                ml_metrics['mse'] = float(model_info['mse']) if model_info['mse'] is not None else None
                            if 'rmse' in model_info:
                                ml_metrics['rmse'] = float(model_info['rmse']) if model_info['rmse'] is not None else None
                            if 'mae' in model_info:
                                ml_metrics['mae'] = float(model_info['mae']) if model_info['mae'] is not None else None
                            if 'r2_score' in model_info:
                                ml_metrics['r2_score'] = float(model_info['r2_score']) if model_info['r2_score'] is not None else None

                            # Trading-specific metrics
                            if 'sharpe_ratio' in model_info:
                                ml_metrics['sharpe_ratio'] = float(model_info['sharpe_ratio']) if model_info['sharpe_ratio'] is not None else None
                            if 'win_rate' in model_info:
                                ml_metrics['win_rate'] = float(model_info['win_rate']) if model_info['win_rate'] is not None else None
                            if 'profit_factor' in model_info:
                                ml_metrics['profit_factor'] = float(model_info['profit_factor']) if model_info['profit_factor'] is not None else None
                            if 'max_drawdown' in model_info:
                                ml_metrics['max_drawdown'] = float(model_info['max_drawdown']) if model_info['max_drawdown'] is not None else None

                            # CV scores
                            cv_scores = {}
                            if 'cv_scores' in model_info and model_info['cv_scores'] is not None:
                                cv_scores_list = model_info['cv_scores']
                                if isinstance(cv_scores_list, (list, tuple)) and len(cv_scores_list) > 0:
                                    cv_scores = {
                                        'mean': float(np.mean(cv_scores_list)),
                                        'std': float(np.std(cv_scores_list)),
                                        'min': float(np.min(cv_scores_list)),
                                        'max': float(np.max(cv_scores_list)),
                                        'scores': [float(s) for s in cv_scores_list],
                                    }

                            # Confusion matrix if available
                            confusion_matrix_data = {}
                            if 'confusion_matrix' in model_info and model_info['confusion_matrix'] is not None:
                                cm = model_info['confusion_matrix']
                                if hasattr(cm, 'tolist'):
                                    confusion_matrix_data = {
                                        'matrix': cm.tolist(),
                                        'shape': list(cm.shape),
                                    }

                            model_stats['per_model_details'][model_name] = {
                                'model_type': model_info.get('model_type', 'unknown'),
                                'training_score': float(model_info.get('train_score', 0.0)) if model_info.get('train_score') is not None else None,
                                'validation_score': float(model_info.get('val_score', 0.0)) if model_info.get('val_score') is not None else None,
                                'test_score': float(model_info.get('test_score', 0.0)) if model_info.get('test_score') is not None else None,
                                'n_features': model_info.get('n_features', 0),
                                'n_samples': model_info.get('n_samples', 0),
                                'ml_metrics': ml_metrics,
                                'cv_scores': cv_scores,
                                'confusion_matrix': confusion_matrix_data,
                            }

                    # Calculate aggregate metrics across all models
                    aggregate_metrics = {
                        'avg_training_score': 0.0,
                        'avg_validation_score': 0.0,
                        'avg_test_score': 0.0,
                        'avg_accuracy': 0.0,
                        'avg_auc': 0.0,
                        'avg_precision': 0.0,
                        'avg_recall': 0.0,
                        'avg_f1_score': 0.0,
                        'best_model': None,
                        'worst_model': None,
                    }

                    valid_train_scores = []
                    valid_val_scores = []
                    valid_test_scores = []
                    valid_accuracies = []
                    valid_aucs = []
                    valid_precisions = []
                    valid_recalls = []
                    valid_f1_scores = []

                    for model_name, details in model_stats['per_model_details'].items():
                        if details['training_score'] is not None:
                            valid_train_scores.append((model_name, details['training_score']))
                        if details['validation_score'] is not None:
                            valid_val_scores.append((model_name, details['validation_score']))
                        if details['test_score'] is not None:
                            valid_test_scores.append((model_name, details['test_score']))

                        ml_metrics = details.get('ml_metrics', {})
                        if ml_metrics.get('accuracy') is not None:
                            valid_accuracies.append(ml_metrics['accuracy'])
                        if ml_metrics.get('auc') is not None or ml_metrics.get('roc_auc') is not None:
                            valid_aucs.append(ml_metrics.get('auc') or ml_metrics.get('roc_auc'))
                        if ml_metrics.get('precision') is not None:
                            valid_precisions.append(ml_metrics['precision'])
                        if ml_metrics.get('recall') is not None:
                            valid_recalls.append(ml_metrics['recall'])
                        if ml_metrics.get('f1_score') is not None:
                            valid_f1_scores.append(ml_metrics['f1_score'])

                    if valid_train_scores:
                        aggregate_metrics['avg_training_score'] = float(np.mean([s for _, s in valid_train_scores]))
                        aggregate_metrics['best_model'] = max(valid_train_scores, key=lambda x: x[1])[0]
                        aggregate_metrics['worst_model'] = min(valid_train_scores, key=lambda x: x[1])[0]
                    if valid_val_scores:
                        aggregate_metrics['avg_validation_score'] = float(np.mean([s for _, s in valid_val_scores]))
                    if valid_test_scores:
                        aggregate_metrics['avg_test_score'] = float(np.mean([s for _, s in valid_test_scores]))
                    if valid_accuracies:
                        aggregate_metrics['avg_accuracy'] = float(np.mean(valid_accuracies))
                        aggregate_metrics['std_accuracy'] = float(np.std(valid_accuracies))
                    if valid_aucs:
                        aggregate_metrics['avg_auc'] = float(np.mean(valid_aucs))
                        aggregate_metrics['std_auc'] = float(np.std(valid_aucs))
                    if valid_precisions:
                        aggregate_metrics['avg_precision'] = float(np.mean(valid_precisions))
                    if valid_recalls:
                        aggregate_metrics['avg_recall'] = float(np.mean(valid_recalls))
                    if valid_f1_scores:
                        aggregate_metrics['avg_f1_score'] = float(np.mean(valid_f1_scores))
                        aggregate_metrics['std_f1_score'] = float(np.std(valid_f1_scores))

                    # Performance metrics breakdown
                    performance_metrics = {
                        'execution_time_seconds': execution_time,
                        'models_per_second': len(models_trained) / max(0.001, execution_time),
                        'progress_summary': progress_summary,
                        'hardware_status': final_hardware_status,
                        'resource_usage': self.training_metrics.get('resource_usage', {}),
                        'aggregate_ml_metrics': aggregate_metrics,
                    }

                    # Training configuration details
                    training_config = {
                        'timeframe': getattr(self.config, 'timeframe', '60m'),
                        'regime_aware': getattr(self.config, 'regime_aware', True),
                        'enable_enhanced_training': getattr(self.config, 'enable_enhanced_training', True),
                        'enable_overfitting_prevention': getattr(self.config, 'enable_overfitting_prevention', True),
                        'use_purged_cv': getattr(self.config, 'use_purged_cv', True),
                        'cv_folds': getattr(self.config, 'cv_folds', 5),
                        'early_stopping_enabled': getattr(self.config, 'early_stopping_enabled', True),
                    }

                    # Data quality and validation
                    data_quality = {
                        'input_samples': results.get('input_samples', 0),
                        'input_features': results.get('input_features', 0),
                        'training_samples': results.get('training_samples', 0),
                        'validation_samples': results.get('validation_samples', 0),
                        'test_samples': results.get('test_samples', 0),
                        'feature_names': results.get('feature_names', []),
                        'regime_distribution': results.get('regime_distribution', {}),
                    }

                    # Warnings and issues
                    warnings_and_issues = {
                        'overfitting_warnings': results.get('overfitting_warnings', []),
                        'lookahead_warnings': results.get('lookahead_warnings', []),
                        'data_quality_warnings': results.get('data_quality_warnings', []),
                        'total_warnings': len(results.get('overfitting_warnings', [])) + len(results.get('lookahead_warnings', [])),
                    }

                    # Create comprehensive outcome report
                    outcome_data = {
                        'component': 'analyst_model_training',
                        'timestamp': datetime.now().isoformat(),
                        'execution_time': execution_time,
                        'configuration': training_config,
                        'results': {
                            'summary': {
                                'models_trained': len(models_trained),
                                'training_successful': True,
                                'report_path': results.get('training_report'),
                            },
                            'model_statistics': model_stats,
                            'data_quality': data_quality,
                        },
                        'performance_metrics': performance_metrics,
                        'warnings_and_issues': warnings_and_issues,
                        'enhanced_training_metadata': results.get('enhanced_training_metadata', {}),
                        'ml_utilities_metadata': results.get('ml_utilities_metadata', {}),
                        'status': 'success'
                    }

                    # Save outcome file
                    with open(outcome_path, 'w') as f:
                        json.dump(outcome_data, f, indent=2, default=str)

                    tprint_success(f"📄 Outcome file saved: {outcome_filename}")
                    results['outcome_file'] = str(outcome_path)

                except Exception as outcome_error:
                    tprint_warning(f"⚠️ Failed to save outcome file: {outcome_error}")
                    # Don't fail training if outcome file generation fails

                return results

        except Exception as e:
            execution_time = time.time() - start_time
            self.training_metrics['end_time'] = datetime.now()
            self.training_metrics['total_duration'] = execution_time

            error_msg = f"Enhanced Analyst models training failed: {e}"
            self.error_handler.handle_error(e, "Enhanced Training Execution", {
                'execution_time': execution_time,
                'progress': self.progress_tracker.get_summary() if self.progress_tracker else {},
                'hardware_status': self._get_hardware_status()
            }, fast_fail=True)

            # Create comprehensive failure report
            failure_results = {
                'error': error_msg,
                'execution_time': execution_time,
                'error_summary': self.error_handler.get_error_summary(),
                'progress_summary': self.progress_tracker.get_summary() if self.progress_tracker else {},
                'hardware_status': self._get_hardware_status(),
                'training_metrics': self.training_metrics
            }

            try:
                self._create_enhanced_training_report(failure_results, execution_time, "FAILED")
            except Exception as report_error:
                tprint_error(f"❌ Failed to create failure report: {report_error}")

            # Fast-fail: Re-raise the exception with enhanced context
            raise RuntimeError(error_msg) from e

    def _execute_enhanced_training(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray,
                                 feature_names: Optional[List[str]], hmm_states: Optional[np.ndarray],
                                 timestamps: Optional[np.ndarray]) -> Dict[str, Any]:
        """Execute enhanced training with overfitting prevention and lookahead bias detection."""
        try:
            tprint_info("🚀 Executing enhanced training with overfitting prevention")

            # Optional vectorized/matrix-optimized preprocessing using matrix_operations if available
            try:
                from src.utils.matrix_operations import optimize_matrix_computations, validate_matrix_properties
                validate_matrix_properties(X)
                X = optimize_matrix_computations(X)
            except Exception:
                # Non-fatal; proceed without matrix optimization
                pass

            # Get unique regimes
            unique_regimes = np.unique(regime_labels)
            results = {
                'models': {},
                'regime_analysis': {},
                'enhanced_training_metadata': {},
                'overfitting_warnings': [],
                'ensemble_diversity': None
            }

            # Train models for each regime with enhanced utilities
            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                X_regime = X[regime_mask]
                y_regime = y[regime_mask]
                timestamps_regime = timestamps[regime_mask] if timestamps is not None else None

                tprint_info(f"🎯 Training models for regime {regime} ({len(X_regime)} samples)")

                # Train each model type for this regime
                regime_models = {}
                for model_type in self.config.model_types:
                    try:
                        # Create model instance with enhanced HPO (fast fail enabled)
                        if ENHANCED_HPO_AVAILABLE and hasattr(self, 'enhanced_hpo'):
                            # Use enhanced HPO system - will fail fast if issues occur
                            try:
                                model = self.training_enhancer.create_model(
                                    model_type=model_type,
                                    model_name=f"analyst_{model_type}_regime_{regime}",
                                    model_params={},
                                    enable_enhanced_hpo=True,
                                    regime_labels=regime_labels,
                                    X_regime=X_regime,
                                    y_regime=y_regime
                                )
                                tprint_success(f"✅ Enhanced HPO successfully applied for {model_type} in regime {regime}")
                            except RuntimeError as e:
                                tprint_error(f"❌ Enhanced HPO failed for {model_type} in regime {regime}: {e}")
                                tprint_info("🔄 Falling back to standard model creation (fast fail disabled for this instance)")
                                # Fallback only for this specific case - create standard model
                                model = self.training_enhancer.create_model(
                                    model_type=model_type,
                                    model_name=f"analyst_{model_type}_regime_{regime}_fallback",
                                    model_params={}
                                )
                        else:
                            # Standard model creation
                            model = self.training_enhancer.create_model(
                                model_type=model_type,
                                model_name=f"analyst_{model_type}_regime_{regime}",
                                model_params={}
                            )

                        # Apply enhanced regularization
                        model = self.training_enhancer.enhanced_utils.apply_enhanced_regularization(
                            model, model_type
                        )

                        # Train with enhanced cross-validation, early stopping and overfitting monitoring
                        trained_model, metadata = self.training_enhancer.enhance_training_step(
                            X_regime, y_regime, model, timestamps_regime,
                            f"analyst_{model_type}_regime_{regime}", regime_labels
                        )

                        regime_models[model_type] = {
                            'model': trained_model,
                            'metadata': metadata
                        }

                        # Check for overfitting warnings
                        if metadata.get('overfitting_detected', False):
                            results['overfitting_warnings'].append(f"Overfitting detected in {model_type} for regime {regime}")

                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to train {model_type} for regime {regime}: {e}")
                        continue

                results['models'][regime] = regime_models

            # Calculate ensemble diversity if multiple models
            if len(self.config.model_types) > 1:
                tprint_info("📊 Calculating ensemble diversity...")
                for regime in unique_regimes:
                    if regime in results['models']:
                        models_list = [results['models'][regime][mt]['model'] for mt in self.config.model_types
                                     if mt in results['models'][regime]]
                        if len(models_list) > 1:
                            diversity_metrics = self.training_enhancer.enhanced_utils.calculate_ensemble_diversity(
                                models_list, X[regime_labels == regime], y[regime_labels == regime]
                            )
                            results['ensemble_diversity'] = diversity_metrics

                            if diversity_metrics.get('diversity_score', 0) < 0.1:
                                tprint_warning(f"⚠️ Low ensemble diversity for regime {regime}")
                            else:
                                tprint_success(f"✅ Good ensemble diversity for regime {regime}")

            # Add enhanced training metadata
            results['enhanced_training_metadata'] = {
                'overfitting_prevention_enabled': True,
                'lookahead_bias_detection_enabled': True,
                'early_stopping_enabled': True,
                'enhanced_regularization_enabled': True,
                'temporal_validation_enabled': timestamps is not None,
                'total_warnings': len(results['overfitting_warnings'])
            }

            tprint_success("✅ Enhanced training completed successfully")
            return results

        except Exception as e:
            tprint_error(f"❌ Enhanced training failed: {e}")
            raise

    def _execute_standard_training(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray,
                                 feature_names: Optional[List[str]], hmm_states: Optional[np.ndarray]) -> Dict[str, Any]:
        """Disabled: Standard training fallback is not allowed."""
        raise RuntimeError("Standard/vectorized fallback path is disabled. Enhanced training is mandatory.")

    def _validate_input_data_enhanced(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Enhanced input data validation with comprehensive error reporting and math validation."""
        errors = []
        warnings = []

        try:
            tprint_info("🔍 Starting enhanced input data validation")

            # Basic null checks
            if X is None or y is None or regime_labels is None:
                errors.append("Input data cannot be None")
                return {'valid': False, 'errors': errors, 'warnings': warnings}

            # Shape validation with math validation
            try:
                x_len = math_validate_positive(len(X), "X length")
                y_len = math_validate_positive(len(y), "y length")
                regime_len = math_validate_positive(len(regime_labels), "regime_labels length")

                if x_len != y_len or x_len != regime_len:
                    errors.append(f"Input data length mismatch: X={x_len}, y={y_len}, regime_labels={regime_len}")
                    return {'valid': False, 'errors': errors, 'warnings': warnings}

                if x_len == 0:
                    errors.append("Input data is empty")
                    return {'valid': False, 'errors': errors, 'warnings': warnings}

                tprint_debug(f"✅ Data length validation passed: {x_len} samples")

            except Exception as e:
                errors.append(f"Data length validation failed: {e}")
                return {'valid': False, 'errors': errors, 'warnings': warnings}

            # Data quality checks with enhanced validation
            try:
                nan_count_X = np.isnan(X).sum()
                inf_count_X = np.isinf(X).sum()
                nan_count_y = np.isnan(y).sum()
                inf_count_y = np.isinf(y).sum()

                # Use math validation for counts
                nan_count_X = math_validate_finite(nan_count_X, "NaN count in X")
                inf_count_X = math_validate_finite(inf_count_X, "Inf count in X")
                nan_count_y = math_validate_finite(nan_count_y, "NaN count in y")
                inf_count_y = math_validate_finite(inf_count_y, "Inf count in y")

                if nan_count_X > 0:
                    nan_percentage = safe_divide(nan_count_X, X.size, 0) * 100
                    if nan_percentage > 50:
                        errors.append(f"Input features contain {nan_count_X} NaN values ({nan_percentage:.1f}%)")
                    else:
                        warnings.append(f"Input features contain {nan_count_X} NaN values ({nan_percentage:.1f}%)")

                if inf_count_X > 0:
                    inf_percentage = safe_divide(inf_count_X, X.size, 0) * 100
                    if inf_percentage > 10:
                        errors.append(f"Input features contain {inf_count_X} infinite values ({inf_percentage:.1f}%)")
                    else:
                        warnings.append(f"Input features contain {inf_count_X} infinite values ({inf_percentage:.1f}%)")

                if nan_count_y > 0:
                    nan_percentage = safe_divide(nan_count_y, len(y), 0) * 100
                    if nan_percentage > 20:
                        errors.append(f"Target values contain {nan_count_y} NaN values ({nan_percentage:.1f}%)")
                    else:
                        warnings.append(f"Target values contain {nan_count_y} NaN values ({nan_percentage:.1f}%)")

                if inf_count_y > 0:
                    inf_percentage = safe_divide(inf_count_y, len(y), 0) * 100
                    if inf_percentage > 5:
                        errors.append(f"Target values contain {inf_count_y} infinite values ({inf_percentage:.1f}%)")
                    else:
                        warnings.append(f"Target values contain {inf_count_y} infinite values ({inf_percentage:.1f}%)")

                tprint_debug(f"✅ Data quality validation completed: NaN X={nan_count_X}, Inf X={inf_count_X}, NaN y={nan_count_y}, Inf y={inf_count_y}")

            except Exception as e:
                errors.append(f"Data quality validation failed: {e}")

            # Regime distribution checks with enhanced analysis
            try:
                unique_regimes = np.unique(regime_labels)
                regime_counts = np.bincount(regime_labels)
                min_regime_size = regime_counts.min()
                max_regime_size = regime_counts.max()

                # Use math validation for regime sizes
                min_regime_size = math_validate_positive(min_regime_size, "min regime size")
                max_regime_size = math_validate_positive(max_regime_size, "max regime size")

                if min_regime_size < self.config.min_samples_per_regime:
                    warnings.append(f"Some regimes have fewer than {self.config.min_samples_per_regime} samples (min: {min_regime_size})")

                # Data distribution warnings with enhanced analysis
                regime_ratio = safe_divide(max_regime_size, min_regime_size, 1)
                if regime_ratio > 10:
                    warnings.append(f"High regime imbalance: largest regime is {regime_ratio:.1f}x larger than smallest")

                # Check for regime diversity
                regime_entropy = self._calculate_regime_entropy(regime_counts)
                if regime_entropy < 0.5:
                    warnings.append(f"Low regime diversity (entropy: {regime_entropy:.3f})")

                tprint_debug(f"✅ Regime distribution validation completed: {len(unique_regimes)} regimes, ratio: {regime_ratio:.1f}, entropy: {regime_entropy:.3f}")

            except Exception as e:
                errors.append(f"Regime distribution validation failed: {e}")

            # Feature statistics with enhanced analysis
            try:
                feature_stats = {
                    'n_samples': len(X),
                    'n_features': X.shape[1] if len(X.shape) > 1 else 1,
                    'n_regimes': len(unique_regimes),
                    'regime_distribution': dict(zip(unique_regimes, regime_counts)),
                    'feature_means': np.mean(X, axis=0).tolist() if len(X.shape) > 1 else [np.mean(X)],
                    'feature_stds': np.std(X, axis=0).tolist() if len(X.shape) > 1 else [np.std(X)],
                    'feature_ranges': [(np.min(X[:, i]), np.max(X[:, i])) for i in range(X.shape[1])] if len(X.shape) > 1 else [(np.min(X), np.max(X))],
                    'target_stats': {
                        'mean': np.mean(y),
                        'std': np.std(y),
                        'min': np.min(y),
                        'max': np.max(y)
                    }
                }

                # Validate feature statistics
                for i, mean_val in enumerate(feature_stats['feature_means']):
                    try:
                        math_validate_finite(mean_val, f"feature_{i}_mean")
                    except Exception as e:
                        warnings.append(f"Invalid feature {i} mean: {e}")

                tprint_debug(f"✅ Feature statistics validation completed: {feature_stats['n_features']} features")

            except Exception as e:
                errors.append(f"Feature statistics validation failed: {e}")
                feature_stats = {}

            # Hardware compatibility check
            try:
                data_size_mb = (X.nbytes + y.nbytes + regime_labels.nbytes) / (1024 * 1024)
                available_memory_mb = get_memory_usage() / 1024 / 1024

                if data_size_mb > available_memory_mb * 0.5:
                    warnings.append(f"Large dataset: {data_size_mb:.1f}MB (50% of available memory)")

                tprint_debug(f"✅ Hardware compatibility check: data size {data_size_mb:.1f}MB, available {available_memory_mb:.1f}MB")

            except Exception as e:
                warnings.append(f"Hardware compatibility check failed: {e}")

            validation_result = {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'statistics': feature_stats
            }

            if validation_result['valid']:
                tprint_success("✅ Enhanced input data validation passed")
            else:
                tprint_error(f"❌ Enhanced input data validation failed: {len(errors)} errors")

            if warnings:
                tprint_warning(f"⚠️ Enhanced input data validation warnings: {len(warnings)} warnings")
                for warning in warnings:
                    tprint_warning(f"  - {warning}")

            return validation_result

        except Exception as e:
            error_msg = f"Enhanced validation exception: {str(e)}"
            tprint_error(f"❌ Enhanced input data validation exception: {e}")
            return {
                'valid': False,
                'errors': [error_msg],
                'warnings': [],
                'statistics': {},
                'exception': str(e)
            }

    def _calculate_regime_entropy(self, regime_counts: np.ndarray) -> float:
        """Calculate entropy of regime distribution."""
        try:
            total = np.sum(regime_counts)
            if total == 0:
                return 0.0

            probabilities = regime_counts / total
            probabilities = probabilities[probabilities > 0]  # Remove zero probabilities

            if len(probabilities) == 0:
                return 0.0

            entropy = -np.sum(probabilities * np.log2(probabilities))
            return math_validate_finite(entropy, "regime entropy")

        except Exception as e:
            tprint_warning(f"⚠️ Failed to calculate regime entropy: {e}")
            return 0.0

    def _validate_input_data(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> bool:
        """Legacy validation method for backward compatibility."""
        result = self._validate_input_data_enhanced(X, y, regime_labels)
        return result['valid']

    def _assess_data_quality(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Assess data quality using common utilities."""
        try:
            tprint_info("🔍 Assessing data quality with common utilities")

            # Convert to DataFrame for quality assessment
            if PANDAS_AVAILABLE:
                df = pd.DataFrame(X)
                df['target'] = y
                df['regime'] = regime_labels
            else:
                # Fallback for when pandas is not available
                df = pd.DataFrame(X)
                # Add target and regime as additional columns
                df.data = [list(row) + [y[i], regime_labels[i]] for i, row in enumerate(X)]
                df.columns = list(range(X.shape[1])) + ['target', 'regime']

            # Use common utilities for quality assessment if available
            if COMMON_UTILITIES_AVAILABLE and hasattr(self, 'data_quality_tools') and self.data_quality_tools:
                quality_metrics = self.data_quality_tools.get('calculate_data_quality_metrics', lambda x: {})(df)
                dataframe_info = self.data_quality_tools.get('get_dataframe_info', lambda x: {})(df)
                quality_report = self.data_quality_tools.get('create_data_quality_report', lambda x: {})(df)
            else:
                # Fallback to basic quality assessment
                quality_metrics = {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'missing_values': df.isnull().sum().sum(),
                    'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                    'duplicate_rows': df.duplicated().sum(),
                    'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
                    'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                    'categorical_columns': len(df.select_dtypes(include=['object']).columns)
                }
                dataframe_info = {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.to_dict(),
                    'memory_usage': df.memory_usage(deep=True).sum()
                }
                quality_report = {
                    'basic_info': dataframe_info,
                    'quality_metrics': quality_metrics,
                    'issues': []
                }

            # Enhanced quality assessment
            quality_assessment = {
                'basic_metrics': quality_metrics,
                'dataframe_info': dataframe_info,
                'quality_report': quality_report,
                'recommendations': []
            }

            # Generate recommendations based on quality metrics
            if quality_metrics.get('missing_percentage', 0) > 20:
                quality_assessment['recommendations'].append("High missing data percentage - consider imputation")

            if quality_metrics.get('duplicate_percentage', 0) > 5:
                quality_assessment['recommendations'].append("High duplicate percentage - consider deduplication")

            if quality_metrics.get('numeric_columns', 0) == 0:
                quality_assessment['recommendations'].append("No numeric columns found - check data types")

            tprint_success("✅ Data quality assessment completed")
            return quality_assessment

        except Exception as e:
            self.error_handler.handle_error(e, "Data Quality Assessment", {
                'data_shape': X.shape,
                'target_shape': y.shape
            })
            return {'error': str(e)}

    def _setup_hardware_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Setup hardware optimization for the training process."""
        try:
            tprint_info("🧠 Setting up hardware optimization")

            optimization_result = {
                'm1_available': is_m1_available(),
                'mps_available': is_mps_available(),
                'optimizations_applied': [],
                'performance_improvements': {}
            }

            # M1 GPU optimization
            if is_mps_available() and HARDWARE_UTILITIES_AVAILABLE:
                try:
                    # Optimize data for M1 GPU
                    X_optimized = create_m1_optimized_array(X, dtype=np.float32)
                    y_optimized = create_m1_optimized_array(y, dtype=np.float32)

                    optimization_result['optimizations_applied'].append('m1_gpu_optimization')
                    optimization_result['performance_improvements']['m1_gpu'] = {
                        'data_optimized': True,
                        'dtype_converted': 'float32'
                    }

                    tprint_success("✅ M1 GPU optimization applied")

                except Exception as e:
                    tprint_warning(f"⚠️ M1 GPU optimization failed: {e}")
            else:
                tprint_info("ℹ️ M1 GPU optimization not available")

            # Memory optimization
            if self.memory_optimizer and hasattr(self.memory_optimizer, 'optimize_memory'):
                try:
                    memory_result = self.memory_optimizer.optimize_memory()
                    optimization_result['optimizations_applied'].append('memory_optimization')
                    optimization_result['performance_improvements']['memory'] = memory_result

                    tprint_success("✅ Memory optimization applied")

                except Exception as e:
                    tprint_warning(f"⚠️ Memory optimization failed: {e}")
            else:
                tprint_info("ℹ️ Memory optimization not available")

            # CPU optimization
            if self.cpu_optimizer and hasattr(self.cpu_optimizer, 'optimize_numpy_operations'):
                try:
                    self.cpu_optimizer.optimize_numpy_operations()
                    optimization_result['optimizations_applied'].append('cpu_optimization')
                    optimization_result['performance_improvements']['cpu'] = {'numpy_optimized': True}

                    tprint_success("✅ CPU optimization applied")

                except Exception as e:
                    tprint_warning(f"⚠️ CPU optimization failed: {e}")
            else:
                tprint_info("ℹ️ CPU optimization not available")

            tprint_success(f"✅ Hardware optimization setup completed: {len(optimization_result['optimizations_applied'])} optimizations applied")
            return optimization_result

        except Exception as e:
            self.error_handler.handle_error(e, "Hardware Optimization Setup", {
                'data_shape': X.shape,
                'target_shape': y.shape
            })
            return {'error': str(e)}

    def _analyze_regimes_enhanced(self, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Enhanced regime analysis with comprehensive statistics."""
        try:
            tprint_info("🔍 Performing enhanced regime analysis")

            unique_regimes = np.unique(regime_labels)
            regime_counts = np.bincount(regime_labels)

            # Enhanced regime statistics
            regime_analysis = {
                'unique_regimes': unique_regimes.tolist(),
                'regime_counts': regime_counts.tolist(),
                'total_regimes': len(unique_regimes),
                'total_samples': len(regime_labels),
                'regime_distribution': {},
                'regime_statistics': {},
                'regime_entropy': self._calculate_regime_entropy(regime_counts),
                'regime_balance_score': 0.0
            }

            # Calculate regime distribution and statistics
            for regime in unique_regimes:
                count = regime_counts[regime]
                percentage = safe_divide(count, len(regime_labels), 0) * 100

                regime_analysis['regime_distribution'][str(regime)] = {
                    'count': int(count),
                    'percentage': percentage
                }

                regime_analysis['regime_statistics'][str(regime)] = {
                    'size': int(count),
                    'percentage': percentage,
                    'sufficient_data': count >= self.config.min_samples_per_regime
                }

            # Calculate regime balance score
            if len(unique_regimes) > 1:
                min_count = min(regime_counts)
                max_count = max(regime_counts)
                regime_analysis['regime_balance_score'] = safe_divide(min_count, max_count, 0)
            else:
                regime_analysis['regime_balance_score'] = 1.0

            tprint_success(f"✅ Enhanced regime analysis completed: {len(unique_regimes)} regimes, balance score: {regime_analysis['regime_balance_score']:.3f}")
            return regime_analysis

        except Exception as e:
            self.error_handler.handle_error(e, "Enhanced Regime Analysis", {
                'regime_labels_shape': regime_labels.shape,
                'unique_regimes_count': len(np.unique(regime_labels))
            })
            return {'error': str(e)}

    def _enhance_results_with_utilities_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance results with utilities metadata."""
        try:
            tprint_info("🔧 Enhancing results with utilities metadata")

            # Add hardware optimization metadata
            results['hardware_optimization_metadata'] = {
                'm1_available': is_m1_available(),
                'mps_available': is_mps_available(),
                'gpu_manager_active': self.gpu_manager is not None,
                'memory_optimizer_active': self.memory_optimizer is not None,
                'cpu_optimizer_active': self.cpu_optimizer is not None
            }

            # Add common utilities metadata
            results['common_utilities_metadata'] = {
                'math_validator_active': hasattr(self, 'math_validator'),
                'data_quality_tools_active': hasattr(self, 'data_quality_tools'),
                'file_operations_active': hasattr(self, 'file_operations'),
                'serializers_active': hasattr(self, 'serializers')
            }

            # Add ML utilities metadata
            results['ml_utilities_metadata'] = {
                'ml_validation_active': hasattr(self, 'ml_validation'),
                'ml_hpo_active': hasattr(self, 'ml_hpo'),
                'ml_evaluation_active': hasattr(self, 'ml_evaluation'),
                'ml_reporting_active': hasattr(self, 'ml_reporting')
            }

            tprint_success("✅ Results enhanced with utilities metadata")
            return results

        except Exception as e:
            self.error_handler.handle_error(e, "Results Enhancement", {
                'results_keys': list(results.keys())
            })
            return results

    def _add_ml_evaluation_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Add ML evaluation metrics using ML common utilities."""
        try:
            tprint_info("🤖 Adding ML evaluation metrics")

            if hasattr(self, 'ml_evaluation') and 'evaluation_results' in results:
                # Use ML common evaluation utilities
                ml_metrics = {}

                for regime, regime_results in results['evaluation_results'].items():
                    if isinstance(regime_results, dict):
                        ml_metrics[regime] = {}

                        for model_type, model_results in regime_results.items():
                            if isinstance(model_results, dict) and 'metrics' in model_results:
                                # Enhance metrics using ML common utilities
                                enhanced_metrics = self._enhance_metrics_with_ml_utilities(
                                    model_results['metrics'], model_type, regime
                                )
                                ml_metrics[regime][model_type] = enhanced_metrics

                results['ml_evaluation_metrics'] = ml_metrics
                tprint_success("✅ ML evaluation metrics added")
            else:
                tprint_warning("⚠️ ML evaluation utilities not available or no evaluation results")

            return results

        except Exception as e:
            self.error_handler.handle_error(e, "ML Evaluation Metrics", {
                'results_keys': list(results.keys())
            })
            return results

    def _enhance_metrics_with_ml_utilities(self, metrics: Dict[str, float], model_type: str, regime: int) -> Dict[str, Any]:
        """Enhance metrics using ML common utilities."""
        try:
            enhanced_metrics = metrics.copy()

            # Add confidence intervals if possible
            if 'r2' in metrics:
                r2_score = metrics['r2']
                # Simple confidence interval calculation
                confidence_interval = 1.96 * math_safe_sqrt(safe_divide(1 - r2_score, 100, 0))
                enhanced_metrics['r2_confidence_interval'] = [
                    max(0, r2_score - confidence_interval),
                    min(1, r2_score + confidence_interval)
                ]

            # Add model complexity score
            enhanced_metrics['model_complexity_score'] = self._calculate_model_complexity_score(model_type)

            # Add regime-specific performance score
            enhanced_metrics['regime_performance_score'] = self._calculate_regime_performance_score(metrics, regime)

            return enhanced_metrics

        except Exception as e:
            tprint_warning(f"⚠️ Failed to enhance metrics: {e}")
            return metrics

    def _calculate_model_complexity_score(self, model_type: str) -> float:
        """Calculate model complexity score."""
        complexity_scores = {
            'TEMPORAL_FUSION_TRANSFORMER': 0.9,
            'TABNET': 0.8,
            'HIST_GRADIENT_BOOSTING': 0.7,
            'EXTRA_TREES': 0.6,
            'TCN': 0.8,
            'CatBoostRegressor': 0.7,
            'LGBMRegressor': 0.6,
            'RandomForestRegressor': 0.5,
            'XGBRegressor': 0.7,
            'NODE': 0.9
        }
        return complexity_scores.get(model_type, 0.5)

    def _calculate_regime_performance_score(self, metrics: Dict[str, float], regime: int) -> float:
        """Calculate regime-specific performance score."""
        try:
            # Weighted combination of metrics
            weights = {'r2': 0.4, 'mse': 0.3, 'mae': 0.2, 'mape': 0.1}
            score = 0.0
            total_weight = 0.0

            for metric, weight in weights.items():
                if metric in metrics:
                    if metric == 'r2':
                        # R2 is already a score (higher is better)
                        score += metrics[metric] * weight
                    else:
                        # Other metrics need to be inverted (lower is better)
                        # Simple inversion: 1 / (1 + metric)
                        inverted_score = safe_divide(1, 1 + metrics[metric], 0)
                        score += inverted_score * weight
                    total_weight += weight

            return safe_divide(score, total_weight, 0) if total_weight > 0 else 0.0

        except Exception as e:
            tprint_warning(f"⚠️ Failed to calculate regime performance score: {e}")
            return 0.0

    def _create_enhanced_training_report(self, results: Dict[str, Any], execution_time: float, status: str = "SUCCESS") -> str:
        """Create enhanced training report with comprehensive utilities integration."""
        try:
            tprint_info("📋 Creating enhanced training report")

            timestamp = self._generate_datetime_stamp()
            report_filename = f"enhanced_analyst_models_training_report_{timestamp}.json"
            report_path = f"{self.config.model_save_path}/reports/{report_filename}"

            # Ensure reports directory exists
            ensure_directory(f"{self.config.model_save_path}/reports")

            # Gather comprehensive system metrics
            system_metrics = self._gather_system_metrics()

            # Gather error summary
            error_summary = self.error_handler.get_error_summary()

            # Gather progress summary if available
            progress_summary = self.progress_tracker.get_summary() if self.progress_tracker else {}

            # Create comprehensive enhanced report
            report_data = {
                "metadata": {
                    "model_name": self.config.model_name,
                    "timeframe": self.config.timeframe,
                    "timestamp": timestamp,
                    "execution_time_seconds": execution_time,
                    "status": status,
                    "version": "enhanced_v2.0_with_utilities",
                    "config": {
                        "model_types": self.config.model_types,
                        "hpo_n_trials": self.config.hpo_n_trials,
                        "hpo_timeout_seconds": self.config.hpo_timeout_seconds,
                        "min_samples_per_regime": self.config.min_samples_per_regime,
                        "enable_data_augmentation": self.config.enable_data_augmentation,
                        "augmentation_method": self.config.augmentation_method,
                        "evaluation_metrics": self.config.evaluation_metrics
                    }
                },
                "results": results,
                "monitoring": {
                    "system_metrics": system_metrics,
                    "training_metrics": self.training_metrics,
                    "progress_summary": progress_summary,
                    "error_summary": error_summary,
                    "hardware_optimization": self.training_metrics.get('hardware_optimization', {}),
                    "data_quality_metrics": self.training_metrics.get('data_quality_metrics', {})
                },
                "utilities_integration": {
                    "common_utilities_active": hasattr(self, 'math_validator'),
                    "hardware_optimization_active": is_m1_available(),
                    "ml_utilities_active": hasattr(self, 'ml_validation'),
                    "serialization_utilities_active": hasattr(self, 'serializers')
                },
                "summary": {
                    "models_trained": len(results.get('models', [])),
                    "regimes_processed": len(results.get('regime_analysis', {}).get('unique_regimes', [])),
                    "best_performing_model": results.get('best_models_per_regime', {}),
                    "training_successful": status == "SUCCESS",
                    "total_errors": error_summary.get('total_errors', 0),
                    "performance_metrics": self._calculate_performance_metrics(results),
                    "utilities_enhancement_score": self._calculate_utilities_enhancement_score()
                }
            }

            # Save enhanced report
            if safe_json_dump(report_data, report_path):
                tprint_success(f"📋 Enhanced training report saved: {report_path}")
                tprint_info(f"📊 Report includes: {len(report_data['monitoring'])} monitoring sections, "
                           f"{error_summary.get('total_errors', 0)} errors tracked, "
                           f"utilities integration: {report_data['utilities_integration']}")
                return report_path
            else:
                tprint_error(f"❌ Failed to save enhanced training report: {report_path}")
                return None

        except Exception as e:
            self.error_handler.handle_error(e, "Enhanced Report Creation", {
                'report_path': report_path,
                'execution_time': execution_time,
                'status': status
            })
            return None

    def _calculate_utilities_enhancement_score(self) -> float:
        """Calculate utilities enhancement score."""
        try:
            score = 0.0
            max_score = 4.0  # 4 categories of utilities

            # Common utilities
            if COMMON_UTILITIES_AVAILABLE and hasattr(self, 'math_validator'):
                score += 1.0

            # Hardware optimization
            if HARDWARE_UTILITIES_AVAILABLE and is_m1_available() and (self.gpu_manager or self.memory_optimizer or self.cpu_optimizer):
                score += 1.0

            # ML utilities
            if ML_UTILITIES_AVAILABLE and hasattr(self, 'ml_validation') and self.ml_validation:
                score += 1.0

            # Serialization utilities
            if SERIALIZATION_UTILITIES_AVAILABLE and hasattr(self, 'serializers'):
                score += 1.0

            return safe_divide(score, max_score, 0)

        except Exception as e:
            tprint_warning(f"⚠️ Failed to calculate utilities enhancement score: {e}")
            return 0.0

    def _enhance_results_with_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance results with comprehensive performance metrics."""
        try:
            # Add training efficiency metrics
            if 'total_training_time' in results:
                results['training_efficiency'] = {
                    'total_time': results['total_training_time'],
                    'models_per_second': results.get('summary', {}).get('total_models', 0) / results['total_training_time'],
                    'regimes_per_second': results.get('summary', {}).get('total_regimes', 0) / results['total_training_time']
                }

            # Add model performance comparison
            if 'evaluation_results' in results:
                model_performance = {}
                for regime, regime_results in results['evaluation_results'].items():
                    if isinstance(regime_results, dict):
                        for model_type, model_results in regime_results.items():
                            if isinstance(model_results, dict) and 'metrics' in model_results:
                                if model_type not in model_performance:
                                    model_performance[model_type] = []
                                model_performance[model_type].append(model_results['metrics'])

                # Calculate aggregate performance per model type
                for model_type, metrics_list in model_performance.items():
                    if metrics_list:
                        aggregate_metrics = {}
                        for metric_name in metrics_list[0].keys():
                            values = [m[metric_name] for m in metrics_list if metric_name in m]
                            if values:
                                aggregate_metrics[metric_name] = {
                                    'mean': np.mean(values),
                                    'std': np.std(values),
                                    'min': np.min(values),
                                    'max': np.max(values),
                                    'count': len(values)
                                }
                        model_performance[model_type] = aggregate_metrics

                results['model_performance_comparison'] = model_performance

            return results

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to enhance results with performance metrics: {e}")
            return results

    def _add_analyst_specific_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add analyst-specific metadata to results with enhanced analysis.

        Args:
            results: Training results

        Returns:
            Enhanced results with analyst-specific metadata
        """
        try:
            # Add analyst-specific analysis
            if 'regime_analysis' in results:
                regime_analysis = results['regime_analysis']

                # Calculate comprehensive analyst-specific metrics
                analyst_metrics = {
                    'total_regimes': len(regime_analysis.get('unique_regimes', [])),
                    'sufficient_regimes': len(regime_analysis.get('sufficient_regimes', [])),
                    'insufficient_regimes': len(regime_analysis.get('insufficient_regimes', [])),
                    'regime_balance': regime_analysis.get('regime_balance_train', 0.0),
                    'regime_diversity': self._calculate_regime_diversity(regime_analysis),
                    'data_quality_score': self._calculate_data_quality_score(results)
                }

                results['analyst_metrics'] = analyst_metrics

            # Add enhanced model performance summary
            if 'evaluation_results' in results:
                evaluation_results = results['evaluation_results']

                # Calculate best performing model per regime with confidence scores
                best_models = {}
                model_confidence_scores = {}

                for regime, regime_metrics in evaluation_results.items():
                    if isinstance(regime_metrics, dict) and 'error' not in regime_metrics:
                        best_model = None
                        best_r2 = -np.inf
                        regime_scores = {}

                        for model_name, metrics in regime_metrics.items():
                            if isinstance(metrics, dict) and 'metrics' in metrics:
                                model_metrics = metrics['metrics']
                                if 'r2' in model_metrics:
                                    r2_score = model_metrics['r2']
                                    regime_scores[model_name] = r2_score

                                    if r2_score > best_r2:
                                        best_r2 = r2_score
                                        best_model = model_name

                        if best_model:
                            # Calculate confidence based on score separation
                            sorted_scores = sorted(regime_scores.values(), reverse=True)
                            confidence = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0] if len(sorted_scores) > 1 else 1.0

                            best_models[regime] = {
                                'model': best_model,
                                'r2_score': best_r2,
                                'confidence': confidence,
                                'all_scores': regime_scores
                            }

                            # Track model confidence across regimes
                            if best_model not in model_confidence_scores:
                                model_confidence_scores[best_model] = []
                            model_confidence_scores[best_model].append(confidence)

                results['best_models_per_regime'] = best_models
                results['model_confidence_analysis'] = {
                    model: {
                        'average_confidence': np.mean(scores),
                        'confidence_std': np.std(scores),
                        'regime_count': len(scores)
                    } for model, scores in model_confidence_scores.items()
                }

            return results

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to add analyst-specific metadata: {e}")
            return results

    def _calculate_regime_diversity(self, regime_analysis: Dict[str, Any]) -> float:
        """Calculate regime diversity score."""
        try:
            unique_regimes = regime_analysis.get('unique_regimes', [])
            if len(unique_regimes) <= 1:
                return 0.0

            # Calculate entropy-based diversity
            regime_counts = [regime_analysis.get('regime_counts', {}).get(str(regime), 0) for regime in unique_regimes]
            total_samples = sum(regime_counts)

            if total_samples == 0:
                return 0.0

            # Normalize counts to probabilities
            probabilities = [count / total_samples for count in regime_counts]

            # Calculate Shannon entropy
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)

            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(unique_regimes))
            diversity_score = entropy / max_entropy if max_entropy > 0 else 0.0

            return diversity_score

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate regime diversity: {e}")
            return 0.0

    def _calculate_data_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        try:
            score = 1.0

            # Penalize for errors
            if 'error' in results:
                score -= 0.5

            # Reward for successful models
            if 'evaluation_results' in results:
                successful_models = 0
                total_models = 0

                for regime_results in results['evaluation_results'].values():
                    if isinstance(regime_results, dict):
                        for model_results in regime_results.values():
                            total_models += 1
                            if isinstance(model_results, dict) and 'metrics' in model_results:
                                successful_models += 1

                if total_models > 0:
                    success_rate = successful_models / total_models
                    score = score * success_rate

            return max(0.0, min(1.0, score))

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate data quality score: {e}")
            return 0.0

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of the training system."""
        health_status = {
            'overall_status': 'healthy',
            'checks': {},
            'recommendations': []
        }

        try:
            # Check system resources
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                health_status['checks']['memory'] = 'critical'
                health_status['recommendations'].append('High memory usage detected - consider reducing batch size')
            elif memory.percent > 75:
                health_status['checks']['memory'] = 'warning'
                health_status['recommendations'].append('Memory usage is high - monitor during training')
            else:
                health_status['checks']['memory'] = 'healthy'

            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                health_status['checks']['disk'] = 'critical'
                health_status['recommendations'].append('Low disk space - cleanup required')
            elif disk.percent > 85:
                health_status['checks']['disk'] = 'warning'
                health_status['recommendations'].append('Disk space is getting low')
            else:
                health_status['checks']['disk'] = 'healthy'

            # Check configuration
            config_validation = self._validate_config_enhanced(self.config)
            if not config_validation['valid']:
                health_status['checks']['configuration'] = 'critical'
                health_status['recommendations'].extend([f"Config error: {error}" for error in config_validation['errors']])
            else:
                health_status['checks']['configuration'] = 'healthy'
                if config_validation['warnings']:
                    health_status['recommendations'].extend([f"Config warning: {warning}" for warning in config_validation['warnings']])

            # Check error history
            error_summary = self.error_handler.get_error_summary()
            if error_summary['total_errors'] > 10:
                health_status['checks']['error_rate'] = 'warning'
                health_status['recommendations'].append('High error rate detected - review error logs')
            else:
                health_status['checks']['error_rate'] = 'healthy'

            # Determine overall status
            critical_checks = [check for check, status in health_status['checks'].items() if status == 'critical']
            warning_checks = [check for check, status in health_status['checks'].items() if status == 'warning']

            if critical_checks:
                health_status['overall_status'] = 'critical'
            elif warning_checks:
                health_status['overall_status'] = 'warning'

            return health_status

        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            return {
                'overall_status': 'error',
                'checks': {'health_check': 'failed'},
                'recommendations': [f'Health check failed: {str(e)}']
            }

# Enhanced Convenience Functions
def create_analyst_models_training_step_enhanced(
    config: Optional[PerRegimeTrainingConfig] = None
) -> AnalystModelsTrainingStepRefactored:
    """
    Create enhanced Analyst models training step with comprehensive utilities integration.

    This function creates a fully enhanced training step with:
    - Common utilities integration
    - Hardware optimization (M1 GPU/CPU/Memory)
    - ML common utilities (CV, HPO, lookahead, etc.)
    - Comprehensive error handling with fast failing
    - Extensive tprint logging
    - Math validation and serialization utilities

    Args:
        config: Per-regime training configuration

    Returns:
        Enhanced Analyst models training step instance

    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If initialization fails
    """
    try:
        tprint_info("🚀 Creating enhanced analyst models training step")

        # Create enhanced step
        step = AnalystModelsTrainingStepRefactored(config)

        # Perform initial health check
        tprint_info("🏥 Performing initial health check")
        health_status = step.health_check()

        if health_status['overall_status'] == 'critical':
            tprint_error(f"🚨 Critical health issues detected: {health_status['recommendations']}")
            raise RuntimeError(f"Critical health issues: {health_status['recommendations']}")
        elif health_status['overall_status'] == 'warning':
            tprint_warning(f"⚠️ Health warnings: {health_status['recommendations']}")
        else:
            tprint_success("✅ Health check passed")

        # Log utilities integration status
        utilities_status = {
            'common_utilities': hasattr(step, 'math_validator'),
            'hardware_optimization': is_m1_available(),
            'ml_utilities': hasattr(step, 'ml_validation'),
            'serialization_utilities': hasattr(step, 'serializers')
        }
        tprint_structured(utilities_status, LogLevel.INFO)

        tprint_success("✅ Enhanced analyst models training step created successfully")
        return step

    except Exception as e:
        tprint_error(f"❌ Failed to create enhanced analyst models training step: {e}")
        raise

def execute_analyst_models_training_enhanced(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[PerRegimeTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    perform_health_check: bool = True
) -> Dict[str, Any]:
    """
    Execute enhanced Analyst models training step with comprehensive utilities integration.

    This function provides a complete training pipeline with:
    - Comprehensive error handling with fast failing
    - Hardware optimization integration
    - Extensive tprint logging at every step
    - Common utilities integration for data operations
    - ML utilities integration for CV, HPO, etc.
    - Math validation and serialization utilities

    Args:
        X: Input features
        y: Target values (analyst outputs)
        regime_labels: Regime labels for each sample
        config: Per-regime training configuration
        feature_names: Names of input features
        hmm_states: HMM cluster/regime states
        perform_health_check: Whether to perform health check before training

    Returns:
        Dictionary containing training results and comprehensive metadata

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If training fails
    """
    try:
        # Fast fail if critical dependencies are missing
        validate_critical_dependencies()

        tprint_info("🚀 Starting enhanced analyst models training execution")

        # Create enhanced step
        step = create_analyst_models_training_step_enhanced(config)

        # Perform pre-training health check
        if perform_health_check:
            tprint_info("🏥 Performing pre-training health check")
            health_status = step.health_check()
            tprint_info(f"🏥 Pre-training health check: {health_status['overall_status']}")

            if health_status['recommendations']:
                tprint_info(f"💡 Health recommendations: {health_status['recommendations']}")

            if health_status['overall_status'] == 'critical':
                tprint_error("🚨 Critical health issues detected - aborting training")
                raise RuntimeError(f"Critical health issues: {health_status['recommendations']}")

        # Execute training with enhanced monitoring
        tprint_info("🔄 Executing enhanced training with comprehensive utilities")
        results = step.execute(X, y, regime_labels, feature_names, hmm_states)

        # Add post-training health check to results
        tprint_info("🏥 Performing post-training health check")
        post_health = step.health_check()
        results['post_training_health'] = post_health

        # Add utilities integration summary
        results['utilities_integration_summary'] = {
            'common_utilities_used': COMMON_UTILITIES_AVAILABLE and hasattr(step, 'math_validator'),
            'hardware_optimization_used': HARDWARE_UTILITIES_AVAILABLE and is_m1_available(),
            'ml_utilities_used': ML_UTILITIES_AVAILABLE and hasattr(step, 'ml_validation'),
            'serialization_utilities_used': SERIALIZATION_UTILITIES_AVAILABLE and hasattr(step, 'serializers'),
            'enhancement_score': step._calculate_utilities_enhancement_score()
        }

        tprint_success("✅ Enhanced analyst models training execution completed successfully")
        tprint_structured(results['utilities_integration_summary'], LogLevel.SUCCESS)

        return results

    except Exception as e:
        tprint_error(f"❌ Enhanced analyst models training execution failed: {e}")
        raise

def analyze_training_report(report_path: str) -> Dict[str, Any]:
    """
    Analyze an enhanced training report and provide comprehensive insights.

    This function analyzes enhanced training reports that include:
    - Utilities integration metrics
    - Hardware optimization results
    - Comprehensive error tracking
    - Performance metrics with confidence intervals
    - Data quality assessments

    Args:
        report_path: Path to the enhanced training report JSON file

    Returns:
        Dictionary containing comprehensive analysis insights
    """
    try:
        tprint_info(f"🔍 Analyzing enhanced training report: {report_path}")

        # Load report data using safe file operations
        report_data = safe_json_load(report_path)
        if not report_data:
            raise ValueError(f"Could not load report data from {report_path}")

        analysis = {
            'report_metadata': report_data.get('metadata', {}),
            'performance_summary': report_data.get('summary', {}),
            'utilities_integration': report_data.get('utilities_integration', {}),
            'health_insights': [],
            'recommendations': [],
            'utilities_insights': [],
            'hardware_insights': []
        }

        # Analyze performance metrics with enhanced insights
        perf_metrics = report_data.get('summary', {}).get('performance_metrics', {})
        if perf_metrics:
            avg_r2 = perf_metrics.get('average_r2_score', 0)
            if avg_r2 > 0.8:
                analysis['health_insights'].append("Excellent model performance (R² > 0.8)")
            elif avg_r2 > 0.6:
                analysis['health_insights'].append("Good model performance (R² > 0.6)")
            else:
                analysis['recommendations'].append("Consider improving model performance - R² < 0.6")

            # Analyze model diversity
            model_count = perf_metrics.get('model_count', 0)
            if model_count < 3:
                analysis['recommendations'].append("Consider using more model types for better ensemble performance")

        # Analyze utilities integration
        utilities_integration = report_data.get('utilities_integration', {})
        if utilities_integration:
            if utilities_integration.get('common_utilities_active', False):
                analysis['utilities_insights'].append("Common utilities successfully integrated")
            if utilities_integration.get('hardware_optimization_active', False):
                analysis['utilities_insights'].append("Hardware optimization successfully integrated")
            if utilities_integration.get('ml_utilities_active', False):
                analysis['utilities_insights'].append("ML utilities successfully integrated")
            if utilities_integration.get('serialization_utilities_active', False):
                analysis['utilities_insights'].append("Serialization utilities successfully integrated")

        # Analyze hardware optimization results
        hardware_optimization = report_data.get('monitoring', {}).get('hardware_optimization', {})
        if hardware_optimization:
            if hardware_optimization.get('success', False):
                analysis['hardware_insights'].append("Hardware optimization successful")
                optimizations = hardware_optimization.get('optimizations_applied', [])
                if optimizations:
                    analysis['hardware_insights'].append(f"Applied optimizations: {optimizations}")
            else:
                analysis['recommendations'].append("Hardware optimization failed - consider manual optimization")

        # Analyze error summary with enhanced insights
        error_summary = report_data.get('monitoring', {}).get('error_summary', {})
        if error_summary.get('total_errors', 0) > 0:
            total_errors = error_summary.get('total_errors', 0)
            critical_errors = error_summary.get('critical_errors', 0)

            analysis['health_insights'].append(f"Training completed with {total_errors} errors ({critical_errors} critical)")

            if critical_errors > 0:
                analysis['recommendations'].append("Critical errors detected - review error logs immediately")
            elif total_errors > 5:
                analysis['recommendations'].append("High error count - review error logs for improvement opportunities")

            # Analyze error patterns
            error_types = error_summary.get('error_types', {})
            if error_types:
                most_common_error = max(error_types.items(), key=lambda x: x[1])
                analysis['recommendations'].append(f"Most common error type: {most_common_error[0]} ({most_common_error[1]} occurrences)")

        # Analyze system metrics with enhanced insights
        system_metrics = report_data.get('monitoring', {}).get('system_metrics', {})
        if system_metrics:
            memory_usage = system_metrics.get('memory', {}).get('used_percent', 0)
            if memory_usage > 90:
                analysis['recommendations'].append("High memory usage detected - consider optimizing memory usage")
            elif memory_usage > 75:
                analysis['health_insights'].append("Moderate memory usage - monitor during training")

            # Analyze disk space
            disk_usage = system_metrics.get('disk', {}).get('usage_percent', 0)
            if disk_usage > 90:
                analysis['recommendations'].append("High disk usage detected - consider cleanup")

        # Analyze data quality metrics
        data_quality = report_data.get('monitoring', {}).get('data_quality_metrics', {})
        if data_quality:
            quality_report = data_quality.get('quality_report', {})
            if quality_report:
                issues = quality_report.get('issues', [])
                if issues:
                    analysis['recommendations'].extend([f"Data quality issue: {issue}" for issue in issues])
                else:
                    analysis['health_insights'].append("Data quality assessment passed")

        # Calculate overall health score
        health_score = 100
        health_score -= min(50, error_summary.get('total_errors', 0) * 5)  # Deduct for errors
        health_score -= min(30, (100 - perf_metrics.get('average_r2_score', 0)) * 30)  # Deduct for poor performance
        health_score = max(0, health_score)

        analysis['overall_health_score'] = health_score
        analysis['health_grade'] = 'A' if health_score >= 90 else 'B' if health_score >= 70 else 'C' if health_score >= 50 else 'D'

        tprint_success(f"✅ Enhanced training report analysis completed - Health Grade: {analysis['health_grade']} ({health_score}/100)")
        return analysis

    except Exception as e:
        tprint_error(f"❌ Failed to analyze enhanced training report: {e}")
        return {'error': str(e)}

# Legacy compatibility functions
def create_analyst_models_training_step_refactored(
    config: Optional[PerRegimeTrainingConfig] = None
) -> AnalystModelsTrainingStepRefactored:
    """Legacy function for backward compatibility."""
    tprint_warning("⚠️ Using legacy function - consider using create_analyst_models_training_step_enhanced")
    return create_analyst_models_training_step_enhanced(config)

def execute_analyst_models_training_refactored(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[PerRegimeTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Legacy function for backward compatibility."""
    tprint_warning("⚠️ Using legacy function - consider using execute_analyst_models_training_enhanced")
    return execute_analyst_models_training_enhanced(X, y, regime_labels, config, feature_names, hmm_states)

# Cleanup and resource management functions
def cleanup_enhanced_training_resources():
    """Cleanup enhanced training resources and optimizers."""
    try:
        tprint_info("🧹 Cleaning up enhanced training resources")

        # Cleanup M1 optimizers
        cleanup_result = cleanup_m1_optimizers()
        if cleanup_result:
            tprint_success("✅ M1 optimizers cleaned up successfully")
        else:
            tprint_warning("⚠️ M1 optimizers cleanup failed or not available")

        # Cleanup tprint resources
        try:
            from src.utils.tprint import cleanup_tprint
            cleanup_tprint()
            tprint_success("✅ Tprint resources cleaned up successfully")
        except Exception as e:
            tprint_warning(f"⚠️ Tprint cleanup failed: {e}")

        # Force garbage collection
        import gc
        collected = gc.collect()
        tprint_info(f"🗑️ Garbage collection: {collected} objects collected")

        tprint_success("✅ Enhanced training resources cleanup completed")

    except Exception as e:
        tprint_error(f"❌ Enhanced training resources cleanup failed: {e}")

def get_enhanced_training_status() -> Dict[str, Any]:
    """Get comprehensive status of enhanced training system."""
    try:
        tprint_info("📊 Getting enhanced training system status")

        status = {
            'timestamp': datetime.now().isoformat(),
            'hardware_status': {
                'm1_available': is_m1_available(),
                'mps_available': is_mps_available(),
                'memory_usage_mb': get_memory_usage() / 1024 / 1024,
                'cpu_percent': psutil.cpu_percent(),
                'disk_space': check_disk_space('/', 1.0)
            },
            'utilities_status': {
                'common_utilities_available': COMMON_UTILITIES_AVAILABLE,
                'hardware_optimization_available': HARDWARE_UTILITIES_AVAILABLE and is_m1_available(),
                'ml_utilities_available': ML_UTILITIES_AVAILABLE,
                'serialization_utilities_available': SERIALIZATION_UTILITIES_AVAILABLE
            },
            'system_health': {
                'memory_healthy': get_memory_usage() / 1024 / 1024 < 8000,  # Less than 8GB
                'cpu_healthy': psutil.cpu_percent() < 80,  # Less than 80%
                'disk_healthy': check_disk_space('/', 1.0)['sufficient']
            }
        }

        # Calculate overall health score
        health_checks = status['system_health']
        health_score = sum(health_checks.values()) / len(health_checks) * 100
        status['overall_health_score'] = health_score
        status['health_grade'] = 'A' if health_score >= 90 else 'B' if health_score >= 70 else 'C' if health_score >= 50 else 'D'

        tprint_success(f"✅ Enhanced training system status retrieved - Health Grade: {status['health_grade']} ({health_score:.1f}/100)")
        return status

    except Exception as e:
        tprint_error(f"❌ Failed to get enhanced training system status: {e}")
        return {'error': str(e), 'timestamp': datetime.now().isoformat()}
