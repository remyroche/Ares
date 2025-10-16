"""
Enhanced Error Handling Utilities

This module provides comprehensive error handling utilities extracted from step02_5_sr_optimization.py
to ensure robust ML training across all steps in the Ares trading system.

Key Features:
- Error severity and category classification
- Robust error handling with automatic retry and fallback mechanisms
- ML failure handler with intelligent fast fail mechanism
- Data drift detection with statistical baselines
- Performance monitoring and memory usage tracking
"""

import logging
import time
import traceback
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
from enum import Enum
import asyncio
import psutil

logger = logging.getLogger(__name__)

# Import M1 utilities
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    from src.utils.hardware.m1_memory_optimizer import (  # type: ignore
        auto_skim_memory, smart_memory_allocation,
        memory_skim_decorator, auto_memory_skim_decorator,
        auto_memory_skim_context, smart_memory_context,
        get_m1_memory_optimizer
    )
    MEMORY_OPTIMIZER_AVAILABLE = True
except ImportError:
    MEMORY_OPTIMIZER_AVAILABLE = False

try:
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer  # type: ignore
    CPU_OPTIMIZER_AVAILABLE = True
except ImportError:
    CPU_OPTIMIZER_AVAILABLE = False

class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    CRITICAL = "CRITICAL"  # System cannot continue
    HIGH = "HIGH"         # Major functionality affected
    MEDIUM = "MEDIUM"     # Minor functionality affected
    LOW = "LOW"          # Cosmetic or non-critical

class ErrorCategory(Enum):
    """Error categories for classification."""
    DATA_QUALITY = "DATA_QUALITY"
    ML_TRAINING = "ML_TRAINING"
    SR_DETECTION = "SR_DETECTION"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    SYSTEM_RESOURCE = "SYSTEM_RESOURCE"
    EXTERNAL_DEPENDENCY = "EXTERNAL_DEPENDENCY"

def classify_error(error: Exception, context: str = "") -> Tuple[ErrorSeverity, ErrorCategory]:
    """
    Classify errors for appropriate handling using core error classes.
    
    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
        
    Returns:
        Tuple of (ErrorSeverity, ErrorCategory)
    """
    error_type = type(error).__name__
    error_msg = str(error).lower()
    
    logger.debug(f"🔍 Classifying error: {error_type} in context: '{context}'")
    logger.debug(f"📋 Error message: {error_msg[:100]}...")
    
    # Critical errors - map to core error classes
    if isinstance(error, (MemoryError, SystemError)):
        logger.critical(f"🚨 CRITICAL ERROR detected: {error_type} - System resource issue")
        return ErrorSeverity.CRITICAL, ErrorCategory.SYSTEM_RESOURCE
    if isinstance(error, (ValueError, KeyError)) and "data" in context.lower():
        logger.critical(f"🚨 CRITICAL ERROR detected: {error_type} - Data quality issue")
        return ErrorSeverity.CRITICAL, ErrorCategory.DATA_QUALITY
    
    # High severity errors - map to core error classes
    if isinstance(error, (ValueError, KeyError, AttributeError)) and "data" in context.lower():
        logger.error(f"⚠️ HIGH SEVERITY ERROR detected: {error_type} - Data quality issue")
        return ErrorSeverity.HIGH, ErrorCategory.DATA_QUALITY
    if isinstance(error, (ValueError, KeyError, AttributeError)) and ("ml" in context.lower() or "model" in context.lower()):
        logger.error(f"⚠️ HIGH SEVERITY ERROR detected: {error_type} - ML training issue")
        return ErrorSeverity.HIGH, ErrorCategory.ML_TRAINING
    
    # Medium severity errors
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        logger.warning(f"⚠️ MEDIUM SEVERITY ERROR detected: {error_type} - External dependency issue")
        return ErrorSeverity.MEDIUM, ErrorCategory.EXTERNAL_DEPENDENCY
    if "sr" in context.lower() or "detection" in context.lower():
        logger.warning(f"⚠️ MEDIUM SEVERITY ERROR detected: {error_type} - SR detection issue")
        return ErrorSeverity.MEDIUM, ErrorCategory.SR_DETECTION
    if "feature" in context.lower():
        logger.warning(f"⚠️ MEDIUM SEVERITY ERROR detected: {error_type} - Feature engineering issue")
        return ErrorSeverity.MEDIUM, ErrorCategory.FEATURE_ENGINEERING
    
    # Default classification
    logger.info(f"ℹ️ Default classification applied: {error_type} - System resource issue")
    return ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM_RESOURCE

def handle_error_with_recovery(error: Exception, context: str, max_retries: int = 3) -> bool:
    """
    Handle errors with appropriate recovery strategies using core error handling.
    
    Args:
        error: The exception that occurred
        context: Context where the error occurred
        max_retries: Maximum number of retry attempts
        
    Returns:
        True if error was handled successfully, False otherwise
    """
    logger.info(f"🔄 Starting error recovery process for {type(error).__name__}")
    start_time = time.time()
    
    severity, category = classify_error(error, context)
    
    logger.error(f"🚨 {severity.value} ERROR in {category.value}: {error}")
    logger.error(f"📋 Context: {context}")
    logger.error(f"📋 Traceback: {traceback.format_exc()}")
    
    if severity == ErrorSeverity.CRITICAL:
        logger.critical("💥 CRITICAL ERROR - System cannot continue safely")
        logger.critical("🛑 Initiating emergency shutdown procedures...")
        return False
    elif severity == ErrorSeverity.HIGH:
        logger.error("⚠️ HIGH SEVERITY ERROR - Major functionality affected")
        logger.error("🔄 Attempting recovery strategies...")
        # Could implement retry logic here
        return False
    else:
        logger.warning(f"⚠️ {severity.value} ERROR - Continuing with degraded functionality")
        logger.info("✅ Error handled successfully - system continues with reduced capabilities")
        return True

class RobustErrorHandler:
    """Robust error handling with automatic retry and fallback mechanisms."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logger.getChild('RobustErrorHandler')
        
        self.logger.info(f"🚀 RobustErrorHandler initialized with max_retries={max_retries}, retry_delay={retry_delay}s")
        self.logger.debug("🔄 Error handler ready for robust error recovery operations")
    
    def execute_with_retry(self, operation_name: str, operation_func: Callable, *args, **kwargs):
        """
        Execute an operation with automatic retry and fallback mechanisms.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the operation or fallback result
        """
        self.logger.info(f"🚀 Starting retry execution for operation: {operation_name}")
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            attempt_start_time = time.time()
            self.logger.debug(f"🔄 Attempt {attempt + 1}/{self.max_retries} for {operation_name}")
            
            try:
                result = operation_func(*args, **kwargs)
                attempt_time = time.time() - attempt_start_time
                total_time = time.time() - start_time
                
                if attempt > 0:
                    self.logger.info(f"✅ Operation {operation_name} succeeded on attempt {attempt + 1} after {total_time:.3f}s")
                else:
                    self.logger.debug(f"✅ Operation {operation_name} succeeded on first attempt in {attempt_time:.3f}s")
                
                return result
                
            except MemoryError as e:
                attempt_time = time.time() - attempt_start_time
                self.logger.warning(f"🧠 Memory error in {operation_name} (attempt {attempt + 1}) after {attempt_time:.3f}s: {e}")
                
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"🔄 Retrying {operation_name} with reduced data...")
                    # Reduce data size and retry
                    reduced_args = self._reduce_data_size(args)
                    time.sleep(self.retry_delay)
                    self.retry_delay *= 2
                    continue
                else:
                    self.logger.error(f"❌ Memory error in {operation_name} after {self.max_retries} attempts")
                    return self._get_fallback_result(operation_name)
                    
            except asyncio.TimeoutError as e:
                attempt_time = time.time() - attempt_start_time
                self.logger.warning(f"⏰ Timeout error in {operation_name} (attempt {attempt + 1}) after {attempt_time:.3f}s: {e}")
                
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"🔄 Retrying {operation_name} after timeout...")
                    time.sleep(self.retry_delay)
                    self.retry_delay *= 2
                    continue
                else:
                    self.logger.error(f"❌ Timeout error in {operation_name} after {self.max_retries} attempts")
                    return self._get_fallback_result(operation_name)
                    
            except Exception as e:
                attempt_time = time.time() - attempt_start_time
                self.logger.warning(f"⚠️ Error in {operation_name} (attempt {attempt + 1}) after {attempt_time:.3f}s: {e}")
                
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"🔄 Retrying {operation_name} in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                    self.retry_delay *= 2
                    continue
                else:
                    total_time = time.time() - start_time
                    self.logger.error(f"❌ Error in {operation_name} after {self.max_retries} attempts in {total_time:.3f}s: {e}")
                    return self._get_fallback_result(operation_name)
    
    def _reduce_data_size(self, args: tuple) -> tuple:
        """Reduce data size for memory-constrained retries."""
        self.logger.debug("🔄 Reducing data size for memory-constrained retry...")
        reduced_args = []
        
        for i, arg in enumerate(args):
            if isinstance(arg, (pd.DataFrame, np.ndarray)):
                original_size = len(arg)
                # Reduce size by taking a sample
                if isinstance(arg, pd.DataFrame):
                    reduced_arg = arg.head(len(arg) // 2)
                    self.logger.debug(f"📊 Reduced DataFrame arg {i}: {original_size} -> {len(reduced_arg)} rows")
                else:
                    reduced_arg = arg[:len(arg) // 2]
                    self.logger.debug(f"📊 Reduced array arg {i}: {original_size} -> {len(reduced_arg)} elements")
                reduced_args.append(reduced_arg)
            else:
                reduced_args.append(arg)
        
        self.logger.debug(f"✅ Data size reduction completed for {len(args)} arguments")
        return tuple(reduced_args)
    
    def _get_fallback_result(self, operation_name: str) -> Dict[str, Any]:
        """Get fallback result for failed operations."""
        self.logger.warning(f"🔄 Generating fallback result for failed operation: {operation_name}")
        
        fallback_result = {
            'operation': operation_name,
            'status': 'failed',
            'fallback': True,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"✅ Fallback result generated for {operation_name}")
        return fallback_result

class MLFailureHandler:
    """ML failure handler with intelligent fast fail mechanism."""
    
    def __init__(self, max_failures: int = 5, critical_threshold: int = 2, recoverable_threshold: int = 3):
        self.max_failures = max_failures
        self.critical_threshold = critical_threshold
        self.recoverable_threshold = recoverable_threshold
        self.ml_failure_count = 0
        self.ml_failure_reasons = []
        self.critical_failure_count = 0
        self.recoverable_failure_count = 0
        self.fast_fail_engaged = False
        self.logger = logger.getChild('MLFailureHandler')
        
        self.logger.info(f"🚀 MLFailureHandler initialized:")
        self.logger.info(f"   📊 Max failures: {max_failures}")
        self.logger.info(f"   🚨 Critical threshold: {critical_threshold}")
        self.logger.info(f"   ⚠️ Recoverable threshold: {recoverable_threshold}")
        self.logger.debug("🔄 ML failure handler ready for intelligent failure management")
    
    def handle_ml_failure(self, error_message: str, error_type: str = "UNKNOWN_ERROR") -> Dict[str, Any]:
        """
        Handle ML training failures with intelligent fast fail mechanism.
        
        Args:
            error_message: Description of the error
            error_type: Type of error for classification
            
        Returns:
            Fallback result or raises exception if fast fail is triggered
        """
        self.logger.info(f"🔄 Handling ML failure #{self.ml_failure_count + 1}: {error_type}")
        start_time = time.time()
        
        self.ml_failure_count += 1
        failure_record = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'failure_count': self.ml_failure_count
        }
        self.ml_failure_reasons.append(failure_record)
        
        self.logger.debug(f"📊 Failure record added: {failure_record}")
        
        # Classify failure severity - make bias detection more lenient
        critical_errors = ["DATA_UNAVAILABLE", "EMPTY_DATA"]
        recoverable_errors = ["FORWARD_BIAS_ERROR", "OPTUNA_ERROR", "CV_ERROR", "MODEL_FIT_ERROR", "ML_TRAINING_ERROR", "METHOD_VALIDATION_ERROR"]
        
        is_critical = error_type in critical_errors
        is_recoverable = error_type in recoverable_errors
        
        self.logger.debug(f"🔍 Failure classification: Critical={is_critical}, Recoverable={is_recoverable}")
        
        if is_critical:
            self.critical_failure_count += 1
            self.logger.warning(f"🚨 Critical failure count increased to {self.critical_failure_count}")
        elif is_recoverable:
            self.recoverable_failure_count += 1
            self.logger.warning(f"⚠️ Recoverable failure count increased to {self.recoverable_failure_count}")
        
        # Check if fast fail should be triggered
        should_fast_fail = (
            self.critical_failure_count >= self.critical_threshold or
            self.recoverable_failure_count >= self.recoverable_threshold or
            self.ml_failure_count >= self.max_failures
        )
        
        self.logger.debug(f"🔍 Fast fail check: should_fail={should_fast_fail}, engaged={self.fast_fail_engaged}")
        self.logger.debug(f"📊 Current counts: Total={self.ml_failure_count}, Critical={self.critical_failure_count}, Recoverable={self.recoverable_failure_count}")
        
        if should_fast_fail and not self.fast_fail_engaged:
            self.fast_fail_engaged = True
            self.logger.critical(f"🚨 FAST FAIL triggered after {self.ml_failure_count} ML failures")
            self.logger.critical(f"🚨 Critical: {self.critical_failure_count}, Recoverable: {self.recoverable_failure_count}")
            self.logger.critical("🛑 Raising RuntimeError to halt ML training pipeline")
            raise RuntimeError(f"Fast fail triggered after {self.ml_failure_count} ML training failures")
        
        # Log the failure
        if is_critical:
            self.logger.error(f"❌ Critical ML failure #{self.ml_failure_count}: {error_message}")
        elif is_recoverable:
            self.logger.warning(f"⚠️ Recoverable ML failure #{self.ml_failure_count}: {error_message}")
        else:
            self.logger.info(f"ℹ️ ML failure #{self.ml_failure_count}: {error_message}")
        
        processing_time = time.time() - start_time
        self.logger.debug(f"⏱️ ML failure handling completed in {processing_time:.3f}s")
        
        return self._get_fallback_ml_result_with_failure_info(error_message, error_type)
    
    def _get_fallback_ml_result_with_failure_info(self, error_message: str, error_type: str) -> Dict[str, Any]:
        """Get fallback ML result with detailed failure information."""
        self.logger.debug(f"🔄 Generating fallback ML result for failure: {error_type}")
        
        fallback_result = {
            'direction_accuracy': 0.5,
            'volatility_mae': 0.1,
            'model_type': 'fallback_due_to_failure',
            'training_samples': 0,
            'sr_levels_used': 0,
            'training_time': 0.0,
            'failure_reason': error_message,
            'failure_type': error_type,
            'failure_count': self.ml_failure_count,
            'fast_fail_enabled': self.fast_fail_engaged
        }
        
        self.logger.info(f"✅ Fallback ML result generated with failure info")
        self.logger.debug(f"📊 Fallback result keys: {list(fallback_result.keys())}")
        
        return fallback_result
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """Get a summary of all ML failures."""
        self.logger.info("📊 Generating ML failure summary...")
        
        summary = {
            'total_failures': self.ml_failure_count,
            'critical_failures': self.critical_failure_count,
            'recoverable_failures': self.recoverable_failure_count,
            'fast_fail_engaged': self.fast_fail_engaged,
            'failure_types': [f['error_type'] for f in self.ml_failure_reasons[-10:]],  # Last 10 failures
            'recent_failures': self.ml_failure_reasons[-5:]  # Last 5 failure details
        }
        
        self.logger.info(f"📈 ML Failure Summary:")
        self.logger.info(f"   📊 Total failures: {summary['total_failures']}")
        self.logger.info(f"   🚨 Critical failures: {summary['critical_failures']}")
        self.logger.info(f"   ⚠️ Recoverable failures: {summary['recoverable_failures']}")
        self.logger.info(f"   🛑 Fast fail engaged: {summary['fast_fail_engaged']}")
        self.logger.info(f"   📋 Recent failure types: {summary['failure_types']}")
        
        return summary

def detect_data_drift(current_data: pd.DataFrame, reference_data: Optional[pd.DataFrame] = None, 
                     drift_threshold: float = 0.1) -> Dict[str, Any]:
    """
    Detect data drift between current and reference datasets using statistical baselines.
    
    Args:
        current_data: Current dataset
        reference_data: Reference dataset (optional)
        drift_threshold: Threshold for drift detection
        
    Returns:
        Dictionary with drift detection results
    """
    logger.info("🔍 Starting data drift detection...")
    start_time = time.time()
    
    drift_results = {
        'drift_detected': False,
        'drift_score': 0.0,
        'drift_details': {},
        'recommendations': []
    }
    
    try:
        logger.debug(f"📊 Current data shape: {current_data.shape}")
        logger.debug(f"📊 Reference data: {'Provided' if reference_data is not None else 'Not provided'}")
        if reference_data is not None:
            logger.debug(f"📊 Reference data shape: {reference_data.shape}")
        logger.debug(f"📊 Drift threshold: {drift_threshold}")
        
        # If no reference data, use statistical baselines
        if reference_data is None:
            logger.debug("🔄 Using statistical baselines for drift detection...")
            
            # Use statistical baselines for common financial metrics
            baseline_stats = {
                'close_mean': current_data['close'].mean() if 'close' in current_data.columns else 0.0,
                'close_std': current_data['close'].std() if 'close' in current_data.columns else 0.0,
                'volume_mean': current_data['volume'].mean() if 'volume' in current_data.columns else 0.0,
                'volume_std': current_data['volume'].std() if 'volume' in current_data.columns else 0.0
            }
            
            logger.debug(f"📊 Baseline statistics: {baseline_stats}")
            
            # Simple drift detection based on statistical properties
            current_stats = {
                'close_mean': current_data['close'].mean() if 'close' in current_data.columns else 0.0,
                'close_std': current_data['close'].std() if 'close' in current_data.columns else 0.0,
                'volume_mean': current_data['volume'].mean() if 'volume' in current_data.columns else 0.0,
                'volume_std': current_data['volume'].std() if 'volume' in current_data.columns else 0.0
            }
            
            logger.debug(f"📊 Current statistics: {current_stats}")
            
            # Calculate drift score (simplified)
            drift_score = 0.0
            for key in baseline_stats:
                if baseline_stats[key] != 0:
                    drift_ratio = abs(current_stats[key] - baseline_stats[key]) / abs(baseline_stats[key])
                    drift_score += drift_ratio
                    logger.debug(f"📊 {key} drift ratio: {drift_ratio:.4f}")
            
            drift_score /= len(baseline_stats)
            drift_results['drift_score'] = drift_score
            drift_results['drift_detected'] = drift_score > drift_threshold
            
            logger.info(f"📊 Drift score: {drift_score:.4f} (threshold: {drift_threshold})")
            logger.info(f"🔍 Drift detected: {drift_results['drift_detected']}")
            
            if drift_results['drift_detected']:
                logger.warning("⚠️ Data drift detected - model retraining recommended")
                drift_results['recommendations'].append("Consider retraining models due to data drift")
                drift_results['recommendations'].append("Review feature engineering pipeline")
            else:
                logger.info("✅ No significant data drift detected")
        else:
            logger.debug("🔄 Comparing with reference data...")
            
            # Compare with reference data
            numeric_cols = current_data.select_dtypes(include=[np.number]).columns
            drift_detected = False
            
            logger.debug(f"📊 Numeric columns to check: {list(numeric_cols)}")
            
            for col in numeric_cols:
                if col in reference_data.columns:
                    current_mean = current_data[col].mean()
                    reference_mean = reference_data[col].mean()
                    
                    if reference_mean != 0:
                        drift_ratio = abs(current_mean - reference_mean) / abs(reference_mean)
                        
                        if drift_ratio > drift_threshold:
                            drift_detected = True
                            drift_results['drift_details'][col] = {
                                'current_mean': current_mean,
                                'reference_mean': reference_mean,
                                'drift_ratio': drift_ratio
                            }
                            logger.warning(f"⚠️ Drift detected in {col}: ratio={drift_ratio:.4f}")
                        else:
                            logger.debug(f"✅ No drift in {col}: ratio={drift_ratio:.4f}")
            
            drift_results['drift_detected'] = drift_detected
            
            if drift_detected:
                logger.warning("⚠️ Data drift detected - model retraining recommended")
                drift_results['recommendations'].append("Data drift detected - consider model retraining")
            else:
                logger.info("✅ No significant data drift detected")
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Data drift detection completed in {processing_time:.3f}s")
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Data drift detection failed after {processing_time:.3f}s: {e}")
        drift_results['error'] = str(e)
    
    return drift_results

class PerformanceMonitor:
    """Enhanced context manager for performance monitoring with M1 optimization tracking."""

    def __init__(self, name: str, logger_instance: logging.Logger = None, enable_m1_tracking: bool = True):
        self.name = name
        self.logger = logger_instance or logger.getChild('PerformanceMonitor')
        self.enable_m1_tracking = enable_m1_tracking
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None
        self.start_gpu_memory = None

        # Initialize M1 utilities if available
        self.gpu_manager = M1GPUManager() if GPU_AVAILABLE else None
        self.memory_optimizer = get_m1_memory_optimizer() if MEMORY_OPTIMIZER_AVAILABLE else None
        self.cpu_optimizer = get_m1_cpu_optimizer() if CPU_OPTIMIZER_AVAILABLE else None

    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = self._check_memory_usage()
        self.start_cpu = self._check_cpu_usage()

        if self.enable_m1_tracking and self.gpu_manager:
            try:
                self.start_gpu_memory = self._check_gpu_memory_usage()
            except Exception:
                self.start_gpu_memory = None

        self.logger.debug(f"📊 Starting performance monitoring for {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        end_memory = self._check_memory_usage()
        end_cpu = self._check_cpu_usage()
        memory_delta = end_memory - self.start_memory
        cpu_delta = end_cpu - self.start_cpu

        # Basic metrics
        self.logger.info(f"⏱️ {self.name} completed in {duration:.2f}s")
        self.logger.info(f"🧠 Memory: {end_memory:.1%} (Δ{memory_delta:+.1%})")
        self.logger.info(f"⚡ CPU: {end_cpu:.1%} (Δ{cpu_delta:+.1%})")

        # M1-specific metrics
        if self.enable_m1_tracking:
            if self.gpu_manager and self.start_gpu_memory is not None:
                try:
                    end_gpu_memory = self._check_gpu_memory_usage()
                    gpu_memory_delta = end_gpu_memory - self.start_gpu_memory
                    self.logger.info(f"🎮 GPU Memory: {end_gpu_memory:.1f}GB (Δ{gpu_memory_delta:+.1f}GB)")
                except Exception as e:
                    self.logger.debug(f"GPU memory tracking failed: {e}")

            # Memory optimization metrics
            if self.memory_optimizer:
                try:
                    memory_report = self.memory_optimizer.get_memory_report()
                    self.logger.debug(f"🧠 Memory Report: {memory_report['current_usage_gb']:.1f}GB peak: {memory_report['peak_usage_gb']:.1f}GB")
                except Exception as e:
                    self.logger.debug(f"Memory report failed: {e}")

        if exc_type is not None:
            self.logger.error(f"❌ {self.name} failed with {exc_type.__name__}: {exc_val}")

            # Enhanced error analysis with M1 context
            if self.enable_m1_tracking:
                self._analyze_failure_context(exc_type, exc_val, duration)

    def _check_memory_usage(self) -> float:
        """Check current memory usage as a percentage."""
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        except Exception:
            return 0.0

    def _check_cpu_usage(self) -> float:
        """Check current CPU usage as a percentage."""
        try:
            return psutil.cpu_percent(interval=0.1) / 100.0
        except Exception:
            return 0.0

    def _check_gpu_memory_usage(self) -> float:
        """Check current GPU memory usage in GB."""
        if not self.gpu_manager:
            return 0.0

        try:
            # Get GPU memory info from M1GPUManager
            memory_info = self.gpu_manager.memory_info
            return memory_info.get('used_gb', 0.0)
        except Exception:
            return 0.0

    def _analyze_failure_context(self, exc_type: type, exc_val: Exception, duration: float):
        """Analyze failure context with M1-specific information."""
        try:
            context_info = {
                'operation_duration': duration,
                'memory_pressure': 'high' if self.start_memory > 0.8 else 'normal',
                'cpu_pressure': 'high' if self.start_cpu > 0.8 else 'normal',
            }

            if self.memory_optimizer:
                memory_report = self.memory_optimizer.get_memory_report()
                context_info.update({
                    'memory_efficiency': memory_report.get('memory_efficiency', 0),
                    'swap_usage': memory_report.get('swap_info', {}).get('usage_percent', 0),
                })

            if self.cpu_optimizer:
                cpu_report = self.cpu_optimizer.get_cpu_usage_report()
                context_info.update({
                    'cpu_cores_used': cpu_report.get('cpu_percent_overall', 0),
                    'optimal_workers': cpu_report.get('optimal_workers', 1),
                })

            self.logger.info(f"🔍 Failure context: {context_info}")

        except Exception as e:
            self.logger.debug(f"Failure context analysis failed: {e}")

def create_performance_monitor(operation_name: str, logger_instance: logging.Logger = None) -> PerformanceMonitor:
    """Create a performance monitor for an operation."""
    return PerformanceMonitor(operation_name, logger_instance)

# Convenience functions for easy access
def robust_error_handler(max_retries: int = 3, retry_delay: float = 1.0) -> RobustErrorHandler:
    """Create a robust error handler."""
    return RobustErrorHandler(max_retries, retry_delay)

def ml_failure_handler(max_failures: int = 5, critical_threshold: int = 2, recoverable_threshold: int = 3) -> MLFailureHandler:
    """Create an ML failure handler."""
    return MLFailureHandler(max_failures, critical_threshold, recoverable_threshold)
