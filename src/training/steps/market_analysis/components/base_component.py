"""
Enhanced Base Component Class for Market Analysis Pipeline Components.

This module provides a comprehensive base class for market analysis pipeline components
with full error handling, logging, M1 optimization, and integration with all utility modules.

Features:
- Comprehensive error handling with no silent failures
- Full tprint logging integration
- M1 hardware optimization (GPU, CPU, Memory)
- Integration with all utility modules
- Mathematical validation and safety checks
- Matrix operations and ML utilities
- Data quality validation and processing
- Serialization utilities
- Hardware-accelerated operations
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import asyncio
import traceback
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Core utilities imports
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_performance, tprint_structured, tprint_timer,
    LogLevel, TPrintConfig
)
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe, validate_dataframe_columns,
    safe_convert_dtypes, calculate_data_quality_metrics, safe_merge_dataframes,
    safe_groupby_operation, safe_apply_function, create_summary_statistics,
    safe_drop_columns, safe_rename_columns, validate_timestamp_column,
    safe_timestamp_conversion, get_dataframe_info, safe_filter_dataframe,
    create_data_quality_report, safe_divide, safe_log, safe_sqrt, safe_power,
    safe_mean, safe_std, safe_float, safe_int, validate_finite, validate_positive,
    validate_range, safe_kelly_calculation, safe_weighted_average,
    safe_percentage_change, safe_matrix_inverse, validate_correlation_matrix,
    safe_to_parquet, safe_read_parquet, list_parquet_files, optimize_dataframe_dtypes,
    ensure_directory, safe_file_exists, safe_json_dump, safe_json_load,
    format_bytes, timed_operation, parallel_map, chunked_iterable,
    integrate_with_m1_optimizers, cleanup_m1_optimizers, memory_checkpoint, gpu_context,
    optimize_memory, get_memory_usage, check_disk_space, validate_file_path,
    get_file_size, CommonUtilities
)
from src.utils.common_utilities import (
    CommonUtilities as BaseCommonUtilities
)
from src.utils.math_validation import (
    MathValidation, safe_divide as math_safe_divide, safe_log as math_safe_log,
    safe_sqrt as math_safe_sqrt, safe_power as math_safe_power, validate_finite as math_validate_finite,
    validate_positive as math_validate_positive, validate_range as math_validate_range,
    safe_correlation, safe_covariance, safe_mean as math_safe_mean, safe_std as math_safe_std,
    safe_percentile, validate_correlation_matrix as math_validate_correlation_matrix,
    safe_matrix_inverse as math_safe_matrix_inverse, MathValidationError
)
from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)
from src.utils.hardware.m1_gpu_utils import (
    get_m1_gpu_manager, is_m1_available, is_mps_available, optimize_dataframe_for_m1,
    create_m1_optimized_array, m1_backtesting_simulate, m1_monte_carlo_simulate
)
from src.utils.hardware.m1_memory_optimizer import (
    get_m1_memory_optimizer, optimize_dataframe_memory, optimize_memory as m1_optimize_memory,
    get_memory_usage as m1_get_memory_usage
)
from src.utils.hardware.m1_cpu_optimizer import (
    get_m1_cpu_optimizer, parallel_map_m1, create_m1_optimized_thread_pool,
    run_cpu_intensive_task, parallel_backtesting_worker, parallel_monte_carlo_simulation,
    run_monte_carlo_batch
)

# Import ML utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    ML_UTILITIES_AVAILABLE = True
except ImportError:
    ML_UTILITIES_AVAILABLE = False
    tprint_warning("ML utilities not available - some features will be disabled")

# Import matrix operations
try:
    from src.utils.matrix_operations.unified_operations import (
        MatrixOperations, safe_matrix_multiply, safe_matrix_inverse as matrix_safe_inverse,
        safe_correlation_matrix, safe_eigenvalue_decomposition, safe_svd_decomposition
    )
    MATRIX_UTILITIES_AVAILABLE = True
except ImportError:
    MATRIX_UTILITIES_AVAILABLE = False
    tprint_warning("Matrix utilities not available - some features will be disabled")

# Import NAS/TAS utilities
try:
    from src.utils.nas_tas.bayesian_tpe_optimizer import BayesianTPEOptimizer as NASBayesianTPE
    from src.utils.nas_tas.unified_evaluator import UnifiedEvaluator
    from src.utils.nas_tas.monte_carlo_engine import MonteCarloEngine
    NAS_TAS_AVAILABLE = True
except ImportError:
    NAS_TAS_AVAILABLE = False
    tprint_warning("NAS/TAS utilities not available - some features will be disabled")

# Import data utilities
try:
    from src.utils.data.klines_parquet import KlineParquetManager
    from src.utils.data.unified_data_utils import UnifiedDataUtils
    from src.utils.data.quality.data_quality import DataQualityManager
    DATA_UTILITIES_AVAILABLE = True
except ImportError:
    DATA_UTILITIES_AVAILABLE = False
    tprint_warning("Data utilities not available - some features will be disabled")

# Import logger
try:
    from src.utils.logger import system_logger
except ImportError:
    import logging
    system_logger = logging.getLogger(__name__)

# Import artifact manager
try:
    from .artifact_manager import ArtifactManager
except ImportError:
    tprint_error("ArtifactManager not available - artifact management will be disabled")
    ArtifactManager = None


@dataclass
class ComponentConfig:
    """Enhanced configuration for pipeline components with full utility integration."""
    # Basic configuration
    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    timeframe: str = "5m"
    data_dir: str = "historical_data"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    force_rerun: bool = False
    validation_enabled: bool = True
    monitoring_enabled: bool = True
    fast_mode: bool = False
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    # Error handling configuration
    enable_comprehensive_error_handling: bool = True
    enable_error_recovery: bool = True
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    fail_fast: bool = False
    
    # Logging configuration
    enable_tprint_logging: bool = True
    log_level: LogLevel = LogLevel.INFO
    enable_performance_logging: bool = True
    enable_debug_logging: bool = False
    
    # M1 optimization configuration
    enable_m1_optimization: bool = True
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    memory_limit_gb: Optional[float] = None
    
    # Data processing configuration
    enable_data_validation: bool = True
    enable_data_quality_checks: bool = True
    enable_missing_data_handling: bool = True
    data_quality_threshold: float = 0.8
    
    # ML and optimization configuration
    enable_ml_utilities: bool = True
    enable_bayesian_optimization: bool = True
    enable_matrix_operations: bool = True
    enable_nas_tas: bool = True
    
    # Performance configuration
    enable_parallel_processing: bool = True
    max_parallel_workers: Optional[int] = None
    enable_caching: bool = True
    cache_size_limit_mb: int = 100
    
    # Artifact management configuration
    enable_artifact_validation: bool = True
    enable_artifact_compression: bool = True
    artifact_compression_level: int = 6
    
    def __post_init__(self):
        """Validate and set default values for configuration."""
        try:
            # Validate basic parameters
            if not self.symbol or not isinstance(self.symbol, str):
                raise ValueError("Symbol must be a non-empty string")
            
            if not self.exchange or not isinstance(self.exchange, str):
                raise ValueError("Exchange must be a non-empty string")
            
            if not self.timeframe or not isinstance(self.timeframe, str):
                raise ValueError("Timeframe must be a non-empty string")
            
            # Validate numeric parameters
            if self.max_retry_attempts < 0:
                self.max_retry_attempts = 0
            
            if self.retry_delay_seconds < 0:
                self.retry_delay_seconds = 0.0
            
            if self.data_quality_threshold < 0 or self.data_quality_threshold > 1:
                self.data_quality_threshold = 0.8
            
            if self.cache_size_limit_mb < 0:
                self.cache_size_limit_mb = 100
            
            if self.artifact_compression_level < 0 or self.artifact_compression_level > 9:
                self.artifact_compression_level = 6
            
            if self.memory_limit_gb is not None and self.memory_limit_gb <= 0:
                self.memory_limit_gb = None
            
            # Set optimal parallel workers if not specified
            if self.max_parallel_workers is None:
                try:
                    import multiprocessing
                    self.max_parallel_workers = min(4, multiprocessing.cpu_count())
                except Exception:
                    self.max_parallel_workers = 2
            
            tprint_debug(f"ComponentConfig initialized successfully: {self.symbol}@{self.exchange}")
            
        except Exception as e:
            tprint_error(f"Failed to initialize ComponentConfig: {e}")
            raise ValueError(f"Invalid ComponentConfig: {e}")


@dataclass
class ComponentResult:
    """Enhanced result from a pipeline component execution with comprehensive metadata."""
    success: bool
    artifacts: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced result fields
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    data_quality_score: float = 0.0
    validation_results: Dict[str, bool] = field(default_factory=dict)
    optimization_applied: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and enhance result data."""
        try:
            # Validate success flag
            if not isinstance(self.success, bool):
                raise ValueError("Success must be a boolean value")
            
            # Validate execution time
            if self.execution_time < 0:
                self.execution_time = 0.0
            
            # Validate memory usage
            if self.memory_usage_mb < 0:
                self.memory_usage_mb = 0.0
            
            # Validate CPU/GPU usage percentages
            if not (0 <= self.cpu_usage_percent <= 100):
                self.cpu_usage_percent = 0.0
            
            if not (0 <= self.gpu_usage_percent <= 100):
                self.gpu_usage_percent = 0.0
            
            # Validate data quality score
            if not (0 <= self.data_quality_score <= 1):
                self.data_quality_score = 0.0
            
            # Add timestamp if not present
            if 'timestamp' not in self.metadata:
                self.metadata['timestamp'] = datetime.now().isoformat()
            
            # Add component info if not present
            if 'component_type' not in self.metadata:
                self.metadata['component_type'] = 'unknown'
            
            tprint_debug(f"ComponentResult initialized: success={self.success}, execution_time={self.execution_time:.3f}s")
            
        except Exception as e:
            tprint_error(f"Failed to initialize ComponentResult: {e}")
            # Set minimal valid result
            self.success = False
            self.error_message = f"Result initialization failed: {e}"
            self.execution_time = 0.0
    
    def add_warning(self, warning_message: str) -> None:
        """Add a warning message to the result."""
        if warning_message and warning_message not in self.warnings:
            self.warnings.append(warning_message)
            tprint_warning(f"Component warning: {warning_message}")
    
    def add_performance_metric(self, metric_name: str, value: Any) -> None:
        """Add a performance metric to the result."""
        self.performance_metrics[metric_name] = value
        tprint_debug(f"Performance metric {metric_name}: {value}")
    
    def add_validation_result(self, validation_name: str, passed: bool) -> None:
        """Add a validation result to the result."""
        self.validation_results[validation_name] = passed
        if not passed:
            self.add_warning(f"Validation failed: {validation_name}")
    
    def add_optimization_applied(self, optimization_name: str) -> None:
        """Record that an optimization was applied."""
        if optimization_name not in self.optimization_applied:
            self.optimization_applied.append(optimization_name)
            tprint_debug(f"Optimization applied: {optimization_name}")
    
    def is_high_quality(self, threshold: float = 0.8) -> bool:
        """Check if the result meets quality standards."""
        return self.data_quality_score >= threshold
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the component result."""
        return {
            'success': self.success,
            'execution_time': self.execution_time,
            'memory_usage_mb': self.memory_usage_mb,
            'data_quality_score': self.data_quality_score,
            'artifact_count': len(self.artifacts),
            'warning_count': len(self.warnings),
            'optimization_count': len(self.optimization_applied),
            'validation_passed': all(self.validation_results.values()) if self.validation_results else True
        }


class BaseMarketAnalysisComponent(ABC):
    """
    Enhanced base class for market analysis pipeline components.
    
    Provides comprehensive functionality including:
    - Full error handling with no silent failures
    - Complete tprint logging integration
    - M1 hardware optimization (GPU, CPU, Memory)
    - Integration with all utility modules
    - Mathematical validation and safety checks
    - Matrix operations and ML utilities
    - Data quality validation and processing
    - Serialization utilities
    - Hardware-accelerated operations
    - Comprehensive artifact management
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the enhanced component with comprehensive configuration."""
        try:
            # Initialize configuration with validation
            self.config = config or ComponentConfig()
            self.component_name = self.__class__.__name__
            
            # Initialize logging
            self.logger = system_logger.getChild(self.component_name)
            self.start_time: Optional[datetime] = None
            self.end_time: Optional[datetime] = None
            
            # Initialize utility managers
            self._initialize_utility_managers()
            
            # Initialize M1 optimizations
            self._initialize_m1_optimizations()
            
            # Initialize artifact manager
            self._initialize_artifact_manager()
            
            # Initialize serialization utilities
            self._initialize_serialization()
            
            # Initialize ML and optimization utilities
            self._initialize_ml_utilities()
            
            # Initialize performance tracking
            self._initialize_performance_tracking()
            
            # Log successful initialization
            tprint_success(f"Enhanced {self.component_name} initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize {self.component_name}: {e}"
            tprint_error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _initialize_utility_managers(self) -> None:
        """Initialize utility managers and common operations."""
        try:
            # Initialize common utilities
            self.common_utils = CommonUtilities()
            self.base_common_utils = BaseCommonUtilities()
            
            # Initialize math validation
            self.math_validator = MathValidation()
            
            # Initialize serializers
            self.json_serializer = JSONSerializer()
            self.pickle_serializer = PickleSerializer()
            self.parquet_serializer = ParquetSerializer()
            self.universal_serializer = UniversalSerializer()
            
            tprint_debug(f"Utility managers initialized for {self.component_name}")
            
        except Exception as e:
            tprint_error(f"Failed to initialize utility managers: {e}")
            raise
    
    def _initialize_m1_optimizations(self) -> None:
        """Initialize M1 hardware optimizations."""
        try:
            if self.config.enable_m1_optimization:
                # Initialize M1 GPU manager
                if self.config.enable_gpu_acceleration:
                    self.m1_gpu_manager = get_m1_gpu_manager()
                    tprint_info(f"M1 GPU manager initialized: MPS available = {is_mps_available()}")
                
                # Initialize M1 memory optimizer
                if self.config.enable_memory_optimization:
                    self.m1_memory_optimizer = get_m1_memory_optimizer(
                        memory_limit_gb=self.config.memory_limit_gb
                    )
                    tprint_info(f"M1 memory optimizer initialized with limit: {self.config.memory_limit_gb} GB")
                
                # Initialize M1 CPU optimizer
                if self.config.enable_cpu_optimization:
                    self.m1_cpu_optimizer = get_m1_cpu_optimizer()
                    cpu_info = self.m1_cpu_optimizer.get_cpu_info()
                    tprint_info(f"M1 CPU optimizer initialized: {cpu_info}")
                
                # Integrate with M1 optimizers
                integration_result = integrate_with_m1_optimizers()
                if integration_result.get('success', False):
                    tprint_success("M1 optimization integration successful")
                else:
                    tprint_warning(f"M1 optimization integration failed: {integration_result.get('error', 'Unknown error')}")
            
            else:
                tprint_info("M1 optimization disabled in configuration")
                
        except Exception as e:
            tprint_warning(f"M1 optimization initialization failed: {e}")
            # Continue without M1 optimization
    
    def _initialize_artifact_manager(self) -> None:
        """Initialize artifact manager."""
        try:
            if ArtifactManager is not None:
                self.artifact_manager = ArtifactManager(
                    base_dir="artifacts",
                    symbol=self.config.symbol,
                    exchange=self.config.exchange,
                    timeframe=self.config.timeframe
                )
                tprint_debug(f"Artifact manager initialized for {self.component_name}")
            else:
                tprint_warning("ArtifactManager not available - artifact management disabled")
                self.artifact_manager = None
                
        except Exception as e:
            tprint_error(f"Failed to initialize artifact manager: {e}")
            self.artifact_manager = None
    
    def _initialize_serialization(self) -> None:
        """Initialize serialization utilities."""
        try:
            self.serializers = {
                'json': self.json_serializer,
                'pickle': self.pickle_serializer,
                'parquet': self.parquet_serializer,
                'universal': self.universal_serializer
            }
            tprint_debug("Serialization utilities initialized")
            
        except Exception as e:
            tprint_warning(f"Serialization initialization failed: {e}")
    
    def _initialize_ml_utilities(self) -> None:
        """Initialize ML and optimization utilities."""
        try:
            self.ml_utilities = {}
            
            if self.config.enable_ml_utilities and ML_UTILITIES_AVAILABLE:
                if self.config.enable_bayesian_optimization:
                    try:
                        self.ml_utilities['bayesian_tpe'] = BayesianTPEOptimizer()
                        tprint_debug("Bayesian TPE optimizer initialized")
                    except Exception as e:
                        tprint_warning(f"Bayesian TPE optimizer initialization failed: {e}")
            
            if self.config.enable_matrix_operations and MATRIX_UTILITIES_AVAILABLE:
                try:
                    self.ml_utilities['matrix_ops'] = MatrixOperations()
                    tprint_debug("Matrix operations initialized")
                except Exception as e:
                    tprint_warning(f"Matrix operations initialization failed: {e}")
            
            if self.config.enable_nas_tas and NAS_TAS_AVAILABLE:
                try:
                    self.ml_utilities['nas_bayesian_tpe'] = NASBayesianTPE()
                    self.ml_utilities['unified_evaluator'] = UnifiedEvaluator()
                    self.ml_utilities['monte_carlo_engine'] = MonteCarloEngine()
                    tprint_debug("NAS/TAS utilities initialized")
                except Exception as e:
                    tprint_warning(f"NAS/TAS utilities initialization failed: {e}")
            
            if self.config.enable_ml_utilities and DATA_UTILITIES_AVAILABLE:
                try:
                    self.ml_utilities['data_quality'] = DataQualityManager()
                    self.ml_utilities['unified_data_utils'] = UnifiedDataUtils()
                    tprint_debug("Data utilities initialized")
                except Exception as e:
                    tprint_warning(f"Data utilities initialization failed: {e}")
            
            tprint_debug(f"ML utilities initialized: {list(self.ml_utilities.keys())}")
            
        except Exception as e:
            tprint_warning(f"ML utilities initialization failed: {e}")
            self.ml_utilities = {}
    
    def _initialize_performance_tracking(self) -> None:
        """Initialize performance tracking."""
        try:
            self.performance_metrics = {}
            self.memory_checkpoints = {}
            
            # Initialize memory tracking
            if hasattr(self, 'm1_memory_optimizer'):
                initial_memory = self.m1_memory_optimizer.get_current_memory_usage_mb()
                self.performance_metrics['initial_memory_mb'] = initial_memory
            
            tprint_debug("Performance tracking initialized")
            
        except Exception as e:
            tprint_warning(f"Performance tracking initialization failed: {e}")
    
    def _validate_input_data(self, data: Any) -> bool:
        """Validate input data with comprehensive checks."""
        try:
            if data is None:
                raise ValueError("Input data cannot be None")
            
            # Check if data is a DataFrame
            if isinstance(data, pd.DataFrame):
                # Validate DataFrame
                if not validate_dataframe(data):
                    raise ValueError("Invalid DataFrame provided")
                
                # Check data quality if enabled
                if self.config.enable_data_quality_checks:
                    quality_metrics = calculate_data_quality_metrics(data)
                    quality_score = 1.0 - (quality_metrics.get('missing_percentage', 0) / 100)
                    
                    if quality_score < self.config.data_quality_threshold:
                        tprint_warning(f"Data quality below threshold: {quality_score:.3f} < {self.config.data_quality_threshold}")
                        return False
                
                # Check for required columns if specified
                if hasattr(self, 'required_columns') and self.required_columns:
                    if not validate_dataframe_columns(data, self.required_columns):
                        raise ValueError(f"DataFrame missing required columns: {self.required_columns}")
            
            # Check if data is a numpy array
            elif isinstance(data, np.ndarray):
                if data.size == 0:
                    raise ValueError("Numpy array cannot be empty")
                
                # Check for non-finite values
                if not np.all(np.isfinite(data)):
                    raise ValueError("Numpy array contains non-finite values")
            
            tprint_debug("Input data validation passed")
            return True
            
        except Exception as e:
            tprint_error(f"Input data validation failed: {e}")
            return False
    
    def _optimize_data_for_processing(self, data: Any) -> Any:
        """Optimize data for processing using available optimizations."""
        try:
            optimized_data = data
            
            # M1 DataFrame optimization
            if isinstance(data, pd.DataFrame) and hasattr(self, 'm1_memory_optimizer'):
                with memory_checkpoint(f"{self.component_name}_dataframe_optimization"):
                    optimized_data = self.m1_memory_optimizer.optimize_dataframe_memory(data)
                    tprint_debug("DataFrame memory optimization applied")
            
            # M1 GPU optimization
            if hasattr(self, 'm1_gpu_manager') and is_mps_available():
                with gpu_context(f"{self.component_name}_gpu_optimization"):
                    optimized_data = optimize_dataframe_for_m1(optimized_data)
                    tprint_debug("GPU optimization applied")
            
            # Data type optimization
            if isinstance(optimized_data, pd.DataFrame):
                optimized_data = optimize_dataframe_dtypes(optimized_data)
                tprint_debug("Data type optimization applied")
            
            return optimized_data
            
        except Exception as e:
            tprint_warning(f"Data optimization failed: {e}")
            return data
    
    def _safe_execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic and comprehensive error handling."""
        last_exception = None
        
        for attempt in range(self.config.max_retry_attempts + 1):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.max_retry_attempts:
                    tprint_warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {self.config.retry_delay_seconds}s...")
                    asyncio.sleep(self.config.retry_delay_seconds)
                else:
                    tprint_error(f"All {self.config.max_retry_attempts + 1} attempts failed")
                    break
        
        # If we get here, all retries failed
        error_msg = f"Function execution failed after {self.config.max_retry_attempts + 1} attempts: {last_exception}"
        tprint_error(error_msg)
        raise RuntimeError(error_msg) from last_exception
    
    def _cleanup_resources(self) -> None:
        """Clean up resources and perform memory optimization."""
        try:
            # Clean up M1 optimizers
            if hasattr(self, 'm1_memory_optimizer'):
                self.m1_memory_optimizer.force_garbage_collection()
            
            # General memory cleanup
            cleanup_result = optimize_memory()
            if cleanup_result.get('success', False):
                tprint_debug("Memory cleanup completed")
            
            # Clean up M1 optimizers
            cleanup_m1_optimizers()
            
        except Exception as e:
            tprint_warning(f"Resource cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        try:
            self._cleanup_resources()
        except Exception:
            pass  # Ignore errors during cleanup
    
    @abstractmethod
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute the component logic with comprehensive error handling and optimization.
        
        This method should be implemented by subclasses to define the specific
        component logic. The base class provides comprehensive error handling,
        logging, and optimization support.
        
        Args:
            data: Input data for the component (DataFrame, numpy array, or other)
            pipeline_state: Current pipeline state dictionary
            
        Returns:
            ComponentResult with execution results, performance metrics, and metadata
            
        Raises:
            RuntimeError: If execution fails after all retry attempts
            ValueError: If input data validation fails
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(f"{self.component_name} must implement execute method")
    
    @abstractmethod
    def get_required_artifacts(self) -> List[str]:
        """
        Get list of required artifacts this component must produce.
        
        Returns:
            List of artifact names that must be present for success
        """
        return []
    
    def get_optional_artifacts(self) -> List[str]:
        """
        Get list of optional artifacts this component may produce.
        
        Returns:
            List of optional artifact names
        """
        return []
    
    def get_component_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about this component.
        
        Returns:
            Dictionary containing component information
        """
        return {
            'component_name': self.component_name,
            'component_type': self.__class__.__name__,
            'config': {
                'symbol': self.config.symbol,
                'exchange': self.config.exchange,
                'timeframe': self.config.timeframe,
                'm1_optimization_enabled': self.config.enable_m1_optimization,
                'gpu_acceleration_enabled': self.config.enable_gpu_acceleration,
                'memory_optimization_enabled': self.config.enable_memory_optimization,
                'cpu_optimization_enabled': self.config.enable_cpu_optimization,
                'ml_utilities_enabled': self.config.enable_ml_utilities,
                'bayesian_optimization_enabled': self.config.enable_bayesian_optimization,
                'matrix_operations_enabled': self.config.enable_matrix_operations,
                'nas_tas_enabled': self.config.enable_nas_tas,
                'parallel_processing_enabled': self.config.enable_parallel_processing,
                'data_validation_enabled': self.config.enable_data_validation,
                'data_quality_checks_enabled': self.config.enable_data_quality_checks
            },
            'required_artifacts': self.get_required_artifacts(),
            'optional_artifacts': self.get_optional_artifacts(),
            'utility_managers_available': {
                'common_utils': hasattr(self, 'common_utils'),
                'math_validator': hasattr(self, 'math_validator'),
                'serializers': hasattr(self, 'serializers'),
                'ml_utilities': hasattr(self, 'ml_utilities') and bool(self.ml_utilities),
                'm1_gpu_manager': hasattr(self, 'm1_gpu_manager'),
                'm1_memory_optimizer': hasattr(self, 'm1_memory_optimizer'),
                'm1_cpu_optimizer': hasattr(self, 'm1_cpu_optimizer'),
                'artifact_manager': hasattr(self, 'artifact_manager') and self.artifact_manager is not None
            },
            'hardware_info': self._get_hardware_info(),
            'initialization_timestamp': self.start_time.isoformat() if self.start_time else None
        }
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information and capabilities."""
        try:
            hardware_info = {
                'm1_available': is_m1_available(),
                'mps_available': is_mps_available(),
                'm1_optimization_enabled': self.config.enable_m1_optimization
            }
            
            if hasattr(self, 'm1_gpu_manager'):
                hardware_info['gpu_info'] = self.m1_gpu_manager.get_gpu_info()
            
            if hasattr(self, 'm1_cpu_optimizer'):
                hardware_info['cpu_info'] = self.m1_cpu_optimizer.get_cpu_info()
            
            if hasattr(self, 'm1_memory_optimizer'):
                hardware_info['memory_info'] = self.m1_memory_optimizer.get_memory_stats()
            
            return hardware_info
            
        except Exception as e:
            tprint_warning(f"Failed to get hardware info: {e}")
            return {'error': str(e)}
    
    def validate_artifacts(self, artifacts: Dict[str, Any]) -> bool:
        """
        Validate that all required artifacts are present and non-empty with comprehensive checks.
        
        Args:
            artifacts: Dictionary of artifacts to validate
            
        Returns:
            True if all required artifacts are present and valid
            
        Raises:
            ValueError: If validation fails with detailed error information
        """
        try:
            if not isinstance(artifacts, dict):
                raise ValueError(f"Artifacts must be a dictionary, got {type(artifacts)}")
            
            required_artifacts = self.get_required_artifacts()
            optional_artifacts = self.get_optional_artifacts()
            
            # Check required artifacts
            missing_artifacts = []
            invalid_artifacts = []
            
            for artifact_name in required_artifacts:
                if artifact_name not in artifacts:
                    missing_artifacts.append(artifact_name)
                    tprint_error(f"Missing required artifact: {artifact_name}")
                    continue
                
                artifact_value = artifacts[artifact_name]
                
                # Comprehensive validation
                if artifact_value is None:
                    invalid_artifacts.append(f"{artifact_name}: None value")
                    tprint_error(f"Required artifact {artifact_name} is None")
                elif isinstance(artifact_value, (list, dict)) and len(artifact_value) == 0:
                    invalid_artifacts.append(f"{artifact_name}: Empty collection")
                    tprint_error(f"Required artifact {artifact_name} is empty")
                elif isinstance(artifact_value, str) and artifact_value.strip() == "":
                    invalid_artifacts.append(f"{artifact_name}: Empty string")
                    tprint_error(f"Required artifact {artifact_name} is empty string")
                elif isinstance(artifact_value, pd.DataFrame) and artifact_value.empty:
                    invalid_artifacts.append(f"{artifact_name}: Empty DataFrame")
                    tprint_error(f"Required artifact {artifact_name} is empty DataFrame")
                elif isinstance(artifact_value, np.ndarray) and artifact_value.size == 0:
                    invalid_artifacts.append(f"{artifact_name}: Empty array")
                    tprint_error(f"Required artifact {artifact_name} is empty array")
                else:
                    # Additional validation for DataFrames
                    if isinstance(artifact_value, pd.DataFrame):
                        # Check data quality
                        if self.config.enable_data_quality_checks:
                            quality_metrics = calculate_data_quality_metrics(artifact_value)
                            quality_score = 1.0 - (quality_metrics.get('missing_percentage', 0) / 100)
                            if quality_score < self.config.data_quality_threshold:
                                invalid_artifacts.append(f"{artifact_name}: Low data quality ({quality_score:.3f})")
                                tprint_warning(f"Artifact {artifact_name} has low data quality: {quality_score:.3f}")
                    
                    # Additional validation for numpy arrays
                    elif isinstance(artifact_value, np.ndarray):
                        if not np.all(np.isfinite(artifact_value)):
                            invalid_artifacts.append(f"{artifact_name}: Non-finite values")
                            tprint_error(f"Artifact {artifact_name} contains non-finite values")
            
            # Check for unexpected artifacts
            all_expected_artifacts = set(required_artifacts + optional_artifacts)
            unexpected_artifacts = set(artifacts.keys()) - all_expected_artifacts
            if unexpected_artifacts:
                tprint_warning(f"Unexpected artifacts found: {list(unexpected_artifacts)}")
            
            # Report validation results
            if missing_artifacts or invalid_artifacts:
                error_details = []
                if missing_artifacts:
                    error_details.append(f"Missing artifacts: {missing_artifacts}")
                if invalid_artifacts:
                    error_details.append(f"Invalid artifacts: {invalid_artifacts}")
                
                error_msg = "; ".join(error_details)
                tprint_error(f"Artifact validation failed: {error_msg}")
                return False
            
            # Log successful validation
            tprint_success(f"Artifact validation passed: {len(required_artifacts)} required artifacts validated")
            
            # Log optional artifacts found
            found_optional = [name for name in optional_artifacts if name in artifacts]
            if found_optional:
                tprint_info(f"Optional artifacts found: {found_optional}")
            
            return True
            
        except Exception as e:
            tprint_error(f"Artifact validation failed with exception: {e}")
            raise ValueError(f"Artifact validation failed: {e}") from e
    
    def _start_execution(self) -> None:
        """Mark the start of execution with comprehensive logging and monitoring."""
        try:
            self.start_time = datetime.now()
            
            # Log execution start
            tprint_info(f"🚀 Starting {self.component_name} execution")
            tprint_debug(f"Configuration: {self.config.symbol}@{self.config.exchange} ({self.config.timeframe})")
            
            # Record initial memory usage
            if hasattr(self, 'm1_memory_optimizer'):
                initial_memory = self.m1_memory_optimizer.get_current_memory_usage_mb()
                self.performance_metrics['initial_memory_mb'] = initial_memory
                tprint_debug(f"Initial memory usage: {initial_memory:.1f} MB")
            
            # Log hardware capabilities
            if hasattr(self, 'm1_gpu_manager'):
                gpu_info = self.m1_gpu_manager.get_gpu_info()
                if gpu_info.get('mps_available'):
                    tprint_info(f"🎮 GPU acceleration available: {gpu_info.get('gpu_name', 'M1 GPU')}")
            
            # Log ML utilities availability
            if self.ml_utilities:
                available_utilities = list(self.ml_utilities.keys())
                tprint_debug(f"ML utilities available: {available_utilities}")
            
            # Log component info
            component_info = self.get_component_info()
            tprint_structured(component_info, LogLevel.DEBUG)
            
        except Exception as e:
            tprint_error(f"Failed to start execution monitoring: {e}")
            # Continue execution even if monitoring fails
    
    def _end_execution(self) -> float:
        """Mark the end of execution and return duration with comprehensive metrics."""
        try:
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds() if self.start_time else 0.0
            
            # Record final memory usage
            if hasattr(self, 'm1_memory_optimizer'):
                final_memory = self.m1_memory_optimizer.get_current_memory_usage_mb()
                self.performance_metrics['final_memory_mb'] = final_memory
                memory_delta = final_memory - self.performance_metrics.get('initial_memory_mb', 0)
                self.performance_metrics['memory_delta_mb'] = memory_delta
                
                if memory_delta > 0:
                    tprint_info(f"Memory usage increased by {memory_delta:.1f} MB")
                elif memory_delta < 0:
                    tprint_info(f"Memory usage decreased by {abs(memory_delta):.1f} MB")
            
            # Log execution completion
            tprint_success(f"✅ Completed {self.component_name} execution in {duration:.3f}s")
            
            # Log performance metrics
            if self.performance_metrics:
                tprint_structured(self.performance_metrics, LogLevel.DEBUG)
            
            # Performance logging
            if self.config.enable_performance_logging:
                tprint_performance(f"{self.component_name} execution", duration)
            
            return duration
            
        except Exception as e:
            tprint_error(f"Failed to complete execution monitoring: {e}")
            return (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0.0
    
    async def save_artifacts(self, artifacts: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Save artifacts using the centralized artifact manager with comprehensive validation and optimization.
        
        Args:
            artifacts: Dictionary of artifacts to save
            metadata: Optional metadata to include
            
        Returns:
            Dictionary mapping artifact names to file paths
            
        Raises:
            RuntimeError: If artifact saving fails
            ValueError: If artifacts validation fails
        """
        try:
            if not artifacts:
                tprint_warning("No artifacts to save")
                return {}
            
            if self.artifact_manager is None:
                raise RuntimeError("Artifact manager not available")
            
            # Validate artifacts before saving
            if self.config.enable_artifact_validation:
                if not self.validate_artifacts(artifacts):
                    raise ValueError("Artifact validation failed")
            
            # Prepare metadata
            enhanced_metadata = metadata or {}
            enhanced_metadata.update({
                'component_name': self.component_name,
                'save_timestamp': datetime.now().isoformat(),
                'config': {
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe
                },
                'performance_metrics': self.performance_metrics,
                'hardware_info': self._get_hardware_info()
            })
            
            # Optimize artifacts for saving if enabled
            optimized_artifacts = artifacts
            if self.config.enable_artifact_compression:
                optimized_artifacts = await self._optimize_artifacts_for_saving(artifacts)
            
            # Save artifacts using artifact manager
            component_name = self.component_name.replace('Component', '').lower()
            saved_files = await self.artifact_manager.save_artifacts(
                component_name, 
                optimized_artifacts, 
                enhanced_metadata
            )
            
            # Log successful save
            tprint_success(f"✅ Saved {len(saved_files)} artifacts: {list(saved_files.keys())}")
            
            # Log file sizes if available
            for artifact_name, file_path in saved_files.items():
                if safe_file_exists(file_path):
                    file_size = get_file_size(file_path)
                    tprint_debug(f"Artifact '{artifact_name}' saved to {file_path} ({format_bytes(file_size)})")
            
            return saved_files
            
        except Exception as e:
            error_msg = f"Failed to save artifacts: {e}"
            tprint_error(error_msg)
            raise RuntimeError(error_msg) from e
    
    async def _optimize_artifacts_for_saving(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize artifacts for efficient saving."""
        try:
            optimized_artifacts = {}
            
            for artifact_name, artifact_value in artifacts.items():
                # Optimize DataFrames
                if isinstance(artifact_value, pd.DataFrame):
                    # Apply memory optimization
                    if hasattr(self, 'm1_memory_optimizer'):
                        optimized_value = self.m1_memory_optimizer.optimize_dataframe_memory(artifact_value)
                    else:
                        optimized_value = optimize_dataframe_dtypes(artifact_value)
                    
                    optimized_artifacts[artifact_name] = optimized_value
                    tprint_debug(f"Optimized DataFrame artifact: {artifact_name}")
                
                # Optimize numpy arrays
                elif isinstance(artifact_value, np.ndarray):
                    # Convert to appropriate dtype for size optimization
                    if artifact_value.dtype == np.float64:
                        optimized_value = artifact_value.astype(np.float32)
                        tprint_debug(f"Optimized array dtype for artifact: {artifact_name} (float64 -> float32)")
                    else:
                        optimized_value = artifact_value
                    
                    optimized_artifacts[artifact_name] = optimized_value
                
                else:
                    # Keep other artifacts as-is
                    optimized_artifacts[artifact_name] = artifact_value
            
            return optimized_artifacts
            
        except Exception as e:
            tprint_warning(f"Artifact optimization failed: {e}")
            return artifacts  # Return original artifacts if optimization fails
    
    async def _execute_with_timing(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute the component with comprehensive timing, error handling, and optimization.
        
        Args:
            data: Input data for the component
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with execution results, performance metrics, and comprehensive metadata
        """
        # Start execution monitoring
        self._start_execution()
        
        result = None
        execution_successful = False
        
        try:
            # Validate input data
            if self.config.enable_data_validation:
                if not self._validate_input_data(data):
                    raise ValueError("Input data validation failed")
            
            # Optimize data for processing
            optimized_data = self._optimize_data_for_processing(data)
            
            # Execute component logic with retry if enabled
            if self.config.enable_error_recovery and self.config.max_retry_attempts > 0:
                result = await self._safe_execute_with_retry(self.execute, optimized_data, pipeline_state)
            else:
                result = await self.execute(optimized_data, pipeline_state)
            
            # Validate result structure
            if not isinstance(result, ComponentResult):
                raise ValueError(f"Component execute method must return ComponentResult, got {type(result)}")
            
            # Add performance metrics to result
            result.performance_metrics.update(self.performance_metrics)
            result.metadata.update({
                'component_name': self.component_name,
                'execution_timestamp': self.start_time.isoformat() if self.start_time else None,
                'hardware_info': self._get_hardware_info(),
                'optimization_applied': list(set(result.optimization_applied))
            })
            
            # Validate artifacts if execution was successful
            if result.success and self.config.enable_artifact_validation:
                try:
                    if not self.validate_artifacts(result.artifacts):
                        result.success = False
                        result.error_message = "Invalid artifacts produced - validation failed"
                        result.add_warning("Artifact validation failed")
                        tprint_error("Component execution succeeded but produced invalid artifacts")
                except Exception as e:
                    result.success = False
                    result.error_message = f"Artifact validation error: {e}"
                    result.add_warning(f"Artifact validation error: {e}")
                    tprint_error(f"Artifact validation failed with exception: {e}")
            
            # Save artifacts if execution was successful
            if result.success and result.artifacts and self.artifact_manager is not None:
                try:
                    with tprint_timer("Artifact saving"):
                        saved_files = await self.save_artifacts(result.artifacts, result.metadata)
                        result.metadata['saved_files'] = saved_files
                        result.add_performance_metric('artifacts_saved', len(saved_files))
                        tprint_success(f"✅ Artifacts saved successfully: {list(saved_files.keys())}")
                except Exception as e:
                    result.success = False
                    result.error_message = f"Artifact saving failed: {e}"
                    result.add_warning(f"Artifact saving failed: {e}")
                    tprint_error(f"❌ Failed to save artifacts: {e}")
                    
                    # Clean up any partial artifacts
                    if self.artifact_manager:
                        component_name = self.component_name.replace('Component', '').lower()
                        try:
                            self.artifact_manager.cleanup_failed_artifacts(component_name)
                        except Exception:
                            pass  # Ignore cleanup errors
            
            # Update execution time and final metrics
            result.execution_time = self._end_execution()
            
            # Add final memory usage
            if hasattr(self, 'm1_memory_optimizer'):
                result.memory_usage_mb = self.m1_memory_optimizer.get_current_memory_usage_mb()
            
            execution_successful = result.success
            
            # Log final result summary
            summary = result.get_summary()
            tprint_structured(summary, LogLevel.INFO)
            
            return result
            
        except Exception as e:
            # Comprehensive error handling
            error_msg = f"Component execution failed: {e}"
            tprint_error(error_msg)
            
            # Log detailed error information
            if self.config.enable_debug_logging:
                tprint_error(f"Full traceback: {traceback.format_exc()}")
            
            # Create error result
            error_result = ComponentResult(
                success=False,
                artifacts={},
                error_message=error_msg,
                execution_time=self._end_execution(),
                metadata={
                    'component_name': self.component_name,
                    'error_type': type(e).__name__,
                    'error_details': str(e),
                    'execution_timestamp': self.start_time.isoformat() if self.start_time else None,
                    'hardware_info': self._get_hardware_info()
                }
            )
            
            # Add error to warnings
            error_result.add_warning(f"Execution failed: {e}")
            
            # Clean up any partial artifacts
            if self.artifact_manager:
                component_name = self.component_name.replace('Component', '').lower()
                try:
                    self.artifact_manager.cleanup_failed_artifacts(component_name)
                except Exception:
                    pass  # Ignore cleanup errors
            
            return error_result
            
        finally:
            # Always perform cleanup
            try:
                self._cleanup_resources()
            except Exception as e:
                tprint_warning(f"Cleanup failed: {e}")
            
            # Log execution summary
            if execution_successful:
                tprint_success(f"🎉 {self.component_name} execution completed successfully")
            else:
                tprint_error(f"💥 {self.component_name} execution failed")
    
    # Utility methods for common operations
    def safe_dataframe_operation(self, df: pd.DataFrame, operation: Callable, *args, **kwargs) -> pd.DataFrame:
        """Safely perform operation on DataFrame with error handling."""
        return safe_dataframe_operation(df, operation, *args, **kwargs)
    
    def safe_math_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """Safely perform mathematical operation with validation."""
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            tprint_error(f"Mathematical operation failed: {e}")
            raise
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame using available optimizations."""
        try:
            optimized_df = df
            
            # M1 memory optimization
            if hasattr(self, 'm1_memory_optimizer'):
                optimized_df = self.m1_memory_optimizer.optimize_dataframe_memory(optimized_df)
            
            # Data type optimization
            optimized_df = optimize_dataframe_dtypes(optimized_df)
            
            # GPU optimization if available
            if hasattr(self, 'm1_gpu_manager') and is_mps_available():
                optimized_df = optimize_dataframe_for_m1(optimized_df)
            
            return optimized_df
            
        except Exception as e:
            tprint_warning(f"DataFrame optimization failed: {e}")
            return df
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'component_name': self.component_name,
            'performance_metrics': self.performance_metrics,
            'hardware_info': self._get_hardware_info(),
            'utility_availability': {
                'm1_gpu_manager': hasattr(self, 'm1_gpu_manager'),
                'm1_memory_optimizer': hasattr(self, 'm1_memory_optimizer'),
                'm1_cpu_optimizer': hasattr(self, 'm1_cpu_optimizer'),
                'ml_utilities': bool(self.ml_utilities),
                'artifact_manager': self.artifact_manager is not None
            },
            'config_summary': {
                'm1_optimization_enabled': self.config.enable_m1_optimization,
                'gpu_acceleration_enabled': self.config.enable_gpu_acceleration,
                'memory_optimization_enabled': self.config.enable_memory_optimization,
                'cpu_optimization_enabled': self.config.enable_cpu_optimization,
                'parallel_processing_enabled': self.config.enable_parallel_processing,
                'data_validation_enabled': self.config.enable_data_validation,
                'data_quality_checks_enabled': self.config.enable_data_quality_checks
            }
        }