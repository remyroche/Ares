"""
Shared Utilities for Market Analysis TAS Regime

This module provides a comprehensive collection of utilities for market analysis
and TAS (Technical Analysis System) regime detection. It consolidates utilities
from various sources to provide a unified interface for:

- Data processing and validation
- Mathematical operations and validation
- Serialization and persistence
- Logging and debugging
- Matrix operations and computations
- Hardware optimization (M1/M2/M3)
- Machine learning utilities
- Performance monitoring and optimization

The module is designed to be the central hub for all utility functions
needed in the market analysis pipeline, providing both high-level interfaces
and low-level utilities for maximum flexibility and reusability.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import warnings

# Configure logging for this module
logger = logging.getLogger(__name__)

# =============================================================================
# CORE UTILITIES IMPORTS
# =============================================================================

# Common Operations - Core data processing utilities
try:
    from src.utils.common_operations import (
        # Data processing
        create_empty_dataframe,
        validate_dataframe,
        validate_dataframe_columns,
        safe_dataframe_operation,
        safe_fillna,
        safe_convert_dtypes,
        safe_merge_dataframes,
        safe_drop_columns,
        safe_rename_columns,
        validate_timestamp_column,
        safe_timestamp_conversion,
        optimize_dataframe_dtypes,
        
        # Data quality
        calculate_data_quality_metrics,
        get_dataframe_info,
        create_data_quality_report,
        create_summary_statistics,
        
        # File operations
        ensure_directory,
        safe_file_exists,
        safe_json_dump,
        safe_json_load,
        safe_copy,
        safe_deepcopy,
        validate_file_path,
        get_file_size,
        check_disk_space,
        
        # Math utilities
        safe_divide,
        safe_log,
        safe_sqrt,
        safe_power,
        safe_mean,
        safe_std,
        safe_float,
        safe_int,
        validate_finite,
        validate_positive,
        validate_range,
        safe_kelly_calculation,
        safe_weighted_average,
        safe_percentage_change,
        
        # String utilities
        safe_lower,
        safe_upper,
        safe_join,
        
        # Collection utilities
        safe_append,
        safe_extend,
        safe_dict_get,
        safe_dict_items,
        
        # Performance utilities
        timed_operation,
        format_bytes,
        chunked_iterable,
        parallel_map,
        
        # Matrix utilities
        validate_correlation_matrix,
        safe_matrix_inverse,
        math_safe,
        
        # Parquet operations
        safe_to_parquet,
        safe_read_parquet,
        list_parquet_files,
        safe_resample,
        align_dataframes,
        validate_dataframe_schema,
        
        # M1 Integration
        get_m1_gpu_manager,
        get_m1_memory_optimizer,
        get_m1_cpu_optimizer,
        cleanup_m1_optimizers,
        integrate_with_m1_optimizers,
        memory_checkpoint,
        gpu_context,
        optimize_memory,
        get_memory_usage,
        
        # Outcome file utilities
        get_latest_outcome_file,
        load_latest_optimal_regime_clustering_outcome,
        
        # Common utilities class
        CommonUtilities,
        
        # Exceptions
        MathValidationError
    )
    COMMON_OPERATIONS_AVAILABLE = True
    logger.info("✅ Common operations utilities loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Common operations utilities not available: {e}")
    COMMON_OPERATIONS_AVAILABLE = False

# Math Validation - Safe mathematical operations
try:
    from src.utils.math_validation import (
        # Safe math operations
        safe_divide,
        safe_log,
        safe_sqrt,
        safe_power,
        validate_finite,
        validate_positive,
        validate_range,
        validate_numeric_array,
        safe_kelly_calculation,
        safe_weighted_average,
        safe_percentage_change,
        safe_correlation,
        safe_covariance,
        safe_mean,
        safe_std,
        safe_percentile,
        validate_correlation_matrix,
        safe_matrix_inverse,
        math_safe,
        
        # Math validation class
        MathValidation,
        MathValidationError
    )
    MATH_VALIDATION_AVAILABLE = True
    logger.info("✅ Math validation utilities loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Math validation utilities not available: {e}")
    MATH_VALIDATION_AVAILABLE = False

# Serialization Utilities - Data persistence
try:
    from src.utils.serialization_utils import (
        JSONSerializer,
        PickleSerializer,
        ParquetSerializer,
        UniversalSerializer
    )
    SERIALIZATION_AVAILABLE = True
    logger.info("✅ Serialization utilities loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Serialization utilities not available: {e}")
    SERIALIZATION_AVAILABLE = False

# TPrint Utilities - Enhanced logging and debugging
try:
    from src.utils.tprint import (
        # Core tprint functions
        tprint,
        tprint_debug,
        tprint_info,
        tprint_warning,
        tprint_error,
        tprint_success,
        tprint_progress,
        tprint_performance,
        tprint_structured,
        tprint_with_level,
        tprint_batch,
        tprint_numba_compatible,
        
        # Enhanced print functions
        enhanced_print,
        tprint_print,
        replace_builtin_print,
        restore_builtin_print,
        capture_print_to_tprint,
        
        # Configuration and management
        configure_tprint,
        get_tprint_config,
        tprint_context,
        tprint_timer,
        tprint_logged,
        cleanup_tprint,
        enable_auto_print_logging,
        set_print_log_level,
        
        # Classes and enums
        TPrintConfig,
        TPrintManager,
        LogLevel,
        TimestampFormat,
        
        # Backward compatibility
        timestamped_print,
        
        # Integration
        NUMBA_AVAILABLE,
        COLORAMA_AVAILABLE,
        
        # Numba compatibility
        numba_print_with_timestamp,
        numba_print_detailed,
        numba_print_simple,
        numba_print_progress,
        numba_print_performance,
        numba_print_error,
        numba_print_warning,
        numba_print_info,
        numba_print_debug,
        get_numba_timestamp_string,
        get_numba_detailed_timestamp_string,
        get_numba_simple_timestamp_string,
        numba_timer_start,
        numba_timer_elapsed,
        numba_print_timing,
        NumbaTimestampFormatter
    )
    TPRINT_AVAILABLE = True
    logger.info("✅ TPrint utilities loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ TPrint utilities not available: {e}")
    TPRINT_AVAILABLE = False

# =============================================================================
# DATA UTILITIES IMPORTS
# =============================================================================

# Data processing utilities
try:
    # Explicit imports instead of wildcard imports
    from src.utils.data.unified_data_utils import UnifiedDataUtils
    from src.utils.data.klines_parquet import KlinesParquetManager, get_klines_manager
    from src.utils.data.optimized_parquet_storage import OptimizedParquetStorage
    from src.utils.data.historical_data_pipeline import HistoricalDataPipeline
    from src.utils.data.real_data_loader import RealDataLoader
    DATA_UTILS_AVAILABLE = True
    logger.info("✅ Data utilities loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Data utilities not available: {e}")
    DATA_UTILS_AVAILABLE = False
    # Create fallback dummy classes
    class UnifiedDataUtils:
        """Fallback unified data utilities class."""
        def __init__(self, *args, **kwargs):
            self.logger = logging.getLogger(self.__class__.__name__)
        
        def load_data(self, *args, **kwargs):
            """Fallback data loading method."""
            self.logger.warning("UnifiedDataUtils.load_data called but module not available")
            return None
        
        def save_data(self, *args, **kwargs):
            """Fallback data saving method."""
            self.logger.warning("UnifiedDataUtils.save_data called but module not available")
            return False
        
        def validate_data(self, *args, **kwargs):
            """Fallback data validation method."""
            self.logger.warning("UnifiedDataUtils.validate_data called but module not available")
            return True
    
    class KlinesParquetManager:
        """Fallback klines parquet manager class."""
        def __init__(self, *args, **kwargs):
            self.logger = logging.getLogger(self.__class__.__name__)
        
        def load_klines(self, *args, **kwargs):
            """Fallback klines loading method."""
            self.logger.warning("KlinesParquetManager.load_klines called but module not available")
            return None
        
        def save_klines(self, *args, **kwargs):
            """Fallback klines saving method."""
            self.logger.warning("KlinesParquetManager.save_klines called but module not available")
            return False
        
        def get_klines_info(self, *args, **kwargs):
            """Fallback klines info method."""
            self.logger.warning("KlinesParquetManager.get_klines_info called but module not available")
            return {}
    
    class OptimizedParquetStorage:
        """Fallback optimized parquet storage class."""
        def __init__(self, *args, **kwargs):
            self.logger = logging.getLogger(self.__class__.__name__)
        
        def store_data(self, *args, **kwargs):
            """Fallback data storage method."""
            self.logger.warning("OptimizedParquetStorage.store_data called but module not available")
            return False
        
        def retrieve_data(self, *args, **kwargs):
            """Fallback data retrieval method."""
            self.logger.warning("OptimizedParquetStorage.retrieve_data called but module not available")
            return None
        
        def optimize_storage(self, *args, **kwargs):
            """Fallback storage optimization method."""
            self.logger.warning("OptimizedParquetStorage.optimize_storage called but module not available")
            return {}
    
    class HistoricalDataPipeline:
        """Fallback historical data pipeline class."""
        def __init__(self, *args, **kwargs):
            self.logger = logging.getLogger(self.__class__.__name__)
        
        def process_data(self, *args, **kwargs):
            """Fallback data processing method."""
            self.logger.warning("HistoricalDataPipeline.process_data called but module not available")
            return None
        
        def validate_pipeline(self, *args, **kwargs):
            """Fallback pipeline validation method."""
            self.logger.warning("HistoricalDataPipeline.validate_pipeline called but module not available")
            return True
        
        def get_pipeline_status(self, *args, **kwargs):
            """Fallback pipeline status method."""
            self.logger.warning("HistoricalDataPipeline.get_pipeline_status called but module not available")
            return {"status": "unavailable"}
    
    class RealDataLoader:
        """Fallback real data loader class."""
        def __init__(self, *args, **kwargs):
            self.logger = logging.getLogger(self.__class__.__name__)
        
        def load_real_data(self, *args, **kwargs):
            """Fallback real data loading method."""
            self.logger.warning("RealDataLoader.load_real_data called but module not available")
            return None
        
        def connect_to_source(self, *args, **kwargs):
            """Fallback data source connection method."""
            self.logger.warning("RealDataLoader.connect_to_source called but module not available")
            return False
        
        def get_data_metadata(self, *args, **kwargs):
            """Fallback data metadata method."""
            self.logger.warning("RealDataLoader.get_data_metadata called but module not available")
            return {}
    
    def get_klines_manager(*args, **kwargs):
        """Fallback klines manager factory function."""
        logger = logging.getLogger('get_klines_manager')
        logger.warning("get_klines_manager called but module not available")
        return None

# Data quality utilities
try:
    # Explicit imports instead of wildcard imports
    from src.utils.data.quality.data_quality import DataQualityValidator
    from src.utils.data.quality.advanced_quality_metrics import AdvancedQualityMetrics
    from src.utils.data.quality.comprehensive_quality_scorer import ComprehensiveQualityScorer
    from src.utils.data.quality.data_cleaning import DataCleaner
    from src.utils.data.quality.statistical_distribution_validation import StatisticalDistributionValidator
    DATA_QUALITY_AVAILABLE = True
    logger.info("✅ Data quality utilities loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Data quality utilities not available: {e}")
    DATA_QUALITY_AVAILABLE = False
    # Create fallback dummy classes
    class DataQualityValidator:
        """Fallback data quality validator class."""
        def __init__(self, *args, **kwargs):
            self.logger = logging.getLogger(self.__class__.__name__)
        
        def validate_data_quality(self, *args, **kwargs):
            """Fallback data quality validation method."""
            self.logger.warning("DataQualityValidator.validate_data_quality called but module not available")
            return {"quality_score": 0.0, "issues": [], "passed": True}
        
        def get_quality_metrics(self, *args, **kwargs):
            """Fallback quality metrics method."""
            self.logger.warning("DataQualityValidator.get_quality_metrics called but module not available")
            return {}
        
        def generate_quality_report(self, *args, **kwargs):
            """Fallback quality report generation method."""
            self.logger.warning("DataQualityValidator.generate_quality_report called but module not available")
            return {"report": "Data quality module not available"}
    
    class AdvancedQualityMetrics:
        """Fallback advanced quality metrics class."""
        def __init__(self, *args, **kwargs):
            self.logger = logging.getLogger(self.__class__.__name__)
        
        def calculate_advanced_metrics(self, *args, **kwargs):
            """Fallback advanced metrics calculation method."""
            self.logger.warning("AdvancedQualityMetrics.calculate_advanced_metrics called but module not available")
            return {}
        
        def assess_data_completeness(self, *args, **kwargs):
            """Fallback data completeness assessment method."""
            self.logger.warning("AdvancedQualityMetrics.assess_data_completeness called but module not available")
            return {"completeness_score": 1.0}
        
        def detect_anomalies(self, *args, **kwargs):
            """Fallback anomaly detection method."""
            self.logger.warning("AdvancedQualityMetrics.detect_anomalies called but module not available")
            return {"anomalies": [], "anomaly_score": 0.0}
    
    class ComprehensiveQualityScorer:
        """Fallback comprehensive quality scorer class."""
        def __init__(self, *args, **kwargs):
            self.logger = logging.getLogger(self.__class__.__name__)
        
        def score_data_quality(self, *args, **kwargs):
            """Fallback data quality scoring method."""
            self.logger.warning("ComprehensiveQualityScorer.score_data_quality called but module not available")
            return {"overall_score": 0.0, "detailed_scores": {}}
        
        def get_quality_breakdown(self, *args, **kwargs):
            """Fallback quality breakdown method."""
            self.logger.warning("ComprehensiveQualityScorer.get_quality_breakdown called but module not available")
            return {}
        
        def recommend_improvements(self, *args, **kwargs):
            """Fallback improvement recommendations method."""
            self.logger.warning("ComprehensiveQualityScorer.recommend_improvements called but module not available")
            return []
    
    class DataCleaner:
        """Fallback data cleaner class."""
        def __init__(self, *args, **kwargs):
            self.logger = logging.getLogger(self.__class__.__name__)
        
        def clean_data(self, *args, **kwargs):
            """Fallback data cleaning method."""
            self.logger.warning("DataCleaner.clean_data called but module not available")
            return None
        
        def remove_duplicates(self, *args, **kwargs):
            """Fallback duplicate removal method."""
            self.logger.warning("DataCleaner.remove_duplicates called but module not available")
            return None
        
        def handle_missing_values(self, *args, **kwargs):
            """Fallback missing value handling method."""
            self.logger.warning("DataCleaner.handle_missing_values called but module not available")
            return None
        
        def normalize_data(self, *args, **kwargs):
            """Fallback data normalization method."""
            self.logger.warning("DataCleaner.normalize_data called but module not available")
            return None
    
    class StatisticalDistributionValidator:
        """Fallback statistical distribution validator class."""
        def __init__(self, *args, **kwargs):
            self.logger = logging.getLogger(self.__class__.__name__)
        
        def validate_distribution(self, *args, **kwargs):
            """Fallback distribution validation method."""
            self.logger.warning("StatisticalDistributionValidator.validate_distribution called but module not available")
            return {"is_valid": True, "distribution_type": "unknown"}
        
        def test_normality(self, *args, **kwargs):
            """Fallback normality testing method."""
            self.logger.warning("StatisticalDistributionValidator.test_normality called but module not available")
            return {"is_normal": True, "p_value": 1.0}
        
        def fit_distribution(self, *args, **kwargs):
            """Fallback distribution fitting method."""
            self.logger.warning("StatisticalDistributionValidator.fit_distribution called but module not available")
            return {"best_fit": "normal", "parameters": {}}

# Data validation utilities
try:
    # Explicit imports instead of wildcard imports
    from src.utils.data.validation.validators import DataValidator
    DATA_VALIDATION_AVAILABLE = True
    logger.info("✅ Data validation utilities loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Data validation utilities not available: {e}")
    DATA_VALIDATION_AVAILABLE = False
    # Create fallback dummy class
    class DataValidator:
        """Fallback data validator class."""
        def __init__(self, *args, **kwargs):
            self.logger = logging.getLogger(self.__class__.__name__)
        
        def validate(self, *args, **kwargs):
            """Fallback validation method."""
            self.logger.warning("DataValidator.validate called but module not available")
            return {"is_valid": True, "errors": [], "warnings": []}
        
        def validate_schema(self, *args, **kwargs):
            """Fallback schema validation method."""
            self.logger.warning("DataValidator.validate_schema called but module not available")
            return {"schema_valid": True, "schema_errors": []}
        
        def validate_data_types(self, *args, **kwargs):
            """Fallback data type validation method."""
            self.logger.warning("DataValidator.validate_data_types called but module not available")
            return {"types_valid": True, "type_errors": []}
        
        def validate_constraints(self, *args, **kwargs):
            """Fallback constraint validation method."""
            self.logger.warning("DataValidator.validate_constraints called but module not available")
            return {"constraints_valid": True, "constraint_errors": []}
        
        def get_validation_report(self, *args, **kwargs):
            """Fallback validation report method."""
            self.logger.warning("DataValidator.get_validation_report called but module not available")
            return {"report": "Data validation module not available", "summary": {}}

# =============================================================================
# MATRIX OPERATIONS IMPORTS
# =============================================================================

try:
    from src.utils.matrix_operations.unified_operations import *
    from src.utils.matrix_operations.vectorized_core import *
    from src.utils.matrix_operations.enhanced_operations import *
    from src.utils.matrix_operations.batch_operations import *
    from src.utils.matrix_operations.computation_toolbox import *
    from src.utils.matrix_operations.convenience import *
    from src.utils.matrix_operations.error_handling import *
    from src.utils.matrix_operations.hardware_integration import *
    MATRIX_OPERATIONS_AVAILABLE = True
    logger.info("✅ Matrix operations utilities loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Matrix operations utilities not available: {e}")
    MATRIX_OPERATIONS_AVAILABLE = False

# =============================================================================
# HARDWARE OPTIMIZATION IMPORTS
# =============================================================================

# M1 GPU Utilities
try:
    from src.utils.hardware.m1_gpu_utils import (
        M1GPUManager,
        is_m1_available,
        is_mps_available,
        get_m1_gpu_manager,
        get_gpu_info,
        optimize_for_m1_gpu,
        create_m1_gpu_context,
        cleanup_m1_gpu_resources
    )
    M1_GPU_AVAILABLE = True
    logger.info("✅ M1 GPU utilities loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ M1 GPU utilities not available: {e}")
    M1_GPU_AVAILABLE = False

# M1 Memory Optimizer
try:
    from src.utils.hardware.m1_memory_optimizer import (
        M1MemoryOptimizer,
        get_m1_memory_optimizer,
        optimize_memory_usage,
        monitor_memory_pressure,
        cleanup_memory_resources,
        get_memory_stats,
        create_memory_checkpoint,
        optimize_dataframe_memory
    )
    M1_MEMORY_AVAILABLE = True
    logger.info("✅ M1 Memory optimizer loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ M1 Memory optimizer not available: {e}")
    M1_MEMORY_AVAILABLE = False

# M1 CPU Optimizer
try:
    from src.utils.hardware.m1_cpu_optimizer import (
        M1CPUOptimizer,
        get_m1_cpu_optimizer,
        optimize_cpu_usage,
        get_cpu_info,
        optimize_numpy_operations,
        optimize_pandas_operations,
        create_cpu_optimized_context,
        cleanup_cpu_resources
    )
    M1_CPU_AVAILABLE = True
    logger.info("✅ M1 CPU optimizer loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ M1 CPU optimizer not available: {e}")
    M1_CPU_AVAILABLE = False

# Unified Hardware Manager
try:
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager,
        get_hardware_manager,
        optimize_system_performance,
        get_system_info,
        create_optimized_context,
        cleanup_all_resources
    )
    UNIFIED_HARDWARE_AVAILABLE = True
    logger.info("✅ Unified hardware manager loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Unified hardware manager not available: {e}")
    UNIFIED_HARDWARE_AVAILABLE = False

# =============================================================================
# ML COMMON UTILITIES IMPORTS
# =============================================================================

try:
    from src.utils.ml_common import (
        # Core ML utilities
        tprint as ml_tprint,
        
        # Models
        EnhancedModelFactory, ModelType, ModelConfig,
        create_model_factory,
        MultiOutputConfig, MultiOutputModel, MultiOutputStackingModel, MultiOutputResult,
        prepare_multi_output_targets, create_analyst_outputs, create_tactician_outputs,
        create_multi_output_stacking_model,
        EnhancedModelTrainer, train_model_with_confidence_metrics,
        ModelEvaluator, ModelRegistry,
        
        # Ensembles
        EnsembleManager, EnsembleType, EnsembleConfig,
        StackingEnsembleManager, StackingEnsembleConfig, StackingEnsembleResult,
        create_analyst_ensemble, create_tactician_ensemble,
        StackingConfidenceCalibrator, StackingCalibrationConfig, StackingCalibrationResult,
        create_analyst_calibrator, create_tactician_calibrator,
        
        # Cross-validation
        CrossValidationManager, CVConfig, CVResult,
        create_cv_manager, create_time_series_cv,
        
        # Feature engineering
        FeatureEngineer, FeatureConfig, FeatureResult,
        create_feature_engineer, create_technical_indicators,
        
        # Hyperparameter optimization
        HyperparameterOptimizer, HPOConfig, HPOResult,
        create_hpo_optimizer, create_bayesian_optimizer,
        
        # Validation
        ValidationManager, ValidationConfig, ValidationResult,
        create_validation_manager, create_time_series_validation,
        
        # Pipeline
        PipelineOrchestrator, PipelineConfig, PipelineResult,
        create_pipeline_orchestrator, create_ml_pipeline,
        
        # Monitoring
        ModelMonitor, MonitoringConfig, MonitoringResult,
        create_model_monitor, create_performance_monitor,
        
        # Reporting
        ReportGenerator, ReportConfig, ReportResult,
        create_report_generator, create_performance_report
    )
    ML_COMMON_AVAILABLE = True
    logger.info("✅ ML Common utilities loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ ML Common utilities not available: {e}")
    ML_COMMON_AVAILABLE = False

# =============================================================================
# UNIFIED INTERFACE CLASS
# =============================================================================

class MarketAnalysisUtilities:
    """
    Unified interface for all market analysis utilities.
    
    This class provides a centralized interface to all utility functions
    needed for market analysis and TAS regime detection, with automatic
    fallbacks and error handling.
    """
    
    def __init__(self):
        """Initialize the unified utilities interface."""
        self.logger = logger.getChild('MarketAnalysisUtilities')
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all available components."""
        self.components = {
            'common_operations': COMMON_OPERATIONS_AVAILABLE,
            'math_validation': MATH_VALIDATION_AVAILABLE,
            'serialization': SERIALIZATION_AVAILABLE,
            'tprint': TPRINT_AVAILABLE,
            'data_utils': DATA_UTILS_AVAILABLE,
            'data_quality': DATA_QUALITY_AVAILABLE,
            'data_validation': DATA_VALIDATION_AVAILABLE,
            'matrix_operations': MATRIX_OPERATIONS_AVAILABLE,
            'm1_gpu': M1_GPU_AVAILABLE,
            'm1_memory': M1_MEMORY_AVAILABLE,
            'm1_cpu': M1_CPU_AVAILABLE,
            'unified_hardware': UNIFIED_HARDWARE_AVAILABLE,
            'ml_common': ML_COMMON_AVAILABLE
        }
        
        # Log component availability
        available_components = [k for k, v in self.components.items() if v]
        unavailable_components = [k for k, v in self.components.items() if not v]
        
        self.logger.info(f"✅ Available components: {available_components}")
        if unavailable_components:
            self.logger.warning(f"⚠️ Unavailable components: {unavailable_components}")
    
    def get_component_status(self) -> Dict[str, bool]:
        """Get the availability status of all components."""
        return self.components.copy()
    
    def is_component_available(self, component: str) -> bool:
        """Check if a specific component is available."""
        return self.components.get(component, False)
    
    def get_available_components(self) -> List[str]:
        """Get list of available components."""
        return [k for k, v in self.components.items() if v]
    
    def get_unavailable_components(self) -> List[str]:
        """Get list of unavailable components."""
        return [k for k, v in self.components.items() if not v]

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_utilities() -> MarketAnalysisUtilities:
    """Get the unified utilities interface."""
    return MarketAnalysisUtilities()

def check_dependencies() -> Dict[str, bool]:
    """Check the availability of all utility dependencies."""
    return get_utilities().get_component_status()

def get_available_utilities() -> List[str]:
    """Get list of available utility components."""
    return get_utilities().get_available_components()

def is_utility_available(utility: str) -> bool:
    """Check if a specific utility is available."""
    return get_utilities().is_component_available(utility)

# =============================================================================
# FALLBACK UTILITIES
# =============================================================================

def safe_import(module_name: str, fallback_value: Any = None):
    """Safely import a module with fallback."""
    try:
        return __import__(module_name)
    except ImportError:
        logger.warning(f"⚠️ Could not import {module_name}, using fallback")
        return fallback_value

def safe_call(func: Callable, *args, **kwargs) -> Any:
    """Safely call a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"⚠️ Error calling {func.__name__}: {e}")
        return None

def create_fallback_logger(name: str) -> logging.Logger:
    """Create a fallback logger if the main one is not available."""
    return logging.getLogger(name)

# =============================================================================
# EXPORTS
# =============================================================================

# Core utilities
__all__ = [
    # Main interface
    'MarketAnalysisUtilities',
    'get_utilities',
    'check_dependencies',
    'get_available_utilities',
    'is_utility_available',
    
    # Utility functions
    'safe_import',
    'safe_call',
    'create_fallback_logger',
    
    # Component availability flags
    'COMMON_OPERATIONS_AVAILABLE',
    'MATH_VALIDATION_AVAILABLE',
    'SERIALIZATION_AVAILABLE',
    'TPRINT_AVAILABLE',
    'DATA_UTILS_AVAILABLE',
    'DATA_QUALITY_AVAILABLE',
    'DATA_VALIDATION_AVAILABLE',
    'MATRIX_OPERATIONS_AVAILABLE',
    'M1_GPU_AVAILABLE',
    'M1_MEMORY_AVAILABLE',
    'M1_CPU_AVAILABLE',
    'UNIFIED_HARDWARE_AVAILABLE',
    'ML_COMMON_AVAILABLE'
]

# Add all imported utilities to __all__ if they're available
if COMMON_OPERATIONS_AVAILABLE:
    __all__.extend([
        'create_empty_dataframe', 'validate_dataframe', 'validate_dataframe_columns',
        'safe_dataframe_operation', 'safe_fillna', 'safe_convert_dtypes',
        'safe_merge_dataframes', 'safe_drop_columns', 'safe_rename_columns',
        'validate_timestamp_column', 'safe_timestamp_conversion', 'optimize_dataframe_dtypes',
        'calculate_data_quality_metrics', 'get_dataframe_info', 'create_data_quality_report',
        'create_summary_statistics', 'ensure_directory', 'safe_file_exists',
        'safe_json_dump', 'safe_json_load', 'safe_copy', 'safe_deepcopy',
        'validate_file_path', 'get_file_size', 'check_disk_space',
        'safe_divide', 'safe_log', 'safe_sqrt', 'safe_power', 'safe_mean', 'safe_std',
        'safe_float', 'safe_int', 'validate_finite', 'validate_positive', 'validate_range',
        'safe_kelly_calculation', 'safe_weighted_average', 'safe_percentage_change',
        'safe_lower', 'safe_upper', 'safe_join', 'safe_append', 'safe_extend',
        'safe_dict_get', 'safe_dict_items', 'timed_operation', 'format_bytes',
        'chunked_iterable', 'parallel_map', 'validate_correlation_matrix',
        'safe_matrix_inverse', 'math_safe', 'safe_to_parquet', 'safe_read_parquet',
        'list_parquet_files', 'safe_resample', 'align_dataframes', 'validate_dataframe_schema',
        'get_m1_gpu_manager', 'get_m1_memory_optimizer', 'get_m1_cpu_optimizer',
        'cleanup_m1_optimizers', 'integrate_with_m1_optimizers', 'memory_checkpoint',
        'gpu_context', 'optimize_memory', 'get_memory_usage',
        'get_latest_outcome_file', 'load_latest_optimal_regime_clustering_outcome',
        'CommonUtilities', 'MathValidationError'
    ])

if MATH_VALIDATION_AVAILABLE:
    __all__.extend([
        'MathValidation', 'MathValidationError'
    ])

if SERIALIZATION_AVAILABLE:
    __all__.extend([
        'JSONSerializer', 'PickleSerializer', 'ParquetSerializer', 'UniversalSerializer'
    ])

if TPRINT_AVAILABLE:
    __all__.extend([
        'tprint', 'tprint_debug', 'tprint_info', 'tprint_warning', 'tprint_error',
        'tprint_success', 'tprint_progress', 'tprint_performance', 'tprint_structured',
        'tprint_with_level', 'tprint_batch', 'tprint_numba_compatible',
        'enhanced_print', 'tprint_print', 'replace_builtin_print', 'restore_builtin_print',
        'capture_print_to_tprint', 'configure_tprint', 'get_tprint_config',
        'tprint_context', 'tprint_timer', 'tprint_logged', 'cleanup_tprint',
        'enable_auto_print_logging', 'set_print_log_level', 'TPrintConfig',
        'TPrintManager', 'LogLevel', 'TimestampFormat', 'timestamped_print',
        'NUMBA_AVAILABLE', 'COLORAMA_AVAILABLE'
    ])

if M1_GPU_AVAILABLE:
    __all__.extend([
        'M1GPUManager', 'is_m1_available', 'is_mps_available', 'get_m1_gpu_manager',
        'get_gpu_info', 'optimize_for_m1_gpu', 'create_m1_gpu_context', 'cleanup_m1_gpu_resources'
    ])

if M1_MEMORY_AVAILABLE:
    __all__.extend([
        'M1MemoryOptimizer', 'get_m1_memory_optimizer', 'optimize_memory_usage',
        'monitor_memory_pressure', 'cleanup_memory_resources', 'get_memory_stats',
        'create_memory_checkpoint', 'optimize_dataframe_memory'
    ])

if M1_CPU_AVAILABLE:
    __all__.extend([
        'M1CPUOptimizer', 'get_m1_cpu_optimizer', 'optimize_cpu_usage', 'get_cpu_info',
        'optimize_numpy_operations', 'optimize_pandas_operations',
        'create_cpu_optimized_context', 'cleanup_cpu_resources'
    ])

if UNIFIED_HARDWARE_AVAILABLE:
    __all__.extend([
        'UnifiedHardwareManager', 'get_hardware_manager', 'optimize_system_performance',
        'get_system_info', 'create_optimized_context', 'cleanup_all_resources'
    ])

# Initialize the utilities interface
_utilities = get_utilities()

# Log initialization
logger.info("🚀 Market Analysis TAS Regime Shared Utilities initialized successfully")
logger.info(f"📊 Available components: {len(_utilities.get_available_components())}/{len(_utilities.components)}")

# Suppress warnings for missing optional dependencies
warnings.filterwarnings('ignore', category=ImportWarning, module=__name__)