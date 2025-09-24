"""
Enhanced Utility Integration for Hybrid NAS-TAS Regime System

This module integrates all the available utility tools from src/utils/ into the
hybrid NAS-TAS regime system for enhanced functionality and performance.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pathlib import Path
import time
from dataclasses import dataclass, field
from enum import Enum

# Import common operations utilities
from src.utils.common_operations import (
    # Data operations
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    safe_merge_dataframes, safe_drop_columns, safe_rename_columns,
    safe_timestamp_conversion, validate_timestamp_column,
    optimize_dataframe_dtypes, safe_fillna,
    
    # Data quality
    calculate_data_quality_metrics, get_dataframe_info, create_data_quality_report,
    create_summary_statistics, validate_dataframe_schema,
    
    # Math operations
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    safe_float, safe_int, validate_finite, validate_positive, validate_range,
    safe_kelly_calculation, safe_weighted_average, safe_percentage_change,
    
    # File operations
    ensure_directory, safe_file_exists, safe_json_dump, safe_json_load,
    safe_to_parquet, safe_read_parquet, list_parquet_files,
    safe_copy, safe_deepcopy, validate_file_path, get_file_size,
    
    # Performance
    timed_operation, format_bytes, chunked_iterable, parallel_map,
    
    # Matrix operations
    validate_correlation_matrix, safe_matrix_inverse, math_safe,
    
    # M1 optimizations
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    integrate_with_m1_optimizers, cleanup_m1_optimizers,
    memory_checkpoint, gpu_context, optimize_memory, get_memory_usage,
    
    # Advanced operations
    safe_resample, align_dataframes, guard_dataframe_nulls,
    validate_file_size, secure_file_path, with_tracing_span, sanitize_string
)

# Import common utilities
from src.utils.common_utilities import (
    CommonUtilities, safe_dataframe_operation as safe_df_op,
    validate_dataframe_columns as validate_df_cols, safe_convert_dtypes as safe_conv_dtypes,
    calculate_data_quality_metrics as calc_quality_metrics,
    safe_merge_dataframes as safe_merge_dfs, safe_groupby_operation,
    safe_apply_function, create_summary_statistics as create_summary_stats,
    safe_drop_columns as safe_drop_cols, safe_rename_columns as safe_rename_cols,
    validate_timestamp_column as validate_ts_col, safe_timestamp_conversion as safe_ts_conv,
    get_dataframe_info as get_df_info, safe_filter_dataframe,
    create_data_quality_report as create_quality_report
)

# Import math validation
from src.utils.math_validation import (
    MathValidation, safe_divide as math_safe_divide, safe_log as math_safe_log,
    safe_sqrt as math_safe_sqrt, safe_power as math_safe_power,
    validate_finite as math_validate_finite, validate_positive as math_validate_positive,
    validate_range as math_validate_range, validate_numeric_array,
    safe_kelly_calculation as math_safe_kelly, safe_weighted_average as math_safe_weighted_avg,
    safe_percentage_change as math_safe_pct_change, safe_correlation, safe_covariance,
    safe_mean as math_safe_mean, safe_std as math_safe_std, safe_percentile,
    validate_correlation_matrix as math_validate_corr_matrix,
    safe_matrix_inverse as math_safe_matrix_inverse, math_safe as math_safe_func,
    MathValidationError
)

# Import serialization utilities
from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

# Import data utilities (conditional imports)
try:
    from src.utils.data.klines_parquet import KlinesParquetManager, get_klines_manager, save_klines_to_parquet, load_klines_from_parquet, validate_klines_data
    KLINES_PARQUET_AVAILABLE = True
except ImportError:
    KLINES_PARQUET_AVAILABLE = False
    KlinesParquetManager = None
    get_klines_manager = None
    save_klines_to_parquet = None
    load_klines_from_parquet = None
    validate_klines_data = None

try:
    from src.utils.data.unified_data_utils import UnifiedDataUtils
    UNIFIED_DATA_UTILS_AVAILABLE = True
except ImportError:
    UNIFIED_DATA_UTILS_AVAILABLE = False
    UnifiedDataUtils = None

# Import additional data processing utilities
try:
    from src.utils.data.feature_engineer import FeatureEngineer
    FEATURE_ENGINEER_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEER_AVAILABLE = False
    FeatureEngineer = None

try:
    from src.utils.data.gap_detector import GapDetector
    GAP_DETECTOR_AVAILABLE = True
except ImportError:
    GAP_DETECTOR_AVAILABLE = False
    GapDetector = None

try:
    from src.utils.data.historical_data_downloader import HistoricalDataDownloader
    HISTORICAL_DOWNLOADER_AVAILABLE = True
except ImportError:
    HISTORICAL_DOWNLOADER_AVAILABLE = False
    HistoricalDataDownloader = None

# Import data quality utilities
try:
    from src.utils.data.quality.comprehensive_quality_scorer import ComprehensiveQualityScorer
    QUALITY_SCORER_AVAILABLE = True
except ImportError:
    QUALITY_SCORER_AVAILABLE = False
    ComprehensiveQualityScorer = None

try:
    from src.utils.data.quality.advanced_quality_metrics import AdvancedQualityMetrics
    ADVANCED_QUALITY_AVAILABLE = True
except ImportError:
    ADVANCED_QUALITY_AVAILABLE = False
    AdvancedQualityMetrics = None

# Import matrix operations (conditional imports)
try:
    from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False
    UnifiedMatrixOperations = None

try:
    from src.utils.matrix_operations.vectorized_core import VectorizedCore
    VECTORIZED_CORE_AVAILABLE = True
except ImportError:
    VECTORIZED_CORE_AVAILABLE = False
    VectorizedCore = None

# Import hardware optimizations (conditional imports)
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager, get_m1_gpu_manager as get_gpu_manager
    M1_GPU_AVAILABLE = True
except ImportError:
    M1_GPU_AVAILABLE = False
    M1GPUManager = None
    get_gpu_manager = None

try:
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer, get_m1_memory_optimizer as get_mem_optimizer
    M1_MEMORY_AVAILABLE = True
except ImportError:
    M1_MEMORY_AVAILABLE = False
    M1MemoryOptimizer = None
    get_mem_optimizer = None

try:
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer, get_m1_cpu_optimizer as get_cpu_optimizer
    M1_CPU_AVAILABLE = True
except ImportError:
    M1_CPU_AVAILABLE = False
    M1CPUOptimizer = None
    get_cpu_optimizer = None

# Import ML common utilities (conditional imports)
try:
    from src.utils.ml_common.common_operations import MLCommonOperations
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False
    MLCommonOperations = None

try:
    from src.utils.ml_common.confidence_metrics import ConfidenceMetrics, calculate_confidence_metrics, calculate_calibration_metrics
    CONFIDENCE_METRICS_AVAILABLE = True
except ImportError:
    CONFIDENCE_METRICS_AVAILABLE = False
    ConfidenceMetrics = None
    calculate_confidence_metrics = None
    calculate_calibration_metrics = None

try:
    from src.utils.ml_common.feature_selection import FeatureSelector
    FEATURE_SELECTION_AVAILABLE = True
except ImportError:
    FEATURE_SELECTION_AVAILABLE = False
    FeatureSelector = None

try:
    from src.utils.ml_common.matrix_cross_validation import MatrixCrossValidation
    MATRIX_CV_AVAILABLE = True
except ImportError:
    MATRIX_CV_AVAILABLE = False
    MatrixCrossValidation = None

# Import additional ML utilities
try:
    from src.utils.ml_common.lookahead_bias_detector import LookaheadBiasDetector
    LOOKAHEAD_DETECTOR_AVAILABLE = True
except ImportError:
    LOOKAHEAD_DETECTOR_AVAILABLE = False
    LookaheadBiasDetector = None

try:
    from src.utils.ml_common.data_drift_detector import DataDriftDetector
    DRIFT_DETECTOR_AVAILABLE = True
except ImportError:
    DRIFT_DETECTOR_AVAILABLE = False
    DataDriftDetector = None

try:
    from src.utils.ml_common.utils import UnifiedCache, get_unified_cache, cached, LookaheadProtection, MLTrainingSafeguards, RobustErrorHandler
    CACHE_UTILS_AVAILABLE = True
except ImportError:
    CACHE_UTILS_AVAILABLE = False
    UnifiedCache = None
    get_unified_cache = None
    cached = None
    LookaheadProtection = None
    MLTrainingSafeguards = None
    RobustErrorHandler = None

try:
    from src.utils.ml_common.ensembles import EnsembleManager, StackingEnsembleManager
    ENSEMBLE_MANAGER_AVAILABLE = True
except ImportError:
    ENSEMBLE_MANAGER_AVAILABLE = False
    EnsembleManager = None
    StackingEnsembleManager = None

try:
    from src.utils.ml_common.hmm_regime_detection import HMMRegimeDetector
    HMM_REGIME_AVAILABLE = True
except ImportError:
    HMM_REGIME_AVAILABLE = False
    HMMRegimeDetector = None

# Setup logging
logger = logging.getLogger(__name__)


class UtilityIntegrationStatus(Enum):
    """Status of utility integration."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass
class UtilityIntegrationConfig:
    """Configuration for utility integration."""
    # Data operations
    enable_data_validation: bool = True
    enable_data_quality_checks: bool = True
    enable_safe_operations: bool = True
    
    # Math operations
    enable_math_validation: bool = True
    enable_safe_math: bool = True
    
    # Serialization
    enable_serialization: bool = True
    default_serialization_format: str = "parquet"
    
    # Hardware optimization
    enable_m1_optimizations: bool = True
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    
    # ML utilities
    enable_ml_common: bool = True
    enable_feature_selection: bool = True
    enable_cross_validation: bool = True
    enable_confidence_metrics: bool = True
    enable_lookahead_detection: bool = True
    enable_drift_detection: bool = True
    enable_ensemble_management: bool = True
    enable_hmm_regime_detection: bool = True
    enable_caching: bool = True

    # Data utilities
    enable_feature_engineering: bool = True
    enable_gap_detection: bool = True
    enable_historical_download: bool = True
    enable_comprehensive_quality: bool = True
    enable_advanced_quality_metrics: bool = True

    # Matrix operations
    enable_matrix_operations: bool = True
    enable_vectorized_operations: bool = True

    # Performance monitoring
    enable_performance_monitoring: bool = True
    enable_memory_monitoring: bool = True


class EnhancedUtilityIntegration:
    """
    Enhanced utility integration manager for hybrid NAS-TAS regime system.
    
    This class integrates all available utility tools from src/utils/ to provide
    enhanced functionality, performance optimization, and robust error handling.
    """
    
    def __init__(self, config: Optional[UtilityIntegrationConfig] = None):
        """Initialize the enhanced utility integration."""
        self.config = config or UtilityIntegrationConfig()
        self.logger = logger.getChild('EnhancedUtilityIntegration')
        
        # Initialize integration status
        self.integration_status = self._check_integration_status()
        
        # Initialize utility managers
        self._initialize_utility_managers()
        
        self.logger.info("🔧 Enhanced Utility Integration initialized")
        self.logger.info(f"📊 Integration Status: {self.integration_status}")
    
    def _check_integration_status(self) -> Dict[str, UtilityIntegrationStatus]:
        """Check the status of all utility integrations."""
        status = {}
        
        # Check data utilities
        status['common_operations'] = UtilityIntegrationStatus.AVAILABLE
        status['common_utilities'] = UtilityIntegrationStatus.AVAILABLE
        status['math_validation'] = UtilityIntegrationStatus.AVAILABLE
        status['serialization'] = UtilityIntegrationStatus.AVAILABLE
        
        # Check data utilities
        status['klines_parquet'] = UtilityIntegrationStatus.AVAILABLE if KLINES_PARQUET_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE
        status['unified_data_utils'] = UtilityIntegrationStatus.AVAILABLE if UNIFIED_DATA_UTILS_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE
        status['feature_engineer'] = UtilityIntegrationStatus.AVAILABLE if FEATURE_ENGINEER_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE
        status['gap_detector'] = UtilityIntegrationStatus.AVAILABLE if GAP_DETECTOR_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE
        status['historical_downloader'] = UtilityIntegrationStatus.AVAILABLE if HISTORICAL_DOWNLOADER_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE
        status['quality_scorer'] = UtilityIntegrationStatus.AVAILABLE if QUALITY_SCORER_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE
        status['advanced_quality'] = UtilityIntegrationStatus.AVAILABLE if ADVANCED_QUALITY_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE

        # Check matrix operations
        status['matrix_operations'] = UtilityIntegrationStatus.AVAILABLE if MATRIX_OPERATIONS_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE
        status['vectorized_core'] = UtilityIntegrationStatus.AVAILABLE if VECTORIZED_CORE_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE

        # Check hardware optimizations
        status['m1_gpu'] = UtilityIntegrationStatus.AVAILABLE if M1_GPU_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE
        status['m1_memory'] = UtilityIntegrationStatus.AVAILABLE if M1_MEMORY_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE
        status['m1_cpu'] = UtilityIntegrationStatus.AVAILABLE if M1_CPU_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE

        # Check ML utilities
        status['ml_common'] = UtilityIntegrationStatus.AVAILABLE if ML_COMMON_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE
        status['confidence_metrics'] = UtilityIntegrationStatus.AVAILABLE if CONFIDENCE_METRICS_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE
        status['feature_selection'] = UtilityIntegrationStatus.AVAILABLE if FEATURE_SELECTION_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE
        status['matrix_cv'] = UtilityIntegrationStatus.AVAILABLE if MATRIX_CV_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE
        status['lookahead_detector'] = UtilityIntegrationStatus.AVAILABLE if LOOKAHEAD_DETECTOR_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE
        status['drift_detector'] = UtilityIntegrationStatus.AVAILABLE if DRIFT_DETECTOR_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE
        status['cache_utils'] = UtilityIntegrationStatus.AVAILABLE if CACHE_UTILS_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE
        status['ensemble_manager'] = UtilityIntegrationStatus.AVAILABLE if ENSEMBLE_MANAGER_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE
        status['hmm_regime'] = UtilityIntegrationStatus.AVAILABLE if HMM_REGIME_AVAILABLE else UtilityIntegrationStatus.UNAVAILABLE
        
        return status
    
    def _initialize_utility_managers(self):
        """Initialize utility managers."""
        # Initialize common utilities
        self.common_utilities = CommonUtilities()
        self.math_validator = MathValidation()
        self.serializer = UniversalSerializer()
        
        # Initialize data utilities
        if self.config.enable_serialization and KLINES_PARQUET_AVAILABLE:
            self.klines_manager = KlinesParquetManager()
        else:
            self.klines_manager = None

        if UNIFIED_DATA_UTILS_AVAILABLE:
            self.unified_data_utils = UnifiedDataUtils()
        else:
            self.unified_data_utils = None

        # Initialize additional data utilities
        if FEATURE_ENGINEER_AVAILABLE:
            self.feature_engineer = FeatureEngineer()
        else:
            self.feature_engineer = None

        if GAP_DETECTOR_AVAILABLE:
            self.gap_detector = GapDetector()
        else:
            self.gap_detector = None

        if HISTORICAL_DOWNLOADER_AVAILABLE:
            self.historical_downloader = HistoricalDataDownloader()
        else:
            self.historical_downloader = None

        if QUALITY_SCORER_AVAILABLE:
            self.quality_scorer = ComprehensiveQualityScorer()
        else:
            self.quality_scorer = None

        if ADVANCED_QUALITY_AVAILABLE:
            self.advanced_quality = AdvancedQualityMetrics()
        else:
            self.advanced_quality = None
        
        # Initialize matrix operations
        if self.config.enable_matrix_operations and MATRIX_OPERATIONS_AVAILABLE:
            self.matrix_operations = UnifiedMatrixOperations()
        else:
            self.matrix_operations = None
            
        if self.config.enable_vectorized_operations and VECTORIZED_CORE_AVAILABLE:
            self.vectorized_core = VectorizedCore()
        else:
            self.vectorized_core = None
        
        # Initialize hardware optimizations
        if self.config.enable_m1_optimizations:
            if M1_GPU_AVAILABLE:
                self.gpu_manager = get_gpu_manager()
            else:
                self.gpu_manager = None
                
            if M1_MEMORY_AVAILABLE:
                self.memory_optimizer = get_mem_optimizer()
            else:
                self.memory_optimizer = None
                
            if M1_CPU_AVAILABLE:
                self.cpu_optimizer = get_cpu_optimizer()
            else:
                self.cpu_optimizer = None
        else:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
        
        # Initialize ML utilities
        if self.config.enable_ml_common and ML_COMMON_AVAILABLE:
            self.ml_common = MLCommonOperations()
        else:
            self.ml_common = None

        if CONFIDENCE_METRICS_AVAILABLE:
            self.confidence_metrics = ConfidenceMetrics()
        else:
            self.confidence_metrics = None

        if FEATURE_SELECTION_AVAILABLE:
            self.feature_selector = FeatureSelector()
        else:
            self.feature_selector = None

        if MATRIX_CV_AVAILABLE:
            self.matrix_cv = MatrixCrossValidation()
        else:
            self.matrix_cv = None

        # Initialize additional ML utilities
        if LOOKAHEAD_DETECTOR_AVAILABLE:
            self.lookahead_detector = LookaheadBiasDetector()
        else:
            self.lookahead_detector = None

        if DRIFT_DETECTOR_AVAILABLE:
            self.drift_detector = DataDriftDetector()
        else:
            self.drift_detector = None

        if CACHE_UTILS_AVAILABLE:
            self.unified_cache = get_unified_cache()
        else:
            self.unified_cache = None

        if ENSEMBLE_MANAGER_AVAILABLE:
            self.ensemble_manager = EnsembleManager()
        else:
            self.ensemble_manager = None

        if HMM_REGIME_AVAILABLE:
            self.hmm_regime_detector = HMMRegimeDetector()
        else:
            self.hmm_regime_detector = None
    
    # =============================================================================
    # DATA OPERATIONS
    # =============================================================================
    
    def safe_dataframe_operation(self, df: pd.DataFrame, operation: Callable, *args, **kwargs) -> pd.DataFrame:
        """Safely perform operation on DataFrame with enhanced error handling."""
        if self.config.enable_safe_operations:
            return safe_dataframe_operation(df, operation, *args, **kwargs)
        else:
            return operation(df, *args, **kwargs)
    
    def validate_dataframe_columns(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate DataFrame columns with enhanced validation."""
        if self.config.enable_data_validation:
            return validate_dataframe_columns(df, required_columns)
        else:
            return True
    
    def calculate_data_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality metrics with enhanced analysis."""
        if self.config.enable_data_quality_checks:
            return calculate_data_quality_metrics(df)
        else:
            return {}
    
    def optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency."""
        return optimize_dataframe_dtypes(df)
    
    def safe_merge_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Safely merge DataFrames with enhanced error handling."""
        return safe_merge_dataframes(df1, df2, **kwargs)
    
    def create_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive data quality report."""
        if self.config.enable_data_quality_checks:
            return create_data_quality_report(df)
        else:
            return {}
    
    # =============================================================================
    # MATH OPERATIONS
    # =============================================================================
    
    def safe_divide(self, a: float, b: float, default: float = 0.0) -> float:
        """Safely divide two numbers with validation."""
        if self.config.enable_math_validation:
            return self.math_validator.safe_divide(a, b, default)
        else:
            return safe_divide(a, b, default)
    
    def safe_log(self, x: float, default: float = 0.0) -> float:
        """Safely calculate logarithm with validation."""
        if self.config.enable_math_validation:
            return self.math_validator.safe_log(x, default)
        else:
            return safe_log(x, default)
    
    def safe_sqrt(self, x: float, default: float = 0.0) -> float:
        """Safely calculate square root with validation."""
        if self.config.enable_math_validation:
            return self.math_validator.safe_sqrt(x, default)
        else:
            return safe_sqrt(x, default)
    
    def validate_finite(self, value: Any, name: str = "value") -> float:
        """Validate that a value is finite."""
        if self.config.enable_math_validation:
            return self.math_validator.validate_finite(value, name)
        else:
            return validate_finite(value, name)
    
    def safe_correlation(self, x: np.ndarray, y: np.ndarray, default: float = 0.0) -> float:
        """Safely calculate correlation with enhanced validation."""
        if self.config.enable_safe_math:
            return safe_correlation(x, y, default)
        else:
            return np.corrcoef(x, y)[0, 1] if len(x) > 1 and len(y) > 1 else default
    
    def safe_covariance(self, x: np.ndarray, y: np.ndarray, default: float = 0.0) -> float:
        """Safely calculate covariance with enhanced validation."""
        if self.config.enable_safe_math:
            return safe_covariance(x, y, default)
        else:
            return np.cov(x, y)[0, 1] if len(x) > 1 and len(y) > 1 else default
    
    # =============================================================================
    # SERIALIZATION
    # =============================================================================
    
    def save_data(self, data: Any, filepath: str, format: str = None) -> bool:
        """Save data using appropriate serialization format."""
        if self.config.enable_serialization:
            if format is None:
                format = self.config.default_serialization_format
            return self.serializer.save(data, filepath, format)
        else:
            return False
    
    def load_data(self, filepath: str) -> Optional[Any]:
        """Load data using automatic format detection."""
        if self.config.enable_serialization:
            return self.serializer.load(filepath)
        else:
            return None
    
    def save_parquet(self, df: pd.DataFrame, filepath: str) -> bool:
        """Save DataFrame to parquet format."""
        return safe_to_parquet(df, filepath)
    
    def load_parquet(self, filepath: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from parquet format."""
        return safe_read_parquet(filepath)
    
    # =============================================================================
    # HARDWARE OPTIMIZATION
    # =============================================================================
    
    def optimize_for_m1(self) -> Dict[str, Any]:
        """Optimize system for M1 hardware."""
        if self.config.enable_m1_optimizations:
            return integrate_with_m1_optimizers()
        else:
            return {'integration_status': 'disabled'}
    
    def get_memory_usage(self) -> float:
        """Get current memory usage."""
        if self.config.enable_memory_monitoring:
            return get_memory_usage()
        else:
            return 0.0
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage."""
        if self.config.enable_memory_optimization and self.memory_optimizer:
            return self.memory_optimizer.optimize_memory()
        else:
            return optimize_memory()
    
    def memory_checkpoint(self, name: str):
        """Create memory checkpoint context manager."""
        if self.config.enable_memory_optimization:
            return memory_checkpoint(name)
        else:
            from contextlib import nullcontext
            return nullcontext()
    
    def gpu_context(self, name: str):
        """Create GPU context manager."""
        if self.config.enable_gpu_acceleration:
            return gpu_context(name)
        else:
            from contextlib import nullcontext
            return nullcontext()
    
    # =============================================================================
    # MATRIX OPERATIONS
    # =============================================================================
    
    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Perform matrix multiplication with optimization."""
        if self.config.enable_matrix_operations and self.matrix_operations:
            return self.matrix_operations.multiply(a, b)
        else:
            return np.dot(a, b)
    
    def matrix_inverse(self, matrix: np.ndarray) -> np.ndarray:
        """Calculate matrix inverse safely."""
        if self.config.enable_safe_math:
            return safe_matrix_inverse(matrix)
        else:
            return np.linalg.inv(matrix)
    
    def vectorized_operation(self, operation: str, *args, **kwargs) -> Any:
        """Perform vectorized operation with optimization."""
        if self.config.enable_vectorized_operations and self.vectorized_core:
            return getattr(self.vectorized_core, operation)(*args, **kwargs)
        else:
            # Fallback to numpy operations
            return getattr(np, operation)(*args, **kwargs)
    
    # =============================================================================
    # ML UTILITIES
    # =============================================================================
    
    def feature_selection(self, X: np.ndarray, y: np.ndarray, method: str = "mutual_info") -> np.ndarray:
        """Perform feature selection."""
        if self.config.enable_feature_selection and self.feature_selector:
            return self.feature_selector.select_features(X, y, method=method)
        else:
            # Fallback: return all features
            return np.arange(X.shape[1])
    
    def cross_validation(self, estimator, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """Perform cross-validation."""
        if self.config.enable_cross_validation and self.matrix_cv:
            return self.matrix_cv.cross_validate(estimator, X, y, cv=cv)
        else:
            # Fallback: basic cross-validation
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(estimator, X, y, cv=cv)
            return {'scores': scores, 'mean': scores.mean(), 'std': scores.std()}
    
    def calculate_confidence_metrics(self, predictions: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
        """Calculate confidence metrics."""
        if self.config.enable_confidence_metrics and self.confidence_metrics:
            return self.confidence_metrics.calculate_metrics(predictions, probabilities)
        elif self.config.enable_confidence_metrics and calculate_confidence_metrics:
            return calculate_confidence_metrics(predictions, probabilities)
        else:
            # Fallback: basic confidence calculation
            return {'mean_confidence': np.mean(probabilities), 'std_confidence': np.std(probabilities)}

    def calculate_calibration_metrics(self, predictions: np.ndarray, probabilities: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
        """Calculate calibration metrics."""
        if self.config.enable_confidence_metrics and calculate_calibration_metrics:
            return calculate_calibration_metrics(predictions, probabilities, true_labels)
        else:
            return {'calibration_error': 0.0}

    def detect_lookahead_bias(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Detect lookahead bias in features."""
        if self.config.enable_ml_common and self.lookahead_detector:
            return self.lookahead_detector.detect_bias(X, y)
        else:
            return {'bias_detected': False, 'bias_score': 0.0}

    def detect_data_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> Dict[str, Any]:
        """Detect data drift between reference and current data."""
        if self.config.enable_ml_common and self.drift_detector:
            return self.drift_detector.detect_drift(reference_data, current_data)
        else:
            return {'drift_detected': False, 'drift_score': 0.0}

    def get_cached_data(self, key: str, default: Any = None):
        """Get data from unified cache."""
        if self.config.enable_ml_common and self.unified_cache:
            return self.unified_cache.get(key, default)
        return default

    def set_cached_data(self, key: str, value: Any):
        """Set data in unified cache."""
        if self.config.enable_ml_common and self.unified_cache:
            return self.unified_cache.set(key, value)
        return False

    def create_ensemble(self, models: List[Any], method: str = "voting") -> Any:
        """Create ensemble model."""
        if self.config.enable_ml_common and self.ensemble_manager:
            return self.ensemble_manager.create_ensemble(models, method)
        return None

    def detect_hmm_regimes(self, data: np.ndarray, n_regimes: int = 3) -> Dict[str, Any]:
        """Detect regimes using HMM."""
        if self.config.enable_ml_common and self.hmm_regime_detector:
            return self.hmm_regime_detector.detect_regimes(data, n_regimes)
        return {'regime_sequence': np.zeros(len(data)), 'regime_probabilities': np.ones((len(data), n_regimes)) / n_regimes}

    def engineer_features(self, data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Engineer features using advanced feature engineering."""
        if self.config.enable_ml_common and self.feature_engineer:
            return self.feature_engineer.engineer_features(data, features)
        return data

    def detect_gaps(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect gaps in time series data."""
        if self.config.enable_ml_common and self.gap_detector:
            return self.gap_detector.detect_gaps(data)
        return {'gaps_found': 0, 'gap_locations': []}

    def download_historical_data(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download historical data."""
        if self.config.enable_ml_common and self.historical_downloader:
            return self.historical_downloader.download_data(symbol, interval, start_date, end_date)
        return pd.DataFrame()

    def score_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Score data quality comprehensively."""
        if self.config.enable_data_quality_checks and self.quality_scorer:
            return self.quality_scorer.score_quality(data)
        return {'overall_score': 0.5, 'quality_issues': []}

    def calculate_advanced_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced quality metrics."""
        if self.config.enable_data_quality_checks and self.advanced_quality:
            return self.advanced_quality.calculate_metrics(data)
        return {'metrics': {}}
    
    # =============================================================================
    # PERFORMANCE MONITORING
    # =============================================================================
    
    def timed_operation(self, func: Callable) -> Callable:
        """Time operation execution."""
        if self.config.enable_performance_monitoring:
            return timed_operation(func)
        else:
            return func
    
    def get_integration_status(self) -> Dict[str, UtilityIntegrationStatus]:
        """Get the status of all utility integrations."""
        return self.integration_status
    
    def get_available_utilities(self) -> List[str]:
        """Get list of available utilities."""
        available = []
        for utility, status in self.integration_status.items():
            if status == UtilityIntegrationStatus.AVAILABLE:
                available.append(utility)
        return available
    
    def get_unavailable_utilities(self) -> List[str]:
        """Get list of unavailable utilities."""
        unavailable = []
        for utility, status in self.integration_status.items():
            if status == UtilityIntegrationStatus.UNAVAILABLE:
                unavailable.append(utility)
        return unavailable
    
    def cleanup_resources(self) -> bool:
        """Clean up resources and optimize memory."""
        try:
            if self.config.enable_m1_optimizations:
                cleanup_m1_optimizers()
            
            if self.memory_optimizer and hasattr(self.memory_optimizer, 'stop_monitoring'):
                self.memory_optimizer.stop_monitoring()
            
            self.logger.info("🧹 Resources cleaned up successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")
            return False


# Factory function for easy initialization
def create_enhanced_utility_integration(config: Optional[UtilityIntegrationConfig] = None) -> EnhancedUtilityIntegration:
    """Create an enhanced utility integration instance."""
    return EnhancedUtilityIntegration(config)


# Convenience functions for common operations
def safe_dataframe_operation_enhanced(df: pd.DataFrame, operation: Callable, *args, **kwargs) -> pd.DataFrame:
    """Enhanced safe DataFrame operation."""
    integration = create_enhanced_utility_integration()
    return integration.safe_dataframe_operation(df, operation, *args, **kwargs)


def validate_dataframe_enhanced(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Enhanced DataFrame validation."""
    integration = create_enhanced_utility_integration()
    return integration.validate_dataframe_columns(df, required_columns)


def optimize_dataframe_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced DataFrame optimization."""
    integration = create_enhanced_utility_integration()
    return integration.optimize_dataframe_dtypes(df)


def save_data_enhanced(data: Any, filepath: str, format: str = None) -> bool:
    """Enhanced data saving."""
    integration = create_enhanced_utility_integration()
    return integration.save_data(data, filepath, format)


def load_data_enhanced(filepath: str) -> Optional[Any]:
    """Enhanced data loading."""
    integration = create_enhanced_utility_integration()
    return integration.load_data(filepath)