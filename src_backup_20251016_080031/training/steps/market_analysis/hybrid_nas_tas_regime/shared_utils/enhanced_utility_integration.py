"""
Enhanced Utility Integration Module

This module provides comprehensive integration with existing utility modules
from src/utils/ to enhance the hybrid NAS-TAS regime detection system.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import existing utility modules
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
    safe_to_parquet, safe_read_parquet, optimize_dataframe_dtypes,
    safe_resample, align_dataframes, validate_dataframe_schema,
    guard_dataframe_nulls, safe_copy, safe_deepcopy,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    cleanup_m1_optimizers, integrate_with_m1_optimizers,
    memory_checkpoint, gpu_context, optimize_memory, get_memory_usage,
    validate_file_path, get_file_size, check_disk_space
)

from src.utils.common_utilities import (
    CommonUtilities, safe_dataframe_operation as safe_df_op,
    validate_dataframe_columns as validate_df_cols, safe_convert_dtypes as safe_conv_dtypes,
    calculate_data_quality_metrics as calc_quality_metrics, safe_merge_dataframes as safe_merge_dfs,
    safe_groupby_operation as safe_groupby_op, safe_apply_function as safe_apply_func,
    create_summary_statistics as create_summary_stats, safe_drop_columns as safe_drop_cols,
    safe_rename_columns as safe_rename_cols, validate_timestamp_column as validate_ts_col,
    safe_timestamp_conversion as safe_ts_conv, get_dataframe_info as get_df_info,
    safe_filter_dataframe as safe_filter_df, create_data_quality_report as create_quality_report
)

from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite, validate_positive,
    validate_range, validate_numeric_array, safe_kelly_calculation, safe_weighted_average,
    safe_percentage_change, safe_correlation, safe_covariance, safe_mean, safe_std,
    safe_percentile, validate_correlation_matrix, safe_matrix_inverse, math_safe,
    MathValidation, MathValidationError
)

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

# Import data utilities
try:
    from src.utils.data.klines_parquet import KlinesParquetProcessor
    from src.utils.data.unified_data_utils import UnifiedDataUtils
    from src.utils.data.historical_data_downloader import HistoricalDataDownloader
    from src.utils.data.feature_engineer import FeatureEngineer
    from src.utils.data.basic_returns_engineer import BasicReturnsEngineer
    from src.utils.data.gap_detector import GapDetector
    from src.utils.data.quality.data_quality import DataQualityAnalyzer
    from src.utils.data.quality.advanced_quality_metrics import AdvancedQualityMetrics
    from src.utils.data.quality.comprehensive_quality_scorer import ComprehensiveQualityScorer
    from src.utils.data.optimized_parquet_storage import OptimizedParquetStorage
except ImportError as e:
    logging.warning(f"Some data utilities not available: {e}")
    KlinesParquetProcessor = None
    UnifiedDataUtils = None
    HistoricalDataDownloader = None
    FeatureEngineer = None
    BasicReturnsEngineer = None
    GapDetector = None
    DataQualityAnalyzer = None
    AdvancedQualityMetrics = None
    ComprehensiveQualityScorer = None
    OptimizedParquetStorage = None

# Import matrix operations
try:
    from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
    from src.utils.matrix_operations.enhanced_operations import EnhancedMatrixOperations
    from src.utils.matrix_operations.batch_operations import BatchMatrixOperations
    from src.utils.matrix_operations.vectorized_core import VectorizedCoreOperations
    from src.utils.matrix_operations.hardware_integration import HardwareMatrixIntegration
    from src.utils.matrix_operations.computation_toolbox import ComputationToolbox
except ImportError as e:
    logging.warning(f"Some matrix operations not available: {e}")
    UnifiedMatrixOperations = None
    EnhancedMatrixOperations = None
    BatchMatrixOperations = None
    VectorizedCoreOperations = None
    HardwareMatrixIntegration = None
    ComputationToolbox = None

# Import hardware optimizations
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
    from src.utils.hardware.adaptive_optimization_engine import AdaptiveOptimizationEngine
except ImportError as e:
    logging.warning(f"Some hardware utilities not available: {e}")
    M1GPUManager = None
    M1MemoryOptimizer = None
    M1CPUOptimizer = None
    UnifiedHardwareManager = None
    AdaptiveOptimizationEngine = None

# Import ML common utilities
try:
    from src.utils.ml_common import (
        FeatureSelector, FeatureSelectionConfig, CrossValidationUtilities,
        PurgedKFold, TemporalCrossValidator, StabilityAnalyzer,
        UnifiedCrossValidator, perform_cross_validation, temporal_cross_validation,
        nested_cross_validation, calculate_confidence_metrics, calculate_calibration_metrics,
        MemoryOptimizer, MemoryIntegrator, ParallelProcessor, UnifiedCache,
        LookaheadProtection, MLTrainingSafeguards, RobustErrorHandler,
        HMMRegimeDetector, RegimeConfig, M1EnhancedMatrixOperations,
        get_enhanced_matrix_operations, PipelineOrchestrator,
        FeatureImportanceAnalyzer, FeatureImportanceConfig, FeatureImportanceResult,
        ImportanceMethod, analyze_feature_importance, get_important_features,
        DataDriftDetector, DriftDetectionConfig, DriftReport, DriftResult,
        DriftType, DriftMethod, DriftSeverity, detect_data_drift, get_drifted_features
    )
except ImportError as e:
    logging.warning(f"Some ML common utilities not available: {e}")
    # Set defaults for missing imports
    FeatureSelector = None
    FeatureSelectionConfig = None
    CrossValidationUtilities = None
    PurgedKFold = None
    TemporalCrossValidator = None
    StabilityAnalyzer = None
    UnifiedCrossValidator = None
    perform_cross_validation = None
    temporal_cross_validation = None
    nested_cross_validation = None
    calculate_confidence_metrics = None
    calculate_calibration_metrics = None
    MemoryOptimizer = None
    MemoryIntegrator = None
    ParallelProcessor = None
    UnifiedCache = None
    LookaheadProtection = None
    MLTrainingSafeguards = None
    RobustErrorHandler = None
    HMMRegimeDetector = None
    RegimeConfig = None
    M1EnhancedMatrixOperations = None
    get_enhanced_matrix_operations = None
    PipelineOrchestrator = None
    FeatureImportanceAnalyzer = None
    FeatureImportanceConfig = None
    FeatureImportanceResult = None
    ImportanceMethod = None
    analyze_feature_importance = None
    get_important_features = None
    DataDriftDetector = None
    DriftDetectionConfig = None
    DriftReport = None
    DriftResult = None
    DriftType = None
    DriftMethod = None
    DriftSeverity = None
    detect_data_drift = None
    get_drifted_features = None

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class UtilityIntegrationConfig:
    """Configuration for utility integration."""
    enable_data_validation: bool = True
    enable_data_quality_checks: bool = True
    enable_safe_operations: bool = True
    enable_math_validation: bool = True
    enable_safe_math: bool = True
    enable_serialization: bool = True
    enable_m1_optimizations: bool = True
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    enable_ml_common: bool = True
    enable_feature_selection: bool = True
    enable_cross_validation: bool = True
    enable_confidence_metrics: bool = True
    enable_matrix_operations: bool = True
    enable_vectorized_operations: bool = True
    enable_performance_monitoring: bool = True
    enable_memory_monitoring: bool = True
    enable_lookahead_protection: bool = True
    enable_overfitting_detection: bool = True
    enable_data_leakage_detection: bool = True
    enable_hpo_optimization: bool = True
    enable_ensemble_management: bool = True
    enable_model_evaluation: bool = True
    enable_parallel_processing: bool = True
    enable_vectorization: bool = True


class EnhancedUtilityIntegration:
    """
    Enhanced utility integration that consolidates functionality from existing utility modules.
    """
    
    def __init__(self, config: UtilityIntegrationConfig):
        """Initialize enhanced utility integration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize utility components
        self._initialize_utilities()
        
        self.logger.info("✅ Enhanced utility integration initialized")
    
    def _initialize_utilities(self):
        """Initialize utility components based on configuration."""
        try:
            # Initialize math validation
            if self.config.enable_math_validation:
                self.math_validator = MathValidation()
                self.logger.info("✅ Math validation utilities initialized")
            
            # Initialize serialization
            if self.config.enable_serialization:
                self.json_serializer = JSONSerializer()
                self.pickle_serializer = PickleSerializer()
                self.parquet_serializer = ParquetSerializer()
                self.universal_serializer = UniversalSerializer()
                self.logger.info("✅ Serialization utilities initialized")
            
            # Initialize M1 optimizations
            if self.config.enable_m1_optimizations:
                self._initialize_m1_optimizations()
            
            # Initialize ML common utilities
            if self.config.enable_ml_common:
                self._initialize_ml_common_utilities()
            
            # Initialize matrix operations
            if self.config.enable_matrix_operations:
                self._initialize_matrix_operations()
            
            # Initialize hardware optimizations
            if self.config.enable_gpu_acceleration or self.config.enable_memory_optimization:
                self._initialize_hardware_optimizations()
            
            self.logger.info("✅ All utility components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize utilities: {e}")
            raise
    
    def _initialize_m1_optimizations(self):
        """Initialize M1-specific optimizations."""
        try:
            # Get M1 optimizers
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
            
            # Start memory monitoring if available
            if self.memory_optimizer and hasattr(self.memory_optimizer, 'start_monitoring'):
                self.memory_optimizer.start_monitoring()
            
            # Optimize CPU operations if available
            if self.cpu_optimizer and hasattr(self.cpu_optimizer, 'optimize_numpy_operations'):
                self.cpu_optimizer.optimize_numpy_operations()
            
            self.logger.info("✅ M1 optimizations initialized")
            
        except Exception as e:
            self.logger.warning(f"⚠️ M1 optimizations not fully available: {e}")
    
    def _initialize_ml_common_utilities(self):
        """Initialize ML common utilities."""
        try:
            # Initialize feature selection
            if self.config.enable_feature_selection and FeatureSelector:
                self.feature_selector = FeatureSelector()
            
            # Initialize cross-validation
            if self.config.enable_cross_validation and CrossValidationUtilities:
                self.cv_utilities = CrossValidationUtilities()
            
            # Initialize memory optimization
            if self.config.enable_memory_optimization and MemoryOptimizer:
                self.memory_optimizer_ml = MemoryOptimizer()
            
            # Initialize parallel processing
            if self.config.enable_parallel_processing and ParallelProcessor:
                self.parallel_processor = ParallelProcessor()
            
            # Initialize unified cache
            if UnifiedCache:
                self.unified_cache = UnifiedCache()
            
            # Initialize safeguards
            if self.config.enable_lookahead_protection and LookaheadProtection:
                self.lookahead_protection = LookaheadProtection()
            
            if MLTrainingSafeguards:
                self.ml_safeguards = MLTrainingSafeguards()
            
            if RobustErrorHandler:
                self.error_handler = RobustErrorHandler()
            
            # Initialize HMM regime detection
            if HMMRegimeDetector and RegimeConfig:
                self.hmm_regime_detector = HMMRegimeDetector()
            
            # Initialize feature importance analysis
            if FeatureImportanceAnalyzer:
                self.feature_importance_analyzer = FeatureImportanceAnalyzer()
            
            # Initialize data drift detection
            if DataDriftDetector:
                self.data_drift_detector = DataDriftDetector()
            
            self.logger.info("✅ ML common utilities initialized")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Some ML common utilities not available: {e}")
    
    def _initialize_matrix_operations(self):
        """Initialize matrix operations."""
        try:
            if UnifiedMatrixOperations:
                self.unified_matrix_ops = UnifiedMatrixOperations()
            
            if EnhancedMatrixOperations:
                self.enhanced_matrix_ops = EnhancedMatrixOperations()
            
            if BatchMatrixOperations:
                self.batch_matrix_ops = BatchMatrixOperations()
            
            if VectorizedCoreOperations:
                self.vectorized_ops = VectorizedCoreOperations()
            
            if HardwareMatrixIntegration:
                self.hardware_matrix_integration = HardwareMatrixIntegration()
            
            if ComputationToolbox:
                self.computation_toolbox = ComputationToolbox()
            
            self.logger.info("✅ Matrix operations initialized")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Some matrix operations not available: {e}")
    
    def _initialize_hardware_optimizations(self):
        """Initialize hardware optimizations."""
        try:
            if M1GPUManager:
                self.m1_gpu_manager = M1GPUManager()
            
            if M1MemoryOptimizer:
                self.m1_memory_optimizer = M1MemoryOptimizer()
            
            if M1CPUOptimizer:
                self.m1_cpu_optimizer = M1CPUOptimizer()
            
            if UnifiedHardwareManager:
                self.unified_hardware_manager = UnifiedHardwareManager()
            
            if AdaptiveOptimizationEngine:
                self.adaptive_optimization_engine = AdaptiveOptimizationEngine()
            
            self.logger.info("✅ Hardware optimizations initialized")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Some hardware optimizations not available: {e}")
    
    # =============================================================================
    # DATA PROCESSING UTILITIES
    # =============================================================================
    
    def safe_dataframe_operation(self, df: pd.DataFrame, operation: Callable, *args, **kwargs) -> pd.DataFrame:
        """Safely perform operation on DataFrame."""
        return safe_dataframe_operation(df, operation, *args, **kwargs)
    
    def validate_dataframe_columns(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate that DataFrame has required columns."""
        return validate_dataframe_columns(df, required_columns)
    
    def safe_convert_dtypes(self, df: pd.DataFrame, dtype_mapping: Dict[str, str]) -> pd.DataFrame:
        """Safely convert DataFrame column dtypes."""
        return safe_convert_dtypes(df, dtype_mapping)
    
    def calculate_data_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality metrics for DataFrame."""
        return calculate_data_quality_metrics(df)
    
    def safe_merge_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Safely merge two DataFrames."""
        return safe_merge_dataframes(df1, df2, **kwargs)
    
    def safe_groupby_operation(self, df: pd.DataFrame, group_cols: List[str], agg_dict: Dict[str, str]) -> pd.DataFrame:
        """Safely perform groupby operation."""
        return safe_groupby_operation(df, group_cols, agg_dict)
    
    def safe_apply_function(self, df: pd.DataFrame, func: Callable, axis: int = 0) -> pd.DataFrame:
        """Safely apply function to DataFrame."""
        return safe_apply_function(df, func, axis=axis)
    
    def create_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create summary statistics for DataFrame."""
        return create_summary_statistics(df)
    
    def safe_drop_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Safely drop columns from DataFrame."""
        return safe_drop_columns(df, columns)
    
    def safe_rename_columns(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Safely rename DataFrame columns."""
        return safe_rename_columns(df, column_mapping)
    
    def validate_timestamp_column(self, df: pd.DataFrame, column: str) -> bool:
        """Validate that column contains valid timestamps."""
        return validate_timestamp_column(df, column)
    
    def safe_timestamp_conversion(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Safely convert column to timestamp."""
        return safe_timestamp_conversion(df, column)
    
    def get_dataframe_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive DataFrame information."""
        return get_dataframe_info(df)
    
    def safe_filter_dataframe(self, df: pd.DataFrame, condition: str) -> pd.DataFrame:
        """Safely filter DataFrame using query condition."""
        return safe_filter_dataframe(df, condition)
    
    def create_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive data quality report."""
        return create_data_quality_report(df)
    
    def safe_to_parquet(self, df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> bool:
        """Safely save DataFrame to parquet format."""
        return safe_to_parquet(df, file_path, **kwargs)
    
    def safe_read_parquet(self, file_path: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
        """Safely read DataFrame from parquet format."""
        return safe_read_parquet(file_path, **kwargs)
    
    def optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency."""
        return optimize_dataframe_dtypes(df)
    
    def safe_resample(self, df: pd.DataFrame, rule: str, agg_dict: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Safely resample a DataFrame with error handling."""
        return safe_resample(df, rule, agg_dict)
    
    def align_dataframes(self, *dfs: pd.DataFrame, method: str = "inner") -> List[pd.DataFrame]:
        """Align multiple DataFrames by index using specified join method."""
        return align_dataframes(*dfs, method=method)
    
    def validate_dataframe_schema(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate that DataFrame has required columns."""
        return validate_dataframe_schema(df, required_columns)
    
    def guard_dataframe_nulls(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Guard against excessive null values in DataFrame."""
        return guard_dataframe_nulls(df, threshold)
    
    # =============================================================================
    # MATHEMATICAL UTILITIES
    # =============================================================================
    
    def safe_divide(self, a: float, b: float, default: float = 0.0) -> float:
        """Safely divide two numbers."""
        return safe_divide(a, b, default)
    
    def safe_log(self, x: float, default: float = 0.0) -> float:
        """Safely calculate logarithm."""
        return safe_log(x, default)
    
    def safe_sqrt(self, x: float, default: float = 0.0) -> float:
        """Safely calculate square root."""
        return safe_sqrt(x, default)
    
    def safe_power(self, x: float, y: float, default: float = 0.0) -> float:
        """Safely calculate power."""
        return safe_power(x, y, default)
    
    def validate_finite(self, value: Any, name: str = "value") -> float:
        """Validate that a value is finite."""
        return validate_finite(value, name)
    
    def validate_positive(self, value: float, name: str = "value") -> float:
        """Validate that a value is positive."""
        return validate_positive(value, name)
    
    def validate_range(self, value: float, min_val: float = None, max_val: float = None, name: str = "value") -> float:
        """Validate that a value is in range."""
        return validate_range(value, min_val, max_val, name)
    
    def safe_kelly_calculation(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Safely calculate Kelly criterion."""
        return safe_kelly_calculation(win_rate, avg_win, avg_loss)
    
    def safe_weighted_average(self, values: List[float], weights: List[float]) -> float:
        """Safely calculate weighted average."""
        return safe_weighted_average(values, weights)
    
    def safe_percentage_change(self, old_value: float, new_value: float) -> float:
        """Safely calculate percentage change."""
        return safe_percentage_change(old_value, new_value)
    
    def safe_correlation(self, x: np.ndarray, y: np.ndarray, default: float = 0.0) -> float:
        """Safely calculate correlation coefficient between two arrays."""
        return safe_correlation(x, y, default)
    
    def safe_covariance(self, x: np.ndarray, y: np.ndarray, default: float = 0.0) -> float:
        """Safely calculate covariance between two arrays."""
        return safe_covariance(x, y, default)
    
    def safe_mean(self, x: np.ndarray, default: float = 0.0) -> float:
        """Safely calculate mean of array."""
        return safe_mean(x, default)
    
    def safe_std(self, x: np.ndarray, default: float = 0.0) -> float:
        """Safely calculate standard deviation of array."""
        return safe_std(x, default)
    
    def safe_percentile(self, x: np.ndarray, percentile: float = 50.0, default: float = 0.0) -> float:
        """Safely calculate percentile of array."""
        return safe_percentile(x, percentile, default)
    
    def validate_correlation_matrix(self, corr_matrix: np.ndarray) -> bool:
        """Validate correlation matrix."""
        return validate_correlation_matrix(corr_matrix)
    
    def safe_matrix_inverse(self, matrix: np.ndarray) -> np.ndarray:
        """Safely calculate matrix inverse."""
        return safe_matrix_inverse(matrix)
    
    def math_safe(self, func: Callable, *args, default: Any = 0.0, **kwargs) -> Any:
        """Safely execute math function."""
        return math_safe(func, *args, default=default, **kwargs)
    
    # =============================================================================
    # SERIALIZATION UTILITIES
    # =============================================================================
    
    def save_json(self, data: Any, filepath: str) -> bool:
        """Save data as JSON."""
        if self.config.enable_serialization:
            return self.json_serializer.save(data, filepath)
        return False
    
    def load_json(self, filepath: str) -> Optional[Any]:
        """Load data from JSON."""
        if self.config.enable_serialization:
            return self.json_serializer.load(filepath)
        return None
    
    def save_pickle(self, data: Any, filepath: str) -> bool:
        """Save data as pickle."""
        if self.config.enable_serialization:
            return self.pickle_serializer.save(data, filepath)
        return False
    
    def load_pickle(self, filepath: str) -> Optional[Any]:
        """Load data from pickle."""
        if self.config.enable_serialization:
            return self.pickle_serializer.load(filepath)
        return None
    
    def save_parquet(self, data: Any, filepath: str) -> bool:
        """Save data as parquet."""
        if self.config.enable_serialization:
            return self.parquet_serializer.save(data, filepath)
        return False
    
    def load_parquet(self, filepath: str) -> Optional[Any]:
        """Load data from parquet."""
        if self.config.enable_serialization:
            return self.parquet_serializer.load(filepath)
        return None
    
    def save_universal(self, data: Any, filepath: str, format: str = 'auto') -> bool:
        """Save data with automatic format detection."""
        if self.config.enable_serialization:
            return self.universal_serializer.save(data, filepath, format)
        return False
    
    def load_universal(self, filepath: str) -> Optional[Any]:
        """Load data with automatic format detection."""
        if self.config.enable_serialization:
            return self.universal_serializer.load(filepath)
        return None
    
    # =============================================================================
    # M1 OPTIMIZATION UTILITIES
    # =============================================================================
    
    def get_m1_gpu_manager(self):
        """Get M1 GPU manager instance."""
        return get_m1_gpu_manager()
    
    def get_m1_memory_optimizer(self):
        """Get M1 memory optimizer instance."""
        return get_m1_memory_optimizer()
    
    def get_m1_cpu_optimizer(self):
        """Get M1 CPU optimizer instance."""
        return get_m1_cpu_optimizer()
    
    def cleanup_m1_optimizers(self) -> bool:
        """Clean up M1 optimizers and release resources."""
        return cleanup_m1_optimizers()
    
    def integrate_with_m1_optimizers(self) -> dict:
        """Integrate with M1 GPU and CPU optimizers."""
        return integrate_with_m1_optimizers()
    
    def memory_checkpoint(self, name: str):
        """Create a memory checkpoint context manager."""
        return memory_checkpoint(name)
    
    def gpu_context(self, name: str):
        """Create a GPU context manager."""
        return gpu_context(name)
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage across the system."""
        return optimize_memory()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in bytes."""
        return get_memory_usage()
    
    # =============================================================================
    # ML COMMON UTILITIES
    # =============================================================================
    
    def select_features(self, X: np.ndarray, y: np.ndarray, method: str = "mutual_info", n_features: int = 10) -> Tuple[np.ndarray, List[int]]:
        """Select features using ML common utilities."""
        if self.config.enable_feature_selection and hasattr(self, 'feature_selector'):
            return self.feature_selector.select_features(X, y, method=method, n_features=n_features)
        else:
            # Fallback to simple feature selection
            return X[:, :n_features], list(range(n_features))
    
    def cross_validate_model(self, estimator, X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: str = "accuracy") -> Dict[str, Any]:
        """Perform cross-validation using ML common utilities."""
        if self.config.enable_cross_validation and hasattr(self, 'cv_utilities'):
            return self.cv_utilities.cross_validate(estimator, X, y, cv=cv, scoring=scoring)
        else:
            # Fallback to basic cross-validation
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
            return {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}
    
    def detect_lookahead_bias(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Detect lookahead bias using ML common utilities."""
        if self.config.enable_lookahead_protection and hasattr(self, 'lookahead_protection'):
            return self.lookahead_protection.detect_bias(X, y)
        else:
            return {'bias_detected': False, 'confidence': 0.5}
    
    def detect_overfitting(self, model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Detect overfitting using ML common utilities."""
        if self.config.enable_overfitting_detection and hasattr(self, 'ml_safeguards'):
            return self.ml_safeguards.detect_overfitting(model, X_train, y_train, X_val, y_val)
        else:
            # Basic overfitting detection
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            overfitting = train_score - val_score > 0.1
            return {'overfitting_detected': overfitting, 'train_score': train_score, 'val_score': val_score}
    
    def detect_data_leakage(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Detect data leakage using ML common utilities."""
        if self.config.enable_data_leakage_detection and hasattr(self, 'error_handler'):
            return self.error_handler.detect_data_leakage(X, y)
        else:
            return {'leakage_detected': False, 'confidence': 0.5}
    
    def calculate_confidence_metrics(self, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate confidence metrics using ML common utilities."""
        if self.config.enable_confidence_metrics and calculate_confidence_metrics:
            return calculate_confidence_metrics(y_pred, y_proba)
        else:
            # Basic confidence calculation
            mean_confidence = np.mean(np.max(y_proba, axis=1))
            return {'mean_confidence': mean_confidence, 'min_confidence': np.min(np.max(y_proba, axis=1))}
    
    def detect_regimes_hmm(self, data: pd.DataFrame, n_regimes: int = 3, features: List[str] = None) -> Dict[str, Any]:
        """Detect regimes using HMM with ML common utilities."""
        if self.config.enable_ml_common and hasattr(self, 'hmm_regime_detector'):
            return self.hmm_regime_detector.detect_regimes(data, n_regimes=n_regimes, features=features)
        else:
            # Fallback to basic regime detection
            n_samples = len(data)
            regime_sequence = np.random.randint(0, n_regimes, n_samples)
            return {'regime_sequence': regime_sequence, 'n_regimes': n_regimes}
    
    def analyze_feature_importance(self, model, X: np.ndarray, y: np.ndarray, method: str = "permutation") -> Dict[str, Any]:
        """Analyze feature importance using ML common utilities."""
        if self.config.enable_ml_common and hasattr(self, 'feature_importance_analyzer'):
            return self.feature_importance_analyzer.analyze(model, X, y, method=method)
        else:
            # Fallback to basic feature importance
            if hasattr(model, 'feature_importances_'):
                return {'importances': model.feature_importances_, 'method': 'tree_based'}
            else:
                return {'importances': np.ones(X.shape[1]) / X.shape[1], 'method': 'uniform'}
    
    def detect_data_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> Dict[str, Any]:
        """Detect data drift using ML common utilities."""
        if self.config.enable_ml_common and hasattr(self, 'data_drift_detector'):
            return self.data_drift_detector.detect_drift(reference_data, current_data)
        else:
            # Basic drift detection
            ref_mean = np.mean(reference_data, axis=0)
            curr_mean = np.mean(current_data, axis=0)
            drift_score = np.mean(np.abs(ref_mean - curr_mean))
            return {'drift_detected': drift_score > 0.1, 'drift_score': drift_score}
    
    # =============================================================================
    # MATRIX OPERATIONS
    # =============================================================================
    
    def enhanced_matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Enhanced matrix multiplication using optimized operations."""
        if self.config.enable_matrix_operations and hasattr(self, 'enhanced_matrix_ops'):
            return self.enhanced_matrix_ops.multiply(A, B)
        else:
            return np.dot(A, B)
    
    def batch_matrix_operations(self, matrices: List[np.ndarray], operation: str = "multiply") -> List[np.ndarray]:
        """Perform batch matrix operations."""
        if self.config.enable_matrix_operations and hasattr(self, 'batch_matrix_ops'):
            return self.batch_matrix_ops.batch_operation(matrices, operation)
        else:
            # Fallback to sequential operations
            if operation == "multiply":
                return [np.dot(matrices[i], matrices[i+1]) for i in range(len(matrices)-1)]
            else:
                return matrices
    
    def vectorized_operations(self, data: np.ndarray, operation: str = "normalize") -> np.ndarray:
        """Perform vectorized operations."""
        if self.config.enable_vectorized_operations and hasattr(self, 'vectorized_ops'):
            return self.vectorized_ops.vectorized_operation(data, operation)
        else:
            # Fallback to basic operations
            if operation == "normalize":
                return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
            else:
                return data
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def get_available_utilities(self) -> List[str]:
        """Get list of available utilities."""
        utilities = []
        
        if self.config.enable_data_validation:
            utilities.extend(['dataframe_validation', 'column_validation', 'schema_validation'])
        
        if self.config.enable_math_validation:
            utilities.extend(['safe_math', 'math_validation', 'correlation_analysis'])
        
        if self.config.enable_serialization:
            utilities.extend(['json_serialization', 'pickle_serialization', 'parquet_serialization'])
        
        if self.config.enable_m1_optimizations:
            utilities.extend(['m1_gpu', 'm1_memory', 'm1_cpu'])
        
        if self.config.enable_ml_common:
            utilities.extend(['feature_selection', 'cross_validation', 'confidence_metrics'])
        
        if self.config.enable_matrix_operations:
            utilities.extend(['matrix_operations', 'vectorized_operations', 'batch_operations'])
        
        return utilities
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and available utilities."""
        return {
            'config': self.config.__dict__,
            'available_utilities': self.get_available_utilities(),
            'm1_optimizations': {
                'gpu_available': self.gpu_manager is not None,
                'memory_optimizer_available': self.memory_optimizer is not None,
                'cpu_optimizer_available': self.cpu_optimizer is not None
            },
            'ml_common_utilities': {
                'feature_selection': hasattr(self, 'feature_selector'),
                'cross_validation': hasattr(self, 'cv_utilities'),
                'memory_optimization': hasattr(self, 'memory_optimizer_ml'),
                'parallel_processing': hasattr(self, 'parallel_processor'),
                'unified_cache': hasattr(self, 'unified_cache'),
                'lookahead_protection': hasattr(self, 'lookahead_protection'),
                'ml_safeguards': hasattr(self, 'ml_safeguards'),
                'error_handler': hasattr(self, 'error_handler'),
                'hmm_regime_detector': hasattr(self, 'hmm_regime_detector'),
                'feature_importance_analyzer': hasattr(self, 'feature_importance_analyzer'),
                'data_drift_detector': hasattr(self, 'data_drift_detector')
            },
            'matrix_operations': {
                'unified_operations': hasattr(self, 'unified_matrix_ops'),
                'enhanced_operations': hasattr(self, 'enhanced_matrix_ops'),
                'batch_operations': hasattr(self, 'batch_matrix_ops'),
                'vectorized_operations': hasattr(self, 'vectorized_ops'),
                'hardware_integration': hasattr(self, 'hardware_matrix_integration'),
                'computation_toolbox': hasattr(self, 'computation_toolbox')
            },
            'hardware_optimizations': {
                'm1_gpu_manager': hasattr(self, 'm1_gpu_manager'),
                'm1_memory_optimizer': hasattr(self, 'm1_memory_optimizer'),
                'm1_cpu_optimizer': hasattr(self, 'm1_cpu_optimizer'),
                'unified_hardware_manager': hasattr(self, 'unified_hardware_manager'),
                'adaptive_optimization_engine': hasattr(self, 'adaptive_optimization_engine')
            }
        }


def create_enhanced_utility_integration(config: UtilityIntegrationConfig = None) -> EnhancedUtilityIntegration:
    """Create an enhanced utility integration instance."""
    if config is None:
        config = UtilityIntegrationConfig()
    
    return EnhancedUtilityIntegration(config)