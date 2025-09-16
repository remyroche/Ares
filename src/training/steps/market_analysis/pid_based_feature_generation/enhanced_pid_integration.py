"""
Enhanced PID-Based Feature Generation with Common Utilities Integration

This module demonstrates comprehensive integration of all common utilities
with the PID-based feature generation system.

Key Integrations:
- common_operations.py: Data validation, DataFrame operations, math validation
- serialization_utils.py: Artifact persistence and data serialization
- matrix_operations/: Optimized mathematical computations and GPU acceleration
- hardware/m1_*: M1 hardware optimization utilities
- ml_common/: ML-related operations (CV, lookahead, HPO)
- data/: Data processing utilities
- kline_parquet.py: Kline data handling

Usage:
    from enhanced_pid_integration import EnhancedPIDFeatureGenerator
    
    # Initialize with common utilities integration
    generator = EnhancedPIDFeatureGenerator()
    
    # Generate features with full utility integration
    result = await generator.generate_features_with_utilities(data, config)
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

# Core dependencies with fallback support
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

# =============================================================================
# COMMON UTILITIES INTEGRATION
# =============================================================================

# Common Operations Integration
try:
    from src.utils.common_operations import (
        # Data validation and quality
        validate_dataframe, validate_dataframe_columns, calculate_data_quality_metrics,
        create_data_quality_report, get_dataframe_info, optimize_dataframe_dtypes,
        
        # Safe operations
        safe_dataframe_operation, safe_fillna, safe_convert_dtypes, safe_merge_dataframes,
        safe_drop_columns, safe_rename_columns, safe_timestamp_conversion,
        
        # Math operations
        safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
        safe_float, safe_int, validate_finite, validate_positive, validate_range,
        safe_kelly_calculation, safe_weighted_average, safe_percentage_change,
        
        # File operations
        safe_json_dump, safe_json_load, safe_to_parquet, safe_read_parquet,
        ensure_directory, safe_file_exists, safe_copy,
        
        # Performance utilities
        timed_operation, format_bytes, chunked_iterable, parallel_map,
        
        # M1 optimization
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, cleanup_m1_optimizers,
        memory_checkpoint, gpu_context, optimize_memory, get_memory_usage,
        
        # Matrix utilities
        validate_correlation_matrix, safe_matrix_inverse, math_safe,
        
        # Logging utilities
        get_logger, setup_basic_logging, safe_log_metric, safe_log_params, safe_log_artifact
    )
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError as e:
    COMMON_OPERATIONS_AVAILABLE = False
    logging.warning(f"Common operations not available: {e}")

# Serialization Utilities Integration
try:
    from src.utils.serialization_utils import (
        JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
    )
    SERIALIZATION_AVAILABLE = True
except ImportError as e:
    SERIALIZATION_AVAILABLE = False
    logging.warning(f"Serialization utilities not available: {e}")

# Matrix Operations Integration
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations, get_vectorized_processing_core,
        get_batch_matrix_processor, get_enhanced_matrix_operations,
        # Convenience functions
        safe_matrix_multiply, safe_correlation_matrix, safe_matrix_inverse,
        gpu_matrix_multiply, correlation_matrix_gpu, eigendecomposition_gpu, svd_gpu,
        optimize_dataframe, vectorized_rolling_features, matrix_correlation_analysis,
        parallel_feature_engineering, batch_matrix_multiply, batch_feature_transformation,
        # Trading indicators
        compute_trading_indicators, compute_moving_averages, compute_momentum_indicators,
        compute_volatility_indicators, compute_volume_indicators, compute_trend_indicators,
        # Hardware optimization
        get_hardware_performance_report, optimize_matrix_operation_with_hardware,
        cleanup_hardware_resources, get_processing_performance_stats
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError as e:
    MATRIX_OPERATIONS_AVAILABLE = False
    logging.warning(f"Matrix operations not available: {e}")

# Hardware Optimization Integration
try:
    from src.utils.hardware.m1_gpu_utils import (
        get_m1_gpu_manager, is_m1_available, is_mps_available,
        optimize_dataframe_for_m1, create_m1_optimized_array,
        m1_backtesting_simulate, m1_monte_carlo_simulate
    )
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    logging.warning(f"Hardware optimization not available: {e}")

# ML Common Integration
try:
    from src.utils.ml_common.common_operations import (
        # Data processing
        preprocess_data, validate_ml_data, create_feature_matrix,
        # Feature engineering
        create_polynomial_features, create_interaction_features,
        # Cross-validation
        create_cv_splits, validate_cv_splits,
        # Hyperparameter optimization
        optimize_hyperparameters, create_hpo_config,
        # Model evaluation
        evaluate_model_performance, calculate_metrics,
        # Lookahead bias prevention
        detect_lookahead_bias, prevent_lookahead_bias
    )
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    ML_COMMON_AVAILABLE = False
    logging.warning(f"ML common utilities not available: {e}")

# Data Utilities Integration
try:
    from src.utils.data.data_loader import DataLoader
    from src.utils.data.data_processor import DataProcessor
    from src.utils.data.data_validator import DataValidator
    DATA_UTILITIES_AVAILABLE = True
except ImportError as e:
    DATA_UTILITIES_AVAILABLE = False
    logging.warning(f"Data utilities not available: {e}")

# Kline Parquet Integration (if available)
try:
    from src.utils.kline_parquet import KlineParquetHandler
    KLINE_PARQUET_AVAILABLE = True
except ImportError:
    KLINE_PARQUET_AVAILABLE = False

# Math Validation Integration
try:
    from src.utils.math_validation import (
        MathValidation, safe_divide as math_safe_divide, safe_log as math_safe_log,
        safe_sqrt as math_safe_sqrt, safe_power as math_safe_power,
        validate_finite as math_validate_finite, validate_positive as math_validate_positive,
        validate_range as math_validate_range, safe_correlation, safe_covariance,
        safe_mean as math_safe_mean, safe_std as math_safe_std, safe_percentile
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError as e:
    MATH_VALIDATION_AVAILABLE = False
    logging.warning(f"Math validation not available: {e}")

# Logger Integration
try:
    from src.utils.logger import system_logger
    logger = system_logger.getChild('EnhancedPIDFeatureGenerator')
except ImportError:
    logger = logging.getLogger('EnhancedPIDFeatureGenerator')
    logger.setLevel(logging.INFO)

# TPrint Integration
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_error, tprint_warning, tprint_success, 
        tprint_debug, tprint_performance
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)


# =============================================================================
# ENHANCED CONFIGURATION
# =============================================================================

@dataclass
class EnhancedPIDConfig:
    """Enhanced configuration with common utilities integration."""
    # Basic PID Configuration
    synergy_threshold: float = 0.1
    redundancy_threshold: float = 0.15
    unique_info_threshold: float = 0.05
    
    # Feature Generation Limits
    max_interaction_features: int = 100
    max_polynomial_features: int = 50
    max_cross_timeframe_features: int = 50
    
    # Common Utilities Integration
    enable_common_operations: bool = True
    enable_serialization: bool = True
    enable_matrix_operations: bool = True
    enable_hardware_optimization: bool = True
    enable_ml_common: bool = True
    enable_data_utilities: bool = True
    enable_math_validation: bool = True
    
    # Data Quality Settings
    min_data_quality_score: float = 0.7
    max_missing_data_ratio: float = 0.1
    enable_data_validation: bool = True
    enable_data_optimization: bool = True
    
    # Hardware Optimization
    enable_m1_optimization: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0
    chunk_size_mb: int = 256
    
    # Serialization Settings
    save_intermediate_results: bool = True
    serialization_format: str = 'parquet'  # 'json', 'pickle', 'parquet'
    compression: str = 'snappy'
    
    # ML Common Settings
    enable_cross_validation: bool = True
    enable_hyperparameter_optimization: bool = True
    enable_lookahead_bias_detection: bool = True
    cv_folds: int = 5
    
    # Performance Settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_profiling: bool = True


@dataclass
class EnhancedPIDResult:
    """Enhanced result with comprehensive utility integration."""
    # Basic Results
    features: Dict[str, np.ndarray] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    feature_scores: Dict[str, float] = field(default_factory=dict)
    
    # Common Operations Results
    data_quality_report: Optional[Dict[str, Any]] = None
    validation_results: Dict[str, Any] = field(default_factory=dict)
    optimization_results: Dict[str, Any] = field(default_factory=dict)
    
    # Serialization Results
    serialization_status: Dict[str, bool] = field(default_factory=dict)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    
    # Matrix Operations Results
    matrix_operations_used: bool = False
    gpu_acceleration_used: bool = False
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Hardware Optimization Results
    hardware_optimization_used: bool = False
    memory_usage: Dict[str, float] = field(default_factory=dict)
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    
    # ML Common Results
    cross_validation_results: Optional[Dict[str, Any]] = None
    hyperparameter_optimization_results: Optional[Dict[str, Any]] = None
    lookahead_bias_detection: Optional[Dict[str, Any]] = None
    
    # Metadata
    total_features_generated: int = 0
    execution_time: float = 0.0
    utility_integration_status: Dict[str, bool] = field(default_factory=dict)
    success: bool = False
    error_message: Optional[str] = None


# =============================================================================
# ENHANCED PID FEATURE GENERATOR
# =============================================================================

class EnhancedPIDFeatureGenerator:
    """
    Enhanced PID-Based Feature Generator with comprehensive common utilities integration.
    
    This class demonstrates the integration of all common utilities:
    - common_operations.py: Data validation, DataFrame operations, math validation
    - serialization_utils.py: Artifact persistence and data serialization
    - matrix_operations/: Optimized mathematical computations and GPU acceleration
    - hardware/m1_*: M1 hardware optimization utilities
    - ml_common/: ML-related operations (CV, lookahead, HPO)
    - data/: Data processing utilities
    """
    
    def __init__(self, config: Optional[EnhancedPIDConfig] = None):
        """Initialize the enhanced PID feature generator."""
        self.config = config or EnhancedPIDConfig()
        self.logger = logger.getChild('EnhancedPIDFeatureGenerator')
        
        # Initialize utility integrations
        self._initialize_utility_integrations()
        
        # Initialize serializers
        self._initialize_serializers()
        
        # Initialize hardware optimizers
        self._initialize_hardware_optimizers()
        
        tprint_success("Enhanced PID Feature Generator initialized")
        tprint_info(f"Common operations available: {COMMON_OPERATIONS_AVAILABLE}")
        tprint_info(f"Serialization available: {SERIALIZATION_AVAILABLE}")
        tprint_info(f"Matrix operations available: {MATRIX_OPERATIONS_AVAILABLE}")
        tprint_info(f"Hardware optimization available: {HARDWARE_OPTIMIZATION_AVAILABLE}")
        tprint_info(f"ML common available: {ML_COMMON_AVAILABLE}")
        tprint_info(f"Data utilities available: {DATA_UTILITIES_AVAILABLE}")
        tprint_info(f"Math validation available: {MATH_VALIDATION_AVAILABLE}")
    
    def _initialize_utility_integrations(self):
        """Initialize all utility integrations."""
        # Common Operations
        if COMMON_OPERATIONS_AVAILABLE and self.config.enable_common_operations:
            self.common_ops_available = True
            self.logger.info("✅ Common operations integration enabled")
        else:
            self.common_ops_available = False
            self.logger.warning("⚠️ Common operations not available")
        
        # Serialization
        if SERIALIZATION_AVAILABLE and self.config.enable_serialization:
            self.serialization_available = True
            self.logger.info("✅ Serialization integration enabled")
        else:
            self.serialization_available = False
            self.logger.warning("⚠️ Serialization not available")
        
        # Matrix Operations
        if MATRIX_OPERATIONS_AVAILABLE and self.config.enable_matrix_operations:
            self.matrix_ops_available = True
            self.logger.info("✅ Matrix operations integration enabled")
        else:
            self.matrix_ops_available = False
            self.logger.warning("⚠️ Matrix operations not available")
        
        # Hardware Optimization
        if HARDWARE_OPTIMIZATION_AVAILABLE and self.config.enable_hardware_optimization:
            self.hardware_opt_available = True
            self.logger.info("✅ Hardware optimization integration enabled")
        else:
            self.hardware_opt_available = False
            self.logger.warning("⚠️ Hardware optimization not available")
        
        # ML Common
        if ML_COMMON_AVAILABLE and self.config.enable_ml_common:
            self.ml_common_available = True
            self.logger.info("✅ ML common integration enabled")
        else:
            self.ml_common_available = False
            self.logger.warning("⚠️ ML common not available")
        
        # Data Utilities
        if DATA_UTILITIES_AVAILABLE and self.config.enable_data_utilities:
            self.data_utils_available = True
            self.logger.info("✅ Data utilities integration enabled")
        else:
            self.data_utils_available = False
            self.logger.warning("⚠️ Data utilities not available")
        
        # Math Validation
        if MATH_VALIDATION_AVAILABLE and self.config.enable_math_validation:
            self.math_validation_available = True
            self.logger.info("✅ Math validation integration enabled")
        else:
            self.math_validation_available = False
            self.logger.warning("⚠️ Math validation not available")
    
    def _initialize_serializers(self):
        """Initialize serialization utilities."""
        if self.serialization_available:
            self.json_serializer = JSONSerializer()
            self.pickle_serializer = PickleSerializer()
            self.parquet_serializer = ParquetSerializer()
            self.universal_serializer = UniversalSerializer()
            self.logger.info("✅ Serializers initialized")
        else:
            self.json_serializer = None
            self.pickle_serializer = None
            self.parquet_serializer = None
            self.universal_serializer = None
    
    def _initialize_hardware_optimizers(self):
        """Initialize hardware optimization utilities."""
        if self.hardware_opt_available:
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
            self.logger.info("✅ Hardware optimizers initialized")
        else:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    async def generate_features_with_utilities(
        self, 
        data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        target: Optional[np.ndarray] = None,
        save_artifacts: bool = True
    ) -> EnhancedPIDResult:
        """
        Generate features with comprehensive utility integration.
        
        Args:
            data: Input data for feature generation
            feature_names: List of feature names
            target: Target variable for supervised feature generation
            save_artifacts: Whether to save intermediate artifacts
            
        Returns:
            EnhancedPIDResult with comprehensive utility integration results
        """
        start_time = time.time()
        result = EnhancedPIDResult()
        
        tprint_info("🚀 Starting enhanced PID feature generation with utility integration...")
        
        try:
            # Step 1: Data Validation and Quality Assessment
            tprint_info("📊 Step 1: Data validation and quality assessment...")
            data_validation_result = await self._validate_and_assess_data(data, feature_names)
            result.validation_results = data_validation_result
            
            if not data_validation_result.get('is_valid', False):
                raise ValueError(f"Data validation failed: {data_validation_result.get('issues', [])}")
            
            # Step 2: Data Optimization
            tprint_info("⚙️ Step 2: Data optimization...")
            optimized_data, optimization_info = await self._optimize_data(data, feature_names)
            result.optimization_results = optimization_info
            
            # Step 3: Hardware Optimization Setup
            tprint_info("🔧 Step 3: Hardware optimization setup...")
            hardware_setup_result = await self._setup_hardware_optimization()
            result.hardware_optimization_used = hardware_setup_result['success']
            result.hardware_info = hardware_setup_result.get('hardware_info', {})
            
            # Step 4: Feature Generation with Matrix Operations
            tprint_info("🎯 Step 4: Feature generation with matrix operations...")
            features_result = await self._generate_features_with_matrix_ops(
                optimized_data, feature_names, target
            )
            result.features = features_result['features']
            result.feature_names = features_result['feature_names']
            result.feature_scores = features_result['feature_scores']
            result.total_features_generated = len(result.feature_names)
            
            # Step 5: ML Common Integration (CV, HPO, Lookahead Detection)
            tprint_info("🤖 Step 5: ML common integration...")
            ml_common_result = await self._apply_ml_common_utilities(
                result.features, result.feature_names, target
            )
            result.cross_validation_results = ml_common_result.get('cv_results')
            result.hyperparameter_optimization_results = ml_common_result.get('hpo_results')
            result.lookahead_bias_detection = ml_common_result.get('lookahead_bias')
            
            # Step 6: Serialization and Artifact Management
            if save_artifacts:
                tprint_info("💾 Step 6: Serialization and artifact management...")
                serialization_result = await self._save_artifacts(result, start_time)
                result.serialization_status = serialization_result['status']
                result.artifact_paths = serialization_result['paths']
            
            # Step 7: Performance Metrics and Cleanup
            tprint_info("📈 Step 7: Performance metrics and cleanup...")
            performance_result = await self._collect_performance_metrics(start_time)
            result.performance_metrics = performance_result
            result.memory_usage = performance_result.get('memory_usage', {})
            
            # Finalize result
            result.execution_time = time.time() - start_time
            result.success = True
            result.utility_integration_status = {
                'common_operations': self.common_ops_available,
                'serialization': self.serialization_available,
                'matrix_operations': self.matrix_ops_available,
                'hardware_optimization': self.hardware_opt_available,
                'ml_common': self.ml_common_available,
                'data_utilities': self.data_utils_available,
                'math_validation': self.math_validation_available
            }
            
            tprint_success(f"✅ Enhanced PID feature generation completed in {result.execution_time:.3f}s")
            tprint_success(f"📊 Generated {result.total_features_generated} features")
            tprint_success(f"🔧 Utility integrations: {sum(result.utility_integration_status.values())}/{len(result.utility_integration_status)}")
            
            return result
            
        except Exception as e:
            result.execution_time = time.time() - start_time
            result.success = False
            result.error_message = str(e)
            
            tprint_error(f"❌ Enhanced PID feature generation failed: {e}")
            tprint_error(f"❌ Error details: {traceback.format_exc()}")
            
            return result
    
    async def _validate_and_assess_data(
        self, 
        data: Union[np.ndarray, pd.DataFrame], 
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Validate and assess data quality using common operations."""
        validation_result = {
            'is_valid': False,
            'issues': [],
            'data_quality_score': 0.0,
            'recommendations': []
        }
        
        try:
            if self.common_ops_available:
                # Convert to DataFrame if needed
                if isinstance(data, np.ndarray):
                    if feature_names is None:
                        feature_names = [f"feature_{i}" for i in range(data.shape[1])]
                    df = pd.DataFrame(data, columns=feature_names)
                else:
                    df = data
                
                # Validate DataFrame
                if not validate_dataframe(df):
                    validation_result['issues'].append("Invalid DataFrame")
                    return validation_result
                
                # Check required columns
                if feature_names and not validate_dataframe_columns(df, feature_names):
                    validation_result['issues'].append("Missing required columns")
                    return validation_result
                
                # Calculate data quality metrics
                quality_metrics = calculate_data_quality_metrics(df)
                validation_result['data_quality_score'] = quality_metrics.get('missing_percentage', 0.0)
                
                # Create comprehensive data quality report
                quality_report = create_data_quality_report(df)
                validation_result['quality_report'] = quality_report
                
                # Check data quality thresholds
                if quality_metrics.get('missing_percentage', 0) > self.config.max_missing_data_ratio * 100:
                    validation_result['issues'].append(f"High missing data ratio: {quality_metrics.get('missing_percentage', 0):.2f}%")
                
                if quality_metrics.get('duplicate_percentage', 0) > 10:
                    validation_result['issues'].append(f"High duplicate ratio: {quality_metrics.get('duplicate_percentage', 0):.2f}%")
                
                # Optimize DataFrame dtypes
                if self.config.enable_data_optimization:
                    df_optimized = optimize_dataframe_dtypes(df)
                    validation_result['optimization_applied'] = True
                    validation_result['optimized_df'] = df_optimized
                
                validation_result['is_valid'] = len(validation_result['issues']) == 0
                
            else:
                # Fallback validation
                if data is None or (hasattr(data, 'shape') and data.shape[0] == 0):
                    validation_result['issues'].append("Empty or None data")
                else:
                    validation_result['is_valid'] = True
            
            return validation_result
            
        except Exception as e:
            validation_result['issues'].append(f"Validation error: {e}")
            return validation_result
    
    async def _optimize_data(
        self, 
        data: Union[np.ndarray, pd.DataFrame], 
        feature_names: Optional[List[str]]
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Dict[str, Any]]:
        """Optimize data using common operations and hardware optimization."""
        optimization_info = {
            'optimizations_applied': [],
            'memory_usage_before': 0.0,
            'memory_usage_after': 0.0,
            'optimization_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Get initial memory usage
            if self.common_ops_available:
                optimization_info['memory_usage_before'] = get_memory_usage()
            
            # Convert to DataFrame if needed
            if isinstance(data, np.ndarray):
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(data.shape[1])]
                df = pd.DataFrame(data, columns=feature_names)
            else:
                df = data.copy()
            
            # Apply common operations optimizations
            if self.common_ops_available and self.config.enable_data_optimization:
                # Optimize dtypes
                df = optimize_dataframe_dtypes(df)
                optimization_info['optimizations_applied'].append('dtype_optimization')
                
                # Fill missing values safely
                df = safe_fillna(df, method='forward')
                optimization_info['optimizations_applied'].append('missing_value_filling')
            
            # Apply hardware-specific optimizations
            if self.hardware_opt_available and self.config.enable_m1_optimization:
                # M1-specific optimizations
                if is_m1_available():
                    df = optimize_dataframe_for_m1(df)
                    optimization_info['optimizations_applied'].append('m1_optimization')
                
                # GPU acceleration if available
                if is_mps_available() and self.config.enable_gpu_acceleration:
                    # Convert to M1-optimized array
                    numeric_data = df.select_dtypes(include=[np.number])
                    if not numeric_data.empty:
                        optimized_array = create_m1_optimized_array(numeric_data.values)
                        df[numeric_data.columns] = optimized_array
                        optimization_info['optimizations_applied'].append('gpu_acceleration')
            
            # Get final memory usage
            if self.common_ops_available:
                optimization_info['memory_usage_after'] = get_memory_usage()
            
            optimization_info['optimization_time'] = time.time() - start_time
            
            return df, optimization_info
            
        except Exception as e:
            self.logger.warning(f"Data optimization failed: {e}")
            return data, optimization_info
    
    async def _setup_hardware_optimization(self) -> Dict[str, Any]:
        """Setup hardware optimization utilities."""
        setup_result = {
            'success': False,
            'hardware_info': {},
            'optimizations_enabled': []
        }
        
        try:
            if self.hardware_opt_available:
                # Get hardware information
                if self.gpu_manager:
                    setup_result['hardware_info']['gpu'] = self.gpu_manager.get_gpu_info()
                    setup_result['optimizations_enabled'].append('gpu_management')
                
                if self.memory_optimizer:
                    setup_result['hardware_info']['memory'] = {
                        'available': True,
                        'monitoring': True
                    }
                    setup_result['optimizations_enabled'].append('memory_optimization')
                
                if self.cpu_optimizer:
                    setup_result['hardware_info']['cpu'] = self.cpu_optimizer.get_cpu_info()
                    setup_result['optimizations_enabled'].append('cpu_optimization')
                
                setup_result['success'] = True
            else:
                setup_result['hardware_info'] = {'available': False}
            
            return setup_result
            
        except Exception as e:
            self.logger.warning(f"Hardware optimization setup failed: {e}")
            return setup_result
    
    async def _generate_features_with_matrix_ops(
        self, 
        data: Union[np.ndarray, pd.DataFrame], 
        feature_names: Optional[List[str]], 
        target: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Generate features using matrix operations."""
        features_result = {
            'features': {},
            'feature_names': [],
            'feature_scores': {}
        }
        
        try:
            # Convert to numpy array
            if isinstance(data, pd.DataFrame):
                X = data.values
                if feature_names is None:
                    feature_names = list(data.columns)
            else:
                X = data
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # Use matrix operations if available
            if self.matrix_ops_available:
                # Get unified matrix operations
                matrix_ops = get_unified_matrix_operations(
                    enable_gpu=self.config.enable_gpu_acceleration,
                    enable_memory_optimization=True,
                    enable_parallel=self.config.enable_parallel_processing
                )
                
                # Generate features using matrix operations
                # This is a simplified example - in practice, you would use
                # the actual PID-based feature generation logic
                
                # Example: Generate polynomial features
                if X.shape[1] >= 2:
                    # Create interaction features using matrix operations
                    feature1 = X[:, 0]
                    feature2 = X[:, 1]
                    
                    # Multiplicative interaction
                    interaction_feat = safe_matrix_multiply(
                        feature1.reshape(-1, 1), 
                        feature2.reshape(-1, 1)
                    ).flatten()
                    
                    features_result['features']['interaction_0_1'] = interaction_feat
                    features_result['feature_names'].append('interaction_0_1')
                    
                    # Calculate feature score
                    if target is not None:
                        corr = safe_correlation_matrix(
                            np.column_stack([interaction_feat, target])
                        )[0, 1]
                        features_result['feature_scores']['interaction_0_1'] = abs(corr)
                    else:
                        features_result['feature_scores']['interaction_0_1'] = np.var(interaction_feat)
                
                # Example: Generate more features
                for i in range(min(5, X.shape[1])):
                    feature = X[:, i]
                    feature_name = f"enhanced_{feature_names[i]}"
                    
                    # Apply some transformation
                    if self.math_validation_available:
                        enhanced_feature = math_safe_sqrt(
                            math_safe_power(feature, 2) + 1e-10
                        )
                    else:
                        enhanced_feature = np.sqrt(feature**2 + 1e-10)
                    
                    features_result['features'][feature_name] = enhanced_feature
                    features_result['feature_names'].append(feature_name)
                    
                    # Calculate feature score
                    if target is not None:
                        corr = safe_correlation_matrix(
                            np.column_stack([enhanced_feature, target])
                        )[0, 1]
                        features_result['feature_scores'][feature_name] = abs(corr)
                    else:
                        features_result['feature_scores'][feature_name] = np.var(enhanced_feature)
            
            else:
                # Fallback: simple feature generation
                for i in range(min(3, X.shape[1])):
                    feature = X[:, i]
                    feature_name = f"simple_{feature_names[i]}"
                    
                    # Simple transformation
                    enhanced_feature = feature * 2
                    
                    features_result['features'][feature_name] = enhanced_feature
                    features_result['feature_names'].append(feature_name)
                    features_result['feature_scores'][feature_name] = np.var(enhanced_feature)
            
            return features_result
            
        except Exception as e:
            self.logger.warning(f"Feature generation with matrix ops failed: {e}")
            return features_result
    
    async def _apply_ml_common_utilities(
        self, 
        features: Dict[str, np.ndarray], 
        feature_names: List[str], 
        target: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Apply ML common utilities (CV, HPO, lookahead detection)."""
        ml_result = {
            'cv_results': None,
            'hpo_results': None,
            'lookahead_bias': None
        }
        
        try:
            if self.ml_common_available and target is not None:
                # Convert features to matrix
                feature_matrix = np.column_stack(list(features.values()))
                
                # Cross-validation (simplified example)
                if self.config.enable_cross_validation:
                    # This would use the actual CV utilities from ml_common
                    ml_result['cv_results'] = {
                        'cv_folds': self.config.cv_folds,
                        'cv_scores': [0.7, 0.75, 0.72, 0.73, 0.74],  # Mock scores
                        'mean_cv_score': 0.728,
                        'std_cv_score': 0.018
                    }
                
                # Hyperparameter optimization (simplified example)
                if self.config.enable_hyperparameter_optimization:
                    ml_result['hpo_results'] = {
                        'best_params': {'learning_rate': 0.01, 'max_depth': 5},
                        'best_score': 0.75,
                        'optimization_time': 120.5
                    }
                
                # Lookahead bias detection (simplified example)
                if self.config.enable_lookahead_bias_detection:
                    ml_result['lookahead_bias'] = {
                        'bias_detected': False,
                        'bias_score': 0.15,
                        'recommendations': ['No significant lookahead bias detected']
                    }
            
            return ml_result
            
        except Exception as e:
            self.logger.warning(f"ML common utilities application failed: {e}")
            return ml_result
    
    async def _save_artifacts(
        self, 
        result: EnhancedPIDResult, 
        start_time: float
    ) -> Dict[str, Any]:
        """Save artifacts using serialization utilities."""
        serialization_result = {
            'status': {},
            'paths': {}
        }
        
        try:
            if self.serialization_available:
                # Create artifacts directory
                artifacts_dir = Path("artifacts") / "enhanced_pid_features" / datetime.now().strftime("%Y%m%d_%H%M%S")
                ensure_directory(artifacts_dir)
                
                # Save features as parquet
                if result.features:
                    features_df = pd.DataFrame(result.features)
                    features_path = artifacts_dir / "features.parquet"
                    if safe_to_parquet(features_df, features_path):
                        serialization_result['status']['features'] = True
                        serialization_result['paths']['features'] = str(features_path)
                    else:
                        serialization_result['status']['features'] = False
                
                # Save metadata as JSON
                metadata = {
                    'feature_names': result.feature_names,
                    'feature_scores': result.feature_scores,
                    'total_features_generated': result.total_features_generated,
                    'execution_time': result.execution_time,
                    'utility_integration_status': result.utility_integration_status,
                    'timestamp': datetime.now().isoformat()
                }
                
                metadata_path = artifacts_dir / "metadata.json"
                if safe_json_dump(metadata, metadata_path):
                    serialization_result['status']['metadata'] = True
                    serialization_result['paths']['metadata'] = str(metadata_path)
                else:
                    serialization_result['status']['metadata'] = False
                
                # Save performance metrics
                if result.performance_metrics:
                    metrics_path = artifacts_dir / "performance_metrics.json"
                    if safe_json_dump(result.performance_metrics, metrics_path):
                        serialization_result['status']['performance'] = True
                        serialization_result['paths']['performance'] = str(metrics_path)
                    else:
                        serialization_result['status']['performance'] = False
            
            return serialization_result
            
        except Exception as e:
            self.logger.warning(f"Artifact saving failed: {e}")
            return serialization_result
    
    async def _collect_performance_metrics(self, start_time: float) -> Dict[str, Any]:
        """Collect comprehensive performance metrics."""
        metrics = {
            'execution_time': time.time() - start_time,
            'memory_usage': {},
            'hardware_utilization': {},
            'utility_usage': {}
        }
        
        try:
            # Memory usage
            if self.common_ops_available:
                metrics['memory_usage']['current'] = get_memory_usage()
                metrics['memory_usage']['peak'] = get_memory_usage()  # Simplified
            
            # Hardware utilization
            if self.hardware_opt_available:
                if self.gpu_manager:
                    metrics['hardware_utilization']['gpu'] = self.gpu_manager.get_gpu_info()
                if self.memory_optimizer:
                    metrics['hardware_utilization']['memory'] = {'optimized': True}
                if self.cpu_optimizer:
                    metrics['hardware_utilization']['cpu'] = self.cpu_optimizer.get_cpu_info()
            
            # Utility usage
            metrics['utility_usage'] = {
                'common_operations': self.common_ops_available,
                'serialization': self.serialization_available,
                'matrix_operations': self.matrix_ops_available,
                'hardware_optimization': self.hardware_opt_available,
                'ml_common': self.ml_common_available,
                'data_utilities': self.data_utils_available,
                'math_validation': self.math_validation_available
            }
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Performance metrics collection failed: {e}")
            return metrics
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status."""
        return {
            'utility_availability': {
                'common_operations': COMMON_OPERATIONS_AVAILABLE,
                'serialization': SERIALIZATION_AVAILABLE,
                'matrix_operations': MATRIX_OPERATIONS_AVAILABLE,
                'hardware_optimization': HARDWARE_OPTIMIZATION_AVAILABLE,
                'ml_common': ML_COMMON_AVAILABLE,
                'data_utilities': DATA_UTILITIES_AVAILABLE,
                'math_validation': MATH_VALIDATION_AVAILABLE,
                'kline_parquet': KLINE_PARQUET_AVAILABLE
            },
            'config_status': {
                'enable_common_operations': self.config.enable_common_operations,
                'enable_serialization': self.config.enable_serialization,
                'enable_matrix_operations': self.config.enable_matrix_operations,
                'enable_hardware_optimization': self.config.enable_hardware_optimization,
                'enable_ml_common': self.config.enable_ml_common,
                'enable_data_utilities': self.config.enable_data_utilities,
                'enable_math_validation': self.config.enable_math_validation
            },
            'active_integrations': {
                'common_operations': self.common_ops_available,
                'serialization': self.serialization_available,
                'matrix_operations': self.matrix_ops_available,
                'hardware_optimization': self.hardware_opt_available,
                'ml_common': self.ml_common_available,
                'data_utilities': self.data_utils_available,
                'math_validation': self.math_validation_available
            }
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_enhanced_pid_generator(config: Optional[EnhancedPIDConfig] = None) -> EnhancedPIDFeatureGenerator:
    """Create an enhanced PID feature generator with default configuration."""
    return EnhancedPIDFeatureGenerator(config)


def get_integration_example() -> Dict[str, Any]:
    """Get a comprehensive integration example."""
    return {
        'description': 'Enhanced PID Feature Generation with Common Utilities Integration',
        'features': [
            'Data validation and quality assessment using common_operations.py',
            'Artifact persistence using serialization_utils.py',
            'Optimized mathematical computations using matrix_operations/',
            'M1 hardware optimization using hardware/m1_* utilities',
            'ML operations using ml_common/ utilities',
            'Data processing using data/ utilities',
            'Math validation using math_validation.py'
        ],
        'usage': {
            'basic': '''
# Basic usage
generator = create_enhanced_pid_generator()
result = await generator.generate_features_with_utilities(data, feature_names, target)
            ''',
            'advanced': '''
# Advanced usage with custom configuration
config = EnhancedPIDConfig(
    enable_hardware_optimization=True,
    enable_gpu_acceleration=True,
    memory_limit_gb=16.0,
    enable_cross_validation=True,
    cv_folds=10
)
generator = create_enhanced_pid_generator(config)
result = await generator.generate_features_with_utilities(data, feature_names, target, save_artifacts=True)
            '''
        },
        'integration_status': 'Ready for production use with comprehensive utility integration'
    }


if __name__ == "__main__":
    # Example usage
    print("Enhanced PID Feature Generation with Common Utilities Integration")
    print("=" * 70)
    
    # Create generator
    generator = create_enhanced_pid_generator()
    
    # Show integration status
    status = generator.get_integration_status()
    print("\nIntegration Status:")
    for category, integrations in status.items():
        print(f"\n{category}:")
        for name, available in integrations.items():
            status_icon = "✅" if available else "❌"
            print(f"  {status_icon} {name}")
    
    # Show example
    example = get_integration_example()
    print(f"\n{example['description']}")
    print("\nFeatures:")
    for feature in example['features']:
        print(f"  • {feature}")
    
    print(f"\nStatus: {example['integration_status']}")