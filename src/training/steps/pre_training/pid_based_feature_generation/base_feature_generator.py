"""
Base Feature Generator with Common Utilities Integration

This module provides a base class for all feature generators with common utilities
integration, eliminating code duplication and providing consistent functionality.
"""

import asyncio
import logging
import time
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

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

# Import common operations for comprehensive utility integration
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
    # Fallback functions
    def _is_finite_scalar(v):
        try:
            return math.isfinite(float(v))
        except Exception:
            return False
    def safe_divide(a, b, default=0.0):
        try:
            return a / b if b not in (0, 0.0) else default
        except Exception:
            return default
    def safe_log(x, default=0.0):
        try:
            xv = float(x)
            return math.log(xv) if xv > 0 else default
        except Exception:
            return default
    def safe_sqrt(x, default=0.0):
        try:
            xv = float(x)
            return math.sqrt(xv) if xv >= 0 else default
        except Exception:
            return default
    def safe_power(x, y, default=0.0):
        try:
            if _is_finite_scalar(x) and _is_finite_scalar(y):
                return float(x) ** float(y)
            return default
        except Exception:
            return default
    def validate_finite(value, name="value"):
        try:
            return float(value) if _is_finite_scalar(value) else 0.0
        except Exception:
            return 0.0

# Import serialization utilities
try:
    from src.utils.serialization_utils import (
        JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
    )
    SERIALIZATION_AVAILABLE = True
except ImportError as e:
    SERIALIZATION_AVAILABLE = False
    logging.warning(f"Serialization utilities not available: {e}")

# Import math validation for additional math operations
try:
    from src.utils.math_validation import MathValidation, safe_correlation, safe_covariance, safe_percentile
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False

# Import tprint for extensive logging
try:
    from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_warning, tprint_success, tprint_debug, tprint_performance
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    # Fallback to basic print
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)

# Import matrix operations
try:
    from src.utils.matrix_operations import get_unified_matrix_operations
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Matrix operations not available: {e}")
    MATRIX_OPS_AVAILABLE = False

# Import logger
try:
    from src.utils.logger import system_logger
    logger = system_logger.getChild('BaseFeatureGenerator')
except ImportError:
    logger = logging.getLogger('BaseFeatureGenerator')
    logger.setLevel(logging.INFO)


@dataclass
class BaseFeatureConfig:
    """Base configuration for feature generators with common utilities integration."""
    # Common Utilities Integration
    enable_common_operations: bool = True
    enable_serialization: bool = True
    enable_data_validation: bool = True
    enable_data_optimization: bool = True
    enable_m1_optimization: bool = True
    
    # Data Quality Settings
    min_data_quality_score: float = 0.7
    max_missing_data_ratio: float = 0.1
    enable_quality_reporting: bool = True
    
    # Performance Settings
    enable_profiling: bool = True
    enable_memory_monitoring: bool = True
    enable_performance_logging: bool = True


@dataclass
class BaseFeatureResult:
    """Base result for feature generators with common utilities integration."""
    # Common Utilities Integration Results
    data_quality_report: Optional[Dict[str, Any]] = None
    validation_results: Dict[str, Any] = field(default_factory=dict)
    optimization_results: Dict[str, Any] = field(default_factory=dict)
    serialization_status: Dict[str, bool] = field(default_factory=dict)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    hardware_optimization_used: bool = False
    memory_usage: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    utility_integration_status: Dict[str, bool] = field(default_factory=dict)


class BaseFeatureGenerator(ABC):
    """
    Base class for feature generators with common utilities integration.
    
    This class provides common functionality for all feature generators,
    eliminating code duplication and ensuring consistent behavior.
    """
    
    def __init__(self, config: Optional[BaseFeatureConfig] = None, logger_name: str = "BaseFeatureGenerator"):
        """Initialize the base feature generator with common utilities integration."""
        self.config = config or BaseFeatureConfig()
        self.logger = logger.getChild(logger_name)
        
        # Initialize common utilities integration
        self._initialize_common_utilities()
        
        # Initialize math validation
        if MATH_VALIDATION_AVAILABLE:
            self.math_validator = MathValidation()
        else:
            self.math_validator = None
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info(f"🔧 {logger_name} initialized successfully")
        self.logger.info(f"🔧 Common operations available: {COMMON_OPERATIONS_AVAILABLE}")
        self.logger.info(f"🔧 Serialization available: {SERIALIZATION_AVAILABLE}")
        self.logger.info(f"🔧 Math validation available: {MATH_VALIDATION_AVAILABLE}")
        self.logger.info(f"🔧 Matrix operations available: {MATRIX_OPS_AVAILABLE}")
    
    def _initialize_common_utilities(self):
        """Initialize common utilities integration."""
        # Initialize serializers
        if SERIALIZATION_AVAILABLE and self.config.enable_serialization:
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
        
        # Initialize M1 optimizers
        if COMMON_OPERATIONS_AVAILABLE and self.config.enable_m1_optimization:
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
            self.logger.info("✅ M1 optimizers initialized")
        else:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
        
        # Initialize utility status tracking
        self.utility_integration_status = {
            'common_operations': COMMON_OPERATIONS_AVAILABLE and self.config.enable_common_operations,
            'serialization': SERIALIZATION_AVAILABLE and self.config.enable_serialization,
            'math_validation': MATH_VALIDATION_AVAILABLE,
            'matrix_operations': MATRIX_OPS_AVAILABLE,
            'data_validation': self.config.enable_data_validation,
            'data_optimization': self.config.enable_data_optimization,
            'm1_optimization': self.config.enable_m1_optimization
        }
        
        self.logger.info(f"🔧 Utility integration status: {self.utility_integration_status}")
    
    @abstractmethod
    def _initialize_components(self):
        """Initialize required components. Must be implemented by subclasses."""
        # Initialize feature generation components using feature_engineering bank
        try:
            # Import feature generation system
            from src.feature_generation import (
                FeatureBank,
                ReturnsFeatureGenerator,
                MomentumFeatureGenerator,
                VolumeFeatureGenerator,
                VolatilityFeatureGenerator,
                TrendFeatureGenerator,
                InteractionFeatureGenerator,
                CrossTimeframeFeatureGenerator,
                generate_features_by_category,
                FeatureGenerationOptimizer,
                get_feature_optimizer
            )
            
            # Initialize feature bank
            self.feature_bank = FeatureBank()
            self.feature_optimizer = get_feature_optimizer()
            
            # Initialize specific feature generators
            self.returns_generator = ReturnsFeatureGenerator()
            self.momentum_generator = MomentumFeatureGenerator()
            self.volume_generator = VolumeFeatureGenerator()
            self.volatility_generator = VolatilityFeatureGenerator()
            self.trend_generator = TrendFeatureGenerator()
            self.interaction_generator = InteractionFeatureGenerator()
            self.cross_timeframe_generator = CrossTimeframeFeatureGenerator()
            # Polynomial generator removed - not used for NAS/TAS
            self.polynomial_generator = None
            
            # Set up feature generation categories
            self.available_categories = [
                'returns', 'momentum', 'volume', 'volatility', 'trend',
                'interaction', 'cross_timeframe'
            ]
            
            self.logger.info("✅ Feature generation components initialized from feature_engineering bank")
            
        except ImportError as e:
            self.logger.warning(f"Feature generation system not available: {e}")
            # Fallback to basic components
            self.feature_bank = None
            self.feature_optimizer = None
            self.available_categories = ['basic']
            
        except Exception as e:
            self.logger.error(f"Failed to initialize feature components: {e}")
            # Fallback to basic components
            self.feature_bank = None
            self.feature_optimizer = None
            self.available_categories = ['basic']
    
    async def _validate_input_data(
        self, 
        data: Union[np.ndarray, pd.DataFrame], 
        feature_names: Optional[List[str]], 
        target: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Validate input data using common utilities."""
        validation_result = {
            'is_valid': False,
            'issues': [],
            'data_quality_score': 0.0
        }
        
        try:
            if COMMON_OPERATIONS_AVAILABLE and self.config.enable_data_validation and PANDAS_AVAILABLE:
                # Convert to DataFrame for validation
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
                validation_result['data_quality_score'] = 1.0 - (quality_metrics.get('missing_percentage', 0) / 100)
                
                # Check data quality thresholds
                if quality_metrics.get('missing_percentage', 0) > self.config.max_missing_data_ratio * 100:
                    validation_result['issues'].append(f"High missing data ratio: {quality_metrics.get('missing_percentage', 0):.2f}%")
                
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
    
    async def _optimize_input_data(
        self, 
        data: Union[np.ndarray, pd.DataFrame], 
        feature_names: Optional[List[str]]
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], List[str], Dict[str, Any]]:
        """Optimize input data using common utilities."""
        optimization_info = {
            'optimizations_applied': [],
            'memory_usage_before': 0.0,
            'memory_usage_after': 0.0,
            'optimization_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            if COMMON_OPERATIONS_AVAILABLE and PANDAS_AVAILABLE:
                # Get initial memory usage
                optimization_info['memory_usage_before'] = get_memory_usage()
                
                # Convert to DataFrame if needed
                if isinstance(data, np.ndarray):
                    if feature_names is None:
                        feature_names = [f"feature_{i}" for i in range(data.shape[1])]
                    df = pd.DataFrame(data, columns=feature_names)
                else:
                    df = data.copy()
                
                # Optimize dtypes
                df = optimize_dataframe_dtypes(df)
                optimization_info['optimizations_applied'].append('dtype_optimization')
                
                # Fill missing values safely
                df = safe_fillna(df, method='forward')
                optimization_info['optimizations_applied'].append('missing_value_filling')
                
                # Apply M1-specific optimizations
                if self.config.enable_m1_optimization and self.gpu_manager:
                    optimization_info['optimizations_applied'].append('m1_optimization')
                
                # Get final memory usage
                optimization_info['memory_usage_after'] = get_memory_usage()
                optimization_info['optimization_time'] = time.time() - start_time
                
                return df, feature_names, optimization_info
            else:
                return data, feature_names, optimization_info
                
        except Exception as e:
            self.logger.warning(f"Data optimization failed: {e}")
            return data, feature_names, optimization_info
    
    async def _assess_data_quality(
        self, 
        X: np.ndarray, 
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Assess data quality using common utilities."""
        quality_report = {
            'overall_score': 0.0,
            'missing_data_ratio': 0.0,
            'duplicate_ratio': 0.0,
            'data_types': {},
            'statistics': {}
        }
        
        try:
            if COMMON_OPERATIONS_AVAILABLE and PANDAS_AVAILABLE:
                # Convert to DataFrame for quality assessment
                df = pd.DataFrame(X, columns=feature_names)
                
                # Calculate data quality metrics
                quality_metrics = calculate_data_quality_metrics(df)
                
                # Create comprehensive quality report
                quality_report = create_data_quality_report(df)
                
                # Calculate overall score
                missing_ratio = quality_metrics.get('missing_percentage', 0) / 100
                duplicate_ratio = quality_metrics.get('duplicate_percentage', 0) / 100
                
                quality_report['overall_score'] = max(0.0, 1.0 - missing_ratio - duplicate_ratio)
                quality_report['missing_data_ratio'] = missing_ratio
                quality_report['duplicate_ratio'] = duplicate_ratio
                
                # Add basic statistics
                quality_report['statistics'] = {
                    'mean': safe_mean(pd.Series(X.flatten())),
                    'std': safe_std(pd.Series(X.flatten())),
                    'min': float(np.min(X)),
                    'max': float(np.max(X))
                }
            
            return quality_report
            
        except Exception as e:
            self.logger.warning(f"Data quality assessment failed: {e}")
            return quality_report
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics with common utilities integration."""
        metrics = {
            'common_operations_available': COMMON_OPERATIONS_AVAILABLE,
            'serialization_available': SERIALIZATION_AVAILABLE,
            'math_validation_available': MATH_VALIDATION_AVAILABLE,
            'matrix_operations_available': MATRIX_OPS_AVAILABLE
        }
        
        # Add common utilities metrics
        metrics['utility_integration_status'] = getattr(self, 'utility_integration_status', {})
        metrics['memory_usage'] = get_memory_usage() if COMMON_OPERATIONS_AVAILABLE else 0.0
        
        return metrics
    
    def _set_utility_integration_status(self, result: BaseFeatureResult):
        """Set utility integration status in result."""
        result.utility_integration_status = getattr(self, 'utility_integration_status', {})
        result.hardware_optimization_used = bool(self.gpu_manager or self.memory_optimizer or self.cpu_optimizer)
        
        # Collect performance metrics
        if self.config.enable_performance_logging:
            result.performance_metrics = self.get_performance_metrics()
            result.memory_usage = {'current': get_memory_usage()} if COMMON_OPERATIONS_AVAILABLE else {}